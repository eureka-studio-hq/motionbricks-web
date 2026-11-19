// Runs NVIDIA MotionBricks (int8 ONNX) with ONNX Runtime Web inside a worker, so the 3D view never stalls.
import * as ort from '../vendor/ort.wasm.min.mjs';

const DB = 'motionbricks-cache', STORE = 'blobs';
let KEY = 'soma';
let session = null;

function idb() {
  return new Promise((res, rej) => {
    const r = indexedDB.open(DB, 1);
    r.onupgradeneeded = () => r.result.createObjectStore(STORE);
    r.onsuccess = () => res(r.result); r.onerror = () => rej(r.error);
  });
}
async function cacheGet() {
  try { const db = await idb(); return await new Promise((res) => { const g = db.transaction(STORE).objectStore(STORE).get(KEY); g.onsuccess = () => res(g.result || null); g.onerror = () => res(null); }); }
  catch { return null; }
}
async function cachePut(buf) {
  try { const db = await idb(); await new Promise((res) => { const t = db.transaction(STORE, 'readwrite'); t.objectStore(STORE).put(buf, KEY); t.oncomplete = res; t.onerror = res; }); }
  catch { /* private mode etc. */ }
}

async function loadModel(parts, total, packed) {
  const cached = await cacheGet();
  if (cached && cached.byteLength === total) { postMessage({ type: 'progress', loaded: packed, total: packed, cached: true }); return new Uint8Array(cached); }
  // the model ships gzip-compressed in parts; stream them in order through the browser's gzip decoder
  let got = 0, i = 0, reader = null;
  const src = new ReadableStream({
    async pull(ctrl) {
      for (;;) {
        if (!reader) {
          if (i >= parts.length) { ctrl.close(); return; }
          const r = await fetch(new URL(parts[i++], import.meta.url));
          if (!r.ok) throw new Error(`model part ${i}: HTTP ${r.status}`);
          reader = r.body.getReader();
        }
        const { done, value } = await reader.read();
        if (done) { reader = null; continue; }
        got += value.byteLength; postMessage({ type: 'progress', loaded: got, total: packed });
        ctrl.enqueue(value); return;
      }
    },
  });
  const out = new Uint8Array(total); let off = 0;
  const dec = src.pipeThrough(new DecompressionStream('gzip')).getReader();
  for (;;) { const { done, value } = await dec.read(); if (done) break; out.set(value, off); off += value.byteLength; }
  if (off !== total) throw new Error(`model size ${off} != ${total}`);
  cachePut(out.buffer);
  return out;
}

const SHAPES = { global_root_values: [1, 8, 5], local_root_values: [1, 8, 4], local_poses: [1, 8, 267],
  has_global_root_values: [1, 8], has_local_root_values: [1, 8], has_local_poses: [1, 8], allowed_pred_num_tokens: [1, 11] };

onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === 'init') {
      const threads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
      ort.env.wasm.numThreads = threads;
      ort.env.wasm.wasmPaths = new URL('../vendor/', import.meta.url).href;
      KEY = 'soma-' + m.key; // a rebuilt model gets a fresh cache entry
      const bytes = await loadModel(m.parts, m.total, m.packed);
      postMessage({ type: 'status', text: 'Compiling the model…' });
      const t0 = performance.now();
      session = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
      postMessage({ type: 'ready', ms: performance.now() - t0, threads });
    } else if (m.type === 'run') {
      const feeds = {};
      for (const k in m.feeds) feeds[k] = new ort.Tensor(m.feeds[k] instanceof Int32Array ? 'int32' : 'float32', m.feeds[k], SHAPES[k]);
      const t0 = performance.now();
      const r = await session.run(feeds);
      const ms = performance.now() - t0;
      const pred = r.pred_global_motions;
      const data = new Float32Array(pred.data);
      postMessage({ type: 'result', id: m.id, pred: data, dims: pred.dims.slice(1), tokens: Number(r.pred_num_tokens.data[0]), ms }, [data.buffer]);
    }
  } catch (err) {
    postMessage({ type: 'error', id: m.id, text: String(err && err.message || err) });
  }
};
