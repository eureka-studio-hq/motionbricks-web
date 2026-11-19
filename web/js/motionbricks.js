// MotionBricks SOMA encode/decode, ported from NVIDIA's Warp kernels (Apache-2.0) to plain JavaScript.
// Transforms are {p:[x,y,z], q:[x,y,z,w]} like Warp. Poses are arrays of transforms in skeleton order.

export const POSES_PER_TOKEN = 4, SAMPLE_RATE = 30, BOUNDARY = 3, MIN_TOK = 6, MAX_TOK = 16;
export const DEFAULT_TOKENS = [8, 9, 10, 11, 12, 13, 14];

// ---------- math (Warp conventions) ----------
export const qmul = (a, b) => [
  a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
  a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
  a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
  a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
];
export const qinv = (q) => [-q[0], -q[1], -q[2], q[3]];
export const qnorm = (q) => { const l = Math.hypot(q[0], q[1], q[2], q[3]) || 1; return [q[0] / l, q[1] / l, q[2] / l, q[3] / l]; };
export const qrot = (q, v) => {
  const [x, y, z, w] = q, [vx, vy, vz] = v;
  const tx = 2 * (y * vz - z * vy), ty = 2 * (z * vx - x * vz), tz = 2 * (x * vy - y * vx);
  return [vx + w * tx + (y * tz - z * ty), vy + w * ty + (z * tx - x * tz), vz + w * tz + (x * ty - y * tx)];
};
export const qaxis = (axis, ang) => { const s = Math.sin(ang / 2); return [axis[0] * s, axis[1] * s, axis[2] * s, Math.cos(ang / 2)]; };
export function qslerp(a, b, t) {
  let d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
  let bb = b;
  if (d < 0) { d = -d; bb = [-b[0], -b[1], -b[2], -b[3]]; }
  if (d > 0.9995) return qnorm([a[0] + t * (bb[0] - a[0]), a[1] + t * (bb[1] - a[1]), a[2] + t * (bb[2] - a[2]), a[3] + t * (bb[3] - a[3])]);
  const th = Math.acos(d), s = Math.sin(th), wa = Math.sin((1 - t) * th) / s, wb = Math.sin(t * th) / s;
  return [wa * a[0] + wb * bb[0], wa * a[1] + wb * bb[1], wa * a[2] + wb * bb[2], wa * a[3] + wb * bb[3]];
}
export function qFromMatrixCols(x, y, z) { // columns x,y,z -> quat (x,y,z,w)
  const m00 = x[0], m10 = x[1], m20 = x[2], m01 = y[0], m11 = y[1], m21 = y[2], m02 = z[0], m12 = z[1], m22 = z[2];
  const tr = m00 + m11 + m22;
  let q;
  if (tr > 0) { const s = Math.sqrt(tr + 1) * 2; q = [(m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s, 0.25 * s]; }
  else if (m00 > m11 && m00 > m22) { const s = Math.sqrt(1 + m00 - m11 - m22) * 2; q = [0.25 * s, (m01 + m10) / s, (m02 + m20) / s, (m21 - m12) / s]; }
  else if (m11 > m22) { const s = Math.sqrt(1 + m11 - m00 - m22) * 2; q = [(m01 + m10) / s, 0.25 * s, (m12 + m21) / s, (m02 - m20) / s]; }
  else { const s = Math.sqrt(1 + m22 - m00 - m11) * 2; q = [(m02 + m20) / s, (m12 + m21) / s, 0.25 * s, (m10 - m01) / s]; }
  return qnorm(q);
}
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const scl = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const nrm = (a) => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };
export const tmul = (a, b) => ({ p: add(a.p, qrot(a.q, b.p)), q: qmul(a.q, b.q) });
export const tinv = (t) => { const qi = qinv(t.q); return { p: scl(qrot(qi, t.p), -1), q: qi }; };
export const T = (p, q) => ({ p: [...p], q: [...q] });

// ---------- skeleton ----------
export class Skeleton {
  constructor(json) {
    this.names = json.names; this.parents = json.parents;
    this.ref = json.reference_local.map((r) => ({ p: r.slice(0, 3), q: r.slice(3, 7) }));
    this.modelJoints = json.model_joints;
    this.modelToPose = this.modelJoints.map((n) => this.names.indexOf(n));
    this.poseToModel = new Int32Array(this.names.length).fill(-1);
    this.modelToPose.forEach((pj, mj) => {
      const par = this.parents[pj];
      if (par === -1 || this.poseToModel[par] >= 0) this.poseToModel[pj] = mj;
    });
    this.rightLeg = this.names.indexOf('RightLeg'); this.leftLeg = this.names.indexOf('LeftLeg');
    this.J = this.names.length;
  }
  world(local, rootTx = { p: [0, 0, 0], q: [0, 0, 0, 1] }) {
    const w = new Array(this.J);
    w[0] = tmul(rootTx, local[0]);
    for (let j = 1; j < this.J; j++) w[j] = tmul(w[this.parents[j]], local[j]);
    return w;
  }
}

function headingFromHips(r, l) {
  let ax = [r[0] - l[0], 0, r[2] - l[2]];
  const len = Math.hypot(ax[0], ax[2]);
  if (len <= 1e-6) return 0;
  ax = scl(ax, 1 / len);
  const f = cross([0, 1, 0], ax);
  return Math.atan2(f[0], f[2]);
}
const rot6d = (x, y) => { const xa = nrm(x), za = nrm(cross(xa, y)), ya = cross(za, xa); return qFromMatrixCols(xa, ya, za); };

// ---------- encode ----------
// context: 4 local poses; targets: 1..4 local poses. modelToRuntime: transform (world = modelToRuntime * model).
export function encode(skel, context, targets, modelToRuntime, opts = {}) {
  const heading = opts.headingAlignment || [1, 0];
  const tokens = opts.allowedTokens || DEFAULT_TOKENS;
  const nT = targets.length, padded = [...targets];
  while (padded.length < 4) padded.push(targets[nT - 1]);
  const rt2m = tinv(modelToRuntime);
  const guide = [...context, ...padded].map((lp) => skel.world(lp, rt2m)); // 8 world poses in model space
  // alignment
  const lc = guide[3], hips = lc[0].p;
  const cur = headingFromHips(lc[skel.rightLeg].p, lc[skel.leftLeg].p);
  const des = Math.atan2(heading[0], heading[1]);
  const align = { p: [hips[0], 0, hips[2]], q: qaxis([0, 1, 0], cur - des) };
  const aq = qinv(align.q), ap = (v) => qrot(aq, sub(v, align.p));
  const nJ = skel.modelJoints.length;
  const g = new Float32Array(8 * 5), l = new Float32Array(8 * 4), lp = new Float32Array(8 * 267);
  for (let f = 0; f < 8; f++) {
    const nf = f === 3 ? f : Math.min(f + 1, 7);
    const W = guide[f], Wn = guide[nf];
    const ch = ap(W[0].p), nh = ap(Wn[0].p);
    const chd = headingFromHips(ap(W[skel.rightLeg].p), ap(W[skel.leftLeg].p));
    const nhd = headingFromHips(ap(Wn[skel.rightLeg].p), ap(Wn[skel.leftLeg].p));
    const dh = Math.atan2(Math.sin(nhd - chd), Math.cos(nhd - chd));
    g.set([ch[0], ch[1], ch[2], Math.cos(chd), Math.sin(chd)], f * 5);
    l.set([dh * SAMPLE_RATE, (nh[0] - ch[0]) * SAMPLE_RATE, (nh[2] - ch[2]) * SAMPLE_RATE, ch[1]], f * 4);
    const root = [ch[0], 0, ch[2]], base = f * 267;
    for (let mj = 0; mj < nJ; mj++) {
      const pj = skel.modelToPose[mj];
      const jp = ap(W[pj].p), jq = qmul(aq, W[pj].q);
      if (mj > 0) { const c = sub(jp, root); lp.set(c, base + (mj - 1) * 3); }
      const xa = qrot(jq, [1, 0, 0]), ya = qrot(jq, [0, 1, 0]);
      lp.set([...xa, ...ya], base + (nJ - 1) * 3 + mj * 6);
    }
  }
  const hasG = new Int32Array(8), hasL = new Int32Array(8), hasP = new Int32Array(8), allowed = new Int32Array(11);
  for (let i = 0; i < 8; i++) {
    const v = i < 4 + nT ? 1 : 0; hasG[i] = v; hasP[i] = v;
    hasL[i] = (i === 3 || i === 4 + nT - 1) ? 0 : v;
  }
  for (const t of tokens) allowed[t - MIN_TOK] = 1;
  return {
    feeds: { global_root_values: g, local_root_values: l, local_poses: lp, has_global_root_values: hasG, has_local_root_values: hasL, has_local_poses: hasP, allowed_pred_num_tokens: allowed },
    align, guide,
  };
}

// ---------- decode + endpoint correction ----------
// pred: Float32Array [64*366] (batch 1), predTokens: int. Returns local poses for valid frames.
export function decode(skel, pred, predTokens, align, modelToRuntime, context, targets, dims = [64, 366]) {
  const [F, D] = dims;
  const raw = Math.min(Math.max(predTokens * 4, 0), F);
  const n = Math.max(0, raw - 2 * BOUNDARY);
  const nJ = skel.modelJoints.length, rotOff = 5 + (nJ - 1) * 3;
  const rd = (fr, off) => { const b = fr * D + off; return rot6d([pred[b], pred[b + 1], pred[b + 2]], [pred[b + 3], pred[b + 4], pred[b + 5]]); };
  const out = [];
  for (let f = 0; f < n; f++) {
    const rf = BOUNDARY + f, b = rf * D;
    const hip = tmul(align, { p: [pred[b], pred[b + 1], pred[b + 2]], q: rd(rf, rotOff) });
    const pose = new Array(skel.J);
    pose[0] = tmul(modelToRuntime, hip);
    for (let j = 1; j < skel.J; j++) {
      const mj = skel.poseToModel[j];
      if (mj < 0) { pose[j] = T(skel.ref[j].p, skel.ref[j].q); continue; }
      const pr = rd(rf, rotOff + mj * 6), par = rd(rf, rotOff + skel.poseToModel[skel.parents[j]] * 6);
      pose[j] = { p: [...skel.ref[j].p], q: qnorm(qmul(qinv(par), pr)) };
    }
    out.push(pose);
  }
  if (n >= 2) {
    const last = n - 1, lc = context[context.length - 1], t0 = targets[0];
    const sd = [], ed = [];
    const rootDelta = (g, d) => ({ p: sub(d.p, g.p), q: qnorm(qmul(qinv(qnorm(g.q)), qnorm(d.q))) });
    sd[0] = rootDelta(out[0][0], lc[0]); ed[0] = rootDelta(out[last][0], t0[0]);
    for (let j = 1; j < skel.J; j++) { sd[j] = tmul(tinv(out[0][j]), lc[j]); ed[j] = tmul(tinv(out[last][j]), t0[j]); }
    const I = [0, 0, 0, 1];
    for (let f = 0; f < n; f++) {
      const ew = f / (n - 1), sw = 1 - ew;
      const r = out[f][0];
      out[f][0] = { p: add(add(r.p, scl(sd[0].p, sw)), scl(ed[0].p, ew)), q: qnorm(qmul(qmul(r.q, qslerp(I, sd[0].q, sw)), qslerp(I, ed[0].q, ew))) };
      for (let j = 1; j < skel.J; j++) {
        const blend = (d, w) => ({ p: scl(d.p, w), q: qslerp(I, d.q, w) });
        out[f][j] = tmul(tmul(out[f][j], blend(sd[j], sw)), blend(ed[j], ew));
      }
    }
  }
  return out;
}
