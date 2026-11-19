// Direct the Hero: NVIDIA MotionBricks generating character motion live, in the browser.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { RoundedBoxGeometry } from 'three/addons/geometries/RoundedBoxGeometry.js';
import { makeKayRig } from './retarget.js';
import { Skeleton, encode, decode, tmul, qrot, qmul, qaxis, qslerp } from './motionbricks.js';

const M2R = { p: [0, 0, 0], q: [0.70710677, 0, 0, 0.70710677] }; // SOMA model space (Y-up) -> runtime (Z-up)
const $ = (id) => document.getElementById(id);

// ---------------------------------------------------------------- data
const need = (url) => fetch(url).then((r) => { if (!r.ok) throw new Error(url); return r; });
const loadGlb = (url) => need(url).then((r) => r.arrayBuffer()).then((buf) => new GLTFLoader().parseAsync(buf, ''));
const KK = ['knight', 'rogue_hooded', 'mage', 'barbarian'];
let skelJson, lib, manifest, kkGltf;
try {
  [skelJson, lib, manifest, ...kkGltf] = await Promise.all([
    need('data/skeleton.json').then((r) => r.json()),
    need('data/pose_library.json').then((r) => r.json()),
    need('model/manifest.json').then((r) => r.json()),
    ...KK.map((k) => loadGlb(`data/characters/${k}.glb`)),
  ]);
} catch (err) {
  // the skeleton, poses and model are built locally from NVIDIA MotionBricks: see the README
  document.getElementById('loadtext').innerHTML = `Missing <code>${err.message}</code>.<br>Build the assets first:<br><code>python tools/build_web_assets.py --motionbricks /path/to/motionbricks</code>`;
  throw err;
}
const SK = new Skeleton(skelJson);
const J = SK.J;
const toPose = (flat) => { const p = new Array(J); for (let j = 0; j < J; j++) { const b = j * 7; p[j] = { p: [flat[b], flat[b + 1], flat[b + 2]], q: [flat[b + 3], flat[b + 4], flat[b + 5], flat[b + 6]] }; } return p; };
const LIB = {}; for (const k in lib.poses) LIB[k] = lib.poses[k].map((w) => w.map(toPose));

// ---------------------------------------------------------------- scene
const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
renderer.shadowMap.enabled = true; renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping; renderer.toneMappingExposure = 1.05;
renderer.outputColorSpace = THREE.SRGBColorSpace;
$('stage').appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b0e14);
scene.fog = new THREE.Fog(0x0b0e14, 14, 42);
const camera = new THREE.PerspectiveCamera(38, innerWidth / innerHeight, 0.1, 200);
camera.position.set(-5.5, 3.6, 6.5);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true; controls.maxPolarAngle = Math.PI * 0.47; controls.minDistance = 2.5; controls.maxDistance = 22;
controls.target.set(0, 0.9, 0);

const hemi = new THREE.HemisphereLight(0x9fb4d9, 0x1a1c22, 0.9); scene.add(hemi);
const key = new THREE.DirectionalLight(0xfff1e0, 2.4); key.position.set(-4, 9, 5); key.castShadow = true;
key.shadow.mapSize.set(2048, 2048); Object.assign(key.shadow.camera, { left: -8, right: 8, top: 8, bottom: -8, near: 0.5, far: 30 });
key.shadow.bias = -0.0004; key.shadow.normalBias = 0.02;
scene.add(key, key.target);
const rim = new THREE.DirectionalLight(0x6fa8ff, 1.2); rim.position.set(5, 4, -6); scene.add(rim);

// checker floor with a soft spotlight falloff
const floorMat = new THREE.ShaderMaterial({
  uniforms: { uFocus: { value: new THREE.Vector2() }, uA: { value: new THREE.Color() }, uB: { value: new THREE.Color() }, uLine: { value: 0.035 }, uFall: { value: 0.75 } },
  vertexShader: `varying vec3 vW; void main(){ vec4 w = modelMatrix*vec4(position,1.); vW=w.xyz; gl_Position=projectionMatrix*viewMatrix*w; }`,
  fragmentShader: `varying vec3 vW; uniform vec2 uFocus; uniform vec3 uA, uB; uniform float uLine, uFall;
    void main(){ vec2 c=floor(vW.xz); float chk=mod(c.x+c.y,2.);
      vec3 col=mix(uA,uB,chk); vec2 g=abs(fract(vW.xz-.5)-.5)/fwidth(vW.xz); float line=1.-min(min(g.x,g.y),1.);
      col+=line*uLine; float d=length(vW.xz-uFocus); col*=1.+uFall*.33-smoothstep(2.,18.,d)*uFall; gl_FragColor=vec4(col,1.);
      #include <colorspace_fragment>
    }`,
});
const floor = new THREE.Mesh(new THREE.PlaneGeometry(160, 160), floorMat); floor.rotation.x = -Math.PI / 2; scene.add(floor);
const shadowCatcher = new THREE.Mesh(new THREE.PlaneGeometry(160, 160), new THREE.ShadowMaterial({ opacity: 0.45 }));
shadowCatcher.rotation.x = -Math.PI / 2; shadowCatcher.position.y = 0.002; shadowCatcher.receiveShadow = true; scene.add(shadowCatcher);

// ---------------------------------------------------------------- worlds
function gradientTex(top, bottom) {
  const c = document.createElement('canvas'); c.width = 4; c.height = 256; const g = c.getContext('2d');
  const gr = g.createLinearGradient(0, 0, 0, 256); gr.addColorStop(0, top); gr.addColorStop(0.62, bottom); gr.addColorStop(1, bottom);
  g.fillStyle = gr; g.fillRect(0, 0, 4, 256); const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; return t;
}
let seed = 7; const rnd = (a = 0, b = 1) => { seed = (seed * 16807) % 2147483647; return a + ((seed - 1) / 2147483646) * (b - a); };
const decor = new THREE.Group(); scene.add(decor);
{ // toy playground set dressing: far enough out that the courses never reach it
  const soft = (c) => new THREE.MeshStandardMaterial({ color: c, roughness: 0.9 });
  const leaf = [soft(0x7fd88a), soft(0x5cc79a), soft(0xa6e07a), soft(0x86d6c4)], trunk = soft(0xc79a73), rock = soft(0xe6d6c3), cloud = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 1, emissive: 0xffffff, emissiveIntensity: 0.25 });
  const ball = new THREE.SphereGeometry(1, 20, 14), cyl = new THREE.CylinderGeometry(0.12, 0.16, 1, 10), rk = new THREE.DodecahedronGeometry(1, 0);
  for (let i = 0; i < 70; i++) {
    const a = rnd(0, Math.PI * 2), r = rnd(34, 75), x = Math.cos(a) * r, z = Math.sin(a) * r, h = rnd(1.6, 3.8), t = new THREE.Group();
    const tr = new THREE.Mesh(cyl, trunk); tr.scale.set(1, h * 0.5, 1); tr.position.y = h * 0.25; t.add(tr);
    const m = leaf[i % 4]; [[0, h * 0.62, 0, h * 0.42], [h * 0.18, h * 0.82, 0.1, h * 0.3], [-h * 0.12, h * 0.9, -0.1, h * 0.26]].forEach(([dx, dy, dz, rr]) => { const b = new THREE.Mesh(ball, m); b.position.set(dx, dy, dz); b.scale.setScalar(rr); b.castShadow = true; t.add(b); });
    t.position.set(x, 0, z); t.userData.r = h * 0.45 + 0.3; decor.add(t);
  }
  for (let i = 0; i < 40; i++) { const m = new THREE.Mesh(rk, rock), a = rnd(0, 6.28), r = rnd(22, 70), sc = rnd(0.25, 0.9); m.position.set(Math.cos(a) * r, sc * 0.3, Math.sin(a) * r); m.scale.set(sc * 1.3, sc * 0.7, sc); m.rotation.y = rnd(0, 6); m.userData.r = sc * 1.3; decor.add(m); }
  for (let i = 0; i < 16; i++) { const c = new THREE.Group(), a = rnd(0, 6.28), r = rnd(20, 60); for (let k = 0; k < 5; k++) { const b = new THREE.Mesh(ball, cloud); b.position.set(k * 1.1 - 2.2, rnd(-0.3, 0.4), rnd(-0.5, 0.5)); b.scale.setScalar(rnd(0.9, 1.6)); c.add(b); } c.position.set(Math.cos(a) * r, rnd(9, 16), Math.sin(a) * r); c.userData.spin = rnd(0.002, 0.006); decor.add(c); }
}
const WORLDS = {
  playground: { bg: gradientTex('#8fcaff', '#ffe9d6'), fog: [0xffe9d6, 22, 80], a: 0xf4e7d4, b: 0xeedbc4, line: 0.03, fall: 0.3, hemi: [0xd2ebff, 0xf3d4b4, 1.5], key: [0xfff0dc, 2.7], rim: [0xffb8d8, 0.7], shadow: [0x8a5a3c, 0.28], exposure: 1.0,
    props: { bar: 0xff7a6b, box: 0x45cfa0, seat: 0xa58bff }, decor: true, ui: 'light' },
  studio: { bg: new THREE.Color(0x0b0e14), fog: [0x0b0e14, 14, 42], a: 0x0e1014, b: 0x15181e, line: 0.035, fall: 0.75, hemi: [0x9fb4d9, 0x1a1c22, 0.9], key: [0xfff1e0, 2.4], rim: [0x6fa8ff, 1.2], shadow: [0x000000, 0.45], exposure: 1.05,
    props: { bar: 0xf2c40c, box: 0xf2c40c, seat: 0xf2c40c }, decor: false, ui: 'dark' },
};
function setWorld(k) {
  const w = WORLDS[k];
  scene.background = w.bg; scene.fog.color.set(w.fog[0]); scene.fog.near = w.fog[1]; scene.fog.far = w.fog[2];
  floorMat.uniforms.uA.value.set(w.a); floorMat.uniforms.uB.value.set(w.b); floorMat.uniforms.uLine.value = w.line; floorMat.uniforms.uFall.value = w.fall;
  hemi.color.set(w.hemi[0]); hemi.groundColor.set(w.hemi[1]); hemi.intensity = w.hemi[2];
  key.color.set(w.key[0]); key.intensity = w.key[1]; rim.color.set(w.rim[0]); rim.intensity = w.rim[1];
  shadowCatcher.material.color.set(w.shadow[0]); shadowCatcher.material.opacity = w.shadow[1];
  renderer.toneMappingExposure = w.exposure; decor.visible = w.decor;
  for (const t in PROP_MAT) PROP_MAT[t].color.set(w.props[t]);
  document.body.classList.toggle('light', w.ui === 'light');
  document.querySelectorAll('[data-world]').forEach((b) => b.classList.toggle('on', b.dataset.world === k));
}

// runtime (Z-up, metres) <-> three (Y-up)
let S = 1; // world scale: KayKit's short legs walk a world scaled down to match their stride
const toThree = (x, y, z = 0) => new THREE.Vector3(x * S, z * S, -y * S);
const fromThree = (v) => [v.x / S, -v.z / S];

// ---------------------------------------------------------------- character rig
const ghostMat = new THREE.MeshStandardMaterial({ color: 0x38e070, emissive: 0x1fc35a, emissiveIntensity: 0.55, roughness: 0.6, transparent: true, opacity: 0.32, depthWrite: false });
const legRT = lib.bind_runtime[SK.names.indexOf('LeftLeg')].p[2];
const kkRig = (g, mat, sh) => { let kneeY = 0; g.scene.traverse((o) => { if (o.isBone && (o.name === 'upperleg.l' || o.name === 'upperlegl')) { o.updateWorldMatrix(true, false); kneeY = new THREE.Vector3().setFromMatrixPosition(o.matrixWorld).y; } }); const sc = kneeY / legRT; const r = makeKayRig(scene, g.scene, { SK, bindRuntime: lib.bind_runtime, scale: sc, material: mat, shadows: sh }); r.scale = sc; return r; };
const KK_LABEL = { knight: 'Knight', rogue_hooded: 'Rogue', mage: 'Mage', barbarian: 'Barbarian' };
const RIGS = {
  ...Object.fromEntries(KK.map((k, i) => { const h = kkRig(kkGltf[i], null, true), gh = kkRig(kkGltf[i], ghostMat, false); return [k, { hero: h, ghost: gh, label: KK_LABEL[k], scale: h.scale }]; })),
};
Object.values(RIGS).forEach((r) => { r.hero.visible(false); r.ghost.visible(false); });
let charH = 1.75; // camera framing follows the character's real height (three units)
let hero = RIGS.knight.hero, ghost = RIGS.knight.ghost; hero.visible(true);
function setCharacter(k) {
  const cur = Object.values(RIGS).find((r) => r.hero === hero); cur.hero.visible(false); cur.ghost.visible(false);
  hero = RIGS[k].hero; ghost = RIGS[k].ghost; hero.visible(true);
  const oldS = S; S = RIGS[k].scale || 1;
  BAR_H = RIGS[k].barH || 1.32; bodyHalfNow = RIGS[k].bodyHalf || 0.35; DIM = dimsFor(bodyHalfNow); updateHalfExt(bodyHalfNow);
  if (typeof props !== 'undefined') for (const p of props) { // re-place and re-size props for this character's world
    { scene.remove(p.g); const g = buildProp(p.type); g.userData.prop = p; g.rotation.y = p.yaw; p.g = g; scene.add(g); }
    p.g.position.copy(toThree(p.x, p.y)); p.g.scale.setScalar(S);
  }
  const oldH = charH; charH = RIGS[k].height || 1.75;
  if (typeof camera !== 'undefined') { const d = camera.position.clone().sub(controls.target); controls.target.multiplyScalar(S / oldS); camera.position.copy(controls.target).add(d.normalize().multiplyScalar(charH * 3.6)); controls.minDistance = 2.5 * charH / 1.75; controls.maxDistance = 22 * charH / 1.75; }
  document.querySelectorAll('[data-char]').forEach((b) => b.classList.toggle('on', b.dataset.char === k));
}

// ---------------------------------------------------------------- pose helpers
const yawOf = (pose) => { // facing from the leg joints, like NVIDIA's ground frame
  const r = pose[0], side = [0, 1, 2].map((i) => qrot(r.q, pose[SK.rightLeg].p)[i] - qrot(r.q, pose[SK.leftLeg].p)[i]);
  return Math.atan2(side[0], -side[1]); // cross(Z, side)
};
function place(windowPoses, x, y, yaw) { // library windows are rooted at the origin facing +X
  const q = qaxis([0, 0, 1], yaw);
  return windowPoses.map((pose) => pose.map((t, j) => (j === 0 ? { p: (() => { const r = qrot(q, t.p); return [r[0] + x, r[1] + y, t.p[2]]; })(), q: qmul(q, t.q) } : t)));
}
const pick = (() => { const c = {}; return (k) => { const a = LIB[k]; c[k] = ((c[k] ?? -1) + 1) % a.length; return a[c[k]]; }; })();

// ---------------------------------------------------------------- props
const PROP = {
  bar: { label: 'Bar', hint: 'he ducks under it' },
  box: { label: 'Box', hint: 'he vaults it' },
  seat: { label: 'Seat', hint: 'he sits on it' },
};
const PROP_MAT = { bar: new THREE.MeshStandardMaterial({ roughness: 0.55 }), box: new THREE.MeshStandardMaterial({ roughness: 0.55 }), seat: new THREE.MeshStandardMaterial({ roughness: 0.55 }) };
let yellow = PROP_MAT.bar;
let BAR_H = 1.32; // underside of the bar, sized to the current character's duck
// prop widths follow the character's measured body width (runtime metres); heights stay tied to what the motion clears
let DIM = { k: 1, barHalf: 0.85, boxHalf: 0.55, seatHalf: 0.45 };
function dimsFor(bodyHalf) { const k = Math.max(1, bodyHalf / 0.35); return { k, barHalf: Math.max(0.85, bodyHalf + 0.45), boxHalf: Math.max(0.55, bodyHalf + 0.2), seatHalf: Math.max(0.45, bodyHalf + 0.1) }; }
const box3 = (sx, sy, sz, x, y, z) => { const r = Math.min(0.06, sx / 2.2, sy / 2.2, sz / 2.2); const m = new THREE.Mesh(new RoundedBoxGeometry(sx, sz, sy, 3, r), yellow); m.position.set(x, z, -y); m.castShadow = m.receiveShadow = true; return m; };
function buildProp(type) { // prop-local runtime frame: +X is the crossing / sitting direction
  const g = new THREE.Group(); yellow = PROP_MAT[type];
  const D = DIM, t = 0.1 * D.k;
  if (type === 'bar') { const h = BAR_H + 0.13 * D.k, w = D.barHalf; g.add(box3(t, t, h, 0, -w, h / 2), box3(t, t, h, 0, w, h / 2), box3(t, 2 * w + t, 0.13 * D.k, 0, 0, BAR_H + 0.065 * D.k)); }
  if (type === 'box') { g.add(box3(0.36, 2 * D.boxHalf, 0.5, 0, 0, 0.25)); }
  if (type === 'seat') { g.add(box3(0.4, 2 * D.seatHalf, 0.4, -0.2, 0, 0.2), box3(0.06 * D.k, 2 * D.seatHalf, 0.5 * D.k, -0.43, 0, 0.4 + 0.25 * D.k)); }
  return g;
}
const props = [];
function addProp(type, x, y, yaw) {
  const g = buildProp(type); g.position.copy(toThree(x, y)); g.rotation.y = yaw; g.userData.s = S; scene.add(g);
  const p = { type, x, y, yaw, g }; g.userData.prop = p; props.push(p); clearSceneryAround(x, y);
  g.scale.setScalar(0.01); popIn.push({ g, t: 0 });
  return p;
}
const popIn = [];
const local = (p, x, y) => { const c = Math.cos(p.yaw), s = Math.sin(p.yaw); return [p.x + c * x - s * y, p.y + s * x + c * y]; };
const toLocal = (p, x, y) => { const dx = x - p.x, dy = y - p.y, c = Math.cos(p.yaw), s = Math.sin(p.yaw); return [c * dx + s * dy, -s * dx + c * dy]; };
const halfExt = { bar: [0.4, 1.25], box: [0.5, 0.95], seat: [0.7, 0.85] }; // prop footprint + body width (updated per character)
function updateHalfExt(bodyHalf = 0.35) { const m = Math.max(0.35, bodyHalf); halfExt.bar = [0.4, DIM.barHalf + 0.1 + m]; halfExt.box = [0.5, DIM.boxHalf + m]; halfExt.seat = [0.7, DIM.seatHalf + m]; }
const extOf = (p) => (p.r ? [p.r, p.r] : halfExt[p.type]);
// scenery the character must walk around (trees, rocks), in runtime coordinates for the current world scale
function sceneryBlockers() {
  if (!decor.visible) return [];
  return decor.children.filter((c) => c.userData.r && c.visible).map((c) => ({ type: 'decor', x: c.position.x / S, y: -c.position.z / S, yaw: 0, r: c.userData.r / S + Math.max(0.35, bodyHalfNow) }));
}
let bodyHalfNow = 0.35;
// a prop course placed among the scenery tidies the scenery out of its way (restored on Clear)
function clearSceneryAround(x, y, pad = 3.5) {
  for (const c of decor.children) { if (!c.userData.r || !c.visible) continue;
    if (Math.hypot(c.position.x / S - x, -c.position.z / S - y) < c.userData.r / S + pad) { c.visible = false; c.userData.cleared = true; } }
}
function segHitsProp(p, ax, ay, bx, by) { // segment vs prop footprint (expanded), in prop-local space
  const [x0, y0] = toLocal(p, ax, ay), [x1, y1] = toLocal(p, bx, by), [hx, hy] = extOf(p);
  let t0 = 0, t1 = 1; const dx = x1 - x0, dy = y1 - y0;
  for (const [pp, qq] of [[-dx, x0 + hx], [dx, hx - x0], [-dy, y0 + hy], [dy, hy - y0]]) {
    if (pp === 0) { if (qq < 0) return null; continue; }
    const r = qq / pp; if (pp < 0) { if (r > t1) return null; if (r > t0) t0 = r; } else { if (r < t0) return null; if (r < t1) t1 = r; }
  }
  return t0;
}

// ---------------------------------------------------------------- the motion track
// track[i] = { local: pose (for context), world: joint world transforms (for rendering) }
const track = [];
let playhead = 0;          // in frames, fractional
let seated = false;
// Posture layer: NVIDIA's sample walk carries a forward-head, rounded-back posture. Lift chest and neck a little
// for display only; the model keeps receiving the original poses as context.
const KIDS = SK.names.map(() => []); SK.parents.forEach((p, j) => { if (p >= 0) KIDS[p].push(j); });
const subtree = (j) => { const out = [j]; for (let i = 0; i < out.length; i++) out.push(...KIDS[out[i]]); return out; };
const POSTURE = [['Chest', 6], ['Neck1', 8], ['Head', -3]].map(([n, d]) => ({ j: SK.names.indexOf(n), rad: (d * Math.PI) / 180, sub: subtree(SK.names.indexOf(n)) }));
let postureOn = true;
function postured(world) {
  if (!postureOn) return world;
  const w = world.map((t) => ({ p: [...t.p], q: [...t.q] }));
  const r = w[0], side = [0, 1, 2].map((i) => w[SK.rightLeg].p[i] - w[SK.leftLeg].p[i]); side[2] = 0;
  const L = Math.hypot(side[0], side[1]) || 1, axis = [side[0] / L, side[1] / L, 0];
  for (const { j, rad, sub } of POSTURE) {
    const k = sittingNowWorld(w) ? 0.5 : 1, dq = qaxis(axis, rad * k), pv = w[j].p;
    for (const c of sub) {
      if (c !== j) { const d = [w[c].p[0] - pv[0], w[c].p[1] - pv[1], w[c].p[2] - pv[2]], rd = qrot(dq, d); w[c].p = [pv[0] + rd[0], pv[1] + rd[1], pv[2] + rd[2]]; }
      w[c].q = qmul(dq, w[c].q);
    }
  }
  return w;
}
const sittingNowWorld = (w) => w[0].p[2] < 0.7;
const push = (pose) => track.push({ local: pose, world: postured(SK.world(pose)) });
const idle0 = place(LIB.idle[0], 0, 0, 0);
for (let i = 0; i < 4; i++) push(idle0[i]);

const queue = [];           // planned targets: { poses, label, kind }
let job = null, jobId = 0;
const stats = { gens: 0, ms: [], seconds: 0 };

function plannerStart() { // where the character will be once everything planned so far has played
  if (queue.length) { const q = queue[queue.length - 1]; return { x: q.x, y: q.y, yaw: q.yaw, seated: q.seated }; }
  const last = track[track.length - 1].local;
  return { x: last[0].p[0], y: last[0].p[1], yaw: yawOf(last), seated };
}
const target = (kind, x, y, yaw, label, extra = {}) => ({ kind, x, y, yaw, label, seated: kind === 'sit', poses: place(pick(kind), x, y, yaw), ...extra });

// measured root speed of each gait in the pose library (m/s)
const SPEED = {}; for (const k of ['walk', 'jog', 'run']) { const v = LIB[k].map((w) => Math.hypot(w[3][0].p[0] - w[0][0].p[0], w[3][0].p[1] - w[0][0].p[1]) * 10); SPEED[k] = v.reduce((a, b) => a + b, 0) / v.length; }
let PACE = 'walk', runMode = false;
const gaitFor = (d) => (d < 1.4 ? 'walk' : PACE);
function planLocomotion(out, from, gx, gy, finalYaw, endIdle = true) {
  let { x, y } = from; const d = Math.hypot(gx - x, gy - y);
  if (d < 0.25) { if (endIdle) out.push(target('idle', gx, gy, finalYaw ?? from.yaw, 'arrive')); return; }
  const yaw = Math.atan2(gy - y, gx - x), kind = gaitFor(d), step = SPEED[kind] * 1.2; // ~1.2 s of motion per generation
  let remain = d;
  while (remain > step * 1.25) { x += Math.cos(yaw) * step; y += Math.sin(yaw) * step; remain -= step; out.push(target(kind, x, y, yaw, kind)); }
  if (endIdle) out.push(target('idle', gx, gy, finalYaw ?? yaw, 'arrive'));
  else if (remain > 0.4) out.push(target(gaitFor(remain), gx, gy, yaw, gaitFor(remain)));
}
// walk/jog from -> to, going around any prop in the way (except the one we are about to use)
function route(out, from, gx, gy, finalYaw, endIdle, skip = null, depth = 0) {
  const hit = depth < 4 && [...props, ...sceneryBlockers()].filter((p) => p !== skip).map((p) => ({ p, t: segHitsProp(p, from.x, from.y, gx, gy) }))
    .filter((h) => h.t !== null && h.t > 0.0).sort((a, b) => a.t - b.t)[0];
  if (!hit) { planLocomotion(out, from, gx, gy, finalYaw, endIdle); return; }
  const p = hit.p, [hx, hy] = extOf(p), m = 0.35;
  const [fx, fy] = toLocal(p, from.x, from.y), [tx, ty] = toLocal(p, gx, gy);
  let best = null;
  for (const side of [1, -1]) { // go round one side: corner nearest the start, then nearest the goal
    const cy = side * (hy + m), c1 = [Math.sign(fx || -1) * (hx + m), cy], c2 = [Math.sign(tx || 1) * (hx + m), cy];
    const pts = c1[0] === c2[0] ? [c1] : [c1, c2];
    let len = 0, prev = [fx, fy]; for (const c of [...pts, [tx, ty]]) { len += Math.hypot(c[0] - prev[0], c[1] - prev[1]); prev = c; }
    if (!best || len < best.len) best = { len, pts };
  }
  let cur = from;
  for (const c of best.pts) { const [wx, wy] = local(p, c[0], c[1]); route(out, cur, wx, wy, null, false, skip, depth + 1); cur = { x: wx, y: wy, yaw: Math.atan2(wy - cur.y, wx - cur.x) }; }
  route(out, cur, gx, gy, finalYaw, endIdle, skip, depth + 1);
}
function planInteraction(out, from, p, travelYaw) {
  const nYaw = p.yaw, dot = Math.cos(travelYaw - nYaw); const dir = p.type === 'seat' ? nYaw : (dot >= 0 ? nYaw : nYaw + Math.PI);
  const at = (dist) => [p.x + Math.cos(dir) * dist, p.y + Math.sin(dir) * dist];
  if (p.type === 'bar') {
    const [ax, ay] = at(-1.75); route(out, from, ax, ay, dir, false, p);
    const g = PACE === 'walk' ? 'walk' : 'jog', out1 = PACE === 'walk' ? 1.6 : 2.2; // rise straight back into the pace
    const fast = { fast: PACE !== 'walk' }; // at speed, a quick duck instead of a crouch-walk
    out.push(target('duck', ...at(-0.7), dir, 'duck under', fast), target('duck', ...at(0.15), dir, 'stay low', fast), target(g, ...at(out1), dir, g));
    return { x: at(out1)[0], y: at(out1)[1], yaw: dir };
  }
  if (p.type === 'box') {
    const keep = PACE; if (PACE === 'walk') PACE = 'jog'; const [ax, ay] = at(-2.25); route(out, from, ax, ay, dir, false, p); PACE = keep;
    if (PACE === 'walk') { out.push(target('vault_apex', ...at(-0.25), dir, 'vault'), target('vault_land', ...at(1.45), dir, 'land')); return { x: at(1.45)[0], y: at(1.45)[1], yaw: dir }; }
    const g = PACE === 'run' ? 'run' : 'jog', d2 = PACE === 'run' ? 3.6 : 2.6; // land running, no stop
    out.push(target('vault_apex', ...at(-0.25), dir, 'vault'), target(g, ...at(d2), dir, g));
    return { x: at(d2)[0], y: at(d2)[1], yaw: dir };
  }
  if (p.type === 'seat') {
    const [fx, fy] = local(p, 1.2, 0); route(out, from, fx, fy, null, false, p);
    const [sx, sy] = local(p, 0.02, 0);
    out.push(target('sit', sx, sy, p.yaw, 'sit'), target('sit', sx, sy, p.yaw, 'sit', { hold: true }));
    return { x: sx, y: sy, yaw: p.yaw, seated: true };
  }
}
function planFrom(goal, final) { return plan(goal, final); }
function plan(goal, final = true) { // goal: {x,y} or {prop}
  const out = []; let from = plannerStart();
  const gx = goal.prop ? goal.prop.x : goal.x, gy = goal.prop ? goal.prop.y : goal.y, dist = Math.hypot(gx - from.x, gy - from.y);
  PACE = goal.pace || (runMode ? 'run' : dist > 9 ? 'run' : dist > 4.5 ? 'jog' : 'walk');
  if (from.seated) {
    const sp = props.find((p) => p.type === 'seat' && Math.hypot(...toLocal(p, from.x, from.y)) < 0.6);
    const [stx, sty] = sp ? local(sp, 0.3, 0) : [from.x + Math.cos(from.yaw) * 0.3, from.y + Math.sin(from.yaw) * 0.3];
    out.push(target('stand', stx, sty, from.yaw, 'stand up'));
    from = { x: stx, y: sty, yaw: from.yaw, seated: false };
  }
  if (goal.prop) { planInteraction(out, from, goal.prop, Math.atan2(goal.prop.y - from.y, goal.prop.x - from.x)); return out; }
  // smart path: deal with the first prop in the way, then carry on
  for (let guard = 0; guard < 4; guard++) {
    const hits = props.map((p) => ({ p, t: segHitsProp(p, from.x, from.y, goal.x, goal.y) })).filter((h) => h.t !== null && h.t > 0.02).sort((a, b) => a.t - b.t);
    if (!hits.length) break;
    const p = hits[0].p, travel = Math.atan2(goal.y - from.y, goal.x - from.x);
    if (p.type === 'seat') { // not in the mood to sit: walk around it to just past it
      const [tx, ty] = toLocal(p, goal.x, goal.y), [hx] = halfExt.seat;
      const [ex, ey] = local(p, Math.sign(tx || 1) * (hx + 0.4), Math.max(-0.2, Math.min(0.2, ty)));
      const exitPt = segHitsProp(p, ex, ey, goal.x, goal.y) === null ? [ex, ey] : [goal.x, goal.y];
      route(out, from, exitPt[0], exitPt[1], null, false); from = { x: exitPt[0], y: exitPt[1], yaw: travel };
      if (exitPt[0] === goal.x && exitPt[1] === goal.y) break;
    } else from = planInteraction(out, from, p, travel);
  }
  route(out, from, goal.x, goal.y, null, final);
  return out;
}

// interrupt: keep what is about to play, drop the rest, re-plan from there
let goals = [], loopGoals = null;  // pending goals after the current one; optional patrol loop
let currentGoal = null;
function command(goal, rest = [], loop = null) { goals = rest; loopGoals = loop; currentGoal = goal; return commandNow(goal); }
function replanIfMoving() { if (currentGoal && (queue.length || job)) commandNow(currentGoal, true); }
function commandNow(goal, quiet = false) {
  const lead = Math.ceil((avgMs() / 1000) * 30 * 1.6) + 6;
  const cut = Math.max(3, Math.floor(playhead) + lead);
  if (cut < track.length - 1) track.length = cut + 1;
  queue.length = 0; if (job) job.stale = true;
  seated = false;
  const last = track[track.length - 1].local;
  seated = sittingNow(last);
  const steps = plan(goal, !goals.length && !loopGoals); queue.push(...steps);
  goals.forEach((g, i) => queue.push(...planFrom(g, i === goals.length - 1 && !loopGoals)));
  if (!quiet) toast(describe(steps));

  pump();
}
const sittingNow = (pose) => pose[0].p[2] < 0.7;
const describe = (steps) => { const l = [...new Set(steps.map((s) => s.label).filter((x) => !['walk', 'jog', 'arrive', 'stay low', 'land', 'walk out'].includes(x)))]; return l.length ? l.join(' → ') : (steps.some((s) => s.kind === 'jog') ? 'jog there' : 'walk there'); };
const SHOW_BEATS = new Set(['duck under', 'vault', 'sit', 'stand up']);
const avgMs = () => (stats.ms.length ? stats.ms.slice(-6).reduce((a, b) => a + b, 0) / Math.min(6, stats.ms.length) : 300);

// ---------------------------------------------------------------- generation (worker)
const worker = new Worker(new URL('./gen-worker.js', import.meta.url), { type: 'module' });
let ready = false;
worker.onmessage = (e) => {
  const m = e.data;
  if (m.type === 'progress') loading(m.loaded / m.total, m.cached ? 'Loaded from this browser’s cache' : `Downloading NVIDIA MotionBricks · ${(m.loaded / 1e6).toFixed(0)} / ${(m.total / 1e6).toFixed(0)} MB`);
  else if (m.type === 'status') loading(1, m.text);
  else if (m.type === 'ready') { ready = true; $('backend').textContent = `WebAssembly · ${m.threads} thread${m.threads > 1 ? 's' : ''}`; doneLoading(); pump(); }
  else if (m.type === 'result') onResult(m);
  else if (m.type === 'error') { console.error(m.text); toast('Generation error: ' + m.text); job = null; }
};
worker.postMessage({ type: 'init', parts: manifest.parts.map((p) => '../model/' + p), total: manifest.bytes, packed: manifest.packed, key: manifest.sha256.slice(0, 16) });

function pump() {
  if (!ready || job) return;
  const ahead = track.length - 1 - playhead;
  if (ahead > 40) return;
  if (!queue.length && loopGoals && loopGoals.length) {
    const g = loopGoals.shift(); loopGoals.push(g); currentGoal = g; queue.push(...plan(g, false));
  }
  if (!queue.length) {
    if (ahead > 8) return;
    currentGoal = null;
    const s = plannerStart(); // keep him alive: idle (or keep sitting) in place
    queue.push(seated || sittingNow(track[track.length - 1].local) ? target('sit', s.x, s.y, s.yaw, 'sit') : target('idle', s.x, s.y, s.yaw, 'idle'));
  }
  const tg = queue.shift();
  window.__mb?.log.push({ label: tg.label, kind: tg.kind, x: +tg.x.toFixed(2), y: +tg.y.toFixed(2), at: track.length });
  const ctx = track.slice(-4).map((f) => f.local);
  const nT = tg.kind === 'idle' || tg.hold ? 1 : 4;
  const targets = tg.poses.slice(0, nT);
  let allowedTokens;
  if (SPEED[tg.kind]) { // locomotion: keep pace instead of letting the model linger
    const a = ctx[3][0].p, b = targets[0][0].p, secs = Math.hypot(b[0] - a[0], b[1] - a[1]) / SPEED[tg.kind];
    const n = Math.round((secs * 30 + 6) / 4); allowedTokens = [n - 1, n, n + 1].filter((t) => t >= 6 && t <= 16);
    if (!allowedTokens.length) allowedTokens = [n < 6 ? 6 : 16];
  }
  if (tg.fast) allowedTokens = [6, 7, 8];
  const { feeds, align } = encode(SK, ctx, targets, M2R, allowedTokens ? { allowedTokens } : {});
  job = { id: ++jobId, tg, ctx, targets, align, stale: false, base: track.length };
  worker.postMessage({ type: 'run', id: job.id, feeds });
  showGhost(tg);
}
function onResult(m) {
  const j = job; job = null;
  if (!j || j.id !== m.id) return pump();
  stats.ms.push(m.ms); stats.gens++;
  if (!j.stale && track.length === j.base) {
    const frames = decode(SK, m.pred, m.tokens, j.align, M2R, j.ctx, j.targets, m.dims);
    const start = track.length;
    for (let i = 1; i < frames.length; i++) { push(frames[i]); track[track.length - 1].kind = j.tg.kind; } // frame 0 equals the last context pose
    if (SHOW_BEATS.has(j.tg.label)) track[start].beat = j.tg.label;
    stats.seconds += (frames.length - 1) / 30;
    seated = j.tg.kind === 'sit';
  }
  hud(m.ms);
  pump();
}

// ---------------------------------------------------------------- ghost of the current target
let ghostFade = 0;
function showGhost(tg) {
  if (!showTargets || tg.label === 'idle' || (tg.kind === 'sit' && tg.hold)) return;
  ghost.set(postured(SK.world(tg.poses[0]))); ghost.visible(true); ghostFade = 1;
}
let showTargets = true;

// ---------------------------------------------------------------- input
const ray = new THREE.Raycaster(), ndc = new THREE.Vector2();
let tool = 'move', previewYaw = null, hoverProp = null;
const preview = { g: null, type: null };
const marker = new THREE.Mesh(new THREE.RingGeometry(0.22, 0.3, 48), new THREE.MeshBasicMaterial({ color: 0xf2c40c, transparent: true, opacity: 0, depthWrite: false }));
marker.rotation.x = -Math.PI / 2; marker.position.y = 0.01; scene.add(marker);
const pathLine = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]), new THREE.LineDashedMaterial({ color: 0xffffff, dashSize: 0.18, gapSize: 0.12, transparent: true, opacity: 0.35 }));
scene.add(pathLine);

function floorHit(ev) {
  const r = renderer.domElement.getBoundingClientRect();
  ndc.set(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1);
  ray.setFromCamera(ndc, camera);
  const ph = ray.intersectObjects(props.map((p) => p.g), true)[0];
  hoverProp = ph ? (() => { let o = ph.object; while (o && !o.userData.prop) o = o.parent; return o?.userData.prop || null; })() : null;
  const hit = ray.intersectObject(floor)[0];
  return hit ? hit.point : null;
}
function heroXY() { const w = currentWorld(); return [w[0].p[0], w[0].p[1]]; }
let downAt = null;
renderer.domElement.addEventListener('pointerdown', (e) => { downAt = [e.clientX, e.clientY]; });
renderer.domElement.addEventListener('pointermove', (e) => {
  const pt = floorHit(e); renderer.domElement.style.cursor = hoverProp && tool === 'move' ? 'pointer' : 'crosshair';
  if (!pt) return;
  const [x, y] = fromThree(pt);
  if (tool !== 'move') {
    if (preview.type !== tool) { if (preview.g) scene.remove(preview.g); preview.g = buildProp(tool); preview.g.traverse((o) => { if (o.isMesh) { o.material = o.material.clone(); o.material.transparent = true; o.material.opacity = 0.45; o.castShadow = false; } }); scene.add(preview.g); preview.type = tool; }
    const [hx, hy] = heroXY(); const yaw = previewYaw ?? (tool === 'seat' ? Math.atan2(hy - y, hx - x) : Math.atan2(y - hy, x - hx));
    preview.g.position.copy(toThree(x, y)); preview.g.scale.setScalar(S); preview.g.rotation.y = yaw; preview.yaw = yaw; preview.x = x; preview.y = y;
    pathLine.visible = false;
  } else {
    const [hx, hy] = heroXY(); const target3 = hoverProp ? toThree(hoverProp.x, hoverProp.y, 0.02) : toThree(x, y, 0.02);
    pathLine.geometry.setFromPoints([toThree(hx, hy, 0.02), target3]); pathLine.computeLineDistances(); pathLine.visible = true;
    const hits = !hoverProp && props.some((p) => segHitsProp(p, hx, hy, x, y) !== null);
    pathLine.material.color.set(hoverProp || hits ? 0xf2c40c : 0xffffff); pathLine.material.opacity = hoverProp || hits ? 0.8 : 0.35;
    $('cursorHint').textContent = hoverProp ? `click: ${PROP[hoverProp.type].hint}` : hits ? 'something in the way, he’ll deal with it' : '';
  }
});
renderer.domElement.addEventListener('pointerup', (e) => {
  if (!downAt || Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]) > 6) return; // that was an orbit drag
  const pt = floorHit(e); if (!pt) return;
  const [x, y] = fromThree(pt);
  if (tool !== 'move' && preview.g) {
    addProp(tool, preview.x, preview.y, preview.yaw); previewYaw = null; replanIfMoving();
    toast(`${PROP[tool].label} placed: ${PROP[tool].hint}`);
    setTool('move'); return;
  }
  if (hoverProp) { command({ prop: hoverProp }); pulse(hoverProp.x, hoverProp.y); return; }
  command({ x, y }); pulse(x, y);
});
addEventListener('wheel', (e) => { if (tool !== 'move' && preview.g) { previewYaw = (preview.yaw ?? 0) + Math.sign(e.deltaY) * 0.2; e.preventDefault(); } }, { passive: false });
addEventListener('keydown', (e) => {
  if (e.key === 'Escape') setTool('move');
  if (e.key.toLowerCase() === 'r' && tool !== 'move') previewYaw = (preview.yaw ?? 0) + Math.PI / 12;
  if (e.key === '1') setTool('bar'); if (e.key === '2') setTool('box'); if (e.key === '3') setTool('seat');
  if (e.key.toLowerCase() === 'g') toggleTargets();
});
function setTool(t) {
  tool = t; document.querySelectorAll('[data-tool]').forEach((b) => b.classList.toggle('on', b.dataset.tool === t));
  if (t === 'move' && preview.g) { scene.remove(preview.g); preview.g = null; preview.type = null; }
  $('cursorHint').textContent = t === 'move' ? '' : `click to place the ${PROP[t].label.toLowerCase()} · scroll or R to rotate · Esc to cancel`;
}
document.querySelectorAll('[data-tool]').forEach((b) => b.addEventListener('click', () => setTool(b.dataset.tool === tool ? 'move' : b.dataset.tool)));
function clearProps() { props.splice(0).forEach((p) => scene.remove(p.g)); decor.children.forEach((c) => { if (c.userData.cleared) { c.visible = true; c.userData.cleared = false; } }); }
$('clear').addEventListener('click', () => { clearProps(); loopGoals = null; replanIfMoving(); toast('Props cleared'); });
function toggleTargets() { showTargets = !showTargets; $('targets').classList.toggle('on', showTargets); if (!showTargets) ghost.visible(false); }
$('targets').addEventListener('click', toggleTargets);
// ---------------------------------------------------------------- patterns
function frame() { // hero position + facing, and a helper to place things relative to him
  const [hx, hy] = heroXY(), yaw = yawOf(track[track.length - 1].local), c = Math.cos(yaw), s = Math.sin(yaw);
  return { yaw, at: (f, l = 0) => [hx + c * f - s * l, hy + s * f + c * l] };
}
const PATTERNS = {
  hurdles: { name: 'Hurdles', desc: 'Four boxes in a row. He vaults every one.', build({ at, yaw }) {
    [5, 11, 17, 23].forEach((d) => addProp('box', ...at(d), yaw)); const [x, y] = at(28); command({ x, y, pace: 'run' }); } },
  limbo: { name: 'Limbo line', desc: 'Three low bars. Duck, rise, duck again.', build({ at, yaw }) {
    [3.5, 7, 10.5].forEach((d) => addProp('bar', ...at(d), yaw)); const [x, y] = at(14); command({ x, y }); } },
  parkour: { name: 'Parkour run', desc: 'Bar, box, bar, box at running pace.', build({ at, yaw }) {
    addProp('bar', ...at(4), yaw); addProp('box', ...at(10), yaw); addProp('bar', ...at(15), yaw); addProp('box', ...at(21), yaw);
    const [x, y] = at(23); command({ x, y, pace: 'run' }); } },
  weave: { name: 'Weave', desc: 'Seats staggered in his path. He threads between them.', build({ at, yaw }) {
    [[3, 0.55], [6, -0.55], [9, 0.55], [12, -0.55]].forEach(([d, l]) => addProp('seat', ...at(d, l), yaw + Math.PI / 2)); const [x, y] = at(15); command({ x, y }); } },
  waiting: { name: 'Waiting room', desc: 'Three seats. He tries each one, then leaves.', build({ at, yaw }) {
    const seats = [-2.2, 0, 2.2].map((l) => addProp('seat', ...at(4.5, l), yaw + Math.PI));
    const [x, y] = at(0, 0); command({ prop: seats[0] }, [{ prop: seats[2] }, { prop: seats[1] }, { x, y }]); } },
  patrol: { name: 'Patrol loop', desc: 'A square beat with a bar and a box. Runs forever.', build({ at, yaw }) {
    const c = [at(0, 0), at(6, 0), at(6, 6), at(0, 6)];
    addProp('bar', ...at(3, 0), yaw); addProp('box', ...at(6, 3), yaw + Math.PI / 2);
    const loop = [c[1], c[2], c[3], c[0]].map(([x, y]) => ({ x, y, pace: 'jog' })); command(loop[0], [], loop.slice(1).concat([loop[0]])); } },
  chaos: { name: 'Chaos', desc: 'Random props, random destinations. Autopilot.', build({ at }) {
    const rnd = (a, b) => a + Math.random() * (b - a), types = ['bar', 'box', 'bar', 'box', 'seat'];
    for (let i = 0; i < 7; i++) { const [x, y] = at(rnd(2.5, 13), rnd(-6, 6)); addProp(types[i % types.length], x, y, rnd(0, Math.PI * 2)); }
    const pts = Array.from({ length: 6 }, () => { const [x, y] = at(rnd(-2, 14), rnd(-7, 7)); return { x, y }; });
    command(pts[0], [], pts.slice(1).concat([pts[0]])); } },
};
const pop = $('patterns');
pop.innerHTML = Object.entries(PATTERNS).map(([k, p]) => `<button data-pattern="${k}"><b>${p.name}</b><span>${p.desc}</span></button>`).join('');
$('patternsBtn').addEventListener('click', (e) => { e.stopPropagation(); pop.classList.toggle('open'); });
addEventListener('click', (e) => { if (!pop.contains(e.target)) pop.classList.remove('open'); });
pop.querySelectorAll('[data-pattern]').forEach((b) => b.addEventListener('click', () => {
  pop.classList.remove('open'); clearProps(); const P = PATTERNS[b.dataset.pattern]; P.build(frame()); toast(`${P.name}: ${P.desc}`);
}));
$('run').addEventListener('click', () => { runMode = !runMode; $('run').classList.toggle('on', runMode); toast(runMode ? 'Run mode: he sprints everywhere' : 'Auto pace: walk short, jog mid, run far'); if (currentGoal) commandNow(currentGoal, true); });
$('posture').addEventListener('click', () => { postureOn = !postureOn; $('posture').classList.toggle('on', postureOn); toast(postureOn ? 'Posture layer on: chest and head lifted a little (display only)' : 'Posture layer off: raw model output'); });

// ---------------------------------------------------------------- HUD + feedback
function hud(ms) {
  $('lastMs').textContent = `${Math.round(ms)} ms`;
  $('avgMs').textContent = `${Math.round(avgMs())} ms`;
  $('gens').textContent = stats.gens;
  $('secs').textContent = `${stats.seconds.toFixed(1)} s`;
  const bar = $('spark'); const v = stats.ms.slice(-28), mx = Math.max(400, ...v);
  bar.innerHTML = v.map((x) => `<i style="height:${Math.max(3, (x / mx) * 26)}px"></i>`).join('');
}
let toastT = 0;
function toast(t) { const el = $('toast'); el.textContent = t; el.classList.add('show'); clearTimeout(toastT); toastT = setTimeout(() => el.classList.remove('show'), 2400); }
function flash(label) { const el = $('beat'); el.textContent = label; el.classList.remove('pop'); void el.offsetWidth; el.classList.add('pop'); }
let pulseT = 0; function pulse(x, y) { marker.position.copy(toThree(x, y, 0.012)); pulseT = 1; }
function loading(f, text) { $('loadbar').style.width = `${(f * 100).toFixed(1)}%`; $('loadtext').textContent = text; }
function doneLoading() { $('loader').classList.add('done'); setTimeout(() => toast('Click anywhere on the floor'), 600); }

// ---------------------------------------------------------------- render loop
function currentWorld() {
  const i = Math.min(Math.floor(playhead), track.length - 1), k = Math.min(i + 1, track.length - 1), a = playhead - i;
  const A = track[i].world, B = track[k].world;
  if (a < 1e-3 || i === k) return A;
  return A.map((t, j) => ({ p: [t.p[0] + (B[j].p[0] - t.p[0]) * a, t.p[1] + (B[j].p[1] - t.p[1]) * a, t.p[2] + (B[j].p[2] - t.p[2]) * a], q: qslerp(t.q, B[j].q, a) }));
}
let clockT = 0; const mood = { surpriseUntil: 0, happyUntil: 0, idleFor: 0, blinkAt: 2, lastKind: '' };
const clock = new THREE.Clock(); const focus = new THREE.Vector3(0, 0.9, 0);
renderer.setAnimationLoop(() => {
  const dt = Math.min(clock.getDelta(), 0.1);
  const before = Math.floor(playhead);
  if (playhead < track.length - 1) playhead = Math.min(playhead + dt * 30, track.length - 1);
  for (let f = before + 1; f <= Math.floor(playhead); f++) if (track[f]?.beat) { flash(track[f].beat); track[f].beat = null; }
  // trim the far past to keep memory flat (keep context frames)
  if (playhead > 600) { const drop = Math.floor(playhead) - 300; track.splice(0, drop); playhead -= drop; if (job) job.base -= drop; }
  pump();
  clockT += dt;
  const w = currentWorld(); hero.set(w);
  const hp = toThree(w[0].p[0], w[0].p[1], 0); hp.y = charH * 0.52;
  focus.lerp(hp, 1 - Math.exp(-dt * 3)); const delta = focus.clone().sub(controls.target);
  controls.target.add(delta); camera.position.add(delta); controls.update();
  key.position.set(focus.x - 4, 9, focus.z + 5); key.target.position.copy(focus);
  floorMat.uniforms.uFocus.value.set(focus.x, focus.z);
  if (ghostFade > 0) { ghostFade = Math.max(0, ghostFade - dt * 0.35); ghostMat.opacity = 0.34 * Math.min(1, ghostFade * 2); if (ghostFade === 0) ghost.visible(false); }
  if (pulseT > 0) { pulseT = Math.max(0, pulseT - dt * 1.4); marker.material.opacity = pulseT; marker.scale.setScalar(1 + (1 - pulseT) * 1.6); }
  for (let i = popIn.length - 1; i >= 0; i--) { const p = popIn[i]; p.t = Math.min(1, p.t + dt * 4); const e = 1 - Math.pow(1 - p.t, 3) * Math.cos(p.t * 5); p.g.scale.setScalar(Math.max(0.01, e) * S); if (p.t >= 1) { p.g.scale.setScalar(S); popIn.splice(i, 1); } }
  decor.children.forEach((c) => { if (c.userData.spin) c.rotation.y += c.userData.spin * dt * 0; c.position.x += c.userData.spin ? dt * 0.25 : 0; if (c.position.x > 80) c.position.x = -80; });
  $('ahead').textContent = `${Math.max(0, (track.length - 1 - playhead) / 30).toFixed(1)} s`;
  renderer.render(scene, camera);
});
addEventListener('resize', () => { camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix(); renderer.setSize(innerWidth, innerHeight); });
setTool('move');
window.__mb = { command, sceneryBlockers, track: () => track, props, toLocal, camera, controls, heroXY, yawOf, get playhead() { return playhead; }, log: [] };
for (const k of KK) { const r = RIGS[k]; S = r.scale; r.hero.visible(true); r.hero.set(postured(SK.world(LIB.duck[0][0]))); r.barH = Math.max(0.9, r.hero.topY() / S + 0.07); r.hero.set(postured(SK.world(LIB.idle[0][0]))); r.height = r.hero.topY(); r.bodyHalf = r.hero.halfWidth() / S; r.hero.visible(false); }
S = 1;
setWorld('playground'); setCharacter('knight');
document.querySelectorAll('[data-world]').forEach((b) => b.addEventListener('click', () => setWorld(b.dataset.world)));
document.querySelectorAll('[data-char]').forEach((b) => b.addEventListener('click', () => setCharacter(b.dataset.char)));
