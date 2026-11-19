// Retarget NVIDIA SOMA motion (runtime space, Z-up metres) onto a KayKit humanoid rig in three.js.
// Rotations transfer as world-space deltas from SOMA's bind pose, corrected for the different rest pose
// (KayKit T-pose vs SOMA A-pose). KayKit keeps its own bone lengths; the hips follow SOMA's leg joints.
import * as THREE from 'three';
import * as SkeletonUtils from 'three/addons/utils/SkeletonUtils.js';

const MAP = { // kaykit bone -> SOMA joint, [SOMA child used for the bone direction, kaykit child]
  hips: ['Hips'], spine: ['Spine1'], chest: ['Chest'], head: ['Head'],
  'upperarm.l': ['LeftArm', 'LeftForeArm', 'lowerarm.l'], 'lowerarm.l': ['LeftForeArm', 'LeftHand', 'wrist.l'], 'wrist.l': ['LeftHand', 'LeftHandMiddleEnd', 'hand.l'],
  'upperarm.r': ['RightArm', 'RightForeArm', 'lowerarm.r'], 'lowerarm.r': ['RightForeArm', 'RightHand', 'wrist.r'], 'wrist.r': ['RightHand', 'RightHandMiddleEnd', 'hand.r'],
  'upperleg.l': ['LeftLeg', 'LeftShin', 'lowerleg.l'], 'lowerleg.l': ['LeftShin', 'LeftFoot', 'foot.l'], 'foot.l': ['LeftFoot', 'LeftToeBase', 'toes.l'],
  'upperleg.r': ['RightLeg', 'RightShin', 'lowerleg.r'], 'lowerleg.r': ['RightShin', 'RightFoot', 'foot.r'], 'foot.r': ['RightFoot', 'RightToeBase', 'toes.r'],
  'toes.l': ['LeftToeBase'], 'toes.r': ['RightToeBase'],
};
const KEEP = /helmet|hat|cape|hood/i; // accessories to keep (weapons and shields are hidden)

export function makeKayRig(THREEscene, srcScene, { SK, bindRuntime, scale, material = null, shadows = true }) {
  const root = SkeletonUtils.clone(srcScene);
  THREEscene.add(root); root.updateMatrixWorld(true);
  let skel = null; const meshes = [];
  root.traverse((o) => {
    if (o.isSkinnedMesh) { skel = skel || o.skeleton; meshes.push(o); o.frustumCulled = false; o.castShadow = shadows; if (material) o.material = material; }
    else if (o.isMesh) { if (!KEEP.test(o.name)) o.visible = false; else { meshes.push(o); o.castShadow = shadows; if (material) o.material = material; } }
  });
  // three.js strips '.' from node names on load (upperleg.l -> upperlegl): look bones up the same way
  const nm = (n) => n.replace(/[\[\]\.:\/]/g, '');
  const bones = skel.bones, byName = Object.fromEntries(bones.map((b) => [b.name, b])), bone = (n) => byName[nm(n)];
  // parent-first order
  const order = []; const seen = new Set();
  const visit = (b) => { if (seen.has(b)) return; if (b.parent && byName[b.parent.name] === b.parent) visit(b.parent); seen.add(b); order.push(b); };
  bones.forEach(visit);
  const bindW = new Map(), bindL = new Map();
  bones.forEach((b) => { bindW.set(b, b.matrixWorld.clone()); bindL.set(b, b.matrix.clone()); });
  const pos = (m) => new THREE.Vector3().setFromMatrixPosition(m), rot = (m) => new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().extractRotation(m));
  // runtime -> three
  const qR = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), -Math.PI / 2);
  const rtPos = (p) => new THREE.Vector3(p[0], p[2], -p[1]);
  const rtQuat = (q) => qR.clone().multiply(new THREE.Quaternion(q[0], q[1], q[2], q[3]));
  const sIdx = (n) => SK.names.indexOf(n);
  const somaBindQinv = {}, align = {};
  for (const [kb, [sj, sChild, kChild]] of Object.entries(MAP)) {
    if (!bone(kb)) continue;
    somaBindQinv[kb] = rtQuat(bindRuntime[sIdx(sj)].q).invert();
    align[kb] = new THREE.Quaternion();
    if (sChild && kChild && bone(kChild)) {
      const dS = rtPos(bindRuntime[sIdx(sChild)].p).sub(rtPos(bindRuntime[sIdx(sj)].p)).normalize();
      const dK = pos(bindW.get(bone(kChild))).sub(pos(bindW.get(bone(kb)))).normalize();
      align[kb].setFromUnitVectors(dK, dS);
    }
  }
  const hips = bone('hips'), legMidK = pos(bindW.get(bone('upperleg.l'))).add(pos(bindW.get(bone('upperleg.r')))).multiplyScalar(0.5);
  const hipsOffset = pos(bindW.get(hips)).sub(legMidK); // hips relative to the leg joints, bind (world)
  const LL = sIdx('LeftLeg'), RL = sIdx('RightLeg');
  const KEYOF = Object.fromEntries(Object.keys(MAP).map((k) => [nm(k), k]));
  bones.forEach((b) => { b.matrixAutoUpdate = false; b.matrixWorldAutoUpdate = false; });
  const W = new Map(), tmpS = new THREE.Vector3(), m4 = new THREE.Matrix4(), one = new THREE.Vector3(1, 1, 1);
  let S = scale;
  function set(world) { // world: SOMA joint world transforms in runtime space
    for (const b of order) {
      const parentW = W.get(b.parent) || b.parent.matrixWorld;
      const kb = KEYOF[b.name], m = kb && MAP[kb];
      if (!m) { W.set(b, parentW.clone().multiply(bindL.get(b))); continue; }
      const t = world[sIdx(m[0])];
      const delta = rtQuat(t.q).multiply(somaBindQinv[kb]);
      const q = delta.clone().multiply(align[kb]).multiply(rot(bindW.get(b)));
      let p;
      if (b === hips) {
        const a = world[LL].p, c = world[RL].p, mid = rtPos([(a[0] + c[0]) / 2, (a[1] + c[1]) / 2, (a[2] + c[2]) / 2]).multiplyScalar(S);
        p = mid.add(hipsOffset.clone().applyQuaternion(delta));
      } else p = new THREE.Vector3().setFromMatrixPosition(m4.copy(parentW).multiply(bindL.get(b)));
      W.set(b, new THREE.Matrix4().compose(p, q, tmpS.setFromMatrixScale(bindW.get(b))));
    }
    for (const b of order) b.matrixWorld.copy(W.get(b));
  }
  return {
    root, meshes, kind: 'kaykit',
    set, visible(v) { root.visible = v; },
    setScale(s) { S = s; },
    halfWidth() { // widest point sideways (three Z) in the idle pose facing +X
      root.updateMatrixWorld(true); let w = 0; const v = new THREE.Vector3();
      for (const mesh of meshes) { const P = mesh.geometry.attributes.position; for (let i = 0; i < P.count; i += 3) { v.fromBufferAttribute(P, i); if (mesh.isSkinnedMesh) mesh.applyBoneTransform(i, v); v.applyMatrix4(mesh.matrixWorld); w = Math.max(w, Math.abs(v.z)); } }
      return w;
    },
    topY() { // highest point of the posed body (for sizing the bar)
      root.updateMatrixWorld(true); let top = -Infinity; const v = new THREE.Vector3();
      for (const mesh of meshes) {
        const g = mesh.geometry, P = g.attributes.position;
        for (let i = 0; i < P.count; i += 3) { v.fromBufferAttribute(P, i); if (mesh.isSkinnedMesh) mesh.applyBoneTransform(i, v); v.applyMatrix4(mesh.matrixWorld); if (v.y > top) top = v.y; }
      }
      return top;
    },
  };
}
