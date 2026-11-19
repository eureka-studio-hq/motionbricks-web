"""Build everything the web demo needs from your own copy of NVIDIA MotionBricks.

    python tools/build_web_assets.py --motionbricks ~/src/motionbricks

Writes, inside web/:
  data/skeleton.json      SOMA skeleton: 77 joints, parents, reference pose, the 30 joints the model sees
  data/pose_library.json  target pose windows (idle, walk, jog, run, duck, sit, stand, vault) + the mesh bind pose
  model/manifest.json     the int8 model, gzip-compressed and split into 19 MB parts the browser streams

The pose library is cut from NVIDIA's sample clips, which are licensed for evaluating NVIDIA technology only.
Keep what this builds internal. See the README's licence section before deploying anywhere public.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
WEB = HERE.parent / "web"
PART_BYTES = 19_000_000


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--motionbricks", required=True, help="path to your clone of NVlabs/motionbricks, with Git LFS files pulled")
    ap.add_argument("--out", default=str(WEB), help="web folder to write into (default: web/)")
    ap.add_argument("--skip-model", action="store_true", help="only rebuild skeleton.json and pose_library.json")
    return ap.parse_args()


args = parse()
sys.path.insert(0, str(HERE))
os.environ["MOTIONBRICKS_DIR"] = args.motionbricks  # mb_cpu imports NVIDIA's package from here
import mb_cpu  # noqa: E402

REPO = mb_cpu.REPO
OUT = Path(args.out)
(OUT / "data").mkdir(parents=True, exist_ok=True)

import numpy as np  # noqa: E402
import warp as wp  # noqa: E402
from scipy.spatial.transform import Rotation as R  # noqa: E402
from motionbricks.io.bvh import load_bvh  # noqa: E402
from motionbricks.utils.space_conversion_utils import FacingDirectionType, SpaceConverter  # noqa: E402

FP32 = REPO / "motionbricks" / "assets" / "onnx" / "soma" / "mmm_v6_soma_fp32.onnx"
N1 = REPO / "motionbricks" / "assets" / "onnx" / "soma" / "soma_n1.json"
USD = REPO / "motionbricks" / "assets" / "human" / "soma_base_skel_minimal_no_root.usd"
CLIPS_DIR = REPO / "assets" / "motions" / "soma"
for f in (FP32, USD, CLIPS_DIR / "locomotion" / "idle" / "idle_loop_001__A268.bvh"):
    if not f.exists() or f.stat().st_size < 1000:
        sys.exit(f"Missing {f}\nRun `git lfs pull` inside {REPO} first (the model, clips and USD are Git LFS files).")

sk = mb_cpu.load_soma_skeleton()
names = list(sk.joint_names)
RL, LL = names.index("RightLeg"), names.index("LeftLeg")
TO_RT = SpaceConverter(FacingDirectionType.MUJOCO).convert_transform(wp.transform_identity())  # SOMA Y-up -> runtime Z-up

# ------------------------------------------------------------------ skeleton.json
model_joints = json.loads(N1.read_text())["joint_names"]
skeleton = {
    "names": names,
    "parents": [int(p) for p in sk.parent_indices.numpy()],
    "reference_local": np.round(sk.reference_local_transforms.numpy().astype(np.float64), 9).tolist(),
    "model_joints": model_joints,
}
(OUT / "data" / "skeleton.json").write_text(json.dumps(skeleton))
print(f"skeleton.json      {len(names)} joints, {len(model_joints)} seen by the model")

# ------------------------------------------------------------------ pose windows
CLIP_FILES = {
    "idle": "locomotion/idle/idle_loop_001__A268",
    "walk": "locomotion/bs_1d_linear/Neutral_walk_forward_002__A057",
    "jog": "locomotion/bs_1d_linear/neutral_jog_ff_180_R_001__A534",
    "run_a": "locomotion/bs_1d_linear/run_loop_180_R_003__A331",
    "run_b": "locomotion/bs_1d_linear/run_loop_180_R_003__A326",
    "duck": "avoidance/neutral_avoid_obstacle_bend_down_walk_ff_180_R_001__A543",
    "sit_loop": "sitting/sit_on_chair_loop",
    "sit_stop": "sitting/sit_on_chair_stop_R_001__A047",
    "jump": "jump/jump_over_obstacle_0_5m_R_001__A415",
}
CLIPS = {k: load_bvh(str(CLIPS_DIR / f"{v}.bvh"), sk) for k, v in CLIP_FILES.items()}
facing_sign = 1.0


def raw_window(clip, t0, offset=TO_RT):
    out = wp.empty((4, sk.num_joints), dtype=wp.transform)
    for i in range(4):
        clip.sample(float(min(t0 + i / 30, clip.duration)), out[i], root_offset=offset)
    return out


def ground_frame(pose):
    """Root XY and facing yaw of one pose, facing taken from the legs (right minus left, crossed with up)."""
    root_q = R.from_quat(pose[0, 3:7])
    side = root_q.apply(pose[RL, :3]) - root_q.apply(pose[LL, :3])
    fwd = np.cross([0, 0, 1], side) * facing_sign
    return pose[0, :2], math.atan2(fwd[1], fwd[0])


def placed_window(clip, t0, x=0.0, y=0.0, yaw_deg=0.0):
    """Four consecutive poses from a clip, moved so the first one stands at (x, y) facing yaw."""
    first = raw_window(clip, t0).numpy()[0]
    (rx, ry), a = ground_frame(first)
    rot = R.from_euler("z", math.radians(yaw_deg) - a)
    tx, ty, _ = np.array([x, y, 0]) - rot.apply([rx, ry, 0])
    place = wp.transform(wp.vec3(tx, ty, 0), wp.quat(*rot.as_quat()))
    return raw_window(clip, t0, wp.mul(place, TO_RT))


# the facing sign is whatever makes the walk clip face the way it travels
w = CLIPS["walk"]
p0, p1 = raw_window(w, 1.0).numpy()[0], raw_window(w, 2.0).numpy()[0]
if np.dot(p1[0, :2] - p0[0, :2], [math.cos(ground_frame(p0)[1]), math.sin(ground_frame(p0)[1])]) < 0:
    facing_sign = -1.0


def scan(clip, pick):
    ts = np.arange(0, clip.duration, 1 / 30)
    hips = [raw_window(clip, t).numpy()[0, 0, 2] for t in ts]
    return float(ts[pick(hips)])


t_duck = scan(CLIPS["duck"], np.argmin)  # deepest point of the duck
t_jump = scan(CLIPS["jump"], np.argmax)  # apex of the vault

lib: dict[str, list] = {}


def add(name, clip, t):
    win = placed_window(CLIPS[clip], float(t)).numpy()
    lib.setdefault(name, []).append(np.round(win, 5).reshape(4, -1).tolist())


for t in (0.5, 2.0, 4.0, 6.0): add("idle", "idle", t)
for t in np.arange(1.0, 2.2, 0.15): add("walk", "walk", t)
for t in np.arange(1.0, 2.0, 0.15): add("jog", "jog", t)
for c in ("run_a", "run_b"):
    for t in np.arange(0.8, min(2.2, CLIPS[c].duration - 0.3), 0.15): add("run", c, t)
add("duck", "duck", t_duck); add("duck", "duck", t_duck - 0.15)
add("sit", "sit_loop", 0.5); add("sit", "sit_loop", 2.5)
add("stand", "sit_stop", CLIPS["sit_stop"].duration - 0.2)
add("vault_apex", "jump", t_jump); add("vault_land", "jump", CLIPS["jump"].duration - 0.3)

# mesh bind pose (USD bindTransforms), in runtime space: retargeting measures everything against it
from pxr import Usd, UsdGeom, UsdSkel  # noqa: E402
stage = Usd.Stage.Open(str(USD))
mpu = UsdGeom.GetStageMetersPerUnit(stage)
skel = next(UsdSkel.Skeleton(p) for p in stage.Traverse() if p.IsA(UsdSkel.Skeleton))
usd_names = [j.split("/")[-1] for j in skel.GetJointsAttr().Get()]
bind = np.array([np.array(m) for m in skel.GetBindTransformsAttr().Get()])  # row-vector matrices, translation in last row
C = R.from_quat([0.70710677, 0, 0, 0.70710677])
bind_runtime = []
for n in names:
    m = bind[usd_names.index(n)]
    bind_runtime.append({"p": np.round(C.apply(m[3, :3] * mpu), 6).tolist(),
                         "q": np.round((C * R.from_matrix(m[:3, :3].T)).as_quat(), 6).tolist()})

(OUT / "data" / "pose_library.json").write_text(json.dumps({
    "poses": lib, "bind_runtime": bind_runtime,
    "note": "Built from NVIDIA MotionBricks sample clips (NVIDIA Sample Data Evaluation License): evaluation use only, do not redistribute.",
}))
print("pose_library.json", {k: len(v) for k, v in lib.items()})

# ------------------------------------------------------------------ browser model
if args.skip_model:
    sys.exit(0)
from onnxruntime.quantization import QuantType, quantize_dynamic  # noqa: E402

(OUT / "model").mkdir(parents=True, exist_ok=True)
int8 = OUT / "model" / "motionbricks-soma-int8.onnx"
t = time.perf_counter()
# Only the matrix multiplies: quantizing the convolutions too visibly breaks the motion.
quantize_dynamic(str(FP32), str(int8), op_types_to_quantize=["MatMul", "Gemm"], weight_type=QuantType.QUInt8,
                 extra_options={"MatMulConstBOnly": True})  # 8-bit weights, activations quantized on the fly
raw = int8.read_bytes()
print(f"int8 model         {FP32.stat().st_size / 1e6:.0f} MB -> {len(raw) / 1e6:.0f} MB in {time.perf_counter() - t:.0f} s")

packed = gzip.compress(raw, compresslevel=9, mtime=0)
for old in (OUT / "model").glob("motionbricks-soma-int8.onnx.gz.part*"):
    old.unlink()
parts = []
for i in range(0, len(packed), PART_BYTES):
    name = f"motionbricks-soma-int8.onnx.gz.part{i // PART_BYTES:03d}"  # no ".gz" suffix: hosts must not decompress it
    (OUT / "model" / name).write_bytes(packed[i:i + PART_BYTES])
    parts.append(name)
(OUT / "model" / "manifest.json").write_text(json.dumps({
    "bytes": len(raw), "packed": len(packed), "sha256": hashlib.sha256(raw).hexdigest(), "parts": parts,
}, indent=2))
if not os.environ.get("KEEP_ONNX"):
    int8.unlink()  # the browser only needs the parts; set KEEP_ONNX=1 to keep it for tools/run_on_cpu.py
print(f"model/             {len(parts)} parts, {len(packed) / 1e6:.0f} MB gzipped")
print("Done. Serve web/ with any static server, e.g.  python -m http.server -d web 8000")
