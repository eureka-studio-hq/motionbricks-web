"""Smallest end-to-end check: MotionBricks generating motion on a CPU, no GPU.

    python tools/run_on_cpu.py --motionbricks ~/src/motionbricks

Starts from an idle pose and asks for a walk that ends 3 m ahead, then a duck 1.5 m further on.
Prints how long each generation took. Set MB_ONNX=web/model/motionbricks-soma-int8.onnx
(built with KEEP_ONNX=1) to time the int8 browser model instead of NVIDIA's fp32 one.
"""
import argparse
import os
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--motionbricks", required=True)
args = ap.parse_args()
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ["MOTIONBRICKS_DIR"] = args.motionbricks  # mb_cpu imports NVIDIA's package from here
import mb_cpu  # noqa: E402

REPO = mb_cpu.REPO

import warp as wp  # noqa: E402
from motionbricks.io.bvh import load_bvh  # noqa: E402
from motionbricks.model.soma.contract import SOMA_N1_FP32_ONNX_PATH  # noqa: E402
from motionbricks.model.soma.engine import MotionBricksSOMAEngine  # noqa: E402
from motionbricks.utils.space_conversion_utils import FacingDirectionType, SpaceConverter  # noqa: E402

sk = mb_cpu.load_soma_skeleton()
clips = REPO / "assets" / "motions" / "soma"
idle = load_bvh(str(clips / "locomotion/idle/idle_loop_001__A268.bvh"), sk)
walk = load_bvh(str(clips / "locomotion/bs_1d_linear/Neutral_walk_forward_002__A057.bvh"), sk)
duck = load_bvh(str(clips / "avoidance/neutral_avoid_obstacle_bend_down_walk_ff_180_R_001__A543.bvh"), sk)
to_rt = SpaceConverter(FacingDirectionType.MUJOCO).convert_transform(wp.transform_identity())


def window(clip, t0, shift_x=0.0):
    """Four poses from a clip, slid forward along the clip's own travel axis by shift_x metres."""
    out = wp.empty((4, sk.num_joints), dtype=wp.transform)
    first = wp.empty(sk.num_joints, dtype=wp.transform); clip.sample(t0, first, root_offset=to_rt)
    p = first.numpy()[0]
    offset = wp.mul(wp.transform(wp.vec3(shift_x - p[0], -p[1], 0.0), wp.quat_identity()), to_rt)
    for i in range(4):
        clip.sample(min(t0 + i / 30, clip.duration), out[i], root_offset=offset)
    return out


t = time.perf_counter()
engine = MotionBricksSOMAEngine(SOMA_N1_FP32_ONNX_PATH, model_to_runtime=to_rt, bake_cuda_graph=False)
print(f"model loaded in {time.perf_counter() - t:.1f} s")
context, target, output = engine.create_input(sk), engine.create_input(sk), engine.create_output(sk)

ctx = window(idle, 0.5)
total = 0
for label, tgt in (("walk 3 m", window(walk, 1.5, 3.0)), ("duck", window(duck, 1.2, 4.5))):
    context.update_from_poses(ctx); target.update_from_poses(tgt)
    t = time.perf_counter(); engine.generate_animation(context, target, output); ms = (time.perf_counter() - t) * 1000
    n = output.num_frames; total += n
    print(f"{label:9s} {n:3d} frames ({n / 30:.2f} s of motion) in {ms:.0f} ms")
    ctx = wp.array(output.poses[:n].numpy()[n - 4:n].copy(), dtype=wp.transform)  # Warp CPU arrays alias: copy
print(f"{total / 30:.1f} s of motion generated on the CPU")
