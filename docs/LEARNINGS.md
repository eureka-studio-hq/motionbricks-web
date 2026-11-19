# What we learned putting MotionBricks in a browser

These are notes from taking NVIDIA MotionBricks from "needs an NVIDIA GPU and TensorRT" to a web page
where you steer a stylised character around an obstacle course. They're grouped by topic, and each one
says what we tried, what happened and what we kept.

## 1. How MotionBricks thinks

- **Inputs and outputs.** You give it the last **4 poses** of the character (the context) and **1–4 target
  poses**. It returns **0.6–1.9 s** of motion at 30 fps that joins them. The length comes in tokens: 6 to
  16, with `frames = tokens × 4 − 6`.
- **It never sees the scene.** No geometry, no obstacles, no text prompt. The targets carry every bit of
  intent. Ducking under a bar means putting a ducking pose under the bar; vaulting a box means putting a
  vault-apex pose above it.
- **You can limit the segment length.** The model accepts a mask of allowed token counts. It's the most
  useful control after the targets themselves: short segments let the character arrive quickly, long ones
  give it room to walk.
- **Endpoint correction.** NVIDIA's decode nudges the last frames onto the target. Keep it in your
  port: without it, small drifts add up over a chain of generations.

## 2. Running it on a CPU

- **TensorRT is only the inference engine.** NVIDIA publishes the model as ONNX, and everything
  around it (encoding, decoding, the Warp kernels) runs fine on a CPU. `tools/mb_cpu.py` stubs out the
  `tensorrt` import and swaps the engine class for an ONNX Runtime session. Nothing else in NVIDIA's
  code changes.
- **Speed.** On a 4-core ARM VM: model load about 1.3 s; each generation of up to 1.7 s of motion takes
  about 30 ms of model time, or about 50 ms including NVIDIA's encode and decode. That's more than 30×
  faster than real time.
- **Gotcha: Warp CPU arrays alias.** A `.numpy()` view of a Warp CPU array shares memory with the array.
  Read the output poses without copying, and the next generation silently overwrites the last one. Always
  `.numpy().copy()`.
- **Gotcha: Python 3.12.** NVIDIA's package uses `StrEnum`, so Python 3.10 fails at import.

## 3. Shrinking the model for the web

| Build | Size | Result |
| --- | --- | --- |
| fp32 (NVIDIA's) | 758 MB | reference |
| 8-bit, matrix multiplies only | 261 MB (210 MB gzipped) | **kept**: motion indistinguishable in our tests |
| 8-bit, convolutions too | 192 MB | rejected: visibly broken motion (jitter, sliding) |

- **Dynamic quantisation** (`onnxruntime.quantization.quantize_dynamic`) with unsigned 8-bit weights, on
  `MatMul`/`Gemm` only, with `MatMulConstBOnly`. Weights are quantised ahead of time; activations are
  quantised on the fly. No calibration data is needed.
- **Delivery.** The file is gzipped and split into 19 MB parts. A worker streams them in order through the
  browser's built-in `DecompressionStream('gzip')`, so the whole model is never held twice in memory. The
  result is cached in IndexedDB, so later visits skip the download.
- **Don't name the parts `.gz`.** Some hosts and CDNs see `.gz`, add `Content-Encoding: gzip`, and the
  browser decompresses before your code does. The decompressor then fails on data that is no longer
  gzip.
- **Speed in the browser.** ONNX Runtime Web in WebAssembly, single-threaded: about 200–260 ms per
  generation. That's slower than the CPU path but still well ahead of playback, because each generation
  yields 0.6–1.9 s of motion. Threads need cross-origin isolation (COOP/COEP headers); the worker uses
  them when it has them.
- **Run inference in a worker.** A 250 ms call on the main thread drops frames you can see. In a worker
  the render loop never notices.

## 4. Porting encode/decode to JavaScript

- `js/motionbricks.js` is a line-by-line port of NVIDIA's Python encode/decode (root features, local poses,
  facing frames, endpoint correction).
- **Verify it with a test vector.** We dumped one generation's inputs, raw outputs and final poses from
  the Python pipeline, then checked the JavaScript against them. The two agree to
  about 1e-5. A port that's slightly wrong doesn't crash; it just looks like the model is a bit wobbly,
  so this check is the only way to know.
- **Spaces.** SOMA model space is Y-up. The runtime (and MotionBricks' world) is Z-up, through the
  quaternion (0.7071068, 0, 0, 0.7071068). three.js is Y-up again. Name every conversion; most of our bugs
  were a missing one.
- **Facing.** Facing comes from the legs, not the hips: the right-leg minus left-leg vector, crossed with
  up. Calibrate the sign once by checking that a walk clip faces the way it travels.

## 5. Steering with targets: the planner

- **Head first.** In the deepest duck, the head is about 0.5 m ahead of the hips. One duck target centred
  on the bar meant the head went straight through it. What worked was two duck targets: the first puts the
  head under the bar, the second keeps the body low until it's clear.
- **Measure, don't guess.** The bar's height comes from the posed duck's top point, for each character.
  The props' widths come from the character's measured body width.
- **Routing.** Straight-line paths are checked against every prop and scenery footprint, widened by the
  body's half-width, and detour around anything they hit. Placing a prop while he's moving triggers a
  replan, so he never walks through something that appeared in his path.
- **The walk–rest–walk problem.** Chaining walk targets made him stop and settle at every target. Two
  fixes: limit the allowed token counts using the distance and the measured speed (walk ≈ 1.45 m/s, jog
  ≈ 1.85, run ≈ 4.3), and land each segment on a moving pose (mid-stride), never an idle one.
- **Fast obstacles.** At running speed, a duck needs the short token counts (6–8), or he slows to a walk
  before the bar.

## 6. Making it look right

- **The hunch is in the data.** The sample motion carries a slight forward lean. We added a display-only
  posture layer: chest +6°, neck +8°, head −3°, halved while sitting. It never feeds back into the model's
  context, so generation is unaffected. The toolbar can switch it off.
- **Retargeting onto chibi characters.** KayKit characters have short legs and big heads. The obvious
  approach, scaling the motion down to their size, made the feet slide, because the stride no longer
  matched the legs. What works is scaling the **world** instead: world scale = KayKit hip height ÷ SOMA
  leg height (about 0.57). Props, routing and the camera all live in that scaled world.
- **Rotations.** We transfer world-space rotation deltas measured from each skeleton's bind pose, with a
  swing alignment on limbs, since the rest poses differ (SOMA binds in an A-pose, its reference pose is a
  T-pose). The hips are placed at the SOMA leg-joint midpoint.

## 7. three.js gotchas

- **`GLTFLoader` renames bones.** A root bone named the same as its armature becomes `Hips_1`, and dots
  are stripped from names (`upperleg.l` → `upperlegl`). Look bones up through a sanitising function.
- **Driving bones directly.** Set `matrixWorld` yourself and turn off `matrixAutoUpdate` and
  `matrixWorldAutoUpdate` on those bones. Otherwise three.js recomputes them from local transforms and
  overwrites yours.
- **Camera framing** follows the character's real height (×3.6), so swapping a tall knight for a short
  one doesn't leave the camera framing the old size.

## 8. Licensing (check this early)

- MotionBricks **code** is Apache-2.0. The **model** is under the NVIDIA Open Model Agreement, which
  allows commercial use and redistribution of modified models.
- The **sample motion clips** are under a separate NVIDIA Sample Data Evaluation License: internal
  evaluation only, and nothing derived from them may be redistributed. Any pose cut from them counts. So
  this repo ships a build script instead of the poses. A public product needs motion data you're licensed
  to ship (your own mocap, a commercial library, or a dataset whose licence allows it).
