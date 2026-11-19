# Direct the Hero: NVIDIA MotionBricks in the browser

Place a bar, a box or a seat, click where to go, and a character ducks, vaults, sits and runs to get there.
Every step is generated live by [NVIDIA MotionBricks](https://github.com/NVlabs/motionbricks), running in the
browser tab. There's no server, and no GPU is needed.

MotionBricks officially needs an NVIDIA GPU and TensorRT. This repo shows it running in two places it wasn't
built for:

- **On a CPU**, through ONNX Runtime: about 50 ms per generation (1.7 s of motion) in a 4-core VM on a Mac.
- **In a web page**, through ONNX Runtime Web: an 8-bit build of the model runs in WebAssembly in about 250 ms
  per generation, which is enough to stay ahead of playback.

On top of the model, the repo adds a small motion planner and retargeting onto stylised game characters
(the free [KayKit Adventurers](https://kaylousberg.itch.io/kaykit-adventurers)).

> **What's in this repo:** our code, the tools and the four KayKit characters.
> **What isn't:** NVIDIA's model and the poses that steer the hero. You build both from your own copy of
> MotionBricks with one command (below). The poses come from NVIDIA's sample clips, which are licensed for
> evaluation only, so they can't be redistributed. See [Licences](#licences).

## Quick start

You need Python 3.12+, Git LFS, and about 2 GB of free disk space.

```bash
# 1. NVIDIA MotionBricks, with its Git LFS files (model, sample clips, skeleton)
git clone https://github.com/NVlabs/motionbricks ~/src/motionbricks
git -C ~/src/motionbricks lfs pull

# 2. This repo and its build tools
git clone https://github.com/eureka-studio-hq/motionbricks-web && cd motionbricks-web
python3 -m venv .venv && . .venv/bin/activate
pip install -r tools/requirements.txt

# 3. Build the skeleton, poses and browser model into web/ (takes a few minutes)
python tools/build_web_assets.py --motionbricks ~/src/motionbricks

# 4. Serve it
python -m http.server -d web 8000   # open http://localhost:8000
```

The first visit downloads the model (about 210 MB, gzipped). After that it loads from the browser's cache.

To run MotionBricks on a CPU without the browser:

```bash
python tools/run_on_cpu.py --motionbricks ~/src/motionbricks
```

## Using it

| Action | How |
| --- | --- |
| Walk somewhere | Click the ground |
| Place a bar, box or seat | `1`, `2`, `3` (or the toolbar), then click; scroll or `R` rotates it |
| Use a prop | Click it: he ducks under bars, vaults boxes and sits on seats |
| Ready-made courses | **Patterns**: hurdles, limbo, parkour, weave, waiting room, patrol, chaos |
| Pace | **Run** toggles walking and running |
| See what the model is given | **Targets** (`G`) shows or hides the target poses |
| Posture | **Posture** lifts the chest and head slightly (display only) |
| Start over | **Clear** removes all props; `Esc` cancels placing |
| Character and world | Knight, Rogue, Mage or Barbarian; Playground or Studio |

The green ghost is the target pose handed to the model. Everything in between is generated.

## How it works

```
click / prop / pattern
        │
        ▼
 planner ──► target poses (where, which way, which pose)      js/app.js
        │
        ▼
 encode ──► ONNX Runtime Web, in a worker ──► decode           js/motionbricks.js, js/gen-worker.js
        │
        ▼
 30 fps joint transforms ──► retarget onto a KayKit rig        js/retarget.js
```

1. **MotionBricks never sees the scene.** It takes the last 4 poses of the character and 1–4 target
   poses, and fills in 0.6–1.9 s of motion between them. To make the hero duck under a bar, the planner
   puts a ducking pose under the bar.
2. **The planner** turns a goal into a chain of targets. It routes around props and scenery, chooses the
   pose for each prop, and limits how long each generated segment can be, which is how the walk, jog and
   run speeds are controlled.
3. **The model runs in a Web Worker**, so the 3D view never stalls. Generation stays about a second ahead
   of playback.
4. **Retargeting** maps NVIDIA's 77-joint SOMA skeleton onto each KayKit rig. The KayKit characters are
   chibi-proportioned, so instead of squashing the motion, the world is scaled down to match their stride.

The full write-up, including what failed, is in [docs/LEARNINGS.md](docs/LEARNINGS.md).

## Who this is for

- **Web game developers.** Generated locomotion and obstacle moves for NPCs or players, with no
  animation graph to build and no server to run.
- **Indie studios without NVIDIA GPUs.** The CPU path (`tools/mb_cpu.py`) runs NVIDIA's own pipeline
  with ONNX Runtime in place of TensorRT. You can prototype, bake animation offline, or batch-generate
  clips on any laptop or CI machine.
- **Animators and technical artists.** Block a shot by placing a few key poses and let the model fill
  in the motion between them.
- **Anyone shipping an ONNX model to the web.** The model loading here may be useful on its own:
  8-bit quantisation of the matrix multiplies only, gzip streaming in parts, and IndexedDB caching.
- **Retargeting onto stylised characters.** `js/retarget.js` drives rigs with very different
  proportions from a realistic skeleton.

## Deploying

`web/` is plain static files with no build step, so it can be served from any static host. With Vercel:

```bash
cd web && npx vercel
```

Before you deploy anywhere public, read the next section. The pose library is built from NVIDIA's
evaluation-only clips, so keep a deployment that includes it internal: local, behind a password or on a
private preview. To go public, replace `web/data/pose_library.json` with poses from motion data you're
licensed to distribute. The format is documented in `tools/build_web_assets.py`.

## Repo layout

```
web/                     the site (static)
  index.html
  js/app.js              scene, props, planner, UI
  js/motionbricks.js     MotionBricks encode/decode, ported from NVIDIA's Python and checked against it
  js/gen-worker.js       ONNX Runtime Web worker: streams, caches and runs the model
  js/retarget.js         SOMA skeleton → KayKit rig
  vendor/                ONNX Runtime Web 1.23.2 (WebAssembly build)
  data/characters/       KayKit Adventurers (CC0), animations stripped
  data/*.json, model/    built by tools/build_web_assets.py (not committed)
tools/
  build_web_assets.py    skeleton + pose library + 8-bit browser model, from your MotionBricks clone
  mb_cpu.py              runs MotionBricks on a CPU: TensorRT stubbed, ONNX Runtime in its place
  run_on_cpu.py          minimal CPU example with timings
docs/LEARNINGS.md        what we learned getting here
```

## Licences

| Part | Licence |
| --- | --- |
| This repo's code | [Apache-2.0](LICENSE) |
| NVIDIA MotionBricks code (ported in `js/motionbricks.js`, used by `tools/`) | Apache-2.0, © NVIDIA |
| MotionBricks model, including the 8-bit build you make | [NVIDIA Open Model Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/) |
| Pose library you build from NVIDIA's sample clips | NVIDIA Sample Data Evaluation License: evaluation only, no redistribution |
| KayKit Adventurers characters | CC0, by [Kay Lousberg](https://www.kaylousberg.com) |
| ONNX Runtime Web | MIT, © Microsoft |
| three.js (loaded from jsDelivr) | MIT |

Details are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). This project isn't affiliated with or endorsed by NVIDIA.

## Credits

Built by [Eureka Studios](https://github.com/eureka-studio-hq). MotionBricks is by NVIDIA Research. Characters are by Kay Lousberg.
