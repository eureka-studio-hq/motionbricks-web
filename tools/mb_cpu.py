"""Run NVIDIA MotionBricks on a CPU, with no GPU and no TensorRT.

Import this module before anything from `motionbricks`. It:
  * stubs out TensorRT (CUDA-only),
  * swaps MotionBricks' inference engine for ONNX Runtime on the CPU,
  * keeps all of NVIDIA's encode/decode code untouched (its Warp kernels run on the CPU too).

Point it at your clone of https://github.com/NVlabs/motionbricks (with Git LFS files pulled)
using the MOTIONBRICKS_DIR environment variable before importing this module.
"""
from __future__ import annotations

import json
import os
import sys
import time
import types
from contextlib import contextmanager
from pathlib import Path
from types import MappingProxyType

import numpy as np

REPO: Path | None = None


def use_repo(path: str | os.PathLike) -> Path:
    """Make NVIDIA's `motionbricks` package importable from a local clone."""
    global REPO
    REPO = Path(path).expanduser().resolve()
    if not (REPO / "motionbricks" / "__init__.py").exists():
        sys.exit(f"{REPO} does not look like a clone of NVlabs/motionbricks")
    for p in (REPO, REPO / "docs" / "examples"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    return REPO


if os.environ.get("MOTIONBRICKS_DIR"):
    use_repo(os.environ["MOTIONBRICKS_DIR"])
else:
    try:
        import motionbricks  # noqa: F401  (already installed, e.g. `pip install -e` of NVIDIA's repo)
    except ImportError:
        sys.exit("Set MOTIONBRICKS_DIR to your clone of https://github.com/NVlabs/motionbricks")


class _Anything:
    def __init__(self, *a, **k): pass
    def __getattr__(self, name): return _Anything()
    def __call__(self, *a, **k): return _Anything()
    def __hash__(self): return id(self)


def _install_stubs():
    trt = types.ModuleType("tensorrt")
    trt.__version__ = "cpu-stub"
    trt.ILogger = type("ILogger", (), {"Severity": int, "__init__": lambda self, *a: None})
    trt.Logger = types.SimpleNamespace(INTERNAL_ERROR=0, ERROR=1, WARNING=2, INFO=3, VERBOSE=4)
    trt.__getattr__ = lambda name: _Anything()
    sys.modules.setdefault("tensorrt", trt)
    try:
        import newton  # noqa: F401  (only needed for the Unitree G1 robot)
    except Exception:
        mod = types.ModuleType("newton"); mod.__getattr__ = lambda name: _Anything()
        sys.modules["newton"] = mod


_install_stubs()

import onnxruntime as ort  # noqa: E402
import warp as wp  # noqa: E402

wp.config.quiet = True
wp.init()
wp.set_device("cpu")

from motionbricks.inference import engine as _ie  # noqa: E402
from motionbricks.inference.types import TensorMode, TensorSpec  # noqa: E402
import motionbricks.model.engine as _me  # noqa: E402
import motionbricks.model.generation as _gen  # noqa: E402


def _init(self, onnx_path, *, build_options=None, stream=None, cache_dir=None,
          force_rebuild=False, logger_severity="INFO", build_progress_monitor=None):
    self._closed = False
    self._cuda_graph = None
    onnx_path = Path(onnx_path)
    manifest = next(onnx_path.parent.glob("*_n1.json"))
    if os.environ.get("MB_ONNX"):  # e.g. the int8 build made for the web
        onnx_path = Path(os.environ["MB_ONNX"])
    specs = [TensorSpec(s["name"], TensorMode(s["mode"]), tuple(s["shape"]), np.dtype(s["dtype"]))
             for s in json.loads(manifest.read_text())["tensor_specs"]]
    self._tensor_specs = MappingProxyType({s.name: s for s in specs})
    self._buffers = {s.name: wp.zeros(s.shape, dtype=wp.dtype_from_numpy(s.dtype), device="cpu") for s in specs}
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    t = time.perf_counter()
    self._session = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])
    self.load_seconds = time.perf_counter() - t
    self._inputs = [i.name for i in self._session.get_inputs()]
    self._outputs = [o.name for o in self._session.get_outputs()]
    self.inference_ms = []


def _enqueue(self):
    feed = {n: self._buffers[n].numpy() for n in self._inputs}
    t = time.perf_counter()
    results = self._session.run(self._outputs, feed)
    self.inference_ms.append((time.perf_counter() - t) * 1000)
    for name, value in zip(self._outputs, results):
        dst = self._buffers[name].numpy()
        dst[...] = 0
        dst[tuple(slice(0, d) for d in value.shape)] = value


def _close(self):
    self._closed = True


_ie.InferenceEngine.__init__ = _init
_ie.InferenceEngine.enqueue_async = _enqueue
_ie.InferenceEngine.synchronize = lambda self: None
_ie.InferenceEngine.bake_cuda_graph = lambda self: None
_ie.InferenceEngine.close = _close
_ie.InferenceEngine.device = property(lambda self: wp.get_device("cpu"))
_ie.InferenceEngine.stream = property(lambda self: None)
_ie.InferenceEngine.cuda_graph_baked = property(lambda self: True)  # no CUDA graph capture on CPU


class CPUTimer:
    def __init__(self, stream=None):
        self.milliseconds = dict.fromkeys(_gen.GENERATION_STAGES, 0.0)
        self._t = {}

    @contextmanager
    def stage(self, name):
        t = time.perf_counter(); yield; self._t[name] = (time.perf_counter() - t) * 1000

    def read(self):
        for k in self.milliseconds:
            self.milliseconds[k] = self._t.get(k, 0.0)
        self._t.clear()


_me.GenerationTimer = CPUTimer
_gen.GenerationTimer = CPUTimer


def load_soma_skeleton():
    """NVIDIA's SOMA skeleton, read from the USD file that ships with MotionBricks."""
    from motionbricks.io.usd import load_skeleton_from_usd
    from motionbricks.io.utils import SOMA_SKELETON_PRIM_PATH, get_asset_file
    return load_skeleton_from_usd(get_asset_file("human", "soma_base_skel_minimal_no_root.usd"), SOMA_SKELETON_PRIM_PATH)
