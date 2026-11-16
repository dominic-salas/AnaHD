"""TensorRT FP16 engines for the networks, and compiled (Triton) code, each with the plain
PyTorch path as the fallback: a clean never fails because an accelerator cannot run.

Engines are built once per network, picture size, GPU and TensorRT version and cached under
``%LOCALAPPDATA%\\AnaHD\\engines``. Compiled code is cached by inductor under
``%LOCALAPPDATA%\\AnaHD\\inductor`` (the first compile of a new picture size takes about a
minute and a half, later runs reuse it).

TensorRT itself is found by :mod:`anahd.gpu.trt` (installed with the app, downloaded into
the AnaHD folder, or missing); engines share its cache folder per GPU, driver and version.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

log = logging.getLogger(__name__)

COMPILE_MODE = "max-autotune-no-cudagraphs"  # graphs are captured around the compiled code


def cache_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / ".cache")
    return Path(base) / "AnaHD"


class Engine:
    """A TensorRT engine for a fixed input: ``run`` copies into its input buffer, executes
    on a stream and returns its (reused) output buffer."""

    def __init__(self, model: nn.Module, example: Tensor, key: str) -> None:
        from anahd.gpu import trt as finder

        trt = finder.module()
        if trt is None:
            raise RuntimeError(finder.status().reason)
        self._in = example.clone()
        with torch.inference_mode():
            self.out: Tensor = model(example).detach().clone()
        gpu = torch.cuda.get_device_name(example.device)
        path = finder.engine_cache_dir(gpu) / f"{key}.engine"
        log_ = trt.Logger(trt.Logger.ERROR)
        if not path.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            onnx = path.with_suffix(".onnx")
            with torch.inference_mode():
                torch.onnx.export(
                    model,
                    (self._in,),
                    str(onnx),
                    input_names=["x"],
                    output_names=["y"],
                    opset_version=17,
                    dynamo=False,
                )
            builder = trt.Builder(log_)
            network = builder.create_network(0)
            if not trt.OnnxParser(network, log_).parse(onnx.read_bytes()):
                raise RuntimeError("TensorRT could not read the model")
            serial = builder.build_serialized_network(network, builder.create_builder_config())
            if serial is None:
                raise RuntimeError("TensorRT could not build the engine")
            path.write_bytes(bytes(serial))
            onnx.unlink()
        self._engine = trt.Runtime(log_).deserialize_cuda_engine(path.read_bytes())
        self._ctx = self._engine.create_execution_context()
        self._ctx.set_tensor_address("x", self._in.data_ptr())
        self._ctx.set_tensor_address("y", self.out.data_ptr())

    def run(self, x: Tensor, stream: int | None = None) -> Tensor:
        self._in.copy_(x)
        if stream is None:
            stream = torch.cuda.current_stream(x.device).cuda_stream
        self._ctx.execute_async_v3(stream)
        return self.out


def build_engine(model: nn.Module, example: Tensor, key: str) -> Engine | None:
    """The engine, or None (run the PyTorch model) on any failure: no TensorRT, no ONNX,
    a build error, an old driver."""
    try:
        return Engine(model, example, key)
    except Exception as exc:
        log.warning("TensorRT is not used (%s); running in PyTorch", exc)
        return None


def compile_available() -> bool:
    """Compiling needs Triton and a CUDA GPU (on Windows the triton-windows wheel brings its
    own C compiler); ``ANAHD_NO_COMPILE=1`` turns it off."""
    if os.environ.get("ANAHD_NO_COMPILE") or not torch.cuda.is_available():
        return False
    try:
        import triton  # type: ignore[import-untyped, unused-ignore]  # noqa: F401
    except ImportError:
        return False
    return True


def use_compile_cache() -> None:
    root = cache_dir()
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(root / "inductor"))
    os.environ.setdefault("TRITON_CACHE_DIR", str(root / "triton"))


class Compiled:
    """``fn`` compiled by torch.compile on first call; if compiling or running it fails
    (no C compiler, an unsupported op), ``fn`` itself runs from then on."""

    def __init__(self, fn: Callable[..., Any]) -> None:
        use_compile_cache()
        self._plain = fn
        self._fn: Callable[..., Any] = torch.compile(fn, mode=COMPILE_MODE, dynamic=False)
        self.failed = False

    def __call__(self, *args: Any) -> Any:
        if not self.failed:
            try:
                return self._fn(*args)
            except Exception as exc:
                log.warning("compiling failed (%s); running without it", exc)
                self.failed = True
        return self._plain(*args)


def maybe_compile(fn: Callable[..., Any], enabled: bool) -> Callable[..., Any]:
    return Compiled(fn) if enabled else fn


def compile_is_warm(tag: str) -> bool:
    """Whether ``tag`` (a network and picture size) was compiled before, so its compile
    cache is there and starting it costs about two seconds instead of a minute and a half."""
    return (cache_dir() / "inductor" / f"{tag}.warm").is_file()


def mark_compiled(tag: str, fn: Callable[..., Any]) -> None:
    if getattr(fn, "failed", True):
        return
    marker = cache_dir() / "inductor" / f"{tag}.warm"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()
