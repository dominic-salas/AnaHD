"""Learned 2x upscale of the cleaned frame (optional, ``--ai``), run with TensorRT.

The model is 2xNomosUni_span_multijpg_ldl by Philip Hofmann (SPAN network, CC BY 4.0):
a fast general-purpose 2x upscaler trained on photos with compression and resampling
damage. It upscales the cleaned 960x720 picture to 1920x1440, which is then reduced to
1440x1080: the extra resolution is supersampled, so edges come out finer than a direct
1.5x resample can make them. Weights are downloaded once and cached; the TensorRT engine
is built once per picture size and cached too (building takes about a minute).
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

from anahd.errors import AnaHDError

MODEL_NAME = "2xNomosUni_span_multijpg_ldl"
MODEL_URL = (
    "https://huggingface.co/Phips/2xNomosUni_span_multijpg_ldl/resolve/main/"
    "2xNomosUni_span_multijpg_ldl.safetensors"
)
MODEL_SHA256 = "a3d35e01b8b71b4b3041ad1686f8ebd7bc4e1f3a10378319c2ac61c78b67012a"
SCALE = 2


def cache_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / ".cache")
    return Path(base) / "anahd"


def _fetch(dest: Path) -> Path:
    if dest.is_file():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(".part")
    try:
        with urllib.request.urlopen(MODEL_URL, timeout=60) as resp, part.open("wb") as out:
            while chunk := resp.read(1 << 20):
                out.write(chunk)
    except OSError as exc:
        raise AnaHDError(f"could not download the upscale model: {exc}") from exc
    if hashlib.sha256(part.read_bytes()).hexdigest() != MODEL_SHA256:
        part.unlink()
        raise AnaHDError("the downloaded upscale model does not match its known checksum")
    part.replace(dest)
    return dest


class Upscaler:
    """(3, h, w) RGB in 0..1 -> (3, 2h, 2w), on the current CUDA stream."""

    def __init__(self, size: tuple[int, int], device: torch.device) -> None:
        w, h = size
        self.size = size
        self.device = device
        weights = _fetch(cache_dir() / "models" / f"{MODEL_NAME}.safetensors")
        from spandrel import ModelLoader  # optional dependency, gpu extra

        model = ModelLoader().load_from_file(str(weights)).model
        self.model = model.to(device).eval().half()
        self.stream = torch.cuda.Stream(device)  # type: ignore[no-untyped-call, unused-ignore]
        self._in = torch.zeros(1, 3, h, w, device=device, dtype=torch.float16)
        self._out = torch.zeros(1, 3, h * SCALE, w * SCALE, device=device, dtype=torch.float16)
        self._trt = self._engine(weights)

    def _engine(self, weights: Path):  # type: ignore[no-untyped-def]
        """A TensorRT execution context, or None to run the model in PyTorch (slower)."""
        try:
            import tensorrt as trt
        except ImportError:
            return None
        w, h = self.size
        gpu = torch.cuda.get_device_name(self.device).replace(" ", "_")
        name = f"{MODEL_NAME}_{MODEL_SHA256[:12]}_{w}x{h}_{gpu}_trt{trt.__version__}.engine"
        path = cache_dir() / "engines" / name
        log = trt.Logger(trt.Logger.ERROR)
        if not path.is_file():
            onnx = path.with_suffix(".onnx")
            path.parent.mkdir(parents=True, exist_ok=True)
            with torch.inference_mode():
                torch.onnx.export(
                    self.model,
                    (self._in,),
                    str(onnx),
                    input_names=["x"],
                    output_names=["y"],
                    opset_version=17,
                    dynamo=False,
                )
            builder = trt.Builder(log)
            network = builder.create_network(0)
            parser = trt.OnnxParser(network, log)
            if not parser.parse(onnx.read_bytes()):
                raise AnaHDError(f"TensorRT could not read the model: {parser.get_error(0)}")
            serial = builder.build_serialized_network(network, builder.create_builder_config())
            if serial is None:
                raise AnaHDError("TensorRT could not build the upscale engine")
            path.write_bytes(bytes(serial))
            onnx.unlink()
        engine = trt.Runtime(log).deserialize_cuda_engine(path.read_bytes())
        ctx = engine.create_execution_context()
        ctx.set_tensor_address("x", self._in.data_ptr())
        ctx.set_tensor_address("y", self._out.data_ptr())
        self._engine_ref = engine  # keep alive while the context is used
        return ctx

    def __call__(self, rgb: Tensor) -> Tensor:
        self._in.copy_(rgb[None])
        main = torch.cuda.current_stream(self.device)
        self.stream.wait_stream(main)
        with torch.cuda.stream(self.stream):
            if self._trt is not None:
                self._trt.execute_async_v3(self.stream.cuda_stream)
            else:
                with torch.inference_mode():
                    self._out.copy_(self.model(self._in))
        main.wait_stream(self.stream)
        return self._out[0].float()


def reduce(rgb: Tensor, size: tuple[int, int]) -> Tensor:
    """(3, H, W) -> (3, height, width) with an antialiased bicubic."""
    w, h = size
    out = F.interpolate(rgb[None], size=(h, w), mode="bicubic", antialias=True, align_corners=False)
    return out[0]
