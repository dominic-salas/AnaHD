"""Learned 2x upscale of the cleaned frame (the classic path's default), run with TensorRT.

The model is 2xNomosUni_span_multijpg_ldl by Philip Hofmann (SPAN network, CC BY 4.0):
a fast general-purpose 2x upscaler trained on photos with compression and resampling
damage. It upscales the cleaned 960x720 picture to 1920x1440 (1280x720 to 2560x1440), the
output size, so nothing is shrunk afterwards. Weights are downloaded once and cached; the
TensorRT engine is built once per picture size and cached too (building takes about a
minute). Without TensorRT the model runs in PyTorch FP16; if that cannot run either,
:func:`make_upscaler` returns None and the caller uses Lanczos.
"""

from __future__ import annotations

import hashlib
import logging
import os
import urllib.request
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

from anahd.errors import AnaHDError
from anahd.gpu.engine import build_engine

log = logging.getLogger(__name__)

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
        key = f"{MODEL_NAME}_{MODEL_SHA256[:12]}_{w}x{h}"
        self._trt = build_engine(self.model, self._in, key)

    def __call__(self, rgb: Tensor) -> Tensor:
        main = torch.cuda.current_stream(self.device)
        self.stream.wait_stream(main)
        with torch.cuda.stream(self.stream):
            self._in.copy_(rgb[None])
            if self._trt is not None:
                out = self._trt.run(self._in, self.stream.cuda_stream)
            else:
                with torch.inference_mode():
                    out = self.model(self._in)
            self._out.copy_(out)
        main.wait_stream(self.stream)
        return self._out[0].float()


def make_upscaler(size: tuple[int, int], device: torch.device) -> Upscaler | None:
    """The SPAN upscaler, or None (use Lanczos) when it cannot be set up: no spandrel, no
    download, no memory."""
    try:
        return Upscaler(size, device)
    except Exception as exc:
        log.warning("the learned upscaler is not available (%s); using Lanczos", exc)
        return None


def reduce(rgb: Tensor, size: tuple[int, int]) -> Tensor:
    """(3, H, W) -> (3, height, width) with an antialiased bicubic."""
    w, h = size
    if tuple(rgb.shape[-2:]) == (h, w):
        return rgb
    out = F.interpolate(rgb[None], size=(h, w), mode="bicubic", antialias=True, align_corners=False)
    return out[0]
