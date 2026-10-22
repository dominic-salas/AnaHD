"""YUV (full range, BT.709) to RGB PNG for previews."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import torch
from torch import Tensor


def yuv_to_rgb(x: Tensor) -> Tensor:
    y, u, v = x[0], x[1] - 0.5, x[2] - 0.5
    r = y + 1.5748 * v
    g = y - 0.1873 * u - 0.4681 * v
    b = y + 1.8556 * u
    return torch.stack((r, g, b)).clamp(0, 1)


def save_rgb_png(path: Path, yuv: Tensor) -> None:
    rgb = (yuv_to_rgb(yuv.float().cpu()) * 255).round().byte().permute(1, 2, 0).numpy()
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[r].tobytes() for r in range(h))

    def chunk(t: bytes, d: bytes) -> bytes:
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")
    path.write_bytes(png)
