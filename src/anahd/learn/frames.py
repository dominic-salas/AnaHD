"""Minimal FFmpeg frame reader for the learning tools (YUV 4:4:4, uint8, full range)."""

from __future__ import annotations

import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Crop:
    w: int
    h: int
    x: int
    y: int


def read_yuv(
    path: Path,
    crop: Crop | None,
    size: tuple[int, int] | None,
    start_s: float = 0.0,
    dur_s: float | None = None,
    scaler: str = "area",
) -> Iterator[NDArray[np.uint8]]:
    """Yield frames as (3, H, W) uint8 YUV, after optional crop then resize to `size` (w, h)."""
    vf: list[str] = []
    if crop is not None:
        vf.append(f"crop={crop.w}:{crop.h}:{crop.x}:{crop.y}")
    if size is not None:
        vf.append(f"scale={size[0]}:{size[1]}:flags={scaler}:in_range=full:out_range=full")
    vf.append("format=yuv444p")
    cmd = ["ffmpeg", "-v", "error", "-nostdin"]
    if start_s > 0:
        cmd += ["-ss", f"{start_s:.3f}"]
    cmd += ["-i", str(path)]
    if dur_s is not None:
        cmd += ["-t", f"{dur_s:.3f}"]
    cmd += ["-vf", ",".join(vf), "-f", "rawvideo", "-pix_fmt", "yuv444p", "-"]
    w, h = size if size is not None else ((crop.w, crop.h) if crop else probe_size(path))
    n = 3 * w * h
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    if proc.stdout is None:
        raise RuntimeError("ffmpeg gave no output pipe")
    try:
        while True:
            buf = proc.stdout.read(n)
            if len(buf) < n:
                break
            yield np.frombuffer(buf, np.uint8).reshape(3, h, w)
    finally:
        proc.stdout.close()
        proc.wait()


def probe_size(path: Path) -> tuple[int, int]:
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    w, h = out.split(",")[:2]
    return int(w), int(h)
