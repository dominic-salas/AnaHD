"""Rough no-reference comparison of two renders of the same clip (CPU):

    python tools/noise_metrics.py A.mp4 B.mp4

Per render: spatial noise (Immerkaer sigma of the luma, levels) and temporal flicker of fine
detail (mean |high-pass(f_t) - high-pass(f_t-1)|, levels). Motion adds to the flicker of both
alike, so only the difference between the two renders means anything.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

W, H = 720, 540


def luma(path: Path) -> np.ndarray:
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"scale={W}:{H}:flags=area"]
    cmd += ["-pix_fmt", "gray", "-f", "rawvideo", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, H, W).astype(np.float32)


def immerkaer(f: np.ndarray) -> float:
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    r = np.abs(cv2.filter2D(f, -1, k))[1:-1, 1:-1]
    return float(np.sqrt(np.pi / 2) * r.mean() / 6)


def main() -> int:
    for p in map(Path, sys.argv[1:]):
        v = luma(p)
        sig = np.mean([immerkaer(f) for f in v[::10]])
        hp = v - np.stack([cv2.GaussianBlur(f, (0, 0), 1.5) for f in v])
        flick = float(np.abs(np.diff(hp, axis=0)).mean())
        print(f"{p.name}: noise {sig:.2f}, fine-detail flicker {flick:.2f}, frames {len(v)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
