"""The encoder writes the frames it is given: AnaHD's GPU frames are already limited-range
BT.709, and FFmpeg must not convert them again (FFmpeg 9 did, taking untagged raw YUV for
BT.601: ~3.5 levels of luma error that only shows on coloured pictures)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from anahd.media import Encoder, VideoInfo

W, H = 320, 240


def _colour_bars() -> bytes:
    """Y, U, V planes of strongly coloured bars (a neutral picture would hide a matrix
    change: BT.601 and BT.709 agree when there is no colour)."""
    y = np.zeros((H, W), np.uint8)
    u = np.zeros((H // 2, W // 2), np.uint8)
    v = np.zeros((H // 2, W // 2), np.uint8)
    bars = [
        (81, 90, 240),
        (145, 54, 34),
        (41, 240, 110),
        (170, 166, 16),
        (106, 202, 222),
        (210, 16, 146),
    ]
    for i, (yy, uu, vv) in enumerate(bars):
        x0, x1 = i * W // 6, (i + 1) * W // 6
        y[:, x0:x1], u[:, x0 // 2 : x1 // 2], v[:, x0 // 2 : x1 // 2] = yy, uu, vv
    return y.tobytes() + u.tobytes() + v.tobytes()


def test_gpu_frames_are_written_without_a_colour_conversion(tmp_path: Path) -> None:
    out = tmp_path / "bars.mp4"
    frame = _colour_bars()
    info = VideoInfo(1280, 720, "60/1", False, False, 1.0)  # an SD-matrix DVR source
    enc = Encoder(out, W, H, info, overwrite=True, ready=True)
    for _ in range(10):
        enc.write_raw(frame)
    enc.close()
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(out), "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"],
        capture_output=True,
        check=True,
    ).stdout
    sent = np.frombuffer(frame, np.uint8).astype(float)
    got = np.frombuffer(raw[: len(frame)], np.uint8).astype(float)
    luma, chroma = slice(0, W * H), slice(W * H, len(frame))
    assert np.abs(got[luma] - sent[luma]).mean() <= 1.5
    assert np.abs(got[chroma] - sent[chroma]).mean() <= 1.5
