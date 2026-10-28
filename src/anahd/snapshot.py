"""Small pictures of a clean while it runs, for the app's live before/after view.

The cleaner hands the writer packed 8-bit 4:2:0 frames (Y, then U, then V); the recording's
raw frames have the same layout at the picture area's size. ``to_bgr`` turns either into a
small BGR image with the right matrix and range, so the two sides of the view differ only
by the clean; ``Throttle`` decides which output frames get a picture (the first, then one
every ``EVERY`` seconds).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

EVERY = 1.5  # seconds between pictures
WIDTH = 640  # pixels across
QUALITY = 85  # JPEG


def to_bgr(
    buf: np.ndarray,
    w: int,
    h: int,
    *,
    full_range: bool,
    bt709: bool,
    width: int = WIDTH,
) -> np.ndarray:
    """A packed 4:2:0 frame (``w`` x ``h``) as uint8 BGR, ``width`` pixels across (aspect
    kept), using the frame's own matrix (BT.709 or BT.601) and range (full or limited)."""
    luma, chroma = w * h, (w // 2) * (h // 2)
    if buf.size < luma + 2 * chroma:
        raise ValueError(f"{buf.size} bytes is less than one {w}x{h} 4:2:0 frame")
    out_w = min(width, w)
    out_h = max(2, round(h * out_w / w))
    y = cv2.resize(buf[:luma].reshape(h, w), (out_w, out_h), interpolation=cv2.INTER_AREA)
    u = buf[luma : luma + chroma].reshape(h // 2, w // 2)
    v = buf[luma + chroma : luma + 2 * chroma].reshape(h // 2, w // 2)
    u = cv2.resize(u, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    v = cv2.resize(v, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    yf = y.astype(np.float32)
    cb = u.astype(np.float32) - 128.0
    cr = v.astype(np.float32) - 128.0
    if not full_range:
        yf = (yf - 16.0) * (255.0 / 219.0)
        cb *= 255.0 / 224.0
        cr *= 255.0 / 224.0
    kr, kb = (0.2126, 0.0722) if bt709 else (0.299, 0.114)
    kg = 1.0 - kr - kb
    r = yf + 2.0 * (1.0 - kr) * cr
    b = yf + 2.0 * (1.0 - kb) * cb
    g = (yf - kr * r - kb * b) / kg
    bgr: np.ndarray = np.clip(np.stack([b, g, r], -1), 0, 255).round().astype(np.uint8)
    return bgr


def jpeg(bgr: np.ndarray, quality: int = QUALITY) -> bytes:
    ok, data = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("could not encode the picture")
    return bytes(data.tobytes())


@dataclass
class Throttle:
    """The first call is due, then one at least ``every`` seconds after the last due one."""

    every: float = EVERY
    last: float | None = None

    def due(self, now: float) -> bool:
        if self.last is not None and now - self.last < self.every:
            return False
        self.last = now
        return True
