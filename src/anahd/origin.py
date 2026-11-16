"""Does a recording look digital rather than analog?

AnaHD is made for analog video. A digital (HDZero) recording has none of the analog
damage and is upscaled without the analog cleaning instead (anahd.gpu.digital). Three
signs, from the most to the least certain:

1. HDZero firmware's own file shapes: 90 fps, or a 1920x1080 picture. Analog DVRs record
   720x480, 720x576 or 1280x720 at 50 or 60 fps, so these never fire for them.
2. A 1280x720 recording at 60 fps looks the same for HDZero and for an analog DVR, so the
   picture decides: HDZero carries real detail up to the pixel grid, an analog picture is
   band-limited and upscaled by the goggles. :func:`looks_digital` measures how much of
   the detail in textured tiles sits in the top spectral bands, horizontally and vertically.
   It is tuned to call no analog flight digital (see tests/test_origin.py and the numbers
   in the changelog); a digital clip of a dark or blurry scene can be missed, and is then
   just cleaned as analog.
3. No OSD at all: an analog goggle DVR records the OSD burned in, a digital camera's
   recordings have none. This one only leaves the clip out of CLEAN (INCLUDE ANYWAY
   cleans it); an analog flight recorded with the OSD switched off is flagged too. Known
   limit: the OSD finder looks for bright detail that stays put, so a digital clip of a
   nearly still scene shows some "OSD" and is not flagged by this sign.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from anahd.media import VideoInfo
from anahd.pipeline import Measured

NO_OSD = 0.002  # share of the picture: below this, no OSD was found
HDZERO = "HDZero digital recording"
HDZERO_NOTE = "HDZero digital recording: upscaled without analog cleaning"


@dataclass(frozen=True)
class Origin:
    digital: bool
    osd_share: float
    hdzero: bool = False

    @property
    def reasons(self) -> str:
        if self.hdzero:
            return HDZERO
        return "no OSD" if self.digital else ""


def _fps(info: VideoInfo) -> float:
    num, _, den = info.fps.partition("/")
    try:
        return float(num) / float(den or 1)
    except (ValueError, ZeroDivisionError):
        return 0.0


def is_hdzero(info: VideoInfo) -> bool:
    """HDZero goggles record 90 fps, or a 1920x1080 picture (HDZero firmware dvr.c).
    Analog DVRs record 720x480, 720x576 or 1280x720 at 50/60 fps, so neither rule fires
    for them; 720p at 60 fps is left to the picture (:func:`looks_digital`)."""
    return 89.0 <= _fps(info) <= 91.0 or (info.width == 1920 and info.height == 1080)


def judge(info: VideoInfo, meas: Measured) -> Origin:
    osd = float(meas.osd.mean()) if meas.osd.size else 0.0
    samples = getattr(meas, "samples", None)
    if is_hdzero(info) or (maybe_hdzero_shape(info) and samples and looks_digital(samples)):
        return Origin(True, osd, True)
    return Origin(osd < NO_OSD, osd)


def maybe_hdzero_shape(info: VideoInfo) -> bool:
    """720p60, the one shape HDZero and an analog DVR share."""
    return info.height == 720 and 59.0 <= _fps(info) <= 60.1


# --- the picture's detail ---

TILE = 64
MIN_STD = 5.0  # flat tiles say nothing about detail
DARK, BRIGHT = 25.0, 235.0  # nor do crushed or clipped ones
MIN_TILES = 8  # per frame
_FRAMES = 24  # how many of the measured samples are looked at
# Median frame score above which a recording looks digital. Measured through measure() on 108
# labelled analog recordings (three cameras) the highest median is -1.80 (-2.11 over 300 more
# sampled by hand); this keeps 0.5 of margin. It finds 35 of 40 labelled HDZero recordings
# (the misses are dark or blurry scenes).
THRESHOLD = -1.3

_win = np.outer(np.hanning(TILE), np.hanning(TILE)).astype(np.float32)
_fy = np.abs(np.fft.fftfreq(TILE))
_fx = np.fft.rfftfreq(TILE)
_mid = (_fy >= 0.32) & (_fy < 0.42)
_ref = (_fy >= 0.12) & (_fy < 0.22)
_low_x = _fx < 0.12


def _tiles(plane: NDArray[np.uint8]) -> NDArray[np.float32]:
    h, w = plane.shape
    rows, cols = h // TILE, w // TILE
    t = plane[: rows * TILE, : cols * TILE].reshape(rows, TILE, cols, TILE)
    return np.asarray(t.swapaxes(1, 2).reshape(-1, TILE, TILE), dtype=np.float32)


def frame_score(plane: NDArray[np.uint8]) -> float | None:
    """Vertical detail near the pixel grid against the mid bands, in textured tiles: the
    log ratio of the power in 0.32..0.42 cycles/pixel to that in 0.12..0.22 (75th
    percentile over the tiles). None when the frame has too few textured tiles."""
    t = _tiles(plane[8:-8, 8:-8])
    mean, std = t.mean((1, 2)), t.std((1, 2))
    t = t[(std > MIN_STD) & (mean > DARK) & (mean < BRIGHT)]
    if len(t) < MIN_TILES:
        return None
    spec = np.abs(np.fft.rfft2((t - t.mean((1, 2), keepdims=True)) * _win)) ** 2
    down = spec[:, :, _low_x].sum(2)  # vertical profile of the low horizontal frequencies
    ratio = np.log(down[:, _mid].sum(1) + 1e-6) - np.log(down[:, _ref].sum(1) + 1e-6)
    return float(np.percentile(ratio, 75))


def looks_digital(samples: list[NDArray[np.uint8]]) -> bool:
    """True when the measured luma samples show the detail of a digital picture: the median
    frame score is above :data:`THRESHOLD`."""
    pick = np.linspace(0, len(samples) - 1, min(_FRAMES, len(samples))).round().astype(int)
    scores = [s for i in pick if (s := frame_score(samples[i])) is not None]
    return len(scores) >= 3 and float(np.median(scores)) > THRESHOLD
