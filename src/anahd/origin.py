"""Does a recording look digital rather than analog?

AnaHD is made for analog video; a digital recording gains nothing from it. The sign is
the OSD: an analog goggle DVR records the OSD burned into the picture, and a
digital camera's recordings have none. The picture's shape does not help: a digital
camera is 4:3 like the analog ones (2026-10-20).

The cost of a wrong flag is small by design: a flagged clip is left out of CLEAN until
INCLUDE ANYWAY. An analog flight recorded with the OSD switched off is flagged too.
Known limit: the OSD finder looks for bright detail that stays put, so a digital clip of
a nearly still scene shows some "OSD" and is not flagged (it is then just cleaned;
tests/test_origin.py pins this down). On 23 real recordings: one has no OSD at all
and would be flagged.
"""

from __future__ import annotations

from dataclasses import dataclass

from anahd.media import VideoInfo
from anahd.pipeline import Measured

NO_OSD = 0.002  # share of the picture: below this, no OSD was found
HDZERO = "HDZero digital recording"


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
    for them; 720p at 50/60 fps stays undecided."""
    return 89.0 <= _fps(info) <= 91.0 or (info.width == 1920 and info.height == 1080)


def judge(info: VideoInfo, meas: Measured) -> Origin:
    osd = float(meas.osd.mean()) if meas.osd.size else 0.0
    if is_hdzero(info):
        return Origin(True, osd, True)
    return Origin(osd < NO_OSD, osd)
