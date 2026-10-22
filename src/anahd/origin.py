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


@dataclass(frozen=True)
class Origin:
    digital: bool
    osd_share: float

    @property
    def reasons(self) -> str:
        return "no OSD" if self.digital else ""


def judge(info: VideoInfo, meas: Measured) -> Origin:
    osd = float(meas.osd.mean()) if meas.osd.size else 0.0
    return Origin(osd < NO_OSD, osd)
