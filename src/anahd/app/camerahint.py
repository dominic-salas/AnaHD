"""A quiet suggestion for recordings from a camera the Learned cleaning was not trained on.

The learned model was trained on 60 fps recordings of one camera. A RaceCam R1 Mini
records 4:3 at 50 fps; on its footage the Learned cleaning smears fine detail (bare
branches become soft blobs) where Classic keeps it. The rule is the recording's shape:
a 4:3 picture at 50 fps. 16:9 recordings at 50 fps and all 60 fps recordings are left
alone. It only ever suggests; the cleaning is never switched by itself.
"""

from __future__ import annotations

from anahd.media import VideoInfo

HINT = (
    "This looks like a camera the Learned cleaning wasn't trained on; "
    "Classic may keep fine detail better."
)
FPS = 50.0
FPS_TOLERANCE = 0.5
FOUR_THIRDS = (1920, 1440)  # the output size of a 4:3 picture


def _fps(info: VideoInfo) -> float:
    num, _, den = info.fps.partition("/")
    try:
        return float(num) / float(den or 1)
    except (ValueError, ZeroDivisionError):
        return 0.0


def hint_for(info: VideoInfo, output: tuple[int, int]) -> str:
    """The suggestion for a recording (its probe and the output size AnaHD would make
    from its picture), or an empty string."""
    four_thirds = output[0] * 3 == output[1] * 4
    return HINT if four_thirds and abs(_fps(info) - FPS) <= FPS_TOLERANCE else ""
