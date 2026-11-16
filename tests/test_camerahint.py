"""The suggestion for recordings that look like a RaceCam R1 Mini (4:3 at 50 fps)."""

from __future__ import annotations

import pytest

from anahd.app.camerahint import HINT, hint_for
from anahd.media import VideoInfo

FOUR_THIRDS = (1920, 1440)
WIDE = (2560, 1440)


def _info(fps: str) -> VideoInfo:
    return VideoInfo(1280, 720, fps, False, False, 60.0)


def test_a_4_3_picture_at_50_fps_gets_the_suggestion() -> None:
    assert hint_for(_info("50/1"), FOUR_THIRDS) == HINT


@pytest.mark.parametrize("fps", ["60/1", "60000/1001", "90/1", "30/1"])
def test_other_frame_rates_get_none(fps: str) -> None:
    assert hint_for(_info(fps), FOUR_THIRDS) == ""


def test_a_16_9_picture_at_50_fps_gets_none() -> None:
    assert hint_for(_info("50/1"), WIDE) == ""


def test_a_broken_frame_rate_gets_none() -> None:
    assert hint_for(_info("0/0"), FOUR_THIRDS) == ""
    assert hint_for(_info("x"), FOUR_THIRDS) == ""


def test_the_suggestion_only_suggests() -> None:
    assert "Classic" in HINT and "Learned" in HINT
