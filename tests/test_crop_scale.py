"""Inputs taller than 720 lines are delivered at 720 (an analog picture has ~480)."""

from __future__ import annotations

from anahd.media import Crop
from anahd.pipeline import MAX_PICTURE_H


def test_tall_picture_is_scaled_to_720_keeping_its_shape() -> None:
    c = Crop(240, 0, 1440, 1080).scaled_to(MAX_PICTURE_H)
    assert (c.width, c.height) == (960, 720)
    assert c.filter == "crop=1440:1080:240:0,scale=960:720:flags=area"


def test_hd_and_sd_pictures_are_left_alone() -> None:
    for c in (Crop(160, 0, 960, 720), Crop(0, 0, 720, 480)):
        assert c.scaled_to(MAX_PICTURE_H) is c
        assert "scale" not in c.filter
