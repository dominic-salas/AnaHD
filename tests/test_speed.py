"""The speed shown under the settings: it reproduces the measured renders and warns below
the 10-minute rule (18 fps over a 3-minute flight)."""

from __future__ import annotations

import pytest

from anahd.app.speed import estimate, frame_ms
from anahd.remaster import NO_LOOK, RemasterParams

ALL_FOUR = RemasterParams(motion_blur=39, stabilisation=50, superview="stretch", sharpen=50)


def test_reproduces_the_measured_renders() -> None:
    """tools/remaster_speed.py, 2026-10-21: the parts, added up, give the measured speed
    of the combination they were not measured on."""
    assert 1000 / frame_ms(RemasterParams(), True) == pytest.approx(20.9, rel=0.01)
    assert 1000 / frame_ms(NO_LOOK, True) == pytest.approx(23.6, rel=0.01)
    assert 1000 / frame_ms(ALL_FOUR, True) == pytest.approx(18.9, rel=0.05)  # measured
    assert 1000 / frame_ms(ALL_FOUR, False) == pytest.approx(20.2, rel=0.05)


def test_warns_below_18_fps_and_not_at_the_defaults() -> None:
    fps, warning = estimate(RemasterParams(), True)
    assert fps > 18 and warning is None
    fps, warning = estimate(ALL_FOUR, True)
    assert fps < 18 and warning is not None and "10.1 MINUTES" in warning
    assert estimate(ALL_FOUR.with_(osd_redraw=True), False)[1] is not None


def test_each_effect_costs_time() -> None:
    base = estimate(RemasterParams(), True)[0]
    for change in (
        {"motion_blur": 39.0},
        {"stabilisation": 50.0},
        {"superview": "stretch"},
        {"sharpen": 50.0},
        {"osd_redraw": True},
    ):
        assert estimate(RemasterParams().with_(**change), True)[0] < base, change


def test_costs_are_each_effects_share_of_the_speed() -> None:
    from anahd.app.speed import costs

    shares = costs(RemasterParams(), True)
    assert set(shares) == {"motion_blur", "stabilisation", "superview", "sharpen", "osd_redraw"}
    assert all(0 < v < 15 for v in shares.values()), shares
    on = RemasterParams(motion_blur=39)
    expected = 100 * (1 - estimate(on, True)[0] / estimate(RemasterParams(), True)[0])
    assert shares["motion_blur"] == pytest.approx(expected)
    # an effect that is on shows what it costs as chosen, the same number
    assert costs(on, True)["motion_blur"] == pytest.approx(expected)
