"""Estimated render speed for the remaster settings, shown under the settings panel.

Built from measurements (``tools/remaster_speed.py``, RTX 4070 Ti, a 10 s 1280x720 DVR
recording, 2026-10-21, best of two runs for Learned): the time per frame of the cleaning
with no look, plus what each remaster part adds, plus the fixed start-up and the
stabilisation's pass over the recording, spread over a 3-minute flight. So the number is
the speed of a whole 3-minute flight, the one the 10-minute rule is about.

Measured steady speeds (fps), Learned / Classic: look off 23.6 / 24.8; defaults 20.9 /
21.7; + motion blur 19.7 / 20.6; + stabilisation 20.5 / 21.2; + SuperView 20.5 / 20.6;
+ sharpening 20.7 / 22.3; all four 18.9 / 20.2. The parts added up predict all four at
18.7 / 19.9 fps.
"""

from __future__ import annotations

from anahd.remaster import RemasterParams

RULE_FPS = 18.0  # a 3-minute flight in 10 minutes
FLIGHT_FRAMES = 180 * 60

# ms per frame, from the measurements above
BASE_MS = {True: 1000 / 23.6, False: 1000 / 24.8}  # cleaning only (Learned, Classic)
LOOK_MS = {True: 1000 / 20.9 - 1000 / 23.6, False: 1000 / 21.7 - 1000 / 24.8}
EFFECT_MS = {  # what each effect adds to the defaults
    "motion_blur": {True: 1000 / 19.7 - 1000 / 20.9, False: 1000 / 20.6 - 1000 / 21.7},
    "stabilisation": {True: 1000 / 20.5 - 1000 / 20.9, False: 1000 / 21.2 - 1000 / 21.7},
    "superview": {True: 1000 / 20.5 - 1000 / 20.9, False: 1000 / 20.6 - 1000 / 21.7},
    "sharpen": {True: 1000 / 20.7 - 1000 / 20.9, False: 0.0},  # within noise for Classic
    "osd_redraw": {True: 3.2, False: 3.2},  # engine measurement
}
START_S = 9.0  # measuring the clip, loading models, the first frames
STEADY_PASS_MS = 2.0  # stabilisation's pass over the recording, per frame of the clip


def _on(params: RemasterParams) -> dict[str, bool]:
    return {
        "motion_blur": params.motion_blur > 0,
        "stabilisation": params.stabilisation > 0,
        "superview": params.superview != "off",
        "sharpen": params.sharpen > 0,
        "osd_redraw": params.osd_redraw,
    }


def frame_ms(params: RemasterParams, learned: bool) -> float:
    """Steady time per frame."""
    ms = BASE_MS[learned]
    if params.look_on:
        ms += LOOK_MS[learned]
    for name, on in _on(params).items():
        if on:
            ms += EFFECT_MS[name][learned]
    return ms


def _flight(params: RemasterParams, learned: bool) -> tuple[float, float]:
    """Frames per second and seconds for a 3-minute flight."""
    per_frame = frame_ms(params, learned)
    if params.stabilisation > 0:
        per_frame += STEADY_PASS_MS
    seconds = START_S + FLIGHT_FRAMES * per_frame / 1000
    return FLIGHT_FRAMES / seconds, seconds


# the model counts each effect as on or off; these values switch it
_OFF: dict[str, object] = {
    "motion_blur": 0,
    "stabilisation": 0,
    "superview": "off",
    "sharpen": 0,
    "osd_redraw": False,
}
_ON: dict[str, object] = {
    "motion_blur": 39,
    "stabilisation": 50,
    "superview": "stretch",
    "sharpen": 50,
    "osd_redraw": True,
}


def costs(params: RemasterParams, learned: bool) -> dict[str, float]:
    """Each effect's speed cost in %: how much lower the speed is with it on than off,
    the other settings as chosen."""
    chosen = _on(params)
    out = {}
    for name, off in _OFF.items():
        with_it = params if chosen[name] else params.with_(**{name: _ON[name]})
        on_fps, _ = _flight(with_it, learned)
        off_fps, _ = _flight(params.with_(**{name: off}), learned)
        out[name] = 100 * (1 - on_fps / off_fps)
    return out


def estimate(params: RemasterParams, learned: bool) -> tuple[float, str | None]:
    """Frames per second over a 3-minute flight, and a warning below the 10-minute rule."""
    fps, seconds = _flight(params, learned)
    if fps >= RULE_FPS:
        return fps, None
    minutes = seconds / 60
    return fps, (
        f"Below {RULE_FPS:.0f} fps: a 3-minute flight takes about {minutes:.1f} minutes "
        "with these effects (more than 10)"
    )
