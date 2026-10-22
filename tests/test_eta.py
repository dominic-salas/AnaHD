"""The time left shown while a clip is cleaned counts down steadily."""

from __future__ import annotations

import itertools

import numpy as np

from anahd.app.eta import Eta


def _run(
    startup_s: float, fps: float, total: int, jitter: float, seed: int = 0
) -> list[tuple[float, float | None]]:
    """Reports every 30 frames like the worker: (time, seconds left) as shown."""
    rng = np.random.default_rng(seed)
    eta, shown = Eta(), []
    for done in range(30, total + 1, 30):
        # a slow first stretch (start-up, CUDA graphs, first neighbours), then steady
        t = startup_s * min(done, 120) / 120 + max(done - 120, 0) / fps
        t += rng.uniform(-jitter, jitter)
        shown.append((t, eta.update(done, total, t)[1]))
    return shown


def test_time_left_counts_down_steadily() -> None:
    shown = _run(startup_s=12.0, fps=24.0, total=11032, jitter=0.08)
    settled = [(t, left) for t, left in shown if left is not None]
    assert settled, "never showed a time left"
    for (t0, a), (t1, b) in itertools.pairwise(settled):
        assert b - a <= 1.0, f"jumped up {b - a:.1f} s at {t1:.0f} s"
        assert a - b <= (t1 - t0) + 1.0 + 1e-6, (
            f"jumped down {a - b - (t1 - t0):.1f} s at {t1:.0f} s"
        )
    # and it is right: at 10 % the remaining work takes 90 % of 11032 / 24 s
    t10, left10 = next((t, left) for t, left in settled if t > 60)
    true_left = 11032 / 24.0 - (t10 - 12.0 + 120 / 24.0)
    assert abs(left10 - true_left) < 5.0


def test_no_time_left_until_the_speed_has_settled() -> None:
    eta = Eta()
    assert eta.update(30, 10000, 0.0)[1] is None
    # 4 s at a steady 24 fps: not yet (two 5 s stretches are needed to call it steady)
    for i in range(1, 5):
        assert eta.update(30 + 24 * i, 10000, float(i))[1] is None
    # a further 6 s at the same speed: steady, so a time left is shown, and it is right
    for i in range(5, 11):
        left = eta.update(30 + 24 * i, 10000, float(i))[1]
    assert left is not None
    assert abs(left - (10000 - 30 - 240) / 24) < 1.0


def test_a_speed_still_ramping_up_shows_no_time_left() -> None:
    """While the speed keeps climbing (start-up), any estimate would be too long and then
    fall fast: nothing is shown until it levels off."""
    eta, done, t = Eta(), 0.0, 0.0
    for _ in range(60):  # 30 s, speed rising 2 % per second
        t += 0.5
        done += 0.5 * 10 * 1.02**t
        assert eta.update(int(done), 100000, t)[1] is None
    for _ in range(30):  # then level: shown within ~10 s
        t += 0.5
        done += 0.5 * 10 * 1.02**30
        left = eta.update(int(done), 100000, t)[1]
    assert left is not None


def test_a_retry_starts_the_estimate_over() -> None:
    eta = Eta()
    for i in range(1, 20):
        eta.update(30 * i, 1000, i * 1.25)
    fps, _ = eta.update(30, 1000, 30.0)  # the clip restarted from frame 0
    assert fps == 0.0
