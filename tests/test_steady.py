"""Steadying plans on synthetic camera paths: jitter is removed, tricks are left alone."""

from __future__ import annotations

import math

import numpy as np

from anahd.look import scale_matrix
from anahd.steady import Plan, SteadyConfig, Step, matrix, plan

W, H = 960, 720


def _steps(dx: np.ndarray, da: np.ndarray | None = None) -> list[Step]:
    da = np.zeros_like(dx) if da is None else da
    return [Step(float(x), 0.0, float(a), True) for x, a in zip(dx, da, strict=True)]


def test_small_jitter_on_a_smooth_pan_is_cancelled() -> None:
    n = 120
    rng = np.random.default_rng(1)
    pan = np.full(n, 2.0)
    jitter = rng.normal(0, 1.0, n)
    path = np.cumsum(pan) + jitter
    dx = np.diff(path, prepend=0.0)
    p = plan(_steps(dx), W, H, SteadyConfig())
    steadied = path + p.cx
    inner = slice(20, -20)
    before = np.std(np.diff(path[inner], 2))
    after = np.std(np.diff(steadied[inner], 2))
    assert after < 0.5 * before
    assert np.all(np.abs(p.cx) <= SteadyConfig().max_shift * W + 1e-9)


def test_a_fast_roll_is_not_damped() -> None:
    n = 120
    da = np.zeros(n)
    da[50:70] = math.radians(18.0)  # a 360 degree roll over 20 frames
    p = plan(_steps(np.zeros(n), da), W, H, SteadyConfig())
    assert np.all(p.ca[45:75] == 0.0)
    assert np.all(p.strength[45:75] == 0.0)


def test_unmeasured_frames_are_not_steadied() -> None:
    steps = [Step(0.5 * (-1) ** i, 0.0, 0.0, i != 60) for i in range(120)]
    p = plan(steps, W, H, SteadyConfig())
    assert p.strength[60] == 0.0 and p.cx[60] == 0.0


def test_no_correction_is_a_plain_resize() -> None:
    zeros = np.zeros(3)
    p = Plan(zeros, zeros, zeros, 1.0, zeros)
    assert np.allclose(matrix(p, 0, (W, H), (1440, 1080)), scale_matrix((W, H), (1440, 1080)))
