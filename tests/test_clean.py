"""Cleaning steps on synthetic frames: noise goes down, OSD and stripes behave."""

from __future__ import annotations

import numpy as np

from anahd import clean, look
from anahd.clean import CleanConfig
from anahd.media import Frame
from anahd.picture import output_size
from anahd.pipeline import _windows

H, W = 120, 160
RNG = np.random.default_rng(7)


def _scene(shift: int = 0) -> np.ndarray:
    x = np.arange(W + 64)
    y = np.arange(H)
    base = 128 + 60 * np.sin(x[None, :] / 9.0) * np.cos(y[:, None] / 13.0)
    return base[:, shift : shift + W]


def _frame(y: np.ndarray, noise: float = 0.0) -> Frame:
    noisy = np.clip(y + RNG.normal(0, noise, y.shape) if noise else y, 0, 255).astype(np.uint8)
    chroma = np.full((H // 2, W // 2), 128, np.uint8)
    return Frame(noisy, chroma, chroma.copy())


def test_fusion_reduces_noise_on_a_panning_scene() -> None:
    osd = np.zeros((H, W), bool)
    fuser = clean.Fuser(osd, CleanConfig())
    frames = [fuser.prepare(_frame(_scene(10 + 2 * k), noise=6.0)) for k in range(7)]
    y, _, _ = fuser.fuse(frames[3], [(k - 3, f) for k, f in enumerate(frames) if k != 3])
    truth = _scene(16)
    inner = (slice(12, -12), slice(12, -12))
    before = np.abs(frames[3].y - truth)[inner].mean()
    after = np.abs(y - truth)[inner].mean()
    assert after < 0.7 * before


def test_osd_is_neither_fused_nor_smeared() -> None:
    osd = np.zeros((H, W), bool)
    osd[10:20, 20:60] = True
    fuser = clean.Fuser(osd, CleanConfig())

    def with_text(shift: int, text: int) -> Frame:
        y = _scene(shift)
        y[10:20, 20:60] = text
        return _frame(y)

    center = fuser.prepare(with_text(10, 250))
    nbs = [(d, fuser.prepare(with_text(10 + 3 * d, 0))) for d in (-3, -2, -1, 1, 2, 3)]
    y, _, _ = fuser.fuse(center, nbs)
    assert np.allclose(y[osd], 250)  # the current frame's OSD is kept exactly
    outside = ~osd
    truth = _scene(10)
    assert np.abs(y - truth)[outside][30:].max() < 40  # no black OSD pasted from neighbours


def test_destripe_removes_row_offsets_and_keeps_the_picture() -> None:
    clean_y = _scene().astype(np.float32)
    striped = clean_y.copy()
    striped[40] += 4.0
    striped[71] -= 3.0
    out = clean.destripe(striped, np.zeros((H, W), bool), CleanConfig(stripe_window=W + 1))
    assert abs(out[40] - clean_y[40]).mean() < 2.0
    assert abs(out[71] - clean_y[71]).mean() < 1.5
    assert np.abs(out - clean_y).mean() < 1.0


def test_windows_give_every_frame_its_neighbours() -> None:
    items = list(range(10))
    got = list(_windows(iter(items), 3))
    assert [c for c, _ in got] == items
    assert [d for d, _ in got[0][1]] == [1, 2, 3]
    assert [d for d, _ in got[5][1]] == [-3, -2, -1, 1, 2, 3]
    assert [n for _, n in got[9][1]] == [6, 7, 8]


def test_output_keeps_the_pictures_shape() -> None:
    from anahd.media import Crop

    assert output_size(Crop(162, 0, 956, 720)) == (1440, 1080)
    assert output_size(Crop(0, 0, 1280, 720)) == (1920, 1080)


def test_tone_curve_is_monotonic_and_sharpening_does_not_overshoot() -> None:
    curve = look.tone_curve(look.Levels(20.0, 230.0), look.LookConfig())
    assert np.all(np.diff(curve) >= 0)
    step = np.zeros((40, 40), np.float32)
    step[:, 20:] = 200.0
    big = look.upscale_luma(step, (60, 60), look.LookConfig())
    assert float(np.min(big)) >= -2.5 and float(np.max(big)) <= 202.5
