"""Breakup detection and repair on the GPU. Skipped without torch and a CUDA GPU."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():  # pragma: no cover - depends on the machine
    pytest.skip("no CUDA GPU", allow_module_level=True)

from torch import Tensor  # noqa: E402

from anahd.clean import CleanConfig  # noqa: E402
from anahd.gpu.breakup import (  # noqa: E402
    BreakupConfig,
    fill_rows,
    row_mask,
    second_smallest,
    trimmed_mean,
)
from anahd.gpu.pipeline import Cleaner, GpuConfig, Prepared  # noqa: E402
from anahd.look import LookConfig  # noqa: E402
from anahd.media import Crop, VideoInfo  # noqa: E402
from anahd.pipeline import _windows  # noqa: E402

pytestmark = pytest.mark.gpu
DEV = torch.device("cuda")
W, H = 480, 360  # RAFT needs the half-size frame to stay above ~200x150
BAND = slice(150, 174)  # rows of the snow band


def test_second_smallest_matches_kthvalue() -> None:
    x = torch.rand(5, 30, 40, device=DEV)
    assert torch.equal(second_smallest(x), x.kthvalue(2, 0).values)


def test_trimmed_mean_drops_the_extremes() -> None:
    x = torch.tensor([[1.0], [2.0], [3.0], [100.0]], device=DEV)
    assert trimmed_mean(x).item() == pytest.approx(2.5)


def test_fill_rows_interpolates_between_good_rows() -> None:
    x = torch.arange(10, dtype=torch.float32, device=DEV)[:, None].expand(10, 3).clone()
    bad = torch.zeros(10, dtype=torch.bool, device=DEV)
    bad[3:7] = True
    x[3:7] = 99.0
    out = fill_rows(x, bad)
    assert torch.allclose(out, torch.arange(10, dtype=torch.float32, device=DEV)[:, None])


def test_thin_streaks_are_repaired_narrowly() -> None:
    cfg = BreakupConfig()
    flags = torch.zeros(100, dtype=torch.bool, device=DEV)
    flags[20] = True  # a 1-row streak
    flags[60:70] = True  # a wide band
    m = row_mask(flags, cfg)
    assert torch.count_nonzero(m[:40]).item() <= 7  # the streak, 1 row each side, 2-row fade
    assert torch.count_nonzero(m[40:]).item() >= 10 + 2 * cfg.dilate


def _scene(t: int) -> np.ndarray:
    """A smooth random texture panning 1.5 pixels per frame (luma, 0..255)."""
    rng = np.random.default_rng(7)
    noise = rng.uniform(0, 255, (H + 40, W + 80)).astype(np.float32)
    big = np.asarray(cv2.GaussianBlur(noise, (0, 0), 2.0), dtype=np.float32)
    lo, hi = float(np.min(big)), float(np.max(big))
    big = np.asarray(20.0 + (big - lo) * (215.0 / (hi - lo)), dtype=np.float32)
    m = np.array([[1, 0, -10 - 1.5 * t], [0, 1, -20]], dtype=np.float32)
    return np.asarray(cv2.warpAffine(big, m, (W, H), flags=cv2.INTER_CUBIC), dtype=np.float32)


def _raw(y: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    planes = [np.clip(p, 0, 255).round().astype(np.uint8).ravel() for p in (y, u, v)]
    return np.concatenate(planes)


def _frames(snow_at: int | None) -> tuple[list[np.ndarray], list[np.ndarray]]:
    rng = np.random.default_rng(11)
    raws, clean = [], []
    for t in range(9):
        y = _scene(t)
        u = np.full((H // 2, W // 2), 128.0) + 20.0 * np.sin(np.arange(W // 2) / 9.0)[None, :]
        v = np.full((H // 2, W // 2), 120.0)
        clean.append(y)
        if t == snow_at:  # a band of colour snow
            y = y.copy()
            y[BAND] = rng.uniform(0, 255, y[BAND].shape)
            u, v = u.copy(), v.copy()
            u[BAND.start // 2 : BAND.stop // 2] = rng.uniform(0, 255, (12, W // 2))
        raws.append(_raw(y, u, v))
    return raws, clean


def _cleaner(repair: bool) -> Cleaner:
    info = VideoInfo(W, H, "60/1", True, False, 1.0)
    curve = np.arange(256, dtype=np.float32)
    osd = np.zeros((H, W), bool)
    gcfg = GpuConfig(repair=repair, graphs=False)
    return Cleaner(
        Crop(0, 0, W, H), osd, curve, (W, H), info, CleanConfig(), LookConfig(), gcfg, DEV
    )


def _fused(c: Cleaner, raws: list[np.ndarray]) -> list[Tensor]:
    """The centre planes after fusion (repaired first when the cleaner repairs)."""
    out = []
    with torch.inference_mode():
        prepared: list[Prepared] = [c.prepare(r) for r in raws]
        for centre, nbs in _windows(iter(prepared), c.cfg.radius):
            if c.detector is None:
                y, _, _ = c.fuse(centre, nbs)
            else:
                a = c.repair(c.detector, centre, c.align(centre, nbs))
                y, _, _ = c.blend(centre.y, centre.u, centre.v, a)
            out.append(y.clone())
    return out


def test_a_snow_band_is_repaired() -> None:
    raws, clean = _frames(snow_at=4)
    truth = torch.from_numpy(clean[4]).to(DEV)
    before = _fused(_cleaner(False), raws)[4]
    after = _fused(_cleaner(True), raws)[4]
    inner = slice(8, W - 8)
    err_before = (before[BAND, inner] - truth[BAND, inner]).abs().mean().item()
    err_after = (after[BAND, inner] - truth[BAND, inner]).abs().mean().item()
    assert err_before > 20.0
    assert err_after < err_before / 5


def test_a_clean_sequence_is_left_alone() -> None:
    raws, _ = _frames(snow_at=None)
    off = _fused(_cleaner(False), raws)
    on = _fused(_cleaner(True), raws)
    for a, b in zip(off, on, strict=True):
        assert (a - b).abs().max().item() < 1.0


def test_graph_replay_matches_eager() -> None:
    raws, _ = _frames(snow_at=4)
    eager = _fused(_cleaner(True), raws)
    c = _cleaner(True)
    c.gcfg = GpuConfig(repair=True, graphs=True)
    graphed = _fused(c, raws)
    for a, b in zip(eager, graphed, strict=True):
        assert (a - b).abs().max().item() < 1e-3
