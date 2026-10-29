"""GPU steps against their OpenCV/CPU counterparts. Skipped without torch and a CUDA GPU."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from anahd import clean, look
from anahd.clean import CleanConfig
from anahd.look import LookConfig

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():  # pragma: no cover - depends on the machine
    pytest.skip("no CUDA GPU", allow_module_level=True)

import torch.nn.functional as F  # noqa: E402
from torch import Tensor  # noqa: E402

from anahd.gpu import ops  # noqa: E402
from anahd.gpu.pipeline import to_bt709  # noqa: E402

pytestmark = pytest.mark.gpu
DEV = torch.device("cuda")
RNG = np.random.default_rng(3)


def _t(a: np.ndarray) -> Tensor:
    t: Tensor = torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(DEV)
    return t


def _n(t: Tensor) -> np.ndarray:
    return np.asarray(t.float().cpu().numpy())


def _img(h: int = 72, w: int = 96) -> np.ndarray:
    return np.asarray(RNG.uniform(0, 255, (h, w)), dtype=np.float32)


@pytest.mark.parametrize(("sigma", "size"), [(1.2, (5, 5)), (1.0, None), (2.0, None)])
def test_gaussian_matches_opencv(sigma: float, size: tuple[int, int] | None) -> None:
    x = _img()
    ref = cv2.GaussianBlur(x, size or (0, 0), sigma)
    assert np.abs(_n(ops.gaussian(_t(x), sigma, size=size)) - ref).max() < 1e-3


def test_box_filters_match_opencv() -> None:
    x = _img()
    assert np.abs(_n(ops.box_mean(_t(x), 5)) - cv2.boxFilter(x, -1, (5, 5))).max() < 1e-3
    ref = cv2.boxFilter(x, -1, (41, 1), normalize=False, borderType=cv2.BORDER_REFLECT)
    assert np.abs(_n(ops.row_box_sum(_t(x), 41)) - ref).max() < 1e-2


def test_destripe_and_derainbow_match_cpu() -> None:
    y, u, v = _img(), _img(36, 48), _img(36, 48)
    osd = np.zeros(y.shape, bool)
    osd[4:12, 8:30] = True
    cfg = CleanConfig()
    from anahd.gpu.pipeline import Cleaner

    # Only the per-plane methods are used; build the few fields they read.
    c = Cleaner.__new__(Cleaner)
    c.cfg = cfg
    c.osd = _t(osd).bool()
    c.keep = (~c.osd).float()
    c.osd_half = _t(cv2.resize(osd.astype(np.uint8), (48, 36), interpolation=0)).bool()
    c.alpha = c.shown = c.osd.float()  # a hard switch, as the CPU path makes
    c.shown_half = c.osd_half.float()
    assert np.abs(_n(c.destripe(_t(y))) - clean.destripe(y, osd, cfg)).max() < 1e-2
    gu, gv = c.derainbow(_t(y), _t(u), _t(v))
    ru, rv = clean.derainbow(y, u, v, _n(c.osd_half).astype(bool), cfg)
    assert np.abs(_n(gu) - ru).max() < 1e-2
    assert np.abs(_n(gv) - rv).max() < 1e-2


@pytest.mark.parametrize("kind", ["lanczos4", "linear", "cubic"])
def test_resampler_matches_warp_affine(kind: str) -> None:
    # Video-like detail: on white noise OpenCV's 1/32-pixel position rounding dominates.
    x = np.asarray(cv2.GaussianBlur(_img(), (0, 0), 1.0), dtype=np.float32)
    size = (144, 108)
    flags = {"lanczos4": cv2.INTER_LANCZOS4, "linear": cv2.INTER_LINEAR, "cubic": cv2.INTER_CUBIC}
    m = look.scale_matrix((x.shape[1], x.shape[0]), size)
    ref = cv2.warpAffine(x, m, size, flags=flags[kind], borderMode=cv2.BORDER_REPLICATE)
    got = _n(ops.Resampler((x.shape[1], x.shape[0]), size, kind, DEV)(_t(x)))
    # OpenCV quantizes sample positions to 1/32 pixel; the GPU path does not.
    assert np.abs(got - ref)[4:-4, 4:-4].mean() < 0.25


def test_tone_curve_matches_cpu() -> None:
    curve = look.tone_curve(look.Levels(10.0, 230.0), LookConfig())
    y = _img()
    got = _n(ops.apply_curve(_t(y), _t(curve)))
    assert np.abs(got - look.apply_tone(y, curve)).max() < 1e-3


def test_colour_conversion_keeps_grey_grey() -> None:
    m = to_bt709(False)
    out = m @ np.array([0.5, 0.0, 0.0])
    assert np.allclose(out, [0.5, 0.0, 0.0], atol=1e-9)


def test_median3_matches_opencv() -> None:
    x = np.asarray(RNG.uniform(0, 255, (40, 50)).round(), dtype=np.float32)
    ref = cv2.medianBlur(x, 3)  # float input: edge pixels repeated
    assert np.abs(_n(ops.median3(_t(x))) - ref).max() < 1e-3


def test_dilate_is_a_square_max_in_two_passes() -> None:
    x = _t((_img() > 200).astype(np.float32))
    for r in (1, 4, 28):
        square = F.max_pool2d(x[None, None], 2 * r + 1, 1, r)[0, 0]
        assert torch.equal(ops.dilate(x, r), square), r
