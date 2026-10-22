"""Breakup bands: rows of colour snow where the analog signal dropped out for a moment.

A band in the centre frame disagrees with every aligned neighbour, while a band in one
neighbour disagrees only with the centre's view of that neighbour. So per pixel the damage
is the second-smallest difference to the nearest neighbours, and per row its mean outside
the OSD. Snow also has much more fine horizontal detail than the same rows of the
neighbours.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from anahd.gpu import ops

FAR = 255.0  # difference given to a slot that did not sample picture


@dataclass(frozen=True)
class BreakupConfig:
    reach: int = 2  # neighbours within this many frames vote on the centre's damage
    smooth: int = 5  # rows averaged before deciding
    floor: float = 1.5  # levels: least reference row (the lower quartile) for normalising
    ratio: float = 4.0  # a band disagrees this many times more than the frame's median row
    level: float = 6.0  # ...and by at least this many levels
    snow: float = 1.4  # ...and has this much more fine detail than the neighbours
    certain: float = 20.0  # levels: rows disagreeing this much are flagged...
    certain_snow: float = 1.1  # ...unless they are no busier than the neighbours (a flash)
    heavy_snow: float = 2.2  # rows this much busier than the neighbours are flagged when...
    heavy_ratio: float = 2.0  # ...they also disagree this many times the reference row
    heavy_level: float = 3.0  # ...and by at least this many levels
    # Repair extent: a band of at least ``wide`` rows grows by ``dilate`` rows on each side
    # and fades out over ``feather`` more; thinner streaks grow by 1 and fade over 2, so the
    # rows next to a streak keep their own detail.
    wide: int = 5
    dilate: int = 3
    feather: int = 4
    still: float = 6.0  # levels: repaired pixels this steady unwarped are kept unwarped
    # Per pixel (colour snow and sparkle, which need not fill whole rows): analog colour is
    # smooth, so fine chroma detail well above the cleanest neighbour's there is damage.
    pixel_ratio: float = 1.5  # centre noise above this many times the cleanest neighbour's...
    pixel_margin: float = 2.0  # ...plus this many levels starts the repair...
    pixel_soft: float = 3.0  # ...which is full this many levels further on
    pixel_noise: float = 4.0  # levels of chroma noise at which a source's weight halves
    speck: float = 18.0  # levels a luma speck differs from its 3x3 median...
    speck_far: float = 24.0  # ...and from the neighbours' consensus
    osd_margin: float = 8.0  # inside the OSD, damage must exceed the margin by this much
    osd_centre: float = 3.0  # and the centre's own OSD counts this many times more
    blotch: float = 14.0  # levels (|dU| + |dV|) from the neighbours' consensus colour
    bridge: int = 3  # most consecutive whole-bad frames rebuilt; longer outages are left
    # Colour loss: the centre keeps less than ``colour_loss`` of the saturation its
    # neighbours agree on (and they have at least ``colour_min`` levels of it).
    colour_loss: float = 0.35
    colour_min: float = 4.0
    whole: float = 0.6  # share of rows flagged that makes the whole frame bad
    whole_level: float = 8.0  # levels: a frame whose median row disagrees this much...
    whole_snow: float = 1.15  # ...with this much extra fine detail is bad as a whole


@dataclass(frozen=True)
class RowSignals:
    disagree: Tensor  # (H,) mean second-smallest luma difference to the neighbours, levels
    snow: Tensor  # (H,) centre's fine horizontal detail over the neighbours' (ratio)
    chroma: Tensor | None = None  # (H,) like disagree, for chroma (|du| + |dv|); scan only


class Detector:
    def __init__(self, keep: Tensor, cfg: BreakupConfig | None = None) -> None:
        self.cfg = cfg or BreakupConfig()
        self.keep = keep  # (H, W) 1 outside the OSD
        self.count = keep.sum(-1).clamp_min(1.0)
        self.keep_half = ops.area_half(keep)
        self.count_half = self.keep_half.sum(-1).clamp_min(1.0)
        self.keep_hf = keep[:, 1:] * keep[:, :-1]
        self.count_hf = self.keep_hf.sum(-1).clamp_min(1.0)
        self._sel: dict[tuple[int, ...], Tensor | slice] = {}

    def _rows(self, x: Tensor) -> Tensor:
        return (x * self.keep).sum(-1) / self.count

    def signals(
        self,
        cy: Tensor,
        cu: Tensor,
        cv: Tensor,
        wy: Tensor,
        wu: Tensor,
        wv: Tensor,
        valid: Tensor,
        offsets: list[int],
        dist: Tensor,
        chroma: bool = False,
    ) -> RowSignals:
        """Centre planes (H, W) / (H/2, W/2); warped neighbours (S, ...) with their slot
        offsets and distance weights (0 marks a missing slot)."""
        near = [
            i
            for i, d in enumerate(offsets)
            if abs(d) <= self.cfg.reach and i == offsets.index(d)  # skip repeated slots
        ]
        sel = slots(near, cy.device, self._sel)
        wy, valid = wy[sel], valid[sel]
        if chroma:
            wu, wv = wu[sel], wv[sel]
        missing = dist[sel] <= 0.0
        k = 2 if len(near) >= 2 else 1

        ref = ops.gaussian(cy, 1.2, size=(5, 5))
        diff = (ops.gaussian(wy, 1.2, size=(5, 5)) - ref).abs()
        diff = torch.where((valid < 0.5) | missing, FAR, diff)
        d = second_smallest(diff) if k == 2 else diff.amin(0)
        disagree = self._rows(torch.where(d >= FAR, 0.0, d))

        crows = None
        if chroma:
            cdiff = (wu - cu).abs() + (wv - cv).abs()
            cvalid = ops.area_half(valid)
            cdiff = torch.where((cvalid < 0.5) | missing, FAR, cdiff)
            cd = second_smallest(cdiff) if k == 2 else cdiff.amin(0)
            crow = (torch.where(cd >= FAR, 0.0, cd) * self.keep_half).sum(-1) / self.count_half
            crows = crow.repeat_interleave(2)

        def hf(y: Tensor) -> Tensor:
            g = (y[..., :, 1:] - y[..., :, :-1]).abs() * self.keep_hf
            return g.sum(-1) / self.count_hf

        snow = hf(cy) / hf(wy).median(0).values.clamp_min(0.5)
        return RowSignals(disagree, snow, crows)

    def saturation(self, u: Tensor, v: Tensor) -> Tensor:
        """Mean chroma distance from grey outside the OSD: (H/2, W/2) -> (), or batched."""
        s = ((u - 128.0).abs() + (v - 128.0).abs()) * self.keep_half
        return s.sum((-2, -1)) / self.keep_half.sum()

    def colour_lost(
        self, cu: Tensor, cv: Tensor, nu: Tensor, nv: Tensor, dist: Tensor
    ) -> tuple[Tensor, Tensor]:
        """The centre lost its colour while at least two neighbours (chroma planes, S slots)
        kept it: () bool, and (S,) which neighbours have colour."""
        sat = self.saturation(nu, nv)
        has = (sat >= self.cfg.colour_min) & (dist.flatten() > 0.0)
        ref = torch.where(has, sat, 0.0).sum() / has.sum().clamp_min(1)
        lost = (has.sum() >= 2) & (self.saturation(cu, cv) < self.cfg.colour_loss * ref)
        return lost, has


def flag_rows(s: RowSignals, cfg: BreakupConfig) -> Tensor:
    """(H,) bool: rows of the centre that are breakup. All rows when the frame is bad as a
    whole: then the median row is damaged too, so it cannot serve as the reference."""
    dis = smooth_rows(s.disagree, cfg.smooth)
    snow = smooth_rows(s.snow, cfg.smooth)
    median = s.disagree.median()
    base = dis.quantile(0.25).clamp_min(cfg.floor)  # bursts damage the median row too
    band = (dis > (cfg.ratio * base).clamp_min(cfg.level)) & (snow > cfg.snow)
    flags = band | ((dis > cfg.certain) & (snow > cfg.certain_snow))
    flags |= (snow > cfg.heavy_snow) & (dis > (cfg.heavy_ratio * base).clamp_min(cfg.heavy_level))
    whole = (median > cfg.whole_level) & (s.snow.median() > cfg.whole_snow)
    whole = whole | (flags.float().mean() > cfg.whole)
    return flags | whole


def _grow(x: Tensor, r: int) -> Tensor:
    return F.max_pool1d(x[None, None], 2 * r + 1, stride=1, padding=r)[0, 0] if r else x


def _fade(x: Tensor, dilate: int, feather: int) -> Tensor:
    """``x`` grown by ``dilate`` rows, then ramping down to 0 over ``feather`` rows."""
    x = _grow(x, dilate)
    r = feather // 2
    if r == 0:
        return x
    return torch.maximum(x, smooth_rows(_grow(x, r), 2 * r + 1))


def row_mask(flags: Tensor, cfg: BreakupConfig) -> Tensor:
    """(H,) 0..1 repair strength from flagged rows."""
    x = flags.float()
    r = cfg.wide // 2
    wide = _grow(-_grow(-x, r), r)  # opening: only bands of at least ``wide`` rows
    return torch.maximum(_fade(x, 1, 2), _fade(wide, cfg.dilate, cfg.feather)).clamp(0.0, 1.0)


def fill_rows(x: Tensor, bad: Tensor) -> Tensor:
    """Rows of ``x`` (..., H, W) where ``bad`` (H,) is set, replaced per column by linear
    interpolation between the nearest good rows above and below (or the one that exists)."""
    h = bad.shape[0]
    r = torch.arange(h, device=x.device)
    good = ~bad
    above = torch.where(good, r, -1).cummax(0).values
    below = torch.where(good, r, h).flip(0).cummin(0).values.flip(0)
    has_a, has_b = above >= 0, below < h
    a, b = above.clamp(0, h - 1), below.clamp(0, h - 1)
    t = torch.where(
        has_a & has_b,
        (r - a).float() / (b - a).clamp_min(1).float(),
        torch.where(has_a, 0.0, 1.0),
    )[:, None]
    return torch.where(bad[:, None], torch.lerp(x[..., a, :], x[..., b, :], t), x)


def slots(
    index: list[int], device: torch.device, cache: dict[tuple[int, ...], Tensor | slice]
) -> Tensor | slice:
    """``index`` as a slice when it is a run (no copy when used), else a cached tensor."""
    key = tuple(index)
    if key not in cache:
        run = list(range(index[0], index[0] + len(index))) == index
        cache[key] = slice(index[0], index[-1] + 1) if run else torch.tensor(index, device=device)
    return cache[key]


def second_smallest(x: Tensor) -> Tensor:
    """Elementwise second-smallest over dim 0 of (S, ...), S >= 2, without sorting."""
    m1 = torch.minimum(x[0], x[1])
    m2 = torch.maximum(x[0], x[1])
    for i in range(2, x.shape[0]):
        m2 = torch.minimum(m2, torch.maximum(m1, x[i]))
        m1 = torch.minimum(m1, x[i])
    return m2


def trimmed_mean(x: Tensor) -> Tensor:
    """Elementwise mean over dim 0 of (S, ...) without the smallest and largest value
    (a cheap stand-in for the median); plain mean for S < 3."""
    if x.shape[0] < 3:
        return x.mean(0)
    return (x.sum(0) - x.amin(0) - x.amax(0)) / (x.shape[0] - 2)


def chroma_noise(u: Tensor, v: Tensor) -> Tensor:
    """Local fine chroma detail, (..., H/2, W/2) -> same: mean |c - blur(c)| of U plus V over
    5x5. Low in clean analog video (colour is band-limited), high in colour snow."""
    n = (u - ops.gaussian(u, 1.5)).abs() + (v - ops.gaussian(v, 1.5)).abs()
    return ops.box_mean(n, 5)


def smooth_rows(x: Tensor, size: int) -> Tensor:
    """Box mean over ``size`` rows of a (H,) or (N, H) row signal."""
    r = size // 2
    b = x.reshape(-1, 1, x.shape[-1])
    b = F.pad(b, (r, r), mode="replicate")
    return F.avg_pool1d(b, size, stride=1).reshape(x.shape)
