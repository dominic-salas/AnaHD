"""Harvest real analog residuals (source minus cleaned picture) from real recordings.

The residual of the current cleaner on flat, still-ish regions is the camera + link noise
itself: grain, line noise, dot crawl, rainbow. It is high-passed so tone and colour changes
made by the cleaner do not leak in, and taken only where the cleaned picture is flat (no
edge halos or warp errors), outside the OSD, and not from frames with breakup.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from torch import Tensor


def _gauss_blur(x: Tensor, sigma: float) -> Tensor:
    k = int(sigma * 3) * 2 + 1
    pos = torch.arange(k, dtype=x.dtype, device=x.device) - k // 2
    g = torch.exp(-0.5 * (pos / sigma) ** 2)
    g = g / g.sum()
    c = x.shape[1]
    x = F.pad(x, (k // 2, k // 2, k // 2, k // 2), mode="reflect")
    x = F.conv2d(x, g.view(1, 1, 1, k).repeat(c, 1, 1, 1), groups=c)
    return F.conv2d(x, g.view(1, 1, k, 1).repeat(c, 1, 1, 1), groups=c)


def harvest(
    src: Tensor,
    clean: Tensor,
    osd: Tensor,
    patch: int = 96,
    max_patches: int = 6,
    flat_quantile: float = 0.35,
) -> list[NDArray[np.float16]]:
    """src, clean: (3, H, W) float YUV in [0, 1] at the same size; osd: (H, W) bool (True = OSD).

    Returns up to `max_patches` (3, patch, patch) float16 signed residual patches.
    """
    res = (src - clean).unsqueeze(0)
    res = res - _gauss_blur(res, 6.0)
    y = clean[:1].unsqueeze(0)
    gx = (y[..., :, 1:] - y[..., :, :-1]).abs()
    gy = (y[..., 1:, :] - y[..., :-1, :]).abs()
    grad = F.pad(gx, (0, 1)) + F.pad(gy, (0, 0, 0, 1))
    grad = _gauss_blur(grad, 2.0)
    pg = F.avg_pool2d(grad, patch, patch)[0, 0]
    po = F.max_pool2d(osd.float().view(1, 1, *osd.shape), patch, patch)[0, 0]
    # Reject clipped (crushed blacks / blown whites) areas: no noise survives there.
    yl = F.avg_pool2d(y, patch, patch)[0, 0]
    pe = F.avg_pool2d(res[:, :1] ** 2, patch, patch)[0, 0].sqrt()
    ok = (po < 0.5) & (yl > 0.06) & (yl < 0.94)
    if not bool(ok.any()):
        return []
    thr = torch.quantile(pg[ok], flat_quantile)
    ok &= pg <= thr
    # Breakup / glitch guard: residual energy far above the frame's typical level.
    med = pe[ok].median() if bool(ok.any()) else pe.median()
    ok &= pe < med * 3.0
    idx = ok.flatten().nonzero().flatten()
    if idx.numel() == 0:
        return []
    idx = idx[torch.randperm(idx.numel())[:max_patches]]
    cols = pg.shape[1]
    out = []
    for i in idx.tolist():
        r, c = divmod(i, cols)
        p = res[0, :, r * patch : (r + 1) * patch, c * patch : (c + 1) * patch]
        out.append(p.cpu().numpy().astype(np.float16))
    return out


def row_spikes(
    src: Tensor, clean: Tensor, osd: Tensor, ratio: float = 8.0, level: float = 6.0, edge: int = 4
) -> Tensor:
    """(H,) bool: rows whose residual energy (5-row mean) is ``ratio`` times the frame's
    90th percentile row and at least ``level`` levels RMS: breakup the repair did not catch.
    Ordinary texture is heavy-tailed (the 90th percentile is ~10x the median), and flat frames
    have a tiny percentile (menu text the OSD mask missed then counts), hence both tests.
    The ``edge`` rows at the top and bottom (the DVR's frame border) are ignored."""
    keep = ~osd
    e = (((src - clean)[0] ** 2) * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1)
    e = F.avg_pool1d(e.view(1, 1, -1), 5, 1, 2, count_include_pad=False).view(-1)
    spikes = (e > torch.quantile(e, 0.9).clamp(min=1e-6) * ratio) & (e > (level / 255.0) ** 2)
    spikes[:edge] = False
    spikes[-edge:] = False
    return spikes


def harvest_bands(
    src: Tensor, clean: Tensor, osd: Tensor, rows: Tensor, max_bands: int = 3
) -> list[NDArray[np.float16]]:
    """Cut real breakup bands: ``rows`` (H,) bool, the rows the breakup detector repaired.

    Returns (3, h, W) float16 signed residual strips (h = band height, 2..96 rows): the
    damage the repair took out (snow, torn lines, black dropouts, colour loss).
    """
    res = src - clean
    keep = ~osd
    bad = F.max_pool1d(rows.float().view(1, 1, -1), 3, 1, 1).view(-1)  # close 1-row gaps
    out: list[NDArray[np.float16]] = []
    r, h = 0, bad.numel()
    while r < h and len(out) < max_bands:
        if bad[r] > 0:
            r1 = r
            while r1 < h and bad[r1] > 0:
                r1 += 1
            if 2 <= r1 - r <= 96:
                lo, hi = max(r - 1, 0), min(r1 + 1, h)
                strip = res[:, lo:hi] * keep[lo:hi]
                out.append(strip.cpu().numpy().astype(np.float16))
            r = r1
        r += 1
    return out
