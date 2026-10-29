"""Image operations on float32 CUDA tensors, matching the OpenCV calls of the CPU path.

Planes are (H, W) or a batch (N, H, W). Borders follow OpenCV's defaults: filters reflect
without repeating the edge pixel (BORDER_REFLECT_101), resamples repeat the edge pixel.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from torch import Tensor

F64 = NDArray[np.float64]


def _batch(x: Tensor) -> Tensor:
    """(H, W) or (N, H, W) -> (N, 1, H, W)."""
    return x.reshape(-1, 1, x.shape[-2], x.shape[-1])


def gaussian_kernel(sigma: float, size: int | None = None) -> NDArray[np.float32]:
    """OpenCV's kernel: for float images ``size`` defaults to round(8 sigma + 1), made odd."""
    if size is None:
        size = round(sigma * 8 + 1) | 1
    t = np.arange(size, dtype=np.float64) - (size - 1) / 2
    k = np.exp(-0.5 * (t / sigma) ** 2)
    return np.asarray(k / k.sum(), dtype=np.float32)


@lru_cache(maxsize=64)
def _kernel_tensor(sigma: float, size: int | None, device: str) -> Tensor:
    return torch.from_numpy(gaussian_kernel(sigma, size)).to(device)


def _reflect101(x: Tensor, r: int, axis: int) -> Tensor:
    """Pad (N, 1, H, W) by ``r`` on both sides of ``axis`` (2 or 3), reflect-101."""
    if r == 0:
        return x
    n = x.shape[axis]
    if r < n:
        pad = (r, r, 0, 0) if axis == 3 else (0, 0, r, r)
        return F.pad(x, pad, mode="reflect")
    idx = torch.arange(-r, n + r, device=x.device)
    period = max(2 * (n - 1), 1)
    idx = idx.remainder(period)
    idx = torch.where(idx >= n, period - idx, idx)
    return x.index_select(axis, idx)


def _conv_axis(x: Tensor, k: Tensor, axis: int) -> Tensor:
    r = (k.numel() - 1) // 2
    x = _reflect101(x, r, axis)
    shape = (1, 1, 1, -1) if axis == 3 else (1, 1, -1, 1)
    return F.conv2d(x, k.view(shape))


def gaussian(
    x: Tensor, sigma_x: float, sigma_y: float | None = None, size: tuple[int, int] | None = None
) -> Tensor:
    """cv2.GaussianBlur(x, size or (0, 0), sigma_x, sigmaY=sigma_y)."""
    sy = sigma_x if sigma_y is None else sigma_y
    kw, kh = (None, None) if size is None else size
    b = _batch(x)
    dev = str(x.device)
    if kw != 1:
        b = _conv_axis(b, _kernel_tensor(sigma_x, kw, dev), 3)
    if kh != 1:
        b = _conv_axis(b, _kernel_tensor(sy, kh, dev), 2)
    return b.reshape(x.shape)


def box_mean(x: Tensor, k: int) -> Tensor:
    """cv2.boxFilter(x, -1, (k, k)): normalized, reflect-101."""
    r = k // 2
    b = _reflect101(_reflect101(_batch(x), r, 3), r, 2)
    return F.avg_pool2d(b, k, stride=1).reshape(x.shape)


def row_box_sum(x: Tensor, k: int) -> Tensor:
    """cv2.boxFilter(x, -1, (k, 1), normalize=False, borderType=BORDER_REFLECT)."""
    r = k // 2
    w = x.shape[-1]
    idx = torch.arange(-r, w + r, device=x.device).remainder(2 * w)
    idx = torch.where(idx >= w, 2 * w - 1 - idx, idx)
    c = torch.cumsum(x.index_select(-1, idx).double(), dim=-1)
    c = F.pad(c, (1, 0))
    return (c[..., k:] - c[..., :-k]).to(x.dtype)


def area_half(x: Tensor) -> Tensor:
    """cv2.resize to half size with INTER_AREA (means of 2x2 blocks)."""
    out = F.avg_pool2d(_batch(x), 2)
    return out.reshape(*x.shape[:-2], out.shape[-2], out.shape[-1])


def dilate(x: Tensor, r: int) -> Tensor:
    """(H, W) max over the (2r+1)x(2r+1) square around each pixel (edges: the part
    inside), as a row pass then a column pass: the same result as one square pass, at
    a small part of its cost for a large ``r``."""
    k = 2 * r + 1
    rows = F.max_pool2d(x[None, None], (1, k), 1, (0, r))
    return F.max_pool2d(rows, (k, 1), 1, (r, 0))[0, 0]


def dilate3(x: Tensor) -> Tensor:
    return F.max_pool2d(_batch(x), 3, stride=1, padding=1).reshape(x.shape)


def median3(x: Tensor) -> Tensor:
    """3x3 median, edge pixels repeated: a min/max network over the 9 shifted planes
    (Paeth's median of 9), no sorting."""
    b = F.pad(_batch(x), (1, 1, 1, 1), mode="replicate")
    h, w = b.shape[-2] - 2, b.shape[-1] - 2
    p = [b[..., i : i + h, j : j + w] for i in range(3) for j in range(3)]

    def sort2(i: int, k: int) -> None:
        p[i], p[k] = torch.minimum(p[i], p[k]), torch.maximum(p[i], p[k])

    for i, k in (
        (1, 2),
        (4, 5),
        (7, 8),
        (0, 1),
        (3, 4),
        (6, 7),
        (1, 2),
        (4, 5),
        (7, 8),
        (0, 3),
        (5, 8),
        (4, 7),
        (3, 6),
        (1, 4),
        (2, 5),
        (4, 7),
        (4, 2),
        (6, 4),
        (4, 2),
    ):
        sort2(i, k)
    return p[4].reshape(x.shape)


def erode3(x: Tensor) -> Tensor:
    return -dilate3(-x)


# Resampling as matrices (out = Ry @ x @ Rx^T): exact for separable kernels, fast on a GPU.


def _lanczos4(t: F64) -> F64:
    a = 4.0
    safe = np.where(np.abs(t) < 1e-9, 1.0, t)
    pt = math.pi * safe
    out = a * np.sin(pt) * np.sin(pt / a) / (pt * pt)
    out = np.where(np.abs(t) < 1e-9, 1.0, out)
    return np.asarray(np.where(np.abs(t) < a, out, 0.0), dtype=np.float64)


def _cubic(t: F64) -> F64:
    a = -0.75  # OpenCV's (and grid_sample's) bicubic
    x = np.abs(t)
    near = ((a + 2) * x - (a + 3)) * x * x + 1
    far = ((a * x - 5 * a) * x + 8 * a) * x - 4 * a
    return np.asarray(np.where(x <= 1, near, np.where(x < 2, far, 0.0)), dtype=np.float64)


def _linear(t: F64) -> F64:
    return np.asarray(np.clip(1.0 - np.abs(t), 0.0, None), dtype=np.float64)


_KERNELS: dict[str, tuple[Callable[[F64], F64], int]] = {
    "lanczos4": (_lanczos4, 4),
    "cubic": (_cubic, 2),
    "linear": (_linear, 1),
}


def resample_matrix(
    n_in: int, n_out: int, scale: float, offset: float, kind: str
) -> NDArray[np.float32]:
    """(n_out, n_in) weights: output pixel o samples the input at (o - offset) / scale, as
    cv2.warpAffine does for one axis of an axis-aligned matrix; edge pixels repeat."""
    fn, reach = _KERNELS[kind]
    pos = (np.arange(n_out, dtype=np.float64) - offset) / scale
    base = np.floor(pos)
    taps = np.arange(-reach + 1, reach + 1)
    weights = fn(pos[:, None] - (base[:, None] + taps[None, :]))
    weights /= weights.sum(axis=1, keepdims=True)
    cols = np.clip(base[:, None] + taps[None, :], 0, n_in - 1).astype(np.int64)
    m = np.zeros((n_out, n_in), dtype=np.float64)
    rows = np.arange(n_out)
    for j in range(len(taps)):
        np.add.at(m, (rows, cols[:, j]), weights[:, j])
    return np.asarray(m, dtype=np.float32)


class Resampler:
    """A plain resize from ``src`` to ``dst`` (width, height), pixel centres aligned."""

    def __init__(
        self, src: tuple[int, int], dst: tuple[int, int], kind: str, device: torch.device
    ) -> None:
        (w, h), (big_w, big_h) = src, dst
        sx, sy = big_w / w, big_h / h
        rx = resample_matrix(w, big_w, sx, (sx - 1) / 2, kind)
        ry = resample_matrix(h, big_h, sy, (sy - 1) / 2, kind)
        self.rxt = torch.from_numpy(np.ascontiguousarray(rx.T)).to(device)
        self.ry = torch.from_numpy(ry).to(device)

    def __call__(self, x: Tensor) -> Tensor:
        return self.ry @ (x @ self.rxt)


def guided(guide: Tensor, src: Tensor, radius: int, eps: float) -> Tensor:
    k = 2 * radius + 1
    mean_i = box_mean(guide, k)
    mean_p = box_mean(src, k)
    cov = box_mean(guide * src, k) - mean_i * mean_p
    var = box_mean(guide * guide, k) - mean_i * mean_i
    a = cov / (var + eps)
    b = mean_p - a * mean_i
    return box_mean(a, k) * guide + box_mean(b, k)


def apply_curve(y: Tensor, curve: Tensor) -> Tensor:
    """Linear interpolation in a 256-entry lookup (look.apply_tone)."""
    idx = y.clamp(0.0, 255.0)
    lo = idx.floor()
    frac = idx - lo
    lo_i = lo.long()
    hi_i = (lo_i + 1).clamp(max=255)
    return curve[lo_i] * (1.0 - frac) + curve[hi_i] * frac


class Filler:
    """Fills a fixed hole mask from its surroundings: normalized convolution at the finest
    scale that reaches each pixel. Stands in for cv2.inpaint before optical flow."""

    SIGMAS = (2.0, 4.0, 8.0, 16.0, 32.0)

    def __init__(self, mask: Tensor) -> None:
        hole = mask.bool()
        self.any = bool(hole.any())
        self.known = (~hole).float()
        self.dens: list[Tensor] = []
        self.take: list[Tensor] = []
        left = hole.clone()
        for s in self.SIGMAS:
            den = gaussian(self.known, s)
            take = left & (den > 0.05)
            left &= ~take
            self.dens.append(den.clamp_min(1e-6))
            self.take.append(take)
        self.take[-1] |= left
        self.used = [bool(t.any()) for t in self.take]  # decided once, no syncs per frame

    def __call__(self, x: Tensor) -> Tensor:
        if not self.any:
            return x
        out = x
        xk = x * self.known
        for s, den, take, used in zip(self.SIGMAS, self.dens, self.take, self.used, strict=True):
            if used:
                out = torch.where(take, gaussian(xk, s) / den, out)
        return out
