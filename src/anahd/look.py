"""Tone, colour and the upscale to 1080 lines.

Tone levels are measured once per recording (so brightness does not pump), then applied
gently. The upscale is a single Lanczos resample of the luma with clamped sharpening (it
cannot overshoot its neighbourhood, so no halos), and the half-size chroma is upsampled
guided by the luma so colour edges follow luma edges instead of bleeding.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from anahd.media import Plane

F32 = NDArray[np.float32]
BoolMask = NDArray[np.bool_]


@dataclass(frozen=True)
class LookConfig:
    levels_strength: float = 0.5  # 0 = keep the DVR's levels, 1 = full auto-levels
    black_percentile: float = 0.5
    white_percentile: float = 99.7
    contrast: float = 0.12  # gentle S-curve
    vibrance: float = 0.25  # saturation boost for muted colours, less for saturated ones
    sharpen: float = 0.7
    sharpen_sigma: float = 1.0
    sharpen_margin: float = 2.0  # levels the sharpened luma may exceed its 3x3 neighbourhood
    chroma_eps: float = 25.0  # guided-filter smoothness for chroma upsampling (levels^2)


@dataclass(frozen=True)
class Levels:
    black: float
    white: float


def measure_levels(samples: list[Plane], osd: BoolMask, cfg: LookConfig) -> Levels:
    picture = np.concatenate([s[~osd].ravel() for s in samples])
    black = float(np.percentile(picture, cfg.black_percentile))
    white = float(np.percentile(picture, cfg.white_percentile))
    if white - black < 32.0:  # nearly flat footage: leave it alone
        return Levels(0.0, 255.0)
    return Levels(black, white)


def tone_curve(levels: Levels, cfg: LookConfig) -> NDArray[np.float32]:
    """A 256-entry lookup from DVR luma to adjusted luma (full range, 0-255)."""
    x = np.arange(256, dtype=np.float32)
    stretched = np.clip((x - levels.black) / (levels.white - levels.black), 0.0, 1.0)
    t = (1.0 - cfg.levels_strength) * (x / 255.0) + cfg.levels_strength * stretched
    s_curve = t + cfg.contrast * (t - 0.5) * (1.0 - np.abs(2.0 * t - 1.0))
    return np.asarray(np.clip(s_curve, 0.0, 1.0) * 255.0, dtype=np.float32)


def apply_tone(y: F32, curve: NDArray[np.float32]) -> F32:
    idx = np.clip(y, 0.0, 255.0)
    lo = np.floor(idx).astype(np.int32)
    hi = np.minimum(lo + 1, 255)
    frac = idx - lo
    return np.asarray(curve[lo] * (1.0 - frac) + curve[hi] * frac, dtype=np.float32)


def apply_vibrance(u: F32, v: F32, cfg: LookConfig) -> tuple[F32, F32]:
    cu, cv_ = u - 128.0, v - 128.0
    sat = np.sqrt(cu * cu + cv_ * cv_)
    gain = 1.0 + cfg.vibrance * np.clip(1.0 - sat / 64.0, 0.0, 1.0)
    return (
        np.asarray(np.clip(128.0 + cu * gain, 0.0, 255.0), dtype=np.float32),
        np.asarray(np.clip(128.0 + cv_ * gain, 0.0, 255.0), dtype=np.float32),
    )


Affine = NDArray[np.float64]  # 2x3, source pixel -> output pixel


def scale_matrix(src: tuple[int, int], dst: tuple[int, int]) -> Affine:
    """The plain resize from ``src`` to ``dst`` (width, height), pixel centres aligned."""
    sx, sy = dst[0] / src[0], dst[1] / src[1]
    return np.array([[sx, 0.0, (sx - 1) / 2], [0.0, sy, (sy - 1) / 2]], dtype=np.float64)


def half_matrix(m: Affine) -> Affine:
    """The same mapping for half-size planes (4:2:0 chroma), pixel centres aligned."""
    out = m.copy()
    out[:, 2] = (m[:, :2] @ np.array([0.5, 0.5]) + m[:, 2] - 0.5) / 2
    return out


def upscale_luma(y: F32, size: tuple[int, int], cfg: LookConfig, m: Affine | None = None) -> F32:
    """Resample to ``size`` in one step (a plain resize, or the affine ``m``), then sharpen."""
    if m is None:
        m = scale_matrix((y.shape[1], y.shape[0]), size)
    big = _warp(y, m, size, cv2.INTER_LANCZOS4)
    detail = big - cv2.GaussianBlur(big, (0, 0), cfg.sharpen_sigma)
    sharp = big + cfg.sharpen * detail
    # Bound the result by the neighbourhood of a ringing-free (bilinear) resample: this
    # removes Lanczos ringing and sharpening halos alike, while edges stay steep.
    plain = _warp(y, m, size, cv2.INTER_LINEAR)
    kernel = np.ones((3, 3), np.uint8)
    hi = cv2.dilate(plain, kernel) + cfg.sharpen_margin
    lo = cv2.erode(plain, kernel) - cfg.sharpen_margin
    return np.asarray(np.clip(sharp, lo, hi), dtype=np.float32)


def _warp(x: F32, m: Affine, size: tuple[int, int], interp: int) -> F32:
    return np.asarray(
        cv2.warpAffine(x, m, size, flags=interp, borderMode=cv2.BORDER_REPLICATE),
        dtype=np.float32,
    )


def _guided(guide: F32, src: F32, radius: int, eps: float) -> F32:
    k = (2 * radius + 1, 2 * radius + 1)
    mean_i = cv2.boxFilter(guide, -1, k)
    mean_p = cv2.boxFilter(src, -1, k)
    cov = cv2.boxFilter(guide * src, -1, k) - mean_i * mean_p
    var = cv2.boxFilter(guide * guide, -1, k) - mean_i * mean_i
    a = cov / (var + eps)
    b = mean_p - a * mean_i
    return np.asarray(cv2.boxFilter(a, -1, k) * guide + cv2.boxFilter(b, -1, k), dtype=np.float32)


def upscale_chroma(c: F32, luma_big: F32, cfg: LookConfig, m: Affine | None = None) -> F32:
    """Half-size chroma to half the output size, its edges snapped to the upscaled luma.

    ``m`` is the full-size luma mapping; the chroma uses the matching half-size one.
    """
    h, w = luma_big.shape
    size = (w // 2, h // 2)
    mc = scale_matrix((c.shape[1], c.shape[0]), size) if m is None else half_matrix(m)
    up = _warp(c, mc, size, cv2.INTER_CUBIC)
    guide = np.asarray(cv2.resize(luma_big, size, interpolation=cv2.INTER_AREA), dtype=np.float32)
    return _guided(guide, up, 2, cfg.chroma_eps)


def to_plane(x: F32) -> Plane:
    return np.asarray(np.clip(np.rint(x), 0, 255), dtype=np.uint8)
