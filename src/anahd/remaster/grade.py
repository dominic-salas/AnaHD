"""The C03 film look as one 33^3 lookup table, plus the per-pixel parts a table cannot
hold (vignette, glow). An approximation of the "Analog Restore" DaVinci grade
(protten, 2026-10-10): Film Look Creator settings, the Shadows and sky nodes.

Colour is handled as BT.709 R'G'B' in 0..1. Hues are HSV hues (0 red, 60 yellow,
120 green, 240 blue); rotations are applied in the Cb/Cr plane, which turns the same way.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from .params import RemasterParams

KR, KB = 0.2126, 0.0722
KG = 1.0 - KR - KB
LUT_SIZE = 33


@dataclass(frozen=True)
class FilmLook:
    """The approved grade (Film Look Creator values from the project file)."""

    # Re-tuned 2026-10-21 for the corrected BT.709 pipeline (main e5d965a): the values that
    # make it look like the approved reference renders (analog_look/recheck). The project file
    # had 5767 K, tint 50, contrast 1.12; the approved renders went through the old
    # double colour conversion, so the same numbers no longer give the same picture.
    source_k: float = 5542.0  # source white balance, kelvin
    tint: float = 10.0  # Resolve's units are not published: 100 is taken as a 5 % green cut
    pre_sat: float = 0.85
    contrast: float = 1.1128
    pivot: float = 0.435
    richness: float = 1.53
    post_sat: float = 1.58
    saturation: float = 1.04  # overall, after the grade (the re-tune's +4 %)
    colour_blend: float = 0.75
    # Hue sectors: centre deg, width deg, rotation deg, saturation.
    sectors: tuple[tuple[float, float, float, float], ...] = (
        (0, 60, 3.9, 0.95),
        (20, 20, 1.5, 1.0),
        (45, 45, -2.0, 1.1),
        (75, 45, 15.5, 0.83),
        (200, 60, -2.5, 1.0),
        (240, 60, 0.0, 0.9),
        (315, 50, 0.0, 0.84),
    )
    lum_sat: tuple[tuple[float, float], ...] = ((0.0, 1.2), (0.33, 1.1), (0.66, 1.0), (1.0, 0.7))
    # "Shadows" node: HSL key on blue..magenta, saturation > 2.9 %, luminance < 66 %.
    # Its correction is not visible in the screenshots; taken as this much desaturation.
    shadow_desat: float = 0.5
    sky_lift: float = 1.06  # "sky" node: Hue vs Lum gain around sky blue
    vignette: float = 0.12  # light lost in the far corners
    bloom: float = 0.06
    halation: float = 0.18 * 0.25


LOOK = FilmLook()


def to_rgb(y: Tensor, cb: Tensor, cr: Tensor) -> Tensor:
    """BT.709 Y (0..1) and Cb, Cr (centred on 0) -> (3, ...) R'G'B'."""
    r = y + 2 * (1 - KR) * cr
    b = y + 2 * (1 - KB) * cb
    g = (y - KR * r - KB * b) / KG
    return torch.stack((r, g, b))


def to_ycc(rgb: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    r, g, b = rgb[0], rgb[1], rgb[2]
    y = KR * r + KG * g + KB * b
    return y, (b - y) / (2 * (1 - KB)), (r - y) / (2 * (1 - KR))


def _daylight_xy(t: float) -> tuple[float, float]:
    """CIE daylight locus, 4000-25000 K."""
    t = min(max(t, 4000.0), 25000.0)
    if t <= 7000:
        x = -4.6070e9 / t**3 + 2.9678e6 / t**2 + 0.09911e3 / t + 0.244063
    else:
        x = -2.0064e9 / t**3 + 1.9018e6 / t**2 + 0.24748e3 / t + 0.237040
    return x, -3.0 * x * x + 2.87 * x - 0.275


def wb_gains(source_k: float, tint: float) -> np.ndarray:
    """Linear RGB gains taking white under ``source_k`` to D65 (von Kries in RGB), then the
    tint (positive = towards magenta)."""
    m = np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]])

    def rgb(xy: tuple[float, float]) -> np.ndarray:
        x, y = xy
        return np.asarray(m @ np.array([x / y, 1.0, (1 - x - y) / y]))

    g = rgb(_daylight_xy(6504.0)) / rgb(_daylight_xy(source_k))
    g = g / g[1]
    g[1] *= 1.0 - 0.05 * tint / 100.0
    return np.asarray(g / g.max())


def _s_curve(x: Tensor, c: float, p: float) -> Tensor:
    """Slope ``c`` at pivot ``p``; 0 and 1 stay put."""
    x = x.clamp(0, 1)
    lo = p * (x / p).clamp_min(1e-6) ** c
    hi = 1 - (1 - p) * ((1 - x) / (1 - p)).clamp_min(1e-6) ** c
    return torch.where(x < p, lo, hi)


def _band(h: Tensor, lo: float, hi: float, soft: float) -> Tensor:
    """1 for hues in lo..hi (degrees), falling to 0 over ``soft`` degrees outside."""
    mid, half = (lo + hi) / 2, (hi - lo) / 2
    d = (((h - mid + 180) % 360) - 180).abs() - half
    return (1 - d.clamp_min(0) / soft).clamp(0, 1)


def _hue(rgb: Tensor) -> Tensor:
    mx, _ = rgb.max(0)
    mn, _ = rgb.min(0)
    d = (mx - mn).clamp_min(1e-6)
    r, g, b = rgb
    h = torch.where(
        mx == r, ((g - b) / d) % 6, torch.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)
    )
    return h * 60


def grade(rgb: Tensor, p: RemasterParams, look: FilmLook = LOOK) -> Tensor:
    """The colour grade on any (3, ...) R'G'B' tensor (the LUT is this, sampled)."""
    k_src = look.source_k + 15.0 * p.warmth  # warmth +100: as if shot under 1500 K bluer light
    gains = torch.tensor(wb_gains(k_src, look.tint + p.tint), dtype=rgb.dtype, device=rgb.device)
    shape = (3,) + (1,) * (rgb.dim() - 1)
    rgb = ((rgb.clamp(0, 1) ** 2.4) * gains.view(shape)).clamp(0, 1) ** (1 / 2.4)
    y, cb, cr = to_ycc(rgb)
    rgb = _s_curve(
        to_rgb(y, cb * look.pre_sat, cr * look.pre_sat),
        1 + (look.contrast - 1) * p.contrast / 100,
        look.pivot,
    )
    base = rgb
    y, cb, cr = to_ycc(rgb)
    sat = torch.sqrt(cb * cb + cr * cr)
    # Richness: more colour, with saturated colours made denser, as film dyes do.
    rich = math.sqrt(look.richness)
    cb, cr = cb * rich, cr * rich
    y = y * (1 - 0.35 * (look.richness - 1) * sat.clamp(0, 0.5))
    h = _hue(to_rgb(y, cb, cr).clamp(0, 1))
    rot = torch.zeros_like(h)
    sf = torch.ones_like(h)
    for c, w, dr, ds in look.sectors:
        dh = ((h - c + 180) % 360) - 180
        wt = torch.where(
            dh.abs() < w, 0.5 * (1 + torch.cos(math.pi * dh / w)), torch.zeros_like(dh)
        )
        rot = rot + wt * dr
        sf = sf * (1 + wt * (ds - 1))
    a = torch.deg2rad(rot)
    cb, cr = cb * torch.cos(a) - cr * torch.sin(a), cb * torch.sin(a) + cr * torch.cos(a)
    xs = torch.tensor([q for q, _ in look.lum_sat], device=rgb.device, dtype=rgb.dtype)
    vs = torch.tensor([v for _, v in look.lum_sat], device=rgb.device, dtype=rgb.dtype)
    yc = y.clamp(0, 1)
    i = torch.bucketize(yc, xs[1:-1])
    ls = vs[i] + (vs[i + 1] - vs[i]) * (yc - xs[i]) / (xs[i + 1] - xs[i])
    k = sf * ls * look.post_sat
    key = _band(h, 225.0, 330.0, 35.0) * ((0.81 - y) / 0.15).clamp(0, 1)
    key = key * ((sat - 0.029) / 0.05).clamp(0, 1)
    k = k * (1 - look.shadow_desat * p.shadows / 100 * key)
    y = y * (1 + (look.sky_lift - 1) * _band(h, 185.0, 235.0, 25.0))
    graded = to_rgb(y, cb * k, cr * k)
    out = base + (graded - base) * look.colour_blend
    # Saturation slider: around the graded result (100 = as graded, 0 = black and white).
    oy, ocb, ocr = to_ycc(out)
    s = p.saturation / 100 * look.saturation
    return to_rgb(oy, ocb * s, ocr * s).clamp(0, 1)


def build_lut(p: RemasterParams, device: torch.device, n: int = LUT_SIZE) -> Tensor:
    """(3, n, n, n) indexed [c, b, g, r], as grid_sample reads it (x = r, y = g, z = b)."""
    t = torch.linspace(0, 1, n, device=device)
    b, g, r = torch.meshgrid(t, t, t, indexing="ij")
    return grade(torch.stack((r, g, b)), p)


def apply_lut(lut: Tensor, rgb: Tensor) -> Tensor:
    grid = (rgb.clamp(0, 1) * 2 - 1).permute(1, 2, 0)[None, None]
    out = F.grid_sample(lut[None], grid, mode="bilinear", padding_mode="border", align_corners=True)
    return out[0, :, 0]
