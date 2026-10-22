"""The remaster stage: after the restorer, on frames at the output size.

Each frame waits for the two frames after it (colour clean-up and motion blur look at its
neighbours), so the stage runs two frames behind the cleaner. It reuses the cleaner's own
optical flow (centre -> neighbour, measured for the restorer anyway), so following the
motion costs only the warps.

Order: colour clean-up (temporal chroma, then a 3-frame mean-colour deflicker) -> lens
colour fix -> sharpening -> grade (LUT, vignette, glow) -> motion blur -> stabilisation ->
SuperView -> packed BT.709 4:2:0. The OSD is never graded, blurred, steadied or cleaned.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from anahd.gpu import ops
from anahd.look import scale_matrix
from anahd.steady import Plan, SteadyConfig, _zoom_for, matrix

from .grade import LOOK, FilmLook, apply_lut, build_lut, to_rgb, to_ycc
from .osd import Redraw
from .params import RemasterParams

RADIUS = 2  # colour clean-up neighbours on each side
MATCH = 8.0  # luma levels at which a neighbour's colour weight halves
FALLOFF = 0.1  # weight lost per frame of distance
BLUR_TAPS = 8
# The C03's radial colour cast, measured on 8 dev clips (analog_look/shading.json,
# 2026-10-21): fit within each row against r^2 and r^4, in 8-bit limited-range levels at
# the corner-normalised radius. 100 % removes this share of it.
SHADING_U = (10.47, 6.29)
SHADING_V = (-12.51, 8.1)
SHADING_SHARE = 0.75
SUPERVIEW_W = 1920
SUPERVIEW_CENTRE = 1.05  # horizontal stretch at the centre (1 = natural)
OSD_OUT = 0.12  # osd_out: extra vertical magnification at the left and right edges


@dataclass
class Frame:
    """One cleaned frame at the output size, full range 0..1, chroma centred on 0."""

    y: Tensor  # (H, W)
    u: Tensor  # (H/2, W/2)
    v: Tensor
    flows: dict[int, Tensor] = field(default_factory=dict)  # d -> (2, H/2, W/2), chroma px
    index: int = 0
    mean: Tensor | None = None  # mean colour after the temporal clean-up


@dataclass
class Timings:
    frames: int = 0
    ms: dict[str, float] = field(default_factory=dict)


class Remaster:
    """``push`` each cleaned frame (with the cleaner's flows); get back the finished ones."""

    def __init__(
        self,
        size: tuple[int, int],
        osd: Tensor,
        params: RemasterParams,
        device: torch.device,
        steady: SteadyWarp | None = None,
        look: FilmLook = LOOK,
        redraw: Redraw | None = None,
        timed: bool = False,
    ) -> None:
        w, h = size
        self.w, self.h, self.dev = w, h, device
        self.p, self.look = params, look
        # The OSD boxes at the output size; per frame only their static strokes are kept.
        self.boxes = F.interpolate(osd.float()[None, None], size=(h, w), mode="nearest")[0, 0] > 0
        self.steady = steady
        self.redraw = redraw  # the experimental OSD redraw (osd.for_clip), or None
        self.win: deque[Frame] = deque()
        self.count = 0
        self.c = 0  # index in ``win`` of the next frame to finish
        self.capture: int | None = None
        self.captured: Tensor | None = None
        self.timed = timed
        self.timings = Timings()
        hh, hw = h // 2, w // 2
        ys, xs = torch.meshgrid(
            torch.arange(hh, device=device, dtype=torch.float32),
            torch.arange(hw, device=device, dtype=torch.float32),
            indexing="ij",
        )
        self.gx_h, self.gy_h = xs, ys
        # The radius in the camera's own 4:3 picture, whatever shape the DVR stored it in
        # (a 16:9-stretched or 720x480 recording gets the same cast on the same scene
        # points; for a 4:3 output this is the plain radius over the half-diagonal). Its
        # strength is not estimated per clip: on the 8 dev clips (same camera) a per-clip fit
        # ranged from 0.11x to 1.62x of the pooled cast, scene colour swamping the lens.
        r2 = (
            (4 * (xs - (hw - 1) / 2) / (hw / 2)) ** 2 + (3 * (ys - (hh - 1) / 2) / (hh / 2)) ** 2
        ) / 25
        share = SHADING_SHARE * params.lens_colour_fix / 100 / 224.0
        self.cast_u = share * (SHADING_U[0] * r2 + SHADING_U[1] * r2 * r2)
        self.cast_v = share * (SHADING_V[0] * r2 + SHADING_V[1] * r2 * r2)
        yf, xf = torch.meshgrid(
            torch.arange(h, device=device, dtype=torch.float32),
            torch.arange(w, device=device, dtype=torch.float32),
            indexing="ij",
        )
        self.gx, self.gy = xf, yf
        rr = ((xf - (w - 1) / 2) ** 2 + (yf - (h - 1) / 2) ** 2) / ((w / 2) ** 2 + (h / 2) ** 2)
        self.vignette = 1 - look.vignette * params.vignette / 100 * rr**1.5
        self.lut = build_lut(params, device) if params.look_strength > 0 else None
        self.sv_grid = superview_grid(w, h, params.superview, device)
        self.out_size = (SUPERVIEW_W, h) if self.sv_grid is not None else (w, h)

    # ----- the queue -----

    def push(self, f: Frame) -> list[Tensor]:
        """Add the next cleaned frame; returns the frames that are now finished (packed)."""
        f.index = self.count
        self.count += 1
        self.win.append(f)
        out = []
        while self.c + RADIUS < len(self.win):
            out.append(self._next())
        return out

    def flush(self) -> list[Tensor]:
        """The last frames, once the recording has ended."""
        out = []
        while self.c < len(self.win):
            out.append(self._next())
        return out

    c = 0  # index in ``win`` of the next frame to finish

    def _next(self) -> Tensor:
        out = self.process(list(self.win), self.c)
        self.c += 1
        while self.c > RADIUS:  # keep only what later frames still look back at
            self.win.popleft()
            self.c -= 1
        return out

    # ----- one frame -----

    def _tick(self, key: str, t0: float) -> float:
        if not self.timed:
            return t0
        import time

        torch.cuda.synchronize(self.dev)
        t1 = time.perf_counter()
        self.timings.ms[key] = self.timings.ms.get(key, 0.0) + 1000 * (t1 - t0)
        return t1

    def osd_now(self, win: list[Frame], c: int) -> Tensor:
        """This frame's OSD pixels: inside the OSD boxes, the bright strokes and the colourful
        icon pixels that stay put across the window, grown by their 1-pixel outline; not the
        whole box. Whatever lies behind the overlay (even a plain, still wall) is graded like
        the rest of the picture, so no lighter box shows around the text and icons."""
        y = win[c].y
        still = torch.ones_like(y, dtype=torch.bool)
        for j in range(max(0, c - RADIUS), min(len(win), c + RADIUS + 1)):
            if j != c:
                still &= (win[j].y - y).abs() < 6 / 255
        chroma = win[c].u.abs() + win[c].v.abs()
        colourful = F.interpolate(chroma[None, None], size=y.shape, mode="nearest")[0, 0] > 0.08
        bright = y > 0.72
        core = (bright | (still & colourful)) & self.boxes
        grown = F.max_pool2d(core.float()[None, None], 3, 1, 1)[0, 0]
        return ops.gaussian(grown, 0.8).clamp(0, 1) * self.boxes

    def process(self, win: list[Frame], c: int) -> Tensor:
        import time

        p = self.p
        t0 = time.perf_counter() if self.timed else 0.0
        f = win[c]
        y, u, v = f.y, f.u, f.v
        osd = self.osd_now(win, c)
        osd_h = F.avg_pool2d(osd[None, None], 2)[0, 0]
        keep_h = 1 - osd_h
        t0 = self._tick("osd", t0)

        # 1. colour clean-up: chroma only, from the aligned neighbours, then deflicker
        if p.colour_cleanup > 0:
            amount = p.colour_cleanup / 100
            yh = F.avg_pool2d(y[None, None], 2)[0, 0]
            num = torch.stack((u, v))
            den = torch.ones_like(u)
            for d, flow in f.flows.items():
                j = c + d
                if abs(d) > RADIUS or not 0 <= j < len(win):
                    continue
                n = win[j]
                nyh = F.avg_pool2d(n.y[None, None], 2)[0, 0]
                wn = self._warp(torch.stack((nyh, n.u, n.v)), flow)
                diff = ops.gaussian((wn[0] - yh).abs() * 255.0, 1.5)
                px, py = self.gx_h + flow[0], self.gy_h + flow[1]
                hh, hw = u.shape
                inside = ((px >= 0) & (px <= hw - 1) & (py >= 0) & (py <= hh - 1)).float()
                wt = amount * inside * (1 - FALLOFF * abs(d)) / (1 + (diff / MATCH) ** 4) * keep_h
                num += wt * wn[1:]
                den += wt
            u, v = num[0] / den, num[1] / den
            mean = torch.stack(((u * keep_h).sum(), (v * keep_h).sum())) / keep_h.sum().clamp_min(1)
            f.mean = mean
            ms = [
                win[j].mean
                for j in (c - 1, c, c + 1)
                if 0 <= j < len(win) and win[j].mean is not None
            ]
            target = torch.stack([m for m in ms if m is not None]).mean(0)
            u = u + amount * (target[0] - mean[0]) * keep_h
            v = v + amount * (target[1] - mean[1]) * keep_h
        t0 = self._tick("colour_cleanup", t0)

        # 2. lens colour fix
        if p.lens_colour_fix > 0:
            u = u - self.cast_u * keep_h
            v = v - self.cast_v * keep_h
        t0 = self._tick("lens", t0)

        # 3. sharpening (luma), then the grade
        if p.sharpen > 0:
            amt = 0.7 * p.sharpen / 100
            sharp = y + amt * (y - ops.gaussian(y, 1.0))
            hi = ops.dilate3(y) + 2 / 255
            lo = ops.erode3(y) - 2 / 255
            y = torch.where(osd > 0, y, torch.minimum(torch.maximum(sharp, lo), hi))
        uf = F.interpolate(
            torch.stack((u, v))[None], size=(self.h, self.w), mode="bilinear", align_corners=False
        )[0]
        rgb_in = to_rgb(y, uf[0], uf[1]).clamp(0, 1)
        rgb = rgb_in
        if self.lut is not None:
            graded = apply_lut(self.lut, rgb_in) * self.vignette
            if p.bloom > 0:
                graded = self._glow(graded, osd)
            rgb = rgb_in + (graded - rgb_in) * (p.look_strength / 100)
        if p.black_level != 0:
            rgb = black_level(rgb, p.black_level)
        rgb = (rgb * (1 - osd) + rgb_in * osd).clamp(0, 1)
        t0 = self._tick("grade", t0)

        # 4. motion blur
        if p.motion_blur > 0 and (1 in f.flows or -1 in f.flows):
            rgb = self._blur(rgb, f, osd)
        t0 = self._tick("motion_blur", t0)

        # 5. stabilisation: the picture moves, the OSD stays
        if self.steady is not None:
            moved = self.steady.warp(rgb, f.index)
            rgb = moved * (1 - osd) + rgb * osd
        t0 = self._tick("stabilisation", t0)

        # 6. OSD redraw (experimental): the font drawn crisply where the old text was read
        if self.redraw is not None:
            ly = to_ycc(rgb)[0]
            rgb = self.redraw.draw(rgb, ly, self.boxes)
        t0 = self._tick("osd_redraw", t0)

        # 7. SuperView
        if self.sv_grid is not None:
            rgb = F.grid_sample(
                rgb[None], self.sv_grid, mode="bicubic", padding_mode="border", align_corners=True
            )[0].clamp(0, 1)
        t0 = self._tick("superview", t0)
        if self.capture == f.index:  # the preview keeps this frame's picture
            self.captured = rgb
        out = pack(rgb)
        self._tick("pack", t0)
        self.timings.frames += 1
        return out

    def _warp(self, x: Tensor, flow: Tensor) -> Tensor:
        h, w = x.shape[-2:]
        gx = (self.gx_h + flow[0]) * (2.0 / (w - 1)) - 1.0
        gy = (self.gy_h + flow[1]) * (2.0 / (h - 1)) - 1.0
        grid = torch.stack((gx, gy), -1)[None]
        return F.grid_sample(
            x[None], grid, mode="bilinear", padding_mode="border", align_corners=True
        )[0]

    def _glow(self, rgb: Tensor, osd: Tensor) -> Tensor:
        lk, s = self.look, self.p.bloom / 100
        lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
        hi = ((lum - 0.7) / 0.3).clamp(0, 1) ** 2 * (1 - osd)
        small = F.avg_pool2d((rgb * hi)[None], 4)[0]
        size = (self.h, self.w)
        glow = F.interpolate(ops.gaussian(small, 6.0)[None], size=size, mode="bilinear")[0]
        hal = F.interpolate(ops.gaussian(small[0:1], 2.5)[None], size=size, mode="bilinear")[0]
        tint = torch.tensor([1.0, 0.35, 0.1], device=rgb.device).view(3, 1, 1)
        return 1 - (1 - rgb) * (1 - lk.bloom * s * glow) * (1 - lk.halation * s * hal * tint)

    def _blur(self, rgb: Tensor, f: Frame, osd: Tensor) -> Tensor:
        shutter = self.p.motion_blur / 100
        size = (self.h, self.w)

        def full(d: int) -> Tensor | None:
            fl = f.flows.get(d)
            if fl is None:
                return None
            return F.interpolate(fl[None], size=size, mode="bilinear", align_corners=False)[0] * 2.0

        fwd, bwd = full(1), full(-1)
        fwd = fwd if fwd is not None else (-bwd if bwd is not None else None)
        bwd = bwd if bwd is not None else (-fwd if fwd is not None else None)
        if fwd is None or bwd is None:
            return rgb
        acc = torch.zeros_like(rgb)
        for i in range(BLUR_TAPS):
            s = (i + 0.5) / BLUR_TAPS - 0.5  # -0.5..0.5 of the shutter
            fl = fwd if s >= 0 else bwd
            a = abs(s) * shutter
            gx = (self.gx + fl[0] * a) * (2.0 / (self.w - 1)) - 1.0
            gy = (self.gy + fl[1] * a) * (2.0 / (self.h - 1)) - 1.0
            acc += F.grid_sample(
                rgb[None], torch.stack((gx, gy), -1)[None], mode="bilinear",
                padding_mode="border", align_corners=True,
            )[0]  # fmt: skip
        return acc / BLUR_TAPS * (1 - osd) + rgb * osd


BLACK_RANGE = 0.08  # the black point moves by this much (of full scale) at +-100


def black_level(rgb: Tensor, amount: float) -> Tensor:
    """Black level: negative lifts the shadows (black becomes |amount|/100 x BLACK_RANGE,
    everything scaled up from there), positive crushes them (that much of the darkest
    range becomes black, the rest stretched to full). White stays white."""
    b = BLACK_RANGE * abs(amount) / 100
    if amount < 0:
        return b + (1 - b) * rgb
    return ((rgb - b) / (1 - b)).clamp(0, 1)


def superview_grid(w: int, h: int, mode: str, device: torch.device) -> Tensor | None:
    """Sampling grid for SuperView at 1920 x h: source x = a*u + (1-a)*u^3, so the centre
    is stretched by SUPERVIEW_CENTRE and the edges take the rest of the 4:3 -> 16:9 widening.
    ``osd_out`` also magnifies the picture vertically towards the left and right edges."""
    if mode == "off":
        return None
    ow = SUPERVIEW_W
    u = torch.linspace(-1, 1, ow, device=device)
    a = min(ow / (w * SUPERVIEW_CENTRE), 1.45)
    gx = (a * u + (1 - a) * u**3).view(1, ow).expand(h, ow)
    gy = torch.linspace(-1, 1, h, device=device).view(h, 1).expand(h, ow)
    if mode == "osd_out":
        gy = gy / (1 + OSD_OUT * u.view(1, ow) ** 2)
    return torch.stack((gx, gy), -1)[None]


def pack(rgb: Tensor) -> Tensor:
    """R'G'B' 0..1 -> limited-range BT.709 yuv420p bytes."""
    y, cb, cr = to_ycc(rgb.clamp(0, 1))
    cb = F.avg_pool2d(cb[None, None], 2)[0, 0]
    cr = F.avg_pool2d(cr[None, None], 2)[0, 0]
    oy = (16 + 219 * y).round().clamp(0, 255)
    ou = (128 + 224 * cb).round().clamp(0, 255)
    ov = (128 + 224 * cr).round().clamp(0, 255)
    return torch.cat((oy.flatten(), ou.flatten(), ov.flatten())).to(torch.uint8)


class SteadyWarp:
    """Per-frame output -> output warps from a steadying :class:`anahd.steady.Plan`."""

    def __init__(
        self, mats: Tensor, size: tuple[int, int], zoom: float = 1.0, strength: float = 0.0
    ) -> None:
        self.mats = mats  # (N, 2, 3): output pixel -> where to sample, in output pixels
        self.w, self.h = size
        self.zoom, self.strength = zoom, strength  # for the result line
        ys, xs = torch.meshgrid(
            torch.arange(self.h, device=mats.device, dtype=torch.float32),
            torch.arange(self.w, device=mats.device, dtype=torch.float32),
            indexing="ij",
        )
        self.pos = torch.stack((xs, ys, torch.ones_like(xs)), -1)  # (H, W, 3)

    @classmethod
    def from_plan(
        cls,
        plan: Plan,
        amount: float,
        src: tuple[int, int],
        size: tuple[int, int],
        device: torch.device,
    ) -> SteadyWarp:
        """``plan``'s corrections scaled by ``amount`` (0..1), as output -> output warps: a
        steadied output pixel shows what the unsteadied frame has at P(M_i^-1(q)), with M_i
        the plan's source -> output matrix and P the plain resize."""
        cx, cy, ca = plan.cx * amount, plan.cy * amount, plan.ca * amount
        w, h = src
        zoom = _zoom_for(cx, cy, ca, w, h, SteadyConfig()) if amount > 0 else 1.0
        scaled = Plan(cx, cy, ca, zoom, plan.strength * amount)
        plain = np.vstack([scale_matrix(src, size), [0.0, 0.0, 1.0]])
        mats = []
        for i in range(len(cx)):
            m = np.vstack([matrix(scaled, i, src, size), [0.0, 0.0, 1.0]])
            mats.append((plain @ np.linalg.inv(m))[:2])
        t = torch.tensor(np.array(mats), dtype=torch.float32, device=device)
        strength = float(scaled.strength.mean()) if len(cx) else 0.0
        return cls(t, size, zoom, strength)

    def warp(self, rgb: Tensor, i: int) -> Tensor:
        if i >= len(self.mats):
            return rgb
        src = self.pos @ self.mats[i].T  # (H, W, 2)
        gx = src[..., 0] * (2.0 / (self.w - 1)) - 1.0
        gy = src[..., 1] * (2.0 / (self.h - 1)) - 1.0
        return F.grid_sample(
            rgb[None], torch.stack((gx, gy), -1)[None], mode="bicubic",
            padding_mode="border", align_corners=True,
        )[0].clamp(0, 1)  # fmt: skip
