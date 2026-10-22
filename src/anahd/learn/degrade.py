"""On-the-fly analog degradation for training the learned restorer.

Everything works on GPU batches of short clips shaped (B, T, 3, H, W): YUV, full range,
Y in [0, 1], U and V centred on 0.5. Clean frames go in, frames that look like a real analog
DVR recording come out. Two sources are mixed:

* real residuals harvested from his own clips (source minus cleaned picture, see
  `noisebank.py`), so grain, line noise, dot crawl and rainbow match the C03 + his VTX;
* synthetic composite-video artefacts in the spirit of ntsc-rs (chroma bleed and delay,
  ringing, cross-colour rainbow, dot crawl, streaky noise, snow, line jitter, breakup bands).

Per-clip traits (bleed width, carrier frequency, strength) are drawn once per clip and kept
across its frames, as a real link would; per-frame noise is drawn independently, and the
carrier phase advances frame to frame so rainbow and dot crawl flicker like the real thing.
Spatial sizes are in pixels of the frame the model sees at inference (960x720 active area),
because the real residuals are harvested at that scale.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class DegradeConfig:
    p_real: float = 0.85  # chance a clip gets real harvested residuals
    real_gain: tuple[float, float] = (0.5, 1.6)
    p_synth_noise: float = 0.5
    luma_sigma: tuple[float, float] = (0.001, 0.012)  # DVR grain is low (~0.5-0.9 of 255)
    chroma_sigma: tuple[float, float] = (0.002, 0.02)
    p_bleed: float = 0.8
    p_ringing: float = 0.5
    p_rainbow: float = 0.6
    p_dotcrawl: float = 0.6
    p_snow: float = 0.5
    p_jitter: float = 0.4
    p_soften: float = 0.5
    soft_w: tuple[float, float] = (0.4, 1.1)  # horizontal blur sigma, pixels
    soft_v: tuple[float, float] = (0.0, 0.0)  # vertical blur sigma (0: none)
    # Leftover breakup after repair (still clearly visible on most real clips), per frame.
    p_band_centre: float = 0.3
    p_band_neighbour: float = 0.15
    p_real_band: float = 0.85  # real harvested strip vs synthetic band
    # A burst: bands in the centre and both adjacent frames (the neighbours are damaged too).
    p_burst: float = 0.0


def _hblur(x: Tensor, width: Tensor) -> Tensor:
    """Horizontal blur with a per-sample width (pixels), via a Gaussian kernel."""
    b = x.shape[0]
    k = 13
    pos = torch.arange(k, device=x.device, dtype=x.dtype) - k // 2
    sig = width.clamp(min=0.3).view(b, 1)
    ker = torch.exp(-0.5 * (pos.view(1, k) / sig) ** 2)
    ker = ker / ker.sum(dim=1, keepdim=True)
    c, h, w = x.shape[1], x.shape[2], x.shape[3]
    xp = F.pad(x.reshape(1, b * c, h, w), (k // 2, k // 2, 0, 0), mode="replicate")
    ker = ker.repeat_interleave(c, dim=0).view(b * c, 1, 1, k)
    return F.conv2d(xp, ker, groups=b * c).view(b, c, h, w)


def _hshift(x: Tensor, shift: Tensor) -> Tensor:
    """Shift each sample (or each row, if shift is (B, H)) right by `shift` pixels, subpixel."""
    b, _, h, w = x.shape
    ys = torch.linspace(-1, 1, h, device=x.device, dtype=x.dtype)
    xs = torch.linspace(-1, 1, w, device=x.device, dtype=x.dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    s = shift.view(b, -1, 1) if shift.dim() == 2 else shift.view(b, 1, 1)
    gx = gx.unsqueeze(0) - s * (2.0 / max(w - 1, 1))
    grid = torch.stack((gx, gy.unsqueeze(0).expand_as(gx)), dim=-1)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=True)


def _u(b: int, lo: float, hi: float, dev: torch.device) -> Tensor:
    return torch.empty(b, device=dev).uniform_(lo, hi)


def _on(b: int, p: float, dev: torch.device) -> Tensor:
    return (torch.rand(b, device=dev) < p).float()


class AnalogDegrader:
    def __init__(
        self, cfg: DegradeConfig, noise_bank: Tensor | None, bands: list[Tensor] | None = None
    ) -> None:
        """noise_bank: (N, 3, P, P) signed residual patches (Y, U, V), on the training device.
        bands: real breakup strips (3, h, W) cut from real recordings
        (`noisebank.harvest_bands`)."""
        self.cfg = cfg
        self.bank = noise_bank
        self.bands = bands or []

    def __call__(self, clean: Tensor) -> Tensor:
        with torch.no_grad():
            return self._degrade(clean)

    def _degrade(self, clean: Tensor) -> Tensor:
        cfg = self.cfg
        b, t, _, h, w = clean.shape
        dev, dt = clean.device, clean.dtype
        x = clean.clone()

        # Per-clip traits.
        bleed_on = _on(b, cfg.p_bleed, dev)
        bleed_w = _u(b, 0.8, 4.0, dev)
        bleed_shift = _u(b, -0.5, 2.5, dev) * bleed_on
        ring_on = _on(b, cfg.p_ringing, dev)
        ring_amp = _u(b, 0.1, 0.6, dev) * ring_on
        ring_off = _u(b, 1.0, 3.0, dev)
        rb_amp = _u(b, 0.15, 0.9, dev) * _on(b, cfg.p_rainbow, dev)
        dc_amp = _u(b, 0.03, 0.18, dev) * _on(b, cfg.p_dotcrawl, dev)
        carrier = _u(b, 0.15, 0.30, dev)  # cycles per pixel at 960 px wide
        line_step = torch.where(torch.rand(b, device=dev) < 0.5, 0.5, 0.25)  # NTSC-ish / PAL-ish
        soft_on = _on(b, cfg.p_soften, dev)
        soft_w = _u(b, *cfg.soft_w, dev)
        soft_v = _u(b, *cfg.soft_v, dev) if cfg.soft_v[1] > 0 else None
        real_on = _on(b, cfg.p_real, dev) if self.bank is not None else torch.zeros(b, device=dev)
        real_gain = _u(b, *cfg.real_gain, dev) * real_on
        sn_on = _on(b, cfg.p_synth_noise, dev)
        ls = _u(b, *cfg.luma_sigma, dev) * sn_on
        cs = _u(b, *cfg.chroma_sigma, dev) * sn_on
        snow_d = _u(b, 0.0002, 0.004, dev) * _on(b, cfg.p_snow, dev)
        jit = _u(b, 0.1, 0.8, dev) * _on(b, cfg.p_jitter, dev)
        phase0 = _u(b, 0.0, 2 * math.pi, dev)
        burst = _on(b, cfg.p_burst, dev)

        xs = torch.arange(w, device=dev, dtype=dt).view(1, 1, w)
        rows = torch.arange(h, device=dev, dtype=dt).view(1, h, 1)

        out = []
        for i in range(t):
            f = x[:, i]
            y, uv = f[:, :1], f[:, 1:] - 0.5

            # Camera / link softness, then chroma bandwidth loss and delay.
            soft = _hblur(y, soft_w)
            if soft_v is not None:
                soft = _hblur(soft.transpose(-1, -2), soft_v).transpose(-1, -2)
            y = torch.lerp(y, soft, soft_on.view(b, 1, 1, 1))
            uv_b = _hshift(_hblur(uv, bleed_w * 1.5), bleed_shift)
            uv = torch.lerp(uv, uv_b, bleed_on.view(b, 1, 1, 1))

            # Ringing: overshoot echo after edges.
            hp = y - _hblur(y, torch.full_like(ring_amp, 1.5))
            y = y + ring_amp.view(b, 1, 1, 1) * _hshift(hp, ring_off)

            # Composite carrier phase: per pixel, per line, per frame.
            phase = (
                2 * math.pi * carrier.view(b, 1, 1) * xs
                + 2 * math.pi * line_step.view(b, 1, 1) * rows
                + phase0.view(b, 1, 1)
                + math.pi * i
            )
            cos, sin = torch.cos(phase).unsqueeze(1), torch.sin(phase).unsqueeze(1)

            # Rainbow: fine luma detail decoded as colour.
            ybp = _hblur(y, torch.full_like(rb_amp, 0.6)) - _hblur(y, torch.full_like(rb_amp, 2.0))
            uv = uv + rb_amp.view(b, 1, 1, 1) * ybp * torch.cat((cos, sin), dim=1)

            # Dot crawl: chroma transitions leaking into luma.
            uvh = uv - _hblur(uv, torch.full_like(dc_amp, 2.5))
            leak = (uvh[:, :1] * cos + uvh[:, 1:] * sin) * 4.0
            y = y + dc_amp.view(b, 1, 1, 1) * leak

            # Streaky noise: horizontally correlated, chroma coarser than luma.
            ny = _hblur(torch.randn_like(y), _u(b, 0.3, 1.5, dev)) * 1.6
            nc = _hblur(torch.randn_like(uv), _u(b, 0.8, 3.0, dev)) * 2.0
            rowy = torch.randn(b, 1, h, 1, device=dev, dtype=dt) * 0.25
            y = y + ls.view(b, 1, 1, 1) * (ny + rowy)
            uv = uv + cs.view(b, 1, 1, 1) * nc

            # Real harvested residuals.
            if self.bank is not None and bool(real_on.any()):
                res = self._bank_crop(self.bank, b, h, w, dev, dt)
                g = real_gain.view(b, 1, 1, 1)
                y = y + g * res[:, :1]
                uv = uv + g * res[:, 1:]

            # Snow: short bright/dark horizontal streaks.
            seeds = (torch.rand(b, 1, h, w, device=dev) < snow_d.view(b, 1, 1, 1)).to(dt)
            sign = torch.where(torch.rand(b, 1, h, w, device=dev) < 0.7, 1.0, -1.0).to(dt)
            streak = _hblur(seeds * sign, _u(b, 1.0, 5.0, dev)) * 6.0
            y = y + streak.clamp(-1, 1) * 0.6

            # Line jitter (time base error), small and per row.
            rj = torch.randn(b, h, device=dev, dtype=dt)
            rj = F.avg_pool1d(rj.unsqueeze(1), 5, 1, 2, count_include_pad=False).squeeze(1)
            f2 = _hshift(torch.cat((y, uv), dim=1), rj * jit.view(b, 1) * 2.0)

            # Leftover breakup bands.
            p = cfg.p_band_centre if i == t // 2 else cfg.p_band_neighbour
            sel = _on(b, p, dev)
            if abs(i - t // 2) <= 1:
                sel = torch.maximum(sel, burst)
            if bool(sel.any()):
                f2 = self._bands(f2, sel)
            out.append(f2)

        o = torch.stack(out, dim=1)
        o[:, :, 1:] += 0.5
        return o.clamp(0.0, 1.0)

    @staticmethod
    def _bank_crop(
        bank: Tensor, b: int, h: int, w: int, dev: torch.device, dt: torch.dtype
    ) -> Tensor:
        n, _, p, _ = bank.shape
        idx = torch.randint(0, n, (b,), device=dev)
        patch = bank[idx].to(dt)
        # Tile patches up to the crop size, random offset, random horizontal flip only
        # (vertical structure of line noise must stay horizontal).
        reps_y, reps_x = math.ceil(h / p) + 1, math.ceil(w / p) + 1
        tiled = patch.repeat(1, 1, reps_y, reps_x)
        oy, ox = int(torch.randint(0, p, (1,))), int(torch.randint(0, p, (1,)))
        crop = tiled[:, :, oy : oy + h, ox : ox + w]
        flip = torch.rand(b, device=dev) < 0.5
        return torch.where(flip.view(b, 1, 1, 1), crop.flip(-1), crop)

    def _bands(self, f: Tensor, sel: Tensor) -> Tensor:
        """1-3 breakup bands per selected sample: real strips when available, else synthetic
        (snow, torn and shifted lines, black dropouts, colour loss)."""
        _, _, h, w = f.shape
        dev = f.device
        out = f.clone()
        for k in sel.nonzero().flatten().tolist():
            for _ in range(int(torch.randint(1, 4, (1,)))):
                if self.bands and float(torch.rand(1)) < self.cfg.p_real_band:
                    strip = self.bands[int(torch.randint(0, len(self.bands), (1,)))]
                    sh, sw = strip.shape[1], strip.shape[2]
                    bh = min(sh, h)
                    x0 = int(torch.randint(0, max(sw - w, 0) + 1, (1,)))
                    y0 = int(torch.randint(0, h - bh + 1, (1,)))
                    piece = strip[:, :bh, x0 : x0 + w].to(f.dtype)
                    if piece.shape[2] < w:
                        piece = F.pad(piece, (0, w - piece.shape[2]))
                    out[k, :, y0 : y0 + bh] += piece * float(torch.empty(1).uniform_(0.7, 1.2))
                else:
                    out[k] = self._synth_band(out[k], dev)
        return out

    def _synth_band(self, f: Tensor, dev: torch.device) -> Tensor:
        _, h, w = f.shape
        bh = int(torch.randint(2, max(3, h // 6), (1,)))
        y0 = int(torch.randint(0, h - bh + 1, (1,)))
        band = f[:, y0 : y0 + bh].clone()
        # Torn lines: each row shifted, more at the band centre.
        tear = float(torch.empty(1).uniform_(0, 12))
        prof = torch.sin(torch.linspace(0, math.pi, bh, device=dev)) * tear
        band = _hshift(band.unsqueeze(0), (prof + torch.randn(bh, device=dev)).view(1, bh))[0]
        # Snow: dense bright/dark horizontal streaks.
        dens = float(torch.empty(1).uniform_(0.02, 0.25))
        seeds = (torch.rand(1, 1, bh, w, device=dev) < dens).to(f.dtype)
        sign = torch.where(torch.rand(1, 1, bh, w, device=dev) < 0.6, 1.0, -1.0).to(f.dtype)
        width = torch.tensor([float(torch.empty(1).uniform_(1, 6))], device=dev)
        streak = _hblur(seeds * sign, width)
        gain = float(torch.empty(1).uniform_(0.2, 0.8))
        band[:1] = band[:1] + (streak[0] * 4).clamp(-1, 1) * gain
        # Black dropout pixels.
        drop = torch.rand(bh, w, device=dev) < float(torch.empty(1).uniform_(0, 0.08))
        band[0] = torch.where(drop, torch.zeros_like(band[0]), band[0])
        # Colour loss or wrong colour.
        if float(torch.rand(1)) < 0.5:
            band[1:] = 0.5 + (band[1:] - 0.5) * float(torch.empty(1).uniform_(0, 0.4))
        sigma = torch.tensor([0.04, 0.03, 0.03], device=dev).view(3, 1, 1)
        band = band + torch.randn_like(band) * sigma
        # Feather into the frame.
        a = torch.ones(bh, device=dev)
        if bh > 4:
            a[0] = a[-1] = 0.5
        f = f.clone()
        f[:, y0 : y0 + bh] = torch.lerp(f[:, y0 : y0 + bh], band, a.view(1, bh, 1))
        return f
