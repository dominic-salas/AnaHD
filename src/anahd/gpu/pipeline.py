"""One recording in, one clean HD file out, on the GPU.

The recording is measured on the CPU exactly as in :mod:`anahd.pipeline` (picture area, OSD,
levels). Frames are then decoded in one thread, cleaned on the GPU, and encoded in another,
so decoding, cleaning and encoding overlap.
"""

from __future__ import annotations

import queue
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from anahd import snapshot as snap
from anahd.clean import CleanConfig
from anahd.errors import AnaHDError, MediaError
from anahd.gpu import ops
from anahd.gpu.breakup import (
    BreakupConfig,
    Detector,
    chroma_noise,
    fill_rows,
    flag_rows,
    row_mask,
    slots,
    trimmed_mean,
)
from anahd.gpu.device import cuda_device
from anahd.gpu.flow import Encoded, Flow
from anahd.gpu.upscale import Upscaler, reduce
from anahd.look import LookConfig
from anahd.media import Crop, Encoder, VideoInfo, probe, read_raw
from anahd.model.infer import LearnedRestorer
from anahd.pipeline import Measured, Options, Result, _windows, measure
from anahd.remaster.params import RemasterParams
from anahd.remaster.stage import Frame as RFrame
from anahd.remaster.stage import Remaster, SteadyWarp

LEARNED_OFFSETS = (-2, -1, 1, 2)  # neighbours the learned restorer sees, around the centre


@dataclass(frozen=True)
class GpuConfig:
    flow_iters: int = 6  # RAFT refinement steps (more barely helps once refined below)
    # Lucas-Kanade refinement of RAFT's flow at the source size: RAFT-small runs at half
    # size and misses sub-pixel motion; these steps bring alignment past the CPU's DIS.
    refine_iters: int = 3
    refine_window: int = 5
    refine_damping: float = 0.5  # (levels/pixel)^2; keeps flat areas from drifting
    ai: bool = False  # learned 2x upscale (gpu.upscale) instead of Lanczos + sharpening
    repair: bool = False  # replace breakup bands and colour loss from the neighbours
    graphs: bool = True  # replay the repair from CUDA graphs (off: plain eager, for tests)
    breakup: BreakupConfig = field(default_factory=BreakupConfig)
    # Learned restorer (anahd.model) checkpoint: replaces fusion, destripe and derainbow with
    # a small network over the aligned frames at -2..+2. None: the hand-made steps.
    learned: Path | None = None
    learned_strength: float = 1.0  # 0 = the repaired centre frame, 1 = the network's output
    # Upload frames from pinned memory without a blocking copy, so the host does not wait for
    # the GPU to drain each frame (the output is the same, bit for bit; 1% faster measured).
    pinned_upload: bool = True


@dataclass
class RepairStats:
    frames: int = 0  # frames with any rows repaired
    rows: int = 0
    whole: int = 0  # whole-bad frames rebuilt
    left: int = 0  # whole-bad frames left alone (outage too long, or no motion known yet)
    colour: int = 0  # frames whose lost colour was restored
    pixels: int = 0  # frames with more than 1% of their pixels repaired as colour snow


@dataclass
class Aligned:
    """Neighbours warped onto a centre frame; S slots (missing ones repeat the last)."""

    flow: Tensor  # (S, 2, H, W) centre -> neighbour, pixels
    y: Tensor  # (S, H, W) warped luma
    u: Tensor  # (S, H/2, W/2) warped chroma
    v: Tensor
    valid: Tensor  # (S, H, W) 0..1: the warp sampled picture (not OSD, not outside)
    offsets: list[int]  # frame distance of each slot
    dist: Tensor  # (S, 1, 1) distance weight, 0 for missing slots
    ny: Tensor  # (S, H, W) unwarped neighbour planes, for re-warping
    nu: Tensor
    nv: Tensor
    nn: Tensor | None = None  # (S, H/2, W/2) chroma noise of each neighbour, unwarped
    wn: Tensor | None = None  # the same, warped
    _blur: Tensor | None = None

    @property
    def blur(self) -> Tensor:
        """Warped luma lightly blurred, as fusion compares it (computed once)."""
        if self._blur is None:
            self._blur = ops.gaussian(self.y, 1.2, size=(5, 5))
        return self._blur


@dataclass
class Prepared:
    y: Tensor  # (H, W) float, full range
    u: Tensor  # (H/2, W/2)
    v: Tensor
    enc: Encoded
    yb: Tensor  # luma lightly blurred, 0..1, for flow refinement
    noise: Tensor | None = None  # chroma noise (breakup.chroma_noise), when repairing


def _ycc_matrix(kr: float, kb: float) -> np.ndarray:
    """Y'CbCr (y in 0..1, c in -0.5..0.5) -> R'G'B'."""
    kg = 1.0 - kr - kb
    return np.array(
        [
            [1.0, 0.0, 2 * (1 - kr)],
            [1.0, -2 * (1 - kb) * kb / kg, -2 * (1 - kr) * kr / kg],
            [1.0, 2 * (1 - kb), 0.0],
        ]
    )


def to_bt709(bt709_source: bool) -> np.ndarray:
    """3x3 from the source's full-range Y'CbCr to BT.709 Y'CbCr (normalized)."""
    src = _ycc_matrix(0.2126, 0.0722) if bt709_source else _ycc_matrix(0.299, 0.114)
    dst = np.linalg.inv(_ycc_matrix(0.2126, 0.0722))
    return np.asarray(dst @ src, dtype=np.float64)


class Cleaner:
    """Per-recording state on the GPU; ``prepare`` each frame, then ``clean`` each centre."""

    def __init__(
        self,
        crop: Crop,
        osd: np.ndarray,
        curve: np.ndarray,
        size: tuple[int, int],
        info: VideoInfo,
        clean: CleanConfig,
        lk: LookConfig,
        gcfg: GpuConfig,
        device: torch.device,
        params: RemasterParams | None = None,
    ) -> None:
        self.dev = device
        self.gcfg = gcfg
        self.params = params or RemasterParams()
        # Clean-up strength: the restorer's (or fusion's) change over the plain centre frame.
        learned_k = gcfg.learned_strength if gcfg.learned is not None else 1.0
        self.cleanup = learned_k * self.params.cleanup_strength / 100
        self.last_flow: tuple[Tensor, list[int]] | None = None  # for the remaster stage
        self.kx = torch.tensor([[[[-0.5, 0.0, 0.5]]]], device=device, dtype=torch.float16)
        self.cfg, self.look, self.info = clean, lk, info
        w, h = crop.width, crop.height
        self.w, self.h = w, h
        self.size = size

        def t(a: np.ndarray) -> Tensor:
            return torch.from_numpy(np.ascontiguousarray(a)).to(device)

        self.osd = t(osd)
        self.keep = (~self.osd).float()
        self.keep_half = ops.area_half(self.keep)
        nearest = cv2.INTER_NEAREST
        self.osd_half = t(
            cv2.resize(osd.astype(np.uint8), (w // 2, h // 2), interpolation=nearest) > 0
        )
        small = (round(w * clean.flow_scale), round(h * clean.flow_scale))
        self.small = small
        self.fill = ops.Filler(
            t(cv2.resize(osd.astype(np.uint8), small, interpolation=nearest) > 0)
        )
        self.flow = Flow(small, 2 * clean.radius, gcfg.flow_iters, device)
        self.curve = t(curve.astype(np.float32))
        ys, xs = torch.meshgrid(
            torch.arange(h, device=device, dtype=torch.float32),
            torch.arange(w, device=device, dtype=torch.float32),
            indexing="ij",
        )
        self.grid = torch.stack([xs, ys], 0)
        hy, hx = torch.meshgrid(
            torch.arange(h // 2, device=device, dtype=torch.float32),
            torch.arange(w // 2, device=device, dtype=torch.float32),
            indexing="ij",
        )
        self.grid_half = torch.stack([hx, hy], 0)
        big_w, big_h = size
        self.up_lanczos = ops.Resampler((w, h), size, "lanczos4", device)
        self.up_linear = ops.Resampler((w, h), size, "linear", device)
        self.up_chroma = ops.Resampler((w // 2, h // 2), (big_w // 2, big_h // 2), "cubic", device)
        self.m709 = to_bt709(info.bt709)
        self.ai = Upscaler((w, h), device) if gcfg.ai else None
        kr, kb = (0.2126, 0.0722) if info.bt709 else (0.299, 0.114)
        self.rgb_from = torch.from_numpy(_ycc_matrix(kr, kb)).float().to(device)
        self.ycc709 = (
            torch.from_numpy(np.linalg.inv(_ycc_matrix(0.2126, 0.0722))).float().to(device)
        )
        repair = gcfg.repair and self.params.static_repair > 0
        self.detector = Detector(self.keep, gcfg.breakup) if repair else None
        self.learned: LearnedRestorer | None = None
        if gcfg.learned is not None:
            # Full strength here: the clean-up sliders mix it with the plain frame (_mix).
            self.learned = LearnedRestorer(gcfg.learned, device, 1.0)
        self._near_cache: dict[tuple[int, ...], list[int | None]] = {}
        self._stats = torch.zeros(6, dtype=torch.long, device=device)  # RepairStats fields
        self._velocity = torch.zeros(2, 2, h, w, device=device)  # last good flow to -1, +1
        self._known = torch.zeros((), dtype=torch.bool, device=device)
        self._bad_run = torch.zeros((), dtype=torch.long, device=device)
        self._flags = torch.zeros(h, dtype=torch.bool, device=device)  # last frame's rows
        self._slot_cache: dict[tuple[int | None, ...], Tensor] = {}
        self._sel_cache: dict[tuple[int, ...], Tensor | slice] = {}
        self._dist_cache: dict[tuple[int, ...], Tensor] = {}
        self._graphs: dict[tuple[object, ...], _Graph] = {}
        self._pinned: list[Tensor] = []  # sync-free uploads (see _pinned_upload)
        self._pinned_events: list[torch.cuda.Event | None] = []
        self._pinned_next = 0

    # --- per frame ---

    def upload(self, raw: np.ndarray) -> tuple[Tensor, Tensor, Tensor]:
        w, h = self.w, self.h
        if self.gcfg.pinned_upload:
            buf = self._pinned_upload(raw).float()
        else:
            buf = torch.from_numpy(raw).to(self.dev, non_blocking=False).float()
        luma, chroma = w * h, (w // 2) * (h // 2)
        y = buf[:luma].view(h, w)
        u = buf[luma : luma + chroma].view(h // 2, w // 2)
        v = buf[luma + chroma :].view(h // 2, w // 2)
        if not self.info.full_range:
            y = ((y - 16.0) * (255.0 / 219.0)).clamp(0, 255).round()
            u = ((u - 128.0) * (255.0 / 224.0) + 128.0).clamp(0, 255).round()
            v = ((v - 128.0) * (255.0 / 224.0) + 128.0).clamp(0, 255).round()
        return y, u, v

    def _pinned_upload(self, raw: np.ndarray) -> Tensor:
        """``raw`` on the GPU without the host waiting: copied into one of a few pinned
        buffers, then sent asynchronously. A buffer is refilled only once the copy that
        last read it has finished (its event), which in practice has long happened."""
        if not self._pinned or self._pinned[0].numel() != raw.size:
            self._pinned = [
                torch.empty(raw.size, dtype=torch.uint8, pin_memory=True)
                for _ in range(PINNED_SLOTS)
            ]
            self._pinned_events = [None] * PINNED_SLOTS
        i = self._pinned_next
        self._pinned_next = (i + 1) % PINNED_SLOTS
        event = self._pinned_events[i]
        if event is not None:
            event.synchronize()
        host = self._pinned[i]
        host.numpy()[:] = raw.reshape(-1)
        dev = host.to(self.dev, non_blocking=True)
        done = torch.cuda.Event()  # type: ignore[no-untyped-call, unused-ignore]
        done.record()
        self._pinned_events[i] = done
        return dev

    def prepare(self, raw: np.ndarray) -> Prepared:
        y, u, v = self.upload(raw)
        small = ops.area_half(y).round()
        if self.small != (small.shape[1], small.shape[0]):
            small = F.interpolate(y[None, None], size=(self.small[1], self.small[0]), mode="area")[
                0, 0
            ].round()
        small = self.fill(small)
        noise = None if self.detector is None else chroma_noise(u, v)  # once per frame
        return Prepared(y, u, v, self.flow.encode(small), ops.gaussian(y, 1.0) / 255.0, noise)

    def align(self, centre: Prepared, neighbours: list[tuple[int, Prepared]]) -> Aligned:
        """Each neighbour warped onto the centre (``neighbours`` must not be empty)."""
        cfg = self.cfg
        h, w = self.h, self.w
        slots = self.flow.slots
        flows = self.flow.flows(centre.enc, [nb.enc for _, nb in neighbours])
        n = len(neighbours)
        idx = [min(i, n - 1) for i in range(slots)]
        offsets = [neighbours[i][0] for i in idx]
        dist = self._dist(tuple(d for d, _ in neighbours))
        full = F.interpolate(flows, size=(h, w), mode="bilinear", align_corners=False)
        full = self.refine(
            centre.yb, torch.stack([neighbours[i][1].yb for i in idx]), full / cfg.flow_scale
        )
        return self.warp(
            full,
            torch.stack([neighbours[i][1].y for i in idx]),
            torch.stack([neighbours[i][1].u for i in idx]),
            torch.stack([neighbours[i][1].v for i in idx]),
            offsets,
            dist,
            None if centre.noise is None else torch.stack([_noise(neighbours[i][1]) for i in idx]),
        )

    def _dist(self, offsets: tuple[int, ...]) -> Tensor:
        """(S, 1, 1) distance weights for the neighbours at ``offsets``, 0 for missing slots;
        built once per window shape."""
        if offsets not in self._dist_cache:
            dist = torch.zeros(self.flow.slots, 1, 1)
            for i, d in enumerate(offsets):
                dist[i, 0, 0] = max(0.0, 1.0 - self.cfg.distance_falloff * abs(d))
            self._dist_cache[offsets] = dist.to(self.dev)
        return self._dist_cache[offsets]

    def warp(
        self,
        full: Tensor,
        ny: Tensor,
        nu: Tensor,
        nv: Tensor,
        offsets: list[int],
        dist: Tensor,
        nn: Tensor | None = None,
    ) -> Aligned:
        h, w = self.h, self.w
        g = _norm_grid(self.grid[None] + full, w, h)  # full: (S, 2, H, W)
        warped = F.grid_sample(
            ny[:, None], g, mode="bicubic", padding_mode="border", align_corners=True
        )[:, 0]
        valid = F.grid_sample(
            self.keep.expand(len(offsets), 1, h, w),
            g,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )[:, 0].clamp(0.0, 1.0)
        half = ops.area_half(full) * 0.5
        gh = _norm_grid(self.grid_half[None] + half, w // 2, h // 2)
        wu = F.grid_sample(
            nu[:, None], gh, mode="bilinear", padding_mode="border", align_corners=True
        )[:, 0]
        wv = F.grid_sample(
            nv[:, None], gh, mode="bilinear", padding_mode="border", align_corners=True
        )[:, 0]
        wn = None
        if nn is not None:
            wn = F.grid_sample(
                nn[:, None], gh, mode="bilinear", padding_mode="border", align_corners=True
            )[:, 0]
        return Aligned(full, warped, wu, wv, valid, offsets, dist, ny, nu, nv, nn, wn)

    def fuse(self, centre: Prepared, neighbours: list[tuple[int, Prepared]]) -> tuple[Tensor, ...]:
        if not neighbours:
            return centre.y, centre.u, centre.v
        return self.blend(centre.y, centre.u, centre.v, self.align(centre, neighbours))

    def blend(self, cy: Tensor, cu: Tensor, cv: Tensor, a: Aligned) -> tuple[Tensor, ...]:
        """Temporal fusion of the centre planes with the warped neighbours."""
        cfg = self.cfg
        warped, wu, wv = a.y, a.u, a.v
        ref = ops.gaussian(cy, 1.2, size=(5, 5))
        diff = (a.blur - ref).abs()
        match = 1.0 / (1.0 + (diff / cfg.match_sigma) ** 4)
        weight = match * a.valid * self.keep * a.dist
        y = (cy + (weight * warped).sum(0)) / (1.0 + weight.sum(0))

        centre_u, centre_v = cu, cv
        ref_u = ops.gaussian(centre_u, 0.8, size=(3, 3))
        ref_v = ops.gaussian(centre_v, 0.8, size=(3, 3))
        cdiff = torch.maximum(
            (ops.gaussian(wu, 0.8, size=(3, 3)) - ref_u).abs(),
            (ops.gaussian(wv, 0.8, size=(3, 3)) - ref_v).abs(),
        )
        cmatch = 1.0 / (1.0 + (cdiff / cfg.chroma_sigma) ** 4)
        cweight = ops.area_half(weight) * cmatch * self.keep_half
        csum = 1.0 + cweight.sum(0)
        u = (centre_u + (cweight * wu).sum(0)) / csum
        v = (centre_v + (cweight * wv).sum(0)) / csum
        return y, u, v

    def repair(self, det: Detector, centre: Prepared, a: Aligned) -> Aligned:
        """Replace the centre's breakup rows, lost colour and colour snow from the warped
        neighbours, in place, so later frames also see the repaired frame as a neighbour.
        Returns the alignment to fuse with, re-warped where the centre's flow was replaced.

        Two steps, each replayed from a CUDA graph with a full window (they are many small
        kernels): detection, then the fix. The one host sync per frame picks between the fix
        with rows (flow repair, re-warp) and the cheaper one without, which most frames take.
        With no damage the masks are zero and the output equals the unrepaired one."""
        if a.nn is None or a.wn is None:
            raise AnaHDError("repair needs the chroma noise maps (prepared with a detector)")
        offsets = tuple(a.offsets)
        args = (centre.y, centre.u, centre.v, a.flow, a.y, a.u, a.v, a.valid, a.ny, a.nu, a.nv)
        noise = (a.nn, a.wn)

        def detect(*t: Tensor) -> tuple[Tensor, ...]:
            return self._detect(det, list(offsets), *t)

        found = self._run(("detect", offsets), detect, (*args[:8], a.nu, a.nv))
        # Is the row path needed at all? Reading that back makes the host wait for the GPU,
        # but always running the row path instead measured 4% slower (2026-10-28).
        rows = bool(found[-1])

        def fix(*t: Tensor) -> tuple[Tensor, ...]:
            return self._fix(det, list(offsets), rows, *t)

        out = self._run(("fix", rows, offsets), fix, (*args, *noise, *found[:-1]))
        centre.y, centre.u, centre.v = out[:3]
        flow, y, u, v, valid, blur = out[3:]
        return Aligned(flow, y, u, v, valid, a.offsets, a.dist, a.ny, a.nu, a.nv, a.nn, a.wn, blur)

    def _run(
        self,
        key: tuple[object, ...],
        fn: Callable[..., tuple[Tensor, ...]],
        args: tuple[Tensor, ...],
    ) -> list[Tensor]:
        offsets = key[-1]
        if not isinstance(offsets, tuple):
            raise TypeError("the last key item must be the window's offsets")
        if not (self.gcfg.graphs and len(set(offsets)) == len(offsets)):
            return list(fn(*args))
        if key not in self._graphs:
            self._graphs[key] = _Graph(fn, args, self._state())
        return [x.clone() for x in self._graphs[key](*args)]

    def _state(self) -> list[Tensor]:
        return [self._stats, self._velocity, self._known, self._bad_run, self._flags]

    def _detect(
        self, det: Detector, offsets: list[int], cy: Tensor, cu: Tensor, cv: Tensor, *t: Tensor
    ) -> tuple[Tensor, ...]:
        """Rows to repair, whether the whole frame is rebuilt, and lost colour."""
        cfg = det.cfg
        dist = self._dist(tuple(offsets))
        _, wy, wu, wv, valid, nu, nv = t
        s = det.signals(cy, cu, cv, wy, wu, wv, valid, offsets, dist)
        flags = flag_rows(s, cfg)
        lost, coloured = det.colour_lost(cu, cv, nu, nv, dist)
        whole = flags.all()
        self._bad_run.copy_(torch.where(whole, self._bad_run + 1, 0))
        rebuild = whole & (self._bad_run <= cfg.bridge) & self._known
        flags = flags & ~(whole & ~rebuild)  # too long an outage, or motion unknown: leave it
        return flags, rebuild, whole, lost, coloured & lost, flags.any()

    def _fix(
        self,
        det: Detector,
        offsets: list[int],
        rows: bool,
        cy: Tensor,
        cu: Tensor,
        cv: Tensor,
        *t: Tensor,
    ) -> tuple[Tensor, ...]:
        cfg = det.cfg
        dist = self._dist(tuple(offsets))
        flow, wy, wu, wv, valid, ny, nu, nv, nn, wn, flags, rebuild, whole, lost, coloured = t
        if rows:
            # The centre's own flow is noise where it is damaged: interpolate it from the
            # clean rows, or for a whole frame assume the motion goes on unchanged.
            back, fwd = self._velocity[0], self._velocity[1]
            steady = torch.stack([(fwd if d > 0 else back) * abs(d) for d in offsets])
            flow = torch.where(rebuild, steady, fill_rows(flow, flags))
            a = self.warp(flow, ny, nu, nv, offsets, dist, nn)
        else:
            a = Aligned(flow, wy, wu, wv, valid, offsets, dist, ny, nu, nv, nn, wn)
        if -1 in offsets and 1 in offsets:
            vel = torch.stack([a.flow[offsets.index(-1)], a.flow[offsets.index(1)]])
            self._velocity.copy_(torch.where(whole, self._velocity, vel))
            self._known.copy_(self._known | ~whole)
        self._flags.copy_(flags)
        n = flags.sum()
        left = whole & ~rebuild
        self._stats[:5] += torch.stack(
            [(n > 0).long(), n, rebuild.long(), left.long(), lost.long()]
        )
        if rows:
            cy, cu, cv = self._replace_rows(cy, cu, cv, a, row_mask(flags, cfg))
        cu, cv = self._restore_colour(cu, cv, a, coloured)
        cy, cu, cv, share = self._repair_pixels(cfg, cy, cu, cv, a)
        self._stats[5] += (share > 0.01).long()
        return cy, cu, cv, a.flow, a.y, a.u, a.v, a.valid, a.blur

    @property
    def repaired_rows(self) -> Tensor:
        """(H,) bool: the rows of the last repaired frame flagged as breakup."""
        return self._flags

    @property
    def stats(self) -> RepairStats:
        return RepairStats(*(int(x) for x in self._stats.tolist()))

    def _slots(self, a: Aligned) -> Tensor | slice:
        """The distinct neighbour slots (missing ones repeat the last)."""
        uniq = [i for i, d in enumerate(a.offsets) if i == a.offsets.index(d)]
        return slots(uniq, self.dev, self._sel_cache)

    def _distinct(self, a: Aligned) -> Tensor:
        """(S,) 1 for the distinct neighbour slots, 0 for repeats."""
        key = (None, *a.offsets)
        if key not in self._slot_cache:
            m = torch.zeros(len(a.offsets))
            m[[i for i, d in enumerate(a.offsets) if i == a.offsets.index(d)]] = 1.0
            self._slot_cache[key] = m.to(self.dev)
        return self._slot_cache[key]

    def _replace_rows(
        self, cy: Tensor, cu: Tensor, cv: Tensor, a: Aligned, m: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        sel = self._slots(a)
        wy, valid, dist = a.y[sel], a.valid[sel], a.dist[sel]
        blur = a.blur[sel]
        agree = 1.0 / (1.0 + ((blur - trimmed_mean(blur)).abs() / self.cfg.match_sigma) ** 4)
        w = agree * valid * dist
        total = w.sum(0)
        rep = (w * wy).sum(0) / total.clamp_min(1e-6)
        # The OSD stays where it is: take it unwarped from the previous frame, which is
        # already repaired (else from the nearest). Overlay text the OSD mask missed is
        # found as pixels that hold still in the unwarped neighbours.
        src = a.offsets.index(-1) if -1 in a.offsets else a.offsets.index(min(a.offsets, key=abs))
        agree_still = (a.ny[sel] - a.ny[src]).abs() < self.gcfg.breakup.still
        still = (agree_still.sum(0) >= 3).float()  # the source and two more neighbours
        overlay = torch.maximum(1.0 - self.keep, still)
        mm = m[:, None] * (1.0 - overlay) * (total > 0.05)
        osd = m[:, None] * overlay
        cy = cy + (rep - cy) * mm + (a.ny[src] - cy) * osd
        wh = ops.area_half(w)
        th = wh.sum(0).clamp_min(1e-6)
        mh = ops.area_half(mm)
        oh = ops.area_half(osd)
        cu = cu + ((wh * a.u[sel]).sum(0) / th - cu) * mh + (a.nu[src] - cu) * oh
        cv = cv + ((wh * a.v[sel]).sum(0) / th - cv) * mh + (a.nv[src] - cv) * oh
        return cy, cu, cv

    def _repair_pixels(
        self, cfg: BreakupConfig, cy: Tensor, cu: Tensor, cv: Tensor, a: Aligned
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Replace pixels of colour snow and sparkle, wherever they are: each from the
        neighbours that are clean at that spot (or a smoothed centre when none is)."""
        use = self._distinct(a)[:, None, None] * a.dist
        valid_h = ops.area_half(a.valid)
        if a.wn is None or a.nn is None:
            raise AnaHDError("repair needs the chroma noise maps (prepared with a detector)")
        nn = a.wn  # (S, h, w) chroma noise of each warped neighbour
        cn = chroma_noise(cu, cv)
        usable = (valid_h > 0.5) & (use > 0.0)
        floor = torch.where(usable, nn, 1e3).amin(0).clamp_max(1e3)
        start = cfg.pixel_ratio * floor + cfg.pixel_margin
        m = ((cn - start) / cfg.pixel_soft).clamp(0.0, 1.0)
        # Blotches of wrong colour are smooth, so they show as disagreement with what the
        # neighbours agree on rather than as fine detail.
        dev = (cu - trimmed_mean(a.u)).abs() + (cv - trimmed_mean(a.v)).abs()
        m = torch.maximum(m, ((dev - cfg.blotch) / cfg.pixel_soft).clamp(0.0, 1.0) * (floor < 1e3))
        m = ops.box_mean(ops.dilate3(m), 3) * self.keep_half  # OSD fringes are not snow
        share = m.mean()

        def clean(n: Tensor) -> Tensor:
            return 1.0 / (1.0 + (n / cfg.pixel_noise) ** 4)

        q = clean(nn) * valid_h * use  # (S, h, w)
        qc = clean(cn)
        eps = 0.05  # the centre smoothed, for spots no source is clean at
        den = q.sum(0) + qc + eps
        out = []
        for c, warped in ((cu, a.u), (cv, a.v)):
            est = ((q * warped).sum(0) + qc * c + eps * ops.gaussian(c, 3.0)) / den
            out.append(c + (est - c) * m)

        def full(x: Tensor) -> Tensor:
            return F.interpolate(x.reshape(-1, 1, *x.shape[-2:]), size=cy.shape, mode="bilinear")[
                :, 0
            ]

        mf = full(m)[0]
        qf = full(q) * a.valid
        ref = trimmed_mean(a.blur)
        agree = 1.0 / (1.0 + ((a.blur - ref).abs() / self.cfg.match_sigma) ** 4)
        w = qf * agree
        qcf = full(qc)[0]
        den_y = w.sum(0) + qcf + eps
        est_y = ((w * a.y).sum(0) + qcf * cy + eps * ops.gaussian(cy, 1.0)) / den_y
        cy = cy + (est_y - cy) * mf

        # Sparkle: isolated luma specks that stand out both from their own surroundings and
        # from what the aligned neighbours agree on.
        tm = trimmed_mean(a.y)
        speck = (cy - ops.median3(cy)).abs()
        far = (cy - tm).abs()
        ms = (torch.minimum(speck - cfg.speck, far - cfg.speck_far) / cfg.pixel_soft).clamp(0, 1)
        ms = ops.box_mean(ops.dilate3(ms), 3) * self.keep * (a.valid.amin(0) > 0.5)
        cy = cy + (tm - cy) * ms

        # The OSD does not move: repair it from the unwarped neighbours clean there. Readouts
        # change between frames and warnings blink, so only confidently damaged OSD pixels
        # are touched, and the centre keeps extra weight there.
        osd = 1.0 - self.keep_half
        nno = a.nn
        floor_o = torch.where(use > 0.0, nno, 1e3).amin(0)
        start_o = cfg.pixel_ratio * floor_o + cfg.osd_margin
        mo = ((cn - start_o) / cfg.pixel_soft).clamp(0, 1)
        mo = ops.box_mean(ops.dilate3(mo), 3) * osd
        qo = clean(nno) * use
        qc = cfg.osd_centre * clean(cn)
        den_o = qo.sum(0) + qc + eps
        for i, plain in ((0, a.nu), (1, a.nv)):
            c = out[i]
            est = ((qo * plain).sum(0) + (qc + eps) * c) / den_o
            out[i] = c + (est - c) * mo
        qof = full(qo)
        qcf = full(qc)[0]
        est_y = ((qof * a.ny).sum(0) + (qcf + eps) * cy) / (qof.sum(0) + qcf + eps)
        cy = cy + (est_y - cy) * full(mo)[0]
        return cy, out[0], out[1], share

    def _restore_colour(
        self, cu: Tensor, cv: Tensor, a: Aligned, coloured: Tensor
    ) -> tuple[Tensor, Tensor]:
        use = self._distinct(a) * coloured.float()
        w = ops.area_half(a.valid) * a.dist * use[:, None, None]
        total = w.sum(0)
        ok = (total > 0.05).float() * self.keep_half
        total = total.clamp_min(1e-6)
        cu = cu + ((w * a.u).sum(0) / total - cu) * ok
        cv = cv + ((w * a.v).sum(0) / total - cv) * ok
        return cu, cv

    def refine(self, a: Tensor, b: Tensor, flow: Tensor) -> Tensor:
        """Lucas-Kanade steps from ``flow``: (H, W) centre, (S, H, W) neighbours, both
        blurred and in 0..1; flow (S, 2, H, W) in pixels."""
        g = self.gcfg
        h, w = a.shape
        win, damp = g.refine_window, g.refine_damping / 255.0**2
        kx = self.kx
        ref = a[None, None].half()
        for _ in range(g.refine_iters):
            grid = _norm_grid(self.grid[None] + flow, w, h)
            wb = F.grid_sample(
                b[:, None], grid, mode="bilinear", padding_mode="border", align_corners=True
            ).half()
            gx = F.conv2d(F.pad(wb, (1, 1, 0, 0), mode="replicate"), kx)
            gy = F.conv2d(F.pad(wb, (0, 0, 1, 1), mode="replicate"), kx.transpose(2, 3))
            dt = wb - ref
            t = torch.cat([gx * gx, gy * gy, gx * gy, gx * dt, gy * dt], 1)
            t = F.avg_pool2d(t, (1, win), 1, (0, win // 2), count_include_pad=False)
            t = F.avg_pool2d(t, (win, 1), 1, (win // 2, 0), count_include_pad=False).float()
            sxx, syy, sxy, bx, by = t[:, 0] + damp, t[:, 1] + damp, t[:, 2], t[:, 3], t[:, 4]
            det = sxx * syy - sxy * sxy
            du = (sxy * by - syy * bx) / det
            dv = (sxy * bx - sxx * by) / det
            flow = flow + torch.stack([du, dv], 1).clamp(-1.0, 1.0)
        return flow

    def destripe(self, y: Tensor) -> Tensor:
        cfg = self.cfg
        vertical = ops.gaussian(y, 0.1, 1.5, size=(1, 7))
        detail = (y - vertical).clamp(-cfg.stripe_clip, cfg.stripe_clip) * self.keep
        total = ops.row_box_sum(detail, cfg.stripe_window)
        count = ops.row_box_sum(self.keep, cfg.stripe_window)
        out = y - total / count.clamp_min(1.0)
        return torch.where(self.osd, y, out)

    def derainbow(self, y: Tensor, u: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        cfg = self.cfg
        yh = ops.area_half(y)
        texture = ops.gaussian((yh - ops.gaussian(yh, 1.0)).abs(), 2.0)
        cut = cfg.rainbow_cut * (texture / cfg.rainbow_texture).clamp(0.0, 1.0)
        cut = torch.where(self.osd_half, torch.zeros_like(cut), cut)
        out = []
        for c in (u, v):
            low = ops.gaussian(c, 1.5)
            out.append(low + (c - low) * (1.0 - cut))
        return out[0], out[1]

    def vibrance(self, u: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        cu, cv = u - 128.0, v - 128.0
        sat = torch.sqrt(cu * cu + cv * cv)
        gain = 1.0 + self.look.vibrance * (1.0 - sat / 64.0).clamp(0.0, 1.0)
        return (128.0 + cu * gain).clamp(0, 255), (128.0 + cv * gain).clamp(0, 255)

    def upscale(self, y: Tensor, u: Tensor, v: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        lk = self.look
        big = self.up_lanczos(y)
        sharp = big + lk.sharpen * (big - ops.gaussian(big, lk.sharpen_sigma))
        plain = self.up_linear(y)
        hi = ops.dilate3(plain) + lk.sharpen_margin
        lo = ops.erode3(plain) - lk.sharpen_margin
        big = torch.minimum(torch.maximum(sharp, lo), hi)
        guide = ops.area_half(big)
        bu = ops.guided(guide, self.up_chroma(u), 2, lk.chroma_eps)
        bv = ops.guided(guide, self.up_chroma(v), 2, lk.chroma_eps)
        return big, bu, bv

    def resize_learned(self, y: Tensor, u: Tensor, v: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """An upscaling network's planes to the output size: it already made the detail, so
        only an antialiased resample (no Lanczos ringing, no sharpening)."""
        bw, bh = self.size

        def rs(p: Tensor, h: int, w: int) -> Tensor:
            x = F.interpolate(p[None, None], size=(h, w), mode="bicubic", antialias=True)
            return x[0, 0]

        return rs(y, bh, bw), rs(u, bh // 2, bw // 2), rs(v, bh // 2, bw // 2)

    def to_output(self, y: Tensor, u: Tensor, v: Tensor) -> Tensor:
        """Round to 8 bit full range (as the CPU path hands to FFmpeg), then convert to
        limited-range BT.709 4:2:0 and pack Y, U, V into one uint8 buffer."""
        y = y.clamp(0, 255).round()
        u = u.clamp(0, 255).round()
        v = v.clamp(0, 255).round()
        m = self.m709
        yn, un, vn = y / 255.0, (u - 128.0) / 255.0, (v - 128.0) / 255.0
        uf = F.interpolate(un[None, None], size=y.shape, mode="bilinear", align_corners=False)[0, 0]
        vf = F.interpolate(vn[None, None], size=y.shape, mode="bilinear", align_corners=False)[0, 0]
        yh = ops.area_half(yn)
        y2 = m[0, 0] * yn + m[0, 1] * uf + m[0, 2] * vf
        u2 = m[1, 0] * yh + m[1, 1] * un + m[1, 2] * vn
        v2 = m[2, 0] * yh + m[2, 1] * un + m[2, 2] * vn
        out_y = (16.0 + 219.0 * y2).round().clamp(0, 255)
        out_u = (128.0 + 224.0 * u2).round().clamp(0, 255)
        out_v = (128.0 + 224.0 * v2).round().clamp(0, 255)
        return torch.cat([out_y.flatten(), out_u.flatten(), out_v.flatten()]).to(torch.uint8)

    def upscale_ai(self, ai: Upscaler, y: Tensor, u: Tensor, v: Tensor) -> Tensor:
        """Learned 2x on RGB, reduced to the output size, packed like :meth:`to_output`."""
        h, w = y.shape

        def up(c: Tensor) -> Tensor:
            return F.interpolate(c[None, None], size=(h, w), mode="bilinear")[0, 0]

        ycc = torch.stack([y / 255.0, (up(u) - 128.0) / 255.0, (up(v) - 128.0) / 255.0])
        rgb = torch.einsum("ij,jhw->ihw", self.rgb_from, ycc).clamp(0.0, 1.0)
        big = reduce(ai(rgb).clamp(0.0, 1.0), self.size)
        out = torch.einsum("ij,jhw->ihw", self.ycc709, (big * 255.0).round() / 255.0)
        out_y = (16.0 + 219.0 * out[0]).round().clamp(0, 255)
        out_u = (128.0 + 224.0 * ops.area_half(out[1])).round().clamp(0, 255)
        out_v = (128.0 + 224.0 * ops.area_half(out[2])).round().clamp(0, 255)
        return torch.cat([out_y.flatten(), out_u.flatten(), out_v.flatten()]).to(torch.uint8)

    def restore(
        self, centre: Prepared, neighbours: list[tuple[int, Prepared]]
    ) -> tuple[Tensor, Tensor, Tensor]:
        """The cleaned picture before the look and the upscale: fusion (after repair, when
        on), destripe and derainbow. Full-range planes in 0..255, chroma at half size."""
        a: Aligned | None = None
        if neighbours:
            a = self.align(centre, neighbours)
            if self.detector is not None:
                a = self._repair_some(centre, a)
        self.last_flow = None if a is None else (a.flow, list(a.offsets))
        if self.learned is not None:
            # The network replaces fusion, destripe and derainbow; only alignment (and repair)
            # are needed for it.
            return self._learned(self.learned, centre, a)
        cy, cu, cv = centre.y, centre.u, centre.v
        y, u, v = (cy, cu, cv) if a is None else self.blend(cy, cu, cv, a)
        yd = self.destripe(y)
        ud, vd = self.derainbow(yd, u, v)
        p, k = self.params, self.cleanup
        if k == 1.0 and p.noise == 100 and p.scan_lines == 100:
            return yd, ud, vd
        n, sl = p.noise / 100, p.scan_lines / 100
        y = cy + k * (n * (y - cy) + sl * (yd - y))
        return y, cu + k * n * (ud - cu), cv + k * n * (vd - cv)

    def _repair_some(self, centre: Prepared, a: Aligned) -> Aligned:
        """Breakup repair, mixed with the unrepaired centre by the static-repair slider."""
        if self.detector is None:
            return a
        share = self.params.static_repair / 100
        if share >= 1.0:
            return self.repair(self.detector, centre, a)
        before = (centre.y.clone(), centre.u.clone(), centre.v.clone())
        a = self.repair(self.detector, centre, a)
        centre.y = before[0] + share * (centre.y - before[0])
        centre.u = before[1] + share * (centre.u - before[1])
        centre.v = before[2] + share * (centre.v - before[2])
        return a

    def _mix(self, out: Tensor, centre: Tensor) -> Tensor:
        """The restorer's output (3, H, W, 0..255) mixed with the plain centre frame by the
        clean-up sliders. Its change over the centre is split into a row-structured part
        (horizontal lines: smooth along a row, changing from row to row; luma only) for
        scan-line removal, and the rest for noise and grain."""
        p, k = self.params, self.cleanup
        if k == 1.0 and p.noise == 100 and p.scan_lines == 100:
            return out
        r = out - centre
        ry = r[0]
        along = ops.gaussian(ry, 16.0, 1.0, size=(129, 1))
        rows = along - ops.gaussian(along, 1.0, 3.0, size=(1, 19))
        n, sl = p.noise / 100, p.scan_lines / 100
        y = centre[0] + k * (sl * rows + n * (ry - rows))
        return torch.stack((y, centre[1] + k * n * r[1], centre[2] + k * n * r[2]))

    def _near(self, offsets: list[int]) -> list[int | None]:
        """For the frames at -2, -1, +1, +2: the slot holding that frame, or the nearest one
        available (same side first); None for the centre itself."""
        key = tuple(offsets)
        if key not in self._near_cache:
            have = sorted(set(offsets) | {0})
            pick: list[int | None] = []
            for d in LEARNED_OFFSETS:

                def rank(x: int, d: int = d) -> tuple[int, bool]:
                    return abs(x - d), (x > 0) != (d > 0)

                o = min(have, key=rank)
                pick.append(None if o == 0 else offsets.index(o))
            self._near_cache[key] = pick
        return self._near_cache[key]

    def learned_stack(self, centre: Prepared, a: Aligned | None) -> Tensor:
        """(5, 3, H, W) YUV 4:4:4 in 0..1: the (repaired) centre and its neighbours at -2..+2
        warped onto it, as the learned restorer sees them."""
        near = self._near(a.offsets) if a is not None else [None] * len(LEARNED_OFFSETS)
        order = (*near[:2], None, *near[2:])
        ys = [centre.y if i is None or a is None else a.y[i] for i in order]
        us = [centre.u if i is None or a is None else a.u[i] for i in order]
        vs = [centre.v if i is None or a is None else a.v[i] for i in order]
        uv = F.interpolate(
            torch.stack(us + vs)[:, None],
            size=(self.h, self.w),
            mode="bilinear",
            align_corners=False,
        )[:, 0]
        n = len(ys)
        return torch.stack((torch.stack(ys), uv[:n], uv[n:]), 1) / 255.0

    def _learned(
        self,
        net: LearnedRestorer,
        centre: Prepared,
        a: Aligned | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """The network over the (repaired) centre and its aligned neighbours. The OSD keeps
        the centre, as fusion does (it gives the neighbours no weight there)."""
        stack = self.learned_stack(centre, a)
        s = net.scale
        if s == 1:
            out = self._mix(net(stack) * 255.0, stack[2] * 255.0)
            y = torch.where(self.osd, centre.y, out[0])
            u = torch.where(self.osd_half, centre.u, ops.area_half(out[1]))
            v = torch.where(self.osd_half, centre.v, ops.area_half(out[2]))
            return y, u, v
        # Upscaling network: planes come back at twice the size (chroma at the source luma
        # size), the OSD keeps the centre upscaled the same way.
        big_osd = self.osd.repeat_interleave(s, 0).repeat_interleave(s, 1)
        out = net(stack, big_osd) * 255.0
        plain = F.interpolate(stack[2:3], scale_factor=s, mode="bicubic", align_corners=False)
        out = self._mix(out, plain[0].clamp(0.0, 1.0) * 255.0)
        return out[0], ops.area_half(out[1]), ops.area_half(out[2])

    def to_709(self, y: Tensor, u: Tensor, v: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Full-range planes in the source's matrix (0..255) -> BT.709 Y in 0..1 and Cb, Cr
        centred on 0 (chroma at half size): what the remaster stage works on."""
        m = self.m709
        yn, un, vn = y.clamp(0, 255) / 255.0, (u - 128.0) / 255.0, (v - 128.0) / 255.0
        uf = F.interpolate(un[None, None], size=y.shape, mode="bilinear", align_corners=False)[0, 0]
        vf = F.interpolate(vn[None, None], size=y.shape, mode="bilinear", align_corners=False)[0, 0]
        yh = ops.area_half(yn)
        y2 = m[0, 0] * yn + m[0, 1] * uf + m[0, 2] * vf
        u2 = m[1, 0] * yh + m[1, 1] * un + m[1, 2] * vn
        v2 = m[2, 0] * yh + m[2, 1] * un + m[2, 2] * vn
        return y2.clamp(0, 1), u2, v2

    def clean_planes(
        self, centre: Prepared, neighbours: list[tuple[int, Prepared]]
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Like :meth:`clean`, but BT.709 planes at the output size (see :meth:`to_709`)
        instead of packed bytes, for the remaster stage."""
        y, u, v = self.restore(centre, neighbours)
        y = ops.apply_curve(y, self.curve)
        u, v = self.vibrance(u, v)
        if self.learned is not None and self.learned.scale != 1:
            return self.to_709(*self.resize_learned(y, u, v))
        if self.ai is not None:
            h, w = y.shape

            def up(c: Tensor) -> Tensor:
                return F.interpolate(c[None, None], size=(h, w), mode="bilinear")[0, 0]

            ycc = torch.stack([y / 255.0, (up(u) - 128.0) / 255.0, (up(v) - 128.0) / 255.0])
            rgb = torch.einsum("ij,jhw->ihw", self.rgb_from, ycc).clamp(0.0, 1.0)
            big = reduce(self.ai(rgb).clamp(0.0, 1.0), self.size)
            out = torch.einsum("ij,jhw->ihw", self.ycc709, big)
            return out[0].clamp(0, 1), ops.area_half(out[1]), ops.area_half(out[2])
        return self.to_709(*self.upscale(y, u, v))

    def flows_at(self, size: tuple[int, int], reach: int = 2) -> dict[int, Tensor]:
        """The last restored frame's flow to its neighbours within ``reach``, resampled to
        ``size`` (w, h) and in its pixels: what the remaster stage follows motion with."""
        if self.last_flow is None:
            return {}
        flow, offsets = self.last_flow
        w, h = size
        scale = torch.tensor([w / self.w, h / self.h], device=flow.device).view(2, 1, 1)
        out: dict[int, Tensor] = {}
        for i, d in enumerate(offsets):
            if abs(d) > reach or d in out:
                continue
            f = F.interpolate(flow[i : i + 1], size=(h, w), mode="bilinear", align_corners=False)
            out[d] = f[0] * scale
        return out

    def clean(self, centre: Prepared, neighbours: list[tuple[int, Prepared]]) -> Tensor:
        y, u, v = self.restore(centre, neighbours)
        y = ops.apply_curve(y, self.curve)
        u, v = self.vibrance(u, v)
        if self.learned is not None and self.learned.scale != 1:
            return self.to_output(*self.resize_learned(y, u, v))
        if self.ai is not None:
            return self.upscale_ai(self.ai, y, u, v)
        return self.to_output(*self.upscale(y, u, v))


class _Graph:
    """``fn`` captured in a CUDA graph for inputs shaped like ``example``; ``state`` (tensors
    ``fn`` updates in place) is restored after the warm-up runs, so they leave no trace."""

    def __init__(
        self,
        fn: Callable[..., tuple[Tensor, ...]],
        example: tuple[Tensor, ...],
        state: list[Tensor],
    ) -> None:
        self.inputs = [t.clone() for t in example]
        saved = [t.clone() for t in state]
        dev = example[0].device
        side = torch.cuda.Stream(dev)  # type: ignore[no-untyped-call, unused-ignore]
        side.wait_stream(torch.cuda.current_stream(dev))
        with torch.cuda.stream(side):
            for _ in range(2):
                fn(*self.inputs)
        torch.cuda.current_stream(dev).wait_stream(side)
        for t, v in zip(state, saved, strict=True):
            t.copy_(v)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.out = fn(*self.inputs)

    def __call__(self, *args: Tensor) -> tuple[Tensor, ...]:
        for dst, src in zip(self.inputs, args, strict=True):
            dst.copy_(src)
        self.graph.replay()
        return self.out


def _noise(p: Prepared) -> Tensor:
    if p.noise is None:
        raise AnaHDError("frame prepared without its chroma noise map")
    return p.noise


def _norm_grid(pos: Tensor, w: int, h: int) -> Tensor:
    """(S, 2, H, W) pixel positions -> grid_sample's (S, H, W, 2), align_corners=True."""
    gx = pos[:, 0] * (2.0 / (w - 1)) - 1.0
    gy = pos[:, 1] * (2.0 / (h - 1)) - 1.0
    return torch.stack([gx, gy], -1)


_END = object()


def background(it: Iterator[np.ndarray], depth: int = 16) -> Iterator[np.ndarray]:
    """Run ``it`` in a thread, ``depth`` items ahead; errors surface in the consumer."""
    q: queue.Queue[object] = queue.Queue(depth)
    stop = threading.Event()

    def work() -> None:
        try:
            for item in it:
                while not stop.is_set():
                    try:
                        q.put(item, timeout=0.1)
                        break
                    except queue.Full:
                        continue
                if stop.is_set():
                    return
            q.put(_END)
        except BaseException as exc:  # handed to the consumer, re-raised there
            q.put(exc)

    t = threading.Thread(target=work, daemon=True)
    t.start()
    try:
        while True:
            item = q.get()
            if item is _END:
                return
            if isinstance(item, BaseException):
                raise item
            yield item  # type: ignore[misc]
    finally:
        stop.set()


Progress = Callable[[int], None]
Stage = Callable[[str], None]  # what the clean is doing, before the first frame
# A live picture: the recording's frame and the cleaned one (JPEG), and the output frame's
# number in this call (from 0).
Picture = Callable[[bytes, bytes, int], None]
PINNED_SLOTS = 4  # frames in flight to the GPU on the sync-free upload
RAW_KEEP = 32  # recent raw frames kept to pair with the output (it lags by the window)


def enhance(
    src: Path,
    out: Path,
    opts: Options,
    *,
    gcfg: GpuConfig | None = None,
    overwrite: bool = False,
    nvenc: bool = False,
    progress: Progress | None = None,
    measured: Measured | None = None,
    stage_report: Stage | None = None,
    picture: Picture | None = None,
) -> Result:
    """``measured``: the whole recording's measurement, when cleaning it part by part.
    ``stage_report`` hears what is being done before the first frame; ``picture`` gets a
    before/after pair of the output now and then (see :mod:`anahd.snapshot`)."""
    say = stage_report or (lambda _what: None)
    if opts.steady is not None:
        raise AnaHDError("--steady runs on the CPU only; add --cpu")
    device = cuda_device()
    gcfg = gcfg or GpuConfig()
    info = probe(src)
    if measured is None:
        say("Measuring the picture area and OSD")
    meas = measured or measure(src, info, opts)
    say("Loading the models")
    params = opts.remaster
    cleaner = Cleaner(
        meas.crop,
        meas.osd,
        meas.curve,
        meas.size,
        info,
        opts.clean,
        opts.look,
        gcfg,
        device,
        params,
    )
    stage: Remaster | None = None
    if params.look_on:
        warp = steady_warp(src, info, meas, opts, device) if params.stabilisation > 0 else None
        osd = torch.from_numpy(meas.osd).to(device)
        redraw = None
        if params.osd_redraw:
            from anahd.remaster.osdlayout import analyze_clip

            num, _, den = info.fps.partition("/")
            fps = int(num) / int(den or 1)
            say("Reading the OSD")
            try:
                layout = analyze_clip(
                    src,
                    params.osd_font_path or None,
                    cli=params.osd_cli_path or None,
                    measured=meas,
                )
            except (OSError, ValueError) as exc:
                raise AnaHDError(f"OSD redraw: {exc}") from exc
            if layout is not None:  # no OSD found: the recording's own OSD stays
                redraw = layout.redraw(
                    meas.size, osd, fps, opts.start or 0.0, draw_live=params.osd_redraw_live
                )
                # Held numbers read a segment at a time over the whole stretch (osdlive);
                # the same pass verifies the timers against every readable tick.
                redraw.timeline = layout.read_live(src, opts.start, opts.duration)
                if redraw.timeline is not None:
                    redraw.timer_ok = redraw.timeline.timer_ok
        stage = Remaster(meas.size, osd, params, device, steady=warp, redraw=redraw)
    w, h = stage.out_size if stage is not None else meas.size
    frame_bytes = w * h + 2 * ((w // 2) * (h // 2))
    encoder = Encoder(out, w, h, info, overwrite=overwrite, ready=True, nvenc=nvenc)
    # Live pictures: output frame n is the recording's frame n (the remaster stage only
    # delays frames), so the main loop keeps the last few raw frames and, when a picture is
    # due, hands the writer the matching one with the output; the writer thread makes the
    # pair once the output's bytes have landed.
    recent: deque[tuple[int, np.ndarray]] = deque(maxlen=RAW_KEEP)
    throttle = snap.Throttle()
    raw_w, raw_h = meas.crop.width, meas.crop.height

    def make_picture(frame: np.ndarray, extra: object) -> None:
        if picture is None or not isinstance(extra, tuple):
            return
        n, raw = extra
        before = snap.to_bgr(raw, raw_w, raw_h, full_range=info.full_range, bt709=info.bt709)
        after = snap.to_bgr(frame, w, h, full_range=False, bt709=True)  # the encoder's format
        # In the output's shape, so the line splits one picture (SuperView, stabilisation
        # zoom and non-square SD pixels change the shape; the Results view does the same).
        before = cv2.resize(before, (after.shape[1], after.shape[0]), interpolation=cv2.INTER_AREA)
        picture(snap.jpeg(before), snap.jpeg(after), n)

    writer = _Writer(encoder, frame_bytes, device, make_picture if picture else None)
    half = (meas.size[0] // 2, meas.size[1] // 2)
    made = 0  # output frames handed to the writer

    def put(packed: Tensor) -> None:
        nonlocal made
        extra = None
        if picture is not None and throttle.due(time.monotonic()):
            raw = next((r for n, r in recent if n == made), None)
            extra = None if raw is None else (made, raw)
        writer.put(packed, extra)
        made += 1

    def remember(frames: Iterator[np.ndarray]) -> Iterator[np.ndarray]:
        for n, raw in enumerate(frames):
            if picture is not None:
                recent.append((n, raw))
            yield raw

    try:
        with torch.inference_mode():
            raws = background(
                read_raw(src, info, meas.crop, start=opts.start, duration=opts.duration)
            )
            prepared = (cleaner.prepare(raw) for raw in remember(raws))
            say("Cleaning")
            for done, (centre, nbs) in enumerate(_windows(prepared, opts.clean.radius), 1):
                if stage is None:
                    put(cleaner.clean(centre, nbs))
                else:
                    y, u, v = cleaner.clean_planes(centre, nbs)
                    for packed in stage.push(RFrame(y, u, v, cleaner.flows_at(half))):
                        put(packed)
                if progress is not None:
                    progress(done)
            if stage is not None:
                for packed in stage.flush():
                    put(packed)
        writer.close()
        encoder.close()
    except BaseException:
        writer.abort()
        encoder.abort()
        raise
    notes = ""
    if gcfg.repair:
        st = cleaner.stats
        notes = (
            f"repaired {st.rows} rows in {st.frames} frames ({st.whole} rebuilt whole, "
            f"{st.left} left alone), colour restored in {st.colour}, "
            f"snow pixels repaired in {st.pixels}"
        )
    zoom = stage.steady.zoom if stage is not None and stage.steady is not None else 1.0
    steadied = stage.steady.strength if stage is not None and stage.steady is not None else 0.0
    return Result(
        encoder.frames, meas.crop, (w, h), float(meas.osd.mean()), zoom, steadied, notes=notes
    )


def steady_warp(
    src: Path, info: VideoInfo, meas: Measured, opts: Options, device: torch.device
) -> SteadyWarp:
    """The A2 steadying (``anahd.steady``) measured over the recording, as per-frame warps
    at the output size. ``stabilisation`` (0..100) scales the corrections; tricks and fast
    moves get none at any setting (the plan fades them to zero)."""
    from anahd import steady as st
    from anahd.media import read_frames
    from anahd.pipeline import _full_range

    cfg = st.SteadyConfig()
    crop = meas.crop
    ignore = meas.osd | st.static_mask(meas.samples)
    frames = read_frames(src, info, crop, start=opts.start, duration=opts.duration)
    steps = st.measure((_full_range(f, info).y for f in frames), ignore, cfg)
    plan = st.plan(steps, crop.width, crop.height, cfg)
    return SteadyWarp.from_plan(
        plan, opts.remaster.stabilisation / 100, (crop.width, crop.height), meas.size, device
    )


class _Writer:
    """Copies finished frames to pinned host memory without waiting, and feeds the encoder
    from a thread once each copy has landed. A frame ``put`` with ``extra`` is also handed,
    with its landed bytes, to ``picture`` on that thread (the live view's pictures), before
    its buffer is reused."""

    SLOTS = 6

    def __init__(
        self,
        encoder: Encoder,
        frame_bytes: int,
        device: torch.device,
        picture: Callable[[np.ndarray, object], None] | None = None,
    ) -> None:
        self.encoder = encoder
        self.device = device
        self.picture = picture
        self._picture_failed = False
        self.bufs = [
            torch.empty(frame_bytes, dtype=torch.uint8, pin_memory=True) for _ in range(self.SLOTS)
        ]
        self.free: queue.Queue[int] = queue.Queue()
        for i in range(self.SLOTS):
            self.free.put(i)
        self.todo: queue.Queue[tuple[int, torch.cuda.Event, object] | None] = queue.Queue()
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def put(self, packed: Tensor, extra: object = None) -> None:
        if self.error is not None:
            raise self.error
        i = self.free.get()
        self.bufs[i].copy_(packed, non_blocking=True)
        event = torch.cuda.Event()  # type: ignore[no-untyped-call, unused-ignore]
        event.record()
        self.todo.put((i, event, extra))

    def _run(self) -> None:
        while True:
            item = self.todo.get()
            if item is None:
                return
            i, event, extra = item
            try:
                if self.error is None:
                    event.synchronize()
                    self.encoder.write_raw(self.bufs[i].numpy().data)
                    if extra is not None and self.picture is not None:
                        self._take_picture(self.bufs[i].numpy(), extra)
            except BaseException as exc:  # reported on the next put or at close
                self.error = exc
            finally:
                self.free.put(i)

    def _take_picture(self, frame: np.ndarray, extra: object) -> None:
        """A failed picture is skipped (and said once); the render goes on."""
        if self.picture is None:
            return
        try:
            self.picture(frame, extra)
        except (cv2.error, ValueError) as exc:
            if not self._picture_failed:
                self._picture_failed = True
                print(f"\nanahd: live picture skipped ({exc})", file=sys.stderr)

    def close(self) -> None:
        self.todo.put(None)
        self.thread.join()
        if self.error is not None:
            raise self.error

    def abort(self) -> None:
        self.error = self.error or MediaError("stopped")
        self.todo.put(None)
        self.thread.join(timeout=5)
