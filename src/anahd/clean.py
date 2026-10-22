"""Cleaning at the source resolution: motion-compensated fusion, destriping, rainbow removal.

Fusion aligns up to ``radius`` frames on each side of the current one with dense optical
flow and averages them, pixel by pixel, weighted by how well each neighbour matches.
Where a neighbour does not match (occlusion, a failed flow, a breakup frame) its weight
falls to zero, so moving edges stay sharp and nothing ghosts. OSD pixels never take part:
the current frame's OSD is kept as is, and pixels that came from a neighbour's OSD are
dropped, so the text cannot smear across the moving picture.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from anahd.media import Frame

F32 = NDArray[np.float32]
BoolMask = NDArray[np.bool_]


@dataclass(frozen=True)
class CleanConfig:
    radius: int = 3  # frames on each side (±3 = the 2026-10-12 blind-test winner)
    match_sigma: float = 7.0  # luma difference (8-bit levels) at which a neighbour's weight halves
    chroma_sigma: float = 6.0
    distance_falloff: float = 0.1  # weight lost per frame of distance
    flow_scale: float = 0.5  # flow is measured at this fraction of the source size
    # Destriping: horizontal lines are row offsets that persist along a row.
    stripe_clip: float = 4.0
    stripe_window: int = 241
    # Rainbow (cross-colour): fine chroma detail on top of fine luma detail.
    rainbow_cut: float = 0.8  # how much chroma detail is removed where luma is textured
    rainbow_texture: float = 6.0  # luma detail (levels) treated as fully textured


@dataclass
class Prepared:
    """A source frame with what fusion needs from it, computed once per frame."""

    y: F32
    u: F32
    v: F32
    flow_luma: NDArray[np.uint8]  # small, OSD inpainted, for optical flow


def prepare(
    frame: Frame, osd: BoolMask, osd_small: NDArray[np.uint8], cfg: CleanConfig
) -> Prepared:
    h, w = frame.y.shape
    small = cv2.resize(
        frame.y,
        (round(w * cfg.flow_scale), round(h * cfg.flow_scale)),
        interpolation=cv2.INTER_AREA,
    )
    if osd.any():
        small = cv2.inpaint(small, osd_small, 3, cv2.INPAINT_TELEA)
    return Prepared(
        y=frame.y.astype(np.float32),
        u=frame.u.astype(np.float32),
        v=frame.v.astype(np.float32),
        flow_luma=np.asarray(small, dtype=np.uint8),
    )


class Fuser:
    def __init__(self, osd: BoolMask, cfg: CleanConfig) -> None:
        self.cfg = cfg
        self.osd = osd
        h, w = osd.shape
        self.osd_f = osd.astype(np.float32)
        small = (round(w * cfg.flow_scale), round(h * cfg.flow_scale))
        self.osd_small = np.asarray(
            cv2.resize(osd.astype(np.uint8) * 255, small, interpolation=cv2.INTER_NEAREST),
            dtype=np.uint8,
        )
        self.keep = (~osd).astype(np.float32)  # where neighbours may contribute at all
        self.keep_half = np.asarray(
            cv2.resize(self.keep, (w // 2, h // 2), interpolation=cv2.INTER_AREA), dtype=np.float32
        )
        self.dis = cv2.DISOpticalFlow.create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        self.grid = (gx, gy)
        hx, hy = np.meshgrid(
            np.arange(w // 2, dtype=np.float32), np.arange(h // 2, dtype=np.float32)
        )
        self.grid_half = (hx, hy)

    def prepare(self, frame: Frame) -> Prepared:
        return prepare(frame, self.osd, self.osd_small, self.cfg)

    def fuse(
        self, center: Prepared, neighbours: Sequence[tuple[int, Prepared]]
    ) -> tuple[F32, F32, F32]:
        """Weighted average of ``center`` and its aligned ``(distance, frame)`` neighbours."""
        cfg = self.cfg
        h, w = center.y.shape
        sum_y, sum_u, sum_v = center.y.copy(), center.u.copy(), center.v.copy()
        wsum = np.ones_like(center.y)
        wsum_c = np.ones_like(center.u)
        ref_blur = cv2.GaussianBlur(center.y, (5, 5), 1.2)
        ref_u = cv2.GaussianBlur(center.u, (3, 3), 0.8)
        ref_v = cv2.GaussianBlur(center.v, (3, 3), 0.8)
        for distance, nb in neighbours:
            flow = self.dis.calc(center.flow_luma, nb.flow_luma, None)  # type: ignore[call-overload]
            flow = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR) / cfg.flow_scale
            mx, my = self.grid[0] + flow[..., 0], self.grid[1] + flow[..., 1]
            warped = cv2.remap(nb.y, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            # Pixels that land outside the frame or come from the neighbour's OSD are unusable.
            valid = cv2.remap(self.keep, mx, my, cv2.INTER_LINEAR, borderValue=0.0)
            diff = np.abs(cv2.GaussianBlur(warped, (5, 5), 1.2) - ref_blur)
            match = 1.0 / (1.0 + (diff / cfg.match_sigma) ** 4)
            weight = match * np.clip(valid, 0.0, 1.0) * self.keep
            weight *= max(0.0, 1.0 - cfg.distance_falloff * abs(distance))
            sum_y += weight * warped
            wsum += weight

            half = cv2.resize(flow, (w // 2, h // 2), interpolation=cv2.INTER_AREA) * 0.5
            hx, hy = self.grid_half[0] + half[..., 0], self.grid_half[1] + half[..., 1]
            wu = cv2.remap(nb.u, hx, hy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            wv = cv2.remap(nb.v, hx, hy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            cdiff = np.maximum(
                np.abs(cv2.GaussianBlur(wu, (3, 3), 0.8) - ref_u),
                np.abs(cv2.GaussianBlur(wv, (3, 3), 0.8) - ref_v),
            )
            cmatch = 1.0 / (1.0 + (cdiff / cfg.chroma_sigma) ** 4)
            cweight = cv2.resize(weight, (w // 2, h // 2), interpolation=cv2.INTER_AREA) * cmatch
            cweight *= self.keep_half
            sum_u += cweight * wu
            sum_v += cweight * wv
            wsum_c += cweight
        return sum_y / wsum, sum_u / wsum_c, sum_v / wsum_c


def destripe(y: F32, osd: BoolMask, cfg: CleanConfig) -> F32:
    """Remove horizontal line offsets: the part of each row's fine vertical detail that
    stays the same along a long stretch of the row. Real detail averages out over the
    window; large edges are clipped so they hardly move the estimate."""
    vertical = cv2.GaussianBlur(y, (1, 7), sigmaX=0.1, sigmaY=1.5)
    detail = np.clip(y - vertical, -cfg.stripe_clip, cfg.stripe_clip)
    detail[osd] = 0.0
    counted = (~osd).astype(np.float32)
    k = (cfg.stripe_window, 1)
    total = cv2.boxFilter(detail, -1, k, normalize=False, borderType=cv2.BORDER_REFLECT)
    n = cv2.boxFilter(counted, -1, k, normalize=False, borderType=cv2.BORDER_REFLECT)
    stripe = total / np.maximum(n, 1.0)
    out = y - stripe
    out[osd] = y[osd]
    return np.asarray(out, dtype=np.float32)


def derainbow(y: F32, u: F32, v: F32, osd_half: BoolMask, cfg: CleanConfig) -> tuple[F32, F32]:
    """Cut fine chroma detail where the luma is finely textured (analog cross-colour)."""
    h, w = u.shape
    yh = np.asarray(cv2.resize(y, (w, h), interpolation=cv2.INTER_AREA), dtype=np.float32)
    texture = cv2.GaussianBlur(np.abs(yh - cv2.GaussianBlur(yh, (0, 0), 1.0)), (0, 0), 2.0)
    cut = cfg.rainbow_cut * np.clip(texture / cfg.rainbow_texture, 0.0, 1.0)
    cut[osd_half] = 0.0
    out = []
    for c in (u, v):
        low = cv2.GaussianBlur(c, (0, 0), 1.5)
        out.append(np.asarray(low + (c - low) * (1.0 - cut), dtype=np.float32))
    return out[0], out[1]
