"""Light steadying from the video alone: take out small jitter, never damp a move.

Camera motion between consecutive frames (shift and roll) is measured from tracked
features in the scene, away from the OSD and anything fixed to the quad (props, frame
arms). The path is smoothed over a few frames, and each frame is nudged toward the
smooth path by at most a few pixels and a fraction of a degree. When the quad moves
fast (a flip, a roll, a hard turn) or the measurement is unsure, the nudge fades to
zero, so tricks and big moves pass through untouched. The picture is zoomed in just
enough per recording to hide the frame edges the nudges expose.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from anahd.look import Affine
from anahd.media import Plane

F64 = NDArray[np.float64]
BoolMask = NDArray[np.bool_]


@dataclass(frozen=True)
class SteadyConfig:
    scale: float = 0.5  # motion is measured at this fraction of the source size
    smooth_sigma: float = 4.0  # frames (~67 ms at 60 fps): only fast wobble is removed
    max_shift: float = 0.01  # of the picture width
    max_angle: float = 0.5  # degrees
    # Inter-frame motion above which steadying fades out (tricks, fast turns).
    # Smooth fast flight is fine to steady; only very fast moves and rolls/flips are not.
    calm_shift: float = 0.015  # of the width per frame: full correction below this
    busy_shift: float = 0.04  # no correction above this
    calm_angle: float = 0.5  # degrees per frame (30 deg/s)
    busy_angle: float = 1.5  # 90 deg/s: any roll or flip is well above this
    min_inliers: int = 30
    min_inlier_ratio: float = 0.3
    ransac_px: float = 3.0  # at the measuring size; analog noise makes tracks jittery
    max_zoom: float = 1.04


@dataclass(frozen=True)
class Step:
    """Camera motion from the previous frame to this one, in source pixels and radians."""

    dx: float
    dy: float
    da: float
    ok: bool


@dataclass(frozen=True)
class Plan:
    """Per-frame correction (source pixels, radians) and the recording's zoom."""

    cx: F64
    cy: F64
    ca: F64
    zoom: float
    strength: F64  # 0..1 per frame, for reports


class MotionMeter:
    """Measures :class:`Step` between consecutive frames."""

    def __init__(self, ignore: BoolMask, cfg: SteadyConfig) -> None:
        self.cfg = cfg
        h, w = ignore.shape
        self.size = (round(w * cfg.scale), round(h * cfg.scale))
        small = cv2.resize(ignore.astype(np.uint8), self.size, interpolation=cv2.INTER_NEAREST)
        grown = cv2.dilate(small, np.ones((9, 9), np.uint8))
        self.feature_mask = np.asarray((grown == 0).astype(np.uint8) * 255, dtype=np.uint8)
        self.center = np.array([self.size[0] / 2, self.size[1] / 2], dtype=np.float32)
        self.prev: NDArray[np.uint8] | None = None

    def step(self, y: Plane) -> Step:
        small = np.asarray(cv2.resize(y, self.size, interpolation=cv2.INTER_AREA), dtype=np.uint8)
        prev, self.prev = self.prev, small
        if prev is None:
            return Step(0.0, 0.0, 0.0, False)
        pts = cv2.goodFeaturesToTrack(
            prev, maxCorners=300, qualityLevel=0.01, minDistance=12, mask=self.feature_mask
        )
        if pts is None or len(pts) < self.cfg.min_inliers:
            return Step(0.0, 0.0, 0.0, False)
        nxt, status, _ = cv2.calcOpticalFlowPyrLK(  # type: ignore[call-overload]
            prev, small, pts, None, winSize=(21, 21)
        )
        good = status.ravel() == 1
        a, b = pts[good].reshape(-1, 2) - self.center, nxt[good].reshape(-1, 2) - self.center
        if len(a) < self.cfg.min_inliers:
            return Step(0.0, 0.0, 0.0, False)
        m, inliers = cv2.estimateAffinePartial2D(
            a, b, method=cv2.RANSAC, ransacReprojThreshold=self.cfg.ransac_px
        )
        if m is None or inliers is None:
            return Step(0.0, 0.0, 0.0, False)
        count = int(inliers.sum())
        ok = count >= self.cfg.min_inliers and count >= self.cfg.min_inlier_ratio * len(a)
        k = 1.0 / self.cfg.scale
        return Step(float(m[0, 2]) * k, float(m[1, 2]) * k, math.atan2(m[1, 0], m[0, 0]), ok)


def plan(steps: list[Step], width: int, height: int, cfg: SteadyConfig) -> Plan:
    """Corrections that pull each frame toward the smoothed path, faded out when busy."""
    dx = np.array([s.dx for s in steps])
    dy = np.array([s.dy for s in steps])
    da = np.array([s.da for s in steps])
    ok = np.array([s.ok for s in steps], dtype=np.float64)
    ok[0] = 1.0  # the first frame has no predecessor; it is the path's origin
    tx, ty, ta = np.cumsum(dx), np.cumsum(dy), np.cumsum(da)
    corr = [_smooth(t, cfg.smooth_sigma) - t for t in (tx, ty, ta)]

    shift = np.hypot(dx, dy) / width
    angle = np.degrees(np.abs(da))
    calm = np.minimum(
        _ramp(shift, cfg.calm_shift, cfg.busy_shift), _ramp(angle, cfg.calm_angle, cfg.busy_angle)
    )
    # A frame is only steadied if its whole smoothing window is calm and measured.
    reach = max(1, round(2 * cfg.smooth_sigma))
    trusted = _window_min(calm * ok, reach)
    # Ease in and out of steadied stretches, but never into an untrusted frame.
    strength = np.minimum(np.clip(_smooth(trusted, cfg.smooth_sigma / 2), 0.0, 1.0), trusted)

    max_px = cfg.max_shift * width
    cx = np.clip(corr[0] * strength, -max_px, max_px)
    cy = np.clip(corr[1] * strength, -max_px, max_px)
    max_rad = math.radians(cfg.max_angle)
    ca = np.clip(corr[2] * strength, -max_rad, max_rad)
    return Plan(cx, cy, ca, _zoom_for(cx, cy, ca, width, height, cfg), strength)


def _smooth(x: F64, sigma: float) -> F64:
    if len(x) < 2 or sigma <= 0:
        return x.copy()
    radius = max(1, round(3 * sigma))
    t = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (t / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(x, radius, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _ramp(x: F64, lo: float, hi: float) -> F64:
    return np.clip((hi - x) / (hi - lo), 0.0, 1.0)


def _window_min(x: F64, reach: int) -> F64:
    padded = np.pad(x, reach, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * reach + 1)
    return np.asarray(windows.min(axis=1), dtype=np.float64)


def _zoom_for(cx: F64, cy: F64, ca: F64, width: int, height: int, cfg: SteadyConfig) -> float:
    """Smallest zoom (capped) that keeps the corrected picture's edges out of view."""
    if len(cx) == 0:
        return 1.0
    corner = math.hypot(width / 2, height / 2)
    need_x = np.abs(cx) + corner * np.abs(np.sin(ca))
    need_y = np.abs(cy) + corner * np.abs(np.sin(ca))
    zoom = 1.0 + 2.0 * max(float(np.max(need_x)) / width, float(np.max(need_y)) / height)
    return min(zoom, cfg.max_zoom)


def measure(frames: Iterable[Plane], ignore: BoolMask, cfg: SteadyConfig) -> list[Step]:
    meter = MotionMeter(ignore, cfg)
    return [meter.step(y) for y in frames]


def matrix(p: Plan, i: int, src: tuple[int, int], dst: tuple[int, int]) -> Affine:
    """Source pixel -> output pixel for frame ``i``: correction, zoom, then the scale-up."""
    w, h = src
    sx, sy = dst[0] / w, dst[1] / h
    a = float(p.ca[i])
    rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    lin = np.diag([sx, sy]) @ (p.zoom * rot)
    ctr = np.array([w / 2, h / 2])
    shift = np.array([p.cx[i], p.cy[i]])
    # Continuous coordinates (pixel centres at +0.5), so no correction == a plain resize.
    offset = np.diag([sx, sy]) @ (p.zoom * (rot @ (np.array([0.5, 0.5]) - ctr) + shift) + ctr)
    out = np.zeros((2, 3))
    out[:, :2] = lin
    out[:, 2] = offset - 0.5
    return out


def static_mask(samples: list[Plane], change: int = 10, stability: float = 0.8) -> BoolMask:
    """Pixels that hardly change across the recording: OSD, props and frame arms in view."""
    stack = np.stack(samples).astype(np.int16)
    steady = (np.abs(np.diff(stack, axis=0)) < change).mean(axis=0) >= stability
    return np.asarray(steady, dtype=bool)
