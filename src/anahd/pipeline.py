"""One recording in, one clean HD file out.

The recording is measured once (picture area, OSD, levels), then each frame is cleaned
in a sliding window of its neighbours.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar

import cv2
import numpy as np

from anahd import clean, look, steady
from anahd.clean import CleanConfig
from anahd.errors import MediaError
from anahd.look import LookConfig
from anahd.media import Crop, Encoder, Frame, VideoInfo, count_frames, probe, read_frames
from anahd.picture import detect_crop, detect_osd, output_size
from anahd.remaster.params import RemasterParams
from anahd.steady import SteadyConfig

SAMPLES = 80  # frames spread over the recording for picture area, OSD and levels
MAX_PICTURE_H = 720  # taller pictures are delivered at this height (aspect kept)

Progress = Callable[[int], None]
T = TypeVar("T")


@dataclass(frozen=True)
class Options:
    clean: CleanConfig = field(default_factory=CleanConfig)
    look: LookConfig = field(default_factory=LookConfig)
    steady: SteadyConfig | None = None  # None: steadying off
    start: float | None = None
    duration: float | None = None
    # The sliders (GPU pipeline; the CPU pipeline uses only the cleaning it has).
    remaster: RemasterParams = field(default_factory=RemasterParams)


@dataclass(frozen=True)
class Result:
    frames: int
    crop: Crop
    size: tuple[int, int]
    osd_fraction: float
    zoom: float = 1.0
    steadied: float = 0.0  # mean steadying strength over the frames
    notes: str = ""  # one line on what was repaired, when repair was on


def _full_range(frame: Frame, info: VideoInfo) -> Frame:
    """Expand a limited-range frame to full range so all later steps see one scale."""
    if info.full_range:
        return frame

    def expand(p: np.ndarray, lo: float, span: float, mid: float) -> np.ndarray:
        return np.asarray(np.clip((p.astype(np.float32) - lo) * span + mid, 0, 255), np.uint8)

    return Frame(
        y=expand(frame.y, 16.0, 255.0 / 219.0, 0.0),
        u=expand(frame.u, 128.0, 255.0 / 224.0, 128.0),
        v=expand(frame.v, 128.0, 255.0 / 224.0, 128.0),
    )


@dataclass(frozen=True)
class Measured:
    """What is measured once per recording, before any frame is cleaned."""

    crop: Crop
    osd: np.ndarray  # bool, at the crop's size
    curve: np.ndarray  # tone lookup
    size: tuple[int, int]  # output width, height
    samples: list[np.ndarray]  # cropped luma samples


def _samples(src: Path, info: VideoInfo, opts: Options) -> list[np.ndarray]:
    """About SAMPLES luma frames spread over the recording. Keyframes are used when there
    are enough of them (a long flight: no full decode needed), else frames at even times."""
    keys = [
        _full_range(f, info).y.copy()
        for f in read_frames(src, info, start=opts.start, duration=opts.duration, keyframes=True)
    ]
    if len(keys) >= SAMPLES:
        pick = np.linspace(0, len(keys) - 1, SAMPLES).round().astype(int)
        return [keys[i] for i in pick]
    span = opts.duration if opts.duration is not None else max(info.duration - (opts.start or 0), 1)
    every = max(span / SAMPLES, 1.0 / 60.0)
    return [
        _full_range(f, info).y.copy()
        for f in read_frames(src, info, start=opts.start, duration=opts.duration, every=every)
    ]


def measure(src: Path, info: VideoInfo, opts: Options) -> Measured:
    samples = _samples(src, info, opts)
    crop = detect_crop(samples)
    cropped = [s[crop.y : crop.y + crop.height, crop.x : crop.x + crop.width] for s in samples]
    # An analog picture holds at most ~480 lines: a taller file (an upscaled re-upload) is
    # cleaned at 720 lines, as fast as an HD goggle DVR, with nothing lost.
    small = crop.scaled_to(MAX_PICTURE_H)
    if small is not crop:
        size = (small.width, small.height)
        cropped = [cv2.resize(s, size, interpolation=cv2.INTER_AREA) for s in cropped]
        crop = small
    osd = detect_osd(cropped)
    levels = look.measure_levels(cropped, osd, opts.look)
    curve = look.tone_curve(levels, opts.look)
    size = output_size(crop, sar=info.sar, frame=(info.width, info.height))
    return Measured(crop, osd, curve, size, cropped)


def enhance(
    src: Path,
    out: Path,
    opts: Options,
    *,
    overwrite: bool = False,
    progress: Progress | None = None,
    measured: Measured | None = None,
    stage_report: Callable[[str], None] | None = None,
) -> Result:
    """``measured``: the recording's measurement, when cleaning parts of it one by one
    (cuts), so every part gets the same crop, OSD mask and tone curve. ``stage_report``
    hears what is being done before the first frame."""
    say = stage_report or (lambda _what: None)
    info = probe(src)
    if measured is None:
        say("Measuring the picture area and OSD")
    meas = measured or measure(src, info, opts)
    crop, osd, curve, size, cropped = meas.crop, meas.osd, meas.curve, meas.size, meas.samples
    osd_half = np.asarray(
        cv2.resize(
            osd.astype(np.uint8),
            (crop.width // 2, crop.height // 2),
            interpolation=cv2.INTER_NEAREST,
        ),
        dtype=bool,
    )

    plan: steady.Plan | None = None
    if opts.steady is not None:
        ignore = osd | steady.static_mask(cropped)
        steps = steady.measure(
            (
                _full_range(f, info).y
                for f in read_frames(src, info, crop, start=opts.start, duration=opts.duration)
            ),
            ignore,
            opts.steady,
        )
        plan = steady.plan(steps, crop.width, crop.height, opts.steady)
    src_size = (crop.width, crop.height)
    plain = look.scale_matrix(src_size, size)
    osd_u8 = osd.astype(np.uint8)
    osd_half_u8 = osd_half.astype(np.uint8)
    osd_alpha, alpha_half = None, None
    if plan is not None and osd.any():
        feather = cv2.GaussianBlur(osd.astype(np.float32), (0, 0), 2.0)
        feather = np.maximum(feather, osd.astype(np.float32))
        osd_alpha = np.asarray(
            cv2.warpAffine(feather, plain, size, flags=cv2.INTER_LINEAR), dtype=np.float32
        )
        alpha_half = np.asarray(
            cv2.resize(osd_alpha, (size[0] // 2, size[1] // 2), interpolation=cv2.INTER_AREA),
            dtype=np.float32,
        )

    fuser = clean.Fuser(osd, opts.clean)
    encoder = Encoder(out, size[0], size[1], info, overwrite=overwrite)
    try:
        frames = (
            fuser.prepare(_full_range(f, info))
            for f in read_frames(src, info, crop, start=opts.start, duration=opts.duration)
        )
        say("Cleaning")
        for done, (center, neighbours) in enumerate(_windows(frames, opts.clean.radius), 1):
            y, u, v = fuser.fuse(center, neighbours)
            y = clean.destripe(y, osd, opts.clean)
            u, v = clean.derainbow(y, u, v, osd_half, opts.clean)
            y = look.apply_tone(y, curve)
            u, v = look.apply_vibrance(u, v, opts.look)
            if plan is None:
                big = look.upscale_luma(y, size, opts.look)
                bu = look.upscale_chroma(u, big, opts.look)
                bv = look.upscale_chroma(v, big, opts.look)
            else:
                m = steady.matrix(plan, done - 1, src_size, size)
                # The picture moves, the OSD must not: take the OSD out of the picture
                # before the warp, then paste it back from the unmoved frame.
                big = look.upscale_luma(_inpaint(y, osd_u8), size, opts.look, m)
                bu = look.upscale_chroma(_inpaint(u, osd_half_u8), big, opts.look, m)
                bv = look.upscale_chroma(_inpaint(v, osd_half_u8), big, opts.look, m)
                if osd_alpha is not None and alpha_half is not None:
                    big0 = look.upscale_luma(y, size, opts.look, plain)
                    big = _blend(big0, big, osd_alpha)
                    bu = _blend(look.upscale_chroma(u, big0, opts.look), bu, alpha_half)
                    bv = _blend(look.upscale_chroma(v, big0, opts.look), bv, alpha_half)
            encoder.write(Frame(y=look.to_plane(big), u=look.to_plane(bu), v=look.to_plane(bv)))
            if progress is not None:
                progress(done)
        encoder.close()
    except BaseException:
        encoder.abort()
        raise
    zoom = plan.zoom if plan is not None else 1.0
    steadied = float(plan.strength.mean()) if plan is not None and len(plan.strength) else 0.0
    return Result(encoder.frames, crop, size, float(osd.mean()), zoom, steadied)


def _blend(top: np.ndarray, under: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    return np.asarray(top * alpha + under * (1.0 - alpha), dtype=np.float32)


def _inpaint(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return x
    filled = cv2.inpaint(look.to_plane(x), mask, 3, cv2.INPAINT_TELEA)
    return np.asarray(filled, dtype=np.float32)


def _windows(frames: Iterator[T], radius: int) -> Iterator[tuple[T, list[tuple[int, T]]]]:
    """Each frame with up to ``radius`` neighbours on each side (fewer at the ends)."""
    window: deque[T] = deque()
    center = 0  # index of the current frame inside ``window``
    exhausted = False
    while True:
        while not exhausted and len(window) - center <= radius:
            nxt = next(frames, None)
            if nxt is None:
                exhausted = True
            else:
                window.append(nxt)
        if center >= len(window):
            return
        neighbours = [
            (i - center, window[i])
            for i in range(max(0, center - radius), min(len(window), center + radius + 1))
            if i != center
        ]
        yield window[center], neighbours
        if center >= radius:
            window.popleft()
        else:
            center += 1


def verify(src_frames: int, out: Path) -> int:
    got = count_frames(out)
    if got != src_frames:
        raise MediaError(f"{out}: {got} frames written, {src_frames} expected")
    return got
