"""One frame through the whole remaster, before cleaning a clip: what the sliders will give.

``preview_frame(clip, t, params)`` decodes the frames around ``t``, runs the clean-up on
the five frames the remaster stage needs (each with its own neighbours), then the remaster
stage, and returns the result next to the plain recording in the same geometry.

Cache (pass the same :class:`PreviewCache`): per clip, the measurement (picture area, OSD,
tone curve) and the GPU cleaner (RAFT, the restorer and their CUDA graphs); per moment and
clean-up settings, the cleaned frames and their flows. So moving a Colour, Look or Effects
slider reruns only the remaster stage; a Clean slider reruns the clean-up of 5 frames;
a new moment decodes 11 frames and cleans 5. Stabilisation measures the camera path over
one second around the frame.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray

from anahd.media import VideoInfo, probe, read_frames, read_raw
from anahd.picture import osd_alpha
from anahd.pipeline import Measured, Options, _full_range, _windows, measure

from .grade import to_rgb
from .osd import Redraw
from .params import RemasterParams
from .stage import RADIUS, Frame, Remaster, SteadyWarp, superview_grid

CLEAN_RADIUS = 3  # the restorer's neighbours (CleanConfig.radius)
STEADY_REACH = 0.5  # seconds each side measured for the preview's stabilisation

RGB = NDArray[np.uint8]


@dataclass
class _Clip:
    info: VideoInfo
    meas: Measured
    cleaner: object  # anahd.gpu.pipeline.Cleaner (imported lazily: torch)
    fps: float
    detector: object  # the cleaner's breakup detector, switched off when static_repair = 0


@dataclass
class _Moment:
    frames: list[Frame]  # cleaned, at the output size
    centre: int  # which one is the frame asked for
    source: torch.Tensor  # the recording's frame, plain upscale, R'G'B' 0..1
    first: int  # frame number of frames[0] in the recording


@dataclass
class PreviewCache:
    """Pass the same one to every call for a clip; see the module notes for what is kept."""

    clips: dict[tuple[Path, bool], _Clip] = field(default_factory=dict)
    moments: dict[tuple[object, ...], _Moment] = field(default_factory=dict)
    keep: int = 4  # moments kept (the latest ones)
    redraws: dict[tuple[int, str, str, bool], Redraw | None] = field(
        default_factory=dict
    )  # per font


@dataclass(frozen=True)
class Preview:
    source: RGB  # the recording at this moment: picture area, plain upscale (+ SuperView)
    result: RGB  # the remastered frame
    seconds: float  # the frame's time in the recording
    timings: dict[str, float]  # ms: setup, decode+clean, steady, remaster


def _clean_key(p: RemasterParams) -> tuple[float, ...]:
    return (p.cleanup_strength, p.scan_lines, p.noise, p.static_repair)


def _clip(cache: PreviewCache, path: Path, learned: bool) -> _Clip:
    key = (path.resolve(), learned)
    if key not in cache.clips:
        from anahd.gpu.device import cuda_device
        from anahd.gpu.pipeline import Cleaner, GpuConfig
        from anahd.model import weights

        info = probe(path)
        meas = measure(path, info, Options())
        num, _, den = info.fps.partition("/")
        fps = int(num) / int(den or 1)
        gcfg = GpuConfig(repair=True, learned=weights.resolve(None) if learned else None)
        cleaner = Cleaner(
            meas.crop, meas.osd, meas.curve, meas.size, info,
            Options().clean, Options().look, gcfg, cuda_device(), regions=meas.osd_regions,
        )  # fmt: skip
        cache.clips[key] = _Clip(info, meas, cleaner, fps, cleaner.detector)
    return cache.clips[key]


def _moment(c: _Clip, path: Path, n: int, p: RemasterParams) -> _Moment:
    """Frames n-2..n+2 cleaned (decoding n-5..n+5 for their neighbours)."""
    cleaner = c.cleaner
    cleaner.params = p  # type: ignore[attr-defined]
    cleaner.cleanup = p.cleanup_strength / 100  # type: ignore[attr-defined]
    cleaner.detector = c.detector if p.static_repair > 0 else None  # type: ignore[attr-defined]
    reach = RADIUS + CLEAN_RADIUS
    first = max(0, n - reach)
    start = None if first == 0 else (first - 0.5) / c.fps
    duration = (n + reach - first + 1) / c.fps
    raws = list(read_raw(path, c.info, c.meas.crop, start=start, duration=duration))
    if not raws:
        raise ValueError(f"{path}: no frame at {n / c.fps:.2f} s")
    n = min(n, first + len(raws) - 1)
    prepared = [cleaner.prepare(r) for r in raws]  # type: ignore[attr-defined]
    w, h = c.meas.size
    want = range(max(first, n - RADIUS), min(first + len(raws), n + RADIUS + 1))
    frames = []
    for i, (centre, nbs) in enumerate(_windows(iter(prepared), CLEAN_RADIUS)):
        if first + i not in want:
            continue
        y, u, v = cleaner.clean_planes(centre, nbs)  # type: ignore[attr-defined]
        flows = cleaner.flows_at((w // 2, h // 2))  # type: ignore[attr-defined]
        frames.append(Frame(y, u, v, flows, osd=cleaner.shown))  # type: ignore[attr-defined]
    y0, u0, v0 = cleaner.upload(raws[n - first])  # type: ignore[attr-defined]
    big = cleaner.to_709(*_plain(y0, u0, v0, (w, h)))  # type: ignore[attr-defined]
    uf = F.interpolate(torch.stack(big[1:])[None], size=(h, w), mode="bilinear")[0]
    source = to_rgb(big[0], uf[0], uf[1]).clamp(0, 1)
    return _Moment(frames, n - want.start, source, want.start)


def _plain(
    y: torch.Tensor, u: torch.Tensor, v: torch.Tensor, size: tuple[int, int]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    w, h = size

    def rs(p: torch.Tensor, hh: int, ww: int) -> torch.Tensor:
        return F.interpolate(p[None, None], size=(hh, ww), mode="bicubic", antialias=True)[0, 0]

    return rs(y, h, w), rs(u, h // 2, w // 2), rs(v, h // 2, w // 2)


def _steady(c: _Clip, path: Path, m: _Moment, amount: float) -> SteadyWarp:
    """The A2 steadying measured over a second around the frame, as warps for the five
    frames of the moment."""
    from anahd import steady as st

    cfg = st.SteadyConfig()
    reach = round(STEADY_REACH * c.fps)
    begin = max(0, m.first - reach)
    start = None if begin == 0 else (begin - 0.5) / c.fps
    count = len(m.frames) + 2 * reach
    crop = c.meas.crop
    frames = read_frames(path, c.info, crop, start=start, duration=count / c.fps)
    ignore = c.meas.osd | st.static_mask(c.meas.samples)
    steps = st.measure((_full_range(f, c.info).y for f in frames), ignore, cfg)
    plan = st.plan(steps, crop.width, crop.height, cfg)
    off = m.first - begin
    sl = slice(off, off + len(m.frames))
    sub = st.Plan(plan.cx[sl], plan.cy[sl], plan.ca[sl], plan.zoom, plan.strength[sl])
    return SteadyWarp.from_plan(
        sub, amount, (crop.width, crop.height), c.meas.size, torch.device("cuda")
    )


def preview_frame(
    clip: Path,
    t: float,
    params: RemasterParams | None = None,
    cache: PreviewCache | None = None,
    *,
    learned: bool = True,
) -> Preview:
    """The frame at ``t`` seconds of ``clip`` remastered with ``params``, and the plain
    recording in the same geometry (picture area, output size, SuperView)."""
    p = params or RemasterParams()
    cache = cache or PreviewCache()
    ms: dict[str, float] = {}
    t0 = time.perf_counter()
    with torch.inference_mode():
        c = _clip(cache, clip, learned)
        t1 = time.perf_counter()
        ms["setup"] = 1000 * (t1 - t0)
        n = max(0, round(t * c.fps))
        key = (clip.resolve(), learned, n, _clean_key(p))
        if key not in cache.moments:
            cache.moments[key] = _moment(c, clip, n, p)
            while len(cache.moments) > cache.keep:
                cache.moments.pop(next(iter(cache.moments)))
        m = cache.moments[key]
        t2 = time.perf_counter()
        ms["decode_clean"] = 1000 * (t2 - t1)
        warp = _steady(c, clip, m, p.stabilisation / 100) if p.stabilisation > 0 else None
        t3 = time.perf_counter()
        ms["steady"] = 1000 * (t3 - t2)
        osd = torch.from_numpy(osd_alpha(c.meas.osd)).to("cuda")
        redraw = _redraw(cache, c, p, clip, m.first) if p.osd_redraw else None
        stage = Remaster(c.meas.size, osd, p, torch.device("cuda"), steady=warp, redraw=redraw)
        stage.capture = m.centre
        for f in m.frames:
            stage.push(Frame(f.y, f.u, f.v, dict(f.flows)))
        stage.flush()
        if stage.captured is None:
            raise RuntimeError("the remaster stage did not reach the previewed frame")
        source = m.source
        grid = superview_grid(c.meas.size[0], c.meas.size[1], p.superview, source.device)
        if grid is not None:
            source = F.grid_sample(source[None], grid, mode="bicubic", align_corners=True)[0]
        result, before = _to_u8(stage.captured), _to_u8(source)
        torch.cuda.synchronize()
        ms["remaster"] = 1000 * (time.perf_counter() - t3)
    return Preview(before, result, (m.first + m.centre) / c.fps, ms)


def _redraw(
    cache: PreviewCache, c: _Clip, p: RemasterParams, clip: Path, first: int
) -> Redraw | None:
    """The clip's OSD redraw from its analysed layout (``analyze_clip``, cached per clip
    and font), set to the previewed frames' time."""
    from .osdlayout import analyze_clip

    key = (id(c), p.osd_font_path, p.osd_cli_path, p.osd_redraw_live)
    if key not in cache.redraws:
        layout = analyze_clip(
            clip, p.osd_font_path or None, cli=p.osd_cli_path or None, measured=c.meas
        )
        osd = torch.from_numpy(osd_alpha(c.meas.osd)).to("cuda")
        cache.redraws[key] = (
            None
            if layout is None
            else layout.redraw(c.meas.size, osd, c.fps, draw_live=p.osd_redraw_live)
        )
    r = cache.redraws[key]
    if r is not None:  # each preview reads its five frames afresh, at their own time
        r.reset()
        r.seek(first)
    return r


def _to_u8(rgb: torch.Tensor) -> RGB:
    return np.ascontiguousarray(
        (rgb.clamp(0, 1) * 255).round().byte().permute(1, 2, 0).cpu().numpy()
    )
