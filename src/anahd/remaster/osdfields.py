"""OSD fields and how each behaves over a recording (for the experimental OSD redraw).

A field is a run of changing cells on one row of the OSD grid (fixed separators such as
':' or '.' may sit inside it). One pass over the recording reads every field twice a
second, with the time of each reading, and decides what kind of field it is:

- **timer**: the digits read as MM:SS and count up one second per second, with pauses
  allowed (a fly timer stops while disarmed). Its text is then generated for every frame
  from the clock, so it is crisp in every frame and ticks on the recording's own frame.
  The tick phase is measured on full-rate frames around one change.
- **slow**: changes rarely (VTX band/channel/power, a mode): shown as read, but a new
  value must hold for half a second.
- **live**: everything else (voltage, current, throttle, RSSI/LQ): read every frame, a new
  value shown once it holds for 3 frames, passed through while it changes.
"""

from __future__ import annotations

import itertools
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from anahd.media import Crop, VideoInfo, read_frames

EVERY = 0.5  # seconds between the readings of the first pass
SLOW_RATE = 0.2  # changes per second below which a field is slow
SLOW_HOLD = 0.5  # seconds a slow field's new value must hold
TIMER_FIT = 0.9  # share of consecutive readings a timer must agree with


@dataclass
class Segment:
    """A stretch of a timer: from ``t`` (s), shown value = floor(time + offset) while
    running, or ``value`` while paused."""

    t: float
    running: bool
    offset: float = 0.0
    value: int = 0


@dataclass
class Field:
    row: int
    cols: list[int]  # the value's cells, left to right
    kind: str = "live"  # timer | slow | live
    seps: list[int] = field(default_factory=list)  # separators (':' '.'): left as recorded
    readings: list[tuple[float, str]] = field(default_factory=list)
    segments: list[Segment] = field(default_factory=list)
    changes: int = 0

    def label(self) -> str:
        cols = sorted(self.cols + self.seps)
        sep = f", separators at {self.seps}" if self.seps else ""
        return f"row {self.row} cols {cols[0]}-{cols[-1]}{sep}"


def find_fields(fixed: Tensor, inside: Tensor, rows: int, cols: int) -> list[Field]:
    """Runs of changing OSD cells on each row; a single fixed cell between two runs (a
    ':' or '.') joins them into one field."""
    f = fixed.view(rows, cols).cpu().numpy()
    ins = inside.view(rows, cols).cpu().numpy()
    out: list[Field] = []
    for r in range(rows):
        changing = [c for c in range(cols) if ins[r, c] and f[r, c] < 0]
        run: list[int] = []
        for c in changing:
            if run and (c - run[-1] == 1 or (c - run[-1] == 2 and f[r, c - 1] >= 0)):
                run.append(c)
            else:
                if run:
                    out.append(Field(r, run))
                run = [c]
        if run:
            out.append(Field(r, run))
    return out


def _text(idx: Tensor) -> str:
    return "".join(chr(int(i)) if 32 <= int(i) < 127 else f"<{int(i):02x}>" for i in idx)


def _seconds(text: str) -> int | None:
    """MMSS or MSS digits -> seconds; SS alone (the minutes being fixed cells in this
    recording) -> seconds within the minute."""
    if not re.fullmatch(r"\d{2,4}", text):
        return None
    m, s = (int(text[:-2]) if len(text) > 2 else 0), int(text[-2:])
    return m * 60 + s if s < 60 else None


def survey(
    src: Path,
    info: VideoInfo,
    crop: Crop,
    fields: list[Field],
    read: object,
    fps: float,
    start: float | None,
    duration: float | None,
) -> None:
    """Read every field twice a second over the recording and classify it (in place).
    ``read(y) -> (idx, ncc)`` reads all cells of a luma frame (H, W, 0..1)."""
    from anahd.pipeline import _full_range

    from .osd import READ

    t0 = start or 0.0
    dev = torch.device("cuda")
    taken: list[tuple[float, Tensor, Tensor]] = []
    for k, frame in enumerate(
        read_frames(src, info, crop, start=start, duration=duration, every=EVERY)
    ):
        y = torch.from_numpy(_full_range(frame, info).y.copy()).to(dev).float() / 255
        idx, ncc = read(y)  # type: ignore[operator]
        taken.append((t0 + k * EVERY, idx, ncc >= READ))
    classify_from(fields, taken, src, info, crop, read, fps)


def classify_from(
    fields: list[Field],
    taken: list[tuple[float, Tensor, Tensor]],
    src: Path,
    info: VideoInfo,
    crop: Crop,
    read: object,
    fps: float,
) -> None:
    """Classify the fields (in place) from timed readings already taken: (time, glyph per
    cell, read with confidence per cell)."""
    if not taken:
        return
    dev = taken[0][1].device
    sure = torch.stack([s for _, _, s in taken]).float().mean(0)  # per cell
    for fl in fields:
        # Separators: cells of the field that hardly ever read while the rest do (':'
        # and '.' are too small to read in single frames). Left as recorded.
        share = [float(sure[fl.row * 30 + c]) for c in fl.cols]
        if max(share) >= 0.5:
            fl.seps = [c for c, v in zip(fl.cols, share, strict=True) if v < 0.3]
            fl.cols = [c for c in fl.cols if c not in fl.seps]
        if not fl.cols:
            continue
        cells = torch.tensor([fl.row * 30 + c for c in fl.cols], device=dev)
        for t, idx, ok in taken:
            if bool(ok[cells].all()):
                fl.readings.append((t, _text(idx[cells])))
    for fl in fields:
        if fl.cols:
            _classify(fl, src, info, crop, read, fps)


def _classify(fl: Field, src: Path, info: VideoInfo, crop: Crop, read: object, fps: float) -> None:
    r = fl.readings
    if len(r) < 6:
        fl.kind = "live"
        return
    fl.changes = sum(1 for a, b in itertools.pairwise(r) if a[1] != b[1])
    span = r[-1][0] - r[0][0]
    # A stray misread (a torn frame) does not stop a timer being seen: up to one reading
    # in ten may fail to parse; it is left out.
    parsed = [(t, _seconds(s)) for t, s in r]
    secs = [(t, s) for t, s in parsed if s is not None]
    if len(secs) >= TIMER_FIT * len(parsed) and len(secs) >= 6:
        pairs = list(itertools.pairwise(secs))
        ok = 0
        for (ta, sa), (tb, sb) in pairs:
            if sa is None or sb is None:
                continue
            d = sb - sa
            if d == 0 or abs(d - (tb - ta)) <= 1.0:
                ok += 1
        moving = sum(1 for (_, sa), (_, sb) in pairs if sb != sa)
        if pairs and ok >= TIMER_FIT * len(pairs) and moving >= 0.5 * len(pairs):
            fl.kind = "timer"
            fl.segments = _timer_segments([(t, s) for t, s in secs if s is not None])
            _tick_phase(fl, src, info, crop, read, fps)
            return
    fl.kind = "slow" if span > 0 and fl.changes / span < SLOW_RATE else "live"


def _timer_segments(secs: list[tuple[float, int]]) -> list[Segment]:
    """Running and paused stretches. While running, the offset interval each reading
    allows (s <= t + o < s + 1) is intersected; its middle is the offset."""
    segs: list[Segment] = []
    lo, hi = -math.inf, math.inf
    for i, (t, s) in enumerate(secs):
        running = i + 1 < len(secs) and secs[i + 1][1] != s
        paused = i > 0 and secs[i - 1][1] == s and not running
        if paused:
            if not segs or segs[-1].running:
                segs.append(Segment(t, False, value=s))
            lo, hi = -math.inf, math.inf
            continue
        if not segs or not segs[-1].running:
            segs.append(Segment(t, True))
            lo, hi = -math.inf, math.inf
        lo, hi = max(lo, s - t), min(hi, s + 1 - t)
        segs[-1].offset = (lo + hi) / 2 if hi > lo else s - t + 0.5
    # The last reading or two can repeat a frame at the end of the recording: not a pause.
    if len(segs) > 1 and not segs[-1].running and segs[-1].t >= secs[-2][0]:
        segs.pop()
    return segs


def _tick_phase(
    fl: Field, src: Path, info: VideoInfo, crop: Crop, read: object, fps: float
) -> None:
    """Pin each running stretch's offset to the frames where the reading actually
    changes: full-rate frames around up to six ticks spread over the stretch, the median
    of what they say (one misread change cannot move it)."""
    from anahd.pipeline import _full_range

    from .osd import READ

    dev = torch.device("cuda")
    cells = torch.tensor([fl.row * 30 + c for c in fl.cols], device=dev)
    ends = [s.t for s in fl.segments[1:]] + [fl.readings[-1][0]]
    for seg, end in zip(fl.segments, ends, strict=True):
        if not seg.running or end - seg.t < 1.0:
            continue
        found = []
        for at in np.linspace(seg.t + 0.5, end - 0.5, 6):
            tick = math.ceil(at + seg.offset) - seg.offset  # the next expected change
            begin = max(tick - 0.6, 0.0)
            last: int | None = None
            for k, frame in enumerate(read_frames(src, info, crop, start=begin, duration=1.2)):
                y = torch.from_numpy(_full_range(frame, info).y.copy()).to(dev).float() / 255
                idx, ncc = read(y)  # type: ignore[operator]
                if not bool((ncc[cells] >= READ).all()):
                    continue
                v = _seconds(_text(idx[cells]))
                if v is None:
                    continue
                if last is not None and v == last + 1:
                    found.append(v - (begin + k / fps) + 0.5 / fps)
                    break
                last = v
        if found:
            seg.offset = float(np.median(found))


def timer_value(fl: Field, t: float) -> int | None:
    seg = None
    for s in fl.segments:
        if s.t <= t + 1e-6:
            seg = s
    if seg is None:
        seg = fl.segments[0] if fl.segments else None
    if seg is None:
        return None
    return math.floor(t + seg.offset) if seg.running else seg.value


def timer_glyphs(fl: Field, value: int) -> list[int]:
    """The field's digits for ``value`` seconds (MMSS, as many digits as it has cells)."""
    n = len(fl.cols)
    if n <= 2:  # seconds only: they wrap, so a value before the model's zero is fine
        return [ord(c) for c in f"{value % 60:02d}"[-n:]]
    m, s = divmod(max(value, 0), 60)
    text = f"{m:0{n - 2}d}{s:02d}"
    return [ord(c) for c in text[-n:]]


def describe(fields: list[Field]) -> list[str]:
    out = []
    for fl in fields:
        seen = sorted({s for _, s in fl.readings})
        extra = ""
        if fl.kind == "timer":
            extra = " segments " + ", ".join(
                f"{'run' if s.running else 'pause'}@{s.t:.1f}s"
                + (f" o={s.offset:+.3f}" if s.running else f" ={s.value}")
                for s in fl.segments
            )
        out.append(
            f"{fl.label()}: {fl.kind} ({len(fl.readings)} readings, {fl.changes} changes, "
            f"values {seen[:4]}{'...' if len(seen) > 4 else ''}){extra}"
        )
    return out


__all__ = ["Field", "describe", "find_fields", "survey", "timer_glyphs", "timer_value"]
