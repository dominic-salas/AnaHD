"""Cut markers: stretches of a recording to leave out of the cleaned output.

A cut is a time range in seconds; what is left between the cuts is kept, in order, and
joined into one output file. Cutting the very start or end is how a clip is trimmed.
Frame numbers are worked out once here, so the app, the CLI and the tests agree on
exactly which frames are kept.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from anahd.errors import AnaHDError

Cut = tuple[float, float]  # seconds, start < end


@dataclass(frozen=True)
class Span:
    """Frames [start, end) of the recording that are kept."""

    start: int
    end: int

    @property
    def frames(self) -> int:
        return self.end - self.start


_TIME = r"(?:(\d+):)?(\d+(?:\.\d+)?)"


def parse_time(text: str) -> float:
    m = re.fullmatch(_TIME, text.strip())
    if m is None:
        raise ValueError(f"not a time: {text!r} (use seconds or m:ss)")
    return 60 * int(m.group(1) or 0) + float(m.group(2))


def parse_cut(text: str) -> Cut:
    """``"12.5-20"`` or ``"1:02-1:10.5"`` -> (start, end) in seconds."""
    parts = text.split("-")
    if len(parts) != 2:
        raise ValueError(f"not a cut: {text!r} (use START-END, e.g. 12.5-20 or 1:02-1:10)")
    a, b = parse_time(parts[0]), parse_time(parts[1])
    if b <= a:
        raise ValueError(f"cut {text!r} ends before it starts")
    return a, b


def normalise(cuts: Iterable[Cut], duration: float) -> list[Cut]:
    """Clamped to the recording, sorted, with overlapping or touching cuts merged."""
    clamped = sorted(
        (max(a, 0.0), min(b, duration)) for a, b in cuts if min(b, duration) > max(a, 0.0)
    )
    merged: list[Cut] = []
    for a, b in clamped:
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged


def keep_spans(cuts: Iterable[Cut], frames: int, fps: float) -> list[Span]:
    """The frames left between the cuts. A frame is cut when its time falls in a cut."""
    spans: list[Span] = []
    pos = 0
    for a, b in normalise(cuts, frames / fps):
        first_cut, after_cut = _ceil(a * fps), _ceil(b * fps)
        if after_cut <= first_cut:
            continue  # no frame falls inside this cut: nothing to leave out, no split
        if first_cut > pos:
            spans.append(Span(pos, min(first_cut, frames)))
        pos = max(pos, after_cut)
    if pos < frames:
        spans.append(Span(pos, frames))
    return spans


def _ceil(x: float) -> int:
    """Ceiling that ignores float noise (2.0000000001 frames is frame 2)."""
    r = round(x)
    return r if abs(x - r) < 1e-6 else int(x) + (x > int(x))


def output_to_source(seconds: float, spans: Sequence[Span], fps: float) -> float:
    """Where a moment of the output comes from in the recording."""
    frame = seconds * fps
    for s in spans:
        if frame < s.frames:
            return (s.start + frame) / fps
        frame -= s.frames
    if not spans:
        raise AnaHDError("nothing is kept")
    return (spans[-1].end - 1) / fps
