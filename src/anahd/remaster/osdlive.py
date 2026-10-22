"""Live numbers read a whole stretch at a time (segment reading).

Between two changes a live value is constant for several frames (Betaflight redraws
it a few times a second), and the OSD sits still on screen. So instead of reading each
frame alone, a pass over the frames cuts every number field into **segments** where its
cells keep the same picture, and reads each segment from the average of all its frames:
the analog noise averages out and the digits' margin grows.

A cut is made where the field's cells stop looking like the running segment (mean
correlation of the cells against the segment's average under SIM_CUT) for two frames in
a row; one bad frame (a tear) is skipped, not cut. A segment shorter than MIN_FRAMES, or a
cell whose best digit does not lead the runner-up by SEG_LEAD_SHARE of the lead it would
have on a perfect picture of itself (each font's digits differ by more or less), stays
unsure: it is never drawn (no stale or guessed digits). A segment is also cut where a
cell reads a different digit with confidence, and cells holding nothing are left out of
the comparison, so a value is never carried across a change.

The result is a timeline, per frame and cell, of the digit to draw or -1.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import torch
from torch import Tensor

from .osd import COLS, READ, Font, Grid, sample_cells
from .osdfields import Field

SIM_CUT = 0.8  # mean cell correlation with the segment below which the value changed
BLANK_STD = 0.08  # a cell's pixel spread under which it holds no glyph
MIN_FRAMES = 3  # frames a segment needs to be read at all
SEG_LEAD_SHARE = 0.7  # of a digit's perfect lead it must reach on the segment's average
FRAME_READ = 0.75  # a single frame's digit counts as a confident read from this score
FRAME_LEAD = 0.2  # ...and this lead over the runner-up
LOOSE_SHARE = 0.5  # of a digit's perfect lead, for the reads that flag a mixed segment
DIGITS = "0123456789"


@dataclass
class Timeline:
    values: Tensor  # (frames, rows*COLS) int16: digit glyph to draw, -1 = unsure/not a number
    segments: int  # segments read (for reports)
    # Per timer field (index in the layout's fields): drawn only when every readable tick
    # in the stretch matched the clock model ([matches, mismatches] in timer_check).
    timer_ok: dict[int, bool] = field(default_factory=dict)
    timer_check: dict[int, list[int]] = field(default_factory=dict)


def _similar(a: Tensor, b: Tensor) -> bool:
    """Whether cells ``a`` look like cells ``b`` (both (k, P)): the mean correlation over
    the cells that hold something in either (a blank cell is noise against noise, which
    says nothing; a digit that appears or disappears is a change, caught by its contrast)."""
    sa, sb = a.std(1), b.std(1)
    held = (sa > BLANK_STD) | (sb > BLANK_STD)
    if not bool(held.any()):
        return True
    return float(_ncc(a[held], b[held]).mean()) >= SIM_CUT


def _ncc(a: Tensor, b: Tensor) -> Tensor:
    """Row-wise correlation of (k, P) against (k, P)."""
    a = a - a.mean(1, keepdim=True)
    b = b - b.mean(1, keepdim=True)
    out: Tensor = (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1)).clamp_min(1e-6)
    return out


def read_segments(
    frames: Iterable[Tensor], grid: Grid, font: Font, fields: list[Field]
) -> Timeline:
    """``frames``: luma (H, W, 0..1) at the picture size, in order. One pass, keeping only
    each field's running segment (a whole flight fits in memory). Every field given is
    read (numbers, timers, single held digits)."""
    numbers = [fl for fl in fields if fl.cols]
    dev = font.device
    cells = [torch.tensor([fl.row * COLS + c for c in fl.cols], device=dev) for fl in numbers]
    cutters = [_Cutter() for _ in numbers]
    found: list[tuple[int, int, int, Tensor, list[set[int]]]] = []  # field, a, b, mean, seen
    digits = torch.tensor([ord(c) for c in DIGITS], device=dev)
    keep = font.readable
    only = torch.zeros_like(keep)
    only[digits] = True
    count = 0
    try:
        font.readable = only
        perfect = _perfect_leads(font, digits)
        for y in frames:
            sampled = sample_cells(y, grid)
            idx, ncc = font.read(sampled)
            sure = (ncc >= FRAME_READ) & (font.margin >= FRAME_LEAD)
            # Looser single-frame reads, only to catch a segment that holds two values.
            loose = (ncc >= READ) & (font.margin >= LOOSE_SHARE * perfect[idx.clamp(0, 255)])
            for k, at in enumerate(cells):
                read = [int(g) if bool(ok) else -1 for g, ok in zip(idx[at], sure[at], strict=True)]
                seen = [
                    int(g) if bool(ok) else -1 for g, ok in zip(idx[at], loose[at], strict=True)
                ]
                for a, b, mean, kinds in cutters[k].push(sampled[at], read, seen):
                    found.append((k, a, b, mean, kinds))
            count += 1
        for k, cutter in enumerate(cutters):
            for a, b, mean, kinds in cutter.close():
                found.append((k, a, b, mean, kinds))
        n = grid.rows * COLS
        values = torch.full((count, n), -1, dtype=torch.int16)
        segs = 0
        for k, a, b, mean, kinds in found:
            if b - a < MIN_FRAMES:
                continue
            idx, ncc = font.read(mean)
            # The lead a digit needs is a share of the lead it has over its nearest
            # look-alike when matched perfectly (each font's digits differ by more or less).
            need = SEG_LEAD_SHARE * perfect[idx.clamp(0, 255)]
            ok = ((ncc >= READ) & (font.margin >= need)).tolist()
            segs += 1
            fl = numbers[k]
            for j, c in enumerate(fl.cols):
                # A segment whose frames read two different digits held a change: unsure.
                if ok[j] and kinds[j] <= {int(idx[j])}:
                    values[a:b, fl.row * COLS + c] = int(idx[j])
    finally:
        font.readable = keep
    return Timeline(values, segs)


def _perfect_leads(font: Font, digits: Tensor) -> Tensor:
    """Per glyph (256): a digit template read against the digits, its lead over the
    runner-up (reference font: '5' 0.19 to '1' 0.46)."""
    out = torch.ones(256, device=digits.device)
    font.read(font.tmpl[digits])
    out[digits] = font.margin
    return out


class _Cutter:
    """Cuts one field's frames into segments [a, b) where its value holds. A frame is off
    the segment when its cells stop looking like the segment's average (SIM_CUT) or when a
    cell reads a different digit with confidence than the segment did (a '2' becoming a
    '3' looks alike, but reads apart). A single frame off (a tear) stays inside the segment
    but is not averaged; two within two frames end it at the first of them (frames between
    may be too blurred to read), so a value is never carried across a change. Each segment
    also returns, per cell, the digits its frames read (loosely): more than one means it
    held a change and is not drawn."""

    def __init__(self) -> None:
        self.i = -1
        self.a = 0
        self.total: Tensor | None = None
        self.count = 0
        self.ref: list[int] = []  # per cell, the digit the segment reads (-1: none yet)
        self.kinds: list[set[int]] = []
        self.recent: list[tuple[int, tuple[Tensor, list[int], list[int]]]] = []
        self.last_miss: int | None = None  # the last frame found off the segment

    def _start(self, at: int, frames: list[tuple[Tensor, list[int], list[int]]]) -> None:
        self.a, self.count = at, len(frames)
        self.total = torch.stack([r for r, _, _ in frames]).sum(0)
        self.ref = [-1] * len(frames[0][1])
        self.kinds = [set() for _ in frames[0][1]]
        for _, d, seen in frames:
            self._note(d, seen)

    def _note(self, d: list[int], seen: list[int]) -> None:
        self.ref = [r if r >= 0 else x for r, x in zip(self.ref, d, strict=True)]
        for k, x in zip(self.kinds, seen, strict=True):
            if x >= 0:
                k.add(x)

    def _fits(self, r: Tensor, d: list[int]) -> bool:
        if self.total is None:
            return False
        if any(x >= 0 and y >= 0 and x != y for x, y in zip(d, self.ref, strict=True)):
            return False
        return _similar(r, self.total / self.count)

    def _done(self, b: int) -> tuple[int, int, Tensor, list[set[int]]]:
        total = self.total if self.total is not None else torch.zeros(0)
        return (self.a, b, total / self.count, self.kinds)

    def push(
        self, r: Tensor, d: list[int], seen: list[int]
    ) -> list[tuple[int, int, Tensor, list[set[int]]]]:
        self.i += 1
        out: list[tuple[int, int, Tensor, list[set[int]]]] = []
        frame = (r, d, seen)
        self.recent = [*self.recent[-2:], (self.i, frame)]
        if self.total is None:
            self._start(self.i, [frame])
        elif self._fits(r, d):
            self.total, self.count = self.total + r, self.count + 1
            self._note(d, seen)
        elif self.last_miss is not None and self.i - self.last_miss <= 2:
            # A second frame off the segment within two (frames in between may be too
            # blurred to read either way): the segment ended at the first of them.
            first = self.last_miss
            out.append(self._done(first))
            since = [f for k, f in self.recent if k >= first]
            agree = [
                f for f in since
                if not any(x >= 0 and y >= 0 and x != y for x, y in zip(f[1], d, strict=True))
            ]  # fmt: skip
            self._start(first, agree or [frame])
            self.last_miss = None
        else:
            self.last_miss = self.i
        return out

    def close(self) -> list[tuple[int, int, Tensor, list[set[int]]]]:
        if self.total is None:
            return []
        return [self._done(self.i + 1)]
