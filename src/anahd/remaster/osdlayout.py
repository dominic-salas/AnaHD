"""What a clip's OSD looks like, worked out once when the clip is added, from
a few frames around the middle.

``analyze_clip(path, font)`` samples the frames every half second over ten seconds around
the clip's middle, drops the ones hit by breakup, finds the character grid, takes the
layout from the cleanest frame (only cells that hold OSD there, and in most of the other
clean frames, may ever be drawn), and sorts the cells:

- **fixed**: one glyph all the time (labels, icons, the craft name, sprites). With the
  pilot's font they are drawn as that glyph; without one, from the clip's own sharpened
  median picture of them;
- **fields**: timers (generated from the clock), held values (changing rarely) and live
  values (read every frame); only with a font, since reading needs one.

It returns an :class:`OsdLayout`, with an annotated picture for the app's clips page, and
caches it per clip, font and file time. Nothing in here is specific to one pilot's OSD.
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from torch import Tensor

from anahd.media import Crop, probe, read_frames
from anahd.pipeline import Measured, Options, _full_range, measure

from . import osdcli
from .osd import (
    COLS,
    CONFIDENT,
    READ,
    STATIC,
    TEMPLATE,
    VALUE_CHARS,
    Font,
    Grid,
    Layout,
    Redraw,
    _distinct,
    _pitches,
    find_grid,
    has_osd,
    sample_cells,
)
from .osdcells import Cells, find_cells
from .osdfields import Field, classify_from, timer_value
from .osdlive import Timeline, read_segments

TIMER_MIN_TICKS = 10  # readable frames a timer must be checked on before it is drawn
WINDOW = 10.0  # seconds around the middle that are sampled
EVERY = 0.5
DIFFERS = 0.15  # mean |sample - median| over a cell's opaque OSD pixels: another glyph

KIND_COLOURS = {  # BGR, for the annotated picture
    "fixed": (90, 220, 90),
    "timer": (40, 210, 250),
    "slow": (240, 200, 60),
    "live": (230, 90, 230),
    "separator": (150, 150, 150),
    "group": (40, 120, 255),  # orange: a picture drawn (or left) as a whole
}


@dataclass
class OsdLayout:
    """A clip's OSD: grid, the cells that may be drawn, what each is, and a picture."""

    grid: Grid
    cells: Tensor  # (rows*COLS,) bool: the layout; nothing outside it is ever drawn
    fixed: Tensor  # glyph per cell, TEMPLATE (fixed, no font) or -1 (changing)
    fields: list[Field]
    font: Font | None
    template: tuple[Tensor, Tensor] | None  # white value, opacity at the picture size
    picture: tuple[int, int]  # w, h of the picture area the grid is in
    reference_time: float  # seconds: the frame the layout was read from
    seconds: float  # how long the analysis took
    preview: NDArray[np.uint8] = field(repr=False)  # annotated reference frame, RGB
    groups: Tensor | None = None  # per cell: the multi-glyph picture it belongs to, or -1
    steps: dict[str, float] = field(default_factory=dict)  # seconds per analysis step
    crop: Crop | None = None  # the picture area the grid is in
    vtx: Tensor | None = None  # cells of the VTX line ("R:8:400"): its digits are redrawn

    def redraw(
        self,
        size: tuple[int, int],
        osd: Tensor,
        fps: float,
        t0: float = 0.0,
        draw_live: bool = False,
    ) -> Redraw:
        """The renderer for this clip at the output ``size``: this layout is the authority.
        ``draw_live``: also redraw the moving numbers (off: they are cleaned in place)."""
        lay = Layout(self.cells, 0, np.zeros((1, 1), np.uint8))
        return Redraw(
            self.font, self.grid, size, self.picture, osd, self.fixed, self.fields, fps, t0,
            layout=lay, template=self.template, device=self.cells.device, groups=self.groups,
            draw_live=draw_live, vtx=self.vtx,
        )  # fmt: skip

    def read_live(
        self, path: Path, start: float | None = None, duration: float | None = None
    ) -> Timeline | None:
        """The live numbers of this stretch of the clip read segment by segment (osdlive):
        one extra decode pass over it. None without a font."""
        if self.font is None:
            return None
        info = probe(path)
        crop = self.crop
        # The same pass checks the timers: a shadow renderer runs the clock model over every
        # frame and compares it with each tick the recording shows readably.
        ones = torch.ones(
            self.picture[1], self.picture[0], dtype=torch.bool, device=self.cells.device
        )
        shadow = self.redraw(self.picture, ones, _fps(info.fps), start or 0.0, draw_live=True)

        def frames() -> Iterator[Tensor]:
            for f in read_frames(path, info, crop, start=start, duration=duration):
                y = torch.from_numpy(_full_range(f, info).y.copy()).to(self.cells.device).float()
                y = y / 255
                shadow.update(y)
                yield y

        # Lone digits that were constant in the analysis (the VTX channel) are read too:
        # a pilot can change channel or power mid-flight.
        lone = [
            Field(i // COLS, [i % COLS], kind="slow")
            for i in (self.cells & (self.fixed >= ord("0")) & (self.fixed <= ord("9")))
            .nonzero()[:, 0]
            .tolist()
        ]
        timeline = read_segments(frames(), self.grid, self.font, self.fields + lone)
        timeline.timer_check = dict(shadow.timer_check)
        timeline.timer_ok = {
            j: good >= TIMER_MIN_TICKS and bad == 0 for j, (good, bad) in shadow.timer_check.items()
        }
        return timeline

    def summary(self) -> dict[str, object]:
        kinds: dict[str, int] = {}
        for fl in self.fields:
            kinds[fl.kind] = kinds.get(fl.kind, 0) + 1
        return {
            "grid": f"{self.grid.rows} rows x {COLS} cols of {self.grid.cell_w:.1f} x "
            f"{self.grid.cell_h:.1f} px",
            "cells": int(self.cells.sum()),
            "fixed": int((self.fixed != -1).sum()),
            "groups": 0 if self.groups is None else int(self.groups.max()) + 1,
            "group_cells": 0 if self.groups is None else int((self.groups >= 0).sum()),
            "fields": kinds,
            "font": self.font is not None,
            "reference_time": round(self.reference_time, 2),
            "seconds": round(self.seconds, 2),
            "steps": {k: round(v, 2) for k, v in self.steps.items()},
        }


class _Read(NamedTuple):
    t: float
    y: Tensor
    cells: Tensor
    idx: Tensor | None
    ncc: Tensor | None
    share: float


_CACHE: dict[tuple[str, float, str, str], OsdLayout | None] = {}


def analyze_clip(
    path: Path,
    font: Path | str | None = None,
    *,
    cli: Path | str | None = None,
    device: torch.device | None = None,
    measured: Measured | None = None,
) -> OsdLayout | None:
    """The clip's OSD layout, or None when it has no OSD (or none that forms a grid).
    ``cli``: an optional Betaflight CLI dump/diff; cells away from every element it shows
    are left out of the layout (see :mod:`anahd.remaster.osdcli`). Cached per clip (path
    and file time), font and dump."""
    path = Path(path)
    key = (str(path.resolve()), path.stat().st_mtime, str(font or ""), str(cli or ""))
    if key not in _CACHE:
        dump = Path(cli).read_text(encoding="utf-8", errors="replace") if cli else None
        _CACHE[key] = _analyze(path, Path(font) if font else None, device, measured, dump)
    return _CACHE[key]


def _analyze(
    path: Path,
    font_path: Path | None,
    device: torch.device | None,
    measured: Measured | None,
    dump: str | None = None,
) -> OsdLayout | None:
    began = time.perf_counter()
    steps: dict[str, float] = {}

    def step(name: str) -> None:
        steps[name] = time.perf_counter() - began - sum(steps.values())

    dev = device or torch.device("cuda")
    info = probe(path)
    meas = measured or measure(path, info, Options())
    if not meas.osd.any():
        return None
    mask = torch.from_numpy(meas.osd).to(dev)
    window = min(WINDOW, info.duration)
    every = min(EVERY, window / 20)  # a short clip is sampled more densely
    start = max(0.0, info.duration / 2 - window / 2)
    frames: list[tuple[float, Tensor]] = []
    for k, f in enumerate(
        read_frames(path, info, meas.crop, start=start, duration=window, every=every)
    ):
        y = torch.from_numpy(_full_range(f, info).y.copy()).to(dev).float() / 255
        frames.append((start + k * every, y))
    if len(frames) < 4:
        return None
    step("frames")
    font = Font(font_path, dev) if font_path else None
    still = torch.stack([y for _, y in frames]).median(0).values
    grid = find_grid(still, mask, font) if font is not None else find_grid_plain(still, mask)
    if grid is None:
        return None
    step("grid")
    inside = sample_cells(mask.float(), grid).mean(1) > 0.05

    # Breakup frames read far worse than the typical one: left out from here on.
    reads: list[_Read] = []
    for t, y in frames:
        cells = sample_cells(y, grid)
        if font is not None:
            idx, ncc = font.read(cells)
            reads.append(
                _Read(t, y, cells, idx, ncc, float(((ncc >= READ) & inside).float().mean()))
            )
        else:
            reads.append(
                _Read(t, y, cells, None, None, float((has_osd(cells) & inside).float().mean()))
            )
    typical = float(np.median([r.share for r in reads]))
    clean = [r for r in reads if r.share >= 0.6 * typical] or reads
    still = torch.stack([r.y for r in clean]).median(0).values

    # The layout: cells with OSD content in the cleanest frame and in most clean frames.
    best = sorted(range(len(clean)), key=lambda i: -clean[i].share)[:5]
    content = [has_osd(clean[i].cells) & inside for i in best]
    cells = content[0] & (torch.stack(content).sum(0) >= min(3, len(content)))
    if dump:  # the pilot's own settings: only ever narrows the layout
        cells &= osdcli.allowed_cells(osdcli.parse(dump), grid.rows).to(dev)
    reference_time = clean[best[0]].t

    # Fixed or changing is judged over the whole clip (AnaHD's measurement samples), not
    # the ten seconds: a timer's tens digit or a slow value may hold for all of them.
    template = None
    groups = None
    if font is not None:
        codes, groups = static_groups(meas.samples, still, mask, grid, font)
        step("groups")
        # Not limited to AnaHD's OSD mask: a row torn by breakup in many frames (the top
        # row of day_overcast) can fall outside it while its text reads fine; the finder's
        # own checks (consistent reads, the still picture, white strokes) keep scenery out.
        anywhere = torch.ones_like(inside)
        found = find_cells(meas.samples, still, anywhere, grid, font, groups, codes, in_mask=inside)
        step("cells")
        cells = torch.tensor([k != "" for k in found.kind], device=dev)
        if dump:  # the pilot's own settings: only ever narrows the layout
            cells &= osdcli.allowed_cells(osdcli.parse(dump), grid.rows).to(dev)
        fixed = torch.where(cells, found.glyph, torch.full_like(found.glyph, -1))
        if dump:  # names the dump spells out are drawn from it, whatever was read
            named = osdcli.static_text(dump, grid.rows)
            if named:
                at = torch.tensor(list(named), device=dev)
                groups[at] = int(groups.max()) + 1
                fixed[at] = torch.tensor(list(named.values()), device=dev)
                cells[at] = True
        idx, ncc = font.read(sample_cells(still, grid))
        seen = set(idx[(ncc >= CONFIDENT) & cells].tolist()) | {ord(c) for c in "0123456789-.:%"}
        font.limit(seen | set(fixed[fixed >= 0].tolist()))
        fields = _fields(found.kind, cells, grid.rows)

        def read(y: Tensor) -> tuple[Tensor, Tensor]:
            return font.read(sample_cells(y, grid))

        timed = [
            (r.t, r.idx, r.ncc >= READ) for r in clean if r.idx is not None and r.ncc is not None
        ]
        classify_from(fields, timed, path, info, meas.crop, read, _fps(info.fps))
        _settle_kinds(fields, found.kind)
        _timer_minutes(fields, found, fixed, timed)
        cells = cells | (fixed >= 0)
        groups = torch.where(fixed >= 0, groups, torch.full_like(groups, -1))  # digits moved
        groups = _element_groups(groups, fixed, cells, grid.rows)
        step("fields")
    else:
        whole = [
            sample_cells(torch.from_numpy(s).to(dev).float() / 255, grid) for s in meas.samples
        ]
        fixed, template = _fixed_templates(whole, inside, still, cells, grid)
        fields = []
    preview = _annotate(path, info, meas, reference_time, grid, cells, fixed, fields, groups)
    step("picture")
    return OsdLayout(
        grid, cells, fixed, fields, font, template, (meas.crop.width, meas.crop.height),
        reference_time, time.perf_counter() - began, preview, groups, steps, meas.crop,
        _vtx_line(fixed, cells, grid.rows),
    )  # fmt: skip


def _fps(rate: str) -> float:
    num, _, den = rate.partition("/")
    return int(num) / int(den or 1)


def find_grid_plain(still: Tensor, osd: Tensor) -> Grid | None:
    """The grid without a font: the pitch from the strokes' autocorrelation (as with a
    font), the offset where the gaps between characters fall on the cell borders."""
    h, w = still.shape
    darkest = -F.max_pool2d(-still[None, None], 5, 1, 2)[0, 0]
    strokes = ((still > 0.6) & (still - darkest > 0.3) & osd).float()
    if strokes.sum() < 50:
        return None
    xs = _distinct([*_pitches(strokes, 24.0, 40.0), w / 30, w / 32])
    ys = _distinct([*_pitches(strokes.T.contiguous(), 40.0, 64.0), h / 13, h / 16])
    col = strokes.sum(0).cpu().numpy()
    row = strokes.sum(1).cpu().numpy()
    best: tuple[float, Grid] | None = None
    for px in xs:
        ox, cx = _phase(col, px)
        for py in ys:
            oy, cy = _phase(row, py)
            rows = min(16, max(13, round(h / py)))
            x0 = ox - px * round((ox - (w - COLS * px) / 2) / px)
            y0 = oy - py * round((oy - (h - rows * py) / 2) / py)
            score = cx + cy
            if best is None or score > best[0]:
                best = (score, Grid(px, py, x0, y0, rows, score))
    return best[1] if best else None


def _phase(profile: NDArray[np.float32], pitch: float, bins: int = 32) -> tuple[float, float]:
    """Where, within a cell of ``pitch``, the strokes are thinnest (the border between
    characters), and how clear that is (0..1)."""
    x = np.arange(len(profile))
    ph = ((x / pitch) % 1 * bins).astype(int)
    hist = np.bincount(ph, weights=profile, minlength=bins)
    hist = hist / np.maximum(np.bincount(ph, minlength=bins), 1)
    smooth = np.convolve(np.r_[hist[-2:], hist, hist[:2]], np.ones(5) / 5, "valid")
    k = int(np.argmin(smooth))
    contrast = float((smooth.max() - smooth.min()) / max(smooth.max(), 1e-9))
    return (k + 0.5) / bins * pitch, contrast


GROUP_VOTE = 0.5  # a sample's reading counts toward a static cell from this correlation
GROUP_STILL = 0.6  # every cell of a group must match its glyph this well on the still
GROUP_MEAN = 0.75  # and the group as a whole this well
GROUP_ANCHOR = 0.85  # with at least one contrasty glyph matched this well


def _fields(kind: list[str], cells: Tensor, rows: int) -> list[Field]:
    """Runs of held or of live cells on a row (a run never mixes the two: numbers)."""
    on = cells.tolist()

    def k(r: int, c: int) -> str:
        return kind[r * COLS + c] if on[r * COLS + c] else ""

    out: list[Field] = []
    for r in range(rows):
        c = 0
        while c < COLS:
            if k(r, c) not in ("held", "live"):
                c += 1
                continue
            end = c
            while end + 1 < COLS and k(r, end + 1) == k(r, c):
                end += 1
            out.append(Field(r, list(range(c, end + 1))))
            c = end + 1
    return out


def _vtx_line(fixed: Tensor, cells: Tensor, rows: int) -> Tensor:
    """The VTX element: a row run that starts with a band letter followed by ':' (Betaflight
    draws it as band:channel:power, e.g. "R:8:400"). Its digits are the only numbers the
    default redraw draws."""
    out = torch.zeros_like(cells)
    on, fx = cells.tolist(), fixed.tolist()
    for r in range(rows):
        c = 0
        while c < COLS:
            if not on[r * COLS + c]:
                c += 1
                continue
            end = c
            while end + 1 < COLS and on[r * COLS + end + 1]:
                end += 1
            first, second = fx[r * COLS + c], fx[r * COLS + c + 1] if c + 1 <= end else -1
            if ord("A") <= first <= ord("Z") and second == ord(":"):
                out[r * COLS + c : r * COLS + end + 1] = True
            c = end + 1
    return out


def _element_groups(groups: Tensor, fixed: Tensor, cells: Tensor, rows: int) -> Tensor:
    """Fixed cells outside a picture are grouped by element: the run of OSD cells they sit
    in on their row (an icon, its number and its ':' or '.'). A group is shown while at
    least half of its fixed cells match (Redraw), so a thin '-', ':' or '.' that reads
    poorly in a frame is drawn with the icon beside it, not dropped alone."""
    out = groups.clone()
    nxt = int(groups.max()) + 1 if bool((groups >= 0).any()) else 0
    on, fx, gr = cells.tolist(), fixed.tolist(), groups.tolist()
    for r in range(rows):
        c = 0
        while c < COLS:
            if not on[r * COLS + c]:
                c += 1
                continue
            end = c
            while end + 1 < COLS and on[r * COLS + end + 1]:
                end += 1
            members = [
                r * COLS + k
                for k in range(c, end + 1)
                if fx[r * COLS + k] >= 0 and gr[r * COLS + k] < 0
            ]
            if members:
                out[members] = nxt
                nxt += 1
            c = end + 1
    return out


def _settle_kinds(fields: list[Field], kind: list[str]) -> None:
    """A field recognised as a timer stays one; otherwise the cells' class decides: held
    (slow) or live, whatever the field's readings in the ten-second window said."""
    for fl in fields:
        k = kind[fl.row * COLS + fl.cols[0]] if fl.cols else ""
        if fl.kind == "timer":
            continue
        fl.kind = "slow" if k == "held" else "live"


def _timer_minutes(
    fields: list[Field], found: Cells, fixed: Tensor, timed: list[tuple[float, Tensor, Tensor]]
) -> None:
    """A seconds-only timer whose left neighbour is a ':' gets the minutes digit left of
    it (so the model counts minutes too); the tens-of-minutes digit is held."""
    taken = {(fl.row, c) for fl in fields for c in fl.cols}
    for fl in list(fields):
        if fl.kind != "timer" or len(fl.cols) != 2:
            continue
        r, c0 = fl.row, fl.cols[0]
        if c0 < 3 or int(fixed[r * COLS + c0 - 1]) != ord(":"):
            continue
        units = r * COLS + c0 - 2
        m = int(found.mode[units])
        if not (ord("0") <= m <= ord("9")):
            continue
        fixed[units] = -1
        for f2 in fields:
            if f2 is not fl and (r, c0 - 2) in {(f2.row, c) for c in f2.cols}:
                f2.cols.remove(c0 - 2)
        fl.cols.insert(0, c0 - 2)
        _shift_minutes(fl, units, timed)
        tens = units - 1
        if (r, c0 - 3) not in taken and ord("0") <= int(found.mode[tens]) <= ord("9"):
            fixed[tens] = -1
            fields.append(Field(r, [c0 - 3], kind="slow"))
    fields[:] = [fl for fl in fields if fl.cols]


def _shift_minutes(fl: Field, cell: int, timed: list[tuple[float, Tensor, Tensor]]) -> None:
    """Make the timer model count whole minutes (its seconds were read from SS digits
    only, so its minute count was arbitrary): the shift on which most confident readings
    of the minutes digit in the window agree (the minute may roll over in the window)."""
    votes: dict[int, int] = {}
    for t, idx, ok in timed:
        g, v = int(idx[cell]), timer_value(fl, t)
        if not bool(ok[cell]) or v is None or not ord("0") <= g <= ord("9"):
            continue
        shift = 60 * (g - ord("0")) + v % 60 - v  # so that minute shows g at time t
        votes[shift] = votes.get(shift, 0) + 1
    if not votes:
        return
    delta = max(votes, key=lambda k: votes[k])
    for seg in fl.segments:
        if seg.running:
            seg.offset += delta
        else:
            seg.value += delta


def static_groups(
    samples: Sequence[NDArray[np.uint8]], still: Tensor, mask: Tensor, grid: Grid, font: Font
) -> tuple[Tensor, Tensor]:
    """Pictures made of several glyphs (some fonts form images when
    glyphs are put together). Reads every cell with the whole font, near-solid tiles
    included, over the clip's samples; a cell is static when one glyph wins at least 80 %
    of the samples that read it and that glyph matches the still picture. Touching static
    cells (8-neighbours) form a group, kept only when it matches well as a whole and holds
    a contrasty glyph (so a patch of plain sky never becomes a group). Returns per cell the
    glyph and the group index (-1: none)."""
    dev = still.device
    n = grid.rows * COLS
    inside = sample_cells(mask.float(), grid).mean(1) > 0.05
    votes = torch.zeros(n, 256, device=dev)
    sure = torch.zeros(n, 256, device=dev)  # confident reads with the readable glyphs
    reads, shares = [], []
    for frame in samples:
        cells = sample_cells(torch.from_numpy(frame).to(dev).float() / 255, grid)
        idx, ncc = font.read_all(cells)
        r_idx, r_ncc = font.read(cells)
        reads.append((idx, ncc, r_idx, r_ncc))
        shares.append(float(((ncc >= GROUP_VOTE) & inside).float().mean()))
    typical = float(np.median(shares)) if shares else 0.0
    clean = [r for r, sh in zip(reads, shares, strict=True) if sh >= 0.6 * typical]
    for idx, ncc, r_idx, r_ncc in clean:
        ok = (ncc >= GROUP_VOTE) & inside
        votes[ok.nonzero()[:, 0], idx[ok]] += 1
        ok = (r_ncc >= READ) & inside
        sure[ok.nonzero()[:, 0], r_idx[ok]] += 1
    total = votes.sum(1)
    best, glyph = votes.max(1)
    m = max(len(clean), 1)
    s_idx, s_ncc = font.read_all(sample_cells(still, grid))
    # Digits and other value glyphs must hold like a fixed cell (STATIC); picture tiles,
    # often torn by colour noise, 80 %.
    value = torch.zeros(256, dtype=torch.bool, device=dev)
    value[[ord(c) for c in VALUE_CHARS]] = True
    need = torch.where(value[glyph], STATIC, 0.8)
    # A digit must also be there in nearly every sample: one that blanks out at times (the
    # third digit of a link quality of 100) belongs to a number, never to a picture.
    there = ~value[glyph] | (total >= 0.9 * m)
    static = (total >= 0.5 * m) & (best >= need * total) & (s_idx == glyph) & (s_ncc >= GROUP_STILL)
    static &= inside & there
    # A cell that is confidently read as two different glyphs is changing (a timer digit
    # whose blurred median happens to match some low-contrast tile): never static.
    top2 = sure.topk(2, dim=1).values
    static &= ~((top2[:, 1] >= 0.05 * sure.sum(1)) & (top2[:, 1] >= 2))
    grid_static = static.view(grid.rows, COLS).cpu().numpy().astype(np.uint8)
    count, labels = cv2.connectedComponents(grid_static, connectivity=8)
    labels_t = torch.from_numpy(labels.astype(np.int64)).to(dev).view(-1)
    groups = torch.full((n,), -1, dtype=torch.long, device=dev)
    k = 0
    for lab in range(1, count):
        members = (labels_t == lab).nonzero()[:, 0]
        if len(members) < 2:
            continue  # single glyphs are the fixed cells' business
        q = s_ncc[members]
        contrasty = font.readable[glyph[members]] & (q >= GROUP_ANCHOR)
        if float(q.mean()) >= GROUP_MEAN and bool(contrasty.any()):
            groups[members] = k
            k += 1
    return glyph, groups


def _fixed_templates(
    samples: Sequence[Tensor],
    inside: Tensor,
    still: Tensor,
    cells: Tensor,
    grid: Grid,
) -> tuple[Tensor, tuple[Tensor, Tensor]]:
    """Without a font: layout cells that look the same in (nearly) every sample of the
    clip are fixed, and drawn from the median picture, sharpened: white strokes and their
    black outline, opaque; the rest transparent. A sample differs in a cell when, over the
    median's strokes and outline (opaque, so the scene never shows there), it is off by
    more than DIFFERS on average (synthetic OSDs: same glyph <= 0.09, another >= 0.20)."""
    share = [float((has_osd(c) & inside).float().mean()) for c in samples]
    typical = float(np.median(share))
    clean = [c for c, sh in zip(samples, share, strict=True) if sh >= 0.6 * typical]
    stack = torch.stack(clean)  # (F, cells, P)
    median = stack.median(0).values
    stroke = median > 0.55
    ring = F.max_pool1d(stroke.float()[:, None], 5, 1, 2)[:, 0] > 0
    opaque = stroke | ((median < 0.3) & ring)
    off = ((stack - median).abs() * opaque).sum(2) / opaque.sum(1).clamp_min(1)
    differs = (off > DIFFERS).float().mean(0)
    present = torch.stack([has_osd(c) for c in clean]).float().mean(0)
    fixed_mask = cells & stroke.any(1) & (differs <= 1 - STATIC) & (present >= 0.8)
    fixed = torch.where(fixed_mask, TEMPLATE, -1).long()
    # The template layer: the still, contrast-stretched; opacity from white strokes and
    # the dark outline next to them.
    sharp = (still + 0.8 * (still - F.avg_pool2d(still[None, None], 3, 1, 1)[0, 0])).clamp(0, 1)
    white = ((sharp - 0.55) / 0.15).clamp(0, 1)
    near = F.max_pool2d(white[None, None], 5, 1, 2)[0, 0]
    dark = ((0.35 - sharp) / 0.15).clamp(0, 1) * near
    alpha = torch.maximum(white, dark)
    value = torch.where(white >= dark, torch.ones_like(sharp), torch.zeros_like(sharp))
    return fixed, (value, alpha)


def _annotate(
    path: Path,
    info: object,
    meas: Measured,
    t: float,
    grid: Grid,
    cells: Tensor,
    fixed: Tensor,
    fields: list[Field],
    groups: Tensor | None = None,
) -> NDArray[np.uint8]:
    """The reference frame with a box per layout cell, coloured by kind, a label per
    field and an outline around each multi-glyph picture: what was found, to show on the
    clips page."""
    frame = next(iter(read_frames(path, info, meas.crop, start=t, duration=0.05)), None)  # type: ignore[arg-type]
    if frame is None:
        return np.zeros((1, 1, 3), np.uint8)
    h, w = frame.y.shape
    u = cv2.resize(frame.u, (w, h))
    v = cv2.resize(frame.v, (w, h))
    bgr = cv2.cvtColor(np.dstack([frame.y, v, u]), cv2.COLOR_YCrCb2BGR)
    big = cv2.resize(bgr, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    kind: dict[int, str] = {}
    for i in cells.nonzero()[:, 0].tolist():
        g = int(fixed[i])
        kind[i] = "separator" if g in (ord(":"), ord(".")) else ("fixed" if g != -1 else "live")
    for fl in fields:
        for c in fl.cols:
            kind[fl.row * COLS + c] = fl.kind
        for c in fl.seps:
            kind[fl.row * COLS + c] = "separator"
    xs = grid.col_x(torch.arange(COLS + 1, dtype=torch.float64)).tolist()
    for i, k in kind.items():
        r, c = divmod(i, COLS)
        x0, x1 = int(xs[c] * 2), int(xs[c + 1] * 2)
        y0, y1 = int((grid.y0 + r * grid.cell_h) * 2), int((grid.y0 + (r + 1) * grid.cell_h) * 2)
        cv2.rectangle(big, (x0, y0), (x1 - 1, y1 - 1), KIND_COLOURS.get(k, (255, 255, 255)), 2)
    if groups is not None:
        for gid in range(int(groups.max()) + 1):
            rc = [divmod(i, COLS) for i in (groups == gid).nonzero()[:, 0].tolist()]
            r0, r1 = min(r for r, _ in rc), max(r for r, _ in rc) + 1
            c0, c1 = min(c for _, c in rc), max(c for _, c in rc) + 1
            p0 = (int(xs[c0] * 2) - 4, int((grid.y0 + r0 * grid.cell_h) * 2) - 4)
            p1 = (int(xs[c1] * 2) + 3, int((grid.y0 + r1 * grid.cell_h) * 2) + 3)
            cv2.rectangle(big, p0, p1, KIND_COLOURS["group"], 3)
    for fl in fields:
        c = fl.cols[0]
        x, y = int(xs[c] * 2), int((grid.y0 + fl.row * grid.cell_h) * 2) - 6
        cv2.putText(big, fl.kind, (x, max(y, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(
            big, fl.kind, (x, max(y, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, KIND_COLOURS[fl.kind], 2
        )
    legend = "green fixed | yellow timer | cyan held | magenta live | grey sep. | orange picture"
    cv2.putText(big, legend, (10, big.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
    cv2.putText(
        big, legend, (10, big.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
    )
    return np.asarray(cv2.cvtColor(big, cv2.COLOR_BGR2RGB), np.uint8)


__all__ = ["OsdLayout", "analyze_clip", "find_grid_plain"]
