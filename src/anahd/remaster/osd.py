"""Experimental: redraw the OSD crisply from the pilot's Betaflight font.

The analog OSD is a grid of 30 columns x 16 (PAL) or 13 (NTSC) rows of 12x18-pixel glyphs
(see :mod:`osdfont`), laid over the camera picture by the VTX and then blurred, smeared and
coloured by the link. Here:

1. **Grid** (once per clip): the cell size and offset that make the static OSD (the median
   of the measurement's samples) read best as font glyphs.
2. **Reading** (each frame): every cell is compared with all 256 glyphs by normalised
   correlation over the glyph's opaque pixels, on the frame resampled to twice the font's
   raster. A cell keeps showing its glyph until a new one has been read in ``STABLE``
   frames in a row (blinking warnings blink; noise does not flicker the text). Cells read
   poorly are left as they are (the original pixels pass through).
3. **Drawing**: under each read glyph the old, blurred text is filled from the surrounding
   picture, then the glyph is drawn white with its black outline, as the font has it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from torch import Tensor

from anahd.media import Crop, VideoInfo

from .osdfields import (
    SLOW_HOLD,
    Field,
    _seconds,
    find_fields,
    survey,
    timer_glyphs,
    timer_value,
)

if TYPE_CHECKING:
    from .osdlive import Timeline
from .osdfont import BLACK, CLEAR, GLYPH_H, GLYPH_W, WHITE, read_mcm

COLS = 30
ROWS = {"PAL": 16, "NTSC": 13}
UP = 2  # reading resolution: the font raster doubled
STABLE = 3  # frames a new reading must hold before it is shown
TEMPLATE = -2  # a fixed cell drawn from the clip's own picture of it (no font given)
STATIC = 0.95  # share of a clip's confident readings one glyph needs for a cell to be fixed
PRESENT = 0.4  # score at which a fixed cell's own glyph counts as showing in a frame
SHOWING = 0.3  # share of fixed cells showing for the OSD to count as on
READ = 0.65  # score a glyph needs to be read at all (hand-checked on real clips)
VALUE_CHARS = "0123456789.:%-"
VOLT = 0x06  # Betaflight's volt symbol
SURE = 0.85  # live cells: a reading this good (and clearly ahead) is drawn in its own frame
FRINGE = 0.12  # colour spread (max-min of RGB) from which a pixel counts as a colour fringe
LEAD = 0.1  # ...when it leads the runner-up glyph by this much
LOST = 0.5  # a frame reading under this share of the recent average is not redrawn
BLANK = 0  # glyph 0 is blank in Betaflight fonts (and in the reference font)
UNEXPLAINED = 0.6  # score lost for bright strokes a glyph does not cover (share of them)
CONFIDENT = 0.75  # a cell read at least this well counts towards finding the grid


@dataclass(frozen=True)
class Grid:
    """Cell size and the top-left corner of cell (0, 0), in picture pixels. ``bend``:
    analog line timing stretches the picture unevenly along a line, so column ``c``
    (continuous) starts at x0 + c * cell_w + bend * c * (c - COLS)."""

    cell_w: float
    cell_h: float
    x0: float
    y0: float
    rows: int
    score: float = 0.0
    bend: float = 0.0

    def cells(self) -> tuple[int, int]:
        return self.rows, COLS

    def col_x(self, c: Tensor) -> Tensor:
        """Picture x of continuous column position ``c``."""
        return self.x0 + c * self.cell_w + self.bend * c * (c - COLS)

    def x_col(self, x: Tensor) -> Tensor:
        """Continuous column position at picture x (the inverse of :meth:`col_x`)."""
        if abs(self.bend) < 1e-9:
            return (x - self.x0) / self.cell_w
        a, b = self.bend, self.cell_w - self.bend * COLS
        return (-b + torch.sqrt((b * b + 4 * a * (x - self.x0)).clamp_min(0))) / (2 * a)

    def scaled(self, sx: float, sy: float) -> Grid:
        return Grid(
            self.cell_w * sx, self.cell_h * sy, self.x0 * sx, self.y0 * sy, self.rows,
            self.score, self.bend * sx,
        )  # fmt: skip

    def with_(self, **kw: float) -> Grid:
        from dataclasses import replace

        return replace(self, **kw)  # type: ignore[arg-type]


class Font:
    """The glyphs as templates for reading and as RGBA tiles for drawing."""

    def __init__(self, path: Path, device: torch.device) -> None:
        g = read_mcm(path)
        self.glyphs = g
        self.device = device
        t = torch.from_numpy(g).to(device)
        opaque = (t != CLEAR).float()
        value = (t == WHITE).float()
        # Reading templates at UP x the raster, softened like the analog picture.
        big = F.interpolate(value[:, None], scale_factor=UP, mode="nearest")
        mask = F.interpolate(opaque[:, None], scale_factor=UP, mode="nearest")
        soft = _blur(big, 0.8 * UP) * mask
        self.mask = mask.flatten(1)  # (256, P)
        self.tmpl = soft.flatten(1)
        n = self.mask.sum(1).clamp_min(1)
        self.count = n
        mean = self.tmpl.sum(1) / n
        centred = (self.tmpl - mean[:, None]) * self.mask
        self.centred = centred
        self.norm = centred.pow(2).sum(1).sqrt().clamp_min(1e-6)
        # Glyphs that are almost all one value cannot be told apart by correlation.
        # Readable: enough opaque pixels with contrast, and not near-solid white (logo tiles
        # and bar segments, which match any bright patch).
        white = (value.flatten(1) * opaque.flatten(1)).sum(1) / opaque.flatten(1).sum(1).clamp_min(
            1
        )
        self.readable = (centred.abs().sum(1) / n > 0.1) & (n >= 8) & (white < 0.6)
        self.value, self.opaque = value, opaque
        # Where each glyph draws anything (grown by a pixel): bright strokes outside it are
        # left unexplained by that glyph (a ':' is not a '.').
        self.covers = (F.max_pool2d(mask, 3, 1, 1) > 0).float().flatten(1)

    def match(self, cells: Tensor, codes: Tensor) -> Tensor:
        """cells: (N, P), codes: (N,) -> each cell's correlation with its own glyph, from
        the whole font (``readable`` does not apply: a sprite tile may be near-solid)."""
        m = self.mask[codes]
        n = self.count[codes]
        s1 = (cells * m).sum(1)
        var = ((cells * cells * m).sum(1) - s1 * s1 / n).clamp_min(1e-6)
        return (cells * self.centred[codes]).sum(1) / (var.sqrt() * self.norm[codes])

    def read_all(self, cells: Tensor) -> tuple[Tensor, Tensor]:
        """Like :meth:`read`, over every glyph that draws anything (``readable`` aside)."""
        keep = self.readable
        self.readable = self.mask.sum(1) > 0
        try:
            return self.read(cells)
        finally:
            self.readable = keep

    def limit(self, glyphs: set[int]) -> None:
        """Read only these glyphs from now on."""
        keep = torch.zeros_like(self.readable)
        keep[list(glyphs)] = True
        self.readable = self.readable & keep

    def read(self, cells: Tensor) -> tuple[Tensor, Tensor]:
        """cells: (N, P) luma 0..1 -> best glyph index and its correlation, per cell."""
        m = self.mask  # (G, P)
        s1 = cells @ m.T  # sum of x over each glyph's mask: (N, G)
        s2 = (cells * cells) @ m.T
        mean = s1 / self.count
        var = (s2 - s1 * mean).clamp_min(1e-6)
        cov = cells @ self.centred.T
        ncc = cov / (var.sqrt() * self.norm)
        bright = ((cells - 0.7) / 0.2).clamp(0, 1)
        total = bright.sum(1, keepdim=True)
        unexplained = (total - bright @ self.covers.T) / total.clamp_min(1.0)
        score = ncc - UNEXPLAINED * unexplained
        score = torch.where(self.readable[None], score, torch.full_like(score, -1.0))
        self.scores = score
        top = score.topk(2, dim=1)
        self.margin = top.values[:, 0] - top.values[:, 1]  # lead over the runner-up
        return top.indices[:, 0], top.values[:, 0]


def _blur(x: Tensor, sigma: float) -> Tensor:
    r = max(1, round(3 * sigma))
    t = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    k = k / k.sum()
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1))
    return F.conv2d(F.pad(x, (0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1))


def sample_cells(y: Tensor, grid: Grid) -> Tensor:
    """The grid's cells of luma ``y`` (H, W, 0..1) resampled to UP x the font raster:
    (rows*COLS, P)."""
    h, w = y.shape
    rows = grid.rows
    rh, rw = rows * GLYPH_H * UP, COLS * GLYPH_W * UP
    xs = grid.col_x((torch.arange(rw, device=y.device) + 0.5) / (GLYPH_W * UP))
    ys = grid.y0 + (torch.arange(rh, device=y.device) + 0.5) * grid.cell_h / (GLYPH_H * UP)
    gx = (xs - 0.5) / (w - 1) * 2 - 1
    gy = (ys - 0.5) / (h - 1) * 2 - 1
    g = torch.stack(torch.meshgrid(gy, gx, indexing="ij")[::-1], -1)[None]
    raster = F.grid_sample(
        y[None, None], g, mode="bilinear", padding_mode="border", align_corners=True
    )
    cells = raster.view(rows, GLYPH_H * UP, COLS, GLYPH_W * UP).permute(0, 2, 1, 3)
    return cells.reshape(rows * COLS, -1)


COARSE = 4.0  # px between offsets in the first pass of the grid search
KEEP = 12  # best coarse candidates refined at 2 px


def find_grid(still: Tensor, osd: Tensor, font: Font) -> Grid | None:
    """The grid that reads the static OSD best. ``still``: (H, W) luma 0..1 where only
    the OSD stays sharp (the median of samples); ``osd``: the OSD mask at that size.

    The OSD chip draws on the analog raster, so the cell pitch is fixed per recording:
    it is measured from the text itself (the autocorrelation of the bright strokes along
    each line of text, and down each column), then the offset is the one at which the
    cells read best as glyphs, refined to a fraction of a pixel."""
    h, w = still.shape
    # OSD strokes: bright pixels, and bright pixels with the black outline close by
    # (scenery behind the OSD may be bright, but is not outlined). Pitch candidates from
    # both, plus the standard PAL/NTSC cells; the glyph matching chooses.
    darkest = -F.max_pool2d(-still[None, None], 5, 1, 2)[0, 0]
    bright = ((still > 0.6) & osd).float()
    outlined = bright * (still - darkest > 0.3).float()
    xs = [w / 30, w / 32]
    ys = [h / 13, h / 16]
    for s_ in (bright, outlined):
        xs += _pitches(s_, 24.0, 40.0)
        ys += _pitches(s_.T.contiguous(), 40.0, 64.0)
    # Coarse to fine: every pitch pair at 6 px offset steps, then the best few at 2 px
    # steps around their offset (the same search as a full 2 px sweep, ~6x faster).
    coarse: list[Grid] = []
    for pitch_x in _distinct(xs):
        for pitch_y in _distinct(ys):
            rows = min(16, max(13, round(h / pitch_y)))
            for oy in np.arange(0.0, pitch_y, COARSE):
                for ox in np.arange(0.0, pitch_x, COARSE):
                    x0 = ox - pitch_x * round((ox - (w - COLS * pitch_x) / 2) / pitch_x)
                    y0 = oy - pitch_y * round((oy - (h - rows * pitch_y) / 2) / pitch_y)
                    g = Grid(pitch_x, pitch_y, x0, y0, rows)
                    coarse.append(g.with_(score=_score(still, osd, g, font)))
    if not coarse:
        return None
    best = max(coarse, key=lambda g: g.score)
    for g0 in sorted(coarse, key=lambda g: -g.score)[:KEEP]:
        for dy in np.arange(-COARSE + 2.0, COARSE - 1.0, 2.0):
            for dx in np.arange(-COARSE + 2.0, COARSE - 1.0, 2.0):
                g = g0.with_(x0=g0.x0 + float(dx), y0=g0.y0 + float(dy))
                score = _score(still, osd, g, font)
                if score > best.score:
                    best = g.with_(score=score)
    best = _bend(still, osd, best, font)
    occupied = _occupied(osd, best)
    _, ncc = font.read(sample_cells(still, best)[occupied])
    sure = occupied.nonzero()[:, 0][ncc >= CONFIDENT]
    if len(sure) < 3:
        return None
    return _refine(still, best, sure, font)


def _distinct(values: list[float], apart: float = 0.6) -> list[float]:
    out: list[float] = []
    for v in values:
        if all(abs(v - o) >= apart for o in out):
            out.append(v)
    return out


def _pitches(strokes: Tensor, lo: float, hi: float, k: int = 3) -> list[float]:
    """Candidate repeat distances (pixels, sub-pixel) of the strokes along the rows of
    ``strokes``: the ``k`` highest local peaks of their summed, normalised autocorrelation
    between lo and hi (sparse text can put a glyph's own structure above the pitch, so the
    glyph matching chooses among them)."""
    rows = strokes[strokes.sum(1) > 6]
    if len(rows) < 3:
        return []
    rows = rows - rows.mean(1, keepdim=True)
    n = rows.shape[1]
    f = torch.fft.rfft(rows, 2 * n)
    ac = torch.fft.irfft(f * f.conj(), 2 * n)[:, :n]
    ac = (ac / ac[:, :1].clamp_min(1e-9)).sum(0).cpu().numpy()
    a, b = max(1, round(lo)), min(n - 2, round(hi))
    peaks = [i for i in range(a, b + 1) if ac[i] >= ac[i - 1] and ac[i] >= ac[i + 1]]
    peaks.sort(key=lambda i: -ac[i])
    out = []
    for i in peaks[:k]:
        before, at, after = ac[i - 1], ac[i], ac[i + 1]
        den = before - 2 * at + after
        step = 0.5 * (before - after) / den if den != 0 else 0.0
        out.append(float(i + max(-0.5, min(0.5, step))))
    return out


def _bend(still: Tensor, osd: Tensor, g: Grid, font: Font) -> Grid:
    """The bend (and a nudge of the offset) under which the most OSD cells read with
    confidence: text at both edges of the picture, not just where the offset was found."""
    occupied = _occupied(osd, g)

    def count(c: Grid) -> tuple[int, float]:
        _, ncc = font.read(sample_cells(still, c)[occupied])
        return int((ncc >= CONFIDENT).sum()), float(ncc.mean())

    best, key = g, count(g)
    # Coarse (twice the steps), then the original steps around the coarse best.
    for dw in np.arange(-0.6, 0.601, 0.2):
        for bend in np.arange(-0.04, 0.0401, 0.01):
            for dx in np.arange(-12.0, 12.01, 2.0):
                c = g.with_(cell_w=g.cell_w + float(dw), bend=float(bend), x0=g.x0 + float(dx))
                k = count(c)
                if k > key:
                    best, key = c, k
    g = best
    for dw in (-0.1, 0.0, 0.1):
        for db in (-0.005, 0.0, 0.005):
            for dx in (-1.0, 0.0, 1.0):
                c = g.with_(cell_w=g.cell_w + dw, bend=g.bend + db, x0=g.x0 + dx)
                k = count(c)
                if k > key:
                    best, key = c, k
    g = best
    for dh in np.arange(-1.0, 1.01, 0.25):
        for dy in np.arange(-6.0, 6.01, 1.0):
            c = g.with_(cell_h=g.cell_h + float(dh), y0=g.y0 + float(dy))
            k = count(c)
            if k > key:
                best, key = c, k
    return best


def _refine(still: Tensor, g: Grid, cells: Tensor, font: Font) -> Grid:
    """Sub-pixel placement: the offset and cell size (x and y apart) at which the cells
    already read confidently match their glyphs best on average."""

    def fit(c: Grid) -> float:
        _, ncc = font.read(sample_cells(still, c)[cells])
        return float(ncc.mean())

    best, score = g, fit(g)
    for step in (0.5, 0.25, 0.1):
        moved = True
        while moved:
            moved = False
            for dx, dy, sw, sh, db in (
                (step, 0, 0, 0, 0), (-step, 0, 0, 0, 0), (0, step, 0, 0, 0), (0, -step, 0, 0, 0),
                (0, 0, step / 30, 0, 0), (0, 0, -step / 30, 0, 0),
                (0, 0, 0, step / 16, 0), (0, 0, 0, -step / 16, 0),
                (0, 0, 0, 0, step / 225), (0, 0, 0, 0, -step / 225),
            ):  # fmt: skip
                c = best.with_(
                    cell_w=best.cell_w + sw, cell_h=best.cell_h + sh,
                    x0=best.x0 + dx, y0=best.y0 + dy, bend=best.bend + db,
                )  # fmt: skip
                s = fit(c)
                if s > score + 1e-4:
                    best, score, moved = c, s, True
    return _place(still, best.with_(score=score), cells, font)


def _place(still: Tensor, g: Grid, cells: Tensor, font: Font) -> Grid:
    """Final placement in the picture itself: the confidently read glyphs drawn at the
    picture's size (softened like the analog picture) against the still picture, over
    their opaque pixels; offsets and sizes refined to 0.05 px."""
    idx, _ = font.read(sample_cells(still, g)[cells])
    shown = torch.full((g.rows * COLS,), -1, dtype=torch.long, device=still.device)
    shown[cells] = idx
    h, w = still.shape

    def fit(c: Grid) -> float:
        value, alpha, _ = _layer(font, shown, c, (w, h))
        value = _blur(value[None, None], 1.0)[0, 0]
        m = alpha > 0.5
        a, b = value[m], still[m]
        a, b = a - a.mean(), b - b.mean()
        return float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-6))

    best, score = g, fit(g)
    for step in (0.4, 0.2, 0.1, 0.05):
        moved = True
        while moved:
            moved = False
            for dx, dy, sw, sh, db in (
                (step, 0, 0, 0, 0), (-step, 0, 0, 0, 0), (0, step, 0, 0, 0), (0, -step, 0, 0, 0),
                (0, 0, step / 30, 0, 0), (0, 0, -step / 30, 0, 0),
                (0, 0, 0, step / 16, 0), (0, 0, 0, -step / 16, 0),
                (0, 0, 0, 0, step / 225), (0, 0, 0, 0, -step / 225),
            ):  # fmt: skip
                c = best.with_(
                    cell_w=best.cell_w + sw, cell_h=best.cell_h + sh,
                    x0=best.x0 + dx, y0=best.y0 + dy, bend=best.bend + db,
                )  # fmt: skip
                s = fit(c)
                if s > score + 1e-5:
                    best, score, moved = c, s, True
    return best.with_(score=score)


def _layer(
    font: Font, shown: Tensor, g: Grid, size: tuple[int, int], ss: int = 1
) -> tuple[Tensor, Tensor, Tensor]:
    """The font raster for ``shown`` (glyph per cell, -1 = none) placed on grid ``g`` at
    ``size`` (w, h): white value, opacity, and the cells it covers."""
    w, h = size
    rows = g.rows
    known = shown >= 0
    idx = shown.clamp_min(0)
    val = font.value[idx] * known[:, None, None]
    opa = font.opaque[idx] * known[:, None, None]

    def raster(x: Tensor) -> Tensor:
        x = x.view(rows, COLS, GLYPH_H, GLYPH_W).permute(0, 2, 1, 3)
        return x.reshape(rows * GLYPH_H, COLS * GLYPH_W)

    cover = known.float().view(rows, COLS).repeat_interleave(GLYPH_H, 0)
    cover = cover.repeat_interleave(GLYPH_W, 1)
    stack = torch.stack((raster(val), raster(opa), cover))[None]
    # ss > 1: crisp pixels with box antialiasing (nearest on an ss-times finer grid, then
    # averaged); ss = 1: bilinear (soft, for fitting against the analog picture).
    dev = stack.device
    ys = (torch.arange(h * ss, device=dev) + 0.5) / ss
    xs = (torch.arange(w * ss, device=dev) + 0.5) / ss
    gy = ((ys - g.y0) / g.cell_h * GLYPH_H) / (rows * GLYPH_H) * 2 - 1
    gx = g.x_col(xs) / COLS * 2 - 1
    grid = torch.stack(torch.meshgrid(gy, gx, indexing="ij")[::-1], -1)[None]
    mode = "nearest" if ss > 1 else "bilinear"
    out = F.grid_sample(stack, grid, mode=mode, padding_mode="zeros", align_corners=False)
    if ss > 1:
        out = F.avg_pool2d(out, ss)
    out = out[0]
    return out[0], out[1], out[2]


def _score(still: Tensor, osd: Tensor, g: Grid, font: Font) -> float:
    """How well the OSD reads on grid ``g``: the summed confidence of its cells (blank
    cells inside the OSD's boxes neither help nor hurt)."""
    occupied = _occupied(osd, g)
    if int(occupied.sum()) < 3:
        return -1.0
    _, ncc = font.read(sample_cells(still, g)[occupied])
    return float((ncc - 0.5).clamp_min(0).mean())


def _occupied(osd: Tensor, g: Grid) -> Tensor:
    """Cells whose middle third holds OSD pixels."""
    share = sample_cells(osd.float(), g).view(g.rows * COLS, GLYPH_H * UP, GLYPH_W * UP)
    mid = share[
        :, GLYPH_H * UP // 3 : 2 * GLYPH_H * UP // 3, GLYPH_W * UP // 4 : 3 * GLYPH_W * UP // 4
    ]
    return mid.mean((1, 2)) > 0.3


class Redraw:
    """Per clip: reads each frame's cells and draws the font over the picture."""

    def __init__(
        self,
        font: Font | None,
        grid: Grid,
        size: tuple[int, int],
        src: tuple[int, int],
        osd: Tensor | None = None,
        fixed: Tensor | None = None,
        fields: list[Field] | None = None,
        fps: float = 60.0,
        t0: float = 0.0,
        layout: Layout | None = None,
        template: tuple[Tensor, Tensor] | None = None,
        groups: Tensor | None = None,
        device: torch.device | None = None,
        draw_live: bool = True,
        vtx: Tensor | None = None,
    ) -> None:
        self.font, self.grid = font, grid
        # Without a font: the fixed cells are drawn from ``template`` (white value and
        # opacity at the picture size, the clip's own sharpened median picture of them).
        self.template = template
        self.w, self.h = size
        sx, sy = size[0] / src[0], size[1] / src[1]
        self.out_grid = grid.scaled(sx, sy)
        n = grid.rows * COLS
        dev = font.device if font is not None else (device or torch.device("cuda"))
        self.shown = torch.full((n,), -1, dtype=torch.long, device=dev)  # -1: pass through
        self.cand = torch.full((n,), -1, dtype=torch.long, device=dev)
        self.held = torch.zeros(n, dtype=torch.long, device=dev)
        self._cache: tuple[Tensor, tuple[Tensor, Tensor, Tensor]] | None = None
        self.recent = 0.0  # running share of OSD cells read
        self.lost = False  # this frame reads too poorly to redraw its changing cells
        self.present = True  # the OSD is on in this frame
        # Cells whose glyph never changes in this recording (glyph index, -1 = changing).
        self.fixed = fixed if fixed is not None else torch.full((n,), -1, device=dev)
        self.is_template = self.fixed == TEMPLATE
        self.is_fixed = (self.fixed >= 0) | self.is_template
        # Fields and their kinds (osdfields): timers are generated from the clock, slow
        # fields need half a second before a new value shows, live ones STABLE frames.
        self.fields = fields or []
        self.fps, self.t0, self.k = fps, t0, 0
        self._last_read: dict[int, int | None] = {}
        self._fixed_cells = (self.fixed >= 0).nonzero()[:, 0]  # glyph cells (a font)
        self._timer_cells = torch.tensor(
            [fl.row * COLS + c for fl in self.fields if fl.kind == "timer" for c in fl.cols],
            dtype=torch.long,
            device=dev,
        )
        self.need = torch.full((n,), STABLE, dtype=torch.long, device=dev)
        self.sep = torch.zeros(n, dtype=torch.bool, device=dev)  # always left as recorded
        self.live = ~self.is_fixed  # cells read frame by frame (not fixed, slow or timer)
        self.timeline: Timeline | None = None  # osdlive segment reading, set by the caller
        self.timeline_start = 0  # the frame (as counted by seek) of the timeline's first
        # Constant-only (the default): moving numbers are never redrawn,
        # only cleaned in place; held values (VTX power) and verified timers are.
        self.draw_live = draw_live
        # Per timer field (index in fields): verified exact over the clip (osdlayout
        # .read_live); None = not checked, and an unchecked timer is not drawn either.
        self.timer_ok: dict[int, bool] | None = None
        self.timer_check: dict[int, list[int]] = {}  # field index -> [matches, mismatches]
        self.vtx = vtx if vtx is not None else torch.zeros(n, dtype=torch.bool, device=dev)
        self._live_cells = torch.zeros(n, dtype=torch.bool, device=dev)
        for fl in self.fields:
            if fl.kind == "live":
                for c in fl.cols:
                    self._live_cells[fl.row * COLS + c] = True
        # With a timeline every digit the clip can change is drawn from it: numbers,
        # timers and lone digits that were constant in the analysis (VTX channel) alike.
        digit = torch.zeros(256, dtype=torch.bool, device=dev)
        digit[[ord(c) for c in "0123456789"]] = True
        self._fixed_digits = (self.fixed >= 0) & digit[self.fixed.clamp(0, 255)]
        self._timer_cells_mask = torch.zeros(n, dtype=torch.bool, device=dev)
        for fl in self.fields:
            if fl.kind == "timer":
                for c in fl.cols:
                    self._timer_cells_mask[fl.row * COLS + c] = True
        self._is_number = torch.zeros(n, dtype=torch.bool, device=dev)
        for fl in self.fields:
            if fl.kind != "timer":
                for c in fl.cols:
                    self._is_number[fl.row * COLS + c] = True
        self.value_glyph = torch.zeros(256, dtype=torch.bool, device=dev)
        self.value_glyph[[ord(c) for c in VALUE_CHARS] + [VOLT]] = True
        if not fields and not bool((self.fixed >= 0).any()):
            self.value_glyph[:] = True  # no layout known: any glyph may be read anywhere
        for fl in self.fields:
            if fl.kind in ("slow", "timer"):
                for c in fl.cols:
                    self.live[fl.row * COLS + c] = False
            for c in fl.seps:
                self.sep[fl.row * COLS + c] = True
            if fl.kind == "slow":
                for c in fl.cols:
                    self.need[fl.row * COLS + c] = max(STABLE, round(SLOW_HOLD * fps))
        # Cells overlapping the OSD mask (at the picture size); all cells when none given.
        if osd is None:
            self.inside = torch.ones(n, dtype=torch.bool, device=dev)
        else:
            share = sample_cells(osd.float(), grid).mean(1)
            self.inside = share > 0.05
        self.layout = layout
        if layout is not None:  # only the cells of the reference frame's layout, ever
            self.inside = layout.cells.clone()  # the layout is the authority
        self.seen_recently = torch.zeros(3, n, dtype=torch.bool, device=dev)
        self.own_recently = torch.zeros(3, n, dtype=torch.bool, device=dev)
        # Pictures made of several glyphs (sprites, the craft name, icons): a group is
        # drawn or left as recorded as a whole, never half of it (osdlayout.static_groups).
        self.groups = groups if groups is not None else torch.full((n,), -1, device=dev)
        self.in_group = self.groups >= 0
        self.n_groups = int(self.groups.max()) + 1 if bool(self.in_group.any()) else 0
        # Per group, the other cells of its row run (an element's numbers beside its
        # labels): when the segment pass draws one of them, the labels are on too, so a
        # torn row does not end up half redrawn.
        self._group_runs: list[Tensor] = []
        for gid in range(self.n_groups):
            run: list[int] = []
            for i in (self.groups == gid).nonzero()[:, 0].tolist():
                r, c = divmod(i, COLS)
                lo = hi = c
                while lo > 0 and bool(self.inside[r * COLS + lo - 1]):
                    lo -= 1
                while hi < COLS - 1 and bool(self.inside[r * COLS + hi + 1]):
                    hi += 1
                run += [r * COLS + k for k in range(lo, hi + 1)]
            self._group_runs.append(torch.tensor(sorted(set(run)), dtype=torch.long, device=dev))

    def reset(self) -> None:
        """Forget what was read (a new stretch of frames; :meth:`seek` sets its time)."""
        self.shown.fill_(-1)
        self.cand.fill_(-1)
        self.held.zero_()
        self.recent, self.lost = 0.0, False

    def update(self, y: Tensor) -> Tensor:
        """Read the cells of this output frame's luma (H, W, 0..1); returns what is shown."""
        cells = sample_cells(y, self.out_grid)
        if self.font is None:  # the fixed cells' pictures only, where the OSD shows them
            self.seen_recently = torch.roll(self.seen_recently, 1, 0)
            self.seen_recently[0] = has_osd(cells)
            keep = self.seen_recently.any(0) & self.inside & self.is_template
            self._keep = keep
            self.k += 1
            return torch.where(keep, self.fixed, torch.full_like(self.fixed, -1))
        idx, ncc = self.font.read(cells)
        sure = ncc >= READ
        bright = cells.max(1).values > 0.55
        reading = torch.where(sure, idx, torch.where(bright, -1, BLANK))
        # Only where AnaHD found the OSD: the scene never turns into glyphs. Cells that
        # change show only what values are written with (digits, '.', ':', '%', '-', the
        # volt symbol); the craft name, sprites and icons are fixed cells, never read here.
        valid = self.inside & (self.is_fixed | self.value_glyph[idx.clamp_min(0)])
        reading = torch.where(valid, reading, torch.full_like(reading, -1))
        idx = torch.where(valid, idx, torch.full_like(idx, -1))
        # What the host needs this frame, fetched in one transfer: the share of OSD cells
        # read, the share of fixed cells showing their glyph, and the timers' cells.
        stats = [(sure & self.inside).float().sum() / self.inside.float().sum().clamp_min(1)]
        fx = self._fixed_cells
        own_fixed = (
            self.font.match(cells[fx], self.fixed[fx]) >= PRESENT if fx.numel() else None
        )  # each fixed cell showing its own glyph (icons and ':' included: any glyph)
        if own_fixed is not None:
            stats.append(own_fixed.float().mean())
        else:
            stats.append(torch.ones((), device=idx.device))
        host = torch.cat(
            [
                torch.stack(stats),
                idx[self._timer_cells].float(),
                (ncc[self._timer_cells] >= SURE).float(),
            ]
        ).tolist()
        share, showing = host[0], host[1]
        n = self._timer_cells.numel()
        timer_idx, timer_sure = host[2 : 2 + n], host[2 + n :]
        # A breakup band or a torn frame reads far less than the frames around it: its
        # changing cells are passed through. Signal loss or disarm: the whole OSD is off.
        self.lost = self.recent > 0 and share < LOST * self.recent
        self.recent = share if self.recent == 0 else 0.9 * self.recent + 0.1 * share
        self.present = showing >= SHOWING
        same = reading == self.cand
        self.held = torch.where(same, self.held + 1, torch.ones_like(self.held))
        self.cand = reading
        switch = self.held >= self.need
        # A cell whose reading has moved off what is shown passes through until the new
        # reading has held: never an old value drawn crisply over a changing one. (Keeping
        # the last stable value instead, for up to 0.5 s, drew 4 wrong digits of 276 in
        # the hand check: the current draw changes faster than that.)
        moving = (reading != self.shown) & ~switch
        self.shown = torch.where(switch, self.cand, torch.where(moving, -1, self.shown))
        # Live values change from frame to frame: a reading this frame is sure of (high
        # score, well ahead of the runner-up) is drawn in this frame, never held over.
        instant = self.live & (idx >= 0) & (ncc >= SURE) & (self.font.margin >= LEAD)
        self.shown = torch.where(instant, idx, self.shown)
        at = self.k - self.timeline_start
        self._timeline_drawn: Tensor | None = None
        if self.timeline is not None and 0 <= at < len(self.timeline.values):
            # Numbers come from the segment reading (osdlive), never from this frame alone.
            v = self.timeline.values[at].to(self.shown.device).long()
            from_tl = self._is_number | self._fixed_digits | self._timer_cells_mask
            self.shown = torch.where(from_tl, v, self.shown)
            self._timeline_drawn = from_tl & (v >= 0)
        # The default redraws no number but the VTX line's: every
        # other digit, timers included, stays as recorded (faded).
        numbers = self._is_number | self._fixed_digits | self._timer_cells_mask
        not_drawn = numbers & ~self.vtx if not self.draw_live else torch.zeros_like(numbers)
        if not self.draw_live and self.timeline is None:
            not_drawn = numbers  # the VTX digits too: they are read per stretch, or not at all
        self.shown = torch.where(not_drawn, -1, self.shown)
        changing = torch.full_like(self.shown, -1) if self.lost else self.shown
        changing = torch.where(self.sep, torch.full_like(changing, -1), changing)
        use_fixed = self.fixed >= 0
        if self.timeline is not None or not self.draw_live:
            use_fixed = use_fixed & ~self._fixed_digits
        out = torch.where(use_fixed, self.fixed, changing)
        # Never draw where the recording shows nothing: a cell is drawn only while this
        # frame or one of the two before it has OSD strokes there.
        self.seen_recently = torch.roll(self.seen_recently, 1, 0)
        self.seen_recently[0] = has_osd(cells)
        # A fixed cell is drawn only while its own glyph shows (the OSD changes state:
        # disarmed, warnings, the stats screen): never carried into a state without it.
        own = torch.zeros_like(self.inside)
        if own_fixed is not None:
            own[fx] = own_fixed
        if self.n_groups:  # a group shows while at least half of its glyphs do
            gi = self.in_group.nonzero()[:, 0]
            hit = (self.font.match(cells[gi], self.fixed[gi]) >= PRESENT).float()
            gid = self.groups[gi]
            votes = torch.zeros(self.n_groups, device=hit.device).index_add_(0, gid, hit)
            size = torch.zeros(self.n_groups, device=hit.device).index_add_(0, gid, hit * 0 + 1)
            on = votes >= 0.5 * size
            if self._timeline_drawn is not None:
                for g, run in enumerate(self._group_runs):
                    if run.numel() and bool((self._timeline_drawn[run]).any()):
                        on[g] = True
            own[gi] = on[gid]
        self.own_recently = torch.roll(self.own_recently, 1, 0)
        self.own_recently[0] = own
        keep = (self.seen_recently.any(0) | self.in_group) & self.inside
        keep = keep & (~self.is_fixed | self.own_recently.any(0))
        out = torch.where(keep, out, torch.full_like(out, -1))
        self._keep = keep
        t = self.t0 + self.k / self.fps
        self.k += 1
        at = 0
        for j, fl in enumerate(self.fields):
            if fl.kind != "timer":
                continue
            k = len(fl.cols)
            got = timer_idx[at : at + k]
            ok = all(v > 0 for v in timer_sure[at : at + k]) and all(g >= 0 for g in got)
            at += k
            self._resync(fl, _seconds("".join(chr(int(g)) for g in got)) if ok else None, t)
            value = timer_value(fl, t)
            if value is None:
                continue
            if ok:  # every readable tick against the model (a frame of slack either way)
                near = [timer_value(fl, t + d / self.fps) for d in (-1, 0, 1)]
                want = {tuple(timer_glyphs(fl, x)) for x in near if x is not None}
                check = self.timer_check.setdefault(j, [0, 0])
                check[0 if tuple(int(g) for g in got) in want else 1] += 1
            if not self.draw_live or not (self.timer_ok or {}).get(j, True):
                for c in fl.cols:
                    out[fl.row * COLS + c] = -1
                continue
            if self.timeline is not None:
                continue  # a verified timer follows the recording's own ticks (the timeline)
            for c, g in zip(fl.cols, timer_glyphs(fl, value), strict=True):
                out[fl.row * COLS + c] = g
        # Timers too: only where the recording shows them right now.
        return torch.where(self._keep, out, torch.full_like(out, -1))

    def _resync(self, fl: Field, value: int | None, t: float) -> None:
        """A recording can drop or repeat frames, and the timer ticks on real time: when
        the field is seen to tick (read with confidence, one more than in the frame
        before) at a different frame than the model says, the model is re-anchored there."""
        previous = self._last_read.get(id(fl))
        self._last_read[id(fl)] = value
        if value is None or previous is None or value != previous + 1:
            return
        if timer_value(fl, t) == value and timer_value(fl, t - 1 / self.fps) == previous:
            return  # the model ticks on this very frame already
        for seg in reversed(fl.segments):
            if seg.t <= t + 1e-6:
                if seg.running:
                    seg.offset = value - t + 0.5 / self.fps
                break

    def seek(self, frame: int) -> None:
        """The next frame given to :meth:`update` is ``frame`` frames after ``t0``."""
        self.k = frame

    def layer(self, shown: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """The font for ``shown`` at the output size: white value, opacity and the area of
        the cells that were read (glyph or blank)."""
        if self._cache is None or not torch.equal(self._cache[0], shown):
            self._cache = (shown.clone(), self._compose(shown))
        return self._cache[1]

    def _compose(self, shown: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        size = (self.w, self.h)
        if self.font is not None:
            glyphs = torch.where(shown >= 0, shown, torch.full_like(shown, -1))
            value, alpha, cover = _layer(self.font, glyphs, self.out_grid, size, 3)
        else:
            value = alpha = cover = torch.zeros(self.h, self.w, device=shown.device)
        if self.template is not None and bool((shown == TEMPLATE).any()):
            cells = _cell_cover(shown == TEMPLATE, self.out_grid, size)
            tv, ta = (
                F.interpolate(t[None, None], size=(self.h, self.w), mode="bilinear")[0, 0]
                for t in self.template
            )
            ta = ta * cells
            value = value * (1 - ta) + tv * ta
            alpha = torch.maximum(alpha, ta)
            cover = torch.maximum(cover, cells)
        return value, alpha, cover

    def draw(self, rgb: Tensor, y: Tensor, boxes: Tensor) -> Tensor:
        """``rgb`` (3, H, W) with the font drawn over the redrawn cells. Nothing is wiped or
        synthesised underneath (a filled background shows as a blotch):
        the recorded pixels stay, with the old text's colour fringes taken out (grey at
        their own luma), and the glyph, outline included, covers its own footprint. Number
        cells left as recorded get the same fringe clean-up and a softened ring."""
        shown = self.update(y)
        del boxes
        if not self.present:
            return rgb
        value, alpha, cover = self.layer(shown)
        lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
        bright = (lum > 0.6).float()
        near = F.max_pool2d(bright[None, None], 7, 1, 3)[0, 0]  # the old text and its halo
        # Only the coloured pixels there are fringes: a bright plain sky behind the text is
        # left alone (greying all of it showed as a grey box).
        chroma = rgb.max(0).values - rgb.min(0).values
        fringe = near * ((chroma - FRINGE) / FRINGE).clamp(0, 1)
        grey = lum.expand_as(rgb)
        # Redrawn cells: fringes out around the old strokes, the glyph on top.
        m = (cover > 0.5).float() * fringe
        out = rgb * (1 - m) + grey * m
        out = out * (1 - alpha) + value[None] * alpha
        # Number cells left as recorded ("faded"): fringes out and the ringing beside the
        # strokes softened; the digit's shape is the recording's, so it can never read wrong.
        faded = self._is_number | self._live_cells | self._timer_cells_mask | self._fixed_digits
        unsure = faded & self._keep & (shown < 0)
        if bool(unsure.any()):
            m = _cell_cover(unsure, self.out_grid, (self.w, self.h)) * fringe
            soft = _blur(grey[None, :1], 0.6)[0].expand_as(out)
            ring = (near - bright).clamp(0, 1) * m  # just beside the strokes
            cleaned = grey * (1 - ring) + soft * ring
            out = out * (1 - m) + cleaned * m
        return out.clamp(0, 1)


def _cell_cover(cells: Tensor, g: Grid, size: tuple[int, int]) -> Tensor:
    """(H, W) 1 inside the grid cells marked in ``cells`` (rows*COLS bool), at ``size``."""
    w, h = size
    rows = g.rows
    m = cells.float().view(rows, COLS)
    m = m.repeat_interleave(GLYPH_H, 0).repeat_interleave(GLYPH_W, 1)[None, None]
    dev = m.device
    ys = torch.arange(h, device=dev) + 0.5
    xs = torch.arange(w, device=dev) + 0.5
    gy = ((ys - g.y0) / g.cell_h) / rows * 2 - 1
    gx = g.x_col(xs) / COLS * 2 - 1
    grid = torch.stack(torch.meshgrid(gy, gx, indexing="ij")[::-1], -1)[None]
    return F.grid_sample(m, grid, mode="nearest", padding_mode="zeros", align_corners=False)[0, 0]


def still_picture(samples: list[NDArray[np.uint8]], device: torch.device) -> Tensor:
    """The median of the measurement's luma samples: the scene blurs away, the OSD stays."""
    stack = torch.from_numpy(np.stack(samples)).to(device).float() / 255
    return stack.median(0).values


__all__ = ["BLACK", "WHITE", "Font", "Grid", "Redraw", "find_grid", "sample_cells", "still_picture"]


def for_clip(
    font_path: str,
    samples: list[NDArray[np.uint8]],
    osd: NDArray[np.bool_],
    size: tuple[int, int],
    device: torch.device,
    source: tuple[Path, VideoInfo, Crop, float, float | None, float | None] | None = None,
) -> Redraw | None:
    """The redraw for one recording (``samples``/``osd`` from its measurement, at the
    picture size; ``size``: the output size), or None when no OSD grid reads well."""
    if not font_path:
        raise ValueError("the OSD redraw needs the pilot's font (.mcm): set osd_font_path")
    path = Path(font_path)
    if not path.is_file():
        raise ValueError(f"OSD font {path} not found")
    font = Font(path, device)
    still = still_picture(samples, device)
    grid = find_grid(still, torch.from_numpy(osd).to(device), font)
    if grid is None:
        return None
    h, w = still.shape
    # The clip's vocabulary: glyphs read with confidence on its still picture, plus digits
    # and the symbols numbers use. Anything else (logo tiles matching a torn frame) is
    # never read, whatever it scores.
    mask = torch.from_numpy(osd).to(device)
    idx, ncc = font.read(sample_cells(still, grid))
    seen = set(idx[(ncc >= CONFIDENT) & (sample_cells(mask.float(), grid).mean(1) > 0.05)].tolist())
    seen |= {ord(c) for c in "0123456789-.:%"}
    layout = find_layout(samples, mask, grid, font)
    fixed = fixed_cells(samples, still, mask, grid, font, layout.cells)
    font.limit(seen)
    redraw = Redraw(font, grid, size, (w, h), mask, fixed, layout=layout)
    if source is not None:  # the fields and how each behaves over the recording
        src, info, crop, fps, start, duration = source
        fields = find_fields(fixed, redraw.inside, grid.rows, COLS)

        def read(y: Tensor) -> tuple[Tensor, Tensor]:
            return font.read(sample_cells(y, grid))

        survey(src, info, crop, fields, read, fps, start, duration)
        redraw = Redraw(
            font, grid, size, (w, h), mask, fixed, fields, fps, start or 0.0, layout=layout
        )
    return redraw


def has_osd(cells: Tensor) -> Tensor:
    """Per cell (sampled with :func:`sample_cells`): does it hold OSD-like content, bright
    strokes with their dark outline? Plain sky, walls and blank cells do not."""
    bright = (cells > 0.6).float().mean(1)
    dark = (cells < 0.3).float().mean(1)
    return (bright > 0.04) & (dark > 0.04)


@dataclass(frozen=True)
class Layout:
    """Which cells the OSD occupies, read off the clip's cleanest frame, as a
    screenshot shows where the OSD elements are."""

    cells: Tensor  # (rows*COLS,) bool
    reference: int  # index of the reference sample
    frame: NDArray[np.uint8]  # that sample (picture-size luma), for the review picture


def find_layout(samples: list[NDArray[np.uint8]], mask: Tensor, grid: Grid, font: Font) -> Layout:
    """The reference is the sample that reads best (most OSD cells read with
    confidence: little breakup, the OSD on). A cell is in the layout when it holds OSD
    content in the reference and in at least 3 of the 5 best samples."""
    dev = mask.device
    inside = sample_cells(mask.float(), grid).mean(1) > 0.05
    share, content = [], []
    for frame in samples:
        cells = sample_cells(torch.from_numpy(frame).to(dev).float() / 255, grid)
        _, ncc = font.read(cells)
        share.append(
            float(((ncc >= READ) & inside).float().sum() / inside.float().sum().clamp_min(1))
        )
        content.append(has_osd(cells) & inside)
    order = sorted(range(len(samples)), key=lambda i: -share[i])
    best = order[: min(5, len(order))]
    votes = torch.stack([content[i] for i in best]).sum(0)
    cells = content[best[0]] & (votes >= min(3, len(best)))
    return Layout(cells, best[0], samples[best[0]])


def fixed_cells(
    samples: list[NDArray[np.uint8]],
    still: Tensor,
    mask: Tensor,
    grid: Grid,
    font: Font,
    layout: Tensor | None = None,
) -> Tensor:
    """Per cell, the glyph that never changes in this recording, or -1. A cell inside the
    OSD is fixed when one glyph wins at least STATIC of the samples that read it with
    confidence, and they are at least half the samples (the OSD may be off at the start
    or end). Small glyphs (':', '.') that frames read too faintly are judged on the still
    picture (the median of the samples), where they are sharp."""
    dev = still.device
    inside = sample_cells(mask.float(), grid).mean(1) > 0.05
    if layout is not None:
        inside = inside & layout
    n = grid.rows * COLS
    votes = torch.zeros(n, 256, device=dev)
    present = torch.zeros(n, device=dev)
    reads, shares = [], []
    for frame in samples:
        cells = sample_cells(torch.from_numpy(frame).to(dev).float() / 255, grid)
        idx, ncc = font.read(cells)
        reads.append((idx, ncc, has_osd(cells)))
        shares.append(float(((ncc >= READ) & inside).float().mean()))
    # Samples hit by breakup read far worse than the clip's typical sample: left out.
    typical = float(np.median(shares)) if shares else 0.0
    clean = [r for r, sh in zip(reads, shares, strict=True) if sh >= 0.6 * typical]
    for idx, ncc, there in clean:
        sure = (ncc >= READ) & inside
        votes[sure.nonzero()[:, 0], idx[sure]] += 1
        present += there.float()
    total = votes.sum(1)
    best, glyph = votes.max(1)
    m = max(len(clean), 1)
    fixed = (total >= 0.5 * m) & (best >= STATIC * total) & (present >= 0.8 * m)
    out = torch.where(fixed, glyph, torch.full_like(glyph, -1))
    # A fixed glyph must also be there on the still picture (the median of the samples,
    # where only the OSD stays sharp): a cell that reads one glyph by chance in noisy or
    # torn samples, over plain scenery, is not drawn.
    idx, ncc = font.read(sample_cells(still, grid))
    cells = (out >= 0).nonzero()[:, 0]
    if len(cells):
        there = font.scores[cells, out[cells]] >= READ
        out[cells[~there]] = -1
    # Punctuation from the still picture.
    small = torch.tensor([ord(c) for c in ":."], device=dev)
    punct = inside & (out < 0) & torch.isin(idx, small) & (ncc >= 0.45) & (present >= 0.8 * m)
    return torch.where(punct, idx, out)
