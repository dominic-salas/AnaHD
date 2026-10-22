"""Which cells of the character grid hold OSD, and what each one is, from a clip's own
frames (the layout comes from the footage alone). Checked against a hand-made
box-by-box verdicts on urban (tests/test_osd_truth.py).

Every cell is read in every measurement sample (spread over the whole clip, in time
order) twice: against the glyphs that can be told apart (the readable ones plus icons
with both white and outline: the RSSI and LQ icons, the clock), and against the digits
alone. Then, per cell:

- **separator**: ':' or '.' on the still picture (too small to read in single frames,
  so judged by their own glyph's match), next to an OSD cell;
- **OSD at all**: one glyph read confidently in at least PRESENT_SHARE of the clean
  samples and matching the still picture (STILL_MATCH), or a digit in at least half of
  them, or part of a multi-glyph picture (osdlayout.static_groups). A cell over plain sky
  reads nothing consistently or not like the still, so it never gets in (v4's false flags);
- **fixed**: anything that is not a digit (labels, icons, the battery hearts, '-'), and a
  digit that never changes, stands alone and is there in nearly every sample (the VTX
  channel);
- **numbers**: runs of digit cells. A number is **held** when its leading digit flips in
  at most HELD_FLIPS of the samples (VTX power "400", link quality "100", the volt
  reading's units), **live** otherwise (current, throttle, RSSI).

Timers are found among the numbers afterwards (osdfields.classify_from), and the minutes
left of a timer's ':' are joined to it there (osdlayout).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from .osd import COLS, READ, STATIC, Font, Grid, sample_cells

PRESENT_SHARE = 0.3  # of clean samples the most read glyph must take for a cell to be OSD
STILL_MATCH = 0.75  # a label's glyph against the still picture (urban: real >= 0.79,
# sky 0.65)
FIXED_SHARE = 0.5  # ...or its share of the frames' confident reads
# Artificial-horizon bars and progress-bar segments (Betaflight SYM_AH_BAR9_*, SYM_PB_*)
# move with the craft or the value: never a label, and look-alikes of '-'.
NEVER_LABELS = list(range(0x80, 0x89)) + list(range(0x8A, 0x90))
WHITE = 0.55  # a label's brightest pixel on the still picture, at least
FIXED_LEAD = 0.05  # a label's glyph lead over the runner-up on the still picture
HELD_FLIPS = 0.12  # share of consecutive confident reads that differ, up to held
SEP_STILL = 0.6  # a separator's own glyph match on the still picture
SEP_FRAMES = 0.45  # ...and in single frames
SEP_SHARE = 0.6  # ...in this share of the clean samples
SEP_SHARE_BETWEEN = 0.3  # ...or this share, between two OSD cells (a row torn by breakup)
SEP_RIVAL = 0.1  # how much better another glyph may match the still than the separator
DIGITS = "0123456789"
TEXT = set(DIGITS + "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-:.%/")  # plain text


@dataclass
class Cells:
    """Per cell (rows*COLS): the class and, for fixed and separator cells, the glyph."""

    kind: list[str]  # "", "fixed", "separator", "held", "live"
    glyph: Tensor  # glyph per cell, -1 where it changes or holds no OSD
    mode: Tensor  # the most read glyph per cell (digits for number cells; -1: none)
    flips: list[float]  # per cell: share of consecutive digit reads that differ


def distinct_glyphs(font: Font) -> Tensor:
    """Glyphs a cell can be matched to reliably: the readable ones, plus icons with both
    enough white and enough outline (not tiny, not near-solid) that ``readable`` drops."""
    op = font.opaque.flatten(1)
    n = op.sum(1)
    white = (font.value.flatten(1) * op).sum(1) / n.clamp_min(1)
    icon = (n >= 20) & (white >= 0.1) & (white <= 0.75)
    out = font.readable | icon
    out[[ord(":"), ord("."), *NEVER_LABELS]] = False
    return out


def _read_with(font: Font, cells: Tensor, glyphs: Tensor) -> tuple[Tensor, Tensor]:
    keep = font.readable
    font.readable = glyphs
    try:
        return font.read(cells)
    finally:
        font.readable = keep


def _votes(stack: Tensor) -> tuple[Tensor, Tensor]:
    """(F, n) glyph or -1 -> per cell, the most read glyph's count and the glyph."""
    n = stack.shape[1]
    votes = torch.zeros(n, 256, device=stack.device)
    for r in stack:
        ok = r >= 0
        votes[ok.nonzero()[:, 0], r[ok]] += 1
    best, mode = votes.max(1)
    return best, torch.where(best > 0, mode, torch.full_like(mode, -1))


def find_cells(
    samples: Sequence[NDArray[np.uint8]],
    still: Tensor,
    inside: Tensor,
    grid: Grid,
    font: Font,
    groups: Tensor,
    group_glyphs: Tensor,
    in_mask: Tensor | None = None,
) -> Cells:
    """``inside``: the cells that may hold OSD at all; ``in_mask``: those inside AnaHD's
    OSD mask. Outside the mask only text is taken (letters, digits, ':' '.'), never an
    icon or tile, which plain scenery matches far more easily."""
    dev = still.device
    n = grid.rows * COLS
    rows = grid.rows
    distinct = distinct_glyphs(font)
    only_digits = torch.zeros_like(distinct)
    only_digits[[ord(c) for c in DIGITS]] = True
    sep_codes = torch.tensor([ord(":"), ord(".")], device=dev)
    reads, digits, seps, shares = [], [], [], []
    for frame in samples:
        cells = sample_cells(torch.from_numpy(frame).to(dev).float() / 255, grid)
        idx, ncc = _read_with(font, cells, distinct)
        sure = (ncc >= READ) & inside
        reads.append(torch.where(sure, idx, torch.full_like(idx, -1)))
        d_idx, d_ncc = _read_with(font, cells, only_digits)
        digits.append(torch.where((d_ncc >= READ) & inside, d_idx, torch.full_like(d_idx, -1)))
        seps.append(
            torch.stack([font.match(cells, torch.full_like(idx, int(s))) for s in sep_codes])
        )
        shares.append(float(sure.float().mean()))
    typical = float(np.median(shares)) if shares else 0.0
    clean = [i for i, sh in enumerate(shares) if sh >= 0.6 * typical]
    m = max(len(clean), 1)
    stack = torch.stack([reads[i] for i in clean])  # (F, n), time order
    dstack = torch.stack([digits[i] for i in clean])
    best, mode = _votes(stack)
    d_best, d_mode = _votes(dstack)

    digit = torch.zeros(256, dtype=torch.bool, device=dev)
    digit[[ord(c) for c in DIGITS]] = True
    # A digit cell reads as some digit in most samples, and its most read glyph overall is
    # a digit, or no glyph holds it (a live value spreads over all ten).
    d_total = (dstack >= 0).sum(0)
    steady = best >= PRESENT_SHARE * m
    mode_digit = (mode >= 0) & digit[mode.clamp_min(0)]
    is_digit = (d_total >= 0.5 * m) & (mode_digit | ~steady)
    # A cell whose most read glyph is a digit is part of a number even when it is blank
    # half the time (the third digit of a link quality that is 100 only some of the time).
    is_digit |= steady & mode_digit
    # A label or icon must also match its glyph on the still picture (the median, where
    # only the OSD stays sharp): sky that happens to read one near-plain glyph does not.
    still_cells = sample_cells(still, grid)
    own_still = font.match(still_cells, mode.clamp_min(0))
    # ...and show white strokes there: OSD glyphs are white with a black outline, so a
    # dark static patch (the quad's own frame in view) that matches a glyph is not one.
    own = (own_still >= STILL_MATCH) & (still_cells.max(1).values >= WHITE)
    present = (steady & (is_digit | own)) | is_digit | (groups >= 0)

    # Separators: their own glyph on the still picture and in most frames.
    sep_still = torch.stack(
        [font.match(still_cells, torch.full((n,), int(s), device=dev)) for s in sep_codes]
    )  # (2, n)
    sep_frames = (torch.stack([seps[i] for i in clean]) >= SEP_FRAMES).float().mean(0)
    which = sep_still.argmax(0)
    p = present.view(rows, COLS)
    beside = torch.zeros_like(p)
    beside[:, 1:] |= p[:, :-1]
    beside[:, :-1] |= p[:, 1:]
    between = torch.zeros_like(p)  # OSD on both sides, like the ':' of "r:8:200"
    between[:, 1:-1] = p[:, :-2] & p[:, 2:]
    frames_share = sep_frames.gather(0, which[None])[0]
    share_needed = torch.where(between.view(-1), SEP_SHARE_BETWEEN, SEP_SHARE)
    is_sep = (sep_still.max(0).values >= SEP_STILL) & (frames_share >= share_needed)
    # A cell also read as some other glyph may still be a separator (a ':' that reads as a
    # sprite tile), but never a digit, and only if the separator matches the still picture
    # nearly as well as that glyph (a 'V' is not a ':').
    is_sep &= inside & beside.view(-1) & ~is_digit
    is_sep &= ~steady | (sep_still.max(0).values >= own_still - SEP_RIVAL)
    # A label's glyph must stand out on the still picture: a look-alike within FIXED_LEAD
    # (two dashes, a ':' against a sprite tile) leaves the cell as recorded.
    allowed = distinct.clone()
    allowed[sep_codes] = True
    s_idx, _ = _read_with(font, still_cells, allowed)
    lead = font.margin
    conf = (stack >= 0).sum(0).clamp_min(1)
    share = best / conf  # the mode's share of a cell's confident reads
    clear = (s_idx == mode) & ((lead >= FIXED_LEAD) | (share >= FIXED_SHARE))

    kind = [""] * n
    glyph = torch.full((n,), -1, dtype=torch.long, device=dev)
    pres, dig, sep_l, grp = present.tolist(), is_digit.tolist(), is_sep.tolist(), groups.tolist()
    clear_l = clear.tolist()
    mask_l = (in_mask if in_mask is not None else inside).tolist()
    const = (d_best >= STATIC * (dstack >= 0).sum(0).clamp_min(1)).tolist()
    # A lone digit is fixed only if it is there nearly always: the third digit of a link
    # quality of 100 is blank whenever it drops below 100, so it is a number, gated per frame.
    always = (d_total >= 0.75 * m).tolist()
    flips = _flips(dstack)
    for r in range(rows):
        c = 0
        while c < COLS:
            i = r * COLS + c
            if sep_l[i]:
                kind[i], glyph[i] = "separator", sep_codes[which[i]]
            elif grp[i] >= 0:
                kind[i], glyph[i] = "fixed", group_glyphs[i]
            elif pres[i] and not dig[i]:
                if clear_l[i] and (mask_l[i] or chr(int(mode[i])) in TEXT):
                    kind[i], glyph[i] = "fixed", mode[i]
            elif pres[i]:  # a number: this digit and the digits right of it
                end = c
                while end + 1 < COLS and dig[i + end + 1 - c] and not sep_l[i + end + 1 - c]:
                    end += 1
                if end == c and const[i] and always[i]:
                    kind[i], glyph[i] = "fixed", d_mode[i]
                else:
                    k = "held" if flips[i] <= HELD_FLIPS else "live"
                    for j in range(i, i + end - c + 1):
                        kind[j] = k
                c = end
            c += 1
    return Cells(kind, glyph, torch.where(is_digit, d_mode, mode), flips)


def _flips(stack: Tensor) -> list[float]:
    """Per cell, the share of consecutive confident reads (time order) that differ."""
    out = []
    for col in stack.T.cpu().numpy():
        seq = col[col >= 0]
        out.append(float((seq[1:] != seq[:-1]).mean()) if len(seq) > 1 else 0.0)
    return out
