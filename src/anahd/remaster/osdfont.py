"""Betaflight / MAX7456 OSD fonts (.mcm): 256 glyphs of 12x18 pixels, 2 bits each.

The file is the line ``MAX7456`` and then 256 x 64 lines of 8 binary digits: one byte per
line, four pixels per byte (most significant pair first). The first 54 bytes of a glyph
are its 18 rows of 12 pixels; the other 10 are padding. Pixel codes: 00 black, 10 white,
01 and 11 transparent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

GLYPHS, GLYPH_W, GLYPH_H = 256, 12, 18
BLACK, WHITE, CLEAR = 0, 255, 128  # in the returned glyph images

Glyphs = NDArray[np.uint8]  # (256, 18, 12): BLACK, WHITE or CLEAR


def read_mcm(path: Path) -> Glyphs:
    lines = path.read_text(encoding="ascii").split()
    if not lines or lines[0] != "MAX7456":
        raise ValueError(f"{path}: not a MAX7456 font")
    data = lines[1:]
    if len(data) < GLYPHS * 64:
        raise ValueError(f"{path}: {len(data)} data lines, {GLYPHS * 64} expected")
    out = np.full((GLYPHS, GLYPH_H, GLYPH_W), CLEAR, np.uint8)
    for g in range(GLYPHS):
        bits = "".join(data[g * 64 : g * 64 + 54])
        codes = [bits[i : i + 2] for i in range(0, len(bits), 2)]
        px = np.array([{"00": BLACK, "10": WHITE}.get(c, CLEAR) for c in codes], np.uint8)
        out[g] = px.reshape(GLYPH_H, GLYPH_W)
    return out


def contact_sheet(glyphs: Glyphs, scale: int = 3) -> NDArray[np.uint8]:
    """All glyphs on a 16x16 grid (grey = transparent), each labelled by position."""
    cell_w, cell_h = (GLYPH_W + 2) * scale, (GLYPH_H + 2) * scale
    sheet = np.full((16 * cell_h, 16 * cell_w), 90, np.uint8)
    for g in range(GLYPHS):
        r, c = divmod(g, 16)
        big = np.kron(glyphs[g], np.ones((scale, scale), np.uint8))
        big = np.where(big == CLEAR, 90, big).astype(np.uint8)
        y, x = r * cell_h + scale, c * cell_w + scale
        sheet[y : y + GLYPH_H * scale, x : x + GLYPH_W * scale] = big
    return sheet
