"""Optional OSD layout input from a Betaflight CLI ``dump`` or ``diff`` (Advanced setting).

Betaflight stores each OSD element's place as ``set osd_<name>_pos = N``, a 16-bit value
(``src/main/osd/osd.h``, checked against Betaflight master, 2026-10):

- bits 0-4: column (0-31), bit 10: a sixth column bit for HD OSDs (``OSD_X``);
- bits 5-9: row (``OSD_Y``);
- bits 11-13: shown in OSD profile 1, 2, 3 (``OSD_PROFILE_FLAG``);
- bits 14-15: the element's type/variant (``OSD_TYPE``).

An element is anchored at its left cell; its width depends on the element and the value
shown, so the CLI only says *where text may start*. :func:`allowed_cells` turns that into a
guard on the picture-derived layout: it can only take cells away (never adds any), keeping
cells on a row that has a visible element, from its column to ``SPAN`` cells right of it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from .osd import COLS

SPAN = 16  # cells right of an element's anchor that it may cover (craft name, warnings)
_SET = re.compile(r"^\s*set\s+osd_(\w+?)_pos\s*=\s*(\d+)\s*$", re.IGNORECASE | re.MULTILINE)
_NAME = re.compile(
    r"^\s*set\s+(craft_name|pilot_name)\s*=[ 	]?(.*?)\s*$", re.IGNORECASE | re.MULTILINE
)
_OLD_NAME = re.compile(r"^\s*name\s+(.*?)\s*$", re.IGNORECASE | re.MULTILINE)  # before 4.4
_VIDEO = re.compile(r"^\s*set\s+vcd_video_system\s*=\s*(\w+)", re.IGNORECASE | re.MULTILINE)
# osd_*_pos element -> the name setting it shows (Betaflight osd_elements.c: the names are
# drawn upper-cased, ``toUpperCase``)
NAME_ELEMENTS = {"craft_name": "craft_name", "pilot_name": "pilot_name"}


@dataclass(frozen=True)
class Element:
    name: str
    col: int
    row: int
    profiles: tuple[int, ...]  # OSD profiles it is shown in (empty: hidden)

    @property
    def visible(self) -> bool:
        return bool(self.profiles)


def decode(name: str, value: int) -> Element:
    col = (value & 0x1F) | ((value & 0x400) >> 5)
    row = (value >> 5) & 0x1F
    profiles = tuple(p for p in (1, 2, 3) if value & (1 << (p - 1 + 11)))
    return Element(name, col, row, profiles)


def parse(text: str) -> list[Element]:
    """Every ``osd_*_pos`` setting in a CLI dump or diff (``diff`` lists only the changed ones)."""
    return [decode(m[1], int(m[2])) for m in _SET.finditer(text)]


def read(path: Path | str) -> list[Element]:
    return parse(Path(path).read_text(encoding="utf-8", errors="replace"))


def allowed_cells(elements: list[Element], rows: int, profile: int | None = None) -> Tensor:
    """(rows*COLS,) bool: cells a visible element may cover (in ``profile``, or in any)."""
    out = torch.zeros(rows, COLS, dtype=torch.bool)
    for e in elements:
        shown = e.profiles if profile is None else [p for p in e.profiles if p == profile]
        if shown and e.row < rows and e.col < COLS:
            out[e.row, e.col : min(COLS, e.col + SPAN)] = True
    return out.view(-1)


def names(text: str) -> dict[str, str]:
    """``craft_name`` and ``pilot_name`` from a dump or diff (also the pre-4.4 ``name``)."""
    out = {m[1].lower(): m[2] for m in _NAME.finditer(text)}
    if "craft_name" not in out and (m := _OLD_NAME.search(text)) and m[1] != "-":
        out["craft_name"] = m[1]
    return out


def video_rows(text: str) -> int | None:
    """16 for PAL, 13 for NTSC, None when the dump leaves it on AUTO (or doesn't say)."""
    m = _VIDEO.search(text)
    if not m:
        return None
    return {"PAL": 16, "NTSC": 13}.get(m[1].upper())


def static_text(text: str, rows: int) -> dict[int, int]:
    """Cells (row*COLS + col) -> glyph code of the text elements a dump fixes outright:
    the craft and pilot names at their places, as Betaflight draws them (upper-cased).
    Pilots build pictures from custom glyphs in these names; with the dump the picture
    is known exactly, no reading needed."""
    pos = {e.name: e for e in parse(text)}
    out: dict[int, int] = {}
    for element, setting in NAME_ELEMENTS.items():
        e = pos.get(element)
        value = names(text).get(setting)
        if e is None or not e.visible or not value or e.row >= rows:
            continue
        for k, ch in enumerate(value.upper()):
            if e.col + k < COLS and ch != " ":
                out[e.row * COLS + e.col + k] = ord(ch) & 0xFF
    return out
