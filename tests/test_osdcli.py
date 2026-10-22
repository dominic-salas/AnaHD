"""Reading OSD element places from a Betaflight CLI dump (bit layout from Betaflight's osd.h)."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")  # the OSD modules build their masks as tensors

from anahd.remaster.osd import COLS
from anahd.remaster.osdcli import SPAN, allowed_cells, decode, parse

DIFF = """
# diff
# version
# Betaflight / STM32F411 (S411) 4.5.1
profile 0
set osd_vbat_pos = 2477
set osd_rssi_pos = 0
set osd_flymode_pos = 6262
set osd_craft_name_pos = 2059
SET osd_tim_2_pos = 14422
"""


def _pos(col: int, row: int, profiles: tuple[int, ...] = (1,)) -> int:
    v = (col & 0x1F) | ((col & 0x20) << 5) | ((row & 0x1F) << 5)
    for p in profiles:
        v |= 1 << (10 + p)
    return v


def test_decode_matches_betaflight_bits() -> None:
    e = decode("vbat", _pos(23, 13))
    assert (e.col, e.row, e.profiles) == (23, 13, (1,))
    assert decode("x", _pos(45, 2, (2, 3))).col == 45  # the HD column bit
    assert decode("x", _pos(45, 2, (2, 3))).profiles == (2, 3)
    assert not decode("rssi", 0).visible
    assert decode("t", _pos(1, 1) | 0xC000).col == 1  # the type bits don't move it


def test_parse_reads_every_pos_setting() -> None:
    els = {e.name: e for e in parse(DIFF)}
    assert set(els) == {"vbat", "rssi", "flymode", "craft_name", "tim_2"}
    assert (els["vbat"].col, els["vbat"].row) == (13, 13)  # 2477 = 0x800 | 13<<5 | 13
    assert (els["craft_name"].col, els["craft_name"].row) == (11, 0)
    assert not els["rssi"].visible


def test_allowed_cells_only_rows_with_visible_elements() -> None:
    m = allowed_cells(parse(DIFF), 16).view(16, COLS)
    assert m[13, 13] and m[13, 13 + SPAN - 1] and not m[13, 12]
    assert not m[5].any()  # nothing listed on row 5
    assert m[0, 11]  # craft name


def test_names_and_static_text() -> None:
    from anahd.remaster.osdcli import names, static_text, video_rows

    dump = DIFF + "set craft_name = ab!#\nset vcd_video_system = NTSC\n"
    assert names(dump) == {"craft_name": "ab!#"}
    assert video_rows(dump) == 13
    cells = static_text(dump, 16)
    # osd_craft_name_pos 2059: row 0, col 11; upper-cased like Betaflight draws it
    assert cells == {11: ord("A"), 12: ord("B"), 13: ord("!"), 14: ord("#")}
