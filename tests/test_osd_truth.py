"""``analyze_clip`` against a hand-made box-by-box check of urban's OSD layout
(tests/data/osd_truth_urban.json, 2026-10-21): every truth cell found with the right class,
and no cell that is not in the truth. Needs the dev clips and font; skips without them.

Five cells follow the footage instead of the first marks (their "override" says why):
the link quality "29:100" is Betaflight's "rfMode:LQ%", so "29" is held and "100" live.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():  # pragma: no cover - depends on the machine
    pytest.skip("no CUDA GPU", allow_module_level=True)

from anahd.remaster.osd import COLS  # noqa: E402
from anahd.remaster.osdlayout import OsdLayout, analyze_clip  # noqa: E402

pytestmark = [pytest.mark.gpu, pytest.mark.slow]
DEV_CLIPS = Path(os.environ.get("ANAHD_DEV_CLIPS", "dev_clips"))
FONT = Path(os.environ.get("ANAHD_OSD_FONT", "osd_font.mcm"))
TRUTH = Path(__file__).parent / "data" / "osd_truth_urban.json"
SEPARATORS = {ord(":"), ord(".")}


def classes(lay: OsdLayout) -> dict[tuple[int, int], str]:
    """The layout's class per cell, in the truth's terms."""
    out: dict[tuple[int, int], str] = {}
    for i in lay.cells.nonzero()[:, 0].tolist():
        g = int(lay.fixed[i])
        if g >= 0:
            out[divmod(i, COLS)] = "separator" if g in SEPARATORS else "fixed"
        else:
            out[divmod(i, COLS)] = "live"
    for fl in lay.fields:
        kind = {"slow": "held"}.get(fl.kind, fl.kind)
        for c in fl.cols:
            out[(fl.row, c)] = kind
        for c in fl.seps:
            out[(fl.row, c)] = "separator"
    return out


def test_urban_layout_matches_doms_check() -> None:
    clip = DEV_CLIPS / "urban.mp4"
    if not clip.exists() or not FONT.exists():
        pytest.skip("needs the dev clips and OSD font")
    truth = {(c["row"], c["col"]): c for c in json.loads(TRUTH.read_text())["cells"]}
    lay = analyze_clip(clip, FONT)
    assert lay is not None
    found = classes(lay)
    missing = sorted(k for k in truth if k not in found)
    wrong = sorted(
        (k, truth[k]["class"], found[k])
        for k in truth
        if k in found and found[k] not in [truth[k]["class"], *truth[k].get("also", [])]
    )
    extra = sorted(k for k in found if k not in truth)
    glyphs = sorted(
        (k, truth[k]["glyph"], int(lay.fixed[k[0] * COLS + k[1]]))
        for k in truth
        if truth[k]["glyph"] is not None and k in found and found[k] in ("fixed", "separator")
        and int(lay.fixed[k[0] * COLS + k[1]]) != truth[k]["glyph"]
    )  # fmt: skip
    print(f"\n{len(truth)} truth cells: missing {missing}\nwrong class {wrong}\nextra {extra}"
          f"\nwrong glyph {glyphs}")  # fmt: skip
    assert not missing and not wrong and not extra and not glyphs
