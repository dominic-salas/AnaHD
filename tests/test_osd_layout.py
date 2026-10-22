"""``analyze_clip`` on OSDs other than the reference one: two synthetic recordings with a known OSD,
each with its own font, grid and layout (a Betaflight-style PAL layout, an INAV-style NTSC
one), analysed with and without the font. Against the truth: the layout holds only cells
that have text, finds the fixed labels, recognises the timer, and the redraw never draws
in a cell that is blank in the recording. Needs a CUDA GPU; about two minutes.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():  # pragma: no cover - depends on the machine
    pytest.skip("no CUDA GPU", allow_module_level=True)

import synthetic as syn  # noqa: E402
from anahd.media import probe, read_frames  # noqa: E402
from anahd.pipeline import Options, _full_range, measure  # noqa: E402
from anahd.remaster.osd import COLS, TEMPLATE, sample_cells  # noqa: E402
from anahd.remaster.osdfont import CLEAR, GLYPH_H, GLYPH_W, read_mcm  # noqa: E402
from anahd.remaster.osdlayout import analyze_clip  # noqa: E402

pytestmark = [pytest.mark.gpu, pytest.mark.slow]
SECONDS = 11  # analyze_clip samples ten seconds around the middle
CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ.:%-"
Layout = Callable[[int], NDArray[np.int64]]


@dataclass(frozen=True)
class Setup:
    name: str
    face: int  # OpenCV Hershey face the font is drawn with
    x0: int  # grid corner and cell size in the 960x720 picture
    y0: int
    cell_w: int
    cell_h: int
    rows: int
    layout: Layout
    timer: tuple[int, list[int]]  # row and columns of the timer's digits


def _put(cells: NDArray[np.int64], r: int, c: int, text: str) -> None:
    for k, ch in enumerate(text):
        cells[r, c + k] = 0 if ch == " " else ord(ch)


def betaflight_pal(i: int) -> NDArray[np.int64]:
    cells = np.zeros((16, 30), np.int64)
    _put(cells, 1, 2, "RSSI 88")
    _put(cells, 1, 22, f"{3.9 - 0.1 * (i // 120) % 1:.2f}V")
    _put(cells, 13, 2, f"00:{10 + i // 60:02d}")
    _put(cells, 14, 20, "ALT 12M")
    return cells


def inav_ntsc(i: int) -> NDArray[np.int64]:
    cells = np.zeros((13, 30), np.int64)
    _put(cells, 0, 1, f"3S {11.8 - 0.1 * ((i // 90) % 3):.1f}V")
    _put(cells, 0, 20, "GPS 9")
    _put(cells, 6, 12, "--  --")
    _put(cells, 11, 1, "SPD 42")
    _put(cells, 11, 21, f"FLY 01:{20 + i // 60:02d}")
    return cells


SETUPS = [
    Setup("betaflight_pal", cv2.FONT_HERSHEY_SIMPLEX, 30, 0, 30, 45, 16, betaflight_pal, (13, [2, 3, 5, 6])),  # noqa: E501
    Setup("inav_ntsc", cv2.FONT_HERSHEY_DUPLEX, 0, 2, 32, 55, 13, inav_ntsc, (11, [25, 26, 28, 29])),  # noqa: E501
]  # fmt: skip


def write_font(path: Path, face: int) -> None:
    lines = ["MAX7456"]
    for g in range(256):
        img = np.full((GLYPH_H, GLYPH_W), 1, np.uint8)
        if chr(g) in CHARS:
            big = np.zeros((GLYPH_H * 4, GLYPH_W * 4), np.uint8)
            cv2.putText(big, chr(g), (6, 58), face, 1.8, 255, 7)
            white = cv2.resize(big, (GLYPH_W, GLYPH_H), interpolation=cv2.INTER_AREA) > 100
            img[cv2.dilate(white.astype(np.uint8), np.ones((3, 3))) > 0] = 0
            img[white] = 2
        bits = "".join({0: "00", 1: "01", 2: "10"}[int(v)] for v in img.ravel()) + "01" * 40
        lines += [bits[i : i + 8] for i in range(0, len(bits), 8)]
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def truth(i: int, s: Setup, glyphs: NDArray[np.uint8]) -> NDArray[np.uint8]:
    ox, oy = 40 + 0.3 * i, 30 + 0.15 * i
    m = np.array([[1, 0, -ox], [0, 1, -oy]], np.float32)
    warped = cv2.warpAffine(syn.SCENE, m, (syn.OUT_W, syn.OUT_H), flags=cv2.INTER_CUBIC)
    out: NDArray[np.uint8] = np.clip(warped, 0, 255).astype(np.uint8)
    k = syn.OUT_W / syn.PIC_W
    cells = s.layout(i)
    for r in range(cells.shape[0]):
        for c in range(cells.shape[1]):
            g = glyphs[cells[r, c]]
            if (g == CLEAR).all():
                continue
            x, y = round((s.x0 + c * s.cell_w) * k), round((s.y0 + r * s.cell_h) * k)
            big = cv2.resize(
                g, (round(s.cell_w * k), round(s.cell_h * k)), interpolation=cv2.INTER_NEAREST
            )
            region = out[y : y + big.shape[0], x : x + big.shape[1]]
            mask = big[: region.shape[0], : region.shape[1]] != CLEAR
            region[mask] = big[: region.shape[0], : region.shape[1]][mask][:, None]
    return out


@pytest.fixture(scope="module", params=SETUPS, ids=lambda s: s.name)
def recording(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory):  # type: ignore[no-untyped-def]
    return _record(request.param, tmp_path_factory)


def _record(s: Setup, tmp_path_factory: pytest.TempPathFactory):  # type: ignore[no-untyped-def]
    folder = tmp_path_factory.mktemp(s.name)
    font = folder / f"{s.name}.mcm"
    write_font(font, s.face)
    glyphs = read_mcm(font)
    path = folder / "osd.mp4"
    rng = np.random.default_rng(3)
    lay = syn.FOUR_THREE
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s"]
    cmd += [f"{lay.dvr_w}x{lay.dvr_h}", "-r", str(syn.FPS), "-i", "-", "-c:v", "libx264"]
    cmd += ["-crf", "12", "-preset", "veryfast", "-pix_fmt", "yuv420p", str(path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    for i in range(SECONDS * syn.FPS):
        proc.stdin.write(syn.degrade(truth(i, s, glyphs), lay, rng, band=False).tobytes())
    proc.stdin.close()
    assert proc.wait() == 0
    return s, path, font


def _truth_cell(s: Setup, crop_x: int, x: float, y: float) -> tuple[int, int] | None:
    """The truth's cell (row, col) at picture position x, y (picture-area coordinates)."""
    tx = x + (crop_x - syn.PIC_X) - s.x0
    c, r = int(tx // s.cell_w), int((y - s.y0) // s.cell_h)
    return (r, c) if 0 <= r < s.rows and 0 <= c < COLS else None


@pytest.mark.parametrize("with_font", [True, False], ids=["font", "no_font"])
def test_layout_holds_only_text_and_finds_the_fixed_labels_and_timer(
    recording: tuple[Setup, Path, Path], with_font: bool
) -> None:
    s, path, font = recording
    lay = analyze_clip(path, font if with_font else None)
    assert lay is not None, "no OSD layout found"
    meas = measure(path, probe(path), Options())
    print(f"\n{s.name} {'font' if with_font else 'no font'}: {lay.summary()}")
    window = range(int((SECONDS / 2 - 5) * syn.FPS), int((SECONDS / 2 + 5) * syn.FPS))
    ever = np.zeros((s.rows, COLS), bool)
    always = np.ones((s.rows, COLS), bool)
    first = s.layout(window.start)
    same = np.ones((s.rows, COLS), bool)
    for i in window:
        t = s.layout(i) > 0
        ever |= t
        always &= t
        same &= s.layout(i) == first
    g = lay.grid
    xs = g.col_x(torch.arange(COLS, dtype=torch.float64) + 0.5).tolist()
    wrong_cells, found_static, fixed_ok, fixed_bad = [], 0, 0, 0
    bad: list[tuple[int, int, int, int]] = []
    for i in lay.cells.nonzero()[:, 0].tolist():
        r, c = divmod(i, COLS)
        hit = _truth_cell(s, meas.crop.x, xs[c], g.y0 + (r + 0.5) * g.cell_h)
        if hit is None or not ever[hit]:
            wrong_cells.append((r, c))
            continue
        found_static += int(always[hit] & same[hit])
        f = int(lay.fixed[i])
        if f >= 0 or f == TEMPLATE:
            if always[hit] and same[hit] and (f == TEMPLATE or f == first[hit]):
                fixed_ok += 1
            else:
                fixed_bad += 1
                bad.append((r, c, f, int(first[hit])))
    static = int((always & same).sum())
    print(f"layout cells on blank truth: {wrong_cells}; static found {found_static}/{static}; "
          f"fixed right {fixed_ok}, wrong {fixed_bad} {bad}; "
          f"fields {[(fl.label(), fl.kind) for fl in lay.fields]}")  # fmt: skip
    assert wrong_cells == [], "the layout holds cells that never have text"
    assert found_static >= 0.9 * static
    assert fixed_bad == 0 and fixed_ok >= 0.8 * static
    if with_font:
        row, _ = s.timer
        kinds = {
            c: fl.kind for fl in lay.fields if fl.row == _row_of(lay, s, meas, row) for c in fl.cols
        }
        assert "timer" in kinds.values(), (
            f"timer not recognised: {[(fl.label(), fl.kind) for fl in lay.fields]}"
        )


def _row_of(lay, s: Setup, meas, truth_row: int) -> int:  # type: ignore[no-untyped-def]
    y = s.y0 + (truth_row + 0.5) * s.cell_h
    return int((y - lay.grid.y0) // lay.grid.cell_h)


@pytest.mark.parametrize("with_font", [True, False], ids=["font", "no_font"])
def test_the_redraw_never_draws_in_a_blank_cell(recording, with_font: bool) -> None:  # type: ignore[no-untyped-def]
    s, path, font = recording
    lay = analyze_clip(path, font if with_font else None)
    assert lay is not None
    info = probe(path)
    meas = measure(path, info, Options())
    w, h = meas.crop.width, meas.crop.height
    r = lay.redraw((w, h), torch.from_numpy(meas.osd).cuda(), 60.0, 4.0)
    xs = lay.grid.col_x(torch.arange(COLS, dtype=torch.float64) + 0.5).tolist()
    drawn = blank = 0
    for k, f in enumerate(read_frames(path, info, meas.crop, start=4.0, duration=2.0)):
        y = torch.from_numpy(_full_range(f, info).y.copy()).cuda().float() / 255
        shown = r.update(y)
        truth_cells = s.layout(round(4.0 * 60) + k)
        for i in ((shown >= 0) | (shown == TEMPLATE)).nonzero()[:, 0].tolist():
            g = int(shown[i])
            if g == 0:
                continue
            rr, c = divmod(i, COLS)
            hit = _truth_cell(s, meas.crop.x, xs[c], lay.grid.y0 + (rr + 0.5) * lay.grid.cell_h)
            drawn += 1
            blank += int(hit is None or truth_cells[hit] == 0)
    print(f"\n{s.name}: {drawn} drawn cell-frames, {blank} in blank cells")
    assert drawn > 0 and blank == 0
    _ = sample_cells  # keeps the import used by both tests' helpers


VTX_SWITCH = 330  # frame at which the VTX power changes 400 -> 25


def vtx_change(i: int) -> NDArray[np.int64]:
    """A Betaflight-style PAL screen whose VTX power changes 400 -> 25 halfway."""
    cells = np.zeros((16, 30), np.int64)
    _put(cells, 0, 10, "R:8:400" if i < VTX_SWITCH else "R:8:25 ")
    _put(cells, 1, 2, "RSSI 88")
    _put(cells, 13, 2, f"00:{10 + i // 60:02d}")
    _put(cells, 14, 20, "ALT 12M")
    return cells


VTX = Setup("vtx_change", cv2.FONT_HERSHEY_SIMPLEX, 30, 0, 30, 45, 16, vtx_change, (13, [5, 6]))


@pytest.fixture(scope="module")
def vtx_recording(tmp_path_factory: pytest.TempPathFactory):  # type: ignore[no-untyped-def]
    return _record(VTX, tmp_path_factory)


def test_held_vtx_power_follows_a_change_and_never_shows_a_wrong_value(vtx_recording) -> None:  # type: ignore[no-untyped-def]
    """Held values are read per stretch: after the VTX power changes 400 -> 25 the redraw
    shows 25, never 400 carried over, and never a wrong digit in any frame."""
    s, path, font = vtx_recording
    lay = analyze_clip(path, font)
    assert lay is not None
    info = probe(path)
    meas = measure(path, info, Options())
    w, h = meas.crop.width, meas.crop.height
    tl = lay.read_live(path)
    assert tl is not None
    r = lay.redraw((w, h), torch.from_numpy(meas.osd).cuda(), 60.0, 0.0)
    r.timeline, r.timer_ok = tl, tl.timer_ok
    xs = lay.grid.col_x(torch.arange(COLS, dtype=torch.float64) + 0.5).tolist()

    def truth_at(i: int) -> tuple[int, int] | None:
        rr, c = divmod(i, COLS)
        return _truth_cell(s, meas.crop.x, xs[c], lay.grid.y0 + (rr + 0.5) * lay.grid.cell_h)

    power = [i for i in range(lay.grid.rows * COLS) if truth_at(i) in ((0, 14), (0, 15), (0, 16))]
    assert len(power) == 3, "the VTX power cells were not found"
    wrong = shown_before = shown_after = before = after = 0
    for k, f in enumerate(read_frames(path, info, meas.crop)):
        y = torch.from_numpy(_full_range(f, info).y.copy()).cuda().float() / 255
        shown = r.update(y)
        truth_cells = s.layout(k)
        for i in power:
            hit = truth_at(i)
            assert hit is not None
            got = int(shown[i])
            wrong += int(got > 0 and got != int(truth_cells[hit]))
        if abs(k - VTX_SWITCH) <= 30:
            continue
        if k < VTX_SWITCH:
            before += 1
            shown_before += all(int(shown[i]) > 0 for i in power)
        else:
            after += 1
            shown_after += all(int(shown[i]) > 0 for i in power[:2])
    print(f"\nVTX power: wrong {wrong}; drawn {shown_before}/{before} before, "
          f"{shown_after}/{after} after the change")  # fmt: skip
    assert wrong == 0
    assert shown_before >= 0.8 * before and shown_after >= 0.8 * after
