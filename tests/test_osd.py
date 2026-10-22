"""The experimental OSD redraw (anahd.remaster.osd) on a synthetic recording whose OSD is
known: text drawn with a font on the Betaflight grid over the clean scene, then recorded
through the same analog damage as the other tests (tests/synthetic.py). Needs a CUDA GPU.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():  # pragma: no cover - depends on the machine
    pytest.skip("no CUDA GPU", allow_module_level=True)

import synthetic as syn  # noqa: E402
from anahd.remaster import osd  # noqa: E402
from anahd.remaster.osdfont import CLEAR, GLYPH_H, GLYPH_W, read_mcm  # noqa: E402

FRAMES = 90
X0, Y0 = 30, 0  # the grid's corner in the 960x720 picture: 30 x 16 cells of 30 x 45
CELL_W, CELL_H = 30, 45
CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ.:%-"


def write_font(path: Path) -> None:
    """A MAX7456 font: Hershey characters at 12x18, white with a black outline; glyph
    0 blank, glyph ord(c) for each character in CHARS, the rest blank."""
    lines = ["MAX7456"]
    for g in range(256):
        img = np.full((GLYPH_H, GLYPH_W), 1, np.uint8)  # 1 = transparent
        c = chr(g)
        if c in CHARS:
            big = np.zeros((GLYPH_H * 4, GLYPH_W * 4), np.uint8)
            cv2.putText(big, c, (6, 58), cv2.FONT_HERSHEY_SIMPLEX, 1.9, 255, 7)
            white = cv2.resize(big, (GLYPH_W, GLYPH_H), interpolation=cv2.INTER_AREA) > 100
            outline = cv2.dilate(white.astype(np.uint8), np.ones((3, 3))) > 0
            img[outline] = 0
            img[white] = 2
        codes = {0: "00", 1: "01", 2: "10"}
        bits = "".join(codes[int(v)] for v in img.ravel()) + "01" * 4 * 10
        lines += [bits[i : i + 8] for i in range(0, len(bits), 8)]
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def layout(i: int) -> NDArray[np.int64]:
    """Glyph per cell (16 x 30) at frame ``i``: fixed labels, a counting timer, a value
    that changes every 20 frames and a warning that blinks every 15 frames."""
    cells = np.zeros((16, 30), np.int64)

    def put(r: int, c: int, text: str) -> None:
        for k, ch in enumerate(text):
            cells[r, c + k] = 0 if ch == " " else ord(ch)

    put(1, 2, "RSSI 88")
    put(1, 22, f"{3.9 - 0.1 * (i // 20):.2f}V")
    put(13, 2, f"00:{10 + i // 60:02d}")
    put(14, 20, "ALT 12M")
    if (i // 15) % 2 == 0:
        put(7, 11, "LOW BAT")
    return cells


def render(
    base: NDArray[np.uint8], cells: NDArray[np.int64], glyphs: NDArray[np.uint8]
) -> NDArray[np.uint8]:
    """The OSD drawn over ``base`` (1440x1080 BGR) as the VTX would, at output scale."""
    out = base.copy()
    s = syn.OUT_W / syn.PIC_W
    for r in range(cells.shape[0]):
        for c in range(cells.shape[1]):
            g = glyphs[cells[r, c]]
            if (g == CLEAR).all():
                continue
            x, y = round((X0 + c * CELL_W) * s), round((Y0 + r * CELL_H) * s)
            w, h = round(CELL_W * s), round(CELL_H * s)
            big = cv2.resize(g, (w, h), interpolation=cv2.INTER_NEAREST)
            region = out[y : y + h, x : x + w]
            m = big != CLEAR
            region[m] = big[m][:, None]
    return out


def truth(i: int, glyphs: NDArray[np.uint8]) -> NDArray[np.uint8]:
    ox, oy = 40 + 0.3 * i, 30 + 0.15 * i
    m = np.array([[1, 0, -ox], [0, 1, -oy]], np.float32)
    scene = cv2.warpAffine(syn.SCENE, m, (syn.OUT_W, syn.OUT_H), flags=cv2.INTER_CUBIC)
    return render(np.clip(scene, 0, 255).astype(np.uint8), layout(i), glyphs)


@pytest.fixture(scope="module")
def clip(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, NDArray[np.uint8]]:
    folder = tmp_path_factory.mktemp("osd")
    font = folder / "test.mcm"
    write_font(font)
    glyphs = read_mcm(font)
    return record(folder / "osd.mp4", glyphs), font, glyphs


def record(path: Path, glyphs: NDArray[np.uint8]) -> Path:
    """Like syn.write_with, plus what the analog link does to the white OSD text in real
    recordings: the colour lags the brightness by ~3 pixels (colour fringes) and the
    brightness is smeared along the line."""
    import subprocess

    rng = np.random.default_rng(1)
    lay = syn.FOUR_THREE
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24"]
    cmd += ["-s", f"{lay.dvr_w}x{lay.dvr_h}", "-r", str(syn.FPS), "-i", "-", "-c:v", "libx264"]
    cmd += ["-crf", "10", "-preset", "veryfast", "-pix_fmt", "yuv420p", str(path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    for i in range(FRAMES):
        frame = syn.degrade(truth(i, glyphs), lay, rng, band=False)
        ycc = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb).astype(np.float32)
        ycc[..., 0] = np.asarray(cv2.GaussianBlur(ycc[..., 0], (0, 0), sigmaX=1.5, sigmaY=0.1))
        ycc[..., 1:] = np.roll(ycc[..., 1:], 3, axis=1)
        shifted = cv2.cvtColor(np.clip(ycc, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2BGR)
        proc.stdin.write(shifted.tobytes())
    proc.stdin.close()
    assert proc.wait() == 0
    return path


def test_the_text_is_read_correctly(clip: tuple[Path, Path, NDArray[np.uint8]]) -> None:
    """Reading the recording's own picture area (after the analog damage), once the
    reading has settled: no cell holding text is drawn as a wrong glyph, and at least 95 %
    are drawn right (the rest, read without enough confidence, stay as recorded: v3 rule).
    Measured 2026-10-21: 96.0 % right, 0 wrong (v2, which drew every reading: 100 %)."""
    from anahd.media import probe, read_frames
    from anahd.pipeline import Options, _full_range, measure

    path, font_path, _ = clip
    info = probe(path)
    meas = measure(path, info, Options())
    dev = torch.device("cuda")
    font = osd.Font(font_path, dev)
    grid = osd.find_grid(
        osd.still_picture(meas.samples, dev), torch.from_numpy(meas.osd).to(dev), font
    )
    assert grid is not None, "no OSD grid found"
    print(f"\ngrid: cell {grid.cell_w:.2f} x {grid.cell_h:.2f} at ({grid.x0:.1f}, {grid.y0:.1f}), "
          f"{grid.rows} rows, score {grid.score:.2f}")  # fmt: skip
    h, w = meas.osd.shape
    redraw = osd.Redraw(font, grid, (w, h), (w, h))
    right = total = unread = 0
    for i, f in enumerate(read_frames(path, info, meas.crop)):
        y = torch.from_numpy(_full_range(f, info).y.copy()).to(dev).float() / 255
        shown = redraw.update(y).view(grid.rows, osd.COLS).cpu().numpy()
        want = layout(i)
        # Settled: the same text for the last STABLE frames (blinks and changes need them).
        if i < osd.STABLE or any((layout(i - k) != want).any() for k in range(1, osd.STABLE)):
            continue
        text = want[: grid.rows] > 0
        got = shown[: want.shape[0]][text]
        right += int((got == want[: grid.rows][text]).sum())
        unread += int((got == -1).sum())
        total += int(text.sum())
    acc = right / max(total, 1)
    print(f"character accuracy {acc:.4f} over {total} cells ({unread} passed through)")
    assert total > 1000 and right + unread == total and acc >= 0.95


def test_redraw_is_closer_to_the_true_osd_than_passing_it_through(
    clip: tuple[Path, Path, NDArray[np.uint8]],
) -> None:
    from anahd import cli

    path, font_path, glyphs = clip
    plain = path.with_name("plain.mp4")
    drawn = path.with_name("drawn.mp4")
    assert cli.main([str(path), "-o", str(plain)]) == 0
    extra = ["--look-osd-redraw", "--look-osd-font-path", str(font_path)]
    assert cli.main([str(path), "-o", str(drawn), *extra]) == 0
    from anahd.media import probe
    from anahd.pipeline import Options, measure

    c = measure(path, probe(path), Options()).crop
    area = (c.x, c.y, c.width, c.height)  # the truth framed like AnaHD's picture area
    size = (syn.OUT_W, syn.OUT_H)
    scores: dict[str, list[float]] = {"plain": [], "drawn": []}
    for i, (a, b) in enumerate(
        zip(syn.read_bgr(plain, size), syn.read_bgr(drawn, size), strict=True)
    ):
        if i < 5:
            continue
        t = syn.reframe(truth(i, glyphs), *area)
        solid = np.where(glyphs == CLEAR, CLEAR, 255).astype(np.uint8)
        m = syn.reframe(render(np.zeros_like(t), layout(i), solid), *area)[..., 0] > 127
        for name, frame in (("plain", a), ("drawn", b)):
            err = (frame.astype(np.float64) - t)[m]
            scores[name].append(10 * np.log10(255**2 / max(float(np.mean(err**2)), 1e-9)))
    p, d = float(np.mean(scores["plain"])), float(np.mean(scores["drawn"]))
    print(f"\nOSD PSNR: passed through {p:.2f} dB, redrawn {d:.2f} dB")
    assert d > p + 1.0  # measured 2026-10-21: 14.89 -> 17.29 dB
