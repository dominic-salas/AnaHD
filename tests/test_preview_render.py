"""The clips page's preview frame is what CLEAN renders: the app's preview thread and the
app's worker (the command line it runs) agree on the same frame, with settings that are
not the defaults. Needs torch and a CUDA GPU.

Limits, fixed before measuring: luma mean difference <= 2 levels and 99th percentile <= 12
(H.264 at CRF 12 and its 4:2:0 colour cost about a level); colour compared on 16x16 block
means (where 4:2:0 does not count), mean <= 2 levels. A preview with other settings must
differ clearly (>= 6 levels), or the comparison could not tell them apart.
"""

from __future__ import annotations

import os
import queue
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():  # pragma: no cover - depends on the machine
    pytest.skip("no CUDA GPU", allow_module_level=True)
pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEventLoop, QTimer  # noqa: E402
from PySide6.QtGui import QImage  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

import synthetic as syn  # noqa: E402
from anahd.app import worker  # noqa: E402
from anahd.app.framepreview import FramePreviewer, Request, Shown  # noqa: E402
from anahd.remaster import RemasterParams  # noqa: E402

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

CHOSEN = RemasterParams(
    warmth=40,
    saturation=140,
    look_strength=60,
    vignette=150,
    sharpen=30,
    motion_blur=39,
    superview="stretch",
)


def _rgb(img: QImage) -> NDArray[np.uint8]:
    img = img.convertToFormat(QImage.Format.Format_RGB888)
    w, h = img.width(), img.height()
    buf = np.frombuffer(img.constBits(), np.uint8, count=img.bytesPerLine() * h)
    return buf.reshape(h, img.bytesPerLine())[:, : w * 3].reshape(h, w, 3).copy()


def _preview(path: Path, seconds: float, params: RemasterParams) -> Shown:
    QApplication.instance() or QApplication([])
    previewer = FramePreviewer()
    got: list[Shown] = []
    loop = QEventLoop()

    def done(shown: Shown) -> None:
        got.append(shown)
        loop.quit()

    previewer.ready.connect(done)
    previewer.failed.connect(lambda _r, why: pytest.fail(why))
    previewer.request(Request(path, seconds, params, True))
    QTimer.singleShot(120_000, loop.quit)
    loop.exec()
    previewer.stop()
    assert got, "no preview in 2 minutes"
    return got[0]


def _blocks(rgb: NDArray[np.uint8]) -> NDArray[np.float64]:
    h, w = rgb.shape[0] // 16 * 16, rgb.shape[1] // 16 * 16
    return rgb[:h, :w].reshape(h // 16, 16, w // 16, 16, 3).mean(axis=(1, 3))


def _luma(rgb: NDArray[np.uint8]) -> NDArray[np.float64]:
    r, g, b = (rgb[..., i].astype(np.float64) for i in range(3))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def test_preview_frame_equals_the_rendered_frame(tmp_path: Path) -> None:
    clip = syn.write(tmp_path / "flight.mp4", 120)
    shown = _preview(clip.path, 1.0, CHOSEN)
    messages: queue.Queue[tuple[Any, ...]] = queue.Queue()
    out = tmp_path / "AnaHD_flight.mp4"
    worker.work(str(clip.path), str(out), messages, [], False, True, CHOSEN.to_dict())  # type: ignore[arg-type]
    last = None
    while not messages.empty():
        last = messages.get()
    assert last is not None and last[0] == "done", last
    after = _rgb(shown.after)
    h, w = after.shape[:2]
    frames = list(syn.read_bgr(out, (w, h)))
    n = round(shown.seconds * syn.FPS)
    rendered = frames[n][..., ::-1]  # BGR -> RGB
    dy = np.abs(_luma(after) - _luma(rendered))
    blocks = float(np.abs(_blocks(after) - _blocks(rendered)).mean())
    print(
        f"\nframe {n} ({w}x{h}): luma mean {dy.mean():.2f}, p99 {np.percentile(dy, 99):.1f}; "
        f"colour blocks {blocks:.2f}"
    )
    assert (w, h) == (1920, 1080)  # SuperView reached both
    assert dy.mean() <= 2.0 and np.percentile(dy, 99) <= 12
    assert blocks <= 2.0
    # the comparison can tell settings apart: the defaults' preview is clearly different
    other = _rgb(_preview(clip.path, 1.0, RemasterParams(superview="stretch")).after)
    apart = float(np.abs(_blocks(other) - _blocks(rendered)).mean())
    print(f"defaults' preview against this render: colour blocks {apart:.2f}")
    assert apart >= 6.0
