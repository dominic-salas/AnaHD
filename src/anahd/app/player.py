"""The clips page's preview: plays the recording (skipping the cut parts, so it shows what
will be kept) and shows still frames while scrubbing. Frames come from FFmpeg, reduced to
preview size; nothing here touches the GPU."""

from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path

from PySide6.QtCore import QObject, QRect, Qt, QThread, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPaintEvent, QPixmap
from PySide6.QtWidgets import QWidget

from anahd.app import theme
from anahd.app.timeline import skip_cuts
from anahd.cuts import Cut
from anahd.media import NO_WINDOW, require

PREVIEW_H = 540
PREVIEW_FPS = 30


class PlayThread(QThread):
    """Decodes from a moment on and hands out frames at the speed they were recorded."""

    frame = Signal(QImage, float)  # picture, its time in the recording
    ended = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._stop = threading.Event()
        self._proc: subprocess.Popen[bytes] | None = None
        self.path: Path | None = None
        self.start_at = 0.0
        self.duration = 0.0
        self.size = (0, 0)  # recording's width, height
        self.cuts: list[Cut] = []

    def play(
        self, path: Path, start: float, duration: float, size: tuple[int, int], cuts: list[Cut]
    ) -> None:
        self.halt()
        self.path, self.start_at, self.duration, self.size = path, start, duration, size
        self.cuts = list(cuts)
        self._stop.clear()
        self.start()

    def halt(self) -> None:
        self._stop.set()
        proc = self._proc
        if proc is not None and proc.poll() is None:
            proc.kill()
        self.wait()

    def run(self) -> None:
        if self.path is None or self.size[1] == 0:
            return
        w = round(PREVIEW_H * self.size[0] / self.size[1] / 2) * 2
        t = skip_cuts(self.start_at, self.cuts)
        while t < self.duration and not self._stop.is_set():
            t = self._play_from(t, w)
            t = skip_cuts(t, self.cuts)
        if not self._stop.is_set():
            self.ended.emit()

    def _play_from(self, t: float, w: int) -> float:
        """Plays until a cut, the end, or a stop; returns where it got to."""
        cmd = [
            require("ffmpeg"),
            "-nostdin",
            "-v",
            "error",
            "-ss",
            f"{t:.3f}",
            "-i",
            str(self.path),
        ]
        cmd += ["-vf", f"fps={PREVIEW_FPS},scale={w}:{PREVIEW_H}", "-pix_fmt", "rgb24"]
        cmd += ["-f", "rawvideo", "-"]
        self._proc = proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, creationflags=NO_WINDOW
        )
        if proc.stdout is None:  # pragma: no cover - Popen with PIPE always sets it
            return self.duration
        n = w * PREVIEW_H * 3
        began: float | None = None  # the pace is counted from the first picture
        shown = 0
        try:
            while not self._stop.is_set():
                data = proc.stdout.read(n)
                if len(data) < n:
                    return self.duration
                now = t + shown / PREVIEW_FPS
                if skip_cuts(now, self.cuts) != now:
                    return now  # a cut starts here: go on after it
                clock = time.monotonic()
                if began is None:
                    began = clock  # FFmpeg's start-up is not owed back by rushing
                delay = began + shown / PREVIEW_FPS - clock
                if delay > 0:
                    time.sleep(delay)
                elif delay < -2 / PREVIEW_FPS:
                    began = clock - shown / PREVIEW_FPS  # fell behind: go on from here, no rush
                image = QImage(data, w, PREVIEW_H, w * 3, QImage.Format.Format_RGB888).copy()
                self.frame.emit(image, now)
                shown += 1
            return t + shown / PREVIEW_FPS
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait()


class Screen(QWidget):
    """Shows the latest preview frame, as large as fits, in its own shape."""

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(320, 200)
        self._pixmap: QPixmap | None = None
        self.message = "Pick a clip to preview it"
        self.palette_ = theme.DARK

    def show_image(self, image: QImage | QPixmap | None) -> None:
        if isinstance(image, QImage):
            image = QPixmap.fromImage(image)
        self._pixmap = image
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.fillRect(self.rect(), QColor(self.palette_.surface))
        if self._pixmap is None or self._pixmap.isNull():
            painter.setPen(QColor(self.palette_.faint))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.message)
        else:
            size = self._pixmap.size().scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio)
            box = QRect(
                (self.width() - size.width()) // 2,
                (self.height() - size.height()) // 2,
                size.width(),
                size.height(),
            )
            painter.drawPixmap(box, self._pixmap)
        painter.end()
