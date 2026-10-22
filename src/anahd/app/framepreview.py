"""The clips page's one-frame preview: the playhead's frame remastered with the panel's
settings, next to the plain recording, rendered off the UI thread.

It is the render's own code (``anahd.remaster.preview.preview_frame``), so what it shows is
what CLEAN gives. One thread owns the GPU state (``PreviewCache``: the measured clip, the
cleaner, cleaned frames per moment); only the newest request is rendered, older ones that
are still waiting are dropped, and finished frames are kept (latest ``KEEP``) so going
back to a setting or a moment is instant.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtGui import QImage

from anahd.remaster import RemasterParams

KEEP = 24  # finished previews remembered


@dataclass(frozen=True)
class Request:
    path: Path
    seconds: float
    params: RemasterParams
    learned: bool

    @property
    def key(self) -> tuple[object, ...]:
        return (self.path, round(self.seconds, 3), self.params.to_json(), self.learned)


@dataclass(frozen=True)
class Shown:
    request: Request
    before: QImage
    after: QImage
    seconds: float  # the frame actually shown


def to_qimage(rgb: NDArray[np.uint8]) -> QImage:
    h, w, _ = rgb.shape
    data = np.ascontiguousarray(rgb)
    return QImage(data.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()


class FramePreviewer(QThread):
    ready = Signal(object)  # Shown
    failed = Signal(object, str)  # Request, why
    busy = Signal(bool)  # rendering (for the spinner)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._next: Request | None = None
        self._stop = False
        self._done: OrderedDict[tuple[object, ...], Shown] = OrderedDict()

    def request(self, req: Request) -> None:
        """Ask for a frame; a finished one comes back at once (from the thread's signal)."""
        with self._lock:
            self._next = req
        self._wake.set()
        if not self.isRunning():
            self.start()

    def stop(self) -> None:
        self._stop = True
        self._wake.set()
        self.wait()

    def run(self) -> None:
        cache = None
        while not self._stop:
            self._wake.wait()
            self._wake.clear()
            with self._lock:
                req, self._next = self._next, None
            if req is None or self._stop:
                continue
            hit = self._done.get(req.key)
            if hit is not None:
                self._done.move_to_end(req.key)
                self.ready.emit(hit)
                continue
            self.busy.emit(True)
            try:
                from anahd.remaster.preview import PreviewCache, preview_frame  # torch: lazily

                if cache is None:
                    cache = PreviewCache()
                pv = preview_frame(req.path, req.seconds, req.params, cache, learned=req.learned)
                shown = Shown(req, to_qimage(pv.source), to_qimage(pv.result), pv.seconds)
                self._done[req.key] = shown
                while len(self._done) > KEEP:
                    self._done.popitem(last=False)
                self.ready.emit(shown)
            except Exception as exc:
                self.failed.emit(req, f"{type(exc).__name__}: {exc}")
            finally:
                with self._lock:
                    more = self._next is not None
                self.busy.emit(more)
