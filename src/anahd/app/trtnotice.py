"""A quiet strip under the header when TensorRT is not installed.

TensorRT is optional: without it AnaHD runs the same pipeline in plain PyTorch, a little
slower. The strip offers NVIDIA's download page and a one-click download into AnaHD's own
folder, and can be dismissed for good. Nothing is blocked; with TensorRT present the strip
is never shown.
"""

from __future__ import annotations

import webbrowser

from PySide6.QtCore import QSettings, QThread, Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton

from anahd.errors import AnaHDError
from anahd.gpu import trt

DISMISSED = "trt_notice_dismissed"


class _Installer(QThread):
    progress = Signal(float)
    failed = Signal(str)

    def run(self) -> None:
        try:
            trt.install(lambda _label, fraction: self.progress.emit(fraction))
        except AnaHDError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:  # the strip must report, not crash the app
            self.failed.emit(f"unexpected problem: {exc}")


def _flat(text: str) -> QPushButton:
    widget = QPushButton(text)
    widget.setProperty("flat", True)
    return widget


class TensorRTNotice(QFrame):
    def __init__(self, settings: QSettings) -> None:
        super().__init__()
        self._settings = settings
        self._installer: _Installer | None = None
        self.setProperty("role", "header")
        self.setFixedHeight(40)
        self._text = QLabel(
            "TensorRT (free, from NVIDIA) was not found. AnaHD works without it; "
            "with it the optional upscaler runs faster."
        )
        self._text.setProperty("role", "caption")
        self._get = _flat("Download TensorRT (2.2 GB)")
        self._page = _flat("NVIDIA's page")
        self._hide = _flat("Hide")
        self._get.clicked.connect(self._start)
        self._page.clicked.connect(lambda: webbrowser.open(trt.DOWNLOAD_PAGE))
        self._hide.clicked.connect(self._dismiss)
        row = QHBoxLayout(self)
        row.setContentsMargins(24, 0, 16, 0)
        row.addWidget(self._text)
        row.addStretch(1)
        for item in (self._get, self._page, self._hide):
            row.addWidget(item)
        self.setVisible(self._wanted())

    def _wanted(self) -> bool:
        return not trt.available() and not self._settings.value(DISMISSED, False, type=bool)

    def _dismiss(self) -> None:
        self._settings.setValue(DISMISSED, True)
        self.hide()

    def _start(self) -> None:
        self._get.setEnabled(False)
        self._page.setEnabled(False)
        self._text.setText("Downloading TensorRT... 0%")
        self._installer = installer = _Installer()
        installer.progress.connect(lambda f: self._text.setText(f"Downloading TensorRT... {f:.0%}"))
        installer.failed.connect(self._failed)
        installer.finished.connect(self._done)
        installer.start()

    def _failed(self, why: str) -> None:
        self._text.setText(f"TensorRT could not be downloaded: {why}")
        self._get.setText("Try again")
        self._get.setEnabled(True)
        self._page.setEnabled(True)

    def _done(self) -> None:
        if trt.refresh().ok:
            self.hide()  # the next clip uses it; nothing more to say
