"""The window drawn to its top edge: the hit-test rules, the non-Windows no-op, and a real
window on Windows answering hit tests and maximising from its own button."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest
from PySide6.QtWidgets import QApplication, QFrame, QHBoxLayout, QLabel, QMainWindow

from anahd.app import windowchrome as wc

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

WIDTH, HEIGHT, EDGE = 800, 600, 6


def region(x: int, y: int) -> str:
    if y < 40:
        return wc.MAXIMIZE if 700 <= x < 744 else wc.CAPTION if x < 600 else wc.CLIENT
    return wc.CLIENT


def hit(x: int, y: int, resizable: bool = True) -> int:
    return wc.classify(x, y, WIDTH, HEIGHT, EDGE, resizable, region)


def test_lparam_carries_signed_screen_coordinates() -> None:
    assert wc.split_lparam((20 << 16) | 10) == (10, 20)
    assert wc.split_lparam(((-5 & 0xFFFF) << 16) | (-1200 & 0xFFFF)) == (-1200, -5)


def test_edges_and_corners_resize() -> None:
    assert hit(0, 300) == wc.HTLEFT
    assert hit(WIDTH - 1, 300) == wc.HTRIGHT
    assert hit(400, HEIGHT - 1) == wc.HTBOTTOM
    assert hit(400, 0) == wc.HTTOP
    assert hit(0, 0) == wc.HTTOPLEFT
    assert hit(WIDTH - 1, 0) == wc.HTTOPRIGHT
    assert hit(0, HEIGHT - 1) == wc.HTBOTTOMLEFT
    assert hit(WIDTH - 1, HEIGHT - 1) == wc.HTBOTTOMRIGHT


def test_the_top_strip_drags_and_the_maximise_button_is_the_snap_target() -> None:
    assert hit(300, 20) == wc.HTCAPTION
    assert hit(720, 20) == wc.HTMAXBUTTON
    assert hit(650, 20) == wc.HTCLIENT  # a button or field in the bar
    assert hit(400, 300) == wc.HTCLIENT


def test_a_maximised_window_has_no_resize_edges() -> None:
    assert hit(0, 300, resizable=False) == wc.HTCLIENT
    assert hit(0, 0, resizable=False) == wc.HTCAPTION
    assert hit(400, 0, resizable=False) == wc.HTCAPTION


def test_not_windows_changes_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    app = QApplication.instance() or QApplication([])
    assert isinstance(app, QApplication)
    monkeypatch.setattr(wc, "_supported", lambda: False)
    window = QMainWindow()
    header = QFrame()
    row = QHBoxLayout(header)
    row.addWidget(QLabel("logo"))
    before = row.count()
    assert wc.attach(window, header) is None
    assert row.count() == before
    assert header.findChildren(wc.CaptionButtons) == []


REAL_WINDOW = textwrap.dedent(
    """
    import ctypes, sys, time
    from ctypes import wintypes
    from PySide6.QtCore import QPoint
    from PySide6.QtWidgets import (
        QApplication, QFrame, QHBoxLayout, QLabel, QMainWindow, QPushButton, QVBoxLayout, QWidget
    )
    from anahd.app import windowchrome as wc

    app = QApplication([])
    window = QMainWindow()
    window.setWindowTitle("Chrome test")
    header = QFrame()
    header.setFixedHeight(48)
    row = QHBoxLayout(header)
    row.addWidget(QLabel("logo"))
    row.addStretch(1)
    row.addWidget(QPushButton("Action"))
    central = QWidget()
    column = QVBoxLayout(central)
    column.setContentsMargins(0, 0, 0, 0)
    column.addWidget(header)
    column.addWidget(QLabel("content"), 1)
    window.setCentralWidget(central)
    chrome = wc.attach(window, header)
    assert chrome is not None
    window.resize(900, 600)
    window.show()
    end = time.time() + 1
    while time.time() < end:
        app.processEvents()

    user32 = ctypes.WinDLL("user32")
    message_args = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_size_t, ctypes.c_ssize_t]
    user32.SendMessageW.argtypes = message_args
    user32.SendMessageW.restype = ctypes.c_ssize_t
    user32.PostMessageW.argtypes = message_args
    user32.GetWindowRect.argtypes = [ctypes.c_void_p, ctypes.POINTER(wintypes.RECT)]
    user32.GetClientRect.argtypes = [ctypes.c_void_p, ctypes.POINTER(wintypes.RECT)]
    user32.IsZoomed.argtypes = [ctypes.c_void_p]
    handle = int(window.winId())
    outer, inner = wintypes.RECT(), wintypes.RECT()
    user32.GetWindowRect(handle, ctypes.byref(outer))
    user32.GetClientRect(handle, ctypes.byref(inner))
    scale = window.devicePixelRatioF()

    def ask(x: int, y: int) -> int:
        return user32.SendMessageW(handle, 0x84, 0, (x & 0xFFFF) | ((y & 0xFFFF) << 16))

    def pump(seconds: float) -> None:
        end = time.time() + seconds
        while time.time() < end:
            app.processEvents()

    def point(widget, dx=0.5, dy=0.5):
        spot = widget.mapTo(window, QPoint(int(widget.width() * dx), int(widget.height() * dy)))
        return outer.left + round(spot.x() * scale), outer.top + round(spot.y() * scale)

    failures = []

    def check(name, got, want):
        if got != want:
            failures.append(f"{name}: got {got}, wanted {want}")

    # no native caption is left: the client area is the whole window
    check("client width", inner.right - inner.left, outer.right - outer.left)
    check("client height", inner.bottom - inner.top, outer.bottom - outer.top)
    middle = (outer.left + outer.right) // 2
    check("left edge", ask(outer.left + 1, outer.top + 300), wc.HTLEFT)
    check("right edge", ask(outer.right - 2, outer.top + 300), wc.HTRIGHT)
    check("bottom edge", ask(middle, outer.bottom - 2), wc.HTBOTTOM)
    check("top-left corner", ask(outer.left + 1, outer.top + 1), wc.HTTOPLEFT)
    check("empty bar drags", ask(middle, outer.top + 24), wc.HTCAPTION)
    check("action button", ask(*point(row.itemAt(2).widget())), wc.HTCLIENT)
    check("maximise button", ask(*point(chrome.buttons.maximize)), wc.HTMAXBUTTON)
    check("close button", ask(*point(chrome.buttons.close_button)), wc.HTCLIENT)
    check("content", ask(middle, outer.top + 300), wc.HTCLIENT)

    # a press and release on the maximise button maximise, then restore
    x, y = point(chrome.buttons.maximize)
    packed = (x & 0xFFFF) | ((y & 0xFFFF) << 16)
    user32.PostMessageW(handle, 0xA1, wc.HTMAXBUTTON, packed)
    pump(0.2)
    user32.PostMessageW(handle, 0xA2, wc.HTMAXBUTTON, packed)
    pump(1.2)
    check("maximised", bool(user32.IsZoomed(handle)), True)
    user32.GetClientRect(handle, ctypes.byref(inner))
    screen = window.screen().availableGeometry()
    check("maximised client width", inner.right - inner.left, round(screen.width() * scale))
    check("maximised client height", inner.bottom - inner.top, round(screen.height() * scale))
    user32.GetWindowRect(handle, ctypes.byref(outer))
    check("no resize edge when maximised", ask(outer.left + 1, outer.top + 300), wc.HTCLIENT)
    window.showNormal()
    pump(1.0)
    check("restored", bool(user32.IsZoomed(handle)), False)

    if failures:
        print("\\n".join(failures))
        sys.exit(1)
    """
)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows draws and hit-tests this")
def test_a_real_window_answers_hit_tests_and_maximises() -> None:
    env = {key: value for key, value in os.environ.items() if key != "QT_QPA_PLATFORM"}
    done = subprocess.run(
        [sys.executable, "-c", REAL_WINDOW], env=env, capture_output=True, text=True, timeout=90
    )
    assert done.returncode == 0, done.stdout + done.stderr
