"""Colour the native Windows title bar to match the app theme.

The window keeps its native frame, so resize, Snap Layouts and the usual shortcuts keep
working. DWM (the Windows compositor) is asked to draw the caption, its text and the window
border in the theme's colours, and to use its dark controls when the theme is dark.

Only Windows has this; everywhere else, and on Windows builds that lack an attribute, every
call quietly does nothing.
"""

from __future__ import annotations

import ctypes
import sys
from dataclasses import dataclass

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication, QWidget

DWMWA_USE_IMMERSIVE_DARK_MODE = 20
DWMWA_BORDER_COLOR = 34
DWMWA_CAPTION_COLOR = 35
DWMWA_TEXT_COLOR = 36

__all__ = [
    "DWMWA_BORDER_COLOR",
    "DWMWA_CAPTION_COLOR",
    "DWMWA_TEXT_COLOR",
    "DWMWA_USE_IMMERSIVE_DARK_MODE",
    "TitleColours",
    "apply_to_window",
    "colorref",
    "install",
    "is_dark",
]


@dataclass(frozen=True)
class TitleColours:
    """Hex colours (``#rrggbb``) for the caption, its text and the window border."""

    background: str
    text: str
    border: str


def colorref(colour: str) -> int:
    """A hex colour as the Windows COLORREF value, 0x00BBGGRR."""
    value = QColor(colour)
    return value.red() | (value.green() << 8) | (value.blue() << 16)


def is_dark(colour: str) -> bool:
    """Whether a background colour is dark enough to want light title bar controls."""
    return QColor(colour).lightness() < 128


def _supported() -> bool:
    return sys.platform == "win32"


def _set_attribute(handle: int, attribute: int, value: int) -> bool:
    """Set one DWM window attribute; False when this Windows build does not know it."""
    try:
        dwm = ctypes.WinDLL("dwmapi")  # type: ignore[attr-defined,unused-ignore]
        setter = dwm.DwmSetWindowAttribute
        setter.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.c_void_p,
            ctypes.c_uint,
        ]
        setter.restype = ctypes.c_long
        data = ctypes.c_int(value)
        return int(setter(handle, attribute, ctypes.byref(data), ctypes.sizeof(data))) == 0
    except (AttributeError, OSError, ValueError, ctypes.ArgumentError):
        return False


def apply_to_window(widget: QWidget, colours: TitleColours) -> bool:
    """Colour the title bar of one top-level window. False when nothing was applied."""
    if not _supported() or not widget.isWindow():
        return False
    try:
        handle = int(widget.winId())
    except (RuntimeError, TypeError, ValueError):
        return False
    dark = 1 if is_dark(colours.background) else 0
    results = (
        _set_attribute(handle, DWMWA_USE_IMMERSIVE_DARK_MODE, dark),
        _set_attribute(handle, DWMWA_CAPTION_COLOR, colorref(colours.background)),
        _set_attribute(handle, DWMWA_TEXT_COLOR, colorref(colours.text)),
        _set_attribute(handle, DWMWA_BORDER_COLOR, colorref(colours.border)),
    )
    return any(results)


class _TitleBarStyler(QObject):
    """Colours every window and dialog the app shows, and recolours them on a theme change."""

    def __init__(self, colours: TitleColours) -> None:
        super().__init__()
        self.colours = colours

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if (
            event.type() == QEvent.Type.Show
            and isinstance(watched, QWidget)
            and watched.windowType() in (Qt.WindowType.Window, Qt.WindowType.Dialog)
        ):
            apply_to_window(watched, self.colours)
        return False

    def recolour(self, colours: TitleColours) -> None:
        self.colours = colours
        for widget in QApplication.topLevelWidgets():
            if widget.isVisible():
                apply_to_window(widget, colours)


_styler: _TitleBarStyler | None = None


def install(app: QApplication, colours: TitleColours) -> None:
    """Colour title bars with ``colours`` now and for every window shown later.

    Calling it again (a theme switch) recolours the windows already open.
    """
    global _styler
    if not _supported():
        return
    if _styler is None:
        _styler = _TitleBarStyler(colours)
    app.installEventFilter(_styler)  # installing it again is harmless
    _styler.recolour(colours)
