"""Let the app's own top bar be the top of the window, with no separate Windows title bar.

On Windows the native frame stays (resize, shadow, Aero Snap, animations and Snap Layouts keep
working) but its caption area is not drawn: the client area is the whole window, and the top
bar of the app holds the logo, an empty draggable area and the minimise, maximise and close
buttons, drawn here as small shapes. The real window title stays set, so the taskbar and
Alt-Tab still name the app.

This works by answering three window messages for the main window:

* ``WM_NCCALCSIZE`` makes the client area the whole window (less the border Windows hangs
  outside a maximised window),
* ``WM_NCHITTEST`` tells Windows which part is a resize edge, the draggable caption, the
  maximise button (so Windows 11 shows Snap Layouts on hover) or the app itself,
* ``WM_NCLBUTTONDOWN``/``UP`` on the maximise button toggle maximise.

Anywhere else, and in dialogs, nothing changes: ``attach`` does nothing off Windows.
"""

from __future__ import annotations

import ctypes
import sys
from collections.abc import Callable
from ctypes import wintypes
from functools import lru_cache

from PySide6.QtCore import (
    QAbstractNativeEventFilter,
    QByteArray,
    QCoreApplication,
    QEvent,
    QObject,
    QPoint,
    QPointF,
    QRect,
    QRectF,
    Qt,
)
from PySide6.QtGui import QColor, QEnterEvent, QPainter, QPaintEvent, QPen
from PySide6.QtWidgets import (
    QAbstractButton,
    QAbstractScrollArea,
    QAbstractSlider,
    QBoxLayout,
    QComboBox,
    QHBoxLayout,
    QLineEdit,
    QMainWindow,
    QSpinBox,
    QWidget,
)

WM_NCCALCSIZE = 0x0083
WM_NCHITTEST = 0x0084
WM_NCLBUTTONDOWN = 0x00A1
WM_NCLBUTTONUP = 0x00A2
WM_NCMOUSELEAVE = 0x02A2

HTCLIENT = 1
HTCAPTION = 2
HTMAXBUTTON = 9
HTLEFT = 10
HTRIGHT = 11
HTTOP = 12
HTTOPLEFT = 13
HTTOPRIGHT = 14
HTBOTTOM = 15
HTBOTTOMLEFT = 16
HTBOTTOMRIGHT = 17

CLIENT = "client"
CAPTION = "caption"
MAXIMIZE = "maximize"

BUTTON_WIDTH = 44
BUTTON_HEIGHT = 32
EDGE_PX = 6  # the grab zone of a resize edge, at 100% scale

__all__ = [
    "BUTTON_HEIGHT",
    "BUTTON_WIDTH",
    "CAPTION",
    "CLIENT",
    "MAXIMIZE",
    "CaptionButtons",
    "WindowChrome",
    "attach",
    "classify",
    "split_lparam",
]


def split_lparam(value: int) -> tuple[int, int]:
    """The signed screen x and y packed in a mouse message's LPARAM."""
    x = ctypes.c_short(value & 0xFFFF).value
    y = ctypes.c_short((value >> 16) & 0xFFFF).value
    return x, y


def classify(
    x: int,
    y: int,
    width: int,
    height: int,
    edge: int,
    resizable: bool,
    region: Callable[[int, int], str],
) -> int:
    """The WM_NCHITTEST answer for a point (window pixels), given the app's own region there.

    ``region`` maps the point to ``CLIENT``, ``CAPTION`` or ``MAXIMIZE``. Resize edges win over
    everything else, except that they are off while the window is maximised.
    """
    if resizable:
        left, right = x < edge, x >= width - edge
        top, bottom = y < edge, y >= height - edge
        if top and left:
            return HTTOPLEFT
        if top and right:
            return HTTOPRIGHT
        if bottom and left:
            return HTBOTTOMLEFT
        if bottom and right:
            return HTBOTTOMRIGHT
        if left:
            return HTLEFT
        if right:
            return HTRIGHT
        if top:
            return HTTOP
        if bottom:
            return HTBOTTOM
    found = region(x, y)
    if found == MAXIMIZE:
        return HTMAXBUTTON
    if found == CAPTION:
        return HTCAPTION
    return HTCLIENT


# ----- the three small buttons -----


class _GlyphButton(QAbstractButton):
    """One caption button: a quiet shape that gets a soft background under the pointer."""

    def __init__(self, kind: str, parent: QWidget) -> None:
        super().__init__(parent)
        self.kind = kind  # "minimize", "maximize" or "close"
        self.restored = False  # the maximise button shows the restore shape while maximised
        self.hovered = False
        self.setFixedSize(BUTTON_WIDTH, BUTTON_HEIGHT)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setCursor(Qt.CursorShape.ArrowCursor)
        names = {"minimize": "Minimize", "maximize": "Maximize", "close": "Close"}
        self.setAccessibleName(names[kind])

    def set_hovered(self, value: bool) -> None:
        if value != self.hovered:
            self.hovered = value
            self.update()

    def enterEvent(self, event: QEnterEvent) -> None:
        self.set_hovered(True)
        super().enterEvent(event)

    def leaveEvent(self, event: QEvent) -> None:
        self.set_hovered(False)
        super().leaveEvent(event)

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        text = self.palette().color(self.foregroundRole())
        text.setAlphaF(0.7)
        if self.hovered or self.isDown():
            backdrop = QColor(text)
            backdrop.setAlphaF(0.16 if self.isDown() else 0.10)
            if self.kind == "close":
                backdrop = QColor("#c4554f")
                backdrop.setAlphaF(0.85 if self.isDown() else 1.0)
                text = QColor("#ffffff")
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(backdrop)
            painter.drawRoundedRect(QRectF(self.rect()).adjusted(2, 0, -2, 0), 6, 6)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(text, 1.0))
        cx, cy = self.width() / 2, self.height() / 2
        if self.kind == "minimize":
            painter.drawLine(QPointF(cx - 5, cy + 0.5), QPointF(cx + 5, cy + 0.5))
        elif self.kind == "maximize" and not self.restored:
            painter.drawRect(QRectF(cx - 4.5, cy - 4.5, 9, 9))
        elif self.kind == "maximize":
            painter.drawRect(QRectF(cx - 4.5, cy - 2.5, 7, 7))
            painter.drawLine(QPointF(cx - 2.5, cy - 4.5), QPointF(cx + 4.5, cy - 4.5))
            painter.drawLine(QPointF(cx + 4.5, cy - 4.5), QPointF(cx + 4.5, cy + 2.5))
        else:
            painter.drawLine(QPointF(cx - 4.5, cy - 4.5), QPointF(cx + 4.5, cy + 4.5))
            painter.drawLine(QPointF(cx - 4.5, cy + 4.5), QPointF(cx + 4.5, cy - 4.5))


class CaptionButtons(QWidget):
    """Minimise, maximise (or restore) and close, 44 x 32 each."""

    def __init__(self, window: QMainWindow, header: QWidget) -> None:
        super().__init__(header)
        self._window = window
        self.minimize = _GlyphButton("minimize", self)
        self.maximize = _GlyphButton("maximize", self)
        self.close_button = _GlyphButton("close", self)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        for button in (self.minimize, self.maximize, self.close_button):
            row.addWidget(button)
        self.setFixedSize(3 * BUTTON_WIDTH, BUTTON_HEIGHT)
        self.minimize.clicked.connect(window.showMinimized)
        self.maximize.clicked.connect(self.toggle_maximize)
        self.close_button.clicked.connect(window.close)

    def toggle_maximize(self) -> None:
        if self._window.isMaximized():
            self._window.showNormal()
        else:
            self._window.showMaximized()

    def sync(self) -> None:
        maximized = self._window.isMaximized()
        if self.maximize.restored != maximized:
            self.maximize.restored = maximized
            self.maximize.update()


# ----- the window side -----


def _is_interactive(widget: QWidget | None, stop: QWidget) -> bool:
    """Whether the widget (or a parent below ``stop``) is something to click, not to drag."""
    kinds = (
        QAbstractButton,
        QAbstractSlider,
        QAbstractScrollArea,
        QComboBox,
        QLineEdit,
        QSpinBox,
    )
    while widget is not None and widget is not stop:
        if isinstance(widget, kinds) or bool(widget.property("captionClient")):
            return True
        widget = widget.parentWidget()
    return False


class WindowChrome(QObject):
    """The state and the hit test for one main window drawn to its top edge."""

    def __init__(
        self, window: QMainWindow, header: QWidget, buttons: CaptionButtons, right_margin: int
    ) -> None:
        super().__init__(window)
        self.right_margin = right_margin
        self.window = window
        self.header = header
        self.buttons = buttons
        window.installEventFilter(self)
        header.installEventFilter(self)
        self.place()

    def place(self) -> None:
        """Centre the buttons on the top bar's own line, at its right padding."""
        x = self.header.width() - self.right_margin - self.buttons.width()
        self.buttons.move(x, (self.header.height() - self.buttons.height()) // 2)
        self.buttons.raise_()

    def handle(self) -> int:
        return int(self.window.effectiveWinId())  # never creates the native window

    def region(self, x: int, y: int) -> str:
        """The app's own part of the window at a point in the window's logical pixels."""
        point = QPoint(x, y)
        top = self.header.mapTo(self.window, QPoint(0, 0))
        if not QRect(top, self.header.size()).contains(point):
            return CLIENT
        maximize = self.buttons.maximize
        corner = maximize.mapTo(self.window, QPoint(0, 0))
        if QRect(corner, maximize.size()).contains(point):
            return MAXIMIZE
        if _is_interactive(self.window.childAt(point), self.header):
            return CLIENT
        return CAPTION

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if watched is self.header and event.type() in (QEvent.Type.Resize, QEvent.Type.Show):
            self.place()
        if watched is self.window:
            kind = event.type()
            if kind == QEvent.Type.Show:
                _refresh_frame(self.handle())
            elif kind == QEvent.Type.WindowStateChange:
                self.buttons.sync()
        return False

    def set_maximize_hover(self, value: bool) -> None:
        self.buttons.maximize.set_hovered(value)


# ----- the Windows messages -----


class _CalcSizeParams(ctypes.Structure):
    _fields_ = [("rgrc", wintypes.RECT * 3), ("lppos", ctypes.c_void_p)]


class _TrackMouse(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("hwndTrack", ctypes.c_void_p),
        ("dwHoverTime", wintypes.DWORD),
    ]


@lru_cache(maxsize=1)
def _user32() -> ctypes.WinDLL:  # type: ignore[name-defined,unused-ignore]
    lib = ctypes.WinDLL("user32")  # type: ignore[attr-defined,unused-ignore]
    void = ctypes.c_void_p
    lib.IsZoomed.argtypes = [void]
    lib.IsZoomed.restype = wintypes.BOOL
    lib.GetWindowRect.argtypes = [void, ctypes.POINTER(wintypes.RECT)]
    lib.GetWindowRect.restype = wintypes.BOOL
    lib.GetDpiForWindow.argtypes = [void]
    lib.GetDpiForWindow.restype = wintypes.UINT
    lib.GetSystemMetricsForDpi.argtypes = [ctypes.c_int, wintypes.UINT]
    lib.GetSystemMetricsForDpi.restype = ctypes.c_int
    lib.SetWindowPos.argtypes = [void, void] + [ctypes.c_int] * 4 + [wintypes.UINT]
    lib.SetWindowPos.restype = wintypes.BOOL
    lib.TrackMouseEvent.argtypes = [ctypes.POINTER(_TrackMouse)]
    lib.TrackMouseEvent.restype = wintypes.BOOL
    return lib


def _dpi(handle: int) -> int:
    return int(_user32().GetDpiForWindow(handle)) or 96


def _frame_inset(handle: int) -> tuple[int, int]:
    """The width and height of the border Windows hangs outside a maximised window."""
    dpi = _dpi(handle)
    user32 = _user32()
    padded = user32.GetSystemMetricsForDpi(92, dpi)  # SM_CXPADDEDBORDER
    return (
        user32.GetSystemMetricsForDpi(32, dpi) + padded,  # SM_CXSIZEFRAME
        user32.GetSystemMetricsForDpi(33, dpi) + padded,  # SM_CYSIZEFRAME
    )


def _refresh_frame(handle: int) -> None:
    """Make Windows ask for the client area again, now that the filter answers for it."""
    try:
        # SWP_NOSIZE | SWP_NOMOVE | SWP_NOZORDER | SWP_NOACTIVATE | SWP_FRAMECHANGED
        _user32().SetWindowPos(handle, None, 0, 0, 0, 0, 0x0001 | 0x0002 | 0x0004 | 0x0010 | 0x0020)
        # A one pixel frame extension keeps the drop shadow on Windows 10.
        margins = (ctypes.c_int * 4)(0, 0, 1, 0)  # left, right, top, bottom
        dwm = ctypes.WinDLL("dwmapi")  # type: ignore[attr-defined,unused-ignore]
        dwm.DwmExtendFrameIntoClientArea.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        dwm.DwmExtendFrameIntoClientArea(handle, ctypes.byref(margins))
    except (AttributeError, OSError, ValueError, ctypes.ArgumentError):
        pass


class _Filter(QAbstractNativeEventFilter):
    def __init__(self) -> None:
        super().__init__()
        self.chromes: list[WindowChrome] = []

    def _find(self, handle: int | None) -> WindowChrome | None:
        for chrome in self.chromes:
            try:
                if handle and chrome.handle() == handle:
                    return chrome
            except RuntimeError:  # the window is already gone
                continue
        return None

    def nativeEventFilter(
        self, eventType: QByteArray | bytes | bytearray | memoryview, message: int
    ) -> tuple[bool, int]:
        if eventType != b"windows_generic_MSG":
            return False, 0
        msg = wintypes.MSG.from_address(int(message))
        if msg.message not in (
            WM_NCCALCSIZE,
            WM_NCHITTEST,
            WM_NCLBUTTONDOWN,
            WM_NCLBUTTONUP,
            WM_NCMOUSELEAVE,
        ):
            return False, 0
        chrome = self._find(msg.hWnd)
        if chrome is None:
            return False, 0
        return _answer(chrome, int(msg.hWnd or 0), msg.message, msg.wParam, msg.lParam)


def _answer(
    chrome: WindowChrome, handle: int, message: int, wparam: int, lparam: int
) -> tuple[bool, int]:
    user32 = _user32()
    if message == WM_NCCALCSIZE:
        if wparam and user32.IsZoomed(handle):
            params = _CalcSizeParams.from_address(lparam)
            inset_x, inset_y = _frame_inset(handle)
            rect = params.rgrc[0]
            rect.left += inset_x
            rect.right -= inset_x
            rect.top += inset_y
            rect.bottom -= inset_y
        return True, 0
    if message == WM_NCHITTEST:
        x, y = split_lparam(lparam)
        rect = wintypes.RECT()
        user32.GetWindowRect(handle, ctypes.byref(rect))
        scale = max(chrome.window.devicePixelRatioF(), 0.5)
        edge = max(2, round(EDGE_PX * _dpi(handle) / 96))
        hit = classify(
            x - rect.left,
            y - rect.top,
            rect.right - rect.left,
            rect.bottom - rect.top,
            edge,
            not user32.IsZoomed(handle),
            lambda px, py: chrome.region(round(px / scale), round(py / scale)),
        )
        if hit == HTMAXBUTTON:
            if not chrome.buttons.maximize.hovered:
                track = _TrackMouse(ctypes.sizeof(_TrackMouse), 0x2 | 0x10, handle, 0)
                user32.TrackMouseEvent(ctypes.byref(track))  # TME_LEAVE | TME_NONCLIENT
            chrome.set_maximize_hover(True)
        else:
            chrome.set_maximize_hover(False)
        return True, hit
    if message == WM_NCLBUTTONDOWN and wparam == HTMAXBUTTON:
        return True, 0
    if message == WM_NCLBUTTONUP and wparam == HTMAXBUTTON:
        chrome.buttons.toggle_maximize()
        return True, 0
    if message == WM_NCMOUSELEAVE:
        chrome.set_maximize_hover(False)
    return False, 0


_filter: _Filter | None = None


def _supported() -> bool:
    return sys.platform == "win32"


def attach(window: QMainWindow, header: QWidget) -> WindowChrome | None:
    """Draw ``header`` (the app's top bar, already at the top of the window) as the window's
    top edge, with the caption buttons on the bar's own centre line at its right padding.

    Returns None, changing nothing, when this is not Windows.
    """
    global _filter
    if not _supported():
        return None
    layout = header.layout()
    right = 16
    if isinstance(layout, QBoxLayout):  # room for the buttons after the bar's own content
        right = layout.contentsMargins().right()
        layout.addSpacing(3 * BUTTON_WIDTH)
    chrome = WindowChrome(window, header, CaptionButtons(window, header), right)
    if _filter is None:
        _filter = _Filter()
        app = QCoreApplication.instance()
        if app is not None:
            app.installNativeEventFilter(_filter)
    _filter.chromes.append(chrome)
    return chrome
