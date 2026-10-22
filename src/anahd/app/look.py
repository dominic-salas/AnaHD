"""Things drawn in code, so no image or font file has to ship: the app icon, the
dot-matrix title and the status dot (after FPV Editor's)."""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QIcon, QLinearGradient, QPainter, QPaintEvent, QPen, QPixmap
from PySide6.QtWidgets import QWidget

SIZES = (16, 24, 32, 48, 64, 128, 256)


def icon_pixmap(size: int) -> QPixmap:
    """A rounded tile: broken analog scan lines on the left turn into a clean picture on
    the right, with a white "HD" bar across."""
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    s = float(size)
    gradient = QLinearGradient(0, 0, s, s)
    gradient.setColorAt(0.0, QColor("#2b2b2b"))
    gradient.setColorAt(1.0, QColor("#0a0a0a"))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(gradient)
    tile = QRectF(0.04 * s, 0.04 * s, 0.92 * s, 0.92 * s)
    painter.drawRoundedRect(tile, 0.2 * s, 0.2 * s)
    # Scan lines, ragged on the left (analog), straight and even towards the right.
    lines = 5 if size < 48 else 7
    top, bottom = 0.22 * s, 0.78 * s
    step = (bottom - top) / (lines - 1)
    height = max(1.0, step * 0.42)
    for i in range(lines):
        y = top + i * step - height / 2
        jag = (0.06, -0.04, 0.09, -0.07, 0.03, 0.08, -0.05)[i % 7] * s
        painter.setBrush(QColor(229, 50, 45, 150 if i % 2 else 210))
        painter.drawRect(QRectF(0.16 * s + jag, y, 0.34 * s - jag, height))
        painter.setBrush(QColor(237, 237, 237, 235))
        painter.drawRect(QRectF(0.52 * s, y, 0.32 * s, height))
    painter.end()
    return pixmap


def app_icon() -> QIcon:
    icon = QIcon()
    for size in SIZES:
        icon.addPixmap(icon_pixmap(size))
    return icon


# A 5-dot-wide dot-matrix face for the title (FPV Editor's). Rows 0-1 hold ascenders,
# rows 2-6 the x-height and rows 7-8 descenders. Letters not listed are left blank.
ROWS = 9
GLYPHS: dict[str, tuple[str, ...]] = {
    "a": (".....", ".....", ".###.", "....#", ".####", "#...#", ".####", ".....", "....."),
    "d": ("....#", "....#", ".####", "#...#", "#...#", "#...#", ".####", ".....", "....."),
    "e": (".....", ".....", ".###.", "#...#", "#####", "#....", ".###.", ".....", "....."),
    "h": ("#....", "#....", "####.", "#...#", "#...#", "#...#", "#...#", ".....", "....."),
    "n": (".....", ".....", "####.", "#...#", "#...#", "#...#", "#...#", ".....", "....."),
    "o": (".....", ".....", ".###.", "#...#", "#...#", "#...#", ".###.", ".....", "....."),
    ".": (".....", ".....", ".....", ".....", ".....", ".....", "..#..", ".....", "....."),
}


class DotTitle(QWidget):
    """Large lowercase text drawn as a grid of dots; ``accent`` letters in the accent."""

    def __init__(self, text: str, cell: int = 7, accent: str = "") -> None:
        super().__init__()
        self._text = text.lower()
        self._accent_letters = accent
        self._cell = cell
        self._colour = QColor("#ededed")
        self._accent = QColor("#e5322d")
        self.setFixedSize((6 * len(self._text) - 1) * cell, ROWS * cell)

    def set_colours(self, colour: str, accent: str) -> None:
        self._colour, self._accent = QColor(colour), QColor(accent)
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        radius = self._cell * 0.36
        for index, letter in enumerate(self._text):
            rows = GLYPHS.get(letter)
            if rows is None:
                continue
            painter.setBrush(self._accent if letter in self._accent_letters else self._colour)
            for y, row in enumerate(rows):
                for x, bit in enumerate(row):
                    if bit == "#":
                        centre = QPointF((index * 6 + x + 0.5) * self._cell, (y + 0.5) * self._cell)
                        painter.drawEllipse(centre, radius, radius)
        painter.end()


class StatusDot(QWidget):
    """A small dot: still and grey when idle, pulsing in the accent colour while busy."""

    def __init__(self) -> None:
        super().__init__()
        self.setFixedSize(10, 10)
        self._idle = QColor("#6e6e6e")
        self._accent = QColor("#e5322d")
        self._busy = False
        self._phase = 0
        self._timer = QTimer(self)
        self._timer.setInterval(60)
        self._timer.timeout.connect(self._tick)

    def set_colours(self, idle: str, accent: str) -> None:
        self._idle, self._accent = QColor(idle), QColor(accent)
        self.update()

    def set_busy(self, busy: bool) -> None:
        self._busy = busy
        if busy:
            self._timer.start()
        else:
            self._timer.stop()
        self.update()

    def _tick(self) -> None:
        self._phase = (self._phase + 1) % 30
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        colour = QColor(self._accent if self._busy else self._idle)
        if self._busy:
            colour.setAlphaF(0.35 + 0.65 * abs(15 - self._phase) / 15)
        painter.setBrush(colour)
        painter.drawEllipse(QPointF(5, 5), 3.5, 3.5)
        painter.end()


class Spinner(QWidget):
    """A small turning arc, shown while a preview frame renders."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(22, 22)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._angle = 0
        self._colour = QColor("#e5322d")
        self._timer = QTimer(self)
        self._timer.setInterval(40)
        self._timer.timeout.connect(self._tick)
        self.hide()

    def spin(self, on: bool) -> None:
        if on:
            self._timer.start()
            self.show()
            self.raise_()
        else:
            self._timer.stop()
            self.hide()

    def _tick(self) -> None:
        self._angle = (self._angle + 24) % 360
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(QColor(0, 0, 0, 140))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QRectF(0, 0, 22, 22))
        pen = QPen(self._colour, 2.5)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawArc(QRectF(5, 5, 12, 12), -self._angle * 16, 270 * 16)
        painter.end()
