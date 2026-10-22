"""Before/after preview: the same frame as a plain upscale and as AnaHD's output, split
by a line that can be dragged across the picture, with a slider to pick the moment."""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRect, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPaintEvent, QPen, QPixmap
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSlider, QStyle, QVBoxLayout, QWidget

from anahd.app import theme


class Wipe(QWidget):
    """Before on the left of the line, after on the right; drag to move the line."""

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(320, 240)
        self.setMouseTracking(True)
        self._before: QPixmap | None = None
        self._after: QPixmap | None = None
        self._split = 0.5
        self._message = "Pick a clip to compare"
        self.palette_ = theme.DARK
        self.fonts = theme.Fonts("Segoe UI", "Consolas", None)

    def show_pair(self, before: QPixmap | None, after: QPixmap | None, message: str = "") -> None:
        self._before, self._after, self._message = before, after, message
        self.update()

    def _picture(self) -> QRect:
        """Where the frames are drawn: as large as fits, in their own shape."""
        ref = self._before or self._after
        if ref is None or ref.isNull():
            return self.rect()
        size = ref.size().scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio)
        return QRect(
            (self.width() - size.width()) // 2,
            (self.height() - size.height()) // 2,
            size.width(),
            size.height(),
        )

    def _move(self, x: float) -> None:
        box = self._picture()
        self._split = min(max((x - box.left()) / max(box.width(), 1), 0.0), 1.0)
        self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        self._move(event.position().x())

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if event.buttons() & Qt.MouseButton.LeftButton:
            self._move(event.position().x())
        box = self._picture()
        near = abs(event.position().x() - (box.left() + self._split * box.width())) < 12
        self.setCursor(Qt.CursorShape.SplitHCursor if near else Qt.CursorShape.ArrowCursor)

    def paintEvent(self, event: QPaintEvent) -> None:
        p = self.palette_
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.fillRect(self.rect(), QColor(p.surface))
        box = self._picture()
        mono = theme.mono_font(self.fonts, 8)
        painter.setFont(mono)
        if self._before is None and self._after is None:
            painter.setPen(QColor(p.faint))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._message)
            painter.end()
            return
        cut = box.left() + round(self._split * box.width())
        if self._before is not None:
            painter.save()
            painter.setClipRect(QRect(box.left(), box.top(), cut - box.left(), box.height()))
            painter.drawPixmap(box, self._before)
            painter.restore()
        right = QRect(cut, box.top(), box.right() - cut + 1, box.height())
        if self._after is not None:
            painter.save()
            painter.setClipRect(right)
            painter.drawPixmap(box, self._after)
            painter.restore()
        else:
            painter.fillRect(right, QColor(p.surface_2))
            painter.setPen(QColor(p.faint))
            painter.drawText(
                right, Qt.AlignmentFlag.AlignCenter, self._message or "NOT CLEANED YET"
            )
        painter.setPen(QPen(QColor("#ffffff"), 2))
        painter.drawLine(cut, box.top(), cut, box.bottom())
        painter.setBrush(QColor("#ffffff"))
        painter.drawEllipse(QPointF(cut, box.center().y()), 5, 5)

        def tag(text: str, left: bool) -> None:
            width = painter.fontMetrics().horizontalAdvance(text) + 14
            x = box.left() + 10 if left else box.right() - width - 10
            r = QRectF(x, box.top() + 10, width, 20)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(0, 0, 0, 160))
            painter.drawRoundedRect(r, 4, 4)
            painter.setPen(QColor("#ededed"))
            painter.drawText(r, Qt.AlignmentFlag.AlignCenter, text)

        tag("BEFORE  PLAIN UPSCALE", True)
        if self._after is not None:
            tag("AFTER  ANAHD", False)
        painter.end()


class JumpSlider(QSlider):
    """A slider that jumps to where it is clicked (Qt's own steps a page at a time), and
    can be dragged from there."""

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            groove = (
                self.width() if self.orientation() == Qt.Orientation.Horizontal else self.height()
            )
            pos = (
                event.position().x()
                if self.orientation() == Qt.Orientation.Horizontal
                else event.position().y()
            )
            handle = self.style().pixelMetric(QStyle.PixelMetric.PM_SliderLength, None, self)
            value = QStyle.sliderValueFromPosition(
                self.minimum(), self.maximum(), round(pos - handle / 2), max(groove - handle, 1)
            )
            self.setValue(value)
        super().mousePressEvent(event)


class Preview(QWidget):
    """The wipe, a time slider under it and a caption."""

    moment = Signal(float)  # share of the clip, 0..1, when the slider is let go

    def __init__(self) -> None:
        super().__init__()
        self.wipe = Wipe()
        self.slider = JumpSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 1000)
        self.slider.setValue(300)
        self.slider.sliderReleased.connect(self._released)
        self.slider.valueChanged.connect(self._changed)
        self.time = QLabel()
        self.time.setProperty("role", "caption")
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.slider, 1)
        row.addWidget(self.time)
        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(8)
        column.addWidget(self.wipe, 1)
        column.addLayout(row)
        self.duration = 0.0

    @property
    def share(self) -> float:
        return self.slider.value() / 1000

    def set_duration(self, seconds: float) -> None:
        self.duration = seconds
        self._changed(self.slider.value())

    def _changed(self, value: int) -> None:
        t = value / 1000 * self.duration
        self.time.setText(f"{int(t // 60)}:{int(t % 60):02d}")
        if not self.slider.isSliderDown():
            self.moment.emit(value / 1000)

    def _released(self) -> None:
        self.moment.emit(self.share)
