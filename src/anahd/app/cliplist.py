"""The clip list: one card per recording with a thumbnail, its length, and how its
cleaning is going (after FPV Editor's clip list)."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QModelIndex, QObject, QPersistentModelIndex, QRect, QSize, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QDragEnterEvent,
    QDragMoveEvent,
    QDropEvent,
    QPainter,
    QPaintEvent,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QListView,
    QListWidget,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
)

from anahd.app import theme

PATH = Qt.ItemDataRole.UserRole  # str: the recording
DETAILS = Qt.ItemDataRole.UserRole + 1  # str: length, size, rate
STATUS = Qt.ItemDataRole.UserRole + 2  # str: what is happening, upper case
SHARE = Qt.ItemDataRole.UserRole + 3  # float 0..1 while cleaning, else None
THUMB = Qt.ItemDataRole.UserRole + 4  # QPixmap
PROBLEM = Qt.ItemDataRole.UserRole + 5  # bool: the status is a problem (shown in red)

CARD_HEIGHT = 84
THUMB_SIZE = QSize(112, 84)


class ClipList(QListWidget):
    """The recordings, cleaned top to bottom: drop files or folders to add."""

    dropped = Signal(list)

    def __init__(self) -> None:
        super().__init__()
        self.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        fit_cards(self)
        self.card = ClipCard(self)
        self.setItemDelegate(self.card)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setAcceptDrops(True)
        self.setMouseTracking(True)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        if event.mimeData().hasUrls():
            self.dropped.emit([Path(url.toLocalFile()) for url in event.mimeData().urls()])
            event.acceptProposedAction()

    def paintEvent(self, event: QPaintEvent) -> None:
        super().paintEvent(event)
        if self.count() == 0:
            painter = QPainter(self.viewport())
            painter.setPen(self.palette().placeholderText().color())
            painter.drawText(
                self.viewport().rect(),
                Qt.AlignmentFlag.AlignCenter,
                "Drop DVR recordings (.mp4, .ts) or folders here",
            )


class ClipCard(QStyledItemDelegate):
    """Draws a clip as a card: number, thumbnail, name, details, status with progress."""

    def __init__(self, parent: QObject) -> None:
        super().__init__(parent)
        self.palette = theme.DARK
        self.fonts = theme.Fonts("Segoe UI", "Consolas", None)

    def set_theme(self, palette: theme.Palette, fonts: theme.Fonts) -> None:
        self.palette, self.fonts = palette, fonts

    def sizeHint(
        self, option: QStyleOptionViewItem, index: QModelIndex | QPersistentModelIndex
    ) -> QSize:
        return QSize(option.rect.width(), CARD_HEIGHT)

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionViewItem,
        index: QModelIndex | QPersistentModelIndex,
    ) -> None:
        p = self.palette
        state = option.state
        rect: QRect = option.rect.adjusted(0, 3, -1, -4)
        selected = bool(state & QStyle.StateFlag.State_Selected)
        hovered = bool(state & QStyle.StateFlag.State_MouseOver)
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(QColor(p.border_strong if selected or hovered else p.border), 1))
        painter.setBrush(QColor(p.surface_2 if selected else p.surface if hovered else p.bg))
        painter.drawRoundedRect(rect, 8, 8)

        mono = theme.mono_font(self.fonts, 8)
        painter.setFont(mono)
        painter.setPen(QColor(p.faint))
        number = QRect(rect.left() + 14, rect.top(), 28, rect.height())
        painter.drawText(number, Qt.AlignmentFlag.AlignVCenter, f"{index.row() + 1:02d}")

        thumb = QSize(THUMB_SIZE.width(), rect.height() - 16)
        box = QRect(number.right() + 4, rect.top() + 8, thumb.width(), thumb.height())
        painter.setPen(QPen(QColor(p.border), 1))
        painter.setBrush(QColor(p.surface_2))
        painter.drawRoundedRect(box, 4, 4)
        pixmap = index.data(THUMB)
        if isinstance(pixmap, QPixmap) and not pixmap.isNull():
            scaled = pixmap.scaled(
                box.size(),
                Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                Qt.TransformationMode.SmoothTransformation,
            )
            source = QRect(
                (scaled.width() - box.width()) // 2,
                (scaled.height() - box.height()) // 2,
                box.width(),
                box.height(),
            )
            painter.drawPixmap(box, scaled, source)

        # Three lines, each the full width: the name, the details, then the status. A long
        # status gets its own line, so it never squeezes the name.
        text_left = box.right() + 18
        width = rect.right() - 16 - text_left
        lines = [QRect(text_left, rect.top() + 9 + 21 * i, width, 21) for i in range(3)]
        left = Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft

        name_font = painter.font()
        name_font.setFamily(self.fonts.sans)
        name_font.setPointSizeF(10.5)
        name_font.setLetterSpacing(name_font.letterSpacingType(), 0)
        painter.setFont(name_font)
        painter.setPen(QColor(p.text))
        name = painter.fontMetrics().elidedText(
            Path(str(index.data(PATH))).name, Qt.TextElideMode.ElideMiddle, width
        )
        painter.drawText(lines[0], left, name)

        painter.setFont(mono)
        metrics = painter.fontMetrics()
        painter.setPen(QColor(p.faint))
        details = str(index.data(DETAILS) or "READING...")
        painter.drawText(
            lines[1], left, metrics.elidedText(details, Qt.TextElideMode.ElideRight, width)
        )
        status = str(index.data(STATUS) or "")
        painter.setPen(QColor(p.accent if index.data(PROBLEM) else p.muted))
        painter.drawText(
            lines[2], left, metrics.elidedText(status, Qt.TextElideMode.ElideRight, width)
        )

        share = index.data(SHARE)
        if isinstance(share, float):
            # Along the bottom of the card, under the text.
            track = QRect(box.left(), rect.bottom() - 5, rect.right() - 16 - box.left(), 2)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(p.border))
            painter.drawRect(track)
            painter.setBrush(QColor(p.accent))
            painter.drawRect(QRect(track.left(), track.top(), round(track.width() * share), 2))
        painter.restore()


def fit_cards(view: QListWidget) -> None:
    """Cards always as wide as the list: re-laid on every resize, never scrolled sideways."""
    view.setResizeMode(QListView.ResizeMode.Adjust)
    view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    view.setUniformItemSizes(True)


def pixmap_from_png(png: bytes) -> QPixmap | None:
    pixmap = QPixmap()
    return pixmap if png and pixmap.loadFromData(png) else None
