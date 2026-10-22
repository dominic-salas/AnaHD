"""The clip's timeline: where the preview is, and the cut markers.

CUT drops a cut region at the playhead; dragging either edge widens or narrows it;
clicking a cut selects it and Delete removes it. Overlapping cuts merge. Clicking or
dragging anywhere else moves the playhead. A cut at the very start or end trims the clip.
"""

from __future__ import annotations

from collections.abc import Sequence

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QKeyEvent, QMouseEvent, QPainter, QPaintEvent, QPen
from PySide6.QtWidgets import QWidget

from anahd.app import theme
from anahd.cuts import Cut, normalise

NEW_CUT = 2.0  # seconds: the width of a fresh cut marker
EDGE_GRAB = 7  # pixels around a cut's edge that pick up the edge
MIN_CUT = 1 / 60  # a cut is at least one frame wide


def skip_cuts(t: float, cuts: Sequence[Cut]) -> float:
    """Where playback goes on from ``t``: the end of the cut ``t`` falls in, else ``t``."""
    for a, b in cuts:
        if a <= t < b:
            return b
    return t


def kept_seconds(cuts: Sequence[Cut], duration: float) -> float:
    return duration - sum(b - a for a, b in normalise(cuts, duration))


class Timeline(QWidget):
    seeked = Signal(float)  # the user moved the playhead (seconds)
    cuts_changed = Signal(list)  # list[Cut], normalised
    selection_changed = Signal(object)  # index of the selected cut, or None

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumHeight(54)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.duration = 0.0
        self.playhead = 0.0
        self.cuts: list[Cut] = []
        self.selected: int | None = None
        self.editable = True
        self._drag: tuple[str, int] | None = None  # ("start" | "end", cut index) or ("seek", -1)
        self.palette_ = theme.DARK

    # ----- model -----

    def load(self, duration: float, cuts: Sequence[Cut], editable: bool) -> None:
        self.duration = max(duration, 0.0)
        self.cuts = normalise(cuts, self.duration)
        self.playhead = min(self.playhead, self.duration)
        self.editable = editable
        self.select(None)

    def set_playhead(self, t: float) -> None:
        self.playhead = min(max(t, 0.0), self.duration)
        self.update()

    def add_cut_at_playhead(self) -> None:
        """A NEW_CUT-second cut from the playhead (or ending at the clip's end)."""
        if not self.editable or self.duration <= 0:
            return
        a = min(self.playhead, max(self.duration - NEW_CUT, 0.0))
        b = min(a + NEW_CUT, self.duration)
        self._set_cuts([*self.cuts, (a, b)], select_at=self.playhead)

    def set_cut(self, index: int, start: float, end: float) -> None:
        """Move cut ``index`` to exactly (start, end) (the cut editor's fields and buttons)."""
        if not self.editable or not 0 <= index < len(self.cuts):
            return
        start = min(max(start, 0.0), self.duration)
        end = min(max(end, start + MIN_CUT), self.duration)
        if end - start < MIN_CUT:
            return
        cuts = list(self.cuts)
        cuts[index] = (start, end)
        self._set_cuts(cuts, select_at=(start + end) / 2)

    def select(self, index: int | None) -> None:
        self.selected = index
        self.update()
        self.selection_changed.emit(index)

    def remove_selected(self) -> None:
        if self.editable and self.selected is not None:
            cuts = [c for i, c in enumerate(self.cuts) if i != self.selected]
            self.selected = None
            self._set_cuts(cuts)

    def _set_cuts(self, cuts: Sequence[Cut], select_at: float | None = None) -> None:
        self.cuts = normalise(cuts, self.duration)
        if select_at is not None:
            self.selected = next(
                (i for i, (a, b) in enumerate(self.cuts) if a <= select_at <= b), None
            )
        elif self.selected is not None and self.selected >= len(self.cuts):
            self.selected = None
        self.update()
        self.cuts_changed.emit(list(self.cuts))
        self.selection_changed.emit(self.selected)

    # ----- geometry -----

    def _track(self) -> QRectF:
        return QRectF(8, 14, max(self.width() - 16, 1), self.height() - 28)

    def _x(self, t: float) -> float:
        track = self._track()
        return track.left() + track.width() * (t / self.duration if self.duration else 0.0)

    def _t(self, x: float) -> float:
        track = self._track()
        share = (x - track.left()) / track.width()
        return min(max(share, 0.0), 1.0) * self.duration

    def _hit(self, x: float) -> tuple[str, int] | None:
        """The cut edge (or the cut) under ``x``."""
        for i, (a, b) in enumerate(self.cuts):
            if abs(x - self._x(a)) <= EDGE_GRAB:
                return "start", i
            if abs(x - self._x(b)) <= EDGE_GRAB:
                return "end", i
        for i, (a, b) in enumerate(self.cuts):
            if self._x(a) < x < self._x(b):
                return "inside", i
        return None

    # ----- input -----

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self.duration <= 0:
            return
        x = event.position().x()
        hit = self._hit(x) if self.editable else None
        if hit is not None and hit[0] in ("start", "end"):
            self._drag = hit
            self.select(hit[1])
        elif hit is not None:
            self._drag = None
            self.select(hit[1])
        else:
            self._drag = ("seek", -1)
            self.select(None)
            self._seek(x)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        x = event.position().x()
        if self._drag is None:
            hit = self._hit(x) if self.editable else None
            edge = hit is not None and hit[0] in ("start", "end")
            self.setCursor(Qt.CursorShape.SizeHorCursor if edge else Qt.CursorShape.ArrowCursor)
            return
        kind, i = self._drag
        if kind == "seek":
            self._seek(x)
            return
        t = self._t(x)
        a, b = self.cuts[i]
        if kind == "start":
            self.cuts[i] = (min(t, b - MIN_CUT), b)
        else:
            self.cuts[i] = (a, max(t, a + MIN_CUT))
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        drag, self._drag = self._drag, None
        if drag is not None and drag[0] in ("start", "end"):
            a, b = self.cuts[drag[1]]
            self._set_cuts(self.cuts, select_at=(a + b) / 2)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self.remove_selected()
        else:
            super().keyPressEvent(event)

    def _seek(self, x: float) -> None:
        self.set_playhead(self._t(x))
        self.seeked.emit(self.playhead)

    # ----- drawing -----

    def paintEvent(self, event: QPaintEvent) -> None:
        p = self.palette_
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        track = self._track()
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(p.surface_2))
        painter.drawRoundedRect(track, 4, 4)
        accent = QColor(p.accent)
        for i, (a, b) in enumerate(self.cuts):
            rect = QRectF(self._x(a), track.top(), max(self._x(b) - self._x(a), 2), track.height())
            fill = QColor(accent)
            fill.setAlpha(170 if i == self.selected else 95)
            painter.setBrush(fill)
            painter.drawRect(rect)
            # hatching: cut parts read as removed
            painter.save()
            painter.setClipRect(rect)
            painter.setPen(QPen(QColor(0, 0, 0, 70), 1))
            x = rect.left() - rect.height()
            while x < rect.right():
                painter.drawLine(QPointF(x, rect.bottom()), QPointF(x + rect.height(), rect.top()))
                x += 8
            painter.restore()
            painter.setPen(Qt.PenStyle.NoPen)
            if i == self.selected:  # edge handles
                painter.setBrush(QColor("#ffffff"))
                for edge in (rect.left(), rect.right()):
                    painter.drawRoundedRect(
                        QRectF(edge - 2, track.top() - 4, 4, track.height() + 8), 2, 2
                    )
        x = self._x(self.playhead)
        painter.setPen(QPen(QColor(p.text), 2))
        painter.drawLine(QPointF(x, 4), QPointF(x, self.height() - 4))
        painter.end()
