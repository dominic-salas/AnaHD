"""Fine-tuning the selected cut: its start and end as typed times, half-second steps
(as in FPVEdit's trim), "start here" / "end here" from the playhead, and delete. Shown
under the timeline while a cut is selected; otherwise a hint says how to get there."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QLineEdit, QPushButton, QStackedWidget, QWidget

from anahd.cuts import parse_time

STEPS = (-0.5, 0.5)  # seconds per click, like FPVEdit's trim buttons


def mark(field: QLineEdit, invalid: bool) -> None:
    """Outline ``field`` in the accent colour while its text is not a usable time."""
    field.setProperty("invalid", invalid)
    field.style().unpolish(field)
    field.style().polish(field)


def fmt_time(seconds: float) -> str:
    """m:ss.mmm"""
    ms = round(seconds * 1000)
    return f"{ms // 60000}:{ms % 60000 // 1000:02d}.{ms % 1000:03d}"


class CutEditor(QStackedWidget):
    changed = Signal(int, float, float)  # cut index, start, end
    deleted = Signal(int)

    def __init__(self, playhead: Callable[[], float]) -> None:
        super().__init__()
        self._playhead = playhead
        self.index: int | None = None
        self.fps = 60.0
        self._cut = (0.0, 0.0)

        hint = QLabel("Select a cut on the timeline to set its start and end exactly")
        hint.setProperty("role", "eyebrow")
        self.addWidget(hint)

        panel = QWidget()
        row = QHBoxLayout(panel)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        self.title = QLabel()
        self.title.setProperty("role", "caption")
        row.addWidget(self.title)
        row.addSpacing(8)
        self.start = self._field("Start of the cut (m:ss.mmm or seconds)")
        self.end = self._field("End of the cut (m:ss.mmm or seconds)")
        for label, field, edge in (("Start", self.start, 0), ("End", self.end, 1)):
            caption = QLabel(label)
            caption.setProperty("role", "eyebrow")
            row.addWidget(caption)
            row.addWidget(field)
            for step in STEPS:
                way = "later" if step > 0 else "earlier"
                tip = f"Move the cut's {label.lower()} {abs(step):g} s {way}"
                row.addWidget(self._button(f"{step:+g}s", partial(self._nudge, edge, step), tip))
            here = "Start here" if edge == 0 else "End here"
            row.addWidget(self._button(here, partial(self._here, edge), "At the playhead"))
            row.addSpacing(10)
        row.addStretch(1)
        self.delete = self._button("Delete cut", self._delete, "Remove this cut")
        row.addWidget(self.delete)
        self.addWidget(panel)
        self.setFixedHeight(34)

    def _field(self, tip: str) -> QLineEdit:
        field = QLineEdit()
        field.setFixedWidth(84)
        field.setToolTip(tip)
        field.editingFinished.connect(self._typed)
        return field

    def _button(self, text: str, slot: Callable[[], object], tip: str) -> QPushButton:
        b = QPushButton(text)
        b.setToolTip(tip)
        b.clicked.connect(slot)
        return b

    # ----- shown cut -----

    def show_cut(self, index: int | None, count: int, cut: tuple[float, float], fps: float) -> None:
        self.index, self._cut, self.fps = index, cut, fps or 60.0
        if index is None:
            self.setCurrentIndex(0)
            return
        self.title.setText(f"Cut {index + 1} of {count}")
        self.start.setText(fmt_time(cut[0]))
        self.end.setText(fmt_time(cut[1]))
        for field in (self.start, self.end):
            mark(field, False)
        self.setCurrentIndex(1)

    # ----- edits -----

    def _emit(self, start: float, end: float) -> None:
        if self.index is not None:
            self.changed.emit(self.index, start, end)

    def _typed(self) -> None:
        if self.index is None:
            return
        try:
            start, end = parse_time(self.start.text()), parse_time(self.end.text())
        except ValueError:
            for field in (self.start, self.end):
                try:
                    parse_time(field.text())
                except ValueError:
                    mark(field, True)
            return
        if (start, end) != self._cut and end > start:
            self._emit(start, end)
        elif end <= start:
            mark(self.end, True)

    def _nudge(self, edge: int, step: float) -> None:
        a, b = self._cut
        if edge == 0:
            self._emit(a + step, b)
        else:
            self._emit(a, b + step)

    def _here(self, edge: int) -> None:
        t = self._playhead()
        a, b = self._cut
        if edge == 0 and t < b:
            self._emit(t, b)
        elif edge == 1 and t > a:
            self._emit(a, t)

    def _delete(self) -> None:
        if self.index is not None:
            self.deleted.emit(self.index)
