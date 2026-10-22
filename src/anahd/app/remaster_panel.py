"""The remaster settings panel, generated from the settings' own metadata
(``anahd.remaster.schema()`` and ``GROUP_INFO``): nothing here names a setting.

Per kind: a slider (with its value and unit), a choice, a switch, a file. A setting with
a ``toggle`` shows as a checkbox that switches it between ``off`` and its last value
(``on`` the first time) and reveals its control when ticked. ``experimental`` switches are
labelled so; a file or switch that ``needs`` a switch is greyed out until it is on. A
setting can show its speed cost (:meth:`RemasterPanel.set_costs`). Groups fold; a
``collapsed`` group starts folded.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from PySide6.QtCore import QSize, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from anahd.remaster import GROUP_INFO, GROUPS, RemasterParams, schema

MIN_WIDTH = 300  # the settings column's usual width; a wider row widens it
DEBOUNCE_MS = 120  # sliders send one change per pause, not one per pixel of drag


def _fmt(value: float, unit: str) -> str:
    return f"{value:g}{' ' + unit if unit and unit != '%' else unit}"


@dataclass
class _Control:
    item: dict[str, Any]
    get: Callable[[], Any]
    put: Callable[[Any], None]  # show a value without announcing it
    widgets: list[QWidget] = field(default_factory=list)  # greyed out together
    last_on: Any = None  # a toggle's value before it was switched off


class RemasterPanel(QScrollArea):
    changed = Signal(object)  # RemasterParams, after a short pause in the changes

    def __init__(self) -> None:
        super().__init__()
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFrameShape(QScrollArea.Shape.NoFrame)
        self._controls: dict[str, _Control] = {}
        self._costs: dict[str, QLabel] = {}
        self._quiet = False
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(DEBOUNCE_MS)
        self._timer.timeout.connect(lambda: self.changed.emit(self.params()))

        body = QWidget()
        column = QVBoxLayout(body)
        column.setContentsMargins(0, 0, 8, 0)
        column.setSpacing(6)
        items = schema()
        self.sections: dict[str, QWidget] = {}
        for group in GROUPS:
            members = [i for i in items if i["group"] == group]
            if not members:
                continue
            info = GROUP_INFO.get(group, {"title": group})
            header = QPushButton()
            header.setProperty("flat", True)
            header.setStyleSheet("text-align: left; padding: 8px 2px 4px 2px;")
            section = QWidget()
            rows = QVBoxLayout(section)
            rows.setContentsMargins(0, 0, 0, 6)
            rows.setSpacing(4)
            for item in members:
                rows.addWidget(self._build(item))
            title = str(info["title"])

            def fold(
                _: bool = False, s: QWidget = section, h: QPushButton = header, t: str = title
            ) -> None:
                s.setVisible(not s.isVisible())
                h.setText(("▾  " if s.isVisible() else "▸  ") + t)
                self.updateGeometry()  # an opened section may need more width

            section.setVisible(not info.get("collapsed", False))
            header.setText(("▾  " if section.isVisible() else "▸  ") + title)
            header.clicked.connect(fold)
            column.addWidget(header)
            column.addWidget(section)
            self.sections[group] = section
        self.reset_button = QPushButton("Reset to defaults")
        self.reset_button.clicked.connect(self.reset)
        column.addSpacing(8)
        column.addWidget(self.reset_button)
        self.speed = QLabel()
        self.speed.setWordWrap(True)
        self.speed.setProperty("role", "caption")
        column.addWidget(self.speed)
        column.addStretch(1)
        self.setWidget(body)
        self._update_needs()

    def minimumSizeHint(self) -> QSize:
        """Never narrower than the widest row (there is no sideways scrolling), so labels,
        values and buttons are not cut off whatever the font."""
        hint = super().minimumSizeHint()
        body = self.widget()
        if body is None:
            return hint
        width = body.minimumSizeHint().width() + self.verticalScrollBar().sizeHint().width()
        return QSize(max(hint.width(), width, MIN_WIDTH), hint.height())

    # ----- building -----

    def _build(self, item: dict[str, Any]) -> QWidget:
        kind = item["kind"]
        holder = QWidget()
        lay = QVBoxLayout(holder)
        lay.setContentsMargins(0, 2, 0, 2)
        lay.setSpacing(2)
        holder.setToolTip(str(item.get("help", "")))
        if kind == "slider":
            control = self._slider(item, lay)
        elif kind == "choice":
            control = self._choice(item, lay)
        elif kind == "switch":
            control = self._switch(item, lay)
        else:
            control = self._path(item, lay)
        self._controls[item["name"]] = control
        return holder

    def _cost(self, item: dict[str, Any]) -> QLabel:
        """The setting's speed cost, filled in by :meth:`set_costs` (hidden until then)."""
        label = QLabel()
        label.setProperty("role", "caption")
        label.hide()
        self._costs[item["name"]] = label
        return label

    def _toggle_box(self, item: dict[str, Any], lay: QVBoxLayout) -> QCheckBox:
        box = QCheckBox(str(item["toggle"]["label"]))
        box.setToolTip(str(item.get("help", "")))
        row = QHBoxLayout()
        row.addWidget(box)
        row.addStretch(1)
        row.addWidget(self._cost(item))
        lay.addLayout(row)
        return box

    def _slider(self, item: dict[str, Any], lay: QVBoxLayout) -> _Control:
        unit = str(item.get("unit", ""))
        top = QHBoxLayout()
        label = QLabel(str(item["label"]))
        value = QLabel()
        value.setProperty("role", "caption")
        top.addWidget(label)
        top.addStretch(1)
        slider = QSlider(Qt.Orientation.Horizontal)
        step = float(item.get("step", 1.0)) or 1.0
        slider.setRange(round(item["min"] / step), round(item["max"] / step))
        slider.setSingleStep(1)
        slider.setPageStep(max(1, round(10 / step)))
        toggle = item.get("toggle")
        box = self._toggle_box(item, lay) if toggle else None
        if box is None:
            top.addWidget(self._cost(item))
            top.addSpacing(8)
        top.addWidget(value)
        rows = QWidget()
        rlay = QVBoxLayout(rows)
        rlay.setContentsMargins(18 if box else 0, 0, 0, 0)
        rlay.setSpacing(0)
        rlay.addLayout(top)
        rlay.addWidget(slider)
        lay.addWidget(rows)

        def get() -> float:
            if box is not None and not box.isChecked():
                return float(toggle["off"])  # type: ignore[index]
            return slider.value() * step

        def show(v: float) -> None:
            value.setText(_fmt(v, unit))

        def put(v: Any) -> None:
            v = float(v)
            if box is not None:
                on = v != float(toggle["off"])  # type: ignore[index]
                box.setChecked(on)
                rows.setVisible(on)
                if on:
                    control.last_on = v
            if box is None or v != float(toggle["off"]):  # type: ignore[index]
                slider.setValue(round(v / step))
            show(get() if box is None else slider.value() * step)

        control = _Control(item, get, put, [label, slider, value])

        def moved(_v: int) -> None:
            show(slider.value() * step)
            self._edited()

        slider.valueChanged.connect(moved)
        if box is not None:

            def toggled(on: bool) -> None:
                if self._quiet:
                    return
                rows.setVisible(on)
                if on:
                    restore = control.last_on if control.last_on is not None else toggle["on"]  # type: ignore[index]
                    self._quiet = True
                    slider.setValue(round(float(restore) / step))
                    show(float(restore))
                    self._quiet = False
                else:
                    control.last_on = slider.value() * step
                self._edited()

            box.toggled.connect(toggled)
            control.widgets.append(box)
        return control

    def _choice(self, item: dict[str, Any], lay: QVBoxLayout) -> _Control:
        names = dict(item.get("names") or {})
        toggle = item.get("toggle")
        box = self._toggle_box(item, lay) if toggle else None
        combo = QComboBox()
        options = [c for c in item["choices"] if not toggle or c != toggle["off"]]
        for c in options:
            combo.addItem(names.get(c, c), c)
        label = QLabel(str(item["label"]))
        row = QWidget()
        rlay = QHBoxLayout(row)
        rlay.setContentsMargins(18 if box else 0, 0, 0, 0)
        if box is None:
            rlay.addWidget(label)
        rlay.addWidget(combo, 1)
        lay.addWidget(row)

        def get() -> str:
            if box is not None and not box.isChecked():
                return str(toggle["off"])  # type: ignore[index]
            return str(combo.currentData())

        def put(v: Any) -> None:
            on = box is None or v != toggle["off"]  # type: ignore[index]
            if box is not None:
                box.setChecked(on)
                row.setVisible(on)
            if on:
                i = combo.findData(v)
                combo.setCurrentIndex(max(i, 0))

        control = _Control(item, get, put, [combo, label])
        combo.currentIndexChanged.connect(lambda _i: self._edited())
        if box is not None:

            def toggled(on: bool) -> None:
                if self._quiet:
                    return
                row.setVisible(on)
                if on and combo.currentData() is None:
                    combo.setCurrentIndex(max(combo.findData(toggle["on"]), 0))  # type: ignore[index]
                self._edited()

            box.toggled.connect(toggled)
            control.widgets.append(box)
        return control

    def _switch(self, item: dict[str, Any], lay: QVBoxLayout) -> _Control:
        box = QCheckBox(str(item["label"]))
        row = QHBoxLayout()
        row.addWidget(box)
        row.addStretch(1)
        row.addWidget(self._cost(item))
        lay.addLayout(row)
        if item.get("experimental"):
            note = QLabel("Experimental: " + str(item.get("help", "")))
            note.setWordWrap(True)
            note.setProperty("role", "warning")
            lay.addWidget(note)

        def flipped(_on: bool) -> None:
            self._update_needs()
            self._edited()

        box.toggled.connect(flipped)
        return _Control(item, box.isChecked, lambda v: box.setChecked(bool(v)), [box])

    def set_costs(self, costs: dict[str, float]) -> None:
        """Show each setting's speed cost in % (settings not in ``costs`` show none)."""
        for name, label in self._costs.items():
            pct = costs.get(name)
            if pct is None:
                label.hide()
                continue
            label.setText("<1 % speed" if pct < 0.5 else f"-{pct:.0f} % speed")
            label.setToolTip("How much slower the clean is with this on, the rest as chosen")
            label.show()

    def _path(self, item: dict[str, Any], lay: QVBoxLayout) -> _Control:
        label = QLabel(str(item["label"]))
        edit = QLineEdit()
        edit.setReadOnly(True)
        edit.setPlaceholderText("(default)")
        browse = QPushButton("Choose...")
        clear = QPushButton("Clear")
        row = QHBoxLayout()
        row.addWidget(edit, 1)
        row.addWidget(browse)
        row.addWidget(clear)
        lay.addWidget(label)
        lay.addLayout(row)

        def choose() -> None:
            path, _ = QFileDialog.getOpenFileName(
                self, str(item["label"]), "", str(item.get("filter") or "All files (*)")
            )
            if path:
                edit.setText(path)
                self._edited()

        def wipe() -> None:
            edit.clear()
            self._edited()

        browse.clicked.connect(choose)
        clear.clicked.connect(wipe)
        return _Control(
            item, edit.text, lambda v: edit.setText(str(v)), [label, edit, browse, clear]
        )

    # ----- values -----

    def _edited(self) -> None:
        if not self._quiet:
            self._timer.start()

    def _update_needs(self) -> None:
        for control in self._controls.values():
            needs = control.item.get("needs")
            if needs and needs in self._controls:
                on = bool(self._controls[needs].get())
                for w in control.widgets:
                    w.setEnabled(on and self.isEnabled())

    def params(self) -> RemasterParams:
        return RemasterParams.from_dict({n: c.get() for n, c in self._controls.items()})

    def set_params(self, params: RemasterParams, *, announce: bool = False) -> None:
        self._quiet = True
        try:
            for name, value in params.to_dict().items():
                if name in self._controls:
                    self._controls[name].put(value)
        finally:
            self._quiet = False
        self._update_needs()
        if announce:
            self.changed.emit(self.params())

    def reset(self) -> None:
        for control in self._controls.values():
            control.last_on = None
        self.set_params(RemasterParams(), announce=True)

    def setEnabled(self, enabled: bool) -> None:
        super().setEnabled(enabled)
        self._update_needs()
