"""The main window, in two steps.

1 CLIPS: the recordings on the left; on the right a player to check the selected one and
its timeline, where CUT markers leave stretches out (a cut at the start or end trims it).
CLEAN cleans them one after another in a worker process, each with its own progress.
2 RESULTS (once clips are cleaned): each output against its recording, before/after,
with Open output and Show in folder.

Closing the window while clips are being cleaned hides it to the notification area (the
tray): the work goes on, and clicking the AnaHD icon there brings the window back.
Closing it with nothing being cleaned quits completely. AnaHD never quits by itself.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from PySide6.QtCore import QItemSelectionModel, QLocale, QSettings, Qt, QTime
from PySide6.QtGui import QAction, QCloseEvent, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QSystemTrayIcon,
    QVBoxLayout,
    QWidget,
)

from anahd import __version__, bundle
from anahd.app import theme
from anahd.app.cliplist import (
    DETAILS,
    PATH,
    PROBLEM,
    SHARE,
    STATUS,
    THUMB,
    THUMB_SIZE,
    ClipList,
    fit_cards,
    pixmap_from_png,
)
from anahd.app.cutedit import CutEditor
from anahd.app.eta import Eta
from anahd.app.framepreview import FramePreviewer, Request, Shown
from anahd.app.frames import FrameReader, Pair, Probed, Still
from anahd.app.look import DotTitle, Spinner, StatusDot, app_icon
from anahd.app.player import PlayThread, Screen
from anahd.app.preview import Preview, Wipe
from anahd.app.remaster_panel import RemasterPanel
from anahd.app.speed import calibrate, costs, estimate
from anahd.app.timeline import Timeline, kept_seconds
from anahd.app.worker import Runner
from anahd.cli import default_output, is_output, part_outputs
from anahd.cuts import Cut, Span, keep_spans, output_to_source
from anahd.media import Crop
from anahd.origin import Origin
from anahd.remaster import RemasterParams

VIDEO = {".mp4", ".ts", ".mov", ".avi", ".mkv", ".m2ts", ".mts"}

DROP_PAGE, CLIPS_PAGE, RESULTS_PAGE = 0, 1, 2
WAITING, QUEUED, CLEANING, DONE, FAILED = "waiting", "queued", "cleaning", "done", "failed"
PAUSED_HEADING = "Preview · Paused: before / after with your settings; drag the line"
LIVE_HEADING = "Cleaning · live: before / after as it is cleaned; drag the line"
BUSY_HEADING = "Preview · before / after paused while cleaning"
START_S = 9.0  # a queued clip's start-up (measuring, loading the models)


SPEED_RATIOS = "speed_ratio"  # settings group: this machine's measured speed ratios


@dataclass
class Clip:
    path: Path
    item: QListWidgetItem
    state: str = WAITING
    duration: float = 0.0
    output: Path | None = None  # set once it is known (when cleaning starts or it exists)
    fps: float = 60.0
    size: tuple[int, int] = (0, 0)
    cuts: list[Cut] = field(default_factory=list)  # stretches to leave out
    cleaned_cuts: list[Cut] = field(default_factory=list)  # the cuts its output was made with
    outputs: list[Path] = field(default_factory=list)  # every file it was cleaned into
    skip: bool = False  # looks digital: left out of CLEAN unless included
    origin: Origin | None = None
    params: RemasterParams | None = None  # the settings it was cleaned with
    learned: bool = True  # cleaned with the learned engine (else the classic one)

    @property
    def editable(self) -> bool:
        """Cuts can change until the clip is being cleaned (then they are what it gets)."""
        return self.state in (WAITING, FAILED)

    @property
    def output_seconds(self) -> float:
        return kept_seconds(self.cleaned_cuts, self.duration)

    def spans(self, cuts: list[Cut] | None = None) -> list[Span]:
        frames = round(self.duration * self.fps)
        return keep_spans(self.cleaned_cuts if cuts is None else cuts, frames, self.fps)

    def out_to_src(self, seconds: float, part: int | None = None) -> float:
        """The moment of the recording that a moment of an output shows (``part``: the
        file of that kept part, when each part is its own file)."""
        spans = self.spans()
        if not spans:
            return 0.0
        if part is not None:
            return spans[part].start / self.fps + seconds
        return output_to_source(seconds, spans, self.fps)

    def output_length(self, part: int | None) -> float:
        if part is None:
            return self.output_seconds
        spans = self.spans()
        return spans[part].frames / self.fps if part < len(spans) else 0.0


def heading(text: str) -> QLabel:
    label = QLabel(text)
    label.setProperty("role", "heading")
    return label


def eyebrow(text: str = "") -> QLabel:
    label = QLabel(text)
    label.setProperty("role", "eyebrow")
    return label


def button(text: str, slot: Callable[[], object], *, primary: bool = False) -> QPushButton:
    widget = QPushButton(text)
    if primary:
        widget.setProperty("primary", True)
    widget.clicked.connect(slot)
    return widget


def restyle(widget: QWidget, name: str, value: object) -> None:
    """Change a style-sheet property and repaint the widget with it."""
    if widget.property(name) != value:
        widget.setProperty(name, value)
        widget.style().unpolish(widget)
        widget.style().polish(widget)


class Page(QWidget):
    """A plain page on the window's background."""


def videos_in(paths: Sequence[Path]) -> list[Path]:
    """Recordings among ``paths``; folders stand for the recordings directly inside them.
    AnaHD's own outputs are left out."""
    found: list[Path] = []
    for path in paths:
        files = sorted(p for p in path.iterdir() if p.is_file()) if path.is_dir() else [path]
        for f in files:
            if f.suffix.lower() in VIDEO and not is_output(f) and f.is_file():
                found.append(f)
    return found


def clock(seconds: float) -> str:
    return f"{int(seconds // 60)}:{int(seconds % 60):02d}"


def split_page(left: QWidget, right: QWidget, stretch: tuple[int, int]) -> Page:
    split = QSplitter(Qt.Orientation.Horizontal)
    split.addWidget(left)
    split.addWidget(right)
    split.setStretchFactor(0, stretch[0])
    split.setStretchFactor(1, stretch[1])
    split.setSizes([stretch[0] * 1000, stretch[1] * 1000])  # start in this proportion
    page = Page()
    layout = QVBoxLayout(page)
    layout.setContentsMargins(40, 32, 40, 16)
    layout.addWidget(split)
    return page


def column(*parts: QWidget | QHBoxLayout, stretch: QWidget | None = None) -> QWidget:
    holder = QWidget()
    layout = QVBoxLayout(holder)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(10)
    for part in parts:
        if isinstance(part, QHBoxLayout):
            layout.addLayout(part)
        else:
            layout.addWidget(part, 1 if part is stretch else 0)
    return holder


def row(*parts: QWidget | int) -> QHBoxLayout:
    """Widgets side by side; an int is a stretch."""
    layout = QHBoxLayout()
    layout.setSpacing(8)
    for part in parts:
        if isinstance(part, int):
            layout.addStretch(part)
        else:
            layout.addWidget(part)
    return layout


class MainWindow(QMainWindow):
    def __init__(self, settings: QSettings) -> None:
        super().__init__()
        self.setWindowTitle(f"AnaHD {__version__}")
        self.setWindowIcon(app_icon())
        self.resize(1320, 820)
        self.setAcceptDrops(True)
        self._settings = settings
        self._fonts = theme.load_fonts()
        self._clips: dict[Path, Clip] = {}
        self._current: Clip | None = None
        self._began = 0.0
        self._batch: list[Clip] = []
        self._quitting = False
        self._told_tray = False
        self._page = CLIPS_PAGE
        self._playing = False
        self._etas: dict[Path, Eta] = {}
        self._left: tuple[float, float, float] | None = None  # time left, fps, when
        self._previewer = FramePreviewer(self)
        self._previewer.ready.connect(self._preview_ready)
        self._previewer.failed.connect(self._preview_failed)
        self._last_shown: Shown | None = None

        self._runner = Runner(self)
        self._runner.progress.connect(self._progress)
        self._runner.stage.connect(self._stage_reported)
        self._runner.frame.connect(self._live_picture)
        self._runner.finished.connect(self._finished)
        self._runner.failed.connect(self._failed)
        self._reader = FrameReader(self)
        self._reader.probed.connect(self._probed)
        self._reader.failed.connect(self._unreadable)
        self._reader.pair.connect(self._pair)
        self._reader.still.connect(self._still)
        self._reader.checked.connect(self._checked)
        self._reader.start()
        self._player = PlayThread(self)
        self._player.frame.connect(self._played)
        self._player.ended.connect(self._play_ended)

        self._build()
        self._build_tray()
        self._apply_theme(str(settings.value("theme", "dark")))
        geometry = settings.value("geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)  # type: ignore[arg-type, unused-ignore]
        self._refresh()
        self._update_speed()

    # ----- layout -----

    def _build(self) -> None:
        logo = self._logo = QLabel()
        logo.setProperty("role", "logo")
        self._step_clips = QPushButton("1  Clips")
        self._step_results = QPushButton("2  Results")
        self._step_clips.clicked.connect(lambda: self._go(CLIPS_PAGE))
        self._step_results.clicked.connect(lambda: self._go(RESULTS_PAGE))
        for step in (self._step_clips, self._step_results):
            step.setProperty("role", "step")
        self._dot = StatusDot()
        self._header_status = eyebrow()
        self._theme_button = QPushButton()
        self._theme_button.setProperty("flat", True)
        self._theme_button.clicked.connect(self._toggle_theme)
        quit_button = button("Quit", self._quit)
        quit_button.setProperty("flat", True)
        quit_button.setToolTip("Close AnaHD completely")
        header = QFrame()
        header.setProperty("role", "header")
        header.setFixedHeight(56)
        top = QHBoxLayout(header)
        top.setContentsMargins(24, 0, 16, 0)
        top.addWidget(logo)
        top.addStretch(1)
        top.addWidget(self._step_clips)
        top.addWidget(eyebrow("——"))
        top.addWidget(self._step_results)
        top.addStretch(1)
        top.addWidget(self._dot)
        top.addWidget(self._header_status)
        top.addSpacing(12)
        top.addWidget(self._theme_button)
        top.addWidget(quit_button)

        # Empty: one big drop target that says what to do.
        self._drop_zone = Page()
        self._title = DotTitle("anahd", cell=12, accent="hd")
        intro = QLabel(
            "Drop analog FPV DVR recordings (or a whole folder) anywhere on this window. "
            "AnaHD repairs breakup, cleans the noise and upscales each one to HD 60 fps "
            "on your graphics card. Your recordings are never changed."
        )
        intro.setProperty("role", "hero")
        intro.setWordWrap(True)
        intro.setMaximumWidth(560)
        zone = QVBoxLayout(self._drop_zone)
        zone.setContentsMargins(72, 48, 72, 48)
        zone.addStretch(2)
        zone.addWidget(eyebrow("● Runs on this PC. Nothing is uploaded"))
        zone.addSpacing(18)
        zone.addWidget(self._title)
        zone.addSpacing(24)
        zone.addWidget(intro)
        zone.addSpacing(24)
        zone.addLayout(
            row(
                button("Add clips", self._choose_clips, primary=True),
                button("Add a folder", self._choose_folder),
                1,
            )
        )
        zone.addStretch(3)

        # 1 CLIPS: the list, and a player with the cut timeline.
        self._list = ClipList()
        self._list.dropped.connect(self.add_paths)
        self._list.itemSelectionChanged.connect(self._selection_changed)
        self._remove = button("Remove", self._remove_selected)
        left = column(
            eyebrow("Step 1 · Clips"),
            heading("Your recordings"),
            self._list,
            row(
                button("Add more", self._choose_clips),
                button("Add a folder", self._choose_folder),
                self._remove,
                1,
            ),
            stretch=self._list,
        )
        self._screen = Screen()
        self._timeline = Timeline()
        self._timeline.seeked.connect(self._seek)
        self._timeline.cuts_changed.connect(self._cuts_changed)
        self._play = button("Play", self._toggle_play)
        self._time = eyebrow()
        self._cut = button("✂  Cut", self._add_cut)
        self._cut.setToolTip(
            "Leave a stretch out: drops a cut at the playhead; drag its edges to size it. "
            "A cut at the start or end trims the clip."
        )
        self._keeps = eyebrow()
        self._include = button("Include anyway", self._include_clip)
        self._include.setToolTip("Clean this clip although it looks digital")
        self._editor = CutEditor(lambda: self._timeline.playhead)
        self._editor.changed.connect(self._timeline.set_cut)
        self._editor.deleted.connect(self._delete_cut)
        self._timeline.selection_changed.connect(self._cut_selected)
        # Paused: the playhead's frame remastered with the panel's settings, before/after.
        # Playing: the recording as it plays.
        self._frame_wipe = Wipe()
        self._stage = QStackedWidget()
        self._stage.addWidget(self._screen)
        self._stage.addWidget(self._frame_wipe)
        self._spinner = Spinner()
        self._preview_state = eyebrow()
        self._previewer.busy.connect(self._spinner.spin)
        self._previewer.busy.connect(
            lambda on: self._preview_state.setText("Rendering preview..." if on else "")
        )
        self._preview_heading = eyebrow(PAUSED_HEADING)
        heading_row = row(self._preview_heading, 1, self._preview_state, self._spinner)
        # While a clip is cleaning: its latest before/after, where it is, and a Stop at hand.
        self._live_caption = eyebrow()
        self._live_stop = button("Stop", self._stop_all)
        self._live_row = QWidget()
        live_layout = QHBoxLayout(self._live_row)
        live_layout.setContentsMargins(0, 0, 0, 0)
        live_layout.addWidget(self._live_caption, 1)
        live_layout.addWidget(self._live_stop)
        self._live_row.setVisible(False)
        self._live: dict[Path, tuple[bytes, bytes, int, int]] = {}  # latest picture per clip
        right = QWidget()
        rcol = QVBoxLayout(right)
        rcol.setContentsMargins(0, 0, 0, 0)
        rcol.setSpacing(10)
        rcol.addLayout(heading_row)
        rcol.addWidget(self._stage, 1)
        rcol.addWidget(self._live_row)
        rcol.addWidget(self._timeline)
        rcol.addWidget(self._editor)
        rcol.addLayout(row(self._play, self._time, 1, self._keeps, self._include, self._cut))
        self._panel = RemasterPanel()
        self._panel.set_params(self._saved_params())
        self._panel.changed.connect(self._params_changed)
        self._panel.reset_button.clicked.connect(self._forget_speed)
        settings_col = column(eyebrow("Remaster settings"), self._panel, stretch=self._panel)
        middle = split_page(right, settings_col, (8, 3))
        middle_layout = middle.layout()
        if middle_layout is not None:
            middle_layout.setContentsMargins(0, 0, 0, 0)
        self._list_page = split_page(left, middle, (3, 9))

        # 2 RESULTS: each cleaned clip against its recording.
        self._results = QListWidget()
        self._results.setItemDelegate(self._list.card)
        self._results.setVerticalScrollMode(QListWidget.ScrollMode.ScrollPerPixel)
        fit_cards(self._results)
        self._results.itemSelectionChanged.connect(self._result_changed)
        self._preview = Preview()
        self._preview.moment.connect(lambda _share: self._request_preview())
        self._open = button("Open output", self._open_output)
        self._show = button("Show in folder", self._show_output)
        self._preview_caption = eyebrow()
        rleft = column(
            eyebrow("Step 2 · Results"),
            heading("Cleaned clips"),
            self._results,
            stretch=self._results,
        )
        rright = column(
            eyebrow("Before / after · Drag the line to compare"),
            self._preview,
            row(self._preview_caption, 1, self._show, self._open),
            stretch=self._preview,
        )
        self._results_page = split_page(rleft, rright, (5, 7))

        self._pages = QStackedWidget()
        self._pages.addWidget(self._drop_zone)
        self._pages.addWidget(self._list_page)
        self._pages.addWidget(self._results_page)

        # Footer: where outputs go, overall progress, CLEAN / STOP.
        self._out_label = eyebrow()
        self._out_button = button("Change", self._choose_out_dir)
        self._out_button.setProperty("flat", True)
        self._export_label = eyebrow()
        self._export_button = button("Change", self._choose_export)
        self._engine_label = eyebrow()
        self._engine_button = button("Change", self._choose_engine)
        self._engine_button.setProperty("flat", True)
        self._export_button.setProperty("flat", True)
        self._overall = QProgressBar()
        self._overall.setRange(0, 1000)
        self._overall.setTextVisible(False)
        self._overall.setFixedWidth(220)
        self._start = button("Clean all  →", self._start_all, primary=True)
        self._stop = button("Stop", self._stop_all)
        footer = QFrame()
        footer.setProperty("role", "footer")
        footer.setFixedHeight(68)
        bottom = QHBoxLayout(footer)
        bottom.setContentsMargins(40, 0, 40, 0)
        bottom.setSpacing(10)
        bottom.addWidget(self._out_label)
        bottom.addWidget(self._out_button)
        bottom.addSpacing(16)
        bottom.addWidget(self._engine_label)
        bottom.addWidget(self._engine_button)
        bottom.addSpacing(16)
        bottom.addWidget(self._export_label)
        bottom.addWidget(self._export_button)
        bottom.addStretch(1)
        bottom.addWidget(self._overall)
        bottom.addSpacing(12)
        bottom.addWidget(self._stop)
        bottom.addWidget(self._start)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(header)
        layout.addWidget(self._pages, 1)
        layout.addWidget(footer)
        self.setCentralWidget(central)

    def _build_tray(self) -> None:
        self._tray = QSystemTrayIcon(app_icon(), self)
        self._tray.setToolTip("AnaHD")
        menu = QMenu(self)
        open_action = QAction("Open AnaHD", self)
        open_action.triggered.connect(self.bring_back)
        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(self._quit)
        menu.addAction(open_action)
        menu.addSeparator()
        menu.addAction(quit_action)
        self._tray.setContextMenu(menu)
        self._tray.activated.connect(self._tray_clicked)
        self._tray.messageClicked.connect(self.bring_back)
        self._tray.show()

    def _apply_theme(self, name: str) -> None:
        palette = theme.THEMES.get(name, theme.DARK)
        app = QApplication.instance()
        if isinstance(app, QApplication):
            theme.apply(app, palette, self._fonts)
        self._list.card.set_theme(palette, self._fonts)
        self._preview.wipe.palette_ = palette
        self._preview.wipe.fonts = self._fonts
        self._screen.palette_ = palette
        self._frame_wipe.palette_ = palette
        self._frame_wipe.fonts = self._fonts
        self._timeline.palette_ = palette
        self._logo.setText(f"ana<span style='color:{palette.accent}'>.</span>hd")
        self._title.set_colours(palette.text, palette.accent)
        self._dot.set_colours(palette.faint, palette.accent)
        self._theme_button.setText("Light" if palette.name == "dark" else "Dark")
        self._settings.setValue("theme", palette.name)
        self.update()

    def _toggle_theme(self) -> None:
        self._apply_theme("light" if self._settings.value("theme") == "dark" else "dark")

    # ----- clips -----

    def add_paths(self, paths: Sequence[Path]) -> None:
        for path in videos_in([Path(p) for p in paths]):
            key = path.resolve()
            if key in self._clips:
                continue
            item = QListWidgetItem()
            item.setData(PATH, str(key))
            self._list.addItem(item)
            clip = Clip(key, item)
            self._clips[key] = clip
            self._mark_existing(clip)
            self._reader.probe(key)
        if self._list.count() and not self._list.selectedItems():
            self._list.setCurrentRow(0)
        self._refresh()

    def _out_dir(self) -> Path | None:
        value = self._settings.value("out_dir")
        return Path(str(value)) if value else None

    def _target(self, clip: Clip) -> Path:
        return default_output(clip.path, self._out_dir())

    def _learned(self) -> bool:
        return self._settings.value("engine", "learned") != "classic"

    def _choose_engine(self) -> None:
        if self._runner.busy:
            return
        menu = QMenu(self)
        learned = menu.addAction(
            "Learned (v6): trained on one pilot's camera and DVR; the best look there",
            lambda: self._set_engine("learned"),
        )
        classic = menu.addAction(
            "Classic: hand-made steps, the same on any analog recording",
            lambda: self._set_engine("classic"),
        )
        (learned if self._learned() else classic).setCheckable(True)
        (learned if self._learned() else classic).setChecked(True)
        menu.exec(self._engine_button.mapToGlobal(self._engine_button.rect().bottomLeft()))

    def _set_engine(self, engine: str) -> None:
        self._settings.setValue("engine", engine)
        self._refresh()
        self._update_speed()
        self._request_frame()

    def _parts(self) -> bool:
        return self._settings.value("export", "one") == "parts"

    def _targets(self, clip: Clip) -> list[Path]:
        """Every file cleaning ``clip`` now would write."""
        out = self._target(clip)
        if self._parts() and clip.cuts and clip.duration > 0:
            return part_outputs(out, len(clip.spans(clip.cuts)))
        return [out]

    def _mark_existing(self, clip: Clip) -> None:
        """A clip whose outputs are all there counts as done (they are never
        overwritten)."""
        if clip.state in (QUEUED, CLEANING):
            return
        targets = self._targets(clip)
        if all(t.is_file() for t in targets):
            clip.state, clip.output, clip.outputs = DONE, targets[0], targets
            clip.cleaned_cuts = list(clip.cuts) if len(targets) > 1 else []
            self._set_status(clip, "Done  (output exists)")
        elif clip.state == DONE:
            clip.state, clip.output, clip.outputs = WAITING, None, []
            self._set_status(clip, "")

    def _set_status(
        self, clip: Clip, text: str, share: float | None = None, problem: bool = False
    ) -> None:
        clip.item.setData(STATUS, text)
        clip.item.setData(SHARE, share)
        clip.item.setData(PROBLEM, problem)

    def _probed(self, info: Probed, thumb: bytes) -> None:
        clip = self._clips.get(info.path)
        if clip is None:
            return
        clip.duration, clip.fps, clip.size = info.duration, info.fps, info.size
        clip.item.setData(DETAILS, info.details)
        pixmap = pixmap_from_png(thumb)
        if pixmap is not None:
            clip.item.setData(THUMB, pixmap)
        if self._selected() is clip:
            self._load_clip(clip)
        picked = self._result()
        if picked is not None and picked[0] is clip:  # its length was unknown when listed
            self._result_changed()
        self._refresh()

    def _unreadable(self, path: Path, why: str) -> None:
        clip = self._clips.get(path)
        if clip is not None and clip.state == WAITING:
            clip.state = FAILED
            self._set_status(clip, "Can't read", problem=True)
            clip.item.setToolTip(why)
            self._refresh()

    def _selected(self) -> Clip | None:
        items = self._list.selectedItems()
        return self._clips.get(Path(str(items[0].data(PATH)))) if items else None

    def _remove_selected(self) -> None:
        for item in self._list.selectedItems():
            clip = self._clips.get(Path(str(item.data(PATH))))
            if clip is None or clip.state in (QUEUED, CLEANING):
                continue  # stop first
            del self._clips[clip.path]
            self._list.takeItem(self._list.row(item))
        self._refresh()

    def _choose_clips(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self, "Add recordings", "", "Video (*.mp4 *.ts *.mov *.avi *.mkv *.m2ts *.mts)"
        )
        self.add_paths([Path(f) for f in files])

    def _choose_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Add a folder of recordings")
        if folder:
            self.add_paths([Path(folder)])

    def _choose_out_dir(self) -> None:
        if self._runner.busy:
            return
        menu = QMenu(self)
        menu.addAction("Next to each recording", lambda: self._set_out_dir(None))
        menu.addAction("Choose a folder...", self._pick_out_dir)
        menu.exec(self._out_button.mapToGlobal(self._out_button.rect().bottomLeft()))

    def _choose_export(self) -> None:
        if self._runner.busy:
            return
        menu = QMenu(self)
        menu.addAction("One file per clip, cut parts removed", lambda: self._set_export("one"))
        menu.addAction(
            "Each kept part as its own file (_part1, _part2, ...)",
            lambda: self._set_export("parts"),
        )
        menu.exec(self._export_button.mapToGlobal(self._export_button.rect().bottomLeft()))

    def _set_export(self, mode: str) -> None:
        self._settings.setValue("export", mode)
        for clip in self._clips.values():
            self._mark_existing(clip)
        self._refresh()

    def _pick_out_dir(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Save cleaned clips in")
        if folder:
            self._set_out_dir(Path(folder))

    def _set_out_dir(self, folder: Path | None) -> None:
        self._settings.setValue("out_dir", str(folder) if folder else "")
        for clip in self._clips.values():
            self._mark_existing(clip)
        self._refresh()

    # ----- clips page: preview player and cut markers -----

    def _selection_changed(self) -> None:
        self._stop_playing()
        clip = self._selected()
        if clip is not None:
            self._load_clip(clip)
        self._refresh()

    def _load_clip(self, clip: Clip) -> None:
        self._timeline.load(clip.duration, clip.cuts, clip.editable)
        self._screen.message = "Reading the clip..." if clip.duration <= 0 else ""
        self._last_shown = None
        self._stage.setCurrentWidget(self._screen)
        self._show_still(self._timeline.playhead)
        self._show_live()
        self._update_cut_controls()

    # ----- the live view: the clip being cleaned, as it is cleaned -----

    def _stage_reported(self, path: Path, text: str) -> None:
        clip = self._clips.get(path)
        if clip is not None and clip.state == CLEANING and not clip.item.data(SHARE):
            self._set_status(clip, f"{text}...", 0.0)
            self._refresh()

    def _live_picture(self, path: Path, before: bytes, after: bytes, n: int, total: int) -> None:
        clip = self._clips.get(path)
        if clip is None or clip.state != CLEANING:
            return  # a late picture of a clip that has ended
        self._live[path] = (before, after, n, total)
        self._show_live()

    def _show_live(self) -> None:
        """The selected clip's latest live picture, if it is the one being cleaned."""
        clip = self._selected()
        live = self._live.get(clip.path) if clip is not None else None
        on = clip is not None and clip.state == CLEANING and live is not None
        self._live_row.setVisible(on)
        if not on or clip is None or live is None or self._playing:
            self._update_heading()
            return
        before, after, n, total = live
        self._frame_wipe.show_pair(pixmap_from_png(before), pixmap_from_png(after), "")
        self._stage.setCurrentWidget(self._frame_wipe)
        of = f" of {total:,}" if total else ""
        at = clock(n / clip.fps) if clip.fps > 0 else ""
        self._live_caption.setText(
            f"Frame {n:,}{of}" + (f"  ·  {at} into the cleaned clip" if at else "")
        )
        self._update_heading()

    def _update_heading(self) -> None:
        clip = self._selected()
        if not self._live_row.isHidden():
            self._preview_heading.setText(LIVE_HEADING)
        elif self._runner.busy and clip is not None:
            self._preview_heading.setText(BUSY_HEADING)
        else:
            self._preview_heading.setText(PAUSED_HEADING)

    def _show_still(self, seconds: float) -> None:
        clip = self._selected()
        if clip is not None and clip.duration > 0:
            self._reader.frame_at(clip.path, seconds)  # quick, until the preview is ready
            self._request_frame()
        self._time.setText(f"{clock(seconds)} / {clock(clip.duration if clip else 0)}")

    # ----- the one-frame preview -----

    def _saved_params(self) -> RemasterParams:
        try:
            return RemasterParams.from_json(str(self._settings.value("remaster", "{}")))
        except (ValueError, TypeError):
            return RemasterParams()  # a broken saved value: the defaults

    def _params_changed(self, params: RemasterParams) -> None:
        self._settings.setValue("remaster", params.to_json())
        self._update_speed()
        self._request_frame()

    def _speed_key(self, gpu: str, learned: bool) -> str:
        name = gpu.replace("/", " ").replace("\\", " ")
        return f"{SPEED_RATIOS}/{name} {__version__} {'learned' if learned else 'classic'}"

    def _speed_ratio(self, learned: bool) -> float | None:
        """This machine's speed over the estimate's, as measured on the last GPU used."""
        gpu = str(self._settings.value("gpu", "") or "")
        if not gpu:
            return None
        value = self._settings.value(self._speed_key(gpu, learned))
        try:
            return None if value is None else float(value)
        except (TypeError, ValueError):
            return None

    def _learn_speed(self, clip: Clip, gpu: str, frames: int, seconds: float) -> None:
        """After a completed run: the estimate learns this machine's speed (per GPU, app
        version and engine; a stopped, failed or short run teaches nothing)."""
        if not gpu or clip.params is None:
            return
        self._settings.setValue("gpu", gpu)
        key = self._speed_key(gpu, clip.learned)
        old = self._speed_ratio(clip.learned)
        ratio = calibrate(old, frames, seconds, clip.params, clip.learned)
        if ratio is not None and ratio != old:
            self._settings.setValue(key, ratio)
            self._update_speed()

    def _forget_speed(self) -> None:
        self._settings.remove(SPEED_RATIOS)
        self._update_speed()

    def _update_speed(self) -> None:
        params, learned = self._panel.params(), self._learned()
        self._panel.set_costs(costs(params, learned))
        ratio = self._speed_ratio(learned)
        fps, warning = estimate(params, learned, ratio or 1.0)
        where = (
            f"measured on this PC's {self._settings.value('gpu')}"
            if ratio is not None
            else "RTX 4070 Ti, 720p recording"
        )
        text = f"Estimated speed: about {fps:.0f} fps ({where})"
        self._panel.speed.setText(text + (f"\n{warning}" if warning else ""))
        restyle(self._panel.speed, "role", "warning" if warning else "caption")

    def _preview_request(self) -> Request | None:
        clip = self._selected()
        if clip is None or clip.duration <= 0 or self._playing:
            return None
        return Request(clip.path, self._timeline.playhead, self._panel.params(), self._learned())

    def _request_frame(self) -> None:
        if self._runner.busy:
            return  # the GPU is cleaning; the plain frame shows meanwhile
        req = self._preview_request()
        if req is not None:
            self._previewer.request(req)

    def _preview_ready(self, shown: Shown) -> None:
        clip = self._selected()
        if clip is None or clip.path != shown.request.path or self._playing:
            return
        if self._runner.busy:
            return  # asked for before Clean: the live view or the plain frame stays
        self._last_shown = shown
        self._frame_wipe.show_pair(
            QPixmap.fromImage(shown.before), QPixmap.fromImage(shown.after), ""
        )
        self._stage.setCurrentWidget(self._frame_wipe)

    def _preview_failed(self, req: Request, why: str) -> None:
        clip = self._selected()
        if clip is not None and clip.path == req.path:
            self._preview_state.setText("Preview failed (hover)")
            self._preview_state.setToolTip(why)

    def _still(self, still: Still) -> None:
        clip = self._selected()
        if clip is not None and clip.path == still.path and not self._playing:
            self._screen.show_image(pixmap_from_png(still.png) if still.png else None)
            if self._last_shown is None and self._live_row.isHidden():
                self._stage.setCurrentWidget(self._screen)  # no preview yet: the frame

    def _seek(self, seconds: float) -> None:
        if self._playing:
            self._start_playing(seconds)
        else:
            self._show_still(seconds)

    def _toggle_play(self) -> None:
        if self._playing:
            self._stop_playing()
        else:
            self._start_playing(self._timeline.playhead)

    def _start_playing(self, seconds: float) -> None:
        clip = self._selected()
        if clip is None or clip.duration <= 0:
            return
        if seconds >= clip.duration - 0.05:
            seconds = 0.0  # from the end: start over
        self._playing = True
        self._play.setText("Pause")
        self._stage.setCurrentWidget(self._screen)
        self._player.play(clip.path, seconds, clip.duration, clip.size, clip.cuts)

    def _stop_playing(self) -> None:
        was = self._playing
        if self._playing:
            self._player.halt()
        self._playing = False
        self._play.setText("Play")
        if was:
            self._request_frame()  # paused: the preview of where it stopped

    def _played(self, image: QImage, seconds: float) -> None:
        if self._playing:
            self._screen.show_image(image)
            self._timeline.set_playhead(seconds)
            clip = self._selected()
            self._time.setText(f"{clock(seconds)} / {clock(clip.duration if clip else 0)}")

    def _play_ended(self) -> None:
        self._playing = False
        self._play.setText("Play")
        self._request_frame()

    def _add_cut(self) -> None:
        self._timeline.add_cut_at_playhead()

    def _delete_cut(self, index: int) -> None:
        self._timeline.select(index)
        self._timeline.remove_selected()

    def _cut_selected(self, index: int | None) -> None:
        clip = self._selected()
        cuts = self._timeline.cuts
        if index is None or clip is None or not clip.editable or index >= len(cuts):
            self._editor.show_cut(None, 0, (0.0, 0.0), 60.0)
        else:
            self._editor.show_cut(index, len(cuts), cuts[index], clip.fps)

    def _include_clip(self) -> None:
        clip = self._selected()
        if clip is not None and clip.skip:
            clip.skip = False
            self._set_status(clip, "Included (looks digital)")
            self._refresh()

    def _checked(self, path: Path, origin: Origin) -> None:
        clip = self._clips.get(path)
        if clip is None:
            return
        clip.origin = origin
        if origin.digital and clip.state == WAITING:
            clip.skip = True
            self._set_status(clip, f"Looks digital ({origin.reasons}): left out", problem=True)
            clip.item.setToolTip(
                "No OSD was found in this recording: analog goggle recordings have the OSD "
                "burned in, digital ones don't. AnaHD is made for analog video, so it is "
                "left out of Clean. Select it and press Include anyway to clean it too."
            )
        self._refresh()

    def _cuts_changed(self, cuts: list[Cut]) -> None:
        clip = self._selected()
        if clip is not None and clip.editable:
            clip.cuts = list(cuts)
            if self._playing:
                self._player.cuts = list(cuts)  # playback skips the new cuts at once
            self._mark_existing(clip)  # the files it would write depend on the cuts
        self._cut_selected(self._timeline.selected)
        self._update_cut_controls()

    def _update_cut_controls(self) -> None:
        clip = self._selected()
        ready = clip is not None and clip.duration > 0
        editable = ready and clip is not None and clip.editable
        self._cut.setEnabled(editable)
        self._include.setVisible(clip is not None and clip.skip)
        self._play.setEnabled(ready)
        if clip is None or not ready:
            self._keeps.setText("")
        elif not clip.editable:
            self._keeps.setText("Cuts locked: cleaning or cleaned")
        elif clip.cuts:
            keeps = kept_seconds(clip.cuts, clip.duration)
            self._keeps.setText(f"Keeps {clock(keeps)} of {clock(clip.duration)}")
        else:
            self._keeps.setText("No cuts")

    # ----- cleaning -----

    def _start_all(self) -> None:
        self._stop_playing()
        self._batch = [c for c in self._ordered() if c.state in (WAITING, FAILED) and not c.skip]
        for clip in self._batch:
            clip.state = QUEUED
            clip.item.setToolTip("")
            self._set_status(clip, "Queued")
        self._began = time.monotonic()
        self._next()  # the GPU is busy from here: no preview is asked for below
        if self._current is not None:
            self._select(self._current)  # pressing Clean is asking to watch it
        else:
            selected = self._selected()
            if selected is not None:
                self._load_clip(selected)

    def _ordered(self) -> list[Clip]:
        rows = (self._list.item(i) for i in range(self._list.count()))
        return [self._clips[Path(str(item.data(PATH)))] for item in rows]

    def _select(self, clip: Clip) -> None:
        if self._selected() is clip:
            self._load_clip(clip)  # its cuts are locked now
        else:
            flags = QItemSelectionModel.SelectionFlag.ClearAndSelect
            self._list.setCurrentItem(clip.item, flags)  # loads it

    def _next(self) -> None:
        ended = self._current
        self._left = None
        clip = next((c for c in self._ordered() if c.state == QUEUED), None)
        if clip is None:
            self._current = None
            self._batch_done()
            return
        targets = self._targets(clip)
        if all(t.exists() for t in targets):
            clip.state, clip.output, clip.outputs = DONE, targets[0], targets
            self._set_status(clip, "Done  (output exists)")
            self._next()
            return
        self._current = clip
        clip.state, clip.output, clip.outputs = CLEANING, targets[0], targets
        clip.cleaned_cuts = list(clip.cuts)
        self._etas.pop(clip.path, None)
        self._set_status(clip, "Starting...", 0.0)
        clip.params, clip.learned = self._panel.params(), self._learned()
        self._runner.start(
            clip.path,
            self._target(clip),
            clip.cuts,
            len(targets) > 1,
            self._learned(),
            clip.params.to_dict(),
        )
        if ended is not None and self._selected() is ended:
            self._select(clip)  # the user was watching: follow the batch
        self._refresh()

    def _progress(self, path: Path, done: int, total: int, when: float) -> None:
        clip = self._clips.get(path)
        if clip is None:
            return
        eta = self._etas.setdefault(path, Eta())
        fps, left = eta.update(done, total, when)
        share = min(done / total, 1.0) if total else 0.0
        speed = f"{fps:.1f} fps" if fps else ""
        self._left = (left, fps, time.monotonic()) if left is not None else None
        when_done = f"{clock(left)} left" if left is not None else "Estimating..."
        text = f"{share:.0%}   {speed}   {when_done}" if total else f"{done} frames   {speed}"
        self._set_status(clip, text, share)
        self._refresh()

    def _finished(self, path: Path, info: tuple[object, ...]) -> None:
        done = cast(tuple[int, float, tuple[int, int, int, int], str], info[:4])
        frames, took, crop, notes = done
        gpu = str(info[4]) if len(info) > 4 else ""  # older workers send no GPU name
        clip = self._clips.get(path)
        if clip is not None:
            self._learn_speed(clip, gpu, int(frames), float(took))
            clip.state = DONE
            self._show_cleaned_thumb(clip)
            self._set_status(clip, f"Done   {clock(took)}   {frames / max(took, 1e-6):.1f} fps")
            clip.item.setToolTip(notes)
            self._reader.set_crop(path, Crop(*crop))
        self._live.pop(path, None)
        self._next()
        if clip is not None and self._selected() is clip:
            self._load_clip(clip)

    def _show_cleaned_thumb(self, clip: Clip) -> None:
        """A cleaned clip's card shows its last live "after" picture (none received: the
        thumbnail stays as it was)."""
        live = self._live.get(clip.path)
        after = pixmap_from_png(live[1]) if live is not None else None
        if after is not None:
            size = THUMB_SIZE * 2  # sharp on a high-DPI screen
            clip.item.setData(
                THUMB,
                after.scaled(
                    size,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                ),
            )

    def _failed(self, path: Path, message: str) -> None:
        clip = self._clips.get(path)
        if clip is not None:
            clip.state = FAILED
            self._set_status(clip, "Failed  (hover for why)", problem=True)
            clip.item.setToolTip(message)
        self._live.pop(path, None)
        self._next()
        if clip is not None and self._selected() is clip:
            self._load_clip(clip)  # its cuts can be changed again; a preview once idle

    def _stop_all(self) -> None:
        self._runner.stop()
        for clip in self._clips.values():
            if clip.state in (QUEUED, CLEANING):
                clip.state = WAITING
                self._set_status(clip, "Stopped" if clip is self._current else "")
        self._current = None
        self._batch = []
        self._live.clear()
        self._left = None
        selected = self._selected()
        if selected is not None:
            self._load_clip(selected)
        self._refresh()

    def _batch_done(self) -> None:
        batch, self._batch = self._batch, []
        if any(c.state == DONE for c in batch):
            self._go(RESULTS_PAGE)  # cleaning is over: on to the results
        self._refresh()
        if not batch:
            return
        failed = [c for c in batch if c.state == FAILED]
        minutes = (time.monotonic() - self._began) / 60
        # AnaHD never quits by itself. Closed to the tray, it waits there with the results
        # until the window is opened (tray icon or this notification); closing the window
        # then, with nothing rendering, quits completely.
        after = "\nClick to see the results." if not self.isVisible() else ""
        if failed:
            names = ", ".join(c.path.name for c in failed)
            self._tray.showMessage(
                "AnaHD: some clips failed",
                f"{len(failed)} of {len(batch)} failed: {names}{after}",
                QSystemTrayIcon.MessageIcon.Warning,
            )
        else:
            self._tray.showMessage(
                "AnaHD finished",
                f"{len(batch)} clip{'s' if len(batch) > 1 else ''} cleaned in "
                f"{minutes:.0f} min{after}",
                app_icon(),
            )

    # ----- results page: before/after -----

    def _go(self, page: int) -> None:
        if page == RESULTS_PAGE and not self._done_clips():
            return
        if page != CLIPS_PAGE:
            self._stop_playing()
        self._page = page
        self._refresh()

    def _done_clips(self) -> list[Clip]:
        return [c for c in self._ordered() if c.state == DONE and c.output is not None]

    def _result_files(self) -> list[tuple[Clip, Path, int | None]]:
        """(clip, output file, part) for every cleaned file; part is None for a whole clip."""
        files: list[tuple[Clip, Path, int | None]] = []
        for c in self._done_clips():
            outs = c.outputs or ([c.output] if c.output else [])
            for i, out in enumerate(outs):
                files.append((c, out, i if len(outs) > 1 else None))
        return files

    def _sync_results(self) -> None:
        """The results list mirrors the cleaned files, keeping the selection."""
        files = self._result_files()
        shown = [str(self._results.item(i).data(PATH)) for i in range(self._results.count())]
        if shown == [str(out) for _, out, _ in files]:
            for i, (c, _, _) in enumerate(files):
                for role in (DETAILS, THUMB):
                    self._results.item(i).setData(role, c.item.data(role))
            return
        selected = self._result()
        self._results.blockSignals(True)
        self._results.clear()
        for c, out, part in files:
            item = QListWidgetItem()
            item.setData(PATH, str(out))
            item.setData(DETAILS, c.item.data(DETAILS))
            item.setData(THUMB, c.item.data(THUMB))
            item.setData(STATUS, f"Part {part + 1}" if part is not None else "")
            self._results.addItem(item)
        keys = [str(out) for _, out, _ in files]
        row_ = keys.index(str(selected[1])) if selected and str(selected[1]) in keys else 0
        self._results.setCurrentRow(row_ if files else -1)
        self._results.blockSignals(False)
        self._result_changed()

    def _result(self) -> tuple[Clip, Path, int | None] | None:
        items = self._results.selectedItems()
        if not items:
            return None
        key = str(items[0].data(PATH))
        return next((f for f in self._result_files() if str(f[1]) == key), None)

    def _result_changed(self) -> None:
        picked = self._result()
        if picked is not None:
            clip, _, part = picked
            self._preview.set_duration(clip.output_length(part))
        self._request_preview()
        self._refresh_result_buttons()

    def _request_preview(self) -> None:
        picked = self._result()
        if picked is None:
            self._preview.wipe.show_pair(None, None, "Pick a cleaned clip to compare")
            return
        clip, out, part = picked
        if clip.duration > 0:
            out_t = self._preview.share * clip.output_length(part)
            self._reader.preview(clip.path, out, out_t, clip.out_to_src(out_t, part))

    def _pair(self, pair: Pair) -> None:
        picked = self._result()
        if picked is None or picked[0].path != pair.path:
            return
        before = pixmap_from_png(pair.before) if pair.before else None
        after = pixmap_from_png(pair.after) if pair.after else None
        self._preview.wipe.show_pair(before, after, "" if after else "No output")

    def _refresh_result_buttons(self) -> None:
        picked = self._result()
        ready = picked is not None and picked[1].is_file()
        self._open.setEnabled(ready)
        self._show.setEnabled(ready)
        self._preview_caption.setText(str(picked[1]) if picked is not None and ready else "")
        self._preview_caption.setToolTip(str(picked[1]) if picked is not None else "")

    def _open_output(self) -> None:
        picked = self._result()
        if picked is not None and picked[1].is_file():
            if sys.platform == "win32":
                os.startfile(picked[1])  # noqa: S606 - the user's own player
            else:
                subprocess.Popen(["xdg-open", str(picked[1])])

    def _show_output(self) -> None:
        picked = self._result()
        if picked is not None and picked[1].is_file():
            if sys.platform == "win32":
                subprocess.Popen(["explorer", "/select,", str(picked[1])])
            else:
                subprocess.Popen(["xdg-open", str(picked[1].parent)])

    # ----- state shown -----

    def _refresh(self) -> None:
        busy = self._runner.busy
        self._sync_results()
        if self._page == RESULTS_PAGE and not self._done_clips():
            self._page = CLIPS_PAGE
        self._pages.setCurrentIndex(self._page if self._clips else DROP_PAGE)
        restyle(self._step_clips, "state", "on" if self._page == CLIPS_PAGE else "off")
        restyle(self._step_results, "state", "on" if self._page == RESULTS_PAGE else "off")
        self._step_results.setEnabled(bool(self._done_clips()))
        todo = [c for c in self._clips.values() if c.state in (WAITING, FAILED) and not c.skip]
        self._start.setEnabled(bool(todo) and not busy)
        self._start.setText(f"Clean {len(todo)} clip{'s' if len(todo) != 1 else ''}  →")
        self._stop.setVisible(busy)
        self._out_button.setEnabled(not busy)
        self._export_button.setEnabled(not busy)
        self._engine_button.setEnabled(not busy)
        self._panel.setEnabled(not busy)  # what a clip gets is fixed when CLEAN starts
        self._engine_label.setText(
            "Cleaning: Learned (v6)" if self._learned() else "Cleaning: Classic"
        )
        self._export_label.setText(
            "Export: each kept part as its own file"
            if self._parts()
            else "Export: one file per clip"
        )
        out_dir = self._out_dir()
        self._out_label.setText(
            f"Output: {out_dir}" if out_dir else "Output: next to each recording (AnaHD_<name>.mp4)"
        )
        self._refresh_result_buttons()
        self._update_cut_controls()
        self._remove.setEnabled(
            any(c.state not in (QUEUED, CLEANING) for c in self._selected_clips())
        )
        self._dot.set_busy(busy)
        # Overall progress over the batch, counting the clip being cleaned by its share.
        if busy and self._batch:
            parts = [
                1.0
                if c.state in (DONE, FAILED)
                else float(c.item.data(SHARE) or 0.0)
                if c.state == CLEANING
                else 0.0
                for c in self._batch
            ]
            share = sum(parts) / len(parts)
            index = next((i for i, c in enumerate(self._batch) if c.state == CLEANING), 0)
            text = f"Cleaning {index + 1} of {len(self._batch)}   {share:.0%}"
            finish = self._finish_time()
            if finish:
                text += f"   done about {finish}"
            self._overall.setValue(round(share * 1000))
            self._overall.setVisible(True)
            self._header_status.setText(text)
            self._tray.setToolTip(f"AnaHD: {text.lower()}")
        else:
            self._overall.setVisible(False)
            done = sum(c.state == DONE for c in self._clips.values())
            self._header_status.setText(
                f"{done} of {len(self._clips)} cleaned" if self._clips else ""
            )
            self._tray.setToolTip("AnaHD")
            self._live_row.setVisible(False)  # nothing is being cleaned
        self._update_heading()
        self._list.viewport().update()

    def _finish_time(self) -> str:
        """When the batch should be done, once the running clip's time left has settled:
        that, plus each queued clip's kept length at the speed measured on this machine
        (the estimate's speed times estimated / measured is the measured speed), plus its
        start-up. Empty while estimating."""
        if self._left is None:
            return ""
        left, fps, at = self._left
        if fps <= 0:
            return ""
        seconds = max(left - (time.monotonic() - at), 0.0)
        for c in self._batch:
            if c.state == QUEUED and c.duration > 0:
                frames = kept_seconds(c.cuts, c.duration) * (c.fps or 60.0)
                seconds += frames / fps + START_S
        done = QTime.currentTime().addSecs(round(seconds))
        return QLocale().toString(done, QLocale.FormatType.ShortFormat)

    def _selected_clips(self) -> list[Clip]:
        return [
            self._clips[Path(str(i.data(PATH)))]
            for i in self._list.selectedItems()
            if Path(str(i.data(PATH))) in self._clips
        ]

    # ----- window and tray -----

    def bring_back(self) -> None:
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def _tray_clicked(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason in (
            QSystemTrayIcon.ActivationReason.Trigger,
            QSystemTrayIcon.ActivationReason.DoubleClick,
        ):
            if self.isVisible() and not self.isMinimized():
                self.hide()
            else:
                self.bring_back()

    def closeEvent(self, event: QCloseEvent) -> None:
        self._settings.setValue("geometry", self.saveGeometry())
        if self._runner.busy and not self._quitting:
            # Keep cleaning in the background; the tray icon brings the window back.
            event.ignore()
            self._stop_playing()
            self.hide()
            if not self._told_tray:
                self._told_tray = True
                self._tray.showMessage(
                    "AnaHD keeps cleaning",
                    "Click the AnaHD icon in the notification area to open the window.",
                    app_icon(),
                )
            return
        event.accept()
        self._quit_now()

    def _quit(self) -> None:
        """The Quit button and the tray's Quit: ask first if a clip is being cleaned."""
        if self._runner.busy:
            self.bring_back()
            answer = QMessageBox.question(
                self,
                "Stop cleaning?",
                "A clip is still being cleaned. Stop it and quit? The unfinished output is "
                "removed; your recordings are not touched.",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        self._quit_now()

    def _quit_now(self) -> None:
        """The only way out of the app, reached from a user action only (X with nothing
        being cleaned, the Quit button, the tray's Quit). AnaHD never quits by itself:
        tests/test_app.py fails if any other code path calls this or quits the app."""
        self._quitting = True
        self._settings.setValue("geometry", self.saveGeometry())
        self._shutdown()
        self.hide()
        QApplication.quit()

    def _shutdown(self) -> None:
        self._previewer.stop()
        self._stop_playing()
        self._runner.stop()
        self._reader.stop()
        self._tray.hide()


def run(argv: Sequence[str]) -> int:
    bundle.setup()
    app = QApplication(list(argv))
    app.setApplicationName("AnaHD")
    app.setApplicationDisplayName("AnaHD")
    app.setApplicationVersion(__version__)
    app.setWindowIcon(app_icon())
    app.setQuitOnLastWindowClosed(False)  # hiding to the tray must not end the app
    settings = QSettings("AnaHD", "AnaHD")
    window = MainWindow(settings)
    # Recordings dropped on the shortcut open on the clips page, to be checked and cut
    # before CLEAN (they are not cleaned straight away).
    window.add_paths([Path(arg) for arg in argv[1:]])
    window.show()
    return app.exec()
