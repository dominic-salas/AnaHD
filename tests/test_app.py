"""The desktop app, offscreen: adding clips, the close-to-tray rule, the worker's messages.
Skipped without PySide6 (the ``app`` extra)."""

from __future__ import annotations

import ast
import os
import queue
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import (
    QCoreApplication,
    QEventLoop,
    QItemSelectionModel,
    QLocale,
    QPoint,
    QRect,
    QSettings,
    Qt,
    QTimer,
)
from PySide6.QtGui import QCloseEvent, QColor, QImage, QPainter, QStandardItemModel
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QLabel,
    QPushButton,
    QStyleOptionViewItem,
)

from anahd.app import worker
from anahd.app.cliplist import CARD_HEIGHT, DETAILS, PATH, SHARE, STATUS, ClipCard
from anahd.app.preview import Wipe
from anahd.app.timeline import Timeline, skip_cuts
from anahd.app.window import (
    CLEANING,
    CLIPS_PAGE,
    DONE,
    RESULTS_PAGE,
    Clip,
    MainWindow,
    videos_in,
)
from anahd.remaster import RemasterParams


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance() or QApplication([])
    assert isinstance(app, QApplication)
    app.setQuitOnLastWindowClosed(False)
    return app


@pytest.fixture
def window(qapp: QApplication, tmp_path: Path) -> Iterator[MainWindow]:
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    win = MainWindow(settings)
    yield win
    win._quitting = True
    win._shutdown()


def test_videos_in_folders_skips_outputs_and_other_files(tmp_path: Path) -> None:
    for name in ("a.mp4", "b.TS", "AnaHD_a.mp4", "old_anahd.mp4", "notes.txt"):
        (tmp_path / name).write_bytes(b"x")
    single = tmp_path / "c.mov"
    single.write_bytes(b"x")
    assert [p.name for p in videos_in([tmp_path])] == ["a.mp4", "b.TS", "c.mov"]
    assert videos_in([single]) == [single]


def test_adding_clips_and_existing_outputs(window: MainWindow, tmp_path: Path) -> None:
    clip = tmp_path / "flight.mp4"
    clip.write_bytes(b"not really a video")
    (tmp_path / "AnaHD_flight.mp4").write_bytes(b"earlier render")
    other = tmp_path / "other.ts"
    other.write_bytes(b"x")
    window.add_paths([tmp_path])
    window.add_paths([clip])  # again: no duplicate
    assert window._list.count() == 2
    done = window._clips[clip.resolve()]
    assert done.state == DONE and done.output == tmp_path / "AnaHD_flight.mp4"
    assert (tmp_path / "AnaHD_flight.mp4").read_bytes() == b"earlier render"
    assert window._start.isEnabled()  # other.ts is still to do


def test_closing_while_busy_hides_to_the_tray(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    window.show()
    monkeypatch.setattr(type(window._runner), "busy", property(lambda self: True))
    event = QCloseEvent()
    window.closeEvent(event)
    assert not event.isAccepted()
    assert not window.isVisible()
    window.bring_back()
    assert window.isVisible()


def test_closing_when_idle_quits(window: MainWindow, monkeypatch: pytest.MonkeyPatch) -> None:
    quits: list[bool] = []
    monkeypatch.setattr(QApplication, "quit", staticmethod(lambda: quits.append(True)))
    window.show()
    event = QCloseEvent()
    window.closeEvent(event)
    assert event.isAccepted() and quits == [True]
    assert not window._tray.isVisible()  # no icon left behind in the notification area


def _one_done_clip(window: MainWindow, tmp_path: Path) -> None:
    clip = tmp_path / "flight.mp4"
    clip.write_bytes(b"x")
    window.add_paths([clip])
    c = window._clips[clip.resolve()]
    c.state = DONE
    window._batch = [c]


def test_never_quits_by_itself_after_a_batch(
    window: MainWindow, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A batch that finishes with the window closed leaves AnaHD in the tray with the
    results: no quit, and no timer that could quit later."""
    timers: list[object] = []
    quits: list[bool] = []
    monkeypatch.setattr(QTimer, "singleShot", lambda *a: timers.append(a))
    monkeypatch.setattr(QApplication, "quit", staticmethod(lambda: quits.append(True)))
    _one_done_clip(window, tmp_path)
    window.hide()
    window._batch_done()
    assert quits == [] and timers == []
    assert window._tray.isVisible()  # the way back to the results


def test_app_never_exits_without_a_user_action(
    window: MainWindow, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole path of a job finishing with the window closed, then 12 s of the real
    event loop (longer than any timer the app ever had): every way out of the app is
    watched, and none may be taken."""
    exits: list[str] = []
    for cls in (QApplication, QCoreApplication):
        monkeypatch.setattr(cls, "quit", staticmethod(lambda: exits.append("quit")))
        monkeypatch.setattr(cls, "exit", staticmethod(lambda *a: exits.append("exit")))
    monkeypatch.setattr(MainWindow, "_quit_now", lambda self: exits.append("_quit_now"))
    clip = tmp_path / "flight.mp4"
    clip.write_bytes(b"x")
    window.add_paths([clip])
    c = window._clips[clip.resolve()]
    c.state, c.output = CLEANING, tmp_path / "AnaHD_flight.mp4"
    window._batch = [c]
    window.show()
    window.hide()  # closed to the tray while cleaning
    window._runner.finished.emit(c.path, (120, 5.0, (0, 0, 960, 720), ""))  # the job ends
    assert c.state == DONE
    loop = QEventLoop()
    QTimer.singleShot(12_000, loop.quit)
    loop.exec()
    assert exits == []
    assert window._tray.isVisible() and not window.isVisible()


def test_only_user_actions_lead_out_of_the_app() -> None:
    """Source guard: quitting is only in _quit_now, which only the X (closeEvent) and the
    Quit button / tray Quit (_quit) call, and _quit is only wired to those two controls.
    Nothing may close the window from code (that would run closeEvent without a user)."""
    import anahd.app as app_pkg

    allowed = {"quit": {"_quit_now"}, "exit": {"<module>"}, "_quit_now": {"closeEvent", "_quit"}}
    found: list[tuple[str, str, str]] = []
    for path in Path(app_pkg.__file__).parent.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        scopes: list[tuple[ast.AST, str]] = [(tree, "<module>")]
        while scopes:
            node, scope = scopes.pop()
            for child in ast.iter_child_nodes(node):
                name = child.name if isinstance(child, ast.FunctionDef) else scope
                if isinstance(child, ast.Attribute) and child.attr in (*allowed, "_quit", "close"):
                    found.append((path.name, child.attr, name))
                scopes.append((child, name))
    for file, attr, scope in found:
        if attr in allowed:
            assert scope in allowed[attr], f"{file}: {attr} called from {scope}"
        elif attr == "_quit":
            assert scope in {"_build", "_build_tray"}, f"{file}: _quit used in {scope}"
        else:
            assert scope == "_shutdown" or file != "window.py", f"{file}: close() in {scope}"
    assert ("window.py", "_quit", "_build") in found  # the Quit button
    assert ("window.py", "_quit", "_build_tray") in found  # the tray's Quit


def test_quit_button_quits(window: MainWindow, monkeypatch: pytest.MonkeyPatch) -> None:
    quits: list[bool] = []
    monkeypatch.setattr(QApplication, "quit", staticmethod(lambda: quits.append(True)))
    window.show()
    window._quit()  # nothing is being cleaned: no question, straight out
    assert quits == [True] and not window._tray.isVisible()


def test_after_seeing_the_results_closing_quits_completely(
    window: MainWindow, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Opened from the tray after the batch, then closed with X: nothing is rendering, so
    the app quits and leaves no icon behind."""
    quits: list[bool] = []
    monkeypatch.setattr(QApplication, "quit", staticmethod(lambda: quits.append(True)))
    _one_done_clip(window, tmp_path)
    window.hide()
    window._batch_done()
    window.bring_back()
    event = QCloseEvent()
    window.closeEvent(event)
    assert event.isAccepted() and quits == [True]
    assert not window._tray.isVisible()


def test_progress_is_shown_per_clip_and_overall(
    window: MainWindow, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(type(window._runner), "busy", property(lambda self: True))
    a, b = tmp_path / "a.mp4", tmp_path / "b.mp4"
    for p in (a, b):
        p.write_bytes(b"x")
    window.add_paths([a, b])
    ca, cb = window._clips[a.resolve()], window._clips[b.resolve()]
    ca.state, cb.state = "cleaning", "queued"
    window._batch = [ca, cb]
    for i in range(12):  # worker timestamps: 24 fps for 11 s, steady
        window._progress(a.resolve(), 300 + 24 * i, 1000, 100.0 + i)
    status = ca.item.data(STATUS)
    left = (1000 - 300 - 24 * 11) / 24  # 18.2 s
    assert status.startswith("56%") and "24.0 fps" in status, status
    assert f"0:{int(left):02d} left" in status, status
    assert ca.item.data(SHARE) == pytest.approx((300 + 24 * 11) / 1000)
    assert cb.item.data(SHARE) is None  # the queued clip shows no progress
    assert "Cleaning 1 of 2" in window._header_status.text()


def test_worker_reports_failure_instead_of_raising(tmp_path: Path) -> None:
    messages: queue.Queue[tuple[Any, ...]] = queue.Queue()
    worker.work(str(tmp_path / "missing.mp4"), str(tmp_path / "out.mp4"), messages)  # type: ignore[arg-type]
    first = messages.get_nowait()
    assert first[0] == "stage", "the card says something before the slow imports"
    kind, text = messages.get_nowait()
    assert kind == "failed" and "not found" in text
    assert not (tmp_path / "out.mp4").exists()


# ----- clips page: preview and cut markers; results page: before/after -----


def _timeline(duration: float = 10.0) -> Timeline:
    tl = Timeline()
    tl.resize(1016, 60)  # track from x=8, 1000 px wide: 100 px per second
    tl.load(duration, [], editable=True)
    return tl


def _x(t: float) -> int:
    return round(8 + 100 * t)


def test_cut_button_drops_a_cut_at_the_playhead(qapp: QApplication) -> None:
    tl = _timeline()
    tl.set_playhead(3.0)
    tl.add_cut_at_playhead()
    assert tl.cuts == [(3.0, 5.0)] and tl.selected == 0
    tl.set_playhead(9.5)  # near the end: the cut ends at the clip's end
    tl.add_cut_at_playhead()
    assert tl.cuts == [(3.0, 5.0), (8.0, 10.0)]


def test_dragging_a_cut_edge_resizes_it_and_overlaps_merge(qapp: QApplication) -> None:
    tl = _timeline()
    changes: list[list[tuple[float, float]]] = []
    tl.cuts_changed.connect(changes.append)
    tl.set_playhead(3.0)
    tl.add_cut_at_playhead()  # 3-5
    QTest.mousePress(tl, Qt.MouseButton.LeftButton, pos=QPoint(_x(5.0), 30))
    QTest.mouseMove(tl, QPoint(_x(7.0), 30))
    QTest.mouseRelease(tl, Qt.MouseButton.LeftButton, pos=QPoint(_x(7.0), 30))
    assert tl.cuts == [pytest.approx((3.0, 7.0), abs=0.02)]
    tl.set_playhead(6.0)
    tl.add_cut_at_playhead()  # 6-8 overlaps 3-7: one cut 3-8
    assert tl.cuts == [pytest.approx((3.0, 8.0), abs=0.02)]
    assert changes[-1] == tl.cuts


def test_select_and_delete_a_cut_and_click_to_seek(qapp: QApplication) -> None:
    tl = _timeline()
    seeks: list[float] = []
    tl.seeked.connect(seeks.append)
    tl.set_playhead(2.0)
    tl.add_cut_at_playhead()  # 2-4
    QTest.mouseClick(tl, Qt.MouseButton.LeftButton, pos=QPoint(_x(8.0), 30))  # empty: seek
    assert tl.selected is None and seeks[-1] == pytest.approx(8.0, abs=0.02)
    QTest.mouseClick(tl, Qt.MouseButton.LeftButton, pos=QPoint(_x(3.0), 30))  # inside: select
    assert tl.selected == 0
    QTest.keyClick(tl, Qt.Key.Key_Delete)
    assert tl.cuts == []


def test_cuts_cannot_be_edited_when_locked(qapp: QApplication) -> None:
    tl = _timeline()
    tl.load(10.0, [(1.0, 2.0)], editable=False)
    tl.set_playhead(5.0)
    tl.add_cut_at_playhead()
    QTest.mousePress(tl, Qt.MouseButton.LeftButton, pos=QPoint(_x(2.0), 30))
    QTest.mouseMove(tl, QPoint(_x(4.0), 30))
    QTest.mouseRelease(tl, Qt.MouseButton.LeftButton, pos=QPoint(_x(4.0), 30))
    assert tl.cuts == [(1.0, 2.0)]


def test_playback_skips_cut_parts() -> None:
    cuts = [(1.0, 2.0), (5.0, 6.5)]
    assert skip_cuts(0.5, cuts) == 0.5
    assert skip_cuts(1.0, cuts) == 2.0
    assert skip_cuts(5.9, cuts) == 6.5
    assert skip_cuts(6.5, cuts) == 6.5


def test_the_worker_gets_the_clips_cuts() -> None:
    from anahd import cli

    args = cli._parser().parse_args(worker.cli_args("flight.mp4", [(0.0, 1.5), (60.0, 72.25)]))
    assert args.cut == [(0.0, 1.5), (60.0, 72.25)]
    assert cli._parser().parse_args(worker.cli_args("flight.mp4", [])).cut is None


def test_clips_page_before_after_is_the_one_frame_preview(window: MainWindow) -> None:
    """Clips page: one before/after, the playhead's frame with the panel's settings, shown
    while paused (playing shows the recording); the results page has its own."""
    assert window._list_page.findChildren(Wipe) == [window._frame_wipe]
    assert window._results_page.findChildren(Wipe)
    assert window._stage.indexOf(window._frame_wipe) >= 0


def _clip(window: MainWindow, tmp_path: Path, name: str = "flight.mp4") -> Clip:
    path = tmp_path / name
    path.write_bytes(b"x")
    window.add_paths([path])
    c = window._clips[path.resolve()]
    c.duration, c.fps = 10.0, 60.0
    return c


def test_cleaning_sends_the_cuts_and_locks_them(
    window: MainWindow, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started: list[tuple[object, ...]] = []
    monkeypatch.setattr(window._runner, "start", lambda *a: started.append(a))
    c = _clip(window, tmp_path)
    window._list.setCurrentRow(0)
    window._load_clip(c)
    window._timeline.set_playhead(0.0)
    window._add_cut()
    assert c.cuts == [(0.0, 2.0)]  # the button edits the selected clip's cuts
    window._start_all()
    default = RemasterParams().to_dict()
    assert started == [(c.path, tmp_path / "AnaHD_flight.mp4", [(0.0, 2.0)], False, True, default)]
    assert not window._timeline.editable and not window._cut.isEnabled()
    window._timeline.set_playhead(5.0)
    window._add_cut()
    assert c.cuts == [(0.0, 2.0)]


def test_results_page_after_cleaning_compares_the_same_moment(
    window: MainWindow, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After a batch: the results page, and the before frame is taken from where that
    moment of the output came from in the recording (across the cut)."""
    asked: list[tuple[object, ...]] = []
    monkeypatch.setattr(window._reader, "preview", lambda *a: asked.append(a))
    c = _clip(window, tmp_path)
    out = tmp_path / "AnaHD_flight.mp4"
    out.write_bytes(b"video")
    c.state, c.output, c.cleaned_cuts = CLEANING, out, [(0.0, 1.0)]
    window._batch = [c]
    window._runner.finished.emit(c.path, (540, 10.0, (0, 0, 960, 720), ""))
    assert window._pages.currentIndex() == RESULTS_PAGE
    assert window._results.count() == 1 and window._open.isEnabled()
    window._preview.slider.setValue(500)  # the middle of the 9 s output
    path, output, out_t, src_t = asked[-1]
    assert (path, output) == (c.path, out)
    assert out_t == pytest.approx(4.5) and src_t == pytest.approx(5.5)


def test_results_page_is_unreachable_without_results(window: MainWindow, tmp_path: Path) -> None:
    _clip(window, tmp_path)
    window._go(RESULTS_PAGE)
    assert window._pages.currentIndex() == CLIPS_PAGE
    assert not window._step_results.isEnabled()


def _number_in_preview(image: QImage) -> int:
    """The frame number of synthetic.numbered_truth, read from a preview frame (the whole
    1280x720 DVR frame scaled to the preview size)."""
    import synthetic as syn

    scale = image.height() / syn.DVR_H
    n = 0
    for bit in range(syn.CODE_BITS):
        cx = syn.CODE_X + bit * syn.CODE_STEP + syn.CODE_SIZE / 2  # output pixels
        cy = syn.CODE_Y + syn.CODE_SIZE / 2
        x = (syn.PIC_X + cx * syn.PIC_W / syn.OUT_W) * scale
        y = cy * syn.PIC_H / syn.OUT_H * scale
        n |= int(QColor(image.pixel(round(x), round(y))).lightness() > 128) << bit
    return n


def test_preview_plays_in_real_time_and_skips_the_cuts(qapp: QApplication, tmp_path: Path) -> None:
    """The real player on a recording whose frames carry their number: each shown picture
    is the frame of its time, nothing inside the cut is shown, and it plays at speed."""
    import synthetic as syn
    from anahd.app.player import PlayThread

    clip = syn.write_with(tmp_path / "flight.mp4", 180, syn.numbered_truth, syn.FOUR_THREE)
    shown: list[tuple[float, int, float]] = []
    player = PlayThread()
    # timed where the player sends each picture: a busy UI thread (a stall on a shared CI
    # runner) would otherwise deliver queued pictures in a burst and look like fast forward
    player.frame.connect(
        lambda image, t: shown.append((t, _number_in_preview(image), time.monotonic())),
        Qt.ConnectionType.DirectConnection,
    )
    loop = QEventLoop()
    player.ended.connect(loop.quit)
    QTimer.singleShot(8000, loop.quit)
    began = time.monotonic()
    player.play(clip.path, 0.5, 3.0, (syn.DVR_W, syn.DVR_H), [(1.0, 2.0)])
    loop.exec()
    player.halt()
    assert shown, "nothing was played"
    times = [t for t, _, _ in shown]
    assert times == sorted(times)
    assert not any(1.0 <= t < 2.0 for t in times), "played inside the cut"
    assert any(t < 1.0 for t in times) and any(t >= 2.0 for t in times)
    wrong = [(t, n) for t, n, _ in shown if abs(n - t * 60) > 1.01]
    assert wrong == [], f"pictures that are not the frame of their time: {wrong[:5]}"
    assert not any(60 <= n < 120 for _, n, _ in shown), "a frame from inside the cut"
    # Pace: within each stretch (before and after the cut) the video plays at real time,
    # not slow motion, not fast forward. Starting FFmpeg (at the start and again after
    # the cut) takes a machine-dependent moment, checked on its own with a loose bound:
    # GitHub's 2-core runner needed ~1 s per start where this PC needs ~0.2 s.
    stretches = [[s for s in shown if s[0] < 1.0], [s for s in shown if s[0] >= 2.0]]
    for part in stretches:
        assert len(part) >= 5, part
        (t0, _, w0), (t1, _, w1) = part[0], part[-1]
        pace = (t1 - t0) / (w1 - w0)
        print(f"\npace {pace:.2f}x real time over {t1 - t0:.2f} s")
        assert 0.85 <= pace <= 1.15, f"{t1 - t0:.2f} s of video in {w1 - w0:.2f} s"
    first_wait = stretches[0][0][2] - began
    cut_wait = stretches[1][0][2] - stretches[0][-1][2]
    print(f"first picture after {first_wait:.2f} s, past the cut after {cut_wait:.2f} s")
    assert first_wait <= 3.0 and cut_wait <= 3.0


def test_selecting_a_cut_opens_its_editor_for_exact_times(
    window: MainWindow, tmp_path: Path
) -> None:
    c = _clip(window, tmp_path)
    window._list.setCurrentRow(0)
    window._load_clip(c)
    ed = window._editor
    assert ed.currentIndex() == 0  # nothing selected: just the hint
    window._timeline.set_playhead(3.0)
    window._add_cut()  # 3-5, selected
    assert ed.currentIndex() == 1 and ed.start.text() == "0:03.000" and ed.end.text() == "0:05.000"
    ed.start.setText("0:02.250")
    ed.end.setText("6.5")
    ed.start.editingFinished.emit()
    assert c.cuts == [(2.25, 6.5)]
    QTest.mouseClick(
        ed.findChildren(QPushButton, options=Qt.FindChildOption.FindChildrenRecursively)[1],
        Qt.MouseButton.LeftButton,
    )  # the start's +0.5s button
    assert c.cuts == [pytest.approx((2.75, 6.5))]  # +0.5 s on the start
    window._timeline.set_playhead(7.25)
    ed._here(1)  # End here
    assert c.cuts[0][1] == pytest.approx(7.25)
    ed.start.setText("nonsense")
    ed.start.editingFinished.emit()
    assert c.cuts[0][1] == pytest.approx(7.25) and ed.start.property("invalid") is True
    ed._delete()
    assert c.cuts == [] and ed.currentIndex() == 0


def test_each_part_as_its_own_file(
    window: MainWindow, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Chosen before cleaning; the worker is told, and the results list each part file,
    comparing it with its own stretch of the recording."""
    started: list[tuple[object, ...]] = []
    asked: list[tuple[object, ...]] = []
    monkeypatch.setattr(window._runner, "start", lambda *a: started.append(a))
    monkeypatch.setattr(window._reader, "preview", lambda *a: asked.append(a))
    window._set_export("parts")
    c = _clip(window, tmp_path)
    c.cuts = [(4.0, 6.0)]  # keeps 0-4 s and 6-10 s
    window._start_all()
    parts = [tmp_path / "AnaHD_flight_part1.mp4", tmp_path / "AnaHD_flight_part2.mp4"]
    default = RemasterParams().to_dict()
    assert started == [(c.path, tmp_path / "AnaHD_flight.mp4", [(4.0, 6.0)], True, True, default)]
    assert c.outputs == parts
    for p in parts:
        p.write_bytes(b"video")
    window._runner.finished.emit(c.path, (480, 10.0, (0, 0, 960, 720), ""))
    assert window._results.count() == 2
    window._results.setCurrentRow(1)
    window._preview.slider.setValue(500)  # the middle of the 4 s second part
    _, output, out_t, src_t = asked[-1]
    assert output == parts[1] and out_t == pytest.approx(2.0) and src_t == pytest.approx(8.0)
    assert window._preview_caption.text() == str(parts[1])  # where the output is


def test_digital_looking_clips_are_left_out_until_included(
    window: MainWindow, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from anahd.origin import Origin

    started: list[tuple[object, ...]] = []
    monkeypatch.setattr(window._runner, "start", lambda *a: started.append(a))
    c = _clip(window, tmp_path)
    window._list.setCurrentRow(0)
    window._reader.checked.emit(c.path, Origin(True, 0.0))
    assert c.skip and "Looks digital" in c.item.data(STATUS)
    assert not window._start.isEnabled() and window._include.isVisibleTo(window)
    window._start_all()
    assert started == []
    window._include_clip()
    assert not c.skip and window._start.isEnabled()
    window._start_all()
    assert started and started[0][0] == c.path


def test_analog_clips_are_not_left_out(window: MainWindow, tmp_path: Path) -> None:
    from anahd.origin import Origin

    c = _clip(window, tmp_path)
    window._reader.checked.emit(c.path, Origin(False, 0.08))  # an OSD was found
    assert not c.skip and window._start.isEnabled()


def test_classic_cleaning_is_a_choice_made_before_clean(
    window: MainWindow, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """For other pilots' gear: the hand-made steps instead of the learned model, chosen
    before cleaning and passed to the worker as --no-learned."""
    from anahd import cli

    started: list[tuple[object, ...]] = []
    monkeypatch.setattr(window._runner, "start", lambda *a: started.append(a))
    assert window._engine_label.text() == "Cleaning: Learned (v6)"  # the default
    window._set_engine("classic")
    assert window._engine_label.text() == "Cleaning: Classic"
    c = _clip(window, tmp_path)
    window._start_all()
    assert started[0][4] is False
    args = cli._parser().parse_args(worker.cli_args(str(c.path), [], False, learned=False))
    assert args.learned is False
    assert cli._parser().parse_args(worker.cli_args(str(c.path), [])).learned is True


# ----- remaster settings panel and the one-frame preview -----


def test_panel_is_generated_from_the_settings_metadata(window: MainWindow) -> None:
    from anahd.remaster import GROUP_INFO, schema

    panel = window._panel
    assert set(panel._controls) == {i["name"] for i in schema()}  # every setting, no other
    assert list(panel.sections) == [g for g in GROUP_INFO if g in panel.sections]
    assert not panel.sections["Advanced"].isVisibleTo(panel)  # folded at the bottom
    assert panel.sections["Clean"].isVisibleTo(panel)
    assert panel.params() == RemasterParams()  # fresh settings: the approved defaults
    texts = [w.text() for w in panel.findChildren(QLabel)]
    assert any(t.startswith("Experimental") for t in texts)


def test_toggles_switch_off_and_bring_back_the_last_value(window: MainWindow) -> None:
    panel = window._panel
    panel.set_params(RemasterParams(look_strength=70))
    look = next(b for b in panel.findChildren(QCheckBox) if b.text() == "Look")
    assert look.isChecked()
    look.setChecked(False)
    assert panel.params().look_strength == 0
    look.setChecked(True)
    assert panel.params().look_strength == 70  # the value it had, not the default
    blur = next(b for b in panel.findChildren(QCheckBox) if b.text() == "Motion blur")
    assert panel.params().motion_blur == 0 and not blur.isChecked()
    blur.setChecked(True)
    assert panel.params().motion_blur == 39  # the Resolve preset, suggested when ticked
    sv = next(b for b in panel.findChildren(QCheckBox) if b.text() == "SuperView")
    sv.setChecked(True)
    assert panel.params().superview == "stretch"


def test_font_picker_waits_for_its_switch(window: MainWindow) -> None:
    panel = window._panel
    font = panel._controls["osd_font_path"]
    assert not any(w.isEnabled() for w in font.widgets)
    panel._controls["osd_redraw"].put(True)
    panel._update_needs()
    assert all(w.isEnabled() for w in font.widgets)


def test_reset_to_defaults(window: MainWindow) -> None:
    panel = window._panel
    panel.set_params(RemasterParams(warmth=50, superview="osd_out", motion_blur=20))
    panel.reset_button.click()
    assert panel.params() == RemasterParams()


def test_settings_are_remembered_between_runs(qapp: QApplication, tmp_path: Path) -> None:
    settings = QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat)
    first = MainWindow(settings)
    chosen = RemasterParams(warmth=-30, saturation=150, stabilisation=40)
    first._panel.set_params(chosen, announce=True)
    first._quitting = True
    first._shutdown()
    second = MainWindow(QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat))
    try:
        assert second._panel.params() == chosen
    finally:
        second._quitting = True
        second._shutdown()


def test_slider_changes_ask_for_a_new_preview_of_the_playhead_frame(
    window: MainWindow, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from anahd.app.framepreview import Request

    asked: list[Request] = []
    monkeypatch.setattr(window._previewer, "request", asked.append)
    c = _clip(window, tmp_path)
    window._list.setCurrentRow(0)
    window._load_clip(c)
    window._timeline.set_playhead(4.25)
    chosen = RemasterParams(warmth=35)
    window._panel.set_params(chosen, announce=True)
    last = asked[-1]
    assert (last.path, last.seconds, last.params, last.learned) == (c.path, 4.25, chosen, True)
    window._set_engine("classic")
    assert asked[-1].learned is False and asked[-1].params == chosen


def test_panel_is_locked_while_cleaning(
    window: MainWindow, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(type(window._runner), "busy", property(lambda self: True))
    window._refresh()
    assert not window._panel.isEnabled()


def test_spinner_shows_while_a_preview_renders(window: MainWindow) -> None:
    window.show()
    window._previewer.busy.emit(True)
    assert not window._spinner.isHidden() and "Rendering" in window._preview_state.text()
    window._previewer.busy.emit(False)
    assert window._spinner.isHidden()


def test_speed_and_warning_follow_the_settings(window: MainWindow) -> None:
    panel = window._panel
    panel.set_params(RemasterParams(), announce=True)
    assert "about 21 fps" in panel.speed.text() and "Below" not in panel.speed.text()
    panel.set_params(
        RemasterParams(motion_blur=39, stabilisation=50, superview="stretch", sharpen=50),
        announce=True,
    )
    assert "Below 18 fps" in panel.speed.text() and panel.speed.property("role") == "warning"


def test_each_effect_shows_its_speed_cost(window: MainWindow) -> None:
    from anahd.app.speed import costs

    panel = window._panel
    panel.set_params(RemasterParams(), announce=True)
    expected = costs(RemasterParams(), window._learned())
    for name, pct in expected.items():
        label = panel._costs[name]
        assert not label.isHidden() and f"-{pct:.0f} % speed" in label.text(), name
    assert panel._costs["warmth"].isHidden()  # no measured cost: nothing shown
    window._set_engine("classic")
    assert panel._costs["sharpen"].text() == "<1 % speed"  # within noise for Classic


def test_moving_numbers_wait_for_the_osd_redraw(window: MainWindow) -> None:
    panel = window._panel
    live = panel._controls["osd_redraw_live"]
    assert panel.params().osd_redraw is False and panel.params().osd_redraw_live is False
    assert not panel.sections["Advanced"].isVisibleTo(panel)  # in the folded section
    assert not any(w.isEnabled() for w in live.widgets)
    panel._controls["osd_redraw"].put(True)
    panel._update_needs()
    assert all(w.isEnabled() for w in live.widgets)


class _TextLog(QPainter):
    """A painter that records the text drawn with it."""

    def __init__(self, image: QImage) -> None:
        super().__init__(image)
        self.texts: list[str] = []

    def drawText(self, *args: Any) -> None:  # type: ignore[override]
        self.texts.append(str(args[-1]))
        super().drawText(*args)


@pytest.mark.parametrize(
    "status", ["", "Queued", "Done   0:59   16.5 fps", "42%   21.5 fps   0:32 left"]
)
@pytest.mark.parametrize("width", [405, 700])  # the list at its narrowest, and wide
def test_clip_card_keeps_the_name_whole_whatever_the_status(
    qapp: QApplication, status: str, width: int
) -> None:
    card = ClipCard(qapp)
    model = QStandardItemModel(1, 1)
    index = model.index(0, 0)
    model.setData(index, "D:/flights/sunny.mp4", PATH)
    model.setData(index, "0:20   1280x720   60 fps", DETAILS)
    model.setData(index, status, STATUS)
    image = QImage(width, CARD_HEIGHT, QImage.Format.Format_ARGB32)
    option = QStyleOptionViewItem()
    option.rect = QRect(0, 0, width, CARD_HEIGHT)
    painter = _TextLog(image)
    card.paint(painter, option, index)
    painter.end()
    assert "sunny.mp4" in painter.texts  # never shortened by the status
    if width == 700:  # room for everything: nothing is shortened
        assert "0:20   1280x720   60 fps" in painter.texts
        assert status in painter.texts


def test_settings_panel_is_never_cut_off_on_the_right(window: MainWindow) -> None:
    """With every section open, the widest row still fits (there is no sideways scroll)."""
    window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
    window.resize(1720, 980)
    window.show()
    window._pages.setCurrentWidget(window._list_page)  # the clips page, as with a clip in
    panel = window._panel
    for header in panel.findChildren(QPushButton):
        if header.text().startswith("▸"):
            header.click()
    for theme_name in ("dark", "light"):
        window._apply_theme(theme_name)
        QCoreApplication.processEvents()
        QTest.qWait(50)
        body = panel.widget()
        assert body is not None
        assert body.width() <= panel.viewport().width(), theme_name
        assert body.minimumSizeHint().width() <= panel.viewport().width(), theme_name
        # ...and no row is so wide that the panel squeezes the clip list next to it
        assert panel.minimumSizeHint().width() <= 320, theme_name


# ----- the live view: stages, pictures, following the batch -----


class _AliveProc:
    exitcode: int | None = None

    def is_alive(self) -> bool:
        return True

    def join(self, timeout: float | None = None) -> None:
        pass


def _runner(
    tmp_path: Path,
) -> tuple[worker.Runner, queue.Queue[tuple[Any, ...]], dict[str, list[Any]]]:
    r = worker.Runner()
    q: queue.Queue[tuple[Any, ...]] = queue.Queue()
    r._queue, r._proc, r._src = q, _AliveProc(), tmp_path / "a.mp4"  # type: ignore[assignment]
    got: dict[str, list[Any]] = {
        k: [] for k in ("stage", "progress", "frame", "finished", "failed")
    }
    r.stage.connect(lambda *a: got["stage"].append(a))
    r.progress.connect(lambda *a: got["progress"].append(a))
    r.frame.connect(lambda *a: got["frame"].append(a))
    r.finished.connect(lambda *a: got["finished"].append(a))
    r.failed.connect(lambda *a: got["failed"].append(a))
    return r, q, got


def test_ongoing_messages_never_end_a_clip(qapp: QApplication, tmp_path: Path) -> None:
    r, q, got = _runner(tmp_path)
    q.put(("stage", "Starting"))
    q.put(("stage", "Loading the models"))
    q.put(("frame", b"b1", b"a1", 1, 100))
    q.put(("frame", b"b2", b"a2", 90, 100))
    q.put(("progress", 30, 100, 20.0, 5.0))
    q.put(("from a newer version", 1))
    r._poll()
    assert r.busy and not got["failed"] and not got["finished"]
    assert got["stage"] == [(r._src, "Loading the models")]  # the newest of each kind
    assert got["frame"] == [(r._src, b"b2", b"a2", 90, 100)]
    assert got["progress"] == [(r._src, 30, 100, 5.0)]


def test_an_ending_drops_what_was_still_ongoing(qapp: QApplication, tmp_path: Path) -> None:
    r, q, got = _runner(tmp_path)
    q.put(("frame", b"b", b"a", 5, 10))
    q.put(("done", 10, 1.0, (0, 0, 8, 8), ""))
    r._poll()
    assert got["finished"] and not got["frame"] and not r.busy


def test_a_trailing_picture_after_the_worker_exits_is_not_a_failure(
    qapp: QApplication, tmp_path: Path
) -> None:
    r, q, got = _runner(tmp_path)

    class Gone(_AliveProc):
        exitcode = 0

        def is_alive(self) -> bool:
            return False

    r._proc = Gone()  # type: ignore[assignment]
    q.put(("frame", b"b", b"a", 5, 10))
    q.put(("stage", "Cleaning"))
    q.put(("done", 10, 1.0, (0, 0, 8, 8), ""))
    r._poll()
    assert got["finished"] and not got["failed"]


# Regression: ISSUE-001 — Stop pressed while a live picture is being shown turned the
# stopped clip into "Failed: the worker stopped unexpectedly (exit code -15)"
# Found by /qa on 2026-10-29
# Report: .gstack/qa-reports/run-20261029T051405Z/qa-report-anahd-2026-10-29.md
def test_stopping_from_a_picture_handler_is_not_a_failure(
    qapp: QApplication, tmp_path: Path
) -> None:
    r, q, got = _runner(tmp_path)

    class Killable(_AliveProc):
        alive = True

        def is_alive(self) -> bool:
            return self.alive

        def kill(self) -> None:
            self.alive = False

    r._proc = Killable()  # type: ignore[assignment]
    r.stage.connect(lambda *a: r.stop())  # Stop pressed while the first signal is handled
    q.put(("stage", "Cleaning"))
    q.put(("progress", 30, 100, 20.0, 5.0))
    q.put(("frame", b"b", b"a", 5, 10))
    r._poll()
    assert got["stage"] and not r.busy
    assert not got["failed"], "a deliberate Stop is not a failure"
    assert not got["progress"] and not got["frame"], "nothing is reported after Stop"


def _pick(window: MainWindow, row: int) -> None:
    """A click: this clip alone is selected."""
    window._list.setCurrentRow(row, QItemSelectionModel.SelectionFlag.ClearAndSelect)


def _busy(monkeypatch: pytest.MonkeyPatch, window: MainWindow, on: bool = True) -> None:
    monkeypatch.setattr(type(window._runner), "busy", property(lambda self: on))


def _jpeg(color: int) -> bytes:
    import cv2
    import numpy as np

    ok, data = cv2.imencode(".jpg", np.full((48, 64, 3), color, np.uint8))
    assert ok
    return bytes(data.tobytes())


def test_clean_selects_the_clip_and_shows_it_live(
    window: MainWindow, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started: list[Path] = []
    monkeypatch.setattr(window._runner, "start", lambda src, *a: started.append(src))
    requests: list[object] = []
    monkeypatch.setattr(window._previewer, "request", requests.append)
    a, _b = _clip(window, tmp_path, "a.mp4"), _clip(window, tmp_path, "b.mp4")
    _pick(window, 1)  # looking at b
    requests.clear()
    _busy(monkeypatch, window)
    window._start_all()
    assert started == [a.path] and window._selected() is a, "Clean shows what it cleans"
    assert not requests, "no GPU preview while the GPU is cleaning"
    assert window._preview_heading.text() == "Preview · before / after paused while cleaning"

    window._stage_reported(a.path, "Loading the models")
    assert a.item.data(STATUS) == "Loading the models..."
    window._live_picture(a.path, _jpeg(40), _jpeg(200), 1200, 600)
    assert window._stage.currentWidget() is window._frame_wipe
    assert not window._live_row.isHidden()
    assert "Frame 1,200 of 600" in window._live_caption.text()
    assert "0:20 into the cleaned clip" in window._live_caption.text()
    assert window._preview_heading.text().startswith("Cleaning · live")

    _pick(window, 1)  # the user looks at b meanwhile: its plain frame
    assert not requests and window._preview_heading.text().endswith("paused while cleaning")
    _pick(window, 0)  # back to a: the latest picture at once
    assert window._stage.currentWidget() is window._frame_wipe


def test_the_live_view_follows_the_batch_unless_the_user_looked_away(
    window: MainWindow, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started: list[Path] = []
    monkeypatch.setattr(window._runner, "start", lambda src, *a: started.append(src))
    monkeypatch.setattr(window._previewer, "request", lambda req: None)
    a, b, c = (_clip(window, tmp_path, f"{n}.mp4") for n in "abc")
    _busy(monkeypatch, window)
    window._start_all()
    window._finished(a.path, (600, 10.0, (0, 0, 960, 720), ""))
    assert started == [a.path, b.path] and window._selected() is b, "followed to b"
    window._live_picture(a.path, _jpeg(1), _jpeg(2), 5, 600)  # late picture of a: ignored
    assert a.path not in window._live
    _pick(window, 0)  # the user looks at a (done)
    window._failed(b.path, "boom")
    assert started == [a.path, b.path, c.path] and window._selected() is a, "not taken away"


def test_a_preview_asked_for_before_clean_is_not_shown_after(
    window: MainWindow, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from anahd.app.framepreview import Request, Shown

    a = _clip(window, tmp_path, "a.mp4")
    _pick(window, 0)
    _busy(monkeypatch, window)
    image = QImage(64, 48, QImage.Format.Format_RGB888)
    shown = Shown(Request(a.path, 0.0, RemasterParams(), True), image, image, 0.0)
    window._stage.setCurrentWidget(window._screen)
    window._preview_ready(shown)
    assert window._stage.currentWidget() is window._screen


def test_finish_time_uses_the_speed_measured_here(
    window: MainWindow, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from anahd.app import window as win_mod

    a, b = _clip(window, tmp_path, "a.mp4"), _clip(window, tmp_path, "b.mp4")
    a.state, b.state = "cleaning", "queued"
    window._batch = [a, b]
    assert window._finish_time() == "", "nothing while the time left is settling"
    now = time.monotonic()
    # Running at 40 fps (an estimate of 20 fps would have said twice as long): b's 10 s at
    # 60 fps is 600 frames, 15 s here, plus its start-up.
    window._left = (100.0, 40.0, now)
    seen: list[int] = []

    class Clock:
        @staticmethod
        def currentTime() -> Any:
            class T:
                def addSecs(self, s: int) -> int:
                    seen.append(s)
                    return s

            return T()

    monkeypatch.setattr(win_mod, "QTime", Clock)
    monkeypatch.setattr(QLocale, "toString", lambda self, t, f: f"t+{t}")
    assert window._finish_time() == f"t+{seen[-1]}"
    assert abs(seen[-1] - (100 + 15 + win_mod.START_S)) <= 1
    window._left = None  # a retry: its time left is settling again
    assert window._finish_time() == ""
