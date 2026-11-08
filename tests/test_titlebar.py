"""The title bar colours: the colour conversion, the non-Windows no-op, and a real read-back."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest
from PySide6.QtWidgets import QApplication, QWidget

from anahd.app import titlebar

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

COLOURS = titlebar.TitleColours(background="#121212", text="#ededed", border="#2a2a2a")
LIGHT = titlebar.TitleColours(background="#f7f7f5", text="#171717", border="#dededa")


@pytest.fixture(scope="module")
def app() -> QApplication:
    found = QApplication.instance() or QApplication([])
    assert isinstance(found, QApplication)
    return found


class Recorder:
    """Stands in for the DWM call and keeps what it was asked to set."""

    def __init__(self, accept: bool = True) -> None:
        self.calls: list[tuple[int, int]] = []
        self.accept = accept

    def __call__(self, handle: int, attribute: int, value: int) -> bool:
        self.calls.append((attribute, value))
        return self.accept


def test_colorref_is_blue_green_red_order() -> None:
    assert titlebar.colorref("#ff0000") == 0x0000FF
    assert titlebar.colorref("#00ff00") == 0x00FF00
    assert titlebar.colorref("#0000ff") == 0xFF0000
    assert titlebar.colorref("#121212") == 0x121212
    assert titlebar.colorref("#f2a93b") == 0x3BA9F2


def test_dark_and_light_backgrounds() -> None:
    assert titlebar.is_dark("#121212")
    assert not titlebar.is_dark("#f7f7f5")


def test_not_windows_does_nothing(app: QApplication, monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = Recorder()
    monkeypatch.setattr(titlebar, "_supported", lambda: False)
    monkeypatch.setattr(titlebar, "_set_attribute", recorder)
    widget = QWidget()
    widget.show()
    assert titlebar.apply_to_window(widget, COLOURS) is False
    titlebar.install(app, COLOURS)
    assert recorder.calls == []
    widget.close()


def test_windows_sets_the_four_attributes(
    app: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = Recorder()
    monkeypatch.setattr(titlebar, "_supported", lambda: True)
    monkeypatch.setattr(titlebar, "_set_attribute", recorder)
    assert titlebar.apply_to_window(QWidget(), COLOURS) is True
    assert recorder.calls == [
        (titlebar.DWMWA_USE_IMMERSIVE_DARK_MODE, 1),
        (titlebar.DWMWA_CAPTION_COLOR, 0x121212),
        (titlebar.DWMWA_TEXT_COLOR, 0xEDEDED),
        (titlebar.DWMWA_BORDER_COLOR, 0x2A2A2A),
    ]


def test_a_light_theme_turns_dark_controls_off(
    app: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = Recorder()
    monkeypatch.setattr(titlebar, "_supported", lambda: True)
    monkeypatch.setattr(titlebar, "_set_attribute", recorder)
    assert titlebar.apply_to_window(QWidget(), LIGHT) is True
    assert (titlebar.DWMWA_USE_IMMERSIVE_DARK_MODE, 0) in recorder.calls
    assert (titlebar.DWMWA_CAPTION_COLOR, 0xF5F7F7) in recorder.calls


def test_an_old_windows_build_is_ignored_quietly(
    app: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(titlebar, "_supported", lambda: True)
    monkeypatch.setattr(titlebar, "_set_attribute", Recorder(accept=False))
    assert titlebar.apply_to_window(QWidget(), COLOURS) is False


def test_a_child_widget_is_left_alone(app: QApplication, monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = Recorder()
    monkeypatch.setattr(titlebar, "_supported", lambda: True)
    monkeypatch.setattr(titlebar, "_set_attribute", recorder)
    parent = QWidget()
    assert titlebar.apply_to_window(QWidget(parent), COLOURS) is False
    assert recorder.calls == []


def test_a_theme_switch_recolours_open_windows_and_new_ones(
    app: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = Recorder()
    monkeypatch.setattr(titlebar, "_supported", lambda: True)
    monkeypatch.setattr(titlebar, "_set_attribute", recorder)
    first = QWidget()
    titlebar.install(app, COLOURS)
    first.show()
    assert (titlebar.DWMWA_CAPTION_COLOR, 0x121212) in recorder.calls
    recorder.calls.clear()
    titlebar.install(app, LIGHT)
    assert (titlebar.DWMWA_CAPTION_COLOR, 0xF5F7F7) in recorder.calls
    recorder.calls.clear()
    second = QWidget()
    second.show()
    assert (titlebar.DWMWA_CAPTION_COLOR, 0xF5F7F7) in recorder.calls
    titlebar.install(app, COLOURS)
    first.close()
    second.close()


READ_BACK = textwrap.dedent(
    """
    import ctypes, sys
    from PySide6.QtWidgets import QApplication, QWidget
    from anahd.app import titlebar

    app = QApplication([])
    colours = titlebar.TitleColours("#121212", "#ededed", "#2a2a2a")
    widget = QWidget()
    widget.show()
    if not titlebar.apply_to_window(widget, colours):
        sys.exit(3)
    dwm = ctypes.WinDLL("dwmapi")
    handle = ctypes.c_void_p(int(widget.winId()))
    # Windows lets the dark flag be read back; the colour attributes are write-only on some
    # builds, so they are checked only where the read succeeds.
    for attribute, expected, required in (
        (titlebar.DWMWA_USE_IMMERSIVE_DARK_MODE, 1, True),
        (titlebar.DWMWA_CAPTION_COLOR, 0x121212, False),
        (titlebar.DWMWA_TEXT_COLOR, 0xEDEDED, False),
        (titlebar.DWMWA_BORDER_COLOR, 0x2A2A2A, False),
    ):
        value = ctypes.c_int(-1)
        if dwm.DwmGetWindowAttribute(
            handle, attribute, ctypes.byref(value), ctypes.sizeof(value)
        ) != 0:
            if required:
                sys.exit(3)
            continue
        if value.value != expected:
            print(attribute, hex(value.value), hex(expected))
            sys.exit(1)
    """
)


@pytest.mark.skipif(sys.platform != "win32", reason="DWM exists only on Windows")
def test_windows_reports_back_the_colours_it_was_given() -> None:
    env = {key: value for key, value in os.environ.items() if key != "QT_QPA_PLATFORM"}
    done = subprocess.run(
        [sys.executable, "-c", READ_BACK], env=env, capture_output=True, text=True, timeout=60
    )
    if done.returncode == 3:
        pytest.skip("this Windows build does not support the title bar colours")
    assert done.returncode == 0, done.stdout + done.stderr
