"""The app's look, shared with FPV Editor (after dvsalas.com): near-black (or near-white)
surfaces, thin borders, small uppercase mono labels, one red accent.

Colours are tokens in a palette per theme; `stylesheet` turns a palette into Qt style
sheet rules. Fonts: Geist, Geist Mono and Doto when their files are in ``fonts/`` (SIL
Open Font License), otherwise Windows' own Segoe UI and Consolas.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtGui import QColor, QFont, QFontDatabase, QPalette
from PySide6.QtWidgets import QApplication

FONTS = Path(__file__).parent / "fonts"


@dataclass(frozen=True)
class Palette:
    name: str
    bg: str
    surface: str
    surface_2: str
    border: str
    border_strong: str
    text: str
    muted: str
    faint: str
    accent: str
    dot: str  # the dotted backdrop of empty areas


DARK = Palette(
    name="dark",
    bg="#0a0a0a",
    surface="#111111",
    surface_2="#171717",
    border="#222222",
    border_strong="#333333",
    text="#ededed",
    muted="#a1a1a1",
    faint="#6e6e6e",
    accent="#e5322d",
    dot="#1c1c1c",
)
LIGHT = Palette(
    name="light",
    bg="#fafafa",
    surface="#ffffff",
    surface_2="#f2f2f2",
    border="#e6e6e6",
    border_strong="#d4d4d4",
    text="#0a0a0a",
    muted="#5c5c5c",
    faint="#8a8a8a",
    accent="#d71921",
    dot="#e4e4e4",
)
THEMES = {DARK.name: DARK, LIGHT.name: LIGHT}


@dataclass(frozen=True)
class Fonts:
    sans: str
    mono: str
    dot: str | None  # the dot-matrix display face, if installed


def load_fonts() -> Fonts:
    """Register the bundled font files (if any) and pick the families to use."""
    if FONTS.is_dir():
        for path in sorted(FONTS.glob("*.[ot]tf")):
            QFontDatabase.addApplicationFont(str(path))
    families = set(QFontDatabase.families())

    def first(*names: str) -> str:
        return next((name for name in names if name in families), names[-1])

    return Fonts(
        sans=first("Geist", "Segoe UI", "Arial"),
        mono=first("Geist Mono", "Cascadia Mono", "Consolas", "Courier New"),
        dot="Doto" if "Doto" in families else None,
    )


def mono_font(fonts: Fonts, size: float = 9.0, weight: int = 400) -> QFont:
    """The site's label face: small, uppercase in the text itself, slightly spaced."""
    font = QFont(fonts.mono)
    font.setPointSizeF(size)
    font.setWeight(QFont.Weight(weight))
    font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 0.7)
    return font


def apply(app: QApplication, palette: Palette, fonts: Fonts) -> None:
    """Restyle the whole application in ``palette``."""
    app.setStyle("Fusion")  # draws every widget from the palette and style sheet
    qt = QPalette()
    for role, colour in (
        (QPalette.ColorRole.Window, palette.bg),
        (QPalette.ColorRole.WindowText, palette.text),
        (QPalette.ColorRole.Base, palette.surface),
        (QPalette.ColorRole.AlternateBase, palette.surface_2),
        (QPalette.ColorRole.Text, palette.text),
        (QPalette.ColorRole.Button, palette.surface),
        (QPalette.ColorRole.ButtonText, palette.text),
        (QPalette.ColorRole.Highlight, palette.surface_2),
        (QPalette.ColorRole.HighlightedText, palette.text),
        (QPalette.ColorRole.ToolTipBase, palette.surface_2),
        (QPalette.ColorRole.ToolTipText, palette.text),
        (QPalette.ColorRole.PlaceholderText, palette.faint),
        (QPalette.ColorRole.Link, palette.accent),
    ):
        qt.setColor(role, QColor(colour))
    qt.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor(palette.faint))
    qt.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor(palette.faint))
    app.setPalette(qt)
    sans = QFont(fonts.sans)
    sans.setPointSizeF(9.5)
    app.setFont(sans)
    app.setStyleSheet(stylesheet(palette, fonts))


def stylesheet(p: Palette, fonts: Fonts) -> str:
    mono = f'"{fonts.mono}"'
    return f"""
    QMainWindow, QDialog {{ background: {p.bg}; }}
    QWidget {{ color: {p.text}; }}
    QToolTip {{ background: {p.surface_2}; color: {p.text}; border: 1px solid {p.border_strong};
        padding: 6px; }}

    /* small uppercase labels: section numbers, captions, field names */
    QLabel[role="eyebrow"] {{ color: {p.faint}; font-family: {mono}; font-size: 8pt; }}
    QLabel[role="caption"] {{ color: {p.muted}; font-family: {mono}; font-size: 8pt; }}
    QLabel[role="muted"] {{ color: {p.muted}; }}
    QLabel[role="status"] {{ color: {p.muted}; font-family: {mono}; font-size: 8pt; }}
    QLabel[role="step"] {{ color: {p.faint}; font-family: {mono}; font-size: 8pt; }}
    QLabel[role="step-on"] {{ color: {p.text}; font-family: {mono}; font-size: 8pt; }}
    QLabel[role="hero"] {{ color: {p.muted}; font-size: 11pt; }}
    QLabel[role="heading"] {{ color: {p.text}; font-size: 20pt; font-weight: 600; }}
    QLabel[role="big-number"] {{ color: {p.text}; font-family: {mono}; font-size: 40pt; }}
    QPushButton[role="step"] {{ border: none; background: transparent; padding: 6px 4px;
        color: {p.faint}; }}
    QPushButton[role="step"]:enabled {{ color: {p.muted}; }}
    QPushButton[role="step"]:enabled:hover {{ color: {p.text}; }}
    QPushButton[role="step"][state="on"] {{ color: {p.text}; }}
    QLabel[role="logo"] {{ font-family: {mono}; font-size: 12pt; font-weight: 600; }}

    QFrame[role="panel"] {{ background: {p.bg}; border: 1px solid {p.border};
        border-radius: 8px; }}
    QFrame[role="header"] {{ background: {p.bg}; border: none;
        border-bottom: 1px solid {p.border}; }}
    QFrame[role="footer"] {{ background: {p.bg}; border: none;
        border-top: 1px solid {p.border}; }}

    QPushButton {{ background: transparent; color: {p.muted}; border: 1px solid {p.border};
        border-radius: 6px; padding: 6px 12px; font-family: {mono}; font-size: 8pt; }}
    QPushButton:hover {{ color: {p.text}; border-color: {p.border_strong};
        background: {p.surface_2}; }}
    QPushButton:pressed {{ background: {p.surface}; }}
    QPushButton:disabled {{ color: {p.faint}; border-color: {p.border}; background: transparent; }}
    QPushButton[primary="true"] {{ background: {p.text}; color: {p.bg}; border-color: {p.text};
        font-weight: 600; }}
    QPushButton[primary="true"]:hover {{ background: {p.muted}; border-color: {p.muted}; }}
    QPushButton[primary="true"]:disabled {{ background: {p.surface_2}; color: {p.faint};
        border-color: {p.border}; }}
    QPushButton[flat="true"] {{ border-color: transparent; }}
    QPushButton::menu-indicator {{ image: none; width: 0; }}
    QPushButton[flat="true"]:hover {{ border-color: {p.border}; }}

    QListWidget {{ background: transparent; border: none; outline: none; }}
    QListWidget::item {{ color: {p.text}; border: 1px solid transparent; border-radius: 6px;
        padding: 4px; margin: 2px 0; }}
    QListWidget::item:hover {{ background: {p.surface}; border-color: {p.border}; }}
    QListWidget::item:selected {{ background: {p.surface_2}; border-color: {p.border_strong};
        color: {p.text}; }}

    QComboBox, QDoubleSpinBox, QLineEdit {{ background: {p.surface}; color: {p.text};
        border: 1px solid {p.border}; border-radius: 6px; padding: 4px 8px;
        font-family: {mono}; font-size: 8pt; }}
    QComboBox:hover, QDoubleSpinBox:hover {{ border-color: {p.border_strong}; }}
    QComboBox QAbstractItemView {{ background: {p.surface}; border: 1px solid {p.border_strong};
        selection-background-color: {p.surface_2}; color: {p.text}; }}
    QComboBox::drop-down {{ border: none; width: 18px; }}
    QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{ width: 0; border: none; }}

    QProgressBar {{ background: {p.border}; border: none; border-radius: 1px;
        max-height: 2px; min-height: 2px; }}
    QProgressBar::chunk {{ background: {p.accent}; border-radius: 1px; }}

    QScrollBar:vertical {{ background: transparent; width: 8px; margin: 2px; }}
    QScrollBar:horizontal {{ background: transparent; height: 8px; margin: 2px; }}
    QScrollBar::handle {{ background: {p.border_strong}; border-radius: 3px; min-height: 24px;
        min-width: 24px; }}
    QScrollBar::handle:hover {{ background: {p.faint}; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
    QScrollBar::add-page, QScrollBar::sub-page {{ background: none; }}

    QSplitter::handle {{ background: {p.bg}; }}
    QSplitter::handle:horizontal {{ width: 12px; }}
    QSplitter::handle:vertical {{ height: 12px; }}

    QMenuBar {{ background: {p.bg}; color: {p.muted}; font-family: {mono}; font-size: 8pt; }}
    QMenuBar::item {{ padding: 4px 10px; border-radius: 6px; }}
    QMenuBar::item:selected {{ background: {p.surface_2}; color: {p.text}; }}
    QMenu {{ background: {p.surface}; border: 1px solid {p.border_strong}; padding: 4px; }}
    QMenu::item {{ padding: 6px 24px 6px 12px; border-radius: 4px; color: {p.text}; }}
    QMenu::item:selected {{ background: {p.surface_2}; }}
    QMenu::separator {{ height: 1px; background: {p.border}; margin: 4px 6px; }}

    QMessageBox {{ background: {p.bg}; }}
    """
