"""The app's look, after dvsalas.com: charcoal (or warm off-white) surfaces, thin
borders, Space Grotesk text with small Geist Mono labels, one amber accent.

Colours are tokens in a palette per theme, named as on the site; `stylesheet` turns a
palette into Qt style sheet rules. Fonts: Space Grotesk, Geist Mono and Doto from
``fonts/`` (SIL Open Font License), otherwise Windows' own Segoe UI and Consolas.
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
    surface: str  # the site's --panel
    surface_2: str  # --panel-2
    border: str  # --line
    border_strong: str  # --line-strong
    field: str  # --field-line, the border of text fields
    text: str
    muted: str
    faint: str
    accent: str


DARK = Palette(
    name="dark",
    bg="#121212",
    surface="#1a1a1a",
    surface_2="#212121",
    border="#2a2a2a",
    border_strong="#3d3d3d",
    field="#5e5e5e",
    text="#ededed",
    muted="#a3a3a3",
    faint="#8c8c8c",
    accent="#f2a93b",
)
LIGHT = Palette(
    name="light",
    bg="#f7f7f5",
    surface="#efefec",
    surface_2="#e7e7e3",
    border="#dededa",
    border_strong="#c4c4bf",
    field="#8f8f8a",
    text="#171717",
    muted="#5f5f5f",
    faint="#6e6e6e",
    accent="#b45309",
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
        sans=first("Space Grotesk", "Segoe UI", "Arial"),
        mono=first("Geist Mono", "Cascadia Mono", "Consolas", "Courier New"),
        dot="Doto" if "Doto" in families else None,
    )


def mono_font(fonts: Fonts, size: float = 9.0, weight: int = 400) -> QFont:
    """The site's label face: small, sentence case, unspaced."""
    font = QFont(fonts.mono)
    font.setPointSizeF(size)
    font.setWeight(QFont.Weight(weight))
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
        (QPalette.ColorRole.Highlight, palette.accent),  # slider fills, check marks
        (QPalette.ColorRole.HighlightedText, palette.bg),
        (QPalette.ColorRole.ToolTipBase, palette.surface_2),
        (QPalette.ColorRole.ToolTipText, palette.text),
        (QPalette.ColorRole.PlaceholderText, palette.faint),
        (QPalette.ColorRole.Link, palette.accent),
    ):
        qt.setColor(role, QColor(colour))
    qt.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor(palette.faint))
    qt.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor(palette.faint))
    qt.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Highlight, QColor(palette.faint))
    app.setPalette(qt)
    sans = QFont(fonts.sans)
    sans.setPointSizeF(10)
    app.setFont(sans)
    app.setStyleSheet(stylesheet(palette, fonts))


def stylesheet(p: Palette, fonts: Fonts) -> str:
    mono = f'"{fonts.mono}"'
    sans = f'"{fonts.sans}"'
    return f"""
    QMainWindow, QDialog {{ background: {p.bg}; }}
    QWidget {{ color: {p.text}; }}
    QToolTip {{ background: {p.surface_2}; color: {p.text}; border: 1px solid {p.border_strong};
        border-radius: 8px; padding: 6px; }}

    /* small mono labels: section names, captions, field names */
    QLabel[role="eyebrow"] {{ color: {p.faint}; font-family: {mono}; font-size: 9pt; }}
    QLabel[role="caption"] {{ color: {p.muted}; font-family: {mono}; font-size: 9pt; }}
    QLabel[role="muted"] {{ color: {p.muted}; }}
    QLabel[role="status"] {{ color: {p.muted}; font-family: {mono}; font-size: 9pt; }}
    QLabel[role="warning"] {{ color: {p.accent}; font-family: {mono}; font-size: 9pt; }}
    QLabel[role="step"] {{ color: {p.faint}; font-family: {mono}; font-size: 9pt; }}
    QLabel[role="step-on"] {{ color: {p.text}; font-family: {mono}; font-size: 9pt; }}
    QLabel[role="hero"] {{ color: {p.muted}; font-size: 12.5pt; }}
    QLabel[role="heading"] {{ color: {p.text}; font-size: 20pt; font-weight: 600; }}
    QLabel[role="big-number"] {{ color: {p.text}; font-family: {mono}; font-size: 40pt; }}
    QPushButton[role="step"] {{ border: none; background: transparent; padding: 8px 10px;
        color: {p.faint}; font-family: {sans}; font-size: 10.5pt; font-weight: 400; }}
    QPushButton[role="step"]:enabled {{ color: {p.muted}; }}
    QPushButton[role="step"]:enabled:hover {{ color: {p.text}; background: transparent; }}
    QPushButton[role="step"][state="on"] {{ color: {p.text}; }}
    QLabel[role="logo"] {{ font-family: {sans}; font-size: 16pt; font-weight: 600; }}

    QFrame[role="panel"] {{ background: {p.surface}; border: 1px solid {p.border};
        border-radius: 12px; }}
    QFrame[role="header"] {{ background: {p.bg}; border: none;
        border-bottom: 1px solid {p.border}; }}
    QFrame[role="footer"] {{ background: {p.bg}; border: none;
        border-top: 1px solid {p.border}; }}

    /* the site's .btn: panel fill, strong line, the line turns to text colour on hover */
    QPushButton {{ background: {p.surface}; color: {p.text}; border: 1px solid {p.border_strong};
        border-radius: 8px; padding: 6px 14px; font-family: {sans}; font-size: 10pt;
        font-weight: 500; }}
    QPushButton:hover {{ border-color: {p.text}; }}
    QPushButton:pressed {{ background: {p.surface_2}; }}
    QPushButton:disabled {{ color: {p.faint}; border-color: {p.border}; background: {p.bg}; }}
    QPushButton[primary="true"] {{ background: {p.text}; color: {p.bg}; border-color: {p.text};
        font-weight: 600; }}
    QPushButton[primary="true"]:hover {{ background: {"#ffffff" if p.name == "dark" else "#000000"};
        border-color: {"#ffffff" if p.name == "dark" else "#000000"}; }}
    QPushButton[primary="true"]:disabled {{ background: {p.surface_2}; color: {p.faint};
        border-color: {p.border}; }}
    QPushButton[flat="true"] {{ background: transparent; border-color: transparent;
        color: {p.muted}; }}
    QPushButton::menu-indicator {{ image: none; width: 0; }}
    QPushButton[flat="true"]:hover {{ color: {p.text}; }}

    QListWidget {{ background: transparent; border: none; outline: none; }}
    QListWidget::item {{ color: {p.text}; border: 1px solid transparent; border-radius: 8px;
        padding: 4px; margin: 2px 0; }}
    QListWidget::item:hover {{ background: {p.surface}; border-color: {p.border}; }}
    QListWidget::item:selected {{ background: {p.surface_2}; border-color: {p.border_strong};
        color: {p.text}; }}

    QComboBox, QDoubleSpinBox, QLineEdit {{ background: {p.bg}; color: {p.text};
        border: 1px solid {p.field}; border-radius: 8px; padding: 4px 8px;
        font-family: {mono}; font-size: 9pt; selection-background-color: {p.accent}; }}
    QComboBox:hover, QDoubleSpinBox:hover {{ border-color: {p.text}; }}
    QLineEdit:focus, QDoubleSpinBox:focus {{ border-color: {p.text}; }}
    QLineEdit[invalid="true"] {{ border-color: {p.accent}; }}
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

    QMenuBar {{ background: {p.bg}; color: {p.muted}; font-family: {mono}; font-size: 9pt; }}
    QMenuBar::item {{ padding: 4px 10px; border-radius: 8px; }}
    QMenuBar::item:selected {{ background: {p.surface_2}; color: {p.text}; }}
    QMenu {{ background: {p.surface}; border: 1px solid {p.border_strong}; padding: 4px;
        border-radius: 8px; }}
    QMenu::item {{ padding: 6px 24px 6px 12px; border-radius: 6px; color: {p.text}; }}
    QMenu::item:selected {{ background: {p.surface_2}; }}
    QMenu::separator {{ height: 1px; background: {p.border}; margin: 4px 6px; }}

    QMessageBox {{ background: {p.bg}; }}
    """
