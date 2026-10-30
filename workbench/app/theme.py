"""Day and night looks: one table of colours (tokens), turned into a Qt style sheet, a palette and plot settings.

Colours are copied from design/preview/index.html: day = look A "test bench" (light greys, steel blue), night =
look B "control room". A token is a named colour ("panel", "acc", "x"), so widgets ask for a role, not a hex value,
and both themes stay complete: test_theme.py checks every token exists in both.

Qt styles widgets three ways, and all three are set here because each reaches different things:
- QSS, the style sheet (CSS-like rules), for our widgets: square corners, 1-px lines, compact heights;
- QPalette, the standard colour roles, for what QSS doesn't reach (dialogs, Qt's own tooltips and scroll areas);
- pyqtgraph's global options for the plots, with antialiasing OFF so lines stay 1 px sharp (spec §8b).
The font is IBM Plex Sans (SIL Open Font License, fonts/OFL.txt), bundled so it works offline and on every laptop.
"""
from __future__ import annotations

import re
from pathlib import Path
from string import Template

import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QFontDatabase, QPalette

FONT_FILE = Path(__file__).parent / "fonts" / "IBMPlexSans-VF.ttf"
FONT_PX = 12

TOKENS: dict[str, dict[str, str]] = {
    "day": {     # look A, test bench
        "bg": "#DADDE1", "panel": "#FFFFFF", "head": "#EEF0F2", "line": "#C3C8CE", "grid": "#ECEEF0",
        "ink": "#1A1D21", "ink2": "#5E6670", "acc": "#1D5FA8", "acc_wash": "rgba(29, 95, 168, .10)",
        "field": "#FFFFFF",
        "x": "#E03E2D", "y": "#2B9A47", "z": "#0A6FDB", "volt": "#5E6670",
        "s0": "#B8BEC5", "s1": "#4F9A5A", "s2": "#D49A2A", "s3": "#B3261E",
        "wf0": "#FFFFFF", "wf1": "#A9C2E2", "wf2": "#1D5FA8", "wf3": "#0A2342",
        "out": "#F2C200", "out_ink": "#1A1D21",
    },
    "night": {   # look B, control room
        "bg": "#121417", "panel": "#1C1F23", "head": "#22262B", "line": "#31363C", "grid": "#262A2F",
        "ink": "#D3D8DD", "ink2": "#8A939C", "acc": "#6FA8DC", "acc_wash": "rgba(111, 168, 220, .14)",
        "field": "#15181B",
        "x": "#FF6B5E", "y": "#4CC46A", "z": "#4DA3FF", "volt": "#A7AFB7",
        "s0": "#3A4047", "s1": "#4E9A5C", "s2": "#D9A23A", "s3": "#D9574E",
        "wf0": "#1C1F23", "wf1": "#1E3A5C", "wf2": "#4F8CC9", "wf3": "#E8F1FA",
        "out": "#E8B500", "out_ink": "#121417",
    },
}

# The style sheet, written once with $token placeholders. Heights follow the preview: 24 px rows, 22 px fields.
_QSS = Template("""
* { outline: 0; }
QMainWindow, QDialog { background: $bg; color: $ink; }
QWidget { color: $ink; }
QToolTip { background: $head; color: $ink; border: 1px solid $line; padding: 2px 4px; }

QMenuBar { background: $head; border-bottom: 1px solid $line; padding: 0 4px; }
QMenuBar::item { padding: 4px 8px; background: transparent; }
QMenuBar::item:selected { background: $acc_wash; }
QMenu { background: $panel; border: 1px solid $line; padding: 2px 0; }
QMenu::item { padding: 4px 24px 4px 12px; }
QMenu::item:selected { background: $acc_wash; color: $ink; }
QMenu::item:disabled { color: $ink2; }
QMenu::separator { height: 1px; background: $line; margin: 2px 0; }

QToolBar { background: $head; border: 0; border-bottom: 1px solid $line; padding: 0 6px; spacing: 2px; }
QToolBar::separator { width: 1px; background: $line; margin: 5px 6px; }
QToolButton { background: transparent; border: 1px solid transparent; border-radius: 0; padding: 2px 6px; min-height: 18px; }
QToolButton:hover { background: $panel; border-color: $line; }
QToolButton:checked { background: $acc_wash; border-color: $acc; }

QTabBar { background: $head; }
QTabBar::tab { background: transparent; color: $ink2; padding: 6px 14px; border: 0; border-bottom: 2px solid transparent; }
QTabBar::tab:selected { color: $ink; border-bottom-color: $acc; }
QTabBar::tab:hover { color: $ink; }

QDockWidget { titlebar-close-icon: none; }
QDockWidget::title { background: $head; border-bottom: 1px solid $line; padding: 4px 8px; text-align: left; }
QMainWindow::separator { background: $bg; width: 3px; height: 3px; }

QTreeView, QListView, QTableView { background: $panel; border: 0; alternate-background-color: $panel;
    selection-background-color: $acc_wash; selection-color: $ink; show-decoration-selected: 1; }
QTreeView::item, QListView::item { height: 22px; }
QTreeView::item:hover, QListView::item:hover { background: $acc_wash; }
QTreeView::item:selected, QListView::item:selected { background: $acc_wash; color: $ink; }
QHeaderView::section { background: $head; color: $ink2; border: 0; border-bottom: 1px solid $grid; padding: 3px 8px; }

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox { background: $field; border: 1px solid $line; border-radius: 0; padding: 0 5px; min-height: 20px; selection-background-color: $acc; selection-color: $panel; }
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus { border-color: $acc; }
QComboBox QAbstractItemView { background: $panel; border: 1px solid $line; selection-background-color: $acc_wash; selection-color: $ink; }
QPushButton { background: $panel; border: 1px solid $line; border-radius: 0; padding: 0 12px; min-height: 20px; }
QPushButton:hover { border-color: $acc; }
QPushButton:default { background: $acc; border-color: $acc; color: $panel; }
QCheckBox::indicator { width: 12px; height: 12px; border: 1px solid $line; background: $field; }
QCheckBox::indicator:checked { background: $acc; border-color: $acc; }

QStatusBar { background: $head; border-top: 1px solid $line; color: $ink2; }
QStatusBar::item { border: 0; border-right: 1px solid $line; }

QScrollBar:vertical { background: $panel; width: 10px; margin: 0; }
QScrollBar:horizontal { background: $panel; height: 10px; margin: 0; }
QScrollBar::handle { background: $line; min-height: 24px; min-width: 24px; }
QScrollBar::handle:hover { background: $ink2; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: none; }
""")

_font_family: str | None = None                      # loaded once per process


def _qt_colour(value: str) -> str:
    """CSS 'rgba(29, 95, 168, .10)' → Qt 'rgba(29, 95, 168, 26)': Qt reads alpha as 0–255, not 0–1."""
    m = re.fullmatch(r"rgba\((\d+), (\d+), (\d+), ([\d.]+)\)", value)
    return f"rgba({m[1]}, {m[2]}, {m[3]}, {round(float(m[4]) * 255)})" if m else value


def build_qss(tokens: dict[str, str]) -> str:
    """The style sheet for one theme."""
    return _QSS.substitute({k: _qt_colour(v) for k, v in tokens.items()})


def resolve(name: str, app) -> str:
    """'day' / 'night' as given; 'system' follows the OS dark mode; anything else → 'day'."""
    if name in TOKENS:
        return name
    if name == "system" and app.styleHints().colorScheme() == Qt.ColorScheme.Dark:
        return "night"
    return "day"


def _palette(t: dict[str, str]) -> QPalette:
    p = QPalette()
    for role, key in ((QPalette.Window, "bg"), (QPalette.WindowText, "ink"), (QPalette.Base, "panel"),
                      (QPalette.AlternateBase, "head"), (QPalette.Text, "ink"), (QPalette.Button, "panel"),
                      (QPalette.ButtonText, "ink"), (QPalette.ToolTipBase, "head"), (QPalette.ToolTipText, "ink"),
                      (QPalette.Highlight, "acc"), (QPalette.HighlightedText, "panel"),
                      (QPalette.PlaceholderText, "ink2"), (QPalette.Mid, "line"), (QPalette.Link, "acc")):
        p.setColor(role, QColor(t[key]))
    p.setColor(QPalette.Disabled, QPalette.Text, QColor(t["ink2"]))
    p.setColor(QPalette.Disabled, QPalette.WindowText, QColor(t["ink2"]))
    return p


def _load_font() -> str | None:
    global _font_family
    if _font_family is None:
        font_id = QFontDatabase.addApplicationFont(str(FONT_FILE))
        families = QFontDatabase.applicationFontFamilies(font_id) if font_id >= 0 else []
        _font_family = families[0] if families else ""
    return _font_family or None


def apply_theme(app, name: str) -> str:
    """Style the whole app as `name` ('day', 'night' or 'system'); returns the look actually used."""
    look = resolve(name, app)
    t = TOKENS[look]
    family = _load_font()
    if family:                                       # if the file were missing, Qt's default font is kept
        font = QFont(family)
        font.setPixelSize(FONT_PX)
        app.setFont(font)
    app.setPalette(_palette(t))
    app.setStyleSheet(build_qss(t))
    pg.setConfigOptions(antialias=False, background=t["panel"], foreground=t["ink2"])
    return look
