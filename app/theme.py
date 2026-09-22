# -*- coding: utf-8 -*-
"""The dark palette, in one place: Qt widgets, the chips, and the matplotlib canvases.

Qt is themed palette-first on the Fusion style, which redraws every built-in control (spin-box arrows,
combo drop-downs, scroll bars, check marks) from the palette. The stylesheet only does what a palette
cannot say: tab shape, group-box title, table header, rounded buttons. Styling a sub-control in the sheet
makes Qt stop drawing that sub-control's built-in indicator, so the sheet deliberately leaves arrows alone.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QPalette

ASSETS = (Path(__file__).resolve().parent / "assets").as_posix()

# One scale, dark to light, plus the accents. Text is off-white rather than pure white: on a near-black
# ground pure white vibrates, and the muted tone has to stay readable next to it.
C = {
    "bg": "#15171c",           # window ground
    "surface": "#1c1f26",      # panels, tables, inputs
    "surface_alt": "#22262f",  # alternating rows, headers
    "raised": "#2a2f3a",       # buttons
    "border": "#333946",
    "border_soft": "#272c35",
    "text": "#e7eaf0",
    "text_dim": "#9aa4b4",
    "text_faint": "#6b7483",
    "accent": "#4c8dff",
    "accent_dim": "#2f5fb0",
    "pos": "#3fbf87",
    "neg": "#f2645a",
    "warn": "#e0912f",
}

# chips in the header: distinct hues, dark enough to carry white text
CHIP = {
    "mega": "#6f5bd0",
    "loss": "#c2453c",
    "phase": "#c07a22",
    "lag": "#7a6455",
    "manual": "#2f6fd0",
    "live": "#2f8f5f",
}

# the chart palette: brighter than the print colours, which disappear on a dark ground
LINE = {
    "g_star": "#5aa2ff",
    "g_norm": "#5aa2ff",
    "g_model": "#8b95a5",
    "r": "#f2645a",
    "pv": "#3fbf87",
    "book": "#5b6675",
    "explicit": "#4c8dff",
    "fade": "#3fbf87",
    "terminal": "#e0912f",
    "price": "#d8524a",
    "zero": "#5a616e",
}


def _palette() -> QPalette:
    p = QPalette()
    p.setColor(QPalette.ColorRole.Window, QColor(C["bg"]))
    p.setColor(QPalette.ColorRole.WindowText, QColor(C["text"]))
    p.setColor(QPalette.ColorRole.Base, QColor(C["surface"]))
    p.setColor(QPalette.ColorRole.AlternateBase, QColor(C["surface_alt"]))
    p.setColor(QPalette.ColorRole.Text, QColor(C["text"]))
    p.setColor(QPalette.ColorRole.BrightText, QColor("#ffffff"))
    p.setColor(QPalette.ColorRole.Button, QColor(C["raised"]))
    p.setColor(QPalette.ColorRole.ButtonText, QColor(C["text"]))
    p.setColor(QPalette.ColorRole.Highlight, QColor(C["accent"]))
    p.setColor(QPalette.ColorRole.HighlightedText, QColor("#0d1014"))
    p.setColor(QPalette.ColorRole.ToolTipBase, QColor(C["surface_alt"]))
    p.setColor(QPalette.ColorRole.ToolTipText, QColor(C["text"]))
    p.setColor(QPalette.ColorRole.PlaceholderText, QColor(C["text_faint"]))
    p.setColor(QPalette.ColorRole.Link, QColor(C["accent"]))
    p.setColor(QPalette.ColorRole.LinkVisited, QColor(C["accent_dim"]))
    for role in (QPalette.ColorRole.Text, QPalette.ColorRole.ButtonText, QPalette.ColorRole.WindowText):
        p.setColor(QPalette.ColorGroup.Disabled, role, QColor(C["text_faint"]))
    p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Base, QColor(C["border_soft"]))
    p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Button, QColor(C["border_soft"]))
    return p


def stylesheet() -> str:
    return f"""
* {{ font-family: "Malgun Gothic", "Segoe UI", sans-serif; }}
QMainWindow, QWidget {{ background: {C["bg"]}; color: {C["text"]}; }}
QMenuBar {{ background: {C["bg"]}; border-bottom: 1px solid {C["border_soft"]}; }}
QMenuBar::item {{ padding: 5px 11px; background: transparent; }}
QMenuBar::item:selected {{ background: {C["surface_alt"]}; border-radius: 4px; }}
QMenu {{ background: {C["surface"]}; border: 1px solid {C["border"]}; padding: 4px; }}
QMenu::item {{ padding: 5px 22px 5px 14px; border-radius: 4px; }}
QMenu::item:selected {{ background: {C["accent_dim"]}; color: {C["text"]}; }}
QStatusBar {{ background: {C["bg"]}; color: {C["text_dim"]}; border-top: 1px solid {C["border_soft"]}; }}
QStatusBar::item {{ border: none; }}

QGroupBox {{ background: {C["surface"]}; border: 1px solid {C["border_soft"]}; border-radius: 8px;
             margin-top: 20px; padding: 12px 10px 10px 10px; }}
QGroupBox::title {{ subcontrol-origin: margin; subcontrol-position: top left; left: 11px; padding: 0 5px;
                    color: {C["text_dim"]}; font-weight: bold; }}

QLineEdit, QAbstractSpinBox, QComboBox {{ background: {C["surface_alt"]}; border: 1px solid {C["border"]};
    border-radius: 6px; padding: 5px 8px; selection-background-color: {C["accent"]}; selection-color: #0d1014; }}
QLineEdit:focus, QAbstractSpinBox:focus, QComboBox:focus {{ border: 1px solid {C["accent"]}; }}
QLineEdit:disabled, QAbstractSpinBox:disabled {{ background: {C["border_soft"]}; color: {C["text_faint"]}; }}
QComboBox QAbstractItemView {{ background: {C["surface"]}; border: 1px solid {C["border"]};
    selection-background-color: {C["accent_dim"]}; outline: none; }}
/* styling a spin box hands its sub-controls to the sheet, so the arrows have to be supplied here or they
   vanish; the assets are two flat triangles in the muted text colour */
QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {{ subcontrol-origin: border; width: 17px;
    background: transparent; border: none; border-left: 1px solid {C["border"]}; }}
QAbstractSpinBox::up-button {{ subcontrol-position: top right; margin: 1px 1px 0 0; }}
QAbstractSpinBox::down-button {{ subcontrol-position: bottom right; margin: 0 1px 1px 0; }}
QAbstractSpinBox::up-button:hover, QAbstractSpinBox::down-button:hover {{ background: {C["raised"]}; }}
QAbstractSpinBox::up-arrow {{ image: url("{ASSETS}/arrow_up.svg"); width: 9px; height: 6px; }}
QAbstractSpinBox::down-arrow {{ image: url("{ASSETS}/arrow_down.svg"); width: 9px; height: 6px; }}
QAbstractSpinBox::up-arrow:disabled, QAbstractSpinBox::up-arrow:off {{ image: url("{ASSETS}/arrow_up_dim.svg"); }}
QAbstractSpinBox::down-arrow:disabled, QAbstractSpinBox::down-arrow:off {{ image: url("{ASSETS}/arrow_down_dim.svg"); }}
QComboBox::drop-down {{ subcontrol-origin: padding; subcontrol-position: center right; width: 18px;
    border: none; border-left: 1px solid {C["border"]}; }}
QComboBox::down-arrow {{ image: url("{ASSETS}/arrow_down.svg"); width: 9px; height: 6px; }}

QPushButton {{ background: {C["raised"]}; border: 1px solid {C["border"]}; border-radius: 6px;
               padding: 6px 14px; color: {C["text"]}; }}
QPushButton:hover {{ background: #333a47; border-color: #3f4757; }}
QPushButton:pressed {{ background: {C["accent_dim"]}; }}
QPushButton:default {{ border: 1px solid {C["accent"]}; }}
QPushButton:disabled {{ background: {C["border_soft"]}; color: {C["text_faint"]}; border-color: {C["border_soft"]}; }}

QTabWidget::pane {{ background: {C["bg"]}; border: 1px solid {C["border_soft"]}; border-radius: 8px; top: -1px; }}
QTabBar::tab {{ background: transparent; color: {C["text_dim"]}; padding: 7px 16px; margin-right: 3px;
                border: 1px solid transparent; border-top-left-radius: 7px; border-top-right-radius: 7px; }}
QTabBar::tab:hover {{ color: {C["text"]}; }}
QTabBar::tab:selected {{ background: {C["surface"]}; color: {C["text"]};
    border: 1px solid {C["border_soft"]}; border-bottom-color: {C["surface"]}; }}

QTableWidget, QTableView {{ background: {C["surface"]}; alternate-background-color: {C["surface_alt"]};
    gridline-color: {C["border_soft"]}; border: 1px solid {C["border_soft"]}; border-radius: 6px;
    selection-background-color: {C["accent_dim"]}; selection-color: {C["text"]}; }}
QTableWidget::item {{ padding: 3px 6px; }}
QHeaderView::section {{ background: {C["surface_alt"]}; color: {C["text_dim"]}; padding: 5px 6px;
    border: none; border-right: 1px solid {C["border_soft"]}; border-bottom: 1px solid {C["border"]}; font-weight: bold; }}
QTableCornerButton::section {{ background: {C["surface_alt"]}; border: none; }}

QListView {{ background: {C["surface"]}; border: 1px solid {C["border_soft"]}; border-radius: 6px;
    outline: none; padding: 2px; }}
QListView::item {{ padding: 3px 6px; border-radius: 4px; }}
QListView::item:hover {{ background: {C["surface_alt"]}; }}
QListView::item:selected {{ background: {C["accent_dim"]}; color: {C["text"]}; }}

QTextBrowser {{ background: {C["surface"]}; border: 1px solid {C["border_soft"]}; border-radius: 6px; padding: 8px; }}
QScrollArea {{ background: {C["bg"]}; border: none; }}
QScrollBar:vertical {{ background: transparent; width: 11px; margin: 0; }}
QScrollBar:horizontal {{ background: transparent; height: 11px; margin: 0; }}
QScrollBar::handle {{ background: {C["border"]}; border-radius: 5px; min-height: 28px; min-width: 28px; }}
QScrollBar::handle:hover {{ background: #414958; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

QSplitter::handle {{ background: {C["border_soft"]}; }}
QToolTip {{ background: {C["surface_alt"]}; color: {C["text"]}; border: 1px solid {C["border"]};
            padding: 5px 7px; border-radius: 5px; }}
QCheckBox {{ spacing: 7px; }}
QCheckBox::indicator {{ width: 14px; height: 14px; border: 1px solid {C["border"]}; border-radius: 3px;
    background: {C["surface_alt"]}; }}
QCheckBox::indicator:hover {{ border-color: {C["accent"]}; }}
QCheckBox::indicator:checked {{ background: {C["accent"]}; border-color: {C["accent"]};
    image: url("{ASSETS}/check.svg"); }}
QCheckBox::indicator:disabled {{ background: {C["border_soft"]}; border-color: {C["border_soft"]}; }}
QCalendarWidget QWidget {{ alternate-background-color: {C["surface_alt"]}; }}
QCalendarWidget QAbstractItemView:enabled {{ background: {C["surface"]}; color: {C["text"]};
    selection-background-color: {C["accent_dim"]}; selection-color: {C["text"]}; }}
QCalendarWidget QAbstractItemView:disabled {{ color: {C["text_faint"]}; }}
QCalendarWidget QToolButton {{ background: transparent; color: {C["text"]}; border: none; padding: 4px 8px; }}
QCalendarWidget QToolButton:hover {{ background: {C["surface_alt"]}; border-radius: 4px; }}
QCalendarWidget QMenu {{ background: {C["surface"]}; }}
QCalendarWidget QSpinBox {{ background: {C["surface_alt"]}; }}
"""


_applied = False


def apply(app) -> None:
    """Fusion + the dark palette + the polish sheet, and the matplotlib defaults to match.

    Idempotent, and called from the window's constructor rather than only from main(), so that the offscreen
    self-test and the screenshots render what the user sees."""
    global _applied
    if _applied:
        return
    _applied = True
    app.setStyle("Fusion")
    app.setPalette(_palette())
    app.setFont(QFont("Malgun Gothic", 9))
    app.setStyleSheet(stylesheet())
    app.setAttribute(Qt.ApplicationAttribute.AA_DontShowIconsInMenus, False)
    apply_mpl()


def apply_mpl() -> None:
    import matplotlib
    matplotlib.rcParams.update({
        "font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": 9,
        "figure.facecolor": C["surface"], "savefig.facecolor": C["surface"],
        "axes.facecolor": C["surface"], "axes.edgecolor": C["border"],
        "axes.labelcolor": C["text_dim"], "axes.titlecolor": C["text"],
        "text.color": C["text"], "xtick.color": C["text_dim"], "ytick.color": C["text_dim"],
        "grid.color": C["border"], "legend.facecolor": C["surface"], "legend.edgecolor": C["border"],
    })
