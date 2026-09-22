# -*- coding: utf-8 -*-
"""Small shared pieces: number formatting, read-only tables that size to their content, coloured chips."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QRect, Qt
from PySide6.QtWidgets import (QAbstractItemView, QHeaderView, QLabel, QPushButton, QSizePolicy, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)

from app.theme import C

DASH = "—"


def isnum(v) -> bool:
    try:
        return v is not None and bool(np.isfinite(float(v)))
    except (TypeError, ValueError):
        return False


def pct(v, d: int = 1, sign: bool = False) -> str:
    if not isnum(v):
        return DASH
    return f"{float(v) * 100:+.{d}f}%" if sign else f"{float(v) * 100:.{d}f}%"


def bn(v, d: int = 0) -> str:
    """dollars -> 십억 달러. A value that rounds to zero prints as 0, never -0."""
    if not isnum(v):
        return DASH
    s = f"{float(v) / 1e9:,.{d}f}"
    if s.lstrip("-").strip("0.,") == "":
        s = s.lstrip("-")
    return f"{s}십억"


def num(v, d: int = 2) -> str:
    if not isnum(v):
        return DASH
    return f"{float(v):,.{d}f}"


def usd(v, d: int = 2) -> str:
    if not isnum(v):
        return DASH
    return f"${float(v):,.{d}f}"


class AutoTable(QTableWidget):
    """Read-only table whose height follows its rows, so several tables stack inside one scroll area without inner
    scrollbars.

    Two shapes. With stretch_last the last column is a column of prose that takes whatever width is left, and the
    columns before it are capped at NOTE_COL_MAX of the viewport so one long value cannot squeeze the prose into
    an ellipsis. Without it the table is numeric and hugs its content instead of stretching across the window."""

    NOTE_COL_MAX = 0.40

    def __init__(self, headers: list[str], stretch_last: bool = True):
        super().__init__(0, len(headers))
        self.stretch_last = stretch_last
        self._fitting = False
        self.verticalHeader().setVisible(False)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setAlternatingRowColors(True)
        self.setWordWrap(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff if stretch_last else Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.set_headers(headers)

    def set_headers(self, headers: list[str]) -> None:
        self.setColumnCount(len(headers))
        self.setHorizontalHeaderLabels(headers)
        hh = self.horizontalHeader()
        if self.stretch_last and len(headers) > 1:
            # every width, the last one included, is assigned in fit(): a column left on Stretch has no settled
            # width when the row heights are measured, and a wrapped note then gets an ellipsis instead of a line
            hh.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
            hh.setStretchLastSection(False)
        else:
            hh.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
            hh.setStretchLastSection(False)

    def fit(self) -> None:
        """Column widths first, then row heights: a wrapped cell's height depends on how wide its column ended up,
        so measuring the rows before the columns are settled is what leaves text elided."""
        if self._fitting:
            return
        self._fitting = True
        try:
            hh = self.horizontalHeader()
            n = self.columnCount()
            avail = self.viewport().width()
            if self.stretch_last and n > 1 and avail > 260:
                cap = int(avail * self.NOTE_COL_MAX)
                used = 0
                for i in range(n - 1):
                    w = max(60, min(self.sizeHintForColumn(i) + 14, cap))
                    self.setColumnWidth(i, w)
                    used += w
                self.setColumnWidth(n - 1, max(120, avail - used))
                self.setMaximumWidth(16777215)
            elif self.stretch_last and n > 1:
                self.resizeColumnsToContents()      # not laid out yet; resizeEvent will do it properly
                self.setMaximumWidth(16777215)
            else:
                self.resizeColumnsToContents()
                self.setMaximumWidth(hh.length() + 2 * self.frameWidth() + 2)
            self._wrap_rows()
            h = hh.height() + 2 * self.frameWidth() + 2
            for i in range(self.rowCount()):
                h += self.rowHeight(i)
            self.setFixedHeight(max(h, 40))
        finally:
            self._fitting = False

    def _wrap_rows(self) -> None:
        """Row heights measured against the settled column widths.

        resizeRowsToContents asks the delegate for a size hint that does not know how wide the column ended up,
        so a note that needs three lines is given two and shown with an ellipsis. Measuring the wrapped text
        directly is the only way to get it right."""
        fm = self.fontMetrics()
        # the width the view actually lays text into is the column less the stylesheet's cell padding, the
        # style's item margins and the grid line: measured at 20-22px, taken at 24 because under-measuring
        # costs an elided line while over-measuring costs a few pixels of air
        pad_w, pad_h = 24, 10
        for i in range(self.rowCount()):
            need = fm.height()
            for j in range(self.columnCount()):
                it = self.item(i, j)
                if it is None or not it.text():
                    continue
                w = max(40, self.columnWidth(j) - pad_w)
                box = fm.boundingRect(QRect(0, 0, w, 10_000), int(Qt.TextFlag.TextWordWrap), it.text())
                need = max(need, box.height())
            self.setRowHeight(i, need + pad_h)

    def resizeEvent(self, ev) -> None:  # noqa: N802
        super().resizeEvent(ev)
        self.fit()


def make_table(headers: list[str], stretch_last: bool = True) -> AutoTable:
    return AutoTable(headers, stretch_last)


def fill_table(t: AutoTable, rows: list[list], right_cols: set[int] | None = None, max_rows_visible: int = 400,
               tips: list[str] | None = None) -> None:
    """Replace the rows and right-align the numeric columns; the table then sizes itself.

    `tips` is one plain-language explanation per row, shown on hover anywhere in that row. A table of terms
    the reader has not met before needs somewhere to put the definition that is not the table itself."""
    right_cols = right_cols or set()
    rows = rows[:max_rows_visible]
    t.setRowCount(0)
    t.setRowCount(len(rows))
    for i, r in enumerate(rows):
        for j, v in enumerate(r):
            it = QTableWidgetItem("" if v is None else str(v))
            if j in right_cols:
                it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            if tips and i < len(tips) and tips[i]:
                it.setToolTip(tips[i])
            t.setItem(i, j, it)
    t.fit()


class SummaryBox(QLabel):
    """The answer before the evidence: a few sentences at the top of a tab, set larger than the tables under it."""

    def __init__(self) -> None:
        super().__init__()
        self.setWordWrap(True)
        self.setTextFormat(Qt.TextFormat.RichText)
        self.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.setStyleSheet(f"background:{C['surface']}; border:1px solid {C['border_soft']}; border-left:3px solid {C['accent']};"
                           f"border-radius:7px; padding:13px 15px; font-size:13px; color:{C['text']};")


class Collapsible(QWidget):
    """A section folded away by default. The tabs carry both what a decision needs and what it needs only when
    doubted; showing them at one weight is what made the screen unreadable, so the second kind lives in here."""

    def __init__(self, hint: str = "") -> None:
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        self.hint = hint
        self.btn = QPushButton()
        self.btn.setCheckable(True)
        self.btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn.setStyleSheet(f"QPushButton {{ background:transparent; border:none; color:{C['accent']};"
                               f"text-align:left; padding:6px 2px; font-weight:bold; }}"
                               f"QPushButton:hover {{ color:{C['text']}; }}")
        self.body = QWidget()
        self.body_lay = QVBoxLayout(self.body)
        self.body_lay.setContentsMargins(0, 0, 0, 0)
        self.body_lay.setSpacing(8)
        self.body.setVisible(False)
        lay.addWidget(self.btn)
        lay.addWidget(self.body)
        self.btn.toggled.connect(self._toggle)
        self._label()

    def _label(self) -> None:
        self.btn.setText("접기" if self.btn.isChecked() else (f"자세히 보기  —  {self.hint}" if self.hint else "자세히 보기"))

    def _toggle(self, on: bool) -> None:
        self.body.setVisible(on)
        self._label()
        if on:
            for t in self.body.findChildren(AutoTable):
                t.fit()

    def add(self, w) -> None:
        self.body_lay.addWidget(w)


def chip(text: str, bg: str) -> QLabel:
    lab = QLabel(text)
    lab.setStyleSheet(f"background:{bg}; color:#f5f7fa; border-radius:9px; padding:3px 10px; font-weight:bold;")
    return lab


def note(text: str = "", bold: bool = False) -> QLabel:
    """A wrapped, selectable line of reading. Bold ones are statements the eye should land on; the rest are
    the quieter tone, so a screen of tables does not read as one flat wall."""
    lab = QLabel(text)
    lab.setWordWrap(True)
    lab.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    lab.setStyleSheet(f"color:{C['text']}; font-weight:bold;" if bold else f"color:{C['text_dim']};")
    return lab


def header_label(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setStyleSheet(f"color:{C['text']}; font-weight:bold; font-size:13px; margin-top:10px;")
    return lab
