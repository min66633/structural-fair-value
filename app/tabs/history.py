# -*- coding: utf-8 -*-
"""Tab 3, 추이: how the price's premise moved, quarter by quarter, against the fundamentals of the time."""
from __future__ import annotations

import pandas as pd
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QScrollArea, QVBoxLayout, QWidget

from sfv.calc import history, history_series
from sfv.store import Store
from app.charts import MplCanvas, draw_history
from app.widgets import fill_table, make_table, note, num


class HistoryTab(QWidget):
    def __init__(self, store: Store):
        super().__init__()
        self.store = store
        self.cik: int | None = None
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll)
        body = QWidget()
        scroll.setWidget(body)
        lay = QVBoxLayout(body)
        lay.setContentsMargins(14, 16, 14, 14)
        lay.setSpacing(8)
        top = QHBoxLayout()
        top.addWidget(QLabel("분기말 수"))
        self.cb_n = QComboBox()
        self.cb_n.addItems(["8", "20", "전체"])
        self.cb_n.currentIndexChanged.connect(self.refresh)
        top.addWidget(self.cb_n)
        top.addStretch(1)
        lay.addLayout(top)
        self.canvas = MplCanvas(6.0, 4.2)
        lay.addWidget(self.canvas)
        self.lbl = note("실적이 나오고 가격이 움직이면 여기가 바뀐다. 요구 성장이 내려오면 이익이 가격을 따라잡은 것이고, "
                        "올라가면 가격이 이익보다 앞서 간 것이다.")
        lay.addWidget(self.lbl)
        self.tbl = make_table(["월", "재무 기준", "시총 $bn", "요구 성장 g* %", "전망 g_model %", "직전 1년 매출성장 %", "할인율 %"], stretch_last=False)
        lay.addWidget(self.tbl)
        lay.addStretch(1)

    def set_row(self, row: pd.DataFrame, q: dict | None = None) -> None:
        self.cik = int(row.iloc[0]["cik"])
        self.refresh()

    def refresh(self) -> None:
        if self.cik is None:
            return
        if self.cik < 0:
            draw_history(self.canvas, None)
            fill_table(self.tbl, [["수기 입력 기업은 추이가 없음", "", "", "", "", "", ""]])
            return
        n = {0: 8, 1: 20, 2: 0}[self.cb_n.currentIndex()]
        h = history(self.store, self.cik, n)
        draw_history(self.canvas, history_series(self.store, self.cik))
        if h is None:
            fill_table(self.tbl, [["추이 없음", "", "", "", "", "", ""]])
            return
        rows = [[r[0], r[1], num(r[2], 0), num(r[3], 1), num(r[4], 1), num(r[5], 1), num(r[6], 1)] for r in h.itertuples(index=False)]
        fill_table(self.tbl, rows, right_cols={2, 3, 4, 5, 6}, max_rows_visible=60)
