# -*- coding: utf-8 -*-
"""Tab 4, 매수일 대조: the premise on a buy date against what the firm delivered since, and how the premise moved."""
from __future__ import annotations

import pandas as pd
from PySide6.QtCore import QDate
from PySide6.QtWidgets import QDateEdit, QHBoxLayout, QLabel, QPushButton, QScrollArea, QVBoxLayout, QWidget

from sfv.calc import scorecard
from sfv.store import Store
from app.widgets import DASH, bn, fill_table, isnum, make_table, note, num, pct


class ScorecardTab(QWidget):
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
        top.addWidget(QLabel("매수일"))
        self.date = QDateEdit()
        self.date.setCalendarPopup(True)
        self.date.setDisplayFormat("yyyy-MM-dd")
        m = store.month
        self.date.setDate(QDate(m.year - 1, m.month, min(m.day, 28)))
        top.addWidget(self.date)
        self.btn = QPushButton("대조")
        self.btn.clicked.connect(self.refresh)
        top.addWidget(self.btn)
        top.addStretch(1)
        lay.addLayout(top)
        self.lbl_head = note(bold=True)
        lay.addWidget(self.lbl_head)
        self.tbl = make_table(["", "당시", "지금"], stretch_last=False)
        lay.addWidget(self.tbl)
        self.tbl_real = make_table(["실현", "누적", "연율", "요구 누적 (연)", "판정"], stretch_last=False)
        lay.addWidget(self.tbl_real)
        self.lbl_read = note()
        lay.addWidget(self.lbl_read)
        lay.addStretch(1)

    def set_row(self, row: pd.DataFrame, q: dict | None = None) -> None:
        self.cik = int(row.iloc[0]["cik"])
        self.refresh()

    def refresh(self) -> None:
        if self.cik is None:
            return
        if self.cik < 0:
            self.lbl_head.setText("수기 입력 기업은 과거 전제가 없음")
            fill_table(self.tbl, [])
            fill_table(self.tbl_real, [])
            self.lbl_read.setText("")
            return
        since = self.date.date().toString("yyyy-MM-dd")
        sc = scorecard(self.store, self.cik, since)
        if not sc["ok"]:
            first = sc["first_month"].date() if sc["first_month"] is not None else "n/a"
            self.lbl_head.setText(f"{since} 이전 평가월이 없음 (첫 평가월 {first})")
            fill_table(self.tbl, [])
            fill_table(self.tbl_real, [])
            self.lbl_read.setText("")
            return
        self.lbl_head.setText(f"{sc['month0'].date()} 시점의 전제 vs 그 뒤 실현 (재무 {sc['f0'].date()} → {sc['f1'].date()}, {sc['yrs']:.2f}년)")
        rows = [["시총", bn(sc["mktcap0"]), f"{bn(sc['mktcap1'])} ({pct(sc['dp'], 0, True)})"],
                ["조정 P/E", f"{num(sc['pe0'], 1)}배", f"{num(sc['pe1'], 1)}배"],
                ["할인율", pct(sc["r0"]), pct(sc["r1"])],
                ["요구 성장 g*", f"{pct(sc['g0'])}/년" if isnum(sc["g0"]) else "해당 없음",
                 (f"{pct(sc['g1'])}/년 ({sc['dg']*100:+.1f}%p)" if isnum(sc["g1"]) and isnum(sc["dg"]) else f"{pct(sc['g1'])}/년" if isnum(sc["g1"]) else "해당 없음")],
                ["3년 달성 확률", DASH, pct(sc["p_achieve"], 0)]]
        fill_table(self.tbl, rows, right_cols={1, 2})
        if not sc["measurable"]:
            fill_table(self.tbl_real, [["실현치를 잴 만큼 시간이 지나지 않았거나(0.5년 미만) 재무·요구 성장이 없음", "", "", "", ""]])
        else:
            rr = []
            for w in sc["realized"]:
                if w["ok"]:
                    rr.append([w["name"], pct(w["cum"], 1, True), pct(w["ann"], 1, True), f"{pct(sc['req_cum'], 1, True)} ({pct(sc['g0'])})",
                               "달성" if w["achieved"] else "미달"])
                else:
                    rr.append([w["name"], "계산 불가 (음수 또는 결측)", "", "", ""])
            fill_table(self.tbl_real, rr, right_cols={1, 2, 3})
        if sc["direction"] is not None:
            parts = []
            if isnum(sc["pe0"]) and isnum(sc["pe1"]):
                parts.append(f"이익이 가격보다 빨리 자라 조정 P/E가 {sc['pe0']:.1f}→{sc['pe1']:.1f}배로 낮아졌고" if sc["pe1"] < sc["pe0"]
                             else f"가격이 이익보다 빨리 올라 조정 P/E가 {sc['pe0']:.1f}→{sc['pe1']:.1f}배로 높아졌고")
            parts.append(f"할인율은 {pct(sc['r0'])}→{pct(sc['r1'])}")
            self.lbl_read.setText(f"읽기: 요구 성장이 {sc['direction']}. {', '.join(parts)}. 전제가 가벼워졌는지는 위 표, "
                                  f"당시 요구 속도를 채웠는지는 달성·미달로 본다. 실적이 요구를 채웠는데 요구 성장이 그대로라면, "
                                  f"그만큼 가격이 같이 올라 전제가 다시 무거워진 것이다.")
        else:
            self.lbl_read.setText("")
