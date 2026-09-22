# -*- coding: utf-8 -*-
"""Tab 5, 수기 입력: a firm outside the universe (Korean listings, financials, anything without SEC facts) entered
by hand and handed to the same inversion and scenario code. The caveats are printed with the result."""
from __future__ import annotations

import pandas as pd
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QLineEdit, QPushButton, QScrollArea,
                               QVBoxLayout, QWidget)

from sfv.calc import manual_row
from sfv.store import SIZE_LABELS, Store
from app.widgets import note, pct

FF12_KO = {"BusEq": "BusEq 정보기술·장비", "Chems": "Chems 화학", "Durbl": "Durbl 내구소비재", "Enrgy": "Enrgy 에너지", "Hlth": "Hlth 헬스케어",
           "Manuf": "Manuf 제조", "NoDur": "NoDur 비내구소비재", "Other": "Other 기타(건설·운송·서비스)", "Shops": "Shops 유통", "Telcm": "Telcm 통신",
           "Utils": "Utils 유틸리티", "Money": "Money 금융"}


def _spin(lo: float, hi: float, d: int, val: float, suffix: str = "") -> QDoubleSpinBox:
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(d)
    s.setValue(val)
    if suffix:
        s.setSuffix(suffix)
    return s


class ManualTab(QWidget):
    def __init__(self, store: Store, on_row):
        super().__init__()
        self.store = store
        self.on_row = on_row            # callback: the window takes the row as the current firm
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
        lay.addWidget(note("유니버스 밖 기업(한국 기업, 금융업, SEC 자료 없는 기업)을 직접 넣는다. 계산하면 이 기업이 현재 종목이 되어 "
                           "전제·목표주가 탭이 그대로 쓰인다. 통화는 무엇이든 되지만 한 통화로 통일한다."))
        box = QGroupBox("기업")
        f = QFormLayout(box)
        self.ed_name = QLineEdit("수기 입력")
        self.cb_ind = QComboBox()
        for k in store.industry["ff12"].tolist():
            self.cb_ind.addItem(FF12_KO.get(k, k), k)
        self.sp_price = _spin(0.0001, 1e9, 2, 100.0)
        self.sp_shares = _spin(0.0001, 1e6, 3, 100.0, " 백만 주")
        self.sp_equity = _spin(-1e6, 1e6, 3, 10.0, " 십억")
        self.sp_e = _spin(-1e6, 1e6, 3, 1.0, " 십억")
        self.sp_rev = _spin(0.0001, 1e6, 3, 10.0, " 십억")
        self.sp_kint = _spin(0.0, 1e6, 3, 0.0, " 십억")
        self.sp_kint.setToolTip("연구비·판관비를 자본화한 무형자산 잔액. 모르면 0 (조정 장부가 과소 → g* 상향 편향)")
        f.addRow("이름", self.ed_name)
        f.addRow("산업 (FF12)", self.cb_ind)
        f.addRow("주가", self.sp_price)
        f.addRow("주식수", self.sp_shares)
        f.addRow("자기자본 (지배주주)", self.sp_equity)
        f.addRow("TTM 순이익 (조정 가능하면 조정)", self.sp_e)
        f.addRow("TTM 매출", self.sp_rev)
        f.addRow("무형자산 잔액 (자본화, 선택)", self.sp_kint)
        lay.addWidget(box)

        box2 = QGroupBox("모형 입력 (체크를 풀면 직접 지정)")
        f2 = QFormLayout(box2)
        self.sp_beta = _spin(0.0, 3.0, 2, 1.0)
        self.chk_omega = QCheckBox("규모 그룹 중앙값")
        self.chk_omega.setChecked(True)
        self.sp_omega = _spin(0.0, 0.995, 3, 0.5)
        self.chk_xinf = QCheckBox("규모 그룹 중앙값")
        self.chk_xinf.setChecked(True)
        self.sp_xinf = _spin(-0.10, 0.10, 3, 0.0)
        self.chk_payout = QCheckBox("산업 중앙값")
        self.chk_payout.setChecked(True)
        self.sp_payout = _spin(0.0, 1.0, 2, 0.3)
        self.chk_rs = QCheckBox("산업 중앙값 (미국 패널)")
        self.chk_rs.setChecked(True)
        self.sp_rs = _spin(-1.0, 1.0, 3, 0.08)
        self.sp_rf = _spin(0.0, 30.0, 2, float(store.macro["rf"]) * 100, " %")
        self.sp_erp = _spin(0.0, 20.0, 2, float(store.macro["erp"]) * 100, " %")
        for chk, sp in ((self.chk_omega, self.sp_omega), (self.chk_xinf, self.sp_xinf), (self.chk_payout, self.sp_payout), (self.chk_rs, self.sp_rs)):
            sp.setEnabled(False)
            chk.toggled.connect(lambda on, s=sp: s.setEnabled(not on))
        f2.addRow("β", self.sp_beta)
        f2.addRow("지속성 ω", self._pair(self.chk_omega, self.sp_omega))
        f2.addRow("장기 초과 ROE x∞", self._pair(self.chk_xinf, self.sp_xinf))
        f2.addRow("환원율", self._pair(self.chk_payout, self.sp_payout))
        f2.addRow("산업 기준 ROE", self._pair(self.chk_rs, self.sp_rs))
        f2.addRow("국채 금리", self.sp_rf)
        f2.addRow("ERP", self.sp_erp)
        lay.addWidget(box2)

        self.btn = QPushButton("계산하고 현재 종목으로")
        self.btn.clicked.connect(self.compute)
        lay.addWidget(self.btn)
        self.lbl = note()
        lay.addWidget(self.lbl)
        lay.addWidget(note("주의: ① 무형자산 잔액 0이면 조정 장부가가 과소해 요구 성장이 위로 치우친다 (미국 패널은 10년치 연구비·판관비를 자본화). "
                           "② ω·산업 기준 ROE·기저율은 미국 2010~2026 패널값이다. ③ ROE는 입력한 현재 자본 기준이다 (패널은 1년 전 자본). "
                           "④ 자본잠식(조정 장부가 ≤ 0)은 평가 불가."))
        lay.addStretch(1)

    @staticmethod
    def _pair(chk, sp) -> QWidget:
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(chk)
        h.addWidget(sp)
        return w

    def compute(self) -> pd.DataFrame | None:
        try:
            row = manual_row(self.store, self.ed_name.text().strip() or "수기 입력", self.cb_ind.currentData(),
                             float(self.sp_price.value()), float(self.sp_shares.value()) * 1e6, float(self.sp_equity.value()) * 1e9,
                             float(self.sp_e.value()) * 1e9, float(self.sp_rev.value()) * 1e9, k_int=float(self.sp_kint.value()) * 1e9,
                             beta=float(self.sp_beta.value()),
                             omega=None if self.chk_omega.isChecked() else float(self.sp_omega.value()),
                             xinf=None if self.chk_xinf.isChecked() else float(self.sp_xinf.value()),
                             payout=None if self.chk_payout.isChecked() else float(self.sp_payout.value()),
                             roe_star=None if self.chk_rs.isChecked() else float(self.sp_rs.value()),
                             rf=float(self.sp_rf.value()) / 100, erp=float(self.sp_erp.value()) / 100)
        except ValueError as e:
            self.lbl.setText(f"계산 불가: {e}")
            return None
        x = row.iloc[0]
        self.lbl.setText(f"규모 그룹 {SIZE_LABELS.get(x['size_group'], x['size_group'])}: ω {x['omega']:.3f}, x∞ {x['xinf']:+.3f}, 환원율 {x['payout']:.2f}, "
                         f"산업 기준 ROE {x['roe_star_ind']:+.3f}, 할인율 {pct(x['r'])}. 조정 장부가 {x['b_adj']/1e9:,.2f}십억, ROE {pct(x['roe_adj'])}, "
                         f"초과 ROE {float(x['x0']):+.3f}. 전제 탭으로 이동해 읽는다.")
        self.on_row(row)
        return row
