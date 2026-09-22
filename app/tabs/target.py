# -*- coding: utf-8 -*-
"""Tab 2, 목표주가: the forward direction. Your revenue-growth and margin path, discount rate, persistence and
payout -> value -> per-share target, the return the price offers on that path, the base rate of that growth.
For loss makers: fix the margin path and solve the required growth, or fix the growth path and solve the margin."""
from __future__ import annotations

import numpy as np
import pandas as pd
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QLineEdit, QPushButton, QScrollArea,
                               QSpinBox, QVBoxLayout, QWidget)

from sfv.calc import base_rate_lines, scenario, solve_required
from sfv.rim import OMEGA_MAX
from sfv.store import Store
from app.charts import MplCanvas, draw_composition
from app.widgets import bn, fill_table, header_label, isnum, make_table, note, pct, usd


def parse_pct_list(s: str) -> list[float]:
    """'40,30,20' -> [0.40, 0.30, 0.20]; raises ValueError on junk."""
    return [float(v) / 100.0 for v in s.replace(" ", "").split(",") if v != ""]


class TargetTab(QWidget):
    def __init__(self, store: Store):
        super().__init__()
        self.store = store
        self.row: pd.DataFrame | None = None
        self.q: dict | None = None
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

        box = QGroupBox("내 전제 (비운 칸은 모형값)")
        form = QFormLayout(box)
        self.ed_growth = QLineEdit()
        self.ed_growth.setPlaceholderText("연도별 매출 성장률 %, 쉼표 구분. 예 40,30,20,15,12")
        self.ed_margin = QLineEdit()
        self.ed_margin.setPlaceholderText("연도별 조정 순이익률 %, 쉼표 구분. 비우면 현재 이익률, 모자라면 마지막 값 유지")
        self.sp_years = QSpinBox()
        self.sp_years.setRange(1, 20)
        self.sp_years.setValue(5)
        self.sp_years.setToolTip("경로가 이 연수보다 짧으면 마지막 값으로 연장. 풀기에서는 탐색 기간")
        self.sp_r = QDoubleSpinBox()
        self.sp_r.setRange(0.0, 40.0)
        self.sp_r.setDecimals(2)
        self.sp_r.setSuffix(" %")
        self.sp_omega = QDoubleSpinBox()
        self.sp_omega.setRange(0.0, 0.995)
        self.sp_omega.setDecimals(3)
        self.sp_omega.setSingleStep(0.01)
        self.sp_omega.setToolTip("명시 경로 이후 초과 ROE의 연간 잔존율. 모형 기본값은 규모·연령·산업 특성의 기저율(추정치는 0.90에서 자름). "
                                 "여기에 직접 넣은 값은 0.995까지 그대로 적용된다 — 전제 탭의 ω*를 그대로 넣어 볼 수 있다")
        self.sp_payout = QDoubleSpinBox()
        self.sp_payout.setRange(0.0, 1.0)
        self.sp_payout.setDecimals(2)
        self.sp_payout.setSingleStep(0.05)
        self.cb_e0 = QComboBox()
        self.cb_e0.addItems(["후행 조정이익", "정규화 이익 (5년 평균 이익률)", "직접 입력"])
        self.sp_e0 = QDoubleSpinBox()
        self.sp_e0.setRange(-1e6, 1e6)
        self.sp_e0.setDecimals(2)
        self.sp_e0.setSuffix(" 십억 달러")
        self.sp_e0.setEnabled(False)
        self.cb_e0.currentIndexChanged.connect(lambda i: self.sp_e0.setEnabled(i == 2))
        form.addRow("매출 성장 경로", self.ed_growth)
        form.addRow("조정 이익률 경로", self.ed_margin)
        form.addRow("연수", self.sp_years)
        form.addRow("할인율", self.sp_r)
        form.addRow("지속성 ω (경로 이후)", self.sp_omega)
        form.addRow("환원율 (배당+자사주 / 이익)", self.sp_payout)
        e0row = QHBoxLayout()
        e0row.addWidget(self.cb_e0)
        e0row.addWidget(self.sp_e0)
        form.addRow("출발 이익", e0row)
        lay.addWidget(box)

        btns = QHBoxLayout()
        self.btn_calc = QPushButton("목표주가 계산")
        self.btn_calc.setDefault(True)
        self.btn_solve_g = QPushButton("요구 성장 풀기 (이익률 경로 고정)")
        self.btn_solve_m = QPushButton("요구 이익률 풀기 (매출 경로 고정)")
        self.btn_reset = QPushButton("모형값으로 되돌리기")
        for b in (self.btn_calc, self.btn_solve_g, self.btn_solve_m, self.btn_reset):
            btns.addWidget(b)
        btns.addStretch(1)
        lay.addLayout(btns)
        self.btn_calc.clicked.connect(self.compute)
        self.btn_solve_g.clicked.connect(lambda: self.solve("growth"))
        self.btn_solve_m.clicked.connect(lambda: self.solve("margin"))
        self.btn_reset.clicked.connect(self.reset_defaults)

        self.lbl_hint = note()
        lay.addWidget(self.lbl_hint)
        self.lbl_head = note(bold=True)
        lay.addWidget(self.lbl_head)
        self.lbl_summary = note()
        lay.addWidget(self.lbl_summary)
        self.tbl_path = make_table(["연", "매출 (십억)", "성장", "이익률", "조정이익 (십억)", "ROE"], stretch_last=False)
        lay.addWidget(self.tbl_path)
        self.canvas = MplCanvas(6.0, 3.0)
        lay.addWidget(self.canvas)
        self.lbl_base = note()
        lay.addWidget(self.lbl_base)
        lay.addWidget(header_label("풀기 결과 (적자·전환 기업)"))
        self.lbl_solve = note()
        lay.addWidget(self.lbl_solve)
        lay.addStretch(1)
        self.last: dict | None = None

    # ------------------------------------------------------------------ state
    def set_row(self, row: pd.DataFrame, q: dict) -> None:
        self.row, self.q, self.last = row, q, None
        self.reset_defaults()
        self.lbl_head.setText("")
        self.lbl_summary.setText("")
        self.lbl_base.setText("")
        self.lbl_solve.setText("")
        fill_table(self.tbl_path, [])
        self.canvas.clear()
        x = row.iloc[0]
        e0 = float(x["ttm_e_adj"]) if isnum(x["ttm_e_adj"]) else np.nan
        loss = not (e0 > 0)
        self.cb_e0.model().item(1).setEnabled(bool(isnum(q.get("e_norm")) and q["e_norm"] > 0))
        self.lbl_hint.setText(
            (f"적자 기업 (후행 조정이익 {bn(e0, 1)}): 성장 하나로는 역산이 안 된다. 이익률 경로를 넣고 '요구 성장 풀기', "
             f"또는 매출 경로를 넣고 '요구 이익률 풀기'. 시나리오 계산도 된다." if loss else
             f"출발점: 매출 {bn(x['rev0'])}, 조정이익 {bn(e0, 1)} (이익률 {pct(e0 / float(x['rev0']) if isnum(x['rev0']) and float(x['rev0']) > 0 else np.nan)}), "
             f"주가 {usd(x['price_ps'])}. 경로 뒤에는 모형의 감쇠(ω)와 터미널이 붙으니 가치 구성에서 그 비중을 본다."))

    def reset_defaults(self) -> None:
        if self.row is None:
            return
        x = self.row.iloc[0]
        self._d_r, self._d_omega, self._d_payout = float(x["r"]), min(float(x["omega"]), OMEGA_MAX), float(x["payout"])
        self.sp_r.setValue(self._d_r * 100)
        self.sp_omega.setValue(self._d_omega)
        self.sp_payout.setValue(self._d_payout)
        self.cb_e0.setCurrentIndex(0)
        self.sp_e0.setValue(float(x["ttm_e_adj"]) / 1e9 if isnum(x["ttm_e_adj"]) else 0.0)

    @staticmethod
    def _exact(spin: QDoubleSpinBox, default: float, scale: float) -> float:
        """The model's exact value while the box still shows its rounded default; the box's value once edited.
        Keeps the tab's default numbers identical to the command line's."""
        v = float(spin.value())
        return default if abs(v - round(default * scale, spin.decimals())) < 1e-9 else v / scale

    def _discount(self) -> float:
        return self._exact(self.sp_r, self._d_r, 100.0)

    def _omega(self) -> float:
        return self._exact(self.sp_omega, self._d_omega, 1.0)

    def _payout(self) -> float:
        return self._exact(self.sp_payout, self._d_payout, 1.0)

    def _row_with_payout(self) -> pd.DataFrame:
        d = self.row.copy()
        d["payout"] = self._payout()
        return d

    def _e0_bn(self) -> float | None:
        i = self.cb_e0.currentIndex()
        if i == 1 and self.q is not None and isnum(self.q.get("e_norm")):
            return float(self.q["e_norm"]) / 1e9
        if i == 2:
            return float(self.sp_e0.value())
        return None

    def _paths(self) -> tuple[list[float], list[float] | None]:
        g = parse_pct_list(self.ed_growth.text())
        years = int(self.sp_years.value())
        if g and years > len(g):
            g = g + [g[-1]] * (years - len(g))
        mg = parse_pct_list(self.ed_margin.text()) or None
        return g, mg

    # ------------------------------------------------------------------ actions
    def compute(self, label: str = "시나리오") -> dict | None:
        if self.row is None:
            return None
        try:
            g, mg = self._paths()
        except ValueError:
            self.lbl_head.setText("경로는 숫자와 쉼표만: 예 40,30,20")
            return None
        if not g:
            self.lbl_head.setText("매출 성장 경로를 넣어 주세요 (예 40,30,20,15,12)")
            return None
        s = scenario(self._row_with_payout(), self.store, g, mg, self._e0_bn(), self._omega(), self._discount())
        self.last = s
        self._render(s, label)
        return s

    def solve(self, mode: str) -> dict | None:
        if self.row is None:
            return None
        try:
            g, mg = self._paths()
        except ValueError:
            self.lbl_solve.setText("경로는 숫자와 쉼표만: 예 30,40")
            return None
        years = int(self.sp_years.value())
        s = solve_required(self._row_with_payout(), self.store, mode, mg=mg, g=g or None, years=years,
                           omega=self._omega(), discount=self._discount())
        self.lbl_solve.setText(self._solve_text(s))
        return s

    def apply_required_growth(self, g: float) -> None:
        """The inversion's required growth as a ten-year revenue path at today's margin: the same earnings path
        the inversion solved, so the target price comes back as the current price."""
        self.ed_growth.setText(",".join([f"{g*100:.4f}"] * 10))
        self.ed_margin.setText("")
        self.sp_years.setValue(10)
        self.cb_e0.setCurrentIndex(0)
        self.reset_defaults()
        self.compute(label=f"전제 탭의 요구 성장 {g*100:.1f}%를 그대로 넣은 경우 (이익률은 현재 수준 유지)")

    def apply_consensus(self, cs: dict) -> None:
        self.ed_growth.setText(",".join(f"{v*100:.1f}" for v in cs["rev"]))
        self.ed_margin.setText("")
        self.sp_years.setValue(2)
        tgt = f", 목표주가 평균 {usd(cs['target'], 0)}" if isnum(cs.get("target")) else ""
        self.compute(label=f"컨센서스 경로 (yfinance, 애널리스트 {cs['n']}명: 매출 성장 올해 {pct(cs['rev'][0], 0, True)} · 내년 {pct(cs['rev'][1], 0, True)}, "
                           f"EPS 올해 {pct(cs['eps'][0], 0, True)} · 내년 {pct(cs['eps'][1], 0, True)}{tgt}). "
                           f"회계연도 성장률을 TTM 출발점에 적용한 근사, 3년째부터 추정 감쇠")

    # ------------------------------------------------------------------ rendering
    def _render(self, s: dict, label: str) -> None:
        if not s["ok"]:
            self.lbl_head.setText(f"{label}: {s['reason']}")
            self.lbl_summary.setText("")
            fill_table(self.tbl_path, [])
            self.canvas.clear()
            self.lbl_base.setText("")
            return
        gp = "/".join(f"{v*100:.0f}" for v in s["g"])
        mp = "/".join(f"{v*100:.0f}" for v in s["mg"])
        self.lbl_head.setText(f"{label} — 매출 성장 {gp}%, 조정 이익률 {mp}% ({s['H']}년), 이후 추정 감쇠 (ω {s['omega']:.2f}, 할인율 {pct(s['r'])})")
        rows = [[w["year"], f"{w['rev']/1e9:,.1f}", pct(w["g"], 0), pct(w["margin"]), f"{w['E']/1e9:,.1f}", pct(w["roe"], 0)] for w in s["rows"]]
        fill_table(self.tbl_path, rows, right_cols={1, 2, 3, 4, 5})
        if not s["value_ok"]:
            self.lbl_summary.setText(f"가치가 0 이하 ({bn(s['V'])}): 이 경로로는 가격을 설명할 수 없음")
            self.canvas.clear()
            self.lbl_base.setText("")
            return
        head = (f"주당 목표가 {usd(s['target_ps'], 0)}  (현재 {usd(s['price_ps'])}, {pct(s['upside'], 0, True)})" if isnum(s["target_ps"])
                else f"가치/가격 {s['V']/s['P']:.2f}배 ({pct(s['upside'], 0, True)})")
        ret = f"   이 경로에서 가격이 주는 수익률 연 {pct(s['ret'])} (모형 할인율 {pct(s['r'])})" if isnum(s["ret"]) else ""
        self.lbl_summary.setText(
            f"{head}{ret}\n가치 {bn(s['V'])} = 장부 {bn(s['B0'])} + 명시구간 {bn(s['pv_exp'])} + 감쇠구간 {bn(s['pv_fade'])} + 터미널 {bn(s['tv'])} "
            f"(터미널 {int(round(s['tv']/s['V']*100))}%)" + ("   ← 출발 이익을 정규화·직접 입력값으로 대체" if s["e0_override"] else ""))
        draw_composition(self.canvas, s)
        self.lbl_base.setText("\n".join(ln.strip() for ln in base_rate_lines(s["base"])) if s["base"] else "")

    @staticmethod
    def _solve_text(s: dict) -> str:
        if "H" not in s:
            return {"need_margin": "요구 성장 풀기에는 이익률 경로가 필요합니다 (예 30,40)",
                    "need_growth": "요구 이익률 풀기에는 매출 성장 경로가 필요합니다 (예 100,60,40)"}.get(s.get("code"), s.get("reason", ""))
        if s["mode"] == "growth":
            mp = "/".join(f"{v*100:.0f}" for v in s["mg"])
            head = f"조정 이익률 {mp}% ({s['H']}년) 경로일 때 가격이 요구하는 연 매출 성장"
            if not s["ok"]:
                return f"{head}\n{s['reason']}"
            lines = [head, f"요구 매출 성장 연 {pct(s['g'])} × {s['H']}년: 매출 {s['rev0']/1e9:,.1f} → {s['rev_H']/1e9:,.1f}십억, "
                           f"{s['H']}년째 조정이익 {s['E_H']/1e9:,.1f}십억"]
            lines += [ln.strip() for ln in base_rate_lines(s["base"])]
            lines.append("읽기: 이익률 경로를 바꾸면 요구 성장도 바뀐다. 적자 기업의 가격은 규모와 수익성을 함께 전제하므로 하나를 정해야 다른 하나가 나온다.")
            return "\n".join(lines)
        gp = "/".join(f"{v*100:.0f}" for v in s["g"])
        head = f"매출 성장 {gp}% ({s['H']}년) 경로일 때 가격이 요구하는 조정 이익률 (전 기간 일정)"
        if not s["ok"]:
            return f"{head}\n{s['reason']}"
        cur = f" (현재 이익률 {pct(s['m0'])})" if isnum(s["m0"]) else ""
        return "\n".join([head, f"요구 조정 이익률 {pct(s['margin'])}: {s['H']}년째 매출 {s['rev_H']/1e9:,.1f}십억, 조정이익 {s['E_H']/1e9:,.1f}십억{cur}",
                          "읽기: 매출 경로를 바꾸면 요구 이익률도 바뀐다. 산업의 정상 이익률과 견줘 그 이익률이 가능한지 자문한다."])
