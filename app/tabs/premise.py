# -*- coding: utf-8 -*-
"""Tab 1, 전제: what the price requires.

Three layers, in the order a decision actually uses them. The summary says the whole thing in sentences. Under
it sit the two figures a judgement rests on: the required growth with its band, and the return the price offers
at each growth view. Everything else — the persistence, the advantage period, the normalised and average-ERP
readings, the discount-rate ingredients, the grid — is a cross-check on those two and is folded away, because
showing nine numbers at one weight is how a reader ends up not knowing where to start."""
from __future__ import annotations

import numpy as np
import pandas as pd
from PySide6.QtWidgets import QScrollArea, QVBoxLayout, QWidget

from sfv.calc import GRID
from sfv.store import Store
from app.phrasing import base_rate_facts, summary_html
from app.widgets import (DASH, Collapsible, SummaryBox, bn, fill_table, header_label, isnum, make_table, note,
                         num, pct)

CORE_HEADERS = ["항목", "값", "읽기"]

TIPS = {
    "g": "가격이 맞으려면 조정 이익이 앞으로 10년간 매년 몇 %씩 자라야 하는가. 이 도구의 중심 숫자다.\n"
         "구간은 연구비·판관비를 자산으로 올리는 회계 가정을 문헌의 여섯 조합으로 바꿨을 때의 범위다.",
    "base": "요구 성장이 비슷한 구간에 있던 과거 기업-분기 중, 이후 3년간 실제로 그 속도 이상으로 매출을 키운 비율.\n"
            "이 회사의 확률이 아니라 비슷한 처지 기업들의 기저율이다. 2014년 6월 이후 표본.",
    "omega": "초과 수익성이 1년에 얼마나 남는지. 0.9면 올해 초과분의 90%가 내년에 남는다는 뜻.\n"
             "ω*는 가격이 맞으려면 얼마여야 하는가이고, 모형 ω는 비슷한 특성의 기업들이 과거에 보인 속도다.",
    "T": "초과 수익성이 어느 날 갑자기 0이 된다고 보면, 그날까지 몇 년이 남아 있어야 지금 가격이 맞는가.",
    "norm": "지금 이익 대신 최근 5년 평균 이익률에 현재 매출을 곱한 '평상시 이익'으로 같은 역산을 한 값.\n"
            "경기 피크나 저점에서는 지금 이익으로 잰 요구 성장이 왜곡되므로 이 값을 함께 본다.",
    "erp": "위험 프리미엄이 2014년 이후 평균(5.2%) 수준이었다면 요구 성장이 얼마가 되는가.\n"
           "할인율이 1%p 움직이면 요구 성장은 2~3%p 움직인다.",
    "p3": "요구 성장·규모·직전 성장으로 맞춘 로짓 모형이 보는, 3년 뒤 실제 성장이 요구를 넘을 확률.",
    "r": "모형이 쓰는 할인율. 10년 국채에 이 종목의 베타를 곱한 위험 프리미엄을 더한 값이며, 평균적 투자자의 요구수익률이지\n"
         "당신의 요구수익률이 아니다. 자기 요구수익률은 수익률 축에서 직접 견준다.",
    "model_omega": "모형이 기업 특성(규모·연령·산업·무형집약도)으로 추정한 지속성과, 이익 중 주주에게 돌려주는 비율,\n"
                   "그리고 산업 중앙값 대비 초과 ROE. 회사가 무리와 다르다고 보면 목표주가 탭에서 바꾼다.",
    "value": "장부가에 미래 초과이익의 현재가치를 더한 모형 가치. 회계 가정에 따라 종목 중앙값 46% 움직이므로\n"
             "점 추정 가치로 판단하지 말고 요구 성장으로 판단한다.",
    "grid": "지속성을 여러 값으로 바꿔 가며 가격을 모형 가치로 나눈 것. 1.00이면 그 지속성에서 가격이 맞는다.",
}


class PremiseTab(QWidget):
    def __init__(self, store: Store):
        super().__init__()
        self.store = store
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

        self.summary = SummaryBox()
        lay.addWidget(self.summary)

        lay.addWidget(header_label("가격이 요구하는 것, 그리고 그게 흔한 일인지"))
        self.tbl_core = make_table(CORE_HEADERS)
        lay.addWidget(self.tbl_core)

        lay.addWidget(header_label("내 성장 전망별로 이 가격이 주는 연 수익률"))
        self.tbl_axis = make_table(["성장 전망"], stretch_last=False)
        lay.addWidget(self.tbl_axis)
        self.lbl_axis = note()
        lay.addWidget(self.lbl_axis)

        self.more = Collapsible("모형 가치, 지속성, 유지 기간, 정규화 기준, 할인율 구성, 기저율 상세")
        self.tbl_more = make_table(CORE_HEADERS)
        self.more.add(self.tbl_more)
        self.lbl_base_head = note("요구 성장 구간의 3년 달성 기저율 (2014-06 이후)", bold=True)
        self.more.add(self.lbl_base_head)
        self.tbl_base = make_table(["집단", "기업-분기", "3년 달성 비율", "실현 3년 연환산 중앙값"], stretch_last=False)
        self.more.add(self.tbl_base)
        self.lbl_grid_head = note("지속성별 가격/가치 (다른 입력은 모형값 고정; 1.00이면 가격이 맞음)", bold=True)
        self.more.add(self.lbl_grid_head)
        self.tbl_grid = make_table(["지속성 ω"], stretch_last=False)
        self.more.add(self.tbl_grid)
        lay.addWidget(self.more)

        self.lbl_read = note()
        lay.addWidget(self.lbl_read)
        lay.addStretch(1)

    # ------------------------------------------------------------------ fill
    def set_row(self, row: pd.DataFrame, q: dict) -> None:
        self.summary.setText(summary_html(self.store, row, q))
        f = base_rate_facts(self.store, row, q)
        self._core(row, q, f)
        self._axis(q)
        self._more(row, q)
        self._base_table(f)
        self._grid(q)
        self.lbl_read.setText("이 도구는 예측하지 않는다. 가격이 무엇을 전제하는지, 내 전제로는 얼마인지를 나란히 놓을 뿐이다. "
                              "각 줄에 마우스를 올리면 그 항목이 무슨 뜻인지 나온다.")

    def _core(self, row: pd.DataFrame, q: dict, f: dict) -> None:
        g, b = q["g_star"], q["band"]
        gtxt = f"{pct(g)}/년" if isnum(g) else "해당 없음 (적자이거나 +200%/년으로도 부족)"
        if b is not None and isnum(b["g_lo"]):
            gtxt += f"    구간 {pct(b['g_lo'])} ~ {pct(b['g_hi'])}  (폭 {b['width_pp']:.1f}%p)"
        read_g = ("구간이 좁으니 점으로 읽어도 된다. 자기 기대가 이보다 높으면 매수 근거, 낮으면 매도 근거다."
                  if b is not None and b["width_pp"] <= 5.0 else
                  "구간이 넓다. 점이 아니라 구간으로 읽고, 자기 기대가 구간 안이면 모형의 정밀도 밖이니 판단을 보류한다."
                  if b is not None else "회계 가정 구간을 계산하지 못했다.")
        rows = [["가격이 요구하는 10년 이익성장", gtxt, read_g]]
        tips = [TIPS["g"]]

        if f["overall"] is not None and isnum(f["overall"]["rate"]):
            parts = [f"전체 {pct(f['overall']['rate'], 0)}"]
            if f["size"] is not None:
                parts.append(f"{f['size']['group']} {pct(f['size']['rate'], 0)}")
            if f["past"] is not None:
                parts.append(f"직전 1년 {f['past']['group']} {pct(f['past']['rate'], 0)}")
            if isnum(f["p_achieve"]):
                parts.append(f"이 회사 조건부 확률 {pct(f['p_achieve'], 0)}")
            rows.append(["그 속도를 3년간 실제로 낸 비율", "   ·   ".join(parts),
                         "한 자릿수라면 그 전제는 예외를 주장하는 것이다. 회사가 무리와 다르다고 볼 근거가 판단의 핵심이 된다."])
            tips.append(TIPS["base"])
        else:
            rows.append(["그 속도를 3년간 실제로 낸 비율", DASH, "요구 성장이 없어(적자 또는 범위 밖) 해당 구간의 기저율이 없다."])
            tips.append(TIPS["base"])
        fill_table(self.tbl_core, rows, tips=tips)

    def _axis(self, q: dict) -> None:
        if q["views"]:
            self.tbl_axis.set_headers(["성장 전망"] + [pct(g_, 0) for g_ in q["views"]])
            fill_table(self.tbl_axis, [["연 수익률"] + [pct(v) for v in q["returns"]]],
                       right_cols=set(range(1, len(q["views"]) + 1)),
                       tips=["10년간 그 속도로 이익이 자라고 그 뒤 추정 감쇠가 이어진다면, 오늘 가격에 사서 얻는 연 수익률."])
            self.lbl_axis.setText("자기 요구수익률을 먼저 정하고, 그 이상을 주는 성장 전망을 믿을 수 있는지 자문한다. "
                                  f"참고로 모형이 쓰는 할인율은 {pct(q['r'])}이고, 이것은 평균적 투자자의 값이지 당신의 값이 아니다.")
        else:
            self.tbl_axis.set_headers(["성장 전망"])
            fill_table(self.tbl_axis, [["적자 기업이라 수익률 축이 없다. 목표주가 탭의 풀기 기능으로 요구 성장·요구 이익률을 본다."]])
            self.lbl_axis.setText("")

    def _more(self, row: pd.DataFrame, q: dict) -> None:
        x = row.iloc[0]
        rows, tips = [], []
        P, v0 = q["P"], q["v0"]
        rows.append(["시가총액 / 모형 가치 / 가격÷가치", f"{bn(P)} / {bn(v0)} / {num(P / v0 if v0 > 0 else np.nan)}배",
                     "점 추정 가치는 회계 가정에 따라 중앙값 46% 움직인다. 판단은 요구 성장으로 한다."])
        tips.append(TIPS["value"])
        om = q["omega_star"]
        rows.append(["가격이 요구하는 지속성 ω*",
                     f"{om:.3f}  (반감기 {q['half_life']:.0f}년)" if isnum(om) and 0 < om < 1 else "0.995로도 부족 — 이익 규모의 성장이 필요",
                     "기저율: 시총 상위 5% 기업이 실제로 보인 지속성은 0.84~0.89(반감기 4~6년), 유니버스 동일가중 0.56."])
        tips.append(TIPS["omega"])
        T = q["T_star"]
        rows.append(["초과 수익성 유지 기간 T*", f"{T:.0f}년" if isnum(T) else "60년 초과 또는 해당 없음",
                     "초과 수익성이 갑자기 사라진다고 보면 몇 년이 남아 있어야 가격이 맞는가."])
        tips.append(TIPS["T"])
        if q["phase"] is not None:
            rows.append(["평상시 이익 기준 요구 성장",
                         f"{pct(q['g_norm']) + '/년' if isnum(q['g_norm']) else '해당 없음'}   [이익 국면: {q['phase']}]   "
                         f"평상시 이익 {bn(q['e_norm'], 1)} (5년 평균 이익률 {pct(q['margin_avg5'])}) vs 지금 {bn(q['E0'], 1)} ({pct(q['margin_ttm'])})",
                         ("지금 이익이 5년 평균과 30% 넘게 다르다. 경기 피크·저점일 수 있으니 이 값을 함께 읽는다." if q["phase"] != "정상"
                          else "지금 이익이 5년 평균 이익률 수준이다.")])
            tips.append(TIPS["norm"])
        if isnum(q["g_alt"]):
            rows.append([f"평균 위험 프리미엄({pct(self.store.erp_avg)})에서의 요구 성장", f"{pct(q['g_alt'])}/년  (할인율 {pct(q['r_alt'])})",
                         "위험 프리미엄이 2014년 이후 평균 수준이었다면 가격이 요구하는 성장은 이만큼이다."])
            tips.append(TIPS["erp"])
        rows.append(["할인율 = 국채 + 베타 × 위험 프리미엄", f"{pct(x['r'])} = {pct(x['rf'])} + {num(x['beta'])} × {pct(x['erp'])}",
                     "할인율을 바꾸면 요구 성장이 움직인다 (상위 200 기준 ±1%p → 약 2.4%p)."])
        tips.append(TIPS["r"])
        rows.append(["모형이 추정한 지속성 / 환원율 / 초과 ROE",
                     f"{num(x['omega'], 3)} (적용 {num(min(float(x['omega']), 0.90), 3)}) / {num(x['payout'])} / "
                     f"{float(x['x0']):+.3f} (산업 기준 ROE {float(x['roe_star_ind']):+.3f}, 조정 ROE {float(x['roe_adj']):+.3f})",
                     "비슷한 특성의 기업들이 과거에 보인 기저율이다. 이 회사가 다르다고 보면 목표주가 탭에서 바꾼다."])
        tips.append(TIPS["model_omega"])
        fill_table(self.tbl_more, rows, tips=tips)

    def _base_table(self, f: dict) -> None:
        if f["bucket"] is None:
            fill_table(self.tbl_base, [["요구 성장이 없어 구간 기저율 해당 없음", "", "", ""]])
            return
        rows = []
        if f["overall"] is not None:
            o = f["overall"]
            rows.append([f"전체 (요구 성장 {f['bucket']})", f"{int(o['n']):,}" if isnum(o["n"]) else DASH, pct(o["rate"], 0),
                         f"{float(o['median']):.1f}%" if isnum(o["median"]) else DASH])
        if f["size"] is not None:
            rows.append([f"규모: {f['size']['group']} (이 종목 {f.get('rank', 0)}위)", f"{f['size']['n']:,}", pct(f["size"]["rate"], 0), ""])
        if f["past"] is not None:
            rows.append([f"직전 1년 매출성장 {pct(f.get('past_growth', np.nan), 0, True)}: {f['past']['group']}",
                         f"{f['past']['n']:,}", pct(f["past"]["rate"], 0), ""])
        fill_table(self.tbl_base, rows if rows else [["기저율 표 없음", "", "", ""]], right_cols={1, 2, 3},
                   tips=[TIPS["base"]] * max(1, len(rows)))

    def _grid(self, q: dict) -> None:
        if q["grid"]:
            self.tbl_grid.set_headers(["지속성 ω"] + [f"{g_:.2f}" for g_ in GRID])
            fill_table(self.tbl_grid, [["가격/가치"] + [num(pv) for _, pv in q["grid"]]],
                       right_cols=set(range(1, len(GRID) + 1)), tips=[TIPS["grid"]])
        else:
            self.tbl_grid.set_headers(["지속성 ω"])
            fill_table(self.tbl_grid, [["적자이거나 초과 수익성이 장기 수준 아래라 지속성 격자가 의미 없음"]])
