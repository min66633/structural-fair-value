# -*- coding: utf-8 -*-
"""Tab 6, 도움말: one walk through the tool, on the stock currently selected.

A reference list tells you where things are; it does not teach you to use anything. This walks the six steps
with the selected firm's own numbers in them, and ends with a button that proves the two directions are one
engine — put the required growth into the forward tab and the target price comes back as today's price."""
from __future__ import annotations

import html

import numpy as np
import pandas as pd
from PySide6.QtWidgets import QPushButton, QTextBrowser, QVBoxLayout, QWidget

from sfv.store import Store
from app.phrasing import base_rate_facts, per_hundred, two_views
from app.theme import C
from app.widgets import isnum, note


def _css() -> str:
    return f"""
<style>
  body {{ color: {C['text_dim']}; }}
  h3 {{ color: {C['text']}; margin: 18px 0 4px 0; }}
  b {{ color: {C['text']}; }}
  p, li {{ color: {C['text_dim']}; }}
  li {{ margin-bottom: 5px; }}
  .num {{ color: {C['accent']}; font-weight: bold; }}
  .step {{ color: {C['text']}; font-weight: bold; }}
  .box {{ background: {C['surface_alt']}; }}
</style>"""


class HelpTab(QWidget):
    def __init__(self, store: Store, on_demo=None):
        super().__init__()
        self.store = store
        self.on_demo = on_demo
        self.row: pd.DataFrame | None = None
        self.q: dict | None = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 16, 14, 14)
        lay.setSpacing(8)
        self.btn = QPushButton()
        self.btn.setToolTip("요구 성장을 10년 경로로 목표주가 탭에 넣고 계산합니다. 그 탭에 입력해 둔 값은 덮어써집니다.")
        self.btn.clicked.connect(self._demo)
        lay.addWidget(self.btn)
        self.lbl = note()
        lay.addWidget(self.lbl)
        self.view = QTextBrowser()
        self.view.setOpenExternalLinks(True)
        lay.addWidget(self.view, 1)
        self.set_row(None, None)

    def _demo(self) -> None:
        if self.on_demo is not None and self.q is not None and isnum(self.q.get("g_star")):
            self.on_demo(float(self.q["g_star"]))

    # ------------------------------------------------------------------ fill
    def set_row(self, row: pd.DataFrame | None, q: dict | None) -> None:
        self.row, self.q = row, q
        can_demo = q is not None and isnum(q.get("g_star"))
        self.btn.setEnabled(bool(can_demo and self.on_demo is not None))
        if row is not None and can_demo:
            t = html.escape(str(row.iloc[0]["ticker"]))
            self.btn.setText(f"{t}로 직접 확인해 보기 — 요구 성장을 목표주가 탭에 넣으면 목표가가 현재가와 같아진다")
            self.lbl.setText("두 탭이 같은 엔진이라는 뜻이다. 거기서 성장이나 이익률을 자기 견해대로 움직인 만큼이 목표가의 괴리가 된다.")
        else:
            self.btn.setText("요구 성장이 있는 종목을 고르면 여기서 바로 확인해 볼 수 있다")
            self.lbl.setText("")
        self.view.setHtml(_css() + (self._walkthrough(row, q) if row is not None and q is not None else self._generic()))
        self.view.verticalScrollBar().setValue(0)

    # ------------------------------------------------------------------ text
    def _walkthrough(self, row: pd.DataFrame, q: dict) -> str:
        x = row.iloc[0]
        m = self.store.macro
        t = html.escape(str(x["ticker"]))
        name = html.escape(str(x["name"]))
        age = (self.store.month - pd.Timestamp(x["f_end"])).days
        g = q["g_star"]
        b = q["band"]
        f = base_rate_facts(self.store, row, q)
        views = two_views(q)
        n = lambda s: f'<span class="num">{s}</span>'  # noqa: E731
        f_end = str(pd.Timestamp(x["f_end"]).date())
        price = f"${float(x['price_ps']):,.2f}" if isnum(x.get("price_ps", np.nan)) else "패널값 없음"

        out = [f"<h3>지금 보고 있는 것: {t} {name}</h3>",
               "<p>이 도구는 두 방향으로 쓴다. <b>가격이 무엇을 전제하는지 거꾸로 읽고</b>(전제 탭), "
               "<b>내 전제로는 얼마인지 앞으로 계산한다</b>(목표주가 탭). 매수·매도 신호는 없다. "
               "신호를 만들려는 검정이 실패했기 때문이고, 그 사실은 이 도구의 한계로 남겨 두었다.</p>",
               f"<p><span class='step'>1단계 — 언제 것인지 먼저 본다.</span> 맨 위 줄이 답한다. "
               f"이 종목의 실적은 {n(f_end)} 기준이고 {n(str(age) + '일')} 전 것이다. "
               f"주가는 {n(price)}로 패널 월 {m['month']} 기준이며, "
               "[주가·금리 갱신]을 누르면 오늘 값으로 바꿔 즉시 다시 계산한다. 실적은 월 1회 파이프라인을 돌려야 바뀐다.</p>"]

        if isnum(g):
            band_txt = ""
            if b is not None and isnum(b["g_lo"]):
                rng = f"{b['g_lo'] * 100:.1f}~{b['g_hi'] * 100:.1f}%"
                band_txt = f" 회계 가정을 문헌의 다른 조합으로 바꾸면 {n(rng)} 사이다."
            out.append(f"<p><span class='step'>2단계 — 가격이 무엇을 전제하는지 읽는다.</span> "
                       f"요구 성장 {n(f'{g*100:.1f}%')}. 이 회사의 조정 이익이 앞으로 10년 동안 매년 그만큼 자라야 "
                       f"오늘 가격이 맞는다는 뜻이다.{band_txt} "
                       "여기서 '조정 이익'은 연구비와 판관비 일부를 비용이 아니라 자산으로 보고 고쳐 쓴 이익이다. "
                       "회계가 무형자산을 비용으로 털어 버려서 장부가와 이익이 왜곡되는 것을 되돌린 값이다.</p>")
        else:
            out.append("<p><span class='step'>2단계 — 가격이 무엇을 전제하는지 읽는다.</span> "
                       "이 회사는 적자라 성장률 하나로 전제를 읽을 수 없다. 적자 기업의 가격은 "
                       "<b>도달할 규모</b>와 <b>벌게 될 이익률</b>을 동시에 전제하는데, 가격 하나로 둘을 동시에 알 수는 없다. "
                       "그래서 목표주가 탭에서 하나를 정하고 나머지를 푼다(4단계).</p>")

        if f["overall"] is not None and isnum(f["overall"]["rate"]):
            s = (f"<p><span class='step'>3단계 — 그게 흔한 일인지 본다.</span> "
                 f"비슷한 요구를 받았던 기업 중 이후 3년간 실제로 그 속도를 낸 곳은 {n(per_hundred(f['overall']['rate']))}이다.")
            if f["past"] is not None and isnum(f["past"]["rate"]):
                s += (f" 다만 이 회사처럼 직전 1년에 {f['past_growth']*100:+.0f}% 자란 무리만 보면 "
                      f"{n(per_hundred(f['past']['rate']))}이다.")
            s += (" 이건 확률 예측이 아니라 <b>기저율</b>이다. 낮다고 해서 이 회사가 못 한다는 뜻이 아니라, "
                  "'이 회사는 그 소수에 든다'고 볼 근거가 있어야 한다는 뜻이다. 그 근거를 대는 일이 곧 기업 분석이고, "
                  "도구 밖의 일이다.</p>")
            out.append(s)

        if len(views) == 2:
            (ga, ra), (gb, rb) = views
            out.append(f"<p><span class='step'>4단계 — 내 전망을 넣어 본다.</span> 수익률 축을 읽는다. "
                       f"연 {ga*100:.0f}% 성장을 본다면 이 가격은 {n(f'연 {ra*100:.1f}%')}를 주고, "
                       f"{gb*100:.0f}%로 보면 {rb*100:.1f}%다. "
                       "먼저 자기 요구수익률을 정하고(예: 국채보다 몇 %p 위여야 하는가), 그보다 나은 수익률을 주는 "
                       "성장 전망을 믿을 수 있는지 자문한다. 가격 하나는 성장과 요구수익률을 묶은 식 하나라서, "
                       "둘 중 하나를 정해야 다른 하나가 나온다.</p>")

        out.append("<p><span class='step'>5단계 — 목표가를 낸다.</span> 목표주가 탭에서 "
                   "<b>매출 성장 경로</b>와 <b>조정 이익률 경로</b>를 넣으면 이익 경로가 만들어지고, 그 뒤로는 모형이 추정한 "
                   "감쇠와 터미널이 붙어 주당 목표가가 나온다. 예를 들어 <b>40,30,20,15,12</b>와 <b>55,52,50,48</b>은 "
                   "'매출이 5년간 40%에서 12%로 둔화하고 이익률이 55%에서 48%로 내려간다'는 견해다. "
                   "나온 목표가 옆의 <b>가치 구성</b>을 꼭 본다. 목표가의 절반 이상이 내가 넣은 5년이 아니라 그 뒤의 "
                   "감쇠 구간과 터미널에서 나오는 경우가 많고, 그 부분은 내 견해가 아니라 모형의 기저율이다.</p>")
        if not isnum(g):
            out.append("<p>적자 기업은 같은 탭의 <b>요구 성장 풀기</b>(이익률 경로를 고정)나 <b>요구 이익률 풀기</b>"
                       "(매출 경로를 고정)를 쓴다. '이익률 40%를 낸다면 매출이 연 90%씩 커야 한다'처럼, "
                       "믿을 수 있는지 판단 가능한 형태로 바뀐다.</p>")
        out.append("<p><span class='step'>6단계 — 실적이 나오면 다시 읽는다.</span> 추이 탭은 분기마다 요구 성장이 어디로 "
                   "갔는지 보여 준다. 내려왔으면 이익이 가격을 따라잡은 것이고, 올라갔으면 가격이 이익보다 앞서 간 것이다. "
                   "매수일 대조 탭에 산 날짜를 넣으면 그날의 전제와 그 뒤 실제 성장을 나란히 놓는다.</p>")

        out.append("<h3>헷갈리기 쉬운 것 넷</h3><ul>"
                   "<li><b>감쇠는 산업이 줄어드는 게 아니다.</b> 남보다 잘 버는 몫이 경쟁 때문에 평균 쪽으로 수렴한다는 뜻이다. "
                   "매출이 줄어든다는 말이 아니다.</li>"
                   "<li><b>요구 성장은 배당 성장률이 아니다.</b> 회계를 고쳐 쓴 이익의 성장률이다.</li>"
                   "<li><b>기저율은 이 회사의 확률이 아니다.</b> 비슷한 처지였던 기업들이 실제로 어떻게 됐는지의 기록이다.</li>"
                   "<li><b>모형 가치는 참고만 한다.</b> 회계 가정에 따라 종목 중앙값 46% 움직인다. "
                   "반면 요구 성장과 종목 간 순위는 거의 움직이지 않아서, 판단을 거기에 싣는다.</li></ul>")

        out.append(f"<h3>이 화면의 숫자가 어디서 오나</h3><ul>"
                   f"<li>패널 월 {m['month']}, 유니버스 {m['n_firms']:,}종목(미국 상장 비금융, 조정 장부가가 양수인 기업). "
                   f"밖의 기업은 수기 입력 탭에 직접 넣는다.</li>"
                   f"<li>국채 10년 {m['rf']*100:.2f}%, 위험 프리미엄 {m['erp']*100:.2f}%(전년 말 시장 전체에서 역산한 값), "
                   f"2014년 이후 평균 {m['erp_avg']*100:.2f}%.</li>"
                   f"<li>지속성은 {m.get('asof_year', '')}년까지의 결과로 기업 특성에 회귀해 추정한 기저율이다. "
                   f"추정치는 0.90에서 자르지만, 목표주가 탭에 직접 넣은 값은 0.995까지 그대로 쓴다.</li>"
                   f"<li>실적 발표 직후 반영은 아직 없다. SEC 기록 자체가 늦는 기업이 약 12%라 "
                   f"실적 기준일이 150일을 넘으면 맨 위에 <b>재무지연</b>이 붙는다.</li></ul>"
                   "<p>더 깊은 설명은 TUTORIAL.md(기초 개념부터), FRAMEWORK.md(현재 사양), "
                   "reports/phase7_expectations.md(검증 수치).</p>")
        return "".join(out)

    @staticmethod
    def _generic() -> str:
        return ("<h3>종목을 고르면 그 종목의 숫자로 사용법을 안내한다</h3>"
                "<p>왼쪽 목록에서 아무 종목이나 고르면, 이 탭이 그 회사의 실제 값으로 여섯 단계를 걸어간다.</p>")
