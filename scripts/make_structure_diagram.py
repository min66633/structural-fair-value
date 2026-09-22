# -*- coding: utf-8 -*-
"""Draw the tool's structure as one annotated diagram.

The equations poster (make_framework_poster.py) shows the model; this shows
the machine: where the data comes from, what each stage turns it into, which
outputs the user reads and in what order, and what the numbers can and cannot
bear. Plain Korean, no mathtext. Live numbers (valuation month, universe size)
are read from the pipeline outputs.

Output reports/sfv_structure.png
Run: python scripts/make_structure_diagram.py [--dpi 150]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sfv import config as C  # noqa: E402

W, H = 13.0, 14.7
BG, FG, MUT, ACC, RULE = "#fbfaf8", "#1d1b18", "#6b6560", "#7a3b2e", "#ddd8d0"
FILL = {"data": "#f1efeb", "engine": "#f4ede4", "diag": "#efefef", "core": "#f8e9e3", "out": "#eef1ee", "side": "#f6f3ee"}
LX0, LX1, RX0, RX1 = 0.040, 0.640, 0.672, 0.962


def lh(fs: float, mult: float = 1.5) -> float:
    return fs * mult / 72.0 / H


def facts() -> dict:
    E = pd.read_parquet(C.PQ / "expectations_monthly.parquet", columns=["month", "cik", "g_star_10y"])
    m = E["month"].max()
    d = E[E["month"] == m]
    U = pd.read_parquet(C.PQ / "universe_monthly.parquet", columns=["month", "in_universe"])
    n_uni = int(U[(U["month"] == m) & U["in_universe"]].shape[0])
    return {"m": m, "n_val": len(d), "n_g": int(d["g_star_10y"].notna().sum()), "n_uni": n_uni,
            "m0": E["month"].min(), "months": E["month"].nunique()}


class Sheet:
    def __init__(self, dpi: int):
        self.fig = plt.figure(figsize=(W, H), dpi=dpi)
        self.fig.patch.set_facecolor(BG)

    def text(self, x, y, s, fs=10, color=FG, weight="normal", ha="left"):
        self.fig.text(x, y, s, fontsize=fs, color=color, fontweight=weight, ha=ha, va="top")

    def box(self, x0, x1, y_top, num, title, lines, fill, fs=10.0, title_fs=11.5, pad=0.009, accent=ACC):
        """Rounded box with a numbered title and explanation lines; returns the bottom y."""
        # text is drawn top-anchored, so the last line only needs its glyph height, not a full line pitch
        h = pad + lh(title_fs, 1.45) + (len(lines) - 0.35) * lh(fs, 1.5) + pad
        self.fig.add_artist(FancyBboxPatch((x0, y_top - h), x1 - x0, h, boxstyle="round,pad=0.003,rounding_size=0.006",
                                           linewidth=0.9, edgecolor=RULE, facecolor=fill, transform=self.fig.transFigure, zorder=0))
        self.fig.add_artist(Line2D([x0 + 0.002, x0 + 0.002], [y_top - h + 0.004, y_top - 0.004], color=accent, lw=2.6,
                                   transform=self.fig.transFigure, zorder=1))
        y = y_top - pad
        if num:
            self.text(x0 + 0.014, y, num, title_fs, accent, "bold")
            self.text(x0 + 0.014 + 0.028, y, title, title_fs, FG, "bold")
        else:
            self.text(x0 + 0.014, y, title, title_fs, FG, "bold")
        y -= lh(title_fs, 1.45)
        for s in lines:
            self.text(x0 + 0.014, y, s, fs, FG if not s.startswith("  ") else MUT)
            y -= lh(fs, 1.5)
        return y_top - h

    def arrow(self, x, y_from, y_to, label=""):
        self.fig.add_artist(FancyArrowPatch((x, y_from), (x, y_to), transform=self.fig.transFigure, arrowstyle="-|>",
                                            mutation_scale=16, lw=1.3, color=MUT, zorder=2))
        if label:
            self.text(x + 0.012, (y_from + y_to) / 2 + 0.004, label, 9.2, MUT)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--out", default=str(C.REPORTS / "sfv_structure.png"))
    a = ap.parse_args()
    plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False})
    F = facts()
    S = Sheet(a.dpi)
    fig = S.fig

    # ------------------------------------------------------------------ title
    S.text(LX0, 1 - 0.40 / H, "SFV 도구의 구조 — 자료에서 판단까지", 26, FG, "bold")
    S.text(LX0, 1 - 0.90 / H, "왼쪽은 자료가 흘러가는 순서, 오른쪽은 그 흐름을 이해하는 데 필요한 생각. 색이 다른 두 상자(⑤ 역산, ⑥ 정방향)가 사용자가 쓰는 부분이다",
           11.5, MUT)
    S.text(RX1, 1 - 0.42 / H, f"평가 기준월 {F['m'].date()}", 11, MUT, ha="right")
    S.text(RX1, 1 - 0.70 / H, f"유니버스 {F['n_uni']:,}종목 · 평가 {F['n_val']:,}종목 · 요구 성장 산출 {F['n_g']:,}종목", 10, MUT, ha="right")
    fig.add_artist(Line2D([LX0, RX1], [1 - 1.15 / H] * 2, color=FG, lw=1.5, transform=fig.transFigure))
    y = 1 - 1.32 / H
    xm = (LX0 + LX1) / 2
    gap = 0.024

    # ================================================================== main spine
    y1 = S.box(LX0, LX1, y, "①", "원본 자료 — 전부 무료 공개 데이터", [
        "SEC XBRL 재무제표(2009~). 최초 보고치만 쓴다: 나중에 정정된 숫자는 그때 알 수 없었으므로 쓰지 않는다",
        "SEC 13F 기관 보유(2013~), N-PORT 펀드 보유·유출입(2019~), FTD 파일의 상장폐지 종목 가격",
        "yfinance 월말 가격·분할·컨센서스, FRED 10년 국채, Damodaran 내재 위험프리미엄(ERP)",
        "Ewens·Peters·Wang(2025) 산업별 무형자산 파라미터: R&D 상각률, 판관비 중 투자 몫",
    ], FILL["data"])
    S.arrow(xm, y1 - 0.003, y1 - gap + 0.004, "파싱 · 시점 정렬")
    y2 = S.box(LX0, LX1, y1 - gap, "②", "정리 — 시점 규칙을 지킨 월별 패널", [
        "기업-분기 재무 패널: 행마다 공시일을 두어 그달 말에 알 수 있던 값만 쓴다(point in time)",
        f"유니버스: 미국 비금융 보통주, 시총 3억$ 이상, 400일 이내 재무제표 → 월 약 {F['n_uni']:,}종목",
        "  채굴→데이터센터 전환사는 SEC가 금융(SIC 6199)으로 분류해도 포함. 디지털자산 보유회사·거래소는 제외",
        "가격·시총(분할 보정 주식수), 기관·패시브 보유 비중(분기말+46일부터), 펀드 흐름(제출일부터)",
    ], FILL["data"])
    S.arrow(xm, y2 - 0.003, y2 - gap + 0.004, "기업-월 247,833행")
    y3 = S.box(LX0, LX1, y2 - gap, "③", "가치 엔진 — 잔여이익모형(장부 + 초과이익의 현재가치)", [
        "무형자산 자본화: R&D 전액과 판관비 일부를 자산으로 올리고 산업별로 상각 → 조정 장부 B, 조정이익 E, 조정 ROE",
        "지속성 ω: 초과 ROE(산업 평균을 넘는 수익성)가 평균으로 돌아오는 속도를 수천 기업의 기록에서 기업 특성별로 추정",
        "V_F = B + Σ 10년 초과이익의 현재가치(ω로 감쇠) + 터미널.  할인율 r = 국채 + β × ERP.  경계값으로 폭발 방지",
        "V_gr: 앞 3년을 백로그(잔여이행의무) 기반 매출 전망으로 키운 변형. 성장 단계를 명시하는 경로",
    ], FILL["engine"])
    S.arrow(xm, y3 - 0.003, y3 - gap + 0.004, "V_F, V_gr, ω, r")
    y4 = S.box(LX0, LX1, y3 - gap, "④", "분해 — 프리미엄을 성분으로 나눈다 (진단용)", [
        "log(P / V_F) = 구조적 수요 + 일시 수급 + 특성 보정 + 모멘텀 + 펀더멘털 기대 + 산업 + 잔차 ε  (월별 횡단면 회귀)",
        "2014~2026년 검정: 어느 성분도, 어느 벤치마크도 이후 수익률을 예측하지 못했다 → 매매 신호로 쓰지 않는다",
        "잔차 ε는 '동료·특성 대비 얼마나 비싼가'의 진단이지 판단의 근거가 아니다. 초대형주에서는 방향조차 없다",
    ], FILL["diag"])
    S.arrow(xm, y4 - 0.003, y4 - gap + 0.004, "같은 엔진을 거꾸로 푼다")
    y5 = S.box(LX0, LX1, y4 - gap, "⑤", "역산 — 이 가격이 무엇을 전제하는가 (사용자가 읽는 것)", [
        "시장 수준: ΣV_F = ΣP가 되는 공통 지속성 ω*(반감기)와 경쟁우위 기간 T*. 시장 전체가 전제하는 지속성",
        "종목: 요구 성장 g*(10년) + 구간(자본화 파라미터 6조합), 요구 지속성 ω*, 요구 경쟁우위 기간 T*",
        "수익률 축 r*(g): 성장 전망 5~30%별로 지금 가격이 주는 연 수익률. 자기 요구수익률과 비교",
        "기저율: 그 요구를 이후 3년간 실제로 달성한 역사적 비율(구간별·규모별, 종목 특성 조건부 확률)",
        "적자·전환 기업: 이익률 경로를 정하면 요구 매출 성장을, 매출 경로를 정하면 요구 이익률을 푼다",
    ], FILL["core"], accent=ACC)
    S.arrow(xm, y5 - 0.003, y5 - gap + 0.004, "같은 엔진을 앞으로 푼다")
    y6 = S.box(LX0, LX1, y5 - gap, "⑥", "정방향 — 내 견해로는 얼마인가 (목표가)", [
        "연도별 매출 성장 × 조정 이익률 → 이익 경로 → 같은 감쇠·터미널 → 가치, 주당 목표가, 그 경로에서 가격이 주는 수익률",
        "컨센서스 매출 성장(yfinance)을 경로로 넣어 비교. 정규화 이익, 감쇠 속도, 할인율도 바꿀 수 있다. 기존 DCF의 자리",
    ], FILL["core"], accent=ACC)
    S.arrow(xm, y6 - 0.003, y6 - gap + 0.004, "")
    y7 = S.box(LX0, LX1, y6 - gap, "⑦", "출력 — 표, 도구, 기록", [
        f"종목별 표 reports/sfv_report.html · sfv_latest.csv: {F['n_val']:,}종목, 요구 기대·구간·수익률 축·달성 확률·잔차·표시",
        "scripts/what_if.py --ticker: 한 종목의 역산, 수익률 축, 정방향 목표가, 2014년 이후 요구 성장 추이, 매수일 대조표",
        "사양 FRAMEWORK.md, 수식 포스터, 리포트 색인, 코드 리뷰·사고 기록. 월별 갱신 python run_sfv.py (약 11분, 점검 36개)",
    ], FILL["out"])

    # ================================================================== right column (drawn before the strip so the strip can sit below both columns)
    yr = 1 - 1.32 / H
    yr = S.box(RX0, RX1, yr, "", "핵심 생각: 가격 하나, 미지수 둘", [
        "가격은 미래 이익의 현재가치다. 미지수는",
        "성장과 요구수익률(할인율) 둘인데 식은 하나.",
        "그래서 하나를 고정하고 하나를 푼다.",
        "  요구 성장 g*: 할인율을 고정하고 성장을 푼다",
        "  수익률 축 r*(g): 성장을 고정하고 수익률을 푼다",
        "  적자 기업: 규모(매출)와 이익률 중 하나를 고정",
        "도구는 '무엇이 참이어야 이 가격이 맞는가'를",
        "말한다. 참인지는 사용자가 판단한다.",
    ], FILL["side"])
    yr = S.box(RX0, RX1, yr - 0.016, "", "왜 감쇠를 추정하는가", [
        "보통의 DCF는 터미널 성장률 하나가 가치의",
        "절반을 정하고 그 값은 가정이다. 여기서는",
        "초과 수익성이 산업 평균으로 돌아오는 속도를",
        "수천 기업의 실제 기록에서 추정한다.",
        "감쇠는 산업 축소가 아니다. 이익은 계속 크고,",
        "산업 평균을 넘는 부분만 좁아진다.",
        "  형태는 가정, 속도는 추정, 목표 수준은 선택",
    ], FILL["side"])
    yr = S.box(RX0, RX1, yr - 0.016, "", "무엇을 믿고 무엇을 참고만 하나", [
        "상대 순위: 파라미터를 바꿔도 유지",
        "  (순위상관 0.92~0.98) → 믿는다",
        "시장 내재 지속성: 0.95~0.96으로 불변 → 믿는다",
        "요구 성장: 초대형주는 좁고(1~4%p), 소형주는",
        "  넓다(중앙값 6%p) → 구간으로 읽는다",
        "점 추정 가치: 파라미터에 따라 중앙값 46%",
        "  움직인다 → 참고만",
        "할인율 ±1%p = 요구 성장 2~3%p → 요구수익률을",
        "  먼저 정해야 한다",
    ], FILL["side"])
    yr = S.box(RX0, RX1, yr - 0.016, "", "적자·전환 기업은 어떻게", [
        "이익이 없으면 g*·ω*·T*가 정의되지 않는다.",
        "가격이 규모와 이익률을 함께 전제하기 때문.",
        "이익률 경로(예: 30→40%)를 주면 요구 매출",
        "성장이 나오고, 매출 경로를 주면 요구 이익률이",
        "나온다. 과거 어느 달에서도 같은 계산이 된다.",
        "  예: 아이렌 2026-09, 이익률 30→40%면",
        "  5년간 연 90% 성장 필요, 기저율 3%",
    ], FILL["side"])
    yr = S.box(RX0, RX1, yr - 0.016, "", "한계", [
        "예측 모형이 아니다. 요구 성장 미달이 이후",
        "1~4분기 수익률로 처벌되지 않았다(2014~26).",
        "장부가 음수인 기업(파산 후)은 평가 불가.",
        "외국 법인의 IFRS 공시 기간은 파싱되지 않음.",
        "금융업 제외. 12년 한 강세장의 기저율.",
        "회사를 아는 일(제품·경쟁·경영진)은 도구 밖.",
    ], FILL["side"])

    # ================================================================== bottom strip: reading order, below both columns
    yb = min(y7, yr) - 0.030
    ys = S.box(LX0, RX1, yb, "", "읽는 순서", [
        "① 자기 요구수익률을 정한다. 모형 할인율(국채 + β × ERP)은 평균적 투자자의 값이지 당신의 값이 아니다",
        "② 자기 성장 전망의 열에서 가격이 주는 수익률 r*(g)를 읽어 ①과 비교한다. 같은 것을 성장 축으로 읽으면 요구 성장 g*와 그 구간이다",
        "③ 자기 기대가 구간 안이면 보류한다(모형의 정밀도 밖)   ④ 3년 달성 확률로 전망을 점검한다   ⑤ 상대 비교는 믿고, 절대 가치는 참고만 한다",
        "⑥ 실적이 나오면 다시 읽는다. --since 매수일로 그날의 전제와 이후 실현을 대조한다   ⑦ 적자·전환 기업은 이익률 경로를 정하고 요구 매출 성장을 읽는다",
    ], FILL["side"], fs=9.8, title_fs=11)

    # ------------------------------------------------------------------ footer
    yf = min(ys, yr) - 0.014
    fig.add_artist(Line2D([LX0, RX1], [yf, yf], color=RULE, lw=1.0, transform=fig.transFigure))
    S.text(LX0, yf - 0.006, "수식은 reports/sfv_framework.png · 사양 FRAMEWORK.md · 종목별 표 reports/sfv_report.html · 한 종목 python scripts/what_if.py --ticker T",
           9.2, MUT)
    S.text(RX1, yf - 0.006, f"생성 {pd.Timestamp.today().date()}", 9.2, MUT, ha="right")
    out = Path(a.out)
    fig.savefig(out, dpi=a.dpi, facecolor=BG)
    plt.close(fig)
    print(f"main column ends at y={y7:.3f}, strip at y={ys:.3f}, right column at y={yr:.3f} (all must stay above 0.03)")
    if min(ys, yr) < 0.03 or yr < yb:
        print("  WARNING: overflow, or the right column runs into the reading-order strip")
    print(f"wrote {out} ({out.stat().st_size/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
