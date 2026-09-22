# -*- coding: utf-8 -*-
"""Phase 2 checkpoint: demand/flow variables, price multiplier of flow-induced trading.

Writes reports/phase2_module_d.md.

Multiplier design (Lou 2012 style, monthly Fama-MacBeth):
    impact      ret[t-5, t-2]  on fit_q(t)      flows in months t-4..t-2 move prices in the window
    next 3m     ret[t, t+3]    on fit_q(t)      continuation after the flow is public
    next 12m    ret[t, t+12]   on fit_q(t)      reversal horizon
Controls: log market cap. Returns from yfinance adjusted closes (listed names,
survivorship-biased) - fine for a diagnostic, replaced by 13F implied prices
in phase 3. Newey-West t-statistics on the monthly coefficient series.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sfv import config as C  # noqa: E402

pd.set_option("display.width", 220)
L = []


def p(t=""):
    L.append(t)


def nw_t(series: pd.Series, lags: int = 3) -> float:
    s = series.dropna().values
    n = len(s)
    if n < 5:
        return np.nan
    e = s - s.mean()
    v = e @ e / n
    for lag in range(1, lags + 1):
        w = 1 - lag / (lags + 1)
        v += 2 * w * (e[lag:] @ e[:-lag]) / n
    return s.mean() / np.sqrt(max(v, 1e-18) / n)


def fm(df: pd.DataFrame, y: str, xs: list[str], min_n: int = 150) -> pd.DataFrame:
    rows = []
    for m, d in df.groupby("month"):
        d = d.dropna(subset=[y] + xs)
        if len(d) < min_n:
            continue
        X = sm.add_constant(d[xs].values)
        b = np.linalg.lstsq(X, d[y].values, rcond=None)[0]
        rows.append(pd.Series(b, index=["const"] + xs, name=m))
    T = pd.DataFrame(rows)
    return pd.DataFrame({"mean": T.mean(), "nw_t": [nw_t(T[c]) for c in T.columns], "months": len(T)})


def main():
    D = pd.read_parquet(C.PQ / "demand_monthly.parquet")
    R = pd.read_parquet(C.PQ / "rim_monthly.parquet", columns=["cik", "month", "log_pv", "log_pb_adj", "icc"])
    U = pd.read_parquet(C.PQ / "universe_monthly.parquet", columns=["cik", "month", "adj_close", "mktcap", "in_universe"])
    U["month"] = U["month"] + pd.offsets.MonthEnd(0)
    U = U.sort_values(["cik", "month"])
    g = U.groupby("cik")
    U["lr"] = np.log(U["adj_close"]).where(U["adj_close"] > 0)
    for h, name in ((3, "ret_f3"), (12, "ret_f12")):
        fwd = g["lr"].shift(-h) - U["lr"]
        gap = (g["month"].shift(-h) - U["month"]).dt.days
        U[name] = fwd.where(gap.between(h * 28, h * 33)).clip(-2, 2)
    back = U["lr"].shift(0)
    U["ret_win"] = (g["lr"].shift(2) - g["lr"].shift(5)).where((g["month"].shift(2) - g["month"].shift(5)).dt.days.between(84, 96)).clip(-2, 2)
    U["log_size"] = np.log(U["mktcap"])
    X = D.merge(U[["cik", "month", "ret_win", "ret_f3", "ret_f12", "log_size"]], on=["cik", "month"], how="left")
    X = X.merge(R, on=["cik", "month"], how="left")

    p("# 단계 2 — 모듈 D (수요·수급 변수) 체크포인트\n")
    p(f"> 생성 {pd.Timestamp.today().date()}. 유니버스 기업-월 {len(D):,}, {D.month.min().date()} ~ {D.month.max().date()}. "
      "13F 변수는 분기말+46일부터, N-PORT 변수는 제출일부터 사용(point-in-time).")

    p("\n## 1. 변수 정의\n")
    p("| 변수 | 정의 | 출처·기간 |\n|---|---|---|")
    p("| inst_share | 13F 기관 보유 가치 / 시가총액 | 13F 2013Q2~ |")
    p("| big3_share | BlackRock·Vanguard·State Street 보유 가치 / 시가총액 | 13F |")
    p("| ext_share | big3 + Geode·Northern Trust·Schwab·Invesco·BNY | 13F |")
    p("| index_share | 이름에 지수형 표기가 있는 펀드들의 보유 가치 / 시가총액 (펀드별 최신 보고 월별 이월) | N-PORT 2019Q4~ |")
    p("| fund_share | N-PORT 주식형 펀드 전체 보유 가치 / 시가총액 | N-PORT |")
    p("| sp500_proxy | Vanguard 500 Index Fund 또는 iShares Core S&P 500 보유 여부 | N-PORT |")
    p("| nsi_12m | 12개월 분할조정 유통주식수 로그변화 (음수 = 순매입) | XBRL 표지 주식수 |")
    p("| fit_q | Σ_펀드 (분기 순유출입/직전 순자산) × 직전 보유가치 / 시가총액, 후행 3개월 흐름 창 | N-PORT |")

    p("\n## 2. 연도별 기술통계 (유니버스 중앙값, fit_q는 표준편차)\n")
    yr = X.assign(y=X["month"].dt.year).groupby("y").agg(
        firms=("cik", "nunique"), inst=("inst_share", "median"), big3=("big3_share", "median"), ext=("ext_share", "median"),
        index=("index_share", "median"), fund=("fund_share", "median"), sp500=("sp500_proxy", "mean"),
        nsi=("nsi_12m", "median"), fit_sd=("fit_q", "std"), fit_p90=("fit_q", lambda s: s.quantile(.9)))
    p(yr.round(4).to_markdown())
    p("\n큰 그림: big3 비중 상승 추세, 순발행 중앙값 0 근처(대형주는 순매입), FIT 표준편차는 Lou(2012)의 분기 FIT 표준편차와 같은 자릿수여야 한다.")

    p("\n## 3. FIT의 가격 승수 (월별 Fama-MacBeth, Newey-West t)\n")
    rows = {}
    for y, label in (("ret_win", "영향: 흐름 창 [t-5, t-2] 수익률"), ("ret_f3", "이후 3개월 수익률"), ("ret_f12", "이후 12개월 수익률")):
        res = fm(X, y, ["fit_q", "log_size"])
        rows[label] = {"fit_q 계수": res.loc["fit_q", "mean"], "t": res.loc["fit_q", "nw_t"], "months": int(res.loc["fit_q", "months"])}
    tab = pd.DataFrame(rows).T
    p(tab.round(3).to_markdown())
    p("\n해석: 계수는 시가총액 대비 1%의 흐름 유발 매수가 수익률을 몇 %p 움직이는지(승수 M). Lou 2012의 종목 승수 약 1.2, Pavlova·Sikorskaya 0.3~0.5, "
      "Chang·Hong·Liskovich 0.7 수준과 비교한다. 이후 12개월 계수가 음수면 되돌림, 즉 일시적 성분의 근거다. "
      "주의: 흐름 창 수익률 회귀는 펀드 흐름이 성과를 좇는 역인과가 섞여 상향 편향될 수 있다.")

    p("\n## 4. log(P/V_F)와 수요 변수의 횡단면 관계 (월별 FM, 통제: 로그 시총)\n")
    Xv = X.dropna(subset=["log_pv"])
    res = fm(Xv, "log_pv", ["inst_share", "big3_share", "nsi_12m", "log_size"])
    p("13F 표본 (2014-06~):\n")
    p(res.round(3).to_markdown())
    Xn = Xv[Xv["month"] >= "2020-03-31"]
    res2 = fm(Xn, "log_pv", ["index_share", "fund_share", "sp500_proxy", "nsi_12m", "log_size"])
    p("\nN-PORT 표본 (2020-03~):\n")
    p(res2.round(3).to_markdown())
    p("\n이것은 단계 3 분해의 예비 결과다. 양의 계수는 그 수요 변수가 높은 기업이 펀더멘털 대비 높게 거래된다는 뜻이며, "
      "그 프리미엄이 지속적(δ_S)인지 되돌리는지(δ_T)는 수익률 예측 검정으로 판정한다.")

    p("\n## 5. 주의점\n")
    p("- 수익률은 yfinance 조정종가 기준이라 상장폐지 종목이 빠진 생존편향 표본이다. 단계 3에서 13F 내재가격으로 대체한다.")
    p("- index_share는 펀드명 기반 분류라 하한이고, 기관 내부 인덱스 운용과 클로짓 인덱싱은 잡히지 않는다(Chinco·Sammon).")
    p("- FIT는 N-PORT 제출 펀드(뮤추얼펀드·ETF)만 포함하며 2019년 말부터 존재한다.")
    (C.REPORTS / "phase2_module_d.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
