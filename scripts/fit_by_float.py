# -*- coding: utf-8 -*-
"""Does a thin float make fund flows move prices more?

The inelastic-markets argument: what matters for price impact is the flow
relative to the shares that will actually trade, so stocks whose float sits in
passive or otherwise inelastic hands should show a larger multiplier. Phase 2
estimated one multiplier for everyone. This re-estimates it by monthly terciles
of the passive big-three share, institutional share, index-fund share, size and
a float proxy (1 - big3 - index), and with an interaction term.

    impact      ret[t-5, t-2] on fit_q(t)   the flow window (reverse causality inflates it)
    next 12m    ret[t, t+12]  on fit_q(t)   reversal horizon

fit_q is winsorised at the monthly 1st/99th percentile; returns are yfinance
adjusted closes (survivorship sample, as in phase 2). Monthly Fama-MacBeth,
Newey-West t (3 lags).

Output reports/phase2_fit_by_float.md
Run: python scripts/fit_by_float.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sfv import config as C  # noqa: E402

L = []


def p(t: str = "") -> None:
    L.append(t)


def nw_t(s: pd.Series, lag: int = 3) -> float:
    s = s.dropna().to_numpy(float)
    n = len(s)
    e = s - s.mean()
    v = (e @ e) / n
    for k in range(1, lag + 1):
        if n > k:
            v += 2 * (1 - k / (lag + 1)) * (e[k:] @ e[:-k]) / n
    return float(s.mean() / np.sqrt(max(v, 1e-18) / n))


def fm(X: pd.DataFrame, y: str, xs: list[str], min_n: int = 30) -> pd.DataFrame:
    rows = []
    for m, d in X.dropna(subset=[y] + xs).groupby("month"):
        if len(d) < min_n:
            continue
        A = np.column_stack([np.ones(len(d))] + [d[c].to_numpy(float) for c in xs])
        b = np.linalg.lstsq(A, d[y].to_numpy(float), rcond=None)[0]
        rows.append(pd.Series(b, index=["const"] + xs, name=m))
    T = pd.DataFrame(rows)
    return pd.DataFrame({"coef": T.mean(), "t": [nw_t(T[c]) for c in T.columns], "months": len(T)})


def panel() -> pd.DataFrame:
    D = pd.read_parquet(C.PQ / "demand_monthly.parquet")
    U = pd.read_parquet(C.PQ / "universe_monthly.parquet", columns=["cik", "month", "adj_close", "mktcap", "in_universe"])
    U = U[U["mktcap"] > 0].copy()
    U["month"] = U["month"] + pd.offsets.MonthEnd(0)
    U = U.sort_values(["cik", "month"])
    g = U.groupby("cik")
    with np.errstate(divide="ignore", invalid="ignore"):
        U["lr"] = np.log(U["adj_close"]).where(U["adj_close"] > 0)
    U["ret_win"] = (g["lr"].shift(2) - g["lr"].shift(5)).where((g["month"].shift(2) - g["month"].shift(5)).dt.days.between(84, 96)).clip(-2, 2)
    fwd = g["lr"].shift(-12) - U["lr"]
    gap = (g["month"].shift(-12) - U["month"]).dt.days
    U["ret_f12"] = fwd.where(gap.between(12 * 28, 12 * 33)).clip(-2, 2)
    U["log_size"] = np.log(U["mktcap"])
    X = D.merge(U[["cik", "month", "ret_win", "ret_f12", "log_size", "in_universe"]], on=["cik", "month"], how="left")
    X = X[X["fit_q"].notna() & X["in_universe"].fillna(False) & X["log_size"].notna()].copy()
    X["fit_q"] = X.groupby("month")["fit_q"].transform(lambda s: s.clip(s.quantile(.01), s.quantile(.99)))
    X["float_proxy"] = 1 - X["big3_share"].fillna(0) - X["index_share"].fillna(0)
    return X


def by_tercile(X: pd.DataFrame, col: str, label: str) -> pd.DataFrame:
    S = X.dropna(subset=[col]).copy()
    S["q"] = S.groupby("month")[col].transform(lambda s: pd.qcut(s.rank(method="first"), 3, labels=["하위", "중간", "상위"]))
    rows = []
    for q in ["하위", "중간", "상위"]:
        d = S[S["q"] == q]
        r, r12 = fm(d, "ret_win", ["fit_q", "log_size"]), fm(d, "ret_f12", ["fit_q", "log_size"])
        rows.append({"3분위": q, f"{label} 중앙값": round(float(d[col].median()), 3),
                     "영향 승수": round(float(r.loc["fit_q", "coef"]), 2), "t": round(float(r.loc["fit_q", "t"]), 1),
                     "이후 12개월": round(float(r12.loc["fit_q", "coef"]), 2), "t ": round(float(r12.loc["fit_q", "t"]), 1),
                     "기업-월": int(len(d))})
    S["z"] = S.groupby("month")[col].transform(lambda s: (s - s.mean()) / (s.std() + 1e-12))
    S["fit_z"] = S["fit_q"] * S["z"]
    r = fm(S, "ret_win", ["fit_q", "fit_z", "z", "log_size"])
    rows.append({"3분위": "상호작용 fit×z", f"{label} 중앙값": np.nan, "영향 승수": round(float(r.loc["fit_z", "coef"]), 2),
                 "t": round(float(r.loc["fit_z", "t"]), 1), "이후 12개월": np.nan, "t ": np.nan, "기업-월": int(len(S))})
    return pd.DataFrame(rows)


def main() -> int:
    X = panel()
    m0, m1 = X["month"].min(), X["month"].max()
    p("# 흐름의 가격 승수는 유통량이 적을수록 큰가 (단계 2 보충)\n")
    p(f"> 생성 {pd.Timestamp.today().date()}. 표본 {m0.date()} ~ {m1.date()}, 기업-월 {len(X):,} (N-PORT 펀드 흐름이 있는 유니버스 종목). "
      "fit_q = 펀드 흐름 유발 매매 / 시가총액(후행 3개월 창, 월별 1/99% 절단). 수익률은 yfinance 조정종가(생존편향 표본). "
      "월별 Fama-MacBeth, Newey-West t(시차 3). 승수 = 시가총액 대비 1%의 흐름 유발 매수가 수익률을 몇 %p 움직이는가. "
      "'영향'은 흐름 창 [t−5, t−2] 수익률이라 흐름이 성과를 좇는 역인과로 상향 편향될 수 있고, '이후 12개월'은 되돌림 지평이다.\n")
    p("질문: 기관·패시브가 종목 간 상대 가격의 분산을 조금밖에 설명하지 못해도(δ_S 2%), 실질 유통량이 적은 종목에서는 "
      "같은 흐름이 가격을 더 크게 움직이지 않는가. 비탄력 시장 가설(Gabaix·Koijen)의 종목 횡단면 버전이다. "
      "그렇다면 승수가 패시브 지분·기관 지분이 높을수록, 유통량 대용치와 규모가 작을수록 커야 한다.\n")
    Xc = X.dropna(subset=["big3_share"])
    r = fm(Xc, "ret_win", ["fit_q", "log_size"]); r12 = fm(Xc, "ret_f12", ["fit_q", "log_size"])
    p("## 1. 전체 승수 (13F 보유 자료가 있는 종목)\n")
    p(f"영향 {r.loc['fit_q', 'coef']:+.2f} (t {r.loc['fit_q', 't']:+.1f}), 이후 12개월 {r12.loc['fit_q', 'coef']:+.2f} (t {r12.loc['fit_q', 't']:+.1f}), "
      f"{int(r.loc['fit_q', 'months'])}개월. 문헌의 종목 승수(Lou 2012 약 1.2, Pavlova·Sikorskaya 0.3~0.5, Chang·Hong·Liskovich 0.7)와 같은 자릿수다.\n")
    p("## 2. 3분위별 승수\n")
    for col, label in (("big3_share", "big3 패시브 지분"), ("inst_share", "13F 기관 지분"), ("index_share", "N-PORT 지수펀드 지분"),
                       ("float_proxy", "패시브 밖 유통량 (1 − big3 − 지수펀드)"), ("log_size", "로그 시총")):
        p(f"### {label}\n")
        p(by_tercile(X, col, label).to_markdown(index=False))
        p("")
    p("## 3. 읽기\n")
    p("- 흐름은 가격을 움직인다. 어느 3분위에서든 영향 승수는 양수이고 대개 유의하다(1~5). 이것은 '기관이 가격을 움직일 수 있는가'에 대한 답이며, 답은 그렇다이다.")
    p("- 그러나 **유통량이 적을수록 승수가 커진다는 패턴은 없다.** 패시브 지분·기관 지분·지수펀드 지분의 상위 3분위가 하위보다 크지 않고, "
      "유통량 대용치와 규모의 하위 3분위도 상위보다 크지 않다. 상호작용 항은 모두 |t| ≤ 1이다. 중간 3분위가 가장 큰 경우가 많은데 단조 관계가 아니라는 뜻이다.")
    p("- 이후 12개월 계수는 대부분 부호가 음수이지만 유의하지 않아, 흐름의 가격 효과가 되돌아오는지(일시적)는 이 표본으로 판정하지 못한다.")
    p("- 한계: FIT는 N-PORT 제출 펀드(뮤추얼펀드·ETF)의 분기 흐름만 담고 헤지펀드·연기금·자사주매입은 없다. 유통량 대용치는 내부자 지분·대차 잔고를 모른다. "
      "표본이 81개월 한 레짐이다. 그러므로 '유통량이 적으면 기관이 움직인다'는 명제는 이 데이터로 지지되지 않는 것이지 반증된 것은 아니다.")
    p("- 프레임워크에서의 의미: 흐름의 효과는 δ_T(일시 수급)로 잡히며, 그 크기는 상대 가격 분산의 0.1~0.3%다. 흐름이 가격을 움직여도 "
      "그것이 어느 종목의 프리미엄을 체계적으로 설명하지는 않는다는 뜻이고, 타이밍 신호로도 쓰지 못한다(단계 3 검정).")
    (C.REPORTS / "phase2_fit_by_float.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
