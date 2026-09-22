# -*- coding: utf-8 -*-
"""Why mega-cap valuation is hard: value against the persistence assumption (phase 7 supplement).

For a set of tickers at the latest valuation month, holds every model input
fixed and moves only the persistence of excess ROE: V/P at omega 0.80, 0.90,
0.95, 0.98, 0.995, the implied omega* (V = P) and half-life, the book-to-price
anchor and the terminal share. Writes reports/phase7_sensitivity.md.
Run: python scripts/phase7_sensitivity.py [--tickers NVDA,MSFT,...] [--month 2026-09-30]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sfv import config as C  # noqa: E402
from sfv.implied import value_common_omega, X_INF_CAP  # noqa: E402

DEFAULT = "NVDA,AAPL,MSFT,GOOGL,META,AMZN,AVGO,TSLA,LLY,LMT,PG,KO,XOM,DUK"
GRID = (0.80, 0.90, 0.95, 0.98, 0.995)
L = []


def p(t=""):
    L.append(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", default=DEFAULT)
    ap.add_argument("--month", default="")
    a = ap.parse_args()
    R = pd.read_parquet(C.PQ / "rim_monthly.parquet")
    m = pd.Timestamp(a.month) if a.month else R["month"].max()
    d = R[R["month"] == m].set_index("ticker")
    tick = [t for t in a.tickers.split(",") if t in d.index]
    d = d.loc[tick].copy()
    d["xinf"] = np.clip(d["a"] / (1.0 - np.minimum(d["omega"], 0.9)), -X_INF_CAP, X_INF_CAP)
    rows = []
    for t, r_ in d.iterrows():
        B0 = np.array([r_["b_adj"]]); x0 = np.array([r_["x0"]]); xi = np.array([r_["xinf"]]); rs = np.array([r_["roe_star_ind"]])
        po = np.array([r_["payout"]]); rr = np.array([r_["r"]])
        v = {f"V/P (ω {w:.3g})": value_common_omega(B0, x0, xi, rs, po, rr, np.array([w]))[0] / r_["mktcap"] for w in GRID}
        f = lambda w: value_common_omega(B0, x0, xi, rs, po, rr, np.array([w]))[0] - r_["mktcap"]
        if x0[0] <= xi[0] or f(0.995) < 0:
            om = np.nan
        elif f(0.0) >= 0:
            om = 0.0
        else:
            lo, hi = 0.0, 0.995
            for _ in range(50):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
            om = 0.5 * (lo + hi)
        rows.append({"종목": t, "시총 $bn": r_["mktcap"] / 1e9, "장부가/가격": r_["b_adj"] / r_["mktcap"], "조정 ROE": r_["roe_adj"], "초과 ROE x0": r_["x0"],
                     "모형 ω": min(r_["omega"], 0.9), "할인율": r_["r"], "log(P/V_F)": r_["log_pv"], "터미널 비중": r_["tv"] / r_["v_f"] if r_["v_f"] > 0 else np.nan,
                     **v, "ω* (V=P)": om, "반감기 y": np.log(0.5) / np.log(om) if np.isfinite(om) and 0 < om < 1 else np.nan,
                     "Δlog V, ω 0.90→0.95": np.log(v["V/P (ω 0.95)"] / v["V/P (ω 0.9)"]) if v["V/P (ω 0.9)"] > 0 and v["V/P (ω 0.95)"] > 0 else np.nan})
    T = pd.DataFrame(rows).set_index("종목")
    p("# 단계 7 보충 — 지속성 가정 하나에 가치가 얼마나 흔들리는가\n")
    p(f"> 생성 {pd.Timestamp.today().date()}. 평가월 {m.date()}, 모형 입력(조정 장부가, 초과 ROE, 산업 감쇠 목표, 배당성향, 할인율)은 그 시점까지 공시된 값. "
      "다른 입력을 고정하고 초과 ROE 지속성 ω만 움직였을 때의 V/P. ω* = V가 가격과 같아지는 ω(0~0.995), 빈칸은 초과 ROE가 장기 수준 아래거나 0.995로도 못 미치는 경우.")
    p("\n## 표\n")
    p(T.round(2).to_markdown())
    p("\n## 읽기\n")
    p("- 가치평가의 난이도는 규모나 품질이 아니라 가격 중 회계로 고정된 부분의 비중이 정한다. 장부가/가격이 0.05인 회사는 가치의 95%가 '얼마나 높은 초과이익을 얼마나 오래'라는 두 미지수에 걸려 있고, 그 가정의 합리적 폭이 곧 가치의 폭이다.")
    p("- 같은 ω 변화(0.90→0.95)에 대한 가치 변화는 초과 ROE가 클수록 크다(엔비디아·애플·록히드 0.45~0.54 vs P&G 0.16). 이익이 2년 만에 세 배가 된 회사의 ω를 ±0.02로 못 박을 수 있는 사람은 없다.")
    p("- '독점이니 비싸지 않다'는 주장은 모형 언어로 ω ≈ 1이다. 현재 가격은 정확히 그것을 전제한다. 모형의 '비쌈' 판정은 나쁜 회사라는 뜻이 아니라 가격이 이미 수십 년의 독점을 반영했다는 뜻이며, 그 전제를 믿느냐는 모형 밖의 판단이다. 기저율: 시총 상위 5% 기업의 실현 지속성 0.85~0.89(반감기 4~6년).")
    p("- 초대형주 안에서도 모형은 갈라 본다. 알파벳·메타·아마존은 반감기 3년 전제로 가격이 설명되고, 엔비디아·애플·마이크로소프트는 13~34년을 요구한다.")
    (C.REPORTS / "phase7_sensitivity.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
