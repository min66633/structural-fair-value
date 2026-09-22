# -*- coding: utf-8 -*-
"""Did the model's end-2025 verdicts anticipate 2026 year-to-date returns, or lag them?

Signals as of 2025-12-31 (all point in time): log(P/V_F) base, growth+backlog
version, residual net of backlog (RPO firms). Returns: yfinance adjusted
close from 2025-12-31 to the latest month. Lag test: change in log V_F over
2025 versus 2025 returns (contemporaneous) and 2026 returns (leading).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sfv import config as C  # noqa: E402

pd.set_option("display.width", 220)
import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--t0", default="2025-12-31")
_ap.add_argument("--t1", default="")
_a = _ap.parse_args()
T0 = pd.Timestamp(_a.t0)
T1 = pd.Timestamp(_a.t1) if _a.t1 else None


def quintile_table(d: pd.DataFrame, score: str, ret: str, label: str):
    d = d.dropna(subset=[score, ret, "mktcap"]).copy()
    d["q"] = pd.qcut(d[score].rank(method="first"), 5, labels=[f"Q{i}" for i in range(1, 6)])
    r = np.expm1(d[ret])
    ew = r.groupby(d["q"], observed=True).mean()
    vw = r.groupby(d["q"], observed=True).apply(lambda s: np.average(s, weights=d.loc[s.index, "mktcap"]))
    n = d.groupby("q", observed=True).size()
    med = d.groupby("q", observed=True)[score].median()
    t = pd.DataFrame({"n": n, f"{score} 중앙값": med, "동일가중 수익률": ew, "시총가중 수익률": vw})
    print(f"\n== {label}: Q1 = 모형상 가장 저평가 (score 최저) … Q5 = 가장 고평가")
    print(t.round(3).to_string())
    # rank correlation and slope
    from scipy.stats import spearmanr
    rho, pv = spearmanr(d[score], d[ret])
    X = np.column_stack([np.ones(len(d)), d[score].values])
    b = np.linalg.lstsq(X, d[ret].values, rcond=None)[0][1]
    print(f"   Spearman rho {rho:.3f} (p {pv:.3f}), slope {b:.3f}, n {len(d)}   [음의 rho = 모형이 맞는 방향]")


def main():
    U = pd.read_parquet(C.PQ / "universe_monthly.parquet", columns=["cik", "month", "adj_close", "mktcap", "in_universe", "ticker", "name"])
    U["month"] = U["month"] + pd.offsets.MonthEnd(0)
    last = T1 if T1 is not None else U[U["adj_close"].notna()]["month"].max()
    p0 = U[(U["month"] == T0) & U["in_universe"]][["cik", "adj_close", "mktcap", "ticker", "name"]].rename(columns={"adj_close": "ac0"})
    p1 = U[U["month"] == last][["cik", "adj_close"]].rename(columns={"adj_close": "ac1"})
    pm = U[U["month"] == T0 - pd.DateOffset(years=1) + pd.offsets.MonthEnd(0)][["cik", "adj_close"]].rename(columns={"adj_close": "ac_m1"})
    R = p0.merge(p1, on="cik", how="left").merge(pm, on="cik", how="left")
    R["ret_2026"] = np.log(R["ac1"] / R["ac0"])
    R["ret_2025"] = np.log(R["ac0"] / R["ac_m1"])
    print(f"2026 YTD window: {T0.date()} -> {last.date()}; firms with return {R.ret_2026.notna().sum():,}")
    mkt = np.average(np.expm1(R["ret_2026"].dropna()), weights=R.loc[R["ret_2026"].notna(), "mktcap"])
    print(f"universe value-weighted return {mkt*100:.1f}%, equal-weighted {np.expm1(R.ret_2026.dropna()).mean()*100:.1f}%")

    base = pd.read_parquet(C.PQ / "rim_monthly.parquet", columns=["cik", "month", "log_pv", "v_f", "icc"])
    grbl = pd.read_parquet(C.PQ / "rim_monthly_grbl.parquet", columns=["cik", "month", "log_pv_gr", "v_f_gr"])
    b0 = base[base["month"] == T0][["cik", "log_pv", "v_f", "icc"]]
    g0 = grbl[grbl["month"] == T0][["cik", "log_pv_gr"]]
    D = R.merge(b0, on="cik", how="left").merge(g0, on="cik", how="left")
    quintile_table(D, "log_pv", "ret_2026", "기존 V_F, 2025-12-31 판정 → 2026 YTD 수익률")
    quintile_table(D, "log_pv_gr", "ret_2026", "성장+백로그 V_F, 2025-12-31 판정 → 2026 YTD 수익률")
    # residual net of backlog, RPO firms (validation panel + backlog panel at 2025-12-31)
    try:
        BP = pd.read_parquet(C.PQ / "backlog_panel.parquet", columns=["cik", "qend", "eps_res", "eps_A"])
        e0 = BP[BP["qend"] == T0][["cik", "eps_res", "eps_A"]]
        E = D.merge(e0, on="cik", how="inner")
        quintile_table(E, "eps_res", "ret_2026", "RPO 기업: 백로그 제거 잔차 ε_res, 2025-12-31 → 2026 YTD")
    except Exception as ex:
        print("backlog panel not available:", ex)

    # lag test: change in log V_F during 2025 vs returns in 2025 (contemporaneous) and 2026 (leading)
    bm1 = base[base["month"] == T0 - pd.DateOffset(years=1) + pd.offsets.MonthEnd(0)][["cik", "v_f"]].rename(columns={"v_f": "v_f_m1"})
    L = D.merge(bm1, on="cik", how="inner")
    L["dlogV_2025"] = np.log(L["v_f"] / L["v_f_m1"]).where((L["v_f"] > 0) & (L["v_f_m1"] > 0)).clip(-2, 2)
    L = L.dropna(subset=["dlogV_2025", "ret_2025", "ret_2026"])
    from scipy.stats import spearmanr
    r_c, _ = spearmanr(L["dlogV_2025"], L["ret_2025"])
    r_l, _ = spearmanr(L["dlogV_2025"], L["ret_2026"])
    print(f"\n== 지연 검사 (n {len(L):,}): 2025년 중 log V_F 변화 vs 2025년 수익률(동행) Spearman {r_c:.3f} | vs 2026 YTD 수익률(선행) {r_l:.3f}")
    # where were 2026's winners in the model at end-2025?
    D2 = D.dropna(subset=["ret_2026", "log_pv"]).copy()
    D2["ret_dec"] = pd.qcut(D2["ret_2026"].rank(method="first"), 10, labels=False)
    D2["pv_half"] = np.where(D2["log_pv"] <= D2["log_pv"].median(), "저평가 절반", "고평가 절반")
    top = D2[D2["ret_dec"] == 9]
    print(f"\n2026 YTD 수익률 상위 10% ({len(top)}개) 중 2025년 말 모형상 저평가 절반에 있던 비율: {(top.pv_half == '저평가 절반').mean()*100:.1f}%")
    bot = D2[D2["ret_dec"] == 0]
    print(f"2026 YTD 수익률 하위 10% ({len(bot)}개) 중 저평가 절반 비율: {(bot.pv_half == '저평가 절반').mean()*100:.1f}%")
    # mega caps: verdict at end-2025 vs 2026 return
    mega = D2.sort_values("mktcap", ascending=False).head(12)[["ticker", "mktcap", "log_pv", "log_pv_gr", "ret_2026"]]
    mega["mktcap"] = mega["mktcap"] / 1e9
    mega["ret_2026"] = np.expm1(mega["ret_2026"])
    print("\n시총 상위 12 (2025-12-31 판정, 2026 YTD 단순수익률):")
    print(mega.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
