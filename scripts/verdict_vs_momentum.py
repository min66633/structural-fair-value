# -*- coding: utf-8 -*-
"""Is the reversed verdict in 2025 a momentum effect, or is 'expensive' a proxy for growth/quality?

At T0: log(P/V_F) (y), its decomposition components (delta_F = growth/quality controls,
eps = residual), 12-1 month momentum, sales growth, ROE. Returns T0 -> T1.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sfv import config as C  # noqa: E402

pd.set_option("display.width", 220)
ap = argparse.ArgumentParser()
ap.add_argument("--t0", default="2024-12-31")
ap.add_argument("--t1", default="2025-12-31")
ap.add_argument("--decomp-file", default="decomp_monthly.parquet", help="e.g. decomp_monthly_mom.parquet (residual net of momentum)")
a = ap.parse_args()
T0, T1 = pd.Timestamp(a.t0), pd.Timestamp(a.t1)


def qtab(d, score, ret, label):
    d = d.dropna(subset=[score, ret]).copy()
    d["q"] = pd.qcut(d[score].rank(method="first"), 5, labels=[f"Q{i}" for i in range(1, 6)])
    r = np.expm1(d[ret])
    ew = r.groupby(d["q"], observed=True).mean()
    rho, pv = spearmanr(d[score], d[ret])
    print(f"  {label:34s} Q1 {ew.iloc[0]*100:6.1f}%  Q2 {ew.iloc[1]*100:6.1f}%  Q3 {ew.iloc[2]*100:6.1f}%  Q4 {ew.iloc[3]*100:6.1f}%  Q5 {ew.iloc[4]*100:6.1f}%   rho {rho:+.3f} (p {pv:.3f})")


def main():
    U = pd.read_parquet(C.PQ / "universe_monthly.parquet", columns=["cik", "month", "adj_close", "mktcap", "in_universe"])
    U["month"] = U["month"] + pd.offsets.MonthEnd(0)
    px = U.pivot_table(index="cik", columns="month", values="adj_close")
    m12 = T0 - pd.DateOffset(months=12) + pd.offsets.MonthEnd(0)
    m1 = T0 - pd.DateOffset(months=1) + pd.offsets.MonthEnd(0)
    R = pd.DataFrame({"ret": np.log(px[T1] / px[T0]), "mom": np.log(px[m1] / px[m12])})
    R = R.join(U[(U["month"] == T0) & U["in_universe"]].set_index("cik")[["mktcap"]], how="inner")
    D = pd.read_parquet(C.PQ / a.decomp_file)
    has_M = "delta_M_A" in D.columns
    keep = ["y", "delta_S_A", "delta_T_A", "delta_F_A", "eps_A", "sales_g", "roe_adj_w", "rnd_int", "loss", "log_size"] + (["delta_M_A", "eps_noM_A"] if has_M else [])
    D = D[D["month"] == T0].set_index("cik")[keep]
    X = R.join(D, how="inner").dropna(subset=["ret", "y", "mom"])
    print(f"window {T0.date()} -> {T1.date()}, firms {len(X):,}, decomposition {a.decomp_file}" + (" (eps net of momentum)" if has_M else ""))
    print("\n5분위 동일가중 수익률 (Q1 = score 최저):")
    qtab(X, "y", "ret", "log(P/V_F) (Q5 = 가장 비쌈)")
    qtab(X, "eps_A", "ret", "잔차 ε (특성·산업 제거)")
    qtab(X, "delta_F_A", "ret", "δ_F 성장·퀄리티 성분")
    qtab(X, "mom", "ret", "모멘텀 12-1 (Q5 = 가장 강함)")
    qtab(X, "sales_g", "ret", "매출성장")
    qtab(X, "roe_adj_w", "ret", "조정 ROE")
    # what is 'expensive' made of at T0?
    print("\nlog(P/V_F)와 T0 특성의 순위상관:")
    def rho_pair(u, v):
        d = X[[u, v]].dropna()
        return spearmanr(d[u], d[v])[0] if len(d) > 10 else np.nan

    for c, nm in (("mom", "모멘텀"), ("sales_g", "매출성장"), ("roe_adj_w", "조정 ROE"), ("rnd_int", "R&D 집약도"), ("log_size", "규모"), ("loss", "적자")):
        print(f"  {nm:10s} {rho_pair('y', c):+.3f}")
    print(f"  잔차 ε ~ 모멘텀 순위상관 {rho_pair('eps_A', 'mom'):+.3f}" + (f"; 모멘텀 통제 전 잔차(eps_noM) ~ 모멘텀 {rho_pair('eps_noM_A', 'mom'):+.3f}" if has_M else ""))
    # regressions: ret on y alone, y + mom, y + mom + characteristics, and eps + components
    def ols(cols):
        d = X.dropna(subset=cols)
        Xm = np.column_stack([np.ones(len(d))] + [d[c].values for c in cols])
        b, res, *_ = np.linalg.lstsq(Xm, d["ret"].values, rcond=None)
        e = d["ret"].values - Xm @ b
        s2 = e @ e / (len(d) - len(cols) - 1)
        cov = s2 * np.linalg.inv(Xm.T @ Xm)
        t = b / np.sqrt(np.diag(cov))
        return pd.DataFrame({"coef": b[1:], "t": t[1:]}, index=cols).round(3)
    print("\n횡단면 회귀 (종속: 해당 기간 로그수익률):")
    comps = ["eps_A", "delta_F_A", "delta_S_A", "delta_T_A"] + (["delta_M_A"] if has_M else [])
    specs = [["y"], ["y", "mom"], ["y", "mom", "sales_g", "roe_adj_w", "log_size", "loss"], comps, comps + ["mom"]]
    for cols in specs:
        print("  " + " + ".join(cols)); print("  " + ols(cols).to_string().replace("\n", "\n  "))


if __name__ == "__main__":
    main()
