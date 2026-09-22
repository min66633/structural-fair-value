# -*- coding: utf-8 -*-
"""Which channel does backlog work through? Multi-year revenue growth, earnings growth, ROE change."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sfv import config as C  # noqa: E402
from sfv.validate import fm  # noqa: E402

pd.set_option("display.width", 220)


def main():
    I = pd.read_parquet(C.PQ / "intangibles_quarterly.parquet",
                        columns=["cik", "end", "qend", "avail_date", "ttm_revenue", "ttm_e_adj", "roe_adj", "b_adj", "ttm_ok"])
    B = pd.read_parquet(C.PQ / "backlog_quarterly.parquet")
    P = I.merge(B[["cik", "end", "rpo", "rpo_to_rev", "rpo_g", "d_rpo_cov"]], on=["cik", "end"], how="left").sort_values(["cik", "end"])
    U = pd.read_parquet(C.PQ / "universe_monthly.parquet", columns=["cik", "month", "in_universe", "mktcap"])
    U["month"] = U["month"] + pd.offsets.MonthEnd(0)
    P["month"] = P["end"] + pd.offsets.MonthEnd(0)
    P = P.merge(U, on=["cik", "month"], how="left")
    P = P[P["in_universe"].fillna(False) & P["rpo"].notna() & P["ttm_ok"].fillna(False)]
    g = P.groupby("cik")
    P["log_rpo_cov"] = np.log(P["rpo_to_rev"]).where(P["rpo_to_rev"] > 0)
    P["log_size"] = np.log(P["mktcap"])
    P["rev_g_past"] = np.log(P["ttm_revenue"] / g["ttm_revenue"].shift(4)).where(
        (P["end"] - g["end"].shift(4)).dt.days.between(340, 390) & (P["ttm_revenue"] > 0) & (g["ttm_revenue"].shift(4) > 0)).clip(-2, 2)
    for k, nm in ((4, "rev_g_f1"), (8, "rev_g_f2"), (12, "rev_g_f3")):
        fwd = g["ttm_revenue"].shift(-k)
        ok = (g["end"].shift(-k) - P["end"]).dt.days.between(k * 85, k * 97) & (P["ttm_revenue"] > 0) & (fwd > 0)
        P[nm] = np.log(fwd / P["ttm_revenue"]).where(ok).clip(-2, 3)
    fwd_e = g["ttm_e_adj"].shift(-4)
    ok_e = (g["end"].shift(-4) - P["end"]).dt.days.between(340, 390) & (P["ttm_e_adj"] > 0) & (fwd_e > 0)
    P["earn_g_f1"] = np.log(fwd_e / P["ttm_e_adj"]).where(ok_e).clip(-2, 2)
    P["d_roe_f1"] = (g["roe_adj"].shift(-4) - P["roe_adj"]).where((g["end"].shift(-4) - P["end"]).dt.days.between(340, 390)).clip(-1, 1)
    P["qend"] = P["qend"]
    xs = ["rpo_g", "log_rpo_cov", "d_rpo_cov", "rev_g_past", "log_size"]
    rows = {}
    for y, lab, lags in (("rev_g_f1", "매출성장 1년", 4), ("rev_g_f2", "매출성장 2년(누적)", 8), ("rev_g_f3", "매출성장 3년(누적)", 12),
                         ("earn_g_f1", "조정이익 성장 1년 (흑자 기업)", 4), ("d_roe_f1", "조정 ROE 변화 1년", 4)):
        res, _ = fm(P.dropna(subset=[y]), y, xs, lags, min_n=100)
        for c in ["rpo_g", "log_rpo_cov", "d_rpo_cov"]:
            rows[f"{lab} ← {c}"] = {"coef": res.loc[c, "coef"], "t": res.loc[c, "t"], "quarters": int(res.loc[c, "quarters"])}
    T = pd.DataFrame(rows).T.round(3)
    print(T.to_string())
    # how much of revenue is locked in: coverage quintile vs realised 1-3y growth
    P["cov_q"] = pd.qcut(P["rpo_to_rev"], 5, labels=False)
    print(P.groupby("cov_q")[["rpo_to_rev", "rev_g_f1", "rev_g_f2", "rev_g_f3", "earn_g_f1", "d_roe_f1"]].median().round(3).to_string())
    T.to_csv(C.REPORTS / "phase6_channel.csv", encoding="utf-8-sig")


if __name__ == "__main__":
    main()
