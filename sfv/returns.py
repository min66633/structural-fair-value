# -*- coding: utf-8 -*-
"""Survivorship-free quarterly returns from 13F implied prices.

Every 13F line carries value and shares, so the cross-filer median implied
price (sfv.f13, px_med) is a quarter-end price for every stock that at
least a few institutions held - including names that were later delisted,
which a price feed of currently listed tickers cannot provide.

    ret_px_q  = log( px_med(q+1) * split_factor / px_med(q) )
    ret_tot_q = ret_px_q + dividends_ttm / 4 / mktcap(q)     (approximation)

Split factor: yfinance split events for the primary ticker between the two
quarter ends; for names with no price feed (delisted), a price ratio within
5% of 1/k or k (k in the usual split ratios) together with a matching jump
in total institutional shares is treated as a split.

Exits: a firm whose 13F price series ends before the sample end and whose
ticker has no listed price six months later is flagged (exit_flag) at its
last observation. Validation code assigns the delisting-return sensitivity.

Reliability: only quarters with at least 3 filers (median of 3+ prices).

Output data/parquet/returns_quarterly.parquet
Run: python -m sfv.returns
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

from . import config as C

MIN_HOLDERS = 3
SPLIT_K = [1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 10, 15, 20, 25, 30, 40, 50, 100]
PX_TOL = np.log(1.05)
SH_TOL = np.log(1.15)
MAX_DY_Q = 0.05


def main() -> int:
    log = open(C.LOGS / "returns.log", "w", encoding="utf-8")

    def say(msg):
        print(msg, flush=True); log.write(msg + "\n"); log.flush()

    t0 = time.time()
    A = pd.read_parquet(C.PQ / "inst_own_quarterly.parquet", columns=["period", "cusip", "px_med", "n_holders", "inst_shares"])
    U = pd.read_parquet(C.PQ / "universe_monthly.parquet",
                        columns=["cik", "month", "cusip", "ticker", "mktcap", "price_src", "adj_close", "in_universe"])
    U["month"] = U["month"] + pd.offsets.MonthEnd(0)
    prim = U.sort_values("month").groupby("cik").agg(cusip=("cusip", "last"), ticker=("ticker", "last")).reset_index()
    A = A.merge(prim, on="cusip", how="inner")
    A = A[A["n_holders"] >= MIN_HOLDERS].sort_values(["cik", "period"])
    say(f"  13F price cells with >= {MIN_HOLDERS} holders: {len(A):,}  ciks {A.cik.nunique():,}  periods {A.period.min().date()}..{A.period.max().date()}")

    g = A.groupby("cik")
    A["period1"] = g["period"].shift(-1)
    A["px1"] = g["px_med"].shift(-1)
    A["sh1"] = g["inst_shares"].shift(-1)
    A["nh1"] = g["n_holders"].shift(-1)
    gap = (A["period1"] - A["period"]).dt.days
    A = A[gap.between(80, 100)].copy()
    # month keys as columns (positional .values after a later sort would misalign)
    A["m0"] = A["period"] + pd.offsets.MonthEnd(0)
    A["m1"] = A["period1"] + pd.offsets.MonthEnd(0)

    # yfinance split events between the two quarter ends, for listed tickers
    P = pd.read_parquet(C.PQ / "prices_monthly.parquet", columns=["ticker", "month", "splits"])
    P["month"] = pd.to_datetime(P["month"]) + pd.offsets.MonthEnd(0)
    P = P.sort_values(["ticker", "month"])
    P["cum"] = np.log(P["splits"].where(P["splits"] > 0, 1.0)).groupby(P["ticker"]).cumsum()
    cs = P.set_index(["ticker", "month"])["cum"]
    c0 = cs.reindex(pd.MultiIndex.from_arrays([A["ticker"], A["m0"]])).values
    c1 = cs.reindex(pd.MultiIndex.from_arrays([A["ticker"], A["m1"]])).values
    yf_split = c1 - c0                                   # log of the shares multiplier
    A["split_src"] = np.where(np.isfinite(yf_split), "yf", "none")
    lsplit = np.where(np.isfinite(yf_split), yf_split, 0.0)
    # share-count heuristic for names without a price feed
    lr = np.log(A["px1"] / A["px_med"])
    lsh = np.log(A["sh1"] / A["inst_shares"])
    no_feed = ~np.isfinite(yf_split)
    for k in SPLIT_K:
        lk = np.log(k)
        fwd = no_feed & (np.abs(lr + lk) < PX_TOL) & (np.abs(lsh - lk) < SH_TOL)
        rev = no_feed & (np.abs(lr - lk) < PX_TOL) & (np.abs(lsh + lk) < SH_TOL)
        lsplit = np.where(fwd, lk, lsplit)
        lsplit = np.where(rev, -lk, lsplit)
        A.loc[fwd | rev, "split_src"] = "shares"
    A["log_split"] = lsplit
    A["ret_px"] = (np.log(A["px1"] / A["px_med"]) + A["log_split"]).clip(-3, 3)

    # dividend yield approximation from the latest available trailing dividends
    D = pd.read_parquet(C.PQ / "intangibles_quarterly.parquet", columns=["cik", "avail_date", "ttm_dividends_paid"])
    D = D.dropna(subset=["avail_date"]).sort_values("avail_date")
    A = A.sort_values("period")
    A = pd.merge_asof(A, D.rename(columns={"avail_date": "d_avail"}), left_on="period", right_on="d_avail", by="cik", direction="backward")
    A = A.merge(U[["cik", "month", "mktcap"]].rename(columns={"month": "m0"}), on=["cik", "m0"], how="left")
    A["dy_q"] = (A["ttm_dividends_paid"].fillna(0).clip(lower=0) / 4.0 / A["mktcap"]).clip(0, MAX_DY_Q).fillna(0.0)
    A["ret_tot"] = A["ret_px"] + A["dy_q"]

    # exits: last 13F observation with no listed price six months later
    last = A.groupby("cik")["period1"].max().rename("last_p1").reset_index()
    A = A.merge(last, on="cik", how="left")
    yf_last = U[U["price_src"] == "yf"].groupby("cik")["month"].max().rename("yf_last").reset_index()
    A = A.merge(yf_last, on="cik", how="left")
    sample_end = A["period1"].max()
    A["exit_flag"] = (A["period1"] == A["last_p1"]) & (A["last_p1"] < sample_end - pd.Timedelta(days=100)) \
        & (A["yf_last"].isna() | (A["yf_last"] < A["last_p1"] + pd.Timedelta(days=180)))

    # cross-check against yfinance adjusted-close quarterly returns where both exist
    Uy = U[U["adj_close"].notna()][["cik", "month", "adj_close"]]
    A = A.merge(Uy.rename(columns={"month": "m0", "adj_close": "ac0"}), on=["cik", "m0"], how="left")
    A = A.merge(Uy.rename(columns={"month": "m1", "adj_close": "ac1"}), on=["cik", "m1"], how="left")
    A["ret_yf"] = np.log(A["ac1"] / A["ac0"])
    both = A.dropna(subset=["ret_yf"])
    corr = np.corrcoef(both["ret_tot"], both["ret_yf"])[0, 1]
    diff = (both["ret_tot"] - both["ret_yf"])
    say(f"  overlap with yfinance: {len(both):,} obs, corr {corr:.3f}, median |diff| {diff.abs().median():.4f}, "
        f"|diff|>0.10 share {(diff.abs() > 0.10).mean()*100:.2f}%")
    bad = both[diff.abs() > 0.3].sort_values("m0")
    say(f"  large disagreements (>0.30): {len(bad):,} ({len(bad)/len(both)*100:.2f}%) - splits/data errors; sample:")
    say(bad.head(8)[["cik", "ticker", "period", "period1", "px_med", "px1", "log_split", "split_src", "ret_tot", "ret_yf"]].round(3).to_string(index=False))

    out = A[["cik", "cusip", "ticker", "period", "period1", "px_med", "px1", "n_holders", "nh1", "log_split", "split_src",
             "ret_px", "dy_q", "ret_tot", "ret_yf", "exit_flag"]].rename(columns={"period": "qend", "period1": "qend_next"})
    out = out.sort_values(["cik", "qend"]).reset_index(drop=True)
    out.to_parquet(C.PQ / "returns_quarterly.parquet", index=False)
    say(f"done: {len(out):,} cik-quarters, ciks {out.cik.nunique():,}, exits flagged {int(out.exit_flag.sum()):,}, "
        f"splits adjusted {int((out.log_split != 0).sum()):,} (yf {int(((out.log_split != 0) & (out.split_src == 'yf')).sum()):,})  {(time.time()-t0)/60:.1f} min")
    vw = out.merge(U[["cik", "month", "mktcap", "in_universe"]].rename(columns={"month": "qend"}), on=["cik", "qend"], how="left")
    vw = vw[vw["in_universe"].fillna(False)]
    agg = vw.groupby("qend").apply(lambda d: pd.Series({"n": len(d), "vw_ret": np.average(np.expm1(d.ret_tot), weights=d.mktcap),
                                                        "eq_ret": np.expm1(d.ret_tot).mean()}), include_groups=False)
    say("universe value-weighted quarterly return (13F prices + dividend approx):\n" + agg.iloc[::4].round(4).to_string())
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
