# -*- coding: utf-8 -*-
"""Residual income value V_F and implied cost of capital, monthly, point in time.

    V_F,t = B_adj,t + sum_{k=1..T} (ROE_k - r) B_{k-1} / (1+r)^k + TV_T

Annual steps from the latest fundamentals available at month t:
    excess ROE      x_k = a_i + omega_i x_{k-1}          (estimated, sfv.persistence)
    ROE_k           = ROE*_ind + x_k
    book            B_k = B_{k-1} (1 + ROE_k (1 - payout))     clean surplus
    r_i             = r_f(t) + beta_i ERP(t)
    TV_T (base)     residual income keeps decaying at omega with book growth g:
                    RI_T q / (1 - q) / (1+r)^T,  q = omega (1+g) / (1+r)
    TV_T (cons.)    0 - no excess return after year T
    ICC             r that sets V_F(r) = market cap, solved on a grid

Availability rules (nothing after t is used):
    fundamentals   latest quarter with avail_date <= t
    omega, a       as-of year Y whose model could be fitted by 30 June Y+1
                   (all FY-Y reports filed); fallback = that year's median
    ROE*           industry (FF12) median ROE_adj of the latest as-of year
    r_f            10-year Treasury at month end (FRED DGS10)
    ERP            Damodaran implied ERP at the previous year end
    beta           60-month regression on the value-weighted universe return,
                   >= 24 months, clipped to [0.5, 1.5], else 1.0
    payout         (dividends + buybacks - issuance) / E_adj over the last 4
                   quarters, clipped to [0, 1]; 0 when E_adj <= 0

Outputs data/parquet/rim_monthly.parquet
Run: python -m sfv.rim
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

from . import config as C

T_YEARS = 10
BETA_CLIP = (0.5, 1.5)
R_GRID = np.linspace(0.00, 0.40, 81)          # 0.5% steps for the ICC solve
FUND_MAX_AGE = 400
# Economic bounds on the fade path. Without them a linear regression
# extrapolated to extreme characteristics and a zero-payout compounding
# book produce values of trillions for mid-caps (Amazon 2018: $8tn).
OMEGA_MAX = 0.90       # persistence cap
X_INF_CAP = 0.10       # long-run excess ROE a/(1-omega) bounded to +-10pp over the industry median
X0_CAP = 0.50          # initial excess ROE bounded to +-50pp
G_CAP = 0.25           # book growth per year in the explicit horizon
G_TV_CAP = 0.04        # book growth in the terminal phase (nominal GDP-like)
Q_MAX = 0.92           # terminal geometric factor cap (multiplier <= 11.5x RI_T)


def macro_inputs() -> tuple[pd.Series, pd.Series]:
    d = pd.read_csv(C.RAW / "macro" / "DGS10.csv", parse_dates=["observation_date"])
    d["DGS10"] = pd.to_numeric(d["DGS10"], errors="coerce")
    rf = d.set_index("observation_date")["DGS10"].dropna().resample("ME").last() / 100.0
    t = pd.read_html(C.RAW / "macro" / "histimpl.html")[0]
    t.columns = t.iloc[0]
    t = t.iloc[1:]
    erp = pd.Series(pd.to_numeric(t["Implied ERP (FCFE)"].astype(str).str.rstrip("%"), errors="coerce").values / 100.0,
                    index=pd.to_numeric(t["Year"], errors="coerce").values).dropna()
    erp.index = erp.index.astype(int)
    return rf, erp


def betas(U: pd.DataFrame, say) -> pd.DataFrame:
    """Rolling 60-month betas on the value-weighted universe return.

    Only in-universe rows enter: rows outside it can carry absurd market caps
    (unfixed share typos) that would otherwise dominate the weights.
    """
    P = U[U["in_universe"]][["cik", "month", "adj_close", "mktcap"]].dropna(subset=["adj_close"]).sort_values(["cik", "month"])
    P["ret"] = P.groupby("cik")["adj_close"].pct_change()
    gap = P.groupby("cik")["month"].diff().dt.days
    P.loc[(gap > 40) | (P["ret"].abs() > 3.0), "ret"] = np.nan
    P["w"] = P.groupby("cik")["mktcap"].shift(1)
    mk = P.dropna(subset=["ret", "w"]).groupby("month").apply(
        lambda d: np.average(d["ret"], weights=d["w"]), include_groups=False).rename("mkt")
    P = P.merge(mk, on="month", how="left")
    P = P.dropna(subset=["ret", "mkt"])
    rows = []
    for cik, d in P.groupby("cik", sort=False):
        r = d["ret"].to_numpy(); m = d["mkt"].to_numpy(); months = d["month"].to_numpy()
        n = len(d)
        if n < 24:
            continue
        # rolling OLS via cumulative sums (window 60)
        for i in range(23, n):
            lo = max(0, i - 59)
            rr, mm = r[lo:i + 1], m[lo:i + 1]
            if len(rr) < 24:
                continue
            mc = mm - mm.mean(); vc = rr - rr.mean()
            var = (mc @ mc)
            if var <= 0:
                continue
            rows.append((cik, months[i], (mc @ vc) / var, len(rr)))
    B = pd.DataFrame(rows, columns=["cik", "month", "beta_raw", "n_beta"])
    B["beta"] = B["beta_raw"].clip(*BETA_CLIP)
    say(f"  betas: {len(B):,} firm-months, median raw {B.beta_raw.median():.2f}, clipped share {((B.beta_raw < BETA_CLIP[0]) | (B.beta_raw > BETA_CLIP[1])).mean()*100:.1f}%")
    return B[["cik", "month", "beta", "n_beta"]]


def value_paths(B0, x0, a, omega, roe_star, payout, r, T=T_YEARS):
    """Vectorised residual-income PV over T annual steps for arrays of firms.

    Excess ROE follows x_k = x_inf + omega^k (x_0 - x_inf) with the
    steady state x_inf = a/(1-omega) bounded to +-X_INF_CAP; book grows at
    ROE_k (1 - payout) bounded to G_CAP. Returns pv (sum of discounted RI),
    RI_T, the discount factor at T, and g_T. r may be 2-d (firms x grid).
    """
    omega = np.minimum(omega, OMEGA_MAX)
    x_inf = np.clip(a / (1.0 - omega), -X_INF_CAP, X_INF_CAP)
    x = np.clip(x0, -X0_CAP, X0_CAP)
    B_prev = B0.copy()
    if r.ndim == 2:
        pv = np.zeros_like(r)
        disc = np.ones_like(r)
    else:
        pv = np.zeros_like(B0)
        disc = np.ones_like(B0)
    ri_T = None; g_T = None
    for k in range(1, T + 1):
        x = x_inf + omega * (x - x_inf)
        roe = roe_star + x
        if r.ndim == 2:
            ri = (roe[:, None] - r) * B_prev[:, None]
            disc = disc * (1.0 + r)
        else:
            ri = (roe - r) * B_prev
            disc = disc * (1.0 + r)
        pv = pv + ri / disc
        g = np.clip(roe * (1.0 - payout), -0.5, G_CAP)
        B_next = B_prev * (1.0 + g)
        if k == T:
            ri_T, g_T = ri, g
        B_prev = B_next
    return pv, ri_T, disc, g_T


def terminal(ri_T, disc_T, omega, g_T, r):
    """Continuing-decay terminal: RI_T * q/(1-q) / (1+r)^T, q = omega(1+g)/(1+r) <= Q_MAX."""
    omega = np.minimum(omega, OMEGA_MAX)
    g = np.minimum(g_T, G_TV_CAP)
    if r.ndim == 2:
        q = (omega * (1.0 + g))[:, None] / (1.0 + r)
    else:
        q = omega * (1.0 + g) / (1.0 + r)
    q = np.clip(q, 0.0, Q_MAX)
    return ri_T * q / (1.0 - q) / disc_T


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--roe-star", choices=["industry", "coe"], default="industry",
                    help="fade target: industry-year median ROE (default) or the firm's cost of equity with zero long-run excess")
    ap.add_argument("--out", default="rim_monthly.parquet")
    ap.add_argument("--omega-file", default="omega_firm_year.parquet",
                    help="firm-year (omega_hat, a_hat) table; e.g. the backlog-augmented one")
    ap.add_argument("--intangibles-file", default="intangibles_quarterly.parquet")
    ap.add_argument("--persistence-obs", default="persistence_obs.parquet",
                    help="sample the industry fade target is read from; must match --omega-file")
    ap.add_argument("--log-label", default="", help="suffix for the run log")
    a_ = ap.parse_args(argv)
    log = open(C.LOGS / f"rim_{a_.roe_star}{('_' + a_.log_label) if a_.log_label else ''}.log", "w", encoding="utf-8")

    def say(msg):
        print(msg, flush=True); log.write(msg + "\n"); log.flush()

    t0 = time.time()
    say(f"  fade target mode: {a_.roe_star}")
    U = pd.read_parquet(C.PQ / "universe_monthly.parquet")
    U["month"] = U["month"] + pd.offsets.MonthEnd(0)
    rf, erp = macro_inputs()
    say(f"  r_f months {rf.index.min().date()}..{rf.index.max().date()}; ERP years {erp.index.min()}..{erp.index.max()} (last {erp.iloc[-1]:.4f})")
    B = betas(U, say)

    # fundamentals: latest quarter available at t (intangibles table carries B_adj, E_adj, payout inputs)
    I = pd.read_parquet(C.PQ / a_.intangibles_file,
                        columns=["cik", "end", "avail_date", "b_adj", "b_gaap", "k_int", "ttm_e_adj", "ttm_e_gaap",
                                 "roe_adj", "ttm_dividends_paid", "ttm_buybacks", "ttm_equity_issuance", "ttm_ok"])
    I = I.dropna(subset=["avail_date"]).sort_values("avail_date")
    X = U[U["in_universe"]][["cik", "month", "mktcap", "ff12", "ticker", "name"]].sort_values("month")
    X = pd.merge_asof(X, I.rename(columns={"end": "f_end", "avail_date": "f_avail"}), left_on="month", right_on="f_avail",
                      by="cik", direction="backward")
    X = X[(X["month"] - X["f_end"]).dt.days <= FUND_MAX_AGE]
    X = X[(X["b_adj"] > 0) & X["ttm_ok"].fillna(False) & X["roe_adj"].notna()]

    # omega / a: as-of year Y usable from 30 June Y+1
    O = pd.read_parquet(C.PQ / a_.omega_file)
    O["usable_from"] = pd.to_datetime((O["asof_year"] + 1).astype(str) + "-06-30")
    O = O.sort_values("usable_from")
    X = pd.merge_asof(X.sort_values("month"), O[["cik", "usable_from", "asof_year", "omega_hat", "a_hat", "roe_star"]],
                      left_on="month", right_on="usable_from", by="cik", direction="backward")
    # fallback for firms without their own omega: the as-of year's cross-sectional median
    ymed = O.groupby("asof_year").agg(omega_med=("omega_hat", "median"), a_med=("a_hat", "median")).reset_index()
    ymed["usable_from"] = pd.to_datetime((ymed["asof_year"] + 1).astype(str) + "-06-30")
    X = pd.merge_asof(X.sort_values("month"), ymed[["usable_from", "omega_med", "a_med"]].sort_values("usable_from"),
                      left_on="month", right_on="usable_from", direction="backward", suffixes=("", "_y"))
    X["omega_src"] = np.where(X["omega_hat"].notna(), "firm", "year-median")
    X["omega"] = X["omega_hat"].where(X["omega_hat"].notna(), X["omega_med"])
    X["a"] = X["a_hat"].where(X["a_hat"].notna(), X["a_med"])
    # industry ROE* of the latest usable as-of year (from the persistence sample)
    Pobs = pd.read_parquet(C.PQ / a_.persistence_obs, columns=["year", "ff12", "roe_adj", "x_roe_adj"])
    Pobs["roe_star_ind"] = Pobs["roe_adj"] - Pobs["x_roe_adj"]
    rs = Pobs.groupby(["ff12", "year"])["roe_star_ind"].first().reset_index()
    rs["usable_from"] = pd.to_datetime((rs["year"] + 1).astype(str) + "-06-30")
    X = pd.merge_asof(X.sort_values("month"), rs[["ff12", "usable_from", "roe_star_ind"]].sort_values("usable_from"),
                      left_on="month", right_on="usable_from", by="ff12", direction="backward", suffixes=("", "_r"))
    X = X.dropna(subset=["omega", "a", "roe_star_ind"])

    # discount rate. Macro files are refreshed by hand, so a stale DGS10.csv or
    # histimpl.html would otherwise delete the newest months silently (dropna on
    # a NaN rate). Carry the last available value forward instead, flag it and
    # say so; the monthly runner checks file freshness separately.
    X = X.merge(B, on=["cik", "month"], how="left")
    X["beta"] = X["beta"].fillna(1.0)
    X["rf"] = X["month"].map(rf)
    X["erp"] = (X["month"].dt.year - 1).map(erp)
    rf_missing, erp_missing = X["rf"].isna(), X["erp"].isna()
    if rf_missing.any():
        say(f"  WARNING: r_f missing for {int(rf_missing.sum()):,} rows "
            f"({X.loc[rf_missing, 'month'].min().date()}..{X.loc[rf_missing, 'month'].max().date()}); "
            f"DGS10.csv ends {rf.index.max().date()} - carrying the last value ({rf.iloc[-1]:.4f}) forward")
        X.loc[rf_missing, "rf"] = float(rf.iloc[-1])
    if erp_missing.any():
        say(f"  WARNING: ERP missing for {int(erp_missing.sum()):,} rows "
            f"({X.loc[erp_missing, 'month'].min().date()}..{X.loc[erp_missing, 'month'].max().date()}); "
            f"histimpl.html ends {int(erp.index.max())} - carrying the last value ({erp.iloc[-1]:.4f}) forward")
        X.loc[erp_missing, "erp"] = float(erp.iloc[-1])
    X["macro_stale"] = (rf_missing | erp_missing).astype(bool)
    X = X.dropna(subset=["rf", "erp"])
    X["r"] = X["rf"] + X["beta"] * X["erp"]
    # payout
    pay = (X["ttm_dividends_paid"].fillna(0) + X["ttm_buybacks"].fillna(0) - X["ttm_equity_issuance"].fillna(0))
    X["payout"] = (pay / X["ttm_e_adj"]).where(X["ttm_e_adj"] > 0, 0.0).clip(0.0, 1.0)
    if a_.roe_star == "coe":
        # classical residual income assumption: excess return OVER THE COST OF
        # EQUITY fades at omega towards zero (competitive equilibrium). The
        # industry-median target is replaced by r and the long-run excess by 0.
        X["roe_star_ind"] = X["r"]
        X["a"] = 0.0
    X["x0"] = (X["roe_adj"].clip(-1, 1) - X["roe_star_ind"])
    say(f"  valuation rows: {len(X):,} firm-months, firms {X.cik.nunique():,}, {X.month.min().date()}..{X.month.max().date()}; "
        f"omega from firm {(X.omega_src == 'firm').mean()*100:.1f}%")

    # value at r
    B0 = X["b_adj"].to_numpy(float); x0 = X["x0"].to_numpy(float); a = X["a"].to_numpy(float)
    om = X["omega"].to_numpy(float); rs_ = X["roe_star_ind"].to_numpy(float); po = X["payout"].to_numpy(float)
    r = X["r"].to_numpy(float)
    pv, ri_T, disc_T, g_T = value_paths(B0, x0, a, om, rs_, po, r)
    tv = terminal(ri_T, disc_T, om, g_T, r)
    # the stored `omega` is the raw prediction (persistence caps it at 0.95); the
    # fade path applies OMEGA_MAX. Keep both so nothing downstream has to guess
    # which one it is holding.
    X["omega_applied"] = np.minimum(om, OMEGA_MAX)
    X["pv_ri"] = pv; X["tv"] = tv
    X["v_f"] = B0 + pv + tv
    X["v_f_notv"] = B0 + pv
    X["log_pv"] = np.log(X["mktcap"] / X["v_f"]).where(X["v_f"] > 0)
    X["log_pv_notv"] = np.log(X["mktcap"] / X["v_f_notv"]).where(X["v_f_notv"] > 0)
    X["log_pb_adj"] = np.log(X["mktcap"] / X["b_adj"])

    # ICC on a grid: V(r) is monotone decreasing in r; interpolate the crossing.
    # Done in row blocks: rows x grid x several float64 arrays would otherwise
    # take over 1 GB at once on this machine.
    M_all = X["mktcap"].to_numpy(float)
    icc = np.full(len(X), np.nan)
    BLOCK = 50_000
    for s in range(0, len(X), BLOCK):
        e = min(s + BLOCK, len(X))
        Rg = np.broadcast_to(R_GRID, (e - s, len(R_GRID)))
        pvg, riTg, discTg, gTg = value_paths(B0[s:e], x0[s:e], a[s:e], om[s:e], rs_[s:e], po[s:e], Rg)
        tvg = terminal(riTg, discTg, om[s:e], gTg, Rg)
        diff = (B0[s:e, None] + pvg + tvg) - M_all[s:e, None]
        sign_change = (diff[:, :-1] > 0) & (diff[:, 1:] <= 0)
        has = sign_change.any(axis=1)
        j = np.argmax(sign_change, axis=1)
        idx = np.where(has)[0]
        d0 = diff[idx, j[idx]]; d1 = diff[idx, j[idx] + 1]
        blk = np.full(e - s, np.nan)
        blk[idx] = R_GRID[j[idx]] + (R_GRID[j[idx] + 1] - R_GRID[j[idx]]) * d0 / (d0 - d1)
        blk[~has & (diff[:, 0] <= 0)] = R_GRID[0]        # value below market cap even at 2%
        blk[~has & (diff[:, -1] > 0)] = R_GRID[-1]       # value above market cap even at 40%
        icc[s:e] = blk
    X["icc"] = icc
    X["icc_excess"] = X["icc"] - X["rf"]

    keep = ["cik", "month", "ticker", "name", "ff12", "mktcap", "f_end", "f_avail", "b_gaap", "b_adj", "k_int",
            "ttm_e_gaap", "ttm_e_adj", "roe_adj", "roe_star_ind", "x0", "omega", "omega_applied", "a", "omega_src", "asof_year",
            "beta", "rf", "erp", "r", "macro_stale", "payout", "pv_ri", "tv", "v_f", "v_f_notv", "log_pv", "log_pv_notv",
            "log_pb_adj", "icc", "icc_excess"]
    X = X[keep].sort_values(["cik", "month"]).reset_index(drop=True)
    X.to_parquet(C.PQ / a_.out, index=False)
    say(f"done: {len(X):,} rows  {(time.time()-t0)/60:.1f} min")
    def _vw(d, col):
        ok = d[col] > 0
        return np.log(d.loc[ok, "mktcap"].sum() / d.loc[ok, col].sum())

    yr = X.assign(y=X["month"].dt.year).groupby("y").apply(
        lambda d: pd.Series({"firms": d.cik.nunique(), "log_pv_med": d.log_pv.median(),
                             "log_pv_p10": d.log_pv.quantile(.1), "log_pv_p90": d.log_pv.quantile(.9),
                             "log_pv_vw": _vw(d, "v_f"), "log_pv_notv_vw": _vw(d, "v_f_notv"),
                             "v_le_0": (d.v_f <= 0).mean(), "icc_med": d.icc.median(), "r_med": d.r.median(),
                             "omega_med": d.omega.median()}),
        include_groups=False)
    say("by year (vw = value-weighted aggregate, log(sum P / sum V)):\n" + yr.round(3).to_string())
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
