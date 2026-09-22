# -*- coding: utf-8 -*-
"""Growth-stage residual income model: backlog enters through the scale of earnings.

Backlog (remaining performance obligations) predicts one- to three-year
revenue growth, not the level of ROE (scripts/phase6_channel.py). A
book-based RIM whose growth comes only from retained earnings cannot use
that information: an asset-light firm growing revenue 30% with 10% book
growth shows up as rising ROE, which the fade path pulls back down.

Two-stage path per firm-month (fundamentals as available at t):
    years 1-3   E_k = E_{k-1} (1 + g_k),   g from an expanding-window
                forecast of cumulative 1/2/3-year revenue growth on past
                growth, size, industry and (with --backlog) RPO growth,
                coverage and its change. Margins are held constant, so
                earnings grow with revenue. Book: clean surplus.
    years 4-10  the base fade: x_k = x_inf + omega (x_{k-1} - x_inf) from
                the year-3 excess ROE, then the base terminal.
Firms with non-positive trailing adjusted earnings keep the base value.

Point in time: the forecast model used for a quarter ending in year Y is the
one fitted on outcomes realised by the end of year Y-1.

Outputs data/parquet/rim_monthly_gr.parquet  (or _grbl with --backlog)
        data/parquet/growth_forecasts[_bl].parquet, reports/logs/growth_rim*.log
Run: python -m sfv.growth_rim [--backlog]
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pandas as pd

from . import config as C
from .rim import T_YEARS, OMEGA_MAX, X_INF_CAP, G_CAP, terminal

G_CLIP = (-0.5, 0.6)
MIN_FIT = 800
L = []


def build_growth_sample(say) -> pd.DataFrame:
    I = pd.read_parquet(C.PQ / "intangibles_quarterly.parquet",
                        columns=["cik", "end", "qend", "avail_date", "ttm_revenue", "ttm_e_adj", "ttm_ok"])
    I = I[I["ttm_ok"].fillna(False) & (I["ttm_revenue"] > 0)].sort_values(["cik", "end"])
    g = I.groupby("cik")
    lag_ok = (I["end"] - g["end"].shift(4)).dt.days.between(340, 390)
    I["rev_g_past"] = np.log(I["ttm_revenue"] / g["ttm_revenue"].shift(4)).where(lag_ok & (g["ttm_revenue"].shift(4) > 0)).clip(-2, 2)
    for k, nm in ((4, "f1"), (8, "f2"), (12, "f3")):
        fwd = g["ttm_revenue"].shift(-k)
        ok = (g["end"].shift(-k) - I["end"]).dt.days.between(k * 85, k * 97) & (fwd > 0)
        I[nm] = np.log(fwd / I["ttm_revenue"]).where(ok).clip(-2, 3)
        I[nm + "_end"] = g["end"].shift(-k)
    U = pd.read_parquet(C.PQ / "universe_monthly.parquet", columns=["cik", "month", "mktcap", "ff12", "in_universe"])
    U["month"] = U["month"] + pd.offsets.MonthEnd(0)
    I["month"] = I["end"] + pd.offsets.MonthEnd(0)
    I = I.merge(U, on=["cik", "month"], how="left")
    I = I[I["in_universe"].fillna(False)]
    I["log_size"] = np.log(I["mktcap"])
    B = pd.read_parquet(C.PQ / "backlog_quarterly.parquet", columns=["cik", "end", "rpo", "rpo_to_rev", "rpo_g", "d_rpo_cov"])
    I = I.merge(B, on=["cik", "end"], how="left")
    I["has_rpo"] = (I["rpo"] > 0).astype(float)
    I["log_rpo_cov"] = np.log(I["rpo_to_rev"]).where(I["rpo_to_rev"] > 0)
    for c in ("rpo_g", "log_rpo_cov", "d_rpo_cov"):
        I[c] = I[c].where(I["has_rpo"] > 0)
    I["year"] = I["end"].dt.year
    say(f"  growth sample: {len(I):,} firm-quarters, {I.cik.nunique():,} firms, with RPO {int(I.has_rpo.sum()):,}")
    return I.dropna(subset=["rev_g_past", "log_size", "ff12"])


def design(D: pd.DataFrame, ref: pd.DataFrame, backlog: bool) -> np.ndarray:
    cols = [np.ones(len(D)), D["rev_g_past"].values, ((D["log_size"] - ref["log_size"].mean()) / (ref["log_size"].std() + 1e-12)).values]
    for ind in sorted(ref["ff12"].dropna().unique())[1:]:
        cols.append((D["ff12"] == ind).astype(float).values)
    if backlog:
        cols.append(D["has_rpo"].values)
        rr = ref[ref["has_rpo"] > 0]
        for c in ("rpo_g", "log_rpo_cov", "d_rpo_cov"):
            mu, sd = rr[c].mean(), rr[c].std() + 1e-12
            cols.append((((D[c] - mu) / sd).fillna(0.0) * D["has_rpo"]).values)
    return np.column_stack(cols)


def forecasts(I: pd.DataFrame, backlog: bool, say) -> pd.DataFrame:
    """Expanding-window forecasts of cumulative 1/2/3-year revenue growth for every firm-quarter."""
    out = []
    years = sorted(I["year"].unique())
    for Y in years:
        model_year = Y - 1
        preds = {}
        for h, nm in ((1, "f1"), (2, "f2"), (3, "f3")):
            fit = I[(I[nm].notna()) & (I[nm + "_end"].dt.year <= model_year)]
            if backlog:
                fit_b = fit[fit["has_rpo"] > 0]
                use_b = len(fit_b) >= 300
            else:
                use_b = False
            if len(fit) < MIN_FIT:
                continue
            Xf = design(fit, fit, use_b)
            b = np.linalg.lstsq(Xf, fit[nm].values, rcond=None)[0]
            cur = I[I["year"] == Y]
            Xc = design(cur, fit, use_b)
            preds[nm] = pd.Series(Xc @ b, index=cur.index)
        if len(preds) < 3:
            continue
        cur = I[I["year"] == Y]
        P = pd.DataFrame({"cik": cur["cik"], "end": cur["end"], "g_hat1": preds["f1"], "g_hat2": preds["f2"], "g_hat3": preds["f3"]})
        out.append(P)
    F = pd.concat(out, ignore_index=True)
    # annual growth path from cumulative forecasts, clipped
    F["g1"] = F["g_hat1"].clip(*G_CLIP)
    F["g2"] = (F["g_hat2"] - F["g_hat1"]).clip(*G_CLIP)
    F["g3"] = (F["g_hat3"] - F["g_hat2"]).clip(*G_CLIP)
    say(f"  forecasts: {len(F):,} firm-quarters, years {F.end.dt.year.min()}-{F.end.dt.year.max()}; "
        f"g1 median {F.g1.median():.3f} p10 {F.g1.quantile(.1):.3f} p90 {F.g1.quantile(.9):.3f}")
    return F


def value_growth(B0, E0, g1, g2, g3, xinf, omega, roe_star, payout, r, T=T_YEARS):
    """Two-stage value: 3 years of earnings growth, then the fade."""
    omega = np.minimum(omega, OMEGA_MAX)
    B_prev = B0.copy()
    E = E0.copy()
    pv = np.zeros_like(B0)
    disc = np.ones_like(B0)
    for k, gk in enumerate((g1, g2, g3), start=1):
        E = E * (1.0 + gk)
        ri = E - r * B_prev
        disc = disc * (1.0 + r)
        pv = pv + ri / disc
        B_new = B_prev + E * (1.0 - payout)
        if k == 3:
            roe3 = E / B_prev
        B_prev = np.maximum(B_new, 1e-6)
    x = np.clip(roe3 - roe_star, -0.5, 0.5)
    ri_T = None
    for k in range(4, T + 1):
        x = xinf + omega * (x - xinf)
        roe = roe_star + x
        ri = (roe - r) * B_prev
        disc = disc * (1.0 + r)
        pv = pv + ri / disc
        g = np.clip(roe * (1.0 - payout), -0.5, G_CAP)
        if k == T:
            ri_T, g_T = ri, g
        B_prev = B_prev * (1.0 + g)
    tv = terminal(ri_T, disc, omega, g_T, r)
    return B0 + pv + tv, pv, tv


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backlog", action="store_true")
    a_ = ap.parse_args(argv)
    tag = "grbl" if a_.backlog else "gr"
    log = open(C.LOGS / f"growth_rim_{tag}.log", "w", encoding="utf-8")

    def say(msg):
        print(msg, flush=True); log.write(msg + "\n"); log.flush()

    t0 = time.time()
    I = build_growth_sample(say)
    F = forecasts(I, a_.backlog, say)
    F.to_parquet(C.PQ / f"growth_forecasts_{tag}.parquet", index=False)

    R = pd.read_parquet(C.PQ / "rim_monthly.parquet")
    X = R.merge(F[["cik", "end", "g1", "g2", "g3", "g_hat1"]].rename(columns={"end": "f_end"}), on=["cik", "f_end"], how="left")
    ok = X["g1"].notna() & (X["ttm_e_adj"] > 0)
    X["xinf"] = np.clip(X["a"] / (1.0 - np.minimum(X["omega"], OMEGA_MAX)), -X_INF_CAP, X_INF_CAP)
    d = X[ok]
    v, pv, tv = value_growth(d["b_adj"].to_numpy(float), d["ttm_e_adj"].to_numpy(float), d["g1"].to_numpy(float),
                             d["g2"].to_numpy(float), d["g3"].to_numpy(float), d["xinf"].to_numpy(float), d["omega"].to_numpy(float),
                             d["roe_star_ind"].to_numpy(float), d["payout"].to_numpy(float), d["r"].to_numpy(float))
    X["v_f_gr"] = X["v_f"]
    X.loc[ok, "v_f_gr"] = v
    X["gr_applied"] = ok
    X["log_pv_gr"] = np.log(X["mktcap"] / X["v_f_gr"]).where(X["v_f_gr"] > 0)
    keep = ["cik", "month", "ticker", "name", "ff12", "mktcap", "f_end", "b_adj", "ttm_e_adj", "roe_adj", "roe_star_ind", "omega", "a",
            "r", "payout", "g1", "g2", "g3", "g_hat1", "gr_applied", "v_f", "log_pv", "v_f_gr", "log_pv_gr"]
    X = X[keep]
    X.to_parquet(C.PQ / f"rim_monthly_{tag}.parquet", index=False)
    say(f"done: {len(X):,} rows, growth stage applied to {ok.mean()*100:.1f}%  {(time.time()-t0)/60:.1f} min")
    yr = X.assign(y=X["month"].dt.year).groupby("y").apply(lambda d: pd.Series({
        "log_pv_vw_base": np.log(d.mktcap.sum() / d.v_f.clip(lower=0).sum()),
        "log_pv_vw_gr": np.log(d.mktcap.sum() / d.v_f_gr.clip(lower=0).sum()),
        "med_base": d.log_pv.median(), "med_gr": d.log_pv_gr.median(), "g1_med": d.g1.median()}), include_groups=False)
    say(yr.round(3).to_string())
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
