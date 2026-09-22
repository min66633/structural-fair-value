# -*- coding: utf-8 -*-
"""Market-implied expectations per firm-month (reverse residual-income model).

The cross-sectional residual says whether a firm is dear relative to peers;
it does not say what the price assumes. For a firm whose book value is a few
percent of its price, "what the price assumes" is the whole question. This
module inverts the model firm by firm, holding every other input at the
model's value (fundamentals as available at t, point in time):

    omega_star   persistence of excess ROE at which V_F = P (base fade model,
                 industry median target). Solved on [0, 0.995]; NaN with
                 status +1 when even 0.995 falls short - persistence of the
                 current profitability cannot justify the price, growth in
                 the level of profits is required; 0 with status -1 when the
                 price is below the no-persistence value. Only where the
                 current excess ROE is above its long-run level (value
                 increasing in omega). half_life = ln 0.5 / ln omega_star.
    T_star       years the CURRENT excess ROE must last, then vanish, for
                 V = P (competitive advantage period); NaN when > 60 years.
                 Only where ROE > r.
    g_star_10y   constant annual earnings growth the price requires for ten
                 years (margins constant, clean-surplus book), followed by
                 the base ten-year fade from the year-10 excess ROE and the
                 base terminal. The headline "how fast" number. Solved on
                 [-50%, +200%] per year; NaN when trailing adjusted earnings
                 <= 0 or when even +200% falls short.
    g_star_3y    the same over three years followed by the phase-6 fade
                 (growth_rim.value_growth) - the horizon of the fundamentals
                 forecast, kept for the comparison with g_model. Because the
                 fade after year 3 is fast for the median firm, this number
                 is much larger than g_star_10y and is not the headline.
    g_model      the fundamentals-based expectation for the same three years:
                 geometric mean of the expanding-window revenue growth
                 forecasts with backlog (growth_rim --backlog), and v_f_gr
                 the value they imply.
    r_star_gXX   the return axis. One price is one equation in two unknowns
                 (growth and the required return); g_star fixes the return at
                 the model's r_f + beta x ERP and solves for growth. These
                 columns do the converse for growth views of 5..30%/y: the
                 annual return the price offers if the firm grows that fast for
                 ten years and then fades as the model says. Read against your
                 own required return.
    g_star_10y_erpavg
                 g_star with the market-wide ERP at its sample average instead
                 of today's implied value. The implied ERP is backed out of the
                 index level, so an expensive market lowers r and makes every
                 stock's required growth look modest; this column shows how
                 much of g_star is that.
    p_achieve_3y the conditional base rate: a logit of "realised 3-year revenue
                 growth met the required growth" on the required growth, size,
                 past growth, profitability and industry, fitted on every
                 firm-quarter with an observed outcome and applied to the latest
                 month. "Firms that looked like this were asked for this much
                 and delivered it p% of the time." In-sample; a base rate, not
                 a forecast test.

Reading a firm: V_F (no growth beyond the fade) -> v_f_gr (what observable
fundamentals justify) -> P (what the market expects), or in logs
    log(P/V_F) = log(v_f_gr/V_F) + log(P/v_f_gr).
The decision rule "buy if my expectation exceeds the market's" compares the
investor's own growth or persistence view with g_star / omega_star / T_star.
The report checks whether g_star carries information about realised growth
(it should: prices aggregate information), gives base rates for the required
growth being met, and tests whether the gap to g_model predicts returns.

Outputs data/parquet/expectations_monthly.parquet, expectations_calibration.parquet,
        reports/phase7_expectations.md. The per-stock table for the latest month
        is written by sfv.report (reports/sfv_latest.csv, reports/sfv_report.html).
Run: python -m sfv.expectations [--decomp-file decomp_monthly_momexp.parquet] [--top 30]
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pandas as pd
import statsmodels.api as sm

from . import config as C
from .rim import T_YEARS, OMEGA_MAX, X_INF_CAP, X0_CAP, G_CAP, terminal
from .implied import value_common_omega
from .growth_rim import value_growth
from .validate import fm

OMEGA_HI = 0.995
T_MAX = 60
G_LO, G_HI = -0.5, 2.0
H_LONG = 10
G_VIEWS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)   # growth views for the return axis r*(g)
R_LO, R_HI = -0.20, 0.80                          # bracket for the implied return
L = []


def p(t=""):
    L.append(t)


def bisect_vec(fn, lo: float, hi: float, target: np.ndarray, it: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """Element-wise bisection for fn non-decreasing in its argument.

    Returns (x, status): status 0 solved; -1 target <= fn(lo), x = lo; +1 target > fn(hi), x = NaN.
    """
    lo0 = np.full(len(target), lo, dtype=float)
    hi0 = np.full(len(target), hi, dtype=float)
    vlo, vhi = fn(lo0), fn(hi0)
    status = np.where(target <= vlo, -1, np.where(target > vhi, 1, 0))
    lo_, hi_ = lo0.copy(), hi0.copy()
    for _ in range(it):
        mid = 0.5 * (lo_ + hi_)
        below = fn(mid) < target
        lo_ = np.where(below, mid, lo_)
        hi_ = np.where(below, hi_, mid)
    x = 0.5 * (lo_ + hi_)
    x = np.where(status == -1, lo0, x)
    return np.where(status == 1, np.nan, x), status


def value_growth_h(B0, E0, g, H, xinf, omega, roe_star, payout, r, T_fade=T_YEARS):
    """H years of constant earnings growth g, then the base fade for T_fade years from the year-H excess ROE, then the base terminal."""
    omega = np.minimum(omega, OMEGA_MAX)
    B_prev = B0.copy()
    E = E0.copy()
    pv = np.zeros_like(B0)
    disc = np.ones_like(B0)
    roeH = None
    for _ in range(1, H + 1):
        E = E * (1.0 + g)
        ri = E - r * B_prev
        disc = disc * (1.0 + r)
        pv = pv + ri / disc
        roeH = E / B_prev
        B_prev = np.maximum(B_prev + E * (1.0 - payout), 1e-6)
    x = np.clip(roeH - roe_star, -X0_CAP, X0_CAP)
    ri_T = g_T = None
    for k in range(1, T_fade + 1):
        x = xinf + omega * (x - xinf)
        roe = roe_star + x
        ri = (roe - r) * B_prev
        disc = disc * (1.0 + r)
        pv = pv + ri / disc
        gb = np.clip(roe * (1.0 - payout), -0.5, G_CAP)
        if k == T_fade:
            ri_T, g_T = ri, gb
        B_prev = B_prev * (1.0 + gb)
    tv = terminal(ri_T, disc, omega, g_T, r)
    return B0 + pv + tv


def cap_years(B0, x0, roe_star, payout, r, P, T_max: int = T_MAX) -> np.ndarray:
    """Smallest T with B0 + PV(T years of the current excess ROE) >= P; NaN beyond T_max. Mirrors implied.value_cap."""
    x = np.clip(x0, -X0_CAP, X0_CAP)
    roe = roe_star + x
    g = np.clip(roe * (1.0 - payout), -0.5, G_CAP)
    B_prev = B0.copy()
    pv = np.zeros_like(B0)
    disc = np.ones_like(B0)
    T_star = np.full(len(B0), np.nan)
    hit = B0 >= P
    T_star[hit] = 0.0
    for k in range(1, T_max + 1):
        ri = (roe - r) * B_prev
        disc = disc * (1.0 + r)
        pv = pv + ri / disc
        B_prev = B_prev * (1.0 + g)
        now = (B0 + pv >= P) & ~hit
        T_star[now] = float(k)
        hit |= now
    return T_star


def realized_growth() -> pd.DataFrame:
    """Realised cumulative log revenue growth 1 and 3 years after each fundamentals quarter (for the checks),
    and the past year's growth b1 (a conditioning variable for the base rate)."""
    I = pd.read_parquet(C.PQ / "intangibles_quarterly.parquet", columns=["cik", "end", "ttm_revenue", "ttm_ok"])
    I = I[I["ttm_ok"].fillna(False) & (I["ttm_revenue"] > 0)].sort_values(["cik", "end"]).drop_duplicates(["cik", "end"])
    g = I.groupby("cik")
    with np.errstate(divide="ignore", invalid="ignore"):
        for k, nm, lo, hi in ((4, "f1", 340, 390), (12, "f3", 1020, 1170)):
            fwd = g["ttm_revenue"].shift(-k)
            ok = (g["end"].shift(-k) - I["end"]).dt.days.between(lo, hi) & (fwd > 0)
            I[nm] = np.log(fwd / I["ttm_revenue"]).where(ok).clip(-2, 3)
        bwd = g["ttm_revenue"].shift(4)
        okb = (I["end"] - g["end"].shift(4)).dt.days.between(340, 390) & (bwd > 0)
        I["b1"] = np.log(I["ttm_revenue"] / bwd).where(okb).clip(-2, 3)
    return I[["cik", "end", "f1", "f3", "b1"]].rename(columns={"end": "f_end"})


def normalized_earnings() -> pd.DataFrame:
    """Mid-cycle adjusted earnings: TTM revenue x the firm's average adjusted margin over the trailing
    five years (at least eight quarters), keyed by the fundamentals quarter.

    For a firm at a cyclical peak or trough the trailing-year earnings are a poor starting point for the
    required growth; this gives the alternative the report shows alongside, with a flag when the two differ.
    """
    I = pd.read_parquet(C.PQ / "intangibles_quarterly.parquet", columns=["cik", "end", "ttm_revenue", "ttm_e_adj", "ttm_ok"])
    I = I[I["ttm_ok"].fillna(False) & (I["ttm_revenue"] > 0)].sort_values(["cik", "end"]).drop_duplicates(["cik", "end"])
    I["margin_ttm"] = (I["ttm_e_adj"] / I["ttm_revenue"]).clip(-1, 1)
    I["margin_avg5"] = I.groupby("cik")["margin_ttm"].transform(lambda s: s.rolling(20, min_periods=8).mean())
    I["e_norm"] = I["ttm_revenue"] * I["margin_avg5"]
    return I[["cik", "end", "margin_ttm", "margin_avg5", "e_norm"]].rename(columns={"end": "f_end"})


CYCLE_GAP = 0.30      # |trailing / normalised earnings - 1| above this flags an earnings phase


def pct(s: pd.Series) -> pd.Series:
    return s.quantile([.1, .25, .5, .75, .9]).rename(index=lambda q: f"p{int(q*100)}")


def top50_r(Xl: pd.DataFrame, col: str) -> float:
    """Median of a column over the 50 largest firms of a (market-cap sorted) latest-month frame."""
    return float(Xl.head(50)[col].median() * 100)


def solve_growth(fn_value, E0, P, idx_ok):
    n = len(P)
    g_star = np.full(n, np.nan); g_status = np.full(n, np.nan)
    gs, stg = bisect_vec(fn_value, G_LO, G_HI, P[idx_ok])
    g_star[idx_ok] = gs; g_status[idx_ok] = stg
    return g_star, g_status


def solve_return(fn_value, P: np.ndarray, lo: float = R_LO, hi: float = R_HI, it: int = 45) -> tuple[np.ndarray, np.ndarray]:
    """Element-wise bisection for a value DEcreasing in the discount rate: the return the price offers.

    Returns (r, status): 0 solved; -1 the value at `lo` is already below P (return below lo);
    +1 the value at `hi` is still above P (return above hi). Unsolved entries are NaN.
    """
    lo0 = np.full(len(P), lo, dtype=float)
    hi0 = np.full(len(P), hi, dtype=float)
    vlo, vhi = fn_value(lo0), fn_value(hi0)
    status = np.where(vlo < P, -1, np.where(vhi > P, 1, 0))
    lo_, hi_ = lo0.copy(), hi0.copy()
    for _ in range(it):
        mid = 0.5 * (lo_ + hi_)
        above = fn_value(mid) > P          # value still above the price: the rate is too low
        lo_ = np.where(above, mid, lo_)
        hi_ = np.where(above, hi_, mid)
    x = 0.5 * (lo_ + hi_)
    return np.where(status == 0, x, np.nan), status


BUCKET_EDGES = [-1.0, 0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 10.0]
BUCKET_LABELS = ["< 0%", "0~5%", "5~10%", "10~15%", "15~20%", "20~30%", "30~50%", "> 50%"]


def conditional_base_rate(Q: pd.DataFrame, X: pd.DataFrame, say) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Base rate conditioned on what the firm looks like.

    Logit of 1[realised 3-year revenue growth >= the growth the price required] on the
    required growth, log size, the past year's revenue growth, adjusted ROE and FF12
    industry, fitted on every firm-quarter whose 3-year outcome is observed, then
    evaluated for every firm-month with a g* (the latest month included). In-sample by
    construction: it is a base rate, not a forecast test. Returns the probability aligned
    to X, the coefficient table, and the plain version (rate by size group x bucket).
    """
    fit = Q.dropna(subset=["g_star_10y", "f3", "mktcap", "roe_adj"]).copy()
    fit = fit[fit["mktcap"] > 0]
    y = (fit["f3"] >= 3.0 * np.log1p(fit["g_star_10y"])).astype(float).to_numpy()
    inds = sorted(fit["ff12"].dropna().unique())
    b1_med = float(fit["b1"].median())
    ls = np.log(fit["mktcap"])
    mu, sd = float(ls.mean()), float(ls.std() + 1e-12)

    def design(D: pd.DataFrame) -> pd.DataFrame:
        cols = {"const": np.ones(len(D)),
                "req_growth": np.log1p(D["g_star_10y"].clip(-0.5, 2.0)).to_numpy(float),
                "log_size": ((np.log(D["mktcap"]) - mu) / sd).to_numpy(float),
                "past_growth": D["b1"].fillna(b1_med).clip(-1, 2).to_numpy(float),
                "roe_adj": D["roe_adj"].clip(-1, 1).fillna(0.0).to_numpy(float)}
        for ind in inds[1:]:
            cols[f"ind_{ind}"] = (D["ff12"] == ind).astype(float).to_numpy()
        return pd.DataFrame(cols, index=D.index)

    Xf = design(fit)
    try:
        res = sm.Logit(y, Xf.values).fit(disp=0, maxiter=200)
        coef = pd.DataFrame({"coef": np.asarray(res.params), "z": np.asarray(res.tvalues)}, index=Xf.columns)
        pr2 = float(res.prsquared)
    except Exception as ex:  # noqa: BLE001  separation or non-convergence
        say(f"  conditional base rate: plain logit failed ({ex}); using an L2-regularised fit")
        res = sm.Logit(y, Xf.values).fit_regularized(disp=0, alpha=1.0, maxiter=500)
        coef = pd.DataFrame({"coef": np.asarray(res.params), "z": np.nan}, index=Xf.columns)
        pr2 = np.nan
    ok = (X["g_star_10y"].notna() & (X["mktcap"] > 0)).to_numpy()
    p = np.full(len(X), np.nan)
    Xp = design(X[ok])
    p[ok] = 1.0 / (1.0 + np.exp(-(Xp.values @ np.asarray(res.params))))
    # the plain version: achievement rate by size group x required-growth bucket
    fit["rank"] = fit.groupby("qend")["mktcap"].rank(ascending=False)
    fit["size"] = np.where(fit["rank"] <= 200, "시총 상위 200", "201위 이하")
    fit["bucket"] = pd.cut(fit["g_star_10y"], bins=BUCKET_EDGES, labels=BUCKET_LABELS, right=False)
    fit["hit"] = y
    cond = fit.groupby(["size", "bucket"], observed=True).agg(n=("hit", "size"), rate=("hit", "mean")).reset_index()
    cond["bucket"] = cond["bucket"].astype(str)
    say(f"  conditional base rate: logit on {len(fit):,} firm-quarters ({fit['qend'].min().date()}..{fit['qend'].max().date()}), "
        f"pseudo-R2 {pr2:.3f}; probability attached to {int(ok.sum()):,} firm-months")
    return p, coef, cond


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decomp-file", default="decomp_monthly_momexp.parquet", help="decomposition to attach (eps, components)")
    ap.add_argument("--top", type=int, default=30, help="firms in the latest-month table")
    a = ap.parse_args(argv)
    log = open(C.LOGS / "expectations.log", "w", encoding="utf-8")

    def say(msg):
        print(msg, flush=True); log.write(msg + "\n"); log.flush()

    t0 = time.time()
    R = pd.read_parquet(C.PQ / "rim_monthly.parquet",
                        columns=["cik", "month", "ticker", "name", "ff12", "mktcap", "f_end", "b_adj", "ttm_e_adj", "roe_adj",
                                 "roe_star_ind", "x0", "omega", "a", "r", "rf", "beta", "erp", "payout", "v_f", "log_pv"])
    G = pd.read_parquet(C.PQ / "rim_monthly_grbl.parquet", columns=["cik", "month", "g1", "g2", "g3", "gr_applied", "v_f_gr", "log_pv_gr"])
    X = R.merge(G, on=["cik", "month"], how="left")
    X = X[(X["mktcap"] > 0) & (X["b_adj"] > 0) & X["v_f"].notna() & X["r"].notna()].reset_index(drop=True)
    X["xinf"] = np.clip(X["a"] / (1.0 - np.minimum(X["omega"], OMEGA_MAX)), -X_INF_CAP, X_INF_CAP)
    X["x0c"] = X["x0"].clip(-X0_CAP, X0_CAP)
    n = len(X)
    say(f"  firm-months {n:,}, months {X.month.nunique()}, {X.month.min().date()}..{X.month.max().date()}")
    B0 = X["b_adj"].to_numpy(float); x0 = X["x0"].to_numpy(float); xinf = X["xinf"].to_numpy(float)
    rs = X["roe_star_ind"].to_numpy(float); po = X["payout"].to_numpy(float); r = X["r"].to_numpy(float)
    P = X["mktcap"].to_numpy(float); om = X["omega"].to_numpy(float); E0 = X["ttm_e_adj"].to_numpy(float)

    # --- omega*: excess ROE path above its long-run level, value increasing in omega
    om_star = np.full(n, np.nan); om_status = np.full(n, np.nan)
    idx = np.where((X["x0c"] > X["xinf"]).to_numpy())[0]
    xs, st = bisect_vec(lambda w: value_common_omega(B0[idx], x0[idx], xinf[idx], rs[idx], po[idx], r[idx], w), 0.0, OMEGA_HI, P[idx])
    om_star[idx] = xs; om_status[idx] = st
    X["omega_star"] = om_star; X["omega_status"] = om_status
    with np.errstate(divide="ignore", invalid="ignore"):
        hl = np.log(0.5) / np.log(X["omega_star"])
    X["half_life"] = np.where(X["omega_star"] > 0, hl, np.where(X["omega_star"] == 0, 0.0, np.nan))
    say(f"  omega*: eligible {len(idx):,} ({len(idx)/n*100:.1f}%), solved {(st == 0).mean()*100:.1f}%, "
        f"beyond 0.995 (needs profit growth) {(st == 1).mean()*100:.1f}%, below zero-persistence value {(st == -1).mean()*100:.1f}%")

    # --- T*: current excess ROE held then zero; only where ROE > r
    T_star = np.full(n, np.nan)
    idx_t = np.where(((rs + X["x0c"].to_numpy()) > r))[0]
    T_star[idx_t] = cap_years(B0[idx_t], x0[idx_t], rs[idx_t], po[idx_t], r[idx_t], P[idx_t])
    X["T_star"] = T_star
    say(f"  T*: eligible (ROE > r) {len(idx_t):,}, solved <= {T_MAX}y {np.isfinite(T_star[idx_t]).mean()*100:.1f}%")

    # --- g*: constant earnings growth, 10 years (headline) and 3 years (phase-6 horizon)
    idx_g = np.where(E0 > 0)[0]
    g10, s10 = solve_growth(lambda g: value_growth_h(B0[idx_g], E0[idx_g], g, H_LONG, xinf[idx_g], om[idx_g], rs[idx_g], po[idx_g], r[idx_g]), E0, P, idx_g)
    g3, s3 = solve_growth(lambda g: value_growth(B0[idx_g], E0[idx_g], g, g, g, xinf[idx_g], om[idx_g], rs[idx_g], po[idx_g], r[idx_g])[0], E0, P, idx_g)
    X["g_star_10y"] = g10; X["g_status_10y"] = s10; X["g_star_3y"] = g3; X["g_status_3y"] = s3
    say(f"  g* (E>0: {len(idx_g):,}, {len(idx_g)/n*100:.1f}%): 10y solved {(s10 == 0).mean()*100:.1f}%, beyond +200%/y {(s10 == 1).mean()*100:.1f}%, "
        f"below -50%/y {(s10 == -1).mean()*100:.1f}% | 3y solved {(s3 == 0).mean()*100:.1f}%, beyond {(s3 == 1).mean()*100:.1f}%")

    # --- model expectation and the premium split
    gm = ((1 + X["g1"]) * (1 + X["g2"]) * (1 + X["g3"])) ** (1 / 3) - 1
    X["g_model"] = gm.where(X["gr_applied"].fillna(False))
    with np.errstate(divide="ignore", invalid="ignore"):
        X["log_vgr_vf"] = np.log(X["v_f_gr"] / X["v_f"]).where((X["v_f_gr"] > 0) & (X["v_f"] > 0))
    X["log_p_vgr"] = X["log_pv_gr"]
    X["gap_star_model"] = np.log1p(X["g_star_3y"]) - np.log1p(X["g_model"])

    # --- attach the decomposition residual (peer-relative reading)
    try:
        D = pd.read_parquet(C.PQ / a.decomp_file)
        dcols = [c for c in ["eps_A", "delta_S_A", "delta_T_A", "delta_F_A", "delta_M_A", "delta_E_A"] if c in D.columns]
        X = X.merge(D[["cik", "month"] + dcols], on=["cik", "month"], how="left")
        say(f"  decomposition attached from {a.decomp_file}: {dcols}")
    except Exception as ex:  # noqa: BLE001
        dcols = []
        say(f"  decomposition not attached: {ex}")

    # --- the return axis: for a growth view, the annual return the price offers (value decreasing in r)
    rcols = []
    for g in G_VIEWS:
        rr = np.full(n, np.nan)
        sol, _ = solve_return(lambda rv, g=g: value_growth_h(B0[idx_g], E0[idx_g], g, H_LONG, xinf[idx_g], om[idx_g], rs[idx_g], po[idx_g], rv),
                              P[idx_g])
        rr[idx_g] = sol
        c = f"r_star_g{int(round(g * 100)):02d}"
        X[c] = rr; rcols.append(c)
    say(f"  return axis r*(g) for growth views {', '.join(f'{g:.0%}' for g in G_VIEWS)}: "
        f"solved for {np.isfinite(X['r_star_g15']).mean()*100:.1f}% of firm-months at g = 15%")

    # --- the discount-rate reference: g* if the market-wide ERP were at its sample average
    erp_avg = float(X.groupby("month")["erp"].first().mean())
    r_alt = (X["rf"] + X["beta"] * erp_avg).to_numpy(float)
    r_alt = np.where(np.isfinite(r_alt), r_alt, r)
    ga, _ = solve_growth(lambda g: value_growth_h(B0[idx_g], E0[idx_g], g, H_LONG, xinf[idx_g], om[idx_g], rs[idx_g], po[idx_g], r_alt[idx_g]),
                         E0, P, idx_g)
    X["g_star_10y_erpavg"] = ga; X["erp_avg"] = erp_avg
    last_rows = (X["month"] == X["month"].max()).to_numpy()
    erp_now = float(X.loc[last_rows, "erp"].iloc[0])
    say(f"  ERP now {erp_now*100:.2f}% vs sample average {erp_avg*100:.2f}%: at the average ERP the latest month's median g* is "
        f"{np.nanmedian((ga - X['g_star_10y'].to_numpy())[last_rows]) * 100:+.1f}pp from the headline")

    # --- the cyclical alternative: required growth from mid-cycle (five-year average margin) earnings
    X = X.merge(normalized_earnings(), on=["cik", "f_end"], how="left")
    En = X["e_norm"].to_numpy(float)
    idx_n = np.where(np.isfinite(En) & (En > 0))[0]
    gn, _ = solve_growth(lambda g: value_growth_h(B0[idx_n], En[idx_n], g, H_LONG, xinf[idx_n], om[idx_n], rs[idx_n], po[idx_n], r[idx_n]),
                         En, P, idx_n)
    X["g_star_10y_norm"] = gn
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = X["ttm_e_adj"] / X["e_norm"]
    X["cycle_flag"] = (X["e_norm"] > 0) & ((X["ttm_e_adj"] <= 0) | ((ratio - 1.0).abs() > CYCLE_GAP))
    say(f"  normalised earnings: available for {len(idx_n):,} firm-months, g* solved {np.isfinite(gn).mean()*100:.1f}%; "
        f"earnings-phase flag on {X.loc[last_rows, 'cycle_flag'].mean()*100:.0f}% of the latest month")

    # --- quarterly panel with realised growth: the checks below and the conditional base rate
    RG = realized_growth()
    X = X.merge(RG[["cik", "f_end", "b1"]], on=["cik", "f_end"], how="left")
    Q = X[X["month"].dt.month.isin([3, 6, 9, 12])].rename(columns={"month": "qend"}).copy()
    Q = Q.merge(RG[["cik", "f_end", "f1", "f3"]], on=["cik", "f_end"], how="left")
    Q["gs1"] = np.log1p(Q["g_star_10y"]); Q["gs3"] = 3.0 * Q["gs1"]
    p_ach, COEF, COND = conditional_base_rate(Q, X, say)
    X["p_achieve_3y"] = p_ach
    # the same achievement rate cut by the firm's own past-year revenue growth: is a demanding premise more
    # often met among firms that were already growing fast (the "rising star" question)
    Q3 = Q.dropna(subset=["g_star_10y", "f3"]).copy()
    Q3["hit"] = (Q3["f3"] >= 3.0 * np.log1p(Q3["g_star_10y"])).astype(float)
    Q3["bucket"] = pd.cut(Q3["g_star_10y"], bins=BUCKET_EDGES, labels=BUCKET_LABELS, right=False)
    Q3["group"] = pd.cut(np.expm1(Q3["b1"]), bins=[-np.inf, 0.10, 0.30, np.inf], labels=["직전 1년 성장 10% 미만", "10~30%", "30% 이상"])
    PG = Q3.dropna(subset=["group"]).groupby(["group", "bucket"], observed=True).agg(n=("hit", "size"), rate=("hit", "mean")).reset_index()
    PG["bucket"] = PG["bucket"].astype(str); PG["group"] = PG["group"].astype(str)
    COND = pd.concat([COND.rename(columns={"size": "group"}).assign(dim="규모"), PG.assign(dim="직전 성장")], ignore_index=True)
    COND.assign(month=X["month"].max()).to_parquet(C.PQ / "expectations_calibration_cond.parquet", index=False)

    out_cols = ["cik", "month", "ticker", "name", "ff12", "mktcap", "f_end", "b_adj", "ttm_e_adj", "roe_adj", "roe_star_ind", "x0", "xinf",
                "omega", "r", "rf", "beta", "erp", "erp_avg", "payout", "v_f", "v_f_gr", "log_pv", "log_vgr_vf", "log_p_vgr", "g1", "g2", "g3", "g_model",
                "g_star_10y", "g_status_10y", "g_star_3y", "g_status_3y", "g_star_10y_erpavg", "gap_star_model", "omega_star", "omega_status",
                "half_life", "T_star"] + rcols + ["b1", "p_achieve_3y", "margin_ttm", "margin_avg5", "e_norm", "g_star_10y_norm", "cycle_flag"] + dcols
    X = X[out_cols]
    X.to_parquet(C.PQ / "expectations_monthly.parquet", index=False)

    # ------------------------------------------------------------------ report
    last = X["month"].max()
    Xl = X[X["month"] == last].sort_values("mktcap", ascending=False)
    # the decomposition lags the valuation by the 13F filing delay: attach each firm's latest residual (<= 200 days old)
    eps_note = ""
    if "eps_A" in X.columns:
        De = X.dropna(subset=["eps_A"]).sort_values("month").groupby("cik").tail(1)[["cik", "month", "eps_A"]]
        De = De.rename(columns={"month": "eps_month", "eps_A": "eps_last"})
        Xl = Xl.merge(De, on="cik", how="left")
        stale = (last - Xl["eps_month"]).dt.days > 200
        Xl.loc[stale, ["eps_last", "eps_month"]] = np.nan
        em = Xl["eps_month"].max()
        eps_note = f" ε은 분해가 가능한 최근 월({em.date() if pd.notna(em) else 'n/a'}, 13F 지연) 기준."
    else:
        Xl["eps_last"] = np.nan; Xl["eps_month"] = pd.NaT
    p("# 단계 7 — 가격이 요구하는 기대 (종목별 내재 지속성·성장, 역산 잔여이익모형)\n")
    p(f"> 생성 {pd.Timestamp.today().date()}. 표본 {X.month.min().date()} ~ {last.date()}, 기업-월 {n:,}. 모든 입력은 해당 시점까지 공시된 값(point in time). "
      "다른 입력을 모형값에 고정하고 한 변수만 풀어 V = P가 되는 값을 구한다: ω*(초과 ROE 지속성, 감쇠 목표 = 산업 중앙값), "
      "T*(현재 초과 ROE가 유지돼야 하는 연수, 이후 0), g*_10y(10년간 연간 이익성장률, 이후 기존 10년 감쇠와 터미널), "
      "g*_3y(3년간 연간 이익성장률, 이후 단계 6 모형의 감쇠). g_model은 같은 3년에 대한 펀더멘털 기반 전망(확장 윈도우, 백로그 포함)의 연환산 기하평균.")
    p("\n프리미엄 읽기: log(P/V_F) = log(V_gr/V_F) + log(P/V_gr). 앞 항은 관측 가능한 펀더멘털 전망이 정당화하는 부분, 뒤 항은 그 너머로 가격이 요구하는 부분이다. "
      "ε은 횡단면 분해(모멘텀·기대 성분 포함 스펙)의 잔차로, 동료 기업·특성 대비 상대적 프리미엄이다. "
      "g*_3y는 3년 뒤 빠른 감쇠를 전제하므로 중앙값 기업에서 매우 크게 나온다. 헤드라인은 g*_10y다.")

    p(f"\n## 1. {last.date()} 시총 상위 {a.top}\n")
    T = Xl.head(a.top)
    tab = pd.DataFrame({
        "시총 $bn": T["mktcap"] / 1e9, "조정 ROE": T["roe_adj"], "log(P/V_F)": T["log_pv"], "log(V_gr/V_F)": T["log_vgr_vf"], "log(P/V_gr)": T["log_p_vgr"],
        "g_model %/y": T["g_model"] * 100, "g*_3y %/y": T["g_star_3y"] * 100, "g*_10y %/y": T["g_star_10y"] * 100,
        "g* 평균ERP %/y": T["g_star_10y_erpavg"] * 100, "g* 정규화 %/y": T["g_star_10y_norm"] * 100, "할인율 %": T["r"] * 100,
        "r*(성장 15%) %": T["r_star_g15"] * 100, "3y 달성 확률": T["p_achieve_3y"], "ω*": T["omega_star"], "반감기 y": T["half_life"],
        "T* y": T["T_star"], "ε": T["eps_last"]})
    tab.index = T["ticker"].values
    p(tab.round(2).to_markdown())
    p("\n빈칸(nan): g*는 조정이익 ≤ 0이거나 연 +200% 성장으로도 가격에 못 미치는 경우, ω*는 초과 ROE가 장기 수준 아래거나 0.995로도 못 미치는 경우"
      "(현재 수익성의 지속만으로는 가격이 설명되지 않고 이익 규모의 성장이 필요하다는 뜻), T*는 ROE ≤ r이거나 60년 초과." + eps_note)
    p("\n**할인율 %**는 모형의 r = 국채 + β × ERP. **r*(성장 15%)**는 그 회사가 10년간 연 15% 성장하고 이후 기존 감쇠를 따를 때 지금 가격이 주는 연 수익률 "
      "(수익률 축; 성장 5·10·15·20·25·30%에 대해 `r_star_gXX` 열). **g* 평균ERP**는 시장 위험프리미엄이 표본 평균이었을 때의 요구 성장. "
      "**3y 달성 확률**은 규모·산업·최근 성장·수익성이 비슷한 기업이 그 요구 성장을 이후 3년간 달성한 역사적 비율(4-2절). "
      "**g* 정규화**는 출발 이익을 후행 4분기 대신 정규화 이익(최근 5년 평균 조정 이익률 × 현재 매출)으로 놓고 다시 푼 요구 성장. "
      f"두 출발 이익이 {CYCLE_GAP:.0%} 넘게 다르거나 후행 이익이 적자면 '이익 국면' 표시가 붙는다(경기 저점·피크 기업).")
    fl = Xl[Xl["cycle_flag"].fillna(False)]
    if len(fl):
        p(f"\n이익 국면 표시 {len(fl):,}종목({len(fl)/len(Xl)*100:.0f}%). 후행 g*와 정규화 g*의 차이 중앙값 "
          f"{((fl['g_star_10y'] - fl['g_star_10y_norm']).abs().median())*100:.1f}%p. 후행 이익이 정규화보다 높은(피크) 종목 "
          f"{int((fl['ttm_e_adj'] > fl['e_norm']).sum()):,}, 낮은(저점) 종목 {int((fl['ttm_e_adj'] <= fl['e_norm']).sum()):,}.")

    p(f"\n## 1-1. 수익률 축 — 성장 전망별로 가격이 주는 연 수익률 ({last.date()})\n")
    p("가격 하나는 미지수 둘(성장, 요구수익률)의 식 하나다. g*는 요구수익률을 모형 할인율에 고정하고 성장을 푼 것이고, 아래는 반대로 성장 전망을 고정하고 "
      "수익률을 푼 것이다. 자기 성장 전망에서의 수익률이 자기 요구수익률보다 높으면 매수 근거다. 국채 수익률과 비교하면 위험프리미엄이 얼마 남는지 보인다.\n")
    rows = {}
    for g in G_VIEWS:
        c = f"r_star_g{int(round(g * 100)):02d}"
        rows[f"성장 {g:.0%}/y"] = {"전체 중앙값 %": Xl[c].median() * 100, "전체 p25 %": Xl[c].quantile(.25) * 100, "전체 p75 %": Xl[c].quantile(.75) * 100,
                               "시총 상위 50 중앙값 %": top50_r(Xl, c), "풀린 종목": int(Xl[c].notna().sum())}
    p(pd.DataFrame(rows).T.round(1).to_markdown())
    rf_now = float(Xl["rf"].iloc[0]); erp_now_ = float(Xl["erp"].iloc[0]); erp_avg_ = float(Xl["erp_avg"].iloc[0])
    p(f"\n국채 10년 {rf_now*100:.1f}%, 내재 ERP {erp_now_*100:.1f}% (2014년 이후 평균 {erp_avg_*100:.1f}%). "
      f"ERP가 평균이었다면 요구 성장 g*는 중앙값 {np.nanmedian((Xl['g_star_10y_erpavg'] - Xl['g_star_10y']).to_numpy())*100:+.1f}%p 달라진다. "
      "내재 ERP는 지수 수준에서 거꾸로 구한 값이라 시장 전체가 비싸면 낮게 나오고, 그러면 모든 종목의 요구 성장이 낮아 보인다. "
      "종목의 g*는 시장 전체의 요구수익률을 기준으로 잰 상대적 전제이며, 시장 자체의 전제는 단계 4의 ω*로 본다.")

    p(f"\n## 2. {last.date()} 분포\n")
    top50 = Xl.head(50)
    dist = pd.DataFrame({
        "g*_10y %/y (전체)": pct(Xl["g_star_10y"].dropna() * 100), "g*_10y %/y (시총 상위 50)": pct(top50["g_star_10y"].dropna() * 100),
        "g*_3y %/y (전체)": pct(Xl["g_star_3y"].dropna() * 100), "g_model %/y (전체)": pct(Xl["g_model"].dropna() * 100),
        "반감기 y (전체)": pct(Xl["half_life"].dropna()), "반감기 y (상위 50)": pct(top50["half_life"].dropna()), "T* y (전체)": pct(Xl["T_star"].dropna())})
    p(dist.round(1).to_markdown())
    st_all = Xl["omega_status"]
    n_el = int(st_all.notna().sum())
    p(f"\nω* 상태 (해당 기업 {n_el:,}개): 0.9 초과 {((Xl['omega_star'] > 0.9) | (st_all == 1)).sum() / n_el * 100:.1f}%, "
      f"0.995로도 못 미침 {(st_all == 1).sum() / n_el * 100:.1f}%, 무지속성 가치 아래 {(st_all == -1).sum() / n_el * 100:.1f}%. "
      f"g*_10y: 조정이익 > 0 기업 중 +200%/y로도 못 미침 {(Xl['g_status_10y'] == 1).sum() / max(Xl['g_status_10y'].notna().sum(), 1) * 100:.1f}%.")

    p("\n## 3. 시장이 요구하는 기대의 추이 (12월 말 + 최근 월)\n")
    months = sorted(set(X[X["month"].dt.month == 12]["month"].unique()) | {last})
    rows = {}
    for m in months:
        d = X[X["month"] == m]
        w = d["log_p_vgr"].notna()
        lp = d["log_pv"].notna()
        rows[str(m.date())] = {
            "기업 수": len(d), "g*_10y 중앙값 %/y": d["g_star_10y"].median() * 100, "g*_10y p75 %/y": d["g_star_10y"].quantile(.75) * 100,
            "g*_10y 시총가중 %/y": np.average(d["g_star_10y"].dropna(), weights=d.loc[d["g_star_10y"].notna(), "mktcap"]) * 100 if d["g_star_10y"].notna().any() else np.nan,
            "g*_3y 중앙값 %/y": d["g_star_3y"].median() * 100, "g_model 중앙값 %/y": d["g_model"].median() * 100, "반감기 중앙값 y": d["half_life"].median(),
            "ω* > 0.9 비율": ((d["omega_star"] > 0.9) | (d["omega_status"] == 1)).sum() / max(d["omega_status"].notna().sum(), 1),
            "시총가중 log(P/V_gr)": np.average(d.loc[w, "log_p_vgr"], weights=d.loc[w, "mktcap"]) if w.any() else np.nan,
            "시총가중 log(P/V_F)": np.average(d.loc[lp, "log_pv"], weights=d.loc[lp, "mktcap"])}
    p(pd.DataFrame(rows).T.round(2).to_markdown())

    # ------------------------------------------------------------------ checks
    p("\n## 4. 검증 — 가격이 요구하는 성장 g*_10y는 실제 성장을 예측하는가\n")
    Q["gm1"] = np.log1p(Q["g1"]).where(Q["g_model"].notna()); Q["gm3"] = np.log1p(Q["g_model"]).where(Q["g_model"].notna()) * 3.0
    Q["lpv"] = Q["log_pv"]
    rows = {}
    for y, lag, specs in (("f1", 4, (["gs1"], ["gm1"], ["gs1", "gm1"], ["lpv"])), ("f3", 12, (["gs3"], ["gm3"], ["gs3", "gm3"], ["lpv"]))):
        for xs in specs:
            res, _ = fm(Q, y, xs, lag)
            rows[f"{y} ~ {' + '.join(xs)}"] = {**{f"{c} 계수": res.loc[c, "coef"] for c in xs}, **{f"{c} t": res.loc[c, "t"] for c in xs}, "분기": int(res.loc[xs[0], "quarters"])}
    p("실현 누적 로그 매출성장(1년 f1, 3년 f3)을 시점 t의 내재 성장(gs = 지평 × log(1+g*_10y)), 펀더멘털 전망(gm), log(P/V_F)에 분기별 횡단면 회귀 "
      "(Fama-MacBeth, NW t, 시차 = 지평). 표본은 실현값이 있는 분기까지. 계수 1 = 요구 성장이 그대로 실현.\n")
    p(pd.DataFrame(rows).T.round(3).to_markdown())
    say("growth check:\n" + pd.DataFrame(rows).T.round(3).to_string())

    p("\n### 4-1. 기저율 — 가격이 요구한 성장률 g*_10y를 이후 1년·3년에 실제로 달성한 비율\n")
    Qc = Q.dropna(subset=["g_star_10y"]).copy()
    Qc["bucket"] = pd.cut(Qc["g_star_10y"], bins=BUCKET_EDGES, labels=BUCKET_LABELS, right=False)
    rows = {}
    for b, d in Qc.groupby("bucket", observed=True):
        d1 = d.dropna(subset=["f1"]); d3 = d.dropna(subset=["f3"])
        rows[str(b)] = {
            "기업-분기 (1y)": len(d1), "실현 1y 성장 중앙값 %": (np.expm1(d1["f1"]).median() * 100) if len(d1) else np.nan,
            "1y 달성 비율": (d1["f1"] >= d1["gs1"]).mean() if len(d1) else np.nan,
            "기업-분기 (3y)": len(d3), "실현 3y 연환산 중앙값 %": ((np.exp(d3["f3"] / 3.0) - 1).median() * 100) if len(d3) else np.nan,
            "실현 3y 연환산 p25 %": ((np.exp(d3["f3"] / 3.0) - 1).quantile(.25) * 100) if len(d3) else np.nan,
            "실현 3y 연환산 p75 %": ((np.exp(d3["f3"] / 3.0) - 1).quantile(.75) * 100) if len(d3) else np.nan,
            "3y 달성 비율": (d3["f3"] >= d3["gs3"]).mean() if len(d3) else np.nan}
    p("g*_10y 구간별로 이후 실현 매출성장(연환산)과 '실현 ≥ 요구' 비율. 처음 1~3년에 요구 속도를 못 내면 10년 요구는 더 어렵다(필요조건). "
      "달성 비율이 낮을수록 그 구간의 가격은 평균적으로 과한 기대를 담고 있었다. 단, 달성 못 해도 주가가 내린다는 뜻은 아니다(5절).\n")
    CAL = pd.DataFrame(rows).T
    CAL.index.name = "bucket"
    CAL.reset_index().to_parquet(C.PQ / "expectations_calibration.parquet", index=False)
    p(CAL.round(2).to_markdown())
    say("calibration:\n" + CAL.round(2).to_string())

    p("\n기간별 3년 달성 비율 (요구 시점 기준) — 기저율이 시기에 얼마나 좌우되는지:\n")
    Q3 = Qc.dropna(subset=["f3"]).copy()
    Q3["period"] = pd.cut(Q3["qend"], bins=[pd.Timestamp("2014-01-01"), pd.Timestamp("2017-12-31"), pd.Timestamp("2020-12-31"),
                                             pd.Timestamp("2030-12-31")], labels=["2014~2017", "2018~2020", "2021~2023"])
    Q3["hit"] = (Q3["f3"] >= Q3["gs3"]).astype(float)
    per = Q3.groupby(["bucket", "period"], observed=True)["hit"].agg(["mean", "size"])
    PT = per["mean"].unstack("period").reindex(BUCKET_LABELS)
    PN = per["size"].unstack("period").reindex(BUCKET_LABELS)
    PT.columns = [f"{c} 달성 비율" for c in PT.columns]; PN.columns = [f"{c} 관측치" for c in PN.columns]
    p(pd.concat([PT.round(3), PN], axis=1).to_markdown())
    p("\n결과 연도가 3년 뒤이므로 마지막 기간은 2021~2023년에 요구된 성장의 2024~2026년 실현이다. 기간에 따라 달성 비율이 다르면 "
      "기저율은 시대의 산물이고, 다음 시대에 그대로 적용된다는 보장이 없다.")

    p("\n### 4-2. 조건부 기저율 — 어떤 기업이 요구 성장을 달성했는가\n")
    p("위 표는 요구 성장의 구간만 조건으로 둔다. 같은 요구라도 규모·산업·최근 성장·수익성에 따라 달성 비율이 다르므로, "
      "'실현 3년 매출성장 ≥ 요구 성장'을 요구 성장(log(1+g*)), 로그 시총(표준화), 직전 1년 매출성장, 조정 ROE, FF12 산업에 로짓으로 회귀했다. "
      "적합은 3년 결과가 관측된 모든 기업-분기(표본 내), 적용은 모든 기업-월(최신 월 포함). 종목별 표의 **3y 달성 확률**이 이 값이다. "
      "기저율이지 예측 검정이 아니다.\n")
    cf = COEF.copy()
    cf.index = [{"const": "절편", "req_growth": "요구 성장 log(1+g*)", "log_size": "로그 시총 (표준화)", "past_growth": "직전 1년 매출성장",
                 "roe_adj": "조정 ROE"}.get(i, i.replace("ind_", "산업 ")) for i in cf.index]
    p(cf.round(3).to_markdown())
    for dim, intro in (("규모", "규모 그룹 × 요구 성장 구간별 달성 비율(로짓 없이 그대로 센 값):"),
                       ("직전 성장", "직전 1년 매출성장 그룹 × 요구 성장 구간별 달성 비율 — 이미 빨리 자라던 기업은 무거운 요구를 더 자주 채우는가:")):
        cd = COND[COND["dim"] == dim]
        p(f"\n{intro}\n")
        gorder = {"규모": ["시총 상위 200", "201위 이하"], "직전 성장": ["직전 1년 성장 10% 미만", "10~30%", "30% 이상"]}[dim]
        cw = cd.pivot(index="bucket", columns="group", values="rate").reindex(index=BUCKET_LABELS, columns=gorder)
        cn = cd.pivot(index="bucket", columns="group", values="n").reindex(index=BUCKET_LABELS, columns=gorder)
        cw.columns = [f"{c} 달성 비율" for c in cw.columns]; cn.columns = [f"{c} 관측치" for c in cn.columns]
        p(pd.concat([cw.round(3), cn], axis=1).to_markdown())
    pl = Xl["p_achieve_3y"].dropna()
    if len(pl):
        p(f"\n{last.date()} 종목별 달성 확률: 중앙값 {pl.median()*100:.0f}%, p25 {pl.quantile(.25)*100:.0f}%, p75 {pl.quantile(.75)*100:.0f}%; "
          f"시총 상위 50 중앙값 {Xl.head(50)['p_achieve_3y'].median()*100:.0f}%.")

    p("\n## 5. 검증 — 가격이 요구하는 성장과 펀더멘털 전망의 차이가 수익률을 예측하는가 (탐색적)\n")
    try:
        V = pd.read_parquet(C.PQ / "validation_panel_mom.parquet", columns=["cik", "qend", "r1", "r4", "log_size", "mom"])
        QV = Q.merge(V, on=["cik", "qend"], how="inner")
        rows = {}
        for h, lag in ((1, 1), (4, 4)):
            for xs in (["gap_star_model"], ["gs1"], ["gs1", "gm1"]):
                res, _ = fm(QV, f"r{h}", xs + ["log_size", "mom"], lag)
                rows[f"r{h} ~ {' + '.join(xs)} + 통제"] = {**{f"{c} 계수": res.loc[c, "coef"] for c in xs}, **{f"{c} t": res.loc[c, "t"] for c in xs}, "분기": int(res.loc[xs[0], "quarters"])}
        p("13F 내재가격 분기 수익률(상장폐지 포함), 통제 = 로그 시총, 모멘텀. gap = log(1+g*_3y) − log(1+g_model) (같은 3년 지평), gs1 = log(1+g*_10y).\n")
        p(pd.DataFrame(rows).T.round(3).to_markdown())
        say("return check:\n" + pd.DataFrame(rows).T.round(3).to_string())
    except Exception as ex:  # noqa: BLE001
        p(f"validation panel not available: {ex}")

    p("\n## 6. 읽는 법\n")
    p("- g*_10y, ω*, T*는 '가격이 맞으려면 무엇이 참이어야 하는가'다. 투자자의 판단이 g*보다 높은 성장, ω*·T*보다 긴 지속을 믿으면 매수 근거, 반대면 매도 근거가 된다. 모형은 그 믿음의 옳고 그름을 판정하지 않는다.")
    p("- 표현 규칙: g*는 '시장이 믿는 성장률'이 아니라 '이 가격이 참이 되려면 필요한 성장률'이다. 가격이 일관된 전망의 산물이라는 보장이 없고, "
      "횡단면 프리미엄 분산의 44%가 펀더멘털·산업·수요·모멘텀·성장 전망 어느 것으로도 설명되지 않는 잔차다(단계 3 분해). 조건이므로 분기 실적으로 채점할 수 있다.")
    p("- 4절이 보여주듯 요구 성장은 실현 성장에 대한 정보를 담는다. 4-1절의 달성 비율은 '그 조건이 실제로 충족된 기저율'로, 예측시장 확률과 같은 단위로 읽을 수 있다.")
    p("- 5절이 무의미하면 '가격이 요구하는 성장이 펀더멘털 전망보다 높다'는 사실만으로는 수익률을 예측하지 못한다는 뜻이며, 초과수익은 투자자 자신의 정보에서 나와야 한다.")
    p("- 초대형주는 장부가가 가격의 5~20%라 g*·ω*의 폭이 곧 가치의 폭이다. 점 추정 가치 대신 이 표의 요구 기대를 읽는 것이 맞다.")
    p("- g*는 모형 할인율(국채 + β × ERP)에서의 요구 성장이다. 가격 하나는 성장과 요구수익률의 식 하나이므로, 1-1절의 수익률 축(성장 전망 → 가격이 주는 수익률)으로 "
      "같은 가격을 반대편에서 읽을 수 있다. 자기 요구수익률을 먼저 정하고 그 행을 읽는 것이 맞다.")
    p("- 내재 ERP는 시장 전체 가격에서 거꾸로 구한 값이므로 시장이 비싸면 모든 종목의 g*가 낮아 보인다. g* 평균ERP 열이 그 몫을 보여 준다.")
    p("- 기저율 참고: 시총 상위 5% 기업의 실현 지속성은 0.85~0.89(반감기 4~6년), 5년 뒤 초과이익 잔존 57% (단계 4).")
    (C.REPORTS / "phase7_expectations.md").write_text("\n".join(L), encoding="utf-8")
    say(f"done  {(time.time()-t0)/60:.1f} min -> reports/phase7_expectations.md "
        f"(the per-stock table is written by sfv.report: reports/sfv_latest.csv)")
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
