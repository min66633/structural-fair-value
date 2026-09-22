# -*- coding: utf-8 -*-
"""Capitalise internally generated intangibles (Peters & Taylor 2017 rules).

GAAP expenses R&D and the organisation-building part of SG&A as they are
paid, so the book value of an intangible-intensive firm understates the
capital it operates and its ROE overstates the return on that capital. The
fix used here is the standard perpetual-inventory one:

    knowledge capital     G_t = (1 - d_R) G_{t-1} + R&D_t          d_R by industry
    organisation capital  O_t = (1 - d_O) O_{t-1} + 0.30 SG&A_t    d_O = 20%/yr

d_R is not one number. Knowledge decays at 11% a year in motor vehicles and 54%
in computer hardware, and the rates below come from Li and Hall's estimates for
the BEA R&D satellite account, matched to SIC codes by their own correspondence
table. Industries they do not cover keep the traditional 15% benchmark; those
firms do little R&D, so almost all of their intangible capital is organisational
anyway. Use --rnd-rates flat --d-rnd 0.15 for the old single-rate behaviour.
    K_int = G + O
    B_adj = B + K_int
    E_adj = E + (R&D_t + 0.30 SG&A_t) - (d_R G_{t-1} + d_O O_{t-1})

run at quarterly frequency with quarterly depreciation rates
1 - (1-d)^(1/4). The opening stock uses the steady-state approximation
G_0 = first annual R&D / (g + d) with g = 10%, as Peters-Taylor do for firms
entering their sample; XBRL starts in 2009 so every firm enters that way.

Acquired intangibles and goodwill are already on the balance sheet and are
not added again. SG&A is taken as reported; the us-gaap element excludes
R&D when R&D is reported separately. When a firm reports no SG&A line but
selling/marketing and G&A separately, their sum is used.

Missing flow quarters are treated as zero investment for the recursion and
counted; firms with no R&D and no SG&A at all get K_int = 0 (not NaN), so
they stay comparable.

Input   data/parquet/fund_quarterly.parquet
Output  data/parquet/intangibles_quarterly.parquet  one row per (cik, end)
Run: python -m sfv.intangibles
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

D_RND, D_ORG = 0.15, 0.20         # annual depreciation, overridable on the command line
SGA_SHARE = 0.30                  # share of SG&A capitalised
G_PRE = 0.10                      # assumed pre-sample growth for the opening stock

# Knowledge depreciates at very different speeds across industries: a drug
# patent outlives a chip design by a decade, and a single rate hides that.
#
# Source: Li, Wendy C.Y. and Bronwyn H. Hall, "Depreciation of Business R&D
# Capital", Review of Income and Wealth (2020); BEA working paper WP2016-5,
# Table 1 (industry to SIC correspondence) and Table 3 (nonlinear least squares
# estimates). Two estimate columns are reported and both are offered here:
#   compustat  publicly traded firms, 1989-2008, n = 5,847 firms
#   bea        BEA-NSF establishment survey, 1987-2007
# The Compustat column is the matched sample for this project, which values
# publicly traded firms, so it is the default.
#
# Aerospace and motor vehicles are the two exceptions. Their headline estimates
# (34% to 88%) come from assuming an 8.9% ex ante return that the paper itself
# argues does not hold there - US autos earned negative returns over the period
# and aerospace margins ran near 1% - so it re-estimates both at a 1% return and
# reports 6% to 16%. Those re-estimates are the values used here, as the paper
# recommends. The paper also notes R&D disclosure quality is poor in both.
#
# Order matters: later entries override earlier ones where SIC ranges overlap,
# which is how the enumerated communication-equipment codes (3661, 3663, 3669,
# 3679) win over the semiconductor range that also contains them.
RND_RATES: list[tuple[str, list, float, float]] = [
    ("Pharmaceuticals",             [2830, 2831, (2833, 2836)],                  0.151, 0.112),
    ("Computers and peripherals",   [(3570, 3579), (3680, 3689), 3695],          0.535, 0.363),
    ("Semiconductors",              [(3661, 3666), (3669, 3679)],                0.293, 0.226),
    ("Communication equipment",     [3576, 3661, 3663, 3669, 3679],              0.308, 0.192),
    ("Aerospace products and parts", [3720, 3721, 3724, 3728, 3760],             0.159, 0.063),
    ("Motor vehicles and parts",    [3585, 3711, (3713, 3716)],                  0.110, 0.119),
    ("Navigational and measuring instruments",
     [3812, 3822, 3823, 3825, 3826, 3829, 3842, 3844, 3845],                     0.472, 0.329),
    ("Software",                    [7372],                                      0.350, 0.308),
    ("Computer system design",      [7370, 7371, 7373],                          0.314, 0.489),
    ("Scientific research and development services", [8731],                     0.326, 0.295),
]
# Everything outside those ten R&D-intensive industries keeps the traditional
# 15% benchmark. Those firms do little R&D, so the choice barely matters: the
# median firm's intangible capital is almost entirely organisation capital.
D_RND_DEFAULT = D_RND


def sic_rnd_rates(source: str = "compustat") -> dict[int, float]:
    """SIC code -> annual R&D depreciation rate, expanded from the Li-Hall table."""
    if source not in ("compustat", "bea"):
        raise ValueError(f"unknown rate source {source!r}; use 'compustat' or 'bea'")
    col = 2 if source == "compustat" else 3
    out: dict[int, float] = {}
    for row in RND_RATES:
        rate = row[col]
        for code in row[1]:
            lo, hi = code if isinstance(code, tuple) else (code, code)
            for s in range(lo, hi + 1):
                out[s] = rate
    return out


# Ewens, Peters and Wang, "Measuring Intangible Capital with Market Prices",
# Management Science 71(1) 2025. They identify the capitalisation parameters
# from what acquirers actually paid at firm exits, and publish them per SIC code
# at github.com/michaelewens/Intangible-capital-stocks (October 2023 vintage,
# mirrored in data/raw/intangibles/). Three parameters, by Fama-French 5 industry:
#
#   knowDepr   R&D depreciation        0.33 (health) to 0.50 (manufacturing)
#   gamma      share of SG&A that is   0.20 (consumer) to 0.51 (health)
#              investment, not expense
#   organDepr  organisation capital    0.20 everywhere - they do NOT estimate
#              depreciation            this by industry, they fix it
#
# That last line is the answer to "can organisation capital depreciation vary by
# industry": the paper that goes furthest on these parameters holds it at the
# same 20% this project already used. What does vary, and by a lot, is how much
# of SG&A counts as investment at all.
EPW_FILE = C.RAW / "intangibles" / "ewens_peters_wang_2023.csv"


def epw_params() -> pd.DataFrame:
    """SIC-indexed (knowDepr, organDepr, gamma) from Ewens-Peters-Wang."""
    if not EPW_FILE.exists():
        raise FileNotFoundError(
            f"{EPW_FILE} not found. Download capital_accum_parameters_2023.csv from "
            "https://github.com/michaelewens/Intangible-capital-stocks")
    d = pd.read_csv(EPW_FILE)
    d = d.dropna(subset=["sic"]).astype({"sic": "int64"}).drop_duplicates("sic")
    return d.set_index("sic")[["knowDepr", "organDepr", "gamma", "industry5"]]


def q_rate(annual: float) -> float:
    """Quarterly depreciation equivalent to an annual rate."""
    return 1 - (1 - annual) ** 0.25

COLS = ["cik", "end", "qend", "avail_date", "fy", "fp", "ttm_ok",
        "q_rnd", "q_sga", "q_selling_marketing", "q_general_admin", "q_revenue", "q_net_income",
        "q_op_income", "equity", "assets", "goodwill", "intangibles",
        "ttm_revenue", "ttm_net_income", "ttm_rnd", "ttm_sga", "q_dividends_paid", "q_buybacks",
        "q_equity_issuance", "ttm_dividends_paid", "ttm_buybacks", "ttm_equity_issuance"]


def perpetual_inventory(inv: np.ndarray, dq: float, d_annual: float) -> tuple[np.ndarray, np.ndarray]:
    """Stock and amortisation paths for one firm from a quarterly investment series."""
    n = len(inv)
    K = np.zeros(n)
    A = np.zeros(n)
    if n == 0:
        return K, A
    first = np.argmax(inv > 0) if (inv > 0).any() else None
    if first is None:
        return K, A
    # opening stock: annualised first observed investment / (g + d)
    K_prev = 4.0 * inv[first] / (G_PRE + d_annual)
    for t in range(n):
        if t < first:
            K[t] = 0.0
            continue
        A[t] = dq * K_prev
        K[t] = K_prev - A[t] + inv[t]
        K_prev = K[t]
    return K, A


def firm_params(P: pd.DataFrame, params: str, d_rnd: float, d_org: float, sga_share: float, say) -> pd.DataFrame:
    """Per-CIK (d_rnd, d_org, gamma). Firms with no SIC match keep the flat defaults."""
    idx = pd.Index(P["cik"].unique(), name="cik")
    out = pd.DataFrame({"d_rnd": d_rnd, "d_org": d_org, "gamma": sga_share}, index=idx)
    if params == "flat":
        say(f"  parameters: R&D {d_rnd:.0%}/yr, organisation {d_org:.0%}/yr, SG&A capitalised {sga_share:.0%} (flat)")
        return out
    S = pd.read_parquet(C.PQ / "universe_monthly.parquet", columns=["cik", "sic"]).dropna(subset=["sic"])
    sic = S.drop_duplicates("cik").set_index("cik")["sic"].astype("int64").reindex(idx)
    if params == "epw":
        E = epw_params()
        out["d_rnd"] = sic.map(E["knowDepr"]).fillna(d_rnd)
        out["d_org"] = sic.map(E["organDepr"]).fillna(d_org)
        out["gamma"] = sic.map(E["gamma"]).fillna(sga_share)
        ind = sic.map(E["industry5"])
        hit = sic.map(E["gamma"]).notna()
        say(f"  parameters: Ewens-Peters-Wang by SIC — {int(hit.sum()):,} of {len(idx):,} firms matched")
        for name, g in pd.DataFrame({"ind": ind, **out}).dropna(subset=["ind"]).groupby("ind"):
            say(f"    {name:12s} n {len(g):5d}  R&D {g.d_rnd.iloc[0]:.0%}  organisation {g.d_org.iloc[0]:.0%}  SG&A share {g.gamma.iloc[0]:.0%}")
    else:                                    # li-hall R&D rates, conventional organisation parameters
        table = sic_rnd_rates(params)
        out["d_rnd"] = sic.map(table).fillna(d_rnd)
        hit = sic.map(table).notna()
        say(f"  parameters: Li-Hall {params} R&D rates by SIC — {int(hit.sum()):,} of {len(idx):,} firms matched, "
            f"rest {d_rnd:.0%}; organisation {d_org:.0%}/yr, SG&A capitalised {sga_share:.0%}")
    return out


def build(P: pd.DataFrame, say, d_rnd: float = D_RND, d_org: float = D_ORG,
          sga_share: float = SGA_SHARE, params: str = "flat") -> pd.DataFrame:
    P = P.sort_values(["cik", "end"]).reset_index(drop=True)
    # scale typos in flows: a quarter's R&D or SG&A larger than three times
    # both total assets and annual revenue is not a number a firm can spend
    # (one filer's 2025 SG&A came through as $8.2tn and dominated the
    # aggregate). Such quarters are dropped from the flows and counted.
    scale = np.maximum(P["assets"].fillna(0), P["ttm_revenue"].fillna(0)) * 3.0
    scale = scale.where(scale > 0, np.inf)
    for c in ("q_rnd", "q_sga", "q_selling_marketing", "q_general_admin"):
        bad = P[c].abs() > scale
        if bad.any():
            say(f"  {c}: {int(bad.sum()):,} quarters dropped as scale typos")
        P.loc[bad, c] = np.nan
    sga = P["q_sga"]
    alt = P["q_selling_marketing"].fillna(0) + P["q_general_admin"].fillna(0)
    sga = sga.where(sga.notna(), alt.where(alt > 0))
    FP = firm_params(P, params, d_rnd, d_org, sga_share, say)
    gamma = P["cik"].map(FP["gamma"]).to_numpy(float)
    inv_rnd = P["q_rnd"].clip(lower=0).fillna(0.0).to_numpy()
    inv_org = (gamma * sga.clip(lower=0).fillna(0.0).to_numpy())
    P["inv_rnd_q"], P["inv_org_q"] = inv_rnd, inv_org
    P["flow_missing"] = P["q_rnd"].isna() & sga.isna()

    # gaps: a quarter missing in the sequence breaks the recursion; the
    # recursion runs on the observed rows and a >200-day gap is flagged
    gap = P.groupby("cik")["end"].diff().dt.days
    P["gap_flag"] = gap > 200

    K_r = np.zeros(len(P)); A_r = np.zeros(len(P)); K_o = np.zeros(len(P)); A_o = np.zeros(len(P))
    t0 = time.time()
    rates = FP.to_dict("index")
    for i, (cik, idx) in enumerate(P.groupby("cik").indices.items()):
        r = rates.get(cik, {"d_rnd": d_rnd, "d_org": d_org})
        dr, do = float(r["d_rnd"]), float(r["d_org"])
        kr, ar = perpetual_inventory(inv_rnd[idx], q_rate(dr), dr)
        ko, ao = perpetual_inventory(inv_org[idx], q_rate(do), do)
        K_r[idx], A_r[idx], K_o[idx], A_o[idx] = kr, ar, ko, ao
        if (i + 1) % 3000 == 0:
            say(f"  {i+1:,} firms  {(time.time()-t0)/60:.1f} min")
    P["k_rnd"], P["k_org"] = K_r, K_o
    P["k_int"] = P["k_rnd"] + P["k_org"]
    P["amort_q"] = A_r + A_o
    P["inv_int_q"] = P["inv_rnd_q"] + P["inv_org_q"]

    P["b_gaap"] = P["equity"]
    P["b_adj"] = P["equity"] + P["k_int"]
    P["e_q_gaap"] = P["q_net_income"]
    P["e_q_adj"] = P["q_net_income"] + P["inv_int_q"] - P["amort_q"]

    g = P.groupby("cik", sort=False)
    span_ok = (P["end"] - g["end"].shift(3)).dt.days.between(255, 295)
    for src, dst in (("e_q_gaap", "ttm_e_gaap"), ("e_q_adj", "ttm_e_adj"), ("inv_int_q", "ttm_inv_int"), ("amort_q", "ttm_amort")):
        P[dst] = g[src].rolling(4, min_periods=4).sum().reset_index(level=0, drop=True).where(span_ok)
    # ROE on beginning-of-year book (four quarters back); undefined when book <= 0
    b_gaap_lag = g["b_gaap"].shift(4)
    b_adj_lag = g["b_adj"].shift(4)
    year_ok = (P["end"] - g["end"].shift(4)).dt.days.between(340, 390)
    P["roe_gaap"] = (P["ttm_e_gaap"] / b_gaap_lag).where(year_ok & (b_gaap_lag > 0))
    P["roe_adj"] = (P["ttm_e_adj"] / b_adj_lag).where(year_ok & (b_adj_lag > 0))
    P["int_intensity"] = (P["k_int"] / P["b_adj"]).where(P["b_adj"] > 0)
    P["rnd_intensity"] = (P["ttm_rnd"] / P["ttm_revenue"]).where(P["ttm_revenue"] > 0)
    return P


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--d-rnd", type=float, default=D_RND, help="annual R&D depreciation (default 0.15)")
    ap.add_argument("--d-org", type=float, default=D_ORG, help="annual organisation-capital depreciation (default 0.20)")
    ap.add_argument("--sga-share", type=float, default=SGA_SHARE, help="share of SG&A capitalised (default 0.30)")
    ap.add_argument("--params", choices=["epw", "compustat", "bea", "flat"], default="epw",
                    help="epw (default): Ewens-Peters-Wang, all three parameters by SIC from market prices. "
                         "compustat/bea: Li-Hall R&D rates by SIC with conventional organisation parameters. "
                         "flat: one rate everywhere from --d-rnd/--d-org/--sga-share")
    ap.add_argument("--out", default="intangibles_quarterly.parquet")
    a_ = ap.parse_args(argv)
    tag = Path(a_.out).stem.replace("intangibles_quarterly", "").lstrip("_") or "base"
    log = open(C.LOGS / f"intangibles_{tag}.log", "w", encoding="utf-8")

    def say(msg):
        print(msg, flush=True); log.write(msg + "\n"); log.flush()

    t0 = time.time()
    P = pd.read_parquet(C.PQ / "fund_quarterly.parquet", columns=COLS)
    P = P[P["end"] >= "2009-01-01"]
    say(f"panel rows {len(P):,}  ciks {P.cik.nunique():,}")
    P = build(P, say, a_.d_rnd, a_.d_org, a_.sga_share, a_.params)
    P.to_parquet(C.PQ / a_.out, index=False)
    say(f"done: {len(P):,} rows  {(time.time()-t0)/60:.1f} min")
    r = P[P["ttm_ok"] & (P["b_adj"] > 0)]
    say(f"coverage among rows with book>0: roe_gaap {r.roe_gaap.notna().mean()*100:.1f}%  roe_adj {r.roe_adj.notna().mean()*100:.1f}%  "
        f"flow_missing {P.flow_missing.mean()*100:.1f}%")
    say("intangible intensity K_int/B_adj by year (value-weighted by B_adj, firms with book>0):")
    yr = r.assign(y=r["qend"].dt.year).groupby("y").apply(
        lambda d: pd.Series({"firms": d.cik.nunique(), "int_intensity_vw": np.average(d.int_intensity, weights=d.b_adj),
                             "roe_gaap_med": d.roe_gaap.median(), "roe_adj_med": d.roe_adj.median()}), include_groups=False)
    say(yr.round(3).to_string())
    for cik, nm in [(320193, "Apple"), (78003, "Pfizer"), (789019, "Microsoft"), (104169, "Walmart"), (1318605, "Tesla")]:
        a = P[P.cik == cik].sort_values("end").tail(1)
        if len(a):
            a = a.iloc[0]
            say(f"  {nm:10s} {a.end.date()}  B {a.b_gaap/1e9:8.1f}bn  K_int {a.k_int/1e9:7.1f}bn (R&D {a.k_rnd/1e9:6.1f}, org {a.k_org/1e9:6.1f})  "
                f"ROE gaap {a.roe_gaap:6.3f}  adj {a.roe_adj:6.3f}  E_ttm gaap {a.ttm_e_gaap/1e9:7.1f}  adj {a.ttm_e_adj/1e9:7.1f}")
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
