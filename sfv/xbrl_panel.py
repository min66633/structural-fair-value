# -*- coding: utf-8 -*-
"""Long XBRL facts -> point-in-time quarterly fundamentals panel.

The problem this solves: XBRL facts come as (start, end, value) with mixed
durations. A 10-Q for Q3 reports the quarter AND the nine months; a 10-K
reports only the year. Quarterly flows therefore have to be recovered, and
every value must carry the date it first became public.

Rules
  * Only FIRST-reported values are used (which in {first, only}). A later
    restatement never leaks backwards. The restated value is kept in a
    parallel column (`*_last`) for the core items so the size of revisions
    can be measured, but nothing downstream uses it by default.
  * Concept resolution: for one (cik, concept, start, end) the EARLIEST-filed
    tag wins (point in time); tag priority (order in concepts.CONCEPTS) only
    breaks ties within the same filing. The restated column takes the
    latest-filed tag the same way.
  * Quarterly flows: a direct ~3-month fact is used when present. Otherwise
    the quarter is the difference of two cumulative facts with the SAME
    start date whose ends are one quarter apart (Y - 9M gives Q4 from the
    10-K; H - Q1 gives Q2; ...). Same-start differencing is exact and does
    not depend on guessing fiscal calendars.
  * Availability: avail_date = latest `filed` among the core items of the
    row (net income, equity, revenue, assets). For a derived quarter it is
    the later of its two components. Downstream code may use a row only at
    dates >= avail_date.
  * Period key: the fiscal period end `end`, plus `qend` = nearest calendar
    quarter end for cross-sectional grouping (52/53-week filers end on
    e.g. 09-27; Walmart's 01-31 maps to 12-31).
  * TTM: sum of four consecutive quarters whose ends span 255-295 days
    (three quarter gaps; 52/53-week calendars allowed).

Inputs   data/parquet/xbrl_raw/part-*.parquet  (whole CIKs per part)
Outputs  data/parquet/fund_quarterly.parquet   one row per (cik, end)
         data/parquet/shares_dei.parquet       cover-page shares outstanding, dated
         reports/logs/xbrl_panel.log
Run: python -m sfv.xbrl_panel [--parts N]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .concepts import CONCEPTS, DURATION_CONCEPTS, INSTANT_CONCEPTS, TAG_TO_CONCEPTS

CORE = ["net_income", "equity", "revenue", "assets"]
UNIT_FOR = {}
for _c in CONCEPTS:
    if _c.startswith("eps") or _c == "dps_declared":
        UNIT_FOR[_c] = "USD/shares"
    elif _c.startswith("shares") or _c in ("dei_shares_out",):
        UNIT_FOR[_c] = "shares"
    else:
        UNIT_FOR[_c] = "USD"

# duration classes in days
Q_LO, Q_HI = 60, 110


def _nearest_qend(d: pd.Series) -> pd.Series:
    """Nearest calendar quarter end to each date (vectorised)."""
    d = pd.to_datetime(d)
    prev_q = d + pd.offsets.QuarterEnd(0)          # this or next quarter end (>= d)
    prev_q2 = prev_q - pd.offsets.QuarterEnd(1)     # previous quarter end (< d unless d is a qend)
    use_prev = (d - prev_q2).abs() < (prev_q - d).abs()
    return pd.Series(np.where(use_prev, prev_q2, prev_q), index=d.index).astype("datetime64[ns]")


def resolve_concepts(raw: pd.DataFrame) -> pd.DataFrame:
    """Tag rows -> concept rows with priority resolution. Returns first-reported values."""
    raw = raw[raw["which"].isin(["first", "only", "last"])].copy()
    # explode tag -> concepts
    raw["concepts"] = raw["tag"].map(TAG_TO_CONCEPTS)
    raw = raw.dropna(subset=["concepts"]).explode("concepts").rename(columns={"concepts": "concept"})
    raw["prio"] = [CONCEPTS[c]["tags"].index(t) for c, t in zip(raw["concept"], raw["tag"])]
    raw = raw[raw["unit"] == raw["concept"].map(UNIT_FOR)]
    first = raw[raw["which"].isin(["first", "only"])]
    last = raw[raw["which"].isin(["last", "only"])]
    key = ["cik", "concept", "start", "end"]
    # POINT-IN-TIME RULE: the first-reported value of a concept is the one
    # filed EARLIEST across all of its tags; tag priority only breaks ties
    # inside the same filing. Ordering by priority first re-dated Alphabet's
    # 2024 quarters to 2025, because the 2025 10-Q re-tagged the comparative
    # period with a higher-priority tag than the 2024 10-Q had used.
    first = first.sort_values(key + ["filed", "prio"]).drop_duplicates(key, keep="first")
    last = (last.sort_values(key + ["filed", "prio"], ascending=[True] * len(key) + [False, True])
                .drop_duplicates(key, keep="first"))
    first = first.merge(last[key + ["val"]].rename(columns={"val": "val_last"}), on=key, how="left")
    return first[key + ["val", "val_last", "filed", "fy", "fp", "form", "tag"]]


# Duration concepts that are averages or ratios, not sums: a fiscal-year
# value minus a nine-month value is meaningless for them (Meta's derived Q4
# 2025 weighted-average share count came out as 1,000,000). Only directly
# reported quarters are used; the missing fourth quarter stays NaN.
NON_ADDITIVE = {"shares_basic_wavg", "shares_diluted_wavg", "eps_basic", "eps_diluted", "dps_declared"}


def quarterly_flows(D: pd.DataFrame) -> pd.DataFrame:
    """Duration facts -> one value per (cik, concept, end) quarter."""
    D = D.dropna(subset=["start", "end"]).copy()
    D["days"] = (D["end"] - D["start"]).dt.days
    direct = D[D["days"].between(Q_LO, Q_HI)].copy()
    direct["method"] = "direct"
    # for averages, the fiscal-year value stands in for the fourth quarter
    # when no quarterly figure was reported (10-Ks report only the annual
    # weighted-average share count); ranked below a direct quarter
    annual = D[D["concept"].isin(NON_ADDITIVE) & D["days"].between(340, 380)].copy()
    annual["method"] = "annual"

    # same-start differencing: consecutive ends within the same cumulative series
    D = D[~D["concept"].isin(NON_ADDITIVE)]
    D = D.sort_values(["cik", "concept", "start", "end"])
    g = D.groupby(["cik", "concept", "start"], sort=False)
    prev_end = g["end"].shift(1)
    prev_val = g["val"].shift(1)
    prev_val_last = g["val_last"].shift(1)
    prev_filed = g["filed"].shift(1)
    gap = (D["end"] - prev_end).dt.days
    ok = gap.between(Q_LO, Q_HI) & prev_val.notna()
    derived = D[ok].copy()
    derived["val"] = D.loc[ok, "val"] - prev_val[ok]
    derived["val_last"] = D.loc[ok, "val_last"] - prev_val_last[ok]
    derived["filed"] = np.maximum(D.loc[ok, "filed"], prev_filed[ok])
    derived["method"] = "derived"
    derived["days"] = gap[ok]

    Q = pd.concat([direct, derived, annual], ignore_index=True)
    # prefer direct over derived/annual; among duplicates keep the earliest filed
    Q["m_rank"] = (Q["method"] != "direct").astype(int)
    Q = Q.sort_values(["cik", "concept", "end", "m_rank", "filed"]).drop_duplicates(["cik", "concept", "end"], keep="first")
    return Q[["cik", "concept", "end", "val", "val_last", "filed", "fy", "fp", "form", "method"]]


def instants(D: pd.DataFrame) -> pd.DataFrame:
    D = D[D["start"].isna()].copy()
    D = D.sort_values(["cik", "concept", "end", "filed"]).drop_duplicates(["cik", "concept", "end"], keep="first")
    return D[["cik", "concept", "end", "val", "val_last", "filed", "fy", "fp", "form"]]


def build_panel(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    R = resolve_concepts(raw)
    dur = R[R["concept"].isin(DURATION_CONCEPTS)]
    ins = R[R["concept"].isin(INSTANT_CONCEPTS)]
    Q = quarterly_flows(dur)
    I = instants(ins)

    # dei shares outstanding is dated by cover date, not period end: keep aside
    dei = I[I["concept"] == "dei_shares_out"][["cik", "end", "val", "filed", "form"]].rename(
        columns={"end": "asof", "val": "shares_dei"})
    I = I[I["concept"] != "dei_shares_out"]

    def widen(T: pd.DataFrame, prefix: str) -> pd.DataFrame:
        v = T.pivot_table(index=["cik", "end"], columns="concept", values="val", aggfunc="first")
        v.columns = [f"{prefix}{c}" for c in v.columns]
        f = T[T["concept"].isin(CORE)].pivot_table(index=["cik", "end"], columns="concept", values="filed", aggfunc="first")
        f.columns = [f"filed_{c}" for c in f.columns]
        vl = T[T["concept"].isin(CORE)].pivot_table(index=["cik", "end"], columns="concept", values="val_last", aggfunc="first")
        vl.columns = [f"{prefix}{c}_last" for c in vl.columns]
        return v.join(f, how="left").join(vl, how="left")

    W = widen(Q, "q_").join(widen(I, ""), how="outer")
    meta = (pd.concat([Q[["cik", "end", "fy", "fp", "form", "method"]].assign(src="q"),
                       I[["cik", "end", "fy", "fp", "form"]].assign(method="instant", src="i")])
              .sort_values(["cik", "end", "src"]).drop_duplicates(["cik", "end"], keep="first")
              .set_index(["cik", "end"]))
    W = W.join(meta[["fy", "fp", "form"]], how="left")
    W = W.reset_index()
    W["qend"] = _nearest_qend(W["end"])
    filed_cols = [c for c in W.columns if c.startswith("filed_")]
    W["avail_date"] = W[filed_cols].max(axis=1)
    W["n_core"] = W[[f"q_{c}" if c in DURATION_CONCEPTS else c for c in CORE]].notna().sum(axis=1)
    W = W[W["n_core"] >= 2]                       # at least two core items -> a real reporting period
    W = W.sort_values(["cik", "end"]).reset_index(drop=True)

    # TTM sums for duration concepts over 4 consecutive quarters. Four quarter
    # ENDS span three gaps, i.e. ~273 days (255-295 allows 52/53-week years).
    qcols = [c for c in W.columns if c.startswith("q_") and not c.endswith("_last")]
    grp = W.groupby("cik", sort=False)
    span = (W["end"] - grp["end"].shift(3)).dt.days
    ok = span.between(255, 295)
    for c in qcols:
        s = grp[c].rolling(4, min_periods=4).sum().reset_index(level=0, drop=True)
        W["ttm_" + c[2:]] = s.where(ok)
    W["ttm_ok"] = ok.fillna(False)
    return W, dei


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", type=int, default=0, help="process only the first N parts (test)")
    a = ap.parse_args(argv)
    log = open(C.LOGS / "xbrl_panel.log", "w", encoding="utf-8")

    def say(msg):
        print(msg, flush=True); log.write(msg + "\n"); log.flush()

    parts = sorted((C.PQ / "xbrl_raw").glob("part-*.parquet"))
    if a.parts:
        parts = parts[:a.parts]
    say(f"parts: {len(parts)}")
    t0 = time.time()
    panels, deis = [], []
    for i, p in enumerate(parts, 1):
        raw = pd.read_parquet(p)
        raw = raw[raw["end"] >= C.XBRL_START]
        if raw.empty:
            continue
        W, dei = build_panel(raw)
        panels.append(W); deis.append(dei)
        if i % 10 == 0 or i == len(parts):
            say(f"  {i}/{len(parts)}  rows {sum(len(x) for x in panels):,}  {(time.time()-t0)/60:.1f} min")
    P = pd.concat(panels, ignore_index=True)
    Dei = pd.concat(deis, ignore_index=True).sort_values(["cik", "filed", "asof"])
    P.to_parquet(C.PQ / "fund_quarterly.parquet", index=False)
    Dei.to_parquet(C.PQ / "shares_dei.parquet", index=False)
    say(f"done: panel rows {len(P):,}  ciks {P.cik.nunique():,}  "
        f"ends {P.end.min().date()} .. {P.end.max().date()}  dei rows {len(Dei):,}  {(time.time()-t0)/60:.1f} min")
    cov = P[["q_revenue", "q_net_income", "equity", "assets", "q_rnd", "q_sga", "q_cfo", "q_buybacks",
             "q_dividends_paid", "shares_out", "ttm_net_income"]].notna().mean().round(3)
    say("coverage (share of rows with value):\n" + cov.to_string())
    lag = (P["avail_date"] - P["end"]).dt.days
    say(f"availability lag days: median {lag.median():.0f}, p90 {lag.quantile(.9):.0f}")
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
