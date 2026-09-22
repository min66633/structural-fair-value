# -*- coding: utf-8 -*-
"""Monthly investable universe: one row per (cik, month) with point-in-time market cap.

Links the three worlds built in phase 0:
    fundamentals  by CIK      (fund_quarterly, shares_dei)
    ownership     by CUSIP    (13F, N-PORT)     via security_master
    prices        by ticker   (yfinance)  and by CUSIP (FTD monthly last price)

Rules
  * One primary common-stock CUSIP/ticker per CIK: the common-type security
    with the latest last_date, then the most trading days. Other common
    classes of the same issuer are counted (n_common_classes) so multi-class
    firms can be flagged; their market cap uses the primary class price
    times TOTAL shares, which is within a few percent for the usual cases.
  * Price at month end: yfinance close when the primary ticker is listed,
    else the FTD last price of the month (delisted names). Source recorded.
  * Shares are point in time: the latest cover-page count (dei) FILED on or
    before the month end, split-adjusted forward with yfinance split events
    between the cover date and the month end (NVIDIA's 10:1 in June 2024
    would otherwise shrink its market cap 10x until the August 10-Q).
    Fallback: balance-sheet shares_out from the latest available panel row.
  * Fundamentals: the latest panel row with avail_date <= month end.
  * Sanity: market cap / total assets outside [0.01, 200] marks a probable
    mis-link or a shell and is excluded from the universe (counted).
  * in_universe = price present, market cap >= $300m, non-financial
    (SIC 6000-6999), common stock, fundamentals within 400 days, sanity ok.

Output  data/parquet/universe_monthly.parquet
Run: python -m sfv.universe
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

from . import config as C

MIN_MCAP = 300e6
FUND_MAX_AGE = 400          # days between avail_date and month end
# Bitcoin miners turned data-centre operators that the SEC files under SIC 6199 (finance services,
# n.e.c.). Not financial firms; kept in the universe despite the 6000-6999 rule (2026-09-12).
NONFINANCIAL_OVERRIDES = {1878848: "IREN", 1964789: "HUT", 1083301: "WULF", 1167419: "RIOT", 1819989: "CIFR",
                          1839341: "CORZ", 1507605: "MARA", 827876: "CLSK", 1720424: "HIVE", 2042022: "WYFI",
                          1710350: "BTBT", 1755953: "ABTC", 1899123: "BTDR"}
SHARES_MAX_AGE = 400
METHOD_RANK = {"ticker-current": 0, "name-exact": 1, "cusip6-propagated": 2, "name-fuzzy": 3}


def sic_to_ff12(sic) -> str:
    """Fama-French 12 industries from SIC (French's definitions, hard-coded ranges)."""
    if pd.isna(sic):
        return "Other"
    s = int(sic)
    rng = {
        "NoDur": [(100, 999), (2000, 2399), (2700, 2749), (2770, 2799), (3100, 3199), (3940, 3989)],
        "Durbl": [(2500, 2519), (2590, 2599), (3630, 3659), (3710, 3711), (3714, 3714), (3716, 3716),
                  (3750, 3751), (3792, 3792), (3900, 3939), (3990, 3999)],
        "Manuf": [(2520, 2589), (2600, 2699), (2750, 2769), (3000, 3099), (3200, 3569), (3580, 3629),
                  (3700, 3709), (3712, 3713), (3715, 3715), (3717, 3749), (3752, 3791), (3793, 3799),
                  (3830, 3839), (3860, 3899)],
        "Enrgy": [(1200, 1399), (2900, 2999)],
        "Chems": [(2800, 2829), (2840, 2899)],
        "BusEq": [(3570, 3579), (3660, 3692), (3694, 3699), (3810, 3829), (7370, 7379)],
        "Telcm": [(4800, 4899)],
        "Utils": [(4900, 4949)],
        "Shops": [(5000, 5999), (7200, 7299), (7600, 7699)],
        "Hlth": [(2830, 2839), (3693, 3693), (3840, 3859), (8000, 8099)],
        "Money": [(6000, 6999)],
    }
    for name, ranges in rng.items():
        for lo, hi in ranges:
            if lo <= s <= hi:
                return name
    return "Other"


def primary_securities(sm: pd.DataFrame, say) -> pd.DataFrame:
    s = sm[(sm["sec_type"] == "common") & sm["cik"].notna()].copy()
    s["cik"] = s["cik"].astype(int)
    s["mrank"] = s["method"].map(METHOD_RANK).fillna(9)
    # Primary = among the securities still trading near the CIK's latest date, the one with the most
    # trading days. "Latest last_date first" picked Duke's and Southern's exchange-listed baby bonds
    # (DUKB, SOJD: a few hundred days, typed common by the source data) over the common stock (5,000+
    # days) because the bond's last print was two days later (2026-09-13).
    s["recent"] = (s.groupby("cik")["last_date"].transform("max") - s["last_date"]).dt.days <= 120
    s = s.sort_values(["cik", "recent", "n_days", "last_date"], ascending=[True, False, False, False])
    prim = s.drop_duplicates("cik").drop(columns=["recent"]).copy()
    # other common classes still alive within a year of the primary's last date
    alive = s.merge(prim[["cik", "last_date"]].rename(columns={"last_date": "prim_last"}), on="cik")
    alive = alive[alive["last_date"] >= alive["prim_last"] - pd.Timedelta(days=365)]
    ncls = alive.groupby("cik")["cusip"].nunique().rename("n_common_classes")
    prim = prim.merge(ncls, on="cik", how="left")
    # A ticker claimed by several CIKs is fine when their securities' active
    # periods do not overlap (Google Inc -> Alphabet kept GOOGL across a
    # holding-company reorganisation); with overlap, keep the best-linked,
    # most-traded claimant.
    prim = prim.sort_values(["ticker", "mrank", "n_days"], ascending=[True, True, False])
    best_first = prim.groupby("ticker")["first_date"].transform("first")
    best_last = prim.groupby("ticker")["last_date"].transform("first")
    overlap = (np.minimum(prim["last_date"], best_last) - np.maximum(prim["first_date"], best_first)).dt.days > 180
    dup = prim.duplicated("ticker", keep="first") & prim["ticker"].notna() & overlap
    say(f"  primary securities: {len(prim):,} ciks; multi-class {int((prim.n_common_classes > 1).sum()):,}; "
        f"tickers claimed twice with overlapping periods dropped {int(dup.sum())}; "
        f"shared across non-overlapping periods {int((prim.duplicated('ticker', keep=False) & ~overlap).sum())}")
    prim = prim[~dup]
    return prim[["cik", "cusip", "cusip6", "ticker", "method", "score", "n_common_classes", "first_date", "last_date", "n_days"]]


def month_grid(prim: pd.DataFrame, months: pd.DatetimeIndex, say) -> pd.DataFrame:
    """(cik, month) grid restricted to each CIK's filing span.

    Without the restriction a predecessor and its successor CIK would both
    carry the shared ticker's price for the whole sample; with it, Google
    Inc covers 2009-2015 and Alphabet 2015 onwards.
    """
    Q = pd.read_parquet(C.PQ / "fund_quarterly.parquet", columns=["cik", "end", "avail_date"])
    span = Q.groupby("cik").agg(first_end=("end", "min"), last_avail=("avail_date", "max")).reset_index()
    G = pd.MultiIndex.from_product([prim["cik"].unique(), months], names=["cik", "month"]).to_frame(index=False)
    G = G.merge(span, on="cik", how="left")
    # Grace after the last filing. It was one quarter (120 days), which silently dropped
    # every firm whose SEC companyfacts record lags its actual filings - in 2026-09 about
    # 12% of the universe (Coca-Cola, Abbott, NextEra, Duke ...) had a Q1 record in a
    # September bulk file although their Q2 10-Qs were filed in July. The fundamentals
    # rule (has_fund: age <= FUND_MAX_AGE) already decides when a stale record is too old,
    # so the grid uses the same limit. A predecessor CIK that could carry its successor's
    # ticker for longer is handled by the ticker-month dedup below, which keeps the CIK
    # with the fresher fundamentals (Google Inc vs Alphabet in 2016).
    lo = G["first_end"] - pd.Timedelta(days=90)
    hi = G["last_avail"] + pd.Timedelta(days=FUND_MAX_AGE)
    G = G[(G["month"] >= lo) & (G["month"] <= hi)].drop(columns=["first_end", "last_avail"])
    say(f"  grid restricted to filing spans: {len(G):,} cik-months")
    return G.merge(prim, on="cik", how="left")


def attach_prices(G: pd.DataFrame, say) -> pd.DataFrame:
    P = pd.read_parquet(C.PQ / "prices_monthly.parquet", columns=["ticker", "month", "close", "adj_close", "splits", "volume"])
    P["month"] = pd.to_datetime(P["month"]) + pd.offsets.MonthEnd(0)
    G = G.merge(P, on=["ticker", "month"], how="left")
    F = pd.read_parquet(C.PQ / "ftd_prices_monthly.parquet")
    F["month"] = pd.to_datetime(F["month"]) + pd.offsets.MonthEnd(0)
    F = F.rename(columns={"price_last": "px_ftd"})
    G = G.merge(F[["cusip", "month", "px_ftd"]], on=["cusip", "month"], how="left")
    G["price"] = G["close"].where(G["close"].notna(), G["px_ftd"])
    G["price_src"] = np.select([G["close"].notna(), G["px_ftd"].notna()], ["yf", "ftd"], default=None)
    say(f"  price coverage: yf {G.close.notna().mean()*100:.1f}%  ftd-only {(G.close.isna() & G.px_ftd.notna()).mean()*100:.1f}%  "
        f"none {G.price.isna().mean()*100:.1f}% of cik-months")
    return G


def clean_dei(D: pd.DataFrame, say) -> pd.DataFrame:
    """Drop cover-page share counts that are scale typos.

    Early XBRL cover pages sometimes carry the count in the wrong unit
    (Yum 2016-10: 3.67e14 for 3.67e8; Chubb 2010; AEP 2010-11; Oracle
    2012-09). Each is an isolated observation a million times off, with sane
    neighbours. Rule: an observation that differs from BOTH neighbours by
    more than 3x is dropped; the last observation of a series is dropped if it
    differs from its predecessor by more than 100x (a real 3x change without
    a split is rare, a 100x one does not happen).
    """
    return _drop_scale_typos(D, "shares_dei", ["cik", "filed", "asof"], say, "cover-page")


def _drop_scale_typos(D: pd.DataFrame, col: str, order: list[str], say, label: str) -> pd.DataFrame:
    """Drop isolated scale typos in a per-cik share series.

    A value is a typo when it is more than 3x away from the centred rolling
    median of its neighbours (window 5, so two consecutive typos - Chubb
    2010Q1 and 2010Q2 - are still outvoted by three sane neighbours). Series
    with fewer than 3 observations cannot be judged and are kept.
    """
    D = D.sort_values(order).copy()
    x = np.log(D[col])
    med = x.groupby(D["cik"]).transform(lambda s: s.rolling(5, center=True, min_periods=3).median())
    bad = ((x - med).abs() > np.log(3.0)) & med.notna()
    say(f"  {label} cleaning: dropped {int(bad.sum()):,} of {len(D):,} counts as scale typos")
    return D[~bad]


def attach_shares(G: pd.DataFrame, say) -> pd.DataFrame:
    D = pd.read_parquet(C.PQ / "shares_dei.parquet")
    D = D[(D["shares_dei"] > 0)]
    D["filed"] = pd.to_datetime(D["filed"])
    D = clean_dei(D, say).sort_values("filed")
    G = G.sort_values("month")
    G = pd.merge_asof(G, D[["cik", "filed", "asof", "shares_dei"]].rename(columns={"filed": "dei_filed", "asof": "dei_asof"}),
                      left_on="month", right_on="dei_filed", by="cik", direction="backward")
    stale = (G["month"] - G["dei_asof"]).dt.days > SHARES_MAX_AGE
    G.loc[stale, ["shares_dei", "dei_asof", "dei_filed"]] = np.nan
    # Cover-page counts are in the share basis of their cover date; the two
    # price sources are in different bases and each needs its own conversion:
    #   yfinance close  = split-adjusted to the LATEST basis (NVIDIA May 2024
    #                     prints $109.63, post-split) -> shares x all splits
    #                     after the cover date, including future ones
    #   FTD last price  = the actual price at the time -> shares x splits
    #                     between the cover date and the month only
    P = pd.read_parquet(C.PQ / "prices_monthly.parquet", columns=["ticker", "month", "splits"])
    P["month"] = pd.to_datetime(P["month"]) + pd.offsets.MonthEnd(0)
    P = P.sort_values(["ticker", "month"])
    P["logsplit"] = np.log(P["splits"].where(P["splits"] > 0, 1.0))
    P["cumsplit"] = P.groupby("ticker")["logsplit"].cumsum()
    cs = P.set_index(["ticker", "month"])["cumsplit"]
    cum_last = P.groupby("ticker")["cumsplit"].last()
    G["cum_now"] = cs.reindex(pd.MultiIndex.from_arrays([G["ticker"], G["month"]])).values
    G["cum_last"] = G["ticker"].map(cum_last)

    def conv_factor(ref_date: pd.Series) -> np.ndarray:
        """Shares dated ref_date -> basis of the price source (latest for yf, month for FTD)."""
        ref_m = (pd.to_datetime(ref_date) + pd.offsets.MonthEnd(0))
        cum_ref = cs.reindex(pd.MultiIndex.from_arrays([G["ticker"], ref_m])).values
        # a reference month before the price history starts: no split info, take the first known level
        f_yf = np.exp((G["cum_last"] - cum_ref).fillna(0.0))
        f_ftd = np.exp((G["cum_now"] - cum_ref).fillna(0.0))
        return np.where(G["price_src"] == "yf", f_yf, f_ftd)

    factor = conv_factor(G["dei_asof"])
    G["split_factor"] = factor
    G["shares"] = G["shares_dei"] * factor
    G["shares_src"] = np.where(G["shares"].notna(), "dei", None)
    # balance-sheet shares (cleaned the same way; PG&E's are a million times
    # off while its cover page is right) are only a fallback when no cover
    # count is available
    Q = pd.read_parquet(C.PQ / "fund_quarterly.parquet", columns=["cik", "end", "avail_date", "shares_out"])
    Q = Q[Q["shares_out"] > 0].dropna(subset=["avail_date"])
    Q = _drop_scale_typos(Q, "shares_out", ["cik", "end"], say, "balance-sheet").sort_values("avail_date")
    G = pd.merge_asof(G.sort_values("month"), Q.rename(columns={"avail_date": "so_avail", "end": "so_end"}),
                      left_on="month", right_on="so_avail", by="cik", direction="backward")
    stale2 = (G["month"] - G["so_end"]).dt.days > SHARES_MAX_AGE
    G.loc[stale2, "shares_out"] = np.nan
    so_adj = G["shares_out"] * conv_factor(G["so_end"])     # dated by its own balance-sheet date
    G["shares_out_adj"] = so_adj
    G["shares_dei_adj"] = G["shares_dei"] * factor
    use_bs = G["shares"].isna() & G["shares_out"].notna()
    G.loc[use_bs, "shares"] = so_adj[use_bs]
    G.loc[use_bs, "shares_src"] = "balance_sheet"
    # third fallback: basic weighted-average shares of the latest quarter (the
    # EPS denominator, reported by every filer and summed across share
    # classes - multi-class issuers such as Meta tag cover-page and
    # balance-sheet counts per class, which companyfacts drops)
    W = pd.read_parquet(C.PQ / "fund_quarterly.parquet", columns=["cik", "end", "avail_date", "q_shares_basic_wavg"])
    W = W[W["q_shares_basic_wavg"] > 0].dropna(subset=["avail_date"])
    W = _drop_scale_typos(W, "q_shares_basic_wavg", ["cik", "end"], say, "weighted-average").sort_values("avail_date")
    G = pd.merge_asof(G.sort_values("month"), W.rename(columns={"avail_date": "wa_avail", "end": "wa_end"}),
                      left_on="month", right_on="wa_avail", by="cik", direction="backward")
    stale3 = (G["month"] - G["wa_end"]).dt.days > SHARES_MAX_AGE
    G.loc[stale3, "q_shares_basic_wavg"] = np.nan
    use_wa = G["shares"].isna() & G["q_shares_basic_wavg"].notna()
    G.loc[use_wa, "shares"] = (G["q_shares_basic_wavg"] * conv_factor(G["wa_end"]))[use_wa]
    G.loc[use_wa, "shares_src"] = "wavg_basic"
    say(f"  shares coverage: dei {(G.shares_src == 'dei').mean()*100:.1f}%  balance-sheet {(G.shares_src == 'balance_sheet').mean()*100:.1f}%  "
        f"wavg {(G.shares_src == 'wavg_basic').mean()*100:.1f}%  split-converted rows {(factor != 1.0).sum():,}")
    return G.drop(columns=["cum_now", "cum_last", "so_avail", "so_end", "wa_avail", "wa_end", "q_shares_basic_wavg"])


def attach_fundamentals(G: pd.DataFrame, say) -> pd.DataFrame:
    Q = pd.read_parquet(C.PQ / "fund_quarterly.parquet",
                        columns=["cik", "end", "avail_date", "assets", "equity", "ttm_revenue", "ttm_net_income"])
    Q = Q.dropna(subset=["avail_date"]).sort_values("avail_date")
    G = pd.merge_asof(G.sort_values("month"), Q.rename(columns={"end": "fund_end", "avail_date": "fund_avail"}),
                      left_on="month", right_on="fund_avail", by="cik", direction="backward")
    G["fund_age"] = (G["month"] - G["fund_end"]).dt.days
    G["has_fund"] = G["fund_age"] <= FUND_MAX_AGE
    E = pd.read_parquet(C.PQ / "sec_entities.parquet", columns=["cik", "sic", "state_inc", "name"])
    G = G.merge(E, on="cik", how="left")
    G["ff12"] = G["sic"].map(sic_to_ff12)
    # ... and their industry is compute infrastructure (BusEq, where CoreWeave and Applied Digital sit),
    # not Money: the fade target and the industry effects come from the FF12 group, and a Money group
    # made only of loss-making miners would drag their target ROE below zero.
    G.loc[G["cik"].isin(list(NONFINANCIAL_OVERRIDES)), "ff12"] = "BusEq"
    # SIC 6000-6999 is financial, except the bitcoin miners turned data-centre operators that the
    # SEC files under 6199 "finance services, n.e.c.". They run power and compute, not balance
    # sheets, so the blanket rule would wrongly drop them (2026-09-12). Operators only:
    # digital-asset treasury companies and exchanges under 6199 stay excluded.
    G["is_financial"] = G["sic"].between(6000, 6999) & ~G["cik"].isin(list(NONFINANCIAL_OVERRIDES))
    G["foreign_inc"] = G["state_inc"].fillna("").str.match(r"^[A-Z][0-9]$")     # SEC codes for foreign countries
    say(f"  fundamentals within {FUND_MAX_AGE}d: {G.has_fund.mean()*100:.1f}% of cik-months; financial {G.is_financial.mean()*100:.1f}%")
    return G


def main() -> int:
    log = open(C.LOGS / "universe.log", "w", encoding="utf-8")

    def say(msg):
        print(msg, flush=True); log.write(msg + "\n"); log.flush()

    t0 = time.time()
    sm = pd.read_parquet(C.PQ / "security_master.parquet")
    prim = primary_securities(sm, say)
    P = pd.read_parquet(C.PQ / "prices_monthly.parquet", columns=["month"])
    last_month = pd.to_datetime(P["month"]).max() + pd.offsets.MonthEnd(0)
    months = pd.date_range("2010-01-31", last_month, freq="ME")
    G = month_grid(prim, months, say)
    say(f"  grid: {len(G):,} cik-months, {len(months)} months to {last_month.date()}")
    G = attach_prices(G, say)
    G = attach_shares(G, say)
    G = attach_fundamentals(G, say)
    G["mktcap"] = G["price"] * G["shares"]
    G["mcap_assets"] = G["mktcap"] / G["assets"]
    # cross-source guard: when the chosen share count gives an absurd market
    # cap / assets ratio and the alternative source gives a sane one, the
    # chosen one is a typo the time-series rule missed (AEP's combined
    # utility filings carry several entities' cover counts under one CIK)
    alt = np.where(G["shares_src"] == "dei", G["shares_out_adj"], G["shares_dei_adj"])
    alt_ratio = G["price"] * alt / G["assets"]
    swap = (~G["mcap_assets"].between(0.01, 200)) & pd.Series(alt_ratio, index=G.index).between(0.01, 200)
    G.loc[swap, "shares"] = alt[swap]
    G.loc[swap, "shares_src"] = G.loc[swap, "shares_src"].map({"dei": "balance_sheet_x", "balance_sheet": "dei_x", "wavg_basic": "dei_x"})
    G.loc[swap, "mktcap"] = G.loc[swap, "price"] * G.loc[swap, "shares"]
    G.loc[swap, "mcap_assets"] = G.loc[swap, "mktcap"] / G.loc[swap, "assets"]
    say(f"  cross-source share swaps: {int(swap.sum()):,} rows")
    # continuity guard: a month whose ratio is still absurd but whose CIK had
    # a sane count within the previous six months reuses that count (same
    # split basis is assumed - splits and typos rarely coincide)
    G = G.sort_values(["cik", "month"])
    sane = G["mcap_assets"].between(0.01, 200)
    last_ok_sh = G["shares"].where(sane).groupby(G["cik"]).ffill()
    last_ok_m = G["month"].where(sane).groupby(G["cik"]).ffill()
    carry = (~sane) & G["assets"].notna() & G["price"].notna() & last_ok_sh.notna() \
        & ((G["month"] - last_ok_m).dt.days <= 190)
    G.loc[carry, "shares"] = last_ok_sh[carry]
    G.loc[carry, "shares_src"] = "carry_forward"
    G.loc[carry, "mktcap"] = G.loc[carry, "price"] * G.loc[carry, "shares"]
    G.loc[carry, "mcap_assets"] = G.loc[carry, "mktcap"] / G.loc[carry, "assets"]
    say(f"  carry-forward share fixes: {int(carry.sum()):,} rows")
    G["sanity_ok"] = G["mcap_assets"].between(0.01, 200) | G["assets"].isna()
    G["in_universe"] = (G["price"].notna() & (G["mktcap"] >= MIN_MCAP) & ~G["is_financial"].fillna(False)
                        & G["has_fund"].fillna(False) & G["sanity_ok"] & G["assets"].notna())
    keep = ["cik", "month", "cusip", "cusip6", "ticker", "method", "score", "n_common_classes", "price", "price_src",
            "adj_close", "volume", "shares", "shares_src", "split_factor", "mktcap", "mcap_assets", "sanity_ok", "assets", "equity",
            "ttm_revenue", "ttm_net_income", "fund_end", "fund_avail", "fund_age", "has_fund", "sic", "ff12",
            "is_financial", "foreign_inc", "state_inc", "name", "in_universe"]
    G = G[keep].sort_values(["cik", "month"]).reset_index(drop=True)
    G = G[G["price"].notna() | G["in_universe"]]
    # a shared ticker in the hand-over months: keep the CIK with the fresher fundamentals
    G = G.sort_values(["ticker", "month", "fund_end"], ascending=[True, True, False])
    dup = G.duplicated(["ticker", "month"], keep="first") & G["ticker"].notna()
    say(f"  ticker-month duplicates resolved: {int(dup.sum()):,}")
    G = G[~dup].sort_values(["cik", "month"]).reset_index(drop=True)
    G.to_parquet(C.PQ / "universe_monthly.parquet", index=False)

    U = G[G["in_universe"]]
    # the CIK list the universe defines, so targeted extracts (the RPO tags in
    # sfv.xbrl_extract --cik-file) follow the universe instead of a stale copy
    U[["cik"]].drop_duplicates().sort_values("cik").to_parquet(C.PQ / "universe_ciks.parquet", index=False)
    say(f"done: rows with price {len(G):,}; in-universe cik-months {len(U):,}; "
        f"distinct ciks in universe {U.cik.nunique():,}  {(time.time()-t0)/60:.1f} min")
    yr = U.assign(y=U["month"].dt.year).groupby("y").agg(
        firms=("cik", "nunique"), mcap_tn=("mktcap", lambda s: s.sum() / 12 / 1e12),
        yf_share=("price_src", lambda s: (s == "yf").mean()), dei_share=("shares_src", lambda s: (s == "dei").mean()))
    say("in-universe firms per year (mcap_tn = average monthly total, $tn):\n" + yr.round(3).to_string())
    bad = G[G["price"].notna() & (G["mktcap"] >= MIN_MCAP) & ~G["sanity_ok"]]
    say(f"sanity exclusions (mcap>=300m but mcap/assets outside [0.01,200]): {bad.cik.nunique():,} ciks, {len(bad):,} months")
    if len(bad):
        say(bad.sort_values("mktcap", ascending=False).drop_duplicates("cik").head(10)[
            ["cik", "name", "ticker", "month", "mktcap", "assets", "mcap_assets", "method"]].to_string())
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
