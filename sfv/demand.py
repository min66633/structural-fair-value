# -*- coding: utf-8 -*-
"""Module D: structural demand and transient flow variables, monthly, point in time.

Structural (slow-moving) demand for a stock
    inst_share      13F institutional holdings value / market cap
    big3_share      BlackRock + Vanguard + State Street value / market cap
    ext_share       big3 + Geode, Northern Trust, Schwab, Invesco, BNY / market cap
    n_holders       number of 13F filers holding the stock
    index_share     N-PORT holdings of index-named funds / market cap (2019-)
    sp500_proxy     held by the Vanguard 500 Index Fund (2019-)
    nsi_12m         12-month log change in split-adjusted shares outstanding
                    (negative = net buybacks, shrinking supply)

Transient flow pressure (Lou 2012, computed from N-PORT)
    fit_q           sum_j pctflow_j,q * value_ij,q-1 / mktcap_i
                    pctflow_j,q = (sales + reinvestment - redemptions over the
                    quarter) / net assets at the previous report; value_ij,q-1
                    the fund's position at its previous report. Dollar-based so
                    no share-basis conversion is needed.
    fit_2q          sum of the latest two quarters' fit

Availability
    13F measures for period q are used from q + 46 days (statutory deadline).
    N-PORT measures for a report are used from its filing date (about 60 days
    after the fund's fiscal quarter end). At month t the latest available
    observation is used, never older than 200 days.

Outputs data/parquet/demand_monthly.parquet, data/parquet/fit_quarterly.parquet
Run: python -m sfv.demand
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

from . import config as C

F13_LAG_DAYS = 46
MAX_AGE = 200
FLOW_WINSOR = (-0.5, 1.0)


def _mktcap_monthly() -> pd.DataFrame:
    U = pd.read_parquet(C.PQ / "universe_monthly.parquet",
                        columns=["cik", "month", "mktcap", "shares", "price_src", "ticker", "in_universe"])
    U["month"] = U["month"] + pd.offsets.MonthEnd(0)
    return U


def cusip_to_cik() -> pd.DataFrame:
    sm = pd.read_parquet(C.PQ / "security_master.parquet", columns=["cusip", "cik", "sec_type"])
    sm = sm[sm["cik"].notna() & (sm["sec_type"] == "common")]
    sm["cik"] = sm["cik"].astype(int)
    return sm[["cusip", "cik"]]


# ----------------------------------------------------------------- 13F structural
def f13_structural(U: pd.DataFrame, link: pd.DataFrame, say) -> pd.DataFrame:
    A = pd.read_parquet(C.PQ / "inst_own_quarterly.parquet",
                        columns=["period", "cusip", "inst_value_usd", "n_holders", "px_med", "big3_shares", "ext_shares",
                                 "big3_complete", "ext_complete"])
    A = A.merge(link, on="cusip", how="inner")
    A["big3_value"] = A["big3_shares"] * A["px_med"]
    A["ext_value"] = (A["big3_shares"] + A["ext_shares"]) * A["px_med"]
    G = (A.groupby(["cik", "period"], as_index=False)
           .agg(inst_value=("inst_value_usd", "sum"), n_holders=("n_holders", "max"),
                big3_value=("big3_value", "sum"), ext_value=("ext_value", "sum"),
                big3_complete=("big3_complete", "min"), ext_complete=("ext_complete", "min")))
    G["month"] = G["period"] + pd.offsets.MonthEnd(0)
    G = G.merge(U[["cik", "month", "mktcap"]], on=["cik", "month"], how="left")
    G["inst_share"] = (G["inst_value"] / G["mktcap"]).clip(0, 1.5)
    G["big3_share"] = (G["big3_value"] / G["mktcap"]).where(G["big3_complete"]).clip(0, 1)
    G["ext_share"] = (G["ext_value"] / G["mktcap"]).where(G["ext_complete"]).clip(0, 1)
    G["f13_avail"] = G["period"] + pd.Timedelta(days=F13_LAG_DAYS)
    say(f"  13F structural: {len(G):,} cik-quarters, {G.cik.nunique():,} ciks, with mktcap {G.mktcap.notna().mean()*100:.1f}%")
    return G[["cik", "period", "f13_avail", "inst_share", "big3_share", "ext_share", "n_holders"]].rename(columns={"period": "f13_period"})


# ----------------------------------------------------------------- net share issuance
def net_issuance(U: pd.DataFrame, say) -> pd.DataFrame:
    """12-month log change in shares, all converted to the latest split basis."""
    P = pd.read_parquet(C.PQ / "prices_monthly.parquet", columns=["ticker", "month", "splits"])
    P["month"] = pd.to_datetime(P["month"]) + pd.offsets.MonthEnd(0)
    P = P.sort_values(["ticker", "month"])
    P["cum"] = np.log(P["splits"].where(P["splits"] > 0, 1.0)).groupby(P["ticker"]).cumsum()
    last = P.groupby("ticker")["cum"].last()
    cs = P.set_index(["ticker", "month"])["cum"]
    S = U[["cik", "month", "shares", "price_src", "ticker"]].copy()
    cum_now = cs.reindex(pd.MultiIndex.from_arrays([S["ticker"], S["month"]])).values
    cum_last = S["ticker"].map(last).values
    conv = np.where(S["price_src"] == "ftd", np.exp(np.nan_to_num(cum_last - cum_now, nan=0.0)), 1.0)
    S["shares_lb"] = S["shares"] * conv
    S = S.sort_values(["cik", "month"])
    prev = S.groupby("cik")["shares_lb"].shift(12)
    prev_m = S.groupby("cik")["month"].shift(12)
    ok = ((S["month"] - prev_m).dt.days.between(350, 380)) & (prev > 0) & (S["shares_lb"] > 0)
    S["nsi_12m"] = np.log(S["shares_lb"] / prev).where(ok).clip(-1, 1)
    say(f"  net issuance: {S.nsi_12m.notna().sum():,} cik-months; median {S.nsi_12m.median():.4f}, p10 {S.nsi_12m.quantile(.1):.3f}, p90 {S.nsi_12m.quantile(.9):.3f}")
    return S[["cik", "month", "nsi_12m"]]


# ----------------------------------------------------------------- N-PORT: index share and FIT
def nport_measures(U: pd.DataFrame, link: pd.DataFrame, say) -> tuple[pd.DataFrame, pd.DataFrame]:
    F = pd.read_parquet(C.PQ / "nport_funds.parquet")
    F = F[(F["net_assets"] > 0) & F["report_date"].notna()].copy()
    F["equity_ratio"] = F["equity_value_usd"] / F["net_assets"]
    F = F[F["equity_ratio"].fillna(0) >= 0.5]                         # equity funds only
    F = F.sort_values(["series_id", "report_date"])
    flow_cols = [f"{k}_m{i}" for k in ("sales", "reinv", "redem") for i in (1, 2, 3)]
    F[flow_cols] = F[flow_cols].fillna(0.0)
    F["net_flow"] = (F[["sales_m1", "sales_m2", "sales_m3"]].sum(axis=1) + F[["reinv_m1", "reinv_m2", "reinv_m3"]].sum(axis=1)
                     - F[["redem_m1", "redem_m2", "redem_m3"]].sum(axis=1))
    g = F.groupby("series_id")
    F["prev_report"] = g["report_date"].shift(1)
    F["prev_na"] = g["net_assets"].shift(1)
    F["prev_accession"] = g["accession"].shift(1)
    gap = (F["report_date"] - F["prev_report"]).dt.days
    F["pctflow"] = (F["net_flow"] / F["prev_na"]).where(gap.between(60, 130) & (F["prev_na"] > 0)).clip(*FLOW_WINSOR)
    F["is_sp500_fund"] = F["series_name"].str.upper().str.contains("VANGUARD 500 INDEX FUND|ISHARES CORE S&P 500", regex=True)
    say(f"  N-PORT equity funds: {F.series_id.nunique():,} series, reports {len(F):,}, with pctflow {F.pctflow.notna().sum():,}; "
        f"pctflow median {F.pctflow.median():.4f}, p10 {F.pctflow.quantile(.1):.3f}, p90 {F.pctflow.quantile(.9):.3f}")
    # The NEXT report's MONTHLY flows (N-PORT items B.6, three calendar months
    # per report) attach to this report's holdings, so flow-induced trading
    # lands in the exact calendar month it happened. Percentage flow uses the
    # previous report's net assets, as the quarterly version does.
    valid = F["pctflow"].notna() & F["prev_accession"].notna()
    rep_p = (F["report_date"] + pd.offsets.MonthEnd(0)).dt.to_period("M")
    rows = []
    for i in (1, 2, 3):
        rows.append(pd.DataFrame({
            "accession": F.loc[valid, "prev_accession"].values,
            "m": (rep_p[valid] - (3 - i)).values,
            "pctflow_m": ((F[f"sales_m{i}"] + F[f"reinv_m{i}"] - F[f"redem_m{i}"]) / F["prev_na"])[valid].clip(-0.3, 0.5).values,
            "next_filed": F.loc[valid, "filed"].values}))
    nxt = pd.concat(rows, ignore_index=True)
    nxt["avail_month"] = (nxt["next_filed"] + pd.offsets.MonthEnd(0)).dt.to_period("M")

    # Funds report on staggered fiscal quarters (a third of them each calendar
    # month), so a per-report-date aggregate sees only a third of the holders.
    # Each fund's latest report is carried forward month by month until its
    # next report is filed (at most MAX_AGE days after the report date), and
    # all funds are summed per calendar month. Done with +v/-v events per
    # (cik, month) and a cumulative sum: expanding fund x stock rows to months
    # would be 100 million rows on this machine.
    nxt_filed = F.groupby("series_id")["filed"].shift(-1)
    win = pd.DataFrame({"accession": F["accession"].values,
                        "start_p": (F["filed"] + pd.offsets.MonthEnd(0)).dt.to_period("M").values,
                        "stop_p": (np.minimum(nxt_filed.fillna(pd.Timestamp("2100-01-01")) - pd.Timedelta(days=1),
                                              F["report_date"] + pd.Timedelta(days=MAX_AGE)) + pd.offsets.MonthEnd(0)).dt.to_period("M").values})
    win = win[win["stop_p"] >= win["start_p"]]

    ev_rows, fit_rows = [], []
    files = sorted((C.PQ / "nport_holdings").glob("*.parquet"))
    t0 = time.time()
    for i, f in enumerate(files, 1):
        H = pd.read_parquet(f, columns=["accession", "series_id", "report_date", "cusip", "value_usd"])
        H = H.merge(link, on="cusip", how="inner")
        H = H.merge(F[["accession", "is_index_name", "is_sp500_fund"]], on="accession", how="inner")
        H["v_idx"] = H["value_usd"].where(H["is_index_name"].astype(bool), 0.0)
        H["v_sp"] = H["value_usd"].where(H["is_sp500_fund"].astype(bool), 0.0)
        Hw = H.merge(win, on="accession", how="inner")
        on = Hw.groupby(["cik", "start_p"], as_index=False)[["value_usd", "v_idx", "v_sp"]].sum().rename(columns={"start_p": "p"})
        on["n"] = Hw.groupby(["cik", "start_p"]).size().values
        off = Hw.assign(p=Hw["stop_p"] + 1).groupby(["cik", "p"], as_index=False)[["value_usd", "v_idx", "v_sp"]].sum()
        off["n"] = -Hw.groupby(["cik", Hw["stop_p"] + 1]).size().values
        off[["value_usd", "v_idx", "v_sp"]] *= -1.0
        ev_rows.append(pd.concat([on, off], ignore_index=True))
        # FIT contributions: this report's positions x the fund's monthly percentage flows next quarter
        Hn = H.merge(nxt, on="accession", how="inner")
        Hn["contrib"] = Hn["value_usd"] * Hn["pctflow_m"]
        fit = Hn.groupby(["cik", "m", "avail_month"], as_index=False).agg(fit_usd=("contrib", "sum"), n_funds=("accession", "nunique"))
        fit_rows.append(fit)
        del H, Hw, Hn
        if i % 9 == 0 or i == len(files):
            say(f"    {i}/{len(files)} holdings files  {(time.time()-t0)/60:.1f} min")

    EV = pd.concat(ev_rows, ignore_index=True)
    EV = EV.groupby(["cik", "p"], as_index=False)[["value_usd", "v_idx", "v_sp", "n"]].sum()
    EV = EV.sort_values(["cik", "p"])
    # cumulative sum on a complete monthly grid per cik. Iterate the groupby
    # (one pass) rather than filtering EV per cik, which rescans the whole
    # event table for every firm and dominated the runtime of this module.
    cap = pd.Period(U["month"].max(), "M")
    parts = []
    for cik, e in EV.groupby("cik", sort=False):
        lo, hi = e["p"].min(), min(e["p"].max(), cap)
        if hi < lo:
            continue
        idx = pd.period_range(lo, hi, freq="M")
        ee = e.set_index("p")[["value_usd", "v_idx", "v_sp", "n"]].reindex(idx, fill_value=0.0).cumsum()
        ee["cik"] = cik
        parts.append(ee.reset_index().rename(columns={"index": "p"}))
    IX = pd.concat(parts, ignore_index=True)
    IX["month"] = IX["p"].dt.to_timestamp("M")
    IX = IX.rename(columns={"value_usd": "all_fund_value", "v_idx": "idx_value", "v_sp": "sp_value", "n": "n_reports"})
    IX = IX[IX["n_reports"] > 0]
    IX = IX.merge(U[["cik", "month", "mktcap"]], on=["cik", "month"], how="left")
    IX["index_share"] = (IX["idx_value"] / IX["mktcap"]).clip(0, 1)
    IX["fund_share"] = (IX["all_fund_value"] / IX["mktcap"]).clip(0, 1.5)
    IX["sp500_proxy"] = (IX["sp_value"] > 0).astype(float)
    say(f"  index share: {len(IX):,} cik-months; median index_share {IX.index_share.median():.3f}, fund_share {IX.fund_share.median():.3f} "
        f"(with mktcap {IX.mktcap.notna().mean()*100:.1f}%)")

    # FIT by calendar flow month m, known from avail_month. fit_q(t) sums the
    # flow months t-4..t-2 that are known by t (reports are public about two
    # months after a fund's quarter end), so the matching price-impact window
    # is the return over [t-5, t-2].
    FIT = pd.concat(fit_rows, ignore_index=True)
    FIT = FIT.groupby(["cik", "m", "avail_month"], as_index=False).agg(fit_usd=("fit_usd", "sum"), n_funds=("n_funds", "sum"))
    FQ = FIT.rename(columns={"m": "flow_month", "avail_month": "fit_avail"}).copy()
    FQ["flow_month"] = FQ["flow_month"].dt.to_timestamp("M")
    FQ["fit_avail"] = FQ["fit_avail"].dt.to_timestamp("M")
    FIT = FIT.loc[FIT.index.repeat(3)].copy()
    kk = FIT.groupby(level=0).cumcount()
    FIT["month"] = (FIT["m"] + 2 + kk.values)
    FIT = FIT[FIT["avail_month"] <= FIT["month"]]
    FM = FIT.groupby(["cik", "month"], as_index=False).agg(fit_usd=("fit_usd", "sum"), n_funds=("n_funds", "sum"))
    FM["month"] = FM["month"].dt.to_timestamp("M")
    # scale by market cap five months earlier (the start of the flow window)
    FM["month0"] = (FM["month"].dt.to_period("M") - 5).dt.to_timestamp("M")
    FM = FM.merge(U[["cik", "month", "mktcap"]].rename(columns={"month": "month0", "mktcap": "mktcap0"}), on=["cik", "month0"], how="left")
    FM["fit_q"] = (FM["fit_usd"] / FM["mktcap0"]).clip(-0.5, 0.5)
    FM = FM.sort_values(["cik", "month"])
    prev3 = FM.groupby("cik")["fit_q"].shift(3)
    prev3_m = FM.groupby("cik")["month"].shift(3)
    ok3 = (FM["month"] - prev3_m).dt.days.between(80, 100)
    FM["fit_2q"] = FM["fit_q"] + prev3.where(ok3, 0.0)
    say(f"  FIT: {len(FM):,} cik-months, {FM.cik.nunique():,} ciks, {FM.month.min().date()}..{FM.month.max().date()}; "
        f"fit_q median {FM.fit_q.median():.4f}, p10 {FM.fit_q.quantile(.1):.4f}, p90 {FM.fit_q.quantile(.9):.4f}, sd {FM.fit_q.std():.4f}")
    return IX[["cik", "month", "index_share", "fund_share", "sp500_proxy", "n_reports"]], FM[["cik", "month", "fit_usd", "fit_q", "fit_2q", "n_funds", "mktcap0"]], FQ


# ----------------------------------------------------------------- assemble monthly
def assemble(U: pd.DataFrame, S13: pd.DataFrame, NSI: pd.DataFrame, IX: pd.DataFrame, FIT: pd.DataFrame, say) -> pd.DataFrame:
    D = U[U["in_universe"]][["cik", "month", "mktcap"]].sort_values("month")
    D = pd.merge_asof(D, S13.sort_values("f13_avail"), left_on="month", right_on="f13_avail", by="cik", direction="backward")
    stale = (D["month"] - D["f13_period"]).dt.days > MAX_AGE
    D.loc[stale, ["inst_share", "big3_share", "ext_share", "n_holders"]] = np.nan
    D = D.merge(NSI, on=["cik", "month"], how="left")
    D = D.merge(IX[["cik", "month", "index_share", "fund_share", "sp500_proxy", "n_reports"]], on=["cik", "month"], how="left")
    D = D.merge(FIT[["cik", "month", "fit_q", "fit_2q", "n_funds"]], on=["cik", "month"], how="left")
    D = D.sort_values(["cik", "month"]).reset_index(drop=True)
    cov = D[["inst_share", "big3_share", "ext_share", "nsi_12m", "index_share", "sp500_proxy", "fit_q"]].notna().mean()
    say("  monthly coverage (share of in-universe cik-months):\n" + (cov * 100).round(1).to_string())
    return D


def main() -> int:
    log = open(C.LOGS / "demand.log", "w", encoding="utf-8")

    def say(msg):
        print(msg, flush=True); log.write(msg + "\n"); log.flush()

    t0 = time.time()
    U = _mktcap_monthly()
    link = cusip_to_cik()
    S13 = f13_structural(U, link, say)
    NSI = net_issuance(U, say)
    IX, FIT, FQ = nport_measures(U, link, say)
    FQ.to_parquet(C.PQ / "fit_quarterly.parquet", index=False)
    D = assemble(U, S13, NSI, IX, FIT, say)
    D.to_parquet(C.PQ / "demand_monthly.parquet", index=False)
    say(f"done: {len(D):,} cik-months  {(time.time()-t0)/60:.1f} min")
    yr = D.assign(y=D["month"].dt.year).groupby("y").agg(
        inst=("inst_share", "median"), big3=("big3_share", "median"), ext=("ext_share", "median"),
        index=("index_share", "median"), sp500=("sp500_proxy", "mean"), nsi=("nsi_12m", "median"),
        fit_q_sd=("fit_q", "std"), n=("cik", "nunique"))
    say("medians by year:\n" + yr.round(4).to_string())
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
