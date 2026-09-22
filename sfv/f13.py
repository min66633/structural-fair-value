# -*- coding: utf-8 -*-
"""13F structured data sets -> institutional and passive ownership per CUSIP-quarter.

Input: the quarterly SEC 13F structured-data zips in data/raw/13f_bulk (2013Q2 onward; see data/README.md).
Each zip holds the filings MADE in a window, so one zip mixes several report
periods (late filings, amendments). Filings are therefore first pooled across
all zips into one table, de-duplicated per (manager CIK, period), and only
then are holdings aggregated.

De-duplication rule: per (CIK, period) keep the filing with the most holdings
rows (ties -> latest filed). A "restatement" amendment re-lists everything and
wins; a "new holdings" amendment lists only additions and loses to the
original. That is the same rule the existing macro pipeline uses, and the
same approximation: additions-only amendments are dropped rather than merged.

Unit handling: the reported VALUE switched from thousands to dollars with the
2022Q4 period, but filers disagreed on both sides of the boundary, so units
are decided PER FILING from the filing's own median implied price
(VALUE / shares). A median under $1 means thousands. This is the rule that
made independent filers agree to 0.01% in the existing pipeline.

Passive ownership: shares held by a fixed list of index-fund complexes,
identified by manager CIK. The "big three" (BlackRock, Vanguard, State
Street) is the standard definition in the literature; an extended set adds
the other large index managers. Both are lower bounds for true passive
ownership (Chinco & Sammon 2024) and both include some active assets run by
the same complexes; the counts are reported, not hidden.

Outputs
    data/parquet/f13_filings.parquet        one row per kept filing
    data/parquet/inst_own_quarterly.parquet one row per (period, cusip)
    reports/logs/f13.log

Run: python -m sfv.f13 [--limit-zips N]
"""
from __future__ import annotations

import argparse
import re
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

EQUITY_CUSIP = re.compile(r"^[0-9A-Z]{6}[0-9][0-9A-Z][0-9]$")

# Manager complexes treated as passive. Names are matched on FILINGMANAGER_NAME
# (upper-cased, whitespace-normalised) and the matched CIKs are printed so the
# assignment can be audited in the log.
PASSIVE_PATTERNS = {
    "big3": [
        r"^VANGUARD GROUP",
        # Every BlackRock filer. Verified in the 2016Q3 data set: until 2016
        # the group filed as seven separate managers (Institutional Trust,
        # Fund Advisors, Group Ltd, Advisors LLC, Investment Management,
        # Japan, and "BlackRock Inc." with only its own small book) whose
        # tables do not overlap - Apple: 144m + 70m + 48m + 16m + 16m + 8m +
        # 12m = 313m shares = 5.7%, the known stake. From 2017 the parent
        # consolidated everything and the subsidiaries stopped filing, and
        # from 2024Q3 the filer is "BLACKROCK, INC." (CIK 2012383). Matching
        # the prefix covers all three regimes without double counting.
        r"^BLACKROCK",
        r"^STATE STREET CORP",
    ],
    "ext": [
        r"^GEODE CAPITAL MANAGEMENT",   # Fidelity index funds
        r"^NORTHERN TRUST CORP",
        r"^CHARLES SCHWAB INVESTMENT MANAGEMENT",
        r"^INVESCO LTD",                # QQQ and Invesco index products (mixed)
        r"^BANK OF NEW YORK MELLON CORP",
    ],
}


def _member(z: zipfile.ZipFile, name: str):
    for n in z.namelist():
        if n.rsplit("/", 1)[-1].upper() == name.upper():
            return z.open(n)
    raise KeyError(name)


def _read(z, name, usecols, dtype=None):
    return pd.read_csv(_member(z, name), sep="\t", usecols=usecols, dtype=dtype,
                       low_memory=False, encoding="utf-8", encoding_errors="replace")


def list_zips() -> list[Path]:
    zips = sorted(C.F13_CACHE.glob("*.zip"))
    if not zips:
        raise FileNotFoundError(f"no 13F zips in {C.F13_CACHE}")
    return zips


def build_filings(zips: list[Path], say) -> pd.DataFrame:
    """Pool cover/submission/summary pages across zips; de-duplicate per (CIK, period)."""
    rows = []
    for p in zips:
        z = zipfile.ZipFile(p)
        cov = _read(z, "COVERPAGE.tsv", ["ACCESSION_NUMBER", "REPORTCALENDARORQUARTER",
                                         "ISAMENDMENT", "AMENDMENTTYPE", "FILINGMANAGER_NAME"],
                    dtype=str)
        sub = _read(z, "SUBMISSION.tsv", ["ACCESSION_NUMBER", "CIK", "FILING_DATE",
                                          "SUBMISSIONTYPE"], dtype=str)
        smy = _read(z, "SUMMARYPAGE.tsv", ["ACCESSION_NUMBER", "TABLEENTRYTOTAL",
                                           "TABLEVALUETOTAL"], dtype=str)
        d = cov.merge(sub, on="ACCESSION_NUMBER", how="left").merge(smy, on="ACCESSION_NUMBER", how="left")
        d["src"] = p.name
        rows.append(d)
    F = pd.concat(rows, ignore_index=True)
    F["period"] = pd.to_datetime(F["REPORTCALENDARORQUARTER"], errors="coerce",
                                 format="mixed", dayfirst=True)
    F["filed"] = pd.to_datetime(F["FILING_DATE"], errors="coerce", format="mixed", dayfirst=True)
    F["cik"] = pd.to_numeric(F["CIK"], errors="coerce").astype("Int64")
    F["n_entries"] = pd.to_numeric(F["TABLEENTRYTOTAL"], errors="coerce")
    F["manager"] = F["FILINGMANAGER_NAME"].fillna("").str.upper().str.replace(r"\s+", " ", regex=True).str.strip()
    F = F.dropna(subset=["period", "cik"])
    F = F[F["period"] >= C.F13_START]
    # a handful of cover pages carry a non-quarter-end report date (e.g.
    # 2020-08-14); they cannot be aligned and are dropped, counted here
    qe = F["period"] + pd.offsets.QuarterEnd(0)
    stray = (qe - F["period"]).dt.days > 3
    if stray.any():
        say_stray = int(stray.sum())
        F = F[~stray]
        print(f"  dropped {say_stray} filings with non-quarter-end report dates", flush=True)
    # 13F-NT is a notice that another manager reports the holdings; no table
    F = F[~F["SUBMISSIONTYPE"].fillna("").str.upper().str.startswith("13F-NT")]
    n0 = len(F)
    F = F.sort_values(["cik", "period", "n_entries", "filed"])
    F = F.drop_duplicates(subset=["cik", "period"], keep="last")
    say(f"  filings pooled {n0:,} -> kept {len(F):,} after per-(CIK, period) de-duplication; "
        f"periods {F.period.min().date()} .. {F.period.max().date()} ({F.period.nunique()})")
    return F


def assign_passive(F: pd.DataFrame, say) -> pd.DataFrame:
    """Tag manager CIKs as big3 / ext passive by name pattern; print what matched."""
    F = F.copy()
    F["passive_group"] = ""
    for group, pats in PASSIVE_PATTERNS.items():
        rx = re.compile("|".join(pats))
        hit = F["manager"].str.match(rx)
        F.loc[hit & (F["passive_group"] == ""), "passive_group"] = group
    for group in ("big3", "ext"):
        sub = F[F["passive_group"] == group]
        summary = (sub.groupby(["cik", "manager"]).size().reset_index(name="n_filings")
                   .sort_values("n_filings", ascending=False))
        say(f"  passive group {group}: {sub.cik.nunique()} CIKs")
        for _, r in summary.iterrows():
            say(f"     cik {int(r.cik):>8}  {r.manager[:50]:50s} filings {int(r.n_filings)}")
    return F


CHUNK = 400_000


def aggregate_zip(p: Path, keep: pd.DataFrame, say) -> pd.DataFrame | None:
    """Aggregate one zip's INFOTABLE over kept filings -> per (period, cusip) partial sums.

    Streams the table in chunks and collapses each chunk to (filing, cusip)
    sums immediately: an INFOTABLE is up to 400 MB of TSV and this machine
    leaves about 3 GB for a job, so the whole-table read used before was
    killed by the memory watchdog. Partial sums for a filing split across
    chunks add up exactly, so the result is identical to the one-shot path.
    """
    z = zipfile.ZipFile(p)
    keep_acc = set(keep["ACCESSION_NUMBER"])
    parts = []
    try:
        it = pd.read_csv(_member(z, "INFOTABLE.tsv"), sep="\t",
                         usecols=["ACCESSION_NUMBER", "CUSIP", "VALUE", "SSHPRNAMT", "SSHPRNAMTTYPE", "PUTCALL"],
                         dtype={"ACCESSION_NUMBER": str, "CUSIP": str, "SSHPRNAMTTYPE": str, "PUTCALL": str},
                         low_memory=False, encoding="utf-8", encoding_errors="replace", chunksize=CHUNK)
        for ch in it:
            ch = ch[ch["ACCESSION_NUMBER"].isin(keep_acc)]
            ch = ch[ch["PUTCALL"].isna() & (ch["SSHPRNAMTTYPE"].astype(str).str.upper().str.strip() == "SH")]
            cus = ch["CUSIP"].astype(str).str.upper().str.strip()
            ch = ch.assign(CUSIP=cus)[cus.str.match(EQUITY_CUSIP, na=False) & ~cus.str.startswith("00000")]
            if ch.empty:
                continue
            v = pd.to_numeric(ch["VALUE"], errors="coerce")
            s = pd.to_numeric(ch["SSHPRNAMT"], errors="coerce")
            ok = (v > 0) & (s > 0)
            ch = pd.DataFrame({"ACCESSION_NUMBER": ch["ACCESSION_NUMBER"][ok].values,
                               "CUSIP": ch["CUSIP"][ok].values, "VALUE": v[ok].values, "SSHPRNAMT": s[ok].values})
            parts.append(ch.groupby(["ACCESSION_NUMBER", "CUSIP"], as_index=False)[["VALUE", "SSHPRNAMT"]].sum())
    except Exception as ex:
        say(f"    skip {p.name}: {type(ex).__name__}: {ex}")
        return None
    if not parts:
        return None
    # a filer may list one CUSIP on several lines (accounts, discretion types)
    it = pd.concat(parts, ignore_index=True)
    del parts
    it = it.groupby(["ACCESSION_NUMBER", "CUSIP"], as_index=False)[["VALUE", "SSHPRNAMT"]].sum()
    # per-filing unit detection (see module docstring): a filing whose median
    # implied price is under $1 reported VALUE in thousands
    it["px"] = it["VALUE"] / it["SSHPRNAMT"]
    fmed = it.groupby("ACCESSION_NUMBER")["px"].transform("median")
    thousands = fmed <= 1.0
    it.loc[thousands, "px"] *= 1000.0
    it.loc[thousands, "VALUE"] *= 1000.0
    it = it[it["px"].between(0.05, 2e6)]
    it = it.merge(keep[["ACCESSION_NUMBER", "cik", "period", "passive_group"]],
                  on="ACCESSION_NUMBER", how="left")
    pos = it.rename(columns={"SSHPRNAMT": "shares", "VALUE": "value"})[
        ["period", "CUSIP", "cik", "passive_group", "shares", "value"]]
    pos["px"] = pos["value"] / pos["shares"]
    g = pos.groupby(["period", "CUSIP"])
    out = g.agg(inst_shares=("shares", "sum"), inst_value_usd=("value", "sum"),
                n_holders=("cik", "nunique"), px_med=("px", "median")).reset_index()
    for grp, col in (("big3", "big3_shares"), ("ext", "ext_shares")):
        s = pos[pos["passive_group"] == grp].groupby(["period", "CUSIP"])["shares"].sum()
        out[col] = out.set_index(["period", "CUSIP"]).index.map(s).fillna(0.0).values
    # medians cannot be merged across zips exactly; carry the per-zip median and
    # its weight (holders) so the combine step can take a holder-weighted median
    # approximation. In practice one period sits almost entirely in one zip.
    out["src"] = p.name
    return out


def combine(parts: list[pd.DataFrame]) -> pd.DataFrame:
    """Merge per-zip partial aggregates for the same (period, cusip).

    A period is spread over zips when late filers or amendments land in a
    later filing window. Sums (shares, value, holders) add across zips; the
    implied price is taken from the zip with the most holders, which is the
    on-time bulk of filers - a late filer's median is a worse estimate of the
    quarter-end price, not a better one. Fully vectorised: the earlier
    holder-weighted-median apply took longer than the aggregation itself.
    """
    A = pd.concat(parts, ignore_index=True).rename(columns={"CUSIP": "cusip"})
    A = A.sort_values(["period", "cusip", "n_holders", "src"], ascending=[True, True, False, True])
    dom = A.drop_duplicates(["period", "cusip"], keep="first")[["period", "cusip", "px_med"]]
    S = (A.groupby(["period", "cusip"], as_index=False)
           .agg(inst_shares=("inst_shares", "sum"), inst_value_usd=("inst_value_usd", "sum"),
                n_holders=("n_holders", "sum"), big3_shares=("big3_shares", "sum"),
                ext_shares=("ext_shares", "sum"), n_src=("src", "size")))
    S = S.merge(dom, on=["period", "cusip"], how="left")
    # Institutional value is holdings marked at the cross-filer median price.
    # The sum of REPORTED values is kept for reference but is not trusted: a
    # filer whose median implied price is under $1 (penny stocks, warrants)
    # is scaled x1000 by the per-filing unit rule, and a handful of such
    # filings inflated the 2023-2024 aggregate by 40-70% (Apple 2024-06:
    # reported 2.87tn vs shares x median 1.97tn). Shares are never scaled,
    # and a median across thousands of filers is immune to the same error.
    S["inst_value_reported"] = S["inst_value_usd"]
    S["inst_value_usd"] = S["inst_shares"] * S["px_med"]
    S["big3_share_of_inst"] = S["big3_shares"] / S["inst_shares"]
    S["ext_share_of_inst"] = (S["big3_shares"] + S["ext_shares"]) / S["inst_shares"]
    return S.sort_values(["period", "cusip"]).reset_index(drop=True)


def passive_presence(F: pd.DataFrame) -> pd.DataFrame:
    """Per period: how many of the expected passive complexes actually filed.

    A quarter where one of the three is missing (late filing, not yet in the
    data sets) would otherwise read as a collapse of passive ownership. The
    caller masks big3/ext measures where the count is short.
    """
    n_big3 = F[F["passive_group"] == "big3"].groupby("period")["cik"].nunique().rename("n_big3_filers")
    n_ext = F[F["passive_group"] == "ext"].groupby("period")["cik"].nunique().rename("n_ext_filers")
    return pd.concat([n_big3, n_ext], axis=1).fillna(0).astype(int).reset_index()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-zips", type=int, default=0)
    a = ap.parse_args(argv)
    log = open(C.LOGS / "f13.log", "w", encoding="utf-8")

    def say(msg):
        print(msg, flush=True); log.write(msg + "\n"); log.flush()

    zips = list_zips()
    if a.limit_zips:
        zips = zips[-a.limit_zips:]
    say(f"13F zips: {len(zips)}  ({zips[0].name} .. {zips[-1].name})")
    t0 = time.time()
    F = build_filings(zips, say)
    F = assign_passive(F, say)
    F.to_parquet(C.PQ / "f13_filings.parquet", index=False)

    # per-zip partial aggregates are cached so a failed or slow combine never
    # costs the 18-minute aggregation pass again
    part_dir = C.PQ / "f13_parts"
    part_dir.mkdir(parents=True, exist_ok=True)
    parts = []
    for i, p in enumerate(zips, 1):
        keep = F[F["src"] == p.name]
        if keep.empty:
            continue
        cache = part_dir / (p.stem + ".parquet")
        if cache.exists():
            out = pd.read_parquet(cache)
        else:
            out = aggregate_zip(p, keep, say)
            if out is not None:
                out.to_parquet(cache, index=False)
        if out is not None:
            parts.append(out)
        say(f"  {i}/{len(zips)} {p.name}: kept filings {len(keep):,}  "
            f"rows {0 if out is None else len(out):,}  {(time.time()-t0)/60:.1f} min")
    A = combine(parts)
    pres = passive_presence(F)
    A = A.merge(pres, on="period", how="left")
    A["big3_complete"] = A["n_big3_filers"] >= 3
    A["ext_complete"] = A["big3_complete"] & (A["n_ext_filers"] >= 4)
    for col in ("big3_shares", "big3_share_of_inst"):
        A.loc[~A["big3_complete"], col] = np.nan
    for col in ("ext_shares", "ext_share_of_inst"):
        A.loc[~A["ext_complete"], col] = np.nan
    say("passive filer presence by period (big3 expected 3, ext expected 5):")
    say(pres.tail(8).to_string(index=False))
    A.to_parquet(C.PQ / "inst_own_quarterly.parquet", index=False)
    say(f"done: {len(A):,} (period, cusip) rows, periods {A.period.nunique()}, "
        f"cusips {A.cusip.nunique():,}  {(time.time()-t0)/60:.1f} min")
    q = A.groupby("period").agg(cusips=("cusip", "size"), holders_med=("n_holders", "median"),
                                big3_med=("big3_share_of_inst", "median"))
    say(q.tail(12).round(3).to_string())
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
