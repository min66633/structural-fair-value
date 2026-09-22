# -*- coding: utf-8 -*-
"""Form N-PORT structured data sets -> fund holdings, monthly flows, identifiers.

N-PORT is filed monthly by every registered fund (mutual funds and ETFs) but
only the report for the last month of each fiscal quarter is made public,
60 days after period end. Each public report carries:

    FUND_REPORTED_INFO     net assets and, for each of the quarter's three
                           months, sales / reinvestment / redemption flows
                           (Item B.6). This is what makes a Lou (2012)
                           flow-induced-trading measure computable for free.
    FUND_REPORTED_HOLDING  every position: CUSIP, shares (BALANCE, UNIT=NS),
                           USD value, % of net assets, asset category
                           (EC = common equity).
    IDENTIFIERS            ISIN and ticker per holding - a free, dated
                           CUSIP <-> ticker map from 2019 on.

Sizes force a streaming pass: FUND_REPORTED_HOLDING is 0.5-1.5 GB of TSV per
quarter and this machine has 8 GB. Holdings are read in chunks, filtered to
long common-equity positions with a valid 9-character CUSIP, and written per
quarter. Amended reports (NPORT-P/A) replace the original for the same
(series, report date).

Outputs (data/parquet/)
    nport_funds.parquet            one row per kept report (series x quarter)
    nport_holdings/<quarter>.parquet   series x report_date x cusip
    nport_identifiers.parquet      cusip -> ticker / isin / name, dated
    reports/logs/nport.log

Run: python -m sfv.nport [--only 2019q4] [--limit N]
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
INDEX_NAME = re.compile(
    r"\b(?:INDEX|IDX|ETF|S&P ?500|500 INDEX|TOTAL (?:STOCK )?MARKET|RUSSELL|NASDAQ[- ]?100|"
    r"EXTENDED MARKET|ISHARES|SPDR|MSCI|FTSE|CRSP|WILSHIRE|DOW JONES)\b", re.I)
CHUNK = 400_000


def _member(z: zipfile.ZipFile, name: str):
    for n in z.namelist():
        if n.rsplit("/", 1)[-1].upper() == name.upper():
            return z.open(n)
    raise KeyError(name)


def _header(z, name) -> list[str]:
    f = _member(z, name)
    return f.readline().decode("utf-8", "replace").rstrip("\r\n").split("\t")


def _read(z, name, usecols, dtype=None, chunksize=None):
    cols = _header(z, name)
    use = [c for c in usecols if c in cols]
    return pd.read_csv(_member(z, name), sep="\t", usecols=use, dtype=dtype,
                       low_memory=False, encoding="utf-8", encoding_errors="replace",
                       chunksize=chunksize, quoting=3)


FLOW_COLS = [f"{k}_FLOW_MON{i}" for k in ("SALES", "REINVESTMENT", "REDEMPTION") for i in (1, 2, 3)]


def funds_table(z: zipfile.ZipFile) -> pd.DataFrame:
    sub = _read(z, "SUBMISSION.tsv", ["ACCESSION_NUMBER", "FILING_DATE", "SUB_TYPE",
                                      "REPORT_ENDING_PERIOD", "REPORT_DATE", "IS_LAST_FILING"], dtype=str)
    reg = _read(z, "REGISTRANT.tsv", ["ACCESSION_NUMBER", "CIK", "REGISTRANT_NAME"], dtype=str)
    info = _read(z, "FUND_REPORTED_INFO.tsv",
                 ["ACCESSION_NUMBER", "SERIES_NAME", "SERIES_ID", "TOTAL_ASSETS", "NET_ASSETS"] + FLOW_COLS,
                 dtype=str)
    F = info.merge(sub, on="ACCESSION_NUMBER", how="left").merge(reg, on="ACCESSION_NUMBER", how="left")
    F["report_date"] = pd.to_datetime(F["REPORT_DATE"], errors="coerce", format="mixed", dayfirst=True)
    F["fy_end"] = pd.to_datetime(F["REPORT_ENDING_PERIOD"], errors="coerce", format="mixed", dayfirst=True)
    F["filed"] = pd.to_datetime(F["FILING_DATE"], errors="coerce", format="mixed", dayfirst=True)
    F["is_amendment"] = F["SUB_TYPE"].fillna("").str.upper().str.endswith("/A")
    F["cik"] = pd.to_numeric(F["CIK"], errors="coerce").astype("Int64")
    for c in ["TOTAL_ASSETS", "NET_ASSETS"] + FLOW_COLS:
        if c not in F.columns:
            F[c] = np.nan
        F[c] = pd.to_numeric(F[c], errors="coerce")
    F = F.rename(columns={"SERIES_ID": "series_id", "SERIES_NAME": "series_name",
                          "REGISTRANT_NAME": "registrant", "TOTAL_ASSETS": "total_assets",
                          "NET_ASSETS": "net_assets", "ACCESSION_NUMBER": "accession"})
    ren = {f"{k}_FLOW_MON{i}": f"{k.lower()[:5]}_m{i}" for k in ("SALES", "REINVESTMENT", "REDEMPTION") for i in (1, 2, 3)}
    F = F.rename(columns=ren)
    F["is_index_name"] = F["series_name"].fillna("").str.contains(INDEX_NAME)
    F = F.dropna(subset=["report_date", "series_id"])
    # amendments replace originals for the same series and report date
    F = F.sort_values(["series_id", "report_date", "is_amendment", "filed"])
    F = F.drop_duplicates(subset=["series_id", "report_date"], keep="last")
    keep = ["accession", "cik", "registrant", "series_id", "series_name", "report_date", "fy_end",
            "filed", "is_amendment", "total_assets", "net_assets", "is_index_name"] + list(ren.values())
    return F[keep].reset_index(drop=True)


def holdings_table(z: zipfile.ZipFile, accessions: set[str], say) -> pd.DataFrame:
    """Stream FUND_REPORTED_HOLDING and collapse each chunk to (filing, cusip).

    Memory is the constraint (about 3 GB is available to a job). Each chunk
    is filtered to long common-equity positions and aggregated at once, so
    the long text columns never accumulate; one representative HOLDING_ID
    per position is kept for the identifiers join (all lines of one CUSIP in
    one filing carry the same ISIN/ticker).
    """
    parts = []
    n_seen = n_kept = 0
    it = _read(z, "FUND_REPORTED_HOLDING.tsv",
               ["ACCESSION_NUMBER", "HOLDING_ID", "ISSUER_NAME", "ISSUER_CUSIP", "BALANCE", "UNIT",
                "CURRENCY_CODE", "CURRENCY_VALUE", "PERCENTAGE", "PAYOFF_PROFILE", "ASSET_CAT",
                "INVESTMENT_COUNTRY"],
               dtype=str, chunksize=CHUNK)
    for ch in it:
        n_seen += len(ch)
        ch = ch[ch["ACCESSION_NUMBER"].isin(accessions)]
        ch = ch[(ch["ASSET_CAT"].fillna("").str.upper() == "EC")
                & (ch["UNIT"].fillna("").str.upper() == "NS")
                & (ch["PAYOFF_PROFILE"].fillna("Long").str.upper() != "SHORT")]
        if ch.empty:
            continue
        cus = ch["ISSUER_CUSIP"].fillna("").str.upper().str.strip()
        keep = cus.str.match(EQUITY_CUSIP, na=False) & ~cus.str.startswith("00000")
        shares = pd.to_numeric(ch["BALANCE"], errors="coerce")
        value = pd.to_numeric(ch["CURRENCY_VALUE"], errors="coerce")
        keep &= (shares > 0) & (value > 0)
        if not keep.any():
            continue
        small = pd.DataFrame({
            "accession": ch["ACCESSION_NUMBER"][keep].values,
            "cusip": cus[keep].values,
            "holding_id": pd.to_numeric(ch["HOLDING_ID"][keep], errors="coerce").values,
            "shares": shares[keep].values,
            "value_usd": value[keep].values,
            "pct": pd.to_numeric(ch["PERCENTAGE"][keep], errors="coerce").values,
            "issuer_name": ch["ISSUER_NAME"][keep].str.slice(0, 60).values,
            "currency": ch["CURRENCY_CODE"][keep].values,
            "country": ch["INVESTMENT_COUNTRY"][keep].values,
        })
        n_kept += len(small)
        parts.append(small.groupby(["accession", "cusip"], as_index=False)
                     .agg(shares=("shares", "sum"), value_usd=("value_usd", "sum"), pct=("pct", "sum"),
                          holding_id=("holding_id", "first"), issuer_name=("issuer_name", "first"),
                          currency=("currency", "first"), country=("country", "first")))
        del ch, small
    say(f"    holdings rows seen {n_seen:,} -> equity kept {n_kept:,}")
    if not parts:
        return pd.DataFrame()
    H = pd.concat(parts, ignore_index=True)
    del parts
    # positions split across chunk boundaries are re-summed here
    H = (H.groupby(["accession", "cusip"], as_index=False)
           .agg(shares=("shares", "sum"), value_usd=("value_usd", "sum"), pct=("pct", "sum"),
                holding_id=("holding_id", "first"), issuer_name=("issuer_name", "first"),
                currency=("currency", "first"), country=("country", "first")))
    H["holding_id"] = H["holding_id"].astype("Int64")
    return H


def fix_share_units(H: pd.DataFrame, say) -> pd.DataFrame:
    """Repair BALANCE fields reported in the wrong unit.

    All funds mark a position at the same period-end fair value, so for one
    (cusip, report_date) the implied price value/shares agrees to the cent
    across hundreds of funds - except for a few filers whose share count is
    off by a power of ten (2019Q4: two funds of one complex carried 32% of
    all reported Apple shares at an implied $2). Value is the field that
    drives % of net assets and is checked by the filer; shares is not.

    Rule: with at least 3 funds reporting the same (cusip, report_date), a
    row whose implied price is outside [median/1.5, median*1.5] gets
    shares := value / median price, and is flagged. Rows without an anchor
    are left alone and flagged as unanchored.
    """
    H = H.copy()
    H["px"] = H["value_usd"] / H["shares"]
    g = H.groupby(["cusip", "report_date"])["px"]
    med = g.transform("median")
    n = g.transform("size")
    ratio = H["px"] / med
    bad = (n >= 3) & ((ratio > 1.5) | (ratio < 1 / 1.5))
    H["shares_reported"] = H["shares"]
    H["share_flag"] = np.where(n < 3, "unanchored", "ok")
    H.loc[bad, "share_flag"] = "rescaled"
    H.loc[bad, "shares"] = H.loc[bad, "value_usd"] / med[bad]
    H["px_anchor"] = med
    say(f"    share-unit repair: rescaled {int(bad.sum()):,} rows "
        f"({bad.mean()*100:.3f}%), unanchored {(n < 3).mean()*100:.1f}%")
    return H.drop(columns=["px"])


def identifiers_table(z: zipfile.ZipFile, holding_ids: set[int]) -> pd.DataFrame:
    parts = []
    it = _read(z, "IDENTIFIERS.tsv", ["HOLDING_ID", "IDENTIFIER_ISIN", "IDENTIFIER_TICKER"],
               dtype=str, chunksize=CHUNK)
    for ch in it:
        hid = pd.to_numeric(ch["HOLDING_ID"], errors="coerce")
        ch = ch[hid.isin(holding_ids)]
        if ch.empty:
            continue
        parts.append(pd.DataFrame({"holding_id": pd.to_numeric(ch["HOLDING_ID"], errors="coerce").astype("Int64"),
                                   "isin": ch["IDENTIFIER_ISIN"].str.strip().str.upper(),
                                   "ticker": ch["IDENTIFIER_TICKER"].str.strip().str.upper()}))
    if not parts:
        return pd.DataFrame(columns=["holding_id", "isin", "ticker"])
    return pd.concat(parts, ignore_index=True)


def process_zip(p: Path, out_dir: Path, say) -> tuple[pd.DataFrame, pd.DataFrame]:
    z = zipfile.ZipFile(p)
    F = funds_table(z)
    tag = p.stem.replace("_nport", "")
    cached = out_dir / f"{tag}.parquet"
    if cached.exists():
        # resume: the heavy holdings pass for this quarter is already on disk
        H = pd.read_parquet(cached)
        say(f"    (cached holdings {len(H):,} rows)")
    else:
        acc = set(F["accession"])
        H = holdings_table(z, acc, say)
        if H.empty:
            return F, pd.DataFrame()
        I = identifiers_table(z, set(H["holding_id"].dropna().astype(int)))
        I = I.drop_duplicates("holding_id")
        H = H.merge(I, on="holding_id", how="left")
        H = H.merge(F[["accession", "series_id", "report_date"]], on="accession", how="left")
        H = H.dropna(subset=["series_id"])
        H = fix_share_units(H, say)
        H = H[["accession", "series_id", "report_date", "cusip", "shares", "value_usd", "pct", "issuer_name",
               "ticker", "isin", "country", "currency", "shares_reported", "share_flag", "px_anchor"]]
        H.to_parquet(cached, index=False)
    eq = H.groupby("accession").agg(n_equity=("cusip", "size"), equity_value_usd=("value_usd", "sum"))
    F = F.merge(eq, left_on="accession", right_index=True, how="left")
    ids = (H.dropna(subset=["ticker"])
             .groupby(["cusip", "ticker"]).agg(n=("accession", "size"), first_seen=("report_date", "min"),
                                               last_seen=("report_date", "max"), issuer_name=("issuer_name", "first"),
                                               isin=("isin", "first")).reset_index())
    return F, ids


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="process a single quarter tag, e.g. 2019q4")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)
    log = open(C.LOGS / "nport.log", "a" if a.only else "w", encoding="utf-8")

    def say(msg):
        print(msg, flush=True); log.write(msg + "\n"); log.flush()

    zips = sorted(C.NPORT_DIR.glob("*_nport.zip"))
    if a.only:
        zips = [p for p in zips if p.stem.startswith(a.only)]
    if a.limit:
        zips = zips[:a.limit]
    out_dir = C.PQ / "nport_holdings"
    out_dir.mkdir(parents=True, exist_ok=True)
    say(f"N-PORT zips: {len(zips)}")
    t0 = time.time()
    funds, ids = [], []
    for i, p in enumerate(zips, 1):
        say(f"  {i}/{len(zips)} {p.name}")
        try:
            F, I = process_zip(p, out_dir, say)
        except Exception as ex:
            say(f"    FAILED {p.name}: {type(ex).__name__}: {ex}")
            continue
        funds.append(F)
        if len(I):
            ids.append(I)
        say(f"    reports {len(F):,}  index-named {int(F.is_index_name.sum()):,}  "
            f"{(time.time()-t0)/60:.1f} min")
    if not funds:
        say("nothing processed"); return 1
    FF = pd.concat(funds, ignore_index=True)
    FF = FF.sort_values(["series_id", "report_date", "filed"]).drop_duplicates(["series_id", "report_date"], keep="last")
    II = pd.concat(ids, ignore_index=True) if ids else pd.DataFrame()
    if len(II):
        II = (II.groupby(["cusip", "ticker"]).agg(n=("n", "sum"), first_seen=("first_seen", "min"),
                                                  last_seen=("last_seen", "max"), issuer_name=("issuer_name", "first"),
                                                  isin=("isin", "first")).reset_index())
    if a.only:
        # merge into existing outputs rather than overwrite
        fp = C.PQ / "nport_funds.parquet"
        if fp.exists():
            old = pd.read_parquet(fp)
            FF = pd.concat([old[~old.accession.isin(FF.accession)], FF], ignore_index=True)
        ip = C.PQ / "nport_identifiers.parquet"
        if ip.exists() and len(II):
            old = pd.read_parquet(ip)
            II = (pd.concat([old, II]).groupby(["cusip", "ticker"])
                    .agg(n=("n", "sum"), first_seen=("first_seen", "min"), last_seen=("last_seen", "max"),
                         issuer_name=("issuer_name", "first"), isin=("isin", "first")).reset_index())
    FF.to_parquet(C.PQ / "nport_funds.parquet", index=False)
    if len(II):
        II.to_parquet(C.PQ / "nport_identifiers.parquet", index=False)
    say(f"done: reports {len(FF):,}  series {FF.series_id.nunique():,}  "
        f"cusip-ticker pairs {len(II):,}  {(time.time()-t0)/60:.1f} min")
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
