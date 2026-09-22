# -*- coding: utf-8 -*-
"""SEC fails-to-deliver files -> dated CUSIP <-> symbol map (and a price by-product).

Each half-month file lists, per settlement date, every security with an
open fail: CUSIP, trading SYMBOL, description and the prior close PRICE.
Liquid names appear on most days, so across 2009-2026 the files give

  * a CUSIP <-> symbol pairing with first/last dates - covering names that
    were later delisted, which is exactly what the current-registrants
    ticker file cannot do; and
  * a monthly last price per CUSIP as a by-product, useful as a cross-check
    on the 13F implied price for delisted names.

Format: pipe-delimited text inside a zip, header
    SETTLEMENT DATE|CUSIP|SYMBOL|QUANTITY (FAILS)|DESCRIPTION|PRICE

Outputs (data/parquet/)
    ftd_cusip_symbol.parquet   cusip, symbol, description, first_date, last_date, n_days
    ftd_prices_monthly.parquet cusip, month, price_last, n_obs
Run: python -m sfv.ftd
"""
from __future__ import annotations

import io
import re
import sys
import time
import zipfile

import numpy as np
import pandas as pd

from . import config as C

EQUITY_CUSIP = re.compile(r"^[0-9A-Z]{6}[0-9][0-9A-Z][0-9]$")


def read_one(p) -> pd.DataFrame | None:
    z = zipfile.ZipFile(p)
    # the member is usually cnsfailsYYYYMMx.txt, but some months ship it without an extension
    names = [n for n in z.namelist() if not n.endswith("/")]
    if not names:
        return None
    raw = z.read(names[0]).decode("latin-1", "replace")
    df = pd.read_csv(io.StringIO(raw), sep="|", dtype=str, quoting=3, on_bad_lines="skip")
    df.columns = [c.strip().upper() for c in df.columns]
    need = {"SETTLEMENT DATE", "CUSIP", "SYMBOL", "DESCRIPTION", "PRICE"}
    if not need.issubset(df.columns):
        return None
    df = df[list(need)].dropna(subset=["CUSIP", "SYMBOL"])
    df["date"] = pd.to_datetime(df["SETTLEMENT DATE"].str.strip(), format="%Y%m%d", errors="coerce")
    df["cusip"] = df["CUSIP"].str.strip().str.upper()
    df["symbol"] = df["SYMBOL"].str.strip().str.upper()
    df["price"] = pd.to_numeric(df["PRICE"].str.strip(), errors="coerce")
    df["description"] = df["DESCRIPTION"].str.strip()
    df = df.dropna(subset=["date"])
    df = df[df["cusip"].str.match(EQUITY_CUSIP, na=False) & (df["symbol"] != "")]
    return df[["date", "cusip", "symbol", "description", "price"]]


def main() -> int:
    log = open(C.LOGS / "ftd.log", "w", encoding="utf-8")

    def say(msg):
        print(msg, flush=True); log.write(msg + "\n"); log.flush()

    files = sorted(C.FTD_DIR.glob("cnsfails*.zip"))
    say(f"FTD files: {len(files)}")
    t0 = time.time()
    pairs, prices = [], []
    for i, p in enumerate(files, 1):
        try:
            df = read_one(p)
        except Exception as ex:
            say(f"  FAILED {p.name}: {type(ex).__name__}: {ex}")
            continue
        if df is None or df.empty:
            say(f"  empty/unknown format {p.name}")
            continue
        g = (df.groupby(["cusip", "symbol"])
               .agg(first_date=("date", "min"), last_date=("date", "max"), n_days=("date", "nunique"),
                    description=("description", "last")).reset_index())
        pairs.append(g)
        m = df.dropna(subset=["price"]).sort_values("date")
        m["month"] = m["date"].dt.to_period("M").dt.to_timestamp("M")
        pm = m.groupby(["cusip", "month"]).agg(price_last=("price", "last"), n_obs=("price", "size")).reset_index()
        prices.append(pm)
        if i % 50 == 0:
            say(f"  {i}/{len(files)}  {(time.time()-t0)/60:.1f} min")
    P = pd.concat(pairs, ignore_index=True)
    P = (P.groupby(["cusip", "symbol"])
           .agg(first_date=("first_date", "min"), last_date=("last_date", "max"), n_days=("n_days", "sum"),
                description=("description", "last")).reset_index())
    P.to_parquet(C.PQ / "ftd_cusip_symbol.parquet", index=False)
    M = pd.concat(prices, ignore_index=True)
    M = M.sort_values(["cusip", "month"]).groupby(["cusip", "month"], as_index=False).agg(
        price_last=("price_last", "last"), n_obs=("n_obs", "sum"))
    M.to_parquet(C.PQ / "ftd_prices_monthly.parquet", index=False)
    say(f"done: cusip-symbol pairs {len(P):,} (cusips {P.cusip.nunique():,}, symbols {P.symbol.nunique():,})  "
        f"monthly price cells {len(M):,}  {(time.time()-t0)/60:.1f} min")
    multi = P.groupby("cusip")["symbol"].nunique()
    say(f"cusips with >1 symbol over time: {(multi > 1).sum():,}")
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
