# -*- coding: utf-8 -*-
"""Monthly prices from yfinance for the currently listed part of the universe.

This is the SECONDARY price route. It only knows names that still trade, so
any return computed from it is survivorship-biased; the primary, bias-free
route is the 13F implied quarterly price (sfv.f13, px_med) with the FTD
monthly last price as a cross-check. yfinance prices are used for
(a) the live valuation tool, which by construction concerns listed names, and
(b) monthly-frequency auxiliary tests, labelled as biased.

Universe: tickers in the security master linked to a CIK with a 10-K, listed
on Nasdaq / NYSE / NYSE American / CBOE per SEC's exchange file (OTC excluded).

Output  data/parquet/prices_monthly.parquet  ticker, month, close, adj_close, volume, dividends, splits
Run: python -m sfv.prices [--limit N]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings

import numpy as np
import pandas as pd

from . import config as C

warnings.filterwarnings("ignore")
BATCH = 80


def universe() -> pd.DataFrame:
    ex = json.load(open(C.RAW / "company_tickers_exchange.json", encoding="utf-8"))
    E = pd.DataFrame(ex["data"], columns=ex["fields"])
    E = E[E["exchange"].isin(["Nasdaq", "NYSE", "CBOE"])]
    E["ticker"] = E["ticker"].str.upper()
    sm = pd.read_parquet(C.PQ / "security_master.parquet")
    linked = sm.dropna(subset=["cik"])
    linked = linked[linked["has_10k"].fillna(False)]
    U = E[E["cik"].isin(linked["cik"].astype(int))][["ticker", "cik", "name"]].drop_duplicates("ticker")
    return U.sort_values("ticker").reset_index(drop=True)


def fetch(tickers: list[str], start: str) -> pd.DataFrame:
    import yfinance as yf
    ysyms = [t.replace(".", "-") for t in tickers]
    d = yf.download(ysyms, start=start, interval="1mo", group_by="ticker", auto_adjust=False,
                    actions=True, progress=False, threads=True)
    rows = []
    for t, ys in zip(tickers, ysyms):
        try:
            x = d[ys] if len(tickers) > 1 else d
        except (KeyError, TypeError):
            continue
        if x is None or x.empty or "Close" not in x:
            continue
        x = x.dropna(subset=["Close"])
        if x.empty:
            continue
        out = pd.DataFrame({
            "ticker": t,
            "month": pd.to_datetime(x.index).tz_localize(None).to_period("M").to_timestamp("M"),
            "close": x["Close"].to_numpy(dtype=float),
            "adj_close": x["Adj Close"].to_numpy(dtype=float) if "Adj Close" in x else np.nan,
            "volume": x["Volume"].to_numpy(dtype=float) if "Volume" in x else np.nan,
            "dividends": x["Dividends"].to_numpy(dtype=float) if "Dividends" in x else 0.0,
            "splits": x["Stock Splits"].to_numpy(dtype=float) if "Stock Splits" in x else 0.0,
        })
        rows.append(out)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--start", default="2009-01-01")
    a = ap.parse_args(argv)
    log = open(C.LOGS / "prices.log", "w", encoding="utf-8")

    def say(msg):
        print(msg, flush=True); log.write(msg + "\n"); log.flush()

    U = universe()
    if a.limit:
        U = U.head(a.limit)
    say(f"tickers to fetch: {len(U):,}")
    out = C.PQ / "prices_monthly.parquet"
    done = set()
    parts = []
    if out.exists():
        prev = pd.read_parquet(out)
        parts.append(prev)
        done = set(prev["ticker"].unique())
        say(f"  cached tickers: {len(done):,}")
    todo = [t for t in U["ticker"] if t not in done]
    t0 = time.time()
    fails = 0
    for i in range(0, len(todo), BATCH):
        chunk = todo[i:i + BATCH]
        try:
            df = fetch(chunk, a.start)
        except Exception as ex:
            say(f"  batch {i} failed: {type(ex).__name__}: {ex}")
            fails += len(chunk)
            time.sleep(5)
            continue
        if len(df):
            parts.append(df)
        if (i // BATCH) % 10 == 0:
            n = sum(len(p) for p in parts)
            say(f"  {i + len(chunk):,}/{len(todo):,}  rows {n:,}  {(time.time()-t0)/60:.1f} min")
            pd.concat(parts, ignore_index=True).drop_duplicates(["ticker", "month"]).to_parquet(out, index=False)
        time.sleep(0.5)
    P = pd.concat(parts, ignore_index=True).drop_duplicates(["ticker", "month"]) if parts else pd.DataFrame()
    P = P.sort_values(["ticker", "month"])
    P.to_parquet(out, index=False)
    say(f"done: tickers {P.ticker.nunique():,}  rows {len(P):,}  months {P.month.min().date()} .. {P.month.max().date()}  "
        f"fails {fails}  {(time.time()-t0)/60:.1f} min")
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
