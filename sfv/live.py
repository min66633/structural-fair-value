# -*- coding: utf-8 -*-
"""Live inputs for the per-stock tools: the share price, the risk-free rate, the sell-side consensus.

Each fetch returns a dict with `ok`, the value(s), `asof` (a timestamp or date string), `source`, and `error`
(an explanation when ok is False). Nothing here raises for a network problem: the callers keep the panel's
value and show the reason. The desktop app runs these in a worker thread.

    price(ticker)      yfinance last price (fast_info), else the last daily close of the past five days
    rf()               FRED DGS10, the last observation (the 10-year Treasury the discount rate starts from)
    consensus(ticker)  yfinance revenue and EPS growth for this and next fiscal year, and the mean price target
"""
from __future__ import annotations

import io
from datetime import datetime, timezone

import numpy as np
import pandas as pd

FRED_DGS10 = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10"
TIMEOUT = 10


def ysyms(ticker: str) -> list[str]:
    """yfinance spellings to try, in order: the ticker with '.' as '-' (BRK.B -> BRK-B); then, for a symbol that may
    be a share class written without a separator (HEIA, MOGA, BFA), the class split off (HEI-A). The second form is
    only tried when the first returns nothing, since most four-letter tickers ending in A or B are ordinary (NVDA)."""
    t = ticker.strip().upper().replace(".", "-")
    out = [t]
    if "-" not in t and 3 <= len(t) <= 5 and t[-1] in "AB" and t[:-1].isalpha():
        out.append(f"{t[:-1]}-{t[-1]}")
    return out


def _price_of(symbol: str) -> tuple[float, str, str] | None:
    """(price, asof, source) for one yfinance symbol, or None when it has no quote."""
    import yfinance as yf
    t = yf.Ticker(symbol)
    try:
        fi = t.fast_info
        p = float(fi["last_price"]) if fi is not None and fi.get("last_price") is not None else None
    except Exception:  # noqa: BLE001
        p = None
    if p is not None and np.isfinite(p) and p > 0:
        return p, datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "yfinance fast_info"
    h = t.history(period="5d", auto_adjust=False)
    if h is None or h.empty or h["Close"].dropna().empty:
        return None
    last = h.dropna(subset=["Close"]).iloc[-1]
    return float(last["Close"]), pd.Timestamp(h.dropna(subset=["Close"]).index[-1]).strftime("%Y-%m-%d"), "yfinance close"


def price(ticker: str) -> dict:
    out = {"ok": False, "ticker": ticker, "symbol": "", "price": np.nan, "asof": None, "source": "yfinance", "error": ""}
    try:
        for sym in ysyms(ticker):
            got = _price_of(sym)
            if got is not None:
                p, asof, src = got
                out.update(ok=True, symbol=sym, price=p, asof=asof, source=src)
                return out
        out["error"] = "가격 없음 (티커 불일치 또는 상장 폐지)"
        return out
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"[:200]
        return out


def rf() -> dict:
    out = {"ok": False, "rf": np.nan, "asof": None, "source": "FRED DGS10", "error": ""}
    try:
        import requests
        r = requests.get(FRED_DGS10, timeout=TIMEOUT)
        r.raise_for_status()
        d = pd.read_csv(io.StringIO(r.text))
        col = [c for c in d.columns if c.upper() == "DGS10"]
        if not col:
            out["error"] = "예상치 못한 CSV 형식"
            return out
        d[col[0]] = pd.to_numeric(d[col[0]], errors="coerce")
        d = d.dropna(subset=[col[0]])
        if d.empty:
            out["error"] = "관측치 없음"
            return out
        last = d.iloc[-1]
        out.update(ok=True, rf=float(last[col[0]]) / 100.0, asof=str(last.iloc[0]))
        return out
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"[:200]
        return out


def consensus(ticker: str) -> dict | None:
    """Sell-side consensus from yfinance: revenue and EPS growth for the current and next fiscal year, mean target.
    None when yfinance has no estimates (or no network)."""
    import yfinance as yf
    for sym in ysyms(ticker):
        try:
            t = yf.Ticker(sym)
            re_, ee = t.revenue_estimate, t.earnings_estimate
            if re_ is None or ee is None or re_.empty:
                continue
            out = {"rev": [float(re_.loc["0y", "growth"]), float(re_.loc["+1y", "growth"])],
                   "eps": [float(ee.loc["0y", "growth"]), float(ee.loc["+1y", "growth"])],
                   "n": int(re_.loc["+1y", "numberOfAnalysts"]), "symbol": sym}
            try:
                pt = t.analyst_price_targets or {}
                out["target"] = float(pt.get("mean", np.nan))
                out["target_n"] = pt
            except Exception:  # noqa: BLE001
                out["target"] = np.nan
            if not all(np.isfinite(out["rev"])):
                continue
            return out
        except Exception:  # noqa: BLE001
            continue
    return None
