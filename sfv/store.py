# -*- coding: utf-8 -*-
"""One loader for the per-stock tools: the pipeline's parquet directory or the app's compact copy.

    Store()               data/parquet, any valuation month (month="2025-12-31"); default the latest
    Store(APP_DATA)       the compact copy that `python -m sfv.store` writes to app/data (latest month only)

Both give the same attributes, so scripts/what_if.py and the desktop app read through one code path
and cannot disagree on an input:

    firms       one row per stock at the valuation month: the model's inputs (b_adj, ttm_e_adj, x0, xinf,
                omega, roe_star_ind, payout, r = rf + beta x erp), the inversion columns expectations.py
                already computed, rev0 (TTM revenue of the fundamentals quarter), price_ps, ni_gaap,
                shares, fund_age, and the capitalisation band (g_star_lo/hi, log_pv_lo/hi)
    hist        every valuation month per stock: what the history table and the scorecard read
    quarters    quarterly TTM revenue and adjusted earnings per stock (rows with a complete TTM)
    rev_panel   quarters with revenue > 0 and the forward 3-year revenue CAGR and past-year growth
                attached (the base rates)
    band        the capitalisation band as its own table
    base_rates  {"bucket": ..., "cond": ...} the achievement tables of phase7_expectations.md 4-1 and 4-2
    industry    per FF12 industry at the month: firms, ROE benchmark, medians of persistence, long-run
                excess ROE, payout, beta, margin (defaults for a hand-entered firm)
    macro       month, erp_avg, rf, erp, omega vintage, size-group medians, n_firms, exported_at, source

The compact copy holds the same tables with the same column names, so the export is Store().save(dir):
about 15 MB, rebuilt by the runner (stage app_export) whenever the panels change.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from sfv import config as C
from sfv.rim import OMEGA_MAX, X_INF_CAP

APP_DATA = C.ROOT / "app" / "data"

FIRM_BASE = ["cik", "month", "ticker", "name", "ff12", "mktcap", "f_end", "f_avail", "b_gaap", "b_adj", "k_int",
             "ttm_e_gaap", "ttm_e_adj", "roe_adj", "roe_star_ind", "x0", "omega", "omega_applied", "a", "omega_src",
             "asof_year", "beta", "rf", "erp", "r", "payout", "v_f", "log_pv"]
HIST_COLS = ["cik", "month", "f_end", "mktcap", "ttm_e_adj", "g_star_10y", "g_star_10y_norm", "g_model", "r", "b1",
             "omega_star", "T_star", "p_achieve_3y", "log_pv"]
QTR_COLS = ["cik", "end", "ttm_revenue", "ttm_e_adj", "ttm_ok"]
BAND_COLS = ["cik", "g_star_lo", "g_star_hi", "log_pv_lo", "log_pv_hi"]
FILES = {"firms": "firms.parquet", "hist": "hist.parquet", "quarters": "quarters.parquet", "band": "band.parquet",
         "bucket": "base_rates_bucket.parquet", "cond": "base_rates_cond.parquet", "industry": "industry.parquet"}
SIZE_LABELS = {"top5": "시총 상위 5%", "top20": "상위 5~20%", "rest": "나머지"}


def size_group(mktcap: pd.Series) -> pd.Series:
    """top5 / top20 / rest by market-cap rank within the month (the groups persistence.py reports by)."""
    pct = mktcap.rank(pct=True)
    out = pd.Series("rest", index=mktcap.index, dtype=object)
    out[pct > 0.80] = "top20"
    out[pct > 0.95] = "top5"
    return out


def attach_growth(quarters: pd.DataFrame) -> pd.DataFrame:
    """Rows with revenue > 0, one per firm-quarter, with the realised forward 3-year revenue CAGR (cagr) and
    the past-year revenue growth (past) attached. The panel the base rates are read from."""
    I = quarters[quarters["ttm_ok"].fillna(False).astype(bool) & (quarters["ttm_revenue"] > 0)]
    I = I.sort_values(["cik", "end"]).drop_duplicates(["cik", "end"]).copy()
    g = I.groupby("cik")
    fwd = g["ttm_revenue"].shift(-12)
    ok = (g["end"].shift(-12) - I["end"]).dt.days.between(1020, 1170) & (fwd > 0)
    bwd = g["ttm_revenue"].shift(4)
    okb = (I["end"] - g["end"].shift(4)).dt.days.between(340, 390) & (bwd > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        I["cagr"] = ((fwd / I["ttm_revenue"]) ** (1.0 / 3.0) - 1.0).where(ok)
        I["past"] = (I["ttm_revenue"] / bwd - 1.0).where(okb)
    return I


def industry_table(d: pd.DataFrame) -> pd.DataFrame:
    g = d.copy()
    g["margin"] = (g["ttm_e_adj"] / g["rev0"]).where(g["rev0"] > 0)
    return (g.groupby("ff12")
             .agg(n=("cik", "size"), roe_star_ind=("roe_star_ind", "median"), omega_med=("omega", "median"),
                  xinf_med=("xinf", "median"), payout_med=("payout", "median"), beta_med=("beta", "median"),
                  margin_med=("margin", "median"))
             .reset_index())


def macro_dict(d: pd.DataFrame, m: pd.Timestamp, erp_avg: float, source: str) -> dict:
    by = d.groupby(size_group(d["mktcap"]))
    return {"month": str(m.date()), "erp_avg": float(erp_avg), "rf": float(d["rf"].iloc[0]), "erp": float(d["erp"].iloc[0]),
            "asof_year": int(d["asof_year"].mode().iloc[0]) if d["asof_year"].notna().any() else None,
            "n_firms": int(len(d)), "fund_end_max": str(pd.Timestamp(d["f_end"].max()).date()),
            "omega_by_size": {k: float(v) for k, v in by["omega"].median().items()},
            "xinf_by_size": {k: float(v) for k, v in by["xinf"].median().items()},
            "mktcap_floor": {k: float(v) for k, v in by["mktcap"].min().items()},
            "source": source}


class Store:
    def __init__(self, root: str | Path | None = None, month: str = ""):
        self.root = Path(root) if root is not None else C.PQ
        self.compact = (self.root / FILES["firms"]).exists() and (self.root / "macro.json").exists()
        if self.compact:
            if month:
                raise ValueError("the compact copy holds one month; use Store(month=...) on data/parquet")
            self._load_compact()
        else:
            self._load_full(month)
        self.month = pd.Timestamp(self.macro["month"])
        self.erp_avg = float(self.macro["erp_avg"])
        self.rev_panel = attach_growth(self.quarters)

    # ------------------------------------------------------------------ loading
    def _load_full(self, month: str) -> None:
        R = pd.read_parquet(C.PQ / "rim_monthly.parquet", columns=FIRM_BASE)
        m = pd.Timestamp(month) if month else R["month"].max()
        erp_avg = float(R.groupby("month")["erp"].first().mean())     # the sample-average ERP: the discount-rate reference
        d = R[R["month"] == m].copy()
        if d.empty:
            raise ValueError(f"no valuation rows for {m.date()}")
        d["xinf"] = np.clip(d["a"] / (1.0 - np.minimum(d["omega"], OMEGA_MAX)), -X_INF_CAP, X_INF_CAP)
        X = pd.read_parquet(C.PQ / "expectations_monthly.parquet")
        self.hist = X[[c for c in HIST_COLS if c in X.columns]].copy()
        Xm = X[X["month"] == m].drop(columns=[c for c in X.columns if c in d.columns and c != "cik"])
        d = d.merge(Xm, on="cik", how="left")
        # revenue of the same fundamentals quarter (scenario paths start from it) and the share price (per-share targets)
        I = pd.read_parquet(C.PQ / "intangibles_quarterly.parquet", columns=QTR_COLS)
        rev0 = I[["cik", "end", "ttm_revenue"]].rename(columns={"end": "f_end", "ttm_revenue": "rev0"}).drop_duplicates(["cik", "f_end"])
        d = d.merge(rev0, on=["cik", "f_end"], how="left")
        U = pd.read_parquet(C.PQ / "universe_monthly.parquet", columns=["cik", "month", "adj_close", "ttm_net_income", "shares", "fund_age"])
        U = U[U["month"] == m].rename(columns={"adj_close": "price_ps", "ttm_net_income": "ni_gaap"}).drop_duplicates(["cik", "month"])
        d = d.merge(U, on=["cik", "month"], how="left")
        f = C.PQ / "expectations_band.parquet"
        self.band = pd.read_parquet(f)[BAND_COLS] if f.exists() else pd.DataFrame(columns=BAND_COLS)
        d = d.merge(self.band, on="cik", how="left")
        d["size_group"] = size_group(d["mktcap"])
        self.firms = d.reset_index(drop=True)
        f = C.PQ / "universe_ciks.parquet"
        if f.exists():
            I = I[I["cik"].isin(pd.read_parquet(f)["cik"])]
        self.quarters = I[I["ttm_ok"].fillna(False).astype(bool)].reset_index(drop=True)
        self.base_rates = {}
        for k, n in (("bucket", "expectations_calibration.parquet"), ("cond", "expectations_calibration_cond.parquet")):
            self.base_rates[k] = pd.read_parquet(C.PQ / n) if (C.PQ / n).exists() else pd.DataFrame()
        self.industry = industry_table(self.firms)
        self.macro = macro_dict(self.firms, m, erp_avg, "full")

    def _load_compact(self) -> None:
        self.firms = pd.read_parquet(self.root / FILES["firms"])
        self.hist = pd.read_parquet(self.root / FILES["hist"])
        self.quarters = pd.read_parquet(self.root / FILES["quarters"])
        self.band = pd.read_parquet(self.root / FILES["band"])
        self.base_rates = {k: pd.read_parquet(self.root / FILES[k]) for k in ("bucket", "cond")}
        self.industry = pd.read_parquet(self.root / FILES["industry"])
        self.macro = json.loads((self.root / "macro.json").read_text(encoding="utf-8"))
        self.macro["source"] = "compact"

    def save(self, out: str | Path) -> list[Path]:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        written = []
        for k, df in (("firms", self.firms), ("hist", self.hist), ("quarters", self.quarters), ("band", self.band),
                      ("bucket", self.base_rates["bucket"]), ("cond", self.base_rates["cond"]), ("industry", self.industry)):
            p = out / FILES[k]
            df.to_parquet(p, index=False)
            written.append(p)
        m = dict(self.macro)
        m["exported_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        m["source"] = "compact"
        p = out / "macro.json"
        p.write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
        written.append(p)
        return written

    # ------------------------------------------------------------------ lookups
    def firm(self, ticker: str) -> tuple[pd.DataFrame, list[str]]:
        """One-row frame for the ticker (case-insensitive). If absent: an empty frame and up to five tickers
        whose company name contains the text."""
        t = ticker.strip().upper()
        row = self.firms[self.firms["ticker"].str.upper() == t]
        if row.empty:
            near = self.firms[self.firms["name"].str.upper().str.contains(t, na=False, regex=False)]["ticker"].head(5).tolist()
            return row, near
        return row.iloc[[0]], []


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="write the compact copy the desktop app reads")
    ap.add_argument("--out", default=str(APP_DATA))
    ap.add_argument("--month", default="", help="valuation month (default: latest)")
    a = ap.parse_args(argv)
    s = Store(month=a.month)
    files = s.save(a.out)
    total = sum(p.stat().st_size for p in files)
    print(f"app data {s.month.date()}: firms {len(s.firms):,}, history rows {len(s.hist):,}, quarters {len(s.quarters):,} "
          f"-> {a.out} ({total / 1e6:.1f} MB, {len(files)} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
