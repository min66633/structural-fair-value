# -*- coding: utf-8 -*-
"""Spot-check the point-in-time fundamentals panel against well-known figures.

Prints, for a handful of firms with awkward calendars or tag histories, the
recovered quarterly revenue / net income, shares, availability lag, and the
derived-vs-direct mix. Numbers are eyeballed against public filings; the
script does not assert, it exposes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sfv import config as C  # noqa: E402

pd.set_option("display.width", 260)
pd.set_option("display.max_columns", 40)

CASES = {
    320193: "Apple (Sep FYE, 52/53-week)",
    104169: "Walmart (Jan FYE)",
    1045810: "NVIDIA (Jan FYE, splits)",
    19617: "JPMorgan (bank revenue tags)",
    789019: "Microsoft (Jun FYE)",
    1018724: "Amazon",
    1318605: "Tesla",
    1652044: "Alphabet (multi-class)",
}


def main():
    P = pd.read_parquet(C.PQ / "fund_quarterly.parquet")
    D = pd.read_parquet(C.PQ / "shares_dei.parquet")
    print(f"panel rows {len(P):,}  ciks {P.cik.nunique():,}  ends {P.end.min().date()} .. {P.end.max().date()}")
    lag = (P["avail_date"] - P["end"]).dt.days
    print(f"avail lag: median {lag.median():.0f}  p75 {lag.quantile(.75):.0f}  p90 {lag.quantile(.9):.0f}  "
          f">120d share {(lag > 120).mean()*100:.1f}%")
    print(f"qend != end share {(P.qend != P.end).mean()*100:.1f}%   ttm_ok share {P.ttm_ok.mean()*100:.1f}%")
    cols = ["end", "qend", "fy", "fp", "form", "q_revenue", "q_net_income", "q_rnd", "q_sga", "equity", "assets",
            "shares_out", "ttm_revenue", "ttm_net_income", "avail_date"]
    for cik, label in CASES.items():
        a = P[P.cik == cik].sort_values("end")
        print("\n" + "=" * 100 + f"\n{label}  cik={cik}  rows={len(a)}")
        if a.empty:
            continue
        show = a[cols].tail(9).copy()
        for c in ["q_revenue", "q_net_income", "q_rnd", "q_sga", "equity", "assets", "ttm_revenue", "ttm_net_income"]:
            show[c] = (show[c] / 1e9).round(3)
        show["shares_out"] = (show["shares_out"] / 1e6).round(1)
        show["lag"] = (show["avail_date"] - show["end"]).dt.days
        print(show.to_string(index=False))
        d = D[D.cik == cik].tail(3)
        print("dei shares (mn):", [(str(x.asof.date()), round(x.shares_dei / 1e6, 1), str(x.filed.date())) for x in d.itertuples()])
    # revenue tag usage overall: how many rows lack revenue but have net income
    miss = P["q_revenue"].isna() & P["q_net_income"].notna()
    print(f"\nrows with net income but no revenue: {miss.mean()*100:.1f}%")
    # negative equity / tiny firms
    print(f"rows with equity <= 0: {(P.equity <= 0).mean()*100:.1f}%   assets < $10m: {(P.assets < 1e7).mean()*100:.1f}%")


if __name__ == "__main__":
    main()
