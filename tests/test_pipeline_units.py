# -*- coding: utf-8 -*-
"""Unit tests for the tricky pieces of the phase-0 pipeline (synthetic data).

Run:  python -m pytest tests -q     or     python tests/test_pipeline_units.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sfv.xbrl_panel import quarterly_flows, _nearest_qend, build_panel  # noqa: E402
from sfv.nport import fix_share_units  # noqa: E402
from sfv.idmap import classify_security, norm_name  # noqa: E402


def _dur(cik, concept, start, end, val, filed, val_last=None):
    return dict(cik=cik, concept=concept, start=pd.Timestamp(start), end=pd.Timestamp(end), val=float(val),
                val_last=float(val if val_last is None else val_last), filed=pd.Timestamp(filed),
                fy=pd.Timestamp(end).year, fp="Q", form="10-Q", tag="X")


def test_quarterly_flows_same_start_differencing():
    # 10-Q Q1 (direct), 10-Q Q2 direct + H1 cumulative, 10-Q Q3 direct + 9M, 10-K FY only
    rows = [
        _dur(1, "revenue", "2020-01-01", "2020-03-31", 100, "2020-05-01"),
        _dur(1, "revenue", "2020-04-01", "2020-06-30", 110, "2020-08-01"),
        _dur(1, "revenue", "2020-01-01", "2020-06-30", 210, "2020-08-01"),
        _dur(1, "revenue", "2020-07-01", "2020-09-30", 120, "2020-11-01"),
        _dur(1, "revenue", "2020-01-01", "2020-09-30", 330, "2020-11-01"),
        _dur(1, "revenue", "2020-01-01", "2020-12-31", 460, "2021-02-15"),
    ]
    D = pd.DataFrame(rows)
    Q = quarterly_flows(D).set_index("end")
    assert Q.loc["2020-03-31", "val"] == 100 and Q.loc["2020-03-31", "method"] == "direct"
    assert Q.loc["2020-06-30", "val"] == 110 and Q.loc["2020-06-30", "method"] == "direct"
    assert Q.loc["2020-09-30", "val"] == 120
    # Q4 derived from FY - 9M with the 10-K filed date
    assert Q.loc["2020-12-31", "val"] == 130 and Q.loc["2020-12-31", "method"] == "derived"
    assert Q.loc["2020-12-31", "filed"] == pd.Timestamp("2021-02-15")


def test_quarterly_flows_only_cumulative_reports():
    # a filer that reports only year-to-date values: every quarter must be derived
    rows = [
        _dur(2, "cfo", "2020-01-01", "2020-03-31", 10, "2020-05-01"),
        _dur(2, "cfo", "2020-01-01", "2020-06-30", 25, "2020-08-01"),
        _dur(2, "cfo", "2020-01-01", "2020-09-30", 45, "2020-11-01"),
        _dur(2, "cfo", "2020-01-01", "2020-12-31", 70, "2021-02-15"),
    ]
    Q = quarterly_flows(pd.DataFrame(rows)).set_index("end")["val"]
    assert list(Q.loc[["2020-03-31", "2020-06-30", "2020-09-30", "2020-12-31"]]) == [10, 15, 20, 25]


def test_nearest_qend():
    d = pd.Series(pd.to_datetime(["2018-09-29", "2019-01-31", "2020-12-31", "2021-02-27", "2021-11-01"]))
    got = list(_nearest_qend(d).dt.strftime("%Y-%m-%d"))
    assert got == ["2018-09-30", "2018-12-31", "2020-12-31", "2021-03-31", "2021-09-30"]


def test_build_panel_ttm_and_avail():
    # two years of clean quarters for one cik; instants at each end
    rows = []
    q_ends = pd.date_range("2019-03-31", periods=8, freq="QE")
    for i, e in enumerate(q_ends):
        s = (e - pd.offsets.QuarterEnd(1)) + pd.Timedelta(days=1)
        filed = e + pd.Timedelta(days=40)
        rows.append(dict(cik=7, taxonomy="us-gaap", tag="NetIncomeLoss", unit="USD", start=s, end=e, val=10 + i,
                         accn="a", fy=e.year, fp="Q1", form="10-Q", filed=filed, frame=None, n_filings=1, which="only"))
        rows.append(dict(cik=7, taxonomy="us-gaap", tag="Revenues", unit="USD", start=s, end=e, val=100 + i,
                         accn="a", fy=e.year, fp="Q1", form="10-Q", filed=filed, frame=None, n_filings=1, which="only"))
        for tag, v in (("StockholdersEquity", 1000 + i), ("Assets", 2000 + i)):
            rows.append(dict(cik=7, taxonomy="us-gaap", tag=tag, unit="USD", start=pd.NaT, end=e, val=v,
                             accn="a", fy=e.year, fp="Q1", form="10-Q", filed=filed, frame=None, n_filings=1, which="only"))
    raw = pd.DataFrame(rows)
    W, dei = build_panel(raw)
    assert len(W) == 8
    assert W["ttm_net_income"].notna().sum() == 5           # first TTM at the 4th quarter
    assert W.iloc[3]["ttm_net_income"] == 10 + 11 + 12 + 13
    assert (W["avail_date"] - W["end"]).dt.days.eq(40).all()
    assert W.iloc[0]["equity"] == 1000 and W.iloc[7]["assets"] == 2007


def test_fix_share_units_rescales_outlier():
    H = pd.DataFrame({
        "cusip": ["037833100"] * 5, "report_date": [pd.Timestamp("2019-09-30")] * 5,
        "value_usd": [223.97 * 1000, 223.97 * 2000, 223.97 * 500, 223.97 * 800, 223.97 * 3000],
        "shares": [1000.0, 2000.0, 500.0, 800.0, 3000.0 * 100],   # last one reported x100
    })
    out = fix_share_units(H, lambda m: None)
    assert out["share_flag"].tolist()[:4] == ["ok"] * 4
    assert out["share_flag"].iloc[4] == "rescaled"
    assert abs(out["shares"].iloc[4] - 3000.0) < 1e-6
    assert out["shares_reported"].iloc[4] == 300000.0


def test_fix_share_units_unanchored_left_alone():
    H = pd.DataFrame({"cusip": ["X"] * 2, "report_date": [pd.Timestamp("2020-03-31")] * 2,
                      "value_usd": [100.0, 5000.0], "shares": [10.0, 5.0]})
    out = fix_share_units(H, lambda m: None)
    assert out["share_flag"].tolist() == ["unanchored"] * 2
    assert out["shares"].tolist() == [10.0, 5.0]


def test_classify_security():
    assert classify_security("APPLE INC;COM NPV", "AAPL", True) == "common"
    assert classify_security("NATIONAL RETAIL PROPERTIES DEP", "NNNPRD", False) == "preferred"
    assert classify_security("FALCON CAP ACQUISITION CORP WT", "FCACW", False) == "warrant"
    assert classify_security("ISHARES RUSSELL 2000 ETF", "IWM", False) == "fund"
    assert classify_security("XYZ CORP SR NT 5.25% 2030", "XYZ30", False) == "debt"
    assert classify_security("SAP SE SPONSORED ADR", "SAP", True) == "adr"
    assert classify_security("GAMESTOP CORP (HLDG CO) CL A", "GME", True) == "common"


def test_norm_name():
    assert norm_name("Apple Inc.") == "APPLE"
    assert norm_name("JOHNSON & JOHNSON") == "JOHNSON AND JOHNSON"
    assert norm_name("Berkshire Hathaway Inc CL B") == "BERKSHIRE HATHAWAY"


if __name__ == "__main__":
    import inspect
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and inspect.isfunction(fn):
            try:
                fn(); print("PASS", name)
            except AssertionError as ex:
                fails += 1; print("FAIL", name, ex)
            except Exception as ex:
                fails += 1; print("ERROR", name, type(ex).__name__, ex)
    sys.exit(1 if fails else 0)
