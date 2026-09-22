# -*- coding: utf-8 -*-
"""Paths and constants shared by every SFV module.

Nothing here touches the network. The SEC contact goes in the User-Agent of
every EDGAR request because SEC's fair-access policy requires a real contact;
override with the SEC_CONTACT environment variable if the project moves.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Windows console (cp949) chokes on non-ASCII in logs; force UTF-8 once here.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:  # pragma: no cover - pipes, notebooks
        pass

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
PQ = DATA / "parquet"
REPORTS = ROOT / "reports"
LOGS = REPORTS / "logs"
for _d in (RAW, PQ, REPORTS, LOGS):
    _d.mkdir(parents=True, exist_ok=True)

# 13F structured-data bulk zips, one per quarter from 2013Q2 (see data/README.md).
# Point SFV_MACRO_DATA at the parent directory if the cache lives somewhere else.
MACRO_DATA = Path(os.environ.get("SFV_MACRO_DATA", str(RAW)))
F13_CACHE = MACRO_DATA / "13f_bulk"
F13_CUSIP_AGG = MACRO_DATA / "13f_cusip_agg.parquet"

# SEC's fair-access policy requires a real contact in the User-Agent. Set
#   SEC_CONTACT="Your Name your@email"
# before running any module that downloads from EDGAR; SEC rejects the placeholder.
SEC_CONTACT = os.environ.get("SEC_CONTACT", "SFV research (set the SEC_CONTACT environment variable)")
SEC_HEADERS = {"User-Agent": SEC_CONTACT, "Accept-Encoding": "gzip, deflate"}
SEC_MAX_RPS = 8          # SEC allows 10 req/s; stay under it

COMPANYFACTS_ZIP = RAW / "companyfacts.zip"
SUBMISSIONS_ZIP = RAW / "submissions.zip"
NPORT_DIR = RAW / "nport"
FTD_DIR = RAW / "ftd"

# Forms whose XBRL facts enter the fundamentals panel. 8-K is excluded: its
# XBRL is sporadic (recast segments, cover pages) and would make availability
# dates inconsistent across firms. 20-F/40-F filers use IFRS tags and are out
# of scope for the first pass (US domestic common stocks).
PANEL_FORMS = {"10-K", "10-K/A", "10-KT", "10-KT/A", "10-Q", "10-Q/A", "10-QT", "10-QT/A"}

# Sample boundaries
XBRL_START = "2009-01-01"
F13_START = "2013-04-01"
NPORT_START = "2019-07-01"
