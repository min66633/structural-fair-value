# -*- coding: utf-8 -*-
"""SEC submissions.zip -> entity master (SIC, names, tickers, former names).

submissions.zip holds one JSON per CIK for every EDGAR filer, individuals
included (~1 GB compressed). Only company-type entities matter here, so a
member is kept when it carries a SIC code or a ticker. Paging members
(CIK##########-submissions-NNN.json, older filings of heavy filers) are
skipped: the master needs identity fields, not the filing index.

Why this instead of per-CIK API calls: 20k calls at 8/s is 40 minutes and
rate-limit exposure; one download is a minute.

Output
    data/parquet/sec_entities.parquet
        cik, name, sic, sic_desc, tickers (|-joined), exchanges, state_inc,
        fye, category, former_names (json), n_recent_filings,
        first_10k, last_10k (from the recent index only)
Run: python -m sfv.submissions
"""
from __future__ import annotations

import json
import sys
import time
import zipfile

import pandas as pd

from . import config as C


def main() -> int:
    log = open(C.LOGS / "submissions.log", "w", encoding="utf-8")

    def say(msg):
        print(msg, flush=True); log.write(msg + "\n"); log.flush()

    z = zipfile.ZipFile(C.SUBMISSIONS_ZIP)
    names = [n for n in z.namelist() if n.lower().endswith(".json") and "-submissions-" not in n]
    say(f"submissions members: {len(names):,}")
    rows, n_bad = [], 0
    t0 = time.time()
    for i, n in enumerate(names, 1):
        try:
            j = json.loads(z.read(n))
        except Exception:
            n_bad += 1
            continue
        sic = (j.get("sic") or "").strip()
        tickers = [t for t in (j.get("tickers") or []) if t]
        if not sic and not tickers:
            continue
        rec = j.get("filings", {}).get("recent", {}) or {}
        forms = rec.get("form") or []
        dates = rec.get("filingDate") or []
        tenk = [d for f, d in zip(forms, dates) if isinstance(f, str) and f.startswith("10-K")]
        rows.append({
            "cik": int(j.get("cik") or 0),
            "name": j.get("name"),
            "sic": sic,
            "sic_desc": j.get("sicDescription"),
            "tickers": "|".join(tickers),
            "exchanges": "|".join([e or "" for e in (j.get("exchanges") or [])]),
            "state_inc": j.get("stateOfIncorporation"),
            "fye": j.get("fiscalYearEnd"),
            "category": j.get("category"),
            "entity_type": j.get("entityType"),
            "former_names": json.dumps(j.get("formerNames") or [], ensure_ascii=False),
            "n_recent_filings": len(forms),
            "first_10k": min(tenk) if tenk else None,
            "last_10k": max(tenk) if tenk else None,
        })
        if i % 100_000 == 0:
            say(f"  {i:,}/{len(names):,}  kept {len(rows):,}  {(time.time()-t0)/60:.1f} min")
    E = pd.DataFrame(rows)
    E["sic"] = pd.to_numeric(E["sic"], errors="coerce").astype("Int32")
    E.to_parquet(C.PQ / "sec_entities.parquet", index=False)
    say(f"done: kept {len(E):,} entities (bad {n_bad})  with sic {E.sic.notna().sum():,}  "
        f"with ticker {(E.tickers != '').sum():,}  {(time.time()-t0)/60:.1f} min")
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
