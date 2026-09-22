# -*- coding: utf-8 -*-
"""Security master: CUSIP (13F / N-PORT world) <-> CIK (XBRL world) <-> ticker.

Nothing free links a CUSIP to a CIK directly, so the link is layered and every
row records how it was made:

  1. cusip -> ticker, dated      FTD files (2009-2026) and N-PORT identifiers
                                 (2019-2026). Both cover names that were later
                                 delisted.
  2. ticker -> cik               SEC company_tickers.json and the `tickers`
                                 field of submissions (current registrants).
  3. name -> cik                 for tickers no longer registered: normalised
                                 issuer name (FTD description, N-PORT issuer
                                 name) matched to SEC entity names and former
                                 names, restricted to entities that filed a
                                 10-K. Exact normalised match first, then a
                                 token-set similarity with a high threshold.

Multi-class issuers: the link is made per CUSIP9 (security), and cusip6
identifies the issuer. Downstream code aggregates ownership across classes of
one CIK where needed.

Outputs (data/parquet/)
    security_master.parquet   cusip, cusip6, ticker, cik, method, score, names, dates
    ticker_cik.parquet        ticker -> cik with source
Run: python -m sfv.idmap
"""
from __future__ import annotations

import json
import re
import sys
import time
from difflib import SequenceMatcher

import numpy as np
import pandas as pd

from . import config as C

SUFFIX = re.compile(
    r"\b(INC|INCORPORATED|CORP|CORPORATION|CO|COMPANY|LTD|LIMITED|PLC|LLC|LP|L P|NV|SA|AG|SE|"
    r"HLDGS|HOLDING|HOLDINGS|GROUP|GRP|THE|CL|CLASS|COM|COMMON|STOCK|SHS|SHARES|ORD|ORDINARY|NEW|DEL|"
    r"MD|SPONSORED|SPON|ADR|ADS|REIT|TR|TRUST|USD|PAR|VALUE|NAMEN|AKT|A|B|C)\b")


FUND_RX = re.compile(r"\b(ETF|ETN|ETFS|FUND|FD|INDEX|TRUST UNIT|UNIT TR|SPDR|ISHARES|VANGUARD|PROSHARES|DIREXION|"
                     r"INVESCO|WISDOMTREE|VANECK|GLOBAL X|FIRST TRUST|CLOSED END|CEF)\b")
PREF_RX = re.compile(r"\b(PFD|PREF|PREFERRED|DEP SH|DEP SHS|DEPOSITARY|DEPOSITORY|DEP RCPT|DEP RECEIPT|CUM|NON CUM|"
                     r"SER [A-Z]|SERIES [A-Z])\b")
WARR_RX = re.compile(r"\b(WT|WTS|WARRANT|WARRANTS|WS)\b")
RIGHT_RX = re.compile(r"\b(RT|RTS|RIGHT|RIGHTS)\b")
UNIT_RX = re.compile(r"\b(UNIT|UNITS|UT)\b")
DEBT_RX = re.compile(r"\b(NOTE|NOTES|NT|NTS|DEBENTURE|DEB|DEBS|BOND|BD|BDS|SUB NT|SR NT|CV NT|CONV NT|TRUPS|CAP SEC)\b")
ADR_RX = re.compile(r"\b(ADR|ADS|ADRS|ADSS|GDR|SPON|SPONSORED|AMERICAN DEP)\b")


def classify_security(desc, symbol, in_nport_ec: bool) -> str:
    """Rough security type from the FTD description and symbol suffix conventions.

    FTD symbols carry suffixes: W/WS warrants, PR preferred, R rights, U units,
    Q bankruptcy (still common). N-PORT's ASSET_CAT == EC marks common equity
    (ADRs included), which overrides a missing description.
    """
    d = (desc or "").upper()
    s = (symbol or "").upper()
    if FUND_RX.search(d):
        return "fund"
    if PREF_RX.search(d) or (re.search(r"(PR[A-Z]?|-P[A-Z]?|\.P[A-Z]?)$", s) and len(s) > 4 and not in_nport_ec):
        return "preferred"
    if WARR_RX.search(d) or (re.search(r"(W|WS|\.WS|\.W|-WT)$", s) and len(s) > 4 and not in_nport_ec):
        return "warrant"
    if RIGHT_RX.search(d) or (re.search(r"(R|RT|\.RT|\.R)$", s) and len(s) > 4 and not in_nport_ec):
        return "right"
    if UNIT_RX.search(d) or (re.search(r"(U|\.U|\.UN)$", s) and len(s) > 4 and not in_nport_ec):
        return "unit"
    if DEBT_RX.search(d):
        return "debt"
    if ADR_RX.search(d):
        return "adr"
    return "common"


def norm_name(s) -> str:
    if not isinstance(s, str):
        return ""
    s = s.upper().replace("&", " AND ").replace("+", " AND ")
    s = re.sub(r"[^A-Z0-9 ]", " ", s)
    s = SUFFIX.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


def load_inputs(say):
    ent = pd.read_parquet(C.PQ / "sec_entities.parquet")
    ent = ent[ent["cik"] > 0].copy()
    ent["nname"] = ent["name"].map(norm_name)
    ent["has_10k"] = ent["first_10k"].notna()
    # former names -> extra rows for matching
    fn_rows = []
    for cik, fnj in zip(ent["cik"], ent["former_names"]):
        try:
            for f in json.loads(fnj or "[]"):
                nm = norm_name(f.get("name"))
                if nm:
                    fn_rows.append((cik, nm))
        except Exception:
            continue
    former = pd.DataFrame(fn_rows, columns=["cik", "nname"]).drop_duplicates()
    say(f"  sec entities {len(ent):,} (with 10-K {ent.has_10k.sum():,}); former names {len(former):,}")

    tk = json.load(open(C.RAW / "company_tickers.json", encoding="utf-8"))
    t2c = pd.DataFrame([(v["ticker"].upper(), int(v["cik_str"])) for v in tk.values()], columns=["ticker", "cik"])
    t2c["source"] = "company_tickers"
    t2c_sub = ent[ent["tickers"] != ""][["cik", "tickers"]].copy()
    t2c_sub["ticker"] = t2c_sub["tickers"].str.split("|")
    t2c_sub = t2c_sub.explode("ticker")[["ticker", "cik"]]
    t2c_sub["ticker"] = t2c_sub["ticker"].str.upper().str.strip()
    t2c_sub["source"] = "submissions"
    T = pd.concat([t2c, t2c_sub]).drop_duplicates(["ticker", "cik"])
    # a ticker that maps to several CIKs (rare: reused symbols) is ambiguous -> drop
    amb = T.groupby("ticker")["cik"].nunique()
    T = T[~T["ticker"].isin(amb[amb > 1].index)]
    say(f"  ticker->cik current registrants {len(T):,} (ambiguous dropped {(amb > 1).sum()})")

    ftd = pd.read_parquet(C.PQ / "ftd_cusip_symbol.parquet")
    ftd = ftd.rename(columns={"symbol": "ticker", "description": "name_ftd"})
    np_ids = pd.read_parquet(C.PQ / "nport_identifiers.parquet") if (C.PQ / "nport_identifiers.parquet").exists() else pd.DataFrame()
    say(f"  ftd pairs {len(ftd):,}   nport pairs {len(np_ids):,}")
    return ent, former, T, ftd, np_ids


def build_cusip_ticker(ftd: pd.DataFrame, np_ids: pd.DataFrame) -> pd.DataFrame:
    a = ftd[["cusip", "ticker", "first_date", "last_date", "n_days", "name_ftd"]].copy()
    a["src"] = "ftd"
    if len(np_ids):
        b = np_ids.rename(columns={"first_seen": "first_date", "last_seen": "last_date", "n": "n_days",
                                   "issuer_name": "name_nport"})[["cusip", "ticker", "first_date", "last_date", "n_days", "name_nport"]]
        b["src"] = "nport"
        a = pd.concat([a, b], ignore_index=True)
    # normalise symbols: FTD uses e.g. BRK.B / BRKB variants; keep as-is but also a stripped key
    a["ticker"] = a["ticker"].str.upper().str.replace(r"[^A-Z0-9.\-]", "", regex=True)
    a = a[a["ticker"] != ""]
    g = (a.groupby(["cusip", "ticker"])
           .agg(first_date=("first_date", "min"), last_date=("last_date", "max"), n_days=("n_days", "sum"),
                name_ftd=("name_ftd", "first"), name_nport=("name_nport", "first"),
                sources=("src", lambda s: "|".join(sorted(set(s))))).reset_index())
    return g


def match_names(cands: pd.DataFrame, ent: pd.DataFrame, former: pd.DataFrame, say) -> pd.DataFrame:
    """Name -> CIK for CUSIPs whose ticker is not a current registrant."""
    pool = ent[ent["has_10k"]][["cik", "nname"]]
    pool = pd.concat([pool, former[former["cik"].isin(pool["cik"])]]).drop_duplicates()
    pool = pool[pool["nname"].str.len() >= 3]
    exact = pool.groupby("nname")["cik"].agg(lambda s: s.iloc[0] if s.nunique() == 1 else -1)
    out = []
    by_first_token: dict[str, list[tuple[str, int]]] = {}
    for nm, cik in zip(pool["nname"], pool["cik"]):
        by_first_token.setdefault(nm.split(" ")[0], []).append((nm, cik))
    t0 = time.time()
    for i, (cusip, nm) in enumerate(zip(cands["cusip"], cands["nname"])):
        if not nm:
            out.append((cusip, None, "none", 0.0)); continue
        c = exact.get(nm)
        if c is not None and c > 0:
            out.append((cusip, int(c), "name-exact", 1.0)); continue
        best, best_c = 0.0, None
        for pn, pc in by_first_token.get(nm.split(" ")[0], []):
            r = SequenceMatcher(None, nm, pn).ratio()
            if r > best:
                best, best_c = r, pc
        if best >= 0.90 and best_c is not None:
            out.append((cusip, int(best_c), "name-fuzzy", round(best, 3)))
        else:
            out.append((cusip, None, "unmatched", round(best, 3)))
        if (i + 1) % 5000 == 0:
            say(f"    name matching {i+1:,}/{len(cands):,}  {(time.time()-t0)/60:.1f} min")
    return pd.DataFrame(out, columns=["cusip", "cik", "method", "score"])


def main() -> int:
    log = open(C.LOGS / "idmap.log", "w", encoding="utf-8")

    def say(msg):
        print(msg, flush=True); log.write(msg + "\n"); log.flush()

    say("loading inputs")
    ent, former, T, ftd, np_ids = load_inputs(say)
    T.to_parquet(C.PQ / "ticker_cik.parquet", index=False)

    CT = build_cusip_ticker(ftd, np_ids)
    say(f"  cusip-ticker pairs {len(CT):,}  cusips {CT.cusip.nunique():,}")
    # latest ticker per cusip (by last_date, then n_days), preferring a clean
    # symbol: FTD appends XXXX-style suffixes to when-issued and odd lines
    # (GOOGLXXXX next to GOOGL) and those never exist in a price feed
    CT["clean_symbol"] = ~CT["ticker"].str.contains("XXXX|WI$|\\.WI$", regex=True) & (CT["ticker"].str.len() <= 6)
    latest = CT.sort_values(["cusip", "clean_symbol", "last_date", "n_days"]).drop_duplicates("cusip", keep="last")
    latest = latest.drop(columns=["clean_symbol"])

    # step 2: ticker -> cik for current registrants (try exact, then dot/dash variants)
    tmap = dict(zip(T["ticker"], T["cik"]))

    def via_ticker(t):
        for v in (t, t.replace(".", "-"), t.replace("-", "."), t.replace(".", ""), t.replace("-", "")):
            if v in tmap:
                return tmap[v]
        return None

    latest = latest.copy()
    latest["cik"] = latest["ticker"].map(via_ticker)
    latest["method"] = np.where(latest["cik"].notna(), "ticker-current", "")
    latest["score"] = np.where(latest["cik"].notna(), 1.0, 0.0)
    say(f"  linked via current ticker: {latest.cik.notna().sum():,} / {len(latest):,}")

    # step 3: name matching for the rest
    rest = latest[latest["cik"].isna()].copy()
    rest["nname"] = rest["name_nport"].where(rest["name_nport"].notna(), rest["name_ftd"]).map(norm_name)
    M = match_names(rest[["cusip", "nname"]], ent, former, say)
    M = M.set_index("cusip")
    latest.loc[rest.index, "cik"] = M.loc[rest["cusip"], "cik"].values
    latest.loc[rest.index, "method"] = M.loc[rest["cusip"], "method"].values
    latest.loc[rest.index, "score"] = M.loc[rest["cusip"], "score"].values
    say(f"  after name matching: linked {latest.cik.notna().sum():,} / {len(latest):,}  "
        f"({latest.method.value_counts().to_dict()})")

    # Predecessor securities: a CUSIP that stopped trading more than a year
    # before its ticker's last appearance means the ticker moved to a new
    # CUSIP - typically a holding-company reorganisation with a NEW CIK
    # (Google Inc 38259P508 -> Alphabet 02079K305, both GOOGL). Linking the
    # old CUSIP to the current registrant would hand Google's 2010-2015
    # fundamentals to nobody. Re-link such CUSIPs by their own name; keep
    # the ticker link when the name resolves to the same CIK or to nothing.
    tk_last = CT.groupby("ticker")["last_date"].max()
    gap = (latest["ticker"].map(tk_last) - latest["last_date"]).dt.days
    pred = (latest["method"] == "ticker-current") & (gap > 365)
    if pred.any():
        cand = latest.loc[pred, ["cusip", "name_nport", "name_ftd"]].copy()
        cand["nname"] = cand["name_nport"].where(cand["name_nport"].notna(), cand["name_ftd"]).map(norm_name)
        M2 = match_names(cand[["cusip", "nname"]], ent, former, say).set_index("cusip")
        new_cik = M2.loc[latest.loc[pred, "cusip"], "cik"].values
        old_cik = latest.loc[pred, "cik"].values
        relink = pd.notna(new_cik) & (new_cik != old_cik)
        idx = latest.index[pred][relink]
        latest.loc[idx, "cik"] = new_cik[relink]
        latest.loc[idx, "method"] = "name-predecessor"
        latest.loc[idx, "score"] = M2.loc[latest.loc[idx, "cusip"], "score"].values
        say(f"  predecessor cusips (ticker moved on): {int(pred.sum()):,}; re-linked by name to a different CIK: {len(idx):,}")

    latest["cusip6"] = latest["cusip"].str[:6]
    # issuer-level consistency: within a cusip6, if some classes are linked and
    # others are not, propagate the modal CIK to the unlinked classes
    modal = (latest.dropna(subset=["cik"]).groupby("cusip6")["cik"]
                   .agg(lambda s: s.mode().iloc[0]))
    fill = latest["cik"].isna() & latest["cusip6"].isin(modal.index)
    latest.loc[fill, "cik"] = latest.loc[fill, "cusip6"].map(modal)
    latest.loc[fill, "method"] = "cusip6-propagated"
    latest["cik"] = latest["cik"].astype("Int64")
    latest = latest.merge(ent[["cik", "name", "sic", "sic_desc", "state_inc", "has_10k"]].rename(columns={"name": "name_sec"}),
                          on="cik", how="left")
    in_np = latest["sources"].str.contains("nport")
    latest["sec_type"] = [classify_security(d, s, n) for d, s, n in zip(latest["name_ftd"], latest["ticker"], in_np)]
    latest["foreign_cusip"] = latest["cusip"].str[0].str.isalpha()
    say("  security types: " + str(latest["sec_type"].value_counts().to_dict()))
    latest = latest[["cusip", "cusip6", "ticker", "cik", "method", "score", "sec_type", "foreign_cusip",
                     "first_date", "last_date", "n_days", "sources", "name_ftd", "name_nport", "name_sec",
                     "sic", "sic_desc", "state_inc", "has_10k"]]
    latest.to_parquet(C.PQ / "security_master.parquet", index=False)
    CT.to_parquet(C.PQ / "cusip_ticker_history.parquet", index=False)
    say(f"done: security master {len(latest):,} cusips, linked to cik {latest.cik.notna().sum():,}, "
        f"distinct ciks {latest.cik.nunique():,}")
    say(latest["method"].value_counts().to_string())
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
