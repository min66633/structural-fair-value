# -*- coding: utf-8 -*-
"""Data-quality gate: invariants that must hold after every pipeline run.

A monthly-updated valuation tool fails quietly, not loudly: a stale macro file
or a half-written parquet does not raise, it just removes the newest month or
shifts a column. These checks are the tripwire. Each returns PASS, FAIL, WARN
or SKIP (input not built yet) and the run exits non-zero if anything FAILs.

Groups
  keys        one row per documented key in every table
  pit         point in time: no input dated after the month it is used in
  identity    algebraic identities that must hold exactly (component sums,
              premium decomposition, winsorised returns equal their source)
  coverage    firms per month, non-null shares, latest month present
  freshness   raw inputs and macro files recent enough for the latest month
  sanity      value ranges that indicate a broken unit or a runaway path

Outputs reports/selfcheck.md   (exit code 1 if any FAIL)
Run: python -m sfv.selfcheck [--quiet]
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import config as C

MACRO_MAX_AGE_DAYS = 45          # DGS10 / ERP refresh cadence
RAW_MAX_AGE_DAYS = 120           # 13F and N-PORT arrive quarterly
RESULTS: list[tuple[str, str, str, str]] = []     # group, name, status, detail


def record(group: str, name: str, status: str, detail: str = "") -> None:
    RESULTS.append((group, name, status, detail))


def check(group: str, name: str, fn, warn_only: bool = False) -> None:
    """Run one check; fn returns (ok, detail) or raises. Missing input -> SKIP."""
    try:
        ok, detail = fn()
    except FileNotFoundError as ex:
        record(group, name, "SKIP", f"missing input: {getattr(ex, 'filename', ex)}")
        return
    except Exception as ex:  # noqa: BLE001
        record(group, name, "FAIL", f"{type(ex).__name__}: {ex}")
        return
    record(group, name, "PASS" if ok else ("WARN" if warn_only else "FAIL"), detail)


def load(name: str, columns=None) -> pd.DataFrame:
    f = C.PQ / name
    if not f.exists():
        raise FileNotFoundError(str(f))
    return pd.read_parquet(f, columns=columns)


def has(name: str) -> bool:
    return (C.PQ / name).exists()


def file_age_days(path) -> float:
    return (time.time() - path.stat().st_mtime) / 86400.0


# ----------------------------------------------------------------- key uniqueness
KEYS = [("fund_quarterly.parquet", ["cik", "end"]), ("intangibles_quarterly.parquet", ["cik", "end"]),
        ("universe_monthly.parquet", ["cik", "month"]), ("omega_firm_year.parquet", ["cik", "asof_year"]),
        ("rim_monthly.parquet", ["cik", "month"]), ("demand_monthly.parquet", ["cik", "month"]),
        ("returns_quarterly.parquet", ["cik", "qend"]), ("decomp_monthly.parquet", ["cik", "month"]),
        ("decomp_monthly_momexp.parquet", ["cik", "month"]), ("validation_panel_mom.parquet", ["cik", "qend"]),
        ("rim_monthly_grbl.parquet", ["cik", "month"]), ("expectations_monthly.parquet", ["cik", "month"]),
        ("backlog_quarterly.parquet", ["cik", "end"])]


def keys_checks() -> None:
    for f, k in KEYS:
        def fn(f=f, k=k):
            d = load(f, columns=k)
            n = int(d.duplicated(k).sum())
            return n == 0, f"{len(d):,} rows, {n:,} duplicate keys"
        check("keys", f"{f} unique on {'+'.join(k)}", fn)


# ----------------------------------------------------------------- point in time
def pit_checks() -> None:
    def fundamentals():
        R = load("rim_monthly.parquet", columns=["month", "f_avail", "f_end"])
        bad = int((R["f_avail"] > R["month"]).sum())
        age = (R["month"] - R["f_end"]).dt.days
        return bad == 0, f"{bad} rows use fundamentals filed after the month; fundamentals age median {age.median():.0f}d, max {age.max():.0f}d"
    check("pit", "V_F uses only filings available at the month", fundamentals)

    def omega():
        R = load("rim_monthly.parquet", columns=["month", "asof_year"])
        R = R.dropna(subset=["asof_year"])
        usable = pd.to_datetime((R["asof_year"].astype(int) + 1).astype(str) + "-06-30")
        bad = int((usable > R["month"]).sum())
        return bad == 0, f"{bad} rows use a persistence model fitted on data not yet public"
    check("pit", "persistence omega usable from 30 June of the following year", omega)

    def f13():
        if not has("demand_monthly.parquet"):
            raise FileNotFoundError(str(C.PQ / "demand_monthly.parquet"))
        D = load("demand_monthly.parquet", columns=["month", "f13_period", "inst_share"])
        d = D.dropna(subset=["f13_period", "inst_share"])
        lag = (d["month"] - d["f13_period"]).dt.days
        bad = int((lag < 46).sum())
        return bad == 0, f"{bad} rows use 13F holdings before the 46-day deadline; lag median {lag.median():.0f}d"
    check("pit", "13F holdings used from quarter end + 46 days", f13)

    def fwd_returns():
        R = load("returns_quarterly.parquet", columns=["qend", "qend_next"])
        gap = (R["qend_next"] - R["qend"]).dt.days
        return bool((gap > 0).all() and gap.between(80, 100).all()), f"gap min {gap.min()}d max {gap.max()}d"
    check("pit", "quarterly returns are forward and one quarter long", fwd_returns)


# ----------------------------------------------------------------- identities
def identity_checks() -> None:
    def decomp(file="decomp_monthly_momexp.parquet"):
        D = load(file)
        comp = [c for c in D.columns if (c.startswith("delta_") and c.endswith("_A")) or c in ("ind_fe_A", "eps_A")]
        sub = D.dropna(subset=["eps_A"])
        resid = sub["y"] - sub[comp].sum(axis=1)
        sd = resid.groupby(sub["month"]).std().max()
        mean_eps = sub.groupby("month")["eps_A"].mean().abs().max()
        return bool(sd < 1e-9 and mean_eps < 1e-9), f"components {len(comp)}, max within-month sd {sd:.1e}, max |mean eps| {mean_eps:.1e}"
    check("identity", "decomposition components + residual reconstruct log(P/V_F)", decomp)

    def premium():
        E = load("expectations_monthly.parquet", columns=["log_pv", "log_vgr_vf", "log_p_vgr"])
        ok = E[["log_pv", "log_vgr_vf", "log_p_vgr"]].notna().all(axis=1)
        diff = (E.loc[ok, "log_pv"] - E.loc[ok, "log_vgr_vf"] - E.loc[ok, "log_p_vgr"]).abs().max()
        return bool(diff < 1e-9), f"max |log(P/V_F) - log(V_gr/V_F) - log(P/V_gr)| = {diff:.1e} on {int(ok.sum()):,} rows"
    check("identity", "premium splits into fundamental expectation and the rest", premium)

    def winsor():
        V = load("validation_panel_mom.parquet", columns=["cik", "qend", "r1"])
        R = load("returns_quarterly.parquet", columns=["cik", "qend", "ret_tot"])
        d = V.merge(R, on=["cik", "qend"], how="left").dropna(subset=["r1", "ret_tot"])
        g = d.groupby("qend")["ret_tot"]
        exp = d["ret_tot"].clip(g.transform(lambda s: s.quantile(.01)), g.transform(lambda s: s.quantile(.99)))
        diff = (d["r1"] - exp).abs().max()
        return bool(diff < 1e-9), f"max |r1 - winsorised ret_tot| = {diff:.1e} on {len(d):,} rows"
    check("identity", "panel returns are the winsorised 13F returns", winsor)

    def returns_vs_yf():
        R = load("returns_quarterly.parquet", columns=["ret_tot", "ret_yf"]).dropna()
        c = float(np.corrcoef(R["ret_tot"], R["ret_yf"])[0, 1])
        return c > 0.99, f"corr {c:.4f} on {len(R):,} overlapping quarters"
    check("identity", "13F implied returns agree with yfinance where both exist", returns_vs_yf)


# ----------------------------------------------------------------- coverage
def coverage_checks(expect_month: pd.Timestamp | None) -> None:
    def universe():
        U = load("universe_monthly.parquet", columns=["cik", "month", "in_universe", "mktcap"])
        u = U[U["in_universe"]]
        cnt = u.groupby("month").size()
        last = cnt.index.max()
        return bool(cnt.iloc[-1] >= 500), f"latest month {last.date()}: {cnt.iloc[-1]:,} firms (min over sample {cnt.min():,})"
    check("coverage", "in-universe firm count in the latest month", universe)

    def valuation():
        R = load("rim_monthly.parquet", columns=["cik", "month", "v_f", "log_pv"])
        last = R["month"].max()
        d = R[R["month"] == last]
        return bool(len(d) >= 500 and d["log_pv"].notna().mean() > 0.95), \
            f"latest month {last.date()}: {len(d):,} valued firms, log(P/V_F) present {d['log_pv'].notna().mean()*100:.1f}%"
    check("coverage", "valuations in the latest month", valuation)

    def expectations():
        E = load("expectations_monthly.parquet", columns=["cik", "month", "g_star_10y", "omega_star", "T_star"])
        last = E["month"].max()
        d = E[E["month"] == last]
        return bool(d["g_star_10y"].notna().mean() > 0.5), \
            f"latest month {last.date()}: g* {d['g_star_10y'].notna().mean()*100:.0f}%, omega* {d['omega_star'].notna().mean()*100:.0f}%, T* {d['T_star'].notna().mean()*100:.0f}%"
    check("coverage", "required-expectation coverage in the latest month", expectations)

    if expect_month is not None:
        def latest(expect_month=expect_month):
            R = load("rim_monthly.parquet", columns=["month"])
            got = R["month"].max()
            return got >= expect_month, f"expected >= {expect_month.date()}, got {got.date()}"
        check("coverage", "valuation table reaches the expected latest month", latest)

    def chain():
        R = load("rim_monthly.parquet", columns=["cik", "month"])
        pieces = {}
        for f in ("decomp_monthly_momexp.parquet", "expectations_monthly.parquet", "rim_monthly_grbl.parquet"):
            if has(f):
                d = load(f, columns=["cik", "month"])
                pieces[f] = len(d.merge(R, on=["cik", "month"], how="inner")) == len(d)
        ok = all(pieces.values())
        return ok, "; ".join(f"{k}: {'subset' if v else 'HAS ROWS OUTSIDE rim_monthly'}" for k, v in pieces.items())
    check("identity", "downstream tables are subsets of the valuation table", chain)


# ----------------------------------------------------------------- sanity
def sanity_checks() -> None:
    def values():
        R = load("rim_monthly.parquet")
        finite = bool(np.isfinite(R["v_f"]).all())
        pos = float((R["v_f"] > 0).mean())
        om = float(R["omega"].max())                       # raw prediction, capped at 0.95 in sfv.persistence
        applied = float(R["omega_applied"].max()) if "omega_applied" in R.columns else np.nan
        rr = (float(R["r"].min()), float(R["r"].max()))
        ok = finite and pos > 0.9 and om <= 0.9501 and rr[0] > 0 and rr[1] < 0.5
        if "omega_applied" in R.columns:
            ok = ok and applied <= 0.9001
        return bool(ok), (f"V_F finite {finite}, >0 {pos*100:.1f}%, max omega (raw) {om:.3f}, "
                          f"max omega applied in the fade {applied:.3f}, r range [{rr[0]:.3f}, {rr[1]:.3f}]")
    check("sanity", "valuation outputs inside their documented bounds", values)

    def macro_stale():
        R = load("rim_monthly.parquet")
        if "macro_stale" not in R.columns:
            return True, "column not present (table predates the flag)"
        n = int(R["macro_stale"].sum())
        return n == 0, f"{n:,} rows valued with a carried-forward interest rate or ERP"
    check("sanity", "no valuation used a stale macro input", macro_stale, warn_only=True)

    def premium_level():
        R = load("rim_monthly.parquet", columns=["month", "mktcap", "v_f"])
        last = R[R["month"] == R["month"].max()]
        ok = last["v_f"] > 0
        vw = float(np.log(last.loc[ok, "mktcap"].sum() / last.loc[ok, "v_f"].sum()))
        return -1.0 < vw < 2.5, f"latest value-weighted log(P/V_F) = {vw:.2f} (outside [-1, 2.5] means a broken input, not a market call)"
    check("sanity", "aggregate premium in a plausible range", premium_level)

    def shares():
        U = load("universe_monthly.parquet", columns=["cik", "month", "shares", "mktcap", "in_universe"])
        u = U[U["in_universe"]]
        last = u[u["month"] == u["month"].max()]
        big = int((last["mktcap"] > 1e13).sum())        # > $10tn is a share-count typo, not a company
        return big == 0, f"latest month: {big} firms above $10tn market cap, largest {last['mktcap'].max()/1e12:.2f}tn"
    check("sanity", "no market cap implies a share-count typo", shares)


# ----------------------------------------------------------------- freshness
RAW_FILES = [("companyfacts.zip", C.RAW / "companyfacts.zip", RAW_MAX_AGE_DAYS),
             ("submissions.zip", C.RAW / "submissions.zip", RAW_MAX_AGE_DAYS),
             ("DGS10.csv", C.RAW / "macro" / "DGS10.csv", MACRO_MAX_AGE_DAYS),
             ("histimpl.html", C.RAW / "macro" / "histimpl.html", MACRO_MAX_AGE_DAYS)]


def extract_complete() -> None:
    """An interrupted XBRL extract leaves a truncated directory that looks valid."""
    for name in ("xbrl_raw", "xbrl_rpo"):
        def fn(name=name):
            import json as _json
            d = C.PQ / name
            if not d.exists():
                raise FileNotFoundError(str(d))
            n = len(list(d.glob("part-*.parquet")))
            mf = d / "_manifest.json"
            if not mf.exists():
                return True, f"{n} part files, no manifest (written before the atomic swap was added)"
            m = _json.loads(mf.read_text(encoding="utf-8"))
            ok = bool(m.get("complete")) and n == m.get("parts")
            return ok, f"{n} part files, manifest says {m.get('parts')} parts / {m.get('rows'):,} rows, written {m.get('written_at')}"
        check("keys", f"{name} extract is complete", fn)


def freshness_checks() -> None:
    for name, path, max_age in RAW_FILES:
        def fn(path=path, max_age=max_age):
            if not path.exists():
                raise FileNotFoundError(str(path))
            age = file_age_days(path)
            return age <= max_age, f"last modified {age:.0f} days ago (refresh after {max_age})"
        check("freshness", f"raw input {name}", fn, warn_only=True)

    def erp_year():
        from .rim import macro_inputs
        rf, erp = macro_inputs()
        R = load("rim_monthly.parquet", columns=["month"])
        need = int(R["month"].max().year) - 1
        return int(erp.index.max()) >= need and rf.index.max() >= R["month"].max(), \
            f"ERP through {int(erp.index.max())} (need {need}), 10-year yield through {rf.index.max().date()} (need {R['month'].max().date()})"
    check("freshness", "macro series cover the latest valuation month", erp_year, warn_only=True)

    def facts_lag():
        # SEC's companyfacts record can trail a firm's actual filings by months (2026-09: Coca-Cola,
        # Abbott, NextEra had July 10-Qs that the September bulk file still lacked). Such firms stay
        # in the universe on their older quarter, flagged in the report; this warns when many do.
        U = load("universe_monthly.parquet", columns=["month", "in_universe", "fund_age"])
        u = U[(U["month"] == U["month"].max()) & U["in_universe"]]
        share = float((u["fund_age"] > 150).mean()) if len(u) else 0.0
        return share <= 0.10, (f"{share*100:.0f}% of the latest month's universe is valued on a fundamentals record older than "
                               f"150 days (SEC companyfacts lag); median age {u['fund_age'].median():.0f}d")
    check("freshness", "fundamentals records keep up with filings", facts_lag, warn_only=True)


# ----------------------------------------------------------------- report
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--expect-month", default="", help="fail if the valuation table does not reach this month (YYYY-MM-DD)")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    exp = pd.Timestamp(a.expect_month) if a.expect_month else None

    t0 = time.time()
    keys_checks()
    extract_complete()
    pit_checks()
    identity_checks()
    coverage_checks(exp)
    sanity_checks()
    freshness_checks()

    R = pd.DataFrame(RESULTS, columns=["group", "check", "status", "detail"])
    n_fail = int((R["status"] == "FAIL").sum())
    n_warn = int((R["status"] == "WARN").sum())
    n_skip = int((R["status"] == "SKIP").sum())
    order = {"FAIL": 0, "WARN": 1, "SKIP": 2, "PASS": 3}
    L = ["# 데이터 품질 점검 (selfcheck)\n",
         f"> 실행 {datetime.now(timezone.utc).astimezone().strftime('%Y-%m-%d %H:%M')}. "
         f"검사 {len(R)}개 — 통과 {int((R.status == 'PASS').sum())}, 실패 {n_fail}, 경고 {n_warn}, 건너뜀 {n_skip}. "
         f"{'**실패 있음: 아래 FAIL 항목을 먼저 보라.**' if n_fail else '실패 없음.'}\n",
         "FAIL은 산출물을 믿을 수 없다는 뜻이고, WARN은 입력이 오래됐거나 값이 경계에 있다는 뜻이다. "
         "SKIP은 해당 단계를 아직 돌리지 않은 것이다.\n"]
    for g in ["keys", "pit", "identity", "coverage", "sanity", "freshness"]:
        d = R[R["group"] == g]
        if d.empty:
            continue
        titles = {"keys": "키 유일성", "pit": "시점 규칙 (point in time)", "identity": "항등식",
                  "coverage": "커버리지", "sanity": "값 범위", "freshness": "입력 신선도"}
        L.append(f"\n## {titles[g]}\n")
        d = d.assign(o=d["status"].map(order)).sort_values("o")
        L.append(d[["status", "check", "detail"]].to_markdown(index=False))
    (C.REPORTS / "selfcheck.md").write_text("\n".join(L), encoding="utf-8")
    # machine-readable status for the HTML report
    import json
    status = {"run_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="minutes"),
              "n_checks": len(R), "n_fail": n_fail, "n_warn": n_warn, "n_skip": n_skip,
              "issues": [{"status": r["status"], "check": r["check"], "detail": r["detail"]}
                         for _, r in R[R["status"].isin(["FAIL", "WARN"])].iterrows()]}
    (C.REPORTS / "selfcheck_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=1), encoding="utf-8")

    if not a.quiet:
        for _, r in R.assign(o=R["status"].map(order)).sort_values("o").iterrows():
            if r["status"] != "PASS":
                print(f"  [{r['status']}] {r['check']}  {r['detail']}", flush=True)
        print(f"selfcheck: {len(R)} checks, {n_fail} FAIL, {n_warn} WARN, {n_skip} SKIP  "
              f"({time.time()-t0:.0f}s) -> reports/selfcheck.md", flush=True)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
