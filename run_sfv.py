# -*- coding: utf-8 -*-
"""Monthly update runner: rebuild whatever is out of date, in dependency order.

The pipeline is twenty-odd modules that must run in a fixed order, each reading
the previous one's parquet. Running them by hand is how a month gets valued with
last quarter's holdings. This runner knows the order and the file dependencies,
skips a stage whose outputs are newer than every input (including the module's
own source), and runs each stage in its own process because this machine has
about 3 GB of usable memory and pandas does not give it back.

Groups
  ingest   parses the raw downloads: companyfacts, submissions, 13F, N-PORT,
           FTD, yfinance prices. Slow, needs the files to be refreshed by hand
           (see --check), and therefore not part of the default run.
  core     everything from the fundamentals panel to the decision table.
  test     the pre-registered return-prediction tests. A finished finding over
           2014-2026, not a monthly output: the tool reads none of it, so the
           monthly run builds the panel these need and stops there.
  extra    the analysis reports behind FRAMEWORK.md, not needed to trade.

Typical uses
  python run_sfv.py --check                 what is stale and how old the inputs are
  python run_sfv.py                         rebuild the core chain and the report
  python run_sfv.py --group all             after refreshing the raw downloads
  python run_sfv.py --only rim,decompose    one or two stages
  python run_sfv.py --from expectations     that stage and everything after it

Refreshing the raw inputs is manual and outside this script:
  data/raw/companyfacts.zip, submissions.zip   SEC bulk
  data/raw/nport/*.zip, data/raw/ftd/*.zip     SEC bulk
  data/raw/13f_bulk/*.zip                      SEC 13F structured data sets (or $SFV_MACRO_DATA/13f_bulk)
  data/raw/macro/DGS10.csv                     FRED 10-year Treasury
  data/raw/macro/histimpl.html                 Damodaran implied ERP
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from sfv import config as C

ROOT = Path(__file__).resolve().parent
PQ, REP, RAW = C.PQ, C.REPORTS, C.RAW
RPO_TAGS = ("RevenueRemainingPerformanceObligation,RevenueRemainingPerformanceObligationPercentage,"
            "ContractWithCustomerLiability,ContractWithCustomerLiabilityCurrent,ContractWithCustomerLiabilityNoncurrent")


@dataclass
class Stage:
    name: str
    cmd: list[str]
    outputs: list[Path]
    inputs: list[Path] = field(default_factory=list)
    group: str = "core"
    note: str = ""

    @property
    def code(self) -> Path | None:
        """The source file whose edits should invalidate this stage's outputs."""
        if "-m" in self.cmd:
            return ROOT / (self.cmd[self.cmd.index("-m") + 1].replace(".", "/") + ".py")
        for tok in self.cmd:
            if tok.endswith(".py"):
                return ROOT / tok
        return None


def pq(*n: str) -> list[Path]:
    return [PQ / x for x in n]


def rep(*n: str) -> list[Path]:
    return [REP / x for x in n]


MACRO = [RAW / "macro" / "DGS10.csv", RAW / "macro" / "histimpl.html"]
FUND = PQ / "fund_quarterly.parquet"
UNI = PQ / "universe_monthly.parquet"
INTA = PQ / "intangibles_quarterly.parquet"
RIM = PQ / "rim_monthly.parquet"
DEM = PQ / "demand_monthly.parquet"
RET = PQ / "returns_quarterly.parquet"
APP = ROOT / "app" / "data"

STAGES: list[Stage] = [
    # ---------------------------------------------------------------- ingest
    Stage("xbrl_extract", ["-m", "sfv.xbrl_extract"], pq("xbrl_raw"), [C.COMPANYFACTS_ZIP], "ingest",
          "companyfacts.zip -> first-reported XBRL facts"),
    Stage("submissions", ["-m", "sfv.submissions"], pq("sec_entities.parquet"), [C.SUBMISSIONS_ZIP], "ingest",
          "submissions.zip -> SIC, former names, tickers"),
    Stage("f13", ["-m", "sfv.f13"], pq("inst_own_quarterly.parquet", "f13_filings.parquet"), [C.F13_CACHE], "ingest",
          "13F bulk -> institutional holdings and implied prices"),
    Stage("nport", ["-m", "sfv.nport"], pq("nport_funds.parquet", "nport_holdings", "nport_identifiers.parquet"),
          [C.NPORT_DIR], "ingest", "N-PORT -> fund holdings and monthly flows"),
    Stage("ftd", ["-m", "sfv.ftd"], pq("ftd_cusip_symbol.parquet", "ftd_prices_monthly.parquet"), [C.FTD_DIR], "ingest",
          "fails-to-deliver -> prices for delisted names"),
    Stage("idmap", ["-m", "sfv.idmap"], pq("security_master.parquet", "cusip_ticker_history.parquet", "ticker_cik.parquet"),
          pq("sec_entities.parquet", "ftd_cusip_symbol.parquet", "nport_identifiers.parquet", "inst_own_quarterly.parquet")
          + [RAW / "company_tickers.json"], "ingest", "CUSIP <-> ticker <-> CIK"),
    Stage("prices", ["-m", "sfv.prices"], pq("prices_monthly.parquet"),
          pq("security_master.parquet") + [RAW / "company_tickers_exchange.json"], "ingest",
          "yfinance monthly prices, splits, dividends (network)"),
    # ---------------------------------------------------------------- core
    Stage("xbrl_panel", ["-m", "sfv.xbrl_panel"], pq("fund_quarterly.parquet", "shares_dei.parquet"), pq("xbrl_raw"),
          note="quarterly fundamentals panel with availability dates"),
    Stage("universe", ["-m", "sfv.universe"], pq("universe_monthly.parquet", "universe_ciks.parquet"),
          [FUND] + pq("shares_dei.parquet", "security_master.parquet", "prices_monthly.parquet", "ftd_prices_monthly.parquet",
                      "sec_entities.parquet"), note="monthly investable universe and point-in-time market cap"),
    Stage("intangibles", ["-m", "sfv.intangibles"], [INTA], [FUND], note="capitalise R&D and organisation capital"),
    Stage("persistence", ["-m", "sfv.persistence"],
          pq("omega_firm_year.parquet", "persistence_obs.parquet", "persistence_by_size.parquet",
             "persistence_multiyear.parquet", "persistence_fm_yearly.parquet"),
          [INTA, UNI, FUND], note="estimate the fade of excess ROE"),
    Stage("rim", ["-m", "sfv.rim"], [RIM], [UNI, INTA] + pq("omega_firm_year.parquet", "persistence_obs.parquet") + MACRO,
          note="residual income value V_F, industry fade target"),
    Stage("rim_coe", ["-m", "sfv.rim", "--roe-star", "coe", "--out", "rim_monthly_coe.parquet"], pq("rim_monthly_coe.parquet"),
          [UNI, INTA] + pq("omega_firm_year.parquet", "persistence_obs.parquet") + MACRO,
          note="same, cost-of-equity fade target (the conservative bound)"),
    Stage("demand", ["-m", "sfv.demand"], [DEM] + pq("fit_quarterly.parquet"),
          [UNI] + pq("security_master.parquet", "inst_own_quarterly.parquet", "prices_monthly.parquet",
                     "nport_funds.parquet", "nport_holdings"), note="institutional and passive demand, fund flows"),
    Stage("returns", ["-m", "sfv.returns"], [RET],
          pq("inst_own_quarterly.parquet") + [UNI, INTA] + pq("prices_monthly.parquet"),
          note="survivorship-free quarterly returns from 13F implied prices"),
    Stage("decompose", ["-m", "sfv.decompose"], pq("decomp_monthly.parquet", "decomp_coefs.parquet", "decomp_aggregate.parquet"),
          [RIM, DEM, INTA], note="cross-sectional decomposition (pre-registered spec)"),
    Stage("validate", ["-m", "sfv.validate", "--panel-only"], pq("validation_panel.parquet"),
          pq("decomp_monthly.parquet") + [RET, RIM, FUND],
          note="firm-quarter panel with forward returns (the prediction tests are in the 'test' group)"),
    Stage("implied", ["-m", "sfv.implied"], pq("implied_monthly.parquet"),
          [RIM, DEM] + pq("persistence_fm_yearly.parquet", "persistence_by_size.parquet", "persistence_multiyear.parquet"),
          note="market-implied persistence omega* and advantage period T*"),
    Stage("implied_coe", ["-m", "sfv.implied", "--rim-file", "rim_monthly_coe.parquet", "--label", "coe"],
          pq("implied_monthly_coe.parquet"), pq("rim_monthly_coe.parquet") + [DEM], note="same on the conservative value"),
    Stage("xbrl_rpo", ["-m", "sfv.xbrl_extract", "--tags", RPO_TAGS, "--cik-file", str(PQ / "universe_ciks.parquet"),
                       "--parts-dir", str(PQ / "xbrl_rpo")], pq("xbrl_rpo"),
          [C.COMPANYFACTS_ZIP] + pq("universe_ciks.parquet"), note="remaining performance obligations (backlog)"),
    Stage("backlog", ["-m", "sfv.backlog"], pq("backlog_quarterly.parquet"),
          pq("xbrl_rpo", "validation_panel.parquet") + [FUND], note="does backlog explain the residual and future revenue"),
    Stage("growth_rim", ["-m", "sfv.growth_rim"], pq("rim_monthly_gr.parquet", "growth_forecasts_gr.parquet"),
          [INTA, UNI, RIM] + pq("backlog_quarterly.parquet"), note="two-stage growth value"),
    Stage("growth_rim_bl", ["-m", "sfv.growth_rim", "--backlog"], pq("rim_monthly_grbl.parquet", "growth_forecasts_grbl.parquet"),
          [INTA, UNI, RIM] + pq("backlog_quarterly.parquet"), note="two-stage growth value using backlog (the final one)"),
    Stage("decompose_mom", ["-m", "sfv.decompose", "--momentum"],
          pq("decomp_monthly_mom.parquet", "decomp_coefs_mom.parquet", "decomp_aggregate_mom.parquet"),
          [RIM, DEM, INTA, UNI, RET], note="decomposition with the momentum component"),
    Stage("validate_mom", ["-m", "sfv.validate", "--decomp-file", "decomp_monthly_mom.parquet", "--label", "mom",
                           "--panel-only"], pq("validation_panel_mom.parquet"),
          pq("decomp_monthly_mom.parquet") + [RET, RIM, FUND], note="same panel on the momentum specification"),
    Stage("decompose_momexp", ["-m", "sfv.decompose", "--momentum", "--expectation"],
          pq("decomp_monthly_momexp.parquet", "decomp_coefs_momexp.parquet", "decomp_aggregate_momexp.parquet"),
          [RIM, DEM, INTA, UNI, RET] + pq("rim_monthly_grbl.parquet"),
          note="adds the fundamental-expectation component (the residual the report shows)"),
    Stage("expectations", ["-m", "sfv.expectations"],
          pq("expectations_monthly.parquet", "expectations_calibration.parquet", "expectations_calibration_cond.parquet"),
          [RIM, INTA] + pq("rim_monthly_grbl.parquet", "decomp_monthly_momexp.parquet", "validation_panel_mom.parquet"),
          note="invert the model per stock: required growth, the return axis, persistence, advantage period, base rates"),
    # The parameter band (required growth under every credible capitalisation
    # parameter set) is part of the decision table, so it is rebuilt monthly.
    # It re-runs intangibles -> persistence -> rim for five variants: about 3 minutes.
    Stage("intangible_sensitivity", ["scripts/intangible_sensitivity.py"],
          rep("intangible_sensitivity.md") + pq("expectations_band.parquet", "intangible_sensitivity_summary.parquet"),
          [FUND, RIM, INTA, UNI], note="required growth under other capitalisation parameters: the g* band"),
    Stage("selfcheck", ["-m", "sfv.selfcheck"], rep("selfcheck.md", "selfcheck_status.json"),
          [RIM, UNI, DEM, RET, INTA, FUND] + pq("expectations_monthly.parquet", "decomp_monthly_momexp.parquet",
                                                "validation_panel_mom.parquet") + MACRO,
          note="data-quality gate: keys, point in time, identities, coverage, freshness"),
    Stage("report", ["-m", "sfv.report"], rep("sfv_latest.csv", "sfv_report.html"),
          pq("expectations_monthly.parquet", "decomp_monthly_momexp.parquet", "backlog_quarterly.parquet",
             "implied_monthly.parquet", "expectations_calibration.parquet", "expectations_calibration_cond.parquet",
             "expectations_band.parquet") + rep("selfcheck_status.json"),
          note="the decision table: CSV and a self-contained HTML page"),
    # The desktop calculator (python -m app) reads a compact copy of the panels
    # (about 20 MB) instead of the 2 GB parquet directory. Same columns, one loader.
    Stage("app_export", ["-m", "sfv.store"],
          [APP / f for f in ("firms.parquet", "hist.parquet", "quarters.parquet", "band.parquet", "base_rates_bucket.parquet",
                             "base_rates_cond.parquet", "industry.parquet", "macro.json")],
          [RIM, INTA, UNI] + pq("expectations_monthly.parquet", "expectations_band.parquet", "expectations_calibration.parquet",
                                "expectations_calibration_cond.parquet", "universe_ciks.parquet"),
          note="the compact tables the desktop calculator reads (app/data)"),
    # ---------------------------------------------------------------- test
    # The pre-registered verdict is a finished finding over 2014-2026, not a
    # monthly output. These regenerate the record on demand; the tool itself
    # reads none of it.
    Stage("test_preregistered", ["-m", "sfv.validate"], rep("phase3_validation.md"),
          pq("decomp_monthly.parquet") + [RET, RIM, FUND], "test",
          "pre-registered return-prediction tests and the verdict (discard)"),
    Stage("test_momentum", ["-m", "sfv.validate", "--decomp-file", "decomp_monthly_mom.parquet", "--label", "mom"],
          rep("phase3_validation_mom.md"), pq("decomp_monthly_mom.parquet") + [RET, RIM, FUND], "test",
          "same tests with momentum separated out (post-hoc, still discard)"),
    # ---------------------------------------------------------------- extra
    Stage("backlog_robustness", ["scripts/phase5_robustness.py"], pq("backlog_panel.parquet") + rep("phase5_backlog_robustness.md"),
          pq("validation_panel.parquet", "backlog_quarterly.parquet"), "extra"),
    Stage("backlog_robustness_mom", ["scripts/phase5_robustness.py", "--label", "mom"],
          pq("backlog_panel_mom.parquet") + rep("phase5_backlog_robustness_mom.md"),
          pq("validation_panel_mom.parquet", "backlog_quarterly.parquet"), "extra"),
    Stage("vw_reversal", ["scripts/phase3_vw_reversal.py"], rep("phase3_vw_reversal.md"),
          pq("backlog_panel_mom.parquet", "validation_panel_mom.parquet"), "extra",
          "why the residual signal reverses when value-weighted"),
    Stage("sensitivity", ["scripts/phase7_sensitivity.py"], rep("phase7_sensitivity.md"), [RIM], "extra",
          "how far value moves with the persistence assumption alone"),
    Stage("model_sensitivity", ["scripts/model_sensitivity.py"], rep("model_sensitivity.md"),
          [RIM] + pq("decomp_monthly_momexp.parquet"), "extra",
          "does demand mismeasurement propagate, and how often do the bounds bind"),
    Stage("framework_poster", ["scripts/make_framework_poster.py"], rep("sfv_framework.png"),
          pq("implied_monthly.parquet", "expectations_monthly.parquet", "expectations_calibration.parquet",
             "persistence_by_size.parquet", "decomp_monthly_momexp.parquet", "expectations_band.parquet",
             "intangible_sensitivity_summary.parquet") + [RIM, UNI, RAW / "intangibles" / "ewens_peters_wang_2023.csv"],
          "extra", "one-page framework summary with the equations"),
    Stage("structure_diagram", ["scripts/make_structure_diagram.py"], rep("sfv_structure.png"),
          pq("expectations_monthly.parquet") + [UNI], "extra", "one-page structure diagram of the tool, annotated"),
    Stage("fit_by_float", ["scripts/fit_by_float.py"], rep("phase2_fit_by_float.md"), [DEM, UNI], "extra",
          "is the flow price multiplier larger where the float is thin (no)"),
    Stage("channel", ["scripts/phase6_channel.py"], rep("phase6_channel.csv"),
          [INTA] + pq("backlog_quarterly.parquet"), "extra", "backlog predicts revenue scale, not ROE"),
    Stage("value_report", ["scripts/phase6_report.py", "--alt-file", "rim_monthly_grbl.parquet",
                           "--alt-cols", "v_f_gr,log_pv_gr", "--label", "grbl", "--title", "성장+백로그"],
          rep("phase6_value_grbl.md"), pq("rim_monthly_grbl.parquet") + [RIM] + pq("validation_panel.parquet"), "extra"),
    Stage("coverage_report", ["scripts/phase0_coverage.py"], rep("phase0_coverage.md"), [FUND, UNI], "extra"),
    Stage("module_f_report", ["scripts/phase1_report.py"], rep("phase1_module_f.md"), [RIM, INTA], "extra"),
    Stage("module_d_report", ["scripts/phase2_report.py"], rep("phase2_module_d.md"), [DEM, RIM], "extra"),
]
BY_NAME = {s.name: s for s in STAGES}


# ----------------------------------------------------------------- staleness
def newest(p: Path) -> float:
    """Modification time of a file, or of the newest file inside a directory."""
    if not p.exists():
        return -1.0
    if p.is_dir():
        times = [f.stat().st_mtime for f in p.rglob("*") if f.is_file()]
        return max(times) if times else -1.0
    return p.stat().st_mtime


def stale_reason(s: Stage) -> str:
    missing = [o.name for o in s.outputs if newest(o) < 0]
    if missing:
        return f"output missing ({', '.join(missing[:3])})"
    out_t = min(newest(o) for o in s.outputs)
    srcs = list(s.inputs) + ([s.code] if s.code else [])
    newer = [p for p in srcs if p and newest(p) > out_t]
    if newer:
        return f"newer input: {', '.join(p.name for p in newer[:3])}"
    return ""


def fmt_age(p: Path) -> str:
    t = newest(p)
    if t < 0:
        return "missing"
    return f"{(time.time() - t) / 86400:.0f}d"


def check_report(stages: list[Stage]) -> None:
    print(f"\n  raw inputs (refresh by hand before --group all)")
    for p in [C.COMPANYFACTS_ZIP, C.SUBMISSIONS_ZIP, C.NPORT_DIR, C.FTD_DIR, C.F13_CACHE] + MACRO:
        print(f"    {p.name:24s} {fmt_age(p):>8s}  {p}")
    print(f"\n  stages")
    for s in stages:
        r = stale_reason(s)
        print(f"    {s.name:22s} {'STALE' if r else 'ok   '}  {r or 'outputs newer than all inputs'}")
    n = sum(1 for s in stages if stale_reason(s))
    print(f"\n  {n} of {len(stages)} stages would run\n")


# ----------------------------------------------------------------- run
def run_stage(s: Stage, log, timeout: int) -> tuple[bool, float]:
    cmd = [sys.executable, "-u"] + s.cmd
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    t0 = time.time()
    print(f"  > {s.name}: {' '.join(s.cmd)}", flush=True)
    log.write(f"\n===== {s.name}  {' '.join(cmd)}\n")
    log.flush()
    try:
        r = subprocess.run(cmd, cwd=str(ROOT), env=env, timeout=timeout,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                           encoding="utf-8", errors="replace")
        out = r.stdout or ""
        ok = r.returncode == 0
    except subprocess.TimeoutExpired as ex:
        out = (ex.stdout or "") + f"\nTIMEOUT after {timeout}s"
        ok = False
    log.write(out)
    log.flush()
    tail = [ln for ln in out.strip().splitlines() if ln.strip()][-3:]
    for ln in tail:
        print(f"      {ln[:160]}", flush=True)
    return ok, time.time() - t0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="SFV monthly update runner")
    ap.add_argument("--group", default="core", choices=["core", "ingest", "extra", "test", "all"],
                    help="which stages to consider (default core)")
    ap.add_argument("--only", default="", help="comma-separated stage names, in this order")
    ap.add_argument("--from", dest="start", default="", help="start at this stage and run everything after it")
    ap.add_argument("--force", action="store_true", help="run even when outputs are newer than inputs")
    ap.add_argument("--check", action="store_true", help="report input ages and what is stale, then exit")
    ap.add_argument("--list", action="store_true", help="list the stages in order, then exit")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--keep-going", action="store_true", help="continue after a stage fails")
    ap.add_argument("--timeout", type=int, default=7200, help="per-stage timeout in seconds")
    a = ap.parse_args(argv)

    if a.only:
        names = [n.strip() for n in a.only.split(",") if n.strip()]
        unknown = [n for n in names if n not in BY_NAME]
        if unknown:
            print(f"unknown stage(s): {', '.join(unknown)}\nknown: {', '.join(BY_NAME)}")
            return 2
        stages = [BY_NAME[n] for n in names]
    else:
        groups = {"core", "ingest", "extra", "test"} if a.group == "all" else {a.group}
        stages = [s for s in STAGES if s.group in groups]
        if a.start:
            if a.start not in BY_NAME:
                print(f"unknown stage: {a.start}")
                return 2
            order = [s.name for s in STAGES]
            i = order.index(a.start)
            stages = [s for s in stages if order.index(s.name) >= i]

    if a.list:
        for s in STAGES:
            print(f"  {s.group:7s} {s.name:22s} {s.note}")
        return 0
    if a.check:
        check_report(stages)
        return 0

    stamp = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d_%H%M%S")
    log_path = C.LOGS / f"run_sfv_{stamp}.log"
    print(f"SFV runner  group={a.group}{' from=' + a.start if a.start else ''}"
          f"{' force' if a.force else ''}  log {log_path.name}")
    for p in MACRO:
        age = (time.time() - newest(p)) / 86400 if newest(p) > 0 else 1e9
        if age > 45:
            print(f"  WARNING: {p.name} last refreshed {age:.0f} days ago; the newest month may be valued "
                  f"with a carried-forward rate (see reports/selfcheck.md)")

    results, failed = [], False
    with open(log_path, "w", encoding="utf-8") as log:
        log.write(f"SFV runner {stamp}  group={a.group} only={a.only} from={a.start} force={a.force}\n")
        for s in stages:
            reason = "forced" if a.force else stale_reason(s)
            if not reason:
                results.append((s.name, "skip", 0.0, "up to date"))
                print(f"  - {s.name}: up to date", flush=True)
                continue
            if a.dry_run:
                results.append((s.name, "would run", 0.0, reason))
                print(f"  > {s.name}: would run ({reason})", flush=True)
                continue
            ok, dt = run_stage(s, log, a.timeout)
            results.append((s.name, "ok" if ok else "FAILED", dt, reason))
            if not ok:
                failed = True
                print(f"    {s.name} FAILED after {dt/60:.1f} min - see {log_path}", flush=True)
                if not a.keep_going:
                    break

    total = sum(r[2] for r in results)
    print(f"\n  {'stage':22s} {'status':9s} {'min':>6s}  reason")
    for n, st, dt, why in results:
        print(f"  {n:22s} {st:9s} {dt/60:6.1f}  {why}")
    print(f"  total {total/60:.1f} min, log {log_path}")
    if not a.dry_run and any(r[1] == "ok" for r in results):
        for f in (REP / "sfv_report.html", REP / "sfv_latest.csv", REP / "selfcheck.md"):
            if f.exists():
                print(f"  -> {f}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
