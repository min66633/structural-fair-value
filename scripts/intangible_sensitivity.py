# -*- coding: utf-8 -*-
"""Do the intangible capitalisation parameters change the answer?

Three numbers turn R&D and SG&A into an asset: how fast knowledge decays, how
fast organisation capital decays, and how much of SG&A is investment rather
than expense. None is measured in our data; each comes from a paper. This
rebuilds the whole valuation chain under every credible parameter set and
reports what moves - the intangible stock, the aggregate premium, the
market-implied persistence, and a few large stocks.

Two sources, estimated by completely different means:
    Ewens, Peters and Wang (Management Science 2025) infer all three from what
      acquirers paid at firm exits. Five broad industries, full SIC coverage.
    Li and Hall (BEA WP2016-5) estimate R&D depreciation from the returns to
      R&D. Ten narrow R&D-intensive industries, no organisation parameters.

Output reports/intangible_sensitivity.md
Run: python scripts/intangible_sensitivity.py [--skip-build] [--keep]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sfv import config as C  # noqa: E402
from sfv.rim import OMEGA_MAX, X_INF_CAP  # noqa: E402
from sfv.implied import value_common_omega, solve_omega  # noqa: E402
from sfv.expectations import value_growth_h, cap_years, bisect_vec, G_LO, G_HI, H_LONG, OMEGA_HI  # noqa: E402
from sfv.rim import X0_CAP  # noqa: E402
from sfv.intangibles import RND_RATES, sic_rnd_rates, epw_params  # noqa: E402

VARIANTS = [
    ("base", [], "Ewens·Peters·Wang — R&D 33~50%, 조직 20%, SG&A 자본화 20~51% (현재 기본값)"),
    ("lihall", ["--params", "compustat"], "Li·Hall R&D 산업별 11~54% + 조직 20% · SG&A 30%"),
    ("libea", ["--params", "bea"], "Li·Hall BEA-NSF 열 6~49% + 조직 20% · SG&A 30%"),
    ("flat15", ["--params", "flat", "--d-rnd", "0.15"], "R&D 15% 단일 · 조직 20% · SG&A 30% (전통적 관례값)"),
    ("flat30", ["--params", "flat", "--d-rnd", "0.30"], "R&D 30% 단일 · 조직 20% · SG&A 30%"),
    ("no_sga", ["--sga-share", "0.0", "--params", "flat", "--d-rnd", "0.15"], "SG&A 자본화 0% — R&D만"),
]
TICKERS = ["NVDA", "AAPL", "MSFT", "GOOGL", "META", "LLY", "XOM"]
L = []


def p(t: str = "") -> None:
    L.append(t)


def run(cmd: list[str], log) -> None:
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    r = subprocess.run([sys.executable, "-u"] + cmd, cwd=str(ROOT), env=env,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    log.write(f"\n===== {' '.join(cmd)}\n{r.stdout}")
    log.flush()
    if r.returncode != 0:
        raise SystemExit(f"failed: {' '.join(cmd)}\n{(r.stdout or '')[-2000:]}")


def build(tag: str, extra: list[str], log) -> None:
    if tag == "base":
        return
    ip = f"intangibles_quarterly_{tag}.parquet"
    run(["-m", "sfv.intangibles"] + extra + ["--out", ip], log)
    run(["-m", "sfv.persistence", "--intangibles-file", ip, "--label", tag], log)
    run(["-m", "sfv.rim", "--intangibles-file", ip, "--omega-file", f"omega_firm_year_{tag}.parquet",
         "--persistence-obs", f"persistence_obs_{tag}.parquet", "--out", f"rim_monthly_{tag}.parquet",
         "--log-label", tag], log)


def summarise(tag: str) -> dict:
    f = "rim_monthly.parquet" if tag == "base" else f"rim_monthly_{tag}.parquet"
    R = pd.read_parquet(C.PQ / f, columns=["cik", "month", "ticker", "mktcap", "b_adj", "k_int", "x0", "a", "omega",
                                           "roe_star_ind", "payout", "r", "v_f", "log_pv", "roe_adj", "ttm_e_adj"])
    m = R["month"].max()
    d = R[R["month"] == m].reset_index(drop=True)
    ok = d["v_f"] > 0
    B0 = d["b_adj"].to_numpy(float)
    x0 = d["x0"].to_numpy(float)
    xinf = np.clip(d["a"].to_numpy(float) / (1.0 - np.minimum(d["omega"].to_numpy(float), OMEGA_MAX)), -X_INF_CAP, X_INF_CAP)
    rs = d["roe_star_ind"].to_numpy(float); po = d["payout"].to_numpy(float); r = d["r"].to_numpy(float)
    om = np.minimum(d["omega"].to_numpy(float), OMEGA_MAX)
    P = d["mktcap"].to_numpy(float); E0 = d["ttm_e_adj"].to_numpy(float)
    om_star_mkt, _ = solve_omega(B0, x0, xinf, rs, po, r, float(P.sum()))

    # the inversion, firm by firm: what this price requires under these parameters
    g = np.full(len(d), np.nan)
    idx = np.where(E0 > 0)[0]
    gs, st = bisect_vec(lambda gg: value_growth_h(B0[idx], E0[idx], gg, H_LONG, xinf[idx], om[idx], rs[idx], po[idx], r[idx]),
                        G_LO, G_HI, P[idx])
    g[idx] = np.where(st == 0, gs, np.nan)
    ostar = np.full(len(d), np.nan)
    j = np.where(x0 > xinf)[0]
    if len(j):
        v, sj = bisect_vec(lambda w: value_common_omega(B0[j], x0[j], xinf[j], rs[j], po[j], r[j], w), 0.0, OMEGA_HI, P[j])
        ostar[j] = np.where(sj == 0, v, np.nan)
    T = np.full(len(d), np.nan)
    k = np.where((rs + np.clip(x0, -X0_CAP, X0_CAP)) > r)[0]
    if len(k):
        T[k] = cap_years(B0[k], x0[k], rs[k], po[k], r[k], P[k])
    d = d.assign(g_star=g, omega_star=ostar, T_star=T)
    st_ = d.set_index("ticker")
    return {"K_int/B_adj 중앙값": (d["k_int"] / d["b_adj"]).median(),
            "조정 ROE 중앙값": d["roe_adj"].clip(-1, 1).median(),
            "지속성 ω 중앙값": d["omega"].median(),
            "log(P/V) 중앙값": d["log_pv"].median(),
            "시총가중 log(P/V)": float(np.log(d.loc[ok, "mktcap"].sum() / d.loc[ok, "v_f"].sum())),
            "시장 내재 ω*": om_star_mkt,
            "_stocks": {t: float(st_.loc[t, "log_pv"]) for t in TICKERS if t in st_.index},
            "_gstar": {t: float(st_.loc[t, "g_star"]) for t in TICKERS if t in st_.index},
            "_ostar": {t: float(st_.loc[t, "omega_star"]) for t in TICKERS if t in st_.index},
            "_tstar": {t: float(st_.loc[t, "T_star"]) for t in TICKERS if t in st_.index},
            "_panel": d.set_index("cik")[["log_pv", "g_star", "omega_star", "T_star"]],
            "_traits": d.set_index("cik")[["mktcap", "b_adj", "k_int", "roe_adj"]],
            "_month": m}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-build", action="store_true", help="reuse variant parquets already on disk")
    ap.add_argument("--keep", action="store_true", help="keep the variant parquets (about 300 MB) instead of deleting them")
    a = ap.parse_args()
    t0 = time.time()
    log = open(C.LOGS / "intangible_sensitivity.log", "w", encoding="utf-8")
    res = {}
    for tag, extra, label in VARIANTS:
        if not a.skip_build:
            print(f"  building {tag} ...", flush=True)
            build(tag, extra, log)
        res[tag] = summarise(tag)
        print(f"  {tag}: 시총가중 log(P/V) {res[tag]['시총가중 log(P/V)']:+.3f}, 내재 ω* {res[tag]['시장 내재 ω*']:.3f}", flush=True)

    m = res["base"]["_month"]
    p("# 무형자산 자본화 파라미터 민감도\n")
    p(f"> 생성 {pd.Timestamp.today().date()}. 평가 기준월 {m.date()}. R&D 상각률, 조직자본 상각률, SG&A 자본화 몫을 "
      "바꿔 무형자산 → 지속성 추정 → 가치평가 전 과정을 다시 계산했다. 세 값 모두 우리 데이터에서 측정되지 않고 "
      "문헌에서 가져오므로, 그 선택이 결론을 얼마나 움직이는지가 중요하다.\n")
    p("## 변형\n")
    p(pd.DataFrame([{"이름": t, "설명": lab} for t, _, lab in VARIANTS]).to_markdown(index=False))
    U = pd.read_parquet(C.PQ / "universe_monthly.parquet", columns=["cik", "sic", "in_universe", "mktcap", "month"])
    u = U[U["in_universe"] & (U["month"] == U["month"].max())].dropna(subset=["sic"]).copy()
    u["sic"] = u["sic"].astype("int64")

    p("\n### 기본값 — Ewens·Peters·Wang 파라미터 (Management Science 2025)\n")
    p("기업이 인수·상장폐지로 매각될 때 인수자가 실제로 지불한 값에서 세 파라미터를 동시에 식별한다. SIC 전체를 덮으며 "
      "Fama-French 5산업으로 묶인다. **조직자본 상각률은 산업별로 추정하지 않고 20%로 고정한다.** "
      "산업에 따라 크게 다른 것은 상각률이 아니라 SG&A 중 투자로 볼 몫이다.\n")
    E = epw_params()
    cov = u.merge(E.reset_index()[["sic", "industry5"]], on="sic", how="left")
    cnt5 = cov.groupby("industry5").agg(n=("cik", "size"), mcap=("mktcap", "sum"))
    E5 = E.drop_duplicates("industry5").set_index("industry5")
    rows5 = []
    for ind, r in E5.iterrows():
        c = cnt5.loc[ind] if ind in cnt5.index else None
        rows5.append({"산업 (FF5)": ind, "R&D 상각": f"{r['knowDepr']:.0%}", "조직자본 상각": f"{r['organDepr']:.0%}",
                      "SG&A 자본화 몫": f"{r['gamma']:.0%}",
                      "유니버스 종목": int(c["n"]) if c is not None else 0,
                      "시총 조$": round(float(c["mcap"]) / 1e12, 2) if c is not None else 0.0})
    p(pd.DataFrame(rows5).to_markdown(index=False))
    p(f"\n최신 월 {len(u):,}종목 중 {int(cov['industry5'].notna().sum()):,}종목"
      f"({cov['industry5'].notna().mean()*100:.0f}%)이 대응된다.")

    p("\n### 비교 대상 — Li·Hall R&D 상각률 (BEA WP2016-5, 표 1·표 3)\n")
    p("표 1의 SIC 대응을 그대로 쓴다. 항공우주와 자동차는 논문이 제시한 기대수익률 1% 재추정치이며, "
      "논문 자신이 8.9% 가정에서 나온 헤드라인 값(34~88%)이 그 산업에서는 성립하지 않는다고 밝힌다. "
      "조직자본 쪽 파라미터는 이 논문에 없으므로 관례값(20%·30%)을 쓴다.\n")
    tbl = sic_rnd_rates("compustat")
    u["rate"] = u["sic"].map(tbl)
    cnt = u.dropna(subset=["rate"]).groupby("rate").agg(n=("cik", "size"), mcap=("mktcap", "sum"))
    rows = []
    for name, codes, c_rate, b_rate in RND_RATES:
        sic_txt = ", ".join(f"{c[0]}–{c[1]}" if isinstance(c, tuple) else str(c) for c in codes)
        r = cnt.loc[c_rate] if c_rate in cnt.index else None
        rows.append({"산업": name, "SIC": sic_txt, "Compustat": f"{c_rate:.1%}", "BEA-NSF": f"{b_rate:.1%}",
                     "유니버스 종목": int(r["n"]) if r is not None else 0,
                     "시총 조$": round(float(r["mcap"]) / 1e12, 2) if r is not None else 0.0})
    p(pd.DataFrame(rows).to_markdown(index=False))
    matched = u["rate"].notna()
    p(f"\n대응된 기업은 최신 월 {int(matched.sum()):,}종목({matched.mean()*100:.0f}%)이지만 "
      f"**시가총액으로는 {u.loc[matched, 'mktcap'].sum()/u['mktcap'].sum()*100:.0f}%**다. "
      "표에 없는 산업은 전통적 기준값 15%를 유지한다. 그 기업들은 R&D가 거의 없어 선택의 영향도 작다.")

    p("\n## 결과\n")
    rows = {t: {k: v for k, v in res[t].items() if not k.startswith("_")} for t, _, _ in VARIANTS}
    T = pd.DataFrame(rows).T
    p(T.round(3).to_markdown())
    base = T.loc["base"]
    p(f"\n기본값 대비 시총가중 log(P/V) 변동 폭: "
      f"{(T['시총가중 log(P/V)'] - base['시총가중 log(P/V)']).abs().max():.3f} "
      f"(가치로는 최대 {abs(np.expm1(-(T['시총가중 log(P/V)'] - base['시총가중 log(P/V)'])).max())*100:.0f}%), "
      f"시장 내재 지속성 ω* 변동 폭: {(T['시장 내재 ω*'] - base['시장 내재 ω*']).abs().max():.3f}")

    p("\n## 종목별 log(P/V_F)\n")
    S = pd.DataFrame({t: res[t]["_stocks"] for t, _, _ in VARIANTS})
    S["최대-최소"] = S.max(axis=1) - S.min(axis=1)
    p(S.round(3).to_markdown())

    # ---- what actually matters: is the inversion steadier than the point value?
    p("\n## 가정에 덜 흔들리는 것은 무엇인가\n")
    p("종목 판단에 쓰는 세 가지를 같은 변형 범위에서 비교한다. 점 추정 가치 log(P/V_F), 가격이 요구하는 10년 성장률 g*, "
      "가격이 요구하는 초과 ROE 유지 기간 T*. 변동 폭이 작을수록 파라미터 선택에 덜 좌우된다.\n")
    rows = []
    six = {"v": [], "g": [], "T": []}        # numeric ranges of the named stocks, quoted in the reading below
    for t in TICKERS:
        if t not in res["base"]["_stocks"]:
            continue
        pick = lambda key: [x for x in (res[k][key].get(t) for k, _, _ in VARIANTS) if x is not None and np.isfinite(x)]
        lv, gv, tv = pick("_stocks"), pick("_gstar"), pick("_tstar")
        if lv:
            six["v"].append((np.exp(max(lv) - min(lv)) - 1) * 100)
        if gv:
            six["g"].append((max(gv) - min(gv)) * 100)
        if tv:
            six["T"].append(max(tv) - min(tv))
        rows.append({"종목": t,
                     "가치 배수 범위": f"{np.exp(min(lv)):.2f}~{np.exp(max(lv)):.2f}배" if lv else "",
                     "가치 변동": f"{(np.exp(max(lv)-min(lv))-1)*100:.0f}%" if lv else "",
                     "요구 성장 g*": f"{min(gv)*100:.1f}~{max(gv)*100:.1f}%/y" if gv else "",
                     "g* 변동": f"{(max(gv)-min(gv))*100:.1f}%p" if gv else "",
                     "요구 기간 T*": f"{min(tv):.0f}~{max(tv):.0f}년" if tv else "",
                     "T* 변동": f"{max(tv)-min(tv):.0f}년" if tv else ""})
    p(pd.DataFrame(rows).to_markdown(index=False))

    # per-stock band across parameter sets, for the decision table to display
    band = None
    for tag, _, _ in VARIANTS:
        q = res[tag]["_panel"].add_suffix(f"_{tag}")
        band = q if band is None else band.join(q, how="outer")
    out = pd.DataFrame(index=band.index)
    for col in ("log_pv", "g_star", "omega_star", "T_star"):
        cols = [f"{col}_{t}" for t, _, _ in VARIANTS if f"{col}_{t}" in band.columns]
        out[f"{col}_lo"] = band[cols].min(axis=1)
        out[f"{col}_hi"] = band[cols].max(axis=1)
        out[f"{col}_n"] = band[cols].notna().sum(axis=1)
    out["month"] = m
    out.reset_index().to_parquet(C.PQ / "expectations_band.parquet", index=False)
    p(f"\n종목별 범위는 `expectations_band.parquet`에 저장되어 종목별 표(`reports/sfv_latest.csv`)에 그대로 실린다. "
      f"{len(out):,}종목.")

    p("\n### 횡단면 순위는 유지되는가 (기본값과의 순위상관)\n")
    base_panel = res["base"]["_panel"]
    rc = []
    for tag, _, lab in VARIANTS:
        if tag == "base":
            continue
        j = base_panel.join(res[tag]["_panel"], how="inner", lsuffix="_b", rsuffix="_v")
        r = {"변형": tag}
        for col, nm in (("log_pv", "log(P/V_F)"), ("g_star", "요구 성장 g*"), ("T_star", "요구 기간 T*")):
            d2 = j[[f"{col}_b", f"{col}_v"]].dropna()
            r[nm] = round(float(d2.corr(method="spearman").iloc[0, 1]), 3) if len(d2) > 50 else np.nan
        rc.append(r)
    RC = pd.DataFrame(rc)
    p(RC.to_markdown(index=False))
    p("\n순위상관이 1에 가까우면 **어느 종목이 더 비싼가**라는 상대 판단은 파라미터를 바꿔도 그대로라는 뜻이다. 수준은 움직여도 순서는 남는다.")
    rho_cols = ["log(P/V_F)", "요구 성장 g*", "요구 기간 T*"]
    rho_lo, rho_hi = float(RC[rho_cols].min().min()), float(RC[rho_cols].max().max())

    # ---- the band for the typical stock, by size: the named stocks are all mega-caps
    p("\n### 구간 폭의 분포 — 규모별\n")
    p("위의 종목은 전부 초대형주다. 유니버스 전체에서 요구 성장 g*의 구간 폭(파라미터 조합 간 최대−최소)과 "
      "가치 변동(배수 최대/최소 − 1)이 규모에 따라 어떻게 다른지 본다. 최신 월 시총 순위 기준.\n")
    tr = res["base"]["_traits"]
    Wd = out.join(tr, how="left")
    Wd["rank"] = Wd["mktcap"].rank(ascending=False)
    Wd["g_w"] = (Wd["g_star_hi"] - Wd["g_star_lo"]) * 100
    Wd["v_s"] = (np.exp(Wd["log_pv_hi"] - Wd["log_pv_lo"]) - 1) * 100
    bw = []
    for lab, s in (("전체", Wd), ("시총 상위 50", Wd[Wd["rank"] <= 50]), ("상위 200", Wd[Wd["rank"] <= 200]),
                   ("201위 이하", Wd[Wd["rank"] > 200])):
        g_, v_ = s["g_w"].dropna(), s["v_s"].dropna()
        bw.append({"구간": lab, "g* 구간 있는 종목": int(len(g_)), "g* 폭 중앙값 %p": round(float(g_.median()), 1),
                   "p75": round(float(g_.quantile(.75)), 1), "p90": round(float(g_.quantile(.9)), 1),
                   "폭 ≤3%p": f"{(g_ <= 3).mean()*100:.0f}%", "폭 ≤5%p": f"{(g_ <= 5).mean()*100:.0f}%",
                   "가치 변동 중앙값": f"{v_.median():.0f}%", "가치 변동 p90": f"{v_.quantile(.9):.0f}%"})
    BW = pd.DataFrame(bw).set_index("구간")
    p(BW.to_markdown())
    top, rest, allr = BW.loc["상위 200"], BW.loc["201위 이하"], BW.loc["전체"]
    # what the width goes with: the share of value that sits in the book (and so in the capitalisation)
    Wd["int_int"] = Wd["k_int"] / Wd["b_adj"]
    Wd["b_p"] = Wd["b_adj"] / Wd["mktcap"]
    sp = {c: float(Wd[["g_w", c]].dropna().corr(method="spearman").iloc[0, 1]) for c in ("int_int", "b_p", "roe_adj")}
    p(f"\n초대형주에서 얻은 \"요구 성장은 거의 움직이지 않는다\"는 결론은 규모가 작아질수록 약해진다. 상위 200에서는 g* 폭 중앙값 "
      f"{top['g* 폭 중앙값 %p']}%p, 종목의 {top['폭 ≤5%p']}가 5%p 안이지만, 201위 이하에서는 중앙값 {rest['g* 폭 중앙값 %p']}%p, "
      f"상위 10%는 {rest['p90']}%p 이상 벌어지고 5%p 안은 {rest['폭 ≤5%p']}뿐이다. 폭은 규모 자체보다 **가치 중 장부가에 실린 몫**과 함께 간다. "
      f"g* 폭과의 순위상관은 무형자본 비중 K_int/B_adj {sp['int_int']:+.2f}, 장부가/가격 {sp['b_p']:+.2f}, 조정 ROE {sp['roe_adj']:+.2f}. "
      "초대형주는 장부가가 가격의 5~20%라 자본화 조정이 가치에 거의 실리지 않고 요구 성장은 가격이 결정하지만, 장부가 비중이 큰 종목은 "
      "자본화 조정이 곧 가치 변화라서 요구 성장도 같이 움직인다. 그래서 세 번째 규칙(구간으로 읽기)이 소형주에서는 예외가 아니라 기본이다.")

    # summary for other tools (the poster reads the rank correlations and the aggregates from here)
    summ = T.rename(columns={"K_int/B_adj 중앙값": "k_int_b_adj_med", "조정 ROE 중앙값": "roe_adj_med", "지속성 ω 중앙값": "omega_med",
                             "log(P/V) 중앙값": "log_pv_med", "시총가중 log(P/V)": "log_pv_vw", "시장 내재 ω*": "omega_star_mkt"})
    summ.index.name = "variant"
    summ = summ.reset_index()
    summ["label"] = [lab for _, _, lab in VARIANTS]
    summ = summ.merge(RC.rename(columns={"변형": "variant", "log(P/V_F)": "rho_log_pv", "요구 성장 g*": "rho_g_star",
                                         "요구 기간 T*": "rho_T_star"}), on="variant", how="left")
    summ["month"] = m
    summ.to_parquet(C.PQ / "intangible_sensitivity_summary.parquet", index=False)

    def rng(v: list, fmt: str) -> str:
        return f"{min(v):{fmt}}~{max(v):{fmt}}" if v else "n/a"

    p("\n## 읽기 — 그래서 종목 판단을 어떻게 할 것인가\n")
    p(f"1. **점 추정 가치를 믿지 말고 요구 기대를 읽는다.** 위 초대형주에서 같은 변형 범위로 가치는 {rng(six['v'], '.0f')}% 흔들리는데 "
      f"요구 성장 g*는 {rng(six['g'], '.1f')}%p, 요구 기간 T*는 {rng(six['T'], '.0f')}년만 움직인다. \"이 회사가 10년간 연 20%씩 크는가\"라는 질문의 답은 "
      "상각률을 어떻게 잡든 거의 같다. 가치는 파라미터의 함수지만 요구 기대는 거의 가격의 함수다. "
      f"유니버스 전체로는 가치 변동 중앙값 {allr['가치 변동 중앙값']}, g* 폭 중앙값 {allr['g* 폭 중앙값 %p']}%p다.")
    p(f"2. **상대 비교는 안전하다.** 어떤 파라미터 집합을 써도 횡단면 순위상관이 {rho_lo:.2f}~{rho_hi:.2f}이다. "
      "\"메타가 애플보다 싼가\"는 파라미터 선택에 좌우되지 않는다.")
    p(f"3. **폭이 남는 종목은 폭째로 읽는다.** 요구 성장 구간이 넓은 종목은 점이 아니라 구간으로 봐야 한다. 초대형주에서도 4%p(애플·일라이릴리)가 나오고, "
      f"201위 이하에서는 {rest['폭 ≤5%p']}만 5%p 안이다. 그 구간을 `expectations_band.parquet`에 저장해 종목별 표에 함께 싣는다. "
      "자기 기대가 구간 안이면 모형의 정밀도를 넘어선 것이니 판단을 보류한다.")
    p("\n그 밖에 확인된 것:")
    p(f"- 시장 수준 진단은 파라미터에 의존하지 않는다. 내재 지속성 ω*가 모든 변형에서 {T['시장 내재 ω*'].min():.3f}~{T['시장 내재 ω*'].max():.3f} 안에 머문다. "
      "청산잉여관계 때문에 장부에 더한 만큼 이익에서 상각을 빼 1차 효과가 상쇄되기 때문이다.")
    p("- 남는 2차 효과는 ROE를 통해 지속성 추정으로 들어가는 경로다. 상각이 빠르면 무형자본이 작아져 ROE가 높아지고, "
      "초과 ROE의 분포와 지속성 추정이 함께 바뀐다. 위 표의 ω 중앙값 열이 그 크기다.")
    p("- 두 논문의 R&D 상각률은 식별 방법이 달라 크게 어긋난다(제약 15% vs 33%). 그 폭이 이 가정의 추정 불확실성이고, "
      "위 구간이 그것을 그대로 반영한다.")
    (C.REPORTS / "intangible_sensitivity.md").write_text("\n".join(L), encoding="utf-8")
    if not a.keep:
        freed = 0
        for tag, _, _ in VARIANTS:
            if tag == "base":
                continue
            for pat in (f"intangibles_quarterly_{tag}.parquet", f"rim_monthly_{tag}.parquet",
                        f"omega_firm_year_{tag}.parquet", f"persistence_obs_{tag}.parquet",
                        f"persistence_by_size_{tag}.parquet", f"persistence_multiyear_{tag}.parquet",
                        f"persistence_fm_yearly_{tag}.parquet"):
                f = C.PQ / pat
                if f.exists():
                    freed += f.stat().st_size
                    f.unlink()
            r = C.REPORTS / f"phase1_persistence_{tag}.md"
            if r.exists():
                r.unlink()
        print(f"  변형 산출물 삭제 ({freed/1e6:.0f} MB). 남기려면 --keep", flush=True)
    log.close()
    print("\n".join(L))
    print(f"\ndone {(time.time()-t0)/60:.1f} min -> reports/intangible_sensitivity.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
