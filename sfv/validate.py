# -*- coding: utf-8 -*-
"""Module S, part 2: does the decomposition predict returns? Pre-registered tests.

Primary returns: survivorship-free quarterly returns from 13F implied prices
(sfv.returns), horizons 1, 2 and 4 quarters. Regressors at quarter end t are
the decomposition components at t (sfv.decompose) plus controls.

Tests
  1. Fama-MacBeth by quarter (Newey-West t, lags = horizon):
       ret = c + lam_y * log(P/V_F) + controls
       ret = c + lam_S delta_S + lam_T delta_T + lam_F delta_F + lam_eps eps + controls
     Predictions: lam_eps < 0 (mispricing converges), lam_T < 0 (reversal),
     lam_S < 0 and small (structural premium lowers expected return without
     reversing).
  2. Benchmarks with the same controls: log(B_adj/M), E/P, Bartram-Grinblatt
     peer-implied value (monthly OLS of market cap on accounting items), ICC.
  3. Decile portfolios, value-weighted, rebalanced quarterly: long-short
     mean, Sharpe, t. Composite score from expanding-window lambda estimates
     (>= 12 quarters of history) for an out-of-sample long-short.
  4. Delisting sensitivity: exits get a -30% return in the quarter after their
     last observation (a merger would be 0 or positive, a failure worse).
  5. Pre-registered verdict (plan section 2.5).

Outputs reports/phase3_validation[_label].md, data/parquet/validation_panel[_label].parquet
Run: python -m sfv.validate [--decomp-file decomp_monthly_mom.parquet --label mom]
     A labelled run is a post-hoc specification (e.g. the momentum-augmented
     decomposition); its verdict section applies the same criteria but is
     marked exploratory. If the decomposition carries delta_M, it enters the
     component regressions as a fifth component.
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pandas as pd

from . import config as C

HORIZONS = (1, 2, 4)
CONTROLS = ["log_size", "mom"]
MIN_N = 300
MIN_OOS_Q = 12
L = []


def p(t=""):
    L.append(t)


def nw_t(x, lags):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 6:
        return np.nan
    e = x - x.mean()
    v = e @ e / n
    for lag in range(1, lags + 1):
        if lag < n:
            v += 2 * (1 - lag / (lags + 1)) * (e[lag:] @ e[:-lag]) / n
    return x.mean() / np.sqrt(max(v, 1e-18) / n)


def fm(df, y, xs, lags, min_n=MIN_N):
    rows = []
    for q, d in df.groupby("qend"):
        d = d.dropna(subset=[y] + xs)
        if len(d) < min_n:
            continue
        Xm = np.column_stack([np.ones(len(d)), d[xs].values.astype(float)])
        b = np.linalg.lstsq(Xm, d[y].values.astype(float), rcond=None)[0]
        rows.append(pd.Series(b, index=["const"] + xs, name=q))
    T = pd.DataFrame(rows)
    return pd.DataFrame({"coef": T.mean(), "t": [nw_t(T[c].values, lags) for c in T.columns], "quarters": len(T)}), T


def winsor_q(df, col, lo=.01, hi=.99):
    g = df.groupby("qend")[col]
    return df[col].clip(g.transform(lambda s: s.quantile(lo)), g.transform(lambda s: s.quantile(hi)))


# ----------------------------------------------------------------- Bartram-Grinblatt
BG_ITEMS = ["assets", "liabilities", "equity", "cash", "debt_lt", "goodwill", "intangibles", "ppe_net",
            "ttm_revenue", "ttm_net_income", "ttm_cfo", "ttm_rnd", "ttm_sga", "ttm_dividends_paid", "ttm_buybacks", "ttm_capex"]


def bartram_grinblatt(panel: pd.DataFrame, say) -> pd.DataFrame:
    """Peer-implied value: quarterly OLS of market cap on accounting levels; mispricing = log(V_BG / P)."""
    Q = pd.read_parquet(C.PQ / "fund_quarterly.parquet", columns=["cik", "end", "avail_date"] + BG_ITEMS)
    Q = Q.dropna(subset=["avail_date"]).sort_values("avail_date")
    X = pd.merge_asof(panel[["cik", "qend", "mktcap"]].sort_values("qend"), Q.rename(columns={"avail_date": "bg_avail"}),
                      left_on="qend", right_on="bg_avail", by="cik", direction="backward")
    X = X[(X["qend"] - X["end"]).dt.days <= 400]
    out = []
    for q, d in X.groupby("qend"):
        d = d.copy()
        d[BG_ITEMS] = d[BG_ITEMS].fillna(0.0)
        for c in BG_ITEMS + ["mktcap"]:
            d[c] = d[c].clip(d[c].quantile(.005), d[c].quantile(.995))
        if len(d) < MIN_N:
            continue
        Xm = np.column_stack([np.ones(len(d)), d[BG_ITEMS].values.astype(float)])
        b = np.linalg.lstsq(Xm, d["mktcap"].values.astype(float), rcond=None)[0]
        d["v_bg"] = Xm @ b
        out.append(d[["cik", "qend", "v_bg"]])
    B = pd.concat(out, ignore_index=True)
    B = B.merge(panel[["cik", "qend", "mktcap"]], on=["cik", "qend"])
    B["bg_mis"] = np.log(B["v_bg"] / B["mktcap"]).where(B["v_bg"] > 0)
    say(f"  Bartram-Grinblatt: {len(B):,} firm-quarters, fitted>0 {(B.v_bg > 0).mean()*100:.1f}%")
    return B[["cik", "qend", "bg_mis"]]


# ----------------------------------------------------------------- panel
def build_panel(say, decomp_file: str = "decomp_monthly.parquet") -> pd.DataFrame:
    Dm = pd.read_parquet(C.PQ / decomp_file)
    Dm = Dm[Dm["month"].dt.month.isin([3, 6, 9, 12])].rename(columns={"month": "qend"})
    R = pd.read_parquet(C.PQ / "returns_quarterly.parquet")
    # forward returns at horizons 1, 2, 4 quarters (sum of log returns), exits handled two ways
    R = R.sort_values(["cik", "qend"])
    R["r1"] = R["ret_tot"]
    R["r1_dl"] = R["ret_tot"]
    ex = R[R["exit_flag"]].copy()
    ex["qend"] = ex["qend_next"]
    ex["r1"] = np.nan
    ex["r1_dl"] = np.log(0.70)
    Rx = pd.concat([R[["cik", "qend", "r1", "r1_dl"]], ex[["cik", "qend", "r1", "r1_dl"]]], ignore_index=True).sort_values(["cik", "qend"])
    g = Rx.groupby("cik")
    for h in (2, 4):
        for col in ("r1", "r1_dl"):
            s = Rx[col]
            acc = s.copy()
            for k in range(1, h):
                nxt = g[col].shift(-k)
                gapk = (g["qend"].shift(-k) - Rx["qend"]).dt.days
                acc = acc + nxt.where(gapk.between(k * 80, k * 100))
            Rx[f"r{h}" + ("_dl" if col.endswith("_dl") else "")] = acc
    P = Dm.merge(Rx, on=["cik", "qend"], how="left")
    # momentum from 13F prices: log(px_{t-1q} / px_{t-4q})
    px = R[["cik", "qend", "px_med"]].sort_values(["cik", "qend"])
    gp = px.groupby("cik")
    px["mom"] = np.log(gp["px_med"].shift(1) / gp["px_med"].shift(4)).where(
        (px["qend"] - gp["qend"].shift(4)).dt.days.between(340, 390))
    P = P.merge(px[["cik", "qend", "mom"]], on=["cik", "qend"], how="left")
    P["bm_adj"] = -P["log_pb_adj"]
    E = pd.read_parquet(C.PQ / "rim_monthly.parquet", columns=["cik", "month", "ttm_e_adj"]).rename(columns={"month": "qend"})
    P = P.merge(E, on=["cik", "qend"], how="left")
    P["ep"] = (P["ttm_e_adj"] / P["mktcap"]).clip(-1, 1)
    for c in ["r1", "r2", "r4", "r1_dl", "r2_dl", "r4_dl"]:
        if c in P:
            P[c] = winsor_q(P, c)
    B = bartram_grinblatt(P, say)
    P = P.merge(B, on=["cik", "qend"], how="left")
    say(f"  validation panel: {len(P):,} firm-quarters, {P.cik.nunique():,} firms, {P.qend.min().date()}..{P.qend.max().date()}; "
        f"r1 coverage {P.r1.notna().mean()*100:.1f}%")
    return P


# ----------------------------------------------------------------- portfolios
def decile_ls(P: pd.DataFrame, score: str, ret: str = "r1", vw: bool = True, lo_is_long: bool = True, lags: int = 1):
    """Long the lowest-score decile, short the highest (score = overpricing measure). lags: NW lags (4 for r4)."""
    rows = []
    for q, d in P.groupby("qend"):
        d = d.dropna(subset=[score, ret, "mktcap"])
        if len(d) < MIN_N:
            continue
        dec = pd.qcut(d[score].rank(method="first"), 10, labels=False)
        w = d["mktcap"] if vw else pd.Series(1.0, index=d.index)
        r = np.expm1(d[ret])
        lo = np.average(r[dec == 0], weights=w[dec == 0]); hi = np.average(r[dec == 9], weights=w[dec == 9])
        rows.append({"qend": q, "long": lo if lo_is_long else hi, "short": hi if lo_is_long else lo})
    T = pd.DataFrame(rows).set_index("qend")
    T["ls"] = T["long"] - T["short"]
    m, s = T["ls"].mean(), T["ls"].std()
    return {"quarters": len(T), "mean_q": m, "sharpe_ann": (m / s) * 2.0 if s > 0 else np.nan, "t": nw_t(T["ls"].values, lags),
            "long_mean": T["long"].mean(), "short_mean": T["short"].mean()}, T


def composite_oos(P: pd.DataFrame, comps: list[str], ret: str, say):
    """Expanding-window FM lambdas -> predicted return -> decile long-short, OOS from quarter MIN_OOS_Q."""
    qs = sorted(P["qend"].unique())
    rows = []
    for i, q in enumerate(qs):
        hist = P[P["qend"] < q]
        if hist["qend"].nunique() < MIN_OOS_Q:
            continue
        res, _ = fm(hist, ret, comps + CONTROLS, 1)
        lam = res["coef"]
        d = P[P["qend"] == q].dropna(subset=comps + CONTROLS + [ret, "mktcap"])
        if len(d) < MIN_N:
            continue
        d = d.copy()
        d["score"] = d[comps + CONTROLS].values @ lam[comps + CONTROLS].values
        dec = pd.qcut(d["score"].rank(method="first"), 10, labels=False)
        r = np.expm1(d[ret]); w = d["mktcap"]
        hi = np.average(r[dec == 9], weights=w[dec == 9]); lo = np.average(r[dec == 0], weights=w[dec == 0])
        rows.append({"qend": q, "ls": hi - lo})
    T = pd.DataFrame(rows).set_index("qend")
    m, s = T["ls"].mean(), T["ls"].std()
    return {"quarters": len(T), "mean_q": m, "sharpe_ann": (m / s) * 2.0 if s > 0 else np.nan, "t": nw_t(T["ls"].values, 1),
            "start": T.index.min().date()}, T


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decomp-file", default="decomp_monthly.parquet", help="decomposition parquet in data/parquet")
    ap.add_argument("--label", default="", help="output suffix; non-empty marks a post-hoc specification")
    ap.add_argument("--panel-only", action="store_true",
                    help="build the firm-quarter panel that later stages need and stop, without re-running "
                         "the prediction tests. The verdict is a finished finding, not something the monthly "
                         "update should recompute; the test reports stay on disk as the record.")
    a = ap.parse_args()
    sfx = f"_{a.label}" if a.label else ""
    log = open(C.LOGS / f"validate{sfx}.log", "w", encoding="utf-8")

    def say(msg):
        print(msg, flush=True); log.write(msg + "\n"); log.flush()

    t0 = time.time()
    P = build_panel(say, a.decomp_file)
    P.to_parquet(C.PQ / f"validation_panel{sfx}.parquet", index=False)
    if a.panel_only:
        say(f"panel only: {len(P):,} firm-quarters -> validation_panel{sfx}.parquet  {(time.time()-t0)/60:.1f} min")
        log.close()
        return 0
    has_M = "delta_M_A" in P.columns
    p("# 단계 3-2 — 수익률 예측 검정 " + ("(사후 스펙, 탐색적)" if a.label else "(사전 등록)") + "\n")
    p(f"> 생성 {pd.Timestamp.today().date()}. 수익률: 13F 내재가격 분기 수익률(상장폐지 포함) + 배당 근사, 분기별 1%/99% 절단. "
      f"표본 {P.qend.min().date()} ~ {P.qend.max().date()}, 기업-분기 {len(P):,}. 통제변수: 로그 시총, 모멘텀(직전 1~4분기). "
      f"분해 파일: `{a.decomp_file}`.")
    if a.label:
        p("\n> **사후 스펙.** 사전 등록 검정이 실패한 뒤 분해에 " + ("모멘텀 성분 δ_M을 추가한 " if has_M else "") +
          "사양이다. 아래 판정은 사전 등록 기준을 같은 방식으로 적용한 것이지만 사후 수정이므로 탐색적이며, "
          "이후 분기의 표본외 확인 없이는 채택 근거가 되지 않는다.")

    # ---- 1. FM regressions
    p("\n## 1. Fama-MacBeth 예측 회귀 (분기별 횡단면, Newey-West t)\n")
    comp_A = ["delta_S_A", "delta_T_A", "delta_F_A"] + (["delta_M_A"] if has_M else []) + ["eps_A"]
    comp_B = ["delta_S_B", "delta_T_B", "delta_F_B"] + (["delta_M_B"] if has_M else []) + ["eps_B"]
    for h in HORIZONS:
        ret = f"r{h}"
        p(f"\n### 지평 {h}분기 (종속변수 {ret}, 로그 수익률 합)\n")
        res_y, _ = fm(P, ret, ["y"] + CONTROLS, h)
        res_A, _ = fm(P, ret, comp_A + CONTROLS, h)
        PB = P[P["qend"] >= "2020-06-30"]
        res_B, _ = fm(PB, ret, comp_B + CONTROLS, h)
        tab = pd.concat([res_y.add_prefix("raw_"), res_A.add_prefix("A_"), res_B.add_prefix("B_")], axis=1)
        p(tab.round(3).to_markdown())
        say(f"h={h}\n" + tab.round(3).to_string())
    # ---- 2. benchmarks
    p("\n## 2. 벤치마크 (같은 통제변수, 지평 1분기와 4분기)\n")
    rows = {}
    for name, col in (("log(P/V_F) 원값", "y"), ("잔차 ε (스펙 A)", "eps_A"), ("log(B_adj/M)", "bm_adj"), ("E/P (조정이익)", "ep"),
                      ("Bartram-Grinblatt log(V_BG/P)", "bg_mis"), ("ICC", "icc")):
        for h in (1, 4):
            res, _ = fm(P, f"r{h}", [col] + CONTROLS, h)
            rows[f"{name} | h={h}"] = {"coef": res.loc[col, "coef"], "t": res.loc[col, "t"], "quarters": int(res.loc[col, "quarters"])}
    p(pd.DataFrame(rows).T.round(3).to_markdown())
    p("\n부호 기대: log(P/V_F), ε, ICC 이외의 고평가 척도는 음(−), log(B/M)·E/P·BG는 양(+).")

    # ---- 3. portfolios
    p("\n## 3. 10분위 포트폴리오, 시총가중, 분기 리밸런싱 (롱 = 저평가 분위, 숏 = 고평가 분위)\n")
    prow = {}
    for name, col, lo_long in (("ε (스펙 A)", "eps_A", True), ("log(P/V_F)", "y", True), ("−log(B_adj/M)", "log_pb_adj", True),
                                ("−E/P", "ep", False), ("−BG", "bg_mis", False)):
        st, _ = decile_ls(P, col, "r1", True, lo_long)
        prow[name] = st
    p(pd.DataFrame(prow).T.round(3).to_markdown())
    p("\n주: sharpe_ann = 분기 평균/표준편차 × 2. 거래비용 미반영. long_mean/short_mean은 분기 단순수익률.")
    # supplementary: equal-weighted deciles for the residual and the raw ratio, 1 and 4 quarter horizons
    p("\n동일가중 보충 (사전 등록 검정은 시총가중; 동일가중은 중소형주에서의 작동 여부를 보기 위한 보조 표):\n")
    erow = {}
    for name, col in (("ε (스펙 A)", "eps_A"), ("log(P/V_F)", "y")):
        for ret, rl in (("r1", "1분기"), ("r4", "4분기")):
            for vw, wl in ((True, "시총가중"), (False, "동일가중")):
                st, _ = decile_ls(P, col, ret, vw, True, lags=4 if ret == "r4" else 1)
                erow[f"{name} | {rl} | {wl}"] = st
    p(pd.DataFrame(erow).T.round(3).to_markdown())
    p("\n주: 4분기 지평은 분기마다 겹치는 윈도우라 mean_q·sharpe_ann은 4분기 누적 기준이며 t는 NW 4시차.")
    # composite OOS
    p("\n## 4. 표본외 복합점수 (확장 윈도우 λ, 최소 12분기 이력)\n")
    st_c, Tc = composite_oos(P, comp_A, "r1", say)
    st_bg, Tb = decile_ls(P[P["qend"] >= pd.Timestamp(st_c["start"])], "bg_mis", "r1", True, False)
    st_bm, _ = decile_ls(P[P["qend"] >= pd.Timestamp(st_c["start"])], "log_pb_adj", "r1", True, True)
    tab = pd.DataFrame({"복합점수 OOS (스펙 A)": st_c, "Bartram-Grinblatt (같은 기간)": st_bg, "log(B_adj/M) (같은 기간)": st_bm}).T
    p(tab.round(3).to_markdown())
    say(tab.round(3).to_string())
    # ---- delisting sensitivity
    p("\n## 5. 상장폐지 민감도 (퇴출 종목의 다음 분기 수익률 −30% 부과)\n")
    rows = {}
    for h in (1, 4):
        res, _ = fm(P, f"r{h}_dl", comp_A + CONTROLS, h)
        for c in comp_A:
            rows[f"{c} | h={h}"] = {"coef": res.loc[c, "coef"], "t": res.loc[c, "t"]}
    p(pd.DataFrame(rows).T.round(3).to_markdown())

    # ---- verdict
    resA1, _ = fm(P, "r1", comp_A + CONTROLS, 1)
    resA4, _ = fm(P, "r4", comp_A + CONTROLS, 4)
    t_eps = resA1.loc["eps_A", "t"]; t_T = resA1.loc["delta_T_A", "t"]; t_S = resA1.loc["delta_S_A", "t"]
    beats = (st_c["sharpe_ann"] > st_bg["sharpe_ann"]) if np.isfinite(st_c["sharpe_ann"]) and np.isfinite(st_bg["sharpe_ann"]) else False
    PB = P[P["qend"] >= "2020-06-30"]
    resB1, _ = fm(PB, "r1", comp_B + CONTROLS, 1)
    resB4, _ = fm(PB, "r4", comp_B + CONTROLS, 4)
    p("\n## 6. " + ("판정 (사후 스펙 — 사전 등록 기준을 같은 방식으로 적용, 탐색적)" if a.label else "사전 등록 판정") + "\n")
    p("스펙 A (13F, 2014-06~, 46분기) — 판정의 기준 스펙:\n")
    p(f"- λ_ε (h=1): {resA1.loc['eps_A','coef']:.3f}, t = {t_eps:.2f}; (h=4): {resA4.loc['eps_A','coef']:.3f}, t = {resA4.loc['eps_A','t']:.2f}  (기준: 음수, |t| > 2.5)")
    p(f"- λ_T (h=1): {resA1.loc['delta_T_A','coef']:.3f}, t = {t_T:.2f}; (h=4): {resA4.loc['delta_T_A','coef']:.3f}, t = {resA4.loc['delta_T_A','t']:.2f}  (기준: 음수, |t| > 2.5)")
    p(f"- λ_S (h=1): {resA1.loc['delta_S_A','coef']:.3f}, t = {t_S:.2f}  (기대: 음수, 작음)")
    if has_M:
        p(f"- λ_M (h=1): {resA1.loc['delta_M_A','coef']:.3f}, t = {resA1.loc['delta_M_A','t']:.2f}; (h=4): {resA4.loc['delta_M_A','coef']:.3f}, t = {resA4.loc['delta_M_A','t']:.2f}  (모멘텀 성분, 참고)")
    p(f"- 복합점수 OOS 샤프 {st_c['sharpe_ann']:.2f} vs Bartram-Grinblatt {st_bg['sharpe_ann']:.2f} (같은 기간, {st_c['start']}~)")
    p("\n스펙 B (13F + N-PORT, 2020-06~, 20~23분기) — 탐색적, 표본이 짧아 판정에 쓰지 않음:\n")
    for c, nm in (("eps_B", "λ_ε"), ("delta_T_B", "λ_T"), ("delta_S_B", "λ_S"), ("delta_F_B", "λ_F")):
        p(f"- {nm} (h=1): {resB1.loc[c,'coef']:.3f}, t = {resB1.loc[c,'t']:.2f}; (h=4): {resB4.loc[c,'coef']:.3f}, t = {resB4.loc[c,'t']:.2f}")
    eps_ok = (resA1.loc["eps_A", "coef"] < 0) and (t_eps < -2.5)
    T_ok = (resA1.loc["delta_T_A", "coef"] < 0) and (t_T < -2.5)
    if eps_ok and T_ok and beats:
        verdict = "채택"
    elif eps_ok or T_ok:
        verdict = "부분 채택 (유의한 성분만 도구화; 구조 성분은 설명 변수로만 유지)"
    else:
        verdict = "폐기 기준 충족 — 벤치마크 대비 우위 없음. 결과를 그대로 기록한다."
    if a.label:
        verdict += " (사후 스펙 — 탐색적, 사전 등록 판정을 대체하지 않음)"
    p(f"\n**판정: {verdict}**")
    say(f"verdict: {verdict}")
    (C.REPORTS / f"phase3_validation{sfx}.md").write_text("\n".join(L), encoding="utf-8")
    say(f"done  {(time.time()-t0)/60:.1f} min -> reports/phase3_validation{sfx}.md")
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
