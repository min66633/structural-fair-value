# -*- coding: utf-8 -*-
"""Module S, part 1: decompose log(P/V_F) cross-sectionally, month by month.

    log(P/V_F)_it = a_t + b_S' X_S + b_T' X_T + b_F' X_F [+ b_M' X_M] + industry FE + eps_it

    X_S  structural demand    inst_share, big3_share, nsi_12m            (2014-06-)
                              + index_share, sp500_proxy                 (2020-03-, spec B)
    X_T  transient pressure   d_inst_q (quarterly change in inst_share)  (2014-06-)
                              + fit_q                                    (2020-03-, spec B)
    X_F  model-error controls roe_adj, sales growth, R&D intensity, log size,
                              loss dummy, omega  -  characteristics that V_F may
                              mis-price; keeping them separate lets eps be a
                              residual net of known model weaknesses
    X_M  price momentum       12-1 month return and 1-month return, only with
                              --momentum. Post-hoc addition (2026-09-06) after
                              the pre-registered test: log(P/V_F) and eps were
                              found to carry momentum (rank corr 0.2-0.4 with
                              the 12-1 return), so a raw verdict shorts
                              momentum. Results go to separate files (label
                              "mom") so the pre-registered outputs stay intact.
    X_E  fundamentals-based   log(V_gr / V_F): the premium the model's own
         growth expectation   growth path justifies (growth_rim --backlog:
                              expanding-window revenue growth forecasts from
                              past growth, size, industry, backlog), only with
                              --expectation; 0 with a dummy where the growth
                              stage does not apply (no forecast or E <= 0), so
                              the sample is unchanged. The raw forecasts are
                              not used as regressors: they are linear in
                              characteristics already in X_F and the industry
                              dummies, which made their coefficients explode.
                              Diagnostic component: what part of a premium is
                              explained by observable expectations.
    Components are formed from cross-sectionally demeaned regressors, so each
    has zero mean every month; the intercept and industry effects carry the
    level. Everything at month t uses only information known at t, so the
    components are point in time by construction.

Momentum inputs (point in time):
    primary   yfinance adjusted close (splits and dividends) of the primary
              ticker: mom_12_1 = log(P_{m-1}/P_{m-12}), rev_1 = log(P_m/P_{m-1})
    fallback  names without a price feed (delisted, FTD-priced): the 13F
              implied quarter-end price (sfv.returns, split adjusted) at the
              latest quarter end on or before the month, window q-4 -> q-1
              (12-3 months, up to two months stale); the 1-month return falls
              back to the FTD last price, unadjusted for splits (rare there).
    Rows still lacking momentum (< 12 months of history, ~0.6%) drop out.

Also reported: variance shares var(component)/var(y) averaged over months,
the time series of the intercept (the unexplained aggregate level) and of
the value-weighted structural component.

Outputs data/parquet/decomp_monthly[_label].parquet, decomp_coefs[_label].parquet,
        decomp_aggregate[_label].parquet, reports/phase3_decomposition[_label].md
Run: python -m sfv.decompose [--momentum] [--expectation] [--label mom|exp|momexp]
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pandas as pd

from . import config as C

XS_A = ["inst_share", "big3_share", "nsi_12m"]
XS_B = XS_A + ["index_share", "sp500_proxy"]
XT_A = ["d_inst_q"]
XT_B = XT_A + ["fit_q"]
XF = ["roe_adj_w", "sales_g", "rnd_int", "log_size", "loss", "omega"]
XM = ["mom_12_1", "rev_1"]
XE = ["log_vgr_vf", "g_na"]
MOM_CLIP = 2.0      # log units: 12-1 month return bounded to [-2, 2]
REV_CLIP = 1.0      # 1-month return bounded to [-1, 1]
VGR_CLIP = (-1.0, 1.5)   # log(V_gr / V_F) bounded
MIN_N = 300
L = []


def p(t=""):
    L.append(t)


def nw_t(x: np.ndarray, lags: int = 6) -> float:
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 8:
        return np.nan
    e = x - x.mean()
    v = e @ e / n
    for lag in range(1, lags + 1):
        v += 2 * (1 - lag / (lags + 1)) * (e[lag:] @ e[:-lag]) / n
    return x.mean() / np.sqrt(max(v, 1e-18) / n)


def load() -> pd.DataFrame:
    R = pd.read_parquet(C.PQ / "rim_monthly.parquet",
                        columns=["cik", "month", "ticker", "ff12", "mktcap", "f_end", "log_pv", "log_pv_notv", "log_pb_adj",
                                 "roe_adj", "ttm_e_adj", "b_adj", "k_int", "omega", "icc", "v_f"])
    D = pd.read_parquet(C.PQ / "demand_monthly.parquet",
                        columns=["cik", "month", "inst_share", "big3_share", "ext_share", "index_share", "fund_share",
                                 "sp500_proxy", "nsi_12m", "fit_q", "fit_2q"])
    X = R.merge(D, on=["cik", "month"], how="left").sort_values(["cik", "month"])
    g = X.groupby("cik")
    prev_inst = g["inst_share"].shift(3)
    prev_m = g["month"].shift(3)
    X["d_inst_q"] = (X["inst_share"] - prev_inst).where((X["month"] - prev_m).dt.days.between(80, 100))
    # sales growth: trailing revenue vs four quarters earlier, from the latest available quarter
    I = pd.read_parquet(C.PQ / "intangibles_quarterly.parquet", columns=["cik", "end", "avail_date", "ttm_revenue", "k_rnd"])
    I = I.dropna(subset=["avail_date"]).sort_values(["cik", "end"])
    I["ttm_rev_lag4"] = I.groupby("cik")["ttm_revenue"].shift(4)
    lag_ok = (I["end"] - I.groupby("cik")["end"].shift(4)).dt.days.between(340, 390)
    I["sales_g"] = np.log(I["ttm_revenue"] / I["ttm_rev_lag4"]).where(lag_ok & (I["ttm_revenue"] > 0) & (I["ttm_rev_lag4"] > 0))
    I = I.sort_values("avail_date")
    X = pd.merge_asof(X.sort_values("month"), I[["cik", "avail_date", "sales_g", "k_rnd"]], left_on="month", right_on="avail_date",
                      by="cik", direction="backward")
    X["rnd_int"] = (X["k_rnd"] / X["b_adj"]).where(X["b_adj"] > 0).clip(0, 3)
    X["log_size"] = np.log(X["mktcap"])
    X["loss"] = (X["ttm_e_adj"] <= 0).astype(float)
    X["roe_adj_w"] = X["roe_adj"].clip(-1, 1)
    X["sales_g"] = X["sales_g"].clip(-1, 1)
    return X.sort_values(["month", "cik"]).reset_index(drop=True)


def momentum_features(say) -> pd.DataFrame:
    """(cik, month) -> mom_12_1, rev_1, mom_src. See the module docstring for sources."""
    U = pd.read_parquet(C.PQ / "universe_monthly.parquet", columns=["cik", "month", "adj_close", "price"])
    U["month"] = U["month"] + pd.offsets.MonthEnd(0)
    U = U.sort_values(["cik", "month"]).reset_index(drop=True)
    g = U.groupby("cik")

    def lag(col: str, k: int, lo: int, hi: int) -> pd.Series:
        # k rows back, accepted only when the calendar gap is k months (the grid has holes)
        ok = (U["month"] - g["month"].shift(k)).dt.days.between(lo, hi)
        return g[col].shift(k).where(ok)

    ac1, ac12 = lag("adj_close", 1, 25, 35), lag("adj_close", 12, 355, 375)
    p1 = lag("price", 1, 25, 35)
    with np.errstate(divide="ignore", invalid="ignore"):
        U["mom_yf"] = np.log(ac1 / ac12).where((ac1 > 0) & (ac12 > 0))
        U["rev_yf"] = np.log(U["adj_close"] / ac1).where((U["adj_close"] > 0) & (ac1 > 0))
        U["rev_ftd"] = np.log(U["price"] / p1).where((U["price"] > 0) & (p1 > 0))
    # 13F implied prices, quarterly: log(px_{q-1} / px_{q-4}) plus the splits inside the window.
    # log_split at row q is the split between q and q+1, so the window q-4 -> q-1 spans rows q-4, q-3, q-2.
    R = pd.read_parquet(C.PQ / "returns_quarterly.parquet", columns=["cik", "qend", "px_med", "log_split"])
    R = R.sort_values(["cik", "qend"]).drop_duplicates(["cik", "qend"]).reset_index(drop=True)
    gr = R.groupby("cik")
    q_ok = (R["qend"] - gr["qend"].shift(4)).dt.days.between(340, 390)
    spl = sum(gr["log_split"].shift(k).fillna(0.0) for k in (2, 3, 4))
    px1, px4 = gr["px_med"].shift(1), gr["px_med"].shift(4)
    with np.errstate(divide="ignore", invalid="ignore"):
        R["mom_13f"] = (np.log(px1 / px4) + spl).where(q_ok & (px1 > 0) & (px4 > 0))
    # latest quarter end on or before the month
    U["qend_last"] = U["month"].where(U["month"].dt.month.isin([3, 6, 9, 12]), U["month"] - pd.offsets.QuarterEnd(1))
    U = U.merge(R[["cik", "qend", "mom_13f"]].rename(columns={"qend": "qend_last"}), on=["cik", "qend_last"], how="left")
    U["mom_12_1"] = U["mom_yf"].where(U["mom_yf"].notna(), U["mom_13f"]).clip(-MOM_CLIP, MOM_CLIP)
    U["rev_1"] = U["rev_yf"].where(U["rev_yf"].notna(), U["rev_ftd"]).clip(-REV_CLIP, REV_CLIP)
    U["mom_src"] = np.select([U["mom_yf"].notna(), U["mom_13f"].notna()], ["yf", "13f"], default="none")
    say(f"  momentum: {len(U):,} cik-months; 12-1 source yf {(U.mom_src == 'yf').mean()*100:.1f}%  13F fallback "
        f"{(U.mom_src == '13f').mean()*100:.1f}%  none {(U.mom_src == 'none').mean()*100:.1f}%; "
        f"rev_1 yf {U.rev_yf.notna().mean()*100:.1f}%  ftd fallback {(U.rev_yf.isna() & U.rev_ftd.notna()).mean()*100:.1f}%")
    return U[["cik", "month", "mom_12_1", "rev_1", "mom_src"]]


def expectation_features(X: pd.DataFrame, say) -> pd.DataFrame:
    """Attach log(V_gr / V_F) from the growth-stage model with backlog (growth_rim --backlog), point in time."""
    G = pd.read_parquet(C.PQ / "rim_monthly_grbl.parquet", columns=["cik", "month", "v_f_gr", "gr_applied"])
    G = G.drop_duplicates(["cik", "month"])
    X = X.merge(G, on=["cik", "month"], how="left")
    ok = X["gr_applied"].fillna(False) & (X["v_f_gr"] > 0) & (X["v_f"] > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        X["log_vgr_vf"] = np.log(X["v_f_gr"] / X["v_f"]).where(ok).fillna(0.0).clip(*VGR_CLIP)
    X["g_na"] = (~ok).astype(float)
    say(f"  expectation: growth stage applies to {ok.mean()*100:.1f}% of valuation rows (else 0 + dummy); "
        f"log(V_gr/V_F) median {X.loc[ok, 'log_vgr_vf'].median():.3f}, p10 {X.loc[ok, 'log_vgr_vf'].quantile(.1):.3f}, p90 {X.loc[ok, 'log_vgr_vf'].quantile(.9):.3f}")
    return X.drop(columns=["v_f_gr", "gr_applied"])


def decompose_month(d: pd.DataFrame, groups: dict[str, list[str]]) -> tuple[pd.DataFrame, pd.Series, dict]:
    cols = [c for cs in groups.values() for c in cs]
    d = d.dropna(subset=["y"] + cols)
    if len(d) < MIN_N:
        return None, None, None
    Z = d[cols].astype(float)
    Zc = Z - Z.mean()
    ind = pd.get_dummies(d["ff12"], prefix="ind", drop_first=True).astype(float)
    Xm = np.column_stack([np.ones(len(d)), Zc.values, ind.values])
    y = d["y"].values
    b, *_ = np.linalg.lstsq(Xm, y, rcond=None)
    k = len(cols)
    comp = pd.DataFrame(index=d.index)
    pos = 1
    for name, gcols in groups.items():
        comp[f"delta_{name}"] = Zc[gcols].values @ b[pos:pos + len(gcols)]
        pos += len(gcols)
    comp["eps"] = y - Xm @ b
    comp["eps_noF"] = comp["eps"] + comp["delta_F"]          # residual if model-error controls were not used
    if "M" in groups:
        comp["eps_noM"] = comp["eps"] + comp["delta_M"]      # residual if momentum were not controlled
    comp["ind_fe"] = ind.values @ b[1 + k:]
    coefs = pd.Series(b[:1 + k], index=["const"] + cols)
    vy = np.var(y)
    shares = {"n": len(d), "R2": 1 - np.var(comp["eps"]) / vy, "share_ind": np.var(comp["ind_fe"]) / vy,
              "share_eps": np.var(comp["eps"]) / vy}
    for name in groups:
        shares[f"share_{name}"] = np.var(comp[f"delta_{name}"]) / vy
    return comp, coefs, shares


def run_spec(X: pd.DataFrame, groups: dict[str, list[str]], label: str, say) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    comps, coefs, shares = [], [], []
    for m, d in X.groupby("month"):
        comp, b, sh = decompose_month(d, groups)
        if comp is None:
            continue
        comp["month"] = m
        comps.append(comp)
        coefs.append(b.rename(m))
        shares.append(pd.Series(sh, name=m))
    Cm = pd.concat(comps)
    B = pd.DataFrame(coefs)
    S = pd.DataFrame(shares)
    say(f"  spec {label}: {len(B)} months, mean n {S.n.mean():.0f}, mean R2 {S.R2.mean():.3f}")
    return Cm, B, S


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--momentum", action="store_true", help="add the momentum group X_M (post-hoc spec)")
    ap.add_argument("--expectation", action="store_true", help="add the growth-expectation group X_E (diagnostic spec)")
    ap.add_argument("--label", default=None, help="output suffix; default 'mom' / 'exp' / 'momexp' from the flags, none otherwise")
    a = ap.parse_args()
    label = a.label if a.label is not None else ("mom" if a.momentum else "") + ("exp" if a.expectation else "")
    sfx = f"_{label}" if label else ""
    log = open(C.LOGS / f"decompose{sfx}.log", "w", encoding="utf-8")

    def say(msg):
        print(msg, flush=True); log.write(msg + "\n"); log.flush()

    t0 = time.time()
    X = load()
    # winsorise the dependent variable monthly at 1% / 99%
    lo = X.groupby("month")["log_pv"].transform(lambda s: s.quantile(.01))
    hi = X.groupby("month")["log_pv"].transform(lambda s: s.quantile(.99))
    X["y"] = X["log_pv"].clip(lo, hi)
    say(f"  rows {len(X):,}, months {X.month.nunique()}, y winsorised at monthly 1/99%")
    xm = XM if a.momentum else []
    xe = XE if a.expectation else []
    if a.momentum:
        X = X.merge(momentum_features(say), on=["cik", "month"], how="left")
        say(f"  momentum coverage on valuation rows: mom_12_1 {X.mom_12_1.notna().mean()*100:.1f}%  rev_1 {X.rev_1.notna().mean()*100:.1f}%")
    if a.expectation:
        X = expectation_features(X, say)

    groups_A = {"S": XS_A, "T": XT_A, "F": XF}
    groups_B = {"S": XS_B, "T": XT_B, "F": XF}
    if xm:
        groups_A["M"] = xm
        groups_B["M"] = xm
    if xe:
        groups_A["E"] = xe
        groups_B["E"] = xe
    CA, BA, SA = run_spec(X, groups_A, "A (13F, 2014-06-)", say)
    XB = X[X["month"] >= "2020-03-31"]
    CB, BB, SB = run_spec(XB, groups_B, "B (13F + N-PORT, 2020-03-)", say)

    base_cols = ["cik", "month", "ticker", "ff12", "mktcap", "log_pv", "y", "log_pv_notv", "log_pb_adj", "icc", "v_f"]
    out = X[base_cols + XS_B + XT_B + XF + xm + (["mom_src"] if xm else []) + xe].copy()
    comp_cols = ["delta_S", "delta_T", "delta_F", "eps", "eps_noF", "ind_fe"] + (["delta_M", "eps_noM"] if xm else []) + (["delta_E"] if xe else [])
    for lab, Cm in (("A", CA), ("B", CB)):
        out = out.join(Cm[comp_cols].add_suffix(f"_{lab}"), how="left")
    out.to_parquet(C.PQ / f"decomp_monthly{sfx}.parquet", index=False)
    BA.assign(spec="A").to_parquet(C.PQ / f"decomp_coefs{sfx}.parquet")

    title_add = " (모멘텀 성분 추가, 사후 스펙)" if xm and not xe else " (모멘텀·기대 성분 추가, 사후 스펙)" if xm and xe else " (기대 성분 추가, 사후 스펙)" if xe else ""
    p("# 단계 3-1 — log(P/V_F) 횡단면 분해" + title_add + "\n")
    p(f"> 생성 {pd.Timestamp.today().date()}. 종속변수 log(P/V_F), 월별 1%/99% 절단. 산업(FF12) 고정효과. "
      f"성분은 월별로 평균 0이 되도록 회귀변수를 횡단면 중심화해 계산. 스펙 A: 13F 변수, {BA.index.min().date()}~{BA.index.max().date()} ({len(BA)}개월). "
      f"스펙 B: 13F + N-PORT, {BB.index.min().date()}~{BB.index.max().date()} ({len(BB)}개월).")
    if xm:
        p("\n> **사후 스펙.** 사전 등록 검정(`phase3_validation.md`)이 실패한 뒤, log(P/V_F)와 잔차 ε이 직전 12개월 모멘텀과 순위상관 0.2~0.4로 "
          "같이 움직이는 것이 확인되어 모멘텀 성분 δ_M(12-1개월 수익률, 1개월 수익률)을 추가했다. 이 파일의 잔차 ε은 모멘텀을 걷어낸 값이며, "
          "`eps_noM` = ε + δ_M 은 모멘텀을 통제하지 않았을 때의 잔차다. 모멘텀 출처: yfinance 조정종가(상장 종목), 없으면 13F 내재가격(분기, 최대 2개월 지연), "
          f"12개월 이력이 없는 행은 제외. 결과는 탐색적이며 사전 등록 판정을 대체하지 않는다. 절단: 12-1 수익률 ±{MOM_CLIP:.0f}, 1개월 수익률 ±{REV_CLIP:.0f} (로그).")
    if xe:
        p("\n> **기대 성분 δ_E (진단용).** 단계 6의 성장 단계 모형(확장 윈도우 매출성장 전망, 과거 성장·규모·산업·백로그로 적합, 해당 분기 기준 point in time)이 정당화하는 "
          "프리미엄 log(V_gr/V_F)를 성분으로 넣었다. 성장 단계가 적용되지 않는 행(전망 없음 또는 조정이익 ≤ 0)은 0으로 두고 더미(g_na)로 표시해 표본을 바꾸지 않았다. "
          "전망치 자체를 회귀변수로 쓰면 이미 들어 있는 특성·산업 더미의 선형결합이라 계수가 폭발하므로 가치 채널로 넣었다. "
          "δ_E는 '관측 가능한 펀더멘털 전망으로 설명되는 프리미엄'이고, 그 뒤의 잔차 ε은 '펀더멘털·동료 기업으로 설명되지 않는 나머지'다. "
          "잔차를 '시장의 기대'라고 부르지 않는 이유는 그것이 누군가의 전망이라는 증거가 없기 때문이다. "
          "가격에서 읽은 전망치 자체를 넣은 것이 아니므로 순환 문제는 없지만, 예측력 향상을 노린 성분도 아니다.")
    for lab, B, S in (("A", BA, SA), ("B", BB, SB)):
        p(f"\n## 스펙 {lab} — 계수 (월별 횡단면 회귀의 시계열 평균, Newey-West t)\n")
        tab = pd.DataFrame({"mean": B.mean(), "nw_t": [nw_t(B[c].values) for c in B.columns], "months": len(B)})
        p(tab.round(3).to_markdown())
        line = (f"\n분산 몫 (월평균): R² {S.R2.mean():.3f} | 구조 δ_S {S.share_S.mean():.3f} | 일시 δ_T {S.share_T.mean():.3f} | "
                f"모형보정 δ_F {S.share_F.mean():.3f}")
        if xm:
            line += f" | 모멘텀 δ_M {S.share_M.mean():.3f}"
        if xe:
            line += f" | 기대 δ_E {S.share_E.mean():.3f}"
        line += f" | 산업 {S.share_ind.mean():.3f} | 잔차 ε {S.share_eps.mean():.3f}"
        p(line)
        say(tab.round(3).to_string())
        say(line.strip())
    if xm:
        # verification: rank correlation of the residual with momentum, before and after the control (monthly, averaged)
        Xa = out.dropna(subset=["eps_A", "mom_12_1"])
        rho = Xa.groupby("month").apply(lambda d: pd.Series({
            "y": d["y"].corr(d["mom_12_1"], method="spearman"),
            "eps_noM": d["eps_noM_A"].corr(d["mom_12_1"], method="spearman"),
            "eps": d["eps_A"].corr(d["mom_12_1"], method="spearman"),
            "delta_F": d["delta_F_A"].corr(d["mom_12_1"], method="spearman")}), include_groups=False)
        p("\n## 모멘텀 검증 (스펙 A) — 12-1개월 수익률과의 월별 순위상관, 연도 평균\n")
        p("ε은 모멘텀을 회귀변수로 넣었으므로 월별 피어슨 상관이 0이 되도록 만들어졌다. 순위상관이 0 근처면 비선형 잔존도 없다는 뜻이다.\n")
        p(rho.groupby(rho.index.year).mean().round(3).to_markdown())
        say("  rank corr with mom_12_1 (mean over months): " + rho.mean().round(3).to_dict().__repr__())
    if xe:
        Xa = out[out["eps_A"].notna() & (out["g_na"] == 0)]
        rho_e = Xa.groupby("month").apply(lambda d: pd.Series({
            "y": d["y"].corr(d["log_vgr_vf"], method="spearman"), "delta_E": d["delta_E_A"].corr(d["log_vgr_vf"], method="spearman"),
            "eps": d["eps_A"].corr(d["log_vgr_vf"], method="spearman")}) if len(d) > 30 else pd.Series({"y": np.nan, "delta_E": np.nan, "eps": np.nan}),
            include_groups=False)
        p("\n## 기대 성분 검증 (스펙 A) — 성장 단계 적용 기업에서 log(V_gr/V_F)와의 월별 순위상관, 연도 평균\n")
        p(rho_e.groupby(rho_e.index.year).mean().round(3).to_markdown())
        say("  rank corr with log_vgr_vf (mean over months): " + rho_e.mean().round(3).to_dict().__repr__())
    # aggregate: intercept (unexplained level) and value-weighted structural component
    Xa = out.dropna(subset=["delta_S_A"])
    agg_cols = {"vw_log_pv": "y", "vw_delta_S": "delta_S_A", "vw_delta_T": "delta_T_A", "vw_delta_F": "delta_F_A", "vw_eps": "eps_A"}
    if xm:
        agg_cols["vw_delta_M"] = "delta_M_A"
    if xe:
        agg_cols["vw_delta_E"] = "delta_E_A"
    agg = Xa.groupby("month").apply(lambda d: pd.Series(
        {"eq_log_pv": d.y.mean(), **{k: np.average(d[c], weights=d.mktcap) for k, c in agg_cols.items()}}), include_groups=False)
    agg.to_parquet(C.PQ / f"decomp_aggregate{sfx}.parquet")
    p("\n## 집계 환원 (스펙 A, 연말 기준)\n")
    p("시총가중 평균 log(P/V_F)와 그 성분. 성분은 횡단면 평균 0이므로 시총가중 값은 '대형주가 평균 기업 대비 얼마나 더 그 성분을 갖는가'를 뜻한다. "
      "시장 전체의 수준(절편)은 횡단면으로는 식별되지 않는다.\n")
    yr = agg[agg.index.month == 12]
    p(yr.round(3).to_markdown())
    p("\n해석 주의: 계수의 부호는 상관관계다. 어느 성분이 '프리미엄'이고 어느 성분이 '되돌림'인지는 3-2의 수익률 예측 검정이 결정한다.")
    (C.REPORTS / f"phase3_decomposition{sfx}.md").write_text("\n".join(L), encoding="utf-8")
    say(f"done  {(time.time()-t0)/60:.1f} min -> reports/phase3_decomposition{sfx}.md")
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
