# -*- coding: utf-8 -*-
"""Phase 5: does forward-looking backlog information sit inside the residual?

ASC 606 (2018-) makes firms disclose remaining performance obligations (RPO):
contracted revenue not yet recognised - a structured backlog. Three
questions, all cross-sectional, quarterly, point in time:

  1. Does RPO information explain the decomposition residual eps (and
     log(P/V_F)) today?            -> the residual contains expectations
  2. Does it predict fundamentals? next-4-quarter revenue growth on RPO growth
                                   -> the expectations have real content
  3. Does it predict returns?      the RPO-explained part of eps (eps_exp)
                                   versus the rest (eps_res). If prices already
                                   reflect real backlog information, eps_exp
                                   should not predict returns; if the premium is
                                   narrative, high eps_exp should underperform.

Variables (latest fiscal quarter available at t, first-reported values):
    rpo_to_rev   RPO / trailing revenue  (years of backlog coverage), log
    rpo_g        year-on-year log growth of RPO
    d_rpo_cov    change in rpo_to_rev over four quarters
    dfr_g        year-on-year growth of contract liabilities (deferred revenue)

Outputs data/parquet/backlog_quarterly.parquet, reports/phase5_backlog.md
Run: python -m sfv.backlog
"""
from __future__ import annotations

import glob
import sys
import time

import numpy as np
import pandas as pd

from . import config as C
from .validate import fm, nw_t, CONTROLS, MIN_N

L = []


def p(t=""):
    L.append(t)


def rpo_panel(say) -> pd.DataFrame:
    d = pd.concat([pd.read_parquet(f) for f in glob.glob(str(C.PQ / "xbrl_rpo" / "part-*.parquet"))])
    d = d[d["which"].isin(["first", "only"]) & (d["unit"] == "USD") & d["start"].isna()]
    d = d.sort_values(["cik", "tag", "end", "filed"]).drop_duplicates(["cik", "tag", "end"], keep="first")
    rpo = d[d["tag"] == "RevenueRemainingPerformanceObligation"][["cik", "end", "val", "filed"]].rename(columns={"val": "rpo", "filed": "rpo_filed"})
    # contract liabilities: total, else current + noncurrent
    cl = d[d["tag"].isin(["ContractWithCustomerLiability", "ContractWithCustomerLiabilityCurrent", "ContractWithCustomerLiabilityNoncurrent"])]
    cl = cl.pivot_table(index=["cik", "end"], columns="tag", values="val", aggfunc="first").reset_index()
    for c in ["ContractWithCustomerLiability", "ContractWithCustomerLiabilityCurrent", "ContractWithCustomerLiabilityNoncurrent"]:
        if c not in cl:
            cl[c] = np.nan
    cl["dfr"] = cl["ContractWithCustomerLiability"].where(cl["ContractWithCustomerLiability"].notna(),
                                                          cl["ContractWithCustomerLiabilityCurrent"].fillna(0) + cl["ContractWithCustomerLiabilityNoncurrent"].fillna(0))
    cl = cl[cl["dfr"] > 0][["cik", "end", "dfr"]]
    Q = pd.read_parquet(C.PQ / "fund_quarterly.parquet", columns=["cik", "end", "avail_date", "ttm_revenue", "ttm_ok"])
    P = Q.merge(rpo, on=["cik", "end"], how="left").merge(cl, on=["cik", "end"], how="left")
    P = P.sort_values(["cik", "end"])
    g = P.groupby("cik")
    lag_ok = (P["end"] - g["end"].shift(4)).dt.days.between(340, 390)
    P["rpo_to_rev"] = (P["rpo"] / P["ttm_revenue"]).where((P["rpo"] > 0) & (P["ttm_revenue"] > 0))
    P["rpo_g"] = np.log(P["rpo"] / g["rpo"].shift(4)).where(lag_ok & (P["rpo"] > 0) & (g["rpo"].shift(4) > 0)).clip(-2, 2)
    P["d_rpo_cov"] = (P["rpo_to_rev"] - g["rpo_to_rev"].shift(4)).where(lag_ok).clip(-5, 5)
    P["dfr_g"] = np.log(P["dfr"] / g["dfr"].shift(4)).where(lag_ok & (P["dfr"] > 0) & (g["dfr"].shift(4) > 0)).clip(-2, 2)
    # future fundamentals: revenue growth over the next four quarters (for test 2)
    P["rev_g_f4"] = np.log(g["ttm_revenue"].shift(-4) / P["ttm_revenue"]).where(
        (g["end"].shift(-4) - P["end"]).dt.days.between(340, 390) & (P["ttm_revenue"] > 0) & (g["ttm_revenue"].shift(-4) > 0)).clip(-2, 2)
    P["rev_g_past"] = np.log(P["ttm_revenue"] / g["ttm_revenue"].shift(4)).where(lag_ok & (P["ttm_revenue"] > 0) & (g["ttm_revenue"].shift(4) > 0)).clip(-2, 2)
    P["avail"] = P[["avail_date", "rpo_filed"]].max(axis=1)
    out = P[P["rpo"].notna() | P["dfr"].notna()][["cik", "end", "avail", "rpo", "rpo_to_rev", "rpo_g", "d_rpo_cov", "dfr", "dfr_g", "rev_g_f4", "rev_g_past"]]
    say(f"  RPO panel: {out.rpo.notna().sum():,} firm-quarters with RPO, {out[out.rpo.notna()].cik.nunique():,} firms; "
        f"deferred revenue {out.dfr.notna().sum():,} rows")
    return out


def main() -> int:
    log = open(C.LOGS / "backlog.log", "w", encoding="utf-8")

    def say(msg):
        print(msg, flush=True); log.write(msg + "\n"); log.flush()

    t0 = time.time()
    B = rpo_panel(say)
    B.to_parquet(C.PQ / "backlog_quarterly.parquet", index=False)
    V = pd.read_parquet(C.PQ / "validation_panel.parquet")
    V = V[V["qend"] >= "2018-03-31"]
    B = B.dropna(subset=["avail"]).sort_values("avail")
    X = pd.merge_asof(V.sort_values("qend"), B.rename(columns={"end": "rpo_end"}), left_on="qend", right_on="avail", by="cik", direction="backward")
    X = X[(X["qend"] - X["rpo_end"]).dt.days <= 200]
    X["log_rpo_cov"] = np.log(X["rpo_to_rev"]).where(X["rpo_to_rev"] > 0)
    has = X["rpo"].notna()
    say(f"  merged: {len(X):,} firm-quarters 2018-03+, with RPO {has.sum():,} ({X[has].cik.nunique():,} firms), "
        f"share of universe market cap covered: {X.loc[has, 'mktcap'].sum() / X['mktcap'].sum() * 100:.1f}%")
    cov = X.groupby("qend").apply(lambda d: pd.Series({"firms": d.cik.nunique(), "with_rpo": d.rpo.notna().sum(),
                                                       "mcap_share": d.loc[d.rpo.notna(), "mktcap"].sum() / d.mktcap.sum()}), include_groups=False)

    p("# 단계 5 — 백로그(잔여이행의무)로 잔차 설명하기\n")
    p(f"> 생성 {pd.Timestamp.today().date()}. ASC 606 이후 RPO 공시 기업. 표본 {X.qend.min().date()} ~ {X.qend.max().date()}, "
      f"기업-분기 {int(has.sum()):,}, 기업 {X[has].cik.nunique():,}. 모든 변수는 최초 보고치이며 제출일 이후에만 사용.")
    p("\n## 0. 커버리지\n")
    p(cov.iloc[::4].round(3).to_markdown())
    p(f"\n유니버스 시가총액 중 RPO 공시 기업 비중 {X.loc[has, 'mktcap'].sum() / X['mktcap'].sum() * 100:.1f}%. "
      "계약 기반 사업(소프트웨어·항공방산·산업재·통신)에 몰려 있다.")
    XR = X[has].copy()
    d = XR[["rpo_to_rev", "rpo_g", "d_rpo_cov", "dfr_g"]].describe(percentiles=[.1, .5, .9]).T
    p("\n변수 분포:\n")
    p(d[["count", "10%", "50%", "90%"]].round(3).to_markdown())

    p("\n## 1. 잔차 ε 과 log(P/V_F)를 백로그 정보로 설명 (분기별 FM, NW t)\n")
    xs = ["rpo_g", "log_rpo_cov", "d_rpo_cov", "dfr_g"]
    rows = {}
    for y, nm in (("eps_A", "잔차 ε (스펙 A)"), ("y", "log(P/V_F)")):
        res, _ = fm(XR, y, xs + CONTROLS, 1, min_n=150)
        for c in xs:
            rows[f"{nm} ← {c}"] = {"coef": res.loc[c, "coef"], "t": res.loc[c, "t"], "quarters": int(res.loc[c, "quarters"])}
    p(pd.DataFrame(rows).T.round(3).to_markdown())
    p("\n양(+)의 유의한 계수면 백로그가 늘거나 두꺼운 기업이 펀더멘털 대비 높게 거래된다는 뜻이다. 잔차가 미래 정보를 담고 있다는 증거.")

    p("\n## 2. 백로그가 실제 미래 매출을 예측하는가 (다음 4분기 매출성장, FM)\n")
    res, _ = fm(XR, "rev_g_f4", ["rpo_g", "log_rpo_cov", "d_rpo_cov", "rev_g_past"] + CONTROLS, 4, min_n=150)
    p(res.round(3).to_markdown())
    p("\n과거 매출성장을 통제하고도 rpo_g 계수가 양이면 백로그는 '기대'로서 실체가 있다.")

    p("\n## 3. 수익률 예측: ε 중 백로그로 설명되는 부분 vs 나머지\n")
    # split eps into the part fitted by RPO variables (quarter by quarter) and the rest
    parts = []
    for q, dq in XR.groupby("qend"):
        dq = dq.dropna(subset=["eps_A"] + xs)
        if len(dq) < 150:
            continue
        Xm = np.column_stack([np.ones(len(dq)), dq[xs].values.astype(float)])
        b = np.linalg.lstsq(Xm, dq["eps_A"].values, rcond=None)[0]
        fit = Xm @ b - b[0]
        parts.append(pd.DataFrame({"cik": dq["cik"], "qend": q, "eps_exp": fit, "eps_res": dq["eps_A"].values - fit}))
    E = pd.concat(parts, ignore_index=True)
    XR = XR.merge(E, on=["cik", "qend"], how="left")
    rows = {}
    for h in (1, 4):
        res, _ = fm(XR, f"r{h}", ["eps_exp", "eps_res", "delta_S_A", "delta_T_A", "delta_F_A"] + CONTROLS, h, min_n=150)
        for c in ["eps_exp", "eps_res"]:
            rows[f"{c} | h={h}"] = {"coef": res.loc[c, "coef"], "t": res.loc[c, "t"], "quarters": int(res.loc[c, "quarters"])}
        res2, _ = fm(XR, f"r{h}", xs + CONTROLS, h, min_n=150)
        for c in xs:
            rows[f"{c} 직접 | h={h}"] = {"coef": res2.loc[c, "coef"], "t": res2.loc[c, "t"], "quarters": int(res2.loc[c, "quarters"])}
    p(pd.DataFrame(rows).T.round(3).to_markdown())
    p("\n해석 기준: eps_exp가 수익률을 예측하지 않으면 백로그 프리미엄은 이미 정당하게 가격에 있다(기대). "
      "eps_exp 계수가 음이면 백로그에 붙은 프리미엄이 되돌려진다(내러티브 또는 과잉반응). 양이면 시장이 백로그에 과소반응한다.")
    p("\n## 4. 한계\n")
    p("- RPO 공시 기업은 계약형 사업에 치우쳐 있어 결과가 소비재·에너지 등에는 적용되지 않는다.")
    p("- 표본이 2018년 이후 약 33분기라 시계열 검정력이 약하다. 13F 내재가격 수익률 사용(상장폐지 포함).")
    p("- RPO 정의는 회사마다 다르고(기간 제한, 취소 가능 계약 포함 여부) 같은 태그라도 완전히 비교 가능하지 않다.")
    (C.REPORTS / "phase5_backlog.md").write_text("\n".join(L), encoding="utf-8")
    say("\n".join(L))
    say(f"done  {(time.time()-t0)/60:.1f} min")
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
