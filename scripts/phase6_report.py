# -*- coding: utf-8 -*-
"""Phase 6 report: V_F with backlog versus without.

Compares rim_monthly.parquet (base) and rim_monthly_bl.parquet (backlog-
augmented expectations) from 2020-06 on:
  1. aggregate premium and its dispersion among RPO firms
  2. return prediction of log(P/V) base vs backlog, RPO subsample (FM, 13F returns)
  3. examples: backlog-rich firms
Writes reports/phase6_backlog_value.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sfv import config as C  # noqa: E402
from sfv.validate import fm, decile_ls, CONTROLS  # noqa: E402

pd.set_option("display.width", 220)
L = []


def p(t=""):
    L.append(t)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--alt-file", default="rim_monthly_bl.parquet", help="alternative valuation file to compare with the base")
    ap.add_argument("--alt-cols", default="v_f,log_pv,omega,a,icc", help="columns in the alt file mapped to v_f_bl, log_pv_bl, ...")
    ap.add_argument("--label", default="bl")
    ap.add_argument("--title", default="백로그를 반영한 V_F 와 기존 V_F 의 비교")
    ap.add_argument("--start", default="2020-06-30")
    a_ = ap.parse_args()
    cols = ["cik", "month", "ticker", "name", "ff12", "mktcap", "b_adj", "v_f", "log_pv", "omega", "a", "roe_adj", "roe_star_ind", "icc"]
    R0 = pd.read_parquet(C.PQ / "rim_monthly.parquet", columns=cols)
    alt_cols = a_.alt_cols.split(",")
    R1 = pd.read_parquet(C.PQ / a_.alt_file, columns=["cik", "month"] + alt_cols)
    R1 = R1.rename(columns={c: f"{n}_bl" for c, n in zip(alt_cols, ["v_f", "log_pv", "omega", "a", "icc"])})
    for n in ["omega", "a", "icc"]:
        if f"{n}_bl" not in R1:
            R1[f"{n}_bl"] = np.nan
    X = R0.merge(R1, on=["cik", "month"])
    X = X[X["month"] >= a_.start]
    B = pd.read_parquet(C.PQ / "backlog_quarterly.parquet").dropna(subset=["avail"]).sort_values("avail")
    X = pd.merge_asof(X.sort_values("month"), B[["cik", "end", "avail", "rpo", "rpo_to_rev", "rpo_g"]].rename(columns={"end": "rpo_end"}),
                      left_on="month", right_on="avail", by="cik", direction="backward")
    X["has_rpo"] = X["rpo"].notna() & ((X["month"] - X["rpo_end"]).dt.days <= 200)
    X.loc[~X["has_rpo"], ["rpo", "rpo_to_rev", "rpo_g"]] = np.nan
    X["dlog"] = X["log_pv"] - X["log_pv_bl"]          # positive = backlog raises V_F

    p(f"# 단계 6 — {a_.title}\n")
    p(f"> 생성 {pd.Timestamp.today().date()}. 표본 {X.month.min().date()} ~ {X.month.max().date()}, 기업-월 {len(X):,}, RPO 보유 {int(X.has_rpo.sum()):,}.")
    p("\n## 1. 집계와 분산\n")
    yr = X.assign(y=X["month"].dt.year).groupby("y").apply(lambda d: pd.Series({
        "log(ΣP/ΣV) 기존": np.log(d.mktcap.sum() / d.v_f.clip(lower=0).sum()),
        "log(ΣP/ΣV) 백로그": np.log(d.mktcap.sum() / d.v_f_bl.clip(lower=0).sum()),
        "RPO기업 log(P/V) 중앙값 기존": d.loc[d.has_rpo, "log_pv"].median(),
        "RPO기업 log(P/V) 중앙값 백로그": d.loc[d.has_rpo, "log_pv_bl"].median(),
        "RPO기업 log(P/V) 표준편차 기존": d.loc[d.has_rpo, "log_pv"].std(),
        "RPO기업 log(P/V) 표준편차 백로그": d.loc[d.has_rpo, "log_pv_bl"].std(),
        "RPO기업 V 변화율 중앙값": np.expm1(d.loc[d.has_rpo, "dlog"]).median()}), include_groups=False)
    p(yr.round(3).to_markdown())
    p("\n'V 변화율'은 백로그 반영으로 V_F가 몇 % 변했는지의 중앙값. 표준편차가 줄면 백로그가 가격 대비 가치의 흩어짐을 설명한 것이다.")

    # by backlog thickness
    d = X[X["has_rpo"]].copy()
    d["cov_q"] = pd.qcut(d["rpo_to_rev"], 5, labels=["얇음 1", "2", "3", "4", "두꺼움 5"])
    g = d.groupby("cov_q", observed=True).agg(n=("cik", "size"), rpo_to_rev=("rpo_to_rev", "median"),
                                              log_pv_base=("log_pv", "median"), log_pv_bl=("log_pv_bl", "median"), v_change=("dlog", lambda s: np.expm1(s).median()))
    p("\n백로그 두께(RPO/매출) 5분위별 중앙값:\n")
    p(g.round(3).to_markdown())

    # return prediction, RPO subsample, quarterly 13F returns
    p("\n## 2. 수익률 예측: log(P/V) 기존 vs 백로그 (RPO 기업, 분기별 FM, NW t)\n")
    V = pd.read_parquet(C.PQ / "validation_panel.parquet", columns=["cik", "qend", "r1", "r4", "mom", "log_size", "bg_mis", "log_pb_adj"])
    Q = X[X["month"].dt.month.isin([3, 6, 9, 12])].rename(columns={"month": "qend"})
    Q = Q.merge(V, on=["cik", "qend"], how="inner")
    QR = Q[Q["has_rpo"]]
    rows = {}
    for h in (1, 4):
        for nm, col in (("log(P/V) 기존", "log_pv"), ("log(P/V) 백로그", "log_pv_bl"), ("log(P/B_adj)", "log_pb_adj"), ("Bartram-Grinblatt", "bg_mis")):
            res, _ = fm(QR, f"r{h}", [col] + CONTROLS, h, min_n=150)
            rows[f"{nm} | h={h}"] = {"coef": res.loc[col, "coef"], "t": res.loc[col, "t"], "quarters": int(res.loc[col, "quarters"])}
    p(pd.DataFrame(rows).T.round(3).to_markdown())
    prow = {}
    for nm, col, lo_long in (("log(P/V) 기존 동일가중", "log_pv", True), ("log(P/V) 백로그 동일가중", "log_pv_bl", True)):
        st, _ = decile_ls(QR, col, "r1", False, lo_long)
        prow[nm] = st
    p("\n10분위 롱숏 (동일가중, 1분기):\n")
    p(pd.DataFrame(prow).T[["quarters", "mean_q", "sharpe_ann", "t"]].round(3).to_markdown())

    p("\n## 3. 대표 기업 (최근 월)\n")
    last = X[X["month"] == X["month"].max()]
    ex = last[last["ticker"].isin(["MSFT", "BA", "LMT", "CRM", "NVDA", "NOW", "ORCL", "AAPL", "GE", "RTX", "ADBE", "PLTR"])].copy()
    ex = ex.sort_values("mktcap", ascending=False)[["ticker", "mktcap", "rpo", "rpo_to_rev", "rpo_g", "v_f", "v_f_bl", "log_pv", "log_pv_bl", "omega", "omega_bl", "a", "a_bl"]]
    for c in ("mktcap", "rpo", "v_f", "v_f_bl"):
        ex[c] = ex[c] / 1e9
    p(ex.round(3).to_markdown(index=False))
    p("\n단위: 10억$. rpo_to_rev = RPO/연매출(년). omega/a는 기존, _bl은 백로그 반영 기대 모형의 값.")
    p("\n## 4. 읽기\n")
    p("- 백로그를 기대 경로에 넣으면 백로그가 두꺼운 기업의 V_F가 오르고 log(P/V)가 내려간다. 그것이 '항상 고평가' 판정을 얼마나 줄이는지가 1절의 분위표다.")
    p("- 2절이 실용적 판정 기준이다. 백로그 반영 log(P/V)가 기존보다 수익률을 더 잘(더 음의 계수·큰 |t|) 예측하면 가치 척도로서 개선된 것이다.")
    (C.REPORTS / f"phase6_value_{a_.label}.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
