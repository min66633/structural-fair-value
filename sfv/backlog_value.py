# -*- coding: utf-8 -*-
"""Backlog-augmented expectations for the residual income model.

A DCF that ignores contracted backlog will call every backlog-rich firm
overvalued. Here the backlog enters the VALUE, not the residual: the
one-year-ahead excess ROE model of sfv.persistence gets the remaining
performance obligations (RPO) as regressors,

    x_{t+1} = a + (w0 + w'z + w_b'b) x_t + g'z + g_b'b + e
    b = (has_rpo, rpo_g, log_rpo_cov, d_rpo_cov)      zero when no RPO

so a firm's first-year excess ROE and its persistence shift with its backlog.
The firm-year (omega_hat, a_hat) predictions are written in the same format
as omega_firm_year.parquet and fed to sfv.rim with --omega-file, giving
V_F^backlog. Everything is expanding-window and point in time; RPO exists
from 2018, so backlog-augmented predictions start with as-of year 2020.

Outputs data/parquet/omega_firm_year_bl.parquet, reports/phase6_backlog_fit.md
Run: python -m sfv.backlog_value
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd
import statsmodels.api as sm

from . import config as C
from .persistence import annual_sample, Z, DUMMIES, MIN_FIT_YEARS

ZB = ["has_rpo", "rpo_g", "log_rpo_cov", "d_rpo_cov"]
DUMMIES_B = DUMMIES | {"has_rpo"}
FIRST_BL_YEAR = 2020
L = []


def p(t=""):
    L.append(t)


def add_backlog(A: pd.DataFrame, say) -> pd.DataFrame:
    B = pd.read_parquet(C.PQ / "backlog_quarterly.parquet", columns=["cik", "end", "rpo", "rpo_to_rev", "rpo_g", "d_rpo_cov"])
    A = A.merge(B, on=["cik", "end"], how="left")
    A["has_rpo"] = (A["rpo"] > 0).astype(float)
    A["log_rpo_cov"] = np.log(A["rpo_to_rev"]).where(A["rpo_to_rev"] > 0)
    for c in ("rpo_g", "log_rpo_cov", "d_rpo_cov"):
        A[c] = A[c].where(A["has_rpo"] > 0)
    say(f"  firm-years with RPO: {int(A.has_rpo.sum()):,} of {len(A):,} (years {A.loc[A.has_rpo > 0, 'year'].min()}-{A.loc[A.has_rpo > 0, 'year'].max()})")
    return A


def design_bl(A: pd.DataFrame, zs: list[str], fit_ref: pd.DataFrame | None = None):
    """Design matrix with base characteristics standardised on the fit sample and RPO
    variables standardised on RPO firms only, zero elsewhere."""
    ref = A if fit_ref is None else fit_ref
    X = pd.DataFrame({"x": A["x_roe_adj"].values}, index=A.index)
    zz = A[zs].copy()
    for c in zs:
        if c in DUMMIES_B:
            continue
        if c in ZB:
            rr = ref.loc[ref["has_rpo"] > 0, c]
            mu, sd = rr.mean(), rr.std() + 1e-12
            zz[c] = ((zz[c] - mu) / sd).fillna(0.0) * A["has_rpo"]
        else:
            mu, sd = ref[c].mean(), ref[c].std() + 1e-12
            zz[c] = (zz[c] - mu) / sd
    for c in zs:
        X[f"x*{c}"] = X["x"] * zz[c]
        X[c] = zz[c]
    X = sm.add_constant(X)
    return X, X.columns.tolist(), zz


def main() -> int:
    log = open(C.LOGS / "backlog_value.log", "w", encoding="utf-8")

    def say(msg):
        print(msg, flush=True); log.write(msg + "\n"); log.flush()

    t0 = time.time()
    A = annual_sample(say)
    A = add_backlog(A, say)
    zs = Z + ZB
    R = A[A["has_next"]]
    out = []
    coef_last = None
    for T_ in sorted(A["year"].unique()):
        if T_ < FIRST_BL_YEAR:
            continue
        fit = R[R["year"] + 1 <= T_]
        if fit["year"].nunique() < MIN_FIT_YEARS or (fit["has_rpo"] > 0).sum() < 300:
            continue
        X, cols, _ = design_bl(fit, zs)
        res = sm.OLS(fit["y_adj"].values, X.values).fit()
        b = pd.Series(res.params, index=cols)
        coef_last = pd.DataFrame({"coef": res.params, "t": res.tvalues}, index=cols)
        cur = A[A["year"] == T_]
        Xc, _, zz = design_bl(cur, zs, fit_ref=fit)
        omega = b["x"] + sum(b[f"x*{c}"] * zz[c] for c in zs)
        a_hat = b["const"] + sum(b[c] * zz[c] for c in zs)
        out.append(pd.DataFrame({"cik": cur["cik"].values, "asof_year": T_, "fit_through": T_, "n_fit": len(fit),
                                 "omega_hat": omega.values, "a_hat": a_hat.values, "x_t": cur["x_roe_adj"].values,
                                 "roe_adj": cur["roe_adj"].values, "roe_star": (cur["roe_adj"] - cur["x_roe_adj"]).values,
                                 "has_rpo": cur["has_rpo"].values}))
    O = pd.concat(out, ignore_index=True)
    O["omega_hat"] = O["omega_hat"].clip(0.0, 0.95)
    # earlier as-of years: reuse the base predictions so the RIM covers the full period
    base = pd.read_parquet(C.PQ / "omega_firm_year.parquet")
    base = base[base["asof_year"] < O["asof_year"].min()].assign(has_rpo=np.nan)
    O = pd.concat([base, O], ignore_index=True)
    O.to_parquet(C.PQ / "omega_firm_year_bl.parquet", index=False)
    say(f"  backlog-augmented predictions: {len(O):,} firm-years, as-of {O.asof_year.min()}-{O.asof_year.max()}  {(time.time()-t0)/60:.1f} min")

    p("# 단계 6 — 백로그를 넣은 초과 ROE 기대 모형\n")
    p(f"> 생성 {pd.Timestamp.today().date()}. 지속성 회귀에 RPO 변수(보유 더미, 성장, 두께, 변화)를 수준 항과 x 상호작용으로 추가. "
      f"확장 윈도우, as-of {O.asof_year.min()}~{O.asof_year.max()}, 백로그 항은 {FIRST_BL_YEAR}년부터.")
    p("\n## 1. 마지막 적합의 계수 (전체 표본, 결과 연도 ≤ 최신)\n")
    keep = ["const", "x"] + [c for c in coef_last.index if any(k in c for k in ZB)] + ["x*top5", "x*top20", "x*neg"]
    p(coef_last.loc[[c for c in keep if c in coef_last.index]].round(3).to_markdown())
    p("\n읽는 법: rpo_g, log_rpo_cov는 표준화(RPO 기업 내). 수준 항 계수가 양이면 백로그가 클수록 다음 해 초과 ROE가 높다는 뜻이고, "
      "x*항이 양이면 백로그가 클수록 초과이익이 더 오래 간다는 뜻이다. has_rpo는 RPO를 공시한다는 사실 자체의 효과.")
    d = O[O["asof_year"] >= FIRST_BL_YEAR]
    cmp = d.groupby(["asof_year", "has_rpo"]).agg(n=("cik", "size"), omega_med=("omega_hat", "median"), a_med=("a_hat", "median")).round(3)
    p("\n## 2. RPO 유무별 예측 ω·a 중앙값\n")
    p(cmp.to_markdown())
    (C.REPORTS / "phase6_backlog_fit.md").write_text("\n".join(L), encoding="utf-8")
    say("\n".join(L))
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
