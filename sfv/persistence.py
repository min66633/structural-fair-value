# -*- coding: utf-8 -*-
"""Persistence of excess ROE: estimated, not assumed.

The residual income model needs how fast a firm's excess return on book
fades. Textbook practice assumes a number (or "competitive advantage
period"). Here the fade parameter omega is estimated from the panel:

    x_{i,t+1} = a + (w0 + w'z_{i,t}) x_{i,t} + g'z_{i,t} + e

    x = ROE_adj - industry-year median ROE_adj  (excess ROE, winsorised)
    z = intangible intensity, R&D intensity, gross-margin stability,
        log size, log age, industry concentration (HHI), negative-x dummy

Three uses of the same regression:
  1. Description: Fama-MacBeth by fiscal year, so persistence can be
     compared across eras; run on ROE_adj and on GAAP ROE to see whether
     the "superstar" rise in persistence survives capitalising intangibles
     (Ayyagari et al. 2018 say most of the rise in ROIC dispersion does not).
  2. Test: pooled OLS with era dummies interacted with x (firm-clustered SE).
  3. Valuation input: for each as-of year T, the model is fitted on
     observations whose OUTCOME year <= T (expanding window, >= 3 years) and
     omega_i,T = w0 + w'z_i,T is predicted for every firm; nothing after T
     is used, so downstream valuations stay point in time.

Inputs   intangibles_quarterly, universe_monthly, sec_entities
Outputs  data/parquet/persistence_obs.parquet   the annual regression sample
         data/parquet/omega_firm_year.parquet   cik, asof_year, omega_hat, a_hat, x_t
         reports/phase1_persistence.md          tables
Run: python -m sfv.persistence
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd
import statsmodels.api as sm

from . import config as C

WINSOR = (-1.0, 1.0)          # excess ROE bounds
ERAS = [(2010, 2013, "2010-13"), (2014, 2017, "2014-17"), (2018, 2021, "2018-21"), (2022, 2026, "2022-26")]
Z = ["int_intensity", "rnd_intensity", "gm_stability", "log_size", "log_age", "hhi", "neg", "top20", "top5"]
DUMMIES = {"neg", "top20", "top5"}      # 0/1 regressors: not standardised
MIN_FIT_YEARS = 3
L = []


def p(t=""):
    L.append(t)


def annual_sample(say, intang_file: str = "intangibles_quarterly.parquet") -> pd.DataFrame:
    I = pd.read_parquet(C.PQ / intang_file)
    U = pd.read_parquet(C.PQ / "universe_monthly.parquet",
                        columns=["cik", "month", "mktcap", "in_universe", "ff12", "sic"])
    # fiscal-year rows only (one observation per firm-year), book > 0 both sides
    A = I[(I["fp"] == "FY") & I["ttm_ok"] & (I["b_adj"] > 0)].copy()
    A["year"] = A["end"].dt.year
    A = A.sort_values(["cik", "end"]).drop_duplicates(["cik", "year"], keep="last")
    # market cap and universe membership at the fiscal year-end month
    U["month"] = U["month"] + pd.offsets.MonthEnd(0)
    A["month"] = A["end"] + pd.offsets.MonthEnd(0)
    A = A.merge(U, on=["cik", "month"], how="left")
    A = A[A["in_universe"].fillna(False)]
    say(f"  firm-years in universe with FY row and book>0: {len(A):,}  firms {A.cik.nunique():,}  years {A.year.min()}-{A.year.max()}")

    # gross margin stability: std of quarterly gross margin over the last 8 quarters
    Q = pd.read_parquet(C.PQ / "fund_quarterly.parquet", columns=["cik", "end", "q_revenue", "q_cogs", "q_gross_profit"])
    Q = Q.sort_values(["cik", "end"])
    gp = Q["q_gross_profit"].where(Q["q_gross_profit"].notna(), Q["q_revenue"] - Q["q_cogs"])
    Q["gm"] = (gp / Q["q_revenue"]).where(Q["q_revenue"] > 0).clip(-1, 1)
    Q["gm_stability"] = Q.groupby("cik")["gm"].transform(lambda s: s.rolling(8, min_periods=6).std())
    A = A.merge(Q[["cik", "end", "gm_stability"]], on=["cik", "end"], how="left")

    # age since first panel observation (left-truncated at 2009: a rank, not an age)
    first = I.groupby("cik")["end"].min().rename("first_end")
    A = A.merge(first, on="cik", how="left")
    A["age"] = (A["end"] - A["first_end"]).dt.days / 365.25 + 1.0
    A["log_age"] = np.log(A["age"])
    A["log_size"] = np.log(A["mktcap"])

    # industry concentration: revenue HHI within 3-digit SIC per year, universe firms
    A["sic3"] = (A["sic"] // 10).astype("Int64")
    rev = A[A["ttm_revenue"] > 0]
    sh = rev["ttm_revenue"] / rev.groupby(["sic3", "year"])["ttm_revenue"].transform("sum")
    hhi = (sh ** 2).groupby([rev["sic3"], rev["year"]]).sum().rename("hhi").reset_index()
    A = A.merge(hhi, on=["sic3", "year"], how="left")

    # excess ROE vs industry-year median (FF12), winsorised
    for col in ("roe_adj", "roe_gaap"):
        A[col] = A[col].clip(*WINSOR)
        med = A.groupby(["ff12", "year"])[col].transform("median")
        A["x_" + col] = A[col] - med
    A["neg"] = (A["x_roe_adj"] < 0).astype(float)
    A["neg_gaap"] = (A["x_roe_gaap"] < 0).astype(float)
    # size-group dummies: persistence is strongly convex in size (top-5% firms
    # keep ~59% of excess ROE after five years, the average firm ~30%), which a
    # linear log-size interaction understates for the mega caps that dominate
    # aggregate value
    A["top20"] = (A["mktcap"] >= A.groupby("year")["mktcap"].transform(lambda s: s.quantile(.80))).astype(float)
    A["top5"] = (A["mktcap"] >= A.groupby("year")["mktcap"].transform(lambda s: s.quantile(.95))).astype(float)
    # next fiscal year's excess ROE (outcome), same firm, 340-390 days later
    nxt = A[["cik", "end", "x_roe_adj", "x_roe_gaap"]].rename(
        columns={"end": "end_next", "x_roe_adj": "y_adj", "x_roe_gaap": "y_gaap"})
    A = A.merge(nxt, on="cik", how="left")
    d = (A["end_next"] - A["end"]).dt.days
    has_next = d.between(340, 390) & A["y_adj"].notna()
    # one row per firm-year: the one with a next-year outcome when it exists
    A = A.assign(has_next=has_next).sort_values(["cik", "year", "has_next"], ascending=[True, True, False])
    A = A.drop_duplicates(["cik", "year"], keep="first")
    A.loc[~A["has_next"], ["y_adj", "y_gaap", "end_next"]] = np.nan
    A["era"] = pd.cut(A["year"], bins=[e[0] - 1 for e in ERAS] + [ERAS[-1][1]], labels=[e[2] for e in ERAS])
    for c in ("int_intensity", "rnd_intensity", "gm_stability", "hhi"):
        A[c] = A[c].fillna(A[c].median())
    A = A.dropna(subset=["x_roe_adj", "log_size", "log_age"])
    say(f"  characteristics sample: {len(A):,} firm-years ({A.year.min()}-{A.year.max()}); "
        f"with next-year outcome: {int(A.has_next.sum()):,} (outcome years {A.year.min()+1}-{A.year.max()+1})")
    return A


def design(A: pd.DataFrame, xcol: str, negcol: str, zs: list[str]) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    X = pd.DataFrame({"x": A[xcol].values}, index=A.index)
    zz = A[zs].copy()
    if "neg" in zs:
        zz["neg"] = A[negcol]
    # standardise continuous characteristics so interaction slopes are comparable
    for c in zs:
        if c not in DUMMIES:
            zz[c] = (zz[c] - zz[c].mean()) / (zz[c].std() + 1e-12)
    for c in zs:
        X[f"x*{c}"] = X["x"] * zz[c]
        X[c] = zz[c]
    X = sm.add_constant(X)
    return X, X.columns.tolist(), zz


def fama_macbeth(A: pd.DataFrame, xcol: str, ycol: str, negcol: str, zs: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    A = A.dropna(subset=[xcol, ycol, negcol] + [z for z in zs if z != "neg"])
    for yr, d in A.groupby("year"):
        if len(d) < 200:
            continue
        X, cols, _ = design(d, xcol, negcol, zs)
        b = sm.OLS(d[ycol].values, X.values).fit().params
        rows.append(pd.Series(b, index=cols, name=yr))
    T = pd.DataFrame(rows)
    mean = T.mean()
    # Newey-West (lag 2) standard errors on the yearly coefficient series
    se = {}
    for c in T.columns:
        s = T[c].values
        n = len(s)
        e = s - s.mean()
        v = (e @ e) / n
        for lag in (1, 2):
            if n > lag:
                w = 1 - lag / 3.0
                v += 2 * w * (e[lag:] @ e[:-lag]) / n
        se[c] = np.sqrt(max(v, 0) / n)
    out = pd.DataFrame({"mean": mean, "t": mean / pd.Series(se)})
    return out, T


def pooled_eras(A: pd.DataFrame, xcol: str, ycol: str, negcol: str) -> pd.DataFrame:
    """x_{t+1} = a + sum_e (w_e * x) 1{era e} + era dummies, firm-clustered SE."""
    A = A.dropna(subset=[xcol, ycol, negcol])
    X = pd.DataFrame(index=A.index)
    for e in ERAS:
        dmy = (A["era"] == e[2]).astype(float)
        X[f"x*{e[2]}"] = A[xcol] * dmy
        X[f"era_{e[2]}"] = dmy
    X["x*neg"] = A[xcol] * A[negcol]
    res = sm.OLS(A[ycol].values, X.values).fit(cov_type="cluster", cov_kwds={"groups": A["cik"].values})
    return pd.DataFrame({"coef": res.params, "t": res.tvalues}, index=X.columns)


def expanding_omega(A: pd.DataFrame, zs: list[str], say) -> pd.DataFrame:
    """Firm-year omega and intercept predicted from models fitted on outcome years <= T."""
    out = []
    years = sorted(A["year"].unique())
    R = A[A["has_next"]]                       # rows with an observed outcome
    for T_ in years:
        fit = R[R["year"] + 1 <= T_]
        if fit["year"].nunique() < MIN_FIT_YEARS:
            continue
        X, cols, _ = design(fit, "x_roe_adj", "neg", zs)
        b = pd.Series(sm.OLS(fit["y_adj"].values, X.values).fit().params, index=cols)
        cur = A[A["year"] == T_]
        # standardise the current year's characteristics with the FIT sample's moments
        zz = cur[zs].copy()
        zz["neg"] = cur["neg"]
        for c in zs:
            if c not in DUMMIES:
                mu, sd = fit[c].mean(), fit[c].std() + 1e-12
                zz[c] = (zz[c] - mu) / sd
        omega = b["x"] + sum(b[f"x*{c}"] * zz[c] for c in zs)
        a_hat = b["const"] + sum(b[c] * zz[c] for c in zs)
        out.append(pd.DataFrame({"cik": cur["cik"].values, "asof_year": T_, "fit_through": T_,
                                 "n_fit": len(fit), "omega_hat": omega.values, "a_hat": a_hat.values,
                                 "x_t": cur["x_roe_adj"].values, "roe_adj": cur["roe_adj"].values,
                                 "roe_star": (cur["roe_adj"] - cur["x_roe_adj"]).values}))
    O = pd.concat(out, ignore_index=True)
    O["omega_hat"] = O["omega_hat"].clip(0.0, 0.95)
    say(f"  omega predicted for {len(O):,} firm-years, as-of years {O.asof_year.min()}-{O.asof_year.max()}")
    return O


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--intangibles-file", default="intangibles_quarterly.parquet")
    ap.add_argument("--label", default="", help="suffix for every output, for assumption variants")
    a_ = ap.parse_args(argv)
    sfx = f"_{a_.label}" if a_.label else ""
    log = open(C.LOGS / f"persistence{sfx}.log", "w", encoding="utf-8")

    def say(msg):
        print(msg, flush=True); log.write(msg + "\n"); log.flush()

    t0 = time.time()
    A_all = annual_sample(say, a_.intangibles_file)
    A_all.to_parquet(C.PQ / f"persistence_obs{sfx}.parquet", index=False)
    A = A_all[A_all["has_next"]].copy()        # regression sample
    p("# 단계 1 — 초과 ROE 지속성(ω) 추정\n")
    p(f"> 표본: 유니버스 기업의 회계연도 관측치 {len(A):,}개, 기업 {A.cik.nunique():,}개, 결과 연도 {A.year.min()+1}~{A.year.max()+1}. "
      f"x = ROE − FF12 산업·연도 중앙값, [{WINSOR[0]}, {WINSOR[1]}]로 절단. 특성은 표준화.")

    # 1. simple persistence by year: adjusted vs GAAP (no characteristics)
    p("\n## 1. 단순 지속성 x_{t+1} = a + ω x_t, 연도별 (무형자산 조정 vs GAAP)\n")
    rows = []
    for yr, d in A.groupby("year"):
        if len(d) < 200:
            continue
        ba = sm.OLS(d["y_adj"].values, sm.add_constant(d["x_roe_adj"].values)).fit().params
        dg = d.dropna(subset=["x_roe_gaap", "y_gaap"])
        bg = sm.OLS(dg["y_gaap"].values, sm.add_constant(dg["x_roe_gaap"].values)).fit().params
        rows.append({"year_t": yr, "n": len(d), "omega_adj": ba[1], "omega_gaap": bg[1],
                     "x_adj_p90": d["x_roe_adj"].quantile(.9), "x_gaap_p90": dg["x_roe_gaap"].quantile(.9)})
    S = pd.DataFrame(rows).set_index("year_t")
    p(S.round(3).to_markdown())
    say(S.round(3).to_string())

    # 1b. persistence by size group and over longer horizons (adjusted ROE)
    p("\n## 1b. 규모별·다년 지속성 (조정 ROE)\n")
    rows = []
    for yr, d in A.groupby("year"):
        if len(d) < 200:
            continue
        x, y, w = d["x_roe_adj"].values, d["y_adj"].values, d["mktcap"].values

        def ols(x, y, w=None):
            Xm = np.column_stack([np.ones(len(x)), x])
            if w is None:
                return np.linalg.lstsq(Xm, y, rcond=None)[0][1]
            sw = np.sqrt(w)
            return np.linalg.lstsq(Xm * sw[:, None], y * sw, rcond=None)[0][1]
        t20 = d["top20"].values > 0; t5 = d["top5"].values > 0
        rows.append({"year_t": yr, "ew": ols(x, y), "vw": ols(x, y, w), "top20": ols(x[t20], y[t20]), "top5": ols(x[t5], y[t5]),
                     "x_med_top5": np.median(x[t5])})
    SZ = pd.DataFrame(rows).set_index("year_t")
    p("연도별 1년 지속성: ew 동일가중, vw 시총가중, top20/top5 규모 상위 20%·5% 기업만.\n")
    p(SZ.round(3).to_markdown())
    p(f"\n평균: ew {SZ.ew.mean():.3f}, vw {SZ.vw.mean():.3f}, top20 {SZ.top20.mean():.3f}, top5 {SZ.top5.mean():.3f}")
    A2 = A_all.sort_values(["cik", "year"]).copy()
    mh = []
    for k in (2, 3, 5):
        nx = A2.groupby("cik")["x_roe_adj"].shift(-k); ny = A2.groupby("cik")["year"].shift(-k)
        ok = (ny - A2["year"] == k) & nx.notna()
        for lab, m in (("전체", ok), ("상위5%", ok & (A2["top5"] > 0))):
            d = A2[m]
            b = np.linalg.lstsq(np.column_stack([np.ones(m.sum()), d["x_roe_adj"].values]), nx[m].values, rcond=None)[0][1]
            mh.append({"지평(년)": k, "집단": lab, "잔존 비율": b, "연환산 ω": b ** (1 / k), "n": int(m.sum())})
    MH = pd.DataFrame(mh)
    p("\nk년 뒤 초과 ROE의 잔존 비율 (x_{t+k}를 x_t에 회귀한 기울기):\n")
    p(MH.round(3).to_markdown(index=False))
    SZ.to_parquet(C.PQ / f"persistence_by_size{sfx}.parquet")
    MH.to_parquet(C.PQ / f"persistence_multiyear{sfx}.parquet", index=False)
    say(SZ.round(3).to_string()); say(MH.round(3).to_string())

    # 2. Fama-MacBeth with characteristics
    fm_adj, T_adj = fama_macbeth(A, "x_roe_adj", "y_adj", "neg", Z)
    fm_gaap, T_gaap = fama_macbeth(A, "x_roe_gaap", "y_gaap", "neg_gaap", Z)
    p("\n## 2. Fama-MacBeth (연도별 횡단면 회귀의 평균, Newey-West t)\n")
    p("x*특성 계수는 그 특성이 1표준편차 높을 때 ω가 얼마나 달라지는지를 뜻한다.\n")
    both = fm_adj.rename(columns={"mean": "adj_mean", "t": "adj_t"}).join(fm_gaap.rename(columns={"mean": "gaap_mean", "t": "gaap_t"}))
    p(both.round(3).to_markdown())
    say(both.round(3).to_string())
    T_adj.to_parquet(C.PQ / f"persistence_fm_yearly{sfx}.parquet")

    # 3. pooled with era dummies
    p("\n## 3. 시대별 ω (풀링 OLS, 기업 클러스터 SE)\n")
    pe_adj = pooled_eras(A, "x_roe_adj", "y_adj", "neg")
    pe_gaap = pooled_eras(A, "x_roe_gaap", "y_gaap", "neg_gaap")
    both2 = pe_adj.rename(columns={"coef": "adj_coef", "t": "adj_t"}).join(pe_gaap.rename(columns={"coef": "gaap_coef", "t": "gaap_t"}))
    p(both2.loc[[c for c in both2.index if c.startswith("x*")]].round(3).to_markdown())
    say(both2.round(3).to_string())
    # difference test: last era vs first era slope
    p("\n해석 기준: x*2022-26 − x*2010-13 이 양(+)이고 유의하면 초과이익 지속성이 구조적으로 늘어난 것. "
      "조정 ROE에서 그 차이가 GAAP보다 작으면 '슈퍼스타 지속성'의 일부는 무형자산 측정오차다.")

    # 4. expanding-window firm-level omega for valuation (predicted for every
    #    firm-year with characteristics, fitted only on observed outcomes)
    O = expanding_omega(A_all, Z, say)
    O.to_parquet(C.PQ / f"omega_firm_year{sfx}.parquet", index=False)
    p("\n## 4. 밸류에이션용 기업별 ω (확장 윈도우 예측)\n")
    dist = O.groupby("asof_year")["omega_hat"].describe(percentiles=[.1, .5, .9])[["count", "10%", "50%", "90%"]]
    p(dist.round(3).to_markdown())
    say(dist.round(3).to_string())

    (C.REPORTS / f"phase1_persistence{sfx}.md").write_text("\n".join(L), encoding="utf-8")
    say(f"done  {(time.time()-t0)/60:.1f} min  -> reports/phase1_persistence{sfx}.md")
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
