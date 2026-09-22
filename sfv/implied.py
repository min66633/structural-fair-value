# -*- coding: utf-8 -*-
"""Market-level re-diagnosis: what persistence does the price assume?

Cross-sections cannot identify the LEVEL of the market's premium over
fundamental value (the intercept absorbs it). This module works at the
aggregate instead, in the spirit of Damodaran's implied ERP:

  omega*   the common persistence of excess ROE at which the universe's
           total V_F equals its total market cap, holding the discount rate
           at r_f + beta * ERP. Solved monthly by bisection. Compared with
           the persistence actually estimated from realised ROE (sfv.persistence).
  T*       implied competitive-advantage period: the number of years current
           excess ROE would have to persist undecayed (then vanish) to
           justify the price.
  frontier the (r, omega*) trade-off: the implied persistence if the discount
           rate were 1-2 points higher or lower, for the latest month and a
           few historical dates.
  counterfactuals   aggregate log(P/V_F) if the discount rate, the persistence,
           or the industry ROE level had stayed at their 2014 levels - the
           share of the premium's rise each input can account for.
  demand   value-weighted big-three share, index-fund share, net issuance,
           for the time-series picture (descriptive only: 12 years).

Outputs data/parquet/implied_monthly.parquet, reports/phase4_implied.md
Run: python -m sfv.implied
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

from . import config as C
from .rim import T_YEARS, X_INF_CAP, X0_CAP, G_CAP, G_TV_CAP

Q_MAX_IMPLIED = 0.99
L = []


def p(t=""):
    L.append(t)


def value_common_omega(B0, x0, xinf, roe_star, payout, r, omega, T=T_YEARS):
    """Aggregate-friendly value with a common persistence omega (scalar or array)."""
    x = np.clip(x0, -X0_CAP, X0_CAP)
    B_prev = B0.copy()
    pv = np.zeros_like(B0)
    disc = np.ones_like(B0)
    for k in range(1, T + 1):
        x = xinf + omega * (x - xinf)
        roe = roe_star + x
        ri = (roe - r) * B_prev
        disc = disc * (1.0 + r)
        pv = pv + ri / disc
        g = np.clip(roe * (1.0 - payout), -0.5, G_CAP)
        B_next = B_prev * (1.0 + g)
        if k == T:
            ri_T, g_T = ri, g
        B_prev = B_next
    q = np.clip(omega * (1.0 + np.minimum(g_T, G_TV_CAP)) / (1.0 + r), 0.0, Q_MAX_IMPLIED)
    tv = ri_T * q / (1.0 - q) / disc
    return B0 + pv + tv


def value_cap(B0, x0, roe_star, payout, r, T):
    """Value with excess ROE held at x0 for T years, zero after (competitive advantage period)."""
    x = np.clip(x0, -X0_CAP, X0_CAP)
    roe = roe_star + x
    B_prev = B0.copy()
    pv = np.zeros_like(B0)
    disc = np.ones_like(B0)
    g = np.clip(roe * (1.0 - payout), -0.5, G_CAP)
    for k in range(1, T + 1):
        ri = (roe - r) * B_prev
        disc = disc * (1.0 + r)
        pv = pv + ri / disc
        B_prev = B_prev * (1.0 + g)
    return B0 + pv


def solve_omega(B0, x0, xinf, rs, po, r, target, lo=0.0, hi=0.995, it=40):
    """Bisection on the common omega: sum V(omega) = target (V increasing in omega)."""
    vlo = value_common_omega(B0, x0, xinf, rs, po, r, lo).sum()
    vhi = value_common_omega(B0, x0, xinf, rs, po, r, hi).sum()
    if target <= vlo:
        return lo, vlo
    if target >= vhi:
        return hi, vhi
    for _ in range(it):
        mid = 0.5 * (lo + hi)
        v = value_common_omega(B0, x0, xinf, rs, po, r, mid).sum()
        if v < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi), v


def solve_cap(B0, x0, rs, po, r, target, Tmax=60):
    prev_v = value_cap(B0, x0, rs, po, r, 0).sum()
    if target <= prev_v:
        return 0.0
    for T in range(1, Tmax + 1):
        v = value_cap(B0, x0, rs, po, r, T).sum()
        if v >= target:
            return (T - 1) + (target - prev_v) / max(v - prev_v, 1e-9)
        prev_v = v
    return float(Tmax) + 1.0      # "> Tmax"


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--rim-file", default="rim_monthly.parquet")
    ap.add_argument("--label", default="", help="suffix for outputs (e.g. coe)")
    a_ = ap.parse_args(argv)
    suf = f"_{a_.label}" if a_.label else ""
    log = open(C.LOGS / f"implied{suf}.log", "w", encoding="utf-8")

    def say(msg):
        print(msg, flush=True); log.write(msg + "\n"); log.flush()

    t0 = time.time()
    R = pd.read_parquet(C.PQ / a_.rim_file,
                        columns=["cik", "month", "ff12", "mktcap", "b_adj", "x0", "a", "omega", "roe_star_ind", "payout",
                                 "r", "rf", "erp", "beta", "v_f"])
    R = R.dropna(subset=["b_adj", "x0", "a", "omega", "roe_star_ind", "payout", "r", "mktcap"])
    R["xinf"] = np.clip(R["a"] / (1.0 - np.minimum(R["omega"], 0.9)), -X_INF_CAP, X_INF_CAP)
    D = pd.read_parquet(C.PQ / "demand_monthly.parquet", columns=["cik", "month", "big3_share", "index_share", "nsi_12m", "inst_share"])
    R = R.merge(D, on=["cik", "month"], how="left")
    say(f"  firm-months {len(R):,}, months {R.month.nunique()}, {R.month.min().date()}..{R.month.max().date()}")

    # 2014 reference levels for the counterfactuals
    base = R[R["month"].dt.year == 2014]
    rf14, erp14 = base["rf"].mean(), base["erp"].mean()
    om14 = base["omega"].median()
    rs14 = base.groupby("ff12")["roe_star_ind"].mean()
    say(f"  2014 reference: rf {rf14:.4f} erp {erp14:.4f} omega median {om14:.3f}")

    rows = []
    for m, d in R.groupby("month"):
        B0 = d["b_adj"].to_numpy(float); x0 = d["x0"].to_numpy(float); xinf = d["xinf"].to_numpy(float)
        rs = d["roe_star_ind"].to_numpy(float); po = d["payout"].to_numpy(float); r = d["r"].to_numpy(float)
        P = d["mktcap"].to_numpy(float)
        target = P.sum()
        v_base = value_common_omega(B0, x0, xinf, rs, po, r, np.minimum(d["omega"].to_numpy(float), 0.9)).sum()
        om_star, _ = solve_omega(B0, x0, xinf, rs, po, r, target)
        cap_star = solve_cap(B0, x0, rs, po, r, target)
        # frontier: implied omega at r +- 1, 2 points
        fr = {}
        for dr in (-0.02, -0.01, 0.01, 0.02):
            fr[f"omega_star_r{dr:+.2f}"] = solve_omega(B0, x0, xinf, rs, po, r + dr, target)[0]
        # counterfactuals
        r_cf = rf14 + d["beta"].to_numpy(float) * erp14
        v_r14 = value_common_omega(B0, x0, xinf, rs, po, r_cf, np.minimum(d["omega"].to_numpy(float), 0.9)).sum()
        v_om14 = value_common_omega(B0, x0, xinf, rs, po, r, om14).sum()
        rs_cf = d["ff12"].map(rs14).fillna(pd.Series(rs, index=d.index)).to_numpy(float)
        v_rs14 = value_common_omega(B0, x0, xinf, rs_cf, po, r, np.minimum(d["omega"].to_numpy(float), 0.9)).sum()
        w = P / target

        def vw(col):
            """Value-weighted mean over rows where col is present (weights renormalised, not zero-filled)."""
            s = d[col].to_numpy(float)
            ok = np.isfinite(s)
            return float(np.sum(w[ok] * s[ok]) / np.sum(w[ok])) if ok.any() and np.sum(w[ok]) > 0 else np.nan

        rows.append({"month": m, "n": len(d), "sum_P_tn": target / 1e12, "sum_V_tn": v_base / 1e12,
                     "log_pv": np.log(target / v_base), "omega_star": om_star, "cap_star": cap_star,
                     "log_pv_r2014": np.log(target / v_r14), "log_pv_omega2014": np.log(target / v_om14),
                     "log_pv_roestar2014": np.log(target / v_rs14),
                     "rf": d["rf"].iloc[0], "erp": d["erp"].iloc[0], "r_vw": float(w @ r),
                     "omega_vw": float(w @ d["omega"].to_numpy(float)), "roe_adj_vw": float(w @ (rs + x0)),
                     "roe_star_vw": float(w @ rs), "big3_vw": vw("big3_share"), "index_vw": vw("index_share"),
                     "nsi_vw": vw("nsi_12m"), **fr})
    M = pd.DataFrame(rows).set_index("month")
    M.to_parquet(C.PQ / f"implied_monthly{suf}.parquet")
    say(f"  solved {len(M)} months  {(time.time()-t0)/60:.1f} min")

    # industry implied omega at selected dates (only those present in the sample:
    # the reference dates are hardcoded, the sample is not)
    REF_DATES = [pd.Timestamp("2014-12-31"), pd.Timestamp("2019-12-31"), pd.Timestamp("2021-12-31")]
    sel_dates = [m for m in REF_DATES if m in M.index] + [M.index.max()]
    ind_rows = {}
    for m in sel_dates:
        d = R[R["month"] == m]
        for ind, di in d.groupby("ff12"):
            if len(di) < 30:
                continue
            om, _ = solve_omega(di["b_adj"].to_numpy(float), di["x0"].to_numpy(float), di["xinf"].to_numpy(float),
                                di["roe_star_ind"].to_numpy(float), di["payout"].to_numpy(float), di["r"].to_numpy(float),
                                di["mktcap"].sum())
            ind_rows.setdefault(ind, {})[str(m.date())] = om
    IND = pd.DataFrame(ind_rows).T

    # realised persistence for comparison
    try:
        Fy = pd.read_parquet(C.PQ / "persistence_fm_yearly.parquet")
        realised = Fy["x"].rename("realised_omega_fm")
    except Exception:
        realised = None

    p(f"# 단계 4 — 시장 수준 재진단: 가격이 전제하는 지속성{' (' + a_.label + ' 모드)' if a_.label else ''}\n")
    p(f"> 생성 {pd.Timestamp.today().date()}. 입력 {a_.rim_file}. 유니버스 시가총액 합계 = V_F 합계가 되는 공통 지속성 ω*를 매월 역산. "
      f"할인율은 r_f + β·ERP(Damodaran 전년 말)로 고정. 경계값은 단계 1과 같되 터미널 q 상한만 0.99. "
      f"표본 {M.index.min().date()} ~ {M.index.max().date()}, 월 {len(M)}개.")
    p("\n## 1. 연도별 (12월 말 값)\n")
    yr = M[M.index.month == 12][["n", "sum_P_tn", "sum_V_tn", "log_pv", "omega_star", "cap_star", "rf", "erp", "r_vw",
                                  "omega_vw", "roe_adj_vw", "roe_star_vw", "big3_vw", "index_vw", "nsi_vw"]].copy()
    last = M.iloc[[-1]][yr.columns]
    yr = pd.concat([yr, last])
    yr.index = [str(x.date()) for x in yr.index]
    p(yr.round(3).to_markdown())
    p("\n- omega_star: 가격을 정당화하는 공통 지속성. omega_vw: 실현 ROE로 추정한 기업별 ω의 시총가중 평균(단계 1). 둘의 차이가 '시장이 추가로 전제하는 지속성'.")
    p("- cap_star: 현재 초과 ROE가 감쇠 없이 몇 년 지속돼야 가격이 맞는지(년). 61이면 60년 이상.")
    p("- roe_adj_vw: 시총가중 조정 ROE. roe_star_vw: 산업 중앙값 ROE의 시총가중 평균.")
    if realised is not None:
        p("\n실현 지속성(연도별 횡단면 회귀의 x 계수, 단계 1):\n")
        p(realised.round(3).to_frame().T.to_markdown())
    try:
        SZ = pd.read_parquet(C.PQ / "persistence_by_size.parquet")
        MH = pd.read_parquet(C.PQ / "persistence_multiyear.parquet")
        p("\n## 1b. 내재 지속성 vs 규모별 실현 지속성\n")
        cmp = pd.DataFrame({"시장 내재 ω* (12월)": M[M.index.month == 12]["omega_star"].round(3).values},
                           index=[x.year for x in M[M.index.month == 12].index])
        cmp = cmp.join(SZ[["ew", "vw", "top20", "top5"]].rename(columns={"ew": "실현 동일가중", "vw": "실현 시총가중", "top20": "실현 상위20%", "top5": "실현 상위5%"}), how="left")
        p(cmp.round(3).to_markdown())
        p(f"\n평균 실현 지속성: 동일가중 {SZ.ew.mean():.2f}, 시총가중 {SZ.vw.mean():.2f}, 상위20% {SZ.top20.mean():.2f}, 상위5% {SZ.top5.mean():.2f}. "
          "시장 가치를 지배하는 대형주의 실현 지속성은 동일가중 평균보다 훨씬 높다.")
        p("\n다년 잔존 비율:\n")
        p(MH.round(3).to_markdown(index=False))
    except Exception as ex:
        p(f"\n(규모별 지속성 표 없음: {ex})")

    p("\n## 2. 할인율-지속성 프론티어 (최근 월과 과거 시점)\n")
    fr_cols = ["omega_star_r-0.02", "omega_star_r-0.01", "omega_star", "omega_star_r+0.01", "omega_star_r+0.02"]
    sel = M.loc[sel_dates, fr_cols]
    sel.columns = ["r −2%p", "r −1%p", "r 기준", "r +1%p", "r +2%p"]
    sel.index = [str(x.date()) for x in sel.index]
    p(sel.round(3).to_markdown())
    p("\n같은 가격을 설명하는 (할인율, 지속성) 조합. 할인율이 1%p 낮다고 보면 요구되는 지속성이 얼마나 낮아지는지를 보여 준다.")

    p("\n## 3. 반사실 분해: 입력을 2014년 수준에 고정했을 때의 log(ΣP/ΣV_F)\n")
    cf = M[M.index.month == 12][["log_pv", "log_pv_r2014", "log_pv_omega2014", "log_pv_roestar2014"]].copy()
    cf = pd.concat([cf, M.iloc[[-1]][cf.columns]])
    cf.index = [str(x.date()) for x in cf.index]
    cf["금리·ERP 몫"] = cf["log_pv"] - cf["log_pv_r2014"]
    cf["지속성 몫"] = cf["log_pv"] - cf["log_pv_omega2014"]
    cf["산업 ROE 수준 몫"] = cf["log_pv"] - cf["log_pv_roestar2014"]
    p(cf.round(3).to_markdown())
    p("\n읽는 법: '금리·ERP 몫'이 양수면 2014년 할인율이었다면 프리미엄이 그만큼 더 컸을 것(즉 할인율 하락이 프리미엄을 낮춰 준 것), 음수면 반대. "
      "각 몫은 하나씩 바꾼 부분 효과라 합이 전체와 같지 않다.")

    p("\n## 4. 산업별 내재 지속성 ω*\n")
    p(IND.round(3).to_markdown())

    p("\n## 5. 해석과 한계\n")
    p("- ω*는 '고평가 판정'이 아니라 '가격이 전제하는 가정'이다. ω*가 실현 지속성보다 높으면 시장은 초과이익이 과거보다 오래 갈 것이라고 보고 있다. 그 기대가 맞는지는 사후에만 안다.")
    p("- 할인율과 지속성은 동시에 식별되지 않는다. 프론티어가 그 불확실성의 크기다.")
    p("- V_F의 경계값(장부 성장 25%, 장기 초과 ROE ±10%p)이 ω*에도 그대로 걸린다. 경계를 풀면 ω*는 낮아진다.")
    p("- 12년 시계열로는 통계적 판정이 불가능하다. 이 표는 진단이지 검정이 아니다.")
    (C.REPORTS / f"phase4_implied{suf}.md").write_text("\n".join(L), encoding="utf-8")
    say("\n".join(L[-60:]))
    say(f"done  {(time.time()-t0)/60:.1f} min -> reports/phase4_implied{suf}.md")
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
