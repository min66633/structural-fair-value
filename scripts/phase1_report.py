# -*- coding: utf-8 -*-
"""Phase 1 checkpoint report: intangibles, persistence, residual income values.

Writes reports/phase1_module_f.md. Every number is measured from the parquet
outputs; the data period is stated with each table.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sfv import config as C  # noqa: E402

pd.set_option("display.width", 220)
L = []


def p(t=""):
    L.append(t)


def main():
    I = pd.read_parquet(C.PQ / "intangibles_quarterly.parquet")
    U = pd.read_parquet(C.PQ / "universe_monthly.parquet", columns=["cik", "month", "in_universe", "mktcap", "ff12"])
    R = pd.read_parquet(C.PQ / "rim_monthly.parquet")
    O = pd.read_parquet(C.PQ / "omega_firm_year.parquet")

    p("# 단계 1 — 모듈 F (펀더멘털 가치) 체크포인트\n")
    p(f"> 생성 {pd.Timestamp.today().date()}. 유니버스: 미국 비금융 보통주, 시총 3억$ 이상, 재무 400일 이내. "
      f"밸류에이션 표본 {R.month.min().date()} ~ {R.month.max().date()}, 기업 {R.cik.nunique():,}개, 기업-월 {len(R):,}.")

    # ---- 1. universe
    p("\n## 1. 유니버스\n")
    Uu = U[U["in_universe"]]
    yr = Uu.assign(y=Uu["month"].dt.year).groupby("y").agg(firms=("cik", "nunique"), mcap_tn=("mktcap", lambda s: s.sum() / 12 / 1e12))
    p(yr.round(2).T.to_markdown())

    # ---- 2. intangibles
    p("\n## 2. 무형자산 자본화 (R&D 100%·판관비 30%, 상각 15%·20%)\n")
    Iu = I.merge(Uu[["cik", "month"]].assign(month=lambda d: d["month"] + pd.offsets.MonthEnd(0)),
                 left_on=["cik", I["end"] + pd.offsets.MonthEnd(0)], right_on=["cik", "month"], how="inner")
    Iu = Iu[(Iu["b_adj"] > 0) & Iu["ttm_ok"]]
    yi = Iu.assign(y=Iu["qend"].dt.year).groupby("y").apply(
        lambda d: pd.Series({"firms": d.cik.nunique(),
                             "K_int/B_adj (vw)": np.average(d.int_intensity, weights=d.b_adj),
                             "K_int/B_adj (med)": d.int_intensity.median(),
                             "ROE gaap (med)": d.roe_gaap.median(), "ROE adj (med)": d.roe_adj.median(),
                             "ROE gaap (p90)": d.roe_gaap.quantile(.9), "ROE adj (p90)": d.roe_adj.quantile(.9)}),
        include_groups=False)
    p("유니버스 기업, 연도별:\n")
    p(yi.round(3).to_markdown())
    p("\n대표 기업 (최근 분기, 10억$):\n")
    rows = []
    for cik, nm in [(320193, "Apple"), (789019, "Microsoft"), (1652044, "Alphabet"), (1326801, "Meta"), (1045810, "NVIDIA"),
                    (78003, "Pfizer"), (104169, "Walmart"), (1318605, "Tesla"), (34088, "Exxon"), (1800, "Abbott")]:
        a = I[I.cik == cik].sort_values("end").tail(1)
        if len(a):
            a = a.iloc[0]
            rows.append({"기업": nm, "분기말": a.end.date(), "B GAAP": a.b_gaap / 1e9, "K_int": a.k_int / 1e9, "K_R&D": a.k_rnd / 1e9,
                         "K_org": a.k_org / 1e9, "E_ttm GAAP": a.ttm_e_gaap / 1e9, "E_ttm adj": a.ttm_e_adj / 1e9,
                         "ROE GAAP": a.roe_gaap, "ROE adj": a.roe_adj})
    p(pd.DataFrame(rows).round(2).to_markdown(index=False))

    # ---- 3. persistence (from its own report)
    p("\n## 3. 지속성 ω\n")
    pr = (C.REPORTS / "phase1_persistence.md")
    if pr.exists():
        p("자세한 표는 `reports/phase1_persistence.md`. 밸류에이션에 쓰인 기업별 ω 분포:\n")
        d = O.groupby("asof_year")["omega_hat"].describe(percentiles=[.1, .5, .9])[["count", "10%", "50%", "90%"]]
        p(d.round(3).to_markdown())

    # ---- 4. RIM values
    p("\n## 4. 잔여이익모형 가치 V_F 와 log(P/V_F)\n")
    yr2 = R.assign(y=R["month"].dt.year).groupby("y").apply(
        lambda d: pd.Series({"firms": d.cik.nunique(),
                             "log(P/V) 중앙값": d.log_pv.median(), "log(P/V) p10": d.log_pv.quantile(.1), "log(P/V) p90": d.log_pv.quantile(.9),
                             "log(ΣP/ΣV) 시총가중": np.log(d.mktcap.sum() / d.v_f.clip(lower=0).sum()),
                             "터미널 0 가정": np.log(d.mktcap.sum() / d.v_f_notv.clip(lower=0).sum()),
                             "log(P/B_adj) 중앙값": d.log_pb_adj.median(),
                             "ICC 중앙값": d.icc.median(), "r 중앙값": d.r.median(), "r_f": d.rf.median(), "ERP": d.erp.median()}),
        include_groups=False)
    p("연도별 (log 값 0.1 ≈ 10% 프리미엄; 시총가중은 유니버스 전체 시가총액 ÷ 전체 V_F):\n")
    p(yr2.round(3).to_markdown())
    p("\n산업별 (FF12), 2025~2026 평균:\n")
    rec = R[R["month"] >= "2025-01-01"]
    ind = rec.groupby("ff12").apply(lambda d: pd.Series({"firms": d.cik.nunique(), "log(P/V) 중앙값": d.log_pv.median(),
                                                        "시총가중": np.log(d.mktcap.sum() / d.v_f.clip(lower=0).sum()),
                                                        "ICC 중앙값": d.icc.median(), "ω 중앙값": d.omega.median()}), include_groups=False)
    p(ind.round(3).to_markdown())
    last = R[R["month"] == R["month"].max()].sort_values("mktcap", ascending=False)
    p(f"\n최근 월 {R.month.max().date()} 시총 상위 15:\n")
    top = last.head(15)[["ticker", "name", "mktcap", "b_adj", "v_f", "log_pv", "roe_adj", "roe_star_ind", "omega", "r", "icc"]].copy()
    for c in ("mktcap", "b_adj", "v_f"):
        top[c] = top[c] / 1e9
    top["name"] = top["name"].str.slice(0, 22)
    p(top.round(3).to_markdown(index=False))
    p("\n## 5. 읽는 법과 한계\n")
    p("- V_F는 관측된 조정 장부가 + 10년 초과이익 현재가치 + 터미널 항이다. '터미널 0 가정' 열이 V_F의 가정 의존도를 보여 준다.")
    p("- ω는 산업 중앙값 대비 초과 ROE의 1년 지속률이며 기업 특성으로 예측한 값이다. 0.7이면 초과이익의 30%가 매년 사라진다.")
    p("- log(P/V_F)가 양수라고 '고평가'가 아니다. 단계 3에서 이 값이 향후 수익률을 예측하는지, 어느 성분(δ_S, δ_T, ε)이 설명하는지 검정한다.")
    p("- 할인율은 r_f + β×ERP 이고 ERP는 Damodaran 전년 말 내재 ERP다. 2026년은 2025년 말 값(4.23%)을 쓴다.")
    (C.REPORTS / "phase1_module_f.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
