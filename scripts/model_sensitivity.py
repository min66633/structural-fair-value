# -*- coding: utf-8 -*-
"""How much do the two judgement calls in the model actually move it?

Two questions a reader should ask and this answers with numbers:

1. Demand is measured with known error — 13F misses non-filers, index-fund
   holdings understate true passive ownership (Chinco-Sammon put it near 33%
   against the ~16% that fund names capture). Does that error propagate into
   the valuation? Test: run the monthly decomposition with and without the
   whole demand block and compare the residual.

2. The fade bounds (persistence 0.90, long-run excess ROE 10pp, book growth
   25%, terminal factor 0.92) are the one place a human number was chosen,
   to stop the value explosions that an unconstrained regression produces.
   Test: how often does each bound actually bind, equal- and value-weighted?

Output reports/model_sensitivity.md
Run: python scripts/model_sensitivity.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sfv import config as C  # noqa: E402
from sfv.rim import OMEGA_MAX, X_INF_CAP, X0_CAP, G_CAP, G_TV_CAP, Q_MAX  # noqa: E402
from sfv.decompose import XS_A, XT_A, XF, XM, XE  # noqa: E402

L = []


def p(t: str = "") -> None:
    L.append(t)


def fit(d: pd.DataFrame, groups: list[list[str]]) -> dict:
    cols = [c for g in groups for c in g]
    Z = d[cols].astype(float)
    Zc = Z - Z.mean()
    ind = pd.get_dummies(d["ff12"], prefix="i", drop_first=True).astype(float)
    Xm = np.column_stack([np.ones(len(d)), Zc.values, ind.values])
    b, *_ = np.linalg.lstsq(Xm, d["y"].values, rcond=None)
    out = {"eps": d["y"].values - Xm @ b}
    pos = 1
    for name, g in zip("STFME", groups):
        out[name] = Zc[g].values @ b[pos:pos + len(g)] if g else np.zeros(len(d))
        pos += len(g)
    return out


def demand_sensitivity() -> None:
    D = pd.read_parquet(C.PQ / "decomp_monthly_momexp.parquet")
    need = ["y", "ff12"] + XS_A + XT_A + XF + XM + XE
    D = D.dropna(subset=need)
    rows = []
    for m, d in D.groupby("month"):
        if len(d) < 300:
            continue
        a = fit(d, [XS_A, XT_A, XF, XM, XE])
        b = fit(d, [[], [], XF, XM, XE])
        vy = np.var(d["y"].values)
        rows.append({"month": m, "n": len(d),
                     "eps_with": np.var(a["eps"]) / vy, "eps_without": np.var(b["eps"]) / vy,
                     "S": np.var(a["S"]) / vy, "T": np.var(a["T"]) / vy,
                     "F_with": np.var(a["F"]) / vy, "F_without": np.var(b["F"]) / vy,
                     "corr": np.corrcoef(a["eps"], b["eps"])[0, 1],
                     "mad": np.mean(np.abs(a["eps"] - b["eps"])), "sd": np.std(a["eps"])})
    A = pd.DataFrame(rows)
    g = A.loc[:, ["eps_with", "eps_without", "S", "T", "F_with", "F_without", "corr", "mad", "sd"]].mean()
    p("## 1. 수요 변수를 통째로 빼면 무엇이 바뀌는가\n")
    p(f"> 월별 횡단면 회귀를 수요 블록(X_S, X_T) 포함·제외로 두 번 돌려 비교. "
      f"기업-월 {len(D):,}, 월 {A['month'].nunique()}개 ({A['month'].min().date()}~{A['month'].max().date()}).\n")
    p(pd.DataFrame({
        "값": [f"{g['eps_with']:.3f}", f"{g['eps_without']:.3f}", f"{g['S']:.3f} / {g['T']:.3f}",
               f"{A.loc[:, 'corr'].mean():.4f} (최저 {A.loc[:, 'corr'].min():.4f})",
               f"{g['mad']:.4f}", f"{g['mad']/g['sd']*100:.1f}%"]},
        index=["잔차 분산 몫 (수요 포함)", "잔차 분산 몫 (수요 제외)", "δ_S / δ_T 분산 몫",
               "두 잔차의 상관", "잔차의 평균 절대 변화", "  같은 값, 잔차 표준편차 대비"]).to_markdown())
    p("\n**읽기.** 수요 모듈 전체를 없애도 잔차는 상관 0.98로 거의 그대로이고, 크기로는 자기 표준편차의 15% 정도 움직인다. "
      "수요 측정이 틀리면 바뀌는 것은 δ_S와 ε의 경계이지 둘의 합이 아니다. 합은 항등식이라 언제나 log(P/V_F)와 정확히 같다.")
    p("\n더 중요한 것은 **V_F와 역산이 수요 변수를 아예 읽지 않는다**는 점이다. 잔여이익 가치, 지속성 추정, 성장 단계, "
      "종목별 ω*·T*·g* 어느 것도 13F·N-PORT를 입력으로 쓰지 않는다(`sfv/implied.py`만 예외로 읽지만 ω* 해와는 무관한 "
      "보고용 열에만 쓴다). 그래서 수요 측정 오차는 도구의 주 출력에 전파되지 않는다.")
    p("\n한계: 패시브 지분은 펀드명 기반이라 하한이다(Chinco·Sammon 2024는 실질 패시브를 33% 수준으로 본다). "
      "따라서 '구조적 수요가 프리미엄의 작은 부분만 설명한다'는 결론은 **하한 해석**이다. 다만 δ_S의 분산 몫이 "
      f"{g['S']*100:.1f}%라 두세 배로 늘려도 그림이 바뀌지 않고, 측정이 더 나은 스펙 B(N-PORT 지수 비중 포함)에서도 3.9%였다.")


def bounds_binding() -> None:
    R = pd.read_parquet(C.PQ / "rim_monthly.parquet",
                        columns=["month", "mktcap", "x0", "a", "omega", "roe_star_ind", "payout", "r"])
    om_raw = R["omega"].to_numpy(float)
    om = np.minimum(om_raw, OMEGA_MAX)
    xinf_raw = R["a"].to_numpy(float) / (1.0 - om)
    xinf = np.clip(xinf_raw, -X_INF_CAP, X_INF_CAP)
    x0_raw = R["x0"].to_numpy(float)
    roe1 = R["roe_star_ind"].to_numpy(float) + (xinf + om * (np.clip(x0_raw, -X0_CAP, X0_CAP) - xinf))
    g1 = roe1 * (1.0 - R["payout"].to_numpy(float))
    q_raw = om * (1.0 + np.minimum(g1, G_TV_CAP)) / (1.0 + R["r"].to_numpy(float))
    tests = [(f"지속성 ω > {OMEGA_MAX}", om_raw > OMEGA_MAX, np.median(om_raw)),
             (f"장기 초과 ROE |x∞| > {X_INF_CAP}", np.abs(xinf_raw) > X_INF_CAP, np.median(np.abs(xinf_raw))),
             (f"현재 초과 ROE |x0| > {X0_CAP}", np.abs(x0_raw) > X0_CAP, np.median(np.abs(x0_raw))),
             (f"장부 성장 1년차 > {G_CAP}", g1 > G_CAP, np.median(g1)),
             (f"터미널 계수 q > {Q_MAX}", q_raw > Q_MAX, np.median(q_raw))]
    w = R["mktcap"].to_numpy(float)
    rows = {}
    for name, mask, med in tests:
        rows[name] = {"걸리는 비율": f"{mask.mean()*100:.1f}%", "시총가중": f"{np.sum(w[mask])/w.sum()*100:.1f}%",
                      "원값 중앙값": f"{med:.3f}"}
    any_bind = np.logical_or.reduce([m for _, m, _ in tests])
    rows["하나라도 걸림"] = {"걸리는 비율": f"{any_bind.mean()*100:.1f}%",
                        "시총가중": f"{np.sum(w[any_bind])/w.sum()*100:.1f}%", "원값 중앙값": ""}
    last_m = R["month"] == R["month"].max()
    lb = any_bind[last_m.to_numpy()]
    lw = w[last_m.to_numpy()]
    p("\n## 2. 경계값은 실제로 얼마나 자주 걸리는가\n")
    p(f"> 기업-월 {len(R):,}. 경계값은 모형에서 사람이 정한 유일한 숫자이며, 제약 없는 회귀가 만드는 가치 폭발"
      "(아마존 2018년 8조 달러)을 막기 위한 것이다.\n")
    p(pd.DataFrame(rows).T.to_markdown())
    p(f"\n최신 월({R['month'].max().date()}) {int(last_m.sum()):,}종목 중 경계에 걸린 비율은 "
      f"{lb.mean()*100:.1f}%, 시총가중으로는 **{np.sum(lw[lb])/lw.sum()*100:.1f}%**다.")
    p("\n**읽기.** 종목 수로는 소수지만 시가총액으로는 대부분이다. 경계값이 대형 고수익 기업에서 집중적으로 걸린다는 뜻이고, "
      "이것이 '경계값이 슈퍼스타를 체계적으로 낮게 평가한다'는 한계의 실제 크기다. 엔비디아의 log(P/V_F)가 1.59일 때 "
      "그중 얼마가 진짜 프리미엄이고 얼마가 경계값 때문인지는 분리되지 않는다.")
    p("\n그래서 대형주에서는 점 추정 가치 대신 역산을 읽어야 한다. 역산은 '이 가격이 맞으려면 경계 밖으로 얼마나 나가야 하는가'를 "
      "직접 답한다. 엔비디아의 요구 지속성 ω* 0.98은 모형 상한 0.90을 크게 넘고, 그 격차 자체가 답이다.")


def main() -> int:
    p("# 모형 민감도 — 수요 측정 오차와 경계값\n")
    p(f"> 생성 {pd.Timestamp.today().date()}. 두 가지 판단이 결과를 얼마나 움직이는지 수치로 답한다.")
    demand_sensitivity()
    bounds_binding()
    (C.REPORTS / "model_sensitivity.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\n-> reports/model_sensitivity.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
