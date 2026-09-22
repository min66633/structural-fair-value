# -*- coding: utf-8 -*-
"""Why does the residual signal reverse when value-weighted? (phase 3 supplement)

Diagnoses the value-weighted reversal of the decile spread on the residual:
weighting variants (equal, cap, 5% single-name cap, sqrt-cap, excluding the
five largest names per decile, excluding > $200bn), attribution of the top
decile's cap-weighted return to single names, size terciles, size-neutral
ranks and the top-50/100 subsets. Run on the RPO subsample (residual net of
backlog and momentum, backlog_panel_mom) and the full sample (residual net of
momentum, validation_panel_mom). Writes reports/phase3_vw_reversal.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sfv import config as C  # noqa: E402
from sfv.validate import fm, nw_t  # noqa: E402

L = []


def p(t=""):
    L.append(t)


def weights(d: pd.DataFrame, kind: str) -> pd.Series:
    if kind == "ew":
        return pd.Series(1.0, index=d.index)
    w = d["mktcap"].astype(float).copy()
    if kind == "sqrt":
        return np.sqrt(w)
    if kind == "cap5":
        for _ in range(20):
            w = w / w.sum()
            w = w.clip(upper=0.05)
        return w
    return w


def ls(P, score, ret, kind, nb=10, min_n=200, exclude_top=0):
    rows = []
    for q, d in P.groupby("qend"):
        d = d.dropna(subset=[score, ret, "mktcap"])
        if len(d) < min_n:
            continue
        b = pd.qcut(d[score].rank(method="first"), nb, labels=False)
        r = np.expm1(d[ret])
        out = {"qend": q}
        for lab, sel in (("lo", b == 0), ("hi", b == nb - 1)):
            dd = d[sel]
            if exclude_top:
                dd = dd.sort_values("mktcap").iloc[:-exclude_top]
            out[lab] = np.average(r[dd.index], weights=weights(dd, kind))
        rows.append(out)
    T = pd.DataFrame(rows).set_index("qend")
    T["ls"] = T["lo"] - T["hi"]
    return {"분기": len(T), "저평가 분위": T["lo"].mean(), "고평가 분위": T["hi"].mean(), "차이": T["ls"].mean(),
            "t": nw_t(T["ls"].values, 4 if ret == "r4" else 1), "양수 분기 비율": (T["ls"] > 0).mean()}


def attribution(P, score, ret, top=True, k=8, nb=10):
    rows = []
    for q, d in P.groupby("qend"):
        d = d.dropna(subset=[score, ret, "mktcap"])
        if len(d) < 200:
            continue
        b = pd.qcut(d[score].rank(method="first"), nb, labels=False)
        dd = d[b == (nb - 1 if top else 0)].copy()
        dd["w"] = dd["mktcap"] / dd["mktcap"].sum()
        dd["contrib"] = dd["w"] * np.expm1(dd[ret])
        rows.append(dd[["qend", "ticker", "w", "contrib", ret]])
    A = pd.concat(rows)
    nq = A["qend"].nunique()
    g = A.groupby("ticker").agg(분기수=("qend", "size"), 평균비중=("w", "mean"), 분기당기여=("contrib", lambda s: s.sum() / nq),
                                평균수익률=(ret, lambda s: np.expm1(s).mean()))
    return g.sort_values("분기당기여", ascending=not top).head(k), nq


def by_size(P, score, ret, nt=3, nb=5):
    res = {}
    P = P.dropna(subset=[score, ret, "mktcap"]).copy()
    P["ter"] = P.groupby("qend")["mktcap"].transform(lambda s: pd.qcut(s.rank(method="first"), nt, labels=False))
    names = {0: "소형", 1: "중형", 2: "대형"}
    for t in range(nt):
        S = P[P["ter"] == t]
        e = ls(S, score, ret, "ew", nb, min_n=60); v = ls(S, score, ret, "vw", nb, min_n=60)
        f, _ = fm(S, ret, [score, "log_size", "mom"], 4 if ret == "r4" else 1, min_n=60)
        res[f"{names[t]} (중앙값 시총 ${S['mktcap'].median()/1e9:.1f}bn)"] = {
            "동일가중 Q1−Q5": e["차이"], "t (EW)": e["t"], "시총가중 Q1−Q5": v["차이"], "t (VW)": v["t"], "FM 계수": f.loc[score, "coef"], "FM t": f.loc[score, "t"]}
    return pd.DataFrame(res).T


def size_neutral(P, score, ret, nt=5):
    P = P.dropna(subset=[score, ret, "mktcap"]).copy()
    P["ter"] = P.groupby("qend")["mktcap"].transform(lambda s: pd.qcut(s.rank(method="first"), nt, labels=False))
    P["sn"] = P.groupby(["qend", "ter"])[score].rank(pct=True)
    return pd.DataFrame({k: ls(P, "sn", ret, kk) for k, kk in (("동일가중", "ew"), ("시총가중", "vw"), ("시총가중, 상한 5%", "cap5"))}).T


def mega(P, score, ret, n=50):
    rows = []
    for q, d in P.groupby("qend"):
        d = d.dropna(subset=[score, ret, "mktcap"]).sort_values("mktcap", ascending=False).head(n)
        if len(d) < n * 0.8:
            continue
        lo = d[score] <= d[score].quantile(.2); hi = d[score] >= d[score].quantile(.8)
        rows.append({"qend": q, "rho": spearmanr(d[score], d[ret])[0], "q1_q5": np.expm1(d[ret][lo]).mean() - np.expm1(d[ret][hi]).mean()})
    T = pd.DataFrame(rows).set_index("qend")
    return {"분기": len(T), "순위상관 평균": T["rho"].mean(), "순위상관 > 0 비율": (T["rho"] > 0).mean(), "순위상관 t": nw_t(T["rho"].values, 4),
            "Q1−Q5 동일가중": T["q1_q5"].mean(), "Q1−Q5 t": nw_t(T["q1_q5"].values, 4)}


def run(P, score, label):
    p(f"\n## {label}\n")
    p(f"점수 {score}, 4분기 수익률, {P.qend.min().date()}~{P.qend.max().date()}, 기업-분기 {len(P):,}.\n")
    tab = {}
    for kind, nm in (("ew", "동일가중"), ("vw", "시총가중"), ("cap5", "시총가중, 종목 상한 5%"), ("sqrt", "sqrt(시총) 가중")):
        tab[nm] = ls(P, score, "r4", kind)
    tab["시총가중, 분위별 최대 5종목 제외"] = ls(P, score, "r4", "vw", exclude_top=5)
    tab["시총가중, 시총 2,000억$ 초과 제외"] = ls(P[P["mktcap"] < 200e9], score, "r4", "vw")
    p("### 1. 10분위 D0(저평가) − D9(고평가), 4분기 단순수익률 평균\n")
    p(pd.DataFrame(tab).T.round(3).to_markdown())
    a_hi, nq = attribution(P, score, "r4", top=True)
    a_lo, _ = attribution(P, score, "r4", top=False)
    p(f"\n### 2. 시총가중 수익률 기여 (분기당 기여 = 분위 평균수익률에 대한 기여, {nq}분기 평균)\n")
    p("고평가 분위 D9를 끌어올린 종목:\n")
    p(a_hi.round(3).to_markdown())
    p("\n저평가 분위 D0를 끌어내린 종목:\n")
    p(a_lo.round(3).to_markdown())
    p("\n### 3. 규모 3분위 내부 (분기별 시총 3분위, 그 안에서 5분위; FM 통제 = 로그 시총, 모멘텀)\n")
    p(by_size(P, score, "r4").round(3).to_markdown())
    p("\n### 4. 규모 중립 점수 (분기·시총 5분위 내부 백분위 순위) 10분위\n")
    p(size_neutral(P, score, "r4").round(3).to_markdown())
    p("\n### 5. 시총 상위 50·100 안에서만\n")
    p(pd.DataFrame({"상위 50": mega(P, score, "r4", 50), "상위 100": mega(P, score, "r4", 100)}).T.round(3).to_markdown())


def main():
    cols = ["cik", "qend", "ticker", "mktcap", "r4", "r1", "log_size", "mom", "eps_A", "y"]
    p("# 단계 3 보충 — 시총가중 역전의 정체\n")
    p(f"> 생성 {pd.Timestamp.today().date()}. 잔차를 10분위로 나눴을 때 동일가중에서는 저평가 분위가 앞서고 시총가중에서는 뒤지는 현상의 원인 진단. "
      "가중 방식 변형, 종목별 기여 귀속, 규모 3분위, 규모 중립 순위, 초대형주 부분집합. 수익률은 13F 내재가격 4분기 수익률(상장폐지 포함), NW t는 4시차.")
    R = pd.read_parquet(C.PQ / "backlog_panel_mom.parquet", columns=cols + ["eps_res"])
    run(R, "eps_res", "RPO 공시 기업 표본 — 백로그·모멘텀 제거 잔차 (validation_panel_mom → backlog_panel_mom)")
    V = pd.read_parquet(C.PQ / "validation_panel_mom.parquet", columns=cols)
    run(V, "eps_A", "전체 표본 — 모멘텀 제거 잔차 (validation_panel_mom)")
    p("\n## 읽기\n")
    p("- 역전은 '대형주 전반에서 신호가 뒤집힘'이 아니라 '초대형 승자 몇 종목이 고평가 분위의 시총가중 수익률을 끌어올림'이다. RPO 표본에서는 엔비디아 한 종목이 D9 수익률의 대부분을 만들었고, 전체 표본에서는 아마존·테슬라·엔비디아·마스터카드다.")
    p("- 소형·중형 3분위에서는 가중 방식과 무관하게 작동하고, 대형 3분위에서만 시총가중이 반대다. 시총 상위 50 안에서는 잔차와 수익률의 순위상관이 0으로 방향 자체가 없다.")
    p("- 규모 중립 순위로는 고쳐지지 않는다(초대형주끼리 비교해도 같은 종목이 가장 비싸고 가장 많이 올랐다). 종목 비중 상한 5%나 초대형주 제외 같은 포트폴리오 규칙으로는 시총가중에서도 양의 스프레드가 나오지만, 결과를 보고 고른 규칙이라 t값은 과대평가돼 있다.")
    p("- 모형으로는 고칠 수 없다. 이후 실현된 이익 폭발은 어떤 시점 t의 재무제표에도 없던 정보다. 도구에서는 초대형주(시총 상위 50 또는 2,000억$ 초과)의 횡단면 판정을 '검증된 효력 없음'으로 표시하고, 그 구간은 단계 7의 요구 기대(ω*, T*, g*)로 읽는다.")
    (C.REPORTS / "phase3_vw_reversal.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
