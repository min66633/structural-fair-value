# -*- coding: utf-8 -*-
"""Robustness of the residual-net-of-backlog return predictor (phase 5).

Horizons, benchmarks in the same subsample, quarterly coefficient series,
subperiods, decile portfolios (VW and EW), delisting sensitivity, decile
monotonicity and characteristics. Writes reports/phase5_backlog_robustness[_label].md.
--label reads validation_panel_<label>.parquet (e.g. "mom": residual net of momentum)
and writes backlog_panel_<label>.parquet.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sfv import config as C  # noqa: E402
from sfv.validate import fm, decile_ls, CONTROLS  # noqa: E402

pd.set_option("display.width", 220)
XS = ["rpo_g", "log_rpo_cov", "d_rpo_cov", "dfr_g"]
L = []


def p(t=""):
    L.append(t)


def build(sfx: str = ""):
    V = pd.read_parquet(C.PQ / f"validation_panel{sfx}.parquet")
    V = V[V["qend"] >= "2018-03-31"]
    B = pd.read_parquet(C.PQ / "backlog_quarterly.parquet").dropna(subset=["avail"]).sort_values("avail")
    X = pd.merge_asof(V.sort_values("qend"), B.rename(columns={"end": "rpo_end"}), left_on="qend", right_on="avail", by="cik", direction="backward")
    X = X[(X["qend"] - X["rpo_end"]).dt.days <= 200]
    X["log_rpo_cov"] = np.log(X["rpo_to_rev"]).where(X["rpo_to_rev"] > 0)
    XR = X[X["rpo"].notna()].copy()
    parts = []
    for q, dq in XR.groupby("qend"):
        dq = dq.dropna(subset=["eps_A"] + XS)
        if len(dq) < 150:
            continue
        Xm = np.column_stack([np.ones(len(dq)), dq[XS].values.astype(float)])
        b = np.linalg.lstsq(Xm, dq["eps_A"].values, rcond=None)[0]
        fit = Xm @ b - b[0]
        parts.append(pd.DataFrame({"cik": dq["cik"], "qend": q, "eps_exp": fit, "eps_res": dq["eps_A"].values - fit}))
    return XR.merge(pd.concat(parts), on=["cik", "qend"], how="inner")


def decile_table(P: pd.DataFrame, score: str, ret: str) -> pd.DataFrame:
    rows = []
    for q, d in P.groupby("qend"):
        d = d.dropna(subset=[score, ret, "mktcap"])
        if len(d) < 200:
            continue
        d = d.assign(dec=pd.qcut(d[score].rank(method="first"), 10, labels=False))
        r = np.expm1(d[ret])
        ew = r.groupby(d["dec"]).mean()
        vw = r.groupby(d["dec"]).apply(lambda s: np.average(s, weights=d.loc[s.index, "mktcap"]))
        rows.append(pd.concat([ew.rename("ew"), vw.rename("vw")], axis=1).assign(qend=q))
    T = pd.concat(rows)
    out = T.groupby(level=0).agg(ew_mean=("ew", "mean"), vw_mean=("vw", "mean"), n_q=("ew", "size"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="", help="validation panel suffix (e.g. mom)")
    a = ap.parse_args()
    sfx = f"_{a.label}" if a.label else ""
    XR = build(sfx)
    XR.to_parquet(C.PQ / f"backlog_panel{sfx}.parquet", index=False)
    p("# 단계 5 보충 — 잔차(백로그 제거) 수익률 예측의 강건성" + (f" [{a.label}]" if a.label else "") + "\n")
    p(f"> RPO 공시 기업 표본, 기업-분기 {len(XR):,}, 분기 {XR.qend.nunique()} ({XR.qend.min().date()}~{XR.qend.max().date()}). "
      f"eps_exp = 잔차 중 백로그 변수로 설명되는 부분, eps_res = 나머지. 입력 패널: `validation_panel{sfx}.parquet`"
      + (" (잔차 ε은 모멘텀 성분을 걷어낸 사후 스펙)." if a.label == "mom" else "."))
    p("\n## 1. 지평별 FM 계수 (같은 표본의 벤치마크와 비교, NW t)\n")
    rows = {}
    for h in (1, 2, 4):
        ra, _ = fm(XR, f"r{h}", ["eps_A"] + CONTROLS, h, min_n=150)
        rr, T = fm(XR, f"r{h}", ["eps_res", "eps_exp"] + CONTROLS, h, min_n=150)
        rb, _ = fm(XR, f"r{h}", ["bg_mis"] + CONTROLS, h, min_n=150)
        rm, _ = fm(XR, f"r{h}", ["log_pb_adj"] + CONTROLS, h, min_n=150)
        rows[f"h={h}"] = {"ε 전체": f"{ra.loc['eps_A','coef']:.3f} ({ra.loc['eps_A','t']:.2f})",
                          "ε 나머지": f"{rr.loc['eps_res','coef']:.3f} ({rr.loc['eps_res','t']:.2f})",
                          "ε 백로그 설명분": f"{rr.loc['eps_exp','coef']:.3f} ({rr.loc['eps_exp','t']:.2f})",
                          "Bartram·Grinblatt": f"{rb.loc['bg_mis','coef']:.3f} ({rb.loc['bg_mis','t']:.2f})",
                          "log(P/B_adj)": f"{rm.loc['log_pb_adj','coef']:.3f} ({rm.loc['log_pb_adj','t']:.2f})"}
        if h == 4:
            Tq = T["eps_res"]
    p(pd.DataFrame(rows).T.to_markdown())
    p("\n4분기 지평의 분기별 eps_res 계수 (음수 = 저평가 잔차가 더 높은 수익):\n")
    p(Tq.round(3).to_frame("coef").T.to_markdown())
    p(f"\n음수 분기 {int((Tq < 0).sum())} / {len(Tq)}.")
    sub_rows = {}
    for lo, hi, lab in (("2018-01-01", "2021-12-31", "2018-21"), ("2022-01-01", "2026-12-31", "2022-26")):
        sub = XR[(XR["qend"] >= lo) & (XR["qend"] <= hi)]
        r_, _ = fm(sub, "r4", ["eps_res", "eps_exp"] + CONTROLS, 4, min_n=150)
        sub_rows[lab] = {"coef": r_.loc["eps_res", "coef"], "t": r_.loc["eps_res", "t"], "quarters": int(r_.loc["eps_res", "quarters"])}
    r_dl, _ = fm(XR, "r4_dl", ["eps_res", "eps_exp"] + CONTROLS, 4, min_n=150)
    sub_rows["상장폐지 −30% 부과"] = {"coef": r_dl.loc["eps_res", "coef"], "t": r_dl.loc["eps_res", "t"], "quarters": int(r_dl.loc["eps_res", "quarters"])}
    p("\n하위 기간과 상장폐지 민감도 (h=4, eps_res):\n")
    p(pd.DataFrame(sub_rows).T.round(3).to_markdown())

    p("\n## 2. 10분위 수익률 (0 = 잔차 최저 = 가장 저평가)\n")
    for score, lab in (("eps_res", "ε 나머지"), ("eps_A", "ε 전체")):
        for ret, rl in (("r1", "1분기"), ("r4", "4분기")):
            t = decile_table(XR, score, ret)
            p(f"\n{lab}, {rl} 수익률 (단순수익률 평균, 분기 {int(t.n_q.iloc[0])}개):\n")
            tt = t[["ew_mean", "vw_mean"]].T
            tt.columns = [f"D{i}" for i in tt.columns]
            tt["D0−D9"] = tt["D0"] - tt["D9"]
            p(tt.round(4).to_markdown())
    prow = {}
    for nm, col, lo_long, vw in (("ε 나머지 시총가중", "eps_res", True, True), ("ε 나머지 동일가중", "eps_res", True, False),
                                  ("ε 전체 시총가중", "eps_A", True, True), ("−BG 시총가중", "bg_mis", False, True), ("log(P/B_adj) 시총가중", "log_pb_adj", True, True)):
        st, _ = decile_ls(XR, col, "r1", vw, lo_long)
        prow[nm] = {"quarters": st["quarters"], "mean_q": st["mean_q"], "sharpe_ann": st["sharpe_ann"], "t": st["t"]}
    p("\n분기 리밸런싱 D0−D9 롱숏:\n")
    p(pd.DataFrame(prow).T.round(3).to_markdown())

    d = XR.dropna(subset=["eps_res", "r4"]).copy()
    d["dec"] = d.groupby("qend")["eps_res"].transform(lambda s: pd.qcut(s.rank(method="first"), 10, labels=False))
    ch = d.groupby("dec").agg(mktcap_med_bn=("mktcap", lambda s: s.median() / 1e9), loss_share=("loss", "mean"), roe_adj=("roe_adj_w", "mean"),
                              sales_g=("sales_g", "mean"), log_pv=("y", "mean"))
    p("\n분위별 특성 (ε 나머지):\n")
    p(ch.round(3).to_markdown())
    p("\n## 3. 읽기\n")
    p("- 회귀(동일가중 관계)와 동일가중 분위는 일치한다: 계약형 기업 중 잔차가 낮은 기업이 이듬해 더 높은 수익을 냈고, 하위 기간·상장폐지 처리에 강건하다.")
    p("- 시총가중 분위는 반대다. 최고 잔차 분위의 대형 승자들이 이 기간 계속 앞섰다. 이 신호는 지수 수준의 판단이 아니라 중소형 계약형 기업의 선별에서만 작동한다.")
    p("- 사전 등록 검정이 실패한 뒤 나온 탐색적 결과다. 표본 선택(RPO 공시 기업)은 데이터 가용성 때문이지 결과를 보고 고른 것은 아니지만, 확정하려면 이후 분기의 표본외 확인이 필요하다.")
    (C.REPORTS / f"phase5_backlog_robustness{sfx}.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
