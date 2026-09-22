# -*- coding: utf-8 -*-
"""Decile-by-decile returns for eps_res / eps_A in the RPO subsample: is the relation monotone?"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sfv import config as C  # noqa: E402

pd.set_option("display.width", 220)


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
    out["ew_t"] = T.groupby(level=0)["ew"].apply(lambda s: s.mean() / (s.std() / np.sqrt(len(s))))
    return out


def main():
    XR = pd.read_parquet(C.PQ / "backlog_panel.parquet")
    for score in ("eps_res", "eps_A"):
        for ret in ("r1", "r4"):
            t = decile_table(XR, score, ret)
            print(f"\n== {score} deciles (0 = lowest = most undervalued), returns {ret}, quarters {int(t.n_q.iloc[0])}")
            print(t.round(4).to_string())
            print("   spread D0-D9: ew %.4f  vw %.4f" % (t.ew_mean.iloc[0] - t.ew_mean.iloc[-1], t.vw_mean.iloc[0] - t.vw_mean.iloc[-1]))
    # characteristics of the extreme deciles
    d = XR.dropna(subset=["eps_res", "r4"]).copy()
    d["dec"] = d.groupby("qend")["eps_res"].transform(lambda s: pd.qcut(s.rank(method="first"), 10, labels=False))
    ch = d.groupby("dec").agg(mktcap_med=("mktcap", "median"), loss=("loss", "mean"), roe=("roe_adj_w", "mean"),
                              sales_g=("sales_g", "mean"), y=("y", "mean"), exit_share=("r4", lambda s: s.isna().mean()))
    print("\ncharacteristics by eps_res decile:")
    print(ch.round(3).to_string())


if __name__ == "__main__":
    main()
