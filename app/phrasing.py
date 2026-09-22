# -*- coding: utf-8 -*-
"""The numbers said in plain Korean.

The screen used to show nine figures at the same weight and let the reader work out which two the decision
rests on. These functions turn the same figures into sentences with no symbols in them, so the top of a tab
answers "what does this price assume, is that a common thing, and what do I get if I disagree" before the
tables start. Used by the 전제 tab's summary and by the walkthrough in 도움말."""
from __future__ import annotations

import html

import numpy as np
import pandas as pd

from sfv.calc import PAST_GROUPS
from sfv.store import Store
from app.theme import C
from app.widgets import isnum

TOP_N = 200            # the size split the conditional base-rate table uses
FUND_LAG_DAYS = 150    # fundamentals older than this: the SEC record is behind the filing


def base_rate_facts(store: Store, row: pd.DataFrame, q: dict) -> dict:
    """What the base-rate tables say about this firm's required-growth bucket: overall, among firms its size,
    and among firms that grew as fast as it did last year. Every part may be missing."""
    x = row.iloc[0]
    out = {"bucket": q["bucket"], "overall": None, "size": None, "past": None,
           "p_achieve": float(x["p_achieve_3y"]) if isnum(x.get("p_achieve_3y", np.nan)) else np.nan}
    if q["bucket"] is None:
        return out
    B = store.base_rates.get("bucket", pd.DataFrame())
    if len(B) and "bucket" in B.columns:
        r = B[B["bucket"] == q["bucket"]]
        if len(r):
            r = r.iloc[0]
            out["overall"] = {"rate": r.get("3y 달성 비율", np.nan), "n": r.get("기업-분기 (3y)", np.nan),
                              "median": r.get("실현 3y 연환산 중앙값 %", np.nan)}
    Cd = store.base_rates.get("cond", pd.DataFrame())
    if len(Cd) and "dim" in Cd.columns:
        rank = int((store.firms["mktcap"] > float(x["mktcap"])).sum()) + 1
        size_grp = "시총 상위 200" if rank <= TOP_N else "201위 이하"
        pg = float(np.expm1(x["b1"])) if isnum(x.get("b1", np.nan)) else np.nan
        past_grp = next((nm for lo, hi, nm in PAST_GROUPS if lo <= pg < hi), None) if isnum(pg) else None
        for key, dim, grp in (("size", "규모", size_grp), ("past", "직전 성장", past_grp)):
            if grp is None:
                continue
            r = Cd[(Cd["dim"] == dim) & (Cd["group"] == grp) & (Cd["bucket"] == q["bucket"])]
            if len(r):
                out[key] = {"group": grp, "rate": float(r["rate"].iloc[0]), "n": int(r["n"].iloc[0])}
        out["rank"], out["past_growth"] = rank, pg
    return out


def per_hundred(rate: float) -> str:
    """A share as a count out of a hundred: '100곳 중 9곳'. Easier to hold than a percent, and it keeps the
    reader from reading a base rate as this company's probability."""
    return f"100곳 중 {round(float(rate) * 100)}곳"


def two_views(q: dict) -> list[tuple[float, float]]:
    """Two growth views to quote, the two highest that are still below what the price requires: the region
    where the reader is less optimistic than the market, which is where the number is worth knowing."""
    pairs = [(g, r) for g, r in zip(q["views"], q["returns"]) if isnum(r)]
    if not pairs:
        return []
    below = [p for p in pairs if not isnum(q["g_star"]) or p[0] < q["g_star"]]
    picked = (below[-2:] if len(below) >= 2 else pairs[:2])
    return list(reversed(picked)) if len(picked) == 2 else picked


def summary_sentences(store: Store, row: pd.DataFrame, q: dict) -> list[str]:
    """Three or four sentences that answer the whole tab. HTML, because the numbers are bolded."""
    x = row.iloc[0]
    name = html.escape(str(x["name"]))
    out = []

    g, b = q["g_star"], q["band"]
    if isnum(g):
        s = f"오늘 가격은 <b>{name}</b>의 이익이 앞으로 10년 동안 <b>매년 {g*100:.1f}%</b>씩 자란다고 전제합니다."
        if b is not None and isnum(b["g_lo"]):
            s += (f" 회계 가정을 문헌의 다른 조합으로 바꿔도 {b['g_lo']*100:.1f}~{b['g_hi']*100:.1f}% 사이입니다."
                  if b["width_pp"] <= 5.0 else
                  f" 다만 회계 가정에 따라 {b['g_lo']*100:.1f}~{b['g_hi']*100:.1f}%로 넓게 움직이니, 점 하나가 아니라 구간으로 읽으세요.")
        out.append(s)
    elif not (isnum(q["E0"]) and q["E0"] > 0):
        out.append(f"<b>{name}</b>는 지금 적자입니다. 그래서 성장률 하나로는 가격의 전제를 읽을 수 없습니다. "
                   "적자 기업의 가격은 <b>도달할 규모</b>와 <b>벌게 될 이익률</b>을 동시에 전제하는데, "
                   "가격 하나로 두 가지를 동시에 알아낼 수는 없기 때문입니다.")
        rev0 = float(x["rev0"]) if isnum(x.get("rev0", np.nan)) else np.nan
        if isnum(rev0) and rev0 > 0:
            out.append(f"지금 매출은 <b>{rev0/1e9:,.1f}십억</b>이고 조정 이익률은 <b>{q['E0']/rev0*100:+.0f}%</b>입니다. "
                       "목표주가 탭에서 몇 년 뒤 이익률을 정하고 '요구 성장 풀기'를 누르면, 그 이익률에서 가격이 요구하는 "
                       "매출 성장이 나옵니다. 그 숫자를 믿을 수 있는지가 판단입니다.")
    else:
        out.append(f"<b>{name}</b>의 가격은 연 200% 성장으로도 설명되지 않습니다. 목표주가 탭의 풀기 기능으로 읽으세요.")

    f = base_rate_facts(store, row, q)
    if f["overall"] is not None and isnum(f["overall"]["rate"]):
        s = f"그만큼을 요구받은 기업이 이후 3년간 실제로 그 속도를 낸 비율은 <b>{per_hundred(f['overall']['rate'])}</b>입니다."
        if f["past"] is not None and isnum(f["past"]["rate"]):
            s += (f" 다만 이 회사처럼 직전 1년에 {f['past_growth']*100:+.0f}% 자란 무리에서는 "
                  f"<b>{per_hundred(f['past']['rate'])}</b>입니다.")
        out.append(s)

    views = two_views(q)
    if len(views) == 2:
        (ga, ra), (gb, rb) = views
        out.append(f"연 {ga*100:.0f}% 성장을 보신다면 이 가격은 <b>연 {ra*100:.1f}%</b>를 줍니다. "
                   f"{gb*100:.0f}%로 보면 {rb*100:.1f}%입니다.")

    if q["phase"] in ("피크", "저점") and isnum(q["g_norm"]):
        high = q["phase"] == "피크"
        out.append(f"지금 이익이 5년 평균 이익률로 잰 것보다 {'크게 높습니다' if high else '크게 낮습니다'}({q['phase']}). "
                   f"평균 이익률을 쓰면 요구 성장은 <b>{q['g_norm']*100:.1f}%</b>가 됩니다. 어느 쪽이 이 회사의 평상시인지는 판단의 몫입니다.")

    age = (store.month - pd.Timestamp(x["f_end"])).days
    if int(x["cik"]) > 0 and age > FUND_LAG_DAYS:
        out.append(f"실적 기준일이 {age}일 전입니다. SEC 기록이 늦어 최근 분기가 빠져 있을 수 있으니, 그만큼 오래된 이익으로 잰 값입니다.")
    return out


def summary_html(store: Store, row: pd.DataFrame, q: dict) -> str:
    body = "".join(f'<p style="margin:0 0 9px 0;">{s}</p>' for s in summary_sentences(store, row, q))
    return f'<div style="color:{C["text"]};">{body}</div>'
