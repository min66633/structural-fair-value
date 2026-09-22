# -*- coding: utf-8 -*-
"""The tool's user-facing output: one decision table per stock, as CSV and HTML.

Everything upstream produces evidence; this produces the thing you actually
read before a trade. For the latest valuation month it joins

    what the price assumes      g*_10y, omega* and its half-life, T*   (sfv.expectations)
    what fundamentals justify   g_model, log(V_gr/V_F)                 (sfv.growth_rim --backlog)
    how it sits against peers   eps from the momentum+expectation
                                decomposition, and its percentile      (sfv.decompose)
    what the market as a whole
    assumes                     omega*, T*, aggregate premium          (sfv.implied)
    whether to trust any of it  base rates for the required growth,
                                the pre-registered verdict, the
                                mega-cap exclusion, data-quality status

and flags the rows where the cross-sectional reading has no validated edge
(mega caps) or where the model cannot price the firm at all (loss makers,
firms whose price needs profit growth rather than persistence).

The HTML is self-contained: no network, no CDN, sortable and filterable with
plain JavaScript, meant to be opened from disk.

Outputs reports/sfv_latest.csv, reports/sfv_report.html
Run: python -m sfv.report [--month YYYY-MM-DD] [--top 0]
"""
from __future__ import annotations

import argparse
import html
import json
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import config as C

MEGA_CAP_USD = 200e9          # above this the cross-sectional residual has no validated edge
MEGA_CAP_RANK = 50            # ... and so does the top-50 by market cap (phase3_vw_reversal)
COLS = [("ticker", "종목"), ("name", "이름"), ("ff12", "산업"), ("mktcap_bn", "시총 $bn"), ("roe_adj", "조정 ROE"),
        ("log_pv", "log(P/V_F)"), ("log_vgr_vf", "기대분"), ("log_p_vgr", "기대 초과분"),
        ("g_model_pct", "g_model %/y"), ("g_star_10y_pct", "g* 10년 %/y"), ("g_star_lo_pct", "g* 하한"),
        ("g_star_hi_pct", "g* 상한"), ("g_star_erpavg_pct", "g* 평균ERP"), ("g_star_norm_pct", "g* 정규화"),
        ("p_achieve_3y_pct", "3년 달성 확률 %"),
        ("r_model_pct", "모형 할인율 %"), ("r_star_g10_pct", "수익률 | 성장 10%"), ("r_star_g15_pct", "수익률 | 성장 15%"),
        ("r_star_g20_pct", "수익률 | 성장 20%"), ("r_star_g25_pct", "수익률 | 성장 25%"),
        ("omega_star", "ω*"), ("half_life", "반감기 y"), ("T_star", "T* y"), ("eps", "잔차 ε"), ("eps_pct", "ε 백분위"), ("flags", "표시")]
DECIMALS = {"mktcap_bn": 1, "roe_adj": 2, "log_pv": 2, "log_vgr_vf": 2, "log_p_vgr": 2, "g_model_pct": 1, "g_star_10y_pct": 1,
            "g_star_lo_pct": 1, "g_star_hi_pct": 1, "g_star_erpavg_pct": 1, "g_star_norm_pct": 1, "p_achieve_3y_pct": 0, "r_model_pct": 1,
            "r_star_g10_pct": 1, "r_star_g15_pct": 1, "r_star_g20_pct": 1, "r_star_g25_pct": 1,
            "omega_star": 2, "half_life": 1, "T_star": 0, "eps": 2, "eps_pct": 2}
BAND_COLS = {"g_star_lo_pct", "g_star_hi_pct"}


def load_inputs(month: pd.Timestamp | None, say):
    E = pd.read_parquet(C.PQ / "expectations_monthly.parquet")
    m = month if month is not None else E["month"].max()
    if m not in set(E["month"]):
        raise SystemExit(f"month {m.date()} not in expectations_monthly (latest {E['month'].max().date()})")
    X = E[E["month"] == m].copy()
    say(f"  valuation month {m.date()}: {len(X):,} firms")

    # residual: the decomposition lags by the 13F deadline, so take each firm's
    # latest available residual and record how old it is
    eps_month = None
    if (C.PQ / "decomp_monthly_momexp.parquet").exists():
        D = pd.read_parquet(C.PQ / "decomp_monthly_momexp.parquet", columns=["cik", "month", "eps_A"]).dropna(subset=["eps_A"])
        D = D[D["month"] <= m]
        D["eps_pct"] = D.groupby("month")["eps_A"].rank(pct=True)
        last = D.sort_values("month").groupby("cik").tail(1).rename(columns={"month": "eps_month", "eps_A": "eps"})
        X = X.merge(last[["cik", "eps_month", "eps", "eps_pct"]], on="cik", how="left")
        stale = (m - X["eps_month"]).dt.days > 200
        X.loc[stale, ["eps", "eps_pct", "eps_month"]] = np.nan
        eps_month = X["eps_month"].max()
        say(f"  residual from {eps_month.date() if pd.notna(eps_month) else 'n/a'} (13F filing lag), present for {X['eps'].notna().mean()*100:.0f}% of firms")
    else:
        X["eps"] = np.nan; X["eps_pct"] = np.nan; X["eps_month"] = pd.NaT

    # backlog flag
    if (C.PQ / "backlog_quarterly.parquet").exists():
        B = pd.read_parquet(C.PQ / "backlog_quarterly.parquet", columns=["cik", "end", "avail", "rpo", "rpo_to_rev"])
        B = B.dropna(subset=["avail", "rpo"]).sort_values("avail")
        B = B[B["avail"] <= m].groupby("cik").tail(1)
        X = X.merge(B[["cik", "rpo_to_rev"]], on="cik", how="left")
    else:
        X["rpo_to_rev"] = np.nan

    # parameter band: the same stock valued under every credible capitalisation
    # parameter set (scripts/intangible_sensitivity.py). The point value moves a
    # lot with those choices; the required growth barely does, and showing both
    # bounds is what keeps the reader from over-reading a single number.
    band_note = ""
    if (C.PQ / "expectations_band.parquet").exists():
        B = pd.read_parquet(C.PQ / "expectations_band.parquet")
        bm = B["month"].iloc[0] if "month" in B.columns and len(B) else None
        if bm is not None and bm == m:
            X = X.merge(B[["cik", "g_star_lo", "g_star_hi", "g_star_n", "log_pv_lo", "log_pv_hi"]], on="cik", how="left")
            n = int(X["g_star_lo"].notna().sum())
            say(f"  parameter band attached for {n:,} firms ({int(X['g_star_n'].max())} parameter sets)")
            band_note = "있음"
        else:
            say(f"  parameter band is for {bm.date() if bm is not None else 'n/a'}, not {m.date()} - rerun scripts/intangible_sensitivity.py")
    if "g_star_lo" not in X.columns:
        X["g_star_lo"] = np.nan; X["g_star_hi"] = np.nan; X["log_pv_lo"] = np.nan; X["log_pv_hi"] = np.nan

    market = {}
    if (C.PQ / "implied_monthly.parquet").exists():
        M = pd.read_parquet(C.PQ / "implied_monthly.parquet")
        M.index = pd.to_datetime(M.index)
        avail = M.index[M.index <= m]
        if len(avail):
            r = M.loc[avail.max()]
            market = {"month": str(avail.max().date()), "omega_star": float(r["omega_star"]), "cap_star": float(r["cap_star"]),
                      "log_pv": float(r["log_pv"]), "sum_P_tn": float(r["sum_P_tn"]), "sum_V_tn": float(r["sum_V_tn"]),
                      "r_vw": float(r["r_vw"]), "omega_vw": float(r["omega_vw"]), "n": int(r["n"])}
    cal = {}
    if (C.PQ / "expectations_calibration.parquet").exists():
        cal["bucket"] = pd.read_parquet(C.PQ / "expectations_calibration.parquet")
    if (C.PQ / "expectations_calibration_cond.parquet").exists():
        cal["cond"] = pd.read_parquet(C.PQ / "expectations_calibration_cond.parquet")
    status = None
    f = C.REPORTS / "selfcheck_status.json"
    if f.exists():
        status = json.loads(f.read_text(encoding="utf-8"))
    return X, m, market, cal, status, eps_month


def build_table(X: pd.DataFrame) -> pd.DataFrame:
    X = X.sort_values("mktcap", ascending=False).reset_index(drop=True)
    X["rank"] = np.arange(1, len(X) + 1)
    X["mktcap_bn"] = X["mktcap"] / 1e9
    X["g_model_pct"] = X["g_model"] * 100
    X["g_star_10y_pct"] = X["g_star_10y"] * 100
    X["g_star_lo_pct"] = X["g_star_lo"] * 100
    X["g_star_hi_pct"] = X["g_star_hi"] * 100
    # the return axis, the ERP reference and the conditional base rate (sfv.expectations); absent on older outputs
    for src, dst in (("g_star_10y_erpavg", "g_star_erpavg_pct"), ("g_star_10y_norm", "g_star_norm_pct"),
                     ("p_achieve_3y", "p_achieve_3y_pct"), ("r", "r_model_pct"),
                     ("r_star_g10", "r_star_g10_pct"), ("r_star_g15", "r_star_g15_pct"), ("r_star_g20", "r_star_g20_pct"),
                     ("r_star_g25", "r_star_g25_pct")):
        X[dst] = X[src] * 100 if src in X.columns else np.nan
    X["is_mega"] = (X["mktcap"] >= MEGA_CAP_USD) | (X["rank"] <= MEGA_CAP_RANK)
    flags = []
    for _, r in X.iterrows():
        f = []
        if r["is_mega"]:
            f.append("초대형")
        if pd.notna(r.get("ttm_e_adj")) and r["ttm_e_adj"] <= 0:
            f.append("적자")
        if r.get("omega_status") == 1:
            f.append("지속불가")
        if bool(r.get("cycle_flag", False)):
            f.append("이익국면")
        if pd.notna(r.get("f_end")) and (r["month"] - r["f_end"]).days > 150:
            f.append("재무지연")           # the SEC facts record lags the firm's filings; valued on an older quarter
        if pd.notna(r.get("rpo_to_rev")):
            f.append("백로그")
        flags.append(" ".join(f))
    X["flags"] = flags
    return X


def write_csv(X: pd.DataFrame, path) -> None:
    extra = ["rpo_to_rev", "eps_month", "v_f", "v_f_gr", "mktcap", "g_star_3y", "omega_status", "is_mega", "log_pv_lo", "log_pv_hi",
             "rf", "beta", "erp", "erp_avg", "r_star_g05", "r_star_g30"]
    out = X[[c for c, _ in COLS] + [c for c in extra if c in X.columns]].copy()
    out.to_csv(path, index=False, encoding="utf-8-sig")


# ----------------------------------------------------------------- HTML
CSS = """
:root{--bg:#fbfaf8;--fg:#1d1b18;--mut:#6b6560;--line:#e2ddd6;--card:#fff;--warn:#8a5a00;--warnbg:#fff6e0;--bad:#8a1c1c;--badbg:#fdecec;--ok:#1f5c37}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI","Malgun Gothic",sans-serif}
.wrap{max-width:1500px;margin:0 auto;padding:28px 20px 80px}
h1{font-size:24px;margin:0 0 4px;letter-spacing:-.01em}
h2{font-size:15px;margin:30px 0 10px;text-transform:none;color:var(--mut);font-weight:600}
.sub{color:var(--mut);margin:0 0 22px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;margin:0 0 18px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 14px}
.card .k{color:var(--mut);font-size:12px}
.card .v{font-size:21px;font-variant-numeric:tabular-nums;margin-top:3px}
.card .n{color:var(--mut);font-size:11px;margin-top:2px}
.note{background:var(--card);border:1px solid var(--line);border-left:3px solid var(--mut);border-radius:6px;padding:12px 14px;margin:0 0 16px}
.note.warn{background:var(--warnbg);border-left-color:var(--warn)}
.note.bad{background:var(--badbg);border-left-color:var(--bad)}
.note b{font-weight:600}
.controls{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:0 0 10px}
input[type=search],select{font:inherit;padding:7px 10px;border:1px solid var(--line);border-radius:6px;background:var(--card);color:var(--fg)}
input[type=search]{min-width:230px}
label.cb{display:inline-flex;align-items:center;gap:6px;color:var(--mut);font-size:13px}
.count{color:var(--mut);font-size:13px;margin-left:auto}
.tablebox{overflow:auto;max-height:72vh;border:1px solid var(--line);border-radius:8px;background:var(--card)}
table{border-collapse:separate;border-spacing:0;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:6px 10px;text-align:right;white-space:nowrap;border-bottom:1px solid var(--line)}
th:nth-child(-n+3),td:nth-child(-n+3){text-align:left}
td:nth-child(2){max-width:230px;overflow:hidden;text-overflow:ellipsis}
thead th{position:sticky;top:0;background:var(--card);cursor:pointer;user-select:none;font-weight:600;border-bottom:1px solid var(--fg);z-index:1}
thead th:hover{color:var(--mut)}
tbody tr:hover{background:#f4f1ec}
tr.mega td{color:#4a443f}
.tag{display:inline-block;font-size:11px;padding:1px 6px;border-radius:999px;border:1px solid var(--line);color:var(--mut);margin-right:3px}
.tag.mega{background:#efeae2}
.tag.loss{background:var(--badbg);border-color:#eccaca;color:var(--bad)}
.tag.nope{background:var(--warnbg);border-color:#e8d5a8;color:var(--warn)}
.small{color:var(--mut);font-size:12.5px}
td.band{color:var(--mut);font-size:12.5px}
.small table{width:auto;margin-top:6px}
.small th,.small td{border-bottom:1px solid var(--line);padding:4px 10px}
footer{margin-top:34px;color:var(--mut);font-size:12.5px;border-top:1px solid var(--line);padding-top:14px}
"""

JS = """
const F={q:'',ind:'',mega:false,col:3,dir:-1};
// row layout comes from META (built in Python from COLS): text columns 0..2, META.num = [[index, decimals]],
// META.band = indices drawn muted, META.flags = the flags column, META.mega = the is_mega boolean appended last
const fmt=(v,d)=>v===null||v===undefined||Number.isNaN(v)?'':Number(v).toFixed(d);
function tags(s){if(!s)return '';return s.split(' ').filter(Boolean).map(t=>{
 const c=t==='초대형'?'mega':(t==='적자'?'loss':(t==='지속불가'?'nope':''));
 return '<span class="tag '+c+'">'+t+'</span>';}).join('');}
function render(){
 const q=F.q.toLowerCase(),tb=document.getElementById('tb');
 let rows=DATA.filter(r=>(!F.ind||r[2]===F.ind)&&(!F.mega||!r[META.mega])&&(!q||(r[0]||'').toLowerCase().includes(q)||(r[1]||'').toLowerCase().includes(q)));
 const c=F.col,d=F.dir;
 rows.sort((a,b)=>{let x=a[c],y=b[c];
  const xe=(x===null||x===undefined||Number.isNaN(x)),ye=(y===null||y===undefined||Number.isNaN(y));
  if(xe&&ye)return 0; if(xe)return 1; if(ye)return -1;
  if(typeof x==='string')return d*x.localeCompare(y);
  return d*(x-y);});
 tb.innerHTML=rows.map(r=>{
  let td='<td>'+(r[0]||'')+'</td><td>'+(r[1]||'')+'</td><td>'+(r[2]||'')+'</td>';
  for(const [i,dec] of META.num){td+='<td'+(META.band.includes(i)?' class="band"':'')+'>'+fmt(r[i],dec)+'</td>';}
  td+='<td style="text-align:left">'+tags(r[META.flags])+'</td>';
  return '<tr class="'+(r[META.mega]?'mega':'')+'">'+td+'</tr>';}).join('');
 document.getElementById('cnt').textContent=rows.length+' / '+DATA.length+' 종목';
}
function sortBy(i){F.dir=(F.col===i)?-F.dir:(i<3?1:-1);F.col=i;render();}

window.addEventListener('DOMContentLoaded',()=>{
 document.querySelectorAll('thead th').forEach((th,i)=>th.addEventListener('click',()=>sortBy(i)));
 document.getElementById('q').addEventListener('input',e=>{F.q=e.target.value;render();});
 document.getElementById('ind').addEventListener('change',e=>{F.ind=e.target.value;render();});
 document.getElementById('mega').addEventListener('change',e=>{F.mega=e.target.checked;render();});
 render();});
"""


def html_report(X: pd.DataFrame, m: pd.Timestamp, market: dict, cal, status, eps_month, verdict_note: str) -> str:
    cols = [c for c, _ in COLS]
    D = X.reindex(columns=cols + ["is_mega"]).copy()
    data = []
    for _, r in D.iterrows():
        row = []
        for c in cols:
            v = r[c]
            if isinstance(v, str) or c in ("ticker", "name", "ff12", "flags"):
                row.append("" if pd.isna(v) else str(v))
            else:
                row.append(None if pd.isna(v) else round(float(v), 6))
        row.append(bool(r["is_mega"]))
        data.append(row)
    inds = sorted(x for x in X["ff12"].dropna().unique())
    e = html.escape
    meta = {"num": [[i, DECIMALS[c]] for i, c in enumerate(cols) if c in DECIMALS],
            "band": [i for i, c in enumerate(cols) if c in BAND_COLS], "flags": cols.index("flags"), "mega": len(cols)}

    cards = []
    if market:
        cards += [("시장 내재 지속성 ω*", f"{market['omega_star']:.3f}",
                   f"반감기 {np.log(0.5)/np.log(market['omega_star']):.0f}년 · 실현 시총가중 {market['omega_vw']:.2f}" if 0 < market["omega_star"] < 1 else ""),
                  ("내재 경쟁우위 기간 T*", f"{market['cap_star']:.0f}년", "현재 초과 ROE가 유지돼야 하는 기간"),
                  ("시총가중 log(P/V_F)", f"{market['log_pv']:.2f}",
                   f"ΣP {market['sum_P_tn']:.1f}조$ vs ΣV_F {market['sum_V_tn']:.1f}조$"),
                  ("할인율 (시총가중)", f"{market['r_vw']*100:.1f}%", f"유니버스 {market['n']:,} 종목")]
    gs = X["g_star_10y"].dropna()
    if len(gs):
        w = X.loc[gs.index, "mktcap"]
        cards.append(("가격이 요구하는 성장 g*", f"{np.average(gs, weights=w)*100:.1f}%/y",
                      f"중앙값 {gs.median()*100:.1f}%/y · 펀더멘털 전망 중앙값 {X['g_model'].median()*100:.1f}%/y"))
    if "erp" in X.columns and X["erp"].notna().any():
        rf_ = float(X["rf"].dropna().iloc[0]); erp_ = float(X["erp"].dropna().iloc[0])
        erp_avg_ = float(X["erp_avg"].dropna().iloc[0]) if "erp_avg" in X.columns and X["erp_avg"].notna().any() else np.nan
        cards.append(("할인율의 재료", f"{rf_*100:.1f}% + β × {erp_*100:.1f}%",
                      f"국채 10년 + β × 내재 ERP · ERP 2014년 이후 평균 {erp_avg_*100:.1f}%"))
    if "r_star_g15_pct" in X.columns and X["r_star_g15_pct"].notna().any():
        r15 = X["r_star_g15_pct"].dropna()
        cards.append(("성장 15% 전망일 때 가격이 주는 수익률", f"{r15.median():.1f}%/y",
                      f"중앙값 · 시총 상위 50 {X.head(50)['r_star_g15_pct'].median():.1f}%/y"))

    st_html = ""
    if status:
        cls = "bad" if status["n_fail"] else ("warn" if status["n_warn"] else "")
        head = (f"데이터 점검 <b>실패 {status['n_fail']}건</b>" if status["n_fail"]
                else (f"데이터 점검 경고 {status['n_warn']}건" if status["n_warn"] else "데이터 점검 전부 통과"))
        items = "".join(f"<li>[{i['status']}] {e(i['check'])} — {e(i['detail'])}</li>" for i in status["issues"][:8])
        st_html = (f'<div class="note {cls}">{head} · 검사 {status["n_checks"]}개, 건너뜀 {status["n_skip"]}개 '
                   f'(<code>reports/selfcheck.md</code>)' + (f"<ul>{items}</ul>" if items else "") + "</div>")

    cal_html = ""
    if cal and "bucket" in cal and len(cal["bucket"]):
        c = cal["bucket"].copy()
        keep = [x for x in ["bucket", "실현 3y 연환산 중앙값 %", "3y 달성 비율", "기업-분기 (3y)"] if x in c.columns]
        c = c[keep].rename(columns={"bucket": "가격이 요구한 성장 g*", "실현 3y 연환산 중앙값 %": "실현 3년 연환산 중앙값 %",
                                    "3y 달성 비율": "3년 달성 비율", "기업-분기 (3y)": "관측치"})
        for col in c.columns[1:]:
            c[col] = pd.to_numeric(c[col], errors="coerce").round(2)
        cond_html = ""
        if "cond" in cal and len(cal["cond"]):
            cd = cal["cond"].copy()
            if "dim" not in cd.columns:                       # older output: size groups only
                cd = cd.rename(columns={"size": "group"}).assign(dim="규모")
            order = [b for b in ["< 0%", "0~5%", "5~10%", "10~15%", "15~20%", "20~30%", "30~50%", "> 50%"] if b in set(cd["bucket"])]
            intros = {"규모": ("규모별.", "같은 요구 성장이라도 규모에 따라 달성 비율이 다르다.", ["시총 상위 200", "201위 이하"]),
                      "직전 성장": ("직전 성장별.", "이미 빨리 자라던 기업이 무거운 요구를 더 자주 채우는가. 초신성 후보의 전제는 이 무리 안에서 읽는다.",
                                 ["직전 1년 성장 10% 미만", "10~30%", "30% 이상"])}
            for dim, (head, note, groups) in intros.items():
                d0 = cd[cd["dim"] == dim]
                if d0.empty:
                    continue
                w = d0.pivot(index="bucket", columns="group", values="rate").reindex(order)
                nn = d0.pivot(index="bucket", columns="group", values="n").reindex(order)
                t = pd.DataFrame({"가격이 요구한 성장 g*": order})
                for s in [g for g in groups if g in w.columns]:
                    t[f"{s} 달성 비율"] = w[s].round(2).values
                    t[f"{s} 관측치"] = nn[s].values
                cond_html += (f'<p style="margin:10px 0 4px"><b>{head}</b> {note}</p>' + t.to_html(index=False, border=0, na_rep=""))
            cond_html += ('<p class="small" style="margin:8px 0 0">종목별 표의 <b>3년 달성 확률</b>은 요구 성장·규모·산업·직전 성장·수익성을 '
                          '함께 조건으로 넣은 로짓 값이다(<code>reports/phase7_expectations.md</code> 4-2절).</p>')
        cal_html = ('<h2>기저율 — 가격이 요구한 성장을 실제로 달성한 비율</h2>'
                    '<div class="note small">자기 기대를 g*와 비교할 때 쓰는 눈금이다. 요구 성장이 높은 구간일수록 달성 비율이 급격히 떨어진다. '
                    '단, 달성하지 못해도 주가가 내렸다는 뜻은 아니다(단계 7 검증에서 수익률 예측력 없음).'
                    + c.to_html(index=False, border=0, na_rep="") + cond_html + "</div>")

    th = "".join(f"<th>{e(lbl)}</th>" for _, lbl in COLS)
    return f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SFV 종목별 요구 기대 — {m.date()}</title><style>{CSS}</style></head><body><div class="wrap">
<h1>SFV — 가격이 요구하는 기대</h1>
<p class="sub">평가 기준월 {m.date()} · 생성 {datetime.now(timezone.utc).astimezone().strftime('%Y-%m-%d %H:%M')} ·
잔차 기준월 {eps_month.date() if eps_month is not None and pd.notna(eps_month) else '없음'} (13F 공시 지연) ·
모든 입력은 해당 시점까지 공시된 값</p>
{st_html}
<div class="cards">{''.join(f'<div class="card"><div class="k">{e(k)}</div><div class="v">{e(v)}</div><div class="n">{e(n)}</div></div>' for k, v, n in cards)}</div>
<div class="note">{verdict_note}</div>
<h2>종목별 표</h2>
<div class="controls">
 <input type="search" id="q" placeholder="종목코드 또는 이름 검색">
 <select id="ind"><option value="">산업 전체</option>{''.join(f'<option>{e(i)}</option>' for i in inds)}</select>
 <label class="cb"><input type="checkbox" id="mega"> 초대형주 제외</label>
 <span class="count" id="cnt"></span>
</div>
<div class="tablebox"><table><thead><tr>{th}</tr></thead><tbody id="tb"></tbody></table></div>
<p class="small">열 제목을 누르면 정렬된다. <b>기대분</b> = log(V_gr/V_F), 펀더멘털 성장 전망이 정당화하는 프리미엄.
<b>기대 초과분</b> = log(P/V_gr), 그 너머로 가격이 요구하는 몫(누군가의 전망이 아니라 산술적 조건이다). <b>g* 10년</b>은 10년간 매년 그만큼 이익이 커지고 이후 기존 감쇠를 따를 때 가격이 맞는 성장률.
<b>g* 하한·상한</b>은 무형자산 자본화 파라미터를 문헌의 모든 조합으로 바꿔 다시 푼 g*의 범위다. 좁으면 파라미터 선택에 좌우되지 않는다는 뜻이고(초대형주에서 대체로 그렇다),
넓으면 점이 아니라 구간으로 읽어야 한다(중소형주에서 흔하다). 자기 기대가 구간 안이면 판단을 보류한다.
<b>g* 평균ERP</b>는 시장 위험프리미엄이 2014년 이후 평균이었을 때의 요구 성장. <b>g* 정규화</b>는 출발 이익을 후행 4분기 대신 최근 5년 평균 조정 이익률 × 현재 매출로 놓고 다시 푼 요구 성장이며,
두 출발 이익이 30% 넘게 다르거나 후행 이익이 적자면 <b>이익국면</b> 표시가 붙는다(경기 저점·피크 기업은 이 열을 읽는다).
<b>3년 달성 확률</b>은 규모·산업·최근 성장·수익성이 비슷한 기업이 그 요구 성장을 이후 3년간 실제로 달성한 역사적 비율(로짓, 표본 내).
<b>모형 할인율</b> = 국채 10년 + β × 내재 ERP. <b>수익률 | 성장 x%</b>는 그 회사가 10년간 연 x% 성장하고 이후 기존 감쇠를 따를 때 지금 가격이 주는 연 수익률이다.
자기 성장 전망의 열에서 수익률을 읽고 자기 요구수익률과 비교한다.
<b>ω*</b>는 현재 초과 수익성이 그 속도로 감쇠할 때 가격이 맞는 지속성이고, 빈칸은 지속만으로는 가격을 설명할 수 없어 이익 규모의 성장이 필요하다는 뜻이다.
<b>ε</b>은 산업·특성·모멘텀·기대를 걷어낸 동료 대비 잔차이며 백분위가 높을수록 동료 대비 비싸다.</p>
{cal_html}
<footer>산출물 <code>reports/sfv_latest.csv</code> · 방법론 <code>FRAMEWORK.md</code> · 리포트 색인 <code>reports/INDEX.md</code> ·
점검 <code>reports/selfcheck.md</code>. 이 표는 예측이 아니라 가격이 전제하는 가정이다.</footer>
</div><script>const DATA={json.dumps(data, ensure_ascii=False)};const META={json.dumps(meta)};{JS}</script></body></html>"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", default="", help="valuation month (default: latest)")
    a = ap.parse_args(argv)
    log = open(C.LOGS / "report.log", "w", encoding="utf-8")

    def say(msg):
        print(msg, flush=True); log.write(msg + "\n"); log.flush()

    t0 = time.time()
    month = pd.Timestamp(a.month) if a.month else None
    X, m, market, cal, status, eps_month = load_inputs(month, say)
    X = build_table(X)
    csv_path = C.REPORTS / "sfv_latest.csv"
    write_csv(X, csv_path)
    vs = (np.exp(X["log_pv_hi"] - X["log_pv_lo"]) - 1) * 100
    bw = (X["g_star_hi"] - X["g_star_lo"]) * 100
    if vs.notna().any():
        band_txt = (f"무형자산 자본화 파라미터를 문헌의 다른 조합으로 바꾸면 가치가 종목 중앙값 {vs.median():.0f}% 움직인다. "
                    f"같은 변형에서 요구 성장 g*의 폭은 초대형주 중앙값 {bw[X['is_mega']].median():.1f}%p, 그 밖 {bw[~X['is_mega']].median():.1f}%p이고 "
                    "횡단면 순위상관은 0.92~0.98로 유지된다. 그래서 <b>상대 순위는 믿을 수 있고, 절대 가치는 구간으로 읽어야 하며, "
                    "요구 성장도 구간이 넓은 종목(주로 중소형주)에서는 하한·상한을 함께 읽어야 한다</b> "
                    "(<code>reports/intangible_sensitivity.md</code>).")
    else:
        band_txt = "자본화 파라미터 구간이 없다. <code>scripts/intangible_sensitivity.py</code>를 먼저 돌릴 것."
    verdict = ("<b>이 표를 읽는 법.</b> 사전 등록 검정에서 이 모형의 횡단면 판정은 수익률을 예측하지 못했다(판정: 폐기, "
               "<code>reports/phase3_validation.md</code>). 남은 쓸모는 '가격이 무엇을 전제하는가'를 읽는 것이다. "
               "자기 기대가 g*·ω*·T*보다 높으면 매수 근거, 낮으면 매도 근거이며, 모형은 그 판단의 옳고 그름을 말하지 않는다."
               "<br><b>가격 하나는 성장과 요구수익률의 식 하나다.</b> g*는 요구수익률을 모형 할인율(국채 + β × ERP)에 고정하고 성장을 푼 값이고, "
               "'수익률 | 성장 x%' 열은 반대로 성장 전망을 고정하고 가격이 주는 연 수익률을 푼 값이다. 자기 요구수익률을 먼저 정한 뒤 자기 성장 전망의 "
               "열을 읽는 것이 맞다. 내재 ERP는 시장 전체 가격에서 거꾸로 구한 값이라 시장이 비싸면 모든 종목의 g*가 낮아 보이며, "
               "'g* 평균ERP' 열이 그 몫을 보여 준다."
               "<br><b>점 추정 가치 log(P/V_F)를 그대로 믿지 말 것.</b> " + band_txt +
               f"<br><b>초대형</b> 표시(시총 {MEGA_CAP_USD/1e9:.0f}십억$ 이상 또는 상위 {MEGA_CAP_RANK})가 붙은 종목은 "
               "잔차 ε의 횡단면 효력이 검증되지 않았으므로 ε으로 판단하지 말고 요구 기대만 읽어야 한다. 요구 기대 쪽은 초대형주에서 "
               "오히려 가장 안정적이다(장부가가 가격의 5~20%라 자본화 조정이 가치에 거의 실리지 않는다).")
    html_path = C.REPORTS / "sfv_report.html"
    html_path.write_text(html_report(X, m, market, cal, status, eps_month, verdict), encoding="utf-8")
    say(f"  wrote {csv_path.name} ({len(X):,} rows) and {html_path.name} ({html_path.stat().st_size/1024:.0f} KB)")
    say(f"  mega-cap flagged {int(X['is_mega'].sum())}, loss makers {int((X['ttm_e_adj'] <= 0).sum())}, "
        f"g* solved {int(X['g_star_10y'].notna().sum())}, omega* solved {int(X['omega_star'].notna().sum())}")
    say(f"done  {(time.time()-t0)/60:.1f} min -> reports/sfv_report.html")
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
