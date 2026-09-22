# -*- coding: utf-8 -*-
"""Turn the model's assumptions into knobs: what is this worth if I believe X?

The estimated fade comes from what thousands of firms actually did, which is a
base rate, not a forecast. If you think this company's advantage outlasts the
base rate, or that the era has changed, you should be able to say so and see
the number move. That is what this does. The inversion answers the same
question from the other side: what would you have to believe for today's price
to be right.

The calculations live in sfv/calc.py (shared with the desktop app in app/); this
file only parses the arguments and prints.

Knobs (any combination; unset ones keep the model's value)
    --omega P      persistence of excess ROE, 0 to 0.995
    --growth G     constant annual earnings growth for --years years, then the fade
    --years N      horizon for --growth (default 10), or the length a --rev-growth path is extended to
    --roe R        override the starting excess ROE over the industry median
    --discount D   discount rate, replacing r_f + beta x ERP

Scenario (the forward direction: your view -> value -> target price)
    --rev-growth 40,30,20,15,12     revenue growth by year, percent
    --margin 55,55,52,50,48         adjusted net margin by year, percent (last value carries; default = today's)
    --e0 B                          replace the starting adjusted earnings with a normalised figure, $bn
    --consensus                     use the yfinance sell-side revenue growth for this year and next as the path
    Earnings follow revenue x margin for the explicit years, then the estimated fade from the final-year
    ROE and the base terminal. Prints the path, the value and its parts, the per-share target against the
    price, the return the price offers at that path, and how often universe firms actually grew that fast.

Modes
    --ticker T     one stock: model value, your scenario, the inversion (what the
                   price requires), the return axis (the annual return the price
                   offers for each growth view, with --growth added to the grid),
                   the required growth at the sample-average ERP, a grid of
                   persistence values against the price, and the quarter-by-quarter
                   history of the required growth against the fundamentals of the time.
                   --since D adds the scorecard: the premise on that date against the
                   revenue and earnings growth delivered since, and how the premise moved
    --universe     the whole market under a shift: --omega-shift or --discount-shift

Everything reads the latest valuation month of rim_monthly.parquet, so the
inputs are the same point-in-time fundamentals the rest of the pipeline uses.

Run: python scripts/what_if.py --ticker NVDA --omega 0.93
     python scripts/what_if.py --universe --omega-shift 0.03
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sfv.rim import OMEGA_MAX  # noqa: E402
from sfv.implied import value_common_omega, solve_omega  # noqa: E402
from sfv.expectations import H_LONG, OMEGA_HI  # noqa: E402
from sfv import config as C  # noqa: E402
from sfv.store import Store, APP_DATA  # noqa: E402
from sfv.calc import (GRID, firm_arrays, model_value, inversion, scenario, solve_required, history, scorecard,  # noqa: E402
                      base_rate_lines)
from sfv.live import consensus  # noqa: E402


def parse_list(s: str) -> list[float]:
    """'40,30,20' (percent) -> [0.40, 0.30, 0.20]"""
    return [float(v) / 100.0 for v in s.split(",") if v.strip()]


def pct_path(v: list[float]) -> str:
    return "/".join(f"{x*100:.0f}" for x in v)


def print_scenario(s: dict, label: str) -> None:
    if not s["ok"]:
        print(f"\n  {label}: {s['reason']}")
        return
    print(f"\n  {label} — 매출 성장 {pct_path(s['g'])}%, 조정 이익률 {pct_path(s['mg'])}% "
          f"({s['H']}년), 이후 추정 감쇠 (ω {s['omega']:.2f}, 할인율 {s['r']*100:.1f}%)")
    gaap = f", GAAP 순이익률 {s['ni_margin']*100:.1f}%" if np.isfinite(s["ni_margin"]) else ""
    print(f"    출발점: 매출 {s['rev0']/1e9:,.0f}십억, 조정이익 {s['e0']/1e9:,.0f}십억 (조정 이익률 {s['m0']*100:.1f}%{gaap})"
          + ("  ← --e0로 정규화 이익 지정" if s["e0_override"] else ""))
    print("    연   매출($bn)   성장    이익률   조정이익($bn)    ROE")
    for w in s["rows"]:
        print(f"    {w['year']:>2d}   {w['rev']/1e9:>9,.0f}   {w['g']*100:>5.0f}%   {w['margin']*100:>5.1f}%   {w['E']/1e9:>10,.0f}   {w['roe']*100:>5.0f}%")
    if not s["value_ok"]:
        print(f"    가치가 0 이하 ({s['V']/1e9:,.0f}십억): 이 경로로는 가격을 설명할 수 없음")
        return
    V = s["V"]
    print(f"    가치 {V/1e9:,.0f}십억 = 장부 {s['B0']/1e9:,.0f} + 명시구간 {s['pv_exp']/1e9:,.0f} + 감쇠구간 {s['pv_fade']/1e9:,.0f} + 터미널 {s['tv']/1e9:,.0f} "
          f"(터미널 {s['tv']/V*100:.0f}%)")
    head = (f"    주당 목표가 ${s['target_ps']:,.0f} (현재 ${s['price_ps']:,.2f}, {s['upside']:+.0%})" if np.isfinite(s["target_ps"])
            else f"    가치/가격 {V/s['P']:.2f}배 ({s['upside']:+.0%})")
    print(head + (f"   이 경로에서 가격이 주는 수익률 연 {s['ret']*100:.1f}% (모형 할인율 {s['r']*100:.1f}%)" if np.isfinite(s["ret"]) else ""))
    for line in base_rate_lines(s["base"]):
        print(line)


def print_solved(s: dict) -> None:
    if "H" not in s:                      # no path could be built
        if s["code"] == "need_margin":
            print("\n  --solve-growth에는 --margin(이익률 경로)이 필요합니다")
        elif s["code"] == "need_growth":
            print("\n  --solve-margin에는 --rev-growth(매출 성장 경로)가 필요합니다")
        else:
            print(f"\n  역산(경로): {s['reason']}")
        return
    if s["mode"] == "growth":
        print(f"\n  역산(경로) — 조정 이익률 {pct_path(s['mg'])}% ({s['H']}년) 경로일 때 가격이 요구하는 연 매출 성장")
        if not s["ok"]:
            print(f"    {s['reason']}")
            return
        print(f"    요구 매출 성장 연 {s['g']*100:.1f}% × {s['H']}년: 매출 {s['rev0']/1e9:,.1f} → {s['rev_H']/1e9:,.1f}십억, "
              f"{s['H']}년째 조정이익 {s['E_H']/1e9:,.1f}십억")
        for line in base_rate_lines(s["base"]):
            print(line)
        print("    읽기: 이익률 경로를 바꾸면 요구 성장도 바뀐다. 적자 기업의 가격은 규모와 수익성을 함께 전제하므로 하나를 정해야 다른 하나가 나온다.")
    else:
        print(f"\n  역산(경로) — 매출 성장 {pct_path(s['g'])}% ({s['H']}년) 경로일 때 가격이 요구하는 조정 이익률 (전 기간 일정)")
        if not s["ok"]:
            print(f"    {s['reason']}")
            return
        print(f"    요구 조정 이익률 {s['margin']*100:.1f}%: {s['H']}년째 매출 {s['rev_H']/1e9:,.1f}십억, 조정이익 {s['E_H']/1e9:,.1f}십억"
              + (f" (현재 이익률 {s['m0']*100:.1f}%)" if np.isfinite(s["m0"]) else ""))
        print("    읽기: 매출 경로를 바꾸면 요구 이익률도 바뀐다. 산업의 정상 이익률과 견줘 그 이익률이 가능한지 자문한다.")


def print_scorecard(sc: dict) -> None:
    if not sc["ok"]:
        first = sc["first_month"].date() if sc["first_month"] is not None else "n/a"
        print(f"\n  매수일 대조: {sc['since'].date()} 이전 평가월이 없음 (첫 평가월 {first})")
        return
    print(f"\n  매수일 대조 — {sc['month0'].date()} 시점의 전제 vs 그 뒤 실현 (재무 {sc['f0'].date()} → {sc['f1'].date()}, {sc['yrs']:.2f}년)")
    if np.isfinite(sc["g0"]):
        print(f"    당시: 시총 {sc['mktcap0']/1e9:,.0f}십억, 조정 P/E {sc['pe0']:.1f}배, 할인율 {sc['r0']*100:.1f}%, 요구 성장 g* {sc['g0']*100:.1f}%/년")
    else:
        print(f"    당시: 시총 {sc['mktcap0']/1e9:,.0f}십억, 조정 P/E {sc['pe0']:.1f}배, 할인율 {sc['r0']*100:.1f}%, 요구 성장 해당 없음")
    if not sc["measurable"]:
        print("    실현치를 잴 만큼 시간이 지나지 않았거나(0.5년 미만) 재무·요구 성장이 없음")
    else:
        for w in sc["realized"]:
            if w["ok"]:
                print(f"    {w['name']:4s} 누적 {w['cum']*100:+.1f}% (연 {w['ann']*100:+.1f}%)  vs 요구 누적 {sc['req_cum']*100:+.1f}% "
                      f"(연 {sc['g0']*100:.1f}%)  → {'달성' if w['achieved'] else '미달'}")
            else:
                print(f"    {w['name']:4s} 성장률 계산 불가 (음수 또는 결측)")
    tail = f", 3년 달성 확률 {sc['p_achieve']*100:.0f}%" if np.isfinite(sc["p_achieve"]) else ""
    if sc["direction"] is not None:
        print(f"    지금: 시총 {sc['mktcap1']/1e9:,.0f}십억 ({sc['dp']*100:+.0f}%), 조정 P/E {sc['pe1']:.1f}배, 할인율 {sc['r1']*100:.1f}%, "
              f"요구 성장 {sc['g1']*100:.1f}%/년 ({sc['dg']*100:+.1f}%p){tail}")
        parts = []
        if pd.notna(sc["pe0"]) and pd.notna(sc["pe1"]):
            parts.append(f"이익이 가격보다 빨리 자라 조정 P/E가 {sc['pe0']:.1f}→{sc['pe1']:.1f}배로 낮아졌고" if sc["pe1"] < sc["pe0"]
                         else f"가격이 이익보다 빨리 올라 조정 P/E가 {sc['pe0']:.1f}→{sc['pe1']:.1f}배로 높아졌고")
        parts.append(f"할인율은 {sc['r0']*100:.1f}→{sc['r1']*100:.1f}%")
        print(f"    읽기: 요구 성장이 {sc['direction']}. {', '.join(parts)}. 전제가 가벼워졌는지는 위 줄, 당시 요구 속도를 채웠는지는 달성·미달 표시로 본다.")
    else:
        print(f"    지금: 시총 {sc['mktcap1']/1e9:,.0f}십억 ({sc['dp']*100:+.0f}%), 요구 성장 해당 없음{tail}")


def one(store: Store, a) -> None:
    t = a.ticker.upper()
    row, near = store.firm(t)
    m = store.month
    if row.empty:
        print(f"{t}: 최신 평가월 {m.date()} 유니버스에 없음")
        if near:
            print("  비슷한 이름:", ", ".join(near))
        return
    x = row.iloc[0]
    P = float(x["mktcap"])
    print(f"\n{t}  {x['name']}   {m.date()}   산업 {x['ff12']}")
    print(f"  시가총액 {P/1e9:,.0f}십억 달러   조정 장부가 {x['b_adj']/1e9:,.0f}   장부가/가격 {x['b_adj']/P:.2f}")
    print(f"  조정 ROE {x['roe_adj']:+.3f}   산업 기준 ROE {x['roe_star_ind']:+.3f}   초과 ROE {x['x0']:+.3f}")
    print(f"  모형 지속성 {x['omega']:.3f} (상한 적용 {x['omega_applied']:.3f})   환원율 {x['payout']:.2f}")
    print(f"  할인율 {x['r']:.3f} = 국채 {x['rf']:.3f} + β {x['beta']:.2f} × ERP {x['erp']:.3f}   (내재 ERP의 2014년 이후 평균 {store.erp_avg:.3f})")
    v0 = float(model_value(row)[0])
    print(f"\n  모형 가치 {v0/1e9:,.0f}십억   가격/가치 {P/v0:.2f}배")

    yrs = int(a.years) if a.years else H_LONG
    if any(v is not None for v in (a.omega, a.roe, a.discount, a.growth)):
        v1 = float(model_value(row, a.omega, a.roe, a.discount, a.growth, yrs)[0])
        knobs = [f"지속성 {a.omega}" if a.omega is not None else "",
                 f"초과 ROE {a.roe}" if a.roe is not None else "",
                 f"할인율 {a.discount}" if a.discount is not None else "",
                 f"연 성장 {a.growth} × {yrs}년" if a.growth is not None else ""]
        print(f"  시나리오 ({', '.join(k for k in knobs if k)}): 가치 {v1/1e9:,.0f}십억   "
              f"가격/가치 {P/v1:.2f}배   모형 대비 {v1/v0-1:+.0%}")

    # inversion: what the price requires
    q = inversion(row, store, growth_view=a.growth, years=yrs)
    print("\n  가격이 요구하는 것")
    om_star, hl, g_star, T_star, g_norm = q["omega_star"], q["half_life"], q["g_star"], q["T_star"], q["g_norm"]
    print(f"    지속성 ω*        {f'{om_star:.3f}  (반감기 {hl:.0f}년)' if np.isfinite(om_star) and 0 < om_star < 1 else '0.995로도 부족 — 이익 규모의 성장이 필요'}")
    print(f"    10년 이익성장 g*  {f'{g_star*100:.1f}%/년' if np.isfinite(g_star) else '해당 없음 (적자 또는 +200%/년으로도 부족)'}")
    print(f"    초과 ROE 유지 기간 {f'{T_star:.0f}년' if np.isfinite(T_star) else '60년 초과 또는 해당 없음'}")
    print("    참고 기저율: 시총 상위 5% 기업의 실현 지속성 0.84~0.89 (반감기 4~6년), 유니버스 동일가중 0.56")
    if q["phase"] is not None:
        print(f"    정규화 이익 기준: 5년 평균 조정 이익률 {q['margin_avg5']*100:.1f}% × 현재 매출 = {q['e_norm']/1e9:,.1f}십억 "
              f"(후행 {q['E0']/1e9:,.1f}십억, 이익률 {q['margin_ttm']*100:.1f}%) "
              f"→ 요구 성장 {f'{g_norm*100:.1f}%/년' if np.isfinite(g_norm) else '해당 없음'}  [이익 국면: {q['phase']}]")
        if q["phase"] != "정상":
            print("    후행 이익이 5년 평균과 30% 넘게 다르다. 경기 저점·피크일 수 있으니 정규화 기준 요구 성장을 함께 읽는다.")
    b = q["band"]
    if b is not None:
        print("\n  무형자산 자본화 가정을 문헌의 다른 조합으로 바꿨을 때의 범위")
        print(f"    10년 요구성장 g*   {b['g_lo']*100:.1f} ~ {b['g_hi']*100:.1f}%/년")
        if np.isfinite(b["pv_lo"]):
            print(f"    가격/가치          {b['pv_lo']:.2f} ~ {b['pv_hi']:.2f}배  (가치 자체는 {b['value_swing']*100:.0f}% 흔들린다)")
        w = b["width_pp"]
        if w <= 5.0:
            print(f"    요구 성장의 폭이 {w:.1f}%p로 좁다. 가치가 크게 움직여도 요구 성장은 거의 그대로이니 판단은 요구 성장 쪽으로 한다.")
        else:
            print(f"    요구 성장의 폭이 {w:.1f}%p로 넓다. 이 종목은 g*도 점이 아니라 구간으로 읽어야 하고,")
            print("    자기 기대가 구간 안이면 모형의 정밀도를 넘어선 것이니 판단을 보류한다.")

    # the return axis: the same price read from the other side
    if q["views"]:
        print("\n  수익률 축 — 10년 성장 전망별로 이 가격이 주는 연 수익률 (이후 기존 감쇠)")
        print("    성장 전망  " + "".join(f"{g*100:>7.0f}%" for g in q["views"]))
        print("    연 수익률  " + "".join(f"{v*100:>7.1f}%" if np.isfinite(v) else f"{'—':>8s}" for v in q["returns"]))
        print(f"    모형 할인율 {q['r']*100:.1f}%에서 요구 성장 {q['g_star']*100:.1f}%.  ERP가 평균 {store.erp_avg*100:.1f}%였다면(할인율 {q['r_alt']*100:.1f}%) "
              f"요구 성장 {q['g_alt']*100:.1f}%" if np.isfinite(q["g_star"]) else "    요구 성장 해당 없음")
        if a.growth is not None and np.isfinite(q["view_return"]):
            rg = q["view_return"]
            print(f"    → 성장 {float(a.growth)*100:.0f}%/년 × {yrs}년 전망이면 이 가격이 주는 수익률은 연 {rg*100:.1f}%: "
                  f"국채 {x['rf']*100:.1f}% 대비 {(rg - x['rf'])*100:+.1f}%p, 모형 할인율 대비 {(rg - x['r'])*100:+.1f}%p")
        print("    읽기: 가격 하나는 성장과 요구수익률의 식 하나다. 자기 요구수익률을 정한 뒤 그 이상을 주는 성장 전망을 믿는지 자문한다.")

    # forward valuation: the target price at your own revenue and margin path, or at the sell-side consensus
    if a.rev_growth:
        g = parse_list(a.rev_growth)
        if a.years and int(a.years) > len(g):
            g = g + [g[-1]] * (int(a.years) - len(g))
        print_scenario(scenario(row, store, g, parse_list(a.margin) if a.margin else None, a.e0, a.omega, a.discount), "시나리오")
    if a.solve_growth:
        print_solved(solve_required(row, store, "growth", mg=parse_list(a.margin) if a.margin else None, years=a.years,
                                    omega=a.omega, discount=a.discount))
    if a.solve_margin:
        print_solved(solve_required(row, store, "margin", g=parse_list(a.rev_growth) if a.rev_growth else None, years=a.years,
                                    omega=a.omega, discount=a.discount))
    if a.consensus:
        cs = consensus(t)
        if cs is None:
            print("\n  컨센서스: yfinance에서 추정치를 받지 못함 (커버리지 없음 또는 네트워크)")
        else:
            tgt = f", 목표주가 평균 ${cs['target']:,.0f}" if np.isfinite(cs.get("target", np.nan)) else ""
            print(f"\n  컨센서스 (yfinance, 애널리스트 {cs['n']}명): 매출 성장 올해 {cs['rev'][0]*100:+.0f}% · 내년 {cs['rev'][1]*100:+.0f}%, "
                  f"EPS 성장 올해 {cs['eps'][0]*100:+.0f}% · 내년 {cs['eps'][1]*100:+.0f}%{tgt}")
            print("    주의: 회계연도 기준 성장률을 TTM 출발점에 그대로 적용한 근사. 3년째부터는 추정 감쇠")
            print_scenario(scenario(row, store, [cs["rev"][0], cs["rev"][1]], None, None, a.omega, a.discount), "컨센서스 경로")

    if q["grid"] is not None:
        print("\n  지속성별 가치 (다른 입력은 모형값 고정)")
        print("    " + "".join(f"{g:>8.2f}" for g in GRID))
        print("    " + "".join((f"{pv:>8.2f}" if np.isfinite(pv) else f"{'—':>8s}") for _, pv in q["grid"]) + "   ← 가격/가치 (1.00이면 가격이 맞음)")
    else:
        print("\n  지속성별 가치: 적자이거나 초과 ROE가 장기 수준 아래라 지속성 격자가 의미 없음. --rev-growth/--margin 또는 --solve-growth/--solve-margin으로 평가")

    h = history(store, int(x["cik"]), int(a.history))
    if h is not None:
        print(f"\n  요구 성장의 추이 (분기말{' 전체' if int(a.history) == 0 else f' 최근 {int(a.history)}개'}) — 실적이 나오고 가격이 움직이면 여기가 바뀐다")
        print("    " + h.to_string(index=False).replace("\n", "\n    "))
    if a.since:
        print_scorecard(scorecard(store, int(x["cik"]), a.since))


def universe(store: Store, a) -> None:
    d = store.firms
    m = store.month
    P = d["mktcap"].to_numpy(float)
    v0 = model_value(d)
    ok = v0 > 0
    base = np.log(P[ok].sum() / v0[ok].sum())
    print(f"\n유니버스 {len(d):,}종목  {m.date()}")
    print(f"  모형 기준 시총가중 log(P/V) {base:+.3f}  (가격이 가치의 {np.exp(base):.2f}배)")
    om0 = np.minimum(d["omega"].to_numpy(float), OMEGA_MAX)
    B0, x0, xinf, rs, po, r, _ = firm_arrays(d)
    rows = []
    for s in ([a.omega_shift] if a.omega_shift is not None else [0.02, 0.05, 0.10]):
        # the shifted persistence is passed straight through, above the model's own cap
        v = value_common_omega(B0, x0, xinf, rs, po, r, np.clip(om0 + s, 0.0, OMEGA_HI))
        k = v > 0
        rows.append({"조정": f"지속성 {s:+.2f}", "시총가중 log(P/V)": np.log(P[k].sum() / v[k].sum()),
                     "가치 변화": v[k].sum() / v0[k].sum() - 1})
    for s in ([a.discount_shift] if a.discount_shift is not None else [-0.01, -0.02]):
        v = value_common_omega(B0, x0, xinf, rs, po, r + s, om0)
        k = v > 0
        rows.append({"조정": f"할인율 {s:+.2f}", "시총가중 log(P/V)": np.log(P[k].sum() / v[k].sum()),
                     "가치 변화": v[k].sum() / v0[k].sum() - 1})
    T = pd.DataFrame(rows)
    T["시총가중 log(P/V)"] = T["시총가중 log(P/V)"].map("{:+.3f}".format)
    T["가치 변화"] = T["가치 변화"].map("{:+.0%}".format)
    print(T.to_string(index=False))
    om_star, _ = solve_omega(B0, x0, xinf, rs, po, r, float(P.sum()))
    hl = f"  (반감기 {np.log(0.5)/np.log(om_star):.0f}년)" if 0 < om_star < 1 else ""
    print(f"\n  전체 가격을 정당화하는 공통 지속성 ω* = {om_star:.3f}{hl}")
    print("  읽기: 모형의 지속성 추정치는 과거 실현치의 기저율이다. 그 기저율이 지금 시대에 낮다고 보면 위 표에서")
    print("        지속성을 올려 보면 되고, ω*는 '얼마나 올려야 가격이 맞는가'를 한 번에 답한다.")


def main() -> int:
    ap = argparse.ArgumentParser(description="모형 가정을 바꿔 보고 가격이 요구하는 값과 비교한다")
    ap.add_argument("--ticker", default="")
    ap.add_argument("--universe", action="store_true")
    ap.add_argument("--month", default="")
    ap.add_argument("--omega", type=float, default=None, help="지속성 (0~0.995)")
    ap.add_argument("--roe", type=float, default=None, help="산업 대비 초과 ROE")
    ap.add_argument("--discount", type=float, default=None, help="할인율")
    ap.add_argument("--growth", type=float, default=None, help="연 이익 성장률(소수), --years 동안")
    ap.add_argument("--years", type=int, default=None, help="--growth 적용 연수(기본 10); --rev-growth 경로를 마지막 값으로 이 연수까지 연장")
    ap.add_argument("--rev-growth", default="", help="시나리오: 연도별 매출 성장률(%%), 쉼표 구분. 예 40,30,20,15,12")
    ap.add_argument("--margin", default="", help="시나리오: 연도별 조정 순이익률(%%), 쉼표 구분. 모자라면 마지막 값 유지, 없으면 현재 이익률")
    ap.add_argument("--e0", type=float, default=None, help="시나리오: 출발 조정이익을 정규화 값(십억 달러)으로 대체")
    ap.add_argument("--consensus", action="store_true", help="yfinance 컨센서스 매출 성장(올해·내년)으로 경로를 만들어 평가")
    ap.add_argument("--solve-growth", action="store_true", help="적자 기업 역산: --margin 경로와 --years를 주면 가격이 요구하는 연 매출 성장을 푼다")
    ap.add_argument("--solve-margin", action="store_true", help="적자 기업 역산: --rev-growth 경로를 주면 가격이 요구하는 조정 이익률을 푼다")
    ap.add_argument("--since", default="", help="--ticker: 매수일(YYYY-MM-DD). 그날의 전제(요구 성장)와 이후 실현 성장을 대조")
    ap.add_argument("--history", type=int, default=8, help="--ticker: 추이에 보일 분기말 수 (0 = 2014-06 이후 전체)")
    ap.add_argument("--omega-shift", type=float, default=None, help="--universe: 지속성을 일괄 가감")
    ap.add_argument("--discount-shift", type=float, default=None, help="--universe: 할인율을 일괄 가감")
    a = ap.parse_args()
    if not a.ticker and not a.universe:
        ap.error("--ticker 또는 --universe 중 하나가 필요합니다")
    # The full panel when it exists, else the compact copy the desktop app ships with (latest month only).
    root = C.PQ if (C.PQ / "rim_monthly.parquet").exists() else APP_DATA
    store = Store(root, month=a.month)
    if a.ticker:
        one(store, a)
    if a.universe:
        universe(store, a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
