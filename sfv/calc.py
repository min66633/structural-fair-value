# -*- coding: utf-8 -*-
"""Per-stock calculations shared by the command line (scripts/what_if.py) and the desktop app (app/).

Every function is pure: a one-row frame for the stock (a Store.firms row), the Store for the panels it needs,
and plain numbers back (dicts and frames, no printing). The command line and the app format the same dicts,
so the two cannot disagree on a number.

    firm_arrays / model_value   the model's inputs and the base value (persistence, ROE, discount-rate knobs)
    inversion                   what the price requires: omega*, g* (10y), T*, the normalised-earnings g* and
                                the earnings phase, the capitalisation band, the return axis r*(g), g* at the
                                sample-average ERP, the persistence grid
    scenario                    the forward direction: a revenue-growth and margin path -> value -> target
                                price, the return the price offers on that path, the base rate of that growth
    solve_required              the loss-maker inversion: fix the margin path and solve the growth, or fix the
                                growth path and solve the margin
    history / scorecard         the premise over time; the premise on a buy date against what was delivered
    with_price / manual_row     a live price (and rate) on a panel row; a hand-entered firm as a panel-shaped row

Bounds are the model's: the estimated persistence capped at OMEGA_MAX (a stated one honoured up to OMEGA_HI),
growth searched in [G_LO, G_HI], the return in [R_LO, R_HI], the fade T_YEARS years, the terminal geometric
factor at most Q_MAX.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sfv.rim import OMEGA_MAX, X0_CAP, G_CAP, G_TV_CAP, Q_MAX, T_YEARS
from sfv.implied import value_common_omega, solve_omega
from sfv.expectations import (value_growth_h, cap_years, bisect_vec, solve_return, G_LO, G_HI, G_VIEWS, H_LONG, OMEGA_HI,
                              BUCKET_EDGES, BUCKET_LABELS)
from sfv.store import Store

GRID = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.99]
PAST_GROUPS = [(-np.inf, 0.10, "직전 1년 성장 10% 미만"), (0.10, 0.30, "10~30%"), (0.30, np.inf, "30% 이상")]
PHASE_GAP = 0.30          # trailing earnings this far from the five-year-margin figure flag a peak or a trough
SOLVE_G_RANGE = (-0.5, 3.0)   # revenue growth searched for a loss maker
SOLVE_M_RANGE = (-0.5, 1.0)   # adjusted margin searched for a loss maker


# ---------------------------------------------------------------------- the model's inputs and value
def firm_arrays(d: pd.DataFrame, omega=None, roe=None, discount=None):
    """B0, x0, xinf, roe_star, payout, r, omega for the rows of d; unset knobs keep the model's values.

    The persistence cap OMEGA_MAX applies to the model's own estimate, which is noisy at the extremes. A
    persistence the user states deliberately is theirs to state and is passed through up to OMEGA_HI, the
    same bound the inversion searches, so that typing the required omega back in reproduces the price."""
    B0 = d["b_adj"].to_numpy(float)
    x0 = d["x0"].to_numpy(float) if roe is None else np.full(len(d), float(roe))
    xinf = d["xinf"].to_numpy(float)
    rs = d["roe_star_ind"].to_numpy(float)
    po = d["payout"].to_numpy(float)
    r = d["r"].to_numpy(float) if discount is None else np.full(len(d), float(discount))
    om = (np.minimum(d["omega"].to_numpy(float), OMEGA_MAX) if omega is None
          else np.full(len(d), float(np.clip(omega, 0.0, OMEGA_HI))))
    return B0, x0, xinf, rs, po, r, om


def model_value(d: pd.DataFrame, omega=None, roe=None, discount=None, growth=None, years: int = H_LONG) -> np.ndarray:
    """The base value, or with a constant earnings growth for `years` years before the fade.

    Constant growth goes through value_path rather than expectations.value_growth_h, which is the same
    arithmetic on a flat path but caps the persistence: routing it here keeps one meaning for the omega knob
    across all three value routes. The two agree exactly at any omega <= OMEGA_MAX."""
    B0, x0, xinf, rs, po, r, om = firm_arrays(d, omega, roe, discount)
    if growth is None:
        return value_common_omega(B0, x0, xinf, rs, po, r, om)
    E = d["ttm_e_adj"].to_numpy(float)
    E_path = []
    for _ in range(int(years)):
        E = E * (1.0 + float(growth))
        E_path.append(E)
    return value_path(B0, E_path, xinf, om, rs, po, r)[0]


def value_path(B0, E_path, xinf, omega, roe_star, payout, r, T_fade: int = T_YEARS):
    """PV of an explicit adjusted-earnings path, then the base fade from the final-year ROE, then the terminal.

    Same mechanics as expectations.value_growth_h with the path given year by year instead of one growth rate.

    The persistence is used as given, as in implied.value_common_omega: capping it here would silently value a
    stated assumption at something else, and the inversion reports omega* well above OMEGA_MAX. What bounds the
    terminal is Q_MAX on the geometric factor (at most 11.5x the last residual income), which holds whatever
    omega is. firm_arrays is where the model's own noisy estimate gets capped.

    Returns (value, PV of the explicit years, PV of the fade years, terminal value)."""
    B_prev = B0.copy()
    pv_exp = np.zeros_like(B0)
    disc = np.ones_like(B0)
    roeH = None
    for E in E_path:
        ri = E - r * B_prev
        disc = disc * (1.0 + r)
        pv_exp = pv_exp + ri / disc
        roeH = E / B_prev
        B_prev = np.maximum(B_prev + E * (1.0 - payout), 1e-6)
    x = np.clip(roeH - roe_star, -X0_CAP, X0_CAP)
    pv_fade = np.zeros_like(B0)
    ri_T = g_T = None
    for k in range(1, T_fade + 1):
        x = xinf + omega * (x - xinf)
        roe = roe_star + x
        ri = (roe - r) * B_prev
        disc = disc * (1.0 + r)
        pv_fade = pv_fade + ri / disc
        gb = np.clip(roe * (1.0 - payout), -0.5, G_CAP)
        if k == T_fade:
            ri_T, g_T = ri, gb
        B_prev = B_prev * (1.0 + gb)
    # rim.terminal with the omega cap left out (it caps the model's estimate, not a stated assumption);
    # identical to it whenever omega <= OMEGA_MAX, which is every default call
    q = np.clip(omega * (1.0 + np.minimum(g_T, G_TV_CAP)) / (1.0 + r), 0.0, Q_MAX)
    tv = ri_T * q / (1.0 - q) / disc
    return B0 + pv_exp + pv_fade + tv, pv_exp, pv_fade, tv


def implied_return(row: pd.DataFrame, g: float, years: int = H_LONG) -> float:
    """The annual return the price offers if earnings grow at g for `years` years, then the fade."""
    B0, x0, xinf, rs, po, r, om = firm_arrays(row)
    E0 = row["ttm_e_adj"].to_numpy(float)
    if E0[0] <= 0:
        return np.nan
    P = row["mktcap"].to_numpy(float)
    rr, st = solve_return(lambda rv: value_growth_h(B0, E0, float(g), int(years), xinf, om, rs, po, rv), P)
    return float(rr[0]) if st[0] == 0 else np.nan


# ---------------------------------------------------------------------- base rates and normalised earnings
def realized_3y_share(rev_panel: pd.DataFrame, threshold: float, past_growth: float | None = None) -> dict:
    """How often universe firm-quarters (2014-06 on) then grew revenue at >= threshold a year for three years,
    for all firms and, if past_growth is given, among firms whose own past-year growth was in the same group."""
    s = rev_panel[(rev_panel["end"] >= "2014-06-01") & rev_panel["cagr"].notna()]
    out = {"share": float((s["cagr"] >= threshold).mean()) if len(s) else np.nan, "n": int(len(s))}
    if past_growth is not None and np.isfinite(past_growth):
        for lo, hi, name in PAST_GROUPS:
            if lo <= past_growth < hi:
                sg = s[(s["past"] >= lo) & (s["past"] < hi)]
                out.update({"group": name, "share_g": float((sg["cagr"] >= threshold).mean()) if len(sg) else np.nan,
                            "n_g": int(len(sg))})
    return out


def firm_past_growth(rev_panel: pd.DataFrame, cik: int, f_end) -> float:
    """The firm's own past-year revenue growth as of its fundamentals quarter."""
    I = rev_panel[(rev_panel["cik"] == cik) & (rev_panel["end"] <= f_end)].tail(5)
    if len(I) < 5 or not (340 <= (I["end"].iloc[-1] - I["end"].iloc[0]).days <= 390):
        return np.nan
    return float(I["ttm_revenue"].iloc[-1] / I["ttm_revenue"].iloc[0] - 1.0)


def base_rate(rev_panel: pd.DataFrame, cagr: float, h: int, cik: int, f_end) -> dict:
    """The base rate of a required or assumed revenue growth: overall and within the firm's past-growth group."""
    pg = firm_past_growth(rev_panel, cik, f_end)
    br = realized_3y_share(rev_panel, cagr, pg)
    br.update(cagr=float(cagr), h=int(h), past_growth=pg)
    return br


def base_rate_lines(br: dict) -> list[str]:
    """The base-rate sentences shared by the scenario and the loss-maker inversion."""
    lines = []
    if np.isfinite(br["share"]):
        lines.append(f"    이 경로의 {br['h']}년 매출 CAGR {br['cagr']*100:.1f}%: 2014년 이후 유니버스 기업-분기 {br['n']:,}개 중 "
                     f"이후 3년간 그 속도 이상으로 자란 비율 {br['share']*100:.0f}%")
    if "group" in br and np.isfinite(br.get("share_g", np.nan)):
        lines.append(f"    이 회사처럼 직전 1년에 {br['past_growth']*100:+.0f}% 자란 무리({br['group']}, {br['n_g']:,}개) 안에서는 "
                     f"{br['share_g']*100:.0f}%")
    return lines


def normalized_for(rev_panel: pd.DataFrame, cik: int, f_end) -> tuple[float, float, float]:
    """Mid-cycle earnings for one firm: TTM revenue x its five-year average adjusted margin at the fundamentals
    quarter. Returns (e_norm, margin_ttm, margin_avg5); NaN when fewer than eight quarters."""
    I = rev_panel[(rev_panel["cik"] == cik) & (rev_panel["end"] <= f_end)].tail(20)
    if len(I) < 8:
        return np.nan, np.nan, np.nan
    m = (I["ttm_e_adj"] / I["ttm_revenue"]).clip(-1, 1)
    return float(I["ttm_revenue"].iloc[-1] * m.mean()), float(m.iloc[-1]), float(m.mean())


def bucket_of(g: float) -> str | None:
    """The required-growth bucket of the base-rate tables."""
    if not np.isfinite(g):
        return None
    for lo, hi, lab in zip(BUCKET_EDGES[:-1], BUCKET_EDGES[1:], BUCKET_LABELS):
        if lo <= g < hi:
            return lab
    return None


# ---------------------------------------------------------------------- the inversion
def inversion(row: pd.DataFrame, store: Store, growth_view: float | None = None, years: int = H_LONG) -> dict:
    """What the price requires, for one stock at the model's inputs.

    omega_star / half_life   the common persistence that prices the stock (NaN: even 0.995 is not enough)
    g_star                   the 10-year adjusted-earnings growth the price requires (NaN: loss maker or > +200%)
    T_star                   years of excess ROE the price requires (NaN: over 60 or not applicable)
    e_norm, g_norm, phase    the same inversion from mid-cycle earnings, and the earnings phase
    band                     g* and P/V under the other capitalisation parameter sets, or None
    views, returns           the return axis: the annual return the price offers for each 10-year growth view
    g_alt, r_alt             the required growth at the sample-average ERP, and that discount rate
    view_return              the return at growth_view for `years` years (NaN if no view)
    grid                     [(omega, P/V)] at the model's other inputs, or None
    """
    x = row.iloc[0]
    P = float(x["mktcap"])
    B0, x0, xinf, rs, po, r, om = firm_arrays(row)
    E0 = row["ttm_e_adj"].to_numpy(float)
    out = {"P": P, "v0": float(model_value(row)[0]), "E0": float(E0[0]), "B0": float(B0[0]), "x0": float(x0[0]),
           "xinf": float(xinf[0]), "roe_star": float(rs[0]), "payout": float(po[0]), "r": float(r[0]), "omega": float(om[0])}
    om_star = np.nan
    if x0[0] > xinf[0]:
        om_star, _ = solve_omega(B0, x0, xinf, rs, po, r, P)
        if value_common_omega(B0, x0, xinf, rs, po, r, np.array([OMEGA_HI]))[0] < P:
            om_star = np.nan
    out["omega_star"] = float(om_star)
    out["half_life"] = float(np.log(0.5) / np.log(om_star)) if np.isfinite(om_star) and 0 < om_star < 1 else np.nan
    g_star = np.nan
    if E0[0] > 0:
        gs, st = bisect_vec(lambda g: value_growth_h(B0, E0, g, H_LONG, xinf, om, rs, po, r), G_LO, G_HI, np.array([P]))
        g_star = gs[0] if st[0] == 0 else np.nan
    out["g_star"] = float(g_star)
    out["bucket"] = bucket_of(out["g_star"])
    out["T_star"] = (float(cap_years(B0, x0, rs, po, r, np.array([P]))[0])
                     if (rs[0] + np.clip(x0[0], -X0_CAP, X0_CAP)) > r[0] else np.nan)

    out["band"] = None
    if "g_star_lo" in row.columns and pd.notna(x["g_star_lo"]):
        lo, hi = float(x["g_star_lo"]), float(x["g_star_hi"])
        b = {"g_lo": lo, "g_hi": hi, "width_pp": (hi - lo) * 100.0, "pv_lo": np.nan, "pv_hi": np.nan, "value_swing": np.nan}
        if pd.notna(x["log_pv_lo"]):
            b["pv_lo"], b["pv_hi"] = float(np.exp(x["log_pv_lo"])), float(np.exp(x["log_pv_hi"]))
            b["value_swing"] = float(np.exp(x["log_pv_hi"] - x["log_pv_lo"]) - 1.0)
        out["band"] = b

    e_norm, m_ttm, m_avg = normalized_for(store.rev_panel, int(x["cik"]), x["f_end"])
    out.update({"e_norm": e_norm, "margin_ttm": m_ttm, "margin_avg5": m_avg, "g_norm": np.nan, "phase": None})
    if np.isfinite(e_norm) and e_norm > 0:
        gn, stn = bisect_vec(lambda g: value_growth_h(B0, np.array([e_norm]), g, H_LONG, xinf, om, rs, po, r), G_LO, G_HI, np.array([P]))
        out["g_norm"] = float(gn[0]) if stn[0] == 0 else np.nan
        out["phase"] = ("피크" if E0[0] > e_norm * (1 + PHASE_GAP) else "저점" if E0[0] < e_norm * (1 - PHASE_GAP) else "정상")

    out.update({"views": [], "returns": [], "g_alt": np.nan, "r_alt": np.nan, "view_return": np.nan, "view_years": int(years)})
    if E0[0] > 0:
        views = list(G_VIEWS) + ([float(growth_view)] if growth_view is not None and float(growth_view) not in G_VIEWS else [])
        out["views"] = views
        out["returns"] = [implied_return(row, g) for g in views]
        r_alt = np.array([float(x["rf"] + x["beta"] * store.erp_avg)])
        gs2, st2 = bisect_vec(lambda g: value_growth_h(B0, E0, g, H_LONG, xinf, om, rs, po, r_alt), G_LO, G_HI, np.array([P]))
        out["g_alt"] = float(gs2[0]) if st2[0] == 0 else np.nan
        out["r_alt"] = float(r_alt[0])
        if growth_view is not None:
            out["view_return"] = implied_return(row, float(growth_view), int(years))

    out["grid"] = None
    if E0[0] > 0 and float(x["x0"]) > float(x["xinf"]):
        vals = [float(model_value(row, omega=g)[0]) for g in GRID]
        out["grid"] = [(g, (P / v if v > 0 else np.nan)) for g, v in zip(GRID, vals)]
    return out


# ---------------------------------------------------------------------- the forward direction
def _fill_margins(mg: list[float] | None, m0: float, H: int) -> list[float]:
    if mg is None:
        return [m0] * H
    if len(mg) < H:
        return mg + [mg[-1]] * (H - len(mg))
    return mg[:H]


def scenario(row: pd.DataFrame, store: Store, g: list[float], mg: list[float] | None, e0_bn: float | None = None,
             omega: float | None = None, discount: float | None = None) -> dict:
    """Forward valuation from a revenue-growth path and an adjusted-margin path: the target price at your view.
    omega and discount, if given, apply to the fade after the path and to the discounting.

    ok=False with `reason` when no path can be built (no revenue). Otherwise: the path rows (year, rev, g,
    margin, E, roe), the value and its parts, value_ok (False when the value is <= 0), the per-share target,
    the return the price offers on the path, and the base rate of the path's 3-year revenue CAGR."""
    x = row.iloc[0]
    B0, x0, xinf, rs, po, r, om = firm_arrays(row, omega=omega, discount=discount)
    P = float(x["mktcap"])
    rev0 = float(x["rev0"]) if pd.notna(x["rev0"]) else np.nan
    if not (rev0 > 0):
        return {"ok": False, "code": "no_revenue", "reason": "매출 자료가 없어 경로를 만들 수 없음"}
    e0 = e0_bn * 1e9 if e0_bn is not None else float(x["ttm_e_adj"])
    m0 = e0 / rev0
    H = len(g)
    mg = _fill_margins(mg, m0, H)
    rev, E, R_ = [], [], rev0
    for k in range(H):
        R_ = R_ * (1.0 + g[k])
        rev.append(R_)
        E.append(R_ * mg[k])
    E_path = [np.array([e]) for e in E]
    V, pv_exp, pv_fade, tv = (float(v[0]) for v in value_path(B0, E_path, xinf, om, rs, po, r))
    rows, Bk = [], float(B0[0])
    for k in range(H):
        rows.append({"year": k + 1, "rev": rev[k], "g": g[k], "margin": mg[k], "E": E[k], "roe": E[k] / Bk})
        Bk = max(Bk + E[k] * (1.0 - float(po[0])), 1e-6)
    ni = float(x["ni_gaap"]) if pd.notna(x.get("ni_gaap", np.nan)) else np.nan
    out = {"ok": True, "H": H, "g": list(g), "mg": list(mg), "rev0": rev0, "e0": e0, "m0": m0, "e0_override": e0_bn is not None,
           "ni_margin": ni / rev0 if np.isfinite(ni) else np.nan, "omega": float(om[0]), "r": float(r[0]), "payout": float(po[0]),
           "rows": rows, "B0": float(B0[0]), "V": V, "pv_exp": pv_exp, "pv_fade": pv_fade, "tv": tv, "P": P,
           "value_ok": V > 0, "price_ps": np.nan, "target_ps": np.nan, "upside": np.nan, "ret": np.nan, "base": None}
    if V <= 0:
        return out
    ps = float(x["price_ps"]) if pd.notna(x.get("price_ps", np.nan)) else np.nan
    out["price_ps"] = ps
    out["target_ps"] = V / P * ps if np.isfinite(ps) and ps > 0 else np.nan
    out["upside"] = V / P - 1.0
    rr, st = solve_return(lambda rv: value_path(B0, E_path, xinf, om, rs, po, rv)[0], np.array([P]))
    out["ret"] = float(rr[0]) if st[0] == 0 else np.nan
    h3 = min(3, H)
    cagr = (rev[h3 - 1] / rev0) ** (1.0 / h3) - 1.0
    out["base"] = base_rate(store.rev_panel, cagr, h3, int(x["cik"]), x["f_end"])
    return out


def solve_required(row: pd.DataFrame, store: Store, mode: str, mg: list[float] | None = None, g: list[float] | None = None,
                   years: int | None = None, omega: float | None = None, discount: float | None = None) -> dict:
    """The inversion for firms the plain inversion cannot price (loss makers, thin earnings).

    A loss maker's price encodes two things at once, the scale it will reach and the margin it will earn, and one
    price cannot identify both. So fix one and solve the other:
      mode="growth"   given a margin path (mg) and a horizon (years, default 5), the constant annual revenue
                      growth over the horizon that makes the value equal the price
      mode="margin"   given a revenue-growth path (g, extended to `years` with its last value), the constant
                      adjusted margin over the path that makes the value equal the price
    After the explicit years the estimated fade and the base terminal apply, as in every other path."""
    x = row.iloc[0]
    B0, x0, xinf, rs, po, r, om = firm_arrays(row, omega=omega, discount=discount)
    P = np.array([float(x["mktcap"])])
    rev0 = float(x["rev0"]) if pd.notna(x["rev0"]) else np.nan
    if not (rev0 > 0):
        return {"ok": False, "mode": mode, "code": "no_revenue", "reason": "매출 자료가 없음"}
    if mode == "growth":
        if not mg:
            return {"ok": False, "mode": mode, "code": "need_margin", "reason": "이익률 경로가 필요합니다"}
        H = int(years) if years else 5
        mg = (list(mg) + [mg[-1]] * (H - len(mg)))[:H]

        def val(gv):
            out_ = np.zeros_like(gv)
            for i, gg in enumerate(gv):
                E_path, R_ = [], rev0
                for k in range(H):
                    R_ = R_ * (1.0 + gg)
                    E_path.append(np.array([R_ * mg[k]]))
                out_[i] = value_path(B0, E_path, xinf, om, rs, po, r)[0][0]
            return out_

        gs, st = bisect_vec(val, SOLVE_G_RANGE[0], SOLVE_G_RANGE[1], P)
        out = {"ok": st[0] == 0, "mode": mode, "code": "solved" if st[0] == 0 else "unsolved", "H": H, "mg": mg, "rev0": rev0,
               "status": int(st[0]), "omega": float(om[0]), "r": float(r[0])}
        if st[0] != 0:
            out["reason"] = "연 300% 성장으로도 가격에 못 미치거나(이익률 경로가 너무 낮음), −50%에서도 넘침. 이익률 경로를 바꿔 볼 것"
            return out
        gsol = float(gs[0])
        rev_H = rev0 * (1.0 + gsol) ** H
        out.update({"g": gsol, "rev_H": rev_H, "E_H": rev_H * mg[-1],
                    "base": base_rate(store.rev_panel, gsol, min(3, H), int(x["cik"]), x["f_end"])})
        return out
    if not g:
        return {"ok": False, "mode": mode, "code": "need_growth", "reason": "매출 성장 경로가 필요합니다"}
    g = list(g)
    if years and int(years) > len(g):
        g = g + [g[-1]] * (int(years) - len(g))
    H = len(g)
    rev, R_ = [], rev0
    for k in range(H):
        R_ = R_ * (1.0 + g[k])
        rev.append(R_)

    def val_m(mv):
        out_ = np.zeros_like(mv)
        for i, mm in enumerate(mv):
            E_path = [np.array([rv * mm]) for rv in rev]
            out_[i] = value_path(B0, E_path, xinf, om, rs, po, r)[0][0]
        return out_

    ms, st = bisect_vec(val_m, SOLVE_M_RANGE[0], SOLVE_M_RANGE[1], P)
    out = {"ok": st[0] == 0, "mode": mode, "code": "solved" if st[0] == 0 else "unsolved", "H": H, "g": g, "rev0": rev0,
           "status": int(st[0]), "omega": float(om[0]), "r": float(r[0]),
           "m0": float(x["ttm_e_adj"]) / rev0 if pd.notna(x["ttm_e_adj"]) else np.nan}
    if st[0] != 0:
        out["reason"] = "이익률 100%로도 가격에 못 미치거나 −50%에서도 넘침. 매출 경로를 바꿔 볼 것"
        return out
    mm = float(ms[0])
    out.update({"margin": mm, "rev_H": rev[-1], "E_H": rev[-1] * mm})
    return out


# ---------------------------------------------------------------------- the premise over time
def history(store: Store, cik: int, n: int = 8) -> pd.DataFrame | None:
    """How the price's premise moved: required growth by quarter-end against the fundamentals of the time.
    n quarter-ends (0 = every quarter-end since the firm entered the valuation sample, 2014-06 at the earliest)."""
    E = store.hist
    h = E[(E["cik"] == cik) & (E["month"] <= store.month) & E["month"].dt.month.isin([3, 6, 9, 12])].sort_values("month")
    if n > 0:
        h = h.tail(n)
    if h.empty:
        return None
    return pd.DataFrame({"월": h["month"].dt.strftime("%Y-%m"), "재무 기준": h["f_end"].dt.strftime("%Y-%m"),
                         "시총 $bn": (h["mktcap"] / 1e9).round(0), "요구 성장 g* %": (h["g_star_10y"] * 100).round(1),
                         "전망 g_model %": (h["g_model"] * 100).round(1), "직전 1년 매출성장 %": (np.expm1(h["b1"]) * 100).round(1),
                         "할인율 %": (h["r"] * 100).round(1)})


def history_series(store: Store, cik: int) -> pd.DataFrame:
    """Every valuation month for the charts: month, required growth (trailing and normalised), P/V, discount rate."""
    E = store.hist
    h = E[(E["cik"] == cik) & (E["month"] <= store.month)].sort_values("month")
    return h[["month", "f_end", "mktcap", "g_star_10y", "g_star_10y_norm", "g_model", "r", "log_pv", "omega_star", "T_star", "p_achieve_3y"]]


def scorecard(store: Store, cik: int, since: str) -> dict:
    """The premise on a past date against what the firm delivered since.

    Required growth as of the last valuation month <= `since`, then realised revenue and adjusted-earnings growth
    from the fundamentals of that date to the latest fundamentals, cumulative and annualised, against the
    cumulative growth the premise required. Then how the premise itself moved."""
    E = store.hist
    h = E[(E["cik"] == cik) & (E["month"] <= store.month)].sort_values("month")
    t0 = pd.Timestamp(since)
    h0 = h[h["month"] <= t0]
    if h0.empty:
        return {"ok": False, "since": t0, "first_month": (h["month"].min() if len(h) else None)}
    a, b = h0.iloc[-1], h.iloc[-1]
    I = store.quarters
    I = I[(I["cik"] == cik) & I["ttm_ok"].fillna(False).astype(bool)].sort_values("end").set_index("end")

    def at(end):
        s = I[I.index <= end]
        return s.iloc[-1] if len(s) else None

    f0, f1 = at(a["f_end"]), at(b["f_end"])
    yrs = (b["f_end"] - a["f_end"]).days / 365.25
    pe0 = a["mktcap"] / a["ttm_e_adj"] if a["ttm_e_adj"] > 0 else np.nan
    pe1 = b["mktcap"] / b["ttm_e_adj"] if b["ttm_e_adj"] > 0 else np.nan
    out = {"ok": True, "since": t0, "month0": a["month"], "month1": b["month"], "f0": a["f_end"], "f1": b["f_end"], "yrs": yrs,
           "mktcap0": float(a["mktcap"]), "mktcap1": float(b["mktcap"]), "pe0": pe0, "pe1": pe1, "r0": float(a["r"]), "r1": float(b["r"]),
           "g0": float(a["g_star_10y"]) if pd.notna(a["g_star_10y"]) else np.nan,
           "g1": float(b["g_star_10y"]) if pd.notna(b["g_star_10y"]) else np.nan,
           "p_achieve": float(b["p_achieve_3y"]) if pd.notna(b["p_achieve_3y"]) else np.nan,
           "measurable": not (yrs < 0.5 or f0 is None or f1 is None or pd.isna(a["g_star_10y"])),
           "req_cum": np.nan, "realized": [], "dp": float(b["mktcap"] / a["mktcap"] - 1.0), "direction": None, "dg": np.nan}
    if out["measurable"]:
        req = (1 + a["g_star_10y"]) ** yrs - 1
        out["req_cum"] = float(req)
        for nm, k in (("매출", "ttm_revenue"), ("조정이익", "ttm_e_adj")):
            v0, v1 = f0[k], f1[k]
            if pd.notna(v0) and pd.notna(v1) and v0 > 0 and v1 > 0:
                cum = v1 / v0 - 1
                out["realized"].append({"name": nm, "ok": True, "cum": float(cum), "ann": float((v1 / v0) ** (1 / yrs) - 1),
                                        "achieved": bool(cum >= req)})
            else:
                out["realized"].append({"name": nm, "ok": False})
    if np.isfinite(out["g1"]) and np.isfinite(out["g0"]):
        out["dg"] = out["g1"] - out["g0"]
        out["direction"] = ("내려왔다" if out["g1"] < out["g0"] - 0.01 else "올라갔다" if out["g1"] > out["g0"] + 0.01 else "그대로다")
    return out


# ---------------------------------------------------------------------- live inputs and hand-entered firms
def with_price(row: pd.DataFrame, price_ps: float | None = None, rf: float | None = None) -> pd.DataFrame:
    """The row at a live share price and/or risk-free rate. Market cap scales with the price ratio (the panel's
    share count); the discount rate is rebuilt as rf + beta x erp with the panel's ERP."""
    d = row.copy()
    x = d.iloc[0]
    if price_ps is not None and np.isfinite(price_ps) and price_ps > 0 and pd.notna(x["price_ps"]) and float(x["price_ps"]) > 0:
        d["mktcap"] = float(x["mktcap"]) * float(price_ps) / float(x["price_ps"])
        d["price_ps"] = float(price_ps)
    if rf is not None and np.isfinite(rf):
        d["rf"] = float(rf)
        d["r"] = float(rf) + float(x["beta"]) * float(x["erp"])
    return d


def manual_row(store: Store, name: str, ff12: str, price_ps: float, shares: float, equity: float, e_ttm: float, revenue: float,
               k_int: float = 0.0, beta: float = 1.0, omega: float | None = None, xinf: float | None = None,
               payout: float | None = None, roe_star: float | None = None, rf: float | None = None, erp: float | None = None) -> pd.DataFrame:
    """A hand-entered firm as a row with the panel's columns, so inversion() and scenario() run unchanged.

    Defaults come from the industry table (ROE benchmark, payout) and the firm's size group (persistence,
    long-run excess ROE). k_int is the capitalised intangible stock the user computed; 0 means no
    capitalisation, so the adjusted book is understated and g* biased upward. ROE is on the entered
    (current) book, where the panel uses the year-earlier book."""
    ind = store.industry.set_index("ff12")
    if ff12 not in ind.index:
        raise ValueError(f"unknown industry {ff12}")
    b_adj = float(equity) + float(k_int)
    if not b_adj > 0:
        raise ValueError("adjusted book must be positive (negative-equity firms cannot be valued by the model)")
    mk = float(price_ps) * float(shares)
    if not mk > 0:
        raise ValueError("price and shares must be positive")
    floors = store.macro["mktcap_floor"]
    grp = "top5" if mk >= floors.get("top5", np.inf) else "top20" if mk >= floors.get("top20", np.inf) else "rest"
    rs = float(ind.loc[ff12, "roe_star_ind"]) if roe_star is None else float(roe_star)
    om = float(store.macro["omega_by_size"][grp]) if omega is None else float(omega)
    xi = float(store.macro["xinf_by_size"][grp]) if xinf is None else float(xinf)
    po = float(ind.loc[ff12, "payout_med"]) if payout is None else float(np.clip(payout, 0.0, 1.0))
    rf_ = float(store.macro["rf"]) if rf is None else float(rf)
    erp_ = float(store.macro["erp"]) if erp is None else float(erp)
    roe = float(e_ttm) / b_adj
    row = {"cik": -1, "month": store.month, "ticker": "MANUAL", "name": name, "ff12": ff12, "mktcap": mk, "f_end": store.month,
           "b_gaap": float(equity), "b_adj": b_adj, "k_int": float(k_int), "ttm_e_gaap": float(e_ttm), "ttm_e_adj": float(e_ttm),
           "roe_adj": roe, "roe_star_ind": rs, "x0": float(np.clip(roe, -1, 1) - rs), "xinf": xi, "omega": om,
           "omega_applied": min(om, OMEGA_MAX), "a": xi * (1.0 - min(om, OMEGA_MAX)), "beta": float(beta), "rf": rf_, "erp": erp_,
           "r": rf_ + float(beta) * erp_, "erp_avg": store.erp_avg, "payout": po, "rev0": float(revenue), "price_ps": float(price_ps),
           "ni_gaap": np.nan, "shares": float(shares), "fund_age": np.nan, "size_group": grp,
           "g_star_lo": np.nan, "g_star_hi": np.nan, "log_pv_lo": np.nan, "log_pv_hi": np.nan}
    d = pd.DataFrame([row])
    d["v_f"] = model_value(d)
    d["log_pv"] = np.log(mk / d["v_f"]).where(d["v_f"] > 0)
    return d
