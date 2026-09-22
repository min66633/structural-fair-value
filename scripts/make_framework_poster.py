# -*- coding: utf-8 -*-
"""Render the framework and its equations as a single image.

One page that holds the whole model: the identity, the residual income value
with the industry-specific intangible adjustment, the estimated fade, the
growth stage, the cross-sectional decomposition, the inversion that is the
tool's purpose, the latest numbers, and the reading rules (including the
parameter band that says how far the required growth can be trusted).
The pre-registered prediction test is not a section any more: by the user's
decision (2026-09-12) it is a finished record, and the sheet only states its
result as a limit. Numbers are read from the pipeline outputs so the sheet
cannot drift from the data.

Korean text is drawn with Malgun Gothic; the equations go through matplotlib's
mathtext, which has its own fonts, so the two never mix inside one string.
Equation heights are measured, not estimated: a sum or a fraction is far
taller than a line and guessing makes it collide with the next line.

Inputs  implied_monthly, expectations_monthly, expectations_calibration,
        persistence_by_size, decomp_monthly_momexp, rim_monthly, universe_monthly,
        expectations_band, intangible_sensitivity_summary, the EPW parameter CSV
Output  reports/sfv_framework.png
Run: python scripts/make_framework_poster.py [--dpi 150] [--out ...]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sfv import config as C  # noqa: E402
from sfv.intangibles import epw_params  # noqa: E402

BG, FG, ACC, MUT, RULE, BOX = "#fbfaf8", "#1d1b18", "#7a3b2e", "#6b6560", "#ddd8d0", "#f2eee7"
W, H = 13.0, 17.2
L0, L1, R0, R1 = 0.048, 0.478, 0.522, 0.952
IND_KO = {"Manuf.": "제조", "Consumer": "소비재", "High-tech": "하이테크", "Other": "기타", "Health": "헬스케어"}


def fy(inches_from_top: float) -> float:
    """Figure-fraction y for a distance below the top edge, so the layout is fixed in inches."""
    return 1.0 - inches_from_top / H


class Poster:
    def __init__(self, w: float, h: float, dpi: int):
        self.fig = plt.figure(figsize=(w, h), dpi=dpi)
        self.fig.patch.set_facecolor(BG)
        self.W, self.H = w, h
        self.rend = self.fig.canvas.get_renderer()
        self.y = {"L": 0.0, "R": 0.0}
        self.x0 = {"L": L0, "R": R0}
        self.x1 = {"L": L1, "R": R1}

    def lh(self, fs: float, mult: float = 1.5) -> float:
        return fs * mult / 72.0 / self.H

    def measured(self, artist) -> float:
        """Actual drawn height as a figure fraction. Tall equations (sums, fractions,
        nested braces) are far taller than a line height, and guessing collides."""
        return artist.get_window_extent(renderer=self.rend).height / (self.H * self.fig.dpi)

    def text(self, col: str, s: str, fs: float = 10.5, color: str = FG, weight: str = "normal",
             indent: float = 0.0, mult: float = 1.5, ha: str = "left", family: str | None = None):
        x = self.x0[col] + indent
        t = self.fig.text(x, self.y[col], s, fontsize=fs, color=color, fontweight=weight, ha=ha, va="top",
                          **({"family": family} if family else {}))
        self.y[col] -= max(self.measured(t), self.lh(fs, 1.0)) + self.lh(fs, mult - 1.0)

    def eq(self, col: str, s: str, fs: float = 12.5, pad: float = 0.004, color: str = FG, indent: float = 0.012):
        self.y[col] -= pad
        t = self.fig.text(self.x0[col] + indent, self.y[col], s, fontsize=fs, color=color, ha="left", va="top")
        self.y[col] -= self.measured(t) + self.lh(fs, 0.45) + pad

    def head(self, col: str, num: str, title: str, sub: str = ""):
        self.y[col] -= 0.009
        self.fig.text(self.x0[col], self.y[col], num, fontsize=13, color=ACC, fontweight="bold", va="top")
        self.fig.text(self.x0[col] + 0.028, self.y[col], title, fontsize=13, color=FG, fontweight="bold", va="top")
        if sub:
            self.fig.text(self.x1[col], self.y[col] - 0.0016, sub, fontsize=9.5, color=MUT, ha="right", va="top")
        self.y[col] -= self.lh(13, 1.35)
        self.fig.add_artist(Line2D([self.x0[col], self.x1[col]], [self.y[col] + 0.004, self.y[col] + 0.004],
                                   color=RULE, lw=1.0, transform=self.fig.transFigure))
        self.y[col] -= 0.005

    def bullets(self, col: str, items: list[tuple[str, str]], fs: float = 9.8, key_w: float = 0.052):
        for k, v in items:
            self.fig.text(self.x0[col] + 0.006, self.y[col], k, fontsize=fs, color=ACC, va="top")
            self.fig.text(self.x0[col] + 0.006 + key_w, self.y[col], v, fontsize=fs, color=FG, va="top")
            self.y[col] -= self.lh(fs, 1.42)

    def table(self, col: str, cols: list[float], header: list[str], rows: list[list[str]], fs: float = 9.5,
              align: list[str] | None = None):
        align = align or (["left"] + ["right"] * (len(header) - 1))
        for x, h, a in zip(cols, header, align):
            self.fig.text(self.x0[col] + x, self.y[col], h, fontsize=fs - 0.5, color=MUT, ha=a, va="top")
        self.y[col] -= self.lh(fs, 1.3)
        self.fig.add_artist(Line2D([self.x0[col], self.x1[col]], [self.y[col] + 0.004, self.y[col] + 0.004],
                                   color=RULE, lw=0.8, transform=self.fig.transFigure))
        self.y[col] -= 0.003
        for r in rows:
            for x, v, a in zip(cols, r, align):
                self.fig.text(self.x0[col] + x, self.y[col], v, fontsize=fs, color=FG, ha=a, va="top")
            self.y[col] -= self.lh(fs, 1.38)

    def box(self, col: str, lines: list[tuple[str, float, str]], pad: float = 0.007):
        """Tinted block: lines of (text, fontsize, colour)."""
        h = sum(self.lh(fs, 1.45) for _, fs, _ in lines) + 2 * pad
        top = self.y[col]
        self.fig.add_artist(FancyBboxPatch((self.x0[col], top - h), self.x1[col] - self.x0[col], h,
                                           boxstyle="round,pad=0.002,rounding_size=0.004", linewidth=0,
                                           facecolor=BOX, transform=self.fig.transFigure, zorder=0))
        self.fig.add_artist(Line2D([self.x0[col] + 0.0012, self.x0[col] + 0.0012], [top - h + 0.002, top - 0.002],
                                   color=ACC, lw=2.2, transform=self.fig.transFigure, zorder=1))
        self.y[col] -= pad
        for s, fs, cl in lines:
            self.fig.text(self.x0[col] + 0.010, self.y[col], s, fontsize=fs, color=cl, va="top")
            self.y[col] -= self.lh(fs, 1.45)
        self.y[col] -= pad


def band_stats(bd: pd.DataFrame) -> dict:
    g, v = bd["g_w"].dropna(), bd["v_s"].dropna()
    return {"n": int(len(g)), "w_med": float(g.median()) if len(g) else np.nan,
            "w_p90": float(g.quantile(.9)) if len(g) else np.nan,
            "le5": float((g <= 5).mean() * 100) if len(g) else np.nan,
            "v_med": float(v.median()) if len(v) else np.nan}


def facts() -> dict:
    M = pd.read_parquet(C.PQ / "implied_monthly.parquet")
    M.index = pd.to_datetime(M.index)
    r, r14 = M.iloc[-1], M.loc[pd.Timestamp("2014-12-31")]
    E = pd.read_parquet(C.PQ / "expectations_monthly.parquet")
    m = E["month"].max()
    d = E[E["month"] == m]
    g = d["g_star_10y"].dropna()
    SZ = pd.read_parquet(C.PQ / "persistence_by_size.parquet")
    CAL = pd.read_parquet(C.PQ / "expectations_calibration.parquet")
    R = pd.read_parquet(C.PQ / "rim_monthly.parquet", columns=["month", "cik"])
    D = pd.read_parquet(C.PQ / "decomp_monthly_momexp.parquet",
                        columns=["month", "y", "delta_S_A", "delta_T_A", "delta_F_A", "delta_M_A", "delta_E_A", "ind_fe_A", "eps_A"])
    s = D.dropna(subset=["eps_A"])
    vy = s.groupby("month")["y"].var()
    share = {c: float((s.groupby("month")[c].var() / vy).mean())
             for c in ["delta_S_A", "delta_T_A", "delta_F_A", "delta_M_A", "delta_E_A", "ind_fe_A", "eps_A"]}

    # capitalisation parameters by industry, with the universe they apply to
    U = pd.read_parquet(C.PQ / "universe_monthly.parquet", columns=["cik", "month", "sic", "mktcap", "in_universe"])
    u = U[U["in_universe"] & (U["month"] == m)].dropna(subset=["sic"]).copy()
    u["sic"] = u["sic"].astype("int64")
    EP = epw_params()
    u["ind"] = u["sic"].map(EP["industry5"])
    cnt = u.groupby("ind").agg(n=("cik", "size"), cap=("mktcap", "sum"))
    P5 = EP.drop_duplicates("industry5").set_index("industry5")
    epw = []
    for ind in ["Manuf.", "Consumer", "High-tech", "Other", "Health"]:
        if ind not in P5.index:
            continue
        pr = P5.loc[ind]
        n_, cap_ = (int(cnt.loc[ind, "n"]), float(cnt.loc[ind, "cap"]) / 1e12) if ind in cnt.index else (0, 0.0)
        epw.append([IND_KO.get(ind, ind), f"{pr['knowDepr']:.0%}", f"{pr['organDepr']:.0%}", f"{pr['gamma']:.0%}",
                    f"{n_:,}", f"{cap_:.1f}"])
    epw_cov = float(u["ind"].notna().mean() * 100) if len(u) else np.nan

    # the parameter band: the same stock under every credible parameter set
    B = pd.read_parquet(C.PQ / "expectations_band.parquet")
    bm = B["month"].iloc[0] if len(B) else None
    if bm is None or bm != m:
        raise SystemExit(f"expectations_band.parquet is for {bm}, not {m.date()}: run scripts/intangible_sensitivity.py first")
    bd = B.merge(d[["cik", "mktcap", "ticker"]], on="cik", how="left")
    bd["rank"] = bd["mktcap"].rank(ascending=False)
    bd["g_w"] = (bd["g_star_hi"] - bd["g_star_lo"]) * 100
    bd["v_s"] = (np.exp(bd["log_pv_hi"] - bd["log_pv_lo"]) - 1) * 100
    band = {"all": band_stats(bd), "top200": band_stats(bd[bd["rank"] <= 200]), "rest": band_stats(bd[bd["rank"] > 200]),
            "by_ticker": bd.dropna(subset=["ticker"]).drop_duplicates("ticker").set_index("ticker")[["g_star_lo", "g_star_hi"]],
            "n_sets": int(B["g_star_n"].max())}
    S = pd.read_parquet(C.PQ / "intangible_sensitivity_summary.parquet")
    rho = S[["rho_log_pv", "rho_g_star", "rho_T_star"]].dropna(how="all")
    sens = {"n": int(len(S)), "rho_lo": float(rho.min().min()), "rho_hi": float(rho.max().max()),
            "om_lo": float(S["omega_star_mkt"].min()), "om_hi": float(S["omega_star_mkt"].max())}
    # the discount rate's ingredients and the return axis (sfv.expectations)
    macro = {k: (float(d[k].dropna().iloc[0]) if k in d.columns and d[k].notna().any() else np.nan) for k in ("rf", "erp", "erp_avg")}
    if "g_star_10y_erpavg" in d.columns:
        macro["dg_erp"] = float(np.nanmedian((d["g_star_10y_erpavg"] - d["g_star_10y"]).to_numpy()) * 100)
    if "r_star_g15" in d.columns:
        macro["r15_med"] = float(d["r_star_g15"].median() * 100)
        macro["r15_top50"] = float(d.sort_values("mktcap", ascending=False).head(50)["r_star_g15"].median() * 100)

    return {"m": m, "M": M, "r": r, "r14": r14, "hl": float(np.log(0.5) / np.log(r["omega_star"])),
            "g_vw": float(np.average(g, weights=d.loc[g.index, "mktcap"])), "g_med": float(g.median()),
            "gmodel_med": float(d["g_model"].median()), "n_firms": len(d), "SZ": SZ, "CAL": CAL, "share": share,
            "stocks": d.set_index("ticker"), "months": R["month"].nunique(), "rows": len(R),
            "dec_m0": s["month"].min(), "dec_m1": s["month"].max(), "dec_n": s["month"].nunique(),
            "epw": epw, "epw_cov": epw_cov, "band": band, "sens": sens, "macro": macro}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--out", default=str(C.REPORTS / "sfv_framework.png"))
    a = ap.parse_args()
    plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False,
                         "mathtext.fontset": "dejavusans", "mathtext.default": "it"})
    F = facts()
    r, m = F["r"], F["m"]
    P = Poster(W, H, a.dpi)
    fig = P.fig

    # ------------------------------------------------------------------ title
    fig.text(L0, fy(0.40), "구조적 적정가치 (SFV) 프레임워크", fontsize=27, color=FG, fontweight="bold", va="top")
    fig.text(L0, fy(0.90), "가격을 펀더멘털 가치와 수요·특성 성분으로 나누고, 남는 것을 “시장이 전제하는 가정”으로 번역한다",
             fontsize=12.5, color=MUT, va="top")
    fig.text(R1, fy(0.42), f"평가 기준월 {m.date()}", fontsize=11.5, color=MUT, ha="right", va="top")
    fig.text(R1, fy(0.71), f"표본 2014-06 ~ {m.strftime('%Y-%m')} · {F['months']}개월 · 기업-월 {F['rows']:,}",
             fontsize=10.5, color=MUT, ha="right", va="top")
    fig.add_artist(Line2D([L0, R1], [fy(1.16), fy(1.16)], color=FG, lw=1.6, transform=fig.transFigure))
    P.y["L"] = P.y["R"] = fy(1.33)

    # ================================================================== LEFT
    P.head("L", "1", "항등식", "월별 횡단면")
    P.eq("L", r"$\log P_{it} = \log V_{F,it} + \delta_{S,it} + \delta_{T,it} + \delta_{F,it} + \delta_{M,it}$", 12.5)
    P.eq("L", r"$\qquad\qquad\qquad + \ \delta_{E,it} + \mathrm{FE}_{ind,t} + \varepsilon_{it}$", 12.5, pad=0.0)
    P.bullets("L", [
        ("$V_F$", "무형자산 조정 · 지속성 추정 잔여이익 가치"),
        (r"$\delta_S$", "구조적 수요 — 기관·big3 보유 비중, 순발행, 지수펀드 비중"),
        (r"$\delta_T$", "일시 수급 — 기관 비중 분기 변화, 펀드 흐름 유발 매매(FIT)"),
        (r"$\delta_F$", "모형보정 특성 — 조정 ROE, 매출성장, R&D 집약도, 규모, 적자, ω"),
        (r"$\delta_M$", "모멘텀 — 12-1개월 수익률, 1개월 수익률"),
        (r"$\delta_E$", "펀더멘털 기대 — 성장 단계 모형이 정당화하는 프리미엄"),
        (r"$\varepsilon$", "잔차 — 동료·특성으로 설명되지 않는 프리미엄"),
    ], fs=9.8, key_w=0.030)
    P.text("L", "로그를 쓰는 이유 — 성분이 곱셈적이다.  $P = V_F(1+s)(1+t)(1+e)$ 에 로그를 취하면 합이 된다.", 9.3, MUT, mult=1.35)
    P.text("L", "$\\delta_S \\equiv \\log(1+s)$ 처럼 성분은 정의부터 로그 단위여서 따로 로그를 붙이지 않는다. 달러로", 9.3, MUT, mult=1.35)
    P.text("L", "관측되는 $P$ 와 $V_F$ 에만 로그가 붙는다. 즉 $\\log(P/V_F) = \\delta_S + \\dots + \\varepsilon$ 과 같은 식이다.", 9.3, MUT)

    P.head("L", "2", "펀더멘털 가치 $V_F$", "잔여이익모형, T = 10년")
    P.eq("L", r"$V_{F,t} = B_t + \sum_{k=1}^{T} \frac{(\mathrm{ROE}_k - r)\,B_{k-1}}{(1+r)^{k}} + TV_T$", 14.5)
    P.text("L", "무형자산 자본화 — 영구재고법(Peters·Taylor 2017). 쓴 돈 그대로 올리고 매년 상각한다", 9.6, MUT)
    P.eq("L", r"$B_t = B^{\mathrm{GAAP}}_t + K_t, \qquad K_t = G_t + O_t$", 12)
    P.eq("L", r"$G_t = (1-\delta_R^{ind})\,G_{t-1} + \mathrm{RD}_t, \qquad O_t = (1-\delta_O)\,O_{t-1} + \gamma^{ind}\,\mathrm{SGA}_t$", 12, pad=0.0)
    P.eq("L", r"$E_t = E^{\mathrm{GAAP}}_t + (\mathrm{RD}_t + \gamma^{ind}\,\mathrm{SGA}_t) - (\delta_R^{ind} G_{t-1} + \delta_O O_{t-1})$", 12, pad=0.0)
    P.text("L", "$\\delta_R$, $\\gamma$ 는 산업별 (Ewens·Peters·Wang 2025, 인수 가격으로 식별, SIC별), $\\delta_O$ = 0.20 은 공통", 9.4, MUT, mult=1.35)
    P.text("L", "분기 적용률 $1-(1-\\delta)^{1/4}$, 기초 재고 = 첫해 투자 / ($g$+$\\delta$), $g$ = 0.10.  "
                f"유니버스 대응 {F['epw_cov']:.0f}%", 9.4, MUT, mult=1.5)
    P.table("L", [0.006, 0.150, 0.235, 0.318, 0.372, 0.428],
            ["산업 (FF5)", "R&D 상각 δ$_R$", "조직 상각 δ$_O$", "판관비 몫 γ", "종목", "시총 조$"], F["epw"], fs=9.3,
            align=["left", "right", "right", "right", "right", "right"])
    P.text("L", "초과 ROE 감쇠 · 장부 경로(청산잉여) · 터미널 · 할인율", 9.6, MUT)
    P.eq("L", r"$x_k = x_\infty + \omega^{k}(x_0 - x_\infty), \qquad x = \mathrm{ROE} - \mathrm{ROE}^{*}$", 12)
    P.eq("L", r"$x_\infty = \frac{a}{1-\omega}, \qquad \mathrm{ROE}_k = \mathrm{ROE}^{*} + x_k$", 12, pad=0.0)
    P.eq("L", r"$B_k = B_{k-1}\,[\,1 + \mathrm{ROE}_k\,(1-p)\,]$", 12, pad=0.0)
    P.eq("L", r"$TV_T = \mathrm{RI}_T\cdot\frac{q}{1-q}\cdot\frac{1}{(1+r)^{T}}, \qquad q = \frac{\omega\,(1+g_T)}{1+r}$", 12, pad=0.0)
    P.eq("L", r"$r_i = r_f + \beta_i\,\mathrm{ERP}_t$", 12, pad=0.0)
    P.text("L", "$\\mathrm{ROE}^{*}$ 감쇠 목표 = 산업(FF12) 중앙값 조정 ROE. 대안: 자본비용 $r$, 장기 초과이익 0 (하한)", 9.4, MUT, mult=1.35)
    P.text("L", "$p$ 환원율 = (배당 + 자사주매입 - 발행) / $E$, 최근 4분기, [0, 1].  ERP = Damodaran 전년 말 내재 ERP", 9.4, MUT, mult=1.35)
    P.text("L", "경계값  ω ≤ 0.90 · |x∞| ≤ 0.10 · |x0| ≤ 0.50 · 장부성장 ≤ 0.25 · 터미널성장 ≤ 0.04 · q ≤ 0.92", 9.4, MUT, mult=1.5)

    P.head("L", "3", "지속성 ω는 가정하지 않고 추정한다", "확장 윈도우")
    P.eq("L", r"$x_{i,t+1} = a + (\omega_0 + \omega' z_{i,t})\,x_{i,t} + \gamma' z_{i,t} + e_{i,t+1}$", 12.5)
    P.text("L", "$z$ = 무형집약도, R&D 집약도, 매출총이익률 안정성, 로그 규모, 연령, 산업집중도(HHI),", 9.6, MUT, mult=1.35)
    P.text("L", "     음의 초과 ROE 더미, 시총 상위 20% · 상위 5% 더미, 시대 더미", 9.6, MUT, mult=1.45)
    P.text("L", "결과 연도 ≤ T 자료로만 적합하고, 그 추정치는 T+1년 6월 30일부터 쓴다 (point in time)", 9.6, MUT)

    P.head("L", "4", "분해 — 성분을 뽑아내는 회귀", f"{F['dec_m0'].strftime('%Y-%m')} ~ {F['dec_m1'].strftime('%Y-%m')} · {F['dec_n']}개월")
    P.eq("L", r"$\log(P/V_F)_{it} = a_t + b_S'\tilde{X}_S + b_T'\tilde{X}_T + b_F'\tilde{X}_F$", 12.5)
    P.eq("L", r"$\qquad\qquad\qquad + \ b_M'\tilde{X}_M + b_E'\tilde{X}_E + \mathrm{FE}_{ind} + \varepsilon_{it}$", 12.5, pad=0.0)
    P.eq("L", r"$\tilde{X} = X - \bar{X}_t \quad\Rightarrow\quad \delta_S = b_S'\tilde{X}_S, \ \dots, \ \ \overline{\delta}_t = 0$", 12, pad=0.0)
    sh = F["share"]
    P.text("L", "분산 몫 (월평균, 종속변수 log(P/V$_F$)의 분산 대비, 모멘텀·기대 성분 포함 스펙)", 9.6, MUT)
    P.table("L", [0.006, 0.190, 0.250, 0.330],
            ["성분", "몫", "성분", "몫"],
            [["구조 $\\delta_S$", f"{sh['delta_S_A']*100:.1f}%", "모멘텀 $\\delta_M$", f"{sh['delta_M_A']*100:.1f}%"],
             ["일시 $\\delta_T$", f"{sh['delta_T_A']*100:.1f}%", "기대 $\\delta_E$", f"{sh['delta_E_A']*100:.1f}%"],
             ["모형보정 $\\delta_F$", f"{sh['delta_F_A']*100:.1f}%", "산업 FE", f"{sh['ind_fe_A']*100:.1f}%"],
             ["", "", "잔차 $\\varepsilon$", f"{sh['eps_A']*100:.1f}%"]],
            fs=9.5, align=["left", "right", "left", "right"])

    # ================================================================== RIGHT
    P.head("R", "5", "성장 단계 — 백로그를 가치에 넣는 경로", "최종 모형")
    P.eq("R", r"$E_k = E_{k-1}(1+g_k), \quad k = 1,2,3 \qquad \mathrm{RI}_k = E_k - r\,B_{k-1}$", 12.5)
    P.text("R", "$g_k$ = 확장 윈도우 매출성장 전망 (과거 성장 · 규모 · 산업 · 잔여이행의무 RPO), 마진 불변", 9.6, MUT)
    P.text("R", "4년째부터 2절의 감쇠로 넘어간다.  결과를 $V_{gr}$ 로 쓴다", 9.6, MUT)
    P.box("R", [("백로그는 ROE가 아니라 이익의 규모를 예측한다 (매출 1·2·3년 t = 7.6 / 3.2 / 2.1, ROE 변화 t < 1).", 9.4, FG),
                ("그래서 초과 ROE 항이 아니라 성장 단계로 들어간다. 종목 가치는 ±10~50% 바뀌지만", 9.4, FG),
                ("집계 프리미엄은 0.73 → 0.77로 변하지 않는다.", 9.4, FG)])

    P.head("R", "6", "역산 — 가격이 전제하는 가정", "이 도구의 용도")
    P.text("R", "다른 입력을 모형값에 고정하고, 가격을 맞추는 미지수 하나를 푼다", 9.8, MUT)
    P.eq("R", r"$V_F(\omega^{*}) = P \ \Rightarrow\ \omega^{*}, \qquad t_{1/2} = \frac{\ln 0.5}{\ln \omega^{*}}$", 12.5)
    P.eq("R", r"$T^{*} = \min\left\{ T : B_0 + \sum_{k=1}^{T}\frac{(\mathrm{ROE}_0 - r)B_{k-1}}{(1+r)^{k}} \geq P \right\}$", 12.5, pad=0.0)
    P.eq("R", r"$V(g^{*};\,H=10,\ r) = P \ \Rightarrow\ g^{*}_{10y} \qquad V(g;\,H=10,\ r^{*}) = P \ \Rightarrow\ r^{*}(g)$", 12.5, pad=0.0)
    P.eq("R", r"$\sum_i V_{F,i}(\omega^{*}_{mkt}) = \sum_i P_i \quad \mathrm{(market\ level)}$", 12.5, pad=0.0)
    P.eq("R", r"$\log(P/V_F) = \log(V_{gr}/V_F) + \log(P/V_{gr})$", 12, pad=0.0)
    P.text("R", "가격 하나는 성장과 요구수익률의 식 하나다. g*는 r을 모형값(국채 + β·ERP)에 고정하고 성장을 푼 것,", 9.4, MUT, mult=1.35)
    P.text("R", "r*(g)는 성장 전망을 고정하고 수익률을 푼 것. 자기 요구수익률을 정한 뒤 r*(g)의 행을 읽는다.", 9.4, MUT, mult=1.35)
    P.text("R", f"g* 하한·상한 = 2절의 자본화 파라미터를 문헌 조합 {F['band']['n_sets']}개로 바꿔 다시 푼 g*의 최소·최대", 9.4, MUT, mult=1.35)
    P.text("R", "정방향(목표가): 매출 성장·이익률 경로 → 이익 경로 → 같은 감쇠·터미널 → 가치. 역산과 같은 엔진", 9.4, MUT)

    P.head("R", "7", f"결과 — {m.strftime('%Y년 %m월')}", "")
    P.table("R", [0.006, 0.235, 0.320, 0.420],
            ["시장 수준", "2014-12", f"{m.strftime('%Y-%m')}", ""],
            [["시장 내재 지속성 ω*", f"{F['r14']['omega_star']:.3f}", f"{r['omega_star']:.3f}", f"반감기 {F['hl']:.0f}년"],
             ["내재 경쟁우위 기간 T*", f"{F['r14']['cap_star']:.0f}년", f"{r['cap_star']:.0f}년", ""],
             ["시총가중 log(P/V$_F$)", f"{F['r14']['log_pv']:.2f}", f"{r['log_pv']:.2f}", f"ΣP {r['sum_P_tn']:.0f}조 달러"],
             ["요구 성장 g*(10년), 시총가중", "", f"{F['g_vw']*100:.1f}%/y", f"전망 {F['gmodel_med']*100:.1f}%/y"],
             [f"할인율 재료: 국채 {F['macro']['rf']*100:.1f}% + β × ERP {F['macro']['erp']*100:.1f}%", "", "",
              f"ERP 평균 {F['macro']['erp_avg']*100:.1f}%면 g* {F['macro'].get('dg_erp', float('nan')):+.1f}%p"],
             ["실현 지속성 (동일가중 / 상위 5%)", "", f"{F['SZ'].ew.mean():.2f} / {F['SZ'].top5.mean():.2f}", "반감기 4~6년"]],
            fs=9.4, align=["left", "right", "right", "right"])
    P.y["R"] -= 0.006
    st, bt = F["stocks"], F["band"]["by_ticker"]
    rows = []
    for t in ["NVDA", "AAPL", "MSFT", "GOOGL", "META", "AMZN"]:
        if t in st.index:
            x = st.loc[t]
            if t in bt.index and pd.notna(bt.loc[t, "g_star_lo"]):
                band = f"{bt.loc[t, 'g_star_lo']*100:.1f}~{bt.loc[t, 'g_star_hi']*100:.1f}%"
            else:
                band = "—"
            r15 = x["r_star_g15"] if "r_star_g15" in x.index else np.nan
            rows.append([t, f"{x['b_adj']/x['mktcap']:.2f}", f"{x['g_star_10y']*100:.1f}%", band,
                         f"{r15*100:.1f}%" if pd.notna(r15) else "—",
                         f"{x['omega_star']:.2f}" if pd.notna(x["omega_star"]) else "—",
                         f"{x['T_star']:.0f}" if pd.notna(x["T_star"]) else "—"])
    P.table("R", [0.006, 0.100, 0.165, 0.262, 0.348, 0.392, 0.428],
            ["종목", "장부가/가격", "g* 10년", "g* 하한~상한", "r*(성장 15%)", "ω*", "T*"], rows, fs=9.4,
            align=["left", "right", "right", "right", "right", "right", "right"])
    P.y["R"] -= 0.006
    cal = F["CAL"]
    cr = [[str(b), f"{v:.0f}%"] for b, v in zip(cal["bucket"], cal["3y 달성 비율"] * 100)][2:]
    P.table("R", [0.006, 0.125, 0.230, 0.350],
            ["요구 성장", "3년 달성", "요구 성장", "3년 달성"],
            [[cr[i][0], cr[i][1], cr[i + 3][0] if i + 3 < len(cr) else "", cr[i + 3][1] if i + 3 < len(cr) else ""]
             for i in range(3)],
            fs=9.4, align=["left", "right", "left", "right"])
    P.text("R", "요구가 높은 구간일수록 달성 비율이 급격히 떨어진다. 다만 미달이 수익률로 처벌되지는 않았다", 9.0, MUT, mult=1.3)

    P.head("R", "8", "읽는 법과 한계", "")
    bd, sn = F["band"], F["sens"]
    P.bullets("R", [
        ("→", "이것은 예측이 아니라 가격이 전제하는 가정이다. 자기 요구수익률을 정하고, 자기 성장 전망에서"),
        ("", "     가격이 주는 수익률 r*(g)가 그보다 높으면 매수 근거, 낮으면 매도 근거다. 모형은 그 판단의"),
        ("", "     옳고 그름을 말하지 않는다. 할인율 ±1%p는 g*를 2~3%p, 감쇠 속도 ±0.05는 0.5%p 움직인다."),
        ("→", "사전 등록 예측 검정(2014-06~2026-09, 13F 분기 수익률, Fama-MacBeth)에서 어느 성분도,"),
        ("", "     어느 벤치마크도 수익률을 예측하지 못했다 → 종목 선별·예측 도구가 아니다. 검정은"),
        ("", "     월별 실행에서 분리해 기록으로만 둔다 (reports/phase3_validation.md)."),
        ("→", f"자본화 파라미터를 문헌 조합 {sn['n']}개로 바꿔도 횡단면 순위상관 {sn['rho_lo']:.2f}~{sn['rho_hi']:.2f}, 시장 내재"),
        ("", f"     ω* {sn['om_lo']:.3f}~{sn['om_hi']:.3f}. 상대 비교와 시장 수준 진단은 파라미터에 좌우되지 않는다."),
        ("→", f"같은 변형에서 종목 가치는 중앙값 {bd['all']['v_med']:.0f}% 움직인다. 요구 성장 g*의 폭은 시총 상위 200에서"),
        ("", f"     중앙값 {bd['top200']['w_med']:.1f}%p, 201위 이하 {bd['rest']['w_med']:.1f}%p (5%p 안 {bd['rest']['le5']:.0f}%). 점 추정 가치 대신"),
        ("", "     요구 기대를 읽고, 상대 비교는 믿고, 폭이 남으면 g* 하한~상한 구간으로 읽는다."),
        ("→", "초대형주는 장부가가 가격의 5~20%라 가치가 지속성 가정 하나로 5배 벌어진다. 경계값은"),
        ("", "     가치 폭발을 막지만 슈퍼스타를 체계적으로 낮게 평가한다."),
        ("→", "12년 · 한 레짐(초대형 기술주 우위) 표본이라 시장 수준 시계열은 검정 불가."),
    ], fs=9.6, key_w=0.016)

    P.head("R", "9", "데이터와 시점 규칙", "point in time")
    P.bullets("R", [
        ("재무제표", "XBRL 최초 보고치만 사용. 행마다 공시일을 두고 정정치는 따로 보관"),
        ("", "     TTM 창 255~295일, 분기 흐름은 같은 시작일 누적값의 차분"),
        ("가격·시총", "월말 조정종가, 상장폐지 종목은 FTD 월말 가격. 주식수는 표지 공시"),
        ("", "     최신치를 분할 보정, 스케일 오타 제거·교차 소스 교체"),
        ("보유·흐름", "13F는 분기말 + 46일부터, N-PORT는 제출일부터, ω는 T+1년 6월 30일부터"),
        ("검증", "수익률은 13F 내재가격 분기 수익률 (상장폐지 포함, yfinance와 상관 0.996)"),
        ("유니버스", "미국 보통주, 시총 3억$ 이상, 금융 제외, 400일 이내 재무제표"),
    ], fs=9.6, key_w=0.058)

    # ------------------------------------------------------------------ footer
    yf_ = min(P.y["L"], P.y["R"]) - 0.012
    fig.add_artist(Line2D([L0, R1], [yf_, yf_], color=RULE, lw=1.0, transform=fig.transFigure))
    fig.text(L0, yf_ - 0.006, "무료 데이터만 사용 (SEC XBRL · 13F · N-PORT · FTD · yfinance · FRED)  ·  사양 FRAMEWORK.md  ·  "
             "종목별 표 reports/sfv_report.html  ·  실행 python run_sfv.py", fontsize=9.2, color=MUT, va="top")
    fig.text(R1, yf_ - 0.006, f"생성 {pd.Timestamp.today().date()}", fontsize=9.2, color=MUT, ha="right", va="top")

    out = Path(a.out)
    fig.savefig(out, dpi=a.dpi, facecolor=BG)
    plt.close(fig)
    print(f"left column ends at y={P.y['L']:.3f}, right at y={P.y['R']:.3f} (must stay above 0.03)")
    if min(P.y["L"], P.y["R"]) < 0.03:
        print("  WARNING: content overflows the page")
    print(f"wrote {out} ({out.stat().st_size/1024:.0f} KB, {int(W*a.dpi)}x{int(H*a.dpi)} px)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
