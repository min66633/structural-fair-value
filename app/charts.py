# -*- coding: utf-8 -*-
"""matplotlib canvases for the window: the value composition against the price, and the premise over time.

Colours come from app.theme so the figures sit on the same dark ground as the widgets around them."""
from __future__ import annotations

import os

os.environ.setdefault("QT_API", "pyside6")
import matplotlib  # noqa: E402

matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from app.theme import C, LINE, apply_mpl  # noqa: E402

apply_mpl()


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, w: float = 6.0, h: float = 3.2, dpi: int = 100):
        self.fig = Figure(figsize=(w, h), dpi=dpi, constrained_layout=True, facecolor=C["surface"])
        super().__init__(self.fig)
        self.setMinimumHeight(int(h * dpi))
        self.setStyleSheet(f"background-color: {C['surface']}; border: 1px solid {C['border_soft']}; border-radius: 6px;")

    def clear(self) -> None:
        self.fig.clear()
        self.draw_idle()


def _style(ax) -> None:
    ax.set_facecolor(C["surface"])
    for s in ax.spines.values():
        s.set_color(C["border"])
    ax.tick_params(colors=C["text_dim"], which="both")


def draw_composition(c: MplCanvas, s: dict) -> None:
    """Value = book + explicit years + fade years + terminal, stacked, next to the price. Negative parts stack
    below zero so a negative terminal (loss makers) stays visible."""
    c.fig.clear()
    ax = c.fig.add_subplot(111)
    _style(ax)
    parts = [("장부", s["B0"], LINE["book"]), ("명시 구간", s["pv_exp"], LINE["explicit"]),
             ("감쇠 구간", s["pv_fade"], LINE["fade"]), ("터미널", s["tv"], LINE["terminal"])]
    up = down = 0.0
    for lab, v, col in parts:
        v = v / 1e9
        if v >= 0:
            ax.bar(0, v, bottom=up, color=col, label=lab, width=0.55)
            up += v
        else:
            ax.bar(0, v, bottom=down, color=col, label=lab, width=0.55)
            down += v
    ax.bar(1, s["P"] / 1e9, color=LINE["price"], width=0.55, label="가격")
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"가치 {s['V']/1e9:,.0f}", f"가격 {s['P']/1e9:,.0f}"], color=C["text"])
    ax.set_ylabel("십억 달러")
    ax.axhline(0, color=LINE["zero"], lw=0.7)
    leg = ax.legend(loc="upper left", fontsize=8, frameon=False)
    for t in leg.get_texts():
        t.set_color(C["text_dim"])
    ax.set_title(f"터미널 비중 {int(round(s['tv']/s['V']*100))}%   가치/가격 {s['V']/s['P']:.2f}배",
                 fontsize=9, color=C["text"])
    c.draw_idle()


def draw_history(c: MplCanvas, h: pd.DataFrame) -> None:
    """Required growth (trailing and normalised) with the discount rate above; price / value below."""
    c.fig.clear()
    if h is None or h.empty:
        ax = c.fig.add_subplot(111)
        ax.text(0.5, 0.5, "추이 없음", ha="center", va="center", color=C["text_dim"])
        ax.set_axis_off()
        c.draw_idle()
        return
    ax1 = c.fig.add_subplot(211)
    ax2 = c.fig.add_subplot(212, sharex=ax1)
    for ax in (ax1, ax2):
        _style(ax)
    m = h["month"]
    ax1.plot(m, h["g_star_10y"] * 100, color=LINE["g_star"], lw=1.7, label="요구 성장 g* (후행 이익)")
    if "g_star_10y_norm" in h and h["g_star_10y_norm"].notna().any():
        ax1.plot(m, h["g_star_10y_norm"] * 100, color=LINE["g_norm"], lw=1.0, ls="--", label="g* (정규화 이익)")
    if "g_model" in h:
        ax1.plot(m, h["g_model"] * 100, color=LINE["g_model"], lw=1.0, label="모형 전망 g_model")
    ax1.plot(m, h["r"] * 100, color=LINE["r"], lw=1.0, ls=":", label="할인율 r")
    ax1.set_ylabel("%/년")
    ax1.grid(alpha=0.25, color=C["border"])
    pv = np.exp(h["log_pv"])
    ax2.plot(m, pv, color=LINE["pv"], lw=1.5, label="가격 / 모형 가치")
    ax2.axhline(1.0, color=LINE["zero"], lw=0.7)
    ax2.set_yscale("log")
    # plain tick labels: the default log formatter uses mathtext, whose minus sign Malgun Gothic lacks
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax2.yaxis.set_minor_formatter(NullFormatter())
    ax2.set_ylabel("배 (log)")
    ax2.grid(alpha=0.25, color=C["border"])
    for ax, ncol in ((ax1, 2), (ax2, 1)):
        leg = ax.legend(loc="upper left", fontsize=8, frameon=False, ncol=ncol)
        for t in leg.get_texts():
            t.set_color(C["text_dim"])
    c.draw_idle()
