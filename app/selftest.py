# -*- coding: utf-8 -*-
"""Offscreen check of the window: every tab produces numbers for a profitable mega cap, a loss maker, a cyclical
at its peak and a late filer; the hand-entered tab reproduces a panel firm's g*; the target tab's scenario equals
the library's. Exit code 1 on any failure.

    python app/selftest.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ.setdefault("QT_API", "pyside6")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from sfv.calc import inversion, scenario  # noqa: E402
from sfv.store import Store  # noqa: E402
from app.main import MainWindow, default_data_dir  # noqa: E402

FAILS: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("PASS  " if cond else "FAIL  ") + msg)
    if not cond:
        FAILS.append(msg)


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    store = Store(default_data_dir())
    w = MainWindow(store)
    w.show()
    app.processEvents()
    print(f"data {store.root}  month {store.month.date()}  firms {len(store.firms):,}")

    for t in ("NVDA", "IREN", "MU", "KO"):
        w.select_ticker(t)
        app.processEvents()
        x = w.row.iloc[0]
        q = w.q
        check(w.tab_premise.tbl_core.rowCount() == 2, f"{t}: the core table is the two figures a decision uses")
        check(w.tab_premise.tbl_more.rowCount() >= 5, f"{t}: the cross-checks are there but folded away ({w.tab_premise.tbl_more.rowCount()} rows)")
        check(not w.tab_premise.more.body.isVisible(), f"{t}: 자세히 starts collapsed")
        n_sent = w.tab_premise.summary.text().count("<p")
        check(n_sent >= 2, f"{t}: the summary says it in {n_sent} sentences")
        check(all(s not in w.tab_premise.summary.text() for s in ("ω", "T*", "g*", "ERP")),
              f"{t}: and says it without symbols")
        check(w.tab_premise.tbl_base.rowCount() >= 1, f"{t}: base-rate table filled")
        check(w.tab_help.view.toHtml().count("단계") >= 5, f"{t}: the walkthrough carries its steps")
        check(w.tab_help.btn.isEnabled() == bool(np.isfinite(q["g_star"])), f"{t}: the walkthrough's demo button matches whether g* exists")
        check(w.tab_history.tbl.rowCount() >= 1, f"{t}: history table filled ({w.tab_history.tbl.rowCount()} rows)")
        loss = not (float(x["ttm_e_adj"]) > 0)
        if loss:
            check(not np.isfinite(q["g_star"]) and not q["views"], f"{t}: loss maker has no g* and no return axis")
            w.tab_target.ed_margin.setText("30,40")
            w.tab_target.sp_years.setValue(5)
            s = w.tab_target.solve("growth")
            check(bool(s and s.get("ok")) and np.isfinite(s.get("g", np.nan)), f"{t}: required growth solved ({s.get('g', np.nan)*100:.1f}%/y)")
            w.tab_target.ed_growth.setText("100,60,40")
            s = w.tab_target.solve("margin")
            check(bool(s and s.get("ok")) and np.isfinite(s.get("margin", np.nan)), f"{t}: required margin solved ({s.get('margin', np.nan)*100:.1f}%)")
        else:
            check(np.isfinite(q["g_star"]) and len(q["views"]) == 6 and all(np.isfinite(q["returns"])), f"{t}: g* {q['g_star']*100:.1f}% and return axis")
            w.tab_target.ed_growth.setText("20,15,10")
            w.tab_target.ed_margin.setText("")
            w.tab_target.sp_years.setValue(5)
            s = w.tab_target.compute()
            check(bool(s and s["ok"] and s["value_ok"]) and np.isfinite(s["target_ps"]), f"{t}: target price ${s['target_ps']:,.0f} vs ${s['price_ps']:,.2f}")
            ref = scenario(w.row, store, [0.20, 0.15, 0.10, 0.10, 0.10], None, None)
            check(abs(s["V"] / ref["V"] - 1) < 1e-12, f"{t}: tab scenario at the default knobs equals the library's at the model's inputs")
            check(s["r"] == float(x["r"]) and s["omega"] == min(float(x["omega"]), 0.90) and s["payout"] == float(x["payout"]),
                  f"{t}: default discount rate, persistence and payout are the model's exact values ({s['r']*100:.3f}%)")
            w.tab_target.sp_r.setValue(9.0)
            s2 = w.tab_target.compute()
            up = 0.09 < float(x["r"])                      # a lower discount rate must raise the value, a higher one lower it
            check(s2 is not None and abs(s2["r"] - 0.09) < 1e-12 and ((s2["V"] > s["V"]) == up),
                  f"{t}: an edited discount rate is used (model {x['r']*100:.2f}% -> 9%: value {'up' if up else 'down'})")
            w.tab_target.reset_defaults()
        w.tab_score.date.setDate(w.tab_score.date.date().addYears(-1))
        w.tab_score.refresh()
        check(w.tab_score.tbl.rowCount() >= 1 or "없음" in w.tab_score.lbl_head.text(), f"{t}: scorecard rendered ({w.tab_score.lbl_head.text()[:40]})")

    # the hand-entered tab reproduces a panel firm's g* when fed the panel's values
    w.select_ticker("NVDA")
    app.processEvents()
    x = w.row.iloc[0]
    q_nvda = inversion(w.row, store)
    m = w.tab_manual
    m.ed_name.setText("NVDA (수기)")
    m.cb_ind.setCurrentIndex([m.cb_ind.itemData(i) for i in range(m.cb_ind.count())].index(x["ff12"]))
    m.sp_price.setValue(float(x["price_ps"]))
    m.sp_shares.setValue(float(x["mktcap"]) / float(x["price_ps"]) / 1e6)
    m.sp_equity.setValue(float(x["b_adj"]) / 1e9)
    m.sp_e.setValue(float(x["ttm_e_adj"]) / 1e9)
    m.sp_rev.setValue(float(x["rev0"]) / 1e9)
    m.sp_kint.setValue(0.0)
    m.sp_beta.setValue(float(x["beta"]))
    for chk, sp, val in ((m.chk_omega, m.sp_omega, float(x["omega"])), (m.chk_xinf, m.sp_xinf, float(x["xinf"])),
                         (m.chk_payout, m.sp_payout, float(x["payout"])), (m.chk_rs, m.sp_rs, float(x["roe_star_ind"]))):
        chk.setChecked(False)
        sp.setValue(val)
    m.sp_rf.setValue(float(x["rf"]) * 100)
    m.sp_erp.setValue(float(x["erp"]) * 100)
    row = m.compute()
    app.processEvents()
    check(row is not None and int(w.row.iloc[0]["cik"]) < 0, "manual: row became the current firm")
    g_manual = w.q["g_star"] if w.q else np.nan
    check(abs(g_manual - q_nvda["g_star"]) < 2e-3, f"manual NVDA g* {g_manual*100:.2f}% vs panel {q_nvda['g_star']*100:.2f}% (spin-box rounding)")
    check(not w.btn_live.isEnabled(), "manual: live buttons disabled")

    # back to a panel firm, live buttons enabled again
    w.select_ticker("AAPL")
    app.processEvents()
    check(w.btn_live.isEnabled() and int(w.row.iloc[0]["cik"]) > 0, "AAPL: panel firm restored")

    # the walkthrough's demo: the required growth put into the forward tab must return today's price
    for t in ("AAPL", "KO"):
        w.select_ticker(t)
        app.processEvents()
        g = w.q["g_star"]
        w.demo_required_growth(g)
        app.processEvents()
        s = w.tab_target.last
        P = float(w.row.iloc[0]["mktcap"])
        check(w.tabs.currentWidget() is w.tab_target, f"{t}: the demo moves to the 목표주가 tab")
        # the growth is written into the input box at four decimals so it stays readable; the residual is
        # smaller than the displayed precision, so the screen shows an upside of 0%
        check(s is not None and abs(s["V"] / P - 1) < 5e-5,
              f"{t}: the demo reproduces the price ({s['V']/1e9:,.1f} vs {P/1e9:,.1f}bn, upside {s['upside']:+.6%})")

    # --live: the worker-thread path through a real event loop (needs the network; skipped otherwise)
    if "--live" in sys.argv:
        from PySide6.QtCore import QTimer
        g_before = w.q["g_star"]
        state = {"tries": 0}

        def poll():
            state["tries"] += 1
            done = (w.live_price is not None and w.live_rf is not None) or state["tries"] > 60
            if done:
                app.quit()

        w.fetch_live()
        timer = QTimer()
        timer.timeout.connect(poll)
        timer.start(500)
        app.exec()
        timer.stop()
        ok_p = bool(w.live_price and w.live_price.get("ok"))
        ok_r = bool(w.live_rf and w.live_rf.get("ok"))
        print(f"      live price {w.live_price}\n      live rf {w.live_rf}")
        check(ok_p, "live: price fetched through the worker thread")
        check(ok_r, "live: risk-free rate fetched through the worker thread")
        if ok_p:
            x = w.row.iloc[0]
            check(abs(float(x["price_ps"]) - w.live_price["price"]) < 1e-9, f"live: row carries the live price {x['price_ps']:.2f}")
            check(w.q["g_star"] != g_before, f"live: g* re-solved ({g_before*100:.2f}% -> {w.q['g_star']*100:.2f}%)")
        w.reset_live()
        check(w.q["g_star"] == g_before, "live: reset restores the panel premise")
    w.close()
    print(f"\n{len(FAILS)} failure(s)")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
