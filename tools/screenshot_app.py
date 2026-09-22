# -*- coding: utf-8 -*-
"""Offscreen screenshots of the desktop calculator for docs/img (no display needed).

    python tools/screenshot_app.py [--ticker NVDA] [--tabs 전제,목표주가,추이] [--out docs/img]

Renders the window with Qt's offscreen platform, selects the ticker, and saves one PNG per
requested tab, named app_<ascii name>.png. On Windows the offscreen platform finds no fonts
unless QT_QPA_FONTDIR points at the system font directory, so that is set here.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if sys.platform == "win32":
    os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PySide6.QtWidgets import QApplication  # noqa: E402

from sfv.store import Store  # noqa: E402
from app.main import MainWindow, default_data_dir  # noqa: E402

# tab title -> file name
NAMES = {"전제": "premise", "목표주가": "target", "추이": "history", "매수일 대조": "scorecard",
         "수기 입력": "manual", "도움말": "help"}


def main() -> int:
    ap = argparse.ArgumentParser(description="offscreen screenshots of the SFV calculator")
    ap.add_argument("--ticker", default="NVDA")
    ap.add_argument("--tabs", default="전제,목표주가,추이", help="comma-separated tab titles")
    ap.add_argument("--out", default=str(ROOT / "docs" / "img"))
    ap.add_argument("--width", type=int, default=1400)
    ap.add_argument("--height", type=int, default=880)
    ap.add_argument("--rev-growth", default="40,30,20,15,12",
                    help="revenue growth path (percent by year) typed into the target tab before its capture; '' = leave blank")
    ap.add_argument("--margin", default="60,58,55,52,50", help="adjusted margin path (percent by year) for the target tab")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    # Run from the repository root with a relative data path so the status bar shows "app\\data", not a machine path.
    os.chdir(ROOT)
    compact = Path("app") / "data"
    store = Store(compact if (compact / "firms.parquet").exists() else default_data_dir())
    app = QApplication.instance() or QApplication(sys.argv)
    w = MainWindow(store)
    w.resize(a.width, a.height)
    w.show()
    app.processEvents()
    w.select_ticker(a.ticker)
    app.processEvents()

    want = [t.strip() for t in a.tabs.split(",") if t.strip()]
    titles = {w.tabs.tabText(i): i for i in range(w.tabs.count())}
    failed = 0
    for title in want:
        if title not in titles:
            print(f"no tab named {title!r}; have {list(titles)}")
            failed += 1
            continue
        w.tabs.setCurrentIndex(titles[title])
        if title == "목표주가" and a.rev_growth:
            # a scenario on screen says more than empty fields
            tab = w.tab_target
            tab.ed_growth.setText(a.rev_growth)
            tab.ed_margin.setText(a.margin)
            tab.sp_years.setValue(len([v for v in a.rev_growth.split(",") if v.strip()]))
            tab.compute()
        for _ in range(3):
            app.processEvents()
        name = NAMES.get(title, f"tab{titles[title]}")
        path = out / f"app_{name}.png"
        ok = w.grab().save(str(path))
        print(("saved  " if ok else "FAILED ") + str(path))
        failed += 0 if ok else 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
