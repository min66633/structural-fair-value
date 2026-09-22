# -*- coding: utf-8 -*-
"""SFV 계산기: a PySide6 window over sfv.calc.

    python -m app                the window (reads app/data if present, else data/parquet)
    python -m app --data DIR     another data directory (a compact copy or the parquet directory)
    python app/selftest.py       offscreen check that every tab produces numbers

Layout: main.py (window, firm list, header, live data), tabs/ (one file per tab), charts.py (matplotlib
canvases), widgets.py (tables and formatting), workers.py (network fetches off the UI thread).
"""
