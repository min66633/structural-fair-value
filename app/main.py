# -*- coding: utf-8 -*-
"""SFV 계산기: the window. Firm list on the left, the header with the data dates and the live buttons, the tabs.

    python -m app                app/data if present, else data/parquet
    python -m app --data DIR     another data directory
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_API", "pyside6")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
from PySide6.QtCore import QSettings, QSortFilterProxyModel, Qt  # noqa: E402
from PySide6.QtGui import QAction, QStandardItem, QStandardItemModel  # noqa: E402
from PySide6.QtWidgets import (QApplication, QHBoxLayout, QLabel, QLineEdit, QListView, QMainWindow, QMessageBox,  # noqa: E402
                               QPushButton, QSplitter, QTabWidget, QVBoxLayout, QWidget)

from sfv import config as C  # noqa: E402
from sfv import live  # noqa: E402
from sfv.calc import inversion, with_price  # noqa: E402
from sfv.store import APP_DATA, Store  # noqa: E402
from app import theme  # noqa: E402
from app.tabs.help import HelpTab  # noqa: E402
from app.tabs.history import HistoryTab  # noqa: E402
from app.tabs.manual import ManualTab  # noqa: E402
from app.tabs.premise import PremiseTab  # noqa: E402
from app.tabs.scorecard import ScorecardTab  # noqa: E402
from app.tabs.target import TargetTab  # noqa: E402
from app.theme import CHIP  # noqa: E402
from app.widgets import chip, isnum, pct, usd  # noqa: E402
from app.workers import Fetcher  # noqa: E402

MEGA_CAP_USD = 200e9       # the cross-sectional verdict has no validated edge above this (phase3_vw_reversal)
MEGA_CAP_RANK = 50
FUND_LAG_DAYS = 150        # fundamentals older than this at the valuation month: SEC record lag or late filer


class MainWindow(QMainWindow):
    def __init__(self, store: Store):
        super().__init__()
        theme.apply(QApplication.instance())
        self.store = store
        self.settings = QSettings("SFV", "calculator")
        self.row_panel: pd.DataFrame | None = None
        self.row: pd.DataFrame | None = None
        self.q: dict | None = None
        self.live_price: dict | None = None
        self.live_rf: dict | None = None
        self.fetchers: list[Fetcher] = []
        self.setWindowTitle(f"SFV 계산기 — 패널 {store.month.date()} ({'축약' if store.compact else '전체'} 데이터, {store.root})")
        self._build()
        self._menu()
        geo = self.settings.value("geometry")
        if geo is not None:
            self.restoreGeometry(geo)
        else:
            self.resize(1400, 880)
        self.select_ticker(str(self.settings.value("ticker", "NVDA")))

    # ------------------------------------------------------------------ build
    def _build(self) -> None:
        split = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(split)

        left = QWidget()
        ll = QVBoxLayout(left)
        self.search = QLineEdit()
        self.search.setPlaceholderText("티커 또는 회사명")
        self.search.setClearButtonEnabled(True)
        ll.addWidget(self.search)
        self.model = QStandardItemModel()
        self.rows_by_ticker: dict[str, int] = {}
        F = self.store.firms.sort_values("mktcap", ascending=False)
        for t, n, mk in zip(F["ticker"], F["name"], F["mktcap"]):
            it = QStandardItem(f"{t:<6} {n}   ({mk/1e9:,.0f}bn)")
            it.setData(t, Qt.ItemDataRole.UserRole)
            it.setEditable(False)
            self.rows_by_ticker[str(t)] = self.model.rowCount()
            self.model.appendRow(it)
        self.proxy = QSortFilterProxyModel()
        self.proxy.setSourceModel(self.model)
        self.proxy.setFilterCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.search.textChanged.connect(self.proxy.setFilterFixedString)
        self.search.returnPressed.connect(self._search_enter)
        self.list = QListView()
        self.list.setModel(self.proxy)
        self.list.setUniformItemSizes(True)
        self.list.clicked.connect(lambda ix: self.select_ticker(self.proxy.data(ix, Qt.ItemDataRole.UserRole)))
        ll.addWidget(self.list)
        split.addWidget(left)

        right = QWidget()
        rl = QVBoxLayout(right)
        self.lbl_title = QLabel()
        self.lbl_title.setStyleSheet(f"font-size:17px; font-weight:bold; color:{theme.C['text']};")
        rl.addWidget(self.lbl_title)
        hrow = QHBoxLayout()
        self.chips = QHBoxLayout()
        hrow.addLayout(self.chips)
        hrow.addStretch(1)
        self.btn_live = QPushButton("주가·금리 갱신")
        self.btn_live.setToolTip("yfinance 최신가와 FRED DGS10을 받아 즉시 다시 역산 (실적은 그대로)")
        self.btn_live.clicked.connect(self.fetch_live)
        self.btn_cons = QPushButton("컨센서스 불러오기")
        self.btn_cons.setToolTip("yfinance 애널리스트 매출·EPS 성장(올해·내년)으로 목표주가 탭의 경로를 채우고 계산")
        self.btn_cons.clicked.connect(self.fetch_consensus)
        self.btn_reset = QPushButton("패널 값으로")
        self.btn_reset.setToolTip("실시간 주가·금리를 버리고 패널 월의 값으로 되돌림")
        self.btn_reset.clicked.connect(self.reset_live)
        for b in (self.btn_live, self.btn_cons, self.btn_reset):
            hrow.addWidget(b)
        rl.addLayout(hrow)
        self.lbl_dates = QLabel()
        self.lbl_dates.setWordWrap(True)
        self.lbl_dates.setStyleSheet(f"color:{theme.C['text_dim']}; background:{theme.C['surface']}; "
                                     f"border:1px solid {theme.C['border_soft']}; border-radius:6px; padding:6px 9px;")
        rl.addWidget(self.lbl_dates)

        self.tabs = QTabWidget()
        self.tab_premise = PremiseTab(self.store)
        self.tab_target = TargetTab(self.store)
        self.tab_history = HistoryTab(self.store)
        self.tab_score = ScorecardTab(self.store)
        self.tab_manual = ManualTab(self.store, self.set_manual_row)
        self.tab_help = HelpTab(self.store, self.demo_required_growth)
        for w, name in ((self.tab_premise, "전제"), (self.tab_target, "목표주가"), (self.tab_history, "추이"),
                        (self.tab_score, "매수일 대조"), (self.tab_manual, "수기 입력"), (self.tab_help, "도움말")):
            self.tabs.addTab(w, name)
        rl.addWidget(self.tabs, 1)
        split.addWidget(right)
        split.setSizes([330, 1070])
        self.statusBar().showMessage(f"데이터 {self.store.root}  ·  패널 {self.store.month.date()}  ·  유니버스 {len(self.store.firms):,}종목")

    def _menu(self) -> None:
        m = self.menuBar()
        f = m.addMenu("파일")
        a = QAction("데이터 폴더 열기", self)
        a.triggered.connect(lambda: os.startfile(str(self.store.root)) if os.name == "nt" else None)
        f.addAction(a)
        a = QAction("종료", self)
        a.triggered.connect(self.close)
        f.addAction(a)
        d = m.addMenu("데이터")
        for text, fn in (("주가·금리 갱신", self.fetch_live), ("컨센서스 불러오기", self.fetch_consensus), ("패널 값으로 되돌리기", self.reset_live)):
            a = QAction(text, self)
            a.triggered.connect(fn)
            d.addAction(a)
        h = m.addMenu("도움말")
        a = QAction("읽는 법", self)
        a.triggered.connect(lambda: self.tabs.setCurrentWidget(self.tab_help))
        h.addAction(a)
        a = QAction("정보", self)
        a.triggered.connect(lambda: QMessageBox.information(self, "SFV 계산기", self._about()))
        h.addAction(a)

    def _about(self) -> str:
        mc = self.store.macro
        return (f"SFV 구조적 적정가치 계산기\n패널 {mc['month']}, 유니버스 {mc['n_firms']:,}종목\n"
                f"국채 {mc['rf']*100:.2f}%, ERP {mc['erp']*100:.2f}% (평균 {mc['erp_avg']*100:.2f}%), ω 기준 {mc.get('asof_year')}\n"
                f"데이터 {self.store.root}\n계산: sfv/calc.py (명령줄 scripts/what_if.py와 동일)")

    # ------------------------------------------------------------------ firm selection
    def _search_enter(self) -> None:
        t = self.search.text().strip().upper()
        row, near = self.store.firm(t)
        if not row.empty:
            self.select_ticker(t)
        elif self.proxy.rowCount() > 0:
            self.select_ticker(self.proxy.data(self.proxy.index(0, 0), Qt.ItemDataRole.UserRole))
        else:
            self.statusBar().showMessage(f"{t}: 유니버스에 없음" + (f" (비슷한 이름: {', '.join(near)})" if near else "") + ". 수기 입력 탭을 쓴다.")

    def select_ticker(self, ticker: str) -> None:
        row, near = self.store.firm(ticker)
        if row.empty:
            self.statusBar().showMessage(f"{ticker}: 유니버스에 없음" + (f" (비슷한 이름: {', '.join(near)})" if near else ""))
            return
        self.row_panel = row
        self.live_price = None
        self.row = with_price(row, None, self.live_rf["rf"] if self.live_rf and self.live_rf.get("ok") else None)
        self.settings.setValue("ticker", ticker)
        self._highlight(str(row.iloc[0]["ticker"]))
        self.btn_live.setEnabled(True)
        self.btn_cons.setEnabled(True)
        self.refresh()

    def _highlight(self, ticker: str) -> None:
        """Move the list's selection to the chosen firm, so a search or a restored session still shows where you are."""
        src = self.rows_by_ticker.get(ticker)
        if src is None:
            return
        ix = self.proxy.mapFromSource(self.model.index(src, 0))
        if ix.isValid():
            self.list.setCurrentIndex(ix)
            self.list.scrollTo(ix)

    def set_manual_row(self, row: pd.DataFrame) -> None:
        self.row_panel, self.row, self.live_price = row, row, None
        self.btn_live.setEnabled(False)
        self.btn_cons.setEnabled(False)
        self.refresh()
        self.tabs.setCurrentWidget(self.tab_premise)

    def refresh(self) -> None:
        if self.row is None:
            return
        self.q = inversion(self.row, self.store)
        self._header()
        for tab in (self.tab_premise, self.tab_target, self.tab_history, self.tab_score, self.tab_help):
            tab.set_row(self.row, self.q)

    def demo_required_growth(self, g: float) -> None:
        """The 도움말 button: put the required growth into the forward tab as a ten-year path and compute, so the
        target price comes back as today's price. Showing that once explains the tool better than a paragraph."""
        self.tab_target.apply_required_growth(g)
        self.tabs.setCurrentWidget(self.tab_target)
        self.statusBar().showMessage(f"요구 성장 {g*100:.1f}%를 10년 경로로 넣었다. 목표가가 현재가와 같아지는 것이 "
                                     f"두 탭이 한 엔진이라는 뜻이고, 여기서 성장이나 이익률을 바꾼 만큼이 목표가의 괴리다.")

    def _header(self) -> None:
        x = self.row.iloc[0]
        manual = int(x["cik"]) < 0
        self.lbl_title.setText(f"{x['ticker']}  {x['name']}   산업 {x['ff12']}   시총 {x['mktcap']/1e9:,.0f}십억")
        while self.chips.count():
            w = self.chips.takeAt(0).widget()
            if w is not None:
                w.hide()                    # deleteLater() alone leaves the old chip painted until the event loop runs
                w.setParent(None)
                w.deleteLater()
        rank = int((self.store.firms["mktcap"] > float(x["mktcap"])).sum()) + 1
        if not manual and (float(x["mktcap"]) > MEGA_CAP_USD or rank <= MEGA_CAP_RANK):
            self.chips.addWidget(chip(f"초대형 ({rank}위)", CHIP["mega"]))
        if not (isnum(x["ttm_e_adj"]) and float(x["ttm_e_adj"]) > 0):
            self.chips.addWidget(chip("적자", CHIP["loss"]))
        if self.q and self.q.get("phase") not in (None, "정상"):
            self.chips.addWidget(chip(f"이익국면 {self.q['phase']}", CHIP["phase"]))
        age = (self.store.month - pd.Timestamp(x["f_end"])).days
        if not manual and age > FUND_LAG_DAYS:
            self.chips.addWidget(chip(f"재무지연 {age}일", CHIP["lag"]))
        if manual:
            self.chips.addWidget(chip("수기 입력", CHIP["manual"]))
        if self.live_price and self.live_price.get("ok"):
            self.chips.addWidget(chip("실시간 주가", CHIP["live"]))
        pr = (f"주가 {usd(x['price_ps'])} ({self.live_price['asof']}, {self.live_price['source']})" if self.live_price and self.live_price.get("ok")
              else f"주가 {usd(x['price_ps'])} (패널 {self.store.month.date()})")
        rf = (f"국채 {pct(x['rf'], 2)} ({self.live_rf['asof']}, FRED)" if self.live_rf and self.live_rf.get("ok") and not manual
              else f"국채 {pct(x['rf'], 2)} (패널)")
        f_end = pd.Timestamp(x["f_end"]).date()
        self.lbl_dates.setText(f"패널 {self.store.month.date()}  ·  실적 기준 {f_end} ({age}일 전)  ·  {pr}  ·  {rf}  ·  "
                               f"ERP {pct(x['erp'], 2)} ({self.store.month.year - 1}년 말)  ·  ω 기준 {self.store.macro.get('asof_year')}  ·  "
                               f"β {float(x['beta']):.2f}  ·  할인율 {pct(x['r'], 2)}")

    # ------------------------------------------------------------------ live data
    def _panel_ticker(self) -> str | None:
        """The current panel firm's ticker, or None for a hand-entered firm / nothing selected."""
        if self.row_panel is None or int(self.row_panel.iloc[0]["cik"]) < 0:
            return None
        return str(self.row_panel.iloc[0]["ticker"])

    def _start(self, fn, *args, tag: str, on_done) -> None:
        self.fetchers = [f for f in self.fetchers if f.isRunning()]
        w = Fetcher(fn, *args, tag=tag, parent=self)
        w.done.connect(on_done)             # bound method of this window: queued into the UI thread
        self.fetchers.append(w)
        w.start()

    def fetch_live(self) -> None:
        t = self._panel_ticker()
        if t is None:
            return
        self.statusBar().showMessage(f"{t} 주가와 국채 금리를 받는 중 …")
        self.btn_live.setEnabled(False)
        self._start(live.price, t, tag=t, on_done=self._on_price)
        self._start(live.rf, tag="rf", on_done=self._on_rf)

    def _apply_live(self) -> None:
        if self._panel_ticker() is None:
            return
        p = self.live_price["price"] if self.live_price and self.live_price.get("ok") else None
        r = self.live_rf["rf"] if self.live_rf and self.live_rf.get("ok") else None
        self.row = with_price(self.row_panel, p, r)
        self.refresh()

    def _on_price(self, tag: str, res: dict) -> None:
        cur = self._panel_ticker()
        self.btn_live.setEnabled(cur is not None)
        if cur is None or tag != cur:
            return                      # the user moved on; a stale result must not land on another firm
        self.live_price = res
        if res.get("ok"):
            x = self.row_panel.iloc[0]
            self.statusBar().showMessage(f"{tag} 주가 {usd(res['price'])} ({res['asof']}, {res['source']}); 패널 {usd(x['price_ps'])} "
                                         f"→ 시총 {pct(res['price'] / float(x['price_ps']) - 1, 1, True)}. 즉시 다시 역산했다.")
        else:
            self.statusBar().showMessage(f"주가를 받지 못함 ({res.get('error', '')}); 패널 값 유지")
        self._apply_live()

    def _on_rf(self, tag: str, res: dict) -> None:
        self.live_rf = res
        if not res.get("ok"):
            self.statusBar().showMessage(f"국채 금리를 받지 못함 ({res.get('error', '')}); 패널 값 유지")
        self._apply_live()

    def reset_live(self) -> None:
        self.live_price, self.live_rf = None, None
        if self.row_panel is not None:
            self.row = self.row_panel
            self.refresh()
            self.statusBar().showMessage("패널 월의 주가와 금리로 되돌렸다")

    def fetch_consensus(self) -> None:
        t = self._panel_ticker()
        if t is None:
            return
        self.statusBar().showMessage(f"{t} 컨센서스를 받는 중 …")
        self.btn_cons.setEnabled(False)
        self._start(live.consensus, t, tag=t, on_done=self._on_consensus)

    def _on_consensus(self, tag: str, cs) -> None:
        cur = self._panel_ticker()
        self.btn_cons.setEnabled(cur is not None)
        if cur is None or tag != cur:
            return
        if not cs or not isinstance(cs, dict) or "rev" not in cs:
            self.statusBar().showMessage("컨센서스: yfinance에서 추정치를 받지 못함 (커버리지 없음 또는 네트워크)")
            return
        self.tab_target.apply_consensus(cs)
        self.tabs.setCurrentWidget(self.tab_target)
        self.statusBar().showMessage(f"{tag} 컨센서스 경로로 목표주가를 계산했다 (애널리스트 {cs['n']}명)")

    # ------------------------------------------------------------------ lifecycle
    def closeEvent(self, ev) -> None:  # noqa: N802
        self.settings.setValue("geometry", self.saveGeometry())
        for w in list(self.fetchers):
            w.wait(3000)
        super().closeEvent(ev)


def default_data_dir() -> Path:
    return APP_DATA if (APP_DATA / "firms.parquet").exists() else C.PQ


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="SFV 계산기")
    ap.add_argument("--data", default="", help="data directory: the compact copy (app/data) or data/parquet")
    a = ap.parse_args(argv)
    app = QApplication.instance() or QApplication(sys.argv)
    store = Store(Path(a.data) if a.data else default_data_dir())
    w = MainWindow(store)
    w.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
