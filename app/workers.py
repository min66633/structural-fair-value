# -*- coding: utf-8 -*-
"""Network fetches off the UI thread. A Fetcher runs one callable and emits (tag, result); the window keeps a
reference until the thread has finished so it is not collected mid-run.

Connect `done` to a bound method of a QObject that lives in the UI thread (the window): PySide then queues the
call into that thread. A lambda or plain function would run in the worker thread and must not touch widgets."""
from __future__ import annotations

from PySide6.QtCore import QThread, Signal


class Fetcher(QThread):
    done = Signal(str, object)

    def __init__(self, fn, *args, tag: str = "", parent=None):
        super().__init__(parent)
        self.fn, self.args, self.tag = fn, args, tag

    def run(self) -> None:
        try:
            out = self.fn(*self.args)
        except Exception as e:  # noqa: BLE001
            out = {"ok": False, "error": f"{type(e).__name__}: {e}"[:200]}
        self.done.emit(self.tag, out)
