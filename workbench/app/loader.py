"""Run slow work on a background thread, so a 700 MB recording or a whole-file Clean never freezes the window.

The work runs on Qt's shared thread pool. The result comes back through a small QObject that lives on the GUI
thread: a signal emitted from the worker to that object is queued by Qt and delivered on the GUI thread, which is
the only thread allowed to touch widgets.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot

from workbench.core import registry
from workbench.core.recording import Recording

_pending: set[_Relay] = set()                 # keeps each relay alive until its job has answered


class _Relay(QObject):
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, on_done, on_error):
        super().__init__()                    # created on the GUI thread, so its slots run there
        self._on_done, self._on_error = on_done, on_error
        self.done.connect(self._deliver_done)
        self.failed.connect(self._deliver_error)

    @Slot(object)
    def _deliver_done(self, result):
        _pending.discard(self)
        self._on_done(result)

    @Slot(str)
    def _deliver_error(self, message):
        _pending.discard(self)
        self._on_error(message)


class _Job(QRunnable):
    def __init__(self, fn: Callable[[], Any], relay: _Relay):
        super().__init__()
        self.fn, self.relay = fn, relay

    def run(self):                            # worker thread: no widgets here
        try:
            result = self.fn()
        except Exception as e:                # any failure becomes a message, never a crash
            self._send(self.relay.failed, str(e) or type(e).__name__)
        else:
            self._send(self.relay.done, result)

    @staticmethod
    def _send(sig, value):
        try:
            sig.emit(value)
        except RuntimeError:                  # the window closed while this ran: nobody left to tell
            pass


def run_async(fn: Callable[[], Any], on_done: Callable[[Any], None], on_error: Callable[[str], None]) -> None:
    """Call fn() on a worker thread; later call on_done(result) or on_error(message) on the GUI thread."""
    relay = _Relay(on_done, on_error)
    _pending.add(relay)
    QThreadPool.globalInstance().start(_Job(fn, relay))


def load_async(path: str | Path, on_done: Callable[[Recording], None], on_error: Callable[[str], None]) -> None:
    """Read `path` with the format registry on a worker thread."""
    path = Path(path)
    run_async(lambda: registry.read(path), on_done, on_error)
