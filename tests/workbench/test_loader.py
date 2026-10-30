"""Background jobs must never let Qt objects be destroyed (or timers started) on a worker thread: that is undefined
behaviour in Qt and showed up as intermittent hangs (2026-10-25)."""
import time

from workbench.app.loader import run_async


def test_many_jobs_no_cross_thread_qt(qtbot, qtlog):
    done, failed = [], []
    for i in range(300):
        run_async(lambda i=i: (time.sleep(0.0005 * (i % 3)), i)[1], done.append, failed.append)
    qtbot.waitUntil(lambda: len(done) + len(failed) == 300, timeout=10_000)
    qtbot.wait(200)                                        # let workers finish their clean-up
    assert sorted(done) == list(range(300)) and not failed
    bad = [r.message for r in qtlog.records if "thread" in r.message.lower()]
    assert not bad, bad[:3]


def test_errors_arrive_on_the_gui_thread(qtbot):
    import threading
    seen = []
    run_async(lambda: 1 / 0, lambda _: None, lambda msg: seen.append((msg, threading.current_thread().name)))
    qtbot.waitUntil(lambda: len(seen) > 0, timeout=3000)
    assert "division" in seen[0][0] and seen[0][1] == "MainThread"


def test_garbage_is_collected_only_on_the_gui_thread(qtbot):
    """Python's cyclic collector runs on whichever thread allocates; if that is a worker, Qt objects in the garbage
    are destroyed off the GUI thread ("Timers cannot be started from another thread", intermittent hangs,
    2026-10-25). The window moves collection onto a GUI-thread timer (pyqtgraph's GarbageCollector)."""
    import gc
    import threading
    from workbench.app.window import MainWindow
    from workbench.core.settings import DEFAULTS
    w = MainWindow({**DEFAULTS, "recent": [], "layouts": {}})
    qtbot.addWidget(w)
    where = []
    cb = lambda phase, info: where.append(threading.current_thread().name) if phase == "start" else None
    gc.callbacks.append(cb)
    try:
        def make_cycles():
            for _ in range(20_000):
                a = []
                a.append(a)                        # cyclic garbage: only the cyclic collector frees it
            return 1
        done = []
        for _ in range(20):
            run_async(make_cycles, done.append, done.append)
        qtbot.waitUntil(lambda: len(done) == 20, timeout=20_000)
        qtbot.wait(1500)                           # the GUI-thread timer collects within ~1 s
    finally:
        gc.callbacks.remove(cb)
    assert not gc.isenabled()
    assert where and set(where) == {"MainThread"}, set(where)
