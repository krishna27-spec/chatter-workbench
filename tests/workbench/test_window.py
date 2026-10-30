import time

import numpy as np
import pytest
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QMessageBox

from workbench.app import loader
from workbench.app.window import MainWindow
from workbench.core import registry
from workbench.core.recording import Channel, Recording
from workbench.core.settings import DEFAULTS


@pytest.fixture
def win(qtbot):
    registry.load_formats()
    w = MainWindow({**DEFAULTS, "recent": [], "layouts": {}})
    qtbot.addWidget(w)
    w.show()
    return w


def lvm(path, n=200):
    path.write_text("".join(f"{i / 20000:.6f}\t{np.sin(i / 7):.6f}\t{i % 5}\t0\n" for i in range(n)))
    return path


def test_window_opens_with_two_workspaces(win):
    assert [win.ws_tabs.tabText(i) for i in range(win.ws_tabs.count())] == ["Analyse", "Live"]
    win.ws_tabs.setCurrentIndex(1)
    assert win.workspace() == "Live"
    assert win.centre.currentWidget() is win.live_view and win.hardware_dock.isVisible() and not win.tree_dock.isVisible()
    assert win.seg_device.text() == "No device" and win.seg_output.text() == "Output off"


def test_open_lvm_fills_tree(win, qtbot, tmp_path):
    win.open_path(lvm(tmp_path / "run.lvm"))
    qtbot.waitUntil(lambda: len(win.tree.channel_items()) == 3, timeout=5000)
    assert [i.text(0) for i in win.tree.channel_items()] == ["X", "Y", "Z"]
    assert win.seg_rate.text() == "20000 Hz"
    assert win.settings["recent"][0].endswith("run.lvm")
    assert win.recent_menu.actions()[0].text() == "run.lvm"


def test_open_session_folder(win, qtbot, tmp_path, make_session):
    win.open_path(make_session(tmp_path / "session_x"))
    qtbot.waitUntil(lambda: win.rec is not None, timeout=5000)
    tops = [win.tree.topLevelItem(0).child(i).text(0) for i in range(win.tree.topLevelItem(0).childCount())]
    assert tops == ["Signals", "Decisions", "Output", "Actions"]


def test_open_bad_file_shows_message(win, qtbot, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a: calls.append(a))
    bad = tmp_path / "broken.lvm"
    bad.write_text("this is not a recording\n")
    win.open_path(bad)
    qtbot.waitUntil(lambda: len(calls) > 0, timeout=5000)
    assert "broken.lvm" in calls[0][2]
    assert win.rec is None and not win.busy.isVisible()


def test_missing_recent_file_is_forgotten(win, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a: calls.append(a))
    gone = str(tmp_path / "gone.lvm")
    win.settings["recent"] = [gone]
    win.open_path(gone)
    assert calls and "gone.lvm" in calls[0][2]
    assert win.settings["recent"] == []


def test_load_does_not_block(win, qtbot, monkeypatch):
    """A slow reader runs on a worker thread: the GUI's event loop keeps ticking meanwhile (review focus 1)."""
    def slow_read(path):
        time.sleep(1.0)
        return Recording([Channel("X", "V", np.zeros(100, np.float32))], 1000.0)
    monkeypatch.setattr(loader.registry, "read", slow_read)
    ticks = []
    timer = QTimer()
    timer.timeout.connect(lambda: ticks.append(1))
    timer.start(50)
    started = time.perf_counter()
    win.open_path(__file__)                            # exists; the patched reader ignores the content
    assert time.perf_counter() - started < 0.2         # open_path returned at once
    assert win.busy.isVisible()
    qtbot.waitUntil(lambda: win.rec is not None, timeout=5000)
    timer.stop()
    assert len(ticks) >= 10                            # ~20 ticks in 1 s; a frozen loop would give 0–1


def test_theme_switch_from_menu(win, qapp):
    win.theme_actions["night"].trigger()
    assert win.settings["theme"] == "night"
    assert qapp.palette().window().color().name().upper() == "#121417"
    win.theme_actions["day"].trigger()
    assert win.settings["theme"] == "day"
    assert win.theme_actions["day"].isChecked()


def test_layout_saved_per_workspace(win):
    win.ws_tabs.setCurrentIndex(1)
    assert "Analyse" in win.settings["layouts"]
    win.close()
    assert "Live" in win.settings["layouts"]
