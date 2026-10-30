"""The Live workspace, driven like a user would, on a replayed file (no hardware needed) and, where present, the
simulated NI chassis for the output."""
import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMessageBox

from workbench.app.window import MainWindow
from workbench.core import registry
from workbench.core.live.hardware import Module, list_modules
from workbench.core.settings import DEFAULTS

FS = 20_000.0
REAL_SIM = [m for m in list_modules()[0] if m.simulated]
HAS_SIM_AO = any(m.ao for m in REAL_SIM)


def fake_modules():
    return [Module("cDAQ9Mod1", "NI 9252", "cDAQ9", 1, False, [f"cDAQ9Mod1/ai{i}" for i in range(8)], []),
            Module("cDAQ9Mod2", "NI 9263", "cDAQ9", 2, False, [], [f"cDAQ9Mod2/ao{i}" for i in range(4)]),
            Module("cDAQ8Mod1", "NI 9775", "cDAQ8", 1, True, [f"cDAQ8Mod1/ai{i}" for i in range(4)], [])], None


def make_window(qtbot, tmp_path, lister):
    registry.load_formats()
    w = MainWindow({**DEFAULTS, "recent": [], "layouts": {}}, lister=lister)
    w.record_root = tmp_path / "rec"
    qtbot.addWidget(w)
    w.show()
    f = tmp_path / "Acce_20k_live.lvm"
    t = np.arange(int(6 * FS)) / FS
    y = 0.05 * np.sin(2 * np.pi * 4100 * t) + 0.01 * np.random.default_rng(0).normal(size=t.size)
    f.write_text("".join(f"{a:.6f}\t{b:.6f}\t{b:.6f}\t{b:.6f}\n" for a, b in zip(t, y)))
    w.open_path(f)
    qtbot.waitUntil(lambda: w.rec is not None, timeout=5000)
    return w


@pytest.fixture
def win(qtbot, tmp_path):
    w = make_window(qtbot, tmp_path, fake_modules)
    yield w
    w.live.close()


def go_live(win, qtbot):
    win.ws_tabs.setCurrentIndex(1)
    qtbot.waitUntil(lambda: win.hardware.tree.topLevelItemCount() > 0, timeout=5000)


def replay(win, models=False):
    i = [win.inputs.source.itemData(k) for k in range(win.inputs.source.count())].index("__replay__")
    win.inputs.source.setCurrentIndex(i)
    win.inputs.models.setChecked(models)


def test_live_lists_every_module(win, qtbot):
    go_live(win, qtbot)
    assert win.hardware_dock.isVisible() and win.output_dock.isVisible() and not win.spectrum_dock.isVisible()
    assert win.hardware.tree.topLevelItemCount() == 2                      # two chassis
    inputs = [win.inputs.source.itemText(i) for i in range(win.inputs.source.count())]
    assert inputs == ["cDAQ9Mod1  NI 9252", "cDAQ8Mod1  NI 9775  (simulated)", "Replay the file open in Analyse"]
    assert win.inputs.source.currentData() == "cDAQ9Mod1"                  # the suggestion: NI 9252, never simulated
    assert [win.inputs.pins.item(r, 1).text() for r in range(3)] == ["X", "Y", "Z"]
    outputs = [win.outputs.module.itemText(i) for i in range(win.outputs.module.count())]
    assert outputs == ["No output (watch only)", "cDAQ9Mod2  NI 9263"]
    assert win.outputs.module.currentData() == "cDAQ9Mod2"
    roles = [win.outputs.pins.cellWidget(r, 1).currentText() for r in range(4)]
    assert roles == ["spindle", "feed", "manual", "manual"]


def test_choosing_pins(win, qtbot):
    go_live(win, qtbot)
    win.inputs.check_pins(["cDAQ9Mod1/ai2", "cDAQ9Mod1/ai5"])
    assert win.inputs.choice().pins == ["cDAQ9Mod1/ai2", "cDAQ9Mod1/ai5"]
    win.outputs.set_pin("cDAQ9Mod2/ao1", True, "feed")
    win.outputs.set_pin("cDAQ9Mod2/ao3", True, "manual")
    out = win.outputs.choice()
    assert [(p.pin, p.role) for p in out.pins] == [("cDAQ9Mod2/ao0", "spindle"), ("cDAQ9Mod2/ao1", "feed"),
                                                   ("cDAQ9Mod2/ao3", "manual")]


def test_replay_runs_and_fills_the_dashboard(win, qtbot):
    go_live(win, qtbot)
    replay(win)
    win.outputs.module.setCurrentIndex(0)                                  # watch only
    win._live_start()
    assert win.live.running and not win.inputs.start_btn.isEnabled()
    qtbot.waitUntil(lambda: win.live.engine.t_now() > 1.0, timeout=5000)
    qtbot.wait(300)
    assert len(win.live_view._lanes) == 3
    assert "acquisition started" in win.live_view.log.toPlainText()
    assert win.live_view.cards["dom"][0].text().startswith("41")            # the 4100 Hz tone
    win._live_stop()
    assert not win.live.running and win.inputs.start_btn.isEnabled()


def test_replay_cannot_drive_a_real_module(win, qtbot, monkeypatch):
    go_live(win, qtbot)
    replay(win)
    msgs = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a: msgs.append(a[2]))
    win._live_start()                                                      # output = the (fake) real NI 9263
    assert not win.live.running and "cannot drive a real output" in msgs[0]


def test_record_button_makes_a_session_that_opens(win, qtbot, tmp_path):
    go_live(win, qtbot)
    replay(win)
    win.outputs.module.setCurrentIndex(0)
    win._live_start()
    qtbot.waitUntil(lambda: win.live.engine.t_now() > 0.3, timeout=5000)
    win.live_view.record_btn.setChecked(True)
    qtbot.waitUntil(lambda: (win.live.engine.snapshot()["rec_seconds"] or 0) > 0.5, timeout=5000)
    win.live_view.record_btn.setChecked(False)
    win._live_stop()
    folder = next((tmp_path / "rec").iterdir())
    rec = registry.read(folder)
    assert rec.n_samples >= 0.5 * FS and [c.name for c in rec.channels] == ["X", "Y", "Z"]


@pytest.mark.skipif(not HAS_SIM_AO, reason="no simulated NI output module on this machine")
def test_arm_confirm_start_and_stop_from_analyse(qtbot, tmp_path, monkeypatch):
    w = make_window(qtbot, tmp_path, list_modules)
    try:
        go_live(w, qtbot)
        replay(w)
        sim = next(m for m in REAL_SIM if m.ao)
        names = [w.outputs.module.itemData(i) for i in range(w.outputs.module.count())]
        w.outputs.module.setCurrentIndex(names.index(sim.name))            # chosen on purpose: simulated
        w._live_start()
        assert w.live.running and w.live.guard is not None
        monkeypatch.setattr(QMessageBox, "question", lambda *a: QMessageBox.No)
        w._live_arm()
        assert w.live.guard.state == "disarmed"                           # no confirm, no output
        asked = []
        monkeypatch.setattr(QMessageBox, "question", lambda *a: (asked.append(a[2]), QMessageBox.Yes)[1])
        w._live_arm()
        assert "SIMULATED" in asked[0] and w.live.guard.state == "armed"
        w._live_start_output()
        qtbot.waitUntil(lambda: w.stop_out_btn.isVisible(), timeout=2000)
        assert w.live.guard.state == "writing" and "WRITING" in w.seg_output.text()
        w.ws_tabs.setCurrentIndex(0)                                      # Analyse: Stop output still works
        assert w.stop_out_btn.isVisible()
        qtbot.keyClick(w, Qt.Key_Escape)
        assert w.live.guard.state == "disarmed"
        w._live_changed()
        assert not w.stop_out_btn.isVisible() and w.seg_output.text() == "Output off"
    finally:
        w.live.close()


def test_start_with_models_does_not_freeze_the_window(win, qtbot):
    """Loading the three models takes ~0.3 s; it happens on the engine's analysis thread, not the GUI thread."""
    import time
    go_live(win, qtbot)
    replay(win, models=True)
    win.outputs.module.setCurrentIndex(0)
    t = time.perf_counter()
    win._live_start()
    assert time.perf_counter() - t < 0.2
    qtbot.waitUntil(lambda: win.live.engine.models is not None, timeout=10_000)
    i = [win.live_view.scale.itemData(k) for k in range(win.live_view.scale.count())]
    win.live_view.scale.setCurrentIndex(i.index("amplitude"))
    qtbot.wait(600)
    y = win.live_view.spec_curve.getData()[1]
    assert y is not None and y.min() >= 0 and "amplitude" in win.live_view.spec.getAxis("left").labelText
    win._live_stop()
