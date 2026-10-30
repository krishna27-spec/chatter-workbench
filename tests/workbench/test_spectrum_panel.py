from pathlib import Path

import numpy as np
import pytest

from workbench.app.analysis import AnalysisSession
from workbench.app.panels.spectrum import SpectrumPanel
from workbench.app.panels.time_plot import TimePlot
from workbench.app.theme import TOKENS
from workbench.app.window import MainWindow
from workbench.core import registry
from workbench.core.analysis.spectrum import spectrogram
from workbench.core.recording import Channel, Recording
from workbench.core.settings import DEFAULTS

FS = 20_000.0


def tones(*freqs, seconds=4.0, name="Acce_20k_test.lvm"):
    rng = np.random.default_rng(5)
    t = np.arange(int(seconds * FS)) / FS
    y = sum(0.3 * np.sin(2 * np.pi * f * t) for f in freqs) + 0.005 * rng.normal(size=t.size)
    chans = [Channel(c, "V", (y if c == "Y" else 0.01 * rng.normal(size=t.size)).astype(np.float32)) for c in "XYZ"]
    return Recording(chans, FS, source=Path(name))


@pytest.fixture
def setup(qtbot):
    def make(rec):
        plot = TimePlot(TOKENS["day"])
        panel = SpectrumPanel(TOKENS["day"])
        for w in (plot, panel):
            qtbot.addWidget(w)
            w.resize(900, 400)
            w.show()
        plot.set_recording(rec)
        plot.fit()
        plot.regions_changed.connect(panel.regions_changed)
        panel.set_session(AnalysisSession(rec), rec, plot)
        return plot, panel
    return make


def test_region_gives_a_curve_and_its_peak(setup, qtbot):
    plot, panel = setup(tones(2047.0))
    with qtbot.waitSignal(panel.computed, timeout=3000):
        plot.add_region(t0=0.5, t1=3.5)
    assert panel.channel.currentText() == "Y"                       # the models' axis by default
    (r, spec), = panel.results
    assert r.name == "chatter" and spec.n_avg > 20
    assert abs(panel.peak_rows[0][0].f - 2047.0) < 0.5


def test_presets_names_and_colours(setup):
    plot, _ = setup(tones(1000.0))
    names = [plot.add_region(t0=i, t1=i + 0.5).name for i in range(4)]
    assert names == ["chatter", "stable", "idle", "region 4"]
    assert [plot.region_colour(r) for r in plot.regions()[:3]] == [TOKENS["day"][k] for k in ("s3", "s1", "ink2")]


def test_peaks_classified_against_harmonics(setup, qtbot):
    plot, panel = setup(tones(8167.7, 1000.0))                      # rpm 20 000 from the file name
    with qtbot.waitSignal(panel.computed, timeout=3000):
        plot.add_region(t0=0.0, t1=4.0)
    assert panel.rpm.value() == 20_000 and "file name" in panel.rpm_src.text()
    rows = {round(p.f): c for p, c in panel.peak_rows}
    assert rows[8168].kind == "between" and rows[8168].n == 12 and rows[8168].nu == pytest.approx(0.25, abs=0.01)
    assert rows[1000].kind == "harmonic" and rows[1000].k == 3
    assert panel.peak_table.item(0, 7).text() in ("between", "harmonic")


def test_typed_rpm_wins_until_auto(setup, qtbot):
    plot, panel = setup(tones(1000.0))
    plot.add_region(t0=0.0, t1=2.0)
    panel.rpm.setValue(30_000)
    assert panel.rpm_src.text() == "typed"
    with qtbot.waitSignal(panel.computed, timeout=3000):
        panel.rpm_auto.click()
    assert panel.rpm.value() == 20_000


def test_clean_source_is_computed_in_the_background(setup, qtbot):
    plot, panel = setup(tones(3000.0))
    with qtbot.waitSignal(panel.computed, timeout=3000):
        plot.add_region(t0=1.0, t1=3.0)
    with qtbot.waitSignal(panel.computed, timeout=10_000):
        panel.source.setCurrentIndex(1)                            # Clean
    assert panel.analysis.ready("clean") is not None
    assert abs(panel.peak_rows[0][0].f - 3000.0) < 0.5            # cleaning keeps the cutting band


def test_short_region_says_why(setup, qtbot):
    plot, panel = setup(tones(1000.0))
    with qtbot.waitSignal(panel.computed, timeout=3000):
        plot.add_region(t0=1.0, t1=1.005)                          # 100 samples
    assert panel.results == [] and "too short" in panel.message.text()


def test_moving_a_region_recomputes(setup, qtbot):
    plot, panel = setup(tones(1000.0))
    with qtbot.waitSignal(panel.computed, timeout=3000):
        r = plot.add_region(t0=0.0, t1=1.0)
    n_before = panel.results[0][1].n_avg
    with qtbot.waitSignal(panel.computed, timeout=3000):
        plot.update_region(r, t1=3.0)
    assert panel.results[0][1].n_avg > n_before


def test_hidden_region_has_no_curve(setup, qtbot):
    plot, panel = setup(tones(1000.0))
    plot.add_region(t0=0.0, t1=1.0)
    with qtbot.waitSignal(panel.computed, timeout=3000):
        r2 = plot.add_region(t0=2.0, t1=3.0)
    assert len(panel.results) == 2
    with qtbot.waitSignal(panel.computed, timeout=3000):
        plot.update_region(r2, visible=False)
    assert [r.name for r, _ in panel.results] == ["chatter"]


def test_spectrogram_lane_peak_hold(qtbot):
    x = 0.01 * np.random.default_rng(1).normal(size=1_200_000).astype(np.float32)     # 60 s
    t = np.arange(1024) / FS
    x[600_000:601_024] += np.sin(2 * np.pi * 5000 * t).astype(np.float32)             # one 51 ms burst
    rec = Recording([Channel("Y", "V", x)], FS)
    plot = TimePlot(TOKENS["night"])
    qtbot.addWidget(plot)
    plot.resize(900, 500)
    plot.show()
    plot.set_recording(rec)
    plot.set_spectrogram(spectrogram(x, FS), rec.t0, "Y raw [Hz]")
    plot.fit()
    img = plot._sg_image.image
    assert plot.sg_plot is not None and img.shape[0] <= 900 * 2
    k = round(5000 / (FS / 1024))
    assert img[:, k].max() - np.median(img[:, k]) > 30           # the burst survives 2300 frames → ~800 columns


@pytest.fixture
def win(qtbot, tmp_path):
    registry.load_formats()
    w = MainWindow({**DEFAULTS, "recent": [], "layouts": {}})
    qtbot.addWidget(w)
    w.show()
    f = tmp_path / "Acce_20k_x.lvm"
    t = np.arange(40_000) / FS
    f.write_text("".join(f"{a:.6f}\t{b:.6f}\t{b:.6f}\t0\n" for a, b in zip(t, 0.3 * np.sin(2 * np.pi * 2000 * t))))
    w.open_path(f)
    qtbot.waitUntil(lambda: w.rec is not None, timeout=5000)
    return w


def test_window_region_key_and_spectrogram_toggle(win, qtbot):
    with qtbot.waitSignal(win.spectrum.computed, timeout=3000):
        win.act_region.trigger()
    assert len(win.plot.regions()) == 1 and win.spectrum_dock.isVisible()
    win.act_sg.trigger()
    qtbot.waitUntil(lambda: win.plot.sg_plot is not None, timeout=5000)
    win.act_sg.trigger()
    assert win.plot.sg_plot is None


def test_typing_an_rpm_never_floods_the_plot(setup, qtbot):
    """Typing 20000 passes through 2, 20, 200 rpm: at 2 rpm the tooth-passing lines up to 10 kHz number 150 000,
    which froze the app for 100 s (2026-10-25). Marks are drawn only when they can be told apart."""
    import time
    plot, panel = setup(tones(1000.0))
    with qtbot.waitSignal(panel.computed, timeout=3000):
        plot.add_region(t0=0.5, t1=3.5)
    for rpm in (2, 20, 200, 2000, 20000):
        t = time.perf_counter()
        with qtbot.waitSignal(panel.computed, timeout=3000):
            panel.rpm.setValue(rpm)
        assert time.perf_counter() - t < 1.0, f"{rpm} rpm took {time.perf_counter() - t:.1f} s"
        assert len(panel._lines) <= 2 * 400
    assert len(panel._lines) == 45                          # 20 000 rpm: 30 spindle + 15 tooth lines up to 10 kHz
    panel.rpm.setValue(2)
    qtbot.waitSignal(panel.computed, timeout=3000)
    qtbot.waitUntil(lambda: "too low" in panel.harm_info.text(), timeout=3000)


def test_linear_scales(setup, qtbot):
    plot, panel = setup(tones(2047.0))                    # amplitude 0.3 V
    with qtbot.waitSignal(panel.computed, timeout=3000):
        plot.add_region(t0=0.5, t1=3.5)
    i = [panel.scale.itemData(k) for k in range(panel.scale.count())]
    assert i == ["db", "psd", "amplitude"]
    with qtbot.waitSignal(panel.computed, timeout=3000):
        panel.scale.setCurrentIndex(i.index("amplitude"))
    y = panel.plot.getPlotItem().listDataItems()[0].getData()[1]
    assert 0.25 < y.max() <= 0.3 and y.min() >= 0               # linear volts, not dB
    assert "amplitude" in panel.plot.getAxis("left").labelText
    assert panel.peak_table.horizontalHeaderItem(2).text() == "amplitude [V]"
    with qtbot.waitSignal(panel.computed, timeout=3000):
        panel.scale.setCurrentIndex(i.index("psd"))
    y = panel.plot.getPlotItem().listDataItems()[0].getData()[1]
    assert y.min() >= 0 and "V²/Hz" in panel.plot.getAxis("left").labelText
