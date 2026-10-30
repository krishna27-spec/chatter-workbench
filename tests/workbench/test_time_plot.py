import numpy as np
import pytest

from workbench.app.panels.time_plot import TimePlot
from workbench.app.theme import TOKENS
from workbench.core.formats.session import read_session
from workbench.core.recording import Channel, Event, Recording


def make_plot(qtbot, rec, width=800):
    p = TimePlot(TOKENS["day"])
    qtbot.addWidget(p)
    p.resize(width, 500)
    p.show()
    qtbot.waitExposed(p)
    p.set_recording(rec)
    p.fit()
    return p


def xyz(n=20000, fs=20000.0):
    rng = np.random.default_rng(0)
    return Recording([Channel(c, "V", rng.normal(size=n).astype(np.float32)) for c in "XYZ"], fs)


def test_lanes_follow_data(qtbot, tmp_path, make_session):
    p = make_plot(qtbot, xyz())
    assert len(p.lanes) == 3 and p.state_plot is None and p.output_plot is None
    p2 = make_plot(qtbot, read_session(make_session(tmp_path / "s")))
    assert len(p2.lanes) == 3 and p2.state_plot is not None and p2.output_plot is not None


def test_overlaid_one_lane(qtbot):
    p = make_plot(qtbot, xyz())
    p.set_overlaid(True)
    assert len(p.lanes) == 1 and list(p.lanes[0].curves) == ["X", "Y", "Z"]
    p.set_overlaid(False)
    assert len(p.lanes) == 3


@pytest.mark.parametrize("width", [800, 1600])
def test_spike_visible_when_zoomed_out(qtbot, width):
    x = np.zeros(12_000_000, np.float32)               # 10 min at 20 kHz
    x[7_654_321] = 5.0
    p = make_plot(qtbot, Recording([Channel("X", "V", x)], 20000.0), width)
    _, y = p.lanes[0].curves["X"].getData()
    assert len(y) < 2 * width + 2                      # drawn as an envelope, not 12 M points
    assert y.max() == 5.0


def test_raw_samples_when_zoomed_in(qtbot):
    rec = xyz()
    p = make_plot(qtbot, rec)
    p.set_view(1000 / rec.fs, 1099 / rec.fs)          # samples 1000 … 1099
    t, y = p.lanes[0].curves["X"].getData()
    assert np.array_equal(y, rec.channel("X").data[999:1101].astype(np.float64))   # real samples, one extra each side
    assert np.sum((t >= 1000 / rec.fs - 1e-12) & (t <= 1099 / rec.fs + 1e-12)) == 100


def test_cursor_values_are_nearest_samples(qtbot):
    x = np.arange(1000, dtype=np.float32)
    p = make_plot(qtbot, Recording([Channel("X", "V", x)], 100.0))
    p.set_cursors(True)
    p.move_cursor(0, 1.234)                            # nearest sample is 123 (t = 1.23)
    p.move_cursor(1, 5.006)                            # nearest is 501 (t = 5.01)
    v = p.cursor_values()
    assert v["X"] == (123.0, 501.0)
    assert v["dt"] == pytest.approx(3.772)
    assert "Δt" in p.readout.text() and p.readout.isVisible()


def test_markers_only_for_timed_events(qtbot):
    rec = xyz()
    rec.events = [Event(0.25, "chatter"), Event(None, "OUTPUT STARTED", wall="03:22:51"), Event(0.5, "stop")]
    p = make_plot(qtbot, rec)
    assert len(p.markers) == 2 * 3                      # 2 timed events × 3 lanes
    assert sorted({m.value() for m in p.markers}) == [0.25, 0.5]


def test_hidden_channel_removes_lane(qtbot):
    p = make_plot(qtbot, xyz())
    p.set_visible_channels(["X", "Z"])
    assert [l.names for l in p.lanes] == [["X"], ["Z"]]


def test_auto_y_fits_the_visible_data(qtbot):
    x = np.zeros(40_000, np.float32)
    x[:20_000] = np.sin(np.arange(20_000) / 5) * 0.1        # small signal in the first half
    x[39_000] = -10.0                                        # one big transient near the end
    p = make_plot(qtbot, Recording([Channel("X", "V", x)], 20000.0))
    p.set_view(0.0, 0.9)
    lo, hi = p.y_range()
    assert -0.2 < lo < -0.09 and 0.09 < hi < 0.2             # fits the ±0.1 signal, not the −10 V spike
    p.fit()
    assert p.y_range()[0] < -10.0                            # the spike is in view → included (nothing hidden)
    p.set_auto_y(False)
    p.set_view(0.0, 0.9)
    assert p.y_range()[0] < -10.0                            # off: whole-file range


def test_output_lane_ticks_only_used_levels(qtbot, tmp_path, make_session):
    p = make_plot(qtbot, read_session(make_session(tmp_path / "s")))
    ticks = p.output_plot.getAxis("left")._tickLevels[0]
    assert [round(v, 4) for v, _ in ticks] == [1.4286, 1.6]
