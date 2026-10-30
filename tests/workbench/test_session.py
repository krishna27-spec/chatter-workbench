import math
from pathlib import Path

import numpy as np
import pytest

from workbench.core import registry
from workbench.core.formats.session import read_session

ROOT = Path(__file__).resolve().parents[2]
def test_signals_from_raw_lvm(tmp_path, make_session):
    rec = read_session(make_session(tmp_path / "s"))
    assert [c.name for c in rec.channels] == ["X", "Y", "Z"]
    assert rec.fs == 20000.0 and rec.n_samples == 200
    assert rec.meta["session"]["args"]["fs"] == 20000


def test_state_mapping(tmp_path, make_session):
    rec = read_session(make_session(tmp_path / "s"))
    state = rec.tables["decisions"]["state"]
    assert state[:4].tolist() == [0, 1, 2, 3]       # no cutting / stable / stable but one model says chatter / chatter
    assert math.isnan(state[4])                     # blank vote (calibrating)
    assert rec.tables["decisions"]["vote"][4] == ""
    assert rec.tables["decisions"]["level_ratio"].dtype == np.float64


def test_output_table(tmp_path, make_session):
    out = read_session(make_session(tmp_path / "s")).tables["output"]
    assert sorted(out) == ["feed_V", "rpm_cmd", "spindle_V", "t", "writing"]
    assert out["spindle_V"].tolist()[3] == 1.6
    assert out["writing"].tolist() == [0, 0, 1, 1, 0]


def test_events_both_time_forms(tmp_path, make_session):
    ev = read_session(make_session(tmp_path / "s")).events
    assert ev[0].wall == "03:22:40" and ev[0].text.startswith("OUTPUT STARTED")
    assert ev[0].t == pytest.approx(0.0768)          # placed on the recording clock through control.csv's wall_time
    assert ev[1].t == pytest.approx(0.09) and ev[1].wall is None
    assert ev[1].text == "chatter at  8167.7 Hz -> no change"


def test_wall_time_without_control_stays_untimed(tmp_path, make_session):
    ev = read_session(make_session(tmp_path / "s", control=False)).events
    assert ev[0].t is None and ev[0].wall == "03:22:40"


def test_missing_parts_still_open(tmp_path, make_session):
    rec = read_session(make_session(tmp_path / "s", control=False, events=False, meta=False))
    assert rec.n_samples == 200
    assert rec.tables == {} and rec.events == []
    for part in ("control.csv", "events.txt", "session.json"):
        assert part in rec.meta["notice"]


def test_signals_npy_preferred(tmp_path, make_session):
    folder = make_session(tmp_path / "s", raw=False)
    np.save(folder / "signals.npy", np.arange(12, dtype=np.float32).reshape(3, 4))
    rec = read_session(folder)
    assert rec.n_samples == 4 and rec.fs == 20000.0
    assert rec.channels[1].data.tolist() == [4, 5, 6, 7]


def test_not_a_session_folder(tmp_path):
    (tmp_path / "x").mkdir()
    with pytest.raises(ValueError, match="raw.lvm or signals.npy"):
        read_session(tmp_path / "x")


def test_registry_reads_folder(tmp_path, make_session):
    folder = make_session(tmp_path / "s")
    assert registry.read(folder).tables["decisions"]["state"][0] == 0


@pytest.mark.slow
def test_real_session_opens():
    folder = ROOT / "data" / "recordings" / "session_20261023_032238"
    if not folder.exists():
        pytest.skip("real session not on this machine")
    rec = read_session(folder)
    assert [c.name for c in rec.channels] == ["X", "Y", "Z"] and rec.fs == 20000.0
    assert len(rec.tables["decisions"]["t"]) > 0
    assert all(e.t is not None for e in rec.events)  # the OUTPUT STARTED line gets its time from control.csv
