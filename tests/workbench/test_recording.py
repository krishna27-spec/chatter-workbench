import numpy as np
import pytest

from workbench.core.recording import Channel, Event, Recording


def make(n=20000, fs=20000.0, names=("X", "Y", "Z")):
    rng = np.random.default_rng(0)
    return Recording([Channel(c, "V", rng.standard_normal(n).astype(np.float32)) for c in names], fs)


def test_basic_properties():
    r = make()
    assert r.n_samples == 20000
    assert r.duration == 1.0
    assert r.time()[-1] == pytest.approx(0.99995)
    assert r.time().dtype == np.float64


def test_slice_is_a_view():
    r = make()
    s = r.slice(0.25, 0.5)
    assert s.channels[0].data.base is not None          # shares memory with r, nothing copied
    assert s.n_samples == 5000
    assert s.t0 == pytest.approx(0.25)
    assert s.time()[0] == pytest.approx(0.25)


def test_slice_keeps_tables_and_events():
    r = make()
    r.events.append(Event(0.3, "chatter"))
    r.tables["decisions"] = {"t": np.array([0.1, 0.2])}
    s = r.slice(0.25, 0.5)
    assert s.events == r.events and s.tables is r.tables


@pytest.mark.parametrize("build, message", [
    (lambda: Recording([Channel("X", "V", np.zeros(10))], 0.0), "fs"),
    (lambda: Recording([Channel("X", "V", np.zeros(10)), Channel("Y", "V", np.zeros(9))], 1.0),
     "channel 'Y' has 9 samples, expected 10"),
    (lambda: Recording([Channel("X", "V", np.zeros((2, 5)))], 1.0), "1-D"),
    (lambda: Recording([], 1.0), "at least one channel"),
])
def test_rejects_bad_input(build, message):
    with pytest.raises(ValueError, match=message):
        build()


def test_channel_lookup():
    r = make()
    assert r.channel("Z").name == "Z"
    with pytest.raises(KeyError, match="X, Y, Z"):
        r.channel("Q")


def test_index_of_is_clipped():
    r = make()
    assert r.index_of(-1.0) == 0
    assert r.index_of(0.5) == 10000
    assert r.index_of(99.0) == r.n_samples
