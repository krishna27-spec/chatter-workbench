import numpy as np
import pytest

from workbench.core import steps
from workbench.core.recording import Channel, Event, Recording
from workbench.core.steps import OPS, Step, op


@pytest.fixture
def scaler():
    """A throwaway op: multiply every channel by k, counting how often it really runs."""
    calls = []

    @op("_scale", "Scale", "multiply by k", k=2.0)
    def scale(rec, k):
        calls.append(k)
        return steps.map_channels(rec, lambda x: x * k)
    yield calls
    OPS.pop("_scale")


def rec3():
    return Recording([Channel(c, "V", np.arange(5, dtype=np.float32) + i) for i, c in enumerate("XYZ")], 10.0,
                     t0=1.5, tables={"decisions": {"t": np.array([2.0])}}, events=[Event(2.0, "x")])


def test_root_step_is_the_file():
    r = rec3()
    root = Step.root(r)
    assert root.output() is r and root.title == "File"


def test_op_output_keeps_time_and_tables(scaler):
    r = rec3()
    s = Step("_scale", input=Step.root(r))
    out = s.output()
    assert np.array_equal(out.channel("Y").data, (np.arange(5) + 1) * 2.0)
    assert out.channel("Y").data.dtype == np.float32
    assert (out.fs, out.t0, [c.name for c in out.channels]) == (10.0, 1.5, ["X", "Y", "Z"])
    assert out.tables is r.tables and out.events is r.events
    assert np.array_equal(r.channel("Y").data, np.arange(5) + 1)          # the original is never changed


def test_cached_until_params_change(scaler):
    s = Step("_scale", input=Step.root(rec3()))
    s.output(); s.output()
    assert scaler == [2.0]                                                 # computed once
    s.set_params(k=3.0)
    assert not s.is_cached
    assert s.output().channel("X").data[1] == 3.0 and scaler == [2.0, 3.0]


def test_change_clears_steps_below(scaler):
    a = Step("_scale", input=Step.root(rec3()))
    b = Step("_scale", {"k": 10.0}, input=a)
    assert b.output().channel("X").data[1] == 20.0
    a.set_params(k=1.0)
    assert not b.is_cached
    assert b.output().channel("X").data[1] == 10.0


def test_unknown_op_and_param_are_refused(scaler):
    with pytest.raises(KeyError, match="no op"):
        Step("nope", input=Step.root(rec3()))
    with pytest.raises(TypeError, match="k"):
        Step("_scale", {"q": 1}, input=Step.root(rec3()))


def test_ops_load_from_the_package():
    assert steps.load_ops() == []
    assert "clean" in OPS and OPS["clean"].title == "Clean"
