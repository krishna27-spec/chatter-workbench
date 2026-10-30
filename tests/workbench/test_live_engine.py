import time
from pathlib import Path

import numpy as np
import pytest

from workbench.core import registry
from workbench.core.live.engine import LiveEngine
from workbench.core.live.hardware import list_modules
from workbench.core.live.output import MemoryWriter, NiWriter, OutputGuard, PinConfig, SpindleConfig
from workbench.core.live.recorder import SessionRecorder
from workbench.core.live.sources import NiSource, ReplaySource
from workbench.core.recording import Channel, Recording

ROOT = Path(__file__).resolve().parents[2]
LAB = ROOT / "data" / "raw" / "Acce_20k_1_r1.lvm"
FS = 20_000.0


def noise_rec(seconds=6.0):
    rng = np.random.default_rng(3)
    return Recording([Channel(c, "V", (0.01 * rng.normal(size=int(seconds * FS))).astype(np.float32)) for c in "XYZ"],
                     FS, source=Path("noise.lvm"))


def wait_for(cond, timeout=20.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.02)
    return False


# ---------------------------------------------------------------- recorder
def test_recording_opens_in_analyse(tmp_path):
    registry.load_formats()
    rng = np.random.default_rng(0)
    blocks = [rng.normal(size=(3, 512)).astype(np.float32) for _ in range(5)]
    r = SessionRecorder(tmp_path / "wb_x", ["X", "Y", "Z"], ["V"] * 3, FS, {"pins": ["M/ai0", "M/ai1", "M/ai2"]})
    for i, b in enumerate(blocks):
        r.signals(b)
        r.decision({"t": (i + 1) * 0.0256, "state": 3, "vote": "chatter", "extra_trees": "chatter",
                    "random_forest": "chatter", "svm_y": "stable"})
    r.output({"t": 0.05, "rpm_cmd": 20000, "writing": 1, "why": "command", "spindle_V": 1.4286, "feed_V": 2.0})
    r.action(0.06, "user", "start output")
    r.close()
    rec = registry.read(tmp_path / "wb_x")
    assert [c.name for c in rec.channels] == ["X", "Y", "Z"] and rec.fs == FS
    assert np.array_equal(np.stack([c.data for c in rec.channels]), np.concatenate(blocks, 1))   # every bit
    assert rec.tables["decisions"]["state"].tolist() == [3] * 5
    assert rec.tables["output"]["spindle_V"].tolist() == [1.4286]
    assert "user: start output" in [e.text for e in rec.events]
    lvm = registry.read(tmp_path / "wb_x" / "signals.lvm")
    assert np.array_equal(lvm.channel("Y").data, rec.channel("Y").data)                  # 9 digits: exact float32


def test_a_crash_loses_at_most_the_unflushed_part(tmp_path):
    r = SessionRecorder(tmp_path / "wb_c", ["X"], ["V"], FS, {})
    r.signals(np.ones((1, 512), np.float32))
    r.flush()
    r.signals(np.ones((1, 512), np.float32))           # written but header not yet updated: "crash" now
    r._npy.flush()
    x = np.load(tmp_path / "wb_c" / "signals.npy")
    assert x.shape == (1, 512)                          # the flushed part opens as is


# ---------------------------------------------------------------- engine
def test_replay_end_to_end_with_output(tmp_path):
    w = MemoryWriter()
    g = OutputGuard([PinConfig("Sim/ao0", "spindle"), PinConfig("Sim/ao1", "feed")], SpindleConfig(start_rpm=20_000), w)
    eng = LiveEngine(ReplaySource(noise_rec(), speed=4.0), guard=g)
    eng.start()
    assert wait_for(lambda: eng.ring.total >= 20_000)
    g.arm()
    g.start()
    folder = eng.start_recording(tmp_path / "wb_e")
    assert wait_for(lambda: eng.snapshot()["rec_seconds"] > 0.5)
    eng.stop()
    assert w.writes[0] == pytest.approx([20_000 / 14_000, 2.0])          # arm: safe values
    assert g.state == "disarmed" and w.closed                           # stopping acquisition stops output
    rec = registry.read(folder)
    assert rec.n_samples >= 10_000 and "output" in rec.tables
    assert any("output off" in e.text for e in rec.events)


def test_replay_cannot_drive_a_real_module():
    g = OutputGuard([PinConfig("cDAQ2Mod3/ao0", "spindle")], SpindleConfig(), NiWriter())
    with pytest.raises(ValueError, match="replayed file cannot drive a real output"):
        LiveEngine(ReplaySource(noise_rec()), guard=g)


def test_acquisition_error_stops_output():
    class Dying(ReplaySource):
        def read(self):
            if self._sent > 5 * 512:
                raise OSError("USB cable out")
            return super().read()
    w = MemoryWriter()
    g = OutputGuard([PinConfig("Sim/ao0", "spindle")], SpindleConfig(), w)
    eng = LiveEngine(Dying(noise_rec(), realtime=False), guard=g)
    g.arm()
    g.start()
    eng.start()
    assert wait_for(lambda: not eng.running, 5)
    assert "USB cable out" in eng.error and g.state == "disarmed" and w.closed


@pytest.mark.slow
@pytest.mark.skipif(not LAB.exists(), reason="lab file not on this machine")
def test_models_match_realtime_py():
    """The workbench's Top3 gives the same labels as realtime.Top3Predictor, window by window."""
    from rig import realtime
    from workbench.core.live.top3 import Top3
    rec = registry.read(LAB) if registry._readers else (registry.load_formats(), registry.read(LAB))[1]
    x = np.stack([c.data for c in rec.channels]).astype(np.float64)[:, : 20 * 20_000]
    ours, ref = Top3(), realtime.Top3Predictor()
    n_same = n = 0
    for i in range(x.shape[1] // 512):
        a, b = ours.update(x[:, i * 512:(i + 1) * 512]), ref.update(x[:, i * 512:(i + 1) * 512])
        assert (a is None) == (b is None)
        if a is not None:
            n += 1
            n_same += a == b
            assert np.allclose([ours.last_probs[m] for m in ours.models], [ref.last_probs[m] for m in ours.models])
    assert n > 600 and n_same == n


# ---------------------------------------------------------------- the simulated NI chassis
def sim_modules():
    mods, _ = list_modules()
    return [m for m in mods if m.simulated]


@pytest.mark.skipif(not any(m.ai for m in sim_modules()), reason="no simulated NI input module")
def test_ni_source_on_the_simulated_chassis():
    m = next(m for m in sim_modules() if len(m.ai) >= 3)
    src = NiSource(m.ai[:3], fs=FS)
    src.start()
    try:
        a, b = src.read(), src.read()
    finally:
        src.stop()
    assert a.shape == b.shape == (3, 512) and src.actual_fs > 0


@pytest.mark.skipif(not any(m.ao for m in sim_modules()), reason="no simulated NI output module")
def test_ni_writer_on_the_simulated_chassis():
    m = next(m for m in sim_modules() if m.ao)
    g = OutputGuard([PinConfig(m.ao[0], "spindle"), PinConfig(m.ao[1], "feed")], SpindleConfig(), NiWriter())
    g.arm()
    g.start()
    g.set_rpm(25_000)
    g.stop()
    assert g.state == "disarmed" and g.writer.task is None
