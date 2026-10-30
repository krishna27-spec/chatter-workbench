import json
from pathlib import Path

import numpy as np
import pytest

from workbench.core.live import speed_rule as wb_rule
from workbench.core.live.hardware import Module, list_modules, suggest_input, suggest_output
from workbench.core.live.output import V_LIMIT, MemoryWriter, OutputGuard, PinConfig, SpindleConfig
from workbench.core.live.ring import RingBuffer

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "models" / "chatter_model_v2_chatter3"


# ---------------------------------------------------------------- ring buffer
def test_ring_keeps_the_newest_in_order():
    r = RingBuffer(2, 10)
    for k in range(4):
        r.write(np.vstack([np.arange(k * 4, k * 4 + 4)] * 2))          # 0..15 in blocks of 4
    x, first = r.latest(10)
    assert x[0].tolist() == list(range(6, 16)) and first == 6 and r.total == 16
    x, first = r.latest(3)
    assert x[1].tolist() == [13, 14, 15] and first == 13


def test_ring_block_longer_than_capacity():
    r = RingBuffer(1, 5)
    r.write(np.arange(12)[None])
    x, first = r.latest(5)
    assert x[0].tolist() == [7, 8, 9, 10, 11] and r.total == 12 and first == 7


# ---------------------------------------------------------------- hardware
def mod(name, product, sim=False, ai=0, ao=0, slot=1):
    return Module(name, product, "cDAQ2", slot, sim, [f"{name}/ai{i}" for i in range(ai)],
                  [f"{name}/ao{i}" for i in range(ao)])


def test_suggestions_follow_preferences_and_skip_simulated():
    mods = [mod("M1", "NI 9222", ai=4, slot=1), mod("M2", "NI 9263", ao=4, slot=2),
            mod("M3", "NI 9252", ai=8, slot=3), mod("S1", "NI 9252", sim=True, ai=8, slot=4),
            mod("S2", "NI 9263", sim=True, ao=4, slot=5)]
    assert suggest_input(mods).name == "M3" and suggest_output(mods).name == "M2"
    only_sim = [m for m in mods if m.simulated]
    assert suggest_input(only_sim) is None and suggest_output(only_sim) is None


def test_list_modules_never_raises():
    mods, problem = list_modules()
    assert isinstance(mods, list) and (problem is None or isinstance(problem, str))


# ---------------------------------------------------------------- output guard
def guard(*pins, **sp):
    w = MemoryWriter()
    log = []
    g = OutputGuard(list(pins), SpindleConfig(**sp), w, log=lambda kind, **kw: log.append((kind, kw)))
    return g, w, log


def test_nothing_is_written_while_disarmed():
    g, w, _ = guard(PinConfig("M/ao0", "spindle"))
    g.set_rpm(30_000)
    assert w.writes == [] and w.opened is None and not g.armed


def test_arm_writes_safe_values_start_follows_stop_restores():
    g, w, log = guard(PinConfig("M/ao0", "spindle"), PinConfig("M/ao1", "feed"), start_rpm=20_000)
    safe = g.arm()
    assert w.opened == ["M/ao0", "M/ao1"] and g.state == "armed"
    assert safe == pytest.approx([20_000 / 14_000, 3 * 2 * 20_000 / 60_000 * 5 / 5])   # lab formula: 2.0 V feed
    g.set_rpm(28_000)
    assert len(w.writes) == 1                       # armed: command remembered, not written
    g.start()
    assert w.writes[-1] == pytest.approx([2.0, 2.8])
    g.stop()
    assert w.writes[-1] == pytest.approx(safe) and w.closed and g.state == "disarmed"
    assert ("action", {"what": "output off (stop)"}) in log


def test_stop_is_safe_any_time():
    g, w, _ = guard(PinConfig("M/ao0", "spindle"))
    g.stop()
    g.stop()
    assert w.writes == []
    with pytest.raises(RuntimeError, match="Arm"):
        g.start()


def test_clamps_and_the_10_volt_ceiling():
    g, w, _ = guard(PinConfig("M/ao0", "spindle", v_min=1.0, v_max=12.0), PinConfig("M/ao2", "manual", v_max=3.0))
    g.arm()
    g.start()
    g.set_rpm(500_000)                              # 35.7 V asked
    g.set_manual("M/ao2", 9.0)
    assert w.writes[-1] == [V_LIMIT, 3.0]
    g.set_rpm(0)
    assert w.writes[-1][0] == 1.0
    assert all(abs(v) <= V_LIMIT for row in w.writes for v in row)


def test_slew_limit_walks_the_value():
    g, w, _ = guard(PinConfig("M/ao0", "manual", slew=2.0))
    g.arm()
    g.start()
    g.set_manual("M/ao0", 5.0)
    assert g.current == [0.0]                       # with a slew limit only tick() moves it
    g.tick(1.0)
    assert g.current == [2.0]
    g.tick(1.0)
    g.tick(1.0)
    assert g.current == [5.0]


def test_failed_write_still_closes():
    class Broken(MemoryWriter):
        def write(self, values):
            if len(self.writes) >= 1:
                raise OSError("USB unplugged")
            super().write(values)
    w = Broken()
    g = OutputGuard([PinConfig("M/ao0", "spindle")], SpindleConfig(), w)
    g.arm()
    with pytest.raises(OSError):
        g.start()
        g.set_rpm(30_000)
    with pytest.raises(OSError):
        g.stop("error")
    assert w.closed and g.state == "disarmed"


def test_bad_role_combinations_refused():
    with pytest.raises(ValueError, match="feed pin follows"):
        OutputGuard([PinConfig("M/ao1", "feed")])
    with pytest.raises(ValueError, match="at most one spindle"):
        OutputGuard([PinConfig("M/ao0", "spindle"), PinConfig("M/ao1", "spindle")])


# ---------------------------------------------------------------- copied rule = original
def test_speed_rule_matches_speed_control():
    from rig import speed_control as sc
    rng = np.random.default_rng(323)
    for _ in range(300):
        f, rpm = rng.uniform(500, 9500), rng.uniform(10_000, 100_000)
        for d in ("nearest", "down", "up"):
            assert wb_rule.new_speed(f, rpm, 2, 10_000, 100_000, direction=d) == \
                sc.new_speed(f, rpm, 2, 10_000, 100_000, direction=d)
    for f, rpm, paper in ((150.0, 950.0, 694.0), (115.2, 694.0, 531.0), (125.0, 1000.0, 833.3)):   # Liao & Young's runs
        N, _ = wb_rule.new_speed(f, rpm, 4, rpm_min=0, rpm_max=1000, max_step=1.0, direction="down")
        assert abs(N - paper) / paper < 0.005


def test_controller_matches_speed_control_on_a_run():
    from rig import speed_control as sc
    rng = np.random.default_rng(1)

    class Out:                                        # the SpindleOutput interface, no hardware
        armed, v_feed = False, None

        def rpm_range(self):
            return 0.0, 140_000.0

        def set_rpm(self, rpm):
            return rpm / 14_000

        def close(self, rpm):
            pass
    t = np.arange(20 * 20_000) / 20_000
    x = 0.01 * rng.normal(size=(3, t.size))
    x[:, 8 * 20_000:] += 0.2 * np.sin(2 * np.pi * 4117.0 * t[8 * 20_000:])        # chatter from 8 s
    labels = ["no cutting"] * (8 * 20_000 // 512) + ["chatter"] * (t.size // 512)
    events = []
    for mod_ in (sc, wb_rule):
        c = mod_.SpeedController(20_000, 2, Out(), 10_000, 100_000, log=lambda _m: None)
        for i in range(t.size // 512):
            c.update(x[:, i * 512:(i + 1) * 512], i * 512 / 20_000, labels[i])
        events.append([(e["action"], round(e["f"], 6), e.get("rpm_to")) for e in c.events])
    assert events[0] == events[1] and events[0]       # same decisions, and at least one


@pytest.mark.slow
def test_fast_forest_matches_sklearn():
    import joblib
    from workbench.core.live.fast_forest import FastForest
    clf = joblib.load(PKG / "extra_trees.joblib")
    X = np.random.default_rng(2).normal(size=(50, clf.n_features_in_))
    assert np.allclose(FastForest(clf).predict_proba(X), clf.predict_proba(X), atol=1e-12)
