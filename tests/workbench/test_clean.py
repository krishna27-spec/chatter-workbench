"""The Clean op must give what denoise.py gives (the models and the noise study were built on that), to within
float rounding: the multi-core FFT rounds differently (≤ 1e-15 relative), nothing else may differ."""
import sys
from pathlib import Path

import numpy as np
import pytest

from workbench.core.ops.clean import clean, clean_axis
from workbench.core.recording import Channel, Recording

ROOT = Path(__file__).resolve().parents[2]
LAB = ROOT / "data" / "raw" / "Acce_20k_1_r1.lvm"


def same(a, b):
    """Equal up to float rounding: 1e-12 of the signal's size is ~10 000× below one ADC step."""
    return a.shape == b.shape and np.max(np.abs(a - b)) <= 1e-12 * max(1.0, np.max(np.abs(b)))


def reference():
    from pipeline import denoise
    return denoise


def test_same_as_denoise_on_synthetic():
    dn = reference()
    rng = np.random.default_rng(323)
    t = np.arange(40_000) / 20_000
    x = 0.01 * rng.normal(size=t.size) + 0.02 * np.sin(2 * np.pi * 50 * t) + 0.005 * np.sin(2 * np.pi * 4.466 * t)
    x[[1000, 22222]] = 1.0                                 # two electrical spikes
    assert same(clean_axis(x, 20_000.0), dn.denoise(x)[0])


@pytest.mark.slow
@pytest.mark.skipif(not LAB.exists(), reason="lab file not on this machine")
def test_same_as_denoise_on_the_lab_file():
    dn = reference()
    _, x = dn.load_lvm(LAB)
    ref, _ = dn.denoise(x)
    for i in range(3):
        assert same(clean_axis(x[:, i], 20_000.0), ref[:, i])


def test_op_on_a_recording():
    rng = np.random.default_rng(1)
    rec = Recording([Channel(c, "V", rng.normal(size=8000).astype(np.float32)) for c in "XY"], 20_000.0, t0=3.0)
    out = clean(rec)
    assert [c.name for c in out.channels] == ["X", "Y"] and out.t0 == 3.0
    assert out.channels[0].data.dtype == np.float32
    assert abs(float(np.mean(out.channels[0].data))) < 1e-3          # the high-pass removed the mean
    assert out.meta["step"] == "clean"
