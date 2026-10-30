import numpy as np
import pytest

from workbench.core.reference.pyramid import MinMaxPyramid


def brute(x, i0, i1, n_px):
    """The definition, one pixel at a time: min and max of every sample inside the pixel."""
    e = i0 + np.arange(n_px + 1) * (i1 - i0) // n_px
    lo = np.array([np.fmin.reduce(x[a:b]) for a, b in zip(e[:-1], e[1:])])
    hi = np.array([np.fmax.reduce(x[a:b]) for a, b in zip(e[:-1], e[1:])])
    return lo, hi


def test_matches_brute_force():
    rng = np.random.default_rng(323)
    for _ in range(200):
        n = int(rng.integers(1, 300_000))
        x = rng.normal(size=n).astype(np.float32)
        pyr = MinMaxPyramid(x, block=int(rng.choice([2, 4, 16])))
        i0 = int(rng.integers(0, n))
        i1 = int(rng.integers(i0 + 1, n + 1))
        n_px = int(rng.integers(1, 4000))
        lo, hi = pyr.envelope(i0, i1, n_px)
        if i1 - i0 <= n_px:
            assert np.array_equal(lo, x[i0:i1]) and np.array_equal(hi, x[i0:i1])
        else:
            blo, bhi = brute(x, i0, i1, n_px)
            assert np.array_equal(lo, blo) and np.array_equal(hi, bhi)


def test_single_spike_survives():
    x = np.zeros(12_000_000, np.float32)            # 10 min at 20 kHz
    x[7_654_321] = 5.0
    pyr = MinMaxPyramid(x)
    for n_px in (500, 1000, 1920, 3840):
        lo, hi = pyr.envelope(0, len(x), n_px)
        assert hi.max() == 5.0 and lo.min() == 0.0


def test_values_are_real_samples():
    x = np.random.default_rng(1).normal(size=100_000).astype(np.float32)
    lo, hi = MinMaxPyramid(x).envelope(1_234, 98_765, 777)
    assert np.isin(lo, x).all() and np.isin(hi, x).all()      # nothing averaged or interpolated


def test_nan_does_not_hide_data():
    x = np.arange(1000, dtype=np.float64)
    x[::2] = np.nan                                  # every other sample missing
    x[100:200] = np.nan                              # one pixel's worth all missing
    lo, hi = MinMaxPyramid(x, block=4).envelope(0, 1000, 10)
    assert lo[0] == 1 and hi[0] == 99
    assert np.isnan(lo[1]) and np.isnan(hi[1])
    assert lo[2] == 201 and hi[9] == 999


def test_zoomed_in_returns_raw_samples():
    x = np.arange(50, dtype=np.float32)
    lo, hi = MinMaxPyramid(x).envelope(10, 40, 800)
    assert np.array_equal(lo, x[10:40]) and np.array_equal(hi, x[10:40])


def test_bad_range_is_refused():
    pyr = MinMaxPyramid(np.zeros(10))
    with pytest.raises(ValueError):
        pyr.envelope(5, 5, 100)
    with pytest.raises(ValueError):
        pyr.envelope(0, 11, 100)
