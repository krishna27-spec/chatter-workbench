import numpy as np
import pytest

from workbench.core.analysis.spectrum import (classify, column_max, harmonics, peaks, psd, rpm_source,
                                              spectrogram, teeth_source)
from workbench.core.formats.session import read_session
from workbench.core.recording import Channel, Recording

FS = 20_000.0
rng = np.random.default_rng(323)


def tone(f, amp=0.5, seconds=2.0, noise=0.01):
    t = np.arange(int(seconds * FS)) / FS
    return amp * np.sin(2 * np.pi * f * t) + noise * rng.normal(size=t.size)


def test_welch_finds_a_tone_between_bins():
    s = psd(tone(1234.5), FS)                          # 1234.5 Hz is between two 4.88 Hz bins
    assert s.nperseg == 4096 and s.resolution == pytest.approx(FS / 4096)
    assert s.n_avg == 1 + (40_000 - 4096) // 2048
    top = peaks(s.f, s.P)[0]
    assert abs(top.f - 1234.5) < 0.3                   # parabolic refinement: well under a bin
    assert abs(s.f[top.bin] - 1234.5) <= s.resolution / 2


def test_area_under_psd_is_the_variance():
    x = rng.normal(0, 0.2, 200_000)                    # white noise, variance 0.04 V²
    s = psd(x, FS)
    assert np.sum(s.P) * s.resolution == pytest.approx(np.var(x), rel=0.02)   # Parseval


def test_fft_method_uses_the_whole_region():
    s = psd(tone(1000.0, seconds=0.5), FS, method="fft")
    assert s.n_avg == 1 and s.nperseg == 10_000 and s.resolution == pytest.approx(2.0)


def test_short_region_shrinks_the_segment():
    s = psd(rng.normal(size=3000), FS)
    assert s.nperseg == 2048


def test_too_short_is_refused():
    with pytest.raises(ValueError, match="too short.*256"):
        psd(rng.normal(size=200), FS)


def test_peaks_ignore_below_fmin_and_sort_by_prominence():
    x = tone(30.0, amp=1.0) + tone(2000.0, amp=0.3) + tone(5000.0, amp=0.05)
    s = psd(x, FS)
    found = peaks(s.f, s.P)
    assert all(p.f >= 50 for p in found)               # the 30 Hz tone is below the 50 Hz floor
    assert [round(p.f, -1) for p in found[:2]] == [2000.0, 5000.0]
    assert found[0].prominence_db > found[1].prominence_db


def test_harmonics_at_20000_rpm():
    h = harmonics(20_000, 2, fmax=2000)
    assert h.spindle_hz == pytest.approx(333.333, abs=1e-3) and h.tooth_hz == pytest.approx(666.667, abs=1e-3)
    assert np.allclose(h.spindle_lines, [333.333, 666.667, 1000.0, 1333.333, 1666.667, 2000.0], atol=1e-3)   # fmax itself included
    assert np.allclose(h.tooth_lines, [666.667, 1333.333, 2000.0], atol=1e-3)


def test_classify_the_sessions_chatter_peak():
    """scope.py logged 'chatter at 8167.7 Hz' at 20 000 rpm, 2 teeth (session_20261023_032238)."""
    c = classify(8167.7, 20_000, 2, resolution=4.88)
    assert c.k == 25 and c.offset_hz == pytest.approx(8167.7 - 25 * 20_000 / 60)
    assert c.kind == "between"
    assert c.n == 12 and c.nu == pytest.approx(8167.7 / (2 * 20_000 / 60) - 12)    # Liao & Young: n + ν
    assert classify(1000.0 + 3.0, 20_000, 2, resolution=4.88).kind == "harmonic"  # within 2 bins of 3 × 333.3


def test_rpm_sources_in_order(tmp_path, make_session):
    rec = read_session(make_session(tmp_path / "s"))
    assert rpm_source(rec, 0.05) == (20000.0, "control.csv rpm_cmd")
    del rec.tables["output"]
    rec.meta["session"]["args"]["rpm"] = 21000
    assert rpm_source(rec, 0.05) == (21000.0, "session.json")
    plain = Recording([Channel("X", "V", np.zeros(10, np.float32))], FS, source=tmp_path / "Acce_60k_1_r1.lvm")
    assert rpm_source(plain) == (60000.0, "file name")
    plain.source = tmp_path / "run.lvm"
    assert rpm_source(plain) == (None, "unknown")
    assert teeth_source(plain) == (2, "default (lab tool, Z = 2)")
    rec.meta["session"]["args"]["teeth"] = 4
    assert teeth_source(rec) == (4, "session.json")


def test_spectrogram_frames_match_the_models_window():
    x = rng.normal(size=20_000)
    sg = spectrogram(x, FS)
    assert sg.db.shape == (1 + (20_000 - 1024) // 512, 513)
    assert sg.t[0] == pytest.approx(512 / FS) and sg.t[1] - sg.t[0] == pytest.approx(512 / FS)
    assert sg.f[1] == pytest.approx(FS / 1024)
    assert sg.db.dtype == np.float32


def test_peak_hold_keeps_a_one_frame_burst():
    x = 0.01 * rng.normal(size=600_000)                # 30 s of noise
    t = np.arange(1024) / FS
    x[300_000:301_024] += np.sin(2 * np.pi * 5000 * t)  # one 51 ms burst at 5 kHz
    sg = spectrogram(x, FS)
    t_cols, cols = column_max(sg, 0.0, 30.0, n_px=200)  # ~6 frames per column
    k = int(round(5000 / (FS / 1024)))
    burst = cols[:, k]
    assert cols.shape == (200, 513)
    assert burst.max() - np.median(burst) > 30         # the burst column stands > 30 dB above the rest
    assert len(t_cols) == 200


def test_zoomed_in_spectrogram_returns_frames():
    sg = spectrogram(rng.normal(size=40_000), FS)
    t_cols, cols = column_max(sg, 0.2, 0.4, n_px=800)   # 8 frames in view, 800 px
    assert len(t_cols) == len(cols) <= 9 and np.array_equal(cols[0], sg.db[np.searchsorted(sg.t, 0.2)])


def test_amplitude_spectrum_reads_the_sine_amplitude():
    """Linear amplitude (peak, in the signal's unit): a 0.5 V sine reads 0.5 V. A tone exactly on a bin is exact;
    between bins the Hann window reads up to 15 % low (scalloping), which the peak's parabola does not undo."""
    from workbench.core.analysis.spectrum import amplitude
    f_bin = 100 * FS / 4096                               # exactly on bin 100
    s = psd(tone(f_bin, amp=0.5, noise=0.0), FS)
    A = amplitude(s)
    assert A.max() == pytest.approx(0.5, rel=1e-3) and int(A.argmax()) == 100
    assert s.enbw == pytest.approx(1.5 * FS / 4096, rel=1e-3)      # Hann: 1.5 bins
    s2 = psd(tone(f_bin + 0.5 * FS / 4096, amp=0.5, noise=0.0), FS)  # half a bin off
    assert 0.5 * 0.84 < amplitude(s2).max() < 0.5
