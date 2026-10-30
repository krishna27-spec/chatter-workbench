"""Clean: the project's offline cleaning (denoise.py), as a Steps op.

Copied, not imported, so the workbench stands alone (the models/ packages copy denoise.py for the same reason), and
tested bit-for-bit against denoise.denoise() so it cannot drift from what the models and the noise study used:
  1. Hampel-style despike: a sample more than 6 local scales from the median of its 11 neighbours is replaced by
     that median; the scale is the RMS over 201 samples (10 ms), floored at one ADC step.
     (Hampel, J. Am. Stat. Assoc. 69(346), 1974; Pearson et al., EURASIP J. Adv. Signal Process. 2016:87,
     doi:10.1186/s13634-016-0383-6. The RMS scale instead of the MAD is the project's own modification.)
  2. 50 Hz high-pass, 4th-order Butterworth, run forward and backward (sosfiltfilt): zero phase, so events don't move.
  3. Line removal: FFT bins within ±0.3 Hz of the mains harmonics 50…400 Hz and ±(0.15 + 0.002 k) Hz of the
     4.466 Hz machine comb up to 400 Hz are set to zero (the comb width grows because an error in f0 grows with k).
Zero-phase steps need the whole file, so this is an offline op; causal.py is the live version.

Three changes from denoise.py, all for speed and memory on long files (measured on the 17.7 M-sample session):
- the 11-sample median uses scipy.ndimage.median_filter(mode="mirror") instead of numpy's median over a
  sliding_window_view of a reflect-padded copy: the same values (scipy "mirror" = numpy "reflect"), no 11× copy;
- the line mask only compares the few bins near each line, not all 8.8 M bins per line: the same mask, 2.7 s → 1 ms;
- the FFT is scipy.fft on all CPU cores instead of numpy.fft: 3.3 s → 0.4 s per transform. Not bit-identical: the
  rounding differs (≤ 1e-15 relative, far below one ADC step), so the test checks equality within float rounding.
"""
from __future__ import annotations

import numpy as np
import scipy.fft as sfft
from scipy import signal
from scipy.ndimage import median_filter, uniform_filter1d

from workbench.core.recording import Recording
from workbench.core.steps import map_channels, op

COMB_F0, COMB_FMAX = 4.466, 400.0
MAINS_F0, MAINS_FMAX = 50.0, 400.0


def despike(v: np.ndarray, window: int = 11, n_sigma: float = 6.0, scale_window: int = 201):
    med = median_filter(v, size=window, mode="mirror")
    centred = v - uniform_filter1d(v, scale_window, mode="reflect")
    rms = np.sqrt(uniform_filter1d(centred ** 2, scale_window, mode="reflect"))
    step = np.median(np.diff(np.unique(v)))             # one ADC step: flat quantised stretches are not spikes
    mask = np.abs(v - med) > n_sigma * np.maximum(rms, step)
    out = v.copy()
    out[mask] = med[mask]
    return out, mask


def highpass(v: np.ndarray, fs: float, fc: float = 50.0, order: int = 4) -> np.ndarray:
    return signal.sosfiltfilt(signal.butter(order, fc, "hp", fs=fs, output="sos"), v)


def remove_lines(v: np.ndarray, fs: float, freqs: np.ndarray, half_width: np.ndarray) -> np.ndarray:
    n = len(v)
    F = sfft.rfft(v, workers=-1)
    f = np.fft.rfftfreq(n, 1 / fs)
    df = f[1] - f[0]
    kill = np.zeros(len(f), bool)
    for f0, w in zip(freqs, np.broadcast_to(half_width, np.shape(freqs))):
        lo, hi = max(0, int((f0 - w) / df) - 2), int((f0 + w) / df) + 3     # only the bins that can be within w
        kill[lo:hi] |= np.abs(f[lo:hi] - f0) <= w
    F[kill] = 0
    return sfft.irfft(F, n, workers=-1)


def clean_axis(x: np.ndarray, fs: float, hp_fc: float = 50.0, comb_hw: float = 0.15, mains_hw: float = 0.3,
               do_despike: bool = True) -> np.ndarray:
    """One channel through the three stages, in float64 (same arithmetic as denoise.denoise)."""
    v = np.asarray(x, float)
    if do_despike:
        v, _ = despike(v)
    v = highpass(v, fs, hp_fc)
    comb = np.arange(1, int(COMB_FMAX / COMB_F0) + 1) * COMB_F0
    mains = np.arange(1, int(MAINS_FMAX / MAINS_F0) + 1) * MAINS_F0
    comb_w = comb_hw + 0.002 * np.arange(1, len(comb) + 1)
    return remove_lines(v, fs, np.concatenate([comb, mains]),
                        np.concatenate([comb_w, np.full(len(mains), mains_hw)]))


@op("clean", "Clean", "Despike (Hampel), 50 Hz high-pass, remove mains and machine-comb lines: denoise.py, zero-phase",
    hp_fc=50.0, comb_hw=0.15, mains_hw=0.3, do_despike=True)
def clean(rec: Recording, hp_fc=50.0, comb_hw=0.15, mains_hw=0.3, do_despike=True) -> Recording:
    return map_channels(rec, lambda x: clean_axis(x, rec.fs, hp_fc, comb_hw, mains_hw, do_despike), step="clean")
