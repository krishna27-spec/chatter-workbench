"""Frequency content of a stretch: power spectrum, peaks, spindle harmonics, spectrogram.

- psd: Welch's method (Welch 1967, IEEE Trans. Audio Electroacoust. 15(2) 70-73, doi:10.1109/TAU.1967.1161901):
  Hann-windowed segments with 50 % overlap, each segment's periodogram averaged. Averaging K segments cuts the
  scatter of the estimate by about √K; the price is resolution fs/nperseg. method="fft" is the single-window
  periodogram over the whole stretch: finest resolution, no averaging. Density scaling, so the area under P is the
  signal's variance (Parseval) whatever the segment length.
- amplitude: the same spectrum as linear peak amplitude in the signal's unit, A = sqrt(2 · P · ENBW), where ENBW is
  the window's equivalent noise bandwidth fs·Σw²/(Σw)² (1.5 bins for Hann). A sine of amplitude A on a bin reads A;
  between bins Hann reads up to 15 % (1.42 dB) low, the "scalloping loss" (Harris 1978, Proc. IEEE 66(1) 51-83).
- peaks: local maxima of the dB spectrum ranked by prominence (how far a peak rises above the higher of the two
  valleys around it, scipy.signal.find_peaks), frequency refined by fitting a parabola through the dB values of the
  peak bin and its two neighbours (quadratic peak interpolation; J. O. Smith III, Spectral Audio Signal Processing,
  W3K 2011, "Quadratic Interpolation of Spectral Peaks").
- harmonics / classify: spindle frequency f_s = rpm/60 and tooth-passing frequency f_tp = Z·rpm/60. Forced vibration
  sits on multiples of f_s; chatter appears between them. Liao & Young (1996, Int. J. Mach. Tools Manufact. 36(5)
  651-660) write the chatter frequency as f = (n + ν)·f_tp: n whole waves plus a phase fraction ν per tooth period,
  which is what speed_control.py changes the speed to adjust.
- spectrogram: short-time Fourier transform with the models' window (1024 Hann, hop 512: 51 ms / 25.6 ms), and
  column_max to show it on fewer pixels than frames by taking the maximum per pixel column (peak hold), the
  spectrogram form of the min/max rule, so a single-frame burst stays visible when zoomed out.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal

from workbench.core.recording import Recording

MIN_SAMPLES = 256
SG_WIN, SG_HOP = 1024, 512


@dataclass
class Spectrum:
    f: np.ndarray            # Hz
    P: np.ndarray            # power spectral density, unit²/Hz
    resolution: float        # bin spacing, Hz
    nperseg: int
    n_avg: int
    method: str
    enbw: float = 0.0        # equivalent noise bandwidth of the window, Hz


@dataclass
class Peak:
    f: float                 # refined frequency, Hz
    level_db: float          # refined level, dB (10·log10 of unit²/Hz)
    prominence_db: float
    bin: int


@dataclass
class Harmonics:
    spindle_hz: float
    tooth_hz: float
    spindle_lines: np.ndarray
    tooth_lines: np.ndarray


@dataclass
class Classified:
    k: int                   # nearest spindle harmonic number
    offset_hz: float         # f − k·f_s
    kind: str                # "harmonic" or "between"
    n: int                   # Liao & Young: f / f_tp = n + ν
    nu: float


@dataclass
class Spectrogram:
    t: np.ndarray            # frame centres, s from the first sample
    f: np.ndarray            # Hz
    db: np.ndarray           # (frames, bins) float32, 10·log10(PSD)


def psd(x: np.ndarray, fs: float, method: str = "welch", nperseg: int = 4096) -> Spectrum:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) < MIN_SAMPLES:
        raise ValueError(f"Region too short: needs ≥ {MIN_SAMPLES} samples "
                         f"({MIN_SAMPLES / fs * 1e3:.1f} ms at {fs / 1e3:g} kHz), has {len(x)}")
    if method == "fft":
        n, overlap = len(x), 0
    elif method == "welch":
        n = nperseg if len(x) >= nperseg else 2 ** int(math.log2(len(x)))
        overlap = n // 2
    else:
        raise ValueError(f"method must be 'welch' or 'fft', got {method!r}")
    f, P = signal.welch(x, fs, window="hann", nperseg=n, noverlap=overlap, detrend="constant", scaling="density")
    n_avg = 1 + (len(x) - n) // (n - overlap) if method == "welch" else 1
    w = signal.get_window("hann", n)                                   # the window welch() used
    return Spectrum(f, P, fs / n, n, n_avg, method, float(fs * np.sum(w ** 2) / np.sum(w) ** 2))


def amplitude(spec: Spectrum) -> np.ndarray:
    """Peak amplitude per bin in the signal's unit: sqrt(2 · P · ENBW)."""
    return np.sqrt(2.0 * spec.P * spec.enbw)


SCALES = {"db": "PSD [dB re 1 {u}²/Hz]", "psd": "PSD [{u}²/Hz]", "amplitude": "amplitude [{u}]"}


def scaled(spec: Spectrum, scale: str) -> np.ndarray:
    """The spectrum in one of SCALES: dB of the density, the density itself, or linear peak amplitude."""
    if scale == "db":
        return to_db(spec.P)
    if scale == "psd":
        return spec.P
    if scale == "amplitude":
        return amplitude(spec)
    raise ValueError(f"scale must be one of {', '.join(SCALES)}")


def to_db(P: np.ndarray) -> np.ndarray:
    return 10 * np.log10(np.maximum(P, 1e-30))


def peaks(f: np.ndarray, P: np.ndarray, n: int = 10, fmin: float = 50.0, min_prominence_db: float = 6.0) -> list[Peak]:
    db = to_db(P)
    idx, props = signal.find_peaks(db, prominence=min_prominence_db)
    keep = (f[idx] >= fmin) & (idx > 0) & (idx < len(db) - 1)
    idx, prom = idx[keep], props["prominences"][keep]
    order = np.argsort(prom)[::-1][:n]
    df = f[1] - f[0]
    out = []
    for i in order:
        k = int(idx[i])
        a, b, c = db[k - 1], db[k], db[k + 1]
        denom = a - 2 * b + c
        p = 0.5 * (a - c) / denom if denom != 0 else 0.0     # vertex offset in bins, |p| ≤ 0.5 at a true maximum
        out.append(Peak(float(f[k] + p * df), float(b - 0.25 * (a - c) * p), float(prom[i]), k))
    return out


def harmonics(rpm: float, teeth: int, fmax: float) -> Harmonics:
    fs_ = rpm / 60.0
    ftp = teeth * fs_
    return Harmonics(fs_, ftp, np.arange(1, int(fmax / fs_) + 1) * fs_, np.arange(1, int(fmax / ftp) + 1) * ftp)


def classify(f: float, rpm: float, teeth: int, resolution: float) -> Classified:
    fs_ = rpm / 60.0
    k = max(1, int(round(f / fs_)))
    offset = f - k * fs_
    tol = max(2 * resolution, 1.0)            # within two bins of a spindle line counts as that line
    ratio = f / (teeth * fs_)
    n = int(math.floor(ratio))
    return Classified(k, offset, "harmonic" if abs(offset) <= tol else "between", n, ratio - n)


def spectrogram(x: np.ndarray, fs: float, nperseg: int = SG_WIN, hop: int = SG_HOP) -> Spectrogram:
    """dB spectrogram, one row per frame; frames are built in chunks so a 17.7 M-sample channel never needs a
    (frames × 1024) float64 copy at once."""
    x = np.nan_to_num(np.asarray(x, dtype=np.float64))
    if len(x) < nperseg:
        raise ValueError(f"too short for a spectrogram: needs ≥ {nperseg} samples")
    w = signal.windows.hann(nperseg, sym=False)
    scale = 1.0 / (fs * np.sum(w ** 2))                     # density scaling, same as psd()
    frames = sliding_window_view(x, nperseg)[::hop]
    db = np.empty((len(frames), nperseg // 2 + 1), dtype=np.float32)
    for a in range(0, len(frames), 4096):
        seg = frames[a:a + 4096]
        seg = (seg - seg.mean(axis=1, keepdims=True)) * w
        P = np.abs(np.fft.rfft(seg, axis=1)) ** 2 * scale
        P[:, 1:-1] *= 2                                      # one-sided: fold negative frequencies in
        db[a:a + 4096] = to_db(P)
    t = (np.arange(len(frames)) * hop + nperseg / 2) / fs
    return Spectrogram(t, np.fft.rfftfreq(nperseg, 1 / fs), db)


def column_max(sg: Spectrogram, t0: float, t1: float, n_px: int) -> tuple[np.ndarray, np.ndarray]:
    """Frames whose centre lies in [t0, t1], reduced to at most n_px columns by maximum (peak hold).
    Returns (centre time of each column, columns × bins). Fewer frames than pixels → the frames themselves."""
    i0 = int(np.searchsorted(sg.t, t0, "left"))
    i1 = int(np.searchsorted(sg.t, t1, "right"))
    count = i1 - i0
    if count <= n_px:
        return sg.t[i0:i1], sg.db[i0:i1]
    edges = i0 + np.arange(n_px + 1) * count // n_px
    cols = np.maximum.reduceat(sg.db[i0:i1], edges[:-1] - i0, axis=0)
    return (sg.t[edges[:-1]] + sg.t[edges[1:] - 1]) / 2, cols


def rpm_source(rec: Recording, t: float | None = None) -> tuple[float | None, str]:
    """Spindle speed at time t: the controller's command, else the run's setting, else the lab file name."""
    out = rec.tables.get("output", {})
    if t is not None and len(out.get("t", [])) and "rpm_cmd" in out:
        tt, rpm = np.asarray(out["t"], float), np.asarray(out["rpm_cmd"], float)
        i = min(max(int(np.searchsorted(tt, t, "right")) - 1, 0), len(tt) - 1)
        if np.isfinite(rpm[i]) and rpm[i] > 0:
            return float(rpm[i]), "control.csv rpm_cmd"
    args = rec.meta.get("session", {}).get("args", {})
    if args.get("rpm"):
        return float(args["rpm"]), "session.json"
    if rec.source is not None:
        m = re.search(r"(?:^|_)(\d+)k(?:_|$)", rec.source.stem)
        if m:
            return float(m[1]) * 1000, "file name"
    return None, "unknown"


def teeth_source(rec: Recording) -> tuple[int, str]:
    teeth = rec.meta.get("session", {}).get("args", {}).get("teeth")
    if teeth:
        return int(teeth), "session.json"
    return 2, "default (lab tool, Z = 2)"
