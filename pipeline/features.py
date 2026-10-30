"""Hand-made features for one 51 ms window (3 axes x 1024 samples).

Every function takes X of shape (n, 3, WIN), already divided by the file's
idle level, and returns (n, k) features plus their names. Groups:

  level     log RMS per axis: loudness only (the old baseline)
  time      kurtosis, skewness, crest factor, zero-crossing rate: waveform shape
  spectral  centroid, spread, flatness, entropy, roll-off, peakiness, top-3
            share, dominant frequency: shape of the spectrum, loudness-free
  bands     share of energy in ten 1 kHz bands: where the energy sits
  cepstrum  cepstral peak prominence: how strongly the spectrum is a comb of
            equally spaced harmonics, found without knowing the spindle speed
  axes      correlation between axis pairs
  harmonic  REFERENCE ONLY, needs the rpm: notched entropy (the labeller's own
            quantity) and the share of energy away from spindle harmonics

context(F) adds the median of each feature over the previous 0.5 s.
"""
import numpy as np
from scipy.ndimage import median_filter

FS, WIN = 20_000, 1024
F = np.fft.rfftfreq(WIN, 1 / FS)
KEEP = F >= 50                                  # same 50 Hz floor as labels.py
AX = 'XYZ'


def power_spectrum(X):
    """Hann-windowed power spectrum, (n, 3, 513)."""
    Xc = X - X.mean(-1, keepdims=True)
    return np.abs(np.fft.rfft(Xc * np.hanning(WIN), axis=-1)) ** 2


def _names(stems):
    return [f'{s}_{a}' for s in stems for a in AX]


def level(X, P=None):
    return np.log(np.sqrt((X ** 2).mean(-1)) + 1e-6), _names(['logrms'])


def time_shape(X, P=None):
    Xc = X - X.mean(-1, keepdims=True)
    sd = Xc.std(-1) + 1e-9
    z = Xc / sd[..., None]
    kurt = (z ** 4).mean(-1) - 3
    skew = (z ** 3).mean(-1)
    crest = np.abs(Xc).max(-1) / sd
    zcr = (np.diff(np.sign(Xc), axis=-1) != 0).mean(-1)
    return np.concatenate([kurt, skew, np.log(crest), zcr], 1), _names(['kurt', 'skew', 'logcrest', 'zcr'])


def spectral(X, P):
    p = P[..., KEEP]
    f = F[KEEP] / 1000                                        # kHz
    tot = p.sum(-1, keepdims=True) + 1e-20
    q = p / tot                                               # spectrum as a probability
    centroid = (q * f).sum(-1)
    spread = np.sqrt((q * (f - centroid[..., None]) ** 2).sum(-1))
    flat = np.exp(np.log(p + 1e-20).mean(-1)) / p.mean(-1)    # geometric / arithmetic mean
    ent = -(q * np.log(q + 1e-30)).sum(-1) / np.log(q.shape[-1])
    rolloff = f[np.argmax(np.cumsum(q, -1) >= 0.85, -1)]
    peaky = np.log(p.max(-1) / np.median(p, -1))
    top3 = np.sort(q, -1)[..., -3:].sum(-1)
    fdom = f[q.argmax(-1)]
    out = [centroid, spread, np.log(flat), ent, rolloff, peaky, top3, fdom]
    return np.concatenate(out, 1), _names(['centroid', 'spread', 'logflat', 'entropy', 'rolloff',
                                           'peaky', 'top3', 'fdom'])


def bands(X, P):
    edges = np.arange(0, 11) * 1000
    tot = P[..., KEEP].sum(-1) + 1e-20
    out, names = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (F >= max(lo, 50)) & (F < hi)
        out.append(np.log(P[..., m].sum(-1) / tot + 1e-6))
        names += [f'band{lo // 1000}k_{a}' for a in AX]
    return np.concatenate(out, 1), names


def cepstrum(X, P):
    """Cepstral peak prominence over quefrencies 0.5-3.5 ms (spindle 17k-120k rpm).
    A spectrum made of evenly spaced harmonics f0, 2f0, ... is itself periodic
    in frequency, so the FFT of the log spectrum has a peak at 1/f0."""
    c = np.abs(np.fft.irfft(np.log(P + 1e-12), axis=-1))     # real cepstrum, (n, 3, 1024)
    qf = np.arange(c.shape[-1]) / FS
    m = (qf >= 0.5e-3) & (qf <= 3.5e-3)
    seg = c[..., m]
    cpp = np.log(seg.max(-1) / np.median(seg, -1))
    return cpp, _names(['cpp'])


def axes(X, P=None):
    Xc = X - X.mean(-1, keepdims=True)
    Xc = Xc / (np.linalg.norm(Xc, axis=-1, keepdims=True) + 1e-9)
    pairs = [(0, 1), (0, 2), (1, 2)]
    return np.stack([(Xc[:, i] * Xc[:, j]).sum(-1) for i, j in pairs], 1), ['corr_XY', 'corr_XZ', 'corr_YZ']


def harmonic(X, P, rpm):
    """Needs the spindle speed: the labeller's notched entropy and the
    non-harmonic energy share. Used only as an upper-bound reference."""
    fsp = rpm / 60.0
    k = np.round(F / fsp)
    harm = (np.abs(F - k * fsp) <= 1.5 * FS / WIN) & (k >= 1)
    keep = ~harm & KEEP
    pn = P[..., keep]
    q = pn / (pn.sum(-1, keepdims=True) + 1e-20)
    nent = -(q * np.log(q + 1e-30)).sum(-1) / np.log(keep.sum())
    nh = np.log(pn.sum(-1) / (P[..., KEEP].sum(-1) + 1e-20))
    return np.concatenate([nent, nh], 1), _names(['notched_ent', 'nonharm_share'])


# ---- wave 2: more invented / borrowed features --------------------------------

def hjorth(X, P=None):
    """Hjorth (1970) mobility ~ mean frequency, complexity ~ how far from one sine."""
    d1 = np.diff(X, axis=-1)
    d2 = np.diff(d1, axis=-1)
    v0, v1, v2 = X.var(-1) + 1e-12, d1.var(-1) + 1e-12, d2.var(-1) + 1e-12
    mob = np.sqrt(v1 / v0)
    comp = np.sqrt(v2 / v1) / mob
    return np.concatenate([mob, np.log(comp)], 1), _names(['hj_mobility', 'hj_logcomplexity'])


def perm_entropy(X, P=None, order=4):
    """Bandt & Pompe (2002): entropy of the ordinal patterns of `order`
    consecutive samples, normalised to 0..1. A pure tone uses few patterns."""
    n, c, L = X.shape
    seg = np.stack([X[..., i:L - order + 1 + i] for i in range(order)], -1)   # (n, c, L', order)
    ranks = np.argsort(seg, -1)
    code = (ranks * (order ** np.arange(order))).sum(-1)
    out = np.empty((n, c))
    for a in range(c):
        cnt = np.zeros((n, order ** order))
        np.add.at(cnt, (np.repeat(np.arange(n), code.shape[2]), code[:, a].ravel()), 1)
        p = cnt / cnt.sum(1, keepdims=True)
        out[:, a] = -(p * np.log(p + 1e-30)).sum(1) / np.log(24)              # 4! patterns
    return out, _names(['perm_entropy'])


def teager(X, P=None):
    """Teager-Kaiser energy x[n]^2 - x[n-1]x[n+1] (Kaiser 1990). For a sine
    A sin(wn) it equals A^2 sin^2 w, so its ratio to the variance tracks frequency;
    its spread over time tracks bursts of amplitude/frequency change."""
    tk = X[..., 1:-1] ** 2 - X[..., :-2] * X[..., 2:]
    r = np.log(np.abs(tk.mean(-1)) / (X.var(-1) + 1e-12) + 1e-9)
    cv = np.log(tk.std(-1) / (np.abs(tk.mean(-1)) + 1e-12) + 1e-9)
    return np.concatenate([r, cv], 1), _names(['tk_ratio', 'tk_logcv'])


def spectral_kurtosis(X, P=None, nfft=128, hop=32):
    """Antoni (2006): for each frequency, kurtosis of the STFT magnitude over
    time. Stationary tone -> about -1, Gaussian noise -> 0, impulses -> large."""
    n, c, L = X.shape
    idx = np.arange(0, L - nfft + 1, hop)[:, None] + np.arange(nfft)
    S = np.abs(np.fft.rfft(X[..., idx] * np.hanning(nfft), axis=-1))[..., 1:]   # (n, c, frames, 64)
    sk = (S ** 4).mean(2) / ((S ** 2).mean(2) ** 2 + 1e-20) - 2
    return np.concatenate([sk.max(-1), sk.mean(-1)], 1), _names(['sk_max', 'sk_mean'])


def katz_fd(X, P=None):
    """Katz (1988) fractal dimension of the waveform: ~1 smooth/tonal, higher = rough."""
    d = np.abs(np.diff(X, axis=-1))
    Lc = d.sum(-1)
    a = d.mean(-1)
    dist = np.abs(X - X[..., :1]).max(-1)
    nn = Lc / (a + 1e-12)
    return np.log10(nn) / (np.log10(nn) + np.log10(dist / (Lc + 1e-12) + 1e-12)), _names(['katz_fd'])


def wavelet_packet(X, P=None, level=3):
    """Energy share of the 8 level-3 Daubechies-4 wavelet packet nodes
    (1.25 kHz each). Common in chatter papers, e.g. Yao et al. 2010."""
    import pywt
    wp = pywt.WaveletPacket(X, 'db4', mode='periodization', maxlevel=level, axis=-1)
    nodes = [nd.path for nd in wp.get_level(level, 'freq')]
    E = np.stack([(wp[p].data ** 2).sum(-1) for p in nodes], -1)          # (n, 3, 8)
    E = np.log(E / (E.sum(-1, keepdims=True) + 1e-20) + 1e-6)
    return E.transpose(0, 2, 1).reshape(len(X), -1), [f'wp{i}_{a}' for i in range(len(nodes)) for a in AX]


def coherence(X, P=None, nfft=128, hop=64):
    """Largest magnitude-squared coherence between axis pairs above 1 kHz
    (Welch average over the frames of the window). A real structural mode moves
    the axes together at one frequency."""
    L = X.shape[-1]
    idx = np.arange(0, L - nfft + 1, hop)[:, None] + np.arange(nfft)
    S = np.fft.rfft(X[..., idx] * np.hanning(nfft), axis=-1)               # (n, 3, frames, 65)
    f = np.fft.rfftfreq(nfft, 1 / FS)
    m = f >= 1000
    out = []
    for i, j in ((0, 1), (0, 2), (1, 2)):
        Sij = (S[:, i] * np.conj(S[:, j])).mean(1)
        msc = np.abs(Sij) ** 2 / ((np.abs(S[:, i]) ** 2).mean(1) * (np.abs(S[:, j]) ** 2).mean(1) + 1e-20)
        out.append(msc[:, m].max(-1))
    return np.stack(out, 1), ['coh_XY', 'coh_XZ', 'coh_YZ']


def catch22(X, P=None):
    """The 22 'canonical time-series characteristics' of Lubba et al. (2019),
    picked from 7,700 hctsa features as the most useful and least redundant
    across 93 classification problems. Computed by the pycatch22 C library."""
    import pycatch22
    n = len(X)
    out = np.zeros((n, 3, 22))
    names = None
    for i in range(n):
        for a in range(3):
            r = pycatch22.catch22_all(X[i, a].tolist())
            out[i, a] = r['values']
            names = names or r['names']
    out = np.nan_to_num(out)
    return out.transpose(0, 2, 1).reshape(n, -1), [f'c22_{nm}_{a}' for nm in names for a in AX]


def lfcc(X, P, n_filt=24, n_coef=12):
    """Linear-frequency cepstral coefficients: log energy in 24 triangular bands
    spread evenly over 50 Hz-10 kHz, then a DCT. The speech-recognition MFCC
    recipe with linear instead of mel spacing, since vibration has no reason to
    favour low frequencies. Coefficient 0 (overall level) is dropped."""
    from scipy.fft import dct
    centres = np.linspace(50, FS / 2, n_filt + 2)
    fb = np.zeros((n_filt, len(F)))
    for j in range(n_filt):
        lo, c, hi = centres[j:j + 3]
        fb[j] = np.clip(np.minimum((F - lo) / (c - lo), (hi - F) / (hi - c)), 0, None)
    E = np.log(P @ fb.T + 1e-12)                               # (n, 3, n_filt)
    C = dct(E, type=2, norm='ortho', axis=-1)[..., 1:n_coef + 1]
    return C.transpose(0, 2, 1).reshape(len(X), -1), [f'lfcc{k}_{a}' for k in range(1, n_coef + 1) for a in AX]


AGNOSTIC = {'level': level, 'time': time_shape, 'spectral': spectral, 'bands': bands,
            'cepstrum': cepstrum, 'axes': axes,
            'hjorth': hjorth, 'perm_entropy': perm_entropy, 'teager': teager,
            'spec_kurtosis': spectral_kurtosis, 'katz': katz_fd, 'wavelet': wavelet_packet,
            'coherence': coherence, 'catch22': catch22, 'lfcc': lfcc}
WAVE1 = ('level', 'time', 'spectral', 'bands', 'cepstrum', 'axes')


def sequence_features(X, n=19):
    """Features that need the previous windows of the same file (call per file):
      tone_persist  share of the last n windows whose dominant frequency is
                    within 2 bins of this window's (a chatter tone stays put)
      flux          1 - correlation between this and the previous log spectrum
      level_trend   change of log RMS over the last n windows (chatter grows)"""
    P = power_spectrum(X.astype(np.float64))[..., KEEP]
    fd = P.argmax(-1)                                                       # (m, 3)
    ls = np.log(P + 1e-12)
    ls = (ls - ls.mean(-1, keepdims=True)) / (ls.std(-1, keepdims=True) + 1e-9)
    flux = 1 - np.r_[np.ones((1, 3)), (ls[1:] * ls[:-1]).mean(-1)]
    lr = np.log(np.sqrt((X.astype(np.float64) ** 2).mean(-1)) + 1e-6)
    m = len(X)
    tp = np.zeros((m, 3))
    trend = np.zeros((m, 3))
    for i in range(m):
        a = max(0, i - n + 1)
        tp[i] = (np.abs(fd[a:i + 1] - fd[i]) <= 2).mean(0)
        trend[i] = lr[i] - lr[a]
    names = _names(['tone_persist']) + _names(['flux']) + _names(['level_trend'])
    return np.concatenate([tp, flux, trend], 1).astype(np.float32), names


def extract(X, rpm=None, groups=tuple(AGNOSTIC), batch=2048):
    """Feature matrix (n, k), names, and the group of each column."""
    outs, names, group = [], [], []
    for s in range(0, len(X), batch):
        xb = X[s:s + batch].astype(np.float64)
        P = power_spectrum(xb)
        row = []
        for g in groups:
            v, nm = harmonic(xb, P, rpm) if g == 'harmonic' else AGNOSTIC[g](xb, P)
            row.append(v)
            if s == 0:
                names += nm
                group += [g] * len(nm)
        outs.append(np.concatenate(row, 1))
    return np.concatenate(outs).astype(np.float32), names, np.array(group)


def context(Fm, n=19):
    """Trailing median over the previous n windows (n=19 is 0.5 s at hop 512).
    Causal, so it could run live. Call per file, never across file borders."""
    pad = np.concatenate([np.repeat(Fm[:1], n - 1, 0), Fm])
    med = median_filter(pad, size=(n, 1), origin=((n - 1) // 2, 0), mode='nearest')
    return med[n - 1:]
