"""One-line description of every feature column, for the package's features.csv and the reference page.

describe('band7k_Y') -> ('bands', 'Y', 'log share of power in 7-8 kHz ...', 'features.bands')
All features are computed on one 51 ms window (1024 samples) of one axis after cleaning and
division by that axis's idle RMS, unless the description says otherwise.
"""
import re

# stem -> (group, description, function in features.py)
STEMS = {
    'logrms': ('level', 'log of the RMS: loudness in multiples of the idle level', 'level'),
    'kurt': ('time', 'excess kurtosis E[z^4] - 3 of the samples: heavy tails / bursts (0 for Gaussian, -1.5 for a sine)', 'time_shape'),
    'skew': ('time', 'skewness E[z^3]: asymmetry of the waveform', 'time_shape'),
    'logcrest': ('time', 'log crest factor = log(max|x| / std): peakiness of the waveform', 'time_shape'),
    'zcr': ('time', 'zero-crossing rate: share of samples where the sign changes, a rough mean frequency', 'time_shape'),
    'centroid': ('spectral', 'spectral centroid in kHz: power-weighted mean frequency (50 Hz-10 kHz)', 'spectral'),
    'spread': ('spectral', 'spectral spread in kHz: power-weighted std of frequency around the centroid', 'spectral'),
    'logflat': ('spectral', 'log spectral flatness = log(geometric mean / arithmetic mean of power); 0 for white noise, very negative for a tone', 'spectral'),
    'entropy': ('spectral', 'normalised spectral entropy -sum(q log q)/log(N) of the power spectrum as a distribution; 1 = flat, low = few peaks', 'spectral'),
    'rolloff': ('spectral', '85 % roll-off frequency in kHz: below it lies 85 % of the power', 'spectral'),
    'peaky': ('spectral', 'log(max power / median power): how far the strongest line stands above the floor', 'spectral'),
    'top3': ('spectral', 'share of total power in the 3 strongest frequency bins', 'spectral'),
    'fdom': ('spectral', 'dominant frequency in kHz (strongest bin above 50 Hz)', 'spectral'),
    'cpp': ('cepstrum', 'cepstral peak prominence at 0.5-3.5 ms quefrency: log(max / median) of the real cepstrum; high when the spectrum is a comb of evenly spaced harmonics (spindle lines), found without the rpm', 'cepstrum'),
    'corr': ('axes', 'Pearson correlation between the two axes within the window', 'axes'),
    'hj_mobility': ('hjorth', 'Hjorth mobility sqrt(var(dx)/var(x)): proportional to mean frequency (Hjorth 1970)', 'hjorth'),
    'hj_logcomplexity': ('hjorth', 'log Hjorth complexity = mobility(dx)/mobility(x); 1 for a pure sine, larger for broadband', 'hjorth'),
    'perm_entropy': ('perm_entropy', 'permutation entropy, order 4: entropy of the 24 rank patterns of 4 consecutive samples, 0..1; low for a steady tone (Bandt & Pompe 2002)', 'perm_entropy'),
    'tk_ratio': ('teager', 'log(mean Teager-Kaiser energy x[n]^2 - x[n-1]x[n+1] / variance): tracks frequency (Kaiser 1990)', 'teager'),
    'tk_logcv': ('teager', 'log(std / mean) of the Teager-Kaiser energy over the window: bursts of amplitude or frequency change', 'teager'),
    'sk_max': ('spec_kurtosis', 'maximum over frequency of the spectral kurtosis (128-sample STFT frames, hop 32); about -1 steady tone, 0 noise, large impulses (Antoni 2006)', 'spectral_kurtosis'),
    'sk_mean': ('spec_kurtosis', 'mean over frequency of the spectral kurtosis', 'spectral_kurtosis'),
    'katz_fd': ('katz', 'Katz fractal dimension of the waveform, amplitude-only variant; near 1 smooth/tonal, higher rough (Katz 1988)', 'katz_fd'),
    'coh': ('coherence', 'largest magnitude-squared coherence between the two axes above 1 kHz (Welch average of 128-sample frames)', 'coherence'),
    'tone_persist': ('sequence', 'share of the last 19 windows (0.5 s) whose dominant frequency is within 2 bins of this one; needs the stream history', 'sequence_features'),
    'flux': ('sequence', 'spectral flux: 1 - correlation between this and the previous window\'s standardised log spectrum', 'sequence_features'),
    'level_trend': ('sequence', 'change of log RMS over the last 19 windows (0.5 s)', 'sequence_features'),
}

CATCH22 = {
    'DN_HistogramMode_5': 'mode of the z-scored values, 5-bin histogram',
    'DN_HistogramMode_10': 'mode of the z-scored values, 10-bin histogram',
    'CO_f1ecac': 'first 1/e crossing of the autocorrelation function',
    'CO_FirstMin_ac': 'first minimum of the autocorrelation function',
    'CO_HistogramAMI_even_2_5': 'automutual information at lag 2, 5 equal-width bins',
    'CO_trev_1_num': 'time-reversibility statistic: mean of (x[t+1] - x[t])^3',
    'MD_hrv_classic_pnn40': 'share of successive differences larger than 0.04 sigma',
    'SB_BinaryStats_mean_longstretch1': 'longest run of consecutive values above the mean',
    'SB_TransitionMatrix_3ac_sumdiagcov': 'trace of the covariance of a 3-state transition matrix (lag = first autocorrelation zero)',
    'PD_PeriodicityWang_th0_01': 'periodicity estimate (Wang et al. 2007)',
    'CO_Embed2_Dist_tau_d_expfit_meandiff': 'exponential-fit mismatch of successive distances in a 2-D time-delay embedding',
    'IN_AutoMutualInfoStats_40_gaussian_fmmi': 'first minimum of the automutual information function (Gaussian estimator, up to lag 40)',
    'FC_LocalSimple_mean1_tauresrat': 'ratio of autocorrelation decay of the residuals of a 1-step mean forecast to the original',
    'DN_OutlierInclude_p_001_mdrmd': 'timing of successive extreme positive values (median of the index statistic)',
    'DN_OutlierInclude_n_001_mdrmd': 'timing of successive extreme negative values',
    'SP_Summaries_welch_rect_area_5_1': 'share of power in the lowest fifth of frequencies (Welch, rectangular window)',
    'SB_BinaryStats_diff_longstretch0': 'longest run of successive decreases',
    'SB_MotifThree_quantile_hh': 'entropy of pairs of symbols after coding values into 3 equiprobable letters',
    'SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1': 'scaling of fluctuations, rescaled-range fit (proportion of slow time scales)',
    'SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1': 'scaling of fluctuations, detrended fluctuation analysis',
    'SP_Summaries_welch_rect_centroid': 'centroid of the Welch power spectrum',
    'FC_LocalSimple_mean3_stderr': 'error of a 3-sample rolling-mean forecast',
}


def describe(name):
    """(group, axis, description, where) for one feature column name."""
    stem, axis = name.rsplit('_', 1)
    if axis in ('XY', 'XZ', 'YZ'):
        axis = f'{axis[0]}-{axis[1]} pair'
    if name.startswith('c22_'):
        key = stem[4:]
        return 'catch22', axis, f'catch22: {CATCH22[key]} (Lubba et al. 2019, pycatch22)', 'features.catch22'
    m = re.fullmatch(r'band(\d)k', stem)
    if m:
        k = int(m.group(1))
        lo = '50 Hz' if k == 0 else f'{k} kHz'
        return 'bands', axis, f'log share of power in {lo}-{k + 1} kHz (of all power above 50 Hz)', 'features.bands'
    m = re.fullmatch(r'wp(\d)', stem)
    if m:
        k = int(m.group(1))
        return ('wavelet', axis, f'log energy share of wavelet packet node {k} (Daubechies-4, level 3, '
                f'about {1.25 * k:.2f}-{1.25 * (k + 1):.2f} kHz)', 'features.wavelet_packet')
    m = re.fullmatch(r'lfcc(\d+)', stem)
    if m:
        return ('lfcc', axis, f'linear-frequency cepstral coefficient {m.group(1)}: DCT of the log energies '
                'in 24 triangular bands evenly spaced over 50 Hz-10 kHz', 'features.lfcc')
    group, text, fn = STEMS[stem]
    return group, axis, text, f'features.{fn}'
