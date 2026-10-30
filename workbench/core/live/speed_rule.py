"""Liao & Young spindle-speed rule and the two controllers, copied from speed_control.py (the version that ran on
the rig on 2026-10-23), so the workbench stands alone. test_live_core.py checks that they give the same speeds and
events as the original.

Paper: Y. S. Liao and Y. C. Young, "A new on-line spindle speed regulation strategy for chatter control",
Int. J. Mach. Tools Manufact. 36(5), 651-660, 1996, doi:10.1016/0890-6955(95)00076-3.
Rule (their eq. 3): Z N / 60 = f / (n + nu), nu = 0.25 (phase 90 deg between undulations).

Changes from speed_control.py: FS and HOP are constants here instead of coming from labels.py, the default log is
silent (the workbench logs events itself), describe() has no terminal colours, and SpindleOutput is replaced by
output.OutputGuard (same interface: rpm0, rpm_range, volts, set_rpm, v_feed, armed, close).
"""
import numpy as np

FS = 20_000
HOP = 512
NU = 0.25             # best phase factor (paper eq. 8)
FFT_WIN = 2048        # 102 ms at 20 kHz, 9.8 Hz bins (refined by parabolic interpolation)
F_MIN = 100.0         # ignore everything below this (drift, the 50 Hz high-pass region)
F_MAX = 9000.0        # and above this: the top of the band near Nyquist (10 kHz) is not trusted
NOTCH_BINS = 1.5      # half-width of the spindle-harmonic notch, in bins (as labels.py)
V_LIMIT = 10.0        # hard ceiling on any voltage written to the NI 9263, whatever the options say


# ---------------------------------------------------------------- the rule

def candidate_speeds(f, z, nu=NU, n_max=200):
    """All spindle speeds (rpm) that put phase factor nu between cuts, for n = 0..n_max."""
    n = np.arange(n_max + 1)
    return n, 60.0 * f / (z * (n + nu))


def new_speed(f, rpm, z, rpm_min, rpm_max, nu=NU, max_step=0.35, direction='nearest'):
    """Speed closest to rpm that satisfies Z N / 60 = f / (n + nu) inside the limits.

    direction: 'nearest' (the paper's stated rule), 'down' (what its experiments did:
    always the next speed below), or 'up'.
    Returns (N, n), or (None, reason) when no allowed speed exists.
    """
    n, N = candidate_speeds(f, z, nu)
    ok = (N >= rpm_min) & (N <= rpm_max) & (np.abs(N - rpm) <= max_step * rpm)
    if direction == 'down':
        ok &= N < rpm
    elif direction == 'up':
        ok &= N > rpm
    if not ok.any():
        return None, 'no speed with nu=%.2f within the limits and step' % nu
    i = np.flatnonzero(ok)[np.argmin(np.abs(N[ok] - rpm))]
    return float(N[i]), int(n[i])


def phase_factor(f, rpm, z):
    """Present nu = fractional part of f / (Z N / 60): where we sit between two lobes."""
    x = f / (z * rpm / 60.0)
    return x - np.floor(x)


# ---------------------------------------------------------------- chatter frequency

def psd(buf, fs=FS):
    """Hann-windowed power spectrum summed over the axes. buf: (n_ch, FFT_WIN)."""
    x = (buf - buf.mean(1, keepdims=True)) * np.hanning(buf.shape[1])
    P = (np.abs(np.fft.rfft(x, axis=1)) ** 2).sum(0)
    return np.fft.rfftfreq(buf.shape[1], 1 / fs), P


def harmonic_mask(f, rpm, fs=FS, win=FFT_WIN):
    """True on bins within NOTCH_BINS of a spindle harmonic (tooth passing included)."""
    fsp = rpm / 60.0
    k = np.round(f / fsp)
    return (np.abs(f - k * fsp) <= NOTCH_BINS * fs / win) & (k >= 1)


def chatter_frequency(P, P_air, f, rpm, fs=FS):
    """Frequency (Hz) of the largest power above the air spectrum, off the spindle harmonics.

    Returns (f_chatter, prominence) with prominence = that excess / median excess of the
    other usable bins (large = one clear tone, about 1 = nothing stands out).
    f_chatter is None when the largest excess sits on the edge of the usable band (next to
    F_MIN, F_MAX or a notch): that is a slope, not a tone, e.g. chatter above Nyquist.
    """
    usable = (f >= F_MIN) & (f <= F_MAX) & ~harmonic_mask(f, rpm, fs, 2 * (len(f) - 1))
    ex = np.where(usable, P - P_air, -np.inf)
    i = int(np.argmax(ex))
    pos = ex[usable & (ex > -np.inf)]
    prom = float(ex[i] / (np.median(np.abs(pos)) + 1e-30))
    if not (0 < i < len(P) - 1 and usable[i - 1] and usable[i + 1]):
        return None, prom
    # parabolic interpolation of the peak on log power, a standard refinement for a
    # Hann-windowed tone (J. O. Smith, Spectral Audio Signal Processing, "Quadratic
    # Interpolation of Spectral Peaks")
    df = f[1] - f[0]
    fc = f[i]
    if 0 < i < len(P) - 1:
        a, b, c = np.log(P[i - 1:i + 2] + 1e-30)
        den = a - 2 * b + c
        if den < 0:
            fc += 0.5 * (a - c) / den * df
    return float(fc), prom


class SpeedController:
    """Watches the detector and changes the spindle speed when chatter persists.

    States:
      calibrating  first calib_s seconds: average the air spectrum (spindle on, tool out)
      watching     counts consecutive chatter hops; after confirm_hops, measures f and acts
      settling     after a change, waits settle_s for the spindle to reach the speed and
                   for the FFT window to fill with data at that speed
      holding      max_changes reached in this cut (or no allowed speed): no more changes
                   until the tool leaves the workpiece
    """

    def __init__(self, rpm, z, out, rpm_min, rpm_max, fs=FS, nu=NU, max_step=0.35, direction='nearest',
                 calib_s=2.0, confirm_hops=4, settle_s=1.0, max_changes=3, min_change=0.005,
                 n_ch=3, hop=HOP, log=lambda _msg: None):
        self.rpm0 = self.rpm = float(rpm)
        self.z, self.out, self.fs, self.nu = z, out, fs, nu
        out.rpm0 = self.rpm0                    # for the feed override % (lab script step 8.12)
        lo, hi = out.rpm_range()
        self.rpm_min, self.rpm_max = max(rpm_min, lo), min(rpm_max, hi)
        self.max_step, self.direction = max_step, direction
        self.confirm, self.max_changes, self.min_change = confirm_hops, max_changes, min_change
        self.n_calib = max(int(calib_s * fs / hop), 1)
        self.n_settle = int(np.ceil(settle_s * fs / hop))
        self.buf = np.zeros((n_ch, FFT_WIN))
        self.filled = 0
        self.air, self.n_air = None, 0
        self.state, self.count, self.wait, self.changes = 'calibrating', 0, 0, 0
        self.log = log
        self.events = []
        out.set_rpm(self.rpm)                   # start from the operator's speed

    def _push(self, chunk):
        h = chunk.shape[1]
        self.buf = np.roll(self.buf, -h, axis=1)
        self.buf[:, -h:] = chunk
        self.filled += h

    def update(self, chunk, t, label):
        """chunk: (n_ch, hop) raw volts; label: 'no cutting' / 'stable' / 'chatter' / other.

        Returns an event dict when the speed is changed (or a change is refused), else None.
        """
        self._push(np.asarray(chunk, float))
        if self.filled < FFT_WIN:
            return None
        f, P = psd(self.buf, self.fs)

        if self.state == 'calibrating':
            self.air = P if self.air is None else self.air + P
            self.n_air += 1
            if self.n_air >= self.n_calib:
                self.air /= self.n_air
                self.state = 'watching'
            return None

        if label == 'no cutting':               # tool out: a new cut starts afresh
            self.count, self.changes = 0, 0
            if self.state == 'holding':
                self.state = 'watching'

        if self.state == 'settling':
            self.wait -= 1
            if self.wait <= 0:
                self.state = 'watching'
            return None
        if self.state == 'holding':
            return None

        self.count = self.count + 1 if label == 'chatter' else 0
        if self.count < self.confirm:
            return None
        self.count = 0

        fc, prom = chatter_frequency(P, self.air, f, self.rpm, self.fs)
        if fc is None:                          # no clear tone: keep watching, change nothing
            ev = dict(t=t, f=np.nan, prominence=prom, rpm_from=self.rpm, nu_now=np.nan,
                      action='refused', reason='largest excess is on a band edge, not a tone')
            self.events.append(ev)
            self.log(describe(ev))
            return ev
        ev = dict(t=t, f=fc, prominence=prom, rpm_from=self.rpm,
                  nu_now=float(phase_factor(fc, self.rpm, self.z)))
        N, n = new_speed(fc, self.rpm, self.z, self.rpm_min, self.rpm_max, self.nu,
                         self.max_step, self.direction)
        if N is None:
            ev.update(action='refused', reason=n)
            self.state = 'holding'
        elif abs(N - self.rpm) < self.min_change * self.rpm:
            ev.update(action='refused', rpm_to=N, n=n,
                      reason='already at nu=%.2f: speed alone cannot fix this, reduce the depth of cut' % self.nu)
            self.state = 'holding'
        else:
            v = self.out.set_rpm(N)
            self.rpm = N
            self.changes += 1
            ev.update(action='changed', rpm_to=N, n=n, volts=v, feed_volts=self.out.v_feed,
                      armed=self.out.armed)
            self.wait = self.n_settle
            self.state = 'holding' if self.changes >= self.max_changes else 'settling'
        self.events.append(ev)
        self.log(describe(ev))
        return ev

    def close(self):
        """Put the operator's starting speed back on the output and release the channel."""
        self.out.close(self.rpm0)


class LabController(SpeedController):
    """The control law of the lab's script (Chatter control using spindle speed regulation.txt,
    steps 8.10 - 8.13), applied window by window on the live stream:

      chatter      new speed = the nu = 0.25 speed closest to the STARTING speed, from this
                   window's chatter frequency, inside [rpm_min, rpm_max] (no step limit)
      otherwise    back to the starting speed

    So the output follows the vote hop by hop, with no confirmation, settling or change cap
    (the script was written for offline processing of a whole recording). Same calibration,
    chatter frequency and output (spindle V, feed V, override %) as SpeedController.
    """

    def update(self, chunk, t, label):
        self._push(np.asarray(chunk, float))
        if self.filled < FFT_WIN:
            return None
        f, P = psd(self.buf, self.fs)
        if self.state == 'calibrating':
            self.air = P if self.air is None else self.air + P
            self.n_air += 1
            if self.n_air >= self.n_calib:
                self.air /= self.n_air
                self.state = 'watching'
            return None

        target, ev = self.rpm0, None
        if label == 'chatter':
            self.state = 'chatter'
            fc, prom = chatter_frequency(P, self.air, f, self.rpm, self.fs)
            if fc is None:                      # no clear tone in this window: keep the command
                return None
            N, n = new_speed(fc, self.rpm0, self.z, self.rpm_min, self.rpm_max, self.nu, max_step=np.inf)
            ev = dict(t=t, f=fc, prominence=prom, rpm_from=self.rpm,
                      nu_now=float(phase_factor(fc, self.rpm, self.z)))
            if N is None:                       # script: no candidate -> keep the starting speed
                ev.update(reason=n)
            else:
                target = N
                ev.update(n=n)
        else:
            self.state = 'watching'
        if abs(target - self.rpm) < 1e-6:
            return None
        v = self.out.set_rpm(target)
        ev = ev or dict(t=t, f=np.nan, prominence=np.nan, rpm_from=self.rpm, nu_now=np.nan,
                        reason='no chatter: back to the starting speed')
        ev.update(action='changed', rpm_to=target, n=ev.get('n', -1), volts=v, feed_volts=self.out.v_feed,
                  armed=self.out.armed)
        self.rpm = target
        self.changes += 1
        self.events.append(ev)
        self.log(describe(ev))
        return ev


def describe(ev):
    if ev['action'] == 'changed' and np.isnan(ev['f']):          # lab law: chatter gone
        feed = '' if ev['feed_volts'] is None else f", feed {ev['feed_volts']:.3f} V"
        return (f"{ev['t']:8.2f} s  {ev['reason']}  ->  {ev['rpm_to']:.0f} rpm "
                f"({ev['volts']:.3f} V{feed}, {'SENT' if ev['armed'] else 'dry run'})")
    head = (f"{ev['t']:8.2f} s  chatter at {ev['f']:7.1f} Hz (prominence {ev['prominence']:.0f}, "
            f"nu now {ev['nu_now']:.2f}) at {ev['rpm_from']:.0f} rpm")
    if ev['action'] == 'changed':
        tag = 'SENT' if ev['armed'] else 'dry run'
        feed = '' if ev['feed_volts'] is None else f", feed {ev['feed_volts']:.3f} V"
        return (f"{head}  ->  {ev['rpm_to']:.0f} rpm (n={ev['n']}, {ev['volts']:.3f} V{feed}, "
                f"{tag})")
    return f"{head}  ->  no change: {ev['reason']}"
