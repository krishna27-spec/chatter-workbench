"""Live chatter detection from the NI 9775 in the cDAQ-9174 (or a replayed recording).

The detector is the entropy rule from labels.py, made causal so it can run
while the machine cuts:

  * the no-cutting reference (level and per-axis entropy) comes from a
    calibration phase at the start, with the spindle on and the tool in the
    air, instead of from the quietest 20 % of the whole record;
  * the median smoothing looks only backwards (SMOOTH_S below), so a label
    lags the signal by about half the smoothing length plus one window;
  * the "chatter stays chatter while the level holds" rule is latched from
    the moment chatter is first seen and released when cutting stops.

Everything is relative to the calibration, so the input can stay in volts.

Live (cDAQ-9174 "cDAQ2" over USB: NI 9775 in slot 1 reads the accelerometer coupler outputs on
ai0..ai2, NI 9263 in slot 2 drives the spindle/feed inputs; "cDAQ1" is a simulated chassis):
    python -m rig.realtime --rpm 20000
Live now also prints what the three trained models say (weights in models/weights_rt/, made by
train_realtime.py); add --no-models for the rule alone.
Replay a recording through the same code, to check it against labels.py:
    python -m rig.realtime --rpm 20000 --replay data/raw/Acce_20k_1_r1.lvm

Chatter control (speed_control.py, Liao & Young 1996): --control adds a spindle speed
command on the NI 9263. It is a dry run (printed, logged, no voltage) until --arm:
    python -m rig.realtime --rpm 20000 --control --replay data/raw/test/Acce_20k_3_r1.lvm
    python -m rig.realtime --rpm 20000 --control --feed-channel cDAQ2Mod2/ao1 --arm
The machine defaults (Z = 2, 0..10 V = 0..140000 rpm, 10000..100000 rpm, fz = 3 um,
programmed feed 5 mm/s) are those of the lab's script "Chatter control using spindle speed
regulation.txt".
"""
import argparse
import csv
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

from pipeline import labels as lb
from pipeline import causal
from rig.fast_forest import FastForest
from rig import ni_hw
from rig import speed_control as sc

FS = lb.FS
WIN, HOP = lb.WIN, lb.HOP
CALIB_S = 2.0        # spindle on, tool in the air, at the start of every session
SMOOTH_S = 0.25      # backward median; labels.py uses 0.5 s centred


class OnlineDetector:
    """Feed HOP new samples (3, HOP) at a time; get one result per call."""

    def __init__(self, rpm, fs=FS, calib_s=CALIB_S, smooth_s=SMOOTH_S, n_ch=3):
        self.rpm, self.fs = rpm, fs
        self.buf = np.zeros((n_ch, WIN))
        self.filled = 0
        self.n_calib = max(int(calib_s * fs / HOP), 5)
        self.k_smooth = max(int(smooth_s * fs / HOP), 1)
        self.hist_lev = deque(maxlen=self.k_smooth)
        self.hist_ent = deque(maxlen=self.k_smooth)
        self.calib_lev, self.calib_ent = [], []
        self.air_level = self.air_ent = None
        self.onset_level = None          # level when chatter was first seen
        self.n = 0                       # windows processed

    @property
    def calibrated(self):
        return self.air_level is not None

    def _features(self):
        ent, lev = [], []
        for v in self.buf:
            tc, f, P = lb.frame_spectra(v, self.fs)          # one frame
            ent.append(lb.notched_entropy(P, f, self.rpm, self.fs)[0])
            lev.append(np.sqrt(P[0, f >= 50].sum()))
        return np.sqrt(np.sum(np.square(lev))), np.array(ent)

    def update(self, chunk):
        chunk = np.asarray(chunk, float)
        h = chunk.shape[1]
        self.buf = np.roll(self.buf, -h, axis=1)
        self.buf[:, -h:] = chunk
        self.filled += h
        if self.filled < WIN:
            return None
        self.n += 1
        t = (self.filled - WIN / 2) / self.fs              # window centre, s
        level, ent = self._features()
        self.hist_lev.append(level)
        self.hist_ent.append(ent)
        level_s = float(np.median(self.hist_lev))
        ent_s = np.median(np.array(self.hist_ent), axis=0)

        if not self.calibrated:
            self.calib_lev.append(level_s)
            self.calib_ent.append(ent_s)
            if len(self.calib_lev) >= self.n_calib:
                self.air_level = float(np.median(self.calib_lev))
                self.air_ent = np.median(np.array(self.calib_ent), axis=0)
            return dict(t=t, label=None, name='calibrating', level_ratio=np.nan, drop=np.nan)

        ratio = level_s / self.air_level
        drop = float(np.mean(1 - ent_s / self.air_ent))
        cutting = ratio > lb.CUT_FACTOR
        if not cutting:
            label, self.onset_level = lb.AIR, None
        elif drop >= lb.DROP_CHATTER:
            label = lb.CHATTER
            if self.onset_level is None:
                self.onset_level = level_s
        elif self.onset_level is not None and level_s >= self.onset_level:
            label = lb.CHATTER                              # broadband chatter
        elif drop < lb.DROP_STABLE:
            label = lb.STABLE
        else:
            label = lb.UNCERTAIN
        return dict(t=t, label=label, name=lb.NAMES[label], level_ratio=ratio, drop=drop)


class ModelPredictor:
    """The three trained models (level only, 1D CNN, CNN + GRU) on the live stream.

    Same input as in training: the causal cleaner of causal.py, divided by the std of the
    first CALIB_S seconds (spindle on, tool in the air), windows of 1024 samples per axis,
    hop 512. Each model is the mean of the probabilities of the 3 seeds in models/weights_rt/.
    The models answer no cutting / stable / chatter and do not need the spindle speed.
    """
    CLASSES = ['no cutting', 'stable', 'chatter']

    def __init__(self, folder='models/weights_rt', fs=FS, calib_s=CALIB_S, n_ch=3):
        import torch
        from training.predict_test import MODELS
        self.torch, self.cleaner = torch, causal.CausalCleaner(n_ch, fs)
        self.nets = {}
        for name, cls in MODELS.items():
            stem = name.replace(' + ', '_').replace(' ', '_')
            files = sorted(Path(folder).glob(f'{stem}_seed*.pt'))
            if not files:
                raise FileNotFoundError(f'no weights for {name} in {folder}/ (run train_realtime.py)')
            nets = []
            for f in files:
                m = cls(); m.load_state_dict(torch.load(f, map_location='cpu')); nets.append(m.eval())
            self.nets[name] = nets
        self.buf = np.zeros((n_ch, WIN)); self.filled = 0
        self.n_calib = int(calib_s * fs); self.calib = []; self.bg = None

    def update(self, chunk):
        """Returns {model name: class name} once a full window exists after calibration, else None."""
        clean = self.cleaner(chunk)
        h = clean.shape[1]
        self.buf = np.roll(self.buf, -h, axis=1); self.buf[:, -h:] = clean
        self.filled += h
        if self.bg is None:
            self.calib.append(clean)
            if sum(c.shape[1] for c in self.calib) >= self.n_calib:
                self.bg = np.concatenate(self.calib, axis=1)[:, :self.n_calib].std(1)[:, None]; self.calib = []
            return None
        if self.filled < WIN:
            return None
        x = self.torch.from_numpy((self.buf / self.bg).astype(np.float32))[None]
        out = {}
        with self.torch.no_grad():
            for name, nets in self.nets.items():
                p = sum(self.torch.softmax(m(x), 1)[0].numpy() for m in nets) / len(nets)
                out[name] = self.CLASSES[int(p.argmax())]
        return out


class Top3Predictor:
    """The top 3 models of the study (extra trees, random forest, RBF SVM on Y) from a models/
    package, run exactly as that package's predict.py --live: causal cleaning, each axis divided
    by its idle RMS from the first CALIB_S seconds, one probability vector per 51 ms window,
    label = argmax of the mean probability of the last SMOOTH windows. The vote is the class
    named by at least `vote` of the three (else 'stable' while cutting).
    """
    SMOOTH, SEQ = 10, 19                     # as predict.py: 0.26 s of smoothing, 0.5 s of history

    def __init__(self, package='models/chatter_model_v2_chatter3', vote=2, fs=FS, calib_s=CALIB_S, n_ch=3):
        import json
        import joblib
        sys.path.append(str(Path(package).resolve()))      # chatter_models (the SVM pickle needs it)
        import chatter_models as cm
        from pipeline import features as ft
        self.cm, self.ft, self.vote = cm, ft, vote
        meta = json.loads((Path(package) / 'meta.json').read_text())
        self.classes = meta['classes']
        self.models = {}
        for name, m in meta['models'].items():
            clf = joblib.load(Path(package) / f'{name}.joblib')
            if hasattr(clf, 'estimators_') and hasattr(clf.estimators_[0], 'tree_'):
                clf = FastForest(clf)           # same trees and probabilities, ~0.1 ms instead of ~7 ms
            elif hasattr(clf, 'n_jobs'):
                clf.n_jobs = 1
            self.models[name] = (clf, m['feature_names'])
        self.full = any(not set(fn) <= cm.WAVE1_NAMES for _, fn in self.models.values())
        self.cleaner = causal.CausalCleaner(n_ch, fs)
        self.win = np.zeros((n_ch, WIN)); self.n = 0
        self.n_calib = int(calib_s * fs / HOP); self.calib = []; self.idle = None
        self.history = deque(maxlen=self.SEQ)
        self.recent = {m: deque(maxlen=self.SMOOTH) for m in self.models}
        self.last_probs = {}                    # model -> smoothed [P(no cutting), P(stable), P(chatter)]

    def update(self, chunk):
        """Returns {model: label, ..., 'vote': label} once calibrated, else None."""
        self.win = np.concatenate([self.win[:, HOP:], self.cleaner(chunk)], 1)
        self.n += 1
        if self.n < WIN // HOP:
            return None
        if self.idle is None:
            self.calib.append(np.sqrt((self.win ** 2).mean(1)))
            if len(self.calib) >= self.n_calib:
                self.idle = np.mean(self.calib, 0)
            return None
        self.history.append(self.win / self.idle[:, None])
        cur = self.history[-1][None]
        if self.full:
            F, names, _ = self.ft.extract(cur, groups=tuple(self.ft.AGNOSTIC))
            S, s_names = self.ft.sequence_features(np.stack(self.history))
            F, names = np.concatenate([F, S[-1:]], 1), list(names) + s_names
        else:
            F, names = self.cm.wave1(cur)
        idx = {n: i for i, n in enumerate(names)}
        out = {}
        for m, (clf, fn) in self.models.items():
            self.recent[m].append(clf.predict_proba(np.nan_to_num(F[:, [idx[n] for n in fn]]))[0])
            p = np.mean(self.recent[m], 0)
            self.last_probs[m] = p
            out[m] = self.classes[int(p.argmax())]
        votes = list(out.values())
        win = [c for c in self.classes if votes.count(c) >= self.vote]
        out['vote'] = win[0] if win else 'stable'
        return out


# ---------------------------------------------------------------- sources

def replay_source(path, fs=FS, realtime=False):
    """Yield (3, HOP) chunks from an .lvm file or a viewer .f32 file."""
    if path.endswith('.f32'):
        x = np.fromfile(path, '<f4').reshape(3, -1)
    else:
        x = np.loadtxt(path)[:, 1:4].T
    t0 = time.perf_counter()
    for i in range(0, x.shape[1] - HOP + 1, HOP):
        if realtime:
            wait = t0 + (i + HOP) / fs - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
        yield x[:, i:i + HOP]


def daq_source(device, channels='ai0:2', fs=FS, vmax=10.0):
    """Yield (3, HOP) chunks from an NI C Series module, continuously."""
    import nidaqmx
    from nidaqmx.constants import AcquisitionType
    from nidaqmx.errors import DaqReadError
    from nidaqmx.stream_readers import AnalogMultiChannelReader

    # On the cDAQ-9174 + NI 9775, some task starts never deliver a sample (the first read
    # times out with -200284, even after 30 s); a module or chassis reset does not prevent
    # it, but the next start works. So a start whose first block is late is retried.
    for attempt in range(1, 6):
        with nidaqmx.Task() as task:
            task.ai_channels.add_ai_voltage_chan(f'{device}/{channels}', min_val=-vmax, max_val=vmax)
            n_ch = task.number_of_channels
            task.timing.cfg_samp_clk_timing(fs, sample_mode=AcquisitionType.CONTINUOUS,
                                            samps_per_chan=fs)      # 1 s driver buffer
            reader = AnalogMultiChannelReader(task.in_stream)
            task.start()
            buf = np.zeros((n_ch, HOP))
            try:
                reader.read_many_sample(buf, HOP, timeout=2.0)
            except DaqReadError:
                print(f'DAQ start {attempt} gave no samples in 2 s, restarting the task', file=sys.stderr)
                continue
            while True:
                yield buf.copy()
                reader.read_many_sample(buf, HOP, timeout=2.0)
    raise RuntimeError(f'{device}: no samples after 5 task starts')


# ---------------------------------------------------------------- main

COLOURS = {'no cutting': '\033[90m', 'stable': '\033[32m', 'chatter': '\033[1;31m',
           'uncertain': '\033[33m', 'calibrating': '\033[36m'}


def add_control_args(ap):
    """The chatter-control options, shared by realtime.py and scope.py."""
    g = ap.add_argument_group('chatter control (speed_control.py)')
    g.add_argument('--control', action='store_true', help='regulate the spindle speed when chatter is seen')
    g.add_argument('--teeth', type=int, default=2, help='number of cutting edges Z of the tool')
    g.add_argument('--rpm-per-volt', type=float, default=14000.0, help='spindle drive scale (10 V = 140000 rpm)')
    g.add_argument('--ao-offset', type=float, default=0.0, help='volts at 0 rpm on the drive input')
    g.add_argument('--ao-channel', default='auto', help='NI 9263 output wired to the drive speed input (auto: ao0 of the NI 9263 found)')
    g.add_argument('--v-min', type=float, default=0.0, help='lowest voltage ever written')
    g.add_argument('--v-max', type=float, default=10.0, help='highest voltage ever written (hard limit 10 V)')
    g.add_argument('--rpm-min', type=float, default=10000.0, help='lowest speed allowed')
    g.add_argument('--rpm-max', type=float, default=100000.0, help='highest speed allowed')
    g.add_argument('--max-step', type=float, default=0.35, help='largest change per step, fraction of the speed')
    g.add_argument('--direction', choices=['nearest', 'down', 'up'], default='nearest')
    g.add_argument('--law', choices=['liao', 'lab'], default='liao',
                   help="liao: change once on confirmed chatter, settle, hold (default); "
                        "lab: the lab script's rule every window, back to the start speed when not chatter")
    g.add_argument('--trigger', choices=['top3', 'model', 'rule'], default='top3',
                   help='chatter from the vote of the top 3 models (default), the CNN + GRU, or the entropy rule')
    g.add_argument('--package', default='models/chatter_model_v2_chatter3', help='top-3 model package')
    g.add_argument('--vote', type=int, default=2, help='how many of the top 3 must agree on chatter')
    g.add_argument('--settle', type=float, default=1.0, help='seconds to wait after a change')
    g.add_argument('--max-changes', type=int, default=3, help='changes allowed per cut')
    g.add_argument('--feed-channel', help='second NI 9263 output for the feed override: auto (ao1 of the NI 9263 found) or e.g. cDAQ2Mod2/ao1')
    g.add_argument('--fz', type=float, default=3.0, help='feed per tooth kept constant, um')
    g.add_argument('--program-feed', type=float, default=5.0, help='programmed feed, mm/s (= 5 V on the feed input)')
    g.add_argument('--arm', action='store_true',
                   help='allow writing to the NI 9263 (realtime.py: from the start; scope.py: when Start output is pressed)')


def make_controller(a, rpm=None):
    """SpindleOutput + SpeedController from the parsed control options. Returns (out, ctl)."""
    rpm = a.rpm if rpm is None else rpm
    out = sc.SpindleOutput(a.ao_channel, a.rpm_per_volt, a.ao_offset, a.v_min, a.v_max, allowed=a.arm,
                           feed_channel=a.feed_channel, z=a.teeth, fz_um=a.fz, program_feed=a.program_feed)
    return out, new_controller(a, out, rpm)


def new_controller(a, out, rpm):
    """A fresh controller on an existing output (after a gap in the data it calibrates again)."""
    cls = sc.LabController if a.law == 'lab' else sc.SpeedController
    return cls(rpm, a.teeth, out, a.rpm_min, a.rpm_max, max_step=a.max_step,
                              direction=a.direction, calib_s=CALIB_S, settle_s=a.settle,
                              max_changes=a.max_changes)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--rpm', type=float, required=True, help='spindle speed (used to notch its harmonics)')
    ap.add_argument('--device', default='auto', help='NI-DAQmx module name, or auto: found at start (ni_hw.py)')
    ap.add_argument('--channels', default='ai0:2', help='X, Y, Z accelerometer channels')
    ap.add_argument('--replay', help='.lvm or .f32 file to replay instead of the DAQ')
    ap.add_argument('--realtime', action='store_true', help='pace the replay at the true sample rate')
    ap.add_argument('--no-models', action='store_true', help='rule only, do not load the trained models')
    ap.add_argument('--weights', default='models/weights_rt', help='folder with the trained weights')
    ap.add_argument('--log', help='CSV file for one row per window')
    ap.add_argument('--quiet', action='store_true', help='print only label changes')
    add_control_args(ap)
    a = ap.parse_args(argv)
    if a.arm and (not a.control or a.replay):
        ap.error('--arm only with --control on live data')
    if a.replay and a.control:
        a.ao_channel = 'replay/ao0' if a.ao_channel == 'auto' else a.ao_channel     # dry run: never opened
        a.feed_channel = 'replay/ao1' if a.feed_channel == 'auto' else a.feed_channel
    ni_hw.resolve(a)                                # 'auto' -> the modules found now; prints what is used

    src = replay_source(a.replay, realtime=a.realtime) if a.replay else daq_source(a.device, a.channels)
    det = OnlineDetector(a.rpm)
    top3 = Top3Predictor(a.package, a.vote) if a.control and a.trigger == 'top3' else None
    # the top 3 take about 21 ms of the 25.6 ms hop, so the CNN models are not run beside them
    pred = None if a.no_models or top3 else ModelPredictor(a.weights)
    if a.trigger == 'model' and pred is None:
        a.trigger = 'rule'
    ctl = None
    if a.control:
        out, ctl = make_controller(a)
        if a.arm:
            out.start()                             # realtime.py: --arm writes from the start
        print(f"Control {'ARMED on ' + a.ao_channel if a.arm else 'dry run (no voltage)'}: "
              f"{a.rpm:.0f} rpm = {out.volts(a.rpm):.3f} V, limits {ctl.rpm_min:.0f}..{ctl.rpm_max:.0f} rpm, "
              f"trigger {a.trigger}" + (f", feed on {a.feed_channel} = {out.feed_volts(a.rpm):.3f} V"
                                         if a.feed_channel else ''))
    log = csv.writer(open(a.log, 'w', newline='', buffering=1)) if a.log else None
    if log:
        heads = list(top3.models) + ['vote'] if top3 else (list(pred.nets) if pred else [])
        log.writerow(['t', 'label', 'name', 'level_ratio', 'entropy_drop'] + heads
                     + (['rpm_cmd', 'spindle_V', 'feed_V', 'writing', 'f_chatter'] if ctl else []))
    if not a.replay:
        print(f'Calibrating for {CALIB_S:.0f} s: spindle on, tool NOT touching the workpiece.')
    last = None
    try:
        for chunk in src:
            r = det.update(chunk)
            m = pred.update(chunk) if pred else None
            m3 = top3.update(chunk) if top3 else None
            if m3:
                m = m3
            if r is None:
                continue
            ev = None
            if ctl:
                trig = {'top3': 'vote', 'model': 'CNN + GRU'}.get(a.trigger)
                trig = m[trig] if trig and m else ('no cutting' if a.trigger == 'top3' else r['name'])
                ev = ctl.update(chunk, r['t'], trig)
                det.rpm = ctl.rpm                   # the rule notches the harmonics of the new speed
            if log:
                log.writerow([f"{r['t']:.4f}", r['label'], r['name'],
                              f"{r['level_ratio']:.3f}", f"{r['drop']:.4f}"] + (list(m.values()) if m else [])
                             + ([f'{ctl.rpm:.1f}', f'{out.v:.4f}', '' if out.v_feed is None else f'{out.v_feed:.4f}',
                                  int(out.armed), f"{ev['f']:.1f}" if ev else ''] if ctl else []))
            key = (r['name'], tuple(m.values()) if m else None)
            if not a.quiet or key != last:
                tail = '' if not m else '   | ' + '  '.join(f'{k}: {v}' for k, v in m.items())
                print(f"{COLOURS[r['name']]}{r['t']:8.2f} s  rule {r['name']:<11s}"
                      f"  level x{r['level_ratio']:6.2f}  drop {r['drop']:+.3f}{tail}\033[0m")
            last = key
    except KeyboardInterrupt:
        pass
    finally:
        if ctl:
            ctl.close()                             # back to the starting speed, channel released
            print(f'Control: {len(ctl.events)} decisions, output back to {a.rpm:.0f} rpm')
    return det


if __name__ == '__main__':
    sys.exit(main() and 0)
