"""The top-3 models (extra trees, random forest, RBF SVM on Y) run live, as realtime.Top3Predictor (copied).

From a models/ package, exactly as that package's predict.py --live: causal cleaning (the package's causal.py),
each axis divided by its idle RMS from the first 2 s (spindle on, tool in the air), one probability vector per
51 ms window, label = argmax of the mean probability of the last 10 windows (0.26 s). The vote is the class named by
at least 2 of the 3 (else 'stable' while cutting). test_live_engine.py checks it against realtime.py on a lab file.
"""
from __future__ import annotations

import importlib
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np

from workbench.core.live.fast_forest import FastForest

from workbench.paths import model_package

DEFAULT_PACKAGE = model_package()           # models/… in the project, bundled models/… when installed
WIN, HOP = 1024, 512
CALIB_S = 2.0


class Top3:
    SMOOTH, SEQ = 10, 19

    def __init__(self, package: str | Path | None = None, vote: int = 2, fs: float = 20_000.0,
                 calib_s: float = CALIB_S, n_ch: int = 3):
        import joblib
        pkg = Path(package or model_package()).resolve()
        if not (pkg / "meta.json").exists():
            raise FileNotFoundError(f"no model package at {pkg} (needs meta.json)")
        if str(pkg) not in sys.path:
            sys.path.insert(0, str(pkg))            # the SVM pickle needs `chatter_models` importable by that name
        cm = importlib.import_module("chatter_models")
        ft = importlib.import_module("features")
        causal = importlib.import_module("causal")
        self.cm, self.ft, self.vote = cm, ft, vote
        meta = json.loads((pkg / "meta.json").read_text())
        self.classes = meta["classes"]
        self.models = {}
        for name, m in meta["models"].items():
            clf = joblib.load(pkg / f"{name}.joblib")
            if hasattr(clf, "estimators_") and hasattr(clf.estimators_[0], "tree_"):
                clf = FastForest(clf)               # same probabilities, ~0.1 ms instead of ~7 ms
            elif hasattr(clf, "n_jobs"):
                clf.n_jobs = 1
            self.models[name] = (clf, m["feature_names"])
        self.full = any(not set(fn) <= cm.WAVE1_NAMES for _, fn in self.models.values())
        self.cleaner = causal.CausalCleaner(n_ch, fs)
        self.win = np.zeros((n_ch, WIN))
        self.n = 0
        self.n_calib = int(calib_s * fs / HOP)
        self.calib, self.idle = [], None
        self.history = deque(maxlen=self.SEQ)
        self.recent = {m: deque(maxlen=self.SMOOTH) for m in self.models}
        self.last_probs: dict[str, np.ndarray] = {}

    @property
    def calibrating(self) -> bool:
        return self.idle is None

    def recalibrate(self) -> None:
        self.calib, self.idle = [], None
        self.history.clear()
        for d in self.recent.values():
            d.clear()

    def update(self, chunk: np.ndarray) -> dict | None:
        """chunk: (3, 512) volts. Returns {model: label, ..., 'vote': label} once calibrated, else None."""
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
        out["vote"] = win[0] if win else "stable"
        return out


def process_state(out: dict | None) -> int | None:
    """0 no cutting / 1 stable / 2 transition (stable, but a model says chatter) / 3 chatter; None = calibrating.
    The same rule as the session reader's State lane."""
    if out is None:
        return None
    v = out["vote"]
    if v == "no cutting":
        return 0
    if v == "chatter":
        return 3
    return 2 if any(lab == "chatter" for k, lab in out.items() if k != "vote") else 1
