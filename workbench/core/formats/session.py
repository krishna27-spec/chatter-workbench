"""scope.py session folders (data/recordings/session_<date>_<time>/), opened as one Recording.

    raw.lvm       signals, every sample (or signals.npy, the workbench recorder's exact float32 copy)
    control.csv   one row per 25.6 ms decision: vote, each model, speed command, volts, controller state
    events.txt    "24.27 s  text" or "03:22:51  text", one line per decision or Start / Stop
    session.json  the settings scope.py ran with (device, rpm, fs, ...)

The workbench's own recordings (data/recordings/wb_<date>_<time>/, core/live/recorder.py) open too: signals.npy,
decisions.csv (instead of control.csv), output.csv (one row per write) and actions.csv (instead of events.txt).

Becomes: channels from the signals, tables['decisions'] (every control.csv column + a process state 0-3),
tables['output'] (what was sent to the NI 9263), events, meta['session']. Any missing part is left empty
and named in meta['notice'], so a session cut short still opens.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..recording import Channel, Event, Recording
from ..registry import reader
from .lvm import ByteRange, complete_end, default_names, read_lvm

MODELS = ("extra_trees", "random_forest", "svm_y")
OUTPUT_COLUMNS = ("t", "spindle_V", "feed_V", "rpm_cmd", "writing")
_TIMED = re.compile(r"^\s*(-?[\d.]+)\s*s\s{2,}(.*)$")          # "24.27 s  chatter at ..."
_WALL = re.compile(r"^\s*(\d{1,2}:\d{2}:\d{2}(?:\.\d+)?)\s{2,}(.*)$")  # "03:22:51  OUTPUT STARTED ..."


def process_state(vote: np.ndarray, models: list[np.ndarray]) -> np.ndarray:
    """0 no cutting, 1 stable, 2 transition (vote stable but at least one model says chatter), 3 chatter,
    NaN when there is no vote yet (calibration)."""
    state = np.full(len(vote), np.nan)
    any_chatter = np.zeros(len(vote), bool)
    for m in models:
        any_chatter |= m == "chatter"
    state[vote == "no cutting"] = 0
    state[(vote == "stable") & ~any_chatter] = 1
    state[(vote == "stable") & any_chatter] = 2
    state[vote == "chatter"] = 3
    return state


def _read_control(path: Path) -> dict[str, np.ndarray]:
    """control.csv -> {column: array}. Numbers as float64 (blank = NaN), text as str (blank = '')."""
    with open(path, "rb") as f:
        end = complete_end(f, path.stat().st_size)
        f.seek(0)
        df = pd.read_csv(ByteRange(f, end), engine="c")
    cols = {}
    for name in df.columns:
        col = df[name]
        if pd.api.types.is_numeric_dtype(col):
            cols[name] = col.to_numpy(np.float64)
        else:
            cols[name] = col.fillna("").astype(str).to_numpy(object)
    if "vote" in cols:
        cols["state"] = process_state(cols["vote"], [cols[m] for m in MODELS if m in cols])
    return cols


def _seconds_of_day(clock: str) -> float:
    h, m, s = clock.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def _wall_to_t(decisions: dict[str, np.ndarray]):
    """A function clock time -> recording time, using control.csv's wall_time next to its t.
    Picks the first decision at or after that clock time (within one 25.6 ms row). None if not possible."""
    if "wall_time" not in decisions or "t" not in decisions:
        return None
    ok = decisions["wall_time"] != ""
    if not ok.any():
        return None
    wall = np.array([_seconds_of_day(w) for w in decisions["wall_time"][ok]])
    wall += 86400 * np.concatenate([[0], np.cumsum(np.diff(wall) < -43200)])   # a session across midnight
    t = decisions["t"][ok]

    def convert(clock: str) -> float | None:
        s = _seconds_of_day(clock)
        if s < wall[0] - 43200:
            s += 86400
        i = int(np.searchsorted(wall, s))
        return float(t[i]) if i < len(t) else None
    return convert


def _read_events(path: Path, wall_to_t) -> list[Event]:
    events = []
    for line in path.read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        if m := _TIMED.match(line):
            events.append(Event(float(m.group(1)), m.group(2).strip()))
        elif m := _WALL.match(line):
            clock = m.group(1)
            events.append(Event(wall_to_t(clock) if wall_to_t else None, m.group(2).strip(), wall=clock))
        else:
            events.append(Event(None, line.strip()))
    return events


def _read_signals(folder: Path, fs: float | None) -> Recording:
    npy = folder / "signals.npy"
    if npy.exists():
        if not fs:
            raise ValueError(f"{folder.name}: signals.npy needs the sample rate from session.json")
        data = np.load(npy, mmap_mode="r")             # memory-mapped: opens instantly, read when drawn
        names = default_names(data.shape[0])
        return Recording([Channel(n, "V", data[i]) for i, n in enumerate(names)], float(fs))
    return read_lvm(folder / "raw.lvm")


@reader(folder=True)
def read_session(path: Path) -> Recording:
    folder = Path(path)
    if not ((folder / "raw.lvm").exists() or (folder / "signals.npy").exists()):
        raise ValueError(f"{folder.name}: not a session folder (needs raw.lvm or signals.npy)")
    missing = []

    session = {}
    if (folder / "session.json").exists():
        session = json.loads((folder / "session.json").read_text())
    else:
        missing.append("session.json")
    fs = session.get("fs") or session.get("args", {}).get("fs")

    rec = _read_signals(folder, fs)

    decisions = {}
    if (folder / "control.csv").exists():                       # scope.py
        decisions = _read_control(folder / "control.csv")
        rec.tables["decisions"] = decisions
        rec.tables["output"] = {c: decisions[c] for c in OUTPUT_COLUMNS if c in decisions}
    elif (folder / "decisions.csv").exists():                   # the workbench recorder
        decisions = _read_control(folder / "decisions.csv")
        rec.tables["decisions"] = decisions
        if (folder / "output.csv").exists() and (folder / "output.csv").stat().st_size > 0:
            rec.tables["output"] = _read_control(folder / "output.csv")
    else:
        missing.append("control.csv")

    if (folder / "events.txt").exists():
        rec.events = _read_events(folder / "events.txt", _wall_to_t(decisions))
    elif (folder / "actions.csv").exists():
        acts = _read_control(folder / "actions.csv")
        rec.events = [Event(float(t), f"{who}: {what}", wall=str(w) or None)
                      for t, w, who, what in zip(acts["t"], acts["wall_time"], acts["who"], acts["what"])]
    else:
        missing.append("events.txt")

    rec.meta["session"] = session
    if missing:
        note = f"missing {', '.join(missing)} (those lanes stay empty)"
        rec.meta["notice"] = "; ".join(filter(None, [rec.meta.get("notice"), note]))
    return rec
