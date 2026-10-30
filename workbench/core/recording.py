"""The one data object the whole app shares.

Every file reader (core/formats/*.py) returns a Recording, and every panel (app/panels/*.py) draws one,
so a panel never needs to know whether the data came from an .lvm, a .csv or a scope.py session.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Channel:
    """One measured signal: a name, a unit and its samples (1-D, evenly spaced at the recording's fs)."""
    name: str
    unit: str
    data: np.ndarray


@dataclass
class Event:
    """Something that happened during a run (an action, a controller decision).

    t     seconds on the recording's clock, or None when the file only gives a clock time
    wall  that clock time, e.g. "03:22:51", when known
    """
    t: float | None
    text: str
    wall: str | None = None


@dataclass
class Recording:
    """Channels sampled together at fs, plus whatever else the file had.

    t0      time of the first sample, s (non-zero after slicing, or for a file that starts late)
    meta    free-form facts from the file: gaps, notices, session settings
    tables  extra time series at their own rate, e.g. tables['decisions']['state'], each with a 't' column
    events  list of Event
    source  the file or folder it was read from
    """
    channels: list[Channel]
    fs: float
    t0: float = 0.0
    meta: dict = field(default_factory=dict)
    tables: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    events: list[Event] = field(default_factory=list)
    source: Path | None = None

    def __post_init__(self):
        if not self.fs or self.fs <= 0:
            raise ValueError(f"fs must be positive, got {self.fs}")
        if not self.channels:
            raise ValueError("a recording needs at least one channel")
        n = None
        for ch in self.channels:
            if ch.data.ndim != 1:
                raise ValueError(f"channel '{ch.name}' must be 1-D, got shape {ch.data.shape}")
            if n is None:
                n = len(ch.data)
            elif len(ch.data) != n:
                raise ValueError(f"channel '{ch.name}' has {len(ch.data)} samples, expected {n}")

    @property
    def n_samples(self) -> int:
        return len(self.channels[0].data)

    @property
    def duration(self) -> float:
        return self.n_samples / self.fs

    def time(self) -> np.ndarray:
        """Time of every sample, s (float64: float32 cannot tell 1799.99995 from 1800.0)."""
        return self.t0 + np.arange(self.n_samples, dtype=np.float64) / self.fs

    def channel(self, name: str) -> Channel:
        for ch in self.channels:
            if ch.name == name:
                return ch
        raise KeyError(f"no channel '{name}'; this recording has {', '.join(c.name for c in self.channels)}")

    def index_of(self, t: float) -> int:
        """Sample index at time t, kept inside 0 .. n_samples."""
        return int(min(max(round((t - self.t0) * self.fs), 0), self.n_samples))

    def slice(self, t_start: float, t_end: float) -> Recording:
        """The part between two times. The data are numpy views: nothing is copied, so slicing 100 M samples is free."""
        a, b = self.index_of(t_start), self.index_of(t_end)
        return Recording(
            channels=[Channel(c.name, c.unit, c.data[a:b]) for c in self.channels],
            fs=self.fs, t0=self.t0 + a / self.fs, meta=self.meta,
            tables=self.tables, events=self.events, source=self.source)
