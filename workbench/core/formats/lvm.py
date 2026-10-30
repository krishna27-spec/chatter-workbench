"""LabVIEW .lvm text files: read and write.

Layout: tab-separated numbers, one row per sample, column 0 = time (s), the other columns = channels.
An optional LabVIEW header sits on top (one or two blocks ending in ***End_of_Header***, then a line of
column names); the lab's and scope.py's files have none. Exactly 3 channels are named X, Y, Z
(feed, normal-to-feed, axial), any other count ch1..chN. Unit V.

Only fs and t0 are kept from the time column (Recording computes time as t0 + i/fs). The column is still
checked: jumps and restarts go into meta['gaps'], so a broken clock is reported, not hidden.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..recording import Channel, Recording
from ..registry import reader, writer

HEADER_SCAN_LINES = 10_000        # a LabVIEW header is a few dozen lines; stop looking after this
SNAP_REL = 1e-4                   # fs within 100 ppm of a whole number of Hz is taken as that number
WRITE_CHUNK = 1_000_000           # rows per write, so a 30 min file never needs one 1 GB text table


def time_to_fs(t: np.ndarray) -> tuple[float, list[dict]]:
    """Sample rate from a time column, plus every place where the clock does not step evenly.

    1. The median step is the normal step. A median ignores a few huge jumps, a mean would not.
    2. Steps outside 0.5 .. 1.5 x the median are gaps (clock jumped ahead) or restarts (clock went back).
    3. fs = number of normal steps / their summed length. Using the sum, not one step, matters for files
       written with 6 decimals: at 51.2 kHz one step reads as 0.000019 or 0.000020, but the rounding
       errors cancel along the file, so the sum is exact to about 1e-6 s.
    4. fs within 100 ppm of a whole number snaps to it (51200.024 -> 51200).
    """
    if len(t) < 2:
        raise ValueError("needs at least 2 rows to find the sample rate")
    steps = np.diff(t)
    med = float(np.median(steps))
    if not med > 0:
        raise ValueError("the time column does not increase")
    normal = (steps > 0.5 * med) & (steps < 1.5 * med)
    fs = float(normal.sum() / steps[normal].sum())
    if abs(fs - round(fs)) <= SNAP_REL * fs:
        fs = float(round(fs))
    gaps = [{"at_s": float(t[i]), "missing_s": float(steps[i] - 1 / fs),
             "kind": "gap" if steps[i] > 0 else "restart"}
            for i in np.flatnonzero(~normal)]
    return fs, gaps


def _is_number(field: bytes) -> bool:
    try:
        float(field)
        return True
    except ValueError:
        return False


def _data_start(f) -> int:
    """Byte offset of the first row whose first field is a number (skips any LabVIEW header)."""
    pos = 0
    for _ in range(HEADER_SCAN_LINES):
        line = f.readline()
        if not line:
            break
        if _is_number(line.split(b"\t", 1)[0].strip()):
            return pos
        pos += len(line)
    raise ValueError("no rows of numbers found")


def complete_end(f, size: int) -> int:
    """Byte offset just after the last newline: a line cut off mid-write has none."""
    back = min(size, 1 << 16)
    f.seek(size - back)
    tail = f.read(back)
    if tail.endswith(b"\n"):
        return size
    cut = tail.rfind(b"\n")
    return size - back + cut + 1 if cut >= 0 else 0


class ByteRange:
    """Read-only file view from the current position up to byte `end`, so pandas never sees a cut-off line."""

    def __init__(self, f, end: int):
        self.f, self.left = f, end - f.tell()

    def read(self, n: int = -1) -> bytes:
        n = self.left if n is None or n < 0 else min(n, self.left)
        b = self.f.read(n)
        self.left -= len(b)
        return b

    def __iter__(self):                    # pandas checks for this before treating it as a file
        return iter(lambda: self.read(1 << 20), b"")


def default_names(n_ch: int) -> list[str]:
    """X, Y, Z for exactly three channels (feed, normal-to-feed, axial), else ch1..chN."""
    return ["X", "Y", "Z"] if n_ch == 3 else [f"ch{i + 1}" for i in range(n_ch)]


def read_numbers(f, size: int, start: int, sep: str, decimal: str = ".") -> tuple[np.ndarray, list[str]]:
    """Parse the numbers from byte `start` to the last complete line. Returns (table, notices).

    The table is float64, one row per sample. A line cut off mid-write (no newline, or fewer columns
    than the rest) is dropped and reported in the notices.
    """
    notices = []
    end = complete_end(f, size)
    if end < size:
        notices.append("incomplete last line dropped (file cut off while it was being written)")
    if start >= end:
        raise ValueError("no complete rows of numbers found")
    f.seek(start)
    try:
        table = pd.read_csv(ByteRange(f, end), sep=sep, decimal=decimal, header=None, dtype=np.float64,
                            engine="c", skip_blank_lines=True).to_numpy()
    except pd.errors.ParserError as e:
        raise ValueError(str(e)) from e
    if len(table) > 1 and np.isnan(table[-1]).any() and not np.isnan(table[0]).any():
        table = table[:-1]
        notices.append("incomplete last line dropped (fewer columns than the rest)")
    return table, notices


def to_recording(path: Path, table: np.ndarray, names: list[str], units: list[str],
                 notices: list[str], time_scale: float = 1.0) -> Recording:
    """Column 0 = time (times time_scale gives seconds), the rest = channels -> Recording with fs, t0, gaps."""
    if table.shape[1] < 2:
        raise ValueError(f"{path.name}: needs a time column and at least one channel, "
                         f"found {table.shape[1]} column")
    t = table[:, 0] * time_scale
    try:
        fs, gaps = time_to_fs(t)
    except ValueError as e:
        raise ValueError(f"{path.name}: {e}") from e
    channels = [Channel(n, u, table[:, i + 1].astype(np.float32)) for i, (n, u) in enumerate(zip(names, units))]
    meta = {}
    if gaps:
        meta["gaps"] = gaps
    if notices:
        meta["notice"] = "; ".join(notices)
    return Recording(channels, fs, t0=float(t[0]), meta=meta)


@reader(".lvm")
def read_lvm(path: Path) -> Recording:
    path = Path(path)
    try:
        with open(path, "rb") as f:
            table, notices = read_numbers(f, path.stat().st_size, _data_start(f), sep="\t")
    except ValueError as e:
        raise ValueError(f"{path.name}: {e}") from e
    if table.shape[1] > 2 and np.isnan(table[:, -1]).all():     # LabVIEW's empty "Comment" column
        table = table[:, :-1]
    n_ch = table.shape[1] - 1
    return to_recording(path, table, default_names(n_ch), ["V"] * n_ch, notices)


def write_numbers(f, rec: Recording, delimiter: str) -> None:
    """Time with 12 significant digits (0.1 us or better up to 10 000 s), data with 9 (enough to give back
    every float32 bit-for-bit). A million rows at a time, so a 30 min file never needs one 1 GB table."""
    fmt = ["%.12g"] + ["%.9g"] * len(rec.channels)
    for a in range(0, rec.n_samples, WRITE_CHUNK):
        b = min(a + WRITE_CHUNK, rec.n_samples)
        t = rec.t0 + np.arange(a, b, dtype=np.float64) / rec.fs
        np.savetxt(f, np.column_stack([t] + [c.data[a:b] for c in rec.channels]), fmt=fmt, delimiter=delimiter)


@writer(".lvm")
def write_lvm(rec: Recording, path: Path) -> None:
    """Same layout as the lab files: tab-separated, no header (so names and units are not kept; use .csv)."""
    with open(path, "w", newline="\n") as f:
        write_numbers(f, rec, "\t")
