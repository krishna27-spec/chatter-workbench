"""Comma-separated text: read and write, keeping channel names and units.

    t [s],X [V],Y [V],Z [V]
    0,0.00306200003,-0.0159320002,-0.0158900004

One header row of "name [unit]" (unit optional), column 0 = time. Opens directly in Excel or MATLAB.
Also reads what Excel writes with a European or Indian locale (";" between columns, "," as the decimal
point, a byte-order mark in front), time in ms or us, and header-less files (then like .lvm).
Named csv_fmt.py, not csv.py: a module called csv would hide Python's own csv module.

Time, gap and cut-off-line rules are the .lvm reader's (lvm.read_numbers / lvm.to_recording).
"""
from __future__ import annotations

import csv
import io
import re
from pathlib import Path

import numpy as np

from ..recording import Recording
from ..registry import reader, writer
from .lvm import default_names, read_numbers, to_recording, write_numbers

TIME_NAMES = {"t", "time"}
TIME_UNITS = {"": 1.0, "s": 1.0, "ms": 1e-3, "us": 1e-6, "µs": 1e-6, "min": 60.0}
_NAME_UNIT = re.compile(r"^(.*?)\s*\[([^\]]*)\]\s*$")


def split_name_unit(field: str) -> tuple[str, str]:
    """'X [V]' -> ('X', 'V');  'X' -> ('X', '')."""
    m = _NAME_UNIT.match(field.strip())
    return (m.group(1), m.group(2).strip()) if m else (field.strip(), "")


def _sniff_sep(line: str) -> str:
    """The separator is whichever of tab / ; / , appears most in the first line (ties go to ',')."""
    counts = {s: line.count(s) for s in (",", ";", "\t")}
    return max(counts, key=lambda s: (counts[s], s == ","))


def _is_number(field: str, decimal: str) -> bool:
    try:
        float(field.strip().replace(decimal, ".") if decimal != "." else field)
        return True
    except ValueError:
        return False


@reader(".csv")
def read_csv(path: Path) -> Recording:
    path = Path(path)
    with open(path, "rb") as f:
        raw = f.readline()
        try:
            first = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            first = raw.decode("latin-1")
        sep = _sniff_sep(first)
        decimal = "," if sep == ";" else "."
        fields = next(csv.reader(io.StringIO(first), delimiter=sep))
        has_header = not _is_number(fields[0], decimal)
        start = len(raw) if has_header else 0
        try:
            table, notices = read_numbers(f, path.stat().st_size, start, sep=sep, decimal=decimal)
        except ValueError as e:
            raise ValueError(f"{path.name}: {e}") from e

    if not has_header:
        n_ch = table.shape[1] - 1
        return to_recording(path, table, default_names(n_ch), ["V"] * n_ch, notices)

    names, units = zip(*(split_name_unit(x) for x in fields))
    names, units = list(names), list(units)
    if names and names[-1] == "" and len(names) == table.shape[1] and np.isnan(table[:, -1]).all():
        names, units, table = names[:-1], units[:-1], table[:, :-1]      # Excel's trailing separator
    if names[0].lower() not in TIME_NAMES:
        raise ValueError(f"{path.name}: the first column must be time ('t [s]' or 'time'), found '{fields[0]}'. "
                         f"Without it the sample rate is unknown.")
    if units[0] not in TIME_UNITS:
        raise ValueError(f"{path.name}: unknown time unit '{units[0]}'; use one of {', '.join(u for u in TIME_UNITS if u)}")
    if len(names) != table.shape[1]:
        raise ValueError(f"{path.name}: the header has {len(names)} names but the rows have "
                         f"{table.shape[1]} columns of numbers")
    return to_recording(path, table, names[1:], units[1:], notices, time_scale=TIME_UNITS[units[0]])


@writer(".csv")
def write_csv(rec: Recording, path: Path) -> None:
    """Header 't [s],name [unit],...' (a name with a comma gets quoted), then the numbers as in .lvm."""
    with open(path, "w", newline="\n", encoding="utf-8") as f:
        csv.writer(f, lineterminator="\n").writerow(
            ["t [s]"] + [f"{c.name} [{c.unit}]" if c.unit else c.name for c in rec.channels])
        write_numbers(f, rec, ",")
