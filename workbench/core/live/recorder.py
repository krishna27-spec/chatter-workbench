"""Record a live session to a folder that opens straight in Analyse (Live spec §6).

  signals.npy    float32, shape (channels, samples) in Fortran order: sample-interleaved on disk, so each block is
                 appended as it comes. The header (which holds the length) is rewritten every second, so a crash
                 or a pulled cable loses at most the last second, and np.load / the session reader open it as is.
  signals.lvm    time + channels as text, time 12 and data 9 significant digits (the .lvm writer's rules)
  decisions.csv  one row per model decision (25.6 ms): state 0–3, vote, each model's label and P(chatter), controller
  output.csv     one row per write: rpm command, volts per pin, writing on/off, why
  actions.csv    every action: time, who (user / controller / system), what
  session.json   hardware, pins, rate, sensitivities, software version, start time, notes
All files are written by one thread (the engine's recorder thread); this class is not thread-safe on purpose.
"""
from __future__ import annotations

import csv
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

HEADER_BYTES = 256
FLUSH_EVERY_S = 1.0


def npy_header(n_ch: int, n: int) -> bytes:
    """A fixed-size .npy v1.0 header for float32 (n_ch, n), Fortran order. Fixed size, so it can be rewritten in
    place as n grows (numpy's own writer would size it to the current shape)."""
    d = "{'descr': '<f4', 'fortran_order': True, 'shape': (%d, %d), }" % (n_ch, n)
    body = d.encode("latin1")
    hlen = HEADER_BYTES - 10
    body = body + b" " * (hlen - len(body) - 1) + b"\n"
    return b"\x93NUMPY\x01\x00" + hlen.to_bytes(2, "little") + body


class SessionRecorder:
    def __init__(self, folder: Path, names: list[str], units: list[str], fs: float, info: dict):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=False)
        self.names, self.units, self.fs = names, units, fs
        self.n = 0
        self._npy = open(self.folder / "signals.npy", "wb")
        self._npy.write(npy_header(len(names), 0))
        self._lvm = open(self.folder / "signals.lvm", "w", encoding="utf-8")
        self._dec = self._csv("decisions.csv")
        self._out = self._csv("output.csv")
        self._act = self._csv("actions.csv", ["t", "wall_time", "who", "what"])
        self._dec_cols = self._out_cols = None
        self._last_flush = time.monotonic()
        meta = {"started": datetime.now().isoformat(timespec="seconds"), "fs": fs, "channels": names,
                "units": units, "format": "workbench session 1", **info}
        (self.folder / "session.json").write_text(json.dumps(meta, indent=1, default=str))
        self.action(0.0, "system", "recording started")

    def _csv(self, name, header=None):
        f = open(self.folder / name, "w", newline="", encoding="utf-8")
        w = csv.writer(f)
        if header:
            w.writerow(header)
        return f, w

    @property
    def seconds(self) -> float:
        return self.n / self.fs

    def bytes_written(self) -> int:
        return sum(p.stat().st_size for p in self.folder.iterdir() if p.is_file())

    def signals(self, block: np.ndarray) -> None:
        block = np.asarray(block, dtype="<f4")
        self._npy.write(np.ascontiguousarray(block.T).tobytes())          # sample-interleaved = Fortran (ch, n)
        t = (self.n + np.arange(block.shape[1])) / self.fs
        rows = np.column_stack([t, block.T.astype(np.float64)])
        np.savetxt(self._lvm, rows, fmt=["%.12g"] + ["%.9g"] * block.shape[0], delimiter="\t")
        self.n += block.shape[1]
        if time.monotonic() - self._last_flush >= FLUSH_EVERY_S:
            self.flush()

    def decision(self, row: dict) -> None:
        if self._dec_cols is None:
            self._dec_cols = list(row)
            self._dec[1].writerow(self._dec_cols)
        self._dec[1].writerow([row.get(c, "") for c in self._dec_cols])

    def output(self, row: dict) -> None:
        if self._out_cols is None:
            self._out_cols = list(row)
            self._out[1].writerow(self._out_cols)
        self._out[1].writerow([row.get(c, "") for c in self._out_cols])

    def action(self, t: float, who: str, what: str) -> None:
        self._act[1].writerow([f"{t:.4f}", datetime.now().strftime("%H:%M:%S.%f")[:-3], who, what])

    def flush(self) -> None:
        pos = self._npy.tell()
        self._npy.seek(0)
        self._npy.write(npy_header(len(self.names), self.n))
        self._npy.seek(pos)
        for f in (self._npy, self._lvm, self._dec[0], self._out[0], self._act[0]):
            f.flush()
        self._last_flush = time.monotonic()

    def close(self) -> None:
        self.action(self.seconds, "system", "recording stopped")
        self.flush()
        for f in (self._npy, self._lvm, self._dec[0], self._out[0], self._act[0]):
            f.close()
