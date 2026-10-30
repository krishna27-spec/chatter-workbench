"""Where live samples come from: an NI module's chosen pins, or a file replayed at its true rate.

Both deliver blocks of `block` samples per channel (512 = 25.6 ms at 20 kHz, the models' hop) from read(), blocking
until the block is due, so the engine runs the same way on the rig and on a replay.
"""
from __future__ import annotations

import time

import numpy as np

from workbench.core.recording import Recording


class ReplaySource:
    """Plays a Recording's channels at its true rate, looping. A simulated device: nothing comes from hardware."""
    kind = "replay"

    def __init__(self, rec: Recording, block: int = 512, realtime: bool = True, loop: bool = True, speed: float = 1.0):
        self.rec, self.block, self.realtime, self.loop, self.speed = rec, block, realtime, loop, speed
        self.names = [c.name for c in rec.channels]
        self.units = [c.unit for c in rec.channels]
        self.fs = self.actual_fs = rec.fs
        self.label = f"replay of {rec.source.name if rec.source else 'a recording'}"
        self._i = 0
        self._t0 = None

    def start(self) -> None:
        self._i, self._t0, self._sent = 0, time.perf_counter(), 0

    def read(self) -> np.ndarray:
        n = self.rec.n_samples
        if self._i + self.block > n:
            if not self.loop:
                raise EOFError("end of the replayed file")
            self._i = 0
        out = np.stack([c.data[self._i:self._i + self.block] for c in self.rec.channels]).astype(np.float32)
        self._i += self.block
        self._sent += self.block
        if self.realtime:
            wait = self._t0 + self._sent / (self.fs * self.speed) - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
        return out

    def stop(self) -> None:
        pass


class NiSource:
    """Continuous voltage input on the chosen pins of one NI module (one input task per module).

    On the cDAQ-9174 some task starts never deliver a sample (first read times out with -200284, even after 30 s);
    the next start works, so a start whose first block is late is retried (as realtime.daq_source)."""
    kind = "ni"

    def __init__(self, pins: list[str], fs: float = 20_000.0, vmax: float = 10.0, block: int = 512,
                 names: list[str] | None = None, units: list[str] | None = None,
                 per_volt: list[float] | None = None):
        self.pins, self.fs, self.vmax, self.block = pins, float(fs), vmax, block
        self.names = names or [p.split("/")[-1] for p in pins]
        self.units = units or ["V"] * len(pins)
        self.per_volt = np.asarray(per_volt or [1.0] * len(pins), dtype=np.float32)[:, None]
        self.label = ", ".join(pins)
        self.actual_fs = self.fs
        self.task = None
        self._first = None

    def start(self) -> None:
        import nidaqmx
        from nidaqmx.constants import AcquisitionType
        from nidaqmx.errors import DaqReadError
        from nidaqmx.stream_readers import AnalogMultiChannelReader
        for attempt in range(1, 6):
            task = nidaqmx.Task()
            try:
                for p in self.pins:
                    task.ai_channels.add_ai_voltage_chan(p, min_val=-self.vmax, max_val=self.vmax)
                task.timing.cfg_samp_clk_timing(self.fs, sample_mode=AcquisitionType.CONTINUOUS,
                                                samps_per_chan=int(self.fs))      # 1 s driver buffer
                self.actual_fs = float(task.timing.samp_clk_rate)                # what the module really gives
                reader = AnalogMultiChannelReader(task.in_stream)
                task.start()
                buf = np.zeros((len(self.pins), self.block))
                try:
                    reader.read_many_sample(buf, self.block, timeout=2.0)
                except DaqReadError:
                    task.close()
                    continue
            except Exception:
                task.close()
                raise
            self.task, self._reader, self._first = task, reader, buf
            return
        raise RuntimeError(f"{self.label}: no samples after 5 task starts")

    def read(self) -> np.ndarray:
        if self._first is not None:
            buf, self._first = self._first, None
        else:
            buf = np.zeros((len(self.pins), self.block))
            self._reader.read_many_sample(buf, self.block, timeout=2.0)
        return (buf.astype(np.float32) * self.per_volt)

    def stop(self) -> None:
        if self.task is not None:
            self.task.close()
            self.task = None
