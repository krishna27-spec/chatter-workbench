"""The output guard: the only code that writes volts, and the safety state machine around it (Live spec §3).

  Disarmed ──Arm (after the confirm dialog)──► Armed ──Start output──► Writing
      ▲                                                                    │
      └────────────── Stop output (from ANY state): safe values, close ◄───┘

Disarmed: no task open, nothing written. Armed: the task is open and every pin holds its safe value; commands are
remembered but not written. Writing: commands go out. Every value is clamped to the pin's [v_min, v_max] and to the
hard 10 V ceiling (V_LIMIT, as speed_control.py), and moves at most `slew` V/s (tick() walks it there). Stop output
writes the safe values and closes the task; so do close(), an exception in a write, and the app quitting.

Same interface as speed_control.SpindleOutput (rpm0, rpm_range, volts, set_rpm, v, v_feed, armed, close), so the
copied controllers drive it unchanged. The NI 9263 keeps its last voltage if this program dies or the USB cable comes
out, which is why the clamp bounds what any command can do.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np

V_LIMIT = 10.0
ROLES = ("spindle", "feed", "manual")


@dataclass
class PinConfig:
    pin: str                      # e.g. cDAQ2Mod3/ao0
    role: str = "manual"          # spindle / feed / manual
    v_min: float = 0.0
    v_max: float = 10.0
    slew: float | None = None     # V/s; None = jump at once
    safe: float | None = None     # V written on arm and stop; None = the start value of the role
    value: float = 0.0            # manual: target volts


@dataclass
class SpindleConfig:
    rpm_per_volt: float = 14_000.0     # lab: 10 V = 140 000 rpm
    offset_v: float = 0.0
    start_rpm: float = 20_000.0
    teeth: int = 2
    fz_um: float = 3.0                 # feed per tooth
    program_feed: float = 5.0          # mm/s programmed on the machine = 5 V (100 % override)


class MemoryWriter:
    """Records writes instead of sending them (tests, replay sessions)."""
    simulated = True

    def __init__(self):
        self.opened, self.closed, self.writes = None, False, []

    def open(self, pins: list[PinConfig]):
        self.opened = [p.pin for p in pins]

    def write(self, values: list[float]):
        self.writes.append(list(values))

    def close(self):
        self.closed = True


class NiWriter:
    """One NI-DAQmx analog-output task over the chosen pins (a module takes one output task)."""
    simulated = False

    def __init__(self):
        self.task = None

    def open(self, pins: list[PinConfig]):
        import nidaqmx
        task = nidaqmx.Task()
        try:
            for p in pins:
                task.ao_channels.add_ao_voltage_chan(p.pin, min_val=max(min(p.v_min, 0.0), -V_LIMIT),
                                                     max_val=min(p.v_max, V_LIMIT))
            task.start()
        except Exception:
            task.close()
            raise
        self.task = task

    def write(self, values: list[float]):
        self.task.write(values[0] if len(values) == 1 else values)

    def close(self):
        if self.task is not None:
            self.task.close()
            self.task = None


class OutputGuard:
    def __init__(self, pins: list[PinConfig], spindle: SpindleConfig | None = None, writer=None, log=None):
        roles = [p.role for p in pins]
        for r in roles:
            if r not in ROLES:
                raise ValueError(f"unknown role {r!r}; use one of {', '.join(ROLES)}")
        if roles.count("spindle") > 1 or roles.count("feed") > 1:
            raise ValueError("at most one spindle pin and one feed pin")
        if "feed" in roles and "spindle" not in roles:
            raise ValueError("a feed pin follows the spindle: choose a spindle pin too")
        self.pins = pins
        self.sp = spindle or SpindleConfig()
        self.writer = writer if writer is not None else MemoryWriter()
        self.log = log or (lambda kind, **kw: None)
        self.state = "disarmed"
        self.rpm0 = self.sp.start_rpm
        self.rpm = self.sp.start_rpm
        self.target = [self._command(p) for p in pins]     # what the commands ask for
        self.current = list(self.target)                    # what was last written (or would be)
        self._lock = threading.Lock()

    # ---- the SpindleOutput interface (used by the copied controllers)
    @property
    def armed(self) -> bool:
        """True while voltages really follow the commands (as SpindleOutput.armed)."""
        return self.state == "writing"

    @property
    def v(self) -> float | None:
        i = self._index("spindle")
        return None if i is None else self.current[i]

    @property
    def v_feed(self) -> float | None:
        i = self._index("feed")
        return None if i is None else self.current[i]

    def rpm_range(self) -> tuple[float, float]:
        i = self._index("spindle")
        lo, hi = (self.pins[i].v_min, self.pins[i].v_max) if i is not None else (0.0, V_LIMIT)
        hi = min(hi, V_LIMIT)
        return (lo - self.sp.offset_v) * self.sp.rpm_per_volt, (hi - self.sp.offset_v) * self.sp.rpm_per_volt

    def volts(self, rpm: float) -> float:
        i = self._index("spindle")
        return self._clamp(self.pins[i] if i is not None else None, self.sp.offset_v + rpm / self.sp.rpm_per_volt)

    def feed_volts(self, rpm: float) -> float:
        feed = self.sp.fz_um * self.sp.teeth * rpm / 60000.0          # mm/s at fz µm per tooth (lab step 8.13)
        i = self._index("feed")
        return self._clamp(self.pins[i] if i is not None else None, feed * 5.0 / self.sp.program_feed)

    def set_rpm(self, rpm: float) -> float:
        """New spindle speed command; written at once (within slew) if Writing. Returns the spindle volts."""
        with self._lock:
            self.rpm = float(rpm)
            self.target = [self._command(p) for p in self.pins]
            if self.state == "writing":
                self._step(None)
            return self.volts(rpm)

    def close(self, rpm: float | None = None) -> None:
        self.stop("close")

    # ---- the state machine
    def arm(self) -> list[float]:
        """Open the task and write every pin's safe value. Returns those values."""
        with self._lock:
            if self.state != "disarmed":
                return self.current
            self.writer.open(self.pins)
            self.state = "armed"
            self.current = [self._safe(p) for p in self.pins]
            self._write("arm")
            return self.current

    def start(self) -> None:
        with self._lock:
            if self.state != "armed":
                raise RuntimeError("Arm the output first (the confirm dialog), then Start output")
            self.state = "writing"
            self.log("action", what="start output")
            self._step(None)

    def stop(self, why: str = "stop") -> None:
        """Safe values out, task closed, Disarmed. Safe to call in any state, any number of times."""
        with self._lock:
            if self.state == "disarmed":
                return
            try:
                self.current = [self._safe(p) for p in self.pins]
                self._write(why)
            finally:
                try:
                    self.writer.close()
                finally:
                    self.state = "disarmed"
                    self.log("action", what=f"output off ({why})")

    def set_manual(self, pin: str, volts: float) -> None:
        with self._lock:
            for i, p in enumerate(self.pins):
                if p.pin == pin and p.role == "manual":
                    p.value = float(volts)
                    self.target[i] = self._command(p)
            if self.state == "writing":
                self._step(None)

    def tick(self, dt: float) -> None:
        """Move the written values toward the commands at the slew limits (call every hop while Writing)."""
        with self._lock:
            if self.state == "writing":
                self._step(dt)

    # ---- inside
    def _index(self, role):
        return next((i for i, p in enumerate(self.pins) if p.role == role), None)

    @staticmethod
    def _clamp(p: PinConfig | None, v: float) -> float:
        lo, hi = (p.v_min, p.v_max) if p is not None else (0.0, V_LIMIT)
        return float(np.clip(np.clip(v, lo, hi), -V_LIMIT, V_LIMIT))

    def _command(self, p: PinConfig) -> float:
        if p.role == "spindle":
            return self.volts(self.rpm)
        if p.role == "feed":
            return self.feed_volts(self.rpm)
        return self._clamp(p, p.value)

    def _safe(self, p: PinConfig) -> float:
        if p.safe is not None:
            return self._clamp(p, p.safe)
        if p.role == "spindle":
            return self.volts(self.rpm0)
        if p.role == "feed":
            return self.feed_volts(self.rpm0)
        return self._clamp(p, 0.0)

    def _step(self, dt: float | None) -> None:
        new = []
        for p, cur, tgt in zip(self.pins, self.current, self.target):
            if p.slew and dt is not None:
                d = p.slew * dt
                new.append(float(np.clip(tgt, cur - d, cur + d)))
            elif p.slew:
                new.append(cur)                      # with a slew limit, only tick() moves the value
            else:
                new.append(tgt)
        if new != self.current:
            self.current = new
            self._write("command")

    def _write(self, why: str) -> None:
        vals = [self._clamp(p, v) for p, v in zip(self.pins, self.current)]
        assert all(abs(v) <= V_LIMIT for v in vals)                  # never above 10 V, whatever happened above
        try:
            self.writer.write(vals)
        except Exception:
            self.state = "armed"                                       # so stop() below runs its full path
            self.log("action", what=f"write failed ({why}): output stopped")
            raise
        self.log("output", values=vals, why=why, rpm=self.rpm, t=time.time())
