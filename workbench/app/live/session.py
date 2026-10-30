"""Glue between the Live panels and the engine: builds the source, the output guard and the engine from the
panels' choices, and is the one place the window asks to stop the output (Stop output from any workspace)."""
from __future__ import annotations

from dataclasses import dataclass, field

from PySide6.QtCore import QObject, Signal

from workbench.core.live.engine import LiveEngine
from workbench.core.live.output import NiWriter, OutputGuard, PinConfig, SpindleConfig
from workbench.core.live.sources import NiSource, ReplaySource
from workbench.core.recording import Recording


@dataclass
class InputChoice:
    kind: str                                   # "ni" or "replay"
    pins: list[str] = field(default_factory=list)
    names: list[str] = field(default_factory=list)
    units: list[str] = field(default_factory=list)
    per_volt: list[float] = field(default_factory=list)
    fs: float = 20_000.0
    vmax: float = 10.0
    replay: Recording | None = None
    models: bool = True


@dataclass
class OutputChoice:
    module: str | None = None                   # None: no output
    simulated: bool = True
    pins: list[PinConfig] = field(default_factory=list)
    spindle: SpindleConfig = field(default_factory=SpindleConfig)
    mode: str = "constant"                      # constant / liao / lab
    rpm_min: float = 10_000.0
    rpm_max: float = 100_000.0
    max_step: float = 0.35
    direction: str = "nearest"


class LiveSession(QObject):
    changed = Signal()                          # engine started / stopped / output state changed

    def __init__(self):
        super().__init__()
        self.engine: LiveEngine | None = None
        self.input: InputChoice | None = None
        self.output: OutputChoice | None = None

    @property
    def running(self) -> bool:
        return self.engine is not None and self.engine.running

    @property
    def guard(self) -> OutputGuard | None:
        return self.engine.guard if self.engine else None

    def start(self, inp: InputChoice, out: OutputChoice) -> None:
        if self.running:
            return
        if inp.kind == "replay":
            if inp.replay is None:
                raise ValueError("Open a recording in Analyse first, then replay it here.")
            source = ReplaySource(inp.replay)
        else:
            if not inp.pins:
                raise ValueError("Tick at least one input pin.")
            source = NiSource(inp.pins, inp.fs, inp.vmax, names=inp.names, units=inp.units, per_volt=inp.per_volt)
        guard = None
        if out.module and out.pins:
            writer = NiWriter()
            writer.simulated = out.simulated     # the engine lets only a simulated module follow a replay
            guard = OutputGuard(out.pins, out.spindle, writer)
        factory = None
        if inp.models and len(source.names) >= 3:
            from workbench.core.live.top3 import Top3
            fs = source.fs
            factory = lambda: Top3(fs=fs)                   # built on the engine's analysis thread
        control = None if out.mode == "constant" else {
            "law": "lab" if out.mode == "lab" else "liao", "rpm_min": out.rpm_min, "rpm_max": out.rpm_max,
            "max_step": out.max_step, "direction": out.direction}
        self.engine = LiveEngine(source, guard=guard, control=control, models_factory=factory)
        self.input, self.output = inp, out
        self.engine.start()
        self.changed.emit()

    def stop(self) -> None:
        if self.engine is not None:
            self.engine.stop("acquisition stopped by the user")
        self.changed.emit()

    def arm(self) -> list[float]:
        g = self.guard
        if g is None:
            raise RuntimeError("No output pins chosen (choose them, then start acquisition).")
        if not self.running:
            raise RuntimeError("Start acquisition first: the controller needs the input.")
        vals = g.arm()
        self.engine.action("user", "output armed")
        self.changed.emit()
        return vals

    def start_output(self) -> None:
        self.guard.start()
        self.changed.emit()

    def stop_output(self, why: str = "Stop output pressed") -> None:
        g = self.guard
        if g is not None and g.state != "disarmed":
            g.stop(why)
        self.changed.emit()

    def close(self) -> None:
        if self.engine is not None:
            self.engine.close()
