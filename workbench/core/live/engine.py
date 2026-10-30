"""The live engine: acquisition → ring buffer → (recorder, analysis) on background threads (Live spec §4).

  acquire thread   source.read() every 512 samples → ring buffer, and puts the block on two queues. It never waits
                   for the GUI, the disk or the models: the analysis queue is bounded and drops (and counts) blocks
                   if the models fall behind; the recorder queue keeps everything.
  analysis thread  top-3 vote → process state → controller (Liao & Young or the lab law) → output guard (and its
                   slew tick) → decision rows for the log and the recorder.
  recorder thread  writes the session files.
The GUI reads the ring buffer and snapshot() about 30 times a second.

Safety: stopping acquisition, an acquisition error, or close() stops the output (safe values, task closed), because
the controller cannot see what the spindle does without the input. A replayed input may only drive a simulated or
memory output (as scope.py refused --arm with --replay).
"""
from __future__ import annotations

import queue
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np

from workbench.core.live import speed_rule
from workbench.core.live.output import OutputGuard
from workbench.core.live.recorder import SessionRecorder
from workbench.core.live.ring import RingBuffer
from workbench.core.live.top3 import Top3, process_state

ANALYSIS_QUEUE = 200            # blocks (≈ 5 s): beyond this the models are too slow and blocks are dropped


class LiveEngine:
    def __init__(self, source, ring_seconds: float = 60.0, models: Top3 | None = None,
                 guard: OutputGuard | None = None, control: dict | None = None, models_factory=None):
        self.source = source
        self.fs = source.fs
        self.ring = RingBuffer(len(source.names), int(ring_seconds * source.fs))
        self.models = models
        self.models_factory = models_factory     # loaded on the analysis thread (~0.3 s), not the GUI thread
        self.guard = guard
        self.control_opts = control              # None: output holds the start speed (constant mode)
        self.controller = None
        self.decisions: deque = deque(maxlen=2000)
        self.events: deque = deque(maxlen=500)   # controller events and actions, newest last; "n" counts all
        self._n_events = 0
        self.recorder: SessionRecorder | None = None
        self.running = False
        self.error: str | None = None
        self.dropped = 0
        self.last: dict = {}
        self._stop = threading.Event()
        self._aq: queue.Queue = queue.Queue(maxsize=ANALYSIS_QUEUE)
        self._rq: queue.Queue = queue.Queue()
        self._threads: list[threading.Thread] = []
        self._t_start = None
        if guard is not None:
            guard.log = self._guard_log
        if guard is not None and getattr(source, "kind", "") == "replay" and not getattr(guard.writer, "simulated", False):
            raise ValueError("a replayed file cannot drive a real output module: choose a simulated module "
                             "or no output (the controller would act on old data)")

    # ------------------------------------------------------------------ control
    def start(self) -> None:
        if self.running:
            return
        self.source.start()
        self.fs = getattr(self.source, "actual_fs", self.fs)
        self._stop.clear()
        self.error = None
        self.running = True
        self._t_start = time.monotonic()
        self._make_controller()
        self._threads = [threading.Thread(target=f, daemon=True, name=n) for f, n in
                         ((self._acquire, "acquire"), (self._analyse, "analyse"), (self._record, "record"))]
        for t in self._threads:
            t.start()
        self.action("system", f"acquisition started: {self.source.label} at {self.fs:g} Hz")

    def _join_if_dead(self) -> None:
        """After an acquisition error the threads end by themselves; reap them so start() works again."""
        if not self.running and self._threads:
            for t in self._threads:
                t.join(timeout=1.0)
            self._threads = []

    def stop(self, why: str = "acquisition stopped") -> None:
        if not self.running and not self._threads:
            return
        if self.guard is not None:                 # output first, while the recorder thread can still log it
            try:
                self.guard.stop(why)
            except Exception as e:
                self.error = f"output stop: {e}"
        self.action("system", why)
        self.running = False
        self._stop.set()
        for t in self._threads:                    # the recorder thread drains its queue before it ends
            t.join(timeout=3.0)
        self._threads = []
        self.source.stop()
        self.stop_recording()

    def close(self) -> None:
        """App closing: everything off, output safe."""
        self.stop("workbench closed")
        if self.guard is not None:
            self.guard.stop("workbench closed")

    def recalibrate(self) -> None:
        if self.models is not None:
            self.models.recalibrate()
        self._make_controller()
        self.action("user", "recalibrate (spindle on, tool in the air)")

    def start_recording(self, folder: Path | None = None, info: dict | None = None) -> Path:
        if self.recorder is not None:
            return self.recorder.folder
        folder = folder or Path("recordings") / datetime.now().strftime("wb_%Y%m%d_%H%M%S")
        info = {**(info or {}), "source": self.source.label}
        self._rq.put(("open", (folder, self.source.names, self.source.units, self.fs, info)))
        deadline = time.monotonic() + 2.0
        while self.recorder is None and time.monotonic() < deadline and self.running:
            time.sleep(0.005)
        if self.recorder is None and not self.running:        # not running: open it here
            self.recorder = SessionRecorder(folder, self.source.names, self.source.units, self.fs, info)
        return Path(folder)

    def stop_recording(self) -> None:
        if self.running:
            self._rq.put(("close", None))
            deadline = time.monotonic() + 3.0
            while self.recorder is not None and time.monotonic() < deadline:
                time.sleep(0.005)
        elif self.recorder is not None:
            self.recorder.close()
            self.recorder = None

    def action(self, who: str, what: str) -> None:
        self._event(self.t_now(), who, what)

    def _event(self, t: float, who: str, what: str) -> None:
        self._n_events += 1
        self.events.append({"n": self._n_events, "t": t, "who": who, "what": what})
        self._rq.put(("action", (t, who, what)))

    def t_now(self) -> float:
        return self.ring.total / self.fs

    def snapshot(self) -> dict:
        g = self.guard
        return {**self.last, "running": self.running, "t": self.t_now(), "fs": self.fs, "error": self.error,
                "dropped": self.dropped, "recording": str(self.recorder.folder) if self.recorder else None,
                "rec_seconds": self.recorder.seconds if self.recorder else 0.0,
                "output_state": g.state if g else None, "volts": list(g.current) if g else [],
                "rpm": g.rpm if g else None,
                "controller": self.controller.state if self.controller else None}

    # ------------------------------------------------------------------ threads
    def _make_controller(self) -> None:
        self.controller = None
        if self.guard is None:
            return
        o = self.control_opts
        if not o:
            self.guard.set_rpm(self.guard.sp.start_rpm)
            return
        cls = speed_rule.LabController if o.get("law") == "lab" else speed_rule.SpeedController
        self.controller = cls(self.guard.sp.start_rpm, self.guard.sp.teeth, self.guard,
                              o.get("rpm_min", 10_000), o.get("rpm_max", 100_000), fs=self.fs,
                              max_step=o.get("max_step", 0.35), direction=o.get("direction", "nearest"),
                              settle_s=o.get("settle_s", 1.0), max_changes=o.get("max_changes", 1),
                              n_ch=min(3, len(self.source.names)))

    def _acquire(self) -> None:
        try:
            while not self._stop.is_set():
                block = self.source.read()
                self.ring.write(block)
                try:
                    self._aq.put_nowait(block)
                except queue.Full:
                    self.dropped += 1
                self._rq.put(("signals", block))
        except Exception as e:
            self.error = f"acquisition: {type(e).__name__}: {e}".splitlines()[0]
            self.running = False
            self._stop.set()
            if self.guard is not None:
                try:
                    self.guard.stop("acquisition error")
                except Exception:
                    pass

    def _analyse(self) -> None:
        hop_s = 0
        n = 0
        if self.models is None and self.models_factory is not None:
            self.last = {"loading_models": True}
            try:
                self.models = self.models_factory()
            except Exception as e:
                self.error = f"models: {type(e).__name__}: {e}".splitlines()[0]
            self.last = {}
        while not self._stop.is_set() or not self._aq.empty():
            try:
                block = self._aq.get(timeout=0.1)
            except queue.Empty:
                continue
            hop_s = block.shape[1] / self.fs
            t = (n + 1) * block.shape[1] / self.fs
            n += 1
            xyz = np.asarray(block[:3], dtype=np.float64)
            out = self.models.update(xyz) if (self.models is not None and xyz.shape[0] == 3) else None
            state = process_state(out)
            vote = out["vote"] if out else ""
            ev = None
            try:
                if self.controller is not None and xyz.shape[0] == 3:
                    ev = self.controller.update(xyz, t, vote)
                if self.guard is not None:
                    self.guard.tick(hop_s)
            except Exception as e:
                self.error = f"output: {type(e).__name__}: {e}".splitlines()[0]
                try:
                    self.guard.stop("output error")
                except Exception:
                    pass
            row = {"t": round(t, 4), "state": "" if state is None else state, "vote": vote}
            if out and self.models is not None:
                for m in self.models.models:
                    row[m] = out[m]
                    row[f"p_chatter_{m}"] = round(float(self.models.last_probs[m][-1]), 4)
            if self.controller is not None:
                row["controller"] = self.controller.state
            if ev is not None:
                row["chatter_f_Hz"] = round(ev["f"], 2) if np.isfinite(ev["f"]) else ""
                row["nu"] = round(ev["nu_now"], 3) if np.isfinite(ev["nu_now"]) else ""
                self._event(t, "controller", speed_rule.describe(ev))
            self.decisions.append(row)
            self._rq.put(("decision", row))
            self.last = {"state": state, "vote": vote, "labels": {k: v for k, v in (out or {}).items() if k != "vote"},
                         "probs": {m: p.tolist() for m, p in (self.models.last_probs.items() if self.models else [])},
                         "calibrating": self.models.calibrating if self.models else False,
                         "last_event": ev}

    def _record(self) -> None:
        while not self._stop.is_set() or not self._rq.empty():
            try:
                kind, item = self._rq.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                if kind == "open":
                    if self.recorder is None:
                        self.recorder = SessionRecorder(*item)
                        g = self.guard
                        if g is not None and g.state != "disarmed":    # what is already being sent
                            self._guard_log("output", values=list(g.current), why="recording started", rpm=g.rpm)
                elif kind == "close":
                    if self.recorder is not None:
                        self.recorder.close()
                        self.recorder = None
                elif self.recorder is not None:
                    if kind == "signals":
                        self.recorder.signals(item)
                    elif kind == "decision":
                        self.recorder.decision(item)
                    elif kind == "output":
                        self.recorder.output(item)
                    elif kind == "action":
                        self.recorder.action(*item)
            except Exception as e:
                self.error = f"recording: {type(e).__name__}: {e}".splitlines()[0]
                self.recorder = None

    def _guard_log(self, kind: str, **kw) -> None:
        t = self.t_now()
        if kind == "action":
            self._event(t, "system", kw["what"])
        elif kind == "output" and self.guard is not None:
            row = {"t": round(t, 4), "rpm_cmd": round(kw["rpm"], 1), "writing": int(self.guard.state == "writing"),
                   "why": kw["why"]}
            for p, v in zip(self.guard.pins, kw["values"]):
                row[f"{p.role}_V" if p.role != "manual" else f"V_{p.pin.replace('/', '_')}"] = round(v, 5)
            self._rq.put(("output", row))
