"""The analysis state of one open file: its Steps (Raw → Clean) and cached spectrograms, computed in the background.

Panels ask for a source ("raw" or "clean") or a spectrogram and get a callback when it is ready. Work that is
already running is not started twice: later callers join the waiting list.
"""
from __future__ import annotations

from typing import Callable

from PySide6.QtCore import QObject, Signal

from workbench.app.loader import run_async
from workbench.core.analysis.spectrum import Spectrogram, spectrogram
from workbench.core.recording import Recording
from workbench.core.steps import OPS, Step, load_ops

SOURCES = ("raw", "clean")


class AnalysisSession(QObject):
    status = Signal(str)                      # "Cleaning…" / "" for the window's status bar

    def __init__(self, rec: Recording):
        super().__init__()
        self.rec = rec
        if "clean" not in OPS:
            load_ops()
        self.steps = {"raw": Step.root(rec)}
        self.steps["clean"] = Step("clean", input=self.steps["raw"])
        self._waiting: dict[tuple, list] = {}
        self._sg: dict[tuple[str, str], Spectrogram] = {}

    def ready(self, source: str) -> Recording | None:
        """The source's Recording if it is already computed, else None (no work started)."""
        step = self.steps[source]
        return step.output() if step.is_cached else None

    def recording(self, source: str, on_ready: Callable[[Recording], None], on_error: Callable[[str], None]) -> None:
        rec = self.ready(source)
        if rec is not None:
            on_ready(rec)
            return
        step = self.steps[source]
        self._run(("source", source), step.output, on_ready, on_error,
                  f"Cleaning all channels (zero-phase needs the whole file, ~6 s per 700 MB)…")

    def spectrogram(self, source: str, channel: str, on_ready: Callable[[Spectrogram], None],
                    on_error: Callable[[str], None]) -> None:
        key = (source, channel)
        if key in self._sg:
            on_ready(self._sg[key])
            return

        def compute_on(rec: Recording):
            def work():
                return spectrogram(rec.channel(channel).data, rec.fs)

            def keep(sg):
                self._sg[key] = sg
                return sg
            self._run(("sg",) + key, work, lambda sg: on_ready(keep(sg)), on_error, f"Spectrogram of {channel}…")

        self.recording(source, compute_on, on_error)

    def rename_channel(self, old: str, new: str) -> None:
        for key in [k for k in self._sg if k[1] == old]:
            self._sg[(key[0], new)] = self._sg.pop(key)
        clean = self.ready("clean")
        if clean is not None:
            for ch in clean.channels:
                if ch.name == old:
                    ch.name = new

    def _run(self, key, work, on_ready, on_error, message):
        if key in self._waiting:                  # already running: just wait for it too
            self._waiting[key].append((on_ready, on_error))
            return
        self._waiting[key] = [(on_ready, on_error)]
        self.status.emit(message)

        def done(result):
            for ok, _ in self._waiting.pop(key, []):
                ok(result)
            if not self._waiting:
                self.status.emit("")

        def failed(msg):
            for _, bad in self._waiting.pop(key, []):
                bad(msg)
            if not self._waiting:
                self.status.emit("")
        run_async(work, done, failed)
