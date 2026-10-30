"""The main window: menus, toolbar with the Analyse / Live workspaces, the docks and the status bar.

It owns the open Recording and wires the panels together: the session tree selects a channel → Properties shows it
→ edits go back to the Recording, the tree and the time plot. Files load in the background (loader.py).

Two workspaces, each with its own docks: Analyse (Session, Properties, Spectrum; the time plot in the centre) and
Live (Hardware and Inputs on the left, Output on the right; the live dashboard in the centre). Output can be armed
and started only from Live, but Stop output (status-bar button, Esc, Ctrl+.) works everywhere.
"""
from __future__ import annotations

import time
from pathlib import Path

from PySide6.QtCore import QByteArray, Qt, QTimer
from PySide6.QtGui import QAction, QActionGroup, QKeySequence, QShortcut
from PySide6.QtWidgets import (QApplication, QDockWidget, QFileDialog, QLabel, QMainWindow, QMessageBox,
                               QProgressBar, QPushButton, QStackedWidget, QTabBar, QToolBar)

from workbench import __version__
from workbench.app.analysis import AnalysisSession
from workbench.app.live.hardware_panel import HardwarePanel
from workbench.app.live.input_panel import InputPanel
from workbench.app.live.live_view import LiveView
from workbench.app.live.output_panel import OutputPanel
from workbench.app.live.session import LiveSession
from workbench.app.loader import load_async
from workbench.app.panels.properties import Properties
from workbench.app.panels.session_tree import SessionTree
from workbench.app.panels.spectrum import SpectrumPanel
from workbench.app.panels.time_plot import TimePlot
from workbench.app.search import CommandPalette
from workbench.app.theme import TOKENS, apply_theme, resolve
from workbench.core import registry
from workbench.core.recording import Recording
from workbench.core.settings import add_recent
from workbench.paths import recordings_dir

_gc_guard = None


def collect_garbage_on_gui_thread() -> None:
    """Python's cyclic garbage collector runs on whichever thread happens to allocate. If that is a worker (file
    loading, Clean, the Live engine), Qt objects in the garbage (old plot items, lanes) are destroyed off the GUI
    thread, which Qt forbids: it showed as "Timers cannot be started from another thread" and intermittent hangs.
    So automatic collection is switched off and a GUI-thread timer collects instead, every 0.5 s, with Python's
    usual thresholds (pyqtgraph's GarbageCollector, written for exactly this)."""
    global _gc_guard
    if _gc_guard is None:
        from pyqtgraph.util.garbage_collector import GarbageCollector
        _gc_guard = GarbageCollector(interval=0.5)

WORKSPACES = ("Analyse", "Live")


class MainWindow(QMainWindow):
    def __init__(self, settings: dict, problems: list[str] | tuple = (), lister=None):
        super().__init__()
        collect_garbage_on_gui_thread()
        self.settings = settings
        self.problems = list(problems)
        self.rec: Recording | None = None
        self.analysis: AnalysisSession | None = None
        self.setWindowTitle("Chatter Workbench")
        self.resize(1280, 800)
        self.setAcceptDrops(True)
        app = QApplication.instance()
        self.tokens = TOKENS[resolve(settings.get("theme", "system"), app)]

        # centre: the time plot (Analyse) or the live dashboard (Live)
        self.plot = TimePlot(self.tokens)
        self.live_view = LiveView(self.tokens)
        self.centre = QStackedWidget()
        self.centre.addWidget(self.plot)
        self.centre.addWidget(self.live_view)
        self.setCentralWidget(self.centre)

        # docks
        self.tree = SessionTree()
        self.props = Properties()
        self.tree_dock = self._dock("Session", "session_dock", self.tree, Qt.LeftDockWidgetArea)
        self.props_dock = self._dock("Properties", "properties_dock", self.props, Qt.RightDockWidgetArea)
        self.spectrum = SpectrumPanel(self.tokens)
        self.spectrum_dock = self._dock("Spectrum", "spectrum_dock", self.spectrum, Qt.BottomDockWidgetArea)
        self.resizeDocks([self.spectrum_dock], [300], Qt.Vertical)      # the time plot keeps most of the height
        self.tree.channel_selected.connect(self._show_channel)
        self.props.channel_changed.connect(self._channel_changed)
        self.plot.regions_changed.connect(self.spectrum.regions_changed)
        self.spectrum.source_changed.connect(lambda *_: self._update_spectrogram())

        # Live docks
        self.live = LiveSession()
        self.record_root = recordings_dir()
        self.hardware = HardwarePanel(lister) if lister else HardwarePanel()
        self.inputs = InputPanel()
        self.outputs = OutputPanel()
        self.hardware_dock = self._dock("Hardware", "hardware_dock", self.hardware, Qt.LeftDockWidgetArea)
        self.inputs_dock = self._dock("Inputs", "inputs_dock", self.inputs, Qt.LeftDockWidgetArea)
        self.output_dock = self._dock("Output", "output_dock", self.outputs, Qt.RightDockWidgetArea)
        self.tabifyDockWidget(self.hardware_dock, self.inputs_dock)
        self.dock_sets = {"Analyse": [self.tree_dock, self.props_dock, self.spectrum_dock],
                          "Live": [self.hardware_dock, self.inputs_dock, self.output_dock]}
        self.hardware.modules_changed.connect(self.inputs.set_modules)
        self.hardware.modules_changed.connect(self.outputs.set_modules)
        self.inputs.start_requested.connect(self._live_start)
        self.inputs.stop_requested.connect(self._live_stop)
        self.outputs.arm_requested.connect(self._live_arm)
        self.outputs.start_requested.connect(self._live_start_output)
        self.outputs.stop_requested.connect(lambda: self.stop_output("Stop output pressed"))
        self.outputs.manual_changed.connect(self._live_manual)
        self.live_view.recal_btn.clicked.connect(lambda: self.live.engine and self.live.engine.recalibrate())
        self.live_view.record_btn.toggled.connect(self._live_record)
        self.live.changed.connect(self._live_changed)
        self.live_timer = QTimer(self, interval=33)            # ~30 frames a second
        self.live_timer.timeout.connect(self._live_tick)

        self._make_actions()
        self._make_menus()
        self._make_toolbar()
        self._make_status()
        for keys in ("Esc", "Ctrl+."):                          # Stop output from anywhere
            sc = QShortcut(QKeySequence(keys), self)
            sc.setContext(Qt.ApplicationShortcut)
            sc.activated.connect(lambda: self.stop_output("Stop output key"))

        self._workspace = WORKSPACES[0]
        self._defaults = {}
        for ws in ("Live", "Analyse"):                          # each workspace starts with only its own docks
            self._show_docks(ws)
            self._defaults[ws] = self.saveState()
        self._default_state = self._defaults["Analyse"]
        self._restore_layout(self._workspace)
        app.styleHints().colorSchemeChanged.connect(self._os_scheme_changed)

    # ------------------------------------------------------------------ construction
    def _dock(self, title, name, widget, area) -> QDockWidget:
        d = QDockWidget(title, self)
        d.setObjectName(name)                          # saveState() identifies docks by object name
        d.setWidget(widget)
        d.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.addDockWidget(area, d)
        return d

    def _action(self, name, slot=None, shortcut=None, checkable=False, desc="", category="General") -> QAction:
        a = QAction(name, self)
        if shortcut:
            a.setShortcut(QKeySequence(shortcut))
        a.setCheckable(checkable)
        if slot is not None:                           # toggles get checked=True/False, plain commands nothing
            a.triggered.connect((lambda checked=False: slot(checked)) if checkable else (lambda *_: slot()))
        a.setStatusTip(desc)
        self._catalogue.append((name.replace("…", ""), desc, category, a))
        return a

    def _make_actions(self):
        self._catalogue: list[tuple[str, str, str, QAction]] = []
        A = self._action
        self.act_open = A("Open file…", self.open_dialog, "Ctrl+O", desc="Open an .lvm or .csv recording", category="File")
        self.act_open_dir = A("Open session folder…", self.open_folder_dialog, "Ctrl+Shift+O",
                              desc="Open a scope.py session (raw.lvm, control.csv, events.txt)", category="File")
        self.act_quit = A("Quit", self.close, "Ctrl+Q", desc="Close the workbench", category="File")
        self.theme_group = QActionGroup(self)
        self.theme_actions = {}
        for key, label in (("system", "Follow system"), ("day", "Day"), ("night", "Night")):
            a = A(f"Theme: {label.lower()}", lambda _=False, k=key: self.set_theme(k), checkable=True,
                  desc=f"Switch to the {label.lower()} look", category="View")
            a.setText(label)
            self.theme_group.addAction(a)
            self.theme_actions[key] = a
        self.theme_actions.get(self.settings.get("theme", "system"), self.theme_actions["system"]).setChecked(True)
        self.act_reset = A("Reset layout", self.reset_layout, desc="Put every panel back where it started", category="View")
        self.act_search = A("Search…", self.open_search, "Ctrl+K", desc="Find any tool or command by name", category="Tools")
        self.act_problems = A("Problems…", self.show_problems, desc="Formats that failed to load", category="Tools")
        self.act_about = A("About", self.show_about, desc="Version and offline note", category="Help")
        self.act_cursors = A("Cursors", self.plot.set_cursors, "C", checkable=True,
                             desc="Two time cursors with values at each", category="Plot")
        self.act_overlay = A("Overlay channels", self.plot.set_overlaid, "O", checkable=True,
                             desc="Draw all channels in one lane", category="Plot")
        self.act_fit = A("Fit", self.plot.fit, "F", desc="Show the whole recording", category="Plot")
        self.act_auto_y = A("Auto Y", self.plot.set_auto_y, "Y", checkable=True,
                            desc="Vertical range follows the data in view (off: whole-file range)", category="Plot")
        self.act_auto_y.setChecked(True)
        self.act_region = A("Add region", self._add_region, "R",
                            desc="Mark a stretch of time (chatter / stable / idle) for the Spectrum", category="Analyse")
        self.act_sg = A("Spectrogram", lambda on: self._update_spectrogram(), "S", checkable=True,
                        desc="Spectrogram lane of the Spectrum's channel (STFT, peak hold)", category="Analyse")
        self.act_spectrum = A("Spectrum panel", lambda: (self.spectrum_dock.show(), self.spectrum_dock.raise_()),
                              desc="Show the Spectrum panel: power spectrum of each region", category="Analyse")
        self.ws_actions = [A(f"{w} workspace", lambda _=False, i=i: self.ws_tabs.setCurrentIndex(i),
                             desc=f"Switch to the {w} workspace", category="Workspace") for i, w in enumerate(WORKSPACES)]

    def _make_menus(self):
        mb = self.menuBar()
        f = mb.addMenu("&File")
        f.addAction(self.act_open)
        f.addAction(self.act_open_dir)
        self.recent_menu = f.addMenu("Open recent")
        f.addSeparator()
        f.addAction(self.act_quit)
        v = mb.addMenu("&View")
        tm = v.addMenu("Theme")
        for a in self.theme_actions.values():
            tm.addAction(a)
        v.addAction(self.act_reset)
        t = mb.addMenu("&Tools")
        t.addAction(self.act_search)
        t.addAction(self.act_problems)
        h = mb.addMenu("&Help")
        h.addAction(self.act_about)
        self._rebuild_recent()

    def _make_toolbar(self):
        tb = QToolBar("Main")
        tb.setObjectName("main_toolbar")
        tb.setMovable(False)
        self.addToolBar(tb)
        self.ws_tabs = QTabBar()
        for w in WORKSPACES:
            self.ws_tabs.addTab(w)
        self.ws_tabs.setExpanding(False)
        self.ws_tabs.currentChanged.connect(self._switch_workspace)
        tb.addWidget(self.ws_tabs)
        tb.addSeparator()
        self.analyse_actions = [self.act_cursors, self.act_overlay, self.act_auto_y, self.act_fit,
                                self.act_region, self.act_sg]
        for a in [self.act_open] + self.analyse_actions:
            tb.addAction(a)
        self.act_open.setIconText("Open")

    def _make_status(self):
        sb = self.statusBar()
        self.busy = QProgressBar()
        self.busy.setRange(0, 0)                       # 0..0 = "working" animation, no percentage
        self.busy.setMaximumWidth(120)
        self.busy.setMaximumHeight(12)
        self.busy.hide()
        self.seg_device = QLabel("No device")
        self.seg_rate = QLabel("–")
        self.seg_output = QLabel("Output off")
        self.stop_out_btn = QPushButton("■ Stop output")
        self.stop_out_btn.setToolTip("Write the safe values and close the output (Esc, Ctrl+.)")
        self.stop_out_btn.clicked.connect(lambda: self.stop_output("Stop output pressed"))
        self.stop_out_btn.hide()
        for w in (self.busy, self.seg_device, self.seg_rate, self.seg_output, self.stop_out_btn):
            sb.addPermanentWidget(w)                   # permanent: normal items hide while a message shows

    # ------------------------------------------------------------------ files
    def open_dialog(self):
        exts = " ".join(f"*{e}" for e in registry.supported_read_exts())
        path, _ = QFileDialog.getOpenFileName(self, "Open recording", self._last_dir(), f"Recordings ({exts});;All files (*)")
        if path:
            self.open_path(path)

    def open_folder_dialog(self):
        path = QFileDialog.getExistingDirectory(self, "Open session folder", self._last_dir())
        if path:
            self.open_path(path)

    def _last_dir(self) -> str:
        recent = self.settings.get("recent", [])
        return str(Path(recent[0]).parent) if recent else str(Path.home())

    def open_path(self, path) -> None:
        path = Path(path)
        if not path.exists():
            self._forget_recent(path)
            QMessageBox.warning(self, "Could not open", f"{path.name}\n\n{path} does not exist (moved or deleted?).")
            return
        self.busy.show()
        self.statusBar().showMessage(f"Opening {path.name}…")
        load_async(path, lambda rec: self._loaded(path, rec), lambda msg: self._failed(path, msg))

    def _loaded(self, path: Path, rec: Recording) -> None:
        self.busy.hide()
        self.rec = rec
        self.plot.set_recording(rec)
        self.tree.set_recording(rec, self.plot.colours())
        self.inputs.set_replay(rec)
        self.analysis = AnalysisSession(rec)
        self.analysis.status.connect(self._analysis_status)
        self.spectrum.set_session(self.analysis, rec, self.plot)
        self.act_sg.setChecked(False)
        self.seg_rate.setText(f"{rec.fs:g} Hz")
        self.setWindowTitle(f"{path.name} — Chatter Workbench")
        add_recent(self.settings, str(path.resolve()))
        self._rebuild_recent()
        msg = (f"Opened {path.name}: {len(rec.channels)} channels × {rec.n_samples:,} samples, "
               f"{rec.duration:.1f} s")
        if rec.meta.get("notice"):
            msg += f"   Note: {rec.meta['notice']}"
        self.statusBar().showMessage(msg)

    def _failed(self, path: Path, message: str) -> None:
        self.busy.hide()
        self.statusBar().clearMessage()
        text = message if message.startswith(path.name) else f"{path.name}: {message}"
        QMessageBox.warning(self, "Could not open", text)

    def _rebuild_recent(self):
        self.recent_menu.clear()
        recent = self.settings.get("recent", [])
        for f in recent:
            a = self.recent_menu.addAction(Path(f).name)
            a.setToolTip(f)
            a.triggered.connect(lambda _=False, f=f: self.open_path(f))
        self.recent_menu.setEnabled(bool(recent))

    def _forget_recent(self, path: Path):
        self.settings["recent"] = [f for f in self.settings.get("recent", []) if Path(f) != path and f != str(path)]
        self._rebuild_recent()

    def dragEnterEvent(self, ev):
        if ev.mimeData().hasUrls():
            ev.acceptProposedAction()

    def dropEvent(self, ev):
        local = [u.toLocalFile() for u in ev.mimeData().urls() if u.isLocalFile()]
        if local:
            self.open_path(local[0])

    # ------------------------------------------------------------------ channels
    def _show_channel(self, name: str) -> None:
        if self.rec is not None:
            self.props.show_channel(self.rec, name, self.plot.colour_of(name), name in self.plot.visible_channels())

    def _channel_changed(self, old: str, new: str, unit: str, colour: str, visible: bool) -> None:
        if self.rec is None:
            return
        if new != old and new in [c.name for c in self.rec.channels]:
            self.props.reject_name(old)
            self.statusBar().showMessage(f"A channel named {new} already exists.")
            return
        ch = self.rec.channel(old)
        ch.name, ch.unit = new, unit
        if self.analysis is not None and new != old:
            self.analysis.rename_channel(old, new)
        self.plot.update_channel(old, new, colour, visible)
        self.tree.update_channel(old, new, unit, colour)
        self.spectrum.set_channels([c.name for c in self.rec.channels])

    # ------------------------------------------------------------------ analysis
    def _add_region(self) -> None:
        if self.rec is None:
            self.statusBar().showMessage("Open a recording first, then mark a region.")
            return
        self.plot.add_region()
        self.spectrum_dock.show()

    def _update_spectrogram(self) -> None:
        if self.rec is None or self.analysis is None or not self.act_sg.isChecked():
            self.plot.set_spectrogram(None)
            return
        source, channel = self.spectrum.current()
        rec = self.rec

        def show(sg):
            if self.rec is rec and self.act_sg.isChecked() and self.spectrum.current() == (source, channel):
                self.plot.set_spectrogram(sg, rec.t0, f"{channel} {source} [Hz]")
        self.analysis.spectrogram(source, channel, show,
                                  lambda msg: QMessageBox.warning(self, "Spectrogram", msg))

    def _analysis_status(self, message: str) -> None:
        self.busy.setVisible(bool(message))
        if message:
            self.statusBar().showMessage(message)
        else:
            self.statusBar().clearMessage()

    # ------------------------------------------------------------------ view
    def set_theme(self, name: str) -> None:
        look = apply_theme(QApplication.instance(), name)
        self.settings["theme"] = name
        self.theme_actions[name].setChecked(True)
        self.tokens = TOKENS[look]
        self.plot.apply_tokens(self.tokens)
        self.spectrum.apply_tokens(self.tokens)
        self.live_view.apply_tokens(self.tokens)
        self._live_changed()
        if self.rec is not None:
            self.tree.set_recording(self.rec, self.plot.colours())

    def _os_scheme_changed(self, *_):
        if self.settings.get("theme") == "system":
            self.set_theme("system")

    def _show_docks(self, ws: str) -> None:
        for name, docks in self.dock_sets.items():
            for d in docks:
                d.setVisible(name == ws)

    def _switch_workspace(self, index: int) -> None:
        self._save_layout(self._workspace)
        self._workspace = WORKSPACES[index]
        self.centre.setCurrentIndex(index)
        self._restore_layout(self._workspace)
        for a in self.analyse_actions:                          # Analyse's plot tools only where they work
            a.setVisible(self._workspace == "Analyse")
        if self._workspace == "Live":
            self.hardware.start_polling()
            self.live_timer.start()
        else:
            self.hardware.stop_polling()
            self.live_timer.stop()

    def workspace(self) -> str:
        return self._workspace

    def _save_layout(self, ws: str) -> None:
        self.settings.setdefault("layouts", {})[ws] = bytes(self.saveState().toBase64()).decode("ascii")

    def _restore_layout(self, ws: str) -> None:
        saved = self.settings.get("layouts", {}).get(ws)
        self.restoreState(QByteArray.fromBase64(saved.encode("ascii")) if saved else self._defaults[ws])
        self._show_docks(ws)                                    # never another workspace's docks

    def reset_layout(self):
        self.restoreState(self._defaults[self._workspace])
        self._show_docks(self._workspace)
        self.settings.get("layouts", {}).pop(self._workspace, None)

    # ------------------------------------------------------------------ live
    def _live_start(self) -> None:
        try:
            inp, out = self.inputs.choice(), self.outputs.choice()
            self.live.start(inp, out)
        except Exception as e:
            QMessageBox.warning(self, "Cannot start acquisition", str(e))
            return
        eng = self.live.engine
        self.live_view.attach(eng.source.names)
        self.seg_device.setText(eng.source.label)
        self.seg_rate.setText(f"{eng.fs:g} Hz")

    def _live_stop(self) -> None:
        self.live.stop()
        self.live_view.record_btn.setChecked(False)

    def _live_arm(self) -> None:
        g = self.live.guard
        if g is None:
            QMessageBox.warning(self, "Arm output", "No output pins were chosen when acquisition started. Stop "
                                "acquisition, choose the output module and pins, and start again.")
            return
        out = self.live.output
        where = ("SIMULATED module: nothing reaches a pin." if out.simulated else
                 "REAL module: these volts go to the spindle drive / machine inputs.")
        rows = "\n".join(f"  {p.pin}   {p.role:<8}  safe {g._safe(p):.3f} V   allowed {p.v_min:g}…{min(p.v_max, 10):g} V"
                         for p in g.pins)
        text = (f"{where}\n\nArming opens the output and writes the safe values:\n{rows}\n\n"
                f"Speed: {out.mode} from {out.spindle.start_rpm:.0f} rpm. Nothing follows the commands until you "
                "press Start output. Stop output (Esc, Ctrl+.) writes the safe values and closes it.")
        if QMessageBox.question(self, "Arm output?", text) != QMessageBox.Yes:
            return
        try:
            self.live.arm()
        except Exception as e:
            QMessageBox.warning(self, "Arm output", str(e))

    def _live_start_output(self) -> None:
        try:
            self.live.start_output()
        except Exception as e:
            QMessageBox.warning(self, "Start output", str(e))

    def _live_manual(self, pin: str, volts: float) -> None:
        if self.live.guard is not None:
            self.live.guard.set_manual(pin, volts)

    def _live_record(self, on: bool) -> None:
        eng = self.live.engine
        if eng is None or not eng.running:
            if on:
                self.live_view.record_btn.setChecked(False)
                self.statusBar().showMessage("Start acquisition first, then Record.")
            return
        if on:
            hw = {"input": self.live.input.pins or self.live.input.kind, "output": self.live.output.module,
                  "output_pins": [vars(p) for p in self.live.output.pins], "spindle": vars(self.live.output.spindle),
                  "mode": self.live.output.mode, "version": __version__}
            folder = eng.start_recording(self.record_root / f"wb_{time.strftime('%Y%m%d_%H%M%S')}", hw)
            self.statusBar().showMessage(f"Recording to {folder}")
        else:
            eng.stop_recording()

    def stop_output(self, why: str = "Stop output") -> None:
        self.live.stop_output(why)

    def _live_changed(self) -> None:
        g = self.live.guard
        st = g.state if g else None
        running = self.live.running
        self.inputs.set_running(running, self.live.engine.fs if running else None)
        self.outputs.set_state(st, running, list(g.current) if g else None)
        on = st in ("armed", "writing")
        self.stop_out_btn.setVisible(on)
        if on:
            volts = "  ".join(f"{p.role} {v:.3f} V" for p, v in zip(g.pins, g.current))
            self.seg_output.setText(f"OUTPUT {st.upper()}  {volts}")
            self.seg_output.setStyleSheet(f"background: {self.tokens['out']}; color: {self.tokens['out_ink']}; "
                                          "font-weight: 600; padding: 0 6px;")
        else:
            self.seg_output.setText("Output off")
            self.seg_output.setStyleSheet("")
        if not running:
            self.seg_device.setText("No device")

    def _live_tick(self) -> None:
        eng = self.live.engine
        if eng is None:
            return
        g = eng.guard
        out = self.live.output
        rpm = g.rpm if g else (out.spindle.start_rpm if out else None)
        self.live_view.refresh(eng, rpm, out.spindle.teeth if out else 2)
        self._live_changed()
        if eng.error and not eng.running:
            self.statusBar().showMessage(f"Live stopped: {eng.error}")

    def open_search(self):
        CommandPalette(self.actions(), self).exec()

    def actions(self) -> list[tuple[str, str, str, QAction]]:
        """(name, description, category, action) for every command, for the search."""
        return list(self._catalogue)

    def show_problems(self):
        text = "\n".join(self.problems) if self.problems else "Every file format loaded without problems."
        QMessageBox.information(self, "Problems", text)

    def show_about(self):
        QMessageBox.about(self, "About Chatter Workbench",
                          f"Chatter Workbench {__version__}\n\nOffline: nothing leaves this laptop.")

    def closeEvent(self, ev):
        self.live.close()                                       # output safe, task closed, recording finished
        self._save_layout(self._workspace)
        super().closeEvent(ev)
