"""The Spectrum panel: one power spectrum per region on one channel, spindle harmonics, and a peak table that says
which peaks sit between the harmonics (chatter candidates), with Liao & Young's n + ν. The maths is in
core/analysis/spectrum.py; this file only gathers the inputs and draws."""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QIcon, QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox, QHBoxLayout, QHeaderView,
                               QLabel, QPushButton, QSpinBox, QSplitter, QTableWidget, QTableWidgetItem, QToolButton,
                               QVBoxLayout, QWidget)

from workbench.app.loader import run_async
from workbench.core.analysis.spectrum import (SCALES, classify, harmonics, peaks, psd, rpm_source, scaled,
                                              teeth_source)

SEGMENTS = (1024, 2048, 4096, 8192, 16384)
BIG = 2_000_000                               # regions bigger than this (in total) are computed on a worker thread
MAX_MARKS = 400                               # more lines than this can't be told apart (and 150 000 froze the app)
NO_REGION = "Draw a region on the time plot (R, or Add region) to see its spectrum."


def _swatch(colour: str) -> QIcon:
    pm = QPixmap(10, 10)
    pm.fill(QColor(colour))
    return QIcon(pm)


class SpectrumPanel(QWidget):
    computed = Signal()                       # curves and peaks are up to date (used by tests)
    source_changed = Signal(str, str)         # (source, channel): the spectrogram lane follows this choice

    def __init__(self, tokens: dict[str, str], parent=None):
        super().__init__(parent)
        self.tokens = tokens
        self.analysis = None
        self.rec = None
        self.time_plot = None
        self._gen = 0
        self._rpm_manual = False
        self.results: list = []               # [(region, Spectrum)] of the last computation
        self.peak_rows: list = []             # [(Peak, Classified | None)] shown in the peak table
        self._lines: list[pg.InfiniteLine] = []

        # ---- controls
        self.channel = QComboBox()
        self.source = QComboBox()
        self.source.addItem("Raw", "raw")
        self.source.addItem("Clean", "clean")
        self.method = QComboBox()
        self.method.addItem("Welch", "welch")
        self.method.addItem("FFT (one window)", "fft")
        self.segment = QComboBox()
        self.scale = QComboBox()
        self.scale.addItem("dB (PSD)", "db")
        self.scale.addItem("linear PSD", "psd")
        self.scale.addItem("linear amplitude", "amplitude")
        self.log_f = QCheckBox("log f")
        self.rpm = QDoubleSpinBox()
        self.rpm.setRange(0, 300_000)
        self.rpm.setDecimals(0)
        self.rpm.setSingleStep(100)
        self.rpm.setSuffix(" rpm")
        self.rpm.setSpecialValueText("rpm ?")
        self.rpm.setKeyboardTracking(False)          # apply on Enter / leaving the box, not on every keystroke
        self.rpm_src = QLabel()
        self.rpm_auto = QToolButton()
        self.rpm_auto.setText("auto")
        self.rpm_auto.setToolTip("Take the rpm from the recording again")
        self.teeth = QSpinBox()
        self.teeth.setRange(1, 12)
        self.teeth.setPrefix("Z = ")
        self.teeth.setKeyboardTracking(False)
        self.show_spindle = QCheckBox("spindle harmonics")
        self.show_spindle.setChecked(True)
        self.show_tooth = QCheckBox("tooth passing")
        self.show_tooth.setChecked(True)
        self.on_rate(20_000.0)

        row1 = QHBoxLayout()
        for w in (QLabel("Channel"), self.channel, QLabel("Signal"), self.source, QLabel("Method"), self.method,
                  QLabel("Segment"), self.segment, QLabel("Scale"), self.scale, self.log_f):
            row1.addWidget(w)
        row1.addStretch(1)
        row2 = QHBoxLayout()
        for w in (self.rpm, self.rpm_auto, self.rpm_src, self.teeth, self.show_spindle, self.show_tooth):
            row2.addWidget(w)
        row2.addStretch(1)

        # ---- spectrum plot
        self.plot = pg.PlotWidget()
        self.plot.setMenuEnabled(False)
        self.plot.hideButtons()
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.setLabel("bottom", "frequency", units="Hz")
        self.plot.setLabel("left", "PSD [dB]")
        self.plot.getAxis("left").enableAutoSIPrefix(False)
        self.legend = self.plot.addLegend(offset=(-10, 10))
        self.message = QLabel(NO_REGION)
        self.message.setWordWrap(True)

        # ---- regions and peaks
        self.regions = QTableWidget(0, 4)
        self.regions.setHorizontalHeaderLabels(["Region", "Start [s]", "Length [s]", "Show"])
        self.regions.verticalHeader().hide()
        self.regions.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.regions.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.add_btn = QPushButton("Add region (R)")
        self.remove_btn = QPushButton("Remove")
        self.peak_region = QComboBox()
        self.peak_table = QTableWidget(0, 8)
        self.peak_table.setHorizontalHeaderLabels(["#", "f [Hz]", "level [dB]", "prom. [dB]", "harmonic",
                                                   "offset [Hz]", "n + ν", "kind"])
        self.peak_table.verticalHeader().hide()
        self.peak_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.peak_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.harm_info = QLabel()

        side = QWidget()
        sl = QVBoxLayout(side)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.addWidget(self.regions, 2)
        btns = QHBoxLayout()
        btns.addWidget(self.add_btn)
        btns.addWidget(self.remove_btn)
        btns.addStretch(1)
        sl.addLayout(btns)
        pr = QHBoxLayout()
        pr.addWidget(QLabel("Peaks of"))
        pr.addWidget(self.peak_region, 1)
        sl.addLayout(pr)
        sl.addWidget(self.peak_table, 3)
        sl.addWidget(self.harm_info)
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.addWidget(self.plot, 1)
        ll.addWidget(self.message)
        split = QSplitter(Qt.Horizontal)
        split.addWidget(left)
        split.addWidget(side)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 4, 6, 4)
        lay.addLayout(row1)
        lay.addLayout(row2)
        lay.addWidget(split, 1)

        self._timer = QTimer(self, singleShot=True, interval=120)      # dragging a region → one recompute
        self._timer.timeout.connect(self._compute)
        for w in (self.method, self.segment, self.peak_region, self.scale):
            w.currentIndexChanged.connect(self.refresh)
        for w in (self.channel, self.source):
            w.currentIndexChanged.connect(self._source_changed)
        self.log_f.toggled.connect(lambda on: (self.plot.setLogMode(x=on), self.refresh()))
        for w in (self.show_spindle, self.show_tooth):
            w.toggled.connect(self.refresh)
        self.teeth.valueChanged.connect(self.refresh)
        self.rpm.valueChanged.connect(self._rpm_edited)
        self.rpm_auto.clicked.connect(self._rpm_reset)
        self.add_btn.clicked.connect(self._add_region)
        self.remove_btn.clicked.connect(self._remove_region)
        self.regions.itemChanged.connect(self._region_edited)
        self.setEnabled(False)

    # ------------------------------------------------------------------ wiring
    def set_session(self, analysis, rec, time_plot) -> None:
        self.analysis, self.rec, self.time_plot = analysis, rec, time_plot
        self.setEnabled(rec is not None)
        if rec is None:
            return
        self.on_rate(rec.fs)
        self.set_channels([c.name for c in rec.channels])
        teeth, _ = teeth_source(rec)
        self.teeth.blockSignals(True)
        self.teeth.setValue(teeth)
        self.teeth.blockSignals(False)
        self._rpm_manual = False
        self._fill_regions()
        self.refresh()

    def on_rate(self, fs: float) -> None:
        current = self.segment.currentData() or 4096
        self.segment.blockSignals(True)
        self.segment.clear()
        for n in SEGMENTS:
            self.segment.addItem(f"{n}  ({fs / n:.3g} Hz)", n)
        self.segment.setCurrentIndex(SEGMENTS.index(current) if current in SEGMENTS else 2)
        self.segment.blockSignals(False)

    def set_channels(self, names: list[str]) -> None:
        keep = self.channel.currentText()
        self.channel.blockSignals(True)
        self.channel.clear()
        self.channel.addItems(names)
        default = "Y" if "Y" in names else names[0]                 # Y: the axis the top-3 models use
        self.channel.setCurrentText(keep if keep in names else default)
        self.channel.blockSignals(False)

    def current(self) -> tuple[str, str]:
        return self.source.currentData(), self.channel.currentText()

    def apply_tokens(self, tokens: dict[str, str]) -> None:
        self.tokens = tokens
        self.plot.setBackground(tokens["panel"])
        self._fill_regions()
        self.refresh()

    def refresh(self, *_) -> None:
        self._timer.start()

    def regions_changed(self) -> None:
        self._fill_regions()
        self.refresh()

    # ------------------------------------------------------------------ regions table
    def _visible_regions(self):
        return [r for r in self.time_plot.regions() if r.visible] if self.time_plot else []

    def _fill_regions(self) -> None:
        if self.time_plot is None:
            return
        regs = self.time_plot.regions()
        self.regions.blockSignals(True)
        self.regions.setRowCount(len(regs))
        for i, r in enumerate(regs):
            name = QTableWidgetItem(_swatch(self.time_plot.region_colour(r)), r.name)
            name.setData(Qt.UserRole, r.id)
            start = QTableWidgetItem(f"{r.t0:.3f}")
            length = QTableWidgetItem(f"{r.t1 - r.t0:.3f}")
            for it in (start, length):
                it.setFlags(it.flags() & ~Qt.ItemIsEditable)
            show = QTableWidgetItem()
            show.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            show.setCheckState(Qt.Checked if r.visible else Qt.Unchecked)
            for col, it in enumerate((name, start, length, show)):
                self.regions.setItem(i, col, it)
        self.regions.blockSignals(False)
        keep = self.peak_region.currentData()
        self.peak_region.blockSignals(True)
        self.peak_region.clear()
        for r in regs:
            if r.visible:
                self.peak_region.addItem(_swatch(self.time_plot.region_colour(r)), r.name, r.id)
        ids = [self.peak_region.itemData(i) for i in range(self.peak_region.count())]
        if keep in ids:
            self.peak_region.setCurrentIndex(ids.index(keep))
        else:
            names = [self.peak_region.itemText(i) for i in range(self.peak_region.count())]
            self.peak_region.setCurrentIndex(names.index("chatter") if "chatter" in names else 0)
        self.peak_region.blockSignals(False)

    def _region_by_id(self, rid):
        return next((r for r in self.time_plot.regions() if r.id == rid), None)

    def _region_edited(self, item: QTableWidgetItem) -> None:
        r = self._region_by_id(self.regions.item(item.row(), 0).data(Qt.UserRole))
        if r is None:
            return
        if item.column() == 0 and item.text().strip() and item.text().strip() != r.name:
            self.time_plot.update_region(r, name=item.text().strip())
        elif item.column() == 3:
            self.time_plot.update_region(r, visible=item.checkState() == Qt.Checked)

    def _add_region(self) -> None:
        if self.time_plot is not None and self.rec is not None:
            self.time_plot.add_region()

    def _remove_region(self) -> None:
        row = self.regions.currentRow()
        if row >= 0:
            r = self._region_by_id(self.regions.item(row, 0).data(Qt.UserRole))
            if r is not None:
                self.time_plot.remove_region(r)

    # ------------------------------------------------------------------ rpm
    def _peak_region(self):
        return self._region_by_id(self.peak_region.currentData()) if self.time_plot else None

    def _rpm_edited(self, _value) -> None:
        if not self.rpm.signalsBlocked():
            self._rpm_manual = True
            self.rpm_src.setText("typed")
            self.refresh()

    def _rpm_reset(self) -> None:
        self._rpm_manual = False
        self.refresh()

    def _update_rpm(self) -> None:
        if self._rpm_manual or self.rec is None:
            return
        r = self._peak_region()
        rpm, src = rpm_source(self.rec, (r.t0 + r.t1) / 2 if r else None)
        self.rpm.blockSignals(True)
        self.rpm.setValue(rpm or 0)
        self.rpm.blockSignals(False)
        self.rpm_src.setText(f"from {src}" if rpm else "unknown: type the rpm to mark harmonics")

    # ------------------------------------------------------------------ computing
    def _source_changed(self, *_) -> None:
        self.source_changed.emit(*self.current())
        self.refresh()

    def _compute(self) -> None:
        self._gen += 1
        gen = self._gen
        self._update_rpm()
        regs = self._visible_regions()
        if self.analysis is None or not regs:
            self._show(gen, [], NO_REGION)
            return
        source, _ = self.current()
        if self.analysis.ready(source) is None:
            self.message.setText("Cleaning the whole file first…")
        self.analysis.recording(source, lambda rec: self._compute_on(gen, rec, regs),
                                lambda msg: self._show(gen, [], msg))

    def _compute_on(self, gen: int, rec, regs) -> None:
        if gen != self._gen:
            return
        name = self.channel.currentText()
        ch = rec.channel(name)
        method, nperseg = self.method.currentData(), self.segment.currentData()
        spans = [(r, rec.index_of(r.t0), rec.index_of(r.t1)) for r in regs]

        def work():
            out = []
            for r, a, b in spans:
                try:
                    out.append((r, psd(ch.data[a:b], rec.fs, method, nperseg)))
                except ValueError as e:
                    out.append((r, str(e)))
            return out
        if sum(b - a for _, a, b in spans) > BIG:
            self.message.setText("Computing…")
            run_async(work, lambda res: self._show(gen, res, ""), lambda msg: self._show(gen, [], msg))
        else:
            self._show(gen, work(), "")

    def _show(self, gen: int, results, message: str) -> None:
        if gen != self._gen:
            return                                # a newer computation has started: drop this one
        self.plot.clear()
        self.legend.clear()
        self._lines = []
        self.results = [(r, s) for r, s in results if not isinstance(s, str)]
        problems = [f"{r.name}: {s}" for r, s in results if isinstance(s, str)]
        unit = self.rec.channel(self.channel.currentText()).unit if self.rec and self.channel.currentText() else ""
        scale = self.scale.currentData()
        self.plot.setLabel("left", SCALES[scale].format(u=unit or "unit"))
        for r, s in self.results:
            curve = self.plot.plot(s.f[1:], scaled(s, scale)[1:], pen=pg.mkPen(self.time_plot.region_colour(r), width=1),
                                   name=f"{r.name}  ({s.n_avg} avg, {s.resolution:.3g} Hz)")
            curve.setDownsampling(auto=False)
        self._draw_harmonics()
        self._fill_peaks()
        self.message.setText("   ".join(filter(None, [message] + problems)))
        self.computed.emit()

    def _draw_harmonics(self) -> None:
        rpm, teeth = self.rpm.value(), self.teeth.value()
        if not rpm or not self.results:
            self.harm_info.setText("" if rpm else "No rpm: harmonic marks off.")
            return
        fmax = self.results[0][1].f[-1]
        if fmax / (rpm / 60.0) > MAX_MARKS:          # count first: never build or draw an unreadable comb
            self.harm_info.setText(f"{rpm:g} rpm is too low to mark harmonics (f_s = {rpm / 60:.3g} Hz: more than "
                                   f"{MAX_MARKS} lines up to {fmax / 1e3:g} kHz).")
            return
        h = harmonics(rpm, teeth, fmax)
        if self.show_spindle.isChecked():
            for f in h.spindle_lines:
                self._vline(f, pg.mkPen(self.tokens["ink2"], width=1, style=Qt.DotLine))
        if self.show_tooth.isChecked():
            for f in h.tooth_lines:
                self._vline(f, pg.mkPen(self.tokens["acc"], width=1, style=Qt.DashLine))
        self.harm_info.setText(f"spindle f_s = {h.spindle_hz:.2f} Hz (dotted) · tooth passing f_tp = Z·f_s = "
                               f"{h.tooth_hz:.2f} Hz (dashed) · 'between' = more than 2 bins from every k·f_s")

    def _vline(self, f: float, pen) -> None:
        line = pg.InfiniteLine(pos=np.log10(f) if self.log_f.isChecked() else f, angle=90, movable=False, pen=pen)
        line.setZValue(-5)
        self.plot.addItem(line, ignoreBounds=True)
        self._lines.append(line)

    def _fill_peaks(self) -> None:
        self.peak_rows = []
        r = self._peak_region()
        spec = next((s for reg, s in self.results if reg is r), None)
        if spec is not None:
            rpm, teeth = self.rpm.value(), self.teeth.value()
            for p in peaks(spec.f, spec.P):
                self.peak_rows.append((p, classify(p.f, rpm, teeth, spec.resolution) if rpm else None))
        self.peak_table.setRowCount(len(self.peak_rows))
        scale = self.scale.currentData()
        unit = self.rec.channel(self.channel.currentText()).unit if self.rec else ""
        self.peak_table.setHorizontalHeaderItem(2, QTableWidgetItem(
            {"db": "level [dB]", "psd": f"PSD [{unit}²/Hz]", "amplitude": f"amplitude [{unit}]"}[scale]))
        bold = QFont()
        bold.setBold(True)
        for i, (p, c) in enumerate(self.peak_rows):
            P = 10 ** (p.level_db / 10)
            level = {"db": f"{p.level_db:.1f}", "psd": f"{P:.3g}",
                     "amplitude": f"{np.sqrt(2 * P * spec.enbw):.4g}"}[scale]
            cells = [str(i + 1), f"{p.f:.1f}", level, f"{p.prominence_db:.1f}"]
            cells += [f"{c.k} × f_s", f"{c.offset_hz:+.1f}", f"{c.n} + {c.nu:.2f}", c.kind] if c else ["", "", "", ""]
            for col, text in enumerate(cells):
                it = QTableWidgetItem(text)
                it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter if col < 7 else Qt.AlignLeft | Qt.AlignVCenter)
                if c is not None and c.kind == "between":
                    it.setForeground(QBrush(QColor(self.tokens["s3"])))
                    it.setFont(bold)
                self.peak_table.setItem(i, col, it)
