"""Inputs: where the data comes from. An NI module's ticked pins (name, unit, sensitivity, range, rate), or a replay
of the file open in Analyse. The first three ticked pins are X, Y, Z for the models."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QHBoxLayout, QHeaderView, QLabel,
                               QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from workbench.app.live.session import InputChoice
from workbench.core.live.hardware import Module, suggest_input

REPLAY = "__replay__"
RANGES = (10.0, 5.0, 1.0, 0.2)


class InputPanel(QWidget):
    start_requested = Signal()
    stop_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.modules: list[Module] = []
        self.replay_rec = None
        self.source = QComboBox()
        self.pins = QTableWidget(0, 4)
        self.pins.setHorizontalHeaderLabels(["Pin", "Name", "Unit", "Unit per V"])
        self.pins.verticalHeader().hide()
        self.pins.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.rate = QDoubleSpinBox()
        self.rate.setRange(100, 200_000)
        self.rate.setDecimals(0)
        self.rate.setValue(20_000)
        self.rate.setSuffix(" Hz")
        self.range = QComboBox()
        for r in RANGES:
            self.range.addItem(f"±{r:g} V", r)
        self.models = QCheckBox("Top-3 models (needs 3 pins: X, Y, Z)")
        self.models.setChecked(True)
        self.actual = QLabel()
        self.start_btn = QPushButton("Start acquisition")
        self.stop_btn = QPushButton("Stop")
        form = QFormLayout()
        form.addRow("Source", self.source)
        form.addRow("Rate", self.rate)
        form.addRow("Range", self.range)
        form.addRow("", self.models)
        btns = QHBoxLayout()
        btns.addWidget(self.start_btn)
        btns.addWidget(self.stop_btn)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.addLayout(form)
        lay.addWidget(self.pins, 1)
        lay.addWidget(self.actual)
        lay.addLayout(btns)
        self.source.currentIndexChanged.connect(self._fill_pins)
        self.start_btn.clicked.connect(self.start_requested)
        self.stop_btn.clicked.connect(self.stop_requested)
        self.set_running(False)

    def set_modules(self, mods: list[Module]) -> None:
        keep = self.source.currentData()
        self.modules = [m for m in mods if m.ai]
        self.source.blockSignals(True)
        self.source.clear()
        for m in self.modules:
            self.source.addItem(m.label, m.name)
        self.source.addItem("Replay the file open in Analyse", REPLAY)
        names = [self.source.itemData(i) for i in range(self.source.count())]
        if keep in names:
            self.source.setCurrentIndex(names.index(keep))
        else:
            sug = suggest_input(self.modules)
            self.source.setCurrentIndex(names.index(sug.name if sug else REPLAY))   # never a simulated one by itself
        self.source.blockSignals(False)
        if keep not in names or keep is None:
            self._fill_pins()

    def set_replay(self, rec) -> None:
        self.replay_rec = rec
        if self.source.currentData() == REPLAY:
            self._fill_pins()

    def _fill_pins(self, *_):
        key = self.source.currentData()
        self.pins.setRowCount(0)
        if key == REPLAY:
            rec = self.replay_rec
            self.actual.setText(f"Replays {rec.source.name if rec and rec.source else 'nothing yet: open a file in Analyse'}"
                                + (f" at {rec.fs:g} Hz, looping" if rec else ""))
            for i, c in enumerate(rec.channels if rec else []):
                self._row(c.name, c.name, c.unit, 1.0, checked=True, editable=False)
            return
        m = next((m for m in self.modules if m.name == key), None)
        self.actual.setText("SIMULATED module: the samples are made up by the driver." if m and m.simulated else "")
        for i, p in enumerate(m.ai if m else []):
            self._row(p, "XYZ"[i] if i < 3 else p.split("/")[-1], "V", 1.0, checked=i < 3, editable=True)

    def _row(self, pin, name, unit, per_volt, checked, editable):
        r = self.pins.rowCount()
        self.pins.insertRow(r)
        it = QTableWidgetItem(pin.split("/")[-1])
        it.setData(Qt.UserRole, pin)
        it.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
        it.setCheckState(Qt.Checked if checked else Qt.Unchecked)
        self.pins.setItem(r, 0, it)
        for col, val in ((1, name), (2, unit), (3, f"{per_volt:g}")):
            cell = QTableWidgetItem(val)
            if not editable:
                cell.setFlags(cell.flags() & ~Qt.ItemIsEditable)
            self.pins.setItem(r, col, cell)

    def check_pins(self, pins: list[str]) -> None:
        for r in range(self.pins.rowCount()):
            it = self.pins.item(r, 0)
            it.setCheckState(Qt.Checked if it.data(Qt.UserRole) in pins else Qt.Unchecked)

    def choice(self) -> InputChoice:
        rows = [r for r in range(self.pins.rowCount()) if self.pins.item(r, 0).checkState() == Qt.Checked]
        get = lambda r, c: self.pins.item(r, c).text().strip()
        try:
            per_volt = [float(get(r, 3)) for r in rows]
        except ValueError:
            raise ValueError("'Unit per V' must be a number (1 = keep volts)") from None
        if self.source.currentData() == REPLAY:
            return InputChoice("replay", replay=self.replay_rec, models=self.models.isChecked())
        return InputChoice("ni", [self.pins.item(r, 0).data(Qt.UserRole) for r in rows], [get(r, 1) for r in rows],
                           [get(r, 2) for r in rows], per_volt, self.rate.value(), self.range.currentData(),
                           models=self.models.isChecked())

    def is_simulated(self) -> bool:
        key = self.source.currentData()
        return key == REPLAY or any(m.name == key and m.simulated for m in self.modules)

    def set_running(self, on: bool, actual_fs: float | None = None) -> None:
        for w in (self.source, self.pins, self.rate, self.range, self.models, self.start_btn):
            w.setEnabled(not on)
        self.stop_btn.setEnabled(on)
        if on and actual_fs:
            self.actual.setText(f"Running at {actual_fs:g} Hz (the rate the source really gives).")
