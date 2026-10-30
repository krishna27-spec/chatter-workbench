"""Output: where volts are written. Pick an analog-output module, tick its pins and give each a role (spindle speed,
feed override that follows the spindle, or a manual voltage), its limits and safe value; choose how the spindle
speed is set (constant, or the controller: Liao & Young or the lab law). Arm → confirm → Start output; Stop output
is always available. The settings are fixed while acquisition runs (stop it to change them)."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel,
                               QPushButton, QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from workbench.app.live.session import OutputChoice
from workbench.core.live.hardware import Module, suggest_output
from workbench.core.live.output import ROLES, V_LIMIT, PinConfig, SpindleConfig

NONE = "__none__"
COLS = ["Pin", "Role", "Min V", "Max V", "Slew V/s", "Safe V", "Manual V"]


def _spin(lo, hi, val, dec=2, step=0.1, suffix=""):
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(dec)
    s.setSingleStep(step)
    s.setValue(val)
    if suffix:
        s.setSuffix(suffix)
    return s


class OutputPanel(QWidget):
    arm_requested = Signal()
    start_requested = Signal()
    stop_requested = Signal()
    manual_changed = Signal(str, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.modules: list[Module] = []
        self.module = QComboBox()
        self.sim_note = QLabel()
        self.sim_note.setWordWrap(True)
        self.pins = QTableWidget(0, len(COLS))
        self.pins.setHorizontalHeaderLabels(COLS)
        self.pins.verticalHeader().hide()
        self.pins.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.pins.itemChanged.connect(self._cell_changed)

        self.mode = QComboBox()
        self.mode.addItem("Constant speed", "constant")
        self.mode.addItem("Controller: Liao & Young (change once, settle, hold)", "liao")
        self.mode.addItem("Controller: lab law (every window, back when stable)", "lab")
        self.start_rpm = _spin(0, 140_000, 20_000, 0, 500, " rpm")
        self.rpm_per_volt = _spin(1, 1e6, 14_000, 0, 100, " rpm/V")
        self.offset = _spin(-10, 10, 0.0, 3, 0.01, " V")
        self.teeth = QSpinBox()
        self.teeth.setRange(1, 12)
        self.teeth.setValue(2)
        self.fz = _spin(0.01, 100, 3.0, 2, 0.1, " µm")
        self.program_feed = _spin(0.01, 1000, 5.0, 2, 0.5, " mm/s")
        self.rpm_min = _spin(0, 140_000, 10_000, 0, 1000, " rpm")
        self.rpm_max = _spin(0, 140_000, 100_000, 0, 1000, " rpm")
        self.max_step = _spin(0.01, 1.0, 0.35, 2, 0.05)
        self.direction = QComboBox()
        for d in ("nearest", "down", "up"):
            self.direction.addItem(d, d)
        sp = QGroupBox("Spindle and feed")
        f = QFormLayout(sp)
        f.addRow("Speed", self.mode)
        f.addRow("Start speed", self.start_rpm)
        f.addRow("Scale", self.rpm_per_volt)
        f.addRow("Offset", self.offset)
        f.addRow("Teeth Z", self.teeth)
        f.addRow("Feed per tooth", self.fz)
        f.addRow("Programmed feed (= 5 V)", self.program_feed)
        f.addRow("Allowed speeds", self._pair(self.rpm_min, self.rpm_max))
        f.addRow("Largest step (fraction)", self.max_step)
        f.addRow("Direction", self.direction)
        self.settings = [self.module, self.pins, sp]

        self.state = QLabel("Output off")
        self.arm_btn = QPushButton("Arm…")
        self.start_btn = QPushButton("Start output")
        self.stop_btn = QPushButton("Stop output")
        btns = QHBoxLayout()
        for b in (self.arm_btn, self.start_btn, self.stop_btn):
            btns.addWidget(b)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.addWidget(QLabel("Module"))
        lay.addWidget(self.module)
        lay.addWidget(self.sim_note)
        lay.addWidget(self.pins, 1)
        lay.addWidget(sp)
        lay.addWidget(self.state)
        lay.addLayout(btns)
        self.module.currentIndexChanged.connect(self._fill_pins)
        self.arm_btn.clicked.connect(self.arm_requested)
        self.start_btn.clicked.connect(self.start_requested)
        self.stop_btn.clicked.connect(self.stop_requested)
        self.set_state(None, False)

    @staticmethod
    def _pair(a, b):
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(a)
        h.addWidget(QLabel("to"))
        h.addWidget(b)
        return w

    # ---- modules and pins
    def set_modules(self, mods: list[Module]) -> None:
        keep = self.module.currentData()
        self.modules = [m for m in mods if m.ao]
        self.module.blockSignals(True)
        self.module.clear()
        self.module.addItem("No output (watch only)", NONE)
        for m in self.modules:
            self.module.addItem(m.label, m.name)
        names = [self.module.itemData(i) for i in range(self.module.count())]
        if keep in names:
            self.module.setCurrentIndex(names.index(keep))
        else:
            sug = suggest_output(self.modules)
            self.module.setCurrentIndex(names.index(sug.name) if sug else 0)
        self.module.blockSignals(False)
        if keep not in names or keep is None:
            self._fill_pins()

    def selected_module(self) -> Module | None:
        return next((m for m in self.modules if m.name == self.module.currentData()), None)

    def _fill_pins(self, *_):
        m = self.selected_module()
        self.sim_note.setText("SIMULATED module: nothing reaches a pin. Good for trying the controller."
                              if m and m.simulated else "")
        self.pins.blockSignals(True)
        self.pins.setRowCount(0)
        for i, p in enumerate(m.ao if m else []):
            r = self.pins.rowCount()
            self.pins.insertRow(r)
            it = QTableWidgetItem(p.split("/")[-1])
            it.setData(Qt.UserRole, p)
            it.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            it.setCheckState(Qt.Checked if i == 0 else Qt.Unchecked)       # ao0 = spindle, as on the rig
            self.pins.setItem(r, 0, it)
            role = QComboBox()
            for name in ROLES:
                role.addItem(name, name)
            role.setCurrentIndex({0: 0, 1: 1}.get(i, 2))                    # ao0 spindle, ao1 feed, others manual
            self.pins.setCellWidget(r, 1, role)
            for col, val in ((2, "0"), (3, "10"), (4, "0"), (5, ""), (6, "0")):
                self.pins.setItem(r, col, QTableWidgetItem(val))
        self.pins.blockSignals(False)

    def set_pin(self, pin: str, used: bool, role: str | None = None) -> None:
        for r in range(self.pins.rowCount()):
            it = self.pins.item(r, 0)
            if it.data(Qt.UserRole) == pin:
                it.setCheckState(Qt.Checked if used else Qt.Unchecked)
                if role:
                    self.pins.cellWidget(r, 1).setCurrentText(role)

    def _cell_changed(self, item: QTableWidgetItem) -> None:
        if item.column() == 6:
            try:
                self.manual_changed.emit(self.pins.item(item.row(), 0).data(Qt.UserRole), float(item.text()))
            except ValueError:
                pass

    def choice(self) -> OutputChoice:
        m = self.selected_module()
        out = OutputChoice(module=m.name if m else None, simulated=bool(m and m.simulated))
        num = lambda r, c, d=None: float(self.pins.item(r, c).text()) if self.pins.item(r, c).text().strip() else d
        for r in range(self.pins.rowCount()):
            if self.pins.item(r, 0).checkState() != Qt.Checked:
                continue
            try:
                slew = num(r, 4, 0.0)
                out.pins.append(PinConfig(self.pins.item(r, 0).data(Qt.UserRole), self.pins.cellWidget(r, 1).currentData(),
                                          num(r, 2, 0.0), min(num(r, 3, V_LIMIT), V_LIMIT), slew or None,
                                          num(r, 5), num(r, 6, 0.0)))
            except ValueError:
                raise ValueError(f"row {r + 1}: volts must be numbers") from None
        out.spindle = SpindleConfig(self.rpm_per_volt.value(), self.offset.value(), self.start_rpm.value(),
                                    self.teeth.value(), self.fz.value(), self.program_feed.value())
        out.mode = self.mode.currentData()
        out.rpm_min, out.rpm_max = self.rpm_min.value(), self.rpm_max.value()
        out.max_step, out.direction = self.max_step.value(), self.direction.currentData()
        return out

    # ---- state
    def set_state(self, state: str | None, running: bool, volts: list[float] | None = None) -> None:
        for w in self.settings:
            w.setEnabled(not running)
        self.arm_btn.setEnabled(running and state == "disarmed")
        self.start_btn.setEnabled(state == "armed")
        self.stop_btn.setEnabled(state in ("armed", "writing"))
        v = "  ".join(f"{x:.3f} V" for x in (volts or []))
        self.state.setText({None: "Output off (no output pins chosen)" if running else "Output off",
                            "disarmed": "Disarmed: nothing is written",
                            "armed": f"ARMED: safe values held  {v}",
                            "writing": f"WRITING  {v}"}[state])
