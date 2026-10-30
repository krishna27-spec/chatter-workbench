"""Hardware: every chassis and module the NI-DAQmx driver knows right now, real and simulated, with their pins.
Refreshed every 2 s on a worker thread, so plugging or unplugging shows without a restart."""
from __future__ import annotations

from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import QLabel, QPushButton, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget

from workbench.app.loader import run_async
from workbench.core.live.hardware import Module, list_modules


class HardwarePanel(QWidget):
    modules_changed = Signal(list)              # list[Module], whenever what is attached changes

    def __init__(self, lister=list_modules, parent=None):
        super().__init__(parent)
        self.lister = lister
        self.modules: list[Module] = []
        self.problem: str | None = None
        self._busy = False
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Module", "Product", "Pins"])
        self.note = QLabel()
        self.note.setWordWrap(True)
        self.refresh_btn = QPushButton("Refresh")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.addWidget(self.tree, 1)
        lay.addWidget(self.note)
        lay.addWidget(self.refresh_btn)
        self.refresh_btn.clicked.connect(self.refresh)
        self.timer = QTimer(self, interval=2000)
        self.timer.timeout.connect(self.refresh)

    def start_polling(self) -> None:
        self.refresh()
        self.timer.start()

    def stop_polling(self) -> None:
        self.timer.stop()

    def refresh(self) -> None:
        if self._busy:
            return
        self._busy = True
        run_async(self.lister, self._got, lambda msg: self._got(([], msg)))

    def _got(self, result) -> None:
        self._busy = False
        mods, problem = result
        same = [(m.name, m.product, m.simulated) for m in mods] == \
               [(m.name, m.product, m.simulated) for m in self.modules] and problem == self.problem
        self.modules, self.problem = mods, problem
        if same and self.tree.topLevelItemCount():
            return
        self.tree.clear()
        chassis: dict[str, QTreeWidgetItem] = {}
        for m in mods:
            top = chassis.get(m.chassis)
            if top is None:
                top = chassis[m.chassis] = QTreeWidgetItem([m.chassis, "", ""])
                self.tree.addTopLevelItem(top)
            pins = ", ".join(f"{len(getattr(m, k))} {k.upper()}" for k in ("ai", "ao") if getattr(m, k))
            item = QTreeWidgetItem(top, [f"slot {m.slot}  {m.name}", m.product + ("  (simulated)" if m.simulated else ""),
                                         pins])
            for p in m.ai + m.ao:
                QTreeWidgetItem(item, [p.split("/")[-1], "", ""])
        self.tree.expandToDepth(0)
        for i in range(3):
            self.tree.resizeColumnToContents(i)
        sims = sum(m.simulated for m in mods)
        if problem:
            self.note.setText(problem)
        elif not mods:
            self.note.setText("No NI modules found. Plug in the cDAQ (USB) and wait 2 s, or replay a file.")
        else:
            self.note.setText(f"{len(mods)} modules" + (f", {sims} simulated: nothing reaches a pin on those."
                                                       if sims else "."))
        self.modules_changed.emit(mods)
