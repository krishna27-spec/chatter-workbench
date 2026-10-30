"""The right panel: the selected channel's editable name, unit, colour and visibility, plus read-only facts."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QCheckBox, QColorDialog, QFormLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout,
                               QWidget)

from workbench.core.recording import Recording


class Properties(QWidget):
    channel_changed = Signal(str, str, str, str, bool)      # old name, new name, unit, colour, visible

    def __init__(self, parent=None):
        super().__init__(parent)
        self._name: str | None = None
        self._colour = "#000000"
        self.title = QLabel("No channel selected")
        self.title.setContentsMargins(8, 8, 8, 4)
        self.name_edit = QLineEdit()
        self.unit_edit = QLineEdit()
        self.colour_btn = QPushButton()
        self.colour_btn.setFixedWidth(60)
        self.visible_box = QCheckBox("Visible")
        self.facts = {k: QLabel("–") for k in ("Samples", "fs", "Duration", "Min", "Max", "RMS")}

        form = QFormLayout()
        form.setContentsMargins(8, 4, 8, 8)
        form.addRow("Name", self.name_edit)
        form.addRow("Unit", self.unit_edit)
        form.addRow("Colour", self.colour_btn)
        form.addRow("", self.visible_box)
        for k, lbl in self.facts.items():
            form.addRow(k, lbl)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.title)
        lay.addLayout(form)
        lay.addStretch(1)
        self._form_widgets = [self.name_edit, self.unit_edit, self.colour_btn, self.visible_box]
        for w in self._form_widgets:
            w.setEnabled(False)

        self.name_edit.editingFinished.connect(self._emit)
        self.unit_edit.editingFinished.connect(self._emit)
        self.visible_box.toggled.connect(self._emit)
        self.colour_btn.clicked.connect(self._pick_colour)

    def show_channel(self, rec: Recording, name: str, colour: str, visible: bool) -> None:
        ch = rec.channel(name)
        self._name, self._colour = name, colour
        for w in self._form_widgets:
            w.blockSignals(True)
            w.setEnabled(True)
        self.title.setText(f"Channel {name}")
        self.name_edit.setText(ch.name)
        self.unit_edit.setText(ch.unit)
        self.visible_box.setChecked(visible)
        self._paint_colour()
        for w in self._form_widgets:
            w.blockSignals(False)
        x = ch.data
        n = len(x)
        self.facts["Samples"].setText(f"{n:,}")
        self.facts["fs"].setText(f"{rec.fs:g} Hz")
        self.facts["Duration"].setText(f"{rec.duration:.3f} s")
        if n:
            rms = float(np.sqrt(np.nanmean(np.square(x, dtype=np.float64))))
            self.facts["Min"].setText(f"{float(np.nanmin(x)):.6g} {ch.unit}")
            self.facts["Max"].setText(f"{float(np.nanmax(x)):.6g} {ch.unit}")
            self.facts["RMS"].setText(f"{rms:.6g} {ch.unit}")

    def set_colour(self, colour: str) -> None:
        """Change the colour as if picked in the dialog (used by the dialog and by tests)."""
        self._colour = colour
        self._paint_colour()
        self._emit()

    def _paint_colour(self):
        self.colour_btn.setStyleSheet(f"background: {self._colour}; border: 1px solid {self._colour};")

    def _pick_colour(self):
        c = QColorDialog.getColor(QColor(self._colour), self, "Channel colour")
        if c.isValid():
            self.set_colour(c.name().upper())

    def _emit(self):
        if self._name is None:
            return
        new = self.name_edit.text().strip() or self._name
        old, self._name = self._name, new
        self.title.setText(f"Channel {new}")
        self.channel_changed.emit(old, new, self.unit_edit.text().strip(), self._colour, self.visible_box.isChecked())

    def reject_name(self, old: str) -> None:
        """The window refused a rename (name taken): put the old name back."""
        self._name = old
        self.name_edit.blockSignals(True)
        self.name_edit.setText(old)
        self.name_edit.blockSignals(False)
        self.title.setText(f"Channel {old}")
