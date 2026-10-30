"""The left panel: what the open file contains. File → Signals (channels) / Decisions / Output / Actions."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPixmap
from PySide6.QtWidgets import QHeaderView, QTreeWidget, QTreeWidgetItem

from workbench.core.recording import Recording

CHANNEL_ROLE = Qt.UserRole + 1                # the channel name stored on a channel item


def _swatch(colour: str) -> QIcon:
    pm = QPixmap(8, 8)
    pm.fill(QColor(colour))
    return QIcon(pm)


def _rate(fs: float) -> str:
    return f"{fs / 1000:g} kHz" if fs >= 1000 else f"{fs:g} Hz"


class SessionTree(QTreeWidget):
    channel_selected = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setHeaderHidden(True)
        self.setColumnCount(2)
        self.setIndentation(14)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.header().setStretchLastSection(False)
        self.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self.header().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.currentItemChanged.connect(self._on_current)

    def set_recording(self, rec: Recording | None, colours: dict[str, str] | None = None) -> None:
        self.clear()
        if rec is None:
            return
        colours = colours or {}
        root = QTreeWidgetItem([rec.source.name if rec.source else "recording", ""])
        self.addTopLevelItem(root)
        signals = QTreeWidgetItem(root, ["Signals", _rate(rec.fs)])
        for ch in rec.channels:
            item = QTreeWidgetItem(signals, [ch.name, ch.unit])
            item.setData(0, CHANNEL_ROLE, ch.name)
            if ch.name in colours:
                item.setIcon(0, _swatch(colours[ch.name]))
        if "decisions" in rec.tables:
            QTreeWidgetItem(root, ["Decisions", f"{len(rec.tables['decisions'].get('t', [])):,} rows"])
        if "output" in rec.tables:
            QTreeWidgetItem(root, ["Output", "ao0 spindle"])
        if rec.events:
            QTreeWidgetItem(root, ["Actions", str(len(rec.events))])
        self.expandAll()

    def channel_items(self) -> list[QTreeWidgetItem]:
        found = []
        for i in range(self.topLevelItemCount()):
            stack = [self.topLevelItem(i)]
            while stack:
                item = stack.pop()
                if item.data(0, CHANNEL_ROLE):
                    found.append(item)
                stack.extend(item.child(j) for j in range(item.childCount()))
        return sorted(found, key=lambda it: it.parent().indexOfChild(it))

    def update_channel(self, old: str, new: str, unit: str, colour: str) -> None:
        for item in self.channel_items():
            if item.data(0, CHANNEL_ROLE) == old:
                item.setText(0, new)
                item.setText(1, unit)
                item.setData(0, CHANNEL_ROLE, new)
                item.setIcon(0, _swatch(colour))

    def _on_current(self, item, _previous):
        if item is not None and item.data(0, CHANNEL_ROLE):
            self.channel_selected.emit(item.data(0, CHANNEL_ROLE))
