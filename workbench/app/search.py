"""Ctrl+K command search: type part of a tool's name or description, Enter runs it."""
from __future__ import annotations

from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QDialog, QLabel, QLineEdit, QListWidget, QListWidgetItem, QVBoxLayout

EMPTY = "No tool matches. Try a shorter word."


class CommandPalette(QDialog):
    def __init__(self, actions: list[tuple[str, str, str, QAction]], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Search")
        self.setMinimumWidth(460)
        self._actions = actions
        self.edit = QLineEdit()
        self.edit.setPlaceholderText("Search tools and commands…")
        self.list = QListWidget()
        self.empty = QLabel(EMPTY)
        self.empty.setContentsMargins(8, 6, 8, 6)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.addWidget(self.edit)
        lay.addWidget(self.list)
        lay.addWidget(self.empty)
        self.edit.textChanged.connect(self._filter)
        self.edit.installEventFilter(self)
        self.list.itemActivated.connect(self._run)
        self._filter("")

    def shown_names(self) -> list[str]:
        return [self.list.item(i).text().split("    ")[0] for i in range(self.list.count())]

    def _filter(self, text: str) -> None:
        words = text.lower().split()
        self.list.clear()
        for name, desc, category, action in self._actions:
            hay = f"{name} {desc} {category}".lower()
            if all(w in hay for w in words):
                item = QListWidgetItem(f"{name}    {category}")
                item.setToolTip(desc)
                item.setData(Qt.UserRole, action)
                if not action.isEnabled():
                    item.setFlags(item.flags() & ~Qt.ItemIsEnabled)
                self.list.addItem(item)
        has = self.list.count() > 0
        self.list.setVisible(has)
        self.empty.setVisible(not has)
        if has:
            self.list.setCurrentRow(0)

    def eventFilter(self, obj, ev):
        if obj is self.edit and ev.type() == QEvent.KeyPress:
            if ev.key() in (Qt.Key_Down, Qt.Key_Up):
                row = self.list.currentRow() + (1 if ev.key() == Qt.Key_Down else -1)
                self.list.setCurrentRow(max(0, min(row, self.list.count() - 1)))
                return True
            if ev.key() in (Qt.Key_Return, Qt.Key_Enter):
                if self.list.currentItem() is not None:
                    self._run(self.list.currentItem())
                return True
        return super().eventFilter(obj, ev)

    def _run(self, item: QListWidgetItem) -> None:
        action = item.data(Qt.UserRole)
        self.accept()
        if action is not None and action.isEnabled():
            action.trigger()
