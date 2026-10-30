import numpy as np
import pytest
from PySide6.QtCore import Qt

from workbench.app.search import EMPTY, CommandPalette
from workbench.app.window import MainWindow
from workbench.core import registry
from workbench.core.settings import DEFAULTS


@pytest.fixture
def opened(qtbot, tmp_path):
    registry.load_formats()
    w = MainWindow({**DEFAULTS, "recent": [], "layouts": {}})
    qtbot.addWidget(w)
    w.show()
    f = tmp_path / "run.lvm"
    f.write_text("".join(f"{i / 1000:.6f}\t{i}\t{-i}\t{i % 3}\n" for i in range(100)))
    w.open_path(f)
    qtbot.waitUntil(lambda: w.rec is not None, timeout=5000)
    return w


def select(win, name):
    item = next(i for i in win.tree.channel_items() if i.text(0) == name)
    win.tree.setCurrentItem(item)


def test_selecting_a_channel_shows_its_facts(opened):
    select(opened, "Y")
    assert opened.props.name_edit.text() == "Y"
    assert opened.props.facts["Samples"].text() == "100"
    assert opened.props.facts["Min"].text().startswith("-99")


def test_rename_channel_updates_plot_label(opened):
    select(opened, "Y")
    opened.props.name_edit.setText("normal")
    opened.props.unit_edit.setText("m/s²")
    opened.props.name_edit.editingFinished.emit()
    assert opened.rec.channels[1].name == "normal" and opened.rec.channels[1].unit == "m/s²"
    assert opened.plot.lane_titles()[1].startswith("normal")
    assert [i.text(0) for i in opened.tree.channel_items()] == ["X", "normal", "Z"]


def test_rename_to_a_taken_name_is_refused(opened):
    select(opened, "Y")
    opened.props.name_edit.setText("X")
    opened.props.name_edit.editingFinished.emit()
    assert [c.name for c in opened.rec.channels] == ["X", "Y", "Z"]
    assert opened.props.name_edit.text() == "Y"


def test_hide_channel_removes_lane(opened):
    select(opened, "Z")
    opened.props.visible_box.setChecked(False)
    assert [l.names for l in opened.plot.lanes] == [["X"], ["Y"]]
    opened.props.visible_box.setChecked(True)
    assert [l.names for l in opened.plot.lanes] == [["X"], ["Y"], ["Z"]]


def test_colour_change_reaches_the_plot(opened):
    select(opened, "X")
    opened.props.set_colour("#FF00FF")
    assert opened.plot.colour_of("X") == "#FF00FF"


def test_palette_filters_and_triggers(opened, qtbot):
    pal = CommandPalette(opened.actions(), opened)
    qtbot.addWidget(pal)
    qtbot.keyClicks(pal.edit, "theme")
    assert pal.shown_names() == ["Theme: follow system", "Theme: day", "Theme: night"]
    qtbot.keyClick(pal.edit, Qt.Key_Down)
    qtbot.keyClick(pal.edit, Qt.Key_Down)
    qtbot.keyClick(pal.edit, Qt.Key_Return)            # the third one: night
    assert opened.settings["theme"] == "night"


def test_palette_empty_message(opened, qtbot):
    pal = CommandPalette(opened.actions(), opened)
    qtbot.addWidget(pal)
    pal.show()
    qtbot.keyClicks(pal.edit, "xyzzy")
    assert pal.list.count() == 0
    assert pal.empty.isVisible() and pal.empty.text() == EMPTY
