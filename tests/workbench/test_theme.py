import ast
import re
from pathlib import Path

import pyqtgraph as pg
from PySide6.QtGui import QPalette

from workbench.app.theme import TOKENS, apply_theme, build_qss, resolve

ROOT = Path(__file__).resolve().parents[2]
COLOUR = re.compile(r"#[0-9A-F]{6}|rgba\(\d+, \d+, \d+, \.\d+\)")


def test_both_themes_have_every_token():
    assert set(TOKENS) == {"day", "night"}
    assert set(TOKENS["day"]) == set(TOKENS["night"])
    assert {"bg", "panel", "ink", "acc", "x", "y", "z", "s0", "s3", "out"} <= set(TOKENS["day"])
    for theme in TOKENS.values():
        for name, value in theme.items():
            assert COLOUR.fullmatch(value), f"{name} = {value!r}"


def test_channel_colours_match_the_spec():
    assert (TOKENS["day"]["x"], TOKENS["day"]["y"], TOKENS["day"]["z"]) == ("#E03E2D", "#2B9A47", "#0A6FDB")
    assert (TOKENS["night"]["x"], TOKENS["night"]["y"], TOKENS["night"]["z"]) == ("#FF6B5E", "#4CC46A", "#4DA3FF")


def test_qss_uses_tokens():
    qss = build_qss(TOKENS["night"])
    assert "#1C1F23" in qss                              # night panel colour
    assert "rgba(111, 168, 220, 36)" in qss              # acc_wash .14 → alpha 36 of 255 (Qt's form)
    assert "{" in qss and "border-radius" in qss


def test_apply_theme_offscreen(qapp):
    assert apply_theme(qapp, "day") == "day"
    assert qapp.palette().color(QPalette.Window).name().upper() == TOKENS["day"]["bg"]
    assert pg.getConfigOption("antialias") is False      # sharp 1-px lines (spec 8b)
    assert pg.getConfigOption("background") == TOKENS["day"]["panel"]
    assert apply_theme(qapp, "night") == "night"
    assert qapp.palette().color(QPalette.Window).name().upper() == TOKENS["night"]["bg"]
    assert apply_theme(qapp, "system") in ("day", "night")


def test_bundled_font_is_used(qapp):
    apply_theme(qapp, "day")
    assert "Plex" in qapp.font().family()
    assert qapp.font().pixelSize() == 12


def test_unknown_name_is_day(qapp):
    assert resolve("purple", qapp) == "day"
    assert resolve("night", qapp) == "night"


def test_core_never_imports_qt():
    """The layering rule (app → core, never back) as a test: core must run without Qt."""
    bad = []
    for path in sorted((ROOT / "workbench" / "core").rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                    [node.module or ""] if isinstance(node, ast.ImportFrom) and node.level == 0 else []
            for n in names:
                if n.split(".")[0] in ("PySide6", "pyqtgraph", "shiboken6") or n.startswith("workbench.app"):
                    bad.append(f"{path.relative_to(ROOT)}:{node.lineno} imports {n}")
    assert not bad, "\n".join(bad)
