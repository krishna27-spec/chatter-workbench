import json

from workbench.core import settings
from workbench.core.settings import DEFAULTS, add_recent, load, save


def test_missing_file_gives_defaults(tmp_path):
    assert load(tmp_path / "settings.json") == DEFAULTS


def test_defaults_are_a_fresh_copy(tmp_path):
    a = load(tmp_path / "none.json")
    a["recent"].append("x")
    assert DEFAULTS["recent"] == []                       # changing one settings dict never changes the defaults
    assert load(tmp_path / "none.json")["recent"] == []


def test_corrupt_json_gives_defaults_and_keeps_a_copy(tmp_path):
    p = tmp_path / "settings.json"
    p.write_text('{"version": 1, "theme": "ni')          # cut off mid-write
    assert load(p) == DEFAULTS
    assert (tmp_path / "settings.json.bad").read_text() == '{"version": 1, "theme": "ni'   # not silently lost


def test_other_version_gives_defaults(tmp_path):
    p = tmp_path / "settings.json"
    p.write_text(json.dumps({"version": 2, "theme": "night"}))
    assert load(p) == DEFAULTS


def test_missing_and_wrong_keys_are_repaired(tmp_path):
    p = tmp_path / "settings.json"
    p.write_text(json.dumps({"version": 1, "theme": "night", "recent": "not a list", "colour": "red"}))
    s = load(p)
    assert s["theme"] == "night"                          # good values are kept
    assert s["recent"] == [] and s["layouts"] == {}       # wrong type and missing key → default
    assert "colour" not in s                              # unknown keys dropped


def test_unknown_theme_falls_back(tmp_path):
    p = tmp_path / "settings.json"
    p.write_text(json.dumps({"version": 1, "theme": "purple"}))
    assert load(p)["theme"] == "system"


def test_round_trip(tmp_path):
    p = tmp_path / "sub" / "settings.json"                # the folder doesn't exist yet
    s = load(p)
    s["theme"] = "night"
    s["layouts"]["Analyse"] = "AAAA/w=="
    add_recent(s, "/data/run1.lvm")
    save(s, p)
    assert load(p) == s
    assert not list(p.parent.glob("*.tmp"))               # the temporary file was renamed, not left behind


def test_save_is_atomic(tmp_path, monkeypatch):
    """If writing fails half way, the old file must still be complete."""
    p = tmp_path / "settings.json"
    save({**DEFAULTS, "theme": "day"}, p)

    def boom(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(settings.json, "dump", boom)
    try:
        save({**DEFAULTS, "theme": "night"}, p)
    except OSError:
        pass
    monkeypatch.undo()
    assert load(p)["theme"] == "day"


def test_recent_is_unique_and_capped():
    s = {**DEFAULTS, "recent": []}
    for i in range(12):
        add_recent(s, f"/data/run{i}.lvm")
    add_recent(s, "/data/run5.lvm")                       # opened again → moves to the top, no duplicate
    assert s["recent"][0] == "/data/run5.lvm"
    assert len(s["recent"]) == 10
    assert len(set(s["recent"])) == 10
    assert "/data/run0.lvm" not in s["recent"]            # the oldest fell off
