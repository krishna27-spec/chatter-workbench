"""The app's memory between runs: theme, recent files and each workspace's window layout.

One small JSON file, ~/.config/chatter-workbench/settings.json (the XDG convention on Linux; part 5 picks the
Windows folder). Two rules make it safe:
- Loading never fails. A missing, corrupt or newer/older file gives the defaults, so a bad settings file can never
  stop the app from starting (spec review focus 5). A corrupt file is kept as settings.json.bad, not overwritten.
- Saving is atomic: write settings.json.tmp, then os.replace() it over the old file. os.replace is a single rename,
  which the OS does completely or not at all, so a crash or full disk mid-save leaves the old file intact.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

PATH = Path.home() / ".config" / "chatter-workbench" / "settings.json"
DEFAULTS = {"version": 1, "theme": "system", "recent": [], "layouts": {}}
THEMES = ("system", "day", "night")
MAX_RECENT = 10


def load(path: Path | None = None) -> dict:
    """The settings from `path` (default PATH), repaired key by key; never raises."""
    path = Path(path or PATH)
    s = copy.deepcopy(DEFAULTS)                     # a fresh copy, so callers can't change DEFAULTS
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return s
    except (OSError, ValueError):                   # unreadable or not JSON (ValueError covers JSONDecodeError)
        try:
            os.replace(path, path.with_name(path.name + ".bad"))
        except OSError:
            pass
        return s
    if not isinstance(stored, dict) or stored.get("version") != DEFAULTS["version"]:
        return s                                    # another format: don't guess what its keys mean
    for key, default in DEFAULTS.items():           # keep only known keys of the right type
        if isinstance(stored.get(key), type(default)):
            s[key] = stored[key]
    if s["theme"] not in THEMES:
        s["theme"] = DEFAULTS["theme"]
    s["recent"] = [str(f) for f in s["recent"]][:MAX_RECENT]
    return s


def save(settings: dict, path: Path | None = None) -> None:
    """Write atomically: a crash part-way leaves the previous file as it was."""
    path = Path(path or PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2)
            f.flush()
            os.fsync(f.fileno())                    # on disk, not just in the OS cache, before the rename
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)                 # only still there if something failed


def add_recent(settings: dict, file: str | Path) -> None:
    """Put `file` at the top of the recent list: no duplicates, at most MAX_RECENT, oldest dropped."""
    file = str(file)
    settings["recent"] = [file] + [f for f in settings["recent"] if f != file][:MAX_RECENT - 1]
