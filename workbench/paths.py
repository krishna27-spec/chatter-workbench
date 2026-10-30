"""Where things are, both when run from the project folder and when installed (.deb / Windows .exe).

PyInstaller unpacks an installed app into one folder and sets sys.frozen; the data files bundled with it (the model
package, the fonts) sit under sys._MEIPASS. From the project folder they are where the repository keeps them.
Recordings go to the project's data/recordings/ when run from the project, and to Documents/Chatter Workbench/recordings
when installed (an installed program's own folder is read-only: /opt on Linux, Program Files on Windows).
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MODEL_NAME = "chatter_model_v2_chatter3"


def frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def bundle_dir() -> Path:
    """The folder the installed app's data files were unpacked to (the repository when not installed)."""
    return Path(getattr(sys, "_MEIPASS", REPO)) if frozen() else REPO


def model_package() -> Path:
    return bundle_dir() / "models" / MODEL_NAME if frozen() else REPO / "models" / MODEL_NAME


def recordings_dir() -> Path:
    if not frozen():
        return REPO / "data" / "recordings"
    docs = Path.home() / "Documents"
    return (docs if docs.is_dir() else Path.home()) / "Chatter Workbench" / "recordings"
