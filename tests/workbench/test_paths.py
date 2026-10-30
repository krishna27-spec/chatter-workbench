import sys
from pathlib import Path

from workbench import paths


def test_project_folder_paths():
    assert not paths.frozen()
    assert paths.model_package() == paths.REPO / "models" / "chatter_model_v2_chatter3"
    assert (paths.model_package() / "meta.json").exists()
    assert paths.recordings_dir() == paths.REPO / "data" / "recordings"


def test_installed_paths(monkeypatch, tmp_path):
    """Installed by PyInstaller: data from the unpacked bundle, recordings in the user's Documents."""
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path / "bundle"), raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    (tmp_path / "home" / "Documents").mkdir(parents=True)
    assert paths.model_package() == tmp_path / "bundle" / "models" / "chatter_model_v2_chatter3"
    assert paths.recordings_dir() == tmp_path / "home" / "Documents" / "Chatter Workbench" / "recordings"


def test_self_test_passes_here():
    from workbench.selftest import run
    ok, report = run()
    assert ok, report
