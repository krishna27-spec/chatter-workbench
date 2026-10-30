import sys
import textwrap

import numpy as np
import pytest

from workbench.core import registry
from workbench.core.recording import Channel, Recording


@pytest.fixture(autouse=True)
def clean_registry():
    saved = registry._snapshot_for_tests()
    registry._clear_for_tests()
    yield
    registry._restore_for_tests(saved)


def tiny(path):
    return Recording([Channel("X", "V", np.zeros(4, np.float32))], 4.0)


def test_read_dispatches_by_suffix(tmp_path):
    calls = []

    @registry.reader(".abc")
    def read_abc(path):
        calls.append(path)
        return tiny(path)

    for name in ("a.abc", "b.ABC"):
        p = tmp_path / name
        p.write_text("x")
        rec = registry.read(p)
        assert rec.source == p              # read() records where it came from
    assert len(calls) == 2


def test_unknown_suffix_lists_supported(tmp_path):
    registry.reader(".lvm")(tiny)
    registry.reader(".csv")(tiny)
    p = tmp_path / "data.xyz"
    p.write_text("x")
    with pytest.raises(registry.UnsupportedFormat, match=r"\.csv, \.lvm") as e:
        registry.read(p)
    assert isinstance(e.value, ValueError)
    assert "data.xyz" in str(e.value)


def test_missing_file_is_clear(tmp_path):
    registry.reader(".lvm")(tiny)
    with pytest.raises(FileNotFoundError, match="nothing.lvm"):
        registry.read(tmp_path / "nothing.lvm")


def test_folder_reader_used_for_directories(tmp_path):
    @registry.reader(folder=True)
    def read_folder(path):
        return tiny(path)

    rec = registry.read(tmp_path)
    assert rec.source == tmp_path


def test_folder_without_reader(tmp_path):
    with pytest.raises(registry.UnsupportedFormat, match="folder"):
        registry.read(tmp_path)


def test_write_dispatches_by_suffix(tmp_path):
    written = []
    registry.writer(".abc")(lambda rec, path: written.append(path))
    registry.write(tiny(None), tmp_path / "out.abc")
    assert written == [tmp_path / "out.abc"]
    with pytest.raises(registry.UnsupportedFormat):
        registry.write(tiny(None), tmp_path / "out.zzz")


def test_supported_read_exts_sorted():
    registry.reader(".lvm", ".LVM2")(tiny)
    registry.reader(".csv")(tiny)
    assert registry.supported_read_exts() == [".csv", ".lvm", ".lvm2"]


def test_broken_plugin_reported(tmp_path, monkeypatch):
    pkg = tmp_path / "fakefmts"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "good.py").write_text(textwrap.dedent("""
        from workbench.core import registry
        @registry.reader('.good')
        def read_good(path):
            raise NotImplementedError
    """))
    (pkg / "bad.py").write_text("raise RuntimeError('boom')\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    problems = registry.load_formats("fakefmts")
    assert len(problems) == 1
    assert "fakefmts.bad" in problems[0] and "boom" in problems[0]
    assert ".good" in registry.supported_read_exts()      # the good module still loaded
    sys.modules.pop("fakefmts.good", None)
    sys.modules.pop("fakefmts.bad", None)
    sys.modules.pop("fakefmts", None)
