"""Which function reads (or writes) which kind of file.

A format file registers itself with a decorator, e.g. in core/formats/lvm.py:

    @reader('.lvm')
    def read_lvm(path): ... return Recording(...)

and the rest of the app only ever calls read(path) / write(rec, path). Adding a format means adding one
file to core/formats/; nothing else changes. load_formats() imports every file in that folder at start-up.
"""
from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path
from typing import Callable

from .recording import Recording

Reader = Callable[[Path], Recording]
Writer = Callable[[Recording, Path], None]

_readers: dict[str, Reader] = {}        # '.lvm' -> read_lvm
_writers: dict[str, Writer] = {}
_folder_reader: list[Reader] = []       # at most one: reads a whole folder (a scope.py session)


class UnsupportedFormat(ValueError):
    """No reader or writer for this file."""


def _norm(ext: str) -> str:
    ext = ext.lower()
    return ext if ext.startswith('.') else '.' + ext


def reader(*exts: str, folder: bool = False):
    """Decorator: register fn(path) -> Recording for these extensions (or for folders)."""
    def register(fn: Reader) -> Reader:
        for e in exts:
            _readers[_norm(e)] = fn
        if folder:
            _folder_reader[:] = [fn]
        return fn
    return register


def writer(*exts: str):
    """Decorator: register fn(rec, path) for these extensions."""
    def register(fn: Writer) -> Writer:
        for e in exts:
            _writers[_norm(e)] = fn
        return fn
    return register


def supported_read_exts() -> list[str]:
    return sorted(_readers)


def read(path: str | Path) -> Recording:
    """Open any supported file or session folder and return its Recording (with .source set)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")
    if path.is_dir():
        if not _folder_reader:
            raise UnsupportedFormat(f"{path.name}: no reader for folders is loaded")
        fn = _folder_reader[0]
    else:
        fn = _readers.get(path.suffix.lower())
        if fn is None:
            raise UnsupportedFormat(
                f"{path.name}: unsupported file type; can open {', '.join(supported_read_exts()) or 'nothing yet'}")
    rec = fn(path)
    rec.source = path
    return rec


def write(rec: Recording, path: str | Path) -> None:
    path = Path(path)
    fn = _writers.get(path.suffix.lower())
    if fn is None:
        raise UnsupportedFormat(f"{path.name}: cannot save as {path.suffix or 'a file without extension'}; "
                                f"can save {', '.join(sorted(_writers)) or 'nothing yet'}")
    fn(rec, path)


def load_formats(package: str = 'workbench.core.formats') -> list[str]:
    """Import every module in the package so their decorators run.

    A module that fails to import is skipped and reported (one message each), so one broken format
    cannot stop the app from starting; the app lists these under Tools > Problems.
    """
    problems = []
    pkg = importlib.import_module(package)
    for info in pkgutil.iter_modules(pkg.__path__):
        name = f"{package}.{info.name}"
        try:
            importlib.import_module(name)
        except Exception as e:                                  # any error inside a plug-in
            problems.append(f"{name}: {type(e).__name__}: {e}")
    return problems


# ---- for tests only: let a test register throwaway formats without leaking into other tests
def _snapshot_for_tests():
    return dict(_readers), dict(_writers), list(_folder_reader)


def _clear_for_tests():
    _readers.clear(); _writers.clear(); _folder_reader.clear()


def _restore_for_tests(saved):
    _clear_for_tests()
    _readers.update(saved[0]); _writers.update(saved[1]); _folder_reader.extend(saved[2])
