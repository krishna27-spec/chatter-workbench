"""The confidentiality rule as a test: no file in workbench/ may import anything that can talk to a network.

It reads the source with Python's own parser (ast) instead of importing it, so it checks every file, including ones
no other test happens to import, and runs no workbench code at all.
"""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

BANNED_TOP = {"socket", "ssl", "http", "urllib", "urllib3", "requests", "httpx", "ftplib", "smtplib", "poplib",
              "imaplib", "telnetlib", "xmlrpc", "websocket", "websockets", "aiohttp", "paramiko"}
BANNED_FULL = {"PySide6.QtNetwork", "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtWebSockets"}


def imported_names(tree: ast.AST):
    """Every module a file imports, with its line: `import a.b` → a.b; `from a import b` → a and a.b."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, node.lineno
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            yield node.module, node.lineno
            for alias in node.names:
                yield f"{node.module}.{alias.name}", node.lineno       # catches `from PySide6 import QtNetwork`


def violations(folder: Path) -> list[str]:
    found = []
    for path in sorted(folder.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for name, line in imported_names(tree):
            if name.split(".")[0] in BANNED_TOP or name in BANNED_FULL:
                found.append(f"{path.relative_to(folder.parent)}:{line} imports {name}")
    return found


def test_no_network_imports():
    found = violations(ROOT / "workbench")
    assert not found, "network-capable imports in the workbench (nothing may leave this laptop):\n" + "\n".join(found)


def test_guard_catches_every_form(tmp_path):
    """The guard itself must work: each way of writing a network import is caught, harmless ones are not."""
    bad = {"a.py": "import socket\n", "b.py": "import urllib.request\n", "c.py": "from http import client\n",
           "d.py": "from PySide6 import QtNetwork\n", "e.py": "from PySide6.QtNetwork import QTcpSocket\n",
           "f.py": "def f():\n    import requests\n"}
    for name, src in bad.items():
        (tmp_path / name).write_text(src)
    (tmp_path / "ok.py").write_text("import numpy\nfrom PySide6 import QtWidgets\nfrom . import sockets_helper\n")
    found = violations(tmp_path)
    files = {f.split(":")[0].split("/")[-1] for f in found}      # one file can give two hits (http, http.client)
    assert files == set(bad)                                      # every bad file caught, ok.py not
    assert any("f.py:2" in f for f in found)            # an import hidden inside a function is still found
