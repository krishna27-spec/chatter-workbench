"""`chatter-workbench --self-test`: checks that an installed copy has everything it needs, without hardware.

Formats and ops load; the bundled fonts exist; the three models load and give a prediction on 3 s of generated
signal (2 s calibration + 1 s); the NI-DAQmx driver is found (or why not). The report is printed, written to
Documents/Chatter Workbench/self-test.txt, and shown in a window when installed (a Windows app has no console).
"""
from __future__ import annotations

import platform
import sys
import time
import traceback

import numpy as np

from workbench import __version__
from workbench.paths import bundle_dir, frozen, model_package, recordings_dir


def run() -> tuple[bool, str]:
    lines, ok = [], True

    def check(name, fn):
        nonlocal ok
        t = time.perf_counter()
        try:
            msg = fn()
            lines.append(f"OK    {name}: {msg}  ({time.perf_counter() - t:.2f} s)")
        except Exception as e:
            ok = False
            lines.append(f"FAIL  {name}: {type(e).__name__}: {e}")
            lines.append("      " + traceback.format_exc().strip().splitlines()[-1])

    lines.append(f"Chatter Workbench {__version__} self-test · {platform.system()} {platform.release()} · "
                 f"Python {platform.python_version()} · {'installed' if frozen() else 'project folder'}")
    lines.append(f"      data folder: {bundle_dir()}   recordings: {recordings_dir()}")

    def formats():
        from workbench.core import registry, steps
        probs = registry.load_formats() + steps.load_ops()
        if probs:
            raise RuntimeError("; ".join(probs))
        return f"read {', '.join(registry.supported_read_exts())} + session folders; ops {', '.join(steps.OPS)}"

    def fonts():
        from workbench.app.theme import FONT_FILE
        if not FONT_FILE.exists():
            raise FileNotFoundError(FONT_FILE)
        return FONT_FILE.name

    def models():
        from workbench.core.live.top3 import Top3
        m = Top3()
        rng = np.random.default_rng(0)
        out = None
        for _ in range(int(3 * 20_000 / 512)):
            out = m.update(0.01 * rng.normal(size=(3, 512)))
        if out is None:
            raise RuntimeError("no prediction after 3 s")
        return f"{', '.join(m.models)} from {model_package()} → vote '{out['vote']}' on generated noise"

    def daq():
        from workbench.core.live.hardware import list_modules
        mods, problem = list_modules()
        if problem:
            return f"not available ({problem}); Analyse works, Live needs NI-DAQmx from ni.com"
        return f"{len(mods)} modules: " + ", ".join(f"{m.name} {m.product}{' (sim)' if m.simulated else ''}"
                                                   for m in mods) if mods else "driver found, no modules attached"

    check("file formats and Clean", formats)
    check("fonts", fonts)
    check("top-3 models", models)
    check("NI-DAQmx", daq)
    lines.append("ALL OK" if ok else "SOME CHECKS FAILED")
    return ok, "\n".join(lines)


def main() -> int:
    ok, report = run()
    print(report)
    try:
        if not frozen():
            raise OSError("project folder: printed only")
        out = recordings_dir().parent / "self-test.txt"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report + "\n", encoding="utf-8")
    except OSError:
        pass
    if frozen():
        from PySide6.QtWidgets import QApplication, QMessageBox
        app = QApplication.instance() or QApplication(sys.argv)
        box = QMessageBox(QMessageBox.Information if ok else QMessageBox.Warning, "Chatter Workbench self-test", report)
        box.exec()
    return 0 if ok else 1
