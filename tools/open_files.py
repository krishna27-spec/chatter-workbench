"""Baseline timing of the numpy reference (part 1), so part 1b's C++ can show what it gained.

Times registry.read (file -> Recording), MinMaxPyramid build per channel, and one full-width envelope
(what one redraw of a zoomed-out lane costs). Run: env -u PYTHONPATH .venv-wb/bin/python tools/open_files.py
"""
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from workbench.core import registry                       # noqa: E402
from workbench.core.reference.pyramid import MinMaxPyramid  # noqa: E402

FILES = [ROOT / "data/raw/Acce_20k_1_r1.lvm", ROOT / "data/recordings/session_20261023_032238/raw.lvm"]


def main():
    registry.load_formats()
    for path in FILES:
        if not path.exists():
            print(f"{path.name}: missing, skipped")
            continue
        mb = path.stat().st_size / 1e6
        t = time.perf_counter()
        rec = registry.read(path)
        t_read = time.perf_counter() - t
        n = len(rec.channels[0].data)
        t = time.perf_counter()
        pyrs = [MinMaxPyramid(c.data) for c in rec.channels]
        t_pyr = time.perf_counter() - t
        t = time.perf_counter()
        for _ in range(20):
            pyrs[0].envelope(0, n, 1920)
        t_env = (time.perf_counter() - t) / 20
        x = rec.channels[0].data
        t = time.perf_counter()
        np.fmax.reduceat(x, np.arange(0, n, n // 1920))       # naive: scan every sample once
        t_scan = time.perf_counter() - t
        print(f"{path.relative_to(ROOT)}: {mb:.0f} MB, {len(rec.channels)} ch x {n:,} samples")
        print(f"  read            {t_read:7.2f} s   ({mb / t_read:.0f} MB/s)")
        print(f"  pyramid build   {t_pyr:7.3f} s   ({len(rec.channels)} channels, {pyrs[0].levels} levels)")
        print(f"  envelope 1920px {t_env * 1e3:7.2f} ms  (naive full scan of one channel: {t_scan * 1e3:.1f} ms)")


if __name__ == "__main__":
    main()
