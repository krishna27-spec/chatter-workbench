"""Shared test setup.

Qt draws into memory instead of a real screen, so GUI tests run anywhere (and never pop up).
`make_session` builds a tiny scope.py session folder (raw.lvm, control.csv, events.txt, session.json) for any test.
"""
import json
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

HEADER = ("t,wall_time,vote,extra_trees,random_forest,svm_y,rule,level_ratio,entropy_drop,rpm_cmd,"
          "spindle_V,feed_V,feed_override_pct,writing,chatter_f_Hz,nu,controller_state")


def control_row(t, wall, vote, et="", rf="", svm="", spindle=1.4286, writing=0):
    return f"{t},{wall},{vote},{et},{rf},{svm},rule,0.1,0.8,20000.0,{spindle},5.0,100.0,{writing},,,watching"


def _make_session(folder, raw=True, control=True, events=True, meta=True):
    folder.mkdir()
    if raw:
        (folder / "raw.lvm").write_text("".join(f"{i / 20000:.6f}\t{i}\t0\t0\n" for i in range(200)))
    if control:
        rows = [control_row(0.0256, "03:22:38.941", "no cutting", "no cutting", "no cutting", "no cutting"),
                control_row(0.0512, "03:22:39.500", "stable", "stable", "stable", "stable"),
                control_row(0.0768, "03:22:40.100", "stable", "chatter", "stable", "stable", writing=1),
                control_row(0.1024, "03:22:41.000", "chatter", "chatter", "chatter", "stable", spindle=1.6, writing=1),
                control_row(0.1280, "03:22:42.000", "")]
        (folder / "control.csv").write_text(HEADER + "\n" + "\n".join(rows) + "\n")
    if events:
        (folder / "events.txt").write_text("03:22:40  OUTPUT STARTED: writing to ao0\n"
                                           "0.09 s  chatter at  8167.7 Hz -> no change\n")
    if meta:
        (folder / "session.json").write_text(json.dumps({"started": "2026-10-23T03:22:38", "args": {"fs": 20000}}))
    return folder



@pytest.fixture
def make_session():
    """Factory: make_session(folder, raw=True, control=True, events=True, meta=True) -> folder."""
    return _make_session
