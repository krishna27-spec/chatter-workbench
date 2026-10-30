from pathlib import Path

import numpy as np
import pytest

from workbench.core import registry
from workbench.core.formats.lvm import read_lvm, time_to_fs, write_lvm
from workbench.core.recording import Channel, Recording

ROOT = Path(__file__).resolve().parents[2]


def rows(*lines):
    return "".join(line + "\n" for line in lines)


def test_round_trip_keeps_9_digits(tmp_path):
    rng = np.random.default_rng(0)
    data = rng.normal(0, 0.01, (3, 1000)).astype(np.float32)
    rec = Recording([Channel(n, "V", d) for n, d in zip("XYZ", data)], 20000.0)
    write_lvm(rec, tmp_path / "a.lvm")
    back = read_lvm(tmp_path / "a.lvm")
    assert back.fs == 20000.0
    assert [c.name for c in back.channels] == ["X", "Y", "Z"]
    for a, b in zip(rec.channels, back.channels):
        assert b.data.dtype == np.float32
        assert np.array_equal(a.data, b.data)          # every bit, not just "close"


def test_reads_the_lab_files():
    path = ROOT / "data" / "raw" / "Acce_20k_1_r1.lvm"
    if not path.exists():
        pytest.skip("lab file not on this machine")
    rec = read_lvm(path)
    assert [c.name for c in rec.channels] == ["X", "Y", "Z"]
    assert rec.fs == 20000.0
    assert rec.channels[0].data[0] == np.float32(0.003062)
    assert "gaps" not in rec.meta


def test_four_columns_get_numbered_names(tmp_path):
    (tmp_path / "s.lvm").write_text(rows("0\t1\t2\t3\t4", "0.00005\t1\t2\t3\t4", "0.0001\t1\t2\t3\t4"))
    rec = read_lvm(tmp_path / "s.lvm")
    assert [c.name for c in rec.channels] == ["ch1", "ch2", "ch3", "ch4"]


def test_scientific_notation_in_time(tmp_path):
    (tmp_path / "s.lvm").write_text(rows("0.000000\t1", "5.000000E-5\t2", "0.000100\t3"))
    rec = read_lvm(tmp_path / "s.lvm")
    assert rec.fs == 20000.0 and rec.n_samples == 3


def test_truncated_last_line(tmp_path):
    # scope.py killed while writing: the last line is missing its other columns
    (tmp_path / "t.lvm").write_text(rows("0\t0.1\t0.2", "0.00005\t0.1\t0.2", "0.0001\t0.1\t0.2") + "0.00015\t0.1")
    rec = read_lvm(tmp_path / "t.lvm")
    assert rec.n_samples == 3
    assert "incomplete last line" in rec.meta["notice"]


def test_cut_inside_a_number(tmp_path):
    # the cut can land in the middle of a number: "-1.5E" is not a number at all
    (tmp_path / "t.lvm").write_text(rows("0\t0.1", "0.00005\t0.1", "0.0001\t0.1") + "0.00015\t-1.5E")
    rec = read_lvm(tmp_path / "t.lvm")
    assert rec.n_samples == 3
    assert "incomplete last line" in rec.meta["notice"]


def test_gap_detected(tmp_path):
    t = [0.0, 0.00005, 0.0001, 0.5, 0.50005, 0.5001]
    (tmp_path / "g.lvm").write_text(rows(*(f"{x}\t1" for x in t)))
    rec = read_lvm(tmp_path / "g.lvm")
    assert rec.fs == 20000.0                              # one big jump does not move fs
    gap = rec.meta["gaps"][0]
    assert gap["at_s"] == pytest.approx(0.0001)
    assert gap["missing_s"] == pytest.approx(0.5 - 0.0001 - 0.00005)
    assert gap["kind"] == "gap"


def test_clock_restart_detected(tmp_path):
    t = [10.0, 10.00005, 10.0001, 0.0, 0.00005, 0.0001]
    (tmp_path / "r.lvm").write_text(rows(*(f"{x}\t1" for x in t)))
    rec = read_lvm(tmp_path / "r.lvm")
    assert rec.fs == 20000.0
    assert rec.meta["gaps"][0]["kind"] == "restart"


def test_fs_exact_when_time_has_6_decimals():
    # 51.2 kHz written with 6 decimals: each step reads as 0.000019 or 0.000020, never 0.00001953125
    t = np.round(np.arange(51200) / 51200.0, 6)
    fs, gaps = time_to_fs(t)
    assert fs == 51200.0
    assert gaps == []


def test_labview_header_skipped(tmp_path):
    header = rows("LabVIEW Measurement", "Writer_Version\t2", "***End_of_Header***", "",
                  "Channels\t2", "Samples\t3\t3", "***End_of_Header***", "X_Value\tUntitled\tUntitled 1\tComment")
    (tmp_path / "h.lvm").write_text(header + rows("0\t1\t4", "0.00005\t2\t5", "0.0001\t3\t6"))
    rec = read_lvm(tmp_path / "h.lvm")
    assert rec.n_samples == 3
    assert rec.channels[1].data.tolist() == [4, 5, 6]


def test_file_without_numbers_names_the_file(tmp_path):
    (tmp_path / "empty.lvm").write_text("just some text\n")
    with pytest.raises(ValueError, match="empty.lvm"):
        read_lvm(tmp_path / "empty.lvm")


def test_registry_opens_lvm(tmp_path):
    (tmp_path / "a.LVM").write_text(rows("0\t1", "0.00005\t2"))
    assert registry.read(tmp_path / "a.LVM").n_samples == 2
