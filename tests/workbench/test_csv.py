import numpy as np
import pytest

from workbench.core import registry
from workbench.core.formats.csv_fmt import read_csv, write_csv
from workbench.core.recording import Channel, Recording


def rec3(n=1000, seed=0):
    data = np.random.default_rng(seed).normal(0, 0.01, (3, n)).astype(np.float32)
    return Recording([Channel(c, "V", d) for c, d in zip("XYZ", data)], 20000.0)


def test_opens_in_spreadsheet_layout(tmp_path):
    write_csv(rec3(), tmp_path / "a.csv")
    assert (tmp_path / "a.csv").read_text().splitlines()[0] == "t [s],X [V],Y [V],Z [V]"


def test_round_trip_keeps_9_digits(tmp_path):
    rec = rec3()
    write_csv(rec, tmp_path / "a.csv")
    back = read_csv(tmp_path / "a.csv")
    assert back.fs == 20000.0
    for a, b in zip(rec.channels, back.channels):
        assert b.data.dtype == np.float32 and np.array_equal(a.data, b.data)


def test_round_trip_names_and_units(tmp_path):
    rec = Recording([Channel("A", "g", np.arange(5, dtype=np.float32)),
                     Channel("B", "", np.ones(5, np.float32))], 1000.0, t0=2.5)
    write_csv(rec, tmp_path / "a.csv")
    back = read_csv(tmp_path / "a.csv")
    assert [(c.name, c.unit) for c in back.channels] == [("A", "g"), ("B", "")]
    assert back.t0 == 2.5


def test_comma_inside_a_name(tmp_path):
    rec = Recording([Channel("X, feed", "m/s²", np.arange(4, dtype=np.float32))], 100.0)
    write_csv(rec, tmp_path / "a.csv")
    back = read_csv(tmp_path / "a.csv")
    assert (back.channels[0].name, back.channels[0].unit) == ("X, feed", "m/s²")


def test_semicolons_and_decimal_commas(tmp_path):
    # what Excel writes with a European / Indian locale
    (tmp_path / "e.csv").write_text("t [s];X [V];Y [V]\n0;0,5;1\n0,00005;0,25;2\n0,0001;-1,5;3\n")
    rec = read_csv(tmp_path / "e.csv")
    assert rec.fs == 20000.0
    assert rec.channels[0].data.tolist() == [0.5, 0.25, -1.5]


def test_excel_bom_ignored(tmp_path):
    (tmp_path / "b.csv").write_bytes("﻿t [s],X [V]\n0,1\n0.5,2\n".encode("utf-8"))
    rec = read_csv(tmp_path / "b.csv")
    assert rec.channels[0].name == "X" and rec.fs == 2.0


def test_time_in_milliseconds(tmp_path):
    (tmp_path / "m.csv").write_text("time [ms],X\n0,1\n0.05,2\n0.1,3\n")
    rec = read_csv(tmp_path / "m.csv")
    assert rec.fs == 20000.0


def test_no_time_column_is_refused(tmp_path):
    (tmp_path / "n.csv").write_text("X [V],Y [V]\n1,2\n3,4\n")
    with pytest.raises(ValueError, match="n.csv.*time"):
        read_csv(tmp_path / "n.csv")


def test_no_header_reads_like_lvm(tmp_path):
    (tmp_path / "h.csv").write_text("0,1,2,3\n0.00005,1,2,3\n")
    rec = read_csv(tmp_path / "h.csv")
    assert [c.name for c in rec.channels] == ["X", "Y", "Z"]


def test_truncated_last_line(tmp_path):
    (tmp_path / "t.csv").write_text("t [s],X [V]\n0,1\n0.00005,2\n0.0001,3\n0.00015,-1.5E")
    rec = read_csv(tmp_path / "t.csv")
    assert rec.n_samples == 3 and "incomplete last line" in rec.meta["notice"]


def test_gap_reported(tmp_path):
    (tmp_path / "g.csv").write_text("t [s],X\n0,1\n0.00005,1\n0.0001,1\n0.5,1\n0.50005,1\n")
    rec = read_csv(tmp_path / "g.csv")
    assert rec.fs == 20000.0 and rec.meta["gaps"][0]["kind"] == "gap"


def test_header_and_columns_disagree(tmp_path):
    (tmp_path / "d.csv").write_text("t [s],X [V],Y [V]\n0,1\n0.5,2\n")
    with pytest.raises(ValueError, match="3 names.*2 columns"):
        read_csv(tmp_path / "d.csv")


def test_registry_opens_csv(tmp_path):
    write_csv(rec3(10), tmp_path / "r.CSV")
    assert registry.read(tmp_path / "r.CSV").n_samples == 10
