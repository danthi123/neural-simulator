"""Unit tests for tools/battery_status.py -- the read-only battery-readiness accounting behind
tools/status.sh's Batteries section and a battery's harvest step. Hermetic: every test builds its own tmp
`raw/` tree + TSV registry and passes `--root`, so this never reads the real
research/coordination/handoff_batteries.tsv or the real research/findings/raw/ tree.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "battery_status.py"
sys.path.insert(0, str(ROOT / "tools"))

import battery_status as bs  # noqa: E402


def _write_tsv(path: Path, rows: list[tuple[str, str, int, str, str]]) -> None:
    lines = ["name\traw_glob\texpected_rows\tharvest_cmd\tfinding_template"]
    for name, raw_glob, expected, harvest, template in rows:
        lines.append(f"{name}\t{raw_glob}\t{expected}\t{harvest}\t{template}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _make_raw_files(root: Path, rel_dir: str, names: list[str]) -> None:
    d = root / rel_dir
    d.mkdir(parents=True, exist_ok=True)
    for n in names:
        (d / n).write_text("{}", encoding="utf-8")


# ---------------------------------------------------------------------------------------------------------------
# parse_tsv
# ---------------------------------------------------------------------------------------------------------------

def test_parse_tsv_skips_comments_blanks_and_header(tmp_path):
    tsv = tmp_path / "b.tsv"
    tsv.write_text(
        "# a comment\n"
        "\n"
        "name\traw_glob\texpected_rows\tharvest_cmd\tfinding_template\n"
        "fi\traw/_fi/**/*.json\t6\techo hi\ttmpl.md\n",
        encoding="utf-8",
    )
    out = bs.parse_tsv(str(tsv))
    assert len(out) == 1
    assert out[0] == bs.Battery("fi", "raw/_fi/**/*.json", 6, "echo hi", "tmpl.md")


def test_parse_tsv_rejects_wrong_field_count():
    import pytest

    tsv_path = None
    try:
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".tsv", delete=False) as fh:
            fh.write("a\tb\tc\n")
            tsv_path = fh.name
        with pytest.raises(ValueError, match="expected 5 tab-separated fields"):
            bs.parse_tsv(tsv_path)
    finally:
        if tsv_path:
            Path(tsv_path).unlink(missing_ok=True)


def test_parse_tsv_rejects_non_integer_expected_rows(tmp_path):
    import pytest

    tsv = tmp_path / "b.tsv"
    tsv.write_text("fi\traw/_fi/*.json\tsix\techo hi\ttmpl.md\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not an integer"):
        bs.parse_tsv(str(tsv))


# ---------------------------------------------------------------------------------------------------------------
# landed_files / is_running / status_line
# ---------------------------------------------------------------------------------------------------------------

def test_landed_files_excludes_prov_json_sidecars(tmp_path):
    _make_raw_files(tmp_path, "raw/_fi", ["s1.json", "s2.json", "s1.json.prov.json"])
    files = bs.landed_files(str(tmp_path), "raw/_fi/**/*.json")
    assert len(files) == 2
    assert all(not f.endswith(".prov.json") for f in files)


def test_landed_files_recurses_into_subdirs(tmp_path):
    _make_raw_files(tmp_path, "raw/_d6/cell_00", ["a.json"])
    _make_raw_files(tmp_path, "raw/_d6/cell_01", ["a.json"])
    files = bs.landed_files(str(tmp_path), "raw/_d6/**/*.json")
    assert len(files) == 2


def test_landed_files_empty_when_dir_absent(tmp_path):
    assert bs.landed_files(str(tmp_path), "raw/_nope/**/*.json") == []


def test_is_running_substring_match():
    assert bs.is_running("fi", "job for fi seed3 running") is True
    assert bs.is_running("fi", "job for settle_a2_wiring running") is False
    assert bs.is_running("", "anything") is False  # an empty name can never match (guards a blank TSV row)


def test_status_line_ready_when_landed_meets_expected_and_not_running(tmp_path):
    _make_raw_files(tmp_path, "raw/_fi", ["s1.json", "s2.json"])
    b = bs.Battery("fi", "raw/_fi/**/*.json", 2, "echo", "tmpl.md")
    name, landed, expected, state = bs.status_line(b, str(tmp_path), running_text="")
    assert (name, landed, expected, state) == ("fi", 2, 2, "READY")


def test_status_line_waiting_when_landed_below_expected(tmp_path):
    _make_raw_files(tmp_path, "raw/_fi", ["s1.json"])
    b = bs.Battery("fi", "raw/_fi/**/*.json", 6, "echo", "tmpl.md")
    _, landed, expected, state = bs.status_line(b, str(tmp_path), running_text="")
    assert (landed, expected, state) == (1, 6, "WAITING")


def test_status_line_running_overrides_ready(tmp_path):
    """A battery whose full row count has already landed but is STILL named in the running-text must never
    read READY -- it may still be writing its last row (mutation: dropping the `is_running` check entirely
    would make this test fail, since landed >= expected alone would then read READY)."""
    _make_raw_files(tmp_path, "raw/_fi", ["s1.json", "s2.json"])
    b = bs.Battery("fi", "raw/_fi/**/*.json", 2, "echo", "tmpl.md")
    _, _, _, state = bs.status_line(b, str(tmp_path), running_text="node1:running fi seed2")
    assert state == "RUNNING"


# ---------------------------------------------------------------------------------------------------------------
# CLI (subprocess): --list and --harvest
# ---------------------------------------------------------------------------------------------------------------

def _run(args, cwd=None):
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True,
                           cwd=cwd, timeout=30)


def test_cli_list_prints_one_tsv_line_per_battery(tmp_path):
    _write_tsv(tmp_path / "b.tsv", [("fi", "raw/_fi/**/*.json", 2, "echo hi", "tmpl.md"),
                                     ("d6", "raw/_d6/**/*.json", 1, "echo hi", "tmpl.md")])
    _make_raw_files(tmp_path, "raw/_fi", ["s1.json", "s2.json"])
    res = _run(["--tsv", str(tmp_path / "b.tsv"), "--root", str(tmp_path)])
    assert res.returncode == 0, res.stderr
    lines = res.stdout.strip("\n").split("\n")
    assert lines == ["fi\t2\t2\tREADY", "d6\t0\t1\tWAITING"]


def test_cli_running_stdin_marks_a_battery_running(tmp_path):
    _write_tsv(tmp_path / "b.tsv", [("fi", "raw/_fi/**/*.json", 2, "echo hi", "tmpl.md")])
    _make_raw_files(tmp_path, "raw/_fi", ["s1.json", "s2.json"])
    res = subprocess.run([sys.executable, str(SCRIPT), "--tsv", str(tmp_path / "b.tsv"),
                           "--root", str(tmp_path), "--running-stdin"],
                          input="node1:running fi\n", capture_output=True, text=True, timeout=30)
    assert res.returncode == 0, res.stderr
    assert res.stdout.strip() == "fi\t2\t2\tRUNNING"


def test_cli_harvest_lists_landed_files_and_recipe(tmp_path):
    _write_tsv(tmp_path / "b.tsv", [("fi", "raw/_fi/**/*.json", 2, "MY HARVEST CMD", "MY TEMPLATE.md")])
    _make_raw_files(tmp_path, "raw/_fi", ["s1.json", "s2.json"])
    res = _run(["--tsv", str(tmp_path / "b.tsv"), "--root", str(tmp_path), "--harvest", "fi"])
    assert res.returncode == 0, res.stderr
    assert "MY HARVEST CMD" in res.stdout
    assert "MY TEMPLATE.md" in res.stdout
    assert str(tmp_path / "raw" / "_fi" / "s1.json") in res.stdout
    assert str(tmp_path / "raw" / "_fi" / "s2.json") in res.stdout


def test_cli_harvest_unknown_battery_fails_loudly(tmp_path):
    _write_tsv(tmp_path / "b.tsv", [("fi", "raw/_fi/**/*.json", 2, "echo", "tmpl.md")])
    res = _run(["--tsv", str(tmp_path / "b.tsv"), "--root", str(tmp_path), "--harvest", "does-not-exist"])
    assert res.returncode == 1
    assert "no battery named" in res.stderr


def test_cli_malformed_tsv_exits_2_not_a_traceback(tmp_path):
    bad = tmp_path / "bad.tsv"
    bad.write_text("only\tone\trow\n", encoding="utf-8")
    res = _run(["--tsv", str(bad), "--root", str(tmp_path)])
    assert res.returncode == 2
    assert "Traceback" not in res.stderr
