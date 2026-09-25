"""declare_output() -- research/runners/__init__.py's public API for a run to register an output written outside
anything named by --out/--output/--json on argv (2026-09-25, closing the gap that left every open-ended-generation
load-bearing cell's oed_distributional<seed>.json permanently un-sidecared: load_bearing_fraction.py writes that
file from the PARENT process into its declared --out's directory, and _declared_output_paths only ever looked at
paths named by an output FLAG on argv).

Uses the same in-process monkeypatch style as test_runner_provenance_snapshot.py (call _stamp_outputs /
_declared_output_paths directly against a fake _ROOT/_RAW_DIR) rather than test_runner_provenance_ownership.py's
subprocess style, since nothing here needs a fresh interpreter.
"""
import json
import os
import sys

import research.runners as provenance


def _base_rec(root, argv):
    return {
        "run_id": "declare-output-test",
        "runner": "research/runners/example.py",
        "argv": argv,
        "cwd": str(root),
        "git_sha": "deadbeef",
        "git_dirty": False,
        "started": "2026-09-25T00:00:00",
        "env": {"SIM_BACKEND": "numpy"},
    }


def _set_argv(monkeypatch, argv):
    """_stamp_outputs calls _resolve_argv(rec) BEFORE using rec["argv"], and _resolve_argv always overwrites
    rec["argv"] with the CURRENT PROCESS's real sys.argv (that is how a runner invoked as `-m research.runners.X`
    gets its true argv, since `-m` has not yet rewritten sys.argv[0] at import time -- see _resolve_argv's own
    docstring). Under pytest, sys.argv is pytest's own invocation, not the fake argv a test builds by hand, so any
    test that wants argv-based --out detection to fire must monkeypatch sys.argv itself, not just rec["argv"]."""
    monkeypatch.setattr(sys, "argv", argv)


def test_declare_output_is_sidecared_alongside_an_argv_declared_out(tmp_path, monkeypatch):
    raw = tmp_path / "research" / "findings" / "raw"
    raw.mkdir(parents=True)
    declared = raw / "lb.json"
    extra = raw / "oed_distributional.json"      # written by the run but NOT named on argv anywhere
    undeclared = raw / "peer_shard_output.json"   # a different run's fresh file in the same directory
    for p in (declared, extra, undeclared):
        p.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(provenance, "_ROOT", str(tmp_path))
    monkeypatch.setattr(provenance, "_RAW_DIR", str(raw))
    monkeypatch.setattr(provenance, "_START", 0.0)
    monkeypatch.setattr(provenance, "_EXTRA_DECLARED_OUTPUTS", [])

    provenance.declare_output(str(extra))
    argv = ["load_bearing_fraction.py", "--out", str(declared)]
    _set_argv(monkeypatch, argv)
    rec = _base_rec(tmp_path, argv)

    made = provenance._stamp_outputs(rec)

    assert sorted(made) == sorted([str(declared), str(extra)])
    assert (raw / "lb.json.prov.json").exists()
    assert (raw / "oed_distributional.json.prov.json").exists()
    assert not (raw / "peer_shard_output.json.prov.json").exists()
    sidecar = json.loads((raw / "oed_distributional.json.prov.json").read_text())
    assert sidecar["artifact"] == os.path.relpath(str(extra), str(tmp_path))


def test_declare_output_alone_is_treated_as_declared_with_no_output_flag_on_argv(tmp_path, monkeypatch):
    """Without any --out/--output/--json on argv, _declared_output_paths would normally report `seen=False` and
    _stamp_outputs would fall back to scanning raw/ for files freshly written since _START. Give the file an mtime
    OLDER than _START so that fallback provably would NOT catch it, and confirm declare_output() alone still gets
    it sidecared -- i.e. it goes through the explicit-path route, not the fresh-file scan."""
    raw = tmp_path / "research" / "findings" / "raw"
    raw.mkdir(parents=True)
    extra = raw / "oed_distributional_s43.json"
    extra.write_text("{}", encoding="utf-8")
    old_mtime = 1_000_000.0
    os.utime(extra, (old_mtime, old_mtime))

    monkeypatch.setattr(provenance, "_ROOT", str(tmp_path))
    monkeypatch.setattr(provenance, "_RAW_DIR", str(raw))
    monkeypatch.setattr(provenance, "_START", old_mtime + 10.0)  # started AFTER the file's mtime
    monkeypatch.setattr(provenance, "_EXTRA_DECLARED_OUTPUTS", [])

    provenance.declare_output(str(extra))
    argv = ["load_bearing_fraction.py"]  # no output flag at all
    _set_argv(monkeypatch, argv)
    rec = _base_rec(tmp_path, argv)

    made = provenance._stamp_outputs(rec)

    assert made == [str(extra)]
    assert (raw / "oed_distributional_s43.json.prov.json").exists()


def test_declare_output_outside_raw_is_ignored_not_fatal(tmp_path, monkeypatch):
    raw = tmp_path / "research" / "findings" / "raw"
    raw.mkdir(parents=True)
    outside = tmp_path / "elsewhere.json"
    outside.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(provenance, "_ROOT", str(tmp_path))
    monkeypatch.setattr(provenance, "_RAW_DIR", str(raw))
    monkeypatch.setattr(provenance, "_START", 0.0)
    monkeypatch.setattr(provenance, "_EXTRA_DECLARED_OUTPUTS", [])

    provenance.declare_output(str(outside))
    argv = ["load_bearing_fraction.py"]
    _set_argv(monkeypatch, argv)
    rec = _base_rec(tmp_path, argv)

    made = provenance._stamp_outputs(rec)

    assert made == []
    assert not (tmp_path / "elsewhere.json.prov.json").exists()


def test_declare_output_never_raises_on_a_bad_value(monkeypatch):
    monkeypatch.setattr(provenance, "_EXTRA_DECLARED_OUTPUTS", [])
    provenance.declare_output(123)       # not a string; str() still succeeds, so this just records "123"
    provenance.declare_output(object())  # str() succeeds on any object -- never raises into the run it instruments
    assert len(provenance._EXTRA_DECLARED_OUTPUTS) == 2
