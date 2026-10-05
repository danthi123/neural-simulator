"""RULING T1-lr0 (2026-10-05): a learning-off (lr 0) cell of the Step 4a/4b pilot runner declares its behaviour
UNDEFINED in the artifact, so tools/gates/below_chance.py reads the declaration instead of a fabricated negative; a
learning cell never carries it. Pinned failing-first: the chance-level record shape below trips the gate without it.
"""
import json

import pytest

from tools.gates import below_chance

pilot = pytest.importorskip("research.runners._ldc_outcome_timing_pilot")


def _score(success, acc):
    return {"success": success, "response_rate": 0.97, "acc_among_responders": acc, "chance": 0.5}


def _record(lr):
    # the runner's own output shape (run_condition + main), at a chance-level accuracy draw just under 0.5
    row = {"schedule": "credit_immediate", "arm": "full", "seed": 9, "relevant": "early",
           "test": {"early": _score(0.41, 0.4227), "terminal": _score(0.44, 0.4536)},
           "success": 0.41, "response_rate": 0.97, "chance": 0.5, "mem_hold_acc": 1.0,
           "instrument": {"mem_after_clear_hz": 0.0, "mem_hold_acc_probe": 1.0}}
    doc = {"runner_sha256": "x", "params": {"lr": lr, "w_clear_fs": 60.0}, "schedule": "credit_immediate",
           "rows": [row]}
    doc.update(pilot.lr0_declaration(doc["params"]) or {})
    return doc


def _bare(doc):
    return {k: v for k, v in doc.items() if k not in ("behavior_verdict", "behavior_undefined_reason")}


def test_the_chance_level_shape_trips_the_gate_without_the_declaration(tmp_path):
    p = tmp_path / "bare.json"
    p.write_text(json.dumps(_bare(_record(0.0))))
    hits = below_chance.check([str(p)])
    assert len(hits) == 1 and "/rows[0]/test/early" in hits[0], "the fixture must reproduce the T1 STOP"


def test_lr0_record_declares_and_clears_the_gate(tmp_path):
    doc = _record(0.0)
    assert "UNDEFINED" in doc["behavior_verdict"] and doc["behavior_undefined_reason"] == "lr0-diagnostic"
    p = tmp_path / "cell.json"
    p.write_text(json.dumps(doc))
    assert below_chance.check([str(p)]) == []


def test_learning_cell_is_never_declared():
    assert pilot.lr0_declaration({"lr": 5.0}) is None
    assert pilot.lr0_declaration({}) is None
    assert "behavior_verdict" not in _record(5.0)


def test_declare_mode_adds_the_declaration_and_refuses_a_learning_record(tmp_path):
    old = tmp_path / "old.json"
    old.write_text(json.dumps(_bare(_record(0.0))))
    learn = tmp_path / "learn.json"
    learn.write_text(json.dumps(_bare(_record(5.0))))
    assert pilot.main(["--declare-lr0", str(old)]) == 0
    d = json.loads(old.read_text())
    assert d["behavior_undefined_reason"] == "lr0-diagnostic"
    assert d["declared_after_run"]["declaring_runner_sha256"] == pilot.RUNNER_SHA256
    assert d["rows"] == _bare(_record(0.0))["rows"], "no measured value may change"
    assert below_chance.check([str(old)]) == []
    assert pilot.main(["--declare-lr0", str(learn)]) == 2
    assert "behavior_verdict" not in json.loads(learn.read_text())
    assert pilot.main(["--declare-lr0", str(old)]) == 0  # idempotent
