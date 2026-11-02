"""tests for tools/gates/lb_battery_provenance.py (CLASS LBP).

Same convention as tests/test_gate_finding_mechanism_on_main.py: import the REAL gate module the pre-commit
registry calls, so these tests cannot drift from what the hook actually runs. The module's own selftest() is the
registry's trust mechanism; this file adds pytest-level coverage on top so a regression shows up as an ordinary
test failure, not only at commit time."""
from __future__ import annotations

import json
import os

import tools.gates.lb_battery_provenance as lbp_gate


def test_registry_selftest_passes():
    problems = lbp_gate.selftest()
    assert problems == [], "gate selftest reported problems: %r" % problems


def test_registry_discovers_this_gate_with_the_expected_contract():
    from tools.gates import discover
    hits = [t for t in discover() if t[0] == lbp_gate.NAME]
    assert len(hits) == 1, "gates/__init__.discover() did not find exactly one %r module" % lbp_gate.NAME
    name, mod, err = hits[0]
    assert err is None, "the registry reports this gate as broken: %s" % err
    assert mod.CLASS_ID == "LBP"
    assert mod.BLOCKING is True


def _write(tmp_path, tag, obj):
    d = tmp_path / "research" / "findings" / "raw" / "_load_bearing" / "_shards" / tag
    d.mkdir(parents=True, exist_ok=True)
    p = d / "aggregate.json"
    p.write_text(json.dumps(obj))
    return str(p)


COVERABLE = {"per_faculty": {"fac-a": {"seeds_present": [42, 43]}}}


# --- failing direction: the incident this gate exists for --------------------------------------------------

def test_blocks_an_aggregate_with_no_provenance_key_at_all(tmp_path):
    p = _write(tmp_path, "b2a-like", dict(COVERABLE))
    problems = lbp_gate.check([p])
    assert problems and "LBP" in problems[0]


def test_blocks_an_aggregate_explicitly_marked_unverified(tmp_path):
    obj = dict(COVERABLE, provenance={"status": "unverified", "pin": None})
    p = _write(tmp_path, "unverified-tag", obj)
    problems = lbp_gate.check([p])
    assert problems and "unverified" in problems[0].lower()


# --- passing direction: must not cry wolf -------------------------------------------------------------------

def test_passes_an_aggregate_verified_against_a_pin(tmp_path):
    obj = dict(COVERABLE, provenance={"status": "verified", "pin": "a" * 40, "n_invalid": 0})
    p = _write(tmp_path, "verified-tag", obj)
    assert lbp_gate.check([p]) == []


def test_passes_an_explicit_unverified_acceptance_waiver(tmp_path):
    obj = dict(COVERABLE, provenance={"status": "unverified"},
               provenance_unverified_accepted="dev smoke, not a verdict")
    p = _write(tmp_path, "waived-tag", obj)
    assert lbp_gate.check([p]) == []


def test_passes_an_aggregate_with_no_coverable_data_yet(tmp_path):
    p = _write(tmp_path, "empty-tag", {"per_faculty": {}})
    assert lbp_gate.check([p]) == []


def test_ignores_nonexistent_paths_and_empty_staged_list(tmp_path):
    assert lbp_gate.check([]) == []
    assert lbp_gate.check([str(tmp_path / "no" / "such" / "aggregate.json")]) == []


def test_ignores_an_aggregate_json_outside_load_bearing(tmp_path):
    d = tmp_path / "research" / "findings" / "raw" / "somewhere_else"
    d.mkdir(parents=True)
    p = d / "aggregate.json"
    p.write_text(json.dumps(COVERABLE))
    assert lbp_gate.check([str(p)]) == []
