"""RULING iw1-R-pre (cloud Claude session 7, 2026-10-05): tools/iw1_score.py carries the tools.verdict.Verdict
preconditions block beside every R/B/A verdict, ported from tools/iw2_score.py, so the verdict-preconditions gate
accepts the iw1 R files Hermes staged. A PASS rests on every cell being DEFINED; a vacuous A is never a PASS."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import iw1_score as S  # noqa: E402
from tools.gates import verdict_preconditions as G  # noqa: E402


def _cells(monkeypatch, states):
    it = iter(states)
    monkeypatch.setattr(S, "_cell_std", lambda *a, **k: next(it))


def test_R_pass_carries_its_earned_block(monkeypatch, tmp_path):
    _cells(monkeypatch, [("DEFINED", "regressed", {})] * len(S.SEEDS))
    res, code = S.score_R("d6-hebbian-write", "t", "p")
    assert (res["verdict"], code) == ("PASS", S.PASS)
    assert res["preconditions"] and all(p["ok"] for p in res["preconditions"])
    out = tmp_path / "R_d6.json"
    out.write_text(json.dumps(res))
    assert G.check([str(out)]) == []


def test_R_with_an_undefined_cell_is_undefined_not_fail(monkeypatch):
    _cells(monkeypatch, [("DEFINED", "regressed", {})] * (len(S.SEEDS) - 1) + [("UNDEFINED:x", None, {})])
    res, code = S.score_R("d6-hebbian-write", "t", "p")
    assert (res["verdict"], code) == ("UNDEFINED", S.UNDEF)


def test_R_fail_rests_on_defined_cells(monkeypatch):
    _cells(monkeypatch, [("DEFINED", "regressed", {})] * (len(S.SEEDS) - 1) + [("DEFINED", "pass", {})])
    res, code = S.score_R("d6-hebbian-write", "t", "p")
    assert (res["verdict"], code) == ("FAIL", S.FAIL) and all(p["ok"] for p in res["preconditions"])


def test_A_with_nothing_to_attribute_is_never_a_pass():
    res, code = S.score_A({"regressions": [], "undefined": []}, "t", "p")
    assert res["verdict"] == "UNDEFINED" and res.get("verdict_unearned") == "PASS" and code == S.UNDEF
