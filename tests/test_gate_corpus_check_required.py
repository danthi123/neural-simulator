"""Tests for tools/gates/corpus_check_required.py (CLASS CC).

Same convention as tests/test_gate_lb_battery_provenance.py: import the REAL gate module the pre-commit
registry calls, so these tests cannot drift from what the hook actually runs. The module's own selftest() is
the registry's trust mechanism (isolated from this real repo's own logs, see its docstring); this file adds
pytest-level coverage on top, including a direct reproduction of the 2026-09-25 incident this gate's fallback
was built to fix (see the module's docstring for the full account)."""
from __future__ import annotations

import json
import os
import time

import tools.gates.corpus_check_required as cc_gate


def test_registry_selftest_passes():
    problems = cc_gate.selftest()
    assert problems == [], "gate selftest reported problems: %r" % problems


def test_registry_discovers_this_gate_with_the_expected_contract():
    from tools.gates import discover
    hits = [t for t in discover() if t[0] == cc_gate.NAME]
    assert len(hits) == 1, "gates/__init__.discover() did not find exactly one %r module" % cc_gate.NAME
    name, mod, err = hits[0]
    assert err is None, "the registry reports this gate as broken: %s" % err
    assert mod.CLASS_ID == "CC"
    assert mod.BLOCKING is True


def _write(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh)


def test_incident_2026_09_25_gap4_artifact_now_passes_via_a_worktree_log(tmp_path, monkeypatch):
    """Reproduces the exact blocked incident: research/score-gap4-c26-0925 @ ed6758f61's
    .../C25/ckpt/s7_r2_transport_ceiling.json (1.08h elapsed) carried no `corpus_check_fresh` anywhere, even
    though `before_you_build.sh` DID log a real check at 2026-09-25T01:38:31 -- under
    `.claude/worktrees/wf_3d929cf5-198-1/research/queue/.corpus_checks.jsonl`, a worktree log the pre-fix gate
    never looked at. Post-fix, `check()` must accept it.

    Mutation-verify: comment out the `_fallback_evidence` call in `_check_one` (or point the glob only at the
    shared root, never `.claude/worktrees/*`) and this test fails.
    """
    monkeypatch.setattr(cc_gate, "_ROOT", str(tmp_path))
    rel = ("research/findings/raw/gap4/transport_ceiling_readout/companion_revfeaca2f/C25/ckpt/"
           "s7_r2_transport_ceiling.json")
    run_start = int(time.mktime(time.strptime("2026-09-25 02:12:00", "%Y-%m-%d %H:%M:%S")))
    started_str = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(run_start))
    _write(str(tmp_path / rel), {
        "seed": 7, "replicate": 2, "arm": "transport_ceiling", "mode": "bdsp",
        "elapsed_seconds": 3891.6,                                    # 1.08h -- matches the incident report
        "started": started_str,
    })
    check_when = time.mktime(time.strptime("2026-09-25 01:38:31", "%Y-%m-%d %H:%M:%S"))
    wt_log = tmp_path / ".claude" / "worktrees" / "wf_3d929cf5-198-1" / "research" / "queue" / ".corpus_checks.jsonl"
    wt_log.parent.mkdir(parents=True)
    wt_log.write_text(json.dumps({
        "when": check_when,
        "query": "BDSP hard weight clamp saturates hidden feedforward weights one-sided LTP drift gap4 "
                 "transport ceiling collapse",
    }) + "\n")

    assert cc_gate.check([rel]) == [], \
        "the gate still blocks the exact gap4 artifact despite the real 01:38 worktree-log evidence"


def test_an_expensive_artifact_with_no_check_anywhere_still_blocks(tmp_path, monkeypatch):
    """POWER CONTROL for the test above: same shape of artifact, but with NEITHER a shared-root log NOR any
    worktree log -- proves the fix is a genuine fallback (checks for real evidence) and not a gate that has
    quietly stopped blocking anything."""
    monkeypatch.setattr(cc_gate, "_ROOT", str(tmp_path))
    rel = "research/findings/raw/gap4/some_other_expensive_run.json"
    run_start = int(time.time()) - 3600
    started_str = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(run_start))
    _write(str(tmp_path / rel), {"elapsed_seconds": 3891.6, "started": started_str})

    problems = cc_gate.check([rel])
    assert problems, "an expensive artifact with NO corpus check anywhere (shared or worktree) must be BLOCKED"
    assert rel in problems[0]


def test_a_check_logged_only_after_the_run_started_does_not_count(tmp_path, monkeypatch):
    """The fallback must respect "BEFORE the run started", not merely "a log exists somewhere" -- a check run
    once the expensive compute was already under way (or after it finished) proves nothing about whether it
    was consulted beforehand."""
    monkeypatch.setattr(cc_gate, "_ROOT", str(tmp_path))
    rel = "research/findings/raw/gap4/after_the_fact_check.json"
    run_start = int(time.time()) - 3600
    started_str = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(run_start))
    _write(str(tmp_path / rel), {"elapsed_seconds": 3891.6, "started": started_str})
    shared_log = tmp_path / "research" / "queue" / ".corpus_checks.jsonl"
    shared_log.parent.mkdir(parents=True)
    shared_log.write_text(json.dumps({"when": run_start + 200, "query": "checked too late"}) + "\n")

    assert cc_gate.check([rel]), "a corpus check logged AFTER this run started was wrongly accepted"


def test_a_direct_in_artifact_stamp_still_short_circuits_the_fallback(tmp_path, monkeypatch):
    """No regression on the ORIGINAL mechanism: an artifact carrying its own `corpus_check_fresh: true` needs
    no shared/worktree log at all."""
    monkeypatch.setattr(cc_gate, "_ROOT", str(tmp_path))
    rel = "research/findings/raw/gap4/directly_stamped.json"
    _write(str(tmp_path / rel), {"elapsed_seconds": 3891.6, "corpus_check_fresh": True})
    assert cc_gate.check([rel]) == []
