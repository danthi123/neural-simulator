"""Tests for tools/gates/corpus_check_required.py (CLASS CC) -- the un-stamped-artifact fallback added in
research/corpus-check-shared-log (2026-09-25).

`corpus_check_fresh` is only ever written by the provenance door for a path named on argv
(`--out`/`--output`/`--json`) or registered via `declare_output()`. A side artifact a runner writes directly
(a checkpoint into `--ckpt-dir`, for instance) gets no sidecar at all, so `fresh` reads `None`: never given
the chance to say, not "checked and stale". This fallback looks for direct evidence that
`tools/before_you_build.sh` ran before that artifact's OWN run started, in the shared log or any surviving
per-worktree legacy log, ONLY for that `fresh is None` case -- an explicit `corpus_check_fresh: false` still
blocks outright with no fallback attempted.

Same convention as tests/test_gate_lb_battery_provenance.py: import the REAL gate module the pre-commit
registry calls, so these tests cannot drift from what actually runs. HERMETIC: every test monkeypatches
`_ROOT` and `_WORKTREE_LOG_GLOB` to isolated tmp_path locations and `SIM_CORPUS_CHECK_LOG` via monkeypatch.setenv
-- none of it reads or writes this real machine's actual `.git`, `research/queue/`, or
`.claude/worktrees/*/research/queue/.corpus_checks.jsonl` (which carry real, live entries that would
otherwise make a "no evidence anywhere" negative control flaky; this was hit and fixed during development --
see the incident test below for the exact real data it reproduces)."""
from __future__ import annotations

import json
import os
import time

import tools.gates.corpus_check_required as cc_gate

REAL_GAP4_ARTIFACT = ("/home/dant123/Projects/sim/research/findings/raw/gap4/transport_ceiling_readout/"
                      "companion_revfeaca2f/C25/ckpt/s7_r2_transport_ceiling.json")
REAL_GAP4_REL = ("research/findings/raw/gap4/transport_ceiling_readout/companion_revfeaca2f/C25/ckpt/"
                 "s7_r2_transport_ceiling.json")
# The REAL sibling run_id for this exact artifact (C25_s7_r2_transport_ceiling.json.prov.json), confirmed --
# not fabricated -- by cross-checking mtimes during this fix's development: run_id 1790334960 == epoch for
# 2026-09-25T07:16:00 local (EDT), and the ckpt file's own real mtime (08:20) sits at
# started(07:16:00) + elapsed_seconds(3891.6s ~= 64.86min) = 08:20:52 -- i.e. the SAME process/run wrote both
# the declared `--out` (sidecarred) and this un-declared checkpoint side-write (not sidecarred).
REAL_GAP4_RUN_ID = "1790334960-2962287"
REAL_GAP4_RUN_START = 1790334960.0


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


def _isolate(monkeypatch, tmp_path):
    """No test here may see this real machine's actual shared/worktree corpus-check logs."""
    monkeypatch.setattr(cc_gate, "_ROOT", str(tmp_path))
    monkeypatch.setattr(cc_gate, "_WORKTREE_LOG_GLOB",
                         str(tmp_path / "no-such-worktrees-dir" / "*" / "nope.jsonl"))
    monkeypatch.setenv("SIM_CORPUS_CHECK_LOG", str(tmp_path / "shared.jsonl"))


def test_incident_gap4_c25_ckpt_artifact_passes_via_the_real_0138_entry(tmp_path, monkeypatch):
    """Reproduces the exact blocked incident against the REAL artifact bytes (read-only copy, never written
    to): a run recording 1.08h with NEITHER `corpus_check_fresh` NOR its own `run_id`/`started_utc_ns` --
    it is a checkpoint side-write that predates being wired through `declare_output()`. Its sidecar (as it
    would carry if a future fix registers it) carries the REAL sibling run's `run_id`. A
    `before_you_build.sh` check at 2026-09-25T01:38:31 -- well before the real 07:16:00 run start -- must
    make this pass.

    Mutation-verify: comment out the `_fallback_evidence` call in `_check_one`, or the `run_id` branch of
    `_run_start_epoch`, and this test fails.
    """
    _isolate(monkeypatch, tmp_path)
    with open(REAL_GAP4_ARTIFACT, encoding="utf-8") as fh:              # READ-ONLY, never written to
        real_obj = json.load(fh)
    assert "corpus_check_fresh" not in real_obj and "run_id" not in real_obj, \
        "the real incident artifact is expected to carry no stamp of its own -- did it change?"
    assert abs(real_obj["elapsed_seconds"] - 3891.6) < 1e-6

    dest = tmp_path / REAL_GAP4_REL
    _write(str(dest), real_obj)
    _write(str(dest) + ".prov.json", {"run_id": REAL_GAP4_RUN_ID})

    check_when = time.mktime(time.strptime("2026-09-25 01:38:31", "%Y-%m-%d %H:%M:%S"))
    assert check_when < REAL_GAP4_RUN_START
    wt_log = (tmp_path / ".claude" / "worktrees" / "wf_3d929cf5-198-1" / "research" / "queue"
              / ".corpus_checks.jsonl")
    wt_log.parent.mkdir(parents=True)
    wt_log.write_text(json.dumps({
        "when": check_when, "iso": "2026-09-25T01:38:31",
        "query": "BDSP hard weight clamp saturates hidden feedforward weights transport ceiling collapse",
    }) + "\n")
    monkeypatch.setattr(cc_gate, "_WORKTREE_LOG_GLOB",
                         str(tmp_path / ".claude" / "worktrees" / "*" / "research" / "queue"
                             / ".corpus_checks.jsonl"))

    assert cc_gate.check([REAL_GAP4_REL]) == [], \
        "the gate still blocks the exact gap4 artifact despite the real 01:38 worktree-log evidence"


def test_incident_gap4_with_no_run_start_source_at_all_fails_closed(tmp_path, monkeypatch):
    """POWER CONTROL for the incident test: the same real artifact bytes, but with no sidecar at all (so no
    run_id/started_utc_ns/started anywhere) -- even a qualifying log entry cannot be applied because there
    is no run-start to compare it against. Proves the fallback does not fall back to mtime."""
    _isolate(monkeypatch, tmp_path)
    with open(REAL_GAP4_ARTIFACT, encoding="utf-8") as fh:
        real_obj = json.load(fh)
    dest = tmp_path / REAL_GAP4_REL
    _write(str(dest), real_obj)
    with open(str(tmp_path / "shared.jsonl"), "w") as fh:
        fh.write(json.dumps({"when": REAL_GAP4_RUN_START - 3600, "query": "would qualify if run-start known"})
                  + "\n")

    problems = cc_gate.check([REAL_GAP4_REL])
    assert problems, "an artifact with NO run-start source anywhere must fail closed, not pass via mtime"


def test_a_check_logged_only_after_the_run_started_does_not_count(tmp_path, monkeypatch):
    """The fallback must respect "BEFORE the run started", not merely "a log exists somewhere" -- a check
    run once the expensive compute was already under way (or after it finished) proves nothing about
    whether it was consulted beforehand."""
    _isolate(monkeypatch, tmp_path)
    rel = "research/findings/raw/gap4/after_the_fact_check.json"
    run_start = int(time.time()) - 3600
    _write(str(tmp_path / rel), {"elapsed_seconds": 3891.6, "run_id": "%d-1" % run_start})
    shared_log = tmp_path / "shared.jsonl"
    shared_log.write_text(json.dumps({"when": run_start + 200, "query": "checked too late"}) + "\n")

    assert cc_gate.check([rel]), "a corpus check logged AFTER this run started was wrongly accepted"


def test_an_expensive_artifact_with_no_check_anywhere_still_blocks(tmp_path, monkeypatch):
    """POWER CONTROL: same shape of artifact, with a resolvable run-start but NEITHER a shared-root log NOR
    any worktree log -- proves the fix is a genuine fallback (checks for real evidence) and not a gate that
    has quietly stopped blocking anything."""
    _isolate(monkeypatch, tmp_path)
    rel = "research/findings/raw/gap4/some_other_expensive_run.json"
    run_start = int(time.time()) - 3600
    _write(str(tmp_path / rel), {"elapsed_seconds": 3891.6, "run_id": "%d-1" % run_start})

    problems = cc_gate.check([rel])
    assert problems, "an expensive artifact with NO corpus check anywhere (shared or worktree) must be BLOCKED"
    assert rel in problems[0]


def test_a_direct_in_artifact_stamp_still_short_circuits_the_fallback(tmp_path, monkeypatch):
    """No regression on the ORIGINAL mechanism: an artifact carrying its own `corpus_check_fresh: true` needs
    no shared/worktree log, no run_id, nothing else."""
    _isolate(monkeypatch, tmp_path)
    rel = "research/findings/raw/gap4/directly_stamped.json"
    _write(str(tmp_path / rel), {"elapsed_seconds": 3891.6, "corpus_check_fresh": True})
    assert cc_gate.check([rel]) == []


def test_explicit_false_stamp_gets_no_fallback_chance(tmp_path, monkeypatch):
    """`fresh is None` (no stamp) and `fresh is False` (an explicit stale stamp) are DIFFERENT: only the
    former tries the fallback. An artifact that COULD carry a stamp and carries a stale one must not be
    rescued by fallback evidence -- otherwise the fallback would silently defeat the original, direct
    mechanism for every runner that already stamps its own outputs."""
    _isolate(monkeypatch, tmp_path)
    rel = "research/findings/raw/gap4/explicitly_stale.json"
    run_start = int(time.time()) - 3600
    _write(str(tmp_path / rel), {
        "elapsed_seconds": 3891.6, "corpus_check_fresh": False, "run_id": "%d-1" % run_start,
    })
    shared_log = tmp_path / "shared.jsonl"
    shared_log.write_text(json.dumps({"when": run_start - 60, "query": "qualifies, but must not be reached"})
                           + "\n")
    assert cc_gate.check([rel]), \
        "an explicit corpus_check_fresh:false must block outright, with no fallback attempted"


def test_naive_started_string_without_timezone_is_not_used_as_run_start(tmp_path, monkeypatch):
    """A `started` string with no timezone (what every v1 record currently writes, via
    `time.strftime(..., time.localtime(x))`) must NOT be trusted as a run-start source -- an AWS/hostless
    node's wall clock is UTC but the string never says so, and re-parsing it as this machine's local zone
    read 4h late during this fix's own development. With no OTHER source available, this must fail closed."""
    _isolate(monkeypatch, tmp_path)
    rel = "research/findings/raw/gap4/naive_started_only.json"
    run_start = int(time.time()) - 3600
    started_str = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(run_start))
    _write(str(tmp_path / rel), {"elapsed_seconds": 3891.6, "started": started_str})
    shared_log = tmp_path / "shared.jsonl"
    shared_log.write_text(json.dumps({"when": run_start - 60, "query": "would qualify if trusted"}) + "\n")

    assert cc_gate.check([rel]), \
        "a naive (no-timezone) 'started' string was wrongly accepted as a run-start source"


def test_started_string_with_timezone_is_used(tmp_path, monkeypatch):
    """The SAME field, but carrying an explicit UTC offset, IS trusted."""
    _isolate(monkeypatch, tmp_path)
    rel = "research/findings/raw/gap4/tz_started.json"
    run_start = int(time.time()) - 3600
    started_str = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(run_start))
    _write(str(tmp_path / rel), {"elapsed_seconds": 3891.6, "started": started_str})
    shared_log = tmp_path / "shared.jsonl"
    shared_log.write_text(json.dumps({"when": run_start - 60, "query": "qualifies"}) + "\n")

    assert cc_gate.check([rel]) == []


def test_evidence_outside_the_freshness_window_does_not_count(tmp_path, monkeypatch):
    _isolate(monkeypatch, tmp_path)
    rel = "research/findings/raw/gap4/stale_evidence.json"
    run_start = int(time.time())
    _write(str(tmp_path / rel), {"elapsed_seconds": 3891.6, "run_id": "%d-1" % run_start})
    shared_log = tmp_path / "shared.jsonl"
    too_old = run_start - cc_gate.FRESHNESS_WINDOW_S - 3600
    shared_log.write_text(json.dumps({"when": too_old, "query": "way too old"}) + "\n")

    assert cc_gate.check([rel]), "evidence older than the freshness window was wrongly accepted"
