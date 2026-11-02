"""The lean gate set (2026-11-02): which gates run, that each can fail, and the new checks' key cases."""
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import tools.gates as gates  # noqa: E402
import tools.gates.brain_prereq as bp  # noqa: E402
import tools.gates.corpus_check_required as cc  # noqa: E402
import tools.gates.secret_scan as sec  # noqa: E402

KEPT = {"artifact-provenance", "prereg_before_run", "prereg_amendment_order", "single-seed", "host_decision_declared",
        "workflow-cost-tiering", "corpus-check-required", "verdict-preconditions", "terminology", "brain-prereq",
        "secret-scan"}


def test_registry_runs_exactly_the_lean_set_and_never_the_archive():
    found = gates.discover()
    assert all(err is None for _, _, err in found), found
    assert {n for n, _, _ in found} == KEPT


def test_every_kept_gate_selftest_is_clean():
    for name, mod, _ in gates.discover():
        assert mod.selftest() == [], name


def test_archived_gates_stay_importable():
    from tools.gates._archive import lane_starvation, flip_smoke_first  # noqa: F401


def test_brain_prereq_grandfathers_brain_v2_files_but_not_new_ones():
    assert not bp._in_scope("brain/experiments/reinstate_probe.py")
    assert bp._in_scope("brain/experiments/sleep_probe.py")
    assert not bp._in_scope("brain/experiments/results/x.txt")


def test_brain_prereq_requires_a_source_per_capability():
    assert bp.check_prereq_doc("## Sleep\nReadiness: NOT-READY\n")
    assert not bp.check_prereq_doc("## Sleep\n- CAT N.12 sleep consolidation\nReadiness: NOT-READY\n")
    assert not bp.check_prereq_doc("## Sleep\n- doi 10.1038/nn.1234\nReadiness: PARTIAL\n")


def test_record_check_needed_only_for_new_non_data_files(tmp_path, monkeypatch):
    log = tmp_path / "log.jsonl"
    monkeypatch.setenv("SIM_CORPUS_CHECK_LOG", str(log))
    monkeypatch.setattr(cc, "_WORKTREE_LOG_GLOB", str(tmp_path / "none" / "*.jsonl"))
    now = 2_000_000_000.0
    assert cc.check_new_files(["research/findings/2026-11-02-a.md"], now=now)
    assert not cc.check_new_files(["research/findings/raw/a.json", "tools/x.py"], now=now)
    log.write_text(json.dumps({"when": now - 60, "query": "q"}) + "\n")
    assert not cc.check_new_files(["research/findings/2026-11-02-a.md"], now=now)


def test_secret_scan_catches_a_key_in_a_given_file(tmp_path):
    f = tmp_path / "cfg.env"
    f.write_text("OPENROUTER_API_KEY=" + "sk-or-v1-" + "a1" * 20 + "\n")
    assert any("openrouter" in p for p in sec.check([str(f)]))
