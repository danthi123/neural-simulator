"""Tests for research/runners/__init__.py's `_shared_corpus_check_log` / `_corpus_check_state`
(research/corpus-check-shared-log, 2026-09-25): the corpus-check evidence a run's provenance sidecar carries
now comes from the ONE log `tools/before_you_build.sh` writes at the git COMMON dir root (shared across every
worktree of this repo), and its age is measured against the run's OWN `_START` rather than wall-clock time at
sidecar-write (exit).

HERMETIC: `SIM_CORPUS_CHECK_LOG` (monkeypatch.setenv) points every test but one at an isolated tmp file, never
this real machine's `.git` or `research/queue/.corpus_checks.jsonl`. The one exception (git-common-dir
resolution) uses a freshly `git init`-ed throwaway repo, never the real project's.
"""
from __future__ import annotations

import json
import os
import subprocess
import time

import research.runners as provenance


def _write_log(path, entries):
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    with open(path, "w") as fh:
        for e in entries:
            fh.write(json.dumps(e) + "\n")


def test_override_env_var_wins_over_git_resolution(monkeypatch, tmp_path):
    override = str(tmp_path / "custom.jsonl")
    monkeypatch.setenv("SIM_CORPUS_CHECK_LOG", override)
    assert provenance._shared_corpus_check_log() == override


def test_shared_log_path_resolves_via_git_common_dir_when_no_override(monkeypatch, tmp_path):
    """No override: falls back to `git -C _ROOT rev-parse --git-common-dir`. Uses a FRESH throwaway `git
    init` repo, never this project's real `.git`."""
    monkeypatch.delenv("SIM_CORPUS_CHECK_LOG", raising=False)
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    monkeypatch.setattr(provenance, "_ROOT", str(tmp_path))
    resolved = provenance._shared_corpus_check_log()
    assert resolved == os.path.join(str(tmp_path), ".git", "corpus_checks_shared.jsonl")


def test_git_common_dir_is_shared_across_two_worktrees(monkeypatch, tmp_path):
    """The core property this fix exists for: two DIFFERENT checkouts of the SAME repo must resolve the
    SAME shared-log path."""
    main = tmp_path / "main"
    main.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=main, check=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=main, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=main, check=True)
    (main / "f.txt").write_text("x")
    subprocess.run(["git", "add", "f.txt"], cwd=main, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=main, check=True)
    wt = tmp_path / "wt"
    subprocess.run(["git", "worktree", "add", "-q", str(wt), "-b", "side"], cwd=main, check=True)

    monkeypatch.delenv("SIM_CORPUS_CHECK_LOG", raising=False)
    monkeypatch.setattr(provenance, "_ROOT", str(main))
    from_main = provenance._shared_corpus_check_log()
    monkeypatch.setattr(provenance, "_ROOT", str(wt))
    from_wt = provenance._shared_corpus_check_log()

    assert from_main == from_wt, "the main checkout and its worktree resolved DIFFERENT shared-log paths"
    assert from_main == os.path.join(str(main), ".git", "corpus_checks_shared.jsonl")


def test_age_measured_at_run_start_not_at_call_time(monkeypatch, tmp_path):
    """Mutation-verify: replace `_START` with `time.time()` in `_corpus_check_state` and this test fails,
    because the second call (after the fake wall-clock jump) would then read a different age."""
    log = tmp_path / "log.jsonl"
    _write_log(log, [{"when": 1000.0, "query": "checked well before start"}])
    monkeypatch.setenv("SIM_CORPUS_CHECK_LOG", str(log))
    monkeypatch.setattr(provenance, "_START", 1100.0)

    state = provenance._corpus_check_state()
    assert state["corpus_check_age_s"] == 100.0
    assert state["corpus_check_fresh"] is True

    real_time = time.time
    monkeypatch.setattr(time, "time", lambda: real_time() + 999999)
    state2 = provenance._corpus_check_state()
    assert state2["corpus_check_age_s"] == 100.0, \
        "age changed after the wall clock moved -- _corpus_check_state is reading time.time(), not _START"


def test_entry_after_run_start_is_excluded(monkeypatch, tmp_path):
    log = tmp_path / "log.jsonl"
    _write_log(log, [{"when": 1500.0, "query": "logged after this run started"}])
    monkeypatch.setenv("SIM_CORPUS_CHECK_LOG", str(log))
    monkeypatch.setattr(provenance, "_START", 1100.0)

    state = provenance._corpus_check_state()
    assert state["corpus_check_age_s"] is None
    assert state.get("corpus_check_query") is None


def test_newest_qualifying_entry_before_start_wins(monkeypatch, tmp_path):
    log = tmp_path / "log.jsonl"
    _write_log(log, [
        {"when": 500.0, "query": "oldest"},
        {"when": 900.0, "query": "newest before start"},
        {"when": 1200.0, "query": "after start, excluded"},
    ])
    monkeypatch.setenv("SIM_CORPUS_CHECK_LOG", str(log))
    monkeypatch.setattr(provenance, "_START", 1100.0)

    state = provenance._corpus_check_state()
    assert state["corpus_check_query"] == "newest before start"
    assert state["corpus_check_age_s"] == 200.0


def test_stale_beyond_default_window_is_not_fresh(monkeypatch, tmp_path):
    log = tmp_path / "log.jsonl"
    _write_log(log, [{"when": 0.0, "query": "ancient"}])
    monkeypatch.setenv("SIM_CORPUS_CHECK_LOG", str(log))
    monkeypatch.setattr(provenance, "_START", 100000.0)   # far more than 24h after `when`

    state = provenance._corpus_check_state()
    assert state["corpus_check_fresh"] is False


def test_missing_log_returns_none_state(monkeypatch, tmp_path):
    monkeypatch.setenv("SIM_CORPUS_CHECK_LOG", str(tmp_path / "does_not_exist.jsonl"))
    monkeypatch.setattr(provenance, "_START", 100.0)

    state = provenance._corpus_check_state()
    assert state == {"corpus_check_age_s": None, "corpus_check_query": None}
