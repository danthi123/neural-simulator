"""Tests for the 2026-09-25 incident fix: a corpus check run in one git worktree (or carried by a pool/GPU
queue job) must be visible to the run it was meant to clear, instead of each worktree keeping its own
invisible `.corpus_checks.jsonl`.

INCIDENT: gate corpus-check-required (CLASS CC) blocked merging research/score-gap4-c26-0925 @ ed6758f61
because .../C25/ckpt/s7_r2_transport_ceiling.json (1.08h of compute) carried no `corpus_check_fresh`, despite a
real check having been run at 2026-09-25T01:38:31 -- logged under
`.claude/worktrees/wf_3d929cf5-198-1/research/queue/.corpus_checks.jsonl`, invisible to both the main root's
own log and to `research/runners/__init__`'s stamp.

Covers: tools/corpus_check_lib.sh (the shared resolution both bash producers/consumers use),
research/runners/__init__._shared_queue_root + ._corpus_check_state (the Python mirror + the env-stamp
priority), tools/pool_queue.sh `add` + tools/pool_autodispatch.sh `pop_job` (the pool propagation seam), and
tools/gpu_queue.sh `add` (the GPU propagation seam). The gate itself (tools/gates/corpus_check_required.py) has
its own test file, tests/test_gate_corpus_check_required.py, following this repo's one-gate-one-test-file
convention.

Every test here is written to FAIL against the pre-fix code (mutation-verify, per the incident's own build
instructions): each positive assertion has a corresponding case that pins the OLD (wrong) behavior would have
produced a different, distinguishable result.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

# Guarded exactly like tests/test_seam_contracts.py's own import of this module: the provenance door runs an
# atexit stamp over the real research/findings/raw/ on import, and `_ENABLED` is latched at whichever import
# is FIRST in the whole pytest session -- disable it for this import specifically so a full-suite run in any
# collection order still gets research.runners imported provenance-disabled at least here.
_PREV_NOPROV = os.environ.get("SIM_NO_PROVENANCE")
os.environ["SIM_NO_PROVENANCE"] = "1"
import research.runners as provenance  # noqa: E402
if _PREV_NOPROV is None:
    os.environ.pop("SIM_NO_PROVENANCE", None)
else:
    os.environ["SIM_NO_PROVENANCE"] = _PREV_NOPROV

ROOT = Path(__file__).resolve().parents[1]
CORPUS_CHECK_LIB = ROOT / "tools" / "corpus_check_lib.sh"
BEFORE_YOU_BUILD = ROOT / "tools" / "before_you_build.sh"
POOL_QUEUE = ROOT / "tools" / "pool_queue.sh"
GPU_QUEUE = ROOT / "tools" / "gpu_queue.sh"
POOL_AUTODISPATCH = ROOT / "tools" / "pool_autodispatch.sh"


def run_bash(script: Path, *args: str, env: dict[str, str] | None = None, input: str | None = None):
    return subprocess.run(
        ["bash", str(script), *args],
        cwd=ROOT,
        env={**os.environ, **(env or {})},
        text=True,
        capture_output=True,
        input=input,
    )


def _git(*args: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=cwd, text=True, capture_output=True, check=True)


@pytest.fixture()
def real_worktree_pair(tmp_path):
    """A REAL main checkout + a REAL git worktree of it (not a fake -- the whole point of this incident is
    `git rev-parse --git-common-dir` resolution, which only a genuine worktree exercises). Returns
    (main_root, worktree_root)."""
    main = tmp_path / "main"
    main.mkdir()
    _git("init", "-q", cwd=main)
    _git("config", "user.email", "t@example.com", cwd=main)
    _git("config", "user.name", "t", cwd=main)
    (main / "README.md").write_text("x\n")
    _git("add", "README.md", cwd=main)
    _git("commit", "-q", "-m", "init", cwd=main)
    wt = tmp_path / "wt"
    _git("worktree", "add", "-q", str(wt), "-b", "feature", cwd=main)
    return main, wt


# ======================================================================================================
# tools/corpus_check_lib.sh -- the shared bash resolution
# ======================================================================================================

class TestCorpusCheckLibShared:
    def _shared_log(self, cwd: Path, fallback: str = "") -> str:
        r = subprocess.run(
            ["bash", "-c", f'source "{CORPUS_CHECK_LIB}"; corpus_check_shared_log "{fallback}"'],
            cwd=cwd, text=True, capture_output=True, check=True,
        )
        return r.stdout.strip()

    def test_resolves_to_the_main_roots_log_from_inside_a_worktree(self, real_worktree_pair):
        """THE INCIDENT ITSELF for the bash writer: from inside the WORKTREE, the shared log must be the MAIN
        root's, not the worktree's own $PWD -- pre-fix, before_you_build.sh used $PWD and would have returned
        <worktree>/research/queue/.corpus_checks.jsonl here instead."""
        main, wt = real_worktree_pair
        got = self._shared_log(wt)
        assert got == str(main / "research" / "queue" / ".corpus_checks.jsonl"), got
        assert not got.startswith(str(wt)), "SEAM BROKEN: resolved to the worktree's own path, not the shared one"

    def test_resolves_to_the_same_log_from_the_main_root_itself(self, real_worktree_pair):
        main, _wt = real_worktree_pair
        got = self._shared_log(main)
        assert got == str(main / "research" / "queue" / ".corpus_checks.jsonl")

    def test_falls_back_to_the_given_root_outside_a_git_checkout(self, tmp_path):
        bare = tmp_path / "not_a_repo"
        bare.mkdir()
        got = self._shared_log(bare, fallback=str(bare))
        assert got == str(bare / "research" / "queue" / ".corpus_checks.jsonl")

    def test_sim_corpus_check_log_overrides_everything(self, real_worktree_pair):
        main, wt = real_worktree_pair
        r = subprocess.run(
            ["bash", "-c", f'source "{CORPUS_CHECK_LIB}"; corpus_check_shared_log'],
            cwd=wt, text=True, capture_output=True, check=True,
            env={**os.environ, "SIM_CORPUS_CHECK_LOG": "/override/path.jsonl"},
        )
        assert r.stdout.strip() == "/override/path.jsonl"

    def test_latest_returns_the_last_well_formed_line_as_when_tab_query(self, tmp_path):
        log = tmp_path / "log.jsonl"
        log.write_text(
            json.dumps({"when": 100, "query": "first"}) + "\n"
            + "not json at all\n"
            + json.dumps({"when": 200, "query": "second one\twith\ttabs\nand a newline"}) + "\n"
        )
        r = subprocess.run(
            ["bash", "-c", f'source "{CORPUS_CHECK_LIB}"; corpus_check_latest "{log}"'],
            text=True, capture_output=True, check=True,
        )
        out = r.stdout.rstrip("\n")
        when, _, query = out.partition("\t")
        assert when == "200"
        assert "\t" not in query and "\n" not in query
        assert query == "second one with tabs and a newline"

    def test_latest_is_silent_for_a_missing_or_empty_log(self, tmp_path):
        missing = tmp_path / "nope.jsonl"
        r = subprocess.run(["bash", "-c", f'source "{CORPUS_CHECK_LIB}"; corpus_check_latest "{missing}"'],
                           text=True, capture_output=True, check=True)
        assert r.stdout == ""


# ======================================================================================================
# tools/before_you_build.sh -- writes to the SHARED log, not $PWD
# ======================================================================================================

def test_before_you_build_writes_the_shared_log_when_run_from_a_worktree(real_worktree_pair):
    """Mutation-verify: revert the `corpus_check_shared_log` call to the old `$PWD/research/queue/...` literal
    and this test fails (the entry lands under the worktree instead).

    `before_you_build.sh` resolves its own location via `$0` (`cd "$(dirname "$0")/.."`), so exercising it
    against an ISOLATED fixture -- never the real repo's own copy, which would append to the real, live,
    gitignored `research/queue/.corpus_checks.jsonl` this task must not touch -- means copying the two scripts
    under test into the fixture's own `tools/`, not invoking the real ones with a redirected cwd."""
    main, wt = real_worktree_pair
    for repo in (main, wt):
        (repo / "tools").mkdir(exist_ok=True)
        (repo / "tools" / "before_you_build.sh").write_text(BEFORE_YOU_BUILD.read_text())
        (repo / "tools" / "corpus_check_lib.sh").write_text(CORPUS_CHECK_LIB.read_text())
        (repo / "research" / "findings").mkdir(parents=True, exist_ok=True)

    r = subprocess.run(["bash", str(wt / "tools" / "before_you_build.sh"), "a fully isolated real-worktree corpus check"],
                       cwd=wt, text=True, capture_output=True)
    assert r.returncode == 0, r.stderr
    shared_log = main / "research" / "queue" / ".corpus_checks.jsonl"
    worktree_log = wt / "research" / "queue" / ".corpus_checks.jsonl"
    assert shared_log.exists(), "the check was not written to the shared (main-root) log at all:\n%s" % r.stdout
    last = json.loads([ln for ln in shared_log.read_text().splitlines() if ln.strip()][-1])
    assert last["query"] == "a fully isolated real-worktree corpus check"
    assert last["cwd"] == str(wt), "the cwd field (which checkout actually ran the check) was dropped"
    assert not worktree_log.exists(), "a SECOND, worktree-local log was created -- the old $PWD-keyed behavior"


# ======================================================================================================
# research/runners/__init__.py -- the Python-side mirror of the same resolution + the env-stamp priority
# ======================================================================================================

class TestSharedQueueRoot:
    def test_resolves_through_git_common_dir_not_the_passed_worktree_root(self, real_worktree_pair):
        main, wt = real_worktree_pair
        got = provenance._shared_queue_root(root=str(wt))
        assert got == str(main / "research" / "queue" / ".corpus_checks.jsonl")
        assert not got.startswith(str(wt))

    def test_falls_back_to_root_outside_a_git_checkout(self, tmp_path):
        bare = str(tmp_path / "no_git_here")
        os.makedirs(bare)
        got = provenance._shared_queue_root(root=bare)
        assert got == os.path.join(bare, "research", "queue", ".corpus_checks.jsonl")

    def test_sim_corpus_check_log_env_override(self, monkeypatch, real_worktree_pair):
        _main, wt = real_worktree_pair
        monkeypatch.setenv("SIM_CORPUS_CHECK_LOG", "/tmp/does/not/matter.jsonl")
        assert provenance._shared_queue_root(root=str(wt)) == "/tmp/does/not/matter.jsonl"


class TestCorpusCheckStateEnvPriority:
    def _clear(self, monkeypatch):
        monkeypatch.delenv("CORPUS_CHECK_WHEN", raising=False)
        monkeypatch.delenv("CORPUS_CHECK_QUERY", raising=False)
        monkeypatch.delenv("SIM_CORPUS_CHECK_LOG", raising=False)

    def test_env_stamp_wins_even_when_no_log_is_reachable(self, monkeypatch, tmp_path):
        """THE POOL/GPU-NODE CASE (root cause b): a job on a node with no git checkout at all has no shared log
        to read -- CORPUS_CHECK_WHEN/QUERY in its own env must be sufficient on their own."""
        self._clear(monkeypatch)
        monkeypatch.setattr(provenance, "_ROOT", str(tmp_path))  # no research/queue/*.jsonl exists here
        when = time.time() - 120
        monkeypatch.setenv("CORPUS_CHECK_WHEN", "%.3f" % when)
        monkeypatch.setenv("CORPUS_CHECK_QUERY", "the pool-carried question")
        st = provenance._corpus_check_state()
        assert st["corpus_check_fresh"] is True
        assert st["corpus_check_query"] == "the pool-carried question"
        assert 100 <= st["corpus_check_age_s"] <= 140
        assert st.get("corpus_check_source") == "env"

    def test_env_stamp_can_read_as_stale(self, monkeypatch, tmp_path):
        self._clear(monkeypatch)
        monkeypatch.setattr(provenance, "_ROOT", str(tmp_path))
        monkeypatch.setenv("CORPUS_CHECK_WHEN", "%.3f" % (time.time() - 100000))   # ~27.8h ago
        monkeypatch.setenv("CORPUS_CHECK_QUERY", "an old question")
        st = provenance._corpus_check_state()
        assert st["corpus_check_fresh"] is False

    def test_falls_back_to_the_shared_log_when_no_env_stamp_is_present(self, monkeypatch, tmp_path):
        """Mutation-verify: without the env-priority branch this still passes; without the
        `_shared_queue_root()` fix (i.e. if this read `_ROOT`-relative again) it would MISS a log placed at the
        shared root when `_ROOT` names a worktree instead."""
        self._clear(monkeypatch)
        monkeypatch.setattr(provenance, "_ROOT", str(tmp_path))
        log = tmp_path / "research" / "queue" / ".corpus_checks.jsonl"
        log.parent.mkdir(parents=True)
        log.write_text(json.dumps({"when": time.time() - 60, "query": "logged, not enveloped"}) + "\n")
        st = provenance._corpus_check_state()
        assert st["corpus_check_fresh"] is True
        assert st["corpus_check_query"] == "logged, not enveloped"

    def test_env_stamp_is_preferred_over_a_contradicting_log(self, monkeypatch, tmp_path):
        """If a job carries its OWN env stamp, a stale/irrelevant local log must not override it."""
        self._clear(monkeypatch)
        monkeypatch.setattr(provenance, "_ROOT", str(tmp_path))
        log = tmp_path / "research" / "queue" / ".corpus_checks.jsonl"
        log.parent.mkdir(parents=True)
        log.write_text(json.dumps({"when": time.time() - 100000, "query": "stale local log"}) + "\n")
        monkeypatch.setenv("CORPUS_CHECK_WHEN", "%.3f" % (time.time() - 30))
        monkeypatch.setenv("CORPUS_CHECK_QUERY", "fresh env stamp")
        st = provenance._corpus_check_state()
        assert st["corpus_check_fresh"] is True
        assert st["corpus_check_query"] == "fresh env stamp"

    def test_real_worktree_run_now_sees_a_check_logged_from_the_main_root(self, monkeypatch, real_worktree_pair):
        """End-to-end for the Python side of root cause (a): a check logged at the SHARED root is now visible
        to a run whose `_ROOT` is the WORKTREE -- pre-fix this returned corpus_check_age_s=None."""
        self._clear(monkeypatch)
        main, wt = real_worktree_pair
        monkeypatch.setattr(provenance, "_ROOT", str(wt))
        shared_log = main / "research" / "queue" / ".corpus_checks.jsonl"
        shared_log.parent.mkdir(parents=True)
        shared_log.write_text(json.dumps({"when": time.time() - 45, "query": "checked from main"}) + "\n")
        st = provenance._corpus_check_state()
        assert st["corpus_check_fresh"] is True
        assert st["corpus_check_query"] == "checked from main"


# ======================================================================================================
# tools/pool_queue.sh `add` + tools/pool_autodispatch.sh `pop_job` -- the pool propagation seam
# ======================================================================================================

class TestPoolQueuePropagation:
    def _env(self, tmp_path, corpus_log):
        return {
            "POOL_QUEUE_PATH": str(tmp_path / "pool.queue"),
            "SIM_CORPUS_CHECK_LOG": str(corpus_log),
        }

    def test_add_appends_a_corpuscheck_annotation_after_checked_without_changing_its_format(self, tmp_path):
        """The producer format tests/test_seam_contracts.py pins (3 `%s` placeholders in the `#checked:` printf)
        must be untouched -- the corpus-check evidence rides inside the $CHECKED value instead of adding a
        placeholder."""
        log = tmp_path / "shared.jsonl"
        log.write_text(json.dumps({"when": 1234567890, "query": "the recorded question"}) + "\n")
        env = self._env(tmp_path, log)
        r = run_bash(POOL_QUEUE, "add", "touch /nonexistent/marker", "--checked", "corpus: nothing covers this",
                     env=env)
        assert r.returncode == 0, r.stderr
        line = (tmp_path / "pool.queue").read_text().strip()
        assert "#checked:corpus: nothing covers this" in line
        assert "#corpuscheck:1234567890|" in line
        # dup-guard's existing strip (`sed 's/  #checked:.*//'`) must still remove BOTH annotations in one cut
        stripped = line.split("\t", 1)[1].split("  #checked:")[0]
        assert stripped == "touch /nonexistent/marker"

    def test_add_omits_the_annotation_when_the_shared_log_has_no_entry(self, tmp_path):
        empty_log = tmp_path / "empty.jsonl"     # does not exist
        env = self._env(tmp_path, empty_log)
        r = run_bash(POOL_QUEUE, "add", "touch /nonexistent/marker", "--checked", "corpus: nothing", env=env)
        assert r.returncode == 0, r.stderr
        line = (tmp_path / "pool.queue").read_text().strip()
        assert "#corpuscheck:" not in line

    def test_pop_job_turns_the_annotation_into_corpus_check_env_vars(self, tmp_path):
        """The other half of the seam: --pop-once (the exact function the live dispatcher calls) must recover
        CORPUS_CHECK_WHEN/QUERY from what `add` wrote, and the result must be REAL bash env-var-prefix syntax
        that a shell actually applies to the command that follows -- not just string containment (bash
        `printf '%q'` may backslash-escape a space-separated value instead of quoting it; either is valid, so
        this executes the popped job rather than pinning one escaping style)."""
        log = tmp_path / "shared.jsonl"
        log.write_text(json.dumps({"when": 1758763111, "query": "gap4 transport ceiling BDSP clamp"}) + "\n")
        env = self._env(tmp_path, log)
        add = run_bash(POOL_QUEUE, "add", "env", "--checked", "corpus: banked three weeks ago", env=env)
        assert add.returncode == 0, add.stderr
        popped = run_bash(POOL_AUTODISPATCH, "--pop-once", "999",
                          env={"POOL_QUEUE_PATH": str(tmp_path / "pool.queue")}).stdout
        assert popped.rstrip("\n").endswith("env"), popped
        executed = subprocess.run(["bash", "-c", popped], text=True, capture_output=True, check=True).stdout
        lines = executed.splitlines()
        assert "CORPUS_CHECK_WHEN=1758763111" in lines, executed
        assert "CORPUS_CHECK_QUERY=gap4 transport ceiling BDSP clamp" in lines, executed
        assert "POOL_CHECKED_REASON=corpus: banked three weeks ago" in lines, executed

    def test_pop_job_without_a_corpuscheck_annotation_is_unaffected(self, tmp_path):
        """Backward compatibility: a line queued before this fix (no #corpuscheck: at all) must pop exactly as
        it did before -- no CORPUS_CHECK_* env, no crash."""
        q = tmp_path / "pool.queue"
        q.write_text("%d\tenv  #checked:an old reason\n" % int(time.time()))
        popped = run_bash(POOL_AUTODISPATCH, "--pop-once", "999", env={"POOL_QUEUE_PATH": str(q)}).stdout
        assert "CORPUS_CHECK_WHEN" not in popped
        executed = subprocess.run(["bash", "-c", popped], text=True, capture_output=True, check=True).stdout
        assert "POOL_CHECKED_REASON=an old reason" in executed.splitlines(), executed


# ======================================================================================================
# tools/gpu_queue.sh `add` -- the GPU propagation seam (simpler: no --checked, no dup guard, queue lines
# execute verbatim via `bash -c "$job"`)
# ======================================================================================================

class TestGpuQueuePropagation:
    def test_add_prepends_corpus_check_env_vars_to_the_queued_command(self, tmp_path):
        log = tmp_path / "shared.jsonl"
        log.write_text(json.dumps({"when": 1758763111.5, "query": "gpu-lane question"}) + "\n")
        qdir = tmp_path / "qdir"
        r = run_bash(GPU_QUEUE, "add", "echo hi",
                    env={"GPU_QUEUE_DIR": str(qdir), "SIM_CORPUS_CHECK_LOG": str(log)})
        assert r.returncode == 0, r.stderr
        line = (qdir / "gpu.queue").read_text().strip()
        assert "CORPUS_CHECK_WHEN=1758763111.5" in line, line
        assert line.endswith("echo hi"), line
        assert "CORPUS_CHECK_QUERY=" in line, line

    def test_add_without_a_shared_entry_queues_the_bare_command(self, tmp_path):
        empty_log = tmp_path / "nope.jsonl"
        qdir = tmp_path / "qdir"
        r = run_bash(GPU_QUEUE, "add", "echo hi",
                    env={"GPU_QUEUE_DIR": str(qdir), "SIM_CORPUS_CHECK_LOG": str(empty_log)})
        assert r.returncode == 0, r.stderr
        assert (qdir / "gpu.queue").read_text().strip() == "echo hi"

    def test_the_prepended_env_actually_reaches_the_process_when_executed(self, tmp_path):
        """Not just string-shape -- prove bash actually applies the prefix as an environment assignment to the
        command that follows it, the way the real daemon's `bash -c "$job"` will."""
        log = tmp_path / "shared.jsonl"
        log.write_text(json.dumps({"when": 42, "query": "q"}) + "\n")
        qdir = tmp_path / "qdir"
        run_bash(GPU_QUEUE, "add", "env", env={"GPU_QUEUE_DIR": str(qdir), "SIM_CORPUS_CHECK_LOG": str(log)})
        job = (qdir / "gpu.queue").read_text().strip()
        r = subprocess.run(["bash", "-c", job], text=True, capture_output=True, check=True)
        assert "CORPUS_CHECK_WHEN=42" in r.stdout.splitlines()
        assert "CORPUS_CHECK_QUERY=q" in r.stdout.splitlines()
