"""tools/queue_job_shape_check.sh -- the enqueue-time "would this even start?" gate shared by
tools/pool_queue.sh add, tools/gpu_queue.sh add and tools/queue_add.sh.

WHY (2026-09-25). Six SETTLE A2 pool lines were queued as
  'A2 wiring seed 42: mem_gb=8 && cd ~/derisk-pool/revisions/... && ...'
-- a prose label as the first word -- so the node ran `A2`, got "command not found" (rc=127), and nothing
ran, while research/queue/pool.queue.claims and the board both said the jobs were dispatched
(research/FAILURE_LOG.md's 2026-09-25 SETTLE A2 row, previously marked NOT-GATEABLE). Neither of
pool_queue.sh's existing checks catches this: both key off finding `-m research.runners.X` ANYWHERE in the
line and validate THAT module -- a line whose first word is prose/garbage but that mentions a real module
further down (exactly the A2 shape) sails through every one of them.

The fixtures in tests/fixtures/queue_job_shapes/ are a deduplicated (by first two whitespace tokens) sample
of the REAL, historical research/queue/pool.queue.claims and research/queue/gpu_queue.log content on this
machine (see tools/queue_job_shape_check.sh's own header for how they were produced) -- a full replay of
both files in full (2028 + 607 lines respectively, at the time this was written) refused only: the 9 known-bad
lines represented here (6 SETTLE A2 lines collapse to one representative "A2 wiring" shape after dedup; 3
independently-torn lines; one bare `status` job that is ALSO a confirmed real historical rc=127 in
gpu_queue.log, 2026-08-31/09-01, three separate cycles, completely unflagged at the time) and zero others.
"""
from __future__ import annotations

import base64
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECK = ROOT / "tools" / "queue_job_shape_check.sh"
POOL_QUEUE = ROOT / "tools" / "pool_queue.sh"
GPU_QUEUE = ROOT / "tools" / "gpu_queue.sh"
QUEUE_ADD = ROOT / "tools" / "queue_add.sh"
FIXTURES = ROOT / "tests" / "fixtures" / "queue_job_shapes"


def run_check(job: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["bash", str(CHECK), job], cwd=ROOT, capture_output=True, text=True, timeout=15)


def _lines(name: str) -> list[str]:
    text = (FIXTURES / name).read_text(encoding="utf-8")
    return [ln for ln in text.splitlines() if ln.strip()]


# --------------------------------------------------------------------------------------------- unit cases

def test_the_a2_shape_is_refused():
    # The exact defect: a prose label glued onto an otherwise-correct pinned-revision job.
    job = ("A2 wiring seed 42: mem_gb=8 && cd ~/derisk-pool/revisions/5b5ea1b7447b24d0978189fb53557a57b88952d4 "
           "&& env SIM_BACKEND=numpy CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 .venv/bin/python -u -m "
           "research.runners._affect_marker_settle_congruence --run-wiring --seeds 42")
    res = run_check(job)
    assert res.returncode == 1
    assert "REFUSED" in res.stderr
    assert "'A2'" in res.stderr


def test_the_historical_status_job_is_refused():
    # research/queue/gpu_queue.log, 2026-08-31/09-01: a queued job that was the single word `status`
    # (almost certainly `gpu_queue.sh add status`, meant to run `gpu_queue.sh status`) ran as
    # `bash: line 1: status: command not found`, rc=127, in under a second, three separate cycles.
    res = run_check("status")
    assert res.returncode == 1
    assert "'status'" in res.stderr


def test_a_syntax_broken_line_is_refused():
    res = run_check("cd ~/x && echo 'unbalanced")
    assert res.returncode == 1
    assert "syntax" in res.stderr


def test_a_torn_line_missing_its_head_is_refused():
    # A REAL historical torn line from research/queue/pool.queue.claims (a fragment of a much longer
    # `... load_bearing_fraction --only curiosity-followup ...` command, missing everything before
    # "earing_fraction").
    res = run_check("earing_fraction --only curiosity-followup --seed 42 --repeats 2 --out x/lb.json")
    assert res.returncode == 1


def test_empty_and_whitespace_only_are_refused():
    assert run_check("").returncode == 1
    assert run_check("   ").returncode == 1


def test_a_comment_only_line_is_not_refused():
    # Inert, not a "would die on argv[0]" shape -- the syntax check is the one that matters for this case,
    # and a bare comment is valid shell.
    assert run_check("# just a comment").returncode == 0


def test_cd_first_is_accepted():
    # The overwhelming majority shape of every real pool.queue.claims line.
    assert run_check("cd ~/derisk-pool/revisions/deadbeef && echo hi").returncode == 0


def test_leading_assignments_are_skipped_to_find_the_real_first_word():
    assert run_check("SIM_BACKEND=numpy .venv/bin/python -u -m research.runners.foo --help").returncode == 0
    assert run_check("A=1 B=2 C=3 echo hi").returncode == 0


def test_pure_assignment_only_is_accepted():
    assert run_check("FOO=bar").returncode == 0


def test_path_shaped_first_word_is_accepted_without_checking_existence():
    # `.venv/bin/python` need not exist in THIS process's cwd (a worktree has no .venv of its own at all) --
    # existence is the job of the downstream, environment-aware checks (pool_queue.sh's REMOTE VALIDITY,
    # gpu_queue.sh's local argparse check), not this shape gate.
    assert run_check(".venv/bin/python -m pytest -q tests/test_x.py").returncode == 0
    assert run_check("/home/dant123/Projects/sim/.venv/bin/python -u -m research.runners.foo").returncode == 0


def test_shell_keywords_and_builtins_are_accepted():
    for job in (": mem_gb=6 && cd ~/derisk-pool/revisions/deadbeef && echo hi",
                "env SIM_BACKEND=numpy OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/x.py",
                "mkdir -p out && echo done",
                "[ -f x.py ] && echo yes || echo no",
                "if [ -f x.py ]; then echo yes; else echo no; fi",
                "true",
                "bash tools/pool_regression_bundle_v1.sh"):
        res = run_check(job)
        assert res.returncode == 0, f"wrongly refused: {job!r} -- {res.stderr}"


def test_it_never_actually_executes_any_part_of_the_job(tmp_path):
    # Safety property: the DEBUG-trap technique must exit BEFORE the real first command runs, even a
    # destructive-looking one.
    canary = tmp_path / "canary.txt"
    run_check(f"touch {canary} && echo should-not-print")
    assert not canary.exists()


# ------------------------------------------------------------------------------- historical-fixture replay

def test_every_real_historical_good_line_from_pool_queue_claims_is_accepted():
    bad = []
    for ln in _lines("pool_history_good.txt"):
        res = run_check(ln)
        if res.returncode != 0:
            bad.append((ln[:120], res.stderr.strip()[:200]))
    assert not bad, f"{len(bad)} real, correct pool.queue.claims line(s) were wrongly refused: {bad}"


def test_every_real_historical_good_line_from_gpu_queue_log_is_accepted():
    bad = []
    for ln in _lines("gpu_history_good.txt"):
        res = run_check(ln)
        if res.returncode != 0:
            bad.append((ln[:120], res.stderr.strip()[:200]))
    assert not bad, f"{len(bad)} real, correct gpu_queue.log line(s) were wrongly refused: {bad}"


def test_every_real_historical_bad_line_is_refused():
    good = []
    for name in ("pool_history_bad.txt", "gpu_history_bad.txt"):
        for ln in _lines(name):
            res = run_check(ln)
            if res.returncode == 0:
                good.append(ln[:160])
    assert not good, f"{len(good)} known-bad historical line(s) were wrongly accepted: {good}"


# ------------------------------------------------------------------------------------------ producer wiring

def test_pool_queue_add_refuses_the_a2_shape_before_any_network_call(tmp_path):
    # Runs with NO ssh on PATH available (PATH stripped to just what bash/coreutils need) to prove the shape
    # gate fires BEFORE pool_queue.sh's ssh-based MOD/remote-validity checks -- a bad-shaped line must never
    # cost a network round trip.
    queue = tmp_path / "pool.queue"
    job = ("A2 wiring seed 42: mem_gb=8 && cd ~/derisk-pool/revisions/x && env SIM_BACKEND=numpy "
           ".venv/bin/python -u -m research.runners.foo --seeds 42")
    env = {**os.environ, "POOL_QUEUE_PATH": str(queue), "PATH": "/usr/bin:/bin"}
    res = subprocess.run(["bash", str(POOL_QUEUE), "add", job, "--checked", "test"],
                         cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
    assert res.returncode != 0
    assert "'A2'" in res.stderr
    assert queue.read_text() == "" if queue.exists() else True


def test_pool_queue_add_still_accepts_a_correct_line_with_no_module(tmp_path):
    queue = tmp_path / "pool.queue"
    env = {**os.environ, "POOL_QUEUE_PATH": str(queue)}
    res = subprocess.run(["bash", str(POOL_QUEUE), "add", "echo hi", "--checked", "test"],
                         cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
    assert res.returncode == 0, res.stderr
    assert "echo hi  #checked:test" in queue.read_text()


def test_gpu_queue_add_refuses_the_a2_shape(tmp_path):
    res = subprocess.run(["bash", str(GPU_QUEUE), "add",
                          "A2 wiring seed 42: mem_gb=8 && cd ~/x && echo hi"],
                         cwd=ROOT, env={**os.environ, "GPU_QUEUE_DIR": str(tmp_path)},
                         capture_output=True, text=True, timeout=30)
    assert res.returncode != 0
    assert "'A2'" in res.stderr
    assert (tmp_path / "gpu.queue").read_text() == ""


def test_gpu_queue_add_refuses_the_historical_status_job(tmp_path):
    res = subprocess.run(["bash", str(GPU_QUEUE), "add", "status"],
                         cwd=ROOT, env={**os.environ, "GPU_QUEUE_DIR": str(tmp_path)},
                         capture_output=True, text=True, timeout=30)
    assert res.returncode != 0
    assert "'status'" in res.stderr


def test_gpu_queue_add_still_accepts_a_correct_line(tmp_path):
    res = subprocess.run(["bash", str(GPU_QUEUE), "add",
                          "SIM_BACKEND=cupy .venv/bin/python -u -m research.runners.foo --seeds 42"],
                         cwd=ROOT, env={**os.environ, "GPU_QUEUE_DIR": str(tmp_path)},
                         capture_output=True, text=True, timeout=30)
    assert res.returncode == 0, res.stderr
    assert "research.runners.foo" in (tmp_path / "gpu.queue").read_text()


def test_queue_add_gpu_lane_refuses_the_a2_shape_before_appending(tmp_path):
    # queue_add.sh's gpu lane appends DIRECTLY to the queue with no other validation -- this is the ONLY
    # check a gpu-lane line through THIS producer ever gets.
    queue = tmp_path / "gpu.queue"
    env = {**os.environ, "POOL_QUEUE_PATH": str(queue)}
    res = subprocess.run(["bash", str(QUEUE_ADD), "gpu",
                          "A2 wiring seed 42: mem_gb=8 && cd ~/x && echo hi", "reason"],
                         cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
    assert res.returncode != 0
    assert "'A2'" in res.stderr
    assert not queue.exists() or queue.read_text() == ""


def test_queue_add_gpu_lane_still_accepts_a_correct_line(tmp_path):
    queue = tmp_path / "gpu.queue"
    env = {**os.environ, "POOL_QUEUE_PATH": str(queue)}
    res = subprocess.run(["bash", str(QUEUE_ADD), "gpu", "echo hi", "reason"],
                         cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
    assert res.returncode == 0, res.stderr
    assert "echo hi  #checked:reason" in queue.read_text()


# ------------------------------------------------------------------------------------- mutation verification

def test_the_check_fails_in_its_failing_direction_when_step_4_is_disabled(tmp_path):
    # Proves this is a real, specific signal (not a check that can never fail, or one that always fires): with
    # the "does the first word resolve" step short-circuited to always pass, the A2 shape must be ACCEPTED --
    # exactly the regression tools/queue_job_shape_check.sh exists to prevent.
    original = CHECK.read_text(encoding="utf-8")
    marker = 'if type -t "$word" >/dev/null 2>&1; then'
    assert marker in original, "test is stale: queue_job_shape_check.sh's step 4 no longer matches"
    mutated = original.replace(marker, "if true; then", 1)
    assert mutated != original
    mut_path = tmp_path / "queue_job_shape_check_mutated.sh"
    mut_path.write_text(mutated, encoding="utf-8")
    res = subprocess.run(["bash", str(mut_path), "A2 wiring seed 42: mem_gb=8 && cd ~/x && echo hi"],
                         cwd=ROOT, capture_output=True, text=True, timeout=15)
    assert res.returncode == 0, (
        "mutation did not flip the result -- the test is not exercising step 4 "
        f"(stdout={res.stdout!r} stderr={res.stderr!r})")
    # ...and the REAL, unmutated file still refuses the same line (belt and suspenders on the test itself).
    assert run_check("A2 wiring seed 42: mem_gb=8 && cd ~/x && echo hi").returncode == 1
