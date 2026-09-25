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

THE CHECK MUST NEVER RUN THE JOB (2026-09-25 fix round, review HIGH). The first version found the first command
with a DEBUG trap in a bash that ran the job; subshells do not inherit that trap, so a job starting with
`( ... )`, `coproc`, `(cmd) | cat`, `if (cmd)` or `time ( ... )` ran on the machine that queued it. The check is
now a static parser; test_no_shape_ever_executes_any_part_of_the_job drives those shapes (and more) with a
command that would create a marker file, and the marker must stay absent.

FIXTURES. tests/fixtures/queue_job_shapes/ is written by tools/queue_job_shape_replay.py --fixtures-dir: from a
full replay of the REAL research/queue/pool.queue.claims and research/queue/gpu_queue.log on this machine, one
line per distinct first-four-whitespace-token shape, split by verdict (accepted -> *_good.txt, refused ->
*_bad.txt). The full replay (research/coordination/queue_shape_replay_2026-09-25.json, whose header gives the
line counts) refused exactly 12 lines and no others: the 6 SETTLE A2 lines (claims 1923-1928), 3 torn lines
(claims 1638, 1647, 1916) and the bare `status` job (gpu_queue.log 672190, 711341, 760341, a real rc=127 each
time, never flagged then). "Accepted" is not "correct": the good fixture also holds torn lines whose first
surviving word contains a "/" (e.g. `tive-memory/lb.json ...`), which the check accepts by design (see the
PATH-shaped note in the check's header); the good-fixture tests pin that no accepted shape starts being refused.
"""
from __future__ import annotations

import os
import stat
import subprocess
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CHECK = ROOT / "tools" / "queue_job_shape_check.sh"
POOL_QUEUE = ROOT / "tools" / "pool_queue.sh"
GPU_QUEUE = ROOT / "tools" / "gpu_queue.sh"
QUEUE_ADD = ROOT / "tools" / "queue_add.sh"
FIXTURES = ROOT / "tests" / "fixtures" / "queue_job_shapes"

A2_LINE = ("A2 wiring seed 42: mem_gb=8 && cd ~/derisk-pool/revisions/5b5ea1b7447b24d0978189fb53557a57b88952d4 "
           "&& env SIM_BACKEND=numpy CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 .venv/bin/python -u -m "
           "research.runners._affect_marker_settle_congruence --run-wiring --seeds 42")


def run_check(job: str, check: Path = CHECK) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["bash", str(check), job], cwd=ROOT, capture_output=True, text=True, timeout=15)


def _lines(name: str) -> list[str]:
    text = (FIXTURES / name).read_text(encoding="utf-8")
    return [ln for ln in text.splitlines() if ln.strip()]


# ------------------------------------------------------------------------------- the check never runs the job

# Every shape here would create {m} if any part of it ran. The first eight are the ones the DEBUG-trap version
# actually ran when this list was run against it (review HIGH: subshell, pipeline from a subshell, coproc,
# `if (cmd)`, `time ( ... )`, nested subshells); the rest cover the other ways a command can start running
# before, around or instead of the first simple command.
CANARY_SHAPES = [
    "( touch {m} ) && echo x",
    "(touch {m}) | cat",
    "coproc touch {m}",
    "coproc {{ touch {m}; }}",
    "if (touch {m}); then :; fi",
    "time ( touch {m} )",
    "( ( touch {m} ) )",
    "(A2 x; touch {m})",
    "time touch {m}",
    "{{ touch {m}; }} && echo",
    "! touch {m}",
    "while touch {m}; do break; done",
    "until touch {m}; do :; done",
    "X=$(touch {m}) echo",
    "X=`touch {m}` echo",
    "$(touch {m})",
    "((x=1)) && touch {m}",
    "f(){{ touch {m}; }}; f",
    ": && touch {m}",
    "mem_gb=8 && touch {m}",
    "touch {m} & wait",
    "A2 wiring; touch {m}",
    "cat <(touch {m})",
    "for i in 1; do touch {m}; done",
    "case x in x) touch {m};; esac",
    "[[ -n $(touch {m}) ]]",
    "trap 'touch {m}' EXIT",
    ": <<EOF\n$(touch {m})\nEOF",
]


def test_no_shape_ever_executes_any_part_of_the_job(tmp_path):
    markers = []
    for i, shape in enumerate(CANARY_SHAPES):
        marker = tmp_path / f"ran_{i}"
        markers.append((shape, marker))
        run_check(shape.format(m=marker))
    time.sleep(0.5)   # a backgrounded or coprocess shape would need a moment to land its marker
    ran = [shape for shape, marker in markers if marker.exists()]
    assert not ran, f"the shape check EXECUTED part of {len(ran)} queued job(s): {ran}"


def test_a_label_inside_a_subshell_is_refused_not_waved_through():
    # Review HIGH: the DEBUG-trap version returned rc=0 for this (fail-open) after running it.
    res = run_check("(A2 wiring seed 42: mem_gb=8 && cd x)")
    assert res.returncode == 1
    assert "'A2'" in res.stderr


@pytest.mark.parametrize("job", [
    "( A2 x ) && echo hi",
    "(A2 x) | cat",
    "if (A2 x); then :; fi",
    "if A2 x; then :; fi",
    "while A2; do :; done",
    "time ( A2 )",
    "time -p A2",
    "{ A2; }",
    "! A2",
    "( ( A2 ) )",
    ">/dev/null A2 x",
    "2>/dev/null A2 x",
    "{fd}>/dev/null A2",
    "a=(1 2 3) A2",
    '"A2" x',
    "\\A2 x",
])
def test_the_first_real_command_is_found_through_wrappers(job):
    res = run_check(job)
    assert res.returncode == 1, f"accepted {job!r}"
    assert "'A2'" in res.stderr


@pytest.mark.parametrize("job", ["coproc cat", "coproc NAME { cat; }", "coproc ( cat )"])
def test_a_coprocess_is_refused(job):
    res = run_check(job)
    assert res.returncode == 1
    assert "coproc" in res.stderr


# ------------------------------------------------------------------------------------------------ unit cases

def test_the_a2_shape_is_refused():
    # The exact defect: a prose label glued onto an otherwise-correct pinned-revision job.
    res = run_check(A2_LINE)
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


@pytest.mark.parametrize("job", [
    'X="a b" .venv/bin/python -m foo',
    "X='a b' .venv/bin/python -m foo",
    'A="x y" B=\'p q\' C=$(date) echo hi',
    'X="a b" A2x=1 true',
])
def test_quoted_assignment_values_with_spaces_are_accepted(job):
    # Review LOW-MEDIUM: `read -r -a` split `X="a b"` into `X="a` and `b"`, and refused `b"` as the command.
    res = run_check(job)
    assert res.returncode == 0, res.stderr


def test_pure_assignment_only_is_accepted():
    assert run_check("FOO=bar").returncode == 0


@pytest.mark.parametrize("job", ["((x=1)) && echo hi", "((x+1)) && echo hi"])
def test_an_arithmetic_command_first_is_accepted(job):
    # Review LOW-MEDIUM: `((x=1))` was refused as an unknown word. (`((x+1))` is the case that would be misread
    # as nested subshells running a command `x+1` if `((` were not recognised.)
    assert run_check(job).returncode == 0


@pytest.mark.parametrize("job", ["f(){ :; }; f", "f () { :; }; f", "function g { :; }; g"])
def test_a_job_that_defines_its_own_function_is_accepted(job):
    # Review LOW-MEDIUM: `f(){ :; }; f` was refused because `type -t f` ran in the caller's shell.
    assert run_check(job).returncode == 0


def test_the_callers_own_functions_never_make_a_job_look_runnable():
    # Review LOW-MEDIUM: `type -t` ran in the CALLER's shell, so a job whose first word happened to be one of
    # the caller's functions (gpu_queue.sh has `daemon`, `selftest`, ...) was accepted, though the job runs in
    # a shell that has no such function. The lookup now runs in a fresh bash with an EMPTY environment (no
    # exported functions either).
    script = f"""
daemon() {{ :; }}
exported_fn() {{ :; }}
export -f exported_fn
source {CHECK}
queue_job_runnable_check "daemon --flag" >/dev/null && echo DAEMON_ACCEPTED
queue_job_runnable_check "exported_fn" >/dev/null && echo EXPORTED_ACCEPTED
queue_job_runnable_check "cd / && echo hi" >/dev/null && echo CD_ACCEPTED
true
"""
    res = subprocess.run(["bash", "-c", script], cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert "CD_ACCEPTED" in res.stdout, res.stderr
    assert "DAEMON_ACCEPTED" not in res.stdout
    assert "EXPORTED_ACCEPTED" not in res.stdout


@pytest.mark.parametrize("job", [
    "mem_gb=8 && A2 wiring seed 42: cd x",
    ": mem_gb=8 && A2 wiring seed 42: cd x",
    "true && A2",
    ": ; : ; A2",
    ": mem_gb=8 | A2",
    "( : ) && A2",
    "{ :; } && A2",
    "if :; then A2; fi",
    "while :; do A2; done",
    "if ( : ) then A2; fi",
    ": <<EOF\ncd x\nEOF\nA2 junk",
])
def test_a_leading_noop_is_stepped_over(job):
    # Review LOW: 90 historical pool lines start `: mem_gb=N &&` or `mem_gb=N &&`; for those the first version
    # checked only the no-op, so a label placed after it (`: mem_gb=8 && A2 wiring ...`) was accepted.
    res = run_check(job)
    assert res.returncode == 1, f"accepted {job!r}"
    assert "'A2'" in res.stderr


@pytest.mark.parametrize("job", [
    ": mem_gb=8 && cd ~/derisk-pool/revisions/deadbeef && echo hi",
    "mem_gb=8 && cd x",
    ": || A2",                       # A2 runs only if `:` fails, which it cannot
    "! : && A2",                     # `! :` is false: A2 never runs
    "until :; do A2; done",          # the body runs only if the condition fails
    "if :; then :; else A2; fi",     # a branch that does not run
    ": ; : ; : ; : ; : ; : ; : ; : ; : ; A2",   # past the cap of 8 no-ops the check stops looking
    ": <<EOF\nA2 junk\nEOF\ncd x",    # a here-document body is data, not a command
])
def test_what_cannot_run_after_a_noop_is_not_judged(job):
    res = run_check(job)
    assert res.returncode == 0, res.stderr


@pytest.mark.parametrize("job", ["$PY -m foo", '"$HOME"/.venv/bin/python x', "${PY:-python3} x", "{a,b} x"])
def test_a_command_word_built_from_an_expansion_is_accepted(job):
    # Statically unknowable, so never refused.
    assert run_check(job).returncode == 0


def test_only_the_first_real_command_is_judged():
    # A SHAPE check, deliberately: later commands are the downstream checks' business.
    assert run_check("cd x && A2").returncode == 0


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
                "[[ -f x.py ]] && echo yes",
                "for s in 42 43; do echo $s; done",
                "true",
                "bash tools/pool_regression_bundle_v1.sh"):
        res = run_check(job)
        assert res.returncode == 0, f"wrongly refused: {job!r} -- {res.stderr}"


# ------------------------------------------------------------------------------- historical-fixture replay

def test_every_accepted_historical_pool_queue_claims_shape_stays_accepted():
    bad = []
    for ln in _lines("pool_history_good.txt"):
        res = run_check(ln)
        if res.returncode != 0:
            bad.append((ln[:120], res.stderr.strip()[:200]))
    assert not bad, f"{len(bad)} historically accepted pool.queue.claims line(s) are now refused: {bad}"


def test_every_accepted_historical_gpu_queue_log_shape_stays_accepted():
    bad = []
    for ln in _lines("gpu_history_good.txt"):
        res = run_check(ln)
        if res.returncode != 0:
            bad.append((ln[:120], res.stderr.strip()[:200]))
    assert not bad, f"{len(bad)} historically accepted gpu_queue.log line(s) are now refused: {bad}"


def test_every_real_historical_bad_line_is_refused():
    good = []
    for name in ("pool_history_bad.txt", "gpu_history_bad.txt"):
        for ln in _lines(name):
            res = run_check(ln)
            if res.returncode == 0:
                good.append(ln[:160])
    assert not good, f"{len(good)} known-bad historical line(s) were wrongly accepted: {good}"


def test_the_bad_fixtures_are_exactly_the_known_refusals():
    # 6 A2 lines + 3 torn lines in pool.queue.claims; the `status` job in gpu_queue.log.
    pool_bad = _lines("pool_history_bad.txt")
    assert sum(ln.startswith("A2 wiring seed ") for ln in pool_bad) == 6
    assert len(pool_bad) == 9
    assert _lines("gpu_history_bad.txt") == ["status"]


# ------------------------------------------------------------------------------------------ producer wiring

def _ssh_recorder(tmp_path: Path) -> tuple[Path, Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "ssh_calls.log"
    stub = bin_dir / "ssh"
    stub.write_text(f'#!/usr/bin/env bash\necho "$*" >> "{log}"\nexit 255\n')
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return bin_dir, log


def test_pool_queue_add_refuses_the_a2_shape_before_any_network_call(tmp_path):
    # A recording `ssh` stub sits FIRST on PATH, so any ssh call pool_queue.sh makes (its MOD/remote-validity
    # probes) would land in the log; the shape gate must refuse before any of them runs -- a bad-shaped line
    # must never cost a network round trip.
    queue = tmp_path / "pool.queue"
    bin_dir, ssh_log = _ssh_recorder(tmp_path)
    job = ("A2 wiring seed 42: mem_gb=8 && cd ~/derisk-pool/revisions/x && env SIM_BACKEND=numpy "
           ".venv/bin/python -u -m research.runners.foo --seeds 42")
    env = {**os.environ, "POOL_QUEUE_PATH": str(queue), "PATH": f"{bin_dir}:/usr/bin:/bin"}
    res = subprocess.run(["bash", str(POOL_QUEUE), "add", job, "--checked", "test"],
                         cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
    assert res.returncode != 0
    assert "'A2'" in res.stderr
    assert not ssh_log.exists(), f"ssh was called before the shape gate refused: {ssh_log.read_text()}"
    assert not queue.exists() or queue.read_text() == ""


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


@pytest.mark.parametrize("word", ["selftest", "daemon"])
def test_gpu_queue_add_refuses_a_word_that_is_only_one_of_its_own_functions(tmp_path, word):
    # gpu_queue.sh defines `selftest` and `daemon`; the queued job runs in `bash -c`, which has neither.
    res = subprocess.run(["bash", str(GPU_QUEUE), "add", word],
                         cwd=ROOT, env={**os.environ, "GPU_QUEUE_DIR": str(tmp_path)},
                         capture_output=True, text=True, timeout=30)
    assert res.returncode != 0
    assert f"'{word}'" in res.stderr
    assert (tmp_path / "gpu.queue").read_text() == ""


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

def _mutant(tmp_path: Path, marker: str, replacement: str) -> Path:
    original = CHECK.read_text(encoding="utf-8")
    assert marker in original, f"test is stale: {marker!r} no longer appears in queue_job_shape_check.sh"
    mutated = original.replace(marker, replacement, 1)
    mut_path = tmp_path / "queue_job_shape_check_mutated.sh"
    mut_path.write_text(mutated, encoding="utf-8")
    return mut_path


def test_the_check_fails_in_its_failing_direction_when_the_lookup_always_succeeds(tmp_path):
    # With the "does the word resolve" lookup made to always succeed, the A2 shape must be ACCEPTED -- exactly
    # the regression this file exists to prevent -- so the refusals above really come from that lookup.
    mut = _mutant(tmp_path, """'PATH=$1; type -t -- "$2" >/dev/null 2>&1'""", "'true'")
    res = run_check("A2 wiring seed 42: mem_gb=8 && cd ~/x && echo hi", mut)
    assert res.returncode == 0, f"mutation did not flip the result (stderr={res.stderr!r})"
    assert run_check("A2 wiring seed 42: mem_gb=8 && cd ~/x && echo hi").returncode == 1


def test_the_noop_step_over_is_what_catches_a_label_after_mem_gb(tmp_path):
    # With `:`/`true` treated as an ordinary runnable command (no step-over), `: mem_gb=8 && A2 ...` is accepted.
    mut = _mutant(tmp_path, "':'|true) return 2 ;;", "':'|true) return 0 ;;")
    job = ": mem_gb=8 && A2 wiring seed 42: cd x"
    assert run_check(job, mut).returncode == 0, "mutation did not flip the result"
    assert run_check(job).returncode == 1
