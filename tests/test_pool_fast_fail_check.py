"""tools/pool_autodispatch.sh: check_fast_fail -- a READ-ONLY, post-dispatch net that catches a pool job
which died almost instantly with rc=127 ("command not found") or rc=2 (a shell syntax/usage error): the
signature of a job that never actually ran.

WHY (2026-09-25). The SETTLE A2 lines (research/FAILURE_LOG.md's 2026-09-25 row) and, independently, a bare
`status` job in research/queue/gpu_queue.log (2026-08-31/09-01, three separate cycles, rc=127 in well under a
second each time) both died this way with nothing in the dispatcher's OWN log distinguishing them from an
ordinary completion. tools/queue_job_shape_check.sh now refuses that SHAPE at enqueue time (see
tests/test_queue_job_shape_check.py); this is the belt-and-suspenders net for whatever it cannot see from the
enqueue side (a module importable on the shared checkout but not on the job's pinned revision, e.g.).

check_fast_fail never writes anything to the node -- it only reads (greps) the node's own job_status.log,
which remote_launch_command's wrapper still writes with EXACTLY the same v2 format as before (untouched).
Exercised here with a stub `ssh` on PATH (no real network, no real node), mirroring
tests/test_pool_ssh_config_plumbing.py's approach.
"""
from __future__ import annotations

import base64
import os
import stat
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUTODISPATCH = ROOT / "tools" / "pool_autodispatch.sh"


def _make_canned_ssh_stub(tmp_path: Path, canned_stdout: str) -> tuple[Path, Path]:
    """A stub `ssh` that logs its full argv and answers every call with a FIXED canned response -- this
    exercises check_fast_fail's OWN parsing of an ssh response, not real remote grep behaviour (which
    remote_launch_command's wrapper -- untouched -- is responsible for producing correctly)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    log = tmp_path / "ssh.log"
    log.write_text("")
    canned = tmp_path / "canned_stdout.txt"
    canned.write_text(canned_stdout)
    stub = bin_dir / "ssh"
    stub.write_text(f"""#!/usr/bin/env bash
echo "$*" >> "{log}"
cat "{canned}"
exit 0
""")
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return bin_dir, log


def _run_check(job: str, bin_dir: Path, env_extra: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"
    if env_extra:
        env.update(env_extra)
    return subprocess.run(["bash", str(AUTODISPATCH), "--check-fast-fail", "pool40", job],
                          cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)


def _job_b64(job: str) -> str:
    return base64.b64encode(job.encode()).decode()


def test_a_matching_rc127_record_is_logged_loudly(tmp_path):
    job = "status"
    canned = f"v2\t1790000000\t127\t{_job_b64(job)}"
    bin_dir, ssh_log = _make_canned_ssh_stub(tmp_path, canned)
    res = _run_check(job, bin_dir)
    assert res.returncode == 0   # read-only diagnostic call, never fails the caller
    assert "FAST-FAIL" in res.stderr
    assert "rc=127" in res.stderr
    assert "pool40" in ssh_log.read_text()


def test_a_matching_rc2_record_is_logged_loudly(tmp_path):
    job = "cd ~/derisk-pool/sim && bad syntax &&"
    canned = f"v2\t1790000000\t2\t{_job_b64(job)}"
    bin_dir, _ = _make_canned_ssh_stub(tmp_path, canned)
    res = _run_check(job, bin_dir)
    assert "FAST-FAIL" in res.stderr
    assert "rc=2" in res.stderr


def test_a_normal_rc0_record_is_never_flagged(tmp_path):
    # THE FAILING DIRECTION: a genuine, successful completion must never be reported as a fast-fail.
    job = "cd ~/derisk-pool/sim && SIM_BACKEND=numpy .venv/bin/python -u -m research.runners.foo --seeds 42"
    canned = f"v2\t1790000000\t0\t{_job_b64(job)}"
    bin_dir, _ = _make_canned_ssh_stub(tmp_path, canned)
    res = _run_check(job, bin_dir)
    assert res.returncode == 0
    assert "FAST-FAIL" not in res.stderr


def test_a_slower_genuine_failure_rc1_is_never_flagged():
    # Only rc=127/2 (argv[0]/syntax death) are flagged -- an ordinary rc=1 (a runner's own honest failure
    # verdict) is real, useful, EXPECTED research output, not a fast-fail.
    job = "cd ~/derisk-pool/sim && SIM_BACKEND=numpy .venv/bin/python -u -m research.runners.foo --seeds 42"
    import tempfile
    with tempfile.TemporaryDirectory() as t:
        bin_dir, _ = _make_canned_ssh_stub(Path(t), f"v2\t1790000000\t1\t{_job_b64(job)}")
        res = _run_check(job, bin_dir)
    assert "FAST-FAIL" not in res.stderr


def test_no_matching_record_yet_is_silent_not_a_false_alarm(tmp_path):
    # The job hasn't completed/appeared in job_status.log yet (still running, or slower than the launch-sleep
    # window) -- best-effort, never a false positive.
    bin_dir, _ = _make_canned_ssh_stub(tmp_path, "")
    res = _run_check("cd ~/derisk-pool/sim && slow_job_still_running", bin_dir)
    assert res.returncode == 0
    assert "FAST-FAIL" not in res.stderr
    assert res.stderr.strip() == ""


def test_the_probe_is_read_only_grep_never_a_write(tmp_path):
    job = "status"
    bin_dir, ssh_log = _make_canned_ssh_stub(tmp_path, "")
    _run_check(job, bin_dir)
    logged = ssh_log.read_text()
    assert "grep" in logged
    assert "job_status.log" in logged
    assert ">>" not in logged and " > " not in logged   # never redirects output into anything remotely
    # the exact job text, base64-encoded, is what is searched for -- not the raw job text (which could
    # contain shell metacharacters unsafe to splice into the remote command).
    assert _job_b64(job) in logged


def test_ssh_call_carries_pool_ssh_config_when_present(tmp_path):
    bin_dir, ssh_log = _make_canned_ssh_stub(tmp_path, "")
    config = tmp_path / "ssh_config"
    config.write_text("Include ~/.ssh/config\n")
    _run_check("status", bin_dir, {"POOL_SSH_CONFIG": str(config)})
    assert f"-F {config}" in ssh_log.read_text()


def test_ssh_call_omits_dash_F_when_pool_ssh_config_absent(tmp_path):
    bin_dir, ssh_log = _make_canned_ssh_stub(tmp_path, "")
    _run_check("status", bin_dir, {"POOL_SSH_CONFIG": str(tmp_path / "does-not-exist")})
    # Only the ssh-side arguments (everything before the "pool40" hostname) count -- the remote command
    # legitimately contains grep's OWN unrelated "-F" (fixed-string) flag.
    ssh_side = ssh_log.read_text().split("pool40", 1)[0]
    assert " -F " not in f" {ssh_side} "


def test_fast_fail_check_is_skippable_via_env_flag(tmp_path):
    # fill_node's opt-out seam (POOL_SKIP_FAST_FAIL_CHECK): not exercised via --check-fast-fail directly (that
    # seam always calls the function), but the flag's presence in the production code path is pinned by
    # grepping the source -- a cheap, honest static check that the seam exists.
    text = AUTODISPATCH.read_text(encoding="utf-8")
    assert "POOL_SKIP_FAST_FAIL_CHECK" in text
    assert "check_fast_fail" in text


# ------------------------------------------------------------------------------------- mutation verification

def test_fails_in_its_failing_direction_when_the_rc_case_is_disabled(tmp_path):
    original = AUTODISPATCH.read_text(encoding="utf-8")
    marker = "  case \"$rc\" in\n    127|2)"
    assert marker in original, "test is stale: check_fast_fail's rc case no longer matches"
    mutated = original.replace(marker, "  case \"$rc\" in\n    999999)", 1)
    assert mutated != original
    mut_path = tmp_path / "pool_autodispatch_mutated.sh"
    mut_path.write_text(mutated, encoding="utf-8")
    mut_path.chmod(mut_path.stat().st_mode | stat.S_IEXEC)

    job = "status"
    canned = f"v2\t1790000000\t127\t{base64.b64encode(job.encode()).decode()}"
    bin_dir, _ = _make_canned_ssh_stub(tmp_path, canned)
    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"
    res = subprocess.run(["bash", str(mut_path), "--check-fast-fail", "pool40", job],
                         cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
    assert "FAST-FAIL" not in res.stderr, "mutation did not flip the result -- test is not exercising the rc case"
    # ...and the REAL, unmutated script still catches it.
    real_bin_dir, _ = _make_canned_ssh_stub(tmp_path / "real", canned)
    real_res = _run_check(job, real_bin_dir)
    assert "FAST-FAIL" in real_res.stderr
