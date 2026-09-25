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
import time
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


# The dispatch time every canned record below is measured against: fill_node takes `date +%s` just before its
# `ssh -f` launch and check_fast_fail accepts only a record written at or after it.
T0 = 1790000000


def _run_check(job: str, bin_dir: Path, env_extra: dict[str, str] | None = None,
               t0: int | None = T0) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"
    if env_extra:
        env.update(env_extra)
    argv = ["bash", str(AUTODISPATCH), "--check-fast-fail", "pool40", job]
    if t0 is not None:
        argv.append(str(t0))
    return subprocess.run(argv, cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)


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


# --------------------------------------------------------- only THIS dispatch's record counts (review MEDIUM)

def test_a_stale_record_from_an_earlier_attempt_is_never_flagged(tmp_path):
    # job_status.log is append-only: an identical job dispatched again, whose EARLIER attempt died rc=127, must
    # not be reported as a new fast-fail. The review fed the first version a record stamped 1000000000 (2001)
    # and it logged "rc=127 within ~5s of dispatch".
    job = "status"
    bin_dir, _ = _make_canned_ssh_stub(tmp_path, f"v2\t1000000000\t127\t{_job_b64(job)}")
    res = _run_check(job, bin_dir)
    assert "FAST-FAIL" not in res.stderr, res.stderr


def test_the_dispatch_time_is_an_inclusive_lower_bound(tmp_path):
    job = "status"
    one_early, _ = _make_canned_ssh_stub(tmp_path / "early", f"v2\t{T0 - 1}\t127\t{_job_b64(job)}")
    assert "FAST-FAIL" not in _run_check(job, one_early).stderr
    on_time, _ = _make_canned_ssh_stub(tmp_path / "on_time", f"v2\t{T0}\t127\t{_job_b64(job)}")
    assert "FAST-FAIL" in _run_check(job, on_time).stderr


def test_only_the_newest_fresh_record_decides(tmp_path):
    # An earlier fresh attempt died rc=127 and the newest record for this job succeeded: the newest wins.
    job = "status"
    b = _job_b64(job)
    bin_dir, _ = _make_canned_ssh_stub(tmp_path, f"v2\t{T0 + 1}\t127\t{b}\nv2\t{T0 + 2}\t0\t{b}")
    assert "FAST-FAIL" not in _run_check(job, bin_dir).stderr


def test_a_longer_job_sharing_the_base64_prefix_is_never_flagged(tmp_path):
    # The remote probe is a substring grep, so it also returns a LONGER job whose base64 starts with this
    # job's; only an exact match on the record's job field may count.
    job, longer = "status", "status2 --really-a-different-job"
    assert _job_b64(longer).startswith(_job_b64(job))   # the collision this test is about
    bin_dir, _ = _make_canned_ssh_stub(tmp_path, f"v2\t{T0 + 1}\t127\t{_job_b64(longer)}")
    assert "FAST-FAIL" not in _run_check(job, bin_dir).stderr


def test_without_a_dispatch_time_nothing_is_reported(tmp_path):
    # With no t0 there is no way to tell this dispatch's record from an earlier attempt's, so stay silent.
    job = "status"
    bin_dir, _ = _make_canned_ssh_stub(tmp_path, f"v2\t{T0 + 1}\t127\t{_job_b64(job)}")
    res = _run_check(job, bin_dir, t0=None)
    assert "FAST-FAIL" not in res.stderr
    assert res.returncode == 0


# --------------------------------------------- the production call site, through fill_node (review MEDIUM)

def _fill_node_stub(tmp_path: Path, launch_record: str) -> tuple[Path, Path, Path]:
    """A stub `ssh` for fill_node's real loop, isolated under tmp_path (the node's HOME is tmp_path/node):
      * the node_is_idle metrics probe (its command carries 'MemAvailable') -> an idle, roomy node;
      * the `ssh -f` launch (carries JOB_B64='...') -> emulates remote_launch_command's wrapper by appending
        a v2 record for exactly that JOB_B64 to <node HOME>/derisk-pool/sim/job_status.log, per
        `launch_record`: 'fast127' (died rc=127 at once), 'ok' (rc=0), or 'stale127' (no record for this
        attempt yet, but an EARLIER attempt of the identical job died rc=127, stamped 2001);
      * the read-only fast-fail probe (reads job_status.log) -> runs that probe's own command with HOME set to
        the stub node's HOME, so the real grep runs, not a canned answer.
    It never runs the job itself."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "ssh.log"
    log.write_text("")
    node_home = tmp_path / "node"
    (node_home / "derisk-pool" / "sim").mkdir(parents=True)
    status_log = node_home / "derisk-pool" / "sim" / "job_status.log"
    status_log.write_text("")
    stub = bin_dir / "ssh"
    stub.write_text(f"""#!/usr/bin/env bash
echo "$*" >> "{log}"
case "$*" in
  *MemAvailable*) echo "8 1 0 20 1 32"; exit 0 ;;
  *JOB_B64=*)
    b64=$(printf '%s' "$*" | grep -oE "JOB_B64='[^']*'" | head -1 | cut -d"'" -f2)
    case "{launch_record}" in
      fast127) printf 'v2\\t%s\\t127\\t%s\\n' "$(date +%s)" "$b64" >> "{status_log}" ;;
      ok) printf 'v2\\t%s\\t0\\t%s\\n' "$(date +%s)" "$b64" >> "{status_log}" ;;
      stale127) printf 'v2\\t1000000000\\t127\\t%s\\n' "$b64" >> "{status_log}" ;;
    esac
    exit 0 ;;
  *job_status.log*) HOME="{node_home}" bash -c "${{@: -1}}"; exit $? ;;
esac
exit 0
""")
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return bin_dir, log, status_log


def _fill_node(tmp_path: Path, launch_record: str, extra_env: dict[str, str] | None = None):
    now = int(time.time())
    queue = tmp_path / "pool.queue"
    queue.write_text(f"{now}\tstatus  #checked:fast-fail call-site test mem_gb=1\n")
    bin_dir, ssh_log, status_log = _fill_node_stub(tmp_path, launch_record)
    (tmp_path / "tmp").mkdir()
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ.get('PATH', '')}",
        "TMPDIR": str(tmp_path / "tmp"),
        "POOL_QUEUE_PATH": str(queue),
        "POOL_RUNNING_PATH": str(tmp_path / "pool.running"),
        "POOL_RESERVATIONS_PATH": str(tmp_path / "resv"),
        "POOL_EXTRA_NODES_FILE": str(tmp_path / "extra_nodes"),
        "POOL_SSH_CONFIG": str(tmp_path / "does-not-exist"),
        "POOL_RUNNER_MEM_PATH": str(tmp_path / "runner_mem.tsv"),
        "POOL_DISPATCH_LAUNCH_SLEEP": "0",
        **(extra_env or {}),
    }
    res = subprocess.run(["bash", str(AUTODISPATCH), "--fill-node", "pool1"],
                         cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
    return res, ssh_log, status_log, queue


def test_fill_node_reports_a_fast_fail_at_the_production_call_site(tmp_path):
    # The review replaced `check_fast_fail "$NODE" "$JOB"` in fill_node with `true` and every test still
    # passed; this drives the real fill_node loop (via --fill-node) end to end.
    res, ssh_log, status_log, queue = _fill_node(tmp_path, "fast127")
    assert res.returncode == 0, res.stderr
    assert queue.read_text().strip() == ""                    # the job really was popped and launched
    assert "\t127\t" in status_log.read_text()                 # the stub wrapper wrote its rc=127 record
    assert "FAST-FAIL: pool1 rc=127" in res.stderr, res.stderr
    assert "job_status.log" in ssh_log.read_text()             # ...found by the real read-only probe


def test_fill_node_does_not_flag_a_successful_dispatch(tmp_path):
    res, _, status_log, _ = _fill_node(tmp_path, "ok")
    assert res.returncode == 0, res.stderr
    assert "\t0\t" in status_log.read_text()
    assert "FAST-FAIL" not in res.stderr


def test_fill_node_does_not_flag_an_earlier_attempts_record(tmp_path):
    res, _, status_log, _ = _fill_node(tmp_path, "stale127")
    assert res.returncode == 0, res.stderr
    assert "\t127\t" in status_log.read_text()                 # the stale record IS in the log...
    assert "FAST-FAIL" not in res.stderr, res.stderr            # ...and it is not this dispatch's


def test_fill_node_skips_the_probe_when_opted_out(tmp_path):
    res, ssh_log, _, _ = _fill_node(tmp_path, "fast127", {"POOL_SKIP_FAST_FAIL_CHECK": "1"})
    assert res.returncode == 0, res.stderr
    assert "FAST-FAIL" not in res.stderr
    assert "job_status.log" not in ssh_log.read_text()


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
    res = subprocess.run(["bash", str(mut_path), "--check-fast-fail", "pool40", job, str(T0)],
                         cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
    assert "FAST-FAIL" not in res.stderr, "mutation did not flip the result -- test is not exercising the rc case"
    # ...and the REAL, unmutated script still catches it.
    real_bin_dir, _ = _make_canned_ssh_stub(tmp_path / "real", canned)
    real_res = _run_check(job, real_bin_dir)
    assert "FAST-FAIL" in real_res.stderr
