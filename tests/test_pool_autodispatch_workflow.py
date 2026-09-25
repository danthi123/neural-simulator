from __future__ import annotations

import base64
import os
import stat
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
DISPATCHER = ROOT / "tools" / "pool_autodispatch.sh"
WORKFLOW = ROOT / "tools" / "workflow_check.sh"


def run_bash(script: Path, *args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(script), *args],
        cwd=ROOT,
        env={**os.environ, **(env or {})},
        text=True,
        capture_output=True,
        check=True,
    )


def test_memory_reservations_expire_and_jobs_declare_size(tmp_path: Path) -> None:
    # 2026-09-23: one fill cycle sent six growing D6 workers to one 15 GB node because each capacity check saw the
    # RSS snapshot from before the previous launch had grown. Dispatches now reserve their declared size.
    now = int(time.time())
    resv = tmp_path / "resv"
    resv.write_text(f"{now - 60} pool42 5\n{now - 30} pool42 5\n{now - 30} pool41 1\n{now - 5000} pool42 9\n")
    queue = tmp_path / "pool.queue"
    queue.write_text(f"{now}\tpython -m research.runners.x  #checked:reason mem_gb=5\n")
    env = {"POOL_QUEUE_PATH": str(queue), "POOL_RESERVATIONS_PATH": str(resv), "POOL_GROWTH_WINDOW_S": "1200"}
    assert run_bash(DISPATCHER, "--reserved-gb", "pool42", env=env).stdout.strip() == "10"   # the 9 GB row expired
    assert run_bash(DISPATCHER, "--reserved-gb", "pool41", env=env).stdout.strip() == "1"
    assert run_bash(DISPATCHER, "--reserved-gb", "pool40", env=env).stdout.strip() == "0"
    assert run_bash(DISPATCHER, "--peek-est-gb", env=env).stdout.strip() == "5"
    queue.write_text(f"{now}\tpython -m research.runners.x  #checked:reason\n")
    assert run_bash(DISPATCHER, "--peek-est-gb", env={**env, "POOL_JOB_EST_GB": "2"}).stdout.strip() == "2"
    queue.write_text(f"{now}\tbash tools/memcap.sh 8 -- python -m research.runners.x  #checked:reason\n")
    assert run_bash(DISPATCHER, "--peek-est-gb", env=env).stdout.strip() == "8"   # memcap cap is the fallback
    table = tmp_path / "mem.tsv"
    table.write_text("# comment\nload_bearing_fraction\t6\n")
    queue.write_text(f"{now}\tpython -m research.runners.load_bearing_fraction --only x  #checked:reason\n")
    env2 = {**env, "POOL_RUNNER_MEM_PATH": str(table)}
    assert run_bash(DISPATCHER, "--peek-est-gb", env=env2).stdout.strip() == "6"   # per-runner measured peak
    queue.write_text(f"{now}\tpython -m research.runners.other_runner  #checked:reason\n")
    assert run_bash(DISPATCHER, "--peek-est-gb", env=env2).stdout.strip() == "1"   # unknown runner -> default


def test_running_jobs_commit_their_declared_size_for_their_lifetime(tmp_path: Path) -> None:
    # 2026-09-23 20:45: a between-phases snapshot read 9 GB free while two ~6 GB LB jobs ran; a third went out and
    # both nodes thrashed. Each launch now stamps POOL_JOB_ID/POOL_JOB_MEM_GB into the job's inherited environment.
    rendered = run_bash(DISPATCHER, "--render-remote-command", "python -m research.runners.x  #checked:r mem_gb=5",
                        env={"HOME": str(tmp_path)}).stdout
    assert "POOL_JOB_MEM_GB='5'" in rendered and "POOL_JOB_ID='" in rendered
    lines = "POOL_JOB_ID=a POOL_JOB_MEM_GB=6\nPOOL_JOB_ID=b POOL_JOB_MEM_GB=5\n"
    r = subprocess.run(["bash", str(DISPATCHER), "--committed-gb"], input=lines, cwd=ROOT, text=True,
                       capture_output=True, check=True)
    assert r.stdout.strip() == "11"
    r = subprocess.run(["bash", str(DISPATCHER), "--committed-gb"], input="", cwd=ROOT, text=True,
                       capture_output=True, check=True)
    assert r.stdout.strip() == "0"


def test_jobs_default_to_one_math_thread_unless_they_set_their_own(tmp_path: Path) -> None:
    # 2026-09-24: jobs with no thread count ran BLAS on every core (load 61 on 12 cores) and the load gate then
    # blocked all dispatch. The launch now defaults OMP/OPENBLAS/MKL/NUMEXPR threads to 1; a job's own value wins.
    remote_root = tmp_path / "derisk-pool" / "sim"
    remote_root.mkdir(parents=True)
    for job, want in (("echo T=$OMP_NUM_THREADS/$OPENBLAS_NUM_THREADS", "T=1/1"),
                      ("OMP_NUM_THREADS=4 bash -c 'echo T=$OMP_NUM_THREADS/$OPENBLAS_NUM_THREADS'", "T=4/1")):
        out, status = remote_root / "autodispatch.out", remote_root / "job_status.log"
        for f in (out, status):
            f.unlink(missing_ok=True)
        rendered = run_bash(DISPATCHER, "--render-remote-command", job, env={"HOME": str(tmp_path)}).stdout
        env = {k: v for k, v in os.environ.items() if not k.endswith("_NUM_THREADS")}
        subprocess.run(["bash", "-c", rendered], env={**env, "HOME": str(tmp_path)}, text=True, check=True)
        for _ in range(200):
            if status.exists() and status.read_text():
                break
            time.sleep(0.02)
        assert out.read_text().strip() == want


def test_queue_add_waits_for_the_dispatchers_lock(tmp_path: Path) -> None:
    # 2026-09-24: `pool_queue.sh add` appended without the lock pop_job holds while it rewrites the queue
    # (awk > tmp; mv), so an add landing mid-rewrite went to the replaced file and was lost.
    import fcntl
    queue = tmp_path / "pool.queue"
    queue.write_text("")
    with open(str(queue) + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        p = subprocess.Popen(["bash", str(ROOT / "tools" / "pool_queue.sh"), "add", "echo lock-test",
                              "--checked", "test"], cwd=ROOT, env={**os.environ, "POOL_QUEUE_PATH": str(queue)},
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        time.sleep(1.0)
        assert p.poll() is None and queue.read_text() == ""       # blocked on the lock, nothing written
        fcntl.flock(lk, fcntl.LOCK_UN)
    assert p.wait(timeout=30) == 0
    assert "echo lock-test  #checked:test" in queue.read_text()


def test_pop_takes_first_job_that_fits_the_node_budget(tmp_path: Path) -> None:
    now = int(time.time())
    queue = tmp_path / "pool.queue"
    queue.write_text(f"{now}\tbig  #checked:r mem_gb=5\n{now}\tsmall  #checked:r mem_gb=1\n")
    env = {"POOL_QUEUE_PATH": str(queue), "POOL_RUNNING_PATH": str(tmp_path / "pool.running")}
    out = run_bash(DISPATCHER, "--pop-once", "3", env=env).stdout
    assert out.endswith("small")                      # the 5 GB head does not fit a 3 GB budget; the 1 GB job does
    assert "big" in queue.read_text() and "small" not in queue.read_text()
    assert run_bash(DISPATCHER, "--pop-once", "3", env=env).stdout == ""    # nothing left that fits
    assert run_bash(DISPATCHER, "--pop-once", env=env).stdout.endswith("big")  # no budget given -> head


def test_remote_wrapper_records_multiline_job_as_one_v2_row(tmp_path: Path) -> None:
    remote_root = tmp_path / "derisk-pool" / "sim"
    remote_root.mkdir(parents=True)
    queue = tmp_path / "pool.queue"
    job = "printf 'alpha\\nbeta\\tgamma\\n'\nexit 7"
    rendered = run_bash(
        DISPATCHER,
        "--render-remote-command",
        job,
        env={"POOL_QUEUE_PATH": str(queue), "HOME": str(tmp_path)},
    ).stdout

    subprocess.run(
        ["bash", "-c", rendered],
        env={**os.environ, "HOME": str(tmp_path)},
        text=True,
        check=True,
    )
    status = remote_root / "job_status.log"
    for _ in range(100):
        if status.exists() and status.read_text():
            break
        time.sleep(0.02)

    rows = status.read_text().splitlines()
    assert len(rows) == 1
    version, epoch, rc, payload = rows[0].split("\t")
    assert version == "v2"
    assert epoch.isdigit()
    assert rc == "7"
    assert base64.b64decode(payload).decode() == job


def test_status_classifier_rejects_malformed_and_stale_rows(tmp_path: Path) -> None:
    now = 2_000_000_000
    # The job carries an --out path: since 941f00105 (2026-08-26) a job with no output flag classifies as U
    # (unverifiable), and this test's no-flag job had read U ever since, failing on its stale "C" expectation.
    recent_job = "python -m research.runners.x --out research/x.json"
    stale_job = "pytest -q tests/test_old.py"
    log = tmp_path / "job_status.log"
    log.write_text(
        "test_name\tand\tnot\n"
        f"v2\t{now - 30}\t4\t{base64.b64encode(recent_job.encode()).decode()}\n"
        f"v2\t{now - 7200}\t2\t{base64.b64encode(stale_job.encode()).decode()}\n"
        "not-a-version\t1\t00:00:00\tbad\n"
    )

    result = run_bash(
        WORKFLOW,
        "--classify-pool-status",
        str(log),
        str(tmp_path),
        str(now),
        "3600",
    )

    assert result.stdout == f"C\t4\tresearch/x.json\t{recent_job}\n"


def test_legacy_status_time_is_anchored_to_file_mtime(tmp_path: Path) -> None:
    now = 2_000_000_000
    clock = time.strftime("%H:%M:%S", time.localtime(now - 60))
    log = tmp_path / "job_status.log"
    log.write_text(f"1\t{clock}\tlegacy crash\n")
    os.utime(log, (now - 7200, now - 7200))

    result = run_bash(
        WORKFLOW,
        "--classify-pool-status",
        str(log),
        str(tmp_path),
        str(now),
        "3600",
    )

    assert result.stdout == ""


def test_status_classifier_distinguishes_written_artifact(tmp_path: Path) -> None:
    now = 2_000_000_000
    relative_out = "research/findings/raw/result.json"
    artifact = tmp_path / relative_out
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}\n")
    job = f"runner --out {relative_out}"
    payload = base64.b64encode(job.encode()).decode()
    log = tmp_path / "job_status.log"
    log.write_text(f"v2\t{now - 5}\t1\t{payload}\n")

    result = run_bash(
        WORKFLOW,
        "--classify-pool-status",
        str(log),
        str(tmp_path),
        str(now),
        "3600",
    )

    assert result.stdout == f"V\t1\t{relative_out}\n"


def test_queue_health_excludes_dispatcher_stale_records(tmp_path: Path) -> None:
    now = 2_000_000_000
    queue = tmp_path / "pool.queue"
    queue.write_text(
        "# staged work\n"
        f"{now - 10}\trecent command #checked:test\n"
        f"{now + 10}\tfuture clock-skew command #checked:test\n"
        f"{now - 50_000}\tstale command #checked:test\n"
        "command-only malformed row\n"
    )

    result = run_bash(
        WORKFLOW,
        "--queue-health",
        str(queue),
        str(now),
        "43200",
    )

    assert result.stdout == "2\t1\t1\t3\n"


def test_no_ready_work_waiver_is_bounded_and_workboard_tied(tmp_path: Path) -> None:
    waiver = tmp_path / ".lane_waiver"
    board = tmp_path / "workboard.json"
    now = int(time.time())
    waiver.write_text(
        "scope=no-ready-work\n"
        "reason=all current CPU lanes are banked and no new question is authorized\n"
        "expiry=auto-6h\n"
    )
    board.write_text('{"lanes": {"gpu": {"status": "completed", "resource": "local_gpu"}}}\n')

    result = run_bash(WORKFLOW, "--no-ready-work", str(waiver), str(board), str(now))
    assert result.returncode == 0

    board.write_text(
        '{"lanes": {"cpu": {"status": "ready", "resource": "local_cpu_plus_pool"}}}\n'
    )
    result = subprocess.run(
        ["bash", str(WORKFLOW), "--no-ready-work", str(waiver), str(board), str(now)],
        cwd=ROOT,
        env=os.environ.copy(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 1

    board.write_text('{"lanes": {"gpu": {"status": "completed", "resource": "local_gpu"}}}\n')
    waiver.write_text("scope=no-ready-work\nreason=priority preference\n")
    result = subprocess.run(
        ["bash", str(WORKFLOW), "--no-ready-work", str(waiver), str(board), str(now)],
        cwd=ROOT,
        env=os.environ.copy(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 1


# ------------------------------------------------------------- workflow_check.sh honours AWS extra-nodes (fix round #2)

def test_workflow_check_pool_nodes_default_unaffected_when_no_extra_nodes_file(tmp_path: Path) -> None:
    res = run_bash(WORKFLOW, "--print-pool-check-nodes",
                   env={"POOL_EXTRA_NODES_FILE": str(tmp_path / "does-not-exist"),
                        "POOL_SSH_CONFIG": str(tmp_path / "does-not-exist")})
    assert res.returncode == 0, res.stderr
    lines = res.stdout.splitlines()
    assert lines[0].strip() == "pool40 pool41 pool42"
    assert lines[1] == "NOF"


def test_workflow_check_pool_nodes_grows_with_extra_nodes_file_and_honours_dash_f(tmp_path: Path) -> None:
    # THE DEFECT (re-review, MEDIUM): the CLUSTER/CRASH detectors (lines ~335/~427) were hardcoded to
    # pool40/41/42 and a bare ssh -- an AWS pool node's crash was never surfaced here.
    extra = tmp_path / "extra_nodes"
    extra.write_text("pool1\n")
    config = tmp_path / "ssh_config"
    config.write_text("Include ~/.ssh/config\n")
    res = run_bash(WORKFLOW, "--print-pool-check-nodes",
                   env={"POOL_EXTRA_NODES_FILE": str(extra), "POOL_SSH_CONFIG": str(config)})
    assert res.returncode == 0, res.stderr
    lines = res.stdout.splitlines()
    assert lines[0].strip() == "pool40 pool41 pool42 pool1"
    assert lines[1] == "F"


# --------------------------------------------------------------- revision-dir-aware pop_job (2026-09-23 fix)

def _write_ssh_stub_answering_dash_d(tmp_path: Path, missing_shas: set[str]):
    """A stub `ssh` that logs its argv and answers the `[ -f ~/derisk-pool/revisions/<sha>/.provisioned_ok ]`
    probe (fix round #2: a completion MARKER, not bare directory existence -- see revision_available's
    docstring): exit 1 (not provisioned) for any sha in `missing_shas`, else exit 0 (provisioned). Any other
    command exits 0 too, so this also stands in for the plain reachability calls this file's other tests don't
    otherwise exercise."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "ssh.log"
    log.write_text("")
    missing = " ".join(sorted(missing_shas))
    stub = bin_dir / "ssh"
    stub.write_text(f"""#!/usr/bin/env bash
echo "$*" >> "{log}"
for m in {missing}; do
  case "$*" in
    *"revisions/$m"*) exit 1 ;;
  esac
done
exit 0
""")
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return bin_dir, log


def test_pop_job_skips_revision_pinned_job_missing_on_node_leaving_it_queued(tmp_path: Path) -> None:
    # REGRESSION (2026-09-23 fix round): the AWS pool node was provisioned only at ~/derisk-pool/sim from HEAD,
    # but every queued job was pinned to `cd ~/derisk-pool/revisions/<sha> && ...` (pool_provision.sh
    # --isolated). The old pop_job popped the head-fitting job REGARDLESS, so it was removed from the queue,
    # dispatched, `cd` failed on the node, and the result (recorded only in the node's own job_status.log, which
    # pool_sync never pulls) was silently lost. pop_job must SKIP such a job for a node lacking the revision --
    # leaving it in the queue for a node that can actually run it -- not pop-and-lose it.
    now = int(time.time())
    sha = "abc1234"
    queue = tmp_path / "pool.queue"
    queue.write_text(f"{now}\tcd ~/derisk-pool/revisions/{sha} && bash run.sh  #checked:r mem_gb=1\n")
    bin_dir, ssh_log = _write_ssh_stub_answering_dash_d(tmp_path, missing_shas={sha})
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ.get('PATH', '')}",
        "POOL_QUEUE_PATH": str(queue),
        "POOL_RUNNING_PATH": str(tmp_path / "pool.running"),
        "POOL_SSH_CONFIG": str(tmp_path / "does-not-exist"),
    }
    res = subprocess.run(["bash", str(DISPATCHER), "--pop-once", "999", "pool1"],
                          cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
    assert res.returncode == 0, res.stderr
    assert res.stdout == ""                     # nothing handed out -- pool1 cannot run this job
    assert sha in queue.read_text()              # the job was NEVER removed from the queue -- not lost
    assert f"[ -f ~/derisk-pool/revisions/{sha}/.provisioned_ok ]" in ssh_log.read_text()


def test_pop_job_hands_out_revision_pinned_job_when_the_revision_is_present(tmp_path: Path) -> None:
    # The mirror case: a node that DOES have the revision gets the job normally (queue entry removed, job text
    # returned), so the fix does not just make every revision-pinned job unrunnable everywhere.
    now = int(time.time())
    sha = "abc1234"
    queue = tmp_path / "pool.queue"
    queue.write_text(f"{now}\tcd ~/derisk-pool/revisions/{sha} && bash run.sh  #checked:r mem_gb=1\n")
    bin_dir, ssh_log = _write_ssh_stub_answering_dash_d(tmp_path, missing_shas=set())
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ.get('PATH', '')}",
        "POOL_QUEUE_PATH": str(queue),
        "POOL_RUNNING_PATH": str(tmp_path / "pool.running"),
        "POOL_SSH_CONFIG": str(tmp_path / "does-not-exist"),
    }
    res = subprocess.run(["bash", str(DISPATCHER), "--pop-once", "999", "pool40"],
                          cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
    assert res.returncode == 0, res.stderr
    assert sha in res.stdout
    assert sha not in queue.read_text()          # popped -- removed from the queue


def _write_ssh_stub_that_forwards_stdin_like_real_ssh(tmp_path: Path, missing_sha: str):
    """A stub `ssh` that actually DRAINS its own stdin before answering -- unlike every other stub `ssh` in this
    file (`echo "$*" >> log; exit N`, which never touches stdin at all). Real `ssh`, run non-interactively
    WITHOUT `-n`, still opens and forwards its local stdin to the remote command; only `-n` (or a `</dev/null`
    redirect) stops it. This is the one behaviour needed to reproduce the 2026-09-25 incident (pool1+pool2
    starved 07:35-09:59 EDT: 74 already-runnable mem_gb=8 B2b jobs sat queued behind ONE job pinned to a
    not-yet-provisioned revision) -- see revision_available()'s own comment in tools/pool_autodispatch.sh."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "ssh.log"
    log.write_text("")
    stub = bin_dir / "ssh"
    stub.write_text(f"""#!/usr/bin/env bash
echo "$*" >> "{log}"
has_n=0
for a in "$@"; do [ "$a" = "-n" ] && has_n=1; done
# Mirror real ssh: with no -n (and no caller-side </dev/null), it forwards local stdin to the remote side --
# here that means draining whatever the CALLER's fd 0 happens to be at this moment.
[ "$has_n" = 0 ] && cat >/dev/null
case "$*" in
  *"revisions/{missing_sha}"*) exit 1 ;;
esac
exit 0
""")
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return bin_dir, log


def test_pop_job_does_not_let_an_unavailable_revision_probe_swallow_later_queued_candidates(tmp_path: Path) -> None:
    # THE 2026-09-25 INCIDENT: pop_job's revision-pinned check runs INSIDE `while IFS= read -r cand; do ... done
    # < <(awk ...)` -- one candidate per iteration, `continue` past a candidate this node can't run. revision_
    # available()'s ssh call used to omit `-n`, so real ssh (which still opens/forwards ITS OWN stdin even
    # non-interactively) drained the SAME pipe the enclosing `read` was consuming from, on the very first probe
    # -- silently truncating the scan to that one candidate. pop_job returned empty every time, discarding every
    # OTHER admissible job behind it (this queue's second, unpinned, perfectly runnable line), for as long as
    # the blocking revision stayed unprovisioned -- exactly what happened to pool1 AND pool2 (pool2 was never
    # AWS-idle-stopped, ruling out an AWS-specific cause) for ~2.5 h with 74 ready mem_gb=8 B2b lines stranded
    # behind one line pinned to a different, not-yet-provisioned revision.
    now = int(time.time())
    missing_sha = "5b5ea1b"
    queue = tmp_path / "pool.queue"
    queue.write_text(
        f"{now}\tcd ~/derisk-pool/revisions/{missing_sha} && bash blocked.sh  #checked:r mem_gb=1\n"
        f"{now + 1}\tbash good.sh  #checked:r mem_gb=1\n"
    )
    bin_dir, ssh_log = _write_ssh_stub_that_forwards_stdin_like_real_ssh(tmp_path, missing_sha)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ.get('PATH', '')}",
        "POOL_QUEUE_PATH": str(queue),
        "POOL_RUNNING_PATH": str(tmp_path / "pool.running"),
        "POOL_SSH_CONFIG": str(tmp_path / "does-not-exist"),
    }
    res = subprocess.run(["bash", str(DISPATCHER), "--pop-once", "999", "pool1"],
                          cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
    assert res.returncode == 0, res.stderr
    assert "good.sh" in res.stdout, (
        f"pop_job returned {res.stdout!r} -- the unavailable-revision probe swallowed the rest of the queue "
        "scan instead of just being skipped (the 2026-09-25 starvation bug)"
    )
    assert missing_sha in queue.read_text()      # the blocked candidate was never popped -- still queued
    assert "good.sh" not in queue.read_text()    # the good candidate WAS popped -- removed from the queue
    assert " -n " in ssh_log.read_text() or ssh_log.read_text().strip().startswith("-n")


def test_pop_job_without_a_node_arg_skips_the_revision_check_entirely(tmp_path: Path) -> None:
    # Backward compatibility: test seams / callers that never pass [node] (this file's other --pop-once tests)
    # must see UNCHANGED size-only selection -- no ssh call at all, even for a revision-pinned candidate.
    now = int(time.time())
    sha = "abc1234"
    queue = tmp_path / "pool.queue"
    queue.write_text(f"{now}\tcd ~/derisk-pool/revisions/{sha} && bash run.sh  #checked:r mem_gb=1\n")
    bin_dir, ssh_log = _write_ssh_stub_answering_dash_d(tmp_path, missing_shas={sha})
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ.get('PATH', '')}",
        "POOL_QUEUE_PATH": str(queue),
        "POOL_RUNNING_PATH": str(tmp_path / "pool.running"),
    }
    res = subprocess.run(["bash", str(DISPATCHER), "--pop-once", "999"],
                          cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
    assert res.returncode == 0, res.stderr
    assert sha in res.stdout                     # popped -- no node given, no revision gate applied
    assert ssh_log.read_text() == ""             # and no ssh call was made to check


def _write_ssh_stub_dir_exists_but_no_marker(tmp_path: Path, sha: str):
    """REGRESSION (2026-09-23 fix round #2): a stub answering `[ -d ... ]` TRUE (dir exists -- the defect: a
    half-provisioned revision from a FAILED pool_provision.sh run always leaves this true) but `[ -f
    .../.provisioned_ok ]` FALSE (never reached the completion marker). Reproduces the exact bug the re-review
    found: the OLD `-d`-only check would have handed this node the job; the fixed `-f` marker check must not."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "ssh.log"
    log.write_text("")
    stub = bin_dir / "ssh"
    stub.write_text(f"""#!/usr/bin/env bash
echo "$*" >> "{log}"
case "$*" in
  *"[ -d ~/derisk-pool/revisions/{sha} ]"*) exit 0 ;;
  *"provisioned_ok"*) exit 1 ;;
esac
exit 0
""")
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return bin_dir, log


def test_pop_job_treats_a_half_provisioned_revision_dir_as_unavailable(tmp_path: Path) -> None:
    # THE ACTUAL DEFECT (re-review, fix round #2): pool_provision.sh's remote `mkdir -p` creates the revision
    # directory FIRST, before rsync/venv/manifest-verify/sanity-check -- a later failure `continue`s past the
    # node WITHOUT removing it. A bare `-d` check therefore reads "available" for a node whose provision never
    # finished (missing .venv, unverified source, a degenerate sanity build). The fix requires a completion
    # marker (`.provisioned_ok`, written as pool_provision.sh's LAST step for that node) instead.
    now = int(time.time())
    sha = "deadbee"
    queue = tmp_path / "pool.queue"
    queue.write_text(f"{now}\tcd ~/derisk-pool/revisions/{sha} && bash run.sh  #checked:r mem_gb=1\n")
    bin_dir, ssh_log = _write_ssh_stub_dir_exists_but_no_marker(tmp_path, sha)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ.get('PATH', '')}",
        "POOL_QUEUE_PATH": str(queue),
        "POOL_RUNNING_PATH": str(tmp_path / "pool.running"),
        "POOL_SSH_CONFIG": str(tmp_path / "does-not-exist"),
    }
    res = subprocess.run(["bash", str(DISPATCHER), "--pop-once", "999", "pool1"],
                          cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
    assert res.returncode == 0, res.stderr
    assert res.stdout == ""                      # NOT handed out despite the directory existing
    assert sha in queue.read_text()               # left queued for a genuinely-provisioned node
    assert "provisioned_ok" in ssh_log.read_text()


def test_revision_available_cached_probes_ssh_only_once_per_node_sha_pair(tmp_path: Path) -> None:
    # LOW (re-review): pop_job's revision check used to ssh EVERY candidate popped, even for a sha it had
    # already checked against this node earlier in the SAME cycle. Exercised directly here across THREE calls in
    # one process, extracted verbatim from the real script (never re-typed) so this cannot drift from the code
    # it is meant to pin. NOTE: this alone is NOT sufficient evidence the live dispatcher benefits from the
    # cache -- see test_fill_node_probes_a_repeated_revision_only_once_per_cycle below, which goes through the
    # actual `JOB=$(pop_job ...)` command-substitution path the production loop uses (fix round #3, re-review
    # HIGH: a fix round #2 array-based cache passed a test shaped exactly like this one while doing nothing for
    # the live dispatcher, because it never survives that subshell boundary).
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    ssh_log = tmp_path / "ssh.log"
    ssh_log.write_text("")
    stub = bin_dir / "ssh"
    stub.write_text(f"""#!/usr/bin/env bash
echo "$*" >> "{ssh_log}"
exit 0
""")
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    cache_file = tmp_path / "revcache"
    harness = tmp_path / "harness.sh"
    harness.write_text(f"""#!/usr/bin/env bash
set -uo pipefail
SSH_F=()
REV_CACHE_FILE={cache_file}
: > "$REV_CACHE_FILE"
source {ROOT}/tools/pool_revision_marker.sh
eval "$(sed -n '/^revision_available()/,/^}}/p; /^revision_available_cached()/,/^}}/p' {DISPATCHER})"
revision_available_cached node1 abc1234 >/dev/null 2>&1
revision_available_cached node1 abc1234 >/dev/null 2>&1
revision_available_cached node1 abc1234 >/dev/null 2>&1
""")
    harness.chmod(harness.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ.get('PATH', '')}"}
    res = subprocess.run(["bash", str(harness)], cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
    assert res.returncode == 0, res.stderr
    calls = [ln for ln in ssh_log.read_text().splitlines() if "abc1234" in ln]
    assert len(calls) == 1, f"expected exactly ONE ssh probe for a repeated (node, sha) pair, got {len(calls)}: {calls}"


def _write_node_is_idle_and_revision_stub(tmp_path: Path, missing_shas: set[str] = frozenset()):
    """A stub `ssh` that answers BOTH calls fill_node's real loop makes: the node_is_idle metrics probe
    (detected by the 'MemAvailable' marker unique to that command) with a fixed idle-and-roomy reading, and the
    `.provisioned_ok` marker probe per revision_marker_probe_cmd (exit 1 for any sha in `missing_shas`, else exit
    0). Any other ssh call (the `-f -n` background launch) is a plain no-op success."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "ssh.log"
    log.write_text("")
    missing = " ".join(sorted(missing_shas)) or "__none__"
    stub = bin_dir / "ssh"
    stub.write_text(f"""#!/usr/bin/env bash
echo "$*" >> "{log}"
case "$*" in
  *MemAvailable*) echo "8 1 0 20 1 32"; exit 0 ;;
  *provisioned_ok*)
    for m in {missing}; do
      case "$*" in *"revisions/$m/"*) exit 1 ;; esac
    done
    exit 0 ;;
esac
exit 0
""")
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return bin_dir, log


def test_fill_node_probes_a_repeated_revision_only_once_per_cycle(tmp_path: Path) -> None:
    # THE ACTUAL DEFECT (re-review, HIGH, "claimed fix ineffective in production"): the fix round #2 REV_CACHE
    # was a plain bash associative array populated inside pop_job, but the production loop calls
    # `JOB=$(pop_job ...)` -- a command-substitution SUBSHELL -- so every write to that array was discarded the
    # instant the subshell exited, and the live dispatcher re-probed by ssh on EVERY pop regardless. This test
    # drives `fill_node` (the SAME function the production `while true` loop calls) via the --fill-node seam, so
    # it goes through the real subshell boundary the old test never did. Two jobs, same node, same revision.
    now = int(time.time())
    sha = "cafefeed"
    queue = tmp_path / "pool.queue"
    queue.write_text(
        f"{now}\tcd ~/derisk-pool/revisions/{sha} && bash run.sh --a  #checked:r mem_gb=1\n"
        f"{now}\tcd ~/derisk-pool/revisions/{sha} && bash run.sh --b  #checked:r mem_gb=1\n"
    )
    bin_dir, ssh_log = _write_node_is_idle_and_revision_stub(tmp_path)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ.get('PATH', '')}",
        "POOL_QUEUE_PATH": str(queue),
        "POOL_RUNNING_PATH": str(tmp_path / "pool.running"),
        "POOL_RESERVATIONS_PATH": str(tmp_path / "resv"),
        "POOL_SSH_CONFIG": str(tmp_path / "does-not-exist"),
        "POOL_DISPATCH_LAUNCH_SLEEP": "0",
    }
    res = subprocess.run(["bash", str(DISPATCHER), "--fill-node", "pool1"],
                          cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
    assert res.returncode == 0, res.stderr
    assert queue.read_text().strip() == ""   # both jobs dispatched -- the node was never wrongly skipped
    revision_calls = [ln for ln in ssh_log.read_text().splitlines() if "provisioned_ok" in ln]
    assert len(revision_calls) == 1, (
        f"expected exactly ONE revision probe across BOTH pop_job calls within fill_node (the per-cycle cache "
        f"must survive the $(pop_job ...) subshell boundary), got {len(revision_calls)}: {revision_calls}"
    )


def test_fill_node_leaves_job_queued_when_node_lacks_the_revision(tmp_path: Path) -> None:
    # Mirror case, through the same fill_node/--fill-node path: a node missing the revision must never pop the
    # job (it would be removed from the queue with nowhere able to run it).
    now = int(time.time())
    sha = "cafefeed"
    queue = tmp_path / "pool.queue"
    queue.write_text(f"{now}\tcd ~/derisk-pool/revisions/{sha} && bash run.sh  #checked:r mem_gb=1\n")
    bin_dir, ssh_log = _write_node_is_idle_and_revision_stub(tmp_path, missing_shas={sha})
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ.get('PATH', '')}",
        "POOL_QUEUE_PATH": str(queue),
        "POOL_RUNNING_PATH": str(tmp_path / "pool.running"),
        "POOL_RESERVATIONS_PATH": str(tmp_path / "resv"),
        "POOL_SSH_CONFIG": str(tmp_path / "does-not-exist"),
        "POOL_DISPATCH_LAUNCH_SLEEP": "0",
    }
    res = subprocess.run(["bash", str(DISPATCHER), "--fill-node", "pool1"],
                          cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
    assert res.returncode == 0, res.stderr
    assert sha in queue.read_text()   # left queued -- never popped for a node that cannot run it


# --------------------------------------------------------------------- SSH_F re-evaluated every cycle (fix)

def test_ssh_f_is_refreshed_mid_loop_without_a_restart(tmp_path: Path) -> None:
    # REGRESSION (2026-09-23 fix round): SSH_F used to be computed ONCE at process start. The live systemd
    # dispatcher started before any AWS node existed kept calling bare `ssh pool1` (no Host entry for pool1 in
    # the user's own ~/.ssh/config) even AFTER `aws_pool_node.sh up` created .pool_ssh_config -- so pool1 was
    # never reachable while it billed, and "no restart needed" was false. Drive the same loop SHAPE (3 cycles,
    # via the --print-ssh-f-loop test seam, which calls the real refresh_ssh_f each cycle with no queue/ssh side
    # effects) and create the config file BETWEEN cycles 1 and 2, inside ONE long-lived process.
    config = tmp_path / "ssh_config"
    proc = subprocess.Popen(
        ["bash", str(DISPATCHER), "--print-ssh-f-loop", "3", "1"],
        cwd=ROOT, env={**os.environ, "POOL_SSH_CONFIG": str(config)},
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        line1 = proc.stdout.readline().strip()
        assert line1 == "NOF", f"cycle 1 should see no config yet, got {line1!r}"
        config.write_text("Include ~/.ssh/config\n")   # create it WHILE the process is still running
        line2 = proc.stdout.readline().strip()
        assert line2 == "F", f"cycle 2 should pick up the now-existing config without a restart, got {line2!r}"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


# --------------------------------------------------- stale-HostName auto-refresh, dispatcher side (2026-09-25)

def _write_aws_pool_state(state_dir: Path, node: str, key: Path, instance: str = "i-aaa") -> Path:
    key.write_text("fake key\n")
    state = state_dir / f".aws_{node}"
    state.write_text(f"instance={instance}\nregion=us-east-1\nkey={key}\nsg=sg-x\n")
    return state


def _make_always_unreachable_ssh_stub(tmp_path: Path) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "ssh"
    stub.write_text("#!/usr/bin/env bash\nexit 255\n")
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return bin_dir


def _make_refresh_aws_stub(bin_dir: Path, log: Path, new_ip: str = "9.9.9.9") -> None:
    stub = bin_dir / "aws"
    stub.write_text(f"""#!/usr/bin/env bash
echo "$*" >> "{log}"
case "$*" in
  *"State.Name"*) echo running; exit 0 ;;
  *"PublicIpAddress"*) echo "{new_ip}"; exit 0 ;;
esac
echo ok; exit 0
""")
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def test_node_is_idle_self_heals_a_stale_hostname_for_an_aws_managed_node(tmp_path: Path) -> None:
    # LOW (2026-09-25 review): "spec gap" -- only tools/pool_sync.sh's post-sync-failure retry self-healed a
    # stale ip (an AWS pool node's public ip changes on EVERY stop/start, no Elastic IP in this feature); the
    # dispatcher's OWN node_is_idle probe never did, though the spec says every entry point that finds a stale
    # HostName should. A node with a research/queue/.aws_<name> state file, unreachable at its recorded (stale)
    # ip, must trigger ONE `aws_pool_node.sh refresh` -- which rewrites the Host block to the current ip -- even
    # though THIS cycle's capacity check still reports busy/unreachable (the refresh helps the NEXT cycle).
    bin_dir = _make_always_unreachable_ssh_stub(tmp_path)
    aws_log = tmp_path / "aws.log"; aws_log.write_text("")
    _make_refresh_aws_stub(bin_dir, aws_log, new_ip="9.9.9.9")

    state_dir = tmp_path / "state"; state_dir.mkdir()
    _write_aws_pool_state(state_dir, "testnode", tmp_path / "key.pem")
    ssh_config = tmp_path / "pool_ssh_config"
    ssh_config.write_text("Include ~/.ssh/config\nHost testnode\n  HostName 1.2.3.4\n")

    res = run_bash(DISPATCHER, "--node-budget", "testnode", env={
        "PATH": f"{bin_dir}:{os.environ.get('PATH', '')}",
        "POOL_QUEUE_PATH": str(tmp_path / "pool.queue"),
        "POOL_SSH_CONFIG": str(ssh_config),
        "POOL_AWS_STATE_DIR": str(state_dir),
        "POOL_STALE_REFRESH_MARK_DIR": str(tmp_path / "marks"),
    })
    assert "busy/unreachable" in res.stdout   # this cycle's OWN capacity check still failed (ssh never answers)
    assert "State.Name" in aws_log.read_text() and "PublicIpAddress" in aws_log.read_text()
    assert "9.9.9.9" in ssh_config.read_text(), "the Host block was never refreshed to the current ip"
    assert "1.2.3.4" not in ssh_config.read_text()


def test_node_is_idle_refresh_is_rate_limited(tmp_path: Path) -> None:
    # A refresh attempt costs an `aws describe-instances` round trip; node_is_idle runs on every dispatch poll
    # (default every 60s) for every configured node, so an un-rate-limited refresh would hammer `aws` every
    # single cycle for a node that stays unreachable for a mundane reason. A second call within the rate-limit
    # window must NOT call `aws` again.
    bin_dir = _make_always_unreachable_ssh_stub(tmp_path)
    aws_log = tmp_path / "aws.log"; aws_log.write_text("")
    _make_refresh_aws_stub(bin_dir, aws_log, new_ip="9.9.9.9")

    state_dir = tmp_path / "state"; state_dir.mkdir()
    _write_aws_pool_state(state_dir, "testnode", tmp_path / "key.pem")
    ssh_config = tmp_path / "pool_ssh_config"
    ssh_config.write_text("Include ~/.ssh/config\nHost testnode\n  HostName 1.2.3.4\n")
    env = {
        "PATH": f"{bin_dir}:{os.environ.get('PATH', '')}",
        "POOL_QUEUE_PATH": str(tmp_path / "pool.queue"),
        "POOL_SSH_CONFIG": str(ssh_config),
        "POOL_AWS_STATE_DIR": str(state_dir),
        "POOL_STALE_REFRESH_MARK_DIR": str(tmp_path / "marks"),
        "POOL_STALE_REFRESH_RATE_S": "300",
    }
    run_bash(DISPATCHER, "--node-budget", "testnode", env=env)
    first_call_count = len(aws_log.read_text().splitlines())
    assert first_call_count > 0, "the first call never even attempted a refresh"

    run_bash(DISPATCHER, "--node-budget", "testnode", env=env)   # immediately again -- well within the window
    assert len(aws_log.read_text().splitlines()) == first_call_count, (
        "a second call within the rate-limit window made another `aws` call -- not rate-limited")


def test_node_is_idle_never_attempts_refresh_for_a_minipc_node_without_an_aws_state_file(tmp_path: Path) -> None:
    # pool40/41/42 have no research/queue/.aws_<name> state file -- must stay byte-identical to before this
    # feature: node_is_idle fails exactly as before, no `aws` call at all (no `aws` binary on PATH needed).
    bin_dir = _make_always_unreachable_ssh_stub(tmp_path)
    res = run_bash(DISPATCHER, "--node-budget", "pool40", env={
        "PATH": f"{bin_dir}:{os.environ.get('PATH', '')}",
        "POOL_QUEUE_PATH": str(tmp_path / "pool.queue"),
        "POOL_SSH_CONFIG": str(tmp_path / "does-not-exist"),
        "POOL_AWS_STATE_DIR": str(tmp_path / "state"),   # dir does not even exist
        "POOL_STALE_REFRESH_MARK_DIR": str(tmp_path / "marks"),
    })
    assert "busy/unreachable" in res.stdout


def test_queue_flag_check_never_pipes_help_into_grep_q() -> None:
    # 2026-09-24: under `set -o pipefail`, `printf '%s' "$HELP" | grep -q FLAG` FAILS whenever grep exits before printf
    # has written a help text larger than the 64 KB pipe buffer (SIGPIPE, rc 141): a 72 KB --help reported present
    # flags as missing, a different subset on every call, and refused 12 valid pool jobs. Here-strings only.
    import re
    src = (ROOT / "tools" / "pool_queue.sh").read_text()
    code = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))
    assert "set -uo pipefail" in src
    assert not re.search(r"printf[^|\n]*\$HELP[^|\n]*\|\s*grep\s+-q", code)
    big = "usage: x\n" + ("--padding-flag-xyz  " * 5000) + "\n--wanted-flag\n"
    r = subprocess.run(["bash", "-c", 'set -uo pipefail; H="$1"; grep -q -- --wanted-flag <<<"$H" && echo ok', "_", big],
                       text=True, capture_output=True)
    assert r.stdout.strip() == "ok" and len(big) > 65536


def test_queue_checks_pinned_jobs_against_the_pinned_revision() -> None:
    # 2026-09-24: `add` checked a revision-pinned job's flags against MAIN's runner, so every unmerged branch's 6-seed
    # job was refused ("does not even import/parse"). Pinned jobs now skip the local check and are checked against the
    # pinned revision's own --help on the node (verified by hand: a bogus flag is refused, the real lines queue).
    src = (ROOT / "tools" / "pool_queue.sh").read_text()
    assert 'if [ -n "$MOD" ] && [ -z "$PINNED_REV" ]; then' in src          # local check only when NOT pinned
    assert "at the pinned revision does not accept" in src                  # remote flag check exists
    assert src.index('PINNED_REV=$(') < src.index('HELP=$(cd "$ROOT"')     # decided before the local --help runs


def test_queue_add_front_puts_the_line_at_the_head(tmp_path: Path) -> None:
    # 2026-09-24: short seed-7 checks had to wait behind a 186-shard battery in a FIFO queue. FRONT=1 prepends.
    queue = tmp_path / "pool.queue"
    queue.write_text("1\told-job  #checked:x\n")
    env = {**os.environ, "POOL_QUEUE_PATH": str(queue)}
    subprocess.run(["bash", str(ROOT / "tools" / "pool_queue.sh"), "add", "echo back", "--checked", "b"], cwd=ROOT,
                   env=env, check=True, capture_output=True, text=True)
    subprocess.run(["bash", str(ROOT / "tools" / "pool_queue.sh"), "add", "echo front", "--checked", "f"], cwd=ROOT,
                   env={**env, "FRONT": "1"}, check=True, capture_output=True, text=True)
    lines = queue.read_text().splitlines()
    assert "echo front" in lines[0] and "old-job" in lines[1] and "echo back" in lines[2]
