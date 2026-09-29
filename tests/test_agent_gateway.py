"""tools/agent_gateway.sh -- the forced command that is the ENTIRE trust boundary between the sim-agent
container and the workstation's compute (docs/plans/2026-09-29-local-model-transition-PLAN.md section 9).
Hermetic throughout: AGENT_GATEWAY_ROOT points every test at a scratch directory holding stub tools/*.sh
(never the real pool_queue.sh/gpu_add.sh/gpu_queue.sh/game.sh/llm.sh), and AGENT_GATEWAY_SSH stubs `ssh`
entirely -- no real ssh connection, no real pool/GPU job, ever touches this repo's live queues.
"""
import os
import stat
import subprocess
import tempfile

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_GATEWAY = os.path.join(_ROOT, "tools", "agent_gateway.sh")


def _write_exec(path, body):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("#!/usr/bin/env bash\n" + body)
    os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return path


def _scaffold(tmp):
    """A minimal $ROOT/tools/... tree the gateway's hard-coded relative paths expect, each stub just logging
    its own invocation."""
    log = os.path.join(tmp, "calls.log")
    _write_exec(os.path.join(tmp, "tools", "pool_queue.sh"), 'echo "pool_queue.sh $*" >> "%s"\n' % log)
    _write_exec(os.path.join(tmp, "tools", "local_llm", "gpu_add.sh"), 'echo "gpu_add.sh $*" >> "%s"\n' % log)
    _write_exec(os.path.join(tmp, "tools", "gpu_queue.sh"), 'echo "gpu_queue.sh $*" >> "%s"\n' % log)
    _write_exec(os.path.join(tmp, "tools", "game.sh"), 'echo "game.sh $*" >> "%s"\n' % log)
    _write_exec(os.path.join(tmp, "tools", "local_llm", "llm.sh"), 'echo "llm.sh $*" >> "%s"\n' % log)
    os.makedirs(os.path.join(tmp, "research", "findings", "raw"), exist_ok=True)
    os.makedirs(os.path.join(tmp, "research", "queue"), exist_ok=True)
    return log


def _fake_ssh(tmp, state_output="container: running\nactive_profile: workstation\nprofile_kind: local\nclaude_active: no\n", reachable=True):
    """Stands in for `ssh <host> <cmd...>`: logs the call, answers `state` with the given text, and for every
    other subcommand exits 0 (or nonzero, if reachable=False, simulating an unreachable ctl host)."""
    log = os.path.join(tmp, "ssh_calls.log")
    if not reachable:
        # a genuinely unreachable host: no stdout at all, just a nonzero exit (ssh's own ConnectTimeout
        # behaviour) -- still logs that the attempt was made, for the test's own assertions.
        body = 'echo "ssh $*" >> "%s"\nexit 1\n' % log
        return _write_exec(os.path.join(tmp, "fake_ssh.sh"), body), log
    body = """
echo "ssh $*" >> "%s"
args=("$@")
last="${args[-1]:-}"
case "$last" in
  state) cat <<'EOF'
%s
EOF
       exit 0 ;;
  *) exit 0 ;;
esac
""" % (log, state_output.rstrip("\n"))
    return _write_exec(os.path.join(tmp, "fake_ssh.sh"), body), log


def _run(cmd, tmp, ssh_path=None, extra_env=None):
    env = dict(os.environ)
    env["AGENT_GATEWAY_ROOT"] = tmp
    env["AGENT_GATEWAY_LOG"] = os.path.join(tmp, "gateway.log")
    env["SSH_ORIGINAL_COMMAND"] = cmd
    if ssh_path:
        env["AGENT_GATEWAY_SSH"] = ssh_path
    if extra_env:
        env.update(extra_env)
    return subprocess.run(["bash", _GATEWAY], env=env, capture_output=True, text=True, timeout=10)


def test_empty_command_is_refused():
    with tempfile.TemporaryDirectory() as tmp:
        _scaffold(tmp)
        res = _run("", tmp)
        assert res.returncode != 0
        assert "refused" in res.stderr.lower()


def test_unknown_subcommand_is_refused_and_logged():
    with tempfile.TemporaryDirectory() as tmp:
        _scaffold(tmp)
        res = _run("rm -rf /", tmp)
        assert res.returncode != 0
        assert "refused" in res.stderr.lower()
        log = open(os.path.join(tmp, "gateway.log")).read()
        assert "REFUSED" in log
        assert "rm -rf /" in log


def test_pool_submit_forwards_to_pool_queue_add():
    with tempfile.TemporaryDirectory() as tmp:
        log = _scaffold(tmp)
        res = _run("pool-submit 'cd ~ && echo hi' --checked 'corpus: nothing'", tmp)
        assert res.returncode == 0, res.stdout + res.stderr
        calls = open(log).read()
        assert "pool_queue.sh add cd ~ && echo hi --checked corpus: nothing" in calls


def test_gpu_submit_forwards_one_quoted_command():
    with tempfile.TemporaryDirectory() as tmp:
        log = _scaffold(tmp)
        res = _run("gpu-submit 'cd ~/derisk-pool/sim && run --seed 7'", tmp)
        assert res.returncode == 0, res.stdout + res.stderr
        calls = open(log).read()
        assert "gpu_add.sh cd ~/derisk-pool/sim && run --seed 7" in calls


def test_gpu_submit_refuses_more_than_one_argument():
    with tempfile.TemporaryDirectory() as tmp:
        _scaffold(tmp)
        res = _run("gpu-submit foo bar", tmp)
        assert res.returncode != 0
        assert "one quoted command" in res.stderr


def test_status_runs_all_three_status_checks():
    with tempfile.TemporaryDirectory() as tmp:
        log = _scaffold(tmp)
        res = _run("status", tmp)
        assert res.returncode == 0, res.stdout + res.stderr
        calls = open(log).read()
        assert "gpu_queue.sh status" in calls
        assert "pool_queue.sh list" in calls
        assert "game.sh status" in calls


def test_results_sync_allows_a_read_of_an_allowed_root():
    with tempfile.TemporaryDirectory() as tmp:
        _scaffold(tmp)
        rsync_log = os.path.join(tmp, "rsync_calls.log")
        _write_exec(os.path.join(tmp, "fakebin", "rsync"), 'echo "rsync $*" >> "%s"\n' % rsync_log)
        env = dict(os.environ)
        env["PATH"] = os.path.join(tmp, "fakebin") + os.pathsep + env["PATH"]
        res = _run(
            "results-sync --server --sender -logDtprze.iLsfxCIvu . research/findings/raw",
            tmp, extra_env={"PATH": env["PATH"]},
        )
        assert res.returncode == 0, res.stdout + res.stderr
        assert "research/findings/raw" in open(rsync_log).read()


def test_results_sync_refuses_a_write_direction():
    with tempfile.TemporaryDirectory() as tmp:
        _scaffold(tmp)
        res = _run("results-sync --server -logDtprze.iLsfxCIvu . research/findings/raw", tmp)
        assert res.returncode != 0
        assert "read" in res.stderr.lower()


def test_results_sync_refuses_a_path_outside_the_allowed_roots():
    with tempfile.TemporaryDirectory() as tmp:
        _scaffold(tmp)
        res = _run("results-sync --server --sender -logDtprze.iLsfxCIvu . research/coordination", tmp)
        assert res.returncode != 0
        assert "outside the allowed roots" in res.stderr


def test_results_sync_refuses_a_path_escaping_via_dotdot():
    with tempfile.TemporaryDirectory() as tmp:
        _scaffold(tmp)
        res = _run("results-sync --server --sender -logDtprze.iLsfxCIvu . research/../tools", tmp)
        assert res.returncode != 0
        assert "outside the allowed roots" in res.stderr


def test_yield_skips_pause_for_a_cloud_profile():
    with tempfile.TemporaryDirectory() as tmp:
        log = _scaffold(tmp)
        ssh_path, ssh_log = _fake_ssh(tmp, state_output="container: running\nprofile_kind: cloud\n")
        res = _run("yield --max-hours 4", tmp, ssh_path=ssh_path)
        assert res.returncode == 0, res.stdout + res.stderr
        assert "no gpu yield needed" in res.stdout.lower()
        # llm.sh yield must never have been called for a cloud backend
        assert "llm.sh yield" not in (open(log).read() if os.path.exists(log) else "")
        ssh_calls = open(ssh_log).read()
        assert "pause" not in ssh_calls   # only `state` was queried, no pause attempt


def test_yield_pauses_the_harness_then_calls_llm_yield_with_notify_cmd():
    with tempfile.TemporaryDirectory() as tmp:
        log = _scaffold(tmp)
        ssh_path, ssh_log = _fake_ssh(tmp, state_output="container: running\nprofile_kind: local\n")
        res = _run("yield --max-hours 6", tmp, ssh_path=ssh_path)
        assert res.returncode == 0, res.stdout + res.stderr
        ssh_calls = open(ssh_log).read()
        assert " pause" in ssh_calls or ssh_calls.strip().endswith("pause")
        calls = open(log).read()
        assert "llm.sh yield --max-hours 6 --notify-cmd" in calls
        assert "unpause" in calls and " say " in calls   # the notify-cmd itself names both steps


def test_yield_continues_gracefully_when_the_ctl_host_is_unreachable():
    with tempfile.TemporaryDirectory() as tmp:
        log = _scaffold(tmp)
        ssh_path, _ = _fake_ssh(tmp, reachable=False)
        res = _run("yield", tmp, ssh_path=ssh_path)
        # unreachable ctl -> harness_kind can't be determined -> NOT cloud -> still attempts llm.sh yield
        assert res.returncode == 0, res.stdout + res.stderr
        assert "llm.sh yield" in open(log).read()
