"""tools/game.sh (rewritten 2026-09-29 for the local-model-driver setup, docs/plans/2026-09-29-local-model-
transition-PLAN.md section 3): `on` pauses the GPU queue, SIGSTOPs any standalone local research.runners/webapp
python (recording pids for `off` to SIGCONT), stops the local-llm server if it was loaded (recording its profile
+ last session so `off` can bring both back), and sets GAME_MODE; `off` reverses all three, then resumes the
recorded session -- interactively from a TTY, headless (logged) otherwise.

Hermetic throughout: GPU_QUEUE_DIR isolates every state file (GAME_MODE, the frozen-pid list, the llm-state
record) from the live research/queue/; GAME_LLM_SH stubs tools/local_llm/llm.sh entirely (never spawns a real
llama-server/systemd unit); tools/gpu_queue.sh's own `pause --now`/`resume`/`status` are exercised for real but
only ever touch the isolated GPU_QUEUE_DIR. Frozen "processes" are real, short-lived `sleep` children this file
spawns and kills itself -- never anything from the live system. GAME_SSH stubs `ssh` too (the sim-agent harness
pause/unpause, 2026-09-29): `_run`'s own default is a stub that fails INSTANTLY (simulating an unreachable ctl
host, never a real network call) unless a test supplies its own via `env_extra`.
"""
import atexit
import os
import stat
import subprocess
import tempfile
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_GAME = os.path.join(_ROOT, "tools", "game.sh")


def _write_exec(path, body):
    with open(path, "w") as f:
        f.write("#!/bin/bash\n" + body)
    os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return path


_DEFAULT_SSH_DIR = tempfile.mkdtemp(prefix="game_test_default_ssh_")
atexit.register(lambda: __import__("shutil").rmtree(_DEFAULT_SSH_DIR, ignore_errors=True))
_DEFAULT_SSH = _write_exec(os.path.join(_DEFAULT_SSH_DIR, "fake_ssh_unreachable.sh"), "exit 1\n")


def _fake_llm_sh(tmp, up=False, profile="prof-1"):
    """A stand-in for tools/local_llm/llm.sh: `status`'s exit/output is irrelevant here (game.sh never parses
    it), `__current_profile` answers the given profile, and `on`/`claude`/`__stop_graceful` just log their full
    invocation to `log` and exit 0."""
    log = os.path.join(tmp, "fake_llm_sh.log")
    _write_exec(os.path.join(tmp, "fake_llm.sh"), """
echo "CALLED: $*" >> "%s"
case "${1:-}" in
  __current_profile) echo "%s" ;;
  *) exit 0 ;;
esac
""" % (log, profile))
    return os.path.join(tmp, "fake_llm.sh"), log


def _fake_curl(tmp, up):
    bindir = os.path.join(tmp, "fakebin")
    os.makedirs(bindir, exist_ok=True)
    _write_exec(os.path.join(bindir, "curl"), "exit %d\n" % (0 if up else 1))
    return bindir


def _fake_systemctl(tmp, active):
    """A stand-in for `systemctl --user is-active --quiet local-llm` (2026-09-29 fix, same shape as
    tests/test_llm_gpu_yield.py's own _fake_systemctl): answers ONLY the unit's real active/inactive state,
    completely independent of whatever `curl` is stubbed to say -- the whole point of the fix under test."""
    return _write_exec(os.path.join(tmp, "fake_systemctl.sh"), """
case "$*" in
  *"is-active --quiet local-llm"*) exit %d ;;
  *) exit 0 ;;
esac
""" % (0 if active else 1))


def _log_text(path):
    return open(path).read() if os.path.exists(path) else ""


def _run(args, env_extra, timeout=10):
    env = dict(os.environ)
    env["GAME_SSH"] = _DEFAULT_SSH   # overridden below if env_extra sets its own
    env.update(env_extra)
    return subprocess.run(["bash", _GAME] + args, env=env, capture_output=True, text=True, timeout=timeout)


def _spawn_sleeper():
    return subprocess.Popen(["sleep", "60"])


def _is_stopped(pid):
    """True if the process's /proc status shows it is genuinely SIGSTOP'd."""
    try:
        with open("/proc/%d/stat" % pid) as f:
            # field 3 (after the parenthesised comm) is the state letter; 'T' == stopped.
            return f.read().split(") ", 1)[1].split(" ", 1)[0] == "T"
    except FileNotFoundError:
        return False


def test_on_pauses_the_queue_and_sets_game_mode():
    with tempfile.TemporaryDirectory() as tmp:
        llm_sh, _ = _fake_llm_sh(tmp, up=False)
        bindir = _fake_curl(tmp, up=False)
        systemctl = _fake_systemctl(tmp, active=False)
        env = dict(os.environ)
        env["PATH"] = bindir + os.pathsep + env["PATH"]
        env.update({"GPU_QUEUE_DIR": tmp, "GAME_LLM_SH": llm_sh, "LOCAL_LLM_SYSTEMCTL": systemctl})
        res = _run(["on"], env)
        assert res.returncode == 0, res.stdout + res.stderr
        assert os.path.exists(os.path.join(tmp, "GAME_MODE"))
        assert os.path.exists(os.path.join(tmp, "GPU_PAUSE")), "gpu_queue.sh pause --now must have run"


def test_on_off_round_trip_freezes_and_resumes_a_standalone_local_process():
    """SIGSTOP on `on`, SIGCONT on `off` -- proving pids are recorded and genuinely revived, not killed."""
    with tempfile.TemporaryDirectory() as tmp:
        llm_sh, _ = _fake_llm_sh(tmp, up=False)
        bindir = _fake_curl(tmp, up=False)
        systemctl = _fake_systemctl(tmp, active=False)
        env = dict(os.environ)
        env["PATH"] = bindir + os.pathsep + env["PATH"]
        env.update({"GPU_QUEUE_DIR": tmp, "GAME_LLM_SH": llm_sh, "LOCAL_LLM_SYSTEMCTL": systemctl})
        proc = subprocess.Popen(["bash", "-c", 'exec -a "python -u -m research.runners.faketest_standalone" sleep 30'])
        try:
            time.sleep(0.2)
            res_on = _run(["on"], env)
            assert res_on.returncode == 0, res_on.stdout + res_on.stderr
            assert _wait_for(lambda: _is_stopped(proc.pid)), "the standalone local process was not SIGSTOP'd"
            frozen = open(os.path.join(tmp, ".game_frozen_pids")).read()
            assert str(proc.pid) in frozen

            res_off = _run(["off"], env)
            assert res_off.returncode == 0, res_off.stdout + res_off.stderr
            assert _wait_for(lambda: not _is_stopped(proc.pid)), "the process was not SIGCONT'd back to running"
            assert not os.path.exists(os.path.join(tmp, "GAME_MODE"))
            assert not os.path.exists(os.path.join(tmp, "GPU_PAUSE"))
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


def test_on_records_the_llm_state_when_it_was_loaded_and_off_reloads_it():
    with tempfile.TemporaryDirectory() as tmp:
        llm_sh, log = _fake_llm_sh(tmp, up=True, profile="qwen38-27b-iq4nl-mtp")
        bindir = _fake_curl(tmp, up=True)
        systemctl = _fake_systemctl(tmp, active=True)
        state_home = os.path.join(tmp, "state")
        os.makedirs(os.path.join(state_home, "sim-local-llm"))
        open(os.path.join(state_home, "sim-local-llm", "last_session"), "w").write("sess-abc\n")
        env = dict(os.environ)
        env["PATH"] = bindir + os.pathsep + env["PATH"]
        env.update({"GPU_QUEUE_DIR": tmp, "GAME_LLM_SH": llm_sh, "XDG_STATE_HOME": state_home, "LOCAL_LLM_SYSTEMCTL": systemctl})

        res_on = _run(["on"], env)
        assert res_on.returncode == 0, res_on.stdout + res_on.stderr
        assert "CALLED: __stop_graceful 0" in open(log).read(), "on (no --graceful) must stop immediately"
        state = open(os.path.join(tmp, ".game_llm_state")).read()
        assert "GAME_LLM_WAS_ON=1" in state
        assert "GAME_LLM_PROFILE=qwen38-27b-iq4nl-mtp" in state
        assert "GAME_LLM_SESSION=sess-abc" in state

        res_off = _run(["off"], env)
        assert res_off.returncode == 0, res_off.stdout + res_off.stderr
        log_text = open(log).read()
        assert "CALLED: on qwen38-27b-iq4nl-mtp --wait" in log_text, "off must reload the SAME profile"
        assert not os.path.exists(os.path.join(tmp, ".game_llm_state")), "the state record must be consumed"
        # this test process has no controlling tty under pytest -- so `off` must take the headless path.
        assert _wait_for(lambda: "CALLED: claude --resume sess-abc -p" in open(log).read()), (
            "the SAME session must be resumed headless: %s" % log_text)


def test_on_off_decide_llm_state_from_the_unit_not_the_health_port():
    """2026-09-29 incident: `game on` recorded GAME_LLM_WAS_ON=1 while the local-llm systemd unit was INACTIVE,
    because something else (Unsloth Studio, during setup) happened to answer on port 8080 -- so `game off` would
    have loaded the 23 GB model and started a SECOND driver session on top of whatever was actually running
    (CLAUDE.md non-negotiable 12: one driver at a time). The unit is inactive but the health port answers `up`
    (the exact conflicting-signal shape that caused it): `on` must record WAS_ON=0 (never stop a unit that was
    never running), `status` must say "down", and `off` must not reload the model or resume any session."""
    with tempfile.TemporaryDirectory() as tmp:
        llm_sh, log = _fake_llm_sh(tmp, up=False)
        bindir = _fake_curl(tmp, up=True)                  # something else answers on :8080 -- must NOT be trusted
        systemctl = _fake_systemctl(tmp, active=False)      # ground truth: the unit itself is NOT running
        env = dict(os.environ)
        env["PATH"] = bindir + os.pathsep + env["PATH"]
        env.update({"GPU_QUEUE_DIR": tmp, "GAME_LLM_SH": llm_sh, "LOCAL_LLM_SYSTEMCTL": systemctl})

        res_on = _run(["on"], env)
        assert res_on.returncode == 0, res_on.stdout + res_on.stderr
        assert "CALLED: __stop_graceful" not in _log_text(log), "must not stop a unit that was never active"
        state = open(os.path.join(tmp, ".game_llm_state")).read()
        assert "GAME_LLM_WAS_ON=0" in state, state

        res_status = _run(["status"], env)
        assert "local model: down" in res_status.stdout, res_status.stdout

        res_off = _run(["off"], env)
        assert res_off.returncode == 0, res_off.stdout + res_off.stderr
        assert "not loaded before the pause" in res_off.stdout
        log_text = _log_text(log)
        assert "CALLED: on" not in log_text and "CALLED: claude" not in log_text, (
            "off must never reload/resume a session for a unit that was never actually active: %s" % log_text)


def test_status_reports_local_model_up_only_when_the_unit_is_active():
    with tempfile.TemporaryDirectory() as tmp:
        llm_sh, _ = _fake_llm_sh(tmp, up=True, profile="prof-x")
        bindir = _fake_curl(tmp, up=False)          # health port down -- must NOT suppress a true "up"
        systemctl = _fake_systemctl(tmp, active=True)
        env = dict(os.environ)
        env["PATH"] = bindir + os.pathsep + env["PATH"]
        env.update({"GPU_QUEUE_DIR": tmp, "GAME_LLM_SH": llm_sh, "LOCAL_LLM_SYSTEMCTL": systemctl})
        res = _run(["status"], env)
        assert "local model: UP (prof-x)" in res.stdout, res.stdout


def test_on_graceful_delays_the_stop():
    with tempfile.TemporaryDirectory() as tmp:
        llm_sh, log = _fake_llm_sh(tmp, up=True)
        bindir = _fake_curl(tmp, up=True)
        systemctl = _fake_systemctl(tmp, active=True)
        env = dict(os.environ)
        env["PATH"] = bindir + os.pathsep + env["PATH"]
        env.update({"GPU_QUEUE_DIR": tmp, "GAME_LLM_SH": llm_sh, "LOCAL_LLM_SYSTEMCTL": systemctl})
        res = _run(["on", "--graceful"], env)
        assert res.returncode == 0, res.stdout + res.stderr
        assert "CALLED: __stop_graceful 45" in open(log).read()


def test_off_does_not_reload_when_the_model_was_not_loaded():
    with tempfile.TemporaryDirectory() as tmp:
        llm_sh, log = _fake_llm_sh(tmp, up=False)
        bindir = _fake_curl(tmp, up=False)
        systemctl = _fake_systemctl(tmp, active=False)
        env = dict(os.environ)
        env["PATH"] = bindir + os.pathsep + env["PATH"]
        env.update({"GPU_QUEUE_DIR": tmp, "GAME_LLM_SH": llm_sh, "LOCAL_LLM_SYSTEMCTL": systemctl})
        _run(["on"], env)
        res = _run(["off"], env)
        assert res.returncode == 0, res.stdout + res.stderr
        assert "not loaded before the pause" in res.stdout
        assert "CALLED: on" not in _log_text(log)
        assert "CALLED: claude" not in _log_text(log)


def test_status_reports_game_mode_and_frozen_pids():
    with tempfile.TemporaryDirectory() as tmp:
        llm_sh, _ = _fake_llm_sh(tmp, up=False)
        bindir = _fake_curl(tmp, up=False)
        systemctl = _fake_systemctl(tmp, active=False)
        env = dict(os.environ)
        env["PATH"] = bindir + os.pathsep + env["PATH"]
        env.update({"GPU_QUEUE_DIR": tmp, "GAME_LLM_SH": llm_sh, "LOCAL_LLM_SYSTEMCTL": systemctl})
        res0 = _run(["status"], env)
        assert "GAME_MODE: off" in res0.stdout

        proc = subprocess.Popen(["bash", "-c", 'exec -a "python -u -m research.runners.faketest_status" sleep 30'])
        try:
            time.sleep(0.2)
            _run(["on"], env)
            res1 = _run(["status"], env)
            assert "GAME_MODE: ON" in res1.stdout
            assert str(proc.pid) in res1.stdout
        finally:
            _run(["off"], env)
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


def _wait_for(predicate, timeout=8.0, interval=0.05):
    deadline = time.time() + timeout
    ok = predicate()
    while not ok and time.time() < deadline:
        time.sleep(interval)
        ok = predicate()
    return ok


# ---------------------------------------------------------------------------------------------------------------
# sim-agent harness pause/unpause (2026-09-29, docs/plans/2026-09-29-local-model-transition-PLAN.md section 9):
# `on` best-effort pauses the containerized harness via tools/agent_harness_ctl.sh (over ssh); `off` unpauses it
# and delivers a "gaming's over" line. GAME_SSH stubs `ssh` entirely -- no real ssh connection is ever made.
# ---------------------------------------------------------------------------------------------------------------

def _fake_ctl_ssh(tmp, kind="local", reachable=True):
    log = os.path.join(tmp, "ssh_calls.log")
    if not reachable:
        return _write_exec(os.path.join(tmp, "fake_ssh.sh"), 'echo "ssh $*" >> "%s"\nexit 1\n' % log), log
    body = """
echo "ssh $*" >> "%s"
args=("$@")
last="${args[-1]:-}"
case "$last" in
  state) echo "profile_kind: %s" ;;
  *) : ;;
esac
exit 0
""" % (log, kind)
    return _write_exec(os.path.join(tmp, "fake_ssh.sh"), body), log


def test_on_pauses_the_harness_via_ctl_ssh():
    with tempfile.TemporaryDirectory() as tmp:
        llm_sh, _ = _fake_llm_sh(tmp, up=False)
        bindir = _fake_curl(tmp, up=False)
        systemctl = _fake_systemctl(tmp, active=False)
        ssh_path, ssh_log = _fake_ctl_ssh(tmp, kind="local")
        env = dict(os.environ)
        env["PATH"] = bindir + os.pathsep + env["PATH"]
        env.update({"GPU_QUEUE_DIR": tmp, "GAME_LLM_SH": llm_sh, "GAME_SSH": ssh_path, "LOCAL_LLM_SYSTEMCTL": systemctl})
        res = _run(["on"], env)
        assert res.returncode == 0, res.stdout + res.stderr
        calls = _log_text(ssh_log)
        assert "state" in calls
        assert "pause" in calls


def test_on_skips_the_harness_pause_for_a_cloud_profile():
    with tempfile.TemporaryDirectory() as tmp:
        llm_sh, _ = _fake_llm_sh(tmp, up=False)
        bindir = _fake_curl(tmp, up=False)
        systemctl = _fake_systemctl(tmp, active=False)
        ssh_path, ssh_log = _fake_ctl_ssh(tmp, kind="cloud")
        env = dict(os.environ)
        env["PATH"] = bindir + os.pathsep + env["PATH"]
        env.update({"GPU_QUEUE_DIR": tmp, "GAME_LLM_SH": llm_sh, "GAME_SSH": ssh_path, "LOCAL_LLM_SYSTEMCTL": systemctl})
        res = _run(["on"], env)
        assert res.returncode == 0, res.stdout + res.stderr
        assert "no pause needed" in res.stdout.lower()
        # only the `state` query happened, never a `pause`
        assert "pause" not in _log_text(ssh_log)


def test_on_continues_gracefully_when_the_ctl_host_is_unreachable():
    with tempfile.TemporaryDirectory() as tmp:
        llm_sh, _ = _fake_llm_sh(tmp, up=False)
        bindir = _fake_curl(tmp, up=False)
        systemctl = _fake_systemctl(tmp, active=False)
        ssh_path, _ = _fake_ctl_ssh(tmp, reachable=False)
        env = dict(os.environ)
        env["PATH"] = bindir + os.pathsep + env["PATH"]
        env.update({"GPU_QUEUE_DIR": tmp, "GAME_LLM_SH": llm_sh, "GAME_SSH": ssh_path, "LOCAL_LLM_SYSTEMCTL": systemctl})
        res = _run(["on"], env)
        assert res.returncode == 0, res.stdout + res.stderr
        assert os.path.exists(os.path.join(tmp, "GAME_MODE")), "local pause must still proceed despite the unreachable ctl host"


def test_off_unpauses_the_harness_and_sends_a_resume_message():
    with tempfile.TemporaryDirectory() as tmp:
        llm_sh, _ = _fake_llm_sh(tmp, up=False)
        bindir = _fake_curl(tmp, up=False)
        systemctl = _fake_systemctl(tmp, active=False)
        ssh_path, ssh_log = _fake_ctl_ssh(tmp, kind="local")
        env = dict(os.environ)
        env["PATH"] = bindir + os.pathsep + env["PATH"]
        env.update({"GPU_QUEUE_DIR": tmp, "GAME_LLM_SH": llm_sh, "GAME_SSH": ssh_path, "LOCAL_LLM_SYSTEMCTL": systemctl})
        _run(["on"], env)
        res = _run(["off"], env)
        assert res.returncode == 0, res.stdout + res.stderr
        calls = _log_text(ssh_log)
        assert "unpause" in calls
        assert "say" in calls and "runbook" in calls
