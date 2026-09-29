"""tools/game.sh (rewritten 2026-09-29 for the local-model-driver setup, docs/plans/2026-09-29-local-model-
transition-PLAN.md section 3): `on` pauses the GPU queue, SIGSTOPs any standalone local research.runners/webapp
python (recording pids for `off` to SIGCONT), stops the local-llm server if it was loaded (recording its profile
+ last session so `off` can bring both back), and sets GAME_MODE; `off` reverses all three, then resumes the
recorded session -- interactively from a TTY, headless (logged) otherwise.

Hermetic throughout: GPU_QUEUE_DIR isolates every state file (GAME_MODE, the frozen-pid list, the llm-state
record) from the live research/queue/; GAME_LLM_SH stubs tools/local_llm/llm.sh entirely (never spawns a real
llama-server/systemd unit); tools/gpu_queue.sh's own `pause --now`/`resume`/`status` are exercised for real but
only ever touch the isolated GPU_QUEUE_DIR. Frozen "processes" are real, short-lived `sleep` children this file
spawns and kills itself -- never anything from the live system.
"""
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


def _log_text(path):
    return open(path).read() if os.path.exists(path) else ""


def _run(args, env_extra, timeout=10):
    env = dict(os.environ)
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
        env = dict(os.environ)
        env["PATH"] = bindir + os.pathsep + env["PATH"]
        env.update({"GPU_QUEUE_DIR": tmp, "GAME_LLM_SH": llm_sh})
        res = _run(["on"], env)
        assert res.returncode == 0, res.stdout + res.stderr
        assert os.path.exists(os.path.join(tmp, "GAME_MODE"))
        assert os.path.exists(os.path.join(tmp, "GPU_PAUSE")), "gpu_queue.sh pause --now must have run"


def test_on_off_round_trip_freezes_and_resumes_a_standalone_local_process():
    """SIGSTOP on `on`, SIGCONT on `off` -- proving pids are recorded and genuinely revived, not killed."""
    with tempfile.TemporaryDirectory() as tmp:
        llm_sh, _ = _fake_llm_sh(tmp, up=False)
        bindir = _fake_curl(tmp, up=False)
        env = dict(os.environ)
        env["PATH"] = bindir + os.pathsep + env["PATH"]
        env.update({"GPU_QUEUE_DIR": tmp, "GAME_LLM_SH": llm_sh})
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
        state_home = os.path.join(tmp, "state")
        os.makedirs(os.path.join(state_home, "sim-local-llm"))
        open(os.path.join(state_home, "sim-local-llm", "last_session"), "w").write("sess-abc\n")
        env = dict(os.environ)
        env["PATH"] = bindir + os.pathsep + env["PATH"]
        env.update({"GPU_QUEUE_DIR": tmp, "GAME_LLM_SH": llm_sh, "XDG_STATE_HOME": state_home})

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


def test_on_graceful_delays_the_stop():
    with tempfile.TemporaryDirectory() as tmp:
        llm_sh, log = _fake_llm_sh(tmp, up=True)
        bindir = _fake_curl(tmp, up=True)
        env = dict(os.environ)
        env["PATH"] = bindir + os.pathsep + env["PATH"]
        env.update({"GPU_QUEUE_DIR": tmp, "GAME_LLM_SH": llm_sh})
        res = _run(["on", "--graceful"], env)
        assert res.returncode == 0, res.stdout + res.stderr
        assert "CALLED: __stop_graceful 45" in open(log).read()


def test_off_does_not_reload_when_the_model_was_not_loaded():
    with tempfile.TemporaryDirectory() as tmp:
        llm_sh, log = _fake_llm_sh(tmp, up=False)
        bindir = _fake_curl(tmp, up=False)
        env = dict(os.environ)
        env["PATH"] = bindir + os.pathsep + env["PATH"]
        env.update({"GPU_QUEUE_DIR": tmp, "GAME_LLM_SH": llm_sh})
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
        env = dict(os.environ)
        env["PATH"] = bindir + os.pathsep + env["PATH"]
        env.update({"GPU_QUEUE_DIR": tmp, "GAME_LLM_SH": llm_sh})
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
