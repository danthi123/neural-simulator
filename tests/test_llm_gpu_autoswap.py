"""tools/gpu_queue.sh <-> tools/local_llm/llm.sh AUTOSWAP (2026-09-25): the interactive local-llm service and
queued GPU jobs both want the one 3090 and are not coordinated on their own. gpu_queue.sh's dispatcher now stops
the unit itself right before a queued job would contend with it for VRAM, and restarts it once the queue drains
-- see the "LOCAL-LLM AUTOSWAP" block in tools/gpu_queue.sh and the AUTOSWAP note at the top of
tools/local_llm/llm.sh.

Hermetic throughout: every test runs against an isolated GPU_QUEUE_DIR (never the live research/queue/) and
stubs systemctl (LOCAL_LLM_SYSTEMCTL), llm.sh itself (GPU_QUEUE_LLM_SH), and nvidia-smi (GPU_QUEUE_NVIDIA_SMI)
-- no real systemd unit or GPU is ever touched. Exercises gpu_queue.sh's `__llm_stop_for_job` /
`__llm_restore_if_idle` hidden entry points directly (single decision, no daemon loop) and llm.sh's real `on`/
`off` subcommands.
"""
import os
import stat
import subprocess
import tempfile
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_GPU_QUEUE = os.path.join(_ROOT, "tools", "gpu_queue.sh")
_LLM_SH = os.path.join(_ROOT, "tools", "local_llm", "llm.sh")


def _write_exec(path, body):
    with open(path, "w") as f:
        f.write("#!/bin/bash\n" + body)
    os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return path


def _fake_systemctl(tmp, state):
    """A stand-in for `systemctl --user ...`. `state` file holds "active" or "inactive"; `stop`/`start`
    flip it so cmd_off/is-active stay consistent with each other inside one test."""
    state_file = os.path.join(tmp, "fake_unit_state")
    with open(state_file, "w") as f:
        f.write(state)
    path = _write_exec(os.path.join(tmp, "fake_systemctl.sh"), """
STATE_FILE="%s"
case "$*" in
  *"is-active --quiet local-llm"*) [ "$(cat "$STATE_FILE" 2>/dev/null)" = "active" ] ;;
  *"stop local-llm"*) echo inactive > "$STATE_FILE" ;;
  *"start local-llm"*) echo active > "$STATE_FILE" ;;
  *"reset-failed local-llm"*) exit 0 ;;
  *"show -p Description"*) echo "local LLM (test-profile)" ;;
  *) exit 0 ;;
esac
""" % state_file)
    return path, state_file


def _fake_llm_sh(tmp, on_exit=0):
    """A stand-in for tools/local_llm/llm.sh, for testing gpu_queue.sh's OWN coordination logic in isolation
    (never spawns a real llama-server / systemd-run). Every invocation is appended to `log`."""
    log = os.path.join(tmp, "fake_llm_sh.log")
    path = _write_exec(os.path.join(tmp, "fake_llm.sh"), """
echo "CALLED: $*" >> "%s"
case "${1:-}" in
  on) exit %d ;;
  *) exit 0 ;;
esac
""" % (log, on_exit))
    return path, log


def _fake_nvidia_smi(tmp, resident_pid=""):
    path = _write_exec(os.path.join(tmp, "fake_nvidia_smi.sh"), """
case "$*" in
  *--query-compute-apps*) [ -n "%s" ] && echo "%s" ;;
  *--query-gpu*) echo "20000" ;;
esac
""" % (resident_pid, resident_pid))
    return path


def _run(args, env_extra, timeout=15):
    env = dict(os.environ)
    env.update(env_extra)
    return subprocess.run(["bash", _GPU_QUEUE] + args, env=env, capture_output=True, text=True, timeout=timeout)


def _log_text(log_path):
    return open(log_path).read() if os.path.exists(log_path) else ""


def _spawn_fake_resident_pid():
    """A real, living process whose /proc/<pid>/cmdline matches gpu_resident_brain_pids()'s pattern (it reads
    /proc, not nvidia-smi's own process_name field, so a bare fake pid number is never enough -- mirrors
    gpu_queue.sh's own --selftest TEST C/D). Caller must kill() it."""
    proc = subprocess.Popen(["bash", "-c", 'exec -a "python -u -m research.runners.faketest_resident" sleep 60'])
    time.sleep(0.1)
    return proc


def _wait_for(predicate, timeout=5.0, interval=0.05):
    """Poll until `predicate()` is true (restore is deliberately backgrounded, so callers must not assume it
    lands synchronously) or the timeout elapses; returns the final truthiness."""
    deadline = time.time() + timeout
    ok = predicate()
    while not ok and time.time() < deadline:
        time.sleep(interval)
        ok = predicate()
    return ok


# ---------------------------------------------------------------------------------------------------------------
# gpu_queue.sh: llm_stop_for_job
# ---------------------------------------------------------------------------------------------------------------

def test_stop_for_job_stops_and_marks_when_active():
    with tempfile.TemporaryDirectory() as tmp:
        systemctl, _ = _fake_systemctl(tmp, "active")
        llm_sh, log = _fake_llm_sh(tmp)
        res = _run(["__llm_stop_for_job"], {
            "GPU_QUEUE_DIR": tmp, "LOCAL_LLM_SYSTEMCTL": systemctl, "GPU_QUEUE_LLM_SH": llm_sh,
        })
        assert res.returncode == 0, res.stderr
        assert os.path.exists(os.path.join(tmp, ".local_llm_was_on")), "marker was not written for an active unit"
        assert "CALLED: off" in _log_text(log), "llm.sh off was never invoked"


def test_stop_for_job_is_a_noop_when_already_inactive():
    """Never fabricates an 'it was on' memory for a model that was already down."""
    with tempfile.TemporaryDirectory() as tmp:
        systemctl, _ = _fake_systemctl(tmp, "inactive")
        llm_sh, log = _fake_llm_sh(tmp)
        res = _run(["__llm_stop_for_job"], {
            "GPU_QUEUE_DIR": tmp, "LOCAL_LLM_SYSTEMCTL": systemctl, "GPU_QUEUE_LLM_SH": llm_sh,
        })
        assert res.returncode == 0, res.stderr
        assert not os.path.exists(os.path.join(tmp, ".local_llm_was_on"))
        assert "CALLED" not in _log_text(log)


# ---------------------------------------------------------------------------------------------------------------
# gpu_queue.sh: llm_restore_if_idle
# ---------------------------------------------------------------------------------------------------------------

def test_restore_if_idle_noop_without_a_marker():
    with tempfile.TemporaryDirectory() as tmp:
        systemctl, _ = _fake_systemctl(tmp, "inactive")
        llm_sh, log = _fake_llm_sh(tmp)
        nvidia_smi = _fake_nvidia_smi(tmp)
        res = _run(["__llm_restore_if_idle"], {
            "GPU_QUEUE_DIR": tmp, "LOCAL_LLM_SYSTEMCTL": systemctl, "GPU_QUEUE_LLM_SH": llm_sh,
            "GPU_QUEUE_NVIDIA_SMI": nvidia_smi,
        })
        assert res.returncode == 0, res.stderr
        time.sleep(0.3)
        assert "CALLED" not in _log_text(log), "llm.sh was invoked with no marker present"


def test_restore_if_idle_restores_and_consumes_the_marker():
    with tempfile.TemporaryDirectory() as tmp:
        open(os.path.join(tmp, ".local_llm_was_on"), "w").close()
        systemctl, _ = _fake_systemctl(tmp, "inactive")
        llm_sh, log = _fake_llm_sh(tmp)
        nvidia_smi = _fake_nvidia_smi(tmp)   # no resident brain process
        res = _run(["__llm_restore_if_idle"], {
            "GPU_QUEUE_DIR": tmp, "LOCAL_LLM_SYSTEMCTL": systemctl, "GPU_QUEUE_LLM_SH": llm_sh,
            "GPU_QUEUE_NVIDIA_SMI": nvidia_smi,
        })
        assert res.returncode == 0, res.stderr
        assert not os.path.exists(os.path.join(tmp, ".local_llm_was_on")), "marker must be consumed immediately"
        assert _wait_for(lambda: "CALLED: on" in _log_text(log)), "llm.sh on was never invoked (it is backgrounded)"


def test_restore_if_idle_defers_while_paused():
    """The gaming PAUSE sentinel: restoring mid-pause would be exactly backwards, but the marker must NOT be
    lost -- it retries once PAUSE clears."""
    with tempfile.TemporaryDirectory() as tmp:
        open(os.path.join(tmp, ".local_llm_was_on"), "w").close()
        open(os.path.join(tmp, "GPU_PAUSE"), "w").close()
        systemctl, _ = _fake_systemctl(tmp, "inactive")
        llm_sh, log = _fake_llm_sh(tmp)
        nvidia_smi = _fake_nvidia_smi(tmp)
        env = {
            "GPU_QUEUE_DIR": tmp, "LOCAL_LLM_SYSTEMCTL": systemctl, "GPU_QUEUE_LLM_SH": llm_sh,
            "GPU_QUEUE_NVIDIA_SMI": nvidia_smi,
        }
        res = _run(["__llm_restore_if_idle"], env)
        assert res.returncode == 0, res.stderr
        time.sleep(0.3)
        assert os.path.exists(os.path.join(tmp, ".local_llm_was_on")), "PAUSE must defer, not drop, the marker"
        assert "CALLED" not in _log_text(log)

        # PAUSE clears -> the very next idle poll restores it.
        os.remove(os.path.join(tmp, "GPU_PAUSE"))
        res2 = _run(["__llm_restore_if_idle"], env)
        assert res2.returncode == 0, res2.stderr
        assert not os.path.exists(os.path.join(tmp, ".local_llm_was_on"))
        assert _wait_for(lambda: "CALLED: on" in _log_text(log))


def test_restore_if_idle_defers_while_a_brain_process_is_still_gpu_resident():
    """Crash-safety: a job that outlived a dead daemon incarnation must block the restore (double-loading the
    card would be worse than a late reload) but, again, must not lose the marker."""
    with tempfile.TemporaryDirectory() as tmp:
        open(os.path.join(tmp, ".local_llm_was_on"), "w").close()
        systemctl, _ = _fake_systemctl(tmp, "inactive")
        llm_sh, log = _fake_llm_sh(tmp)
        resident = _spawn_fake_resident_pid()
        try:
            nvidia_smi = _fake_nvidia_smi(tmp, resident_pid=str(resident.pid))
            env = {
                "GPU_QUEUE_DIR": tmp, "LOCAL_LLM_SYSTEMCTL": systemctl, "GPU_QUEUE_LLM_SH": llm_sh,
                "GPU_QUEUE_NVIDIA_SMI": nvidia_smi,
            }
            res = _run(["__llm_restore_if_idle"], env)
            assert res.returncode == 0, res.stderr
            time.sleep(0.3)
            assert os.path.exists(os.path.join(tmp, ".local_llm_was_on")), "a resident brain process must defer, not drop, the marker"
            assert "CALLED" not in _log_text(log)
        finally:
            resident.kill()
            resident.wait(timeout=5)

        # the resident process is gone -> the next idle poll restores it.
        nvidia_smi_clear = _fake_nvidia_smi(tmp, resident_pid="")
        env["GPU_QUEUE_NVIDIA_SMI"] = nvidia_smi_clear
        res2 = _run(["__llm_restore_if_idle"], env)
        assert res2.returncode == 0, res2.stderr
        assert not os.path.exists(os.path.join(tmp, ".local_llm_was_on"))
        assert _wait_for(lambda: "CALLED: on" in _log_text(log))


def test_restore_if_idle_is_one_shot_a_failed_restart_is_never_retried():
    """Crash/hang safety: the marker is consumed BEFORE the restart attempt, so a `llm on` that itself fails
    (or would hang) is tried exactly once per drain, never in a tight forever-retry loop."""
    with tempfile.TemporaryDirectory() as tmp:
        open(os.path.join(tmp, ".local_llm_was_on"), "w").close()
        systemctl, _ = _fake_systemctl(tmp, "inactive")
        llm_sh, log = _fake_llm_sh(tmp, on_exit=1)   # `llm on` itself fails
        nvidia_smi = _fake_nvidia_smi(tmp)
        env = {
            "GPU_QUEUE_DIR": tmp, "LOCAL_LLM_SYSTEMCTL": systemctl, "GPU_QUEUE_LLM_SH": llm_sh,
            "GPU_QUEUE_NVIDIA_SMI": nvidia_smi,
        }
        _run(["__llm_restore_if_idle"], env)
        assert _wait_for(lambda: "CALLED: on" in _log_text(log))
        assert not os.path.exists(os.path.join(tmp, ".local_llm_was_on"))

        # run it again on the same (still-idle, still-failed) state: with the marker gone, this must be a
        # total no-op -- proving the earlier failure is not retried on every subsequent idle poll.
        before = _log_text(log)
        res2 = _run(["__llm_restore_if_idle"], env)
        assert res2.returncode == 0
        time.sleep(0.3)
        assert _log_text(log) == before, "a consumed marker must not cause a second restart attempt"


# ---------------------------------------------------------------------------------------------------------------
# llm.sh: `on` refuses/waits while a GPU job is busy; `off` always cancels the pending auto-restore
# ---------------------------------------------------------------------------------------------------------------

def _run_llm(args, env_extra, timeout=10):
    env = dict(os.environ)
    env.update(env_extra)
    return subprocess.run(["bash", _LLM_SH] + args, env=env, capture_output=True, text=True, timeout=timeout)


def test_llm_on_refuses_while_a_gpu_job_is_running():
    with tempfile.TemporaryDirectory() as tmp:
        open(os.path.join(tmp, "gpu.running"), "w").write("12345\tsome-job")
        systemctl, _ = _fake_systemctl(tmp, "inactive")
        # a sentinel that would only appear if llm.sh went ahead and tried to actually launch the model. A
        # REAL, valid profile name (not a placeholder) so a broken refusal-check runs all the way to
        # systemd-run instead of failing earlier for an unrelated reason (an unknown profile name).
        sentinel = os.path.join(tmp, "systemd_run_was_called")
        fake_systemd_run = _write_exec(os.path.join(tmp, "systemd-run"), "touch %s\n" % sentinel)
        env = dict(os.environ)
        env["PATH"] = tmp + os.pathsep + env["PATH"]
        env["GPU_QUEUE_DIR"] = tmp
        env["LOCAL_LLM_SYSTEMCTL"] = systemctl
        res = subprocess.run(["bash", _LLM_SH, "on", "qwen38-27b-iq4nl-mtp"], env=env, capture_output=True, text=True, timeout=10)
        assert res.returncode != 0, res.stdout + res.stderr
        assert "refusing" in res.stdout.lower(), res.stdout
        assert not os.path.exists(sentinel), "llm on must not launch the model while a GPU job is busy"
        del fake_systemd_run


def test_llm_on_already_loaded_short_circuits_before_the_busy_check():
    """A GPU job busy is irrelevant once the model is already up -- nothing to load, nothing to refuse."""
    with tempfile.TemporaryDirectory() as tmp:
        open(os.path.join(tmp, "gpu.running"), "w").write("12345\tsome-job")
        systemctl, _ = _fake_systemctl(tmp, "active")
        res = _run_llm(["on"], {"GPU_QUEUE_DIR": tmp, "LOCAL_LLM_SYSTEMCTL": systemctl})
        assert res.returncode == 0, res.stdout + res.stderr
        assert "already loaded" in res.stdout


def test_llm_on_wait_blocks_while_the_gpu_job_is_running():
    """--wait must actually block (not refuse, not proceed) for as long as the queue is busy."""
    with tempfile.TemporaryDirectory() as tmp:
        running = os.path.join(tmp, "gpu.running")
        open(running, "w").write("12345\tsome-job")
        systemctl, _ = _fake_systemctl(tmp, "inactive")
        env = dict(os.environ)
        env.update({"GPU_QUEUE_DIR": tmp, "LOCAL_LLM_SYSTEMCTL": systemctl, "LLM_WAIT_POLL_SEC": "1"})
        proc = subprocess.Popen(["bash", _LLM_SH, "on", "some-profile", "--wait"], env=env,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            time.sleep(0.6)
            assert proc.poll() is None, "llm on --wait must still be blocked while the GPU job is running"
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


def test_llm_off_cancels_the_pending_autoswap_restore_even_mid_job():
    """The exact scenario named in the task: gpu_queue stopped the model for a job (marker set), the owner
    then runs `llm off` themselves mid-job -- the marker must be gone so the eventual queue-drain restore
    never fires."""
    with tempfile.TemporaryDirectory() as tmp:
        marker = os.path.join(tmp, ".local_llm_was_on")
        open(marker, "w").close()   # as gpu_queue's llm_stop_for_job would have left it
        systemctl, _ = _fake_systemctl(tmp, "inactive")   # already stopped, exactly like the mid-job scenario
        res = _run_llm(["off"], {"GPU_QUEUE_DIR": tmp, "LOCAL_LLM_SYSTEMCTL": systemctl})
        assert res.returncode == 0, res.stdout + res.stderr
        assert not os.path.exists(marker), "a manual `llm off` must cancel gpu_queue's pending auto-restore"

        # and the queue-drain restore that follows must now be a true no-op.
        llm_sh, log = _fake_llm_sh(tmp)
        res2 = _run(["__llm_restore_if_idle"], {
            "GPU_QUEUE_DIR": tmp, "LOCAL_LLM_SYSTEMCTL": systemctl, "GPU_QUEUE_LLM_SH": llm_sh,
            "GPU_QUEUE_NVIDIA_SMI": _fake_nvidia_smi(tmp),
        })
        assert res2.returncode == 0
        time.sleep(0.3)
        assert "CALLED" not in _log_text(log)


def test_llm_off_clears_the_marker_even_when_the_unit_is_still_active():
    with tempfile.TemporaryDirectory() as tmp:
        marker = os.path.join(tmp, ".local_llm_was_on")
        open(marker, "w").close()
        systemctl, _ = _fake_systemctl(tmp, "active")
        res = _run_llm(["off"], {"GPU_QUEUE_DIR": tmp, "LOCAL_LLM_SYSTEMCTL": systemctl})
        assert res.returncode == 0, res.stdout + res.stderr
        assert "unloaded" in res.stdout
        assert not os.path.exists(marker)
