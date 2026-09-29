"""tools/gpu_queue.sh <-> tools/local_llm/llm.sh coordination for the one shared 3090.

2026-09-25 (fix-round same day) introduced an AUTOSWAP: gpu_queue.sh's dispatcher stopped the local-llm unit
itself, automatically, the instant a job was queued, and restored it once the queue drained. 2026-09-29
(docs/plans/2026-09-29-local-model-transition-PLAN.md section 3) retired the automatic stop: with the local
model now the day-to-day DRIVER, an automatic mid-session stop loses whatever it was doing. A queued job is now
HELD (never started) for as long as the unit is active; the driver frees the card only by explicitly choosing to
(`llm.sh yield` -- see tests/test_llm_gpu_yield.py for that half). This file keeps the parts of the ORIGINAL
autoswap machinery that are still live and unrelated to the retired auto-stop:
  - `llm_restore_if_idle`: when the OWNER (not the driver) runs `llm on` while a GPU job is busy, that refusal
    itself records the intent to load once the queue clears (research/queue/.local_llm_was_on) -- this is what
    honours it once the queue actually drains. Nothing in gpu_queue.sh writes that marker any more; only llm.sh's
    own `cmd_on` refusal path does.
  - `llm on`'s busy-refusal/`--wait` behaviour and `llm off`'s marker-clearing -- unaffected by the auto-stop
    retirement (they never depended on the daemon dispatching a job over an active unit).

Hermetic throughout: every test runs against an isolated GPU_QUEUE_DIR (never the live research/queue/) and
stubs systemctl (LOCAL_LLM_SYSTEMCTL), llm.sh itself (GPU_QUEUE_LLM_SH, where used), and nvidia-smi
(GPU_QUEUE_NVIDIA_SMI) -- no real systemd unit or GPU is ever touched. Exercises gpu_queue.sh's
`__llm_restore_if_idle` hidden entry point directly (single decision, no daemon loop) and llm.sh's real
`on`/`off`/`__current_profile` subcommands throughout.
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


def _fake_systemctl(tmp, state, desc="local LLM (test-profile)", stop_works=True):
    """A stand-in for `systemctl --user ...`. `state` file holds "active"/"inactive"; `stop`/`start` flip it
    -- UNLESS stop_works=False, which makes `stop` a no-op (simulates a stop that is issued but does not
    actually take, e.g. a wedged unit -- the HIGH finding's failure mode). `desc` seeds a separate file read by
    `show -p Description`, so a fake `systemd-run` (see tests/test_llm_gpu_yield.py's own
    `_fake_systemd_run_and_curl`) can update it to reflect whatever profile was actually (re)loaded -- this is
    what lets a test verify the SAME profile came back. Returns (systemctl_path, state_file, desc_file)."""
    state_file = os.path.join(tmp, "fake_unit_state")
    desc_file = os.path.join(tmp, "fake_unit_desc")
    with open(state_file, "w") as f:
        f.write(state)
    with open(desc_file, "w") as f:
        f.write(desc)
    # NOTE: no trailing "# comment" on this line -- it would swallow the case clause's own terminating ";;".
    stop_case = 'echo inactive > "$STATE_FILE"' if stop_works else ':'   # ':' == issued but does not take (HIGH finding)
    path = _write_exec(os.path.join(tmp, "fake_systemctl.sh"), """
STATE_FILE="%s"
DESC_FILE="%s"
case "$*" in
  *"is-active --quiet local-llm"*) [ "$(cat "$STATE_FILE" 2>/dev/null)" = "active" ] ;;
  *"stop local-llm"*) %s ;;
  *"start local-llm"*) echo active > "$STATE_FILE" ;;
  *"reset-failed local-llm"*) exit 0 ;;
  *"show -p Description"*) cat "$DESC_FILE" 2>/dev/null ;;
  *) exit 0 ;;
esac
""" % (state_file, desc_file, stop_case))
    return path, state_file, desc_file


def _fake_llm_sh(tmp, on_exit=0, profile="fake-profile"):
    """A stand-in for tools/local_llm/llm.sh, for testing gpu_queue.sh's OWN coordination logic in isolation
    (never spawns a real llama-server / systemd-run). Every invocation is appended to `log`. Answers
    `on` and `__current_profile`."""
    log = os.path.join(tmp, "fake_llm_sh.log")
    path = _write_exec(os.path.join(tmp, "fake_llm.sh"), """
echo "CALLED: $*" >> "%s"
case "${1:-}" in
  on) exit %d ;;
  __current_profile) echo "%s" ;;
  *) exit 0 ;;
esac
""" % (log, on_exit, profile))
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


def _spawn_live_pid():
    """Any real, living process -- a stand-in for a dispatcher/job pid wherever a test needs llm.sh's
    dispatcher_alive()/gpu_job_busy() (kill -0 based) to see something genuinely alive, as opposed to a bare
    placeholder pid number that was never a real process. Caller must terminate it."""
    return subprocess.Popen(["sleep", "60"])


def _dead_pid():
    """A pid guaranteed to be dead (spawned and immediately reaped) -- mirrors gpu_queue.sh's own --selftest
    TEST D idiom for a stale record that no longer corresponds to anything real."""
    p = subprocess.Popen(["true"])
    p.wait()
    return p.pid


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
# gpu_queue.sh: llm_restore_if_idle
# ---------------------------------------------------------------------------------------------------------------

def test_restore_if_idle_noop_without_a_marker():
    with tempfile.TemporaryDirectory() as tmp:
        systemctl, _, _ = _fake_systemctl(tmp, "inactive")
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
        open(os.path.join(tmp, ".local_llm_was_on"), "w").write("some-profile\n")
        systemctl, _, _ = _fake_systemctl(tmp, "inactive")
        llm_sh, log = _fake_llm_sh(tmp)
        nvidia_smi = _fake_nvidia_smi(tmp)   # no resident brain process
        res = _run(["__llm_restore_if_idle"], {
            "GPU_QUEUE_DIR": tmp, "LOCAL_LLM_SYSTEMCTL": systemctl, "GPU_QUEUE_LLM_SH": llm_sh,
            "GPU_QUEUE_NVIDIA_SMI": nvidia_smi,
        })
        assert res.returncode == 0, res.stderr
        assert not os.path.exists(os.path.join(tmp, ".local_llm_was_on")), "marker must be consumed immediately"
        assert _wait_for(lambda: "CALLED: on some-profile" in _log_text(log)), (
            "llm.sh on must be invoked with the SAME profile that was recorded (it is backgrounded)")


def test_restore_if_idle_defers_while_paused():
    """The gaming PAUSE sentinel: restoring mid-pause would be exactly backwards, but the marker must NOT be
    lost -- it retries once PAUSE clears."""
    with tempfile.TemporaryDirectory() as tmp:
        open(os.path.join(tmp, ".local_llm_was_on"), "w").write("some-profile\n")
        open(os.path.join(tmp, "GPU_PAUSE"), "w").close()
        systemctl, _, _ = _fake_systemctl(tmp, "inactive")
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
        assert _wait_for(lambda: "CALLED: on some-profile" in _log_text(log))


def test_restore_if_idle_defers_while_a_brain_process_is_still_gpu_resident():
    """Crash-safety: a job that outlived a dead daemon incarnation must block the restore (double-loading the
    card would be worse than a late reload) but, again, must not lose the marker."""
    with tempfile.TemporaryDirectory() as tmp:
        open(os.path.join(tmp, ".local_llm_was_on"), "w").write("some-profile\n")
        systemctl, _, _ = _fake_systemctl(tmp, "inactive")
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
        assert _wait_for(lambda: "CALLED: on some-profile" in _log_text(log))


def test_restore_if_idle_is_one_shot_a_failed_restart_is_never_retried():
    """Crash/hang safety: the marker is consumed BEFORE the restart attempt, so a `llm on` that itself fails
    (or would hang) is tried exactly once per drain, never in a tight forever-retry loop."""
    with tempfile.TemporaryDirectory() as tmp:
        open(os.path.join(tmp, ".local_llm_was_on"), "w").write("some-profile\n")
        systemctl, _, _ = _fake_systemctl(tmp, "inactive")
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


# The daemon-level HELD / yield-drain / restore-and-resume / deadline-expiry / GAME_MODE-precedence tests that
# used to live here (against the retired auto-stop) now live in tests/test_llm_gpu_yield.py, against the current
# design.

# ---------------------------------------------------------------------------------------------------------------
# llm.sh: `on` refuses/waits while a GPU job is busy; `off` always cancels the pending auto-restore
# ---------------------------------------------------------------------------------------------------------------

def _run_llm(args, env_extra, timeout=10):
    env = dict(os.environ)
    env.update(env_extra)
    return subprocess.run(["bash", _LLM_SH] + args, env=env, capture_output=True, text=True, timeout=timeout)


def test_llm_on_refuses_while_a_gpu_job_is_running():
    """gpu_job_busy() now requires a genuinely LIVE dispatcher AND a live recorded job pid (MEDIUM finding) --
    both are real, running processes here, never a bare placeholder pid."""
    with tempfile.TemporaryDirectory() as tmp:
        dispatcher = _spawn_live_pid()
        job = _spawn_live_pid()
        try:
            open(os.path.join(tmp, "gpu_queue.dpid"), "w").write(str(dispatcher.pid))
            open(os.path.join(tmp, "gpu.running"), "w").write("%d\tsome-job" % job.pid)
            systemctl, _, _ = _fake_systemctl(tmp, "inactive")
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
        finally:
            dispatcher.kill(); dispatcher.wait(timeout=5)
            job.kill(); job.wait(timeout=5)


def test_llm_on_records_intent_when_it_refuses():
    """MEDIUM finding: a refused `llm on` promises an auto-load once the queue drains -- it must WRITE the
    marker itself to make that true (nothing else writes it any more: the auto-stop that used to write it for a
    queued job was retired 2026-09-29 in favour of an explicit yield, so a refusal against an ALREADY-unloaded
    model would otherwise never auto-load anything)."""
    with tempfile.TemporaryDirectory() as tmp:
        dispatcher = _spawn_live_pid()
        job = _spawn_live_pid()
        try:
            open(os.path.join(tmp, "gpu_queue.dpid"), "w").write(str(dispatcher.pid))
            open(os.path.join(tmp, "gpu.running"), "w").write("%d\tsome-job" % job.pid)
            systemctl, _, _ = _fake_systemctl(tmp, "inactive")
            res = _run_llm(["on", "qwen38-27b-iq4nl-mtp"], {"GPU_QUEUE_DIR": tmp, "LOCAL_LLM_SYSTEMCTL": systemctl})
            assert res.returncode != 0
            marker = os.path.join(tmp, ".local_llm_was_on")
            assert os.path.exists(marker), "a refused `llm on` must record the owner's intent so the promised auto-load happens"
            assert open(marker).read().strip() == "qwen38-27b-iq4nl-mtp"
        finally:
            dispatcher.kill(); dispatcher.wait(timeout=5)
            job.kill(); job.wait(timeout=5)


def test_llm_on_does_not_refuse_forever_on_a_stale_running_pid_with_a_dead_dispatcher():
    """MEDIUM finding: a stale gpu.running (dead pid) left behind by a dispatcher that crashed, with no live
    dispatcher at all, must NOT refuse forever -- nothing is left to ever dequeue and contend for the GPU."""
    with tempfile.TemporaryDirectory() as tmp:
        open(os.path.join(tmp, "gpu.running"), "w").write("%d\tsome-job" % _dead_pid())
        # no gpu_queue.dpid and no .gpu_daemon.lock at all -> dispatcher_alive() must be false
        systemctl, _, _ = _fake_systemctl(tmp, "inactive")
        sentinel = os.path.join(tmp, "systemd_run_was_called")
        _write_exec(os.path.join(tmp, "systemd-run"), "touch %s\n" % sentinel)
        env = dict(os.environ)
        env["PATH"] = tmp + os.pathsep + env["PATH"]
        env["GPU_QUEUE_DIR"] = tmp
        env["LOCAL_LLM_SYSTEMCTL"] = systemctl
        res = subprocess.run(["bash", _LLM_SH, "on", "qwen38-27b-iq4nl-mtp"], env=env, capture_output=True, text=True, timeout=10)
        assert "refusing" not in res.stdout.lower(), res.stdout
        assert os.path.exists(sentinel), "a stale record with no live dispatcher must not block loading forever"


def test_llm_on_does_not_refuse_forever_on_a_stale_queue_with_a_dead_dispatcher():
    """Same MEDIUM finding, for a stale non-empty gpu.queue rather than gpu.running: with no live dispatcher,
    nothing will ever dequeue it, so it is not real contention."""
    with tempfile.TemporaryDirectory() as tmp:
        open(os.path.join(tmp, "gpu.queue"), "w").write("SIM_BACKEND=cupy some-command --json raw/o.json\n")
        systemctl, _, _ = _fake_systemctl(tmp, "inactive")
        sentinel = os.path.join(tmp, "systemd_run_was_called")
        _write_exec(os.path.join(tmp, "systemd-run"), "touch %s\n" % sentinel)
        env = dict(os.environ)
        env["PATH"] = tmp + os.pathsep + env["PATH"]
        env["GPU_QUEUE_DIR"] = tmp
        env["LOCAL_LLM_SYSTEMCTL"] = systemctl
        res = subprocess.run(["bash", _LLM_SH, "on", "qwen38-27b-iq4nl-mtp"], env=env, capture_output=True, text=True, timeout=10)
        assert "refusing" not in res.stdout.lower(), res.stdout
        assert os.path.exists(sentinel), "a stale queue with no live dispatcher must not block loading forever"


def test_llm_on_already_loaded_short_circuits_before_the_busy_check():
    """A GPU job busy is irrelevant once the model is already up -- nothing to load, nothing to refuse."""
    with tempfile.TemporaryDirectory() as tmp:
        open(os.path.join(tmp, "gpu.running"), "w").write("12345\tsome-job")
        systemctl, _, _ = _fake_systemctl(tmp, "active")
        res = _run_llm(["on"], {"GPU_QUEUE_DIR": tmp, "LOCAL_LLM_SYSTEMCTL": systemctl})
        assert res.returncode == 0, res.stdout + res.stderr
        assert "already loaded" in res.stdout


def test_llm_on_wait_blocks_while_the_gpu_job_is_running():
    """--wait must actually block (not refuse, not proceed) for as long as the queue is busy (with a genuinely
    LIVE dispatcher + job, matching the new liveness-based gpu_job_busy())."""
    with tempfile.TemporaryDirectory() as tmp:
        dispatcher = _spawn_live_pid()
        job = _spawn_live_pid()
        try:
            open(os.path.join(tmp, "gpu_queue.dpid"), "w").write(str(dispatcher.pid))
            open(os.path.join(tmp, "gpu.running"), "w").write("%d\tsome-job" % job.pid)
            systemctl, _, _ = _fake_systemctl(tmp, "inactive")
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
        finally:
            dispatcher.kill(); dispatcher.wait(timeout=5)
            job.kill(); job.wait(timeout=5)


def test_llm_off_cancels_the_pending_autoswap_restore_even_mid_job():
    """The exact scenario named in the task: `llm on` was refused earlier (marker set, recording the owner's
    intent to load once the GPU frees up), the owner then runs `llm off` themselves before it does -- the marker
    must be gone so the eventual queue-drain restore never fires."""
    with tempfile.TemporaryDirectory() as tmp:
        marker = os.path.join(tmp, ".local_llm_was_on")
        open(marker, "w").write("qwen38-27b-iq4nl-mtp\n")   # as a refused `llm on` would have left it
        systemctl, _, _ = _fake_systemctl(tmp, "inactive")   # already down, exactly like that scenario
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
        open(marker, "w").write("qwen38-27b-iq4nl-mtp\n")
        systemctl, _, _ = _fake_systemctl(tmp, "active")
        res = _run_llm(["off"], {"GPU_QUEUE_DIR": tmp, "LOCAL_LLM_SYSTEMCTL": systemctl})
        assert res.returncode == 0, res.stdout + res.stderr
        assert "unloaded" in res.stdout
        assert not os.path.exists(marker)

