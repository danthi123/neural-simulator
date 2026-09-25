"""tools/gpu_queue.sh <-> tools/local_llm/llm.sh AUTOSWAP (2026-09-25, fix-round same day): the interactive
local-llm service and queued GPU jobs both want the one 3090 and are not coordinated on their own. gpu_queue.sh's
dispatcher stops the unit DIRECTLY via systemctl (never through llm.sh's own `off`, which deletes the very
marker the stop is about to write -- the BLOCKER this round fixed) right before a queued job would contend with
it for VRAM, and restarts it -- with the SAME profile -- once the queue drains. The dispatcher's contention loop
also never dispatches a job while the unit is still active (a loaded model alone can leave VRAM headroom above
MIN_FREE while still holding the card). See the "LOCAL-LLM AUTOSWAP" block in tools/gpu_queue.sh and the
AUTOSWAP note at the top of tools/local_llm/llm.sh.

Hermetic throughout: every test runs against an isolated GPU_QUEUE_DIR (never the live research/queue/) and
stubs systemctl (LOCAL_LLM_SYSTEMCTL), llm.sh itself (GPU_QUEUE_LLM_SH, where used), nvidia-smi
(GPU_QUEUE_NVIDIA_SMI), and (for the end-to-end tests) systemd-run/curl via PATH -- no real systemd unit or GPU
is ever touched. Exercises gpu_queue.sh's `__llm_stop_for_job` / `__llm_restore_if_idle` hidden entry points
directly (single decision, no daemon loop), the REAL daemon loop (`__daemon`) for the two end-to-end tests, and
llm.sh's real `on`/`off`/`__current_profile` subcommands throughout.
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
    `show -p Description`, so a fake `systemd-run` (see `_fake_systemd_run_and_curl`) can update it to reflect
    whatever profile was actually (re)loaded -- this is what lets a test verify the SAME profile came back.
    Returns (systemctl_path, state_file, desc_file)."""
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
    `__current_profile` (used by the real llm_stop_for_job to learn which profile to restore)."""
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


def _fake_systemd_run_and_curl(tmp, state_file, desc_file):
    """Fakes for llm.sh's REAL cmd_on path (used by the end-to-end restore test): `systemd-run` flips the fake
    unit to active and records the profile from its own --description argument (mirrors the real
    systemd-run/unit relationship); `curl` (used only by llm.sh's is_up() health check) succeeds iff the fake
    unit is active -- so the REAL, unstubbed wait_up()/is_up() loop resolves immediately without a real
    llama-server or network socket. Returns a directory to prepend to PATH."""
    bindir = os.path.join(tmp, "fakebin")
    os.makedirs(bindir, exist_ok=True)
    _write_exec(os.path.join(bindir, "systemd-run"), """
for a in "$@"; do
  case "$a" in
    --description=*) printf '%%s' "${a#--description=}" > "%s" ;;
  esac
done
echo active > "%s"
""" % (desc_file, state_file))
    _write_exec(os.path.join(bindir, "curl"), '[ "$(cat "%s" 2>/dev/null)" = "active" ]\n' % state_file)
    return bindir


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
# gpu_queue.sh: llm_stop_for_job
# ---------------------------------------------------------------------------------------------------------------

def test_stop_for_job_stops_and_marks_when_active():
    """BLOCKER regression test: the stop must happen DIRECTLY via systemctl -- never by shelling out to llm.sh's
    own `off`, which deletes this same marker as its own "manual off cancels the pending restore" behavior. The
    marker's contents must carry the profile that was actually running (LOW finding)."""
    with tempfile.TemporaryDirectory() as tmp:
        systemctl, state_file, _ = _fake_systemctl(tmp, "active")
        llm_sh, log = _fake_llm_sh(tmp, profile="qwen38-27b-iq4nl-mtp")
        res = _run(["__llm_stop_for_job"], {
            "GPU_QUEUE_DIR": tmp, "LOCAL_LLM_SYSTEMCTL": systemctl, "GPU_QUEUE_LLM_SH": llm_sh,
        })
        assert res.returncode == 0, res.stderr
        marker = os.path.join(tmp, ".local_llm_was_on")
        assert os.path.exists(marker), "marker was not written for an active unit"
        assert open(marker).read().strip() == "qwen38-27b-iq4nl-mtp", "marker must carry the profile to restore"
        assert open(state_file).read().strip() == "inactive", "the unit must actually be stopped"
        assert "CALLED: off" not in _log_text(log), (
            "must stop DIRECTLY via systemctl, never via llm.sh's own `off` -- off deletes this same marker "
            "(the BLOCKER: the marker written here would be destroyed moments later, and the model would "
            "never auto-restore)")
        assert "CALLED: __current_profile" in _log_text(log)


def test_stop_for_job_is_a_noop_when_already_inactive():
    """Never fabricates an 'it was on' memory for a model that was already down."""
    with tempfile.TemporaryDirectory() as tmp:
        systemctl, _, _ = _fake_systemctl(tmp, "inactive")
        llm_sh, log = _fake_llm_sh(tmp)
        res = _run(["__llm_stop_for_job"], {
            "GPU_QUEUE_DIR": tmp, "LOCAL_LLM_SYSTEMCTL": systemctl, "GPU_QUEUE_LLM_SH": llm_sh,
        })
        assert res.returncode == 0, res.stderr
        assert not os.path.exists(os.path.join(tmp, ".local_llm_was_on"))
        assert "CALLED" not in _log_text(log)


def test_stop_for_job_does_not_write_marker_or_claim_success_when_the_stop_fails():
    """HIGH finding: a stop that is issued but does not take must never be logged/treated as 'stopped', and
    must never write the marker -- a marker for a model that is STILL LOADED would tell llm_restore_if_idle to
    'restore' a unit that was never actually freed."""
    with tempfile.TemporaryDirectory() as tmp:
        systemctl, state_file, _ = _fake_systemctl(tmp, "active", stop_works=False)
        llm_sh, _ = _fake_llm_sh(tmp)
        res = _run(["__llm_stop_for_job"], {
            "GPU_QUEUE_DIR": tmp, "LOCAL_LLM_SYSTEMCTL": systemctl, "GPU_QUEUE_LLM_SH": llm_sh,
        })
        assert res.returncode != 0, "a failed stop must be reported as a failure, not silently swallowed"
        assert not os.path.exists(os.path.join(tmp, ".local_llm_was_on")), "must never mark a stop that didn't take"
        assert "FAILED to stop" in _log_text(os.path.join(tmp, "gpu_queue.log"))
        assert open(state_file).read().strip() == "active", "the unit is still (unintentionally) active"


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


# ---------------------------------------------------------------------------------------------------------------
# gpu_queue.sh __daemon: the contention loop must never dispatch while local-llm is still active, and the
# full stop -> job -> restore cycle must work end to end through the REAL llm.sh (BLOCKER + LOW regression)
# ---------------------------------------------------------------------------------------------------------------

def test_daemon_never_starts_a_job_while_llm_is_active_when_the_stop_keeps_failing():
    """HIGH finding, driven through the REAL daemon loop (not just the single-shot hidden entry points): if
    stopping local-llm keeps failing (a wedged unit), the queued job must NEVER start while it is still active
    -- raw VRAM headroom (MIN_FREE) alone would not catch this, since a loaded model can leave headroom above it
    while still holding the card."""
    with tempfile.TemporaryDirectory() as tmp:
        systemctl, state_file, _ = _fake_systemctl(tmp, "active", stop_works=False)   # stop can never succeed
        llm_sh, _ = _fake_llm_sh(tmp)
        nvidia_smi = _fake_nvidia_smi(tmp)   # plenty of "free" VRAM, as if the loaded model still leaves headroom
        sentinel = os.path.join(tmp, "job_ran")
        subprocess.run(["bash", _GPU_QUEUE, "add", "touch %s" % sentinel],
                        env={**os.environ, "GPU_QUEUE_DIR": tmp}, check=True, capture_output=True, text=True, timeout=10)
        env = dict(os.environ)
        env.update({
            "GPU_QUEUE_DIR": tmp, "LOCAL_LLM_SYSTEMCTL": systemctl, "GPU_QUEUE_LLM_SH": llm_sh,
            "GPU_QUEUE_NVIDIA_SMI": nvidia_smi, "GPU_QUEUE_POLL_SEC": "1",
        })
        daemon = subprocess.Popen(["bash", _GPU_QUEUE, "__daemon"], env=env,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            time.sleep(3)   # several retry cycles at POLL_SEC=1
            assert not os.path.exists(sentinel), "the job started while local-llm was still (unstoppably) active"
            assert open(state_file).read().strip() == "active"
            assert "FAILED to stop" in _log_text(os.path.join(tmp, "gpu_queue.log"))
            assert not os.path.exists(os.path.join(tmp, ".local_llm_was_on")), "must never mark a stop that never took"
        finally:
            daemon.terminate()
            try:
                daemon.wait(timeout=5)
            except subprocess.TimeoutExpired:
                daemon.kill()


def test_end_to_end_real_llm_sh_job_runs_then_queue_drains_then_unit_restored_with_same_profile():
    """The scenario the whole autoswap exists for, driven through the REAL daemon loop and the REAL llm.sh
    (only systemctl/nvidia-smi/systemd-run/curl are stubbed -- no real systemd unit or GPU is ever touched): a
    profile is loaded, a job is queued, the dispatcher stops the model, the job runs, and once the queue drains
    the SAME profile is restored -- proving the BLOCKER (marker deleted by llm.sh's own `off`) and the LOW
    (restore the same profile) fixes together, with neither gpu_queue.sh's own stop/restore functions nor
    llm.sh's on/off/__current_profile stubbed out."""
    with tempfile.TemporaryDirectory() as tmp:
        loaded_profile = "qwen38-27b-iq4nl-mtp"   # a REAL profile name (tools/local_llm/profiles.json)
        systemctl, state_file, desc_file = _fake_systemctl(tmp, "active", desc="local LLM (%s)" % loaded_profile)
        bindir = _fake_systemd_run_and_curl(tmp, state_file, desc_file)
        nvidia_smi = _fake_nvidia_smi(tmp)   # plenty of free VRAM, no resident brain process
        sentinel = os.path.join(tmp, "job_ran")
        subprocess.run(["bash", _GPU_QUEUE, "add", "touch %s" % sentinel],
                        env={**os.environ, "GPU_QUEUE_DIR": tmp}, check=True, capture_output=True, text=True, timeout=10)
        env = dict(os.environ)
        env["PATH"] = bindir + os.pathsep + env["PATH"]
        env.update({
            "GPU_QUEUE_DIR": tmp, "LOCAL_LLM_SYSTEMCTL": systemctl, "GPU_QUEUE_NVIDIA_SMI": nvidia_smi,
            "GPU_QUEUE_POLL_SEC": "1",
            # GPU_QUEUE_LLM_SH intentionally NOT overridden -- gpu_queue.sh must resolve and use the REAL llm.sh.
        })
        daemon = subprocess.Popen(["bash", _GPU_QUEUE, "__daemon"], env=env,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            assert _wait_for(lambda: os.path.exists(sentinel), timeout=10), "the queued job never ran"
            restored = _wait_for(
                lambda: open(state_file).read().strip() == "active"
                and open(desc_file).read().strip() == "local LLM (%s)" % loaded_profile,
                timeout=10)
            assert restored, "unit not restored with the SAME profile once the queue drained (state=%r desc=%r)" % (
                open(state_file).read(), open(desc_file).read())
            assert not os.path.exists(os.path.join(tmp, ".local_llm_was_on")), "marker must be consumed, not left dangling"
        finally:
            daemon.terminate()
            try:
                daemon.wait(timeout=5)
            except subprocess.TimeoutExpired:
                daemon.kill()


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
    marker itself to make that true (previously the marker was only ever written by gpu_queue's own
    llm_stop_for_job, so a refusal against an ALREADY-unloaded model never auto-loaded anything)."""
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
    """The exact scenario named in the task: gpu_queue stopped the model for a job (marker set), the owner
    then runs `llm off` themselves mid-job -- the marker must be gone so the eventual queue-drain restore
    never fires."""
    with tempfile.TemporaryDirectory() as tmp:
        marker = os.path.join(tmp, ".local_llm_was_on")
        open(marker, "w").write("qwen38-27b-iq4nl-mtp\n")   # as gpu_queue's llm_stop_for_job would have left it
        systemctl, _, _ = _fake_systemctl(tmp, "inactive")   # already stopped, exactly like the mid-job scenario
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


def test_user_bus_env_is_supplied_when_the_dispatcher_runs_without_a_session():
    """2026-09-25 LIVE failure: gpu-queue-dispatch is a SYSTEM unit with no XDG_RUNTIME_DIR /
    DBUS_SESSION_BUS_ADDRESS, so `systemctl --user` could not reach the user bus, llm_is_active() read the
    running model as inactive, and a queued job waited forever for VRAM. The stub below behaves like the real
    systemctl: it cannot see the unit unless the user-bus variables are set. gpu_queue.sh must supply them."""
    if not os.path.isdir("/run/user/%d" % os.getuid()) or not os.path.exists("/run/user/%d/bus" % os.getuid()):
        import pytest
        pytest.skip("no user runtime dir / bus on this machine")
    with tempfile.TemporaryDirectory() as tmp:
        state = os.path.join(tmp, "fake_unit_state")
        open(state, "w").write("active")
        systemctl = _write_exec(os.path.join(tmp, "bus_aware_systemctl.sh"), """
if [ -z "$XDG_RUNTIME_DIR" ] || [ -z "$DBUS_SESSION_BUS_ADDRESS" ]; then
  echo "Failed to connect to user scope bus via local transport" >&2; exit 1
fi
case "$*" in
  *"is-active --quiet local-llm"*) [ "$(cat %s)" = "active" ] ;;
  *"stop local-llm"*) echo inactive > %s ;;
  *) exit 0 ;;
esac
""" % (state, state))
        llm_sh, _ = _fake_llm_sh(tmp, profile="qwen38-27b-iq4nl-mtp-128k-q4")
        env = {k: v for k, v in os.environ.items() if k not in ("XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS")}
        env.update({"GPU_QUEUE_DIR": tmp, "LOCAL_LLM_SYSTEMCTL": systemctl, "GPU_QUEUE_LLM_SH": llm_sh})
        res = subprocess.run(["bash", _GPU_QUEUE, "__llm_stop_for_job"], env=env, capture_output=True, text=True,
                             timeout=15)
        assert res.returncode == 0, res.stderr
        assert open(state).read().strip() == "inactive", (
            "the running model was not stopped: gpu_queue.sh did not supply the user-bus environment")
        assert os.path.exists(os.path.join(tmp, ".local_llm_was_on"))
