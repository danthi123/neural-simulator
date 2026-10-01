"""GPU YIELD (2026-09-29, docs/plans/2026-09-29-local-model-transition-PLAN.md section 3): with the local model
now the day-to-day DRIVER of this repo, tools/gpu_queue.sh no longer auto-stops it to run a queued job (that
retired design -- and the tests that pinned it -- live in tests/test_llm_gpu_autoswap.py's history/docstring).
Instead:
  - a queued GPU job is HELD (never started) for as long as the local-llm unit is active and no yield is in force;
  - `tools/local_llm/llm.sh yield` is the driver's OWN explicit handoff: it refuses under GAME_MODE or with an
    empty GPU queue, else it writes research/queue/.llm_yield (profile/deadline) and stops itself gracefully
    (~45s delay, so it can finish replying inside its own chat turn);
  - gpu_queue.sh's dispatcher drains its queue while that marker exists, starts no NEW job once the deadline has
    passed (a job already running is left to finish), and once the window is genuinely drained removes the
    marker and restarts the server for the SAME profile with the model unloaded (`llm on --lazy`);
  - 2026-09-30: the driver is Hermes on homelab-server (it yields through tools/gpu_remote.sh), so the
    workstation draft-worktree refusal and the headless resume of a local session are gone -- a resumed local
    session would be a second driver beside Hermes, whose own sim-driver wake gate starts its next run;
  - GAME_MODE (tools/game.sh) takes precedence over all of this, always.

Hermetic throughout: every test runs against an isolated GPU_QUEUE_DIR (never the live research/queue/) and stubs
systemctl (LOCAL_LLM_SYSTEMCTL), llm.sh itself (GPU_QUEUE_LLM_SH, where used), nvidia-smi (GPU_QUEUE_NVIDIA_SMI),
and curl/systemd-run via PATH where `llm yield` itself is exercised -- no real systemd unit or GPU is ever
touched.
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


def _fake_systemctl(tmp, state, desc="local LLM (test-profile)"):
    """A stand-in for `systemctl --user ...`: `state` file holds "active"/"inactive"; `stop`/`start` flip it.
    Returns (systemctl_path, state_file, desc_file)."""
    state_file = os.path.join(tmp, "fake_unit_state")
    desc_file = os.path.join(tmp, "fake_unit_desc")
    open(state_file, "w").write(state)
    open(desc_file, "w").write(desc)
    path = _write_exec(os.path.join(tmp, "fake_systemctl.sh"), """
STATE_FILE="%s"
DESC_FILE="%s"
case "$*" in
  *"is-active --quiet local-llm"*) [ "$(cat "$STATE_FILE" 2>/dev/null)" = "active" ] ;;
  *"stop local-llm"*) echo inactive > "$STATE_FILE" ;;
  *"start local-llm"*) echo active > "$STATE_FILE" ;;
  *"reset-failed local-llm"*) exit 0 ;;
  *"show -p Description"*) cat "$DESC_FILE" 2>/dev/null ;;
  *) exit 0 ;;
esac
""" % (state_file, desc_file))
    return path, state_file, desc_file


def _fake_nvidia_smi(tmp, resident_pid=""):
    return _write_exec(os.path.join(tmp, "fake_nvidia_smi.sh"), """
case "$*" in
  *--query-compute-apps*) [ -n "%s" ] && echo "%s" ;;
  *--query-gpu*) echo "20000" ;;
esac
""" % (resident_pid, resident_pid))


def _fake_llm_sh_yield(tmp, state_file=None, on_exit=0):
    """A stand-in for tools/local_llm/llm.sh, for testing gpu_queue.sh's OWN drain/restore logic in isolation.
    Every invocation is appended to `log`. `on <profile>` flips the shared fake-systemctl state file to active
    (so a later poll sees the "reload" as real) when `state_file` is given; `claude ...` just logs and exits 0."""
    log = os.path.join(tmp, "fake_llm_sh.log")
    flip = 'echo active > "%s"\n' % state_file if state_file else ""
    _write_exec(os.path.join(tmp, "fake_llm.sh"), """
echo "CALLED: $*" >> "%s"
case "${1:-}" in
  on) %s exit %d ;;
  *) exit 0 ;;
esac
""" % (log, flip, on_exit))
    return os.path.join(tmp, "fake_llm.sh"), log


def _fake_curl_and_systemd_run(tmp, state_file):
    """`curl` (llm.sh's own is_up() health check) succeeds iff the fake unit is active; `systemd-run` just logs
    its full argv (used to verify a graceful stop's --on-active=<n>s delay) and exits 0. Returns a PATH dir."""
    bindir = os.path.join(tmp, "fakebin")
    os.makedirs(bindir, exist_ok=True)
    _write_exec(os.path.join(bindir, "curl"), '[ "$(cat "%s" 2>/dev/null)" = "active" ]\n' % state_file)
    _write_exec(os.path.join(bindir, "systemd-run"),
                'echo "$*" >> "%s"\nexit 0\n' % os.path.join(tmp, "systemd_run.log"))
    return bindir


def _write_yield_marker(tmp, session="sess-1", profile="prof-1", deadline_in=3600, deadline_abs=None):
    deadline = deadline_abs if deadline_abs is not None else int(time.time()) + deadline_in
    open(os.path.join(tmp, ".llm_yield"), "w").write(
        "LLM_YIELD_SESSION=%s\nLLM_YIELD_PROFILE=%s\nLLM_YIELD_DEADLINE=%d\n" % (session, profile, deadline))
    return deadline


def _add_job(tmp, cmd):
    subprocess.run(["bash", _GPU_QUEUE, "add", cmd], env={**os.environ, "GPU_QUEUE_DIR": tmp},
                    check=True, capture_output=True, text=True, timeout=10)


def _daemon_env(tmp, systemctl, nvidia_smi, poll="1", llm_sh=None):
    env = dict(os.environ)
    env.update({"GPU_QUEUE_DIR": tmp, "LOCAL_LLM_SYSTEMCTL": systemctl, "GPU_QUEUE_NVIDIA_SMI": nvidia_smi,
                "GPU_QUEUE_POLL_SEC": poll})
    if llm_sh:
        env["GPU_QUEUE_LLM_SH"] = llm_sh
    return env


def _spawn_daemon(env):
    return subprocess.Popen(["bash", _GPU_QUEUE, "__daemon"], env=env,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _kill(proc):
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


def _run(args, env_extra, timeout=15):
    env = dict(os.environ)
    env.update(env_extra)
    return subprocess.run(["bash", _GPU_QUEUE] + args, env=env, capture_output=True, text=True, timeout=timeout)


def _run_llm(args, env_extra, timeout=10):
    env = dict(os.environ)
    env.update(env_extra)
    return subprocess.run(["bash", _LLM_SH] + args, env=env, capture_output=True, text=True, timeout=timeout)


def _log_text(path):
    return open(path).read() if os.path.exists(path) else ""


def _wait_for(predicate, timeout=8.0, interval=0.05):
    deadline = time.time() + timeout
    ok = predicate()
    while not ok and time.time() < deadline:
        time.sleep(interval)
        ok = predicate()
    return ok


def _spawn_fake_resident_pid():
    proc = subprocess.Popen(["bash", "-c", 'exec -a "python -u -m research.runners.faketest_resident" sleep 60'])
    time.sleep(0.1)
    return proc


# ---------------------------------------------------------------------------------------------------------------
# gpu_queue.sh __daemon: HELD semantics (no yield marker) and the yield window itself
# ---------------------------------------------------------------------------------------------------------------

def test_job_held_while_llm_active_and_no_yield_marker():
    """The core semantics change: a queued job must never dispatch, and the unit must never be touched, while
    local-llm is active and nothing has yielded."""
    with tempfile.TemporaryDirectory() as tmp:
        systemctl, state_file, _ = _fake_systemctl(tmp, "active")
        nvidia_smi = _fake_nvidia_smi(tmp)
        sentinel = os.path.join(tmp, "job_ran")
        _add_job(tmp, "touch %s" % sentinel)
        daemon = _spawn_daemon(_daemon_env(tmp, systemctl, nvidia_smi))
        try:
            time.sleep(3)
            assert not os.path.exists(sentinel), "the job ran while local-llm was active and no yield was in force"
            assert open(state_file).read().strip() == "active", "the driver must never be stopped by the queue itself"
        finally:
            _kill(daemon)


def test_job_runs_once_yielded_and_unit_inactive():
    """Once the driver has yielded (unit down, marker present with time left), the queue drains normally."""
    with tempfile.TemporaryDirectory() as tmp:
        systemctl, _, _ = _fake_systemctl(tmp, "inactive")
        nvidia_smi = _fake_nvidia_smi(tmp)
        _write_yield_marker(tmp, deadline_in=3600)
        sentinel = os.path.join(tmp, "job_ran")
        _add_job(tmp, "touch %s" % sentinel)
        daemon = _spawn_daemon(_daemon_env(tmp, systemctl, nvidia_smi))
        try:
            assert _wait_for(lambda: os.path.exists(sentinel), timeout=10), "the queued job never ran during the yield window"
        finally:
            _kill(daemon)


def test_no_new_job_starts_after_yield_deadline_even_with_unit_inactive():
    """A job already dispatched would be left to finish, but a NEW one must not start once the driver's own
    promised window has expired -- even though the unit is genuinely down (nothing to hold it back but the clock)."""
    with tempfile.TemporaryDirectory() as tmp:
        systemctl, state_file, _ = _fake_systemctl(tmp, "inactive")
        nvidia_smi = _fake_nvidia_smi(tmp)
        llm_sh, llm_log = _fake_llm_sh_yield(tmp, state_file=state_file)
        _write_yield_marker(tmp, deadline_abs=int(time.time()) - 10)
        sentinel = os.path.join(tmp, "job_ran")
        _add_job(tmp, "touch %s" % sentinel)
        daemon = _spawn_daemon(_daemon_env(tmp, systemctl, nvidia_smi, llm_sh=llm_sh))
        try:
            time.sleep(3)
            assert not os.path.exists(sentinel), "a new job started after the yield deadline had already passed"
            # 2026-09-29 fix: an expired window with jobs still queued must RESTORE the driver (it used to hold forever)
            assert _wait_for(lambda: os.path.exists(llm_log) and " on" in (" " + open(llm_log).read()), timeout=10), \
                "the driver was never restored after the yield deadline passed with jobs still queued"
            assert not os.path.exists(os.path.join(tmp, ".llm_yield")), "the yield marker was not consumed"
            time.sleep(2)
            assert not os.path.exists(sentinel), "a held job started while/after the driver was being restored"
        finally:
            _kill(daemon)


def test_yield_window_drains_and_restarts_the_server_lazily_without_resuming_a_session():
    """The full cycle: a queued job runs during the yield window, the window then drains (queue empty, nothing
    running), and the dispatcher removes the marker and restarts the SAME profile with the model unloaded
    (`on <profile> --lazy`). A LLM_YIELD_SESSION line (an older marker) is ignored: no local session is ever
    resumed headless -- Hermes on homelab-server is the driver and starts its own next run."""
    with tempfile.TemporaryDirectory() as tmp:
        systemctl, state_file, _ = _fake_systemctl(tmp, "inactive")
        llm_sh, log = _fake_llm_sh_yield(tmp, state_file=state_file)
        nvidia_smi = _fake_nvidia_smi(tmp)
        _write_yield_marker(tmp, session="sess-42", profile="prof-x", deadline_in=3600)
        sentinel = os.path.join(tmp, "job_ran")
        _add_job(tmp, "touch %s" % sentinel)
        daemon = _spawn_daemon(_daemon_env(tmp, systemctl, nvidia_smi, llm_sh=llm_sh))
        try:
            assert _wait_for(lambda: os.path.exists(sentinel), timeout=10), "the queued job never ran"
            assert _wait_for(lambda: "CALLED: on prof-x --lazy" in _log_text(log), timeout=10), (
                "the SAME profile was not restarted lazily: %s" % _log_text(log))
            assert _wait_for(lambda: not os.path.exists(os.path.join(tmp, ".llm_yield")), timeout=10), "marker must be consumed once drained"
            assert not os.path.exists(os.path.join(tmp, ".llm_yield_job_count")), "the per-window job counter must be cleared with the marker"
            time.sleep(1)
            assert "CALLED: claude" not in _log_text(log), "no local session may be resumed: %s" % _log_text(log)
        finally:
            _kill(daemon)


def test_yield_restore_never_fires_under_game_mode():
    """GAME_MODE takes precedence over a yield's own drain -- checked directly, not only via GPU_PAUSE."""
    with tempfile.TemporaryDirectory() as tmp:
        _write_yield_marker(tmp, deadline_in=3600)
        open(os.path.join(tmp, "GAME_MODE"), "w").close()
        systemctl, _, _ = _fake_systemctl(tmp, "inactive")
        llm_sh, log = _fake_llm_sh_yield(tmp)
        nvidia_smi = _fake_nvidia_smi(tmp)
        res = _run(["__llm_yield_restore_if_drained"], {
            "GPU_QUEUE_DIR": tmp, "LOCAL_LLM_SYSTEMCTL": systemctl, "GPU_QUEUE_LLM_SH": llm_sh,
            "GPU_QUEUE_NVIDIA_SMI": nvidia_smi,
        })
        assert res.returncode == 0, res.stderr
        time.sleep(0.3)
        assert os.path.exists(os.path.join(tmp, ".llm_yield")), "GAME_MODE must defer, not drop, the marker"
        assert "CALLED" not in _log_text(log)


def test_yield_restore_defers_while_a_brain_process_is_still_gpu_resident():
    with tempfile.TemporaryDirectory() as tmp:
        _write_yield_marker(tmp, deadline_in=3600)
        systemctl, _, _ = _fake_systemctl(tmp, "inactive")
        llm_sh, log = _fake_llm_sh_yield(tmp)
        resident = _spawn_fake_resident_pid()
        try:
            nvidia_smi = _fake_nvidia_smi(tmp, resident_pid=str(resident.pid))
            res = _run(["__llm_yield_restore_if_drained"], {
                "GPU_QUEUE_DIR": tmp, "LOCAL_LLM_SYSTEMCTL": systemctl, "GPU_QUEUE_LLM_SH": llm_sh,
                "GPU_QUEUE_NVIDIA_SMI": nvidia_smi,
            })
            assert res.returncode == 0, res.stderr
            time.sleep(0.3)
            assert os.path.exists(os.path.join(tmp, ".llm_yield")), "a resident brain process must defer, not drop, the marker"
        finally:
            resident.kill()
            resident.wait(timeout=5)


def test_llm_is_active_needs_user_bus_env_supplied_by_the_dispatcher():
    """2026-09-25 LIVE failure (see gpu_queue.sh's own comment): without XDG_RUNTIME_DIR/DBUS_SESSION_BUS_ADDRESS,
    `systemctl --user` fails closed and llm_is_active() misreads a genuinely ACTIVE unit as inactive -- under the
    NEW design that would let a queued job wrongly bypass the HELD guard and dispatch straight over a loaded
    model. gpu_queue.sh must supply the bus env itself."""
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
  *) exit 0 ;;
esac
""" % state)
        env = {k: v for k, v in os.environ.items() if k not in ("XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS")}
        env.update({"GPU_QUEUE_DIR": tmp, "LOCAL_LLM_SYSTEMCTL": systemctl})
        res = subprocess.run(["bash", _GPU_QUEUE, "__llm_is_active"], env=env, capture_output=True, text=True, timeout=15)
        assert res.returncode == 0, res.stderr
        assert res.stdout.strip() == "active", (
            "gpu_queue.sh did not supply the user-bus environment -- a genuinely active unit read as inactive")


# ---------------------------------------------------------------------------------------------------------------
# llm.sh: `yield` refusals and its success path (real script; curl/systemd-run stubbed where touched)
# ---------------------------------------------------------------------------------------------------------------

def test_yield_refuses_under_game_mode():
    with tempfile.TemporaryDirectory() as tmp:
        open(os.path.join(tmp, "GAME_MODE"), "w").close()
        open(os.path.join(tmp, "gpu.queue"), "w").write("some job\n")
        res = _run_llm(["yield"], {"GPU_QUEUE_DIR": tmp, "LLM_PORT": "18080"})
        assert res.returncode != 0
        assert "GAME_MODE" in res.stdout
        assert not os.path.exists(os.path.join(tmp, ".llm_yield"))


def test_yield_refuses_with_empty_gpu_queue():
    with tempfile.TemporaryDirectory() as tmp:
        res = _run_llm(["yield"], {"GPU_QUEUE_DIR": tmp, "LLM_PORT": "18080"})
        assert res.returncode != 0
        assert "empty" in res.stdout.lower()
        assert not os.path.exists(os.path.join(tmp, ".llm_yield"))


def test_yield_refuses_when_the_model_is_not_loaded():
    with tempfile.TemporaryDirectory() as tmp:
        open(os.path.join(tmp, "gpu.queue"), "w").write("some job\n")
        res = _run_llm(["yield"], {"GPU_QUEUE_DIR": tmp, "LLM_PORT": "18080"})
        assert res.returncode != 0
        assert "not loaded" in res.stdout.lower()


def test_yield_writes_marker_and_schedules_graceful_stop():
    with tempfile.TemporaryDirectory() as tmp:
        open(os.path.join(tmp, "gpu.queue"), "w").write("some job\n")
        systemctl, state_file, _ = _fake_systemctl(tmp, "active", desc="local LLM (myprofile)")
        bindir = _fake_curl_and_systemd_run(tmp, state_file)
        state_home = os.path.join(tmp, "state")
        os.makedirs(os.path.join(state_home, "sim-local-llm"))
        open(os.path.join(state_home, "sim-local-llm", "last_session"), "w").write("session-xyz\n")
        env = dict(os.environ)
        env["PATH"] = bindir + os.pathsep + env["PATH"]
        env.update({"GPU_QUEUE_DIR": tmp, "LOCAL_LLM_SYSTEMCTL": systemctl,
                    "XDG_STATE_HOME": state_home})
        res = _run_llm(["yield", "--max-hours", "2"], env)
        assert res.returncode == 0, res.stdout + res.stderr
        marker = os.path.join(tmp, ".llm_yield")
        assert os.path.exists(marker)
        content = open(marker).read()
        assert "LLM_YIELD_SESSION" not in content, "a local session must never be recorded for a headless resume"
        assert "LLM_YIELD_PROFILE=myprofile" in content
        deadline_line = next(l for l in content.splitlines() if l.startswith("LLM_YIELD_DEADLINE="))
        deadline = int(deadline_line.split("=", 1)[1])
        assert abs(deadline - (time.time() + 2 * 3600)) < 30, "deadline must reflect --max-hours 2"
        sysrun_log = os.path.join(tmp, "systemd_run.log")
        assert _wait_for(lambda: os.path.exists(sysrun_log))
        assert "--on-active=45s" in open(sysrun_log).read(), "the stop must be scheduled gracefully (~45s), not immediate"


def test_yield_refuses_a_max_hours_value_that_is_not_plain_digits():
    """2026-09-29 security review, H2: `deadline=$(( $(date +%s) + max_hours * 3600 ))` is bash ARITHMETIC
    expansion, which recursively re-expands a variable's STRING VALUE as if it were itself an arithmetic
    expression -- so `--max-hours 'a[$(cmd)]'` made bash evaluate the embedded command substitution and run
    `cmd`, not merely fail to parse as a number (confirmed standalone: `max_hours="a[$(touch marker)]";
    $(( ... + max_hours * 3600 ))` creates `marker` even though the surrounding expression then errors on a bad
    array subscript). --max-hours must be validated as 1-2 plain digits BEFORE it ever reaches `$(( ))`."""
    with tempfile.TemporaryDirectory() as tmp, \
         tempfile.TemporaryDirectory() as scratch:
        marker = os.path.join(scratch, "pwned")
        open(os.path.join(tmp, "gpu.queue"), "w").write("some job\n")
        systemctl, state_file, _ = _fake_systemctl(tmp, "active", desc="local LLM (myprofile)")
        bindir = _fake_curl_and_systemd_run(tmp, state_file)
        env = dict(os.environ)
        env["PATH"] = bindir + os.pathsep + env["PATH"]
        env.update({"GPU_QUEUE_DIR": tmp, "LOCAL_LLM_SYSTEMCTL": systemctl})
        res = _run_llm(["yield", "--max-hours", "a[$(touch %s)]" % marker], env)
        assert res.returncode != 0
        assert "1-12" in res.stdout
        assert not os.path.exists(marker), "--max-hours must never be evaluated as a shell/arithmetic injection"
        assert not os.path.exists(os.path.join(tmp, ".llm_yield")), "no yield marker must be written on refusal"


def test_yield_refuses_a_max_hours_value_over_12():
    """2026-09-29 re-review, M2: the digit-only check alone still let a container-triggered `yield --max-hours
    N` claim an arbitrarily long window (up to 99h) over the shared GPU -- capped at 12."""
    with tempfile.TemporaryDirectory() as tmp:
        open(os.path.join(tmp, "gpu.queue"), "w").write("some job\n")
        systemctl, state_file, _ = _fake_systemctl(tmp, "active", desc="local LLM (myprofile)")
        bindir = _fake_curl_and_systemd_run(tmp, state_file)
        env = dict(os.environ)
        env["PATH"] = bindir + os.pathsep + env["PATH"]
        env.update({"GPU_QUEUE_DIR": tmp, "LOCAL_LLM_SYSTEMCTL": systemctl})
        res = _run_llm(["yield", "--max-hours", "13"], env)
        assert res.returncode != 0
        assert "1-12" in res.stdout
        assert not os.path.exists(os.path.join(tmp, ".llm_yield"))


def test_yield_accepts_a_max_hours_value_of_exactly_12():
    with tempfile.TemporaryDirectory() as tmp:
        open(os.path.join(tmp, "gpu.queue"), "w").write("some job\n")
        systemctl, state_file, _ = _fake_systemctl(tmp, "active", desc="local LLM (myprofile)")
        bindir = _fake_curl_and_systemd_run(tmp, state_file)
        env = dict(os.environ)
        env["PATH"] = bindir + os.pathsep + env["PATH"]
        env.update({"GPU_QUEUE_DIR": tmp, "LOCAL_LLM_SYSTEMCTL": systemctl})
        res = _run_llm(["yield", "--max-hours", "12"], env)
        assert res.returncode == 0, res.stdout + res.stderr
