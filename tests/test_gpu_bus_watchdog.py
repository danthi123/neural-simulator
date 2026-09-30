"""tools/gpu_bus_watchdog.sh -- the hardware-only auto-reboot watchdog for the RTX 3090 "off the bus" hang
(docs/GPU_CRASH_RECOVERY.md). It replaces ONLY the reboot half of the retired gpu-train-watchdog.service
(2026-09-28); it must NEVER restart training, a session, or any other process -- its only two real-world side
effects on a genuine detection are a wall(1) broadcast and a reboot.

Fully hermetic: journalctl and the reboot command are stubbed via this repo's existing TEST-ONLY env-var-override
convention (GPU_WATCHDOG_JOURNALCTL / GPU_WATCHDOG_REBOOT_CMD, matching tools/local_llm/llm.sh's
LOCAL_LLM_SYSTEMCTL and tools/gpu_queue.sh's GPU_QUEUE_NVIDIA_SMI / GPU_QUEUE_LLM_SH); `wall` has no dedicated
override in the script, so it is stubbed via a fake executable prepended to PATH instead, the same PATH-shimming
approach tests/test_llm_gpu_yield.py already uses for curl/systemd-run. No real journalctl, systemd unit, wall
broadcast, or reboot is ever touched.
"""
import os
import stat
import subprocess
import tempfile
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPT = os.path.join(_ROOT, "tools", "gpu_bus_watchdog.sh")


def _shq(s):
    """Single-quote-escape `s` for embedding in a generated shell script body."""
    return "'" + s.replace("'", "'\\''") + "'"


def _write_exec(path, body):
    with open(path, "w") as f:
        f.write("#!/bin/bash\n" + body)
    os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return path


def _fake_journalctl(tmp, lines):
    """A stand-in for `journalctl -k -b 0 -f`: prints `lines` (one per line) then exits. A real `-f` would
    block forever following new kernel messages, but the script only needs to see the lines drain through its
    `while read` loop -- a finite fake keeps these tests fast and deterministic while exercising the exact
    same line-by-line matching path production would use for the since-boot portion of the stream."""
    body = "".join("echo %s\n" % _shq(l) for l in lines)
    return _write_exec(os.path.join(tmp, "fake_journalctl.sh"), body)


def _fake_reboot(tmp, log):
    return _write_exec(os.path.join(tmp, "fake_reboot.sh"), 'echo "REBOOT: $*" >> %s\n' % _shq(log))


def _fake_wall_bindir(tmp, log):
    """Puts a fake `wall` on PATH ahead of the real one (see module docstring)."""
    bindir = os.path.join(tmp, "fakebin")
    os.makedirs(bindir, exist_ok=True)
    _write_exec(os.path.join(bindir, "wall"), 'echo "WALL: $*" >> %s\n' % _shq(log))
    return bindir


def _log_text(path):
    return open(path).read() if os.path.exists(path) else ""


def _run(env_extra, timeout=20):
    env = dict(os.environ)
    env.update(env_extra)
    return subprocess.run(["bash", _SCRIPT], env=env, capture_output=True, text=True, timeout=timeout)


def _base_env(tmp, lines, delay_s="1"):
    """Wires up fake journalctl/reboot/wall + an isolated marker path, all under `tmp`. Returns (env, log,
    marker)."""
    log = os.path.join(tmp, "actions.log")
    journalctl = _fake_journalctl(tmp, lines)
    reboot = _fake_reboot(tmp, log)
    bindir = _fake_wall_bindir(tmp, log)
    marker = os.path.join(tmp, "marker")
    env = {
        "GPU_WATCHDOG_JOURNALCTL": journalctl,
        "GPU_WATCHDOG_REBOOT_CMD": reboot,
        "GPU_WATCHDOG_MARKER": marker,
        "GPU_WATCHDOG_DELAY_S": delay_s,
        "PATH": bindir + os.pathsep + os.environ["PATH"],
    }
    return env, log, marker


# ---------------------------------------------------------------------------------------------------------------
# (a) no crash signature -> total no-op
# ---------------------------------------------------------------------------------------------------------------

def test_no_crash_signature_no_action():
    with tempfile.TemporaryDirectory() as tmp:
        env, log, marker = _base_env(tmp, [
            "Sep 29 10:00:00 host kernel: some entirely benign kernel line",
            "Sep 29 10:00:01 host kernel: NVRM: GPU0 fan speed nominal",
            "Sep 29 10:00:02 host kernel: usb 1-2: new high-speed USB device",
        ])
        res = _run(env)
        assert res.returncode == 0, res.stdout + res.stderr
        assert not os.path.exists(marker), "no marker should be written without a crash signature"
        assert _log_text(log) == "", "neither wall nor reboot should fire without a crash signature: %r" % _log_text(log)


# ---------------------------------------------------------------------------------------------------------------
# (b) crash signature present -> wall, then (after the delay) reboot, in that order; marker written
# ---------------------------------------------------------------------------------------------------------------

def test_crash_signature_triggers_wall_then_reboot_after_delay_and_writes_marker():
    with tempfile.TemporaryDirectory() as tmp:
        env, log, marker = _base_env(tmp, [
            "Sep 29 11:07:00 host kernel: NVRM: GPU0 _scrubWaitAndSave: Timed out waiting for scrub [NV_ERR_TIMEOUT]",
            "Sep 29 11:07:01 host kernel: NVRM: GPU lost from the bus",
        ], delay_s="1")
        start = time.time()
        res = _run(env)
        elapsed = time.time() - start
        assert res.returncode == 0, res.stdout + res.stderr
        assert elapsed >= 0.9, "the reboot must not be instant -- the delay was not honored (%.2fs elapsed)" % elapsed
        assert os.path.exists(marker), "the at-most-once-per-boot marker was never written"
        content = _log_text(log)
        assert "WALL:" in content, "wall(1) was never broadcast: %r" % content
        assert "REBOOT:" in content, "the reboot command was never invoked: %r" % content
        assert content.index("WALL:") < content.index("REBOOT:"), \
            "wall must be broadcast BEFORE the reboot, not after: %r" % content


def test_gpu_has_fallen_off_the_bus_phrasing_alone_triggers():
    """Robustness to a different driver/kernel phrasing of the same event -- no _scrubWaitAndSave line at all."""
    with tempfile.TemporaryDirectory() as tmp:
        env, log, marker = _base_env(tmp, [
            "Sep 29 11:07:00 host kernel: NVRM: Xid (PCI:0000:01:00): 79, GPU has fallen off the bus.",
        ], delay_s="1")
        res = _run(env)
        assert res.returncode == 0, res.stdout + res.stderr
        assert os.path.exists(marker)
        assert "REBOOT:" in _log_text(log)


def test_gpu_lost_from_the_bus_phrasing_alone_triggers():
    with tempfile.TemporaryDirectory() as tmp:
        env, log, marker = _base_env(tmp, [
            "Sep 29 11:07:00 host kernel: NVRM: GPU lost from the bus",
        ], delay_s="1")
        res = _run(env)
        assert res.returncode == 0, res.stdout + res.stderr
        assert os.path.exists(marker)
        assert "REBOOT:" in _log_text(log)


def test_scrubwaitandsave_without_nv_err_timeout_does_not_trigger():
    """The primary signature requires BOTH tokens together -- a bare _scrubWaitAndSave mention (e.g. in a log
    explaining the incident) must not, on its own, cause a reboot."""
    with tempfile.TemporaryDirectory() as tmp:
        env, log, marker = _base_env(tmp, [
            "Sep 29 11:07:00 host kernel: NVRM: GPU0 _scrubWaitAndSave: completed normally",
        ], delay_s="1")
        res = _run(env)
        assert res.returncode == 0, res.stdout + res.stderr
        assert not os.path.exists(marker)
        assert _log_text(log) == ""


# ---------------------------------------------------------------------------------------------------------------
# (c) marker already present this boot -> total no-op even with a crash signature in the log
# ---------------------------------------------------------------------------------------------------------------

def test_marker_already_present_this_boot_is_a_total_noop():
    with tempfile.TemporaryDirectory() as tmp:
        env, log, marker = _base_env(tmp, [
            "Sep 29 11:07:00 host kernel: NVRM: GPU0 _scrubWaitAndSave: Timed out ... [NV_ERR_TIMEOUT]",
        ], delay_s="1")
        with open(marker, "w") as f:
            f.write("already rebooted this boot\n")
        res = _run(env)
        assert res.returncode == 0, res.stdout + res.stderr
        assert _log_text(log) == "", \
            "neither wall nor reboot may fire once this boot has already triggered a reboot: %r" % _log_text(log)


# ---------------------------------------------------------------------------------------------------------------
# (d) the real 2-minute default (no override)
# ---------------------------------------------------------------------------------------------------------------

def test_default_delay_is_120_seconds_when_unset():
    """Asserts the script's OWN default constant rather than actually waiting 120s in a test."""
    text = open(_SCRIPT).read()
    assert 'DELAY_S="${GPU_WATCHDOG_DELAY_S:-120}"' in text, (
        "the documented 2-minute default (120s) must be the literal default used when "
        "GPU_WATCHDOG_DELAY_S is unset -- found no such default assignment in the script")


# ---------------------------------------------------------------------------------------------------------------
# (e) the wall message says why and roughly when
# ---------------------------------------------------------------------------------------------------------------

def test_wall_message_explains_why_and_roughly_when():
    with tempfile.TemporaryDirectory() as tmp:
        env, log, marker = _base_env(tmp, [
            "Sep 29 11:07:00 host kernel: NVRM: GPU0 _scrubWaitAndSave: Timed out ... [NV_ERR_TIMEOUT]",
        ], delay_s="3")
        res = _run(env)
        assert res.returncode == 0, res.stdout + res.stderr
        wall_line = next(l for l in _log_text(log).splitlines() if l.startswith("WALL:"))
        lower = wall_line.lower()
        assert "gpu" in lower and ("bus" in lower or "hang" in lower), \
            "the wall message must explain WHY the machine is rebooting: %r" % wall_line
        assert "3 second" in wall_line, \
            "the wall message must say roughly WHEN the reboot happens: %r" % wall_line
        assert "never restarts training" in lower or "never restart" in lower, \
            "the wall message should reassure that nothing but the OS reboot is automatic: %r" % wall_line


# ---------------------------------------------------------------------------------------------------------------
# marker path override itself
# ---------------------------------------------------------------------------------------------------------------

def test_marker_is_written_at_the_overridden_path_not_the_real_run_directory():
    with tempfile.TemporaryDirectory() as tmp:
        env, log, marker = _base_env(tmp, [
            "Sep 29 11:07:00 host kernel: NVRM: GPU0 _scrubWaitAndSave: Timed out ... [NV_ERR_TIMEOUT]",
        ], delay_s="1")
        assert marker.startswith(tmp), "the test's own marker fixture must live under the temp dir"
        real_run_marker_existed_before = os.path.exists("/run/gpu_bus_watchdog.rebooted")
        res = _run(env)
        assert res.returncode == 0, res.stdout + res.stderr
        assert os.path.exists(marker), "the marker must be written at the overridden GPU_WATCHDOG_MARKER path"
        # the real production marker path must be completely untouched by a test run
        assert os.path.exists("/run/gpu_bus_watchdog.rebooted") == real_run_marker_existed_before, \
            "a test must never touch the real /run marker path"
