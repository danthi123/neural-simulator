"""tools/agent_gateway_submit.py's build_bwrap_argv -- FUNCTIONAL sandbox-escape tests (2026-09-29 re-review,
HIGH-1). The independent re-review found the earlier `--ro-bind / /` design escapable: host /proc was visible
with no PID namespace (so a same-uid process's /proc/<pid>/environ -- live ANTHROPIC/AWS/GH tokens from the
dispatcher, local-llm, Claude Code -- was readable), and /run/user/<uid>/bus was reachable with
DBUS_SESSION_BUS_ADDRESS inherited (so `systemctl --user` could start a transient unit OUTSIDE the sandbox).

These tests run the REAL `bwrap` binary (available on this workstation) against the REAL argv build_bwrap_argv
produces, inside SCRATCH checkout/output directories -- never the live repo's own credentials or the real GPU
(these six use `/usr/bin/true`-style commands, not cupy; the separate real-GPU smoke test lives in
test_agent_gateway_submit_gpu_smoke.py). No network call, no dbus session, no real credential is ever touched;
the whole point is proving the sandbox itself denies reaching them.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import time

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "tools"))
import agent_gateway_submit as submit  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("bwrap") is None, reason="bwrap is not installed on this machine")


def _build(tmp, extra_inner=None):
    checkout = os.path.join(tmp, "checkout")
    job_out = os.path.join(tmp, "checkout_out")
    home = os.path.join(tmp, "fakehome")
    venv = os.path.join(tmp, "fakevenv")
    os.makedirs(checkout, exist_ok=True)
    os.makedirs(job_out, exist_ok=True)
    os.makedirs(os.path.join(home, ".ssh"), exist_ok=True)
    with open(os.path.join(home, ".ssh", "id_ed25519"), "w") as f:
        f.write("PRETEND-PRIVATE-KEY-NEVER-READABLE-FROM-THE-SANDBOX\n")
    os.makedirs(venv, exist_ok=True)   # empty stand-in -- these tests don't need real cupy
    inner = extra_inner or ["/usr/bin/true"]
    argv = submit.build_bwrap_argv(checkout, job_out, inner, home=home, repo_venv=venv)
    return argv, checkout, job_out, home


def _run(argv, timeout=15):
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout)


def test_proc_shows_only_the_sandboxs_own_pids():
    with tempfile.TemporaryDirectory() as tmp:
        argv, checkout, job_out, home = _build(tmp, ["/usr/bin/sh", "-c", "ls /proc | grep -E '^[0-9]+$'"])
        res = _run(argv)
        assert res.returncode == 0, res.stdout + res.stderr
        pids = res.stdout.split()
        assert pids, "expected at least the sandboxed process's own pid"
        # a fresh pid namespace for a single short-lived command shows only pid 1 (and maybe 2) -- never
        # anything resembling a real host pid count in the thousands.
        assert all(int(p) <= 5 for p in pids), "unexpectedly many/large pids visible: %s" % pids


def test_another_processs_environ_is_not_readable():
    marker = "AGENT_SANDBOX_ESCAPE_TEST_SECRET_TOKEN_%d" % os.getpid()
    env = dict(os.environ)
    env[marker] = "super-secret-value"
    proc = subprocess.Popen(["sleep", "30"], env=env)
    try:
        time.sleep(0.2)
        real_pid = proc.pid
        with tempfile.TemporaryDirectory() as tmp:
            argv, checkout, job_out, home = _build(
                tmp, ["/usr/bin/sh", "-c", "cat /proc/%d/environ 2>&1; echo RC=$?" % real_pid])
            res = _run(argv)
            assert marker not in res.stdout, (
                "the sandbox could read another (same-uid, real-host) process's environ: %s" % res.stdout)
            assert "RC=0" not in res.stdout, "reading the real host pid's environ must FAIL, not succeed"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_dbus_and_systemd_user_are_unreachable():
    with tempfile.TemporaryDirectory() as tmp:
        argv, checkout, job_out, home = _build(tmp, [
            "/usr/bin/sh", "-c",
            "test -d /run/user/$(id -u) && echo XDG_RUNTIME_EXISTS; "
            "echo DBUS=[$DBUS_SESSION_BUS_ADDRESS]; "
            "test -d /run && echo RUN_EXISTS"
        ])
        res = _run(argv)
        assert "XDG_RUNTIME_EXISTS" not in res.stdout
        assert "DBUS=[]" in res.stdout, res.stdout   # cleared by --clearenv, never inherited
        assert "RUN_EXISTS" not in res.stdout, "/run must not exist inside the sandbox at all"


def test_network_is_unreachable():
    with tempfile.TemporaryDirectory() as tmp:
        argv, checkout, job_out, home = _build(tmp, [
            "/usr/bin/python3.11", "-c",
            "import socket\n"
            "s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n"
            "s.settimeout(2)\n"
            "try:\n"
            "    s.connect(('1.1.1.1', 443))\n"
            "    print('REACHABLE')\n"
            "except OSError as e:\n"
            "    print('BLOCKED', e)\n"
        ])
        res = _run(argv)
        assert "REACHABLE" not in res.stdout
        assert "BLOCKED" in res.stdout, res.stdout + res.stderr


def test_home_dotfiles_are_absent():
    with tempfile.TemporaryDirectory() as tmp:
        argv, checkout, job_out, home = _build(tmp, [
            "/usr/bin/sh", "-c",
            "echo HOME=$HOME; ls -la $HOME 2>&1; test -f $HOME/.ssh/id_ed25519 && echo KEY_VISIBLE"
        ])
        res = _run(argv)
        assert "HOME=/tmp/home" in res.stdout
        assert "KEY_VISIBLE" not in res.stdout
        assert "PRETEND-PRIVATE-KEY" not in res.stdout


def test_write_outside_the_two_binds_fails():
    with tempfile.TemporaryDirectory() as tmp:
        argv, checkout, job_out, home = _build(tmp, [
            "/usr/bin/sh", "-c",
            "(: > /usr/evil_write_test) 2>&1 || echo USR_REFUSED; "
            "(: > /sys/evil_write_test) 2>&1 || echo SYS_REFUSED; "
            "(: > /tmp/inside_tmpfs_ok) 2>&1 && echo TMP_OK"
        ])
        res = _run(argv)
        assert "USR_REFUSED" in res.stdout
        assert "SYS_REFUSED" in res.stdout
        assert "TMP_OK" in res.stdout   # the fresh --tmpfs /tmp is writable -- not a leak, its own scratch space


def test_only_the_job_out_dir_is_writable_the_checkout_is_ro_bound():
    # 2026-09-29 re-review, L1: the checkout is RO-BOUND (a job has no reason to modify the pinned revision it
    # runs against) -- only its own per-job output subdirectory is writable.
    with tempfile.TemporaryDirectory() as tmp:
        checkout = os.path.join(tmp, "checkout")
        job_out = os.path.join(tmp, "checkout_out")
        argv, checkout, job_out, home = _build(tmp, [
            "/usr/bin/sh", "-c",
            "(: > %s/w1) 2>&1 || echo CHECKOUT_REFUSED; "
            "(: > %s/w2) 2>&1 && echo JOBOUT_WRITABLE" % (checkout, job_out)
        ])
        res = _run(argv)
        assert "CHECKOUT_REFUSED" in res.stdout
        assert "JOBOUT_WRITABLE" in res.stdout
        assert not os.path.exists(os.path.join(checkout, "w1"))
        assert os.path.exists(os.path.join(job_out, "w2")), "the write must be REAL (land on the host path)"
