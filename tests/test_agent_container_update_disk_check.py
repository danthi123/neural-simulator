"""tools/agent_container/update.sh's disk-usage check (2026-09-29 security review, M3): homelab-server is a
SHARED box with no per-container quota on the two named volumes, so update.sh warns loudly (every cycle) when
the filesystem hosting docker's data is over a threshold -- a coarse, always-available floor independent of
tools/inspection_digest.py's finer per-volume breakdown (which needs the ssh ctl path to be up).

Hermetic: AGENT_UPDATE_DF stubs `df` entirely (no real filesystem is ever inspected), and a stub `docker` is put
first on PATH so the script's own build/recreate logic (unrelated to this check, but unconditionally reached
right after it) never invokes a real Docker daemon.
"""
import os
import stat
import subprocess
import tempfile

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_UPDATE = os.path.join(_ROOT, "tools", "agent_container", "update.sh")


def _write_exec(path, body):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("#!/usr/bin/env bash\n" + body)
    os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return path


def _fake_df(tmp, pct):
    # matches `df --output=pcent <path>` -- a header line then the value, right-padded, with a trailing '%'
    # exactly like the real coreutils `df` prints it.
    return _write_exec(os.path.join(tmp, "fake_df.sh"),
                        'echo "Use%%"\necho " %s%%"\n' % pct)


def _fake_docker(tmp):
    log = os.path.join(tmp, "docker_calls.log")
    _write_exec(os.path.join(tmp, "fakebin", "docker"),
                'echo "docker $*" >> "%s"\n'
                'case "$1 $2" in\n'
                '  "inspect -f") echo absent ;;\n'
                '  *) exit 0 ;;\n'
                'esac\n' % log)
    return os.path.join(tmp, "fakebin"), log


def _scaffold(tmp):
    here = os.path.join(tmp, "deployed")
    os.makedirs(here, exist_ok=True)
    with open(os.path.join(here, "compose.yaml"), "w") as f:
        f.write("name: sim-agent\nservices: {}\n")
    return here


def _run(tmp, df_pct, extra_env=None, timeout=15):
    here = _scaffold(tmp)
    df = _fake_df(tmp, df_pct)
    fakebin, docker_log = _fake_docker(tmp)
    env = dict(os.environ)
    env["PATH"] = fakebin + os.pathsep + env["PATH"]
    env["AGENT_UPDATE_DF"] = df
    if extra_env:
        env.update(extra_env)
    res = subprocess.run(["bash", _UPDATE, here], env=env, capture_output=True, text=True, timeout=timeout)
    log_path = os.path.join(here, "update.log")
    log_text = open(log_path).read() if os.path.exists(log_path) else ""
    return res, log_text


def test_logs_disk_usage_percentage_every_cycle():
    with tempfile.TemporaryDirectory() as tmp:
        res, log_text = _run(tmp, df_pct=42)
        assert res.returncode == 0, res.stdout + res.stderr
        assert "disk usage at" in log_text
        assert "42%" in log_text
        assert "WARNING" not in log_text


def test_warns_loudly_when_over_the_threshold():
    with tempfile.TemporaryDirectory() as tmp:
        res, log_text = _run(tmp, df_pct=91, extra_env={"AGENT_UPDATE_DISK_WARN_PCT": "85"})
        assert res.returncode == 0, res.stdout + res.stderr
        assert "WARNING" in log_text
        assert "91%" in log_text
        assert "SHARED server" in log_text


def test_does_not_warn_exactly_at_the_boundary_below_threshold():
    with tempfile.TemporaryDirectory() as tmp:
        res, log_text = _run(tmp, df_pct=84, extra_env={"AGENT_UPDATE_DISK_WARN_PCT": "85"})
        assert res.returncode == 0, res.stdout + res.stderr
        assert "WARNING" not in log_text


def test_disk_check_failure_is_non_fatal_and_build_still_proceeds():
    with tempfile.TemporaryDirectory() as tmp:
        here = _scaffold(tmp)
        fakebin, docker_log = _fake_docker(tmp)
        env = dict(os.environ)
        env["PATH"] = fakebin + os.pathsep + env["PATH"]
        env["AGENT_UPDATE_DF"] = os.path.join(tmp, "no-such-df-binary")
        res = subprocess.run(["bash", _UPDATE, here], env=env, capture_output=True, text=True, timeout=15)
        assert res.returncode == 0, res.stdout + res.stderr
        log_text = open(os.path.join(here, "update.log")).read()
        assert "skipped (non-fatal)" in log_text
        # the build step must still have run despite the disk check being unreadable
        assert "docker compose" in open(docker_log).read() or "docker" in open(docker_log).read()
