"""tools/agent_container/update.sh -- the weekly rebuild + smoke + safe-swap cycle for the sim-agent stack on
srv-host-0 (replaces tests/test_agent_container_update_disk_check.py, 2026-09-30, whose disk-usage checks are kept
below). The script now drives the ONE ~/apps compose project, builds from the owner's clean git checkout, smoke-
tests a changed image before anything can run it, rolls the tag back on failure, and recreates the running
container only when that can neither lose work nor start it unfiltered.

Hermetic: a fake `docker` (first on PATH) records every call and answers from environment knobs; `df` and
`systemctl` are stubbed through the script's own TEST-ONLY overrides; ~/apps and the source checkout are temp dirs.
"""
import os
import re
import shutil
import stat
import subprocess
import tempfile

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_UPDATE = os.path.join(_ROOT, "tools", "agent_container", "update.sh")
_FRAGMENT = os.path.join(_ROOT, "tools", "agent_container", "server-stack", "sim-agent", "compose.yaml")

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")

FAKE_DOCKER = r'''#!/usr/bin/env bash
echo "docker $*" >> "$FAKE_T/docker_calls.log"
case "$1" in
  info) exit 0 ;;
  image)
    case "$2" in
      inspect) if [ -f "$FAKE_T/built" ]; then id="$FAKE_NEW_ID"; else id="$FAKE_OLD_ID"; fi
               [ -n "$id" ] && echo "$id" || exit 1 ;;
      tag) exit 0 ;;
    esac ;;
  compose)
    echo "cwd=$PWD" >> "$FAKE_T/docker_calls.log"
    case "$2" in
      build) touch "$FAKE_T/built"; exit "${FAKE_BUILD_RC:-0}" ;;
      run) exit "${FAKE_SMOKE_RC:-0}" ;;
      up) exit 0 ;;
    esac ;;
  inspect)
    case "$3" in
      *State.Status*) echo "$FAKE_STATUS" ;;
      *Image*) echo "$FAKE_CUR_IMAGE" ;;
    esac ;;
  exec) [ -n "$FAKE_BUSY" ] && echo "$FAKE_BUSY"; exit 0 ;;
  run) exit "${FAKE_PROBE_RC:-0}" ;;
esac
exit 0
'''


def _exec(path, body):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(body)
    os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return path


def _git(repo, *args):
    subprocess.run(["git", "-C", repo, *args], check=True, capture_output=True,
                   env=dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t",
                            GIT_COMMITTER_EMAIL="t@t"))


def _scaffold(tmp, include=True, drift=False):
    apps = os.path.join(tmp, "apps")
    os.makedirs(os.path.join(apps, "sim-agent"))
    with open(os.path.join(apps, "compose.yaml"), "w") as f:
        f.write("include:\n  - searxng/compose.yaml\n%s  - unifi/compose.yaml\n"
                % ("  - sim-agent/compose.yaml\n" if include else ""))
    frag = open(_FRAGMENT).read()
    with open(os.path.join(apps, "sim-agent", "compose.yaml"), "w") as f:
        f.write(frag + ("# hand edit on the box\n" if drift else ""))
    data = os.path.join(tmp, "data-sim-agent")
    src = os.path.join(data, "src")
    os.makedirs(os.path.join(src, "tools", "agent_container", "server-stack", "sim-agent"))
    os.makedirs(os.path.join(src, "tools", "agent_container", "bin"))
    with open(os.path.join(src, "tools", "agent_container", "server-stack", "sim-agent", "compose.yaml"), "w") as f:
        f.write(frag)
    for rel in ("tools/agent_container/Dockerfile", "tools/agent_container/bin/llm", "tools/agent_harness_ctl.sh"):
        with open(os.path.join(src, rel), "w") as f:
            f.write("# %s\n" % rel)
    subprocess.run(["git", "init", "-q", src], check=True)
    _git(src, "add", "-A")
    _git(src, "commit", "-q", "-m", "init")
    return apps, data, src


def _run(tmp, *, status="running", old_id="sha256:old", new_id="sha256:new", cur_image="sha256:old", busy="",
         smoke_rc=0, build_rc=0, fw="active", df_pct=42, include=True, drift=False, dirty=None, extra_env=None,
         probe_rc=0):
    apps, data, src = _scaffold(tmp, include=include, drift=drift)
    if dirty:
        with open(os.path.join(src, dirty), "w") as f:
            f.write("planted\n")
    fakebin = os.path.join(tmp, "fakebin")
    _exec(os.path.join(fakebin, "docker"), FAKE_DOCKER)
    df = _exec(os.path.join(tmp, "fake_df"), '#!/usr/bin/env bash\necho "Use%%"\necho " %s%%"\n' % df_pct)
    systemctl = _exec(os.path.join(tmp, "fake_systemctl"),
                      '#!/usr/bin/env bash\necho "systemctl $*" >> "%s/systemctl.log"\n[ "%s" = active ]\n'
                      % (tmp, fw))
    env = dict(os.environ)
    env.update({
        "PATH": fakebin + os.pathsep + env["PATH"], "FAKE_T": tmp, "FAKE_STATUS": status, "FAKE_OLD_ID": old_id,
        "FAKE_NEW_ID": new_id, "FAKE_CUR_IMAGE": cur_image, "FAKE_BUSY": busy, "FAKE_SMOKE_RC": str(smoke_rc),
        "FAKE_BUILD_RC": str(build_rc), "FAKE_PROBE_RC": str(probe_rc), "SIM_AGENT_APPS_DIR": apps,
        "SIM_AGENT_SRC_DIR": src,
        "SIM_AGENT_DATA_DIR": data, "AGENT_UPDATE_DF": df, "AGENT_UPDATE_SYSTEMCTL": systemctl,
        "AGENT_UPDATE_DISK_ROOTS": data,
    })
    env.update(extra_env or {})
    res = subprocess.run(["bash", _UPDATE], env=env, capture_output=True, text=True, timeout=30)
    log_path = os.path.join(data, "update.log")
    calls_path = os.path.join(tmp, "docker_calls.log")
    return (res, open(log_path).read() if os.path.exists(log_path) else "",
            open(calls_path).read() if os.path.exists(calls_path) else "")


# ------------------------------------------------------------------------ disk usage (2026-09-29 review, M3)

def test_logs_disk_usage_percentage_every_cycle():
    with tempfile.TemporaryDirectory() as tmp:
        res, log, _ = _run(tmp, df_pct=42)
        assert res.returncode == 0, res.stdout + res.stderr
        assert "disk usage at" in log and "42%" in log
        assert "WARNING" not in log


def test_warns_loudly_when_over_the_threshold():
    with tempfile.TemporaryDirectory() as tmp:
        res, log, _ = _run(tmp, df_pct=91, extra_env={"AGENT_UPDATE_DISK_WARN_PCT": "85"})
        assert res.returncode == 0, res.stdout + res.stderr
        assert "WARNING" in log and "91%" in log and "SHARED server" in log


def test_does_not_warn_just_below_the_threshold():
    with tempfile.TemporaryDirectory() as tmp:
        res, log, _ = _run(tmp, df_pct=84, extra_env={"AGENT_UPDATE_DISK_WARN_PCT": "85"})
        assert res.returncode == 0 and "WARNING" not in log


def test_disk_check_failure_is_non_fatal_and_build_still_proceeds():
    with tempfile.TemporaryDirectory() as tmp:
        res, log, calls = _run(tmp, extra_env={"AGENT_UPDATE_DF": os.path.join(tmp, "no-such-df")})
        assert res.returncode == 0, res.stdout + res.stderr
        assert "skipped (non-fatal)" in log
        assert "docker compose build --pull sim-agent" in calls


# ------------------------------------------------------------------------------------------- preconditions

def test_refuses_when_the_apps_project_does_not_include_the_stack():
    with tempfile.TemporaryDirectory() as tmp:
        res, log, calls = _run(tmp, include=False)
        assert res.returncode == 2
        assert "does not include sim-agent/compose.yaml" in log
        assert "compose build" not in calls


@pytest.mark.parametrize("planted", ["tools/agent_container/bin/evil.sh", "tools/agent_harness_ctl.sh"])
def test_refuses_to_build_from_a_dirty_source_checkout(planted):
    # an untracked file in bin/ would be COPYed into the image; an edited ctl script would run as the forced command
    with tempfile.TemporaryDirectory() as tmp:
        res, log, calls = _run(tmp, dirty=planted)
        assert res.returncode == 1
        assert "not a clean git checkout" in log
        assert "compose build" not in calls


def test_warns_but_proceeds_when_the_apps_fragment_drifted_from_the_reviewed_copy():
    with tempfile.TemporaryDirectory() as tmp:
        res, log, calls = _run(tmp, drift=True)
        assert res.returncode == 0, res.stdout + res.stderr
        assert "differs from the reviewed copy" in log
        assert "compose build --pull sim-agent" in calls


def test_every_compose_call_runs_from_the_apps_project():
    with tempfile.TemporaryDirectory() as tmp:
        res, _, calls = _run(tmp)
        cwds = {l for l in calls.splitlines() if l.startswith("cwd=")}
        assert cwds == {"cwd=" + os.path.join(tmp, "apps")}, cwds


# -------------------------------------------------------------------------------------- smoke + rollback

def test_a_changed_image_is_smoke_tested_before_the_swap():
    with tempfile.TemporaryDirectory() as tmp:
        res, log, calls = _run(tmp)
        assert res.returncode == 0, res.stdout + res.stderr
        order = [calls.index(c) for c in ("docker image tag sim-agent:local sim-agent:rollback",
                                          "docker compose build --pull sim-agent",
                                          "docker compose run --rm --no-deps -T sim-agent smoke_test.sh",
                                          "docker compose up -d --no-deps --no-build sim-agent")]
        assert order == sorted(order)


def test_a_failed_smoke_test_rolls_the_tag_back_and_never_touches_the_container():
    with tempfile.TemporaryDirectory() as tmp:
        res, log, calls = _run(tmp, smoke_rc=1)
        assert res.returncode == 1
        assert "docker image tag sim-agent:rollback sim-agent:local" in calls
        assert "compose up" not in calls
        assert "SMOKE TEST FAILED" in log


def test_a_failed_build_changes_nothing():
    with tempfile.TemporaryDirectory() as tmp:
        res, log, calls = _run(tmp, build_rc=1)
        assert res.returncode == 1 and "BUILD FAILED" in log
        assert "compose run" not in calls and "compose up" not in calls


def test_an_unchanged_image_already_running_is_left_alone():
    with tempfile.TemporaryDirectory() as tmp:
        res, log, calls = _run(tmp, old_id="sha256:same", new_id="sha256:same", cur_image="sha256:same")
        assert res.returncode == 0
        assert "compose run" not in calls and "compose up" not in calls
        assert "already running the current image" in log


def test_an_unchanged_image_still_catches_up_a_container_left_on_an_older_one():
    # the previous cycle built + smoked but deferred the swap (busy/paused): this cycle swaps without re-smoking
    with tempfile.TemporaryDirectory() as tmp:
        res, log, calls = _run(tmp, old_id="sha256:new", new_id="sha256:new", cur_image="sha256:older")
        assert res.returncode == 0, res.stdout + res.stderr
        assert "compose run" not in calls
        assert "docker compose up -d --no-deps --no-build sim-agent" in calls


# ------------------------------------------------------------------------------- when the swap may happen

def test_running_idle_with_the_firewall_active_is_recreated():
    with tempfile.TemporaryDirectory() as tmp:
        res, log, calls = _run(tmp, status="running", fw="active")
        assert res.returncode == 0, res.stdout + res.stderr
        assert "docker compose up -d --no-deps --no-build sim-agent" in calls
        assert "is-active --quiet sim-agent-fw.service" in open(os.path.join(tmp, "systemctl.log")).read()


def test_never_recreates_while_the_firewall_unit_is_inactive():
    with tempfile.TemporaryDirectory() as tmp:
        res, log, calls = _run(tmp, status="running", fw="inactive")
        assert res.returncode == 1
        assert "compose up" not in calls
        assert "REFUSED to recreate" in log


def test_defers_while_a_claude_process_is_active():
    with tempfile.TemporaryDirectory() as tmp:
        res, log, calls = _run(tmp, status="running", busy="4242")
        assert res.returncode == 0
        assert "compose up" not in calls and "DEFERRED" in log


def test_defers_while_paused_because_a_frozen_session_may_be_mid_task():
    # 2026-09-30: the old script treated `paused` as safe to recreate, which killed a session the gaming yield
    # had only frozen. Paused now defers like a busy session.
    with tempfile.TemporaryDirectory() as tmp:
        res, log, calls = _run(tmp, status="paused")
        assert res.returncode == 0
        assert "compose up" not in calls and "DEFERRED" in log and "paused" in log


@pytest.mark.parametrize("status", ["exited", "created", ""])
def test_never_starts_a_container_the_owner_stopped_or_never_started(status):
    with tempfile.TemporaryDirectory() as tmp:
        res, log, calls = _run(tmp, status=status)
        assert res.returncode == 0, res.stdout + res.stderr
        assert "compose up" not in calls
        assert "not running" in log


def test_the_idle_probe_pins_path_and_runs_an_absolute_pgrep():
    # 2026-09-30 security review (HIGH #1): an exec inherits the container's environment, and the image PATH used to
    # start with the model-writable home, so a planted ~/.local/bin/pgrep decided whether a live session was killed
    with tempfile.TemporaryDirectory() as tmp:
        res, log, calls = _run(tmp, status="running")
        assert res.returncode == 0, res.stdout + res.stderr
        assert ("docker exec -e PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin sim-agent "
                "/usr/bin/pgrep -f /claude$") in calls


# ------------------------------------- isolation probes from a pristine container (2026-09-30 review, HIGH #1)
# The probes used to run inside the smoke container, which mounts the model-writable home. They now run every
# cycle from a throwaway container with no mounts (server-stack/probe_from_pristine.sh), before any swap.

def _probe_line(calls):
    lines = [l for l in calls.splitlines() if l.startswith("docker run ")]
    assert len(lines) == 1, lines
    return lines[0]


def test_a_changed_image_is_probed_from_a_pristine_container_after_the_smoke_and_before_the_swap():
    with tempfile.TemporaryDirectory() as tmp:
        res, log, calls = _run(tmp)
        assert res.returncode == 0, res.stdout + res.stderr
        line = _probe_line(calls)
        assert "--network apps_sim-agent-net" in line and "--read-only" in line and "--cap-drop ALL" in line
        assert " -v " not in line and "--mount" not in line and "--volume" not in line
        assert "sim-agent:local /usr/local/lib/agent-container/bin/net_probe.sh --must-fail" in line
        order = [calls.index(c) for c in ("docker compose run --rm --no-deps -T sim-agent smoke_test.sh",
                                          "docker run ", "docker compose up -d --no-deps --no-build sim-agent")]
        assert order == sorted(order)
        assert "isolation probe passed" in log


def test_a_failed_isolation_probe_rolls_a_changed_image_back_and_never_swaps():
    with tempfile.TemporaryDirectory() as tmp:
        res, log, calls = _run(tmp, probe_rc=1)
        assert res.returncode == 1
        assert "ISOLATION PROBE FAILED" in log
        assert "docker image tag sim-agent:rollback sim-agent:local" in calls
        assert "compose up" not in calls


def test_the_isolation_probe_runs_and_alerts_even_when_the_image_did_not_change():
    # a weekly regression check of the host firewall itself: nothing to roll back, but the unit must fail (ntfy)
    with tempfile.TemporaryDirectory() as tmp:
        res, log, calls = _run(tmp, old_id="sha256:same", new_id="sha256:same", cur_image="sha256:same", probe_rc=1)
        assert res.returncode == 1
        assert "ISOLATION PROBE FAILED" in log
        assert "docker image tag sim-agent:rollback sim-agent:local" not in calls
        assert "compose up" not in calls


def test_probe_targets_cover_this_host_and_never_require_the_workstation():
    text = open(_UPDATE).read()
    must_fail = re.search(r'^PROBE_MUST_FAIL="([^"]*)"', text, re.MULTILINE).group(1).split()
    must_pass = re.search(r'^PROBE_MUST_PASS="([^"]*)"', text, re.MULTILINE).group(1).split()
    for t in ("172.30.30.1:22", "192.168.0.10:22", "192.168.0.10:445", "192.168.0.10:2049", "192.168.0.10:18443"):
        assert t in must_fail, t
    assert "192.168.0.10:2222" in must_pass
    # the workstation may be asleep or gaming during the weekly rebuild: a miss there must never roll an image back
    assert not any(t.startswith("192.168.0.68:") for t in must_pass)
