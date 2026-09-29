"""tools/agent_harness_ctl.sh -- the forced command that is the ONLY way the workstation can reach into the
sim-agent container from the server side (docs/plans/2026-09-29-local-model-transition-PLAN.md section 9).
Hermetic: AGENT_CTL_DOCKER stubs `docker` entirely -- no real container is ever paused/unpaused/exec'd into.
"""
import os
import stat
import subprocess
import tempfile
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CTL = os.path.join(_ROOT, "tools", "agent_harness_ctl.sh")


def _write_exec(path, body):
    with open(path, "w") as f:
        f.write("#!/usr/bin/env bash\n" + body)
    os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return path


def _fake_docker(tmp, status="running", exec_ok=True,
                  df_v_output="sim-agent_sim-agent-repo   1   245.3MB\nsim-agent_sim-agent-home   1   1.2GB\n"):
    """A stand-in for `docker`: logs every invocation; answers `inspect -f {{.State.Status}}` with `status`;
    `exec ... tmux send-keys` and `exec ... cat ...` / `exec ... pgrep ...` succeed iff exec_ok; `pause`/
    `unpause` always succeed and are logged; `system df -v` answers with `df_v_output` (the same column shape
    real `docker system df -v` prints for its "Local Volumes" section: NAME ... SIZE, SIZE last)."""
    log = os.path.join(tmp, "docker_calls.log")
    df_v_file = os.path.join(tmp, "df_v_output.txt")
    with open(df_v_file, "w") as f:
        f.write(df_v_output)
    body = """
echo "docker $*" >> "%s"
case "$1" in
  inspect) echo "%s" ;;
  pause|unpause) exit 0 ;;
  system) cat "%s" ;;
  exec)
    %s
    case "$*" in
      *"cat /home/agent/.active_profile_kind"*) echo "local" ;;
      *"cat /home/agent/.active_profile"*) echo "workstation" ;;
      *"pgrep"*) exit 1 ;;   # no claude process "active" by default
      *"claude --version"*) echo "2.1.280 (Claude Code)" ;;
      *) exit 0 ;;
    esac
    ;;
  *) exit 0 ;;
esac
""" % (log, status, df_v_file, "" if exec_ok else "exit 1")
    return _write_exec(os.path.join(tmp, "fake_docker.sh"), body), log


def _run(cmd, tmp, docker_path, extra_env=None, timeout=10):
    env = dict(os.environ)
    env["AGENT_CTL_DOCKER"] = docker_path
    env["AGENT_CTL_LOG"] = os.path.join(tmp, "ctl.log")
    env["AGENT_CTL_PAUSE_DELAY_S"] = "0"   # no need to wait out a real delay in tests
    env["SSH_ORIGINAL_COMMAND"] = cmd
    if extra_env:
        env.update(extra_env)
    return subprocess.run(["bash", _CTL], env=env, capture_output=True, text=True, timeout=timeout)


def test_unknown_command_is_refused_and_logged():
    with tempfile.TemporaryDirectory() as tmp:
        docker, _ = _fake_docker(tmp)
        res = _run("rm -rf /", tmp, docker)
        assert res.returncode != 0
        assert "refused" in res.stderr.lower()
        assert "REFUSED" in open(os.path.join(tmp, "ctl.log")).read()


def test_pause_schedules_a_docker_pause_and_returns_immediately():
    with tempfile.TemporaryDirectory() as tmp:
        docker, log = _fake_docker(tmp, status="running")
        res = _run("pause", tmp, docker)
        assert res.returncode == 0, res.stdout + res.stderr
        assert "scheduled" in res.stdout
        # give the backgrounded `sleep 0; docker pause` a moment to land
        for _ in range(30):
            if os.path.exists(log) and "docker pause" in open(log).read():
                break
            time.sleep(0.1)
        assert os.path.exists(log) and "docker pause sim-agent" in open(log).read()


def test_unpause_is_a_noop_when_not_paused():
    with tempfile.TemporaryDirectory() as tmp:
        docker, log = _fake_docker(tmp, status="running")
        res = _run("unpause", tmp, docker)
        assert res.returncode == 0, res.stdout + res.stderr
        assert "nothing to do" in res.stdout
        assert "docker unpause" not in open(log).read()


def test_unpause_calls_docker_unpause_when_paused():
    with tempfile.TemporaryDirectory() as tmp:
        docker, log = _fake_docker(tmp, status="paused")
        res = _run("unpause", tmp, docker)
        assert res.returncode == 0, res.stdout + res.stderr
        assert "unpaused" in res.stdout
        assert "docker unpause sim-agent" in open(log).read()


def test_say_sends_tmux_keys_via_docker_exec():
    with tempfile.TemporaryDirectory() as tmp:
        docker, log = _fake_docker(tmp)
        res = _run("say 'hello from the workstation'", tmp, docker)
        assert res.returncode == 0, res.stdout + res.stderr
        calls = open(log).read()
        assert "tmux send-keys" in calls
        assert "hello from the workstation" in calls


def test_say_refuses_an_overlong_message():
    with tempfile.TemporaryDirectory() as tmp:
        docker, _ = _fake_docker(tmp)
        res = _run("say " + "x" * 301, tmp, docker)
        assert res.returncode != 0
        assert "300" in res.stderr


def test_say_refuses_non_printable_content():
    with tempfile.TemporaryDirectory() as tmp:
        docker, _ = _fake_docker(tmp)
        res = _run("say hello\x01world", tmp, docker)
        assert res.returncode != 0


def test_say_refuses_a_carriage_return():
    # 2026-09-29 security review, LOW: \r is classified as [[:space:]] by POSIX (space/tab/newline/vtab/
    # formfeed/CR all count), so the old printable-OR-space check let it straight through -- and sent literally
    # (no -l, at the time) it acts as an Enter keystroke inside the container's tmux pane, letting a crafted
    # `say` message submit a SECOND, attacker-chosen line to whatever shell is running there.
    with tempfile.TemporaryDirectory() as tmp:
        docker, log = _fake_docker(tmp)
        res = _run("say " + "innocent\rrm -rf /", tmp, docker)
        assert res.returncode != 0
        assert "carriage return" in res.stderr.lower() or "newline" in res.stderr.lower()
        assert "rm -rf" not in (open(log).read() if os.path.exists(log) else "")


def test_say_sends_the_text_literally_and_enter_as_a_separate_call():
    # 2026-09-29 security review, LOW: without `-l`, `tmux send-keys` treats a word that matches a key name
    # (e.g. "C-c") as that KEY, not literal text -- a `say 'C-c'` sent an actual Ctrl-C into the model's
    # terminal instead of three harmless characters. Fixed by sending the text with `-l --` (forced literal)
    # and Enter as its OWN, separate, non-literal call (bundling it into the -l call would type the word
    # "Enter" instead of pressing the key).
    with tempfile.TemporaryDirectory() as tmp:
        docker, log = _fake_docker(tmp)
        res = _run("say C-c", tmp, docker)
        assert res.returncode == 0, res.stdout + res.stderr
        calls = open(log).read()
        assert "tmux send-keys -t agent -l -- C-c" in calls
        assert "tmux send-keys -t agent Enter" in calls
        # the literal call must never be the one that also carries the bare "Enter" argument
        literal_lines = [l for l in calls.splitlines() if "-l --" in l]
        assert literal_lines and all("Enter" not in l for l in literal_lines)


def test_say_fails_loudly_when_docker_exec_fails():
    with tempfile.TemporaryDirectory() as tmp:
        docker, _ = _fake_docker(tmp, exec_ok=False)
        res = _run("say hello", tmp, docker)
        assert res.returncode != 0
        assert "failed" in res.stderr.lower()


def test_state_reports_status_profile_and_kind_when_running():
    with tempfile.TemporaryDirectory() as tmp:
        docker, _ = _fake_docker(tmp, status="running")
        res = _run("state", tmp, docker)
        assert res.returncode == 0, res.stdout + res.stderr
        assert "container: running" in res.stdout
        assert "active_profile: workstation" in res.stdout
        assert "profile_kind: local" in res.stdout
        assert "claude_active: no" in res.stdout
        assert "claude_version: 2.1.280" in res.stdout


def test_state_when_absent_does_not_attempt_docker_exec():
    with tempfile.TemporaryDirectory() as tmp:
        docker, log = _fake_docker(tmp, status="absent")
        res = _run("state", tmp, docker)
        assert res.returncode == 0, res.stdout + res.stderr
        assert "container: absent" in res.stdout
        assert "docker exec" not in open(log).read()


def test_state_reports_volume_disk_usage_in_gb():
    # 2026-09-29 security review, M3: "the digest that warns when the volumes exceed a threshold" reads these
    # two lines. 245.3MB -> ~0.25GB, 1.2GB -> ~1.20GB.
    with tempfile.TemporaryDirectory() as tmp:
        docker, _ = _fake_docker(tmp, status="running")
        res = _run("state", tmp, docker)
        assert res.returncode == 0, res.stdout + res.stderr
        assert "repo_volume_gb: 0.25" in res.stdout, res.stdout
        assert "home_volume_gb: 1.20" in res.stdout, res.stdout


def test_state_reports_volume_disk_usage_even_when_the_container_is_absent():
    # the named volumes persist independently of whether the container itself is running.
    with tempfile.TemporaryDirectory() as tmp:
        docker, _ = _fake_docker(tmp, status="absent")
        res = _run("state", tmp, docker)
        assert res.returncode == 0, res.stdout + res.stderr
        assert "repo_volume_gb: 0.25" in res.stdout, res.stdout
        assert "home_volume_gb: 1.20" in res.stdout, res.stdout


def test_state_volume_usage_is_unknown_when_not_found():
    with tempfile.TemporaryDirectory() as tmp:
        docker, _ = _fake_docker(tmp, status="running", df_v_output="unrelated-volume   1   5MB\n")
        res = _run("state", tmp, docker)
        assert res.returncode == 0, res.stdout + res.stderr
        assert "repo_volume_gb: unknown" in res.stdout
        assert "home_volume_gb: unknown" in res.stdout
