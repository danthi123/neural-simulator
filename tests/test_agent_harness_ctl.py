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


def _fake_docker(tmp, status="running", exec_ok=True):
    """A stand-in for `docker`: logs every invocation; answers `inspect -f {{.State.Status}}` with `status`;
    `exec ... tmux send-keys` and `exec ... cat ...` / `exec ... pgrep ...` succeed iff exec_ok; `pause`/
    `unpause` always succeed and are logged."""
    log = os.path.join(tmp, "docker_calls.log")
    body = """
echo "docker $*" >> "%s"
case "$1" in
  inspect) echo "%s" ;;
  pause|unpause) exit 0 ;;
  exec)
    %s
    case "$*" in
      *"cat /home/agent/.active_profile_kind"*) echo "local" ;;
      *"cat /home/agent/.active_profile"*) echo "workstation" ;;
      *"pgrep"*) exit 1 ;;   # no claude process "active" by default
      *"claude --version"*) echo "2.1.280 (Claude Code)" ;;
      *"readlink /home/agent/.local/bin/claude"*) echo "/home/agent/.local/share/claude/versions/2.1.280" ;;
      *) exit 0 ;;
    esac
    ;;
  *) exit 0 ;;
esac
""" % (log, status, "" if exec_ok else "exit 1")
    return _write_exec(os.path.join(tmp, "fake_docker.sh"), body), log


def _fake_zfs(tmp, referenced, refquota, ok=True):
    """`zfs get -Hp -o value referenced,refquota <dataset>` -> one raw byte count per line, like the real one."""
    log = os.path.join(tmp, "zfs_calls.log")
    body = 'echo "zfs $*" >> "%s"\n' % log
    body += ('printf "%%s\\n%%s\\n" "%s" "%s"\n' % (referenced, refquota)) if ok else "exit 1\n"
    return _write_exec(os.path.join(tmp, "fake_zfs_%s_%s.sh" % (referenced, refquota)), body)


def _fake_systemctl(tmp, state):
    return _write_exec(os.path.join(tmp, "fake_systemctl_%s.sh" % state),
                       'echo "%s"\n[ "%s" = active ]\n' % (state, state))


def _run(cmd, tmp, docker_path, extra_env=None, timeout=10):
    env = dict(os.environ)
    env["AGENT_CTL_DOCKER"] = docker_path
    env["AGENT_CTL_LOG"] = os.path.join(tmp, "ctl.log")
    env["AGENT_CTL_PAUSE_DELAY_S"] = "0"   # no need to wait out a real delay in tests
    # NEVER the real $HOME (2026-09-29 re-review, L3 fix): both default to per-test tmp paths so no test can
    # ever read or write the actual workstation's own active-profile record or profile files.
    env.setdefault("AGENT_CTL_ACTIVE_PROFILE_STATE", os.path.join(tmp, ".sim_agent_active_profile"))
    env.setdefault("AGENT_CTL_PROFILE_DIR", os.path.join(tmp, "no-such-profile-dir"))
    # never the real `zfs` / `systemctl`: a 50 GB-of-64 GB dataset and an active firewall unit by default
    env.setdefault("AGENT_CTL_ZFS", _fake_zfs(tmp, "50000000000", "64000000000"))
    env.setdefault("AGENT_CTL_SYSTEMCTL", _fake_systemctl(tmp, "active"))
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


def _write_profile_env(profile_dir, name, kind):
    os.makedirs(profile_dir, exist_ok=True)
    with open(os.path.join(profile_dir, "%s.env" % name), "w") as f:
        f.write("BASE_URL=http://example\nTOKEN_VAR=\nMODEL=x\nCONTEXT=1\nKIND=%s\n" % kind)


def test_state_reports_status_profile_and_kind_when_running():
    with tempfile.TemporaryDirectory() as tmp:
        docker, _ = _fake_docker(tmp, status="running")
        profile_dir = os.path.join(tmp, "profiles")
        _write_profile_env(profile_dir, "workstation", "local")
        env = {"AGENT_CTL_PROFILE_DIR": profile_dir}
        res_set = _run("set-active-profile workstation", tmp, docker, extra_env=env)
        assert res_set.returncode == 0, res_set.stdout + res_set.stderr
        res = _run("state", tmp, docker, extra_env=env)
        assert res.returncode == 0, res.stdout + res.stderr
        assert "container: running" in res.stdout
        assert "active_profile: workstation" in res.stdout
        assert "profile_kind: local" in res.stdout
        assert "claude_active: no" in res.stdout
        assert "claude_version: 2.1.280" in res.stdout


def test_state_does_not_trust_a_container_writable_profile_file():
    """2026-09-29 re-review, L3: `state` must never ask the CONTAINER what profile/kind is active -- a
    possibly-adversarial model could simply write its own .active_profile_kind to fake a cloud backend and
    dodge the owner's gaming-pause. With no set-active-profile ever called, `docker exec` claiming
    "local"/"workstation" (the fake docker's own default answer for those cat calls) must be ignored."""
    with tempfile.TemporaryDirectory() as tmp:
        docker, log = _fake_docker(tmp, status="running")
        res = _run("state", tmp, docker)
        assert res.returncode == 0, res.stdout + res.stderr
        assert "active_profile: none" in res.stdout
        assert "profile_kind: unknown" in res.stdout
        assert "cat /home/agent/.active_profile" not in open(log).read(), (
            "state must never docker-exec into the container to ask for the active profile at all")


def test_set_active_profile_determines_kind_from_the_servers_own_profile_file():
    with tempfile.TemporaryDirectory() as tmp:
        docker, _ = _fake_docker(tmp)
        profile_dir = os.path.join(tmp, "profiles")
        _write_profile_env(profile_dir, "cloud-example", "cloud")
        env = {"AGENT_CTL_PROFILE_DIR": profile_dir}
        res = _run("set-active-profile cloud-example", tmp, docker, extra_env=env)
        assert res.returncode == 0, res.stdout + res.stderr
        assert "cloud" in res.stdout
        state_path = os.path.join(tmp, ".sim_agent_active_profile")
        content = open(state_path).read()
        assert "PROFILE=cloud-example" in content
        assert "KIND=cloud" in content


def test_set_active_profile_refuses_an_unsafe_name():
    with tempfile.TemporaryDirectory() as tmp:
        docker, _ = _fake_docker(tmp)
        res = _run("set-active-profile ../../etc/passwd", tmp, docker)
        assert res.returncode != 0


def test_set_active_profile_refuses_a_profile_that_does_not_exist():
    with tempfile.TemporaryDirectory() as tmp:
        docker, _ = _fake_docker(tmp)
        res = _run("set-active-profile no-such-profile", tmp, docker)
        assert res.returncode != 0
        assert "no such profile" in res.stderr.lower()


def test_state_strips_control_characters_from_claude_version():
    """2026-09-29 re-review, L3: even the one remaining container-sourced field (claude_version) must never let
    an ANSI escape / control byte reach whoever reads `state`'s output on a terminal."""
    with tempfile.TemporaryDirectory() as tmp:
        docker, _ = _fake_docker(tmp)
        # override the version answer with an embedded ESC sequence + a literal control byte
        body = open(docker).read().replace(
            '*"readlink /home/agent/.local/bin/claude"*) echo "/home/agent/.local/share/claude/versions/2.1.280" ;;',
            '*"readlink /home/agent/.local/bin/claude"*) '
            'printf "/home/agent/.local/share/claude/versions/2.1.280\\x1b[31m EVIL\\x07\\n" ;;')
        assert body != open(docker).read()
        with open(docker, "w") as f:
            f.write(body)
        res = _run("state", tmp, docker)
        assert res.returncode == 0, res.stdout + res.stderr
        assert "\x1b" not in res.stdout
        assert "\x07" not in res.stdout
        assert "EVIL" in res.stdout   # the ordinary printable text around it must survive


def test_state_when_absent_does_not_attempt_docker_exec():
    with tempfile.TemporaryDirectory() as tmp:
        docker, log = _fake_docker(tmp, status="absent")
        res = _run("state", tmp, docker)
        assert res.returncode == 0, res.stdout + res.stderr
        assert "container: absent" in res.stdout
        assert "docker exec" not in open(log).read()


def test_state_reports_the_datasets_usage_against_its_refquota():
    # 2026-09-29 security review M3, rebased onto the ZFS dataset 2026-09-30: tools/inspection_digest.py reads
    # these lines. 50e9 bytes -> 50.00 GB, 64e9 -> 64.00 GB; the query names the sim-agent dataset explicitly.
    with tempfile.TemporaryDirectory() as tmp:
        docker, _ = _fake_docker(tmp, status="running")
        res = _run("state", tmp, docker)
        assert res.returncode == 0, res.stdout + res.stderr
        assert "appdata_gb: 50.00" in res.stdout, res.stdout
        assert "appdata_quota_gb: 64.00" in res.stdout, res.stdout
        calls = open(os.path.join(tmp, "zfs_calls.log")).read()
        assert "get -Hp -o value referenced,refquota data/appdata/sim-agent" in calls


def test_state_reports_appdata_even_when_the_container_is_absent():
    # the dataset persists independently of whether the container itself exists
    with tempfile.TemporaryDirectory() as tmp:
        docker, _ = _fake_docker(tmp, status="absent")
        res = _run("state", tmp, docker)
        assert res.returncode == 0, res.stdout + res.stderr
        assert "appdata_gb: 50.00" in res.stdout, res.stdout


def test_state_reports_no_refquota_as_none():
    with tempfile.TemporaryDirectory() as tmp:
        docker, _ = _fake_docker(tmp, status="running")
        res = _run("state", tmp, docker, extra_env={"AGENT_CTL_ZFS": _fake_zfs(tmp, "1000000000", "0")})
        assert "appdata_gb: 1.00" in res.stdout, res.stdout
        assert "appdata_quota_gb: none" in res.stdout, res.stdout


def test_state_appdata_is_unknown_when_the_dataset_cannot_be_read():
    with tempfile.TemporaryDirectory() as tmp:
        docker, _ = _fake_docker(tmp, status="running")
        res = _run("state", tmp, docker, extra_env={"AGENT_CTL_ZFS": _fake_zfs(tmp, "x", "y", ok=False)})
        assert res.returncode == 0, res.stdout + res.stderr
        assert "appdata_gb: unknown" in res.stdout
        assert "appdata_quota_gb: unknown" in res.stdout


def test_state_reports_the_firewall_unit_state():
    with tempfile.TemporaryDirectory() as tmp:
        docker, _ = _fake_docker(tmp, status="running")
        res = _run("state", tmp, docker)
        assert "firewall: active" in res.stdout, res.stdout
        res = _run("state", tmp, docker, extra_env={"AGENT_CTL_SYSTEMCTL": _fake_systemctl(tmp, "failed")})
        assert "firewall: failed" in res.stdout, res.stdout


# ------------------------------------------ docker exec hygiene (2026-09-30 security review, HIGH #1)
SAFE_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"


def test_every_docker_exec_pins_path_and_names_an_absolute_binary():
    # an exec inherits the container's environment, and the image PATH used to start with the model-writable home
    with tempfile.TemporaryDirectory() as tmp:
        docker, log = _fake_docker(tmp, status="running")
        assert _run("state", tmp, docker).returncode == 0
        assert _run("say 'hello'", tmp, docker).returncode == 0
        execs = [l for l in open(log).read().splitlines() if l.startswith("docker exec")]
        assert len(execs) >= 4, execs
        for l in execs:
            assert l.startswith("docker exec -e PATH=%s sim-agent /" % SAFE_PATH), l


def test_state_reads_the_claude_version_from_its_symlink_and_never_runs_the_binary():
    # the binary lives in the model-writable home: running it from the host-side ctl would run planted code
    with tempfile.TemporaryDirectory() as tmp:
        docker, log = _fake_docker(tmp, status="running")
        res = _run("state", tmp, docker)
        assert res.returncode == 0, res.stdout + res.stderr
        assert "claude_version: 2.1.280\n" in res.stdout
        assert "claude --version" not in open(log).read()


# ------------------------------------------ gitea dataset usage (2026-09-30 security review, MEDIUM storage)
def _fake_zfs_by_dataset(tmp, simagent=("50000000000", "64000000000"), gitea=("2870000000", "0")):
    log = os.path.join(tmp, "zfs_calls.log")
    body = 'echo "zfs $*" >> "%s"\ncase "$*" in\n' % log
    body += '  *data/appdata/gitea*) printf "%%s\\n%%s\\n" "%s" "%s" ;;\n' % gitea
    body += '  *) printf "%%s\\n%%s\\n" "%s" "%s" ;;\nesac\n' % simagent
    return _write_exec(os.path.join(tmp, "fake_zfs_by_dataset.sh"), body)


def test_state_reports_the_gitea_datasets_usage_against_its_quota():
    # the write deploy key can push unlimited research/draft-* data; data/appdata/gitea shares the `data` pool with
    # every other stack, so its usage and quota are reported beside the sim-agent dataset
    with tempfile.TemporaryDirectory() as tmp:
        docker, _ = _fake_docker(tmp, status="running")
        zfs = _fake_zfs_by_dataset(tmp, gitea=("2870000000", "32212254720"))
        res = _run("state", tmp, docker, extra_env={"AGENT_CTL_ZFS": zfs})
        assert res.returncode == 0, res.stdout + res.stderr
        assert "appdata_gb: 50.00" in res.stdout
        assert "gitea_gb: 2.87" in res.stdout, res.stdout
        assert "gitea_quota_gb: 32.21" in res.stdout, res.stdout
        assert "get -Hp -o value used,quota data/appdata/gitea" in open(os.path.join(tmp, "zfs_calls.log")).read()


def test_state_reports_a_gitea_dataset_without_a_quota_as_none():
    with tempfile.TemporaryDirectory() as tmp:
        docker, _ = _fake_docker(tmp, status="absent")
        res = _run("state", tmp, docker, extra_env={"AGENT_CTL_ZFS": _fake_zfs_by_dataset(tmp)})
        assert "gitea_quota_gb: none" in res.stdout, res.stdout
