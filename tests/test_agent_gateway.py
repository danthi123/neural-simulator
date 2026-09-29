"""tools/agent_gateway.sh -- the forced command that is the ENTIRE trust boundary between the sim-agent
container and the workstation's compute (docs/plans/2026-09-29-local-model-transition-PLAN.md section 9).
Hermetic throughout: AGENT_GATEWAY_ROOT points every test at a scratch directory holding stub tools/*.sh (never
the real pool_queue.sh/gpu_queue.sh/game.sh/llm.sh/agent_gateway_submit.py), and AGENT_GATEWAY_SSH stubs `ssh`
entirely -- no real ssh connection, no real pool/GPU job, ever touches this repo's live queues.

2026-09-29 security review rewrite: H1 removed `eval` from argument splitting entirely (tools/
agent_gateway_parse.py, a real python3 script -- NOT stubbed here, since it is itself hermetic/side-effect-free);
H2 replaced free-form `pool-submit '<cmd>' --checked '<reason>'` / `gpu-submit '<cmd>'` with STRUCTURED requests
the gateway forwards, verbatim as argv (never re-joined into a shell string), to tools/agent_gateway_submit.py
(stubbed here -- its own real validation/sandboxing logic is tested in tests/test_agent_gateway_submit.py); H3
removed `results-sync` from this script entirely (moved to a separate rrsync-only ssh key).
"""
import os
import stat
import subprocess
import tempfile

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_GATEWAY = os.path.join(_ROOT, "tools", "agent_gateway.sh")


def _write_exec(path, body):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("#!/usr/bin/env bash\n" + body)
    os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return path


def _scaffold(tmp):
    """A minimal $ROOT/tools/... tree the gateway's hard-coded relative paths expect, each stub just logging
    its own invocation (one argument per log line, `argv:`-prefixed, so a test can tell "one literal token" from
    "several words a shell would have split")."""
    log = os.path.join(tmp, "calls.log")
    _write_exec(os.path.join(tmp, "tools", "pool_queue.sh"), 'echo "pool_queue.sh $*" >> "%s"\n' % log)
    _write_exec(os.path.join(tmp, "tools", "gpu_queue.sh"), 'echo "gpu_queue.sh $*" >> "%s"\n' % log)
    _write_exec(os.path.join(tmp, "tools", "game.sh"), 'echo "game.sh $*" >> "%s"\n' % log)
    _write_exec(os.path.join(tmp, "tools", "local_llm", "llm.sh"), 'echo "llm.sh $*" >> "%s"\n' % log)
    # a STUB agent_gateway_submit.py: NOT the real validator (that lives in tests/test_agent_gateway_submit.py)
    # -- this only proves the gateway forwards the request as a clean argv list to whatever python3 runs it
    # with. It is invoked as `python3 <this file> <argv...>`, so it must be valid PYTHON, not a shell script.
    with open(os.path.join(tmp, "tools", "agent_gateway_submit.py"), "w") as f:
        f.write(
            "import sys\n"
            "with open(%r, 'a') as f:\n"
            "    f.write('submit.py:' + ''.join(' argv[%%s]' %% a for a in sys.argv[1:]) + chr(10))\n"
            % log
        )
    os.makedirs(os.path.join(tmp, "research", "findings", "raw"), exist_ok=True)
    os.makedirs(os.path.join(tmp, "research", "queue"), exist_ok=True)
    return log


def _fake_ssh(tmp, state_output="container: running\nactive_profile: workstation\nprofile_kind: local\nclaude_active: no\n", reachable=True):
    """Stands in for `ssh <host> <cmd...>`: logs the call, answers `state` with the given text, and for every
    other subcommand exits 0 (or nonzero, if reachable=False, simulating an unreachable ctl host)."""
    log = os.path.join(tmp, "ssh_calls.log")
    if not reachable:
        # a genuinely unreachable host: no stdout at all, just a nonzero exit (ssh's own ConnectTimeout
        # behaviour) -- still logs that the attempt was made, for the test's own assertions.
        body = 'echo "ssh $*" >> "%s"\nexit 1\n' % log
        return _write_exec(os.path.join(tmp, "fake_ssh.sh"), body), log
    body = """
echo "ssh $*" >> "%s"
args=("$@")
last="${args[-1]:-}"
case "$last" in
  state) cat <<'EOF'
%s
EOF
       exit 0 ;;
  *) exit 0 ;;
esac
""" % (log, state_output.rstrip("\n"))
    return _write_exec(os.path.join(tmp, "fake_ssh.sh"), body), log


def _run(cmd, tmp, ssh_path=None, extra_env=None):
    env = dict(os.environ)
    env["AGENT_GATEWAY_ROOT"] = tmp
    env["AGENT_GATEWAY_LOG"] = os.path.join(tmp, "gateway.log")
    env["SSH_ORIGINAL_COMMAND"] = cmd
    if ssh_path:
        env["AGENT_GATEWAY_SSH"] = ssh_path
    if extra_env:
        env.update(extra_env)
    return subprocess.run(["bash", _GATEWAY], env=env, capture_output=True, text=True, timeout=10)


def test_empty_command_is_refused():
    with tempfile.TemporaryDirectory() as tmp:
        _scaffold(tmp)
        res = _run("", tmp)
        assert res.returncode != 0
        assert "refused" in res.stderr.lower()


def test_unknown_subcommand_is_refused_and_logged():
    with tempfile.TemporaryDirectory() as tmp:
        _scaffold(tmp)
        res = _run("rm -rf /", tmp)
        assert res.returncode != 0
        assert "refused" in res.stderr.lower()
        log = open(os.path.join(tmp, "gateway.log")).read()
        assert "REFUSED" in log
        assert "rm -rf /" in log


def test_pool_submit_forwards_the_structured_request_as_argv():
    with tempfile.TemporaryDirectory() as tmp:
        log = _scaffold(tmp)
        res = _run("pool-submit --sha " + "a" * 40 + " --runner research.runners.foo "
                   "--checked 'corpus: nothing' -- --seed 7", tmp)
        assert res.returncode == 0, res.stdout + res.stderr
        calls = open(log).read()
        assert "submit.py: argv[pool] argv[--sha] argv[" + "a" * 40 + "]" in calls
        assert "argv[--runner] argv[research.runners.foo]" in calls
        assert "argv[--checked] argv[corpus: nothing]" in calls
        assert "argv[--] argv[--seed] argv[7]" in calls


def test_gpu_submit_forwards_the_structured_request_as_argv():
    with tempfile.TemporaryDirectory() as tmp:
        log = _scaffold(tmp)
        res = _run("gpu-submit --sha " + "b" * 40 + " --runner research.runners.bar -- --seed 7", tmp)
        assert res.returncode == 0, res.stdout + res.stderr
        calls = open(log).read()
        assert "submit.py: argv[gpu] argv[--sha] argv[" + "b" * 40 + "]" in calls
        assert "argv[--runner] argv[research.runners.bar]" in calls


def test_gpu_submit_with_no_request_is_refused():
    with tempfile.TemporaryDirectory() as tmp:
        _scaffold(tmp)
        res = _run("gpu-submit", tmp)
        assert res.returncode != 0
        assert "structured request" in res.stderr


def test_pool_submit_command_substitution_is_never_executed_and_arrives_as_one_literal_token():
    """H1 regression: the exact shape the security review used (`pool-submit x $(cmd) --checked y`) must never
    run `cmd` -- it must arrive at the (stubbed) submit script as ONE inert, literal argv token, exactly as
    `eval` never touching this text would produce."""
    with tempfile.TemporaryDirectory() as tmp:
        log = _scaffold(tmp)
        marker = os.path.join(tmp, "pwned")
        # $(id>marker) has NO internal whitespace, so it is exactly ONE shlex token either way -- isolating the
        # question this test asks (did the substitution get EXECUTED) from ordinary word-splitting.
        res = _run("pool-submit --sha " + "c" * 40 +
                   " --runner research.runners.foo --checked reason -- --out $(id>%s)" % marker, tmp)
        assert res.returncode == 0, res.stdout + res.stderr
        assert not os.path.exists(marker), "a command substitution in the request must NEVER execute"
        calls = open(log).read()
        assert ("argv[$(id>%s)]" % marker) in calls, (
            "the literal, unexpanded text must still reach the (stub) submit script as one token: %s" % calls)


def test_a_control_character_in_an_argument_is_refused_before_reaching_the_submit_script():
    # \x01 survives BOTH bash's own `read -r SUB REST <<< "$CMD"` line-extraction (only \n ends that read; \x01
    # is not an IFS character either) AND shlex's own whitespace splitting (shlex treats " \t\r\n" as
    # inter-token whitespace, so a bare \r would just become a token boundary, never an embedded byte -- \x01
    # is not, and genuinely reaches agent_gateway_parse.py's per-token check embedded WITHIN one token).
    with tempfile.TemporaryDirectory() as tmp:
        log = _scaffold(tmp)
        res = _run("pool-submit --sha " + "d" * 40 +
                   " --runner research.runners.foo --checked reason -- --seed=7\x018", tmp)
        assert res.returncode != 0
        assert "control character" in res.stderr or "could not parse" in res.stderr
        assert not os.path.exists(os.path.join(tmp, "calls.log")) or "submit.py" not in open(log).read()


def test_results_sync_has_moved_off_this_gateway():
    with tempfile.TemporaryDirectory() as tmp:
        _scaffold(tmp)
        res = _run("results-sync --server --sender -logDtprze.iLsfxCIvu . research/findings/raw", tmp)
        assert res.returncode != 0
        assert "rrsync" in res.stderr.lower()


def test_status_runs_all_three_status_checks():
    with tempfile.TemporaryDirectory() as tmp:
        log = _scaffold(tmp)
        res = _run("status", tmp)
        assert res.returncode == 0, res.stdout + res.stderr
        calls = open(log).read()
        assert "gpu_queue.sh status" in calls
        assert "pool_queue.sh list" in calls
        assert "game.sh status" in calls


def test_yield_skips_pause_for_a_cloud_profile():
    with tempfile.TemporaryDirectory() as tmp:
        log = _scaffold(tmp)
        ssh_path, ssh_log = _fake_ssh(tmp, state_output="container: running\nprofile_kind: cloud\n")
        res = _run("yield --max-hours 4", tmp, ssh_path=ssh_path)
        assert res.returncode == 0, res.stdout + res.stderr
        assert "no gpu yield needed" in res.stdout.lower()
        # llm.sh yield must never have been called for a cloud backend
        assert "llm.sh yield" not in (open(log).read() if os.path.exists(log) else "")
        ssh_calls = open(ssh_log).read()
        assert "pause" not in ssh_calls   # only `state` was queried, no pause attempt


def test_yield_pauses_the_harness_then_calls_llm_yield_with_notify_cmd():
    with tempfile.TemporaryDirectory() as tmp:
        log = _scaffold(tmp)
        ssh_path, ssh_log = _fake_ssh(tmp, state_output="container: running\nprofile_kind: local\n")
        res = _run("yield --max-hours 6", tmp, ssh_path=ssh_path)
        assert res.returncode == 0, res.stdout + res.stderr
        ssh_calls = open(ssh_log).read()
        assert " pause" in ssh_calls or ssh_calls.strip().endswith("pause")
        calls = open(log).read()
        assert "llm.sh yield --max-hours 6 --notify-cmd" in calls
        assert "unpause" in calls and " say " in calls   # the notify-cmd itself names both steps


def test_yield_continues_gracefully_when_the_ctl_host_is_unreachable():
    with tempfile.TemporaryDirectory() as tmp:
        log = _scaffold(tmp)
        ssh_path, _ = _fake_ssh(tmp, reachable=False)
        res = _run("yield", tmp, ssh_path=ssh_path)
        # unreachable ctl -> harness_kind can't be determined -> NOT cloud -> still attempts llm.sh yield
        assert res.returncode == 0, res.stdout + res.stderr
        assert "llm.sh yield" in open(log).read()


def test_yield_max_hours_command_substitution_is_never_executed():
    """H1 regression on the `yield` path too -- its own --max-hours parsing used the same `eval "set -- $REST"`
    before this fix."""
    with tempfile.TemporaryDirectory() as tmp:
        log = _scaffold(tmp)
        marker = os.path.join(tmp, "pwned_yield")
        ssh_path, _ = _fake_ssh(tmp, state_output="container: running\nprofile_kind: local\n")
        res = _run("yield --max-hours $(touch %s)" % marker, tmp, ssh_path=ssh_path)
        assert not os.path.exists(marker), "a command substitution in --max-hours must NEVER execute"


def test_log_strips_control_characters_from_the_attacker_controlled_command():
    """2026-09-29 re-review, L3: $CMD is the container's own SSH_ORIGINAL_COMMAND, logged verbatim on every
    request -- an embedded ANSI escape sequence must never reach the log file (a terminal later tailing/catting
    it could have its title rewritten, text hidden, or worse on a vulnerable emulator)."""
    with tempfile.TemporaryDirectory() as tmp:
        _scaffold(tmp)
        res = _run("bogus-command\x1b[31m EVIL\x07", tmp)
        assert res.returncode != 0
        log_text = open(os.path.join(tmp, "gateway.log")).read()
        assert "\x1b" not in log_text
        assert "\x07" not in log_text
        assert "EVIL" in log_text   # the ordinary printable text around it must survive
