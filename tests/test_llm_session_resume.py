"""`llm claude` session bookkeeping (tools/local_llm/llm.sh, 2026-09-25). Claude Code's own --continue resumes the most
recent session in the project folder -- usually the owner's Anthropic desktop session, far larger than the local model's
window. llm therefore records the id of every session it starts and rewrites --continue / -c to --resume <that id>."""
import os
import subprocess
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LLM = os.path.join(ROOT, "tools", "local_llm", "llm.sh")


def _args(state, *a):
    env = dict(os.environ, XDG_STATE_HOME=state)
    out = subprocess.run(["bash", LLM, "__session_args", *a], env=env, capture_output=True, text=True, timeout=10)
    assert out.returncode == 0, out.stderr
    return out.stdout.split(), out.stderr


def _last(state):
    return open(os.path.join(state, "sim-local-llm", "last_session")).read().strip()


def test_new_session_gets_a_recorded_id():
    with tempfile.TemporaryDirectory() as st:
        args, _ = _args(st, "-p", "hi")
        assert args[0] == "--session-id" and args[2:] == ["-p", "hi"]
        assert _last(st) == args[1]


def test_continue_resumes_the_last_local_session_not_claude_codes_most_recent():
    with tempfile.TemporaryDirectory() as st:
        first, _ = _args(st, "-p", "hi")
        args, _ = _args(st, "--continue", "-p", "again")
        assert args[:2] == ["--resume", first[1]]
        assert "--continue" not in args and "-c" not in args, "must never fall through to Claude Code's own --continue"


def test_continue_with_no_record_starts_a_new_recorded_session():
    with tempfile.TemporaryDirectory() as st:
        args, err = _args(st, "-c")
        assert args[0] == "--session-id" and "no previous local session" in err
        assert "-c" not in args


def test_explicit_resume_passes_through_and_becomes_the_last_session():
    with tempfile.TemporaryDirectory() as st:
        args, _ = _args(st, "--resume", "abc-123")
        assert args == ["--resume", "abc-123"]
        assert _last(st) == "abc-123"
