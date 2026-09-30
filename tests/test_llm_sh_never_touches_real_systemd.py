"""2026-09-29 INCIDENT: tests/test_llm_gpu_autoswap.py's two "does not refuse forever" tests correctly stub
LOCAL_LLM_SYSTEMCTL (so llm.sh's is-active/busy CHECKS are safe), but ran to completion with NO local PATH
override -- so `cmd_on`'s actual process launcher, a bare `systemd-run` PATH lookup, reached the REAL binary.
It started a REAL "local-llm" transient unit running llama-swap (from a dev worktree's config), and llm.sh's own
activate_profile() then PUT an unrelated profile onto it and force-loaded ~21GB onto the shared GPU -- colliding
with the owner's real, concurrent use of the machine. tests/conftest.py now globally stubs `systemd-run` (mirror
of its existing LOCAL_LLM_SYSTEMCTL stub) so no test, present or future, can repeat this by forgetting a local
PATH override. This file pins that guarantee directly, not just by unit-testing the stub in isolation.

SAFETY OF THIS FILE ITSELF: every real-invocation test below sets an isolated GPU_QUEUE_DIR (never the live
research/queue/) and an isolated LLM_PORT (never 8080), so even a REGRESSION here degrades to "cmd_on reaches
some inert PATH-stubbed systemd-run, or in the worst case an isolated, off-8080 wait_up() failure" rather than
touching anything real -- the assertions below are what would actually catch that regression, not what
prevents its blast radius; belt AND suspenders.
"""
import os
import shutil
import subprocess
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LLM_SH = os.path.join(ROOT, "tools", "local_llm", "llm.sh")


def test_systemd_run_resolves_to_the_session_stub_not_a_real_binary():
    resolved = shutil.which("systemd-run")
    assert resolved is not None, "no systemd-run on PATH at all -- cannot confirm the session stub is in front"
    assert os.path.join("tests", "_stub_bin", "systemd-run") in resolved or resolved.endswith(
        os.path.join("_stub_bin", "systemd-run")), (
        "systemd-run resolves to %r, not tests/_stub_bin/systemd-run -- the global stub "
        "(tests/conftest.py) is not in front of PATH for this process" % resolved)


def test_the_stub_systemd_run_is_a_true_noop():
    """The stub itself must never actually run its argument -- it launches nothing, ever."""
    sentinel = None
    with tempfile.TemporaryDirectory() as tmp:
        sentinel = os.path.join(tmp, "should_never_exist")
        stub = shutil.which("systemd-run")
        res = subprocess.run([stub, "--unit=x", "bash", "-c", "touch %s" % sentinel], capture_output=True,
                             text=True, timeout=5)
        assert res.returncode == 0
        assert not os.path.exists(sentinel), "the systemd-run stub actually executed its payload command"


def _real_unit_is_active():
    """Ground truth: the REAL `systemctl --user`, never the LOCAL_LLM_SYSTEMCTL stub -- this is the ONE check
    in this whole test suite that is deliberately allowed to touch the real systemd user session, because it is
    the only way to prove a test did NOT leak into it."""
    return subprocess.run(["systemctl", "--user", "is-active", "--quiet", "local-llm"]).returncode == 0


def _cleanup_if_leaked(profile_hint):
    """Best-effort safety net: if a regression somehow still got past the global stub and left a REAL unit
    running, stop it and say so loudly -- a failed assertion must not also leave a live leak behind."""
    if _real_unit_is_active():
        subprocess.run(["systemctl", "--user", "stop", "local-llm"])
        subprocess.run(["systemctl", "--user", "reset-failed", "local-llm"])
        return True
    return False


def test_llm_on_with_no_local_stub_beyond_the_global_one_never_starts_a_real_unit():
    """The exact shape of the incident: LOCAL_LLM_SYSTEMCTL says inactive/not-busy (correctly), GPU_QUEUE_DIR is
    a fresh isolated dir (so gpu_job_busy() is false, same as the real incident's "stale dispatcher" case), and
    NOTHING local stubs systemd-run -- only the global tests/conftest.py protection may save this."""
    leaked_before = _real_unit_is_active()
    with tempfile.TemporaryDirectory() as tmp:
        env = dict(os.environ)   # inherits this process's PATH, which conftest.py has already prepended
        env["GPU_QUEUE_DIR"] = tmp
        env["LLM_PORT"] = "18098"   # belt-and-suspenders: even a leaked real unit would not land on the real port
        try:
            subprocess.run(["bash", LLM_SH, "on", "qwen38-27b-iq4nl-mtp"], env=env, capture_output=True,
                           text=True, timeout=15)
        except subprocess.TimeoutExpired:
            pass
        leaked = _cleanup_if_leaked("qwen38-27b-iq4nl-mtp")
        assert not leaked, (
            "llm.sh reached the REAL systemd-run and started a REAL local-llm unit -- the global stub in "
            "tests/conftest.py did not take effect. (a leaked unit found by this test was just stopped)")
    assert leaked_before == _real_unit_is_active(), "this test must leave the real unit's state exactly as found"
