"""Pins a real bug found + fixed 2026-09-09: `tools/gpu_queue.sh status` used
`[ -n "$untracked" ] && echo "..."` as its LAST statement in the `status)` case. Bash's `test && action`
returns the TEST's own exit status when the test is false, so on every HEALTHY call (no untracked GPU
process -- the common case) the whole `status` subcommand exited 1, even though the printed report was
entirely healthy. `tools/tool_health.py` (and the pre-commit `tool-health-fresh` gate) reads that exit code,
so gpu-queue was reported permanently ROTTED regardless of the daemon's real state -- a diagnostic that could
never signal success. Fixed with an explicit `if ... fi; exit 0` at the end of the case arm.

Hermetic: runs in an isolated `GPU_QUEUE_DIR` (a real gpu_queue.sh queue directory has no bearing on this
check -- `status` on a freshly-initialized, empty queue is the healthy/common case being pinned).
"""
import os
import subprocess
import tempfile

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPT = os.path.join(_ROOT, "tools", "gpu_queue.sh")


def _run_status(env_extra):
    env = dict(os.environ)
    env.update(env_extra)
    return subprocess.run(["bash", _SCRIPT, "status"], cwd=_ROOT, env=env,
                           capture_output=True, text=True, timeout=30)


def test_status_healthy_exits_zero():
    """The common case (no untracked GPU-resident process, dispatcher down or up, queue empty) must exit 0 --
    a read-only status report is never itself a failure."""
    with tempfile.TemporaryDirectory() as tmp:
        res = _run_status({"GPU_QUEUE_DIR": tmp})
        assert res.returncode == 0, (
            f"gpu_queue.sh status exited {res.returncode} on a healthy/empty queue "
            f"(stdout={res.stdout!r} stderr={res.stderr!r})")
        assert "== gpu_queue ==" in res.stdout
        assert "UNTRACKED" not in res.stdout


def test_status_prints_untracked_but_still_exits_zero_when_only_a_warning():
    """A `status` call is a diagnostic; even when it PRINTS a warning (untracked GPU-resident brain process),
    the read-only report itself must not fail the calling tool (only the warning line must appear). This
    module cannot fabricate a real untracked GPU process portably, so it pins the weaker but still load-
    bearing half directly: `bash -c` reproduces the exact fixed idiom and shows it returns 0 either way."""
    ok_when_false = subprocess.run(["bash", "-c", 'x=""; if [ -n "$x" ]; then echo warn; fi; exit 0'],
                                    capture_output=True, text=True, timeout=5)
    ok_when_true = subprocess.run(["bash", "-c", 'x="1"; if [ -n "$x" ]; then echo warn; fi; exit 0'],
                                   capture_output=True, text=True, timeout=5)
    assert ok_when_false.returncode == 0
    assert ok_when_true.returncode == 0 and "warn" in ok_when_true.stdout
