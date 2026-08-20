"""GPU-present smoke of the DEFAULT production chat — the class guard for cupy-only crashes.

WHY (2026-08-19). The default `/api/brain-chat` turn (tiny-demo + the spiking Qwen mouth)
was 400-crashing on the cupy backend for every request (SciPy `.tocoo()` on a cupy-hybrid
CSR, sim/bridge.py), and it was INVISIBLE because every automated check for the chat ran
on numpy, where the bug does not occur. The specific bug now has a CPU-runnable regression
gate (test_set_pathway_weights_backend_safe.py); this test guards the CLASS — a default-on
faculty exercised only on numpy can ship a cupy-only crash — by actually building the
default brain on the GPU and asserting the chat answers (HTTP 200), not errors.

It SKIPS cleanly when no CUDA GPU + cupy are available (CPU CI), so it is a no-op there and
a real guard on any GPU machine. The backend is selected at webapp.server import time from
SIM_BACKEND, so the check runs in a fresh SIM_BACKEND=cupy subprocess rather than mutating
this (numpy-default) process.
"""
import os
import subprocess
import sys

import pytest


def _gpu_available() -> bool:
    try:
        import cupy  # noqa: F401
        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


_SUBPROCESS = r"""
import json
from webapp.server import brain_chat, BrainChatRequest
r = brain_chat(BrainChatRequest(session="gpu_smoke", message="what does the dog chase?"))
code = int(getattr(r, "status_code", 0))
body = json.loads(bytes(r.body))
answer = (body.get("answer") or body.get("response") or "")
assert code == 200, "default chat returned status %s (expected 200): %s" % (code, str(body)[:300])
assert answer.strip(), "default chat returned an empty answer on the GPU path"
print("GPU_SMOKE_OK code=%d answer=%r" % (code, answer[:80]))
"""


@pytest.mark.skipif(not _gpu_available(), reason="no CUDA GPU / cupy — CPU CI, nothing to smoke")
def test_default_production_chat_answers_on_gpu():
    """The default brain_chat must return 200 (not a 400 crash) on the cupy backend."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, SIM_BACKEND="cupy", PYTHONPATH=root)
    proc = subprocess.run(
        [sys.executable, "-u", "-c", _SUBPROCESS],
        cwd=root, env=env, capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0 and "GPU_SMOKE_OK" in proc.stdout, (
        "default production chat did not answer 200 on the cupy backend.\n"
        "stdout tail:\n%s\nstderr tail:\n%s"
        % ("\n".join(proc.stdout.splitlines()[-8:]), "\n".join(proc.stderr.splitlines()[-15:]))
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
