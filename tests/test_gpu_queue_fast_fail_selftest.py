"""tools/gpu_queue.sh: the GPU daemon's FAST-FAIL line, exercised through the script's own `--selftest` TEST E.

WHY (2026-09-25 fix round, review MEDIUM). The daemon logs a loud `⛔ FAST-FAIL` line when a job dies with rc=127
or rc=2 within GPU_QUEUE_FAST_FAIL_S of its START (the historical `status` job sat in gpu_queue.log as an ordinary
DONE(rc=127) line three times). The review changed that block to `if false` and every pytest still passed: only
the manual `--selftest` (TEST E1/E2) noticed, and nothing ran it. This test runs it.

The selftest is isolated by construction: every daemon it starts gets GPU_QUEUE_DIR=<its own mktemp -d scratch
dir> and a fake nvidia-smi, it never dispatches a real brain, and its EXIT-trap cleanup only kills daemons whose
environment names that scratch dir. TMPDIR is pointed at this test's tmp_path, so the scratch dir lives there too.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GPU_QUEUE = ROOT / "tools" / "gpu_queue.sh"


def test_gpu_queue_selftest_fast_fail_legs_pass(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith("GPU_QUEUE_")}
    env["TMPDIR"] = str(tmp_path)
    res = subprocess.run(["bash", str(GPU_QUEUE), "--selftest"], cwd=ROOT, env=env,
                         capture_output=True, text=True, timeout=300)
    out = res.stdout
    assert f"isolated scratch dir: {tmp_path}/" in out, out[:600]   # the selftest ran inside tmp_path
    # E1: a job that dies rc=127 at once is logged as a loud FAST-FAIL; E2 (the failing direction): an rc=0
    # job is never flagged.
    assert "PASS(E1)" in out, out[-3000:]
    assert "PASS(E2)" in out, out[-3000:]
    assert "SELFTEST: PASS" in out, out[-3000:]
    assert res.returncode == 0
    assert not list(tmp_path.glob("gpu_queue_selftest.*")), "the selftest's EXIT-trap cleanup did not run"
