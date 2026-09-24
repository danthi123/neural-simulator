"""A7 (warm-server-prewarm, plan step S18) -- SEED-7 DEV SMOKE runner for the plan's own
required HEAVY check, driven remotely from the pool (the local box is RAM-starved for this).

This does NOT reimplement the check -- it invokes the actual committed pytest test
(tests/test_brain_prewarm_scratch_session.py::test_prewarm_scratch_session_does_not_perturb_default)
as a subprocess with SIM_RUN_HEAVY_CAPABILITY=1 so it runs to completion instead of skipping, and
records the honest pass/fail verdict + stdout/stderr tails to --out (with the automatic
research/runners/__init__.py provenance sidecar, since this is invoked via `-m research.runners.X`).

Dev seed 7 (never a validation seed 42/43/44/100/101/102) -- the pytest test itself pins
BRAIN_CHAT_SEED=7 via os.environ.setdefault, matching the plan's declared dev-smoke seed. This is
a SMOKE / dev check, not an evaluation artifact -- no prereg is required (verify-go / prereg-before-
evaluation-artifact applies to validation-seed GO/NO-GO claims, not to a single dev-seed byte-
identity smoke run of an already-committed unit test).

WHY A RUNNER WRAPPER INSTEAD OF QUEUEING PYTEST DIRECTLY: tools/pool_queue.sh's `add` gate only
accepts `-m research.runners.<module>` commands (it argparse-validates --help and probes node
availability of that exact module) -- there is no path to queue a bare pytest invocation.

Usage:
    SIM_BACKEND=numpy OMP_NUM_THREADS=1 .venv/bin/python -u -m research.runners._a7_prewarm_scratch_seed7_smoke \\
        --out research/findings/raw/_a7_prewarm_scratch_seed7_smoke.json

Exit code mirrors the underlying pytest run's returncode (0 pass, non-zero fail/error) -- the JSON
report is written FIRST regardless, so a non-zero exit never loses the verdict.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

_TEST_TARGET = (
    "tests/test_brain_prewarm_scratch_session.py::"
    "test_prewarm_scratch_session_does_not_perturb_default"
)
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _tail(text, n=60):
    lines = text.splitlines()
    return "\n".join(lines[-n:])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, help="path to write the JSON smoke report to")
    p.add_argument("--timeout", type=int, default=2400,
                   help="wall-clock timeout in seconds for the pytest subprocess (default 2400)")
    args = p.parse_args(argv)

    env = dict(os.environ)
    env["SIM_RUN_HEAVY_CAPABILITY"] = "1"
    env.setdefault("SIM_BACKEND", "numpy")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("BRAIN_CHAT_SEED", "7")

    cmd = [sys.executable, "-m", "pytest", _TEST_TARGET, "-v", "-s"]
    start = time.time()
    try:
        proc = subprocess.run(cmd, cwd=_ROOT, env=env, capture_output=True, text=True,
                               timeout=args.timeout)
        returncode = proc.returncode
        stdout, stderr = proc.stdout, proc.stderr
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        returncode = None
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        timed_out = True
    duration_s = time.time() - start

    report = {
        "runner": "_a7_prewarm_scratch_seed7_smoke",
        "test_target": _TEST_TARGET,
        "seed": 7,
        "kind": "dev_smoke",
        "prereg": None,
        "timed_out": timed_out,
        "returncode": returncode,
        "passed": (returncode == 0) if not timed_out else False,
        "duration_s": duration_s,
        "timeout_s": args.timeout,
        "stdout_tail": _tail(stdout),
        "stderr_tail": _tail(stderr),
    }

    out_path = args.out
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    print(f"wrote {out_path}: passed={report['passed']} returncode={returncode} "
          f"timed_out={timed_out} duration_s={duration_s:.1f}")

    if timed_out:
        return 124
    return returncode if returncode is not None else 1


if __name__ == "__main__":
    sys.exit(main())
