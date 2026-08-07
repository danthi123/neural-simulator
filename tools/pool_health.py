#!/usr/bin/env python3
"""Correct health check for the mini-PC pool (pool40/41/42). Tests what actually matters.

WHY (2026-08-06): the pool is deployed by RSYNC, not `git clone` — `tools/pool_provision.sh` archives HEAD and
rsyncs it into `~/derisk-pool/sim` with a venv (numpy+scipy). There is NO `.git` there BY DESIGN. Two sub-agents
independently ran `git rev-parse HEAD` inside that directory, got `fatal: not a git repository`, and concluded
"pool checkouts are BROKEN — re-provision needed". That false diagnosis was written onto the board and repeated
for a whole session, and it suppressed the very frugality path the pool exists for (route multi-seed validation
to 36 free cores, Claude only at the endpoints). The pool was healthy the entire time.

The RIGHT test is: is the code present, does the venv import the backend, and does the node carry the runner you
intend to dispatch? A node MISSING a recently-added runner is not "broken" — it is STALE, and the fix is a
re-rsync (`pool_provision.sh`), not a repair. This distinguishes the two so the next session does not confuse them.

    python -m tools.pool_health                       # venv + code, all nodes
    python -m tools.pool_health --runner research.runners.my_gate   # also require this runner present
    python -m tools.pool_health --json                # machine-readable, for lane_check/heartbeat to consult

Exit 0 iff every reachable node is HEALTHY (and, if --runner given, carries it).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys

NODES = ["pool40", "pool41", "pool42"]
REMOTE = "~/derisk-pool/sim"


def check_node(node: str, runner: str | None, timeout_s: int = 20) -> dict:
    """SSH once; test code present + venv imports numpy/scipy + (optional) a specific runner file exists."""
    runner_path = ""
    if runner:
        # research.runners.foo -> research/runners/foo.py
        runner_path = REMOTE + "/" + runner.replace(".", "/") + ".py"
    script = (
        'test -d %s/sim && echo CODE_OK || echo CODE_MISSING; '
        '%s/.venv/bin/python -c "import numpy,scipy" >/dev/null 2>&1 && echo VENV_OK || echo VENV_BROKEN; '
        'ls %s/research/runners/*.py 2>/dev/null | wc -l; '
        % (REMOTE, REMOTE, REMOTE)
    )
    if runner_path:
        script += 'test -f %s && echo RUNNER_OK || echo RUNNER_MISSING' % runner_path
    try:
        out = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", node, script],
                             capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return {"node": node, "reachable": False, "healthy": False, "why": "ssh timeout"}
    if out.returncode != 0 and "CODE_" not in out.stdout:
        return {"node": node, "reachable": False, "healthy": False,
                "why": (out.stderr or "unreachable").strip()[:120]}
    text = out.stdout
    code_ok = "CODE_OK" in text
    venv_ok = "VENV_OK" in text
    nrun = next((int(t) for t in text.split() if t.isdigit()), 0)
    runner_ok = (not runner) or ("RUNNER_OK" in text)
    healthy = code_ok and venv_ok
    r = {"node": node, "reachable": True, "healthy": healthy, "code": code_ok,
         "venv": venv_ok, "runners": nrun}
    if runner:
        r["runner_present"] = runner_ok
        r["stale_for_runner"] = healthy and not runner_ok  # healthy node, just missing a recent runner
    if not healthy:
        r["why"] = "code missing" if not code_ok else "venv broken (numpy/scipy import failed)"
        r["fix"] = "bash tools/pool_provision.sh %s" % node
    elif runner and not runner_ok:
        r["why"] = "STALE: healthy but missing %s — re-rsync to update (NOT broken)" % runner
        r["fix"] = "bash tools/pool_provision.sh %s" % node
    return r


def check_all(runner: str | None = None) -> list[dict]:
    return [check_node(n, runner) for n in NODES]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Correct pool health check (venv+code, not git).")
    ap.add_argument("--runner", default=None, help="also require this runner module present on each node")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    rows = check_all(args.runner)
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        for r in rows:
            if not r["reachable"]:
                print("  %-7s ⛔ UNREACHABLE — %s" % (r["node"], r.get("why", "")))
            elif not r["healthy"]:
                print("  %-7s ⛔ BROKEN — %s | fix: %s" % (r["node"], r.get("why"), r.get("fix")))
            elif r.get("stale_for_runner"):
                print("  %-7s ⚠  STALE — %s | %s" % (r["node"], r.get("why"), r.get("fix")))
            else:
                print("  %-7s ✅ HEALTHY (code+venv, %d runners)" % (r["node"], r["runners"]))
    ok = all(r["healthy"] and (not args.runner or r.get("runner_present")) for r in rows)
    return 0 if ok else 1


def selftest() -> list[str]:
    """No live SSH: test the pure PARSER on canned remote output (healthy / broken venv / stale runner)."""
    bad = []

    def parse(text, runner=None):
        code_ok = "CODE_OK" in text
        venv_ok = "VENV_OK" in text
        runner_ok = (not runner) or ("RUNNER_OK" in text)
        healthy = code_ok and venv_ok
        return {"healthy": healthy, "stale": healthy and runner and not runner_ok, "venv": venv_ok}

    if not parse("CODE_OK\nVENV_OK\n1412\n")["healthy"]:
        bad.append("a healthy node did not read as healthy")
    if parse("CODE_OK\nVENV_BROKEN\n1412\n")["healthy"]:
        bad.append("a broken venv (the REAL failure) read as healthy — a silent pass")
    st = parse("CODE_OK\nVENV_OK\n1412\nRUNNER_MISSING", runner="research.runners.new")
    if not st["stale"] or not st["healthy"]:
        bad.append("a healthy-but-missing-runner node was not read as STALE (would be mis-called broken)")
    if not parse("CODE_OK\nVENV_OK\n1412\nRUNNER_OK", runner="research.runners.new")["healthy"]:
        bad.append("a node carrying the runner did not read healthy")
    return bad


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        problems = selftest()
        raise SystemExit("⛔ pool_health selftest FAILED: " + "; ".join(problems) if problems
                         else print("pool_health selftest PASSED (healthy/broken-venv/stale-runner parsing)."))
    raise SystemExit(main())
