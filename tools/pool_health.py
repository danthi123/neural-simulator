#!/usr/bin/env python3
"""Correct health check for the mini-PC pool (nodes: tools/pool_nodes.conf). Tests what actually matters.

WHY (2026-08-06): the pool is deployed by RSYNC, not `git clone` — there is NO `.git` on a node BY DESIGN, and
"fatal: not a git repository" is NOT a broken pool. Two sub-agents once wrote that false diagnosis onto the board
for a whole session. The RIGHT test is: is a COMPLETED revision present, does its venv import the backend, and
does it carry the runner you intend to dispatch. A node MISSING a recently-added runner is not "broken" — it is
STALE, and the fix is a re-provision, not a repair.

WHAT IT ASKS (2026-10-12, RULING misc-1012b): the old `~/derisk-pool/sim` checkout is RETIRED. The pool serves
`~/derisk-pool/revisions/<sha>` directories, each attested by the `.provisioned_ok` completion marker written
LAST by tools/pool_provision.sh (after rsync + venv + manifest-verify). pool_health now asks the SAME marker
question as the dispatcher, via the revision's own tools/pool_revision_marker.sh (the ONE shared predicate —
see its header for the drift defect it fixed), so it can never disagree with what dispatch will actually run.
The newest MARKERED revision is the live one; a newer unmarked dir means provisioning in progress (NOT broken).

The fix hint is the SANCTIONED request path: pool_provision.sh is banned on this host (the server-side
`touch research/queue/provision.req/<sha>` from a research/* branch provisions; RULING compute-placement).

    python -m tools.pool_health                       # marker + venv + code, all nodes
    python -m tools.pool_health --runner research.runners.my_gate   # also require this runner present
    python -m tools.pool_health --json                # machine-readable, for tool_health/heartbeat to consult
    python -m tools.pool_health --selftest            # pure-parser tests on canned remote output (no SSH)

Exit 0 iff every reachable node is HEALTHY (and, if --runner given, carries it).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

# tools/ dir on sys.path so this resolves whether pool_health.py is run directly, as `python -m
# tools.pool_health`, or loaded by path in a test.
_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)
import pool_nodes  # noqa: E402 -- single source of truth: tools/pool_nodes.conf

NODES = pool_nodes.default_nodes()
REVISIONS_ROOT = "~/derisk-pool/revisions"
# The sanctioned fix path. pool_provision.sh is banned on this host (server-side provision.req instead).
FIX_HINT = ("server-side provision: touch research/queue/provision.req/<sha> from a research/* branch "
            "(pool_provision.sh is banned on this host)")


def _remote_script(runner_path: str) -> str:
    """The exact remote shell script: newest-first revision dirs, marker via the revision's OWN
    tools/pool_revision_marker.sh (fallback: the literal default), then venv/runner checks in the
    newest MARKERED revision. Plain semicolons, no nesting traps — it rides one ssh argv element."""
    parts = [
        'REVROOT="$HOME/derisk-pool/revisions"',
        'cands=$(ls -1t "$REVROOT" 2>/dev/null | head -4)',
        '[ -n "$cands" ] || echo NO_REVISIONS',
        'live=""',
        'for s in $cands; do d="$REVROOT/$s"; m=".provisioned_ok"; '
        'if [ -f "$d/tools/pool_revision_marker.sh" ]; then '
        '. "$d/tools/pool_revision_marker.sh" >/dev/null 2>&1; m="${POOL_REVISION_MARKER_FILE:-.provisioned_ok}"; '
        'fi; '
        'if [ -f "$d/$m" ]; then echo "MARKER_OK $s"; [ -z "$live" ] && live="$d"; '
        'else echo "MARKER_MISSING $s"; fi; '
        'done',
        'if [ -n "$live" ]; then echo CODE_OK; '
        '"$live/.venv/bin/python" -c "import numpy,scipy" >/dev/null 2>&1 && echo VENV_OK || echo VENV_BROKEN; '
        'ls "$live"/research/runners/*.py 2>/dev/null | wc -l',
    ]
    if runner_path:
        parts.append('export RUNNER_REL=%s; test -f "$live/$RUNNER_REL" && echo RUNNER_OK || echo RUNNER_MISSING'
                     % runner_path)
    parts.append('fi')
    return "; ".join(parts)


def parse_remote(text: str, runner: str | None = None) -> dict:
    """THE parser (used by check_node AND selftest — a reimplementation in the test would just lie).
    Reads the canned/real remote output: marker states newest-first, then CODE/VENV/count[/RUNNER]."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    marker_ok = [ln.split()[1] for ln in lines if ln.startswith("MARKER_OK ") and len(ln.split()) > 1]
    partial = [ln.split()[1] for ln in lines if ln.startswith("MARKER_MISSING ") and len(ln.split()) > 1]
    code_ok = "CODE_OK" in lines
    venv_ok = "VENV_OK" in lines
    runners = next((int(ln) for ln in lines if ln.isdigit()), 0)
    runner_ok = (not runner) or ("RUNNER_OK" in lines)
    live = bool(marker_ok) and code_ok
    return {
        "no_revisions": "NO_REVISIONS" in lines,
        "marker": bool(marker_ok),
        "revision": (marker_ok[0] if marker_ok else None),
        "partial": partial,
        "code": code_ok,
        "venv": venv_ok,
        "runners": runners,
        "runner_ok": runner_ok,
        "healthy": live and venv_ok,
    }


def check_node(node: str, runner: str | None, timeout_s: int = 20) -> dict:
    """SSH once; ask the revision-marker question + venv import + (optional) runner presence."""
    runner_path = ""
    if runner:
        # research.runners.foo -> research/runners/foo.py, relative to the live revision dir
        runner_path = runner.replace(".", "/") + ".py"
    script = _remote_script(runner_path)
    try:
        out = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", node, script],
                             capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return {"node": node, "reachable": False, "healthy": False, "why": "ssh timeout"}
    if out.returncode != 0 and not any(t in out.stdout for t in ("CODE_", "MARKER_", "NO_REVISIONS")):
        return {"node": node, "reachable": False, "healthy": False,
                "why": (out.stderr or "unreachable").strip()[:120]}
    p = parse_remote(out.stdout, runner)
    r = {"node": node, "reachable": True, "healthy": p["healthy"], "code": p["code"],
         "venv": p["venv"], "runners": p["runners"],
         "marker": p["marker"], "revision": p["revision"], "partial": p["partial"]}
    if runner:
        r["runner_present"] = p["runner_ok"]
        r["stale_for_runner"] = p["healthy"] and not p["runner_ok"]
    if p["no_revisions"]:
        r["why"] = "no revision dirs under %s" % REVISIONS_ROOT
        r["fix"] = FIX_HINT
    elif not p["marker"]:
        r["why"] = "STALE: %d revision dir(s) but none carries the .provisioned_ok marker" % len(p["partial"])
        r["fix"] = FIX_HINT
    elif not p["venv"]:
        r["why"] = "venv broken (numpy/scipy import failed) in revision %s" % p["revision"]
        r["fix"] = FIX_HINT
    elif runner and not p["runner_ok"]:
        r["why"] = "STALE: healthy (revision %s) but missing %s — re-provision (NOT broken)" % (p["revision"], runner)
        r["fix"] = FIX_HINT
    return r


def check_all(runner: str | None = None) -> list[dict]:
    return [check_node(n, runner) for n in NODES]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Correct pool health check: revision marker + venv, not git.")
    ap.add_argument("--runner", default=None, help="also require this runner module present in the live revision")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    rows = check_all(args.runner)
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        for r in rows:
            rev = r.get("revision") or "-"
            if not r["reachable"]:
                print("  %-7s ⛔ UNREACHABLE — %s" % (r["node"], r.get("why", "")))
            elif not r["healthy"]:
                print("  %-7s ⛔ BROKEN — %s | fix: %s" % (r["node"], r.get("why"), r.get("fix")))
            elif r.get("stale_for_runner"):
                print("  %-7s ⚠  STALE — %s | %s" % (r["node"], r.get("why"), r.get("fix")))
            else:
                note = (" (+%d unmarked dir(s) provisioning)" % len(r["partial"])) if r["partial"] else ""
                print("  %-7s ✅ HEALTHY (revision %s, venv, %d runners)%s"
                      % (r["node"], rev, r["runners"], note))
    ok = all(r["healthy"] and (not args.runner or r.get("runner_present")) for r in rows)
    return 0 if ok else 1


def selftest() -> list[str]:
    """No live SSH: canned remote output through the REAL parser (parse_remote) + the retired-path guard."""
    bad = []

    healthy = parse_remote("MARKER_OK abc123\nCODE_OK\nVENV_OK\n1412\n")
    if not healthy["healthy"] or healthy["revision"] != "abc123" or healthy["runners"] != 1412:
        bad.append("a markered+venv node did not read as healthy")
    if healthy["marker"] is not True or healthy["partial"]:
        bad.append("marker/partial flags wrong on a clean node")

    nomarker = parse_remote("MARKER_MISSING abc123\nMARKER_MISSING def456\n")
    if nomarker["healthy"]:
        bad.append("revision dirs without ANY marker read as healthy — the retired-checkout bug reborn")
    if not nomarker["partial"] or nomarker["revision"] is not None:
        bad.append("partial list/revision wrong when nothing is markered")

    venvbad = parse_remote("MARKER_OK abc123\nCODE_OK\nVENV_BROKEN\n1412\n")
    if venvbad["healthy"]:
        bad.append("a broken venv (the REAL failure) read as healthy — a silent pass")

    partial_ok = parse_remote("MARKER_MISSING new99\nMARKER_OK abc123\nCODE_OK\nVENV_OK\n900\n")
    if not partial_ok["healthy"] or partial_ok["revision"] != "abc123" or partial_ok["partial"] != ["new99"]:
        bad.append("a newer unmarked (provisioning-in-progress) dir broke the live-revision read")

    st = parse_remote("MARKER_OK abc123\nCODE_OK\nVENV_OK\n1412\nRUNNER_MISSING", runner="research.runners.new")
    if not st["healthy"] or st["runner_ok"]:
        bad.append("a healthy-but-missing-runner node was not read as STALE (would be mis-called broken)")
    if parse_remote("MARKER_OK abc123\nCODE_OK\nVENV_OK\n1412\nRUNNER_OK",
                    runner="research.runners.new")["healthy"] is not True:
        bad.append("a node carrying the runner did not read healthy")

    if parse_remote("NO_REVISIONS\n")["healthy"]:
        bad.append("an empty revisions root read as healthy")

    # The BANNED fix must never resurface, and the retired checkout must stay retired.
    # Needles are CONCATENATED so this block's own source can never satisfy (or trip) its own test.
    src = open(os.path.join(_TOOLS_DIR, "pool_health.py")).read()
    retired = '= "' + "~/derisk-pool/sim" + '"'
    banned = "bash tools/" + "pool_provision.sh"
    if retired in src:
        bad.append("the retired ~/derisk-pool/sim checkout reference came back")
    if banned in src:
        bad.append("pool_health suggests the banned pool_provision.sh fix")
    if 'REVISIONS_ROOT = "' + "~/derisk-pool/revisions" + '"' not in src:
        bad.append("the revisions-root constant was lost")
    if "provision.req" not in FIX_HINT or "banned" not in FIX_HINT:
        bad.append("FIX_HINT no longer names the sanctioned path + banned-tool constraint")
    return bad


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        problems = selftest()
        if problems:
            raise SystemExit("⛔ pool_health selftest FAILED: " + "; ".join(problems))
        print("pool_health selftest PASSED (marker/venv/stale-runner/partial parsing + banned-fix guard).")
        sys.exit(0)
    raise SystemExit(main())
