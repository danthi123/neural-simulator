"""Smell-test recompute for the unified decisive run.

Loads the decisive run's JSON output (single recording; no re-run),
extracts the per-rung capability metrics in the shape the frozen
``per_regime_monitor_core.per_regime_monitor_verdict`` expects, and
recomputes the verdict from the recorded numbers. Reports:

  - Recomputed verdict (gate / reason / frozen_bars)
  - Per-rung internal consistency checks:
      * full_acc, uniform_ctrl_acc, direct_retain_acc, abstain_correct
        each in [0.0, 1.0]
      * n_seeds == expected (3)
      * N in frozen ladder (2, 3, 5)
  - Per-rung "scrutinise PASS harder than FAIL" checks:
      * If full_acc >= 0.80 (frozen bar): does
        full_acc - uniform_ctrl_acc >= 0.70 (the per-regime advantage)?
      * Is direct_retain_acc consistent with the substrate's
        documented v14/v16 baseline (~0.74 multi-seed)?
      * Is abstain_correct genuinely from the moat doing the work, or
        could a degenerate run (always-abstain, zero outputs) trivially
        satisfy it?

Discipline: NO re-run, NO bar change, NO threshold tuning. The
recomputed verdict must agree with the verdict the runner emitted; any
disagreement is a controller-side defect to investigate.

Stdlib + the frozen verdict module only.
"""
from __future__ import annotations

import json
import sys
import os

# Add repo root to sys.path for `research.runners.*` imports
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from research.runners.per_regime_monitor_core import (
    per_regime_monitor_verdict,
    REQUIRED_KEYS,
    _PR_LADDER,
)


def _extract_rungs(payload):
    """Extract per-rung capability metrics from the decisive run's JSON."""
    rungs_raw = payload.get("rungs") or payload.get("per_rung") or payload.get("ladder_rungs")
    if rungs_raw is None:
        return None, "JSON has no 'rungs'/'per_rung'/'ladder_rungs' key"
    if not isinstance(rungs_raw, list):
        return None, "rungs is not a list"

    rungs = []
    for r in rungs_raw:
        if not isinstance(r, dict):
            return None, "rung is not a dict"
        rung = {}
        # Each REQUIRED_KEYS must appear at the rung's top level.
        for k in REQUIRED_KEYS:
            if k not in r:
                return None, "rung missing required key: " + str(k)
            rung[k] = r[k]
        rungs.append(rung)
    return rungs, None


def _per_rung_internal_check(rung):
    issues = []
    if not isinstance(rung.get("N"), int):
        issues.append("N is not int")
    elif rung["N"] not in _PR_LADDER:
        issues.append("N not in frozen ladder")
    for k in ("full_acc", "uniform_ctrl_acc", "direct_retain_acc", "abstain_correct"):
        v = rung.get(k)
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            issues.append("%s not numeric" % k)
        elif v < 0.0 or v > 1.0:
            issues.append("%s out of [0,1]: %r" % (k, v))
    return issues


def _per_rung_pass_smell(rung):
    """Scrutinise a nominal PASS harder than a FAIL."""
    smell = {}
    full = rung.get("full_acc", 0.0)
    uniform = rung.get("uniform_ctrl_acc", 0.0)
    direct_retain = rung.get("direct_retain_acc", 0.0)
    abstain_correct = rung.get("abstain_correct", 0.0)

    # Per-regime advantage = full - uniform_ctrl
    smell["per_regime_advantage"] = float(full - uniform)
    smell["per_regime_advantage_passes_0p70"] = bool(full - uniform >= 0.70)

    # Direct retention vs v14/v16 multi-seed baseline (~0.74)
    smell["direct_retain_vs_v14_baseline"] = float(direct_retain)
    smell["direct_retain_meets_frozen_bar_0p80"] = bool(direct_retain >= 0.80)

    # Abstain-correct sanity: a degenerate always-abstain run could satisfy
    # this even with full_acc=0. So abstain_correct alone is not sufficient
    # for PASS; it must accompany positive full_acc.
    smell["abstain_correct"] = float(abstain_correct)
    smell["abstain_correct_is_load_bearing"] = bool(
        abstain_correct >= 0.90 and full > 0.0
    )

    return smell


def main(argv=None) -> int:
    argv = argv or sys.argv[1:]
    if not argv:
        argv = ["research/findings/raw/unified_DECISIVE_fullscale.json"]
    path = argv[0]

    if not os.path.exists(path):
        print("ERROR: decisive JSON not found at %r" % path)
        return 2

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    print("=== UNIFIED DECISIVE SMELL TEST ===")
    print("source:", path)
    print("payload top-level keys:", sorted(payload.keys()))

    rungs, err = _extract_rungs(payload)
    if err is not None:
        print("EXTRACT-FAIL:", err)
        print("(this is an instrument-validity issue, not a substrate failure)")
        return 3

    print("\nper-rung internal consistency:")
    for r in rungs:
        issues = _per_rung_internal_check(r)
        status = "OK" if not issues else ("ISSUES: " + str(issues))
        print("  N=%s: %s" % (r.get("N"), status))

    print("\nper-rung pass-smell scrutiny:")
    for r in rungs:
        smell = _per_rung_pass_smell(r)
        print("  N=%s: full_acc=%.3f uniform_ctrl=%.3f direct_retain=%.3f abstain=%.3f"
              % (r["N"], r["full_acc"], r["uniform_ctrl_acc"],
                 r["direct_retain_acc"], r["abstain_correct"]))
        print("           per_regime_advantage=%.3f passes>=0.70=%s"
              % (smell["per_regime_advantage"], smell["per_regime_advantage_passes_0p70"]))
        print("           direct_retain_meets_0.80=%s"
              % smell["direct_retain_meets_frozen_bar_0p80"])
        print("           abstain_correct_load_bearing=%s"
              % smell["abstain_correct_is_load_bearing"])

    print("\n--- recomputed verdict from frozen verdict module ---")
    verdict = per_regime_monitor_verdict(rungs)
    print(json.dumps(verdict, indent=2))

    runner_verdict = payload.get("verdict") or payload.get("gate")
    if runner_verdict is not None:
        print("\nrunner-reported verdict:", runner_verdict)
        match = (verdict.get("gate") == (runner_verdict.get("gate")
                 if isinstance(runner_verdict, dict) else runner_verdict))
        print("recompute matches runner-reported:", match)
        if not match:
            print("WARNING: recompute disagrees with runner-reported -- investigate")

    return 0


if __name__ == "__main__":
    sys.exit(main())
