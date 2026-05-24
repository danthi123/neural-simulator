"""Direction G POST chain: runs after Direction G completes; auto-
recommends next direction based on outcome.

Steps:
1. Read Direction G OUT_JSON (strict top-1 multi-seed)
2. Recommend next:
   - DIRECTION_G_PASS -> dispatch fresh-agent reviewer; if CLEAR,
     record pillar n=105 (substrate theta-gamma+hippocampus sequence
     storage VALIDATED); the catalog's load-bearing positional
     binding (D.04 + D.11 + N.16) is empirically vindicated
   - DIRECTION_G_PARTIAL_HIPPO_HELPS -> hippocampus contributes
     above the cortical-only floor but doesn't fully solve; diagnose
     what's still missing; pre-register cheap-first probe for that
     gap
   - DIRECTION_G_NO_IMPROVEMENT_OVER_CORTICAL -> hippocampus addition
     doesn't change anything; bound is deeper than hippocampus
     absence (likely the weak concept-pool dynamics themselves, per
     pillar n=104 diagnosis); next direction is Direction H (canon
     concept-pool dynamics, with controls for v14/v16 multi-concept
     trainability preservation)
   - DIRECTION_G_HIPPO_NEGATIVE -> hippocampus actually HURT;
     unexpected substrate interaction; characterize precisely

Reuses (skips) smell test if Direction G < bar (the BOUNDARY
case from pillar n=104 already characterized the controls).

~5-15 min wall total.
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))


def main():
    print(f"=== Direction G POST chain ===", flush=True)

    g_p = os.path.join(
        _HERE, "direction_G_hippo_theta_gamma_substrate.json")
    if not os.path.exists(g_p):
        print(f"  [FATAL] Direction G OUT_JSON not found",
              flush=True)
        return 1

    with open(g_p, "r", encoding="utf-8") as f:
        g = json.load(f)
    g_top1 = g.get("strict_top1_mean", float("nan"))
    g_per_seed = g.get("per_seed_acc", [])
    g_verdict = g.get("verdict", "")
    print(f"\n  Direction G strict top-1 multi-seed: {g_top1:.3f}",
          flush=True)
    print(f"  per-seed: {g_per_seed}", flush=True)
    print(f"  Direction G verdict: {g_verdict}", flush=True)

    print(f"\n  Comparison to prior substrate attempts:",
          flush=True)
    print(f"    Direction A v1 (cortical+ec_context, frozen): 0.333",
          flush=True)
    print(f"    Direction A v2 (cortical+ec_context, learned): 0.292",
          flush=True)
    print(f"    Direction E Task 1 (cortical+theta-gamma):    0.250",
          flush=True)
    print(f"    Direction G (HIPPO+theta-gamma):              "
          f"{g_top1:.3f}", flush=True)

    # If PASS, run smell test for full controls verification
    smell_verdict = None
    if g_top1 >= 0.80:
        print(f"\n--- Direction G PASS -> running smell test ---",
              flush=True)
        t0 = time.time()
        try:
            subprocess.run(
                [sys.executable, "-m",
                 "research.findings.raw.direction_G_smell_test"],
                cwd=_REPO_ROOT, check=True)
            smell_p = os.path.join(
                _HERE, "direction_G_smell_test.json")
            if os.path.exists(smell_p):
                with open(smell_p, "r", encoding="utf-8") as f:
                    smell_data = json.load(f)
                smell_verdict = smell_data.get("verdict")
            print(f"  smell test wall: {(time.time()-t0)/60:.1f} min;"
                  f" verdict: {smell_verdict}", flush=True)
        except Exception as e:
            print(f"  smell test failed: {e}", flush=True)

    print(f"\n=== NEXT-DIRECTION RECOMMENDATION ===", flush=True)
    if g_top1 >= 0.80:
        recommendation = "PILLAR_N105_DISPATCH_REVIEWER"
        print(f"  PASS at multi-seed >= 0.80. Dispatch fresh-agent"
              f" adversarial reviewer; if CLEAR, record pillar "
              f"n=105 (substrate theta-gamma + hippocampus sequence"
              f" storage VALIDATED). The catalog's load-bearing "
              f"positional binding mechanism (D.04 + D.11 + N.16)"
              f" empirically vindicated. Next-next: chat REPL "
              f"integration; user-facing sequence demos.",
              flush=True)
    elif g_top1 > 0.40:
        recommendation = "PARTIAL_DIAGNOSE_REMAINING_GAP"
        print(f"  {g_top1:.3f} above prior 0.25-0.33 cluster but"
              f" below {0.80} bar. Hippocampus HELPS; not fully"
              f" sufficient. Diagnose remaining gap: cheap-first"
              f" probe of WHICH hippocampal subcircuit (CA3 "
              f"recurrent vs CA1 sequence cells vs trisynaptic"
              f" loop vs SWR consolidation) is the load-bearing"
              f" piece; identify what additional architectural"
              f" augmentation closes the bound. Honest BOUNDARY"
              f" finding pending detailed diagnosis.", flush=True)
    elif g_top1 > 0.125:
        recommendation = "NO_IMPROVEMENT_PIVOT_TO_DIRECTION_H"
        print(f"  {g_top1:.3f} similar to cortical-only result. "
              f"Hippocampus addition didn't help; bound is deeper"
              f" than hippocampus absence -- likely the weak "
              f"concept-pool dynamics (v14/v16's deliberate "
              f"canon-amplifies-bias-collapse design). Pivot to "
              f"Direction H: canon concept-pool dynamics with "
              f"pre-registered controls for v14/v16 multi-concept"
              f" trainability preservation.", flush=True)
    else:
        recommendation = "HIPPO_HURT_DIAGNOSE_INTERACTION"
        print(f"  {g_top1:.3f} BELOW the cortical-only 0.25 floor."
              f" Hippocampus actually HURT performance. Unexpected"
              f" substrate interaction. Diagnose: is HIPPO substrate"
              f" injecting noise via SWR pathways during retrieval?"
              f" Cheap-first probe: disable SWR consolidation gates"
              f" during retrieval; see if performance recovers.",
              flush=True)

    print(f"\n  RECOMMENDED: {recommendation}", flush=True)
    out = {
        "direction_G_strict_top1_mean": g_top1,
        "direction_G_per_seed": g_per_seed,
        "direction_G_verdict": g_verdict,
        "smell_test_verdict": smell_verdict,
        "next_direction_recommendation": recommendation,
        "comparison": {
            "direction_A_v1": 0.333,
            "direction_A_v2": 0.292,
            "direction_E_task1": 0.250,
            "direction_G": g_top1,
        },
    }
    out_p = os.path.join(_HERE, "direction_G_post_chain.json")
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_p}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
