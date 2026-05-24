"""Direction E substrate Task 1 POST chain: runs smell test +
prepares for adversarial review when Task 1 completes.

Triggered after Task 1's OUT_JSON appears at
research/findings/raw/direction_E_substrate_task1_full.json.

Steps:
1. Read Task 1 strict top-1 multi-seed result + verdict
2. Run smell test (3 controls: wrong-slot, no-stim, no-window)
3. Read smell test verdict
4. Recommend next:
   - PASS_CONTROLS_DECISIVE + main >= 0.80 -> dispatch fresh-agent
     adversarial review; if CLEAR, record pillar n=104 + update
     capability_status; substrate theta-gamma sequence storage
     VALIDATED
   - PASS_COLLAPSES_TO_MULTITAG / WRONG_SLOT_INSENSITIVE / WEAK ->
     precise BOUNDARY; honest propagation; the substrate has the
     theta-gamma mechanism partially but not robustly
   - MAIN_BELOW_BAR -> honest NEGATIVE; substrate dynamics
     fundamentally incompatible with theta-gamma temporal binding;
     the ec_context substrate Direction A AND theta-gamma substrate
     Direction E both fail; pivot to substrate redesign

~15-20 min wall (smell test is the slow part).
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
    print(f"=== Direction E substrate Task 1 POST chain ===",
          flush=True)

    task1_p = os.path.join(
        _HERE, "direction_E_substrate_task1_full.json")
    if not os.path.exists(task1_p):
        print(f"  [FATAL] Task 1 OUT_JSON not found", flush=True)
        return 1

    with open(task1_p, "r", encoding="utf-8") as f:
        t1 = json.load(f)
    main_top1 = t1.get("strict_top1_mean", float("nan"))
    main_per_seed = t1.get("per_seed_acc", [])
    main_verdict = t1.get("verdict", "")
    print(f"\n  Task 1 strict top-1 multi-seed: {main_top1:.3f}",
          flush=True)
    print(f"  per-seed: {main_per_seed}", flush=True)
    print(f"  Task 1 verdict: {main_verdict}", flush=True)

    print(f"\n--- Step 1: SMELL TEST ---", flush=True)
    t0 = time.time()
    subprocess.run(
        [sys.executable, "-m",
         "research.findings.raw.direction_E_substrate_task1_smell_test"],
        cwd=_REPO_ROOT, check=True)
    print(f"  smell test wall: {(time.time()-t0)/60:.1f} min",
          flush=True)

    smell_p = os.path.join(
        _HERE, "direction_E_substrate_task1_smell_test.json")
    if os.path.exists(smell_p):
        with open(smell_p, "r", encoding="utf-8") as f:
            smell_data = json.load(f)
        smell_verdict = smell_data.get("verdict", "")
        print(f"\n  smell test verdict: {smell_verdict}",
              flush=True)
    else:
        smell_verdict = "SMELL_TEST_FAILED_TO_PRODUCE_OUTPUT"

    print(f"\n=== NEXT-DIRECTION RECOMMENDATION ===", flush=True)
    if ("PASS_CONTROLS_DECISIVE" in smell_verdict
            and main_top1 >= 0.80):
        recommendation = "DISPATCH_FRESH_AGENT_REVIEWER_PILLAR_N104"
        print(f"  Task 1 strict top-1 multi-seed {main_top1:.3f}"
              f" >= 0.80 + controls decisive. NEXT: dispatch fresh-"
              f"agent adversarial reviewer; if CLEAR, record "
              f"capability_status pillar n=104 (Direction E "
              f"substrate theta-gamma sequence storage VALIDATED).",
              flush=True)
    elif ("PASS_COLLAPSES" in smell_verdict
            or "PASS_WITH_WEAK_CONTROLS" in smell_verdict):
        recommendation = "BOUNDARY_HONEST_PROPAGATION"
        print(f"  Task 1 strict top-1 PASS but controls show "
              f"collapse / weakness. The mechanism is partially "
              f"present but not robustly so. Honest BOUNDARY "
              f"finding; write findings doc; consider whether the "
              f"weak components are fixable.", flush=True)
    elif "MAIN_BELOW_BAR" in smell_verdict or main_top1 < 0.80:
        recommendation = "NEGATIVE_HONEST_BOTH_SUBSTRATES_BOUNDED"
        print(f"  Task 1 strict top-1 multi-seed {main_top1:.3f}"
              f" < 0.80. Both Direction A (ec_context) AND "
              f"Direction E (theta-gamma) substrate mechanisms "
              f"FAIL the strict bar. The v16 substrate dynamics "
              f"are fundamentally bounded for sequence storage. "
              f"Honest biology-translatable finding: the substrate"
              f" needs architectural changes (e.g., stronger "
              f"concept-pool dynamics; dedicated sequence-binding "
              f"region; or larger substrate) to support reliable "
              f"sequence-position retrieval.", flush=True)
    else:
        recommendation = "MANUAL_DECISION_NEEDED"
        print(f"  Verdict unclear: {smell_verdict}", flush=True)

    print(f"\n  RECOMMENDED LAUNCH: {recommendation}", flush=True)

    out = {
        "task1_strict_top1_mean": main_top1,
        "task1_per_seed": main_per_seed,
        "task1_verdict": main_verdict,
        "smell_test_verdict": smell_verdict,
        "next_direction_recommendation": recommendation,
    }
    out_p = os.path.join(_HERE, "direction_E_task1_post_chain.json")
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_p}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
