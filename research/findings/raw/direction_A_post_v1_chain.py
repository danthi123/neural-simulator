"""Direction A POST-v1 analysis chain: runs strict top-1 postproc +
smell test + decides next-direction launch.

Triggered manually after Direction A v1's OUT_JSON appears (
research/findings/raw/direction_A_ec_context_sequence_full.json).
Encapsulates the complete decision chain so the controller doesn't
manually re-derive each step.

Steps:
1. Verify v1 OUT_JSON exists; read top-3 multi-seed result
2. Run strict-top-1 post-processor on cached per-seed trials JSONs
3. Run smell test (3 anti-cheat controls) with strict-top-1 verdict
4. Print clean summary
5. Recommend next-direction launch:
   - PASS_CONTROLS_DECISIVE_TOP1 -> pillar n=104 + capacity sweep
   - PASS_TOP1_COLLAPSES_TO_MULTITAG -> the cue isn't load-bearing;
     v2 likely won't help; pivot to Direction E substrate
   - PASS_TOP1_COLLAPSES_TO_CUE_ALONE -> engram isn't load-bearing;
     mechanism is just ec_context->pool weights; v2 worth trying
   - PASS_TOP1_WITH_WEAK_CONTROLS -> partial signal; refine + iterate
   - MAIN_TOP1_BELOW_BAR -> launch v2 (plasticity-during-encoding)

No GPU work (loads bridges briefly but doesn't simulate). ~15 min
wall total (smell test is the slow part: 3 controls x 8 sequences
x 3 seeds x 3 repeats).
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def main():
    print(f"=== Direction A POST-v1 analysis chain ===", flush=True)

    v1_out = os.path.join(
        _HERE, "direction_A_ec_context_sequence_full.json")
    if not os.path.exists(v1_out):
        print(f"  [FATAL] v1 OUT_JSON not found: {v1_out}",
              flush=True)
        print(f"  Direction A v1 hasn't completed yet. Wait.",
              flush=True)
        return 1

    with open(v1_out, "r", encoding="utf-8") as f:
        v1_data = json.load(f)
    top3_mean = v1_data.get("slot3_accuracy_mean", float("nan"))
    top3_per_seed = v1_data.get("slot3_accuracy_per_seed", [])
    seeds = v1_data.get("seeds", [])
    print(f"\n--- v1 top-3 result ---", flush=True)
    print(f"  multi-seed top-3 mean: {top3_mean:.3f}", flush=True)
    print(f"  per-seed: {top3_per_seed}", flush=True)
    print(f"  seeds: {seeds}", flush=True)
    print(f"  (NOTE: top-3 is DEGENERATE per reviewer; strict "
          f"top-1 is the load-bearing metric)", flush=True)

    print(f"\n--- Step 1: STRICT TOP-1 postproc ---", flush=True)
    t0 = time.time()
    subprocess.run(
        [sys.executable, "-m",
         "research.findings.raw.direction_A_strict_top1_postproc"],
        cwd=_REPO_ROOT, check=True)
    print(f"  Step 1 wall: {(time.time()-t0)/60:.1f} min",
          flush=True)

    top1_p = os.path.join(_HERE, "direction_A_strict_top1_postproc.json")
    with open(top1_p, "r", encoding="utf-8") as f:
        top1_data = json.load(f)
    top1_mean = top1_data.get("strict_top1_mean", float("nan"))
    top1_verdict = top1_data.get("verdict", "")
    print(f"\n  strict top-1 multi-seed: {top1_mean:.3f}",
          flush=True)
    print(f"  verdict: {top1_verdict}", flush=True)

    print(f"\n--- Step 2: SMELL TEST (top-3 + top-1 controls) ---",
          flush=True)
    t0 = time.time()
    subprocess.run(
        [sys.executable, "-m",
         "research.findings.raw.direction_A_smell_test"],
        cwd=_REPO_ROOT, check=True)
    print(f"  Step 2 wall: {(time.time()-t0)/60:.1f} min",
          flush=True)

    smell_p = os.path.join(_HERE, "direction_A_smell_test.json")
    if os.path.exists(smell_p):
        with open(smell_p, "r", encoding="utf-8") as f:
            smell_data = json.load(f)
        smell_verdict = smell_data.get("verdict", "")
        print(f"\n  smell test verdict: {smell_verdict}",
              flush=True)
        # top-1 control means
        wp_top1 = smell_data.get("ctrl_wrong_position_top1_mean", None)
        ns_top1 = smell_data.get("ctrl_no_stim_top1_mean", None)
        nc_top1 = smell_data.get("ctrl_no_cue_top1_mean", None)
        if wp_top1 is not None:
            print(f"  top-1 control means:", flush=True)
            print(f"    wrong_pos: {wp_top1:.3f}", flush=True)
            print(f"    no_stim:   {ns_top1:.3f}", flush=True)
            print(f"    no_cue:    {nc_top1:.3f}", flush=True)
    else:
        smell_verdict = "SMELL_TEST_FAILED_TO_PRODUCE_OUTPUT"
        print(f"  smell test failed to produce output", flush=True)

    print(f"\n=== NEXT-DIRECTION RECOMMENDATION ===", flush=True)
    if "PASS_CONTROLS_DECISIVE_TOP1" in smell_verdict:
        recommendation = "RECORD_PILLAR_N104_LAUNCH_CAPACITY_SWEEP"
        print(f"  Top-1 multi-seed PASSes >= 0.80 + controls "
              f"decisive. Record pillar n=104 (Direction A "
              f"ec_context substrate sequence storage). Launch "
              f"capacity sweep (slot_count 4,5,6,7).",
              flush=True)
    elif "PASS_TOP1_COLLAPSES_TO_MULTITAG" in smell_verdict:
        recommendation = "PIVOT_TO_DIRECTION_E_SUBSTRATE"
        print(f"  Top-1 PASS but ec_context cue isn't load-bearing"
              f" (Direction A collapses to multitag). v2 won't "
              f"help (plasticity-during-encoding only changes the "
              f"already-frozen cue weights). PIVOT to Direction E"
              f" substrate (theta-gamma temporal phase code).",
              flush=True)
    elif "PASS_TOP1_COLLAPSES_TO_CUE_ALONE" in smell_verdict:
        recommendation = "LAUNCH_V2_HIGH_CONFIDENCE"
        print(f"  Top-1 PASS but engram isn't load-bearing -- the"
              f" mechanism is the ec_context->pool weights. v2 "
              f"(opens those gates during encoding) should "
              f"strengthen this signal further. Launch v2.",
              flush=True)
    elif "PASS_TOP1_WITH_WEAK_CONTROLS" in smell_verdict:
        recommendation = "LAUNCH_V2_MEDIUM_CONFIDENCE"
        print(f"  Top-1 PASS with weak controls -- multiple signals"
              f" contribute. v2 should strengthen the learned-"
              f"pathway component. Launch v2; if v2 doesn't push "
              f"past, pivot to Direction E substrate.", flush=True)
    elif "MAIN_TOP1_BELOW_BAR" in smell_verdict:
        recommendation = "LAUNCH_V2_INVESTIGATE_BOUND"
        print(f"  Top-1 BELOW bar. v2 (plasticity fix) is the "
              f"natural next test. Also run weight inspection to "
              f"confirm v2 hypothesis. If v2 also below bar, "
              f"pivot to Direction E substrate.", flush=True)
    else:
        recommendation = "MANUAL_DECISION_NEEDED"
        print(f"  Smell test verdict unclear: {smell_verdict}. "
              f"Manual decision needed.", flush=True)

    print(f"\n  RECOMMENDED LAUNCH: {recommendation}", flush=True)
    print(f"\n  next commands:", flush=True)
    print(f"  - Weight inspection: python -m research.findings.raw."
          f"direction_A_weight_inspection", flush=True)
    if "V2" in recommendation:
        print(f"  - V2 plasticity-during-encoding: python -m "
              f"research.findings.raw.direction_A_v2_with_plasticity"
              f" (~30 min GPU)", flush=True)
    if "DIRECTION_E" in recommendation or "V2" in recommendation:
        print(f"  - Direction E substrate Task 0: python -m "
              f"research.findings.raw.direction_E_substrate_task0_"
              f"grounding (~5 min GPU)", flush=True)
    if "CAPACITY" in recommendation:
        print(f"  - Capacity sweep: python -m research.findings.raw"
              f".direction_A_capacity_sweep (~30-60 min GPU)",
              flush=True)

    out = {
        "v1_top3_mean": top3_mean,
        "v1_top3_per_seed": top3_per_seed,
        "strict_top1_mean": top1_mean,
        "strict_top1_verdict": top1_verdict,
        "smell_test_verdict": smell_verdict,
        "next_direction_recommendation": recommendation,
    }
    out_p = os.path.join(_HERE, "direction_A_post_v1_chain.json")
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_p}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
