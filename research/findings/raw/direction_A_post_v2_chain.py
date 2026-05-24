"""Direction A POST-v2 analysis chain: runs after v2 completes.

Reads v2's OUT_JSON (strict top-1 multi-seed); recommends final
next-direction:
  - V2_STRICT_TOP1_PASS -> record pillar n=104; substrate sequence
    storage validated via ec_context with plasticity; launch capacity
    sweep + Direction E substrate (complementary mechanism)
  - V2_STRICT_TOP1_ABOVE_CHANCE_BELOW_BAR -> plasticity helps but
    doesn't fully solve; pivot to Direction E substrate (different
    mechanism)
  - V2_STRICT_TOP1_AT_CHANCE -> plasticity fix didn't help at all;
    confirms ec_context substrate cannot do positional binding; pivot
    to Direction E substrate

Also runs weight inspection diagnostic if not already done.

~5 min wall (no extra simulation; just JSON + recommendation logic).
"""
from __future__ import annotations
import json
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))


def main():
    print(f"=== Direction A POST-v2 analysis chain ===",
          flush=True)

    v2_p = os.path.join(_HERE, "direction_A_v2_with_plasticity.json")
    if not os.path.exists(v2_p):
        print(f"  [FATAL] v2 OUT_JSON not found: {v2_p}",
              flush=True)
        return 1

    with open(v2_p, "r", encoding="utf-8") as f:
        v2_data = json.load(f)
    top1_mean = v2_data.get("strict_top1_mean", float("nan"))
    top1_per_seed = v2_data.get("strict_top1_per_seed", [])
    top3_mean = v2_data.get("top3_mean", float("nan"))
    top3_per_seed = v2_data.get("top3_per_seed", [])
    verdict = v2_data.get("verdict", "")
    print(f"\n  v2 strict top-1 multi-seed: {top1_mean:.3f}",
          flush=True)
    print(f"  v2 per-seed top-1: {top1_per_seed}", flush=True)
    print(f"  v2 top-3 multi-seed (for comparison): {top3_mean:.3f}"
          f" per-seed {top3_per_seed}", flush=True)
    print(f"  v2 verdict: {verdict}", flush=True)

    # Also load v1 result for comparison
    v1_p = os.path.join(
        _HERE, "direction_A_ec_context_sequence_full.json")
    v1_top3 = float("nan")
    v1_top1 = float("nan")
    if os.path.exists(v1_p):
        with open(v1_p, "r", encoding="utf-8") as f:
            v1_data = json.load(f)
        v1_top3 = v1_data.get("slot3_accuracy_mean", float("nan"))
    top1_postproc_p = os.path.join(
        _HERE, "direction_A_strict_top1_postproc.json")
    if os.path.exists(top1_postproc_p):
        with open(top1_postproc_p, "r", encoding="utf-8") as f:
            t1d = json.load(f)
        v1_top1 = t1d.get("strict_top1_mean", float("nan"))
    print(f"\n  v1 vs v2 strict top-1: {v1_top1:.3f} -> "
          f"{top1_mean:.3f} (delta {top1_mean - v1_top1:+.3f})",
          flush=True)
    print(f"  v1 vs v2 top-3: {v1_top3:.3f} -> {top3_mean:.3f}",
          flush=True)

    # Weight inspection (optional; for v2 hypothesis confirmation)
    wi_p = os.path.join(_HERE, "direction_A_weight_inspection.json")
    if not os.path.exists(wi_p):
        print(f"\n--- Running weight inspection diagnostic ---",
              flush=True)
        try:
            subprocess.run(
                [sys.executable, "-m",
                 "research.findings.raw.direction_A_weight_inspection"],
                cwd=_REPO_ROOT, check=True)
            if os.path.exists(wi_p):
                with open(wi_p, "r", encoding="utf-8") as f:
                    wi = json.load(f)
                print(f"  weight inspection verdict: "
                      f"{wi.get('verdict', '')}", flush=True)
        except Exception as e:
            print(f"  weight inspection failed: {e}", flush=True)

    print(f"\n=== NEXT-DIRECTION RECOMMENDATION ===", flush=True)
    if "V2_STRICT_TOP1_PASS" in verdict:
        recommendation = "PILLAR_N104_PLUS_LAUNCH_DIRECTION_E_TASK0"
        print(f"  v2 STRICT TOP-1 PASS at {top1_mean:.3f} >= 0.80 ."
              f" Substrate ec_context positional binding works "
              f"WITH plasticity. Record pillar n=104. Launch "
              f"Direction E substrate Task 0 grounding as the "
              f"COMPLEMENTARY mechanism (different positional "
              f"binding primitive; the catalog has both).",
              flush=True)
    elif "V2_STRICT_TOP1_ABOVE_CHANCE_BELOW_BAR" in verdict:
        recommendation = "LAUNCH_DIRECTION_E_SUBSTRATE_TASK0_PLUS_1"
        print(f"  v2 {top1_mean:.3f} above chance but below bar. "
              f"ec_context substrate provides partial positional "
              f"binding but not at the strict bar. Pivot to "
              f"Direction E substrate (Task 0 -> Task 1; theta-"
              f"gamma temporal phase code is the principled "
              f"catalog alternative).", flush=True)
    elif "V2_STRICT_TOP1_AT_CHANCE" in verdict:
        recommendation = "LAUNCH_DIRECTION_E_SUBSTRATE_HIGH_PRIORITY"
        print(f"  v2 at chance ({top1_mean:.3f}). Plasticity fix "
              f"didn't help. The ec_context substrate cannot "
              f"reliably do positional binding via either pure-"
              f"engram (v1) or learned-pathway (v2) mechanisms. "
              f"Direction E substrate (theta-gamma) is the next "
              f"decisive test; if it also fails, the substrate "
              f"itself is fundamentally limited for sequence "
              f"storage and requires substantive architectural "
              f"changes.", flush=True)
    else:
        recommendation = "MANUAL_DECISION_NEEDED"
        print(f"  v2 verdict unclear: {verdict}. Manual decision.",
              flush=True)

    print(f"\n  RECOMMENDED LAUNCH: {recommendation}", flush=True)
    print(f"\n  next commands:", flush=True)
    if "PILLAR_N104" in recommendation:
        print(f"  - Update capability_status.json with pillar n=104"
              f" (similar to n=103 process)", flush=True)
        print(f"  - Launch capacity sweep: python -m research."
              f"findings.raw.direction_A_capacity_sweep", flush=True)
    if "DIRECTION_E" in recommendation:
        print(f"  - Direction E substrate Task 0 grounding: python "
              f"-m research.findings.raw.direction_E_substrate_"
              f"task0_grounding (~5 min GPU)", flush=True)
        print(f"  - Direction E substrate Task 1 FULL: python -m "
              f"research.findings.raw.direction_E_substrate_task1_"
              f"full (~3 hr GPU; kill-safe per-seed)", flush=True)

    out = {
        "v1_top3_mean": v1_top3, "v1_top1_mean": v1_top1,
        "v2_top3_mean": top3_mean, "v2_top1_mean": top1_mean,
        "v2_per_seed_top1": top1_per_seed,
        "v2_verdict": verdict,
        "next_direction_recommendation": recommendation,
    }
    out_p = os.path.join(_HERE, "direction_A_post_v2_chain.json")
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_p}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
