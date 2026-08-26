"""6-SEED FLIP-SOAK for the spiking-SC orienting production organ (the default-ON flip gate).

The task's flip gate: "the orienting probe's 6-seed correct-cardinal stability". For each of 6 seeds this
builds the spiking-SC organ, runs the correct-cardinal battery (the orienting fidelity) INTACT and under
the SC_SCRAMBLE lesion, and runs the embodied FOVEATION loop INTACT and lesioned. It reports the per-seed
numbers and a GATE:

  FLIP-GATE (default-ON) passes iff, across ALL 6 seeds:
    * min INTACT correct-cardinal rate   >= 0.80   (the SC reliably reads the orienting bearing)
    * max LESION correct-cardinal rate   <= 0.45   (scrambled retinotopy collapses toward chance 0.25)
    * min INTACT embodied reach-rate     >= 0.80   (the body foveates the salient target)
    * max LESION embodied reach-rate     <= 0.50   (the lesioned SC random-walks)

This is the NO-REGRESSION / stability soak the parent runs before flipping BRAIN_SPIKING_SC_ORIENT
default-ON. It is the load-bearing evidence, not a headline: the lesion arm proves the read is carried by
the retinotopic spiking sheet at every seed, not a re-hidden host shortcut.

Run (CPU):
    SIM_BACKEND=numpy python -m research.runners._sc_orienting_flip_soak --seeds 42,43,44,45,46,47 \
        --out research/findings/raw/sc_orienting/flip_soak.json
"""
from __future__ import annotations

import os
os.environ.setdefault("SIM_BACKEND", "numpy")

import argparse
import json

from research.runners.sc_orienting_production_organ import get_organ
from research.runners._sc_orienting_production_organ_verify import (
    cardinal_battery, run_episode, EPISODES, LOG_POLAR,
)
from tools.lab import attributable_to

DEFAULT_SEEDS = [42, 43, 44, 45, 46, 47]

# flip-gate thresholds
MIN_INTACT_CARDINAL = 0.80
MAX_LESION_CARDINAL = 0.45
MIN_INTACT_REACH = 0.80
MAX_LESION_REACH = 0.50


def soak_one_seed(seed: int) -> dict:
    """Build the organ at `seed`, run the correct-cardinal battery + embodied foveation loop, INTACT and
    lesioned. Returns the four rates for this seed."""
    organ = get_organ(seed=seed, log_polar=LOG_POLAR)
    i_ok, i_tot = cardinal_battery(organ, lesion=False)
    l_ok, l_tot = cardinal_battery(organ, lesion=True)
    n = len(EPISODES)
    i_reach = sum(run_episode(organ, a, g, lesion=False)[0] for a, g in EPISODES) / n
    l_reach = sum(run_episode(organ, a, g, lesion=True)[0] for a, g in EPISODES) / n
    return {
        "seed": seed,
        "intact_cardinal": i_ok / max(i_tot, 1),
        "lesion_cardinal": l_ok / max(l_tot, 1),
        "intact_reach": float(i_reach),
        "lesion_reach": float(l_reach),
        "n_cardinal": i_tot, "n_episodes": n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS),
                    help="comma-separated seed list (default the 6-seed panel)")
    ap.add_argument("--out", default="research/findings/raw/sc_orienting/flip_soak.json")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    print("=" * 82)
    print("SPIKING-SC ORIENTING — 6-SEED FLIP-SOAK (the default-ON gate)")
    print("=" * 82)
    print(f"{'seed':>5} | {'INTACT card':>11} {'LESION card':>11} | {'INTACT reach':>12} {'LESION reach':>12}")
    rows = []
    for s in seeds:
        r = soak_one_seed(s)
        rows.append(r)
        print(f"{r['seed']:>5} | {r['intact_cardinal']:>11.3f} {r['lesion_cardinal']:>11.3f} | "
              f"{r['intact_reach']:>12.3f} {r['lesion_reach']:>12.3f}", flush=True)

    ic = [r["intact_cardinal"] for r in rows]
    lc = [r["lesion_cardinal"] for r in rows]
    ir = [r["intact_reach"] for r in rows]
    lr = [r["lesion_reach"] for r in rows]
    summary = {
        "seeds": seeds,
        "min_intact_cardinal": min(ic), "mean_intact_cardinal": sum(ic) / len(ic),
        "max_lesion_cardinal": max(lc), "mean_lesion_cardinal": sum(lc) / len(lc),
        "min_intact_reach": min(ir), "mean_intact_reach": sum(ir) / len(ir),
        "max_lesion_reach": max(lr), "mean_lesion_reach": sum(lr) / len(lr),
        "thresholds": {
            "min_intact_cardinal": MIN_INTACT_CARDINAL, "max_lesion_cardinal": MAX_LESION_CARDINAL,
            "min_intact_reach": MIN_INTACT_REACH, "max_lesion_reach": MAX_LESION_REACH,
        },
        "rows": rows,
    }
    gate = (summary["min_intact_cardinal"] >= MIN_INTACT_CARDINAL
            and summary["max_lesion_cardinal"] <= MAX_LESION_CARDINAL
            and summary["min_intact_reach"] >= MIN_INTACT_REACH
            and summary["max_lesion_reach"] <= MAX_LESION_REACH)
    summary["flip_gate"] = "GO" if gate else "NOT-YET"

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)

    print("-" * 82)
    print(f"INTACT correct-cardinal : min {summary['min_intact_cardinal']:.3f}  "
          f"mean {summary['mean_intact_cardinal']:.3f}   (gate: min >= {MIN_INTACT_CARDINAL})")
    print(f"LESION correct-cardinal : max {summary['max_lesion_cardinal']:.3f}  "
          f"mean {summary['mean_lesion_cardinal']:.3f}   (gate: max <= {MAX_LESION_CARDINAL}, chance 0.25)")
    print(f"INTACT embodied reach   : min {summary['min_intact_reach']:.3f}  "
          f"mean {summary['mean_intact_reach']:.3f}   (gate: min >= {MIN_INTACT_REACH})")
    print(f"LESION embodied reach   : max {summary['max_lesion_reach']:.3f}  "
          f"mean {summary['mean_lesion_reach']:.3f}   (gate: max <= {MAX_LESION_REACH})")
    # ATTRIBUTION: the load-bearing subtraction — what fraction of the 6-seed-mean orienting is owned by
    # the intact retinotopic coupling (INTACT) vs. survives the scrambled-retinotopy control (LESION).
    print("attribution (INTACT treatment vs SC_SCRAMBLE control, 6-seed means):")
    summary["attributable_cardinal"] = attributable_to(
        "correct-cardinal (mean)", summary["mean_intact_cardinal"], summary["mean_lesion_cardinal"])
    summary["attributable_reach"] = attributable_to(
        "embodied reach (mean)", summary["mean_intact_reach"], summary["mean_lesion_reach"])
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nFLIP-GATE (6-seed default-ON): {summary['flip_gate']}")
    print(f"wrote {args.out}")
    return 0 if gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
