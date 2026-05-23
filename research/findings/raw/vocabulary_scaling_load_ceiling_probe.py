"""Cheap load-ceiling characterisation for the trained-substrate
vocabulary-scaling decisive run.

The trained-substrate decisive run cleared the frozen 0.80 compositional
bar at loads {2, 3} multi-seed (means 0.842, 0.814) but missed at load
5 (0.756 < 0.80, by 0.044). The pre-registered routing's a-priori
premise (that a BELOW-BAR result here would mean the substrate was
still too sparse) is contradicted by the data: the trained substrate
density is 0.10 (above the validated benchmark's 0.075), recognition is
a clean 1.000 temporally-averaged, and the failure mode is a load
ceiling at higher binding loads. This probe maps that ceiling precisely.

It re-runs the SAME biologized grounded-composition pipeline (imported
byte-unchanged) on the EXISTING per-seed trained activity cache at
loads {2, 3, 4, 5, 6, 7}. No GPU run, no re-train, no new capture --
the cache is reused. The pipeline at loads {2, 3, 5} with the same args
as the decisive run is reproduced as a sanity check; the extended loads
{4, 6, 7} are the new map points.

This is a finer-grained CHARACTERISATION of the BOUNDARY result, NOT a
re-test of the same load set; the frozen 0.80 bar is unchanged and is
NOT being moved. It informs the design of the next pre-registered step
(grounding the symbol in the K-of-N pattern itself -- candidate 2 --
sharpened by knowing where the spiking-symbol noise floor's load
ceiling actually sits).

PRE-REGISTERED reading (fixed; never tuned):
- The sanity loads {2, 3, 5} re-runs reproduce the decisive run's
  multi-seed means within the seeded RNG path (the re-run uses an
  identical-args call so the per-seed values must match
  byte-for-byte).
- The extended loads {4, 6, 7} characterise where the multi-seed mean
  crosses the 0.80 bar. Reported as "highest load with mean >= 0.80"
  (the load ceiling) and "lowest load with mean < 0.80" (the first
  miss). No verdict change relative to the decisive run.

Pure CPU; reads only the recorded activity cache; reuses
`vocabulary_scaling_run.run_pipeline` byte-unchanged. No GPU. Plain
ASCII.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from research.findings.raw.vocabulary_scaling_run import (
    BAR, N_TRIALS, K_RECOG, K_VOCAB, SEEDS,
    run_pipeline, _load_cache,
)

# Extended load ladder for the characterisation.
LOADS_EXT = [2, 3, 4, 5, 6, 7]
# The decisive run's loads (for the byte-for-byte sanity reproduction).
LOADS_DECISIVE = [2, 3, 5]

CACHE_DIR = os.path.join(_HERE, "vocabulary_scaling_trained_cache")
RESULT_JSON = os.path.join(_HERE,
                           "vocabulary_scaling_run_trained_full.json")
OUT_JSON = os.path.join(_HERE,
                        "vocabulary_scaling_load_ceiling_probe.json")


def _per_load_dict(loads, per_load):
    """Pull integrated + composition-only accuracy out of run_pipeline's
    return shape, keyed by int load."""
    return {load: {
        "integrated_accuracy": float(per_load[load]["integrated_accuracy"]),
        "composition_only_accuracy":
            float(per_load[load]["composition_only_accuracy"]),
        "n_composition_only": int(per_load[load]["n_composition_only"]),
        "effective_load": int(per_load[load]["effective_load"]),
    } for load in loads}


def main():
    print("=== vocabulary-scaling load-ceiling characterisation ===")
    print(f"cheap CPU re-run on the EXISTING trained activity cache; "
          f"frozen bar {BAR} (not moved)")
    print(f"sanity loads {LOADS_DECISIVE} (reproduce the decisive run "
          f"byte-for-byte); extended loads {LOADS_EXT}")

    decisive = None
    if os.path.exists(RESULT_JSON):
        with open(RESULT_JSON, "r", encoding="utf-8") as f:
            decisive = json.load(f)

    seeds = list(SEEDS)
    per_seed_sanity = {}
    per_seed_extended = {}

    for seed in seeds:
        path = os.path.join(CACHE_DIR, f"trained_full_seed{seed}.npz")
        if not os.path.exists(path):
            print(f"  seed {seed}: cache absent -- skipping")
            continue
        acts, words, _patterns = _load_cache(path)
        print(f"\n--- seed {seed} ({len(words)} concepts, "
              f"{acts[words[0]].shape[0]} obs/concept) ---")

        # Sanity reproduction: same args as the decisive run.
        per_load_s = run_pipeline(seed, acts, words, LOADS_DECISIVE,
                                  N_TRIALS, K_RECOG, K_VOCAB)
        per_seed_sanity[seed] = _per_load_dict(LOADS_DECISIVE, per_load_s)
        print(f"  SANITY (loads {LOADS_DECISIVE}):")
        for load in LOADS_DECISIVE:
            e = per_seed_sanity[seed][load]
            print(f"    L={load}: int={e['integrated_accuracy']:.4f}  "
                  f"comp-only={e['composition_only_accuracy']:.4f}")

        # Extended ladder.
        per_load_e = run_pipeline(seed, acts, words, LOADS_EXT,
                                  N_TRIALS, K_RECOG, K_VOCAB)
        per_seed_extended[seed] = _per_load_dict(LOADS_EXT, per_load_e)
        print(f"  EXTENDED (loads {LOADS_EXT}):")
        for load in LOADS_EXT:
            e = per_seed_extended[seed][load]
            print(f"    L={load}: int={e['integrated_accuracy']:.4f}  "
                  f"comp-only={e['composition_only_accuracy']:.4f}")

    # --- Aggregates ----------------------------------------------------
    def aggregate(per_seed, loads):
        agg = {}
        for load in loads:
            ints = [per_seed[s][load]["integrated_accuracy"]
                    for s in per_seed]
            comps = [per_seed[s][load]["composition_only_accuracy"]
                     for s in per_seed]
            agg[load] = {
                "mean_integrated": float(np.mean(ints)) if ints else float("nan"),
                "per_seed_integrated": ints,
                "mean_composition_only":
                    float(np.mean(comps)) if comps else float("nan"),
            }
        return agg

    sanity_agg = aggregate(per_seed_sanity, LOADS_DECISIVE)
    ext_agg = aggregate(per_seed_extended, LOADS_EXT)

    # --- Sanity check vs the decisive recording -----------------------
    print("\n=== SANITY (re-runs vs decisive recording) ===")
    sanity_ok = True
    if decisive is not None:
        for load in LOADS_DECISIVE:
            sr = sanity_agg[load]
            dec_agg = decisive["aggregate"].get(str(load), {})
            dec_mean = float(dec_agg.get("mean_integrated", float("nan")))
            dec_per_seed = list(dec_agg.get("per_seed_integrated", []))
            match_mean = abs(sr["mean_integrated"] - dec_mean) < 1e-6
            match_per_seed = all(
                abs(sr["per_seed_integrated"][i] - dec_per_seed[i]) < 1e-6
                for i in range(min(len(sr["per_seed_integrated"]),
                                    len(dec_per_seed))))
            tag = "ok" if (match_mean and match_per_seed) else "MISMATCH"
            print(f"  L={load}: re-run mean={sr['mean_integrated']:.4f} "
                  f"vs decisive {dec_mean:.4f}  [{tag}]")
            if not (match_mean and match_per_seed):
                sanity_ok = False
    else:
        print("  decisive JSON absent -- cannot cross-check the sanity run")
        sanity_ok = False

    # --- Extended ceiling map -----------------------------------------
    print("\n=== EXTENDED LOAD-CEILING MAP ===")
    print("                  multi-seed integrated mean")
    crosses_at = None
    last_above = None
    for load in LOADS_EXT:
        m = ext_agg[load]["mean_integrated"]
        bar_status = ">=" if m >= BAR else "<"
        marker = "  PASS" if m >= BAR else "  miss"
        print(f"  L={load}: per-seed="
              f"{['%.4f' % a for a in ext_agg[load]['per_seed_integrated']]} "
              f"mean={m:.4f}  ({bar_status} {BAR}){marker}")
        if m >= BAR:
            last_above = load
        elif crosses_at is None:
            crosses_at = load

    print(f"\nceiling: highest load with multi-seed mean >= {BAR} "
          f"is L={last_above}; lowest load with mean < {BAR} is "
          f"L={crosses_at}")

    out = {
        "frozen_bar": BAR,
        "seeds": seeds,
        "loads_decisive": LOADS_DECISIVE,
        "loads_extended": LOADS_EXT,
        "n_trials": N_TRIALS, "k_recog": K_RECOG, "k_vocab": K_VOCAB,
        "per_seed_sanity": {str(s): {str(l): v for l, v in d.items()}
                            for s, d in per_seed_sanity.items()},
        "per_seed_extended": {str(s): {str(l): v for l, v in d.items()}
                              for s, d in per_seed_extended.items()},
        "aggregate_sanity": {str(l): v for l, v in sanity_agg.items()},
        "aggregate_extended": {str(l): v for l, v in ext_agg.items()},
        "sanity_reproduces_decisive": bool(sanity_ok),
        "ceiling_highest_pass_load": last_above,
        "ceiling_first_miss_load": crosses_at,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_JSON}")
    return 0 if sanity_ok else 1


if __name__ == "__main__":
    sys.exit(main())
