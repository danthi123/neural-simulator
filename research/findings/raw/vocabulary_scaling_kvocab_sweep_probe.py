"""Cheap K_VOCAB sweep on the activity-grounded pipeline.

The pattern-grounded NEGATIVE plus its built-in geometry diagnostic
established that the compositional algebra's load-bearing requirement
is mean-centered (signed) symbols, that the activity-grounded path
satisfies that requirement naturally via mean-centring, and that the
gap from the activity-grounded L=5 ceiling (0.756 multi-seed) to the
geometry-clean reference (~1.000) is the residual spiking-symbol noise
on top of the right geometry. This probe tests how far averaging more
observations per concept before deriving the symbol pushes the
activity-grounded ceiling toward the geometry-clean reference.

The trained activity cache has M_OBS=16 observations per concept; the
current pipeline uses K_VOCAB=8 for the consolidated symbol. This
probe sweeps K_VOCAB over {1, 2, 4, 8, 16} on the existing cache --
the natural log2 sweep up to the cache's maximum (no oracle-adjacency
added; K_VOCAB=16 just uses all available observations).

PRE-REGISTERED reading (fixed; never tuned):
- Sanity: K_VOCAB=8 reproduces the trained-substrate decisive run's
  multi-seed means BYTE-FOR-BYTE -- 0.8417 / 0.8139 / 0.7560 at loads
  {2, 3, 5}. Any deviation here breaks the sanity contract.
- Noise-bounded interpretation: if K_VOCAB=16 lifts the L=5 multi-
  seed mean substantially toward 1.000 (well above the current
  0.756), the L=5 ceiling is residual spiking-symbol noise that more
  observations close -- biology-translatable: longer temporal
  integration in cortex closes the noise gap.
- Structural-crosstalk interpretation: if K_VOCAB=16 does NOT lift it
  meaningfully, the ceiling has a structural component the noise-
  averaging cannot remove -- sharpens the diagnosis further.

Pure CPU; reads only the recorded activity cache and the runner's
pipeline by import; no GPU, no re-train, no new capture. Plain ASCII.
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
    BAR, LOADS, SEEDS, K_RECOG, N_TRIALS,
    run_pipeline, _load_cache,
)

K_VOCAB_LADDER = [1, 2, 4, 8, 16]

CACHE_DIR = os.path.join(_HERE, "vocabulary_scaling_trained_cache")
DECISIVE_JSON = os.path.join(
    _HERE, "vocabulary_scaling_run_trained_full.json")
OUT_JSON = os.path.join(
    _HERE, "vocabulary_scaling_kvocab_sweep_probe.json")


def main():
    print("=== K_VOCAB sweep on the activity-grounded pipeline ===")
    print(f"cheap CPU; reads the existing trained activity cache; "
          f"frozen bar {BAR} (not moved)")
    print(f"K_VOCAB ladder: {K_VOCAB_LADDER}; K_RECOG fixed at "
          f"{K_RECOG}; loads {LOADS}; seeds {SEEDS}")

    decisive = None
    if os.path.exists(DECISIVE_JSON):
        with open(DECISIVE_JSON, "r", encoding="utf-8") as f:
            decisive = json.load(f)

    per_seed_per_k = {seed: {} for seed in SEEDS}
    for seed in SEEDS:
        path = os.path.join(CACHE_DIR, f"trained_full_seed{seed}.npz")
        if not os.path.exists(path):
            print(f"  seed {seed}: cache absent -- skipping")
            continue
        acts, words, _patterns = _load_cache(path)
        print(f"\n--- seed {seed} ({len(words)} concepts, "
              f"{acts[words[0]].shape[0]} obs/concept) ---")
        for k in K_VOCAB_LADDER:
            per_load = run_pipeline(seed, acts, words, LOADS, N_TRIALS,
                                    K_RECOG, k)
            per_seed_per_k[seed][k] = {
                int(load): {
                    "integrated_accuracy":
                        float(per_load[load]["integrated_accuracy"]),
                    "composition_only_accuracy":
                        float(per_load[load]["composition_only_accuracy"]),
                } for load in LOADS}
            ints = [per_seed_per_k[seed][k][load]["integrated_accuracy"]
                    for load in LOADS]
            print(f"  K_VOCAB={k:>2}: " +
                  "  ".join(f"L={load} int={ints[i]:.4f}"
                            for i, load in enumerate(LOADS)))

    # --- Aggregate ----------------------------------------------------
    print("\n=== MULTI-SEED AGGREGATE (integrated mean) ===")
    print("           " + "  ".join(f"L={load}" for load in LOADS))
    agg = {}
    for k in K_VOCAB_LADDER:
        agg[k] = {}
        cells = []
        for load in LOADS:
            ints = [per_seed_per_k[seed][k][load]["integrated_accuracy"]
                    for seed in SEEDS if k in per_seed_per_k[seed]]
            m = float(np.mean(ints)) if ints else float("nan")
            agg[k][load] = {"mean_integrated": m,
                            "per_seed_integrated": ints}
            mark = ">=" if m >= BAR else "<"
            cells.append(f"{m:.4f} {mark}{BAR}")
        print(f"  K_VOCAB={k:>2}:  " + "  ".join(cells))

    # --- Sanity vs the decisive recording (K_VOCAB=8) -----------------
    print("\n=== SANITY (K_VOCAB=8 vs decisive recording) ===")
    sanity_ok = True
    if decisive is not None and 8 in agg:
        for load in LOADS:
            rc = float(agg[8][load]["mean_integrated"])
            dec = float(decisive["aggregate"].get(str(load), {})
                                .get("mean_integrated", float("nan")))
            match = abs(rc - dec) < 1e-6
            tag = "ok" if match else "MISMATCH"
            print(f"  L={load}: K=8 mean={rc:.4f} vs decisive {dec:.4f} "
                  f"[{tag}]")
            if not match:
                sanity_ok = False
    else:
        print("  decisive JSON absent or K=8 not run -- cannot cross-check")
        sanity_ok = False

    # --- Pre-registered reading ---------------------------------------
    print("\n=== READING ===")
    if 16 in agg and 8 in agg:
        lift_l5 = agg[16][5]["mean_integrated"] - agg[8][5]["mean_integrated"]
        k16_l5 = agg[16][5]["mean_integrated"]
        k16_all_pass = all(agg[16][load]["mean_integrated"] >= BAR
                            for load in LOADS)
        if k16_all_pass:
            reading = "NOISE_BOUNDED_K16_CLEARS_BAR_AT_ALL_LOADS"
            print(f"  K_VOCAB=16 lifts L=5 from "
                  f"{agg[8][5]['mean_integrated']:.4f} to {k16_l5:.4f} "
                  f"(+{lift_l5:.4f}) and CLEARS the 0.80 bar at all "
                  f"loads. Noise-bounded interpretation strongly "
                  f"supported -- biology-translatable: longer temporal "
                  f"integration closes the noise gap. A refined PASS "
                  f"on the activity-grounded path with no oracle-"
                  f"adjacency added (K=16 uses all 16 cached "
                  f"observations -- no tuning, just the maximum the "
                  f"cache supports). Subject to a dedicated adversarial "
                  f"review on the K_VOCAB sweep tool before any "
                  f"capability claim.")
        elif lift_l5 > 0.05:
            reading = "NOISE_BOUNDED_K16_LIFTS_BUT_DOES_NOT_CLEAR"
            print(f"  K_VOCAB=16 lifts L=5 from "
                  f"{agg[8][5]['mean_integrated']:.4f} to {k16_l5:.4f} "
                  f"(+{lift_l5:.4f}) but does not clear the 0.80 bar. "
                  f"The ceiling is noise-bounded in part; more "
                  f"observations help; a structural component may "
                  f"remain.")
        else:
            reading = "STRUCTURAL_CROSSTALK_NOISE_AVERAGING_DOES_NOT_LIFT"
            print(f"  K_VOCAB=16 does not lift L=5 meaningfully "
                  f"({agg[8][5]['mean_integrated']:.4f} -> {k16_l5:.4f}, "
                  f"+{lift_l5:.4f}). The ceiling has a structural "
                  f"component the noise-averaging cannot remove -- "
                  f"sharpens the diagnosis further.")
    else:
        reading = "INCOMPLETE_RUN"
        print("  K=16 or K=8 missing -- reading inconclusive")

    out = {
        "frozen_bar": BAR,
        "seeds": list(SEEDS),
        "loads": LOADS,
        "k_vocab_ladder": K_VOCAB_LADDER,
        "k_recog_fixed": K_RECOG,
        "n_trials": N_TRIALS,
        "per_seed_per_k": {
            str(s): {str(k): {str(load): v for load, v in dk.items()}
                     for k, dk in dsk.items()}
            for s, dsk in per_seed_per_k.items() if dsk},
        "aggregate": {str(k): {str(load): v for load, v in dk.items()}
                      for k, dk in agg.items()},
        "sanity_reproduces_decisive_at_k8": bool(sanity_ok),
        "reading": reading,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_JSON}")
    return 0 if sanity_ok else 1


if __name__ == "__main__":
    sys.exit(main())
