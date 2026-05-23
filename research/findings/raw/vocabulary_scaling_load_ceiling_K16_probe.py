"""Cheap extended load-ceiling map at K_VOCAB=16 on the trained
activity cache.

The K_VOCAB sweep result established that at K_VOCAB=16 (the cache
maximum) the activity-grounded biologized pipeline clears the frozen
0.80 bar at loads {2, 3, 5} multi-seed (refined CAPABILITY PASS,
adversarially reviewed CLEAR, with a thin L=5 margin). This probe
extends that finding by running the SAME pipeline at K_VOCAB=16 on
the wider load ladder {2, 3, 4, 5, 6, 7} -- fully mapping how far the
noise-averaged activity-grounded path extends past load 5.

Sanity contract: loads {2, 3, 5} at K=16 must reproduce the K_VOCAB
sweep's K=16 result for those loads BYTE-FOR-BYTE (multi-seed means
0.9325 / 0.9244 / 0.8640). The new map points are L=4, L=6, L=7.

PRE-REGISTERED reading (fixed; never tuned):
- BOTH L=6 AND L=7 clear the 0.80 bar multi-seed: the noise-averaged
  activity-grounded pipeline extends meaningfully past load 5. A
  strong refined characterisation; the K=16 ceiling sits above L=7
  on this substrate.
- L=6 clears, L=7 misses: the K=16 ceiling sits between loads 6 and
  7. Crisp map.
- L=6 misses: the K=16 ceiling sits between loads 5 and 6 (the K=16
  L=5 result already cleared). Crisp map; the refined-PASS extends
  only one load past the original boundary.
- Any sanity mismatch on loads {2, 3, 5}: the cache or pipeline has
  drifted; investigate before propagating any new claim.

The frozen 0.80 bar is unchanged and is NOT being moved. This is a
finer-grained CHARACTERISATION of the K=16 PASS at higher loads, not
a re-test at a moved bar.

Pure CPU; reads only the recorded activity cache; reuses
`vocabulary_scaling_run.run_pipeline` byte-unchanged at K_VOCAB=16.
No GPU; no re-train; no new capture. Plain ASCII.
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
    BAR, SEEDS, K_RECOG, N_TRIALS,
    run_pipeline, _load_cache,
)

K_VOCAB_TARGET = 16
LOADS_EXT = [2, 3, 4, 5, 6, 7]
# Sanity loads (must match the K_VOCAB sweep result byte-for-byte).
LOADS_SANITY = [2, 3, 5]
SANITY_EXPECTED = {2: 0.9325, 3: 0.9244, 5: 0.8640}   # from K_VOCAB sweep

CACHE_DIR = os.path.join(_HERE, "vocabulary_scaling_trained_cache")
SWEEP_JSON = os.path.join(
    _HERE, "vocabulary_scaling_kvocab_sweep_probe.json")
OUT_JSON = os.path.join(
    _HERE, "vocabulary_scaling_load_ceiling_K16_probe.json")


def main():
    print("=== K=16 extended load-ceiling map ===")
    print(f"cheap CPU; reads the existing trained activity cache; "
          f"frozen bar {BAR} (not moved)")
    print(f"K_VOCAB fixed at {K_VOCAB_TARGET}; K_RECOG fixed at "
          f"{K_RECOG}; loads {LOADS_EXT}; seeds {list(SEEDS)}")

    per_seed = {}
    for seed in SEEDS:
        path = os.path.join(CACHE_DIR, f"trained_full_seed{seed}.npz")
        if not os.path.exists(path):
            print(f"  seed {seed}: cache absent -- skipping")
            continue
        acts, words, _patterns = _load_cache(path)
        print(f"\n--- seed {seed} ({len(words)} concepts, "
              f"{acts[words[0]].shape[0]} obs/concept) ---")
        per_load = run_pipeline(seed, acts, words, LOADS_EXT,
                                N_TRIALS, K_RECOG, K_VOCAB_TARGET)
        per_seed[seed] = {int(load): {
            "integrated_accuracy":
                float(per_load[load]["integrated_accuracy"]),
            "composition_only_accuracy":
                float(per_load[load]["composition_only_accuracy"]),
        } for load in LOADS_EXT}
        for load in LOADS_EXT:
            e = per_seed[seed][load]
            print(f"  L={load}: int={e['integrated_accuracy']:.4f}  "
                  f"comp-only={e['composition_only_accuracy']:.4f}")

    # --- Aggregate ----------------------------------------------------
    agg = {}
    for load in LOADS_EXT:
        ints = [per_seed[s][load]["integrated_accuracy"]
                for s in SEEDS if s in per_seed]
        m = float(np.mean(ints)) if ints else float("nan")
        agg[load] = {"mean_integrated": m, "per_seed_integrated": ints}

    # --- Sanity vs K_VOCAB sweep --------------------------------------
    print("\n=== SANITY (loads {2,3,5} at K=16 vs K_VOCAB sweep) ===")
    sanity_ok = True
    for load in LOADS_SANITY:
        m = agg[load]["mean_integrated"]
        exp = SANITY_EXPECTED[load]
        match = abs(m - exp) < 1e-4
        tag = "ok" if match else "MISMATCH"
        print(f"  L={load}: K=16 mean={m:.4f} vs sweep "
              f"{exp:.4f}  [{tag}]")
        if not match:
            sanity_ok = False

    # --- Extended map -------------------------------------------------
    print("\n=== EXTENDED LOAD-CEILING MAP AT K_VOCAB=16 ===")
    last_above = None
    first_miss = None
    for load in LOADS_EXT:
        m = agg[load]["mean_integrated"]
        marker = "PASS" if m >= BAR else "miss"
        print(f"  L={load}: per-seed="
              f"{['%.4f' % a for a in agg[load]['per_seed_integrated']]} "
              f"mean={m:.4f}  ({'>=' if m >= BAR else '<'} {BAR})  "
              f"{marker}")
        if m >= BAR:
            last_above = load
        elif first_miss is None:
            first_miss = load

    print(f"\nceiling at K=16: highest load with multi-seed mean >= "
          f"{BAR} is L={last_above}; lowest load with mean < {BAR} is "
          f"L={first_miss}")

    out = {
        "frozen_bar": BAR,
        "k_vocab": K_VOCAB_TARGET,
        "k_recog": K_RECOG,
        "n_trials": N_TRIALS,
        "seeds": list(SEEDS),
        "loads_extended": LOADS_EXT,
        "loads_sanity": LOADS_SANITY,
        "sanity_expected_from_sweep": SANITY_EXPECTED,
        "per_seed": {str(s): {str(load): v for load, v in d.items()}
                     for s, d in per_seed.items()},
        "aggregate": {str(load): v for load, v in agg.items()},
        "sanity_matches_sweep": bool(sanity_ok),
        "ceiling_highest_pass_load": last_above,
        "ceiling_first_miss_load": first_miss,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_JSON}")
    return 0 if sanity_ok else 1


if __name__ == "__main__":
    sys.exit(main())
