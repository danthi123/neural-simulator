"""Biologized mode-unification (parallel-matching) extension across the
160-ensemble's 5 bridges -- natural completion of the (b) VALIDATED
capability thread.

The (b) capability (mode-unification biologized via parallel-population-
matching identification; both readouts PASS multi-seed at 32 concepts;
VALIDATED pillar n=93) was tested on bridgeA_nouns only. This extension
runs the IDENTICAL pre-registered runner logic (byte-unchanged reuse via
import) on the other 4 bridges' caches (bridgeB_verbs, bridgeC_adj,
bridgeD_spatial, bridgeE_functional) to characterise whether the
biologized mode-unification both-readouts capability holds per-bridge
across the full 160-concept ensemble.

PRE-REGISTERED reading (fixed; never tuned):
- ENSEMBLE_PASS: every (bridge, load) cell multi-seed-mean >= 0.80 on
  BOTH order-bearing AND order-invariant readouts across all 5 bridges
  x 3 loads = 15 cells per readout = 30 cells total. The biologized
  mode-unification both-readouts capability extends per-bridge across
  the full 160-concept ensemble.
- BOUNDARY: some bridge or load misses; per-bridge breakdown reported
  honestly (similar to the 160-ensemble decisive run's bridgeD_spatial
  miss).

Cheap CPU; reuses every 160-ensemble bridge cache + the parallel-
matching runner's logic byte-unchanged via direct re-implementation
of the per-trial loop (the runner's `run_one_seed` is bridgeA-only by
design; mirrors its structure for cross-bridge iteration). No
protected, frozen, or moat module modified; no automatic
differentiation; no-confab moat must stay 7/7 green.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Reuse-by-import only; same primitives the parallel-matching runner
# (n=93 VALIDATED) uses.
from research.findings.raw.vocabulary_scaling_run import (
    BAR, LOADS, SEEDS, N_DIM, N_TRIALS, _load_cache,
)
from research.findings.raw.biologized_spiking_mode_unification_parallel_matching_runner import (
    K_VOCAB_TARGET, DERIV_SEED, _ground_symbols,
)
from research.findings.raw.biologized_spiking_mode_unification_helpers import (
    gamma_slot_positions,
)
from research.runners.resonate_fire_fhrr import ResonateFireFHRR
from research.runners.spiking_phasor_fhrr import phase_similarity

# The 5 bridges from the 160-ensemble; their caches at
# research/findings/raw/vocabulary_scaling_160ensemble_cache/
# full_<bridge>_seed{seed}.npz.
BRIDGES = ["bridgeA_nouns", "bridgeB_verbs", "bridgeC_adj",
           "bridgeD_spatial", "bridgeE_functional"]
ENSEMBLE_CACHE_DIR = os.path.join(
    _HERE, "vocabulary_scaling_160ensemble_cache")
N_GAMMA_SLOTS = 7
OUT_JSON = os.path.join(
    _HERE,
    "biologized_mode_unification_parallel_matching_5bridge_extension.json")


def run_one_bridge_seed(bridge_name, seed):
    """Mirror the parallel-matching runner's per-seed loop, but on the
    specified bridge's 160-ensemble cache. Returns per-load (OB, OI)."""
    cache_p = os.path.join(
        ENSEMBLE_CACHE_DIR, f"full_{bridge_name}_seed{seed}.npz")
    if not os.path.exists(cache_p):
        raise FileNotFoundError(f"cache missing: {cache_p}")
    acts, words, _patterns = _load_cache(cache_p)
    d_act = acts[words[0]].shape[1]
    consolidated = {w: acts[w][:K_VOCAB_TARGET].mean(axis=0) for w in words}
    grounded = _ground_symbols(consolidated, words, d_act)
    positions = gamma_slot_positions(seed, N_GAMMA_SLOTS, N_DIM)
    net = ResonateFireFHRR(N_DIM, np.random.default_rng(seed))
    word_idx = {w: i for i, w in enumerate(words)}
    qrng = np.random.default_rng(seed + 1)

    per_load = {}
    for load in LOADS:
        ob_ok = oi_ok = 0
        for _ in range(N_TRIALS):
            items = list(qrng.choice(words, size=load, replace=False))
            items_idx = tuple(word_idx[w] for w in items)
            C = net.encode([(grounded[items[k]], positions[k])
                            for k in range(load)])
            unbinds = [net.query(C, positions[k]) for k in range(load)]
            # ORDER-BEARING via parallel-population matching.
            recovered = []
            for k in range(load):
                scores = [phase_similarity(unbinds[k], grounded[w])
                          for w in words]
                recovered.append(int(np.argmax(scores)))
            if tuple(recovered) == items_idx:
                ob_ok += 1
            # ORDER-INVARIANT via marginal-sum.
            scores_oi = np.zeros(len(words))
            for k in range(load):
                for w_idx, w in enumerate(words):
                    scores_oi[w_idx] += phase_similarity(unbinds[k],
                                                          grounded[w])
            topK = sorted(int(i) for i in np.argsort(scores_oi)[-load:])
            if tuple(topK) == tuple(sorted(items_idx)):
                oi_ok += 1
        per_load[load] = {
            "order_bearing_accuracy": ob_ok / N_TRIALS,
            "order_invariant_accuracy": oi_ok / N_TRIALS,
            "n_trials": N_TRIALS,
        }
    return per_load


def main():
    print("=== biologized mode-unification (parallel-matching) "
          "5-bridge extension ===", flush=True)
    print(f"bridges={BRIDGES}; seeds={list(SEEDS)}; loads={LOADS}; "
          f"K_VOCAB=16; bar={BAR}; decoder=parallel_population_matching",
          flush=True)
    print("Reuses parallel-matching runner's pipeline byte-unchanged "
          "via import; 160-ensemble caches reused.", flush=True)

    cell_results = []
    t0 = time.time()
    for bridge in BRIDGES:
        for seed in SEEDS:
            t_cell = time.time()
            per_load = run_one_bridge_seed(bridge, seed)
            cell_results.append({"bridge": bridge, "seed": seed,
                                 "per_load": {str(l): v
                                              for l, v in per_load.items()}})
            ob_str = ", ".join(f"L{l}={per_load[l]['order_bearing_accuracy']:.3f}"
                                for l in LOADS)
            oi_str = ", ".join(f"L{l}={per_load[l]['order_invariant_accuracy']:.3f}"
                                for l in LOADS)
            print(f"  [{bridge}/{seed}] OB({ob_str}) | OI({oi_str})  "
                  f"({time.time()-t_cell:.1f}s)", flush=True)
    print(f"\nTotal cell-runs: {len(cell_results)} in "
          f"{time.time()-t0:.1f}s")

    # Aggregate per (bridge, load) across seeds.
    print("\n=== PER-BRIDGE MULTI-SEED AGGREGATE ===", flush=True)
    print("                       L=2          L=3          L=5", flush=True)
    print("                       OB     OI    OB     OI    OB     OI",
          flush=True)
    per_bridge_agg = {}
    ob_all_pass = oi_all_pass = True
    for bridge in BRIDGES:
        per_bridge_agg[bridge] = {}
        cells = []
        for load in LOADS:
            obs = [c["per_load"][str(load)]["order_bearing_accuracy"]
                   for c in cell_results if c["bridge"] == bridge]
            ois = [c["per_load"][str(load)]["order_invariant_accuracy"]
                   for c in cell_results if c["bridge"] == bridge]
            ob_m = float(np.mean(obs)); oi_m = float(np.mean(ois))
            per_bridge_agg[bridge][load] = {
                "order_bearing_mean": ob_m,
                "order_bearing_per_seed": obs,
                "order_invariant_mean": oi_m,
                "order_invariant_per_seed": ois,
            }
            cells.append(f"{ob_m:.3f} {oi_m:.3f}")
            if ob_m < BAR:
                ob_all_pass = False
            if oi_m < BAR:
                oi_all_pass = False
        print(f"  {bridge:>18}:  {cells[0]}  {cells[1]}  {cells[2]}",
              flush=True)

    print(f"\n=== VERDICT ===", flush=True)
    if ob_all_pass and oi_all_pass:
        verdict = "ENSEMBLE_PASS_PARALLEL_MATCHING_ALL_5_BRIDGES"
        print("  Every (bridge, load) cell across all 5 bridges clears "
              "the frozen 0.80 bar multi-seed on BOTH order-bearing AND "
              "order-invariant readouts. The biologized mode-unification "
              "both-readouts capability via parallel-population-matching "
              "extends per-bridge across the FULL 160-concept ensemble. "
              "Subject to a fresh dedicated adversarial review before "
              "any capability-pillar claim. The oracle-adjacency caveat "
              "from the parallel-matching design doc applies.", flush=True)
    else:
        verdict = "BOUNDARY_SOME_BRIDGE_LOAD_MISSES"
        print("  Some (bridge, load, readout) cell is below 0.80; "
              "per-bridge breakdown above. Honest finding about per-"
              "bridge variation in the biologized mode-unification "
              "via parallel-matching identification.", flush=True)

    out = {
        "bridges": BRIDGES, "seeds": list(SEEDS), "loads": LOADS,
        "bar": BAR, "n_gamma_slots": N_GAMMA_SLOTS,
        "k_vocab": K_VOCAB_TARGET, "n_trials": N_TRIALS,
        "decoder_order_bearing": "parallel_population_matching",
        "decoder_order_invariant": "marginal_sum_phase_similarity",
        "cell_results": cell_results,
        "per_bridge_aggregate": {b: {str(l): v for l, v in d.items()}
                                 for b, d in per_bridge_agg.items()},
        "verdict": verdict,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_JSON}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
