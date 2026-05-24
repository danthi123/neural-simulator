"""Cross-bridge OI load-ceiling map extension to pillar n=95.

n=95 established CROSS_BRIDGE_BOUNDARY: OB perfect every cell;
OI ceilings at L=5 x V=160 (multi-seed-mean 0.79; just below the
0.80 bar). The original probe tested loads {2, 3, 5}. This extension
maps the OI ceiling precisely across loads {2, 3, 4, 5, 6, 7} on
the 160-concept union for both common-mode conditions.

CPU-only (SIM_BACKEND=numpy via the backend-aware batched runner --
or just doesn't matter since the orchestration is identical); reuses
the existing cross_bridge_mode_unification_probe.py runner's
primitives byte-unchanged; reuses the 160-ensemble caches; no GPU
substrate work; ~15-20 min CPU.

This is a CHARACTERISATION extension to the n=95 BOUNDARY pillar;
NOT a new capability claim. The OI load ceiling map sharpens the
biology-translatable insight about where the marginal-sum top-K
ORDER-INVARIANT mechanism crosses the substrate's grounded-symbol
noise floor.

PRE-REGISTERED reading (descriptive only):
- Map: per-load multi-seed-mean OI for both conditions
  (global_mean + per_bridge_mean) over loads {2,3,4,5,6,7}
- Honest: any load with OI multi-seed-mean < 0.80 is BELOW BAR;
  this is descriptive characterisation, not a new PASS/NEGATIVE
  verdict
- The n=95 pillar is unchanged; this provides the load-ceiling
  shape that the n=95 metric text refers to descriptively

Reuses every primitive byte-unchanged; no protected/frozen/moat
module modified; no autograd; no-confab moat must stay 7/7 green.
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

# Reuse the cross-bridge probe's grounding + batched primitive +
# verification helper byte-unchanged.
from research.findings.raw.cross_bridge_mode_unification_probe import (
    _global_ground_symbols, _per_bridge_ground_symbols,
    batched_phase_similarity, verify_batched_equivalent_to_scalar,
    BRIDGES, ENSEMBLE_CACHE_DIR, N_GAMMA_SLOTS,
)
from research.findings.raw.vocabulary_scaling_run import (
    BAR, SEEDS, N_DIM, N_TRIALS,
)
from research.findings.raw.biologized_spiking_mode_unification_helpers import (
    gamma_slot_positions,
)
from research.runners.resonate_fire_fhrr import ResonateFireFHRR
from sim.backend import get_backend, to_host

EXTENDED_LOADS = [2, 3, 4, 5, 6, 7]
OUT_JSON = os.path.join(
    _HERE, "cross_bridge_oi_load_ceiling_map.json")


def run_one_seed_one_condition(seed, condition, xp):
    if condition == "global_mean":
        all_words, grounded, d_act = _global_ground_symbols(seed)
    elif condition == "per_bridge_mean":
        all_words, grounded, d_act = _per_bridge_ground_symbols(seed)
    else:
        raise ValueError(f"unknown condition: {condition}")
    V = len(all_words)
    max_diff, vocab_phase_matrix = verify_batched_equivalent_to_scalar(
        grounded, all_words, xp, rng_seed=seed)
    positions = gamma_slot_positions(seed, N_GAMMA_SLOTS, N_DIM)
    net = ResonateFireFHRR(N_DIM, np.random.default_rng(seed))
    qrng = np.random.default_rng(seed + 1)

    per_load = {}
    for load in EXTENDED_LOADS:
        assert load <= N_GAMMA_SLOTS, (
            f"load {load} > gamma slots {N_GAMMA_SLOTS}")
        ob_ok = oi_ok = 0
        for _ in range(N_TRIALS):
            items_idx = tuple(int(i) for i in
                              qrng.choice(V, size=load, replace=False))
            items = [all_words[i] for i in items_idx]
            C = net.encode([(grounded[items[k]], positions[k])
                            for k in range(load)])
            unbinds = [net.query(C, positions[k]) for k in range(load)]
            recovered = []
            scores_oi_gpu = xp.zeros(V)
            for k in range(load):
                sims_k = batched_phase_similarity(
                    unbinds[k], vocab_phase_matrix, xp)
                recovered.append(int(xp.argmax(sims_k)))
                scores_oi_gpu = scores_oi_gpu + sims_k
            if tuple(recovered) == items_idx:
                ob_ok += 1
            scores_oi_host = to_host(scores_oi_gpu)
            topK = sorted(
                int(i) for i in np.argsort(scores_oi_host)[-load:])
            if tuple(topK) == tuple(sorted(items_idx)):
                oi_ok += 1
        per_load[load] = {
            "order_bearing_accuracy": ob_ok / N_TRIALS,
            "order_invariant_accuracy": oi_ok / N_TRIALS,
            "n_trials": N_TRIALS,
        }
    return per_load, V, max_diff


def main():
    xp, backend_name = get_backend()
    print("=== cross-bridge OI load-ceiling map (extension to n=95) ===",
          flush=True)
    print(f"  backend={backend_name}; loads={EXTENDED_LOADS}; "
          f"seeds={list(SEEDS)}; bar={BAR}; bridges={BRIDGES}",
          flush=True)
    print("  Sharpens the n=95 OI L=5 ceiling characterisation by "
          "mapping loads {2..7}; descriptive only -- the n=95 "
          "BOUNDARY pillar stands.", flush=True)

    results = {"global_mean": [], "per_bridge_mean": []}
    t0 = time.time()
    for condition in ("global_mean", "per_bridge_mean"):
        print(f"\n--- condition: {condition} ---", flush=True)
        for seed in SEEDS:
            t_seed = time.time()
            per_load, V, max_diff = run_one_seed_one_condition(
                seed, condition, xp)
            results[condition].append(
                {"seed": seed, "V": V,
                 "batched_vs_scalar_max_diff": max_diff,
                 "per_load": {str(l): v for l, v in per_load.items()}})
            row = " ".join(f"L{l}:{per_load[l]['order_invariant_accuracy']:.3f}"
                           for l in EXTENDED_LOADS)
            print(f"  [seed={seed} V={V}] OI({row})  "
                  f"({time.time()-t_seed:.1f}s)", flush=True)
    print(f"\nTotal wall-clock: {time.time()-t0:.1f}s",
          flush=True)

    print(f"\n=== MULTI-SEED OI CEILING MAP ===", flush=True)
    print("            " + "   ".join(f"L={l}" for l in EXTENDED_LOADS),
          flush=True)
    agg = {}
    for condition in ("global_mean", "per_bridge_mean"):
        agg[condition] = {}
        row_cells = []
        for load in EXTENDED_LOADS:
            ois = [r["per_load"][str(load)]["order_invariant_accuracy"]
                   for r in results[condition]]
            obs = [r["per_load"][str(load)]["order_bearing_accuracy"]
                   for r in results[condition]]
            ob_m = float(np.mean(obs)); oi_m = float(np.mean(ois))
            agg[condition][load] = {
                "order_bearing_mean": ob_m,
                "order_invariant_mean": oi_m,
                "order_bearing_per_seed": obs,
                "order_invariant_per_seed": ois,
            }
            row_cells.append(f"{oi_m:.3f}")
        print(f"  {condition:>16}:  {'  '.join(row_cells)}", flush=True)

    # Identify the L at which OI crosses 0.80 bar (descriptive).
    print(f"\n=== OI CEILING (descriptive) ===", flush=True)
    for condition in ("global_mean", "per_bridge_mean"):
        last_above = None; first_below = None
        for load in EXTENDED_LOADS:
            oi_m = agg[condition][load]["order_invariant_mean"]
            if oi_m >= BAR:
                last_above = load
            elif first_below is None:
                first_below = load
        print(f"  {condition}: highest L with OI >= 0.80 = {last_above}; "
              f"lowest L below = {first_below}", flush=True)

    out = {
        "backend": backend_name,
        "bridges": BRIDGES, "seeds": list(SEEDS),
        "loads": EXTENDED_LOADS, "bar": BAR,
        "n_gamma_slots": N_GAMMA_SLOTS,
        "decoder_order_bearing": "parallel_population_matching_batched",
        "decoder_order_invariant": "marginal_sum_phase_similarity_batched",
        "vocab_size": "union_of_5_bridges_32_concepts_each",
        "conditions": ["global_mean", "per_bridge_mean"],
        "per_condition_per_seed": results,
        "per_condition_aggregate": {
            c: {str(l): v for l, v in d.items()} for c, d in agg.items()},
        "note": ("Descriptive extension of pillar n=95; the BOUNDARY "
                  "pillar's framing and verdict stand; this maps the "
                  "OI load ceiling shape across {2..7}."),
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_JSON}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
