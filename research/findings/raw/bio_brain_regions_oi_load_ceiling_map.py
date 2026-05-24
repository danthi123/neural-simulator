"""Load-ceiling map for the bio_brain_regions substrate variants
(OPTION 3 / HIPPO-OPTION3 / DLPFC-extension) at loads {2..7}.

Direct extension of pillars n=96/n=97/n=98 -- those tested at loads
{2, 3, 5}. This probe sharpens the load characterisation across the
full L=2..7 range for each substrate variant.

Analogous to the cross-bridge OI load-ceiling map (2026-05-24)
extending pillar n=95. CPU-only; reuses every substrate's cached
trained activity + parallel-matching primitives byte-unchanged.

PRE-REGISTERED reading (descriptive only):
- Per-substrate per-load OB and OI multi-seed-mean across loads
  {2..7}
- Highest-L-PASS + lowest-L-BELOW reported for each substrate
- The pillars' verdicts stand; this provides the load-ceiling
  shape

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

from research.findings.raw.vocabulary_scaling_run import (
    BAR, SEEDS, N_DIM, N_TRIALS,
)
from research.findings.raw.biologized_spiking_mode_unification_parallel_matching_runner import (
    K_VOCAB_TARGET, DERIV_SEED,
)
from research.findings.raw.pattern_separation_grounding_probe import (
    make_deriver,
)
from research.findings.raw.biologized_spiking_mode_unification_helpers import (
    gamma_slot_positions,
)
from research.findings.raw.cross_bridge_mode_unification_probe import (
    batched_phase_similarity, verify_batched_equivalent_to_scalar,
)
from research.findings.raw.mode_unification_on_bio_brain_regions_probe import (
    _load_activity_cache,
)
from research.runners.resonate_fire_fhrr import ResonateFireFHRR
from research.runners.spiking_phasor_fhrr import phases_to_spikes
from research.runners.concept_pool_demo import (
    DIRECTION_VOCAB, NOUN_VOCAB, VERB_VOCAB, ADJECTIVE_VOCAB,
)
from sim.backend import get_backend, to_host

EXTENDED_LOADS = [2, 3, 4, 5, 6, 7]
N_GAMMA_SLOTS = 7
OUT_JSON = os.path.join(
    _HERE, "bio_brain_regions_oi_load_ceiling_map.json")

# Three substrate variants and their activity cache directories.
SUBSTRATES = {
    "OPTION3_no_hippo": os.path.join(
        _HERE, "mode_unification_on_bio_brain_regions_cache"),
    "HIPPO_OPTION3": os.path.join(
        _HERE, "mode_unification_with_hippo_cache"),
    "DLPFC_extension": os.path.join(
        _HERE, "mode_unification_with_hippo_dlpfc_cache"),
}

VOCAB_16 = (list(DIRECTION_VOCAB) + list(NOUN_VOCAB) +
            list(VERB_VOCAB) + list(ADJECTIVE_VOCAB))


def _ground_symbols_from_cache(cache_dir: str, seed: int):
    """Load activity cache + build grounded symbols via the validated
    mean-centred + DERIV_SEED deriver pipeline."""
    cache_p = os.path.join(cache_dir, f"activity_full_seed{seed}.npz")
    if not os.path.exists(cache_p):
        raise FileNotFoundError(f"cache missing: {cache_p}")
    acts = _load_activity_cache(cache_p, VOCAB_16)
    d_act = acts[VOCAB_16[0]].shape[1]
    consolidated = {w: acts[w][:K_VOCAB_TARGET].mean(axis=0)
                    for w in VOCAB_16}
    common = np.mean([consolidated[w] for w in VOCAB_16], axis=0)
    deriver = make_deriver(N_DIM, d_act, DERIV_SEED)
    grounded = {w: phases_to_spikes(deriver(consolidated[w] - common))
                for w in VOCAB_16}
    return grounded, d_act


def run_one_substrate_one_seed(substrate: str, seed: int, xp):
    cache_dir = SUBSTRATES[substrate]
    grounded, d_act = _ground_symbols_from_cache(cache_dir, seed)
    max_diff, vocab_phase_matrix = verify_batched_equivalent_to_scalar(
        grounded, VOCAB_16, xp, rng_seed=seed)
    positions = gamma_slot_positions(seed, N_GAMMA_SLOTS, N_DIM)
    net = ResonateFireFHRR(N_DIM, np.random.default_rng(seed))
    qrng = np.random.default_rng(seed + 1)

    per_load = {}
    V = len(VOCAB_16)
    for load in EXTENDED_LOADS:
        ob_ok = oi_ok = 0
        for _ in range(N_TRIALS):
            items_idx = tuple(int(i) for i in
                              qrng.choice(V, size=load, replace=False))
            items = [VOCAB_16[i] for i in items_idx]
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
            topK = sorted(int(i) for i in np.argsort(scores_oi_host)[-load:])
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
    print("=== bio_brain_regions OI load-ceiling map (extension of "
          "pillars n=96/n=97/n=98) ===", flush=True)
    print(f"  backend={backend_name}; loads={EXTENDED_LOADS}; "
          f"seeds={list(SEEDS)}; bar={BAR}; V={len(VOCAB_16)}",
          flush=True)
    print(f"  Substrates: {list(SUBSTRATES.keys())}", flush=True)

    results = {}
    t0 = time.time()
    for substrate in SUBSTRATES:
        print(f"\n--- substrate: {substrate} ---", flush=True)
        results[substrate] = []
        for seed in SEEDS:
            t_seed = time.time()
            per_load, V, max_diff = run_one_substrate_one_seed(
                substrate, seed, xp)
            results[substrate].append({
                "seed": seed, "V": V,
                "batched_vs_scalar_max_diff": max_diff,
                "per_load": {str(l): v for l, v in per_load.items()},
            })
            row = " ".join(f"L{l}:{per_load[l]['order_invariant_accuracy']:.3f}"
                           for l in EXTENDED_LOADS)
            print(f"  [{substrate}/{seed} V={V} diff={max_diff:.1e}] "
                  f"OI({row}) ({time.time()-t_seed:.1f}s)",
                  flush=True)
    print(f"\nTotal wall-clock: {time.time()-t0:.1f}s", flush=True)

    print(f"\n=== MULTI-SEED OI LOAD-CEILING MAP ===", flush=True)
    print("                  " + "   ".join(f"L={l}" for l in EXTENDED_LOADS),
          flush=True)
    agg = {}
    for substrate in SUBSTRATES:
        agg[substrate] = {}
        oi_row = []
        ob_row = []
        for load in EXTENDED_LOADS:
            obs = [r["per_load"][str(load)]["order_bearing_accuracy"]
                   for r in results[substrate]]
            ois = [r["per_load"][str(load)]["order_invariant_accuracy"]
                   for r in results[substrate]]
            ob_m = float(np.mean(obs)); oi_m = float(np.mean(ois))
            agg[substrate][load] = {
                "order_bearing_mean": ob_m,
                "order_invariant_mean": oi_m,
                "order_bearing_per_seed": obs,
                "order_invariant_per_seed": ois,
            }
            oi_row.append(f"{oi_m:.3f}")
            ob_row.append(f"{ob_m:.3f}")
        print(f"  {substrate:>18}: OB {' '.join(ob_row)}",
              flush=True)
        print(f"  {substrate:>18}: OI {' '.join(oi_row)}",
              flush=True)

    print(f"\n=== OI CEILING (descriptive) ===", flush=True)
    for substrate in SUBSTRATES:
        last_above = None; first_below = None
        for load in EXTENDED_LOADS:
            oi_m = agg[substrate][load]["order_invariant_mean"]
            if oi_m >= BAR:
                last_above = load
            elif first_below is None:
                first_below = load
        print(f"  {substrate}: highest L with OI >= 0.80 = {last_above}; "
              f"lowest L below = {first_below}", flush=True)

    out = {
        "backend": backend_name,
        "substrates": list(SUBSTRATES.keys()),
        "seeds": list(SEEDS), "loads": EXTENDED_LOADS, "bar": BAR,
        "n_gamma_slots": N_GAMMA_SLOTS, "V": len(VOCAB_16),
        "decoder_order_bearing": "parallel_population_matching_batched",
        "decoder_order_invariant": "marginal_sum_phase_similarity_batched",
        "per_substrate_per_seed": results,
        "per_substrate_aggregate": {
            s: {str(l): v for l, v in d.items()}
            for s, d in agg.items()},
        "note": ("Descriptive extension of pillars n=96/n=97/n=98 "
                  "across loads {2..7}; OB perfect every cell expected "
                  "(per pillar n=96/n=97/n=98 results at L=2/3/5); OI "
                  "ceiling shape characterised."),
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_JSON}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
