"""Cross-bridge biologized mode-unification probe -- OPTION 4 from the
(e) completion (autonomous parallel direction while awaiting owner
steer on (c) generative-replay architecture-integration choice).

Tests whether the parallel-matching biologized mode-unification
extends ACROSS bridge boundaries: encode a composite whose K items are
drawn UNIFORMLY from the union of all 5 bridges' 32-concept
vocabularies (160 concepts total), and decode per-slot via the
parallel-matching mechanism over the FULL 160-concept union. This is
the cross-bridge composition direction the 160-ensemble decisive run
explicitly bracketed.

The (e) extension validated per-bridge mode-unification (each bridge's
own 32-word vocabulary; pillar n=94). This probe asks the next
natural question: does the same algebra + same decoder + same
substrate handle composites that SPAN bridges (e.g. apple_noun +
go_verb + big_adj from three different bridges)?

CRITICAL TECHNICAL DECISION (recorded honestly): per-bridge
_ground_symbols subtracts each bridge's OWN common mode (mean across
that bridge's 32 concepts). For a fair cross-bridge geometry, we
re-mean-centre GLOBALLY across all 160 concepts and re-derive the
grounded symbols via the SAME fixed-seed deriver pipeline. This is
more biology-faithful (cortical pooled inhibition normalises across
the whole cortical extent, not per-region) and yields a uniform
phasor space across bridges.

PRE-REGISTERED reading (fixed; never tuned):
- CROSS_BRIDGE_PASS: multi-seed-mean >= the frozen 0.80 bar at every
  load {2, 3, 5} on BOTH order-bearing AND order-invariant readouts,
  with the 160-concept union vocabulary. Cross-bridge mode-unification
  extends the (e) per-bridge capability to bridge-spanning composites.
- CROSS_BRIDGE_BOUNDARY: either readout misses at some load; honest
  per-load breakdown reported. Biology-translatable: the parallel-
  matching mechanism's per-bridge capability does not automatically
  generalise to bridge-spanning vocabulary at this scale; characterise
  precisely.

If PASS: NOT YET a capability pillar -- pending fresh dedicated
adversarial review (matching the (b) and (e) standing discipline).
If NEGATIVE: honest characterisation finding, propagated.

CPU-only; reuses every 160-ensemble cache + parallel-matching
primitives byte-unchanged; no protected/frozen/moat module modified;
no autograd; no-confab moat must stay 7/7 green.
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

# Reuse-by-import only.
from research.findings.raw.vocabulary_scaling_run import (
    BAR, LOADS, SEEDS, N_DIM, N_TRIALS, _load_cache,
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
from research.runners.resonate_fire_fhrr import ResonateFireFHRR
from research.runners.spiking_phasor_fhrr import (
    phases_to_spikes, phase_similarity,
)

BRIDGES = ["bridgeA_nouns", "bridgeB_verbs", "bridgeC_adj",
           "bridgeD_spatial", "bridgeE_functional"]
ENSEMBLE_CACHE_DIR = os.path.join(
    _HERE, "vocabulary_scaling_160ensemble_cache")
N_GAMMA_SLOTS = 7
OUT_JSON = os.path.join(
    _HERE, "cross_bridge_mode_unification_probe.json")


def _global_ground_symbols(seed):
    """Load all 5 bridges' caches for this seed; build the 160-
    concept union; re-mean-centre GLOBALLY across all 160 concepts;
    derive grounded symbols via the SAME fixed-seed deriver
    (DERIV_SEED=90909). Returns (all_words, grounded, d_act) where
    all_words is a list of (bridge, word) tuples in deterministic
    order and grounded is a dict keyed on (bridge, word)."""
    consolidated = {}
    all_words = []
    d_act = None
    for bridge in BRIDGES:
        cache_p = os.path.join(
            ENSEMBLE_CACHE_DIR, f"full_{bridge}_seed{seed}.npz")
        if not os.path.exists(cache_p):
            raise FileNotFoundError(f"cache missing: {cache_p}")
        acts, words, _ = _load_cache(cache_p)
        if d_act is None:
            d_act = acts[words[0]].shape[1]
        elif acts[words[0]].shape[1] != d_act:
            raise ValueError(
                f"d_act mismatch: bridge {bridge} has "
                f"{acts[words[0]].shape[1]} vs expected {d_act}")
        for w in words:
            consolidated[(bridge, w)] = acts[w][:K_VOCAB_TARGET].mean(
                axis=0)
            all_words.append((bridge, w))
    # GLOBAL common-mode removal across all 160 concepts.
    common = np.mean([consolidated[bw] for bw in all_words], axis=0)
    deriver = make_deriver(N_DIM, d_act, DERIV_SEED)
    grounded = {bw: phases_to_spikes(deriver(consolidated[bw] - common))
                for bw in all_words}
    return all_words, grounded, d_act


def _per_bridge_ground_symbols(seed):
    """Comparison condition: per-bridge mean-centring (each bridge's
    own common mode subtracted independently, as the (e) extension
    does). Used as a control to characterise whether the global vs
    per-bridge mean-centring choice matters."""
    all_words = []
    grounded = {}
    d_act = None
    for bridge in BRIDGES:
        cache_p = os.path.join(
            ENSEMBLE_CACHE_DIR, f"full_{bridge}_seed{seed}.npz")
        acts, words, _ = _load_cache(cache_p)
        if d_act is None:
            d_act = acts[words[0]].shape[1]
        consolidated_b = {w: acts[w][:K_VOCAB_TARGET].mean(axis=0)
                          for w in words}
        common_b = np.mean([consolidated_b[w] for w in words], axis=0)
        deriver = make_deriver(N_DIM, d_act, DERIV_SEED)
        for w in words:
            grounded[(bridge, w)] = phases_to_spikes(
                deriver(consolidated_b[w] - common_b))
            all_words.append((bridge, w))
    return all_words, grounded, d_act


def run_one_seed_one_condition(seed, condition):
    """condition in {'global_mean', 'per_bridge_mean'}."""
    if condition == "global_mean":
        all_words, grounded, d_act = _global_ground_symbols(seed)
    elif condition == "per_bridge_mean":
        all_words, grounded, d_act = _per_bridge_ground_symbols(seed)
    else:
        raise ValueError(f"unknown condition: {condition}")
    V = len(all_words)
    positions = gamma_slot_positions(seed, N_GAMMA_SLOTS, N_DIM)
    net = ResonateFireFHRR(N_DIM, np.random.default_rng(seed))
    qrng = np.random.default_rng(seed + 1)

    per_load = {}
    for load in LOADS:
        ob_ok = oi_ok = 0
        for _ in range(N_TRIALS):
            items_idx = tuple(int(i) for i in
                              qrng.choice(V, size=load, replace=False))
            items = [all_words[i] for i in items_idx]
            C = net.encode([(grounded[items[k]], positions[k])
                            for k in range(load)])
            unbinds = [net.query(C, positions[k]) for k in range(load)]
            # ORDER-BEARING: per-slot argmax over the FULL 160-concept
            # union via parallel-population matching.
            recovered = []
            for k in range(load):
                scores = [phase_similarity(unbinds[k], grounded[bw])
                          for bw in all_words]
                recovered.append(int(np.argmax(scores)))
            if tuple(recovered) == items_idx:
                ob_ok += 1
            # ORDER-INVARIANT: marginal-sum across slots over 160-
            # concept union; top-K sorted by index.
            scores_oi = np.zeros(V)
            for k in range(load):
                for v, bw in enumerate(all_words):
                    scores_oi[v] += phase_similarity(unbinds[k],
                                                     grounded[bw])
            topK = sorted(int(i) for i in np.argsort(scores_oi)[-load:])
            if tuple(topK) == tuple(sorted(items_idx)):
                oi_ok += 1
        per_load[load] = {
            "order_bearing_accuracy": ob_ok / N_TRIALS,
            "order_invariant_accuracy": oi_ok / N_TRIALS,
            "n_trials": N_TRIALS,
        }
    return per_load, V


def main():
    print("=== cross-bridge biologized mode-unification probe "
          "(OPTION 4) ===", flush=True)
    print(f"seeds={list(SEEDS)}; loads={LOADS}; bridges={BRIDGES}; "
          f"K_VOCAB={K_VOCAB_TARGET}; bar={BAR}; "
          f"decoder=parallel_population_matching on 160-concept union",
          flush=True)
    print("Two conditions: global_mean (mean-centre across all 160) "
          "and per_bridge_mean (each bridge's own mean; the (e) "
          "extension's mean-centring choice).", flush=True)
    print("Reuses 160-ensemble caches + parallel-matching primitives "
          "byte-unchanged; CPU-only.", flush=True)

    results = {"global_mean": [], "per_bridge_mean": []}
    t0 = time.time()
    for condition in ("global_mean", "per_bridge_mean"):
        print(f"\n--- condition: {condition} ---", flush=True)
        for seed in SEEDS:
            t_seed = time.time()
            per_load, V = run_one_seed_one_condition(seed, condition)
            results[condition].append(
                {"seed": seed, "V": V,
                 "per_load": {str(l): v for l, v in per_load.items()}})
            ob_str = ", ".join(f"L{l}={per_load[l]['order_bearing_accuracy']:.3f}"
                                for l in LOADS)
            oi_str = ", ".join(f"L{l}={per_load[l]['order_invariant_accuracy']:.3f}"
                                for l in LOADS)
            print(f"  [seed={seed} V={V}] OB({ob_str}) | OI({oi_str})  "
                  f"({time.time()-t_seed:.1f}s)", flush=True)
    print(f"\nTotal wall-clock: {time.time()-t0:.1f}s", flush=True)

    # Per-condition multi-seed aggregate + verdict.
    print(f"\n=== MULTI-SEED AGGREGATE PER CONDITION ===", flush=True)
    agg = {}
    verdicts = {}
    for condition in ("global_mean", "per_bridge_mean"):
        agg[condition] = {}
        ob_all_pass = oi_all_pass = True
        per_load_means = []
        for load in LOADS:
            obs = [r["per_load"][str(load)]["order_bearing_accuracy"]
                   for r in results[condition]]
            ois = [r["per_load"][str(load)]["order_invariant_accuracy"]
                   for r in results[condition]]
            ob_m = float(np.mean(obs)); oi_m = float(np.mean(ois))
            agg[condition][load] = {
                "order_bearing_mean": ob_m,
                "order_bearing_per_seed": obs,
                "order_invariant_mean": oi_m,
                "order_invariant_per_seed": ois,
            }
            per_load_means.append(f"L{load}: OB={ob_m:.3f} OI={oi_m:.3f}")
            if ob_m < BAR:
                ob_all_pass = False
            if oi_m < BAR:
                oi_all_pass = False
        print(f"  {condition}:  {'  '.join(per_load_means)}", flush=True)
        if ob_all_pass and oi_all_pass:
            verdicts[condition] = "CROSS_BRIDGE_PASS"
        else:
            verdicts[condition] = "CROSS_BRIDGE_BOUNDARY"

    print(f"\n=== VERDICT ===", flush=True)
    for c, v in verdicts.items():
        print(f"  {c}: {v}", flush=True)

    primary_verdict = verdicts["global_mean"]
    print(f"\nPRIMARY VERDICT (global_mean): {primary_verdict}",
          flush=True)
    if primary_verdict == "CROSS_BRIDGE_PASS":
        print("  Parallel-matching biologized mode-unification extends "
              "ACROSS bridge boundaries: per-slot identification "
              "succeeds on the 160-concept union vocabulary with global "
              "mean-centring. NOT yet a capability claim -- pending "
              "fresh dedicated adversarial review. Oracle-adjacency "
              "caveat from (b) carries forward.", flush=True)
    else:
        print("  Cross-bridge mode-unification does NOT clear the bar "
              "at every load under global mean-centring. Honest "
              "characterisation finding; per-load breakdown above.",
              flush=True)

    out = {
        "bridges": BRIDGES, "seeds": list(SEEDS), "loads": LOADS,
        "bar": BAR, "n_gamma_slots": N_GAMMA_SLOTS,
        "k_vocab": K_VOCAB_TARGET,
        "decoder_order_bearing": "parallel_population_matching",
        "decoder_order_invariant": "marginal_sum_phase_similarity",
        "vocab_size": "union_of_5_bridges_32_concepts_each",
        "conditions": ["global_mean", "per_bridge_mean"],
        "per_condition_per_seed": results,
        "per_condition_aggregate": {
            c: {str(l): v for l, v in d.items()} for c, d in agg.items()},
        "per_condition_verdict": verdicts,
        "primary_verdict": primary_verdict,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_JSON}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
