"""Biologized spiking mode-unification with PARALLEL-POPULATION-
MATCHING order-bearing decoder -- the biology-grounded alternative
to the FHRR-biologization arc's TPAM attractor.

Justified by the biologized-spiking mode-unification NEGATIVE_ORDER_-
INVARIANT_ONLY result (TPAM has V=8-V=20 capacity window; at 32-
concept full-vocabulary mode-unification per-slot identification it
crosses the ceiling) and the built-in diagnostic showing argmax-of-
phase-similarities to substrate-grounded vocab symbols recovers
per-slot items at multi-seed 1.000 on the same data.

THE HONEST ORACLE-ADJACENCY FRAMING (recorded up front)
-------------------------------------------------------
The FHRR-biologization arc's shortcut-3 critique was that the
original composition layer used "argmax over an explicitly stored
vocabulary table" -- engineered hand-supplied. The TPAM attractor
was the biology-faithful replacement (recurrent Hopfield-class
network). TPAM IS biologically grounded (cortical attractor
networks; Amit & Treves 1989).

Parallel-population-matching is a DIFFERENT biology-grounded
mechanism: feedforward similarity comparison across a population of
neurons each tuned to one stored concept, followed by lateral-
inhibition winner-take-all. The KEY distinction that keeps this
biology-grounded rather than engineered: the "vocabulary" here is
the substrate's own derived grounded symbols (mean-centred
consolidated activity → fixed-seed deriver → spike-phase rep), NOT
a hand-supplied engineered table. The "argmax" is the parallel-
population WTA biology naturally implements. Comparing the unbind
output to each substrate-derived concept symbol via phase-similarity
is dendritic integration; picking the maximum is lateral inhibition
WTA. Both biological operations are well-established cortical
mechanisms.

Both biologizations are honest with different scaling properties:
- TPAM: recurrent attractor settling; non-monotonic V=8-V=20
  capacity window on grounded symbols.
- Parallel matching: feedforward similarity + WTA; scales with
  vocab without the attractor capacity ceiling.

This runner tests the second alternative head-to-head on the
identical substrate cache the pre-registered TPAM runner used.

WHAT CHANGES (the ONLY change from the pre-registered mode-
unification runner): the ORDER-BEARING decoder is per-slot
argmax(phase_similarity(unbinds[k], grounded[w]) for w in vocab),
no TPAM, no attractor. The ORDER-INVARIANT decoder is unchanged
(marginal-sum of per-slot phase-similarities, top-K). Both
readouts share the SAME encoded C and the SAME per-slot unbinds.

PRE-REGISTERED reading (fixed; never tuned):
- PASS: BOTH readouts multi-seed-mean >= the frozen 0.80 bar at
  every load {2, 3, 5}. The biologized mode-unification's both-
  readouts capability is realised on the substrate via parallel-
  population-matching identification.
- NEGATIVE_PARALLEL_MATCHING_INSUFFICIENT: either readout misses.
  Would diverge from the diagnostic prediction; investigation
  needed.

Reuse-by-import only; no protected/frozen/moat module modified; no
automatic differentiation; no-confab moat must stay 7/7 green.
Plain ASCII.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Validated substrate + training (byte-unchanged from the mode-
# unification arc; substrate cache is reused).
from research.runners.concept_pool_sparse_distributed import (
    build_sparse_pool_bridge,
)
from research.findings.raw.vocabulary_scaling_run_trained import (
    train_substrate, N_TRAIN_EVENTS,
)
from research.findings.raw.vocabulary_scaling_run import (
    BAR, LOADS, SEEDS, N_DIM, K_RECOG, N_TRIALS, SPARSITY,
    STIM_STEPS, SMOKE_M_OBS, SMOKE_K, SMOKE_TRIALS, SMOKE_STIM_STEPS,
    capture_concept_activity, _save_cache, _load_cache,
)
# 160-ensemble helper for the bridge's vocab + patterns.
from research.findings.raw.vocabulary_scaling_160ensemble_helpers import (
    bridge_vocab_and_patterns,
)
# FHRR-biologization arc's bind/unbind/bundle on resonate-and-fire
# neurons. NOTE: NO ResonateFireTPAM import -- this runner uses
# parallel population matching, NOT the recurrent attractor.
from research.runners.resonate_fire_fhrr import ResonateFireFHRR
from research.runners.spiking_phasor_fhrr import (
    phases_to_spikes, phase_similarity,
)
from research.findings.raw.pattern_separation_grounding_probe import (
    make_deriver,
)
# Mode-unification arc's gamma-slot positions helper.
from research.findings.raw.biologized_spiking_mode_unification_helpers import (
    gamma_slot_positions,
)

# =====================================================================
# Pre-registered constants -- IDENTICAL to the mode-unification arc
# so the comparison is head-to-head (same substrate, same encoding,
# same gamma slots, same K=16 recipe, ONLY the decoder differs).
# =====================================================================
N_GAMMA_SLOTS = 7
N_CONCEPTS_PER_BRIDGE = 32
K_VOCAB_TARGET = 16
M_OBS_FULL = 16
DERIV_SEED = 90909
TEST_BRIDGE = "bridgeA_nouns"

DEFAULT_N_LANG_INPUT = 8192
DEFAULT_N_SHARED_POOL = 2000
DEFAULT_N_SHARED_FS = 300
DEFAULT_PATTERN_SIZE = 100

SMOKE_VOCAB = 8
SMOKE_N_LANG_INPUT = 512
SMOKE_N_SHARED_POOL = 512
SMOKE_N_SHARED_FS = 60
SMOKE_PATTERN_SIZE = 24
SMOKE_TRAIN_EVENTS = 10
SMOKE_LOADS = [2, 3]

# The substrate cache from the mode-unification arc (which itself
# was the 160-ensemble bridgeA_nouns cache, byte-identical to what
# this runner would produce since the substrate recipe is identical).
CACHE_DIR = os.path.join(
    _HERE, "biologized_spiking_mode_unification_cache")


def _cache_path(seed, smoke):
    tag = "smoke" if smoke else "full"
    return os.path.join(CACHE_DIR, f"{tag}_seed{seed}.npz")


def _build_and_capture(seed, smoke, verbose):
    """Reuses the mode-unification substrate cache. Builds + trains
    + captures if absent (byte-identical to what the mode-unification
    runner would produce since the substrate recipe is identical)."""
    m_obs = SMOKE_M_OBS if smoke else M_OBS_FULL
    cache_p = _cache_path(seed, smoke)
    if os.path.exists(cache_p):
        acts, words, patterns = _load_cache(cache_p)
        if verbose:
            print(f"  [seed {seed}] loaded cached activity "
                  f"({len(words)} concepts, "
                  f"{acts[words[0]].shape[0]} obs/concept)", flush=True)
        n_lang = (SMOKE_N_LANG_INPUT if smoke else DEFAULT_N_LANG_INPUT)
        return acts, words, patterns, n_lang, m_obs

    if smoke:
        n_lang = SMOKE_N_LANG_INPUT
        n_pool = SMOKE_N_SHARED_POOL
        n_fs = SMOKE_N_SHARED_FS
        k = SMOKE_PATTERN_SIZE
        n_concepts = SMOKE_VOCAB
        n_train_events = SMOKE_TRAIN_EVENTS
        stim_steps = SMOKE_STIM_STEPS
    else:
        n_lang = DEFAULT_N_LANG_INPUT
        n_pool = DEFAULT_N_SHARED_POOL
        n_fs = DEFAULT_N_SHARED_FS
        k = DEFAULT_PATTERN_SIZE
        n_concepts = N_CONCEPTS_PER_BRIDGE
        n_train_events = N_TRAIN_EVENTS
        stim_steps = STIM_STEPS

    t0 = time.time()
    bridge = build_sparse_pool_bridge(
        seed=seed, n_lang_input=n_lang, n_shared_pool=n_pool,
        n_shared_fs=n_fs, n_lang_output=n_lang, verbose=verbose)
    full_vocab, full_patterns = bridge_vocab_and_patterns(
        TEST_BRIDGE, seed=seed, n_pool=n_pool, k=k)
    words = full_vocab[:n_concepts]
    patterns = [full_patterns[i] for i in range(n_concepts)]
    if verbose:
        print(f"  [seed {seed}] training {TEST_BRIDGE} ({n_concepts} "
              f"concepts x {n_train_events} events)", flush=True)
    train_substrate(
        bridge, patterns, n_lang_input=n_lang, n_concepts=n_concepts,
        seed=seed, n_train_events=n_train_events, sparsity=SPARSITY,
        n_words_for_orthogonal=n_concepts, verbose=verbose)
    if verbose:
        print(f"  [seed {seed}] capturing {m_obs} observations per "
              f"concept", flush=True)
    acts = capture_concept_activity(
        bridge, words, patterns, m_obs=m_obs, n_lang_input=n_lang,
        n_words_for_orthogonal=n_concepts, stim_steps=stim_steps,
        verbose=verbose)
    _save_cache(cache_p, acts, words, patterns)
    if verbose:
        density = float(np.mean([np.mean(acts[w] > 0.0) for w in words]))
        print(f"  [seed {seed}] trained + captured + cached in "
              f"{time.time() - t0:.1f}s (captured pool density "
              f"{density:.4f})", flush=True)
    return acts, words, patterns, n_lang, m_obs


def _ground_symbols(consolidated, words, d_act):
    """Mean-centred activity -> deriver -> spike-phase, per concept.
    Identical to the FHRR-biologization arc's grounded-symbol
    derivation."""
    common = np.mean([consolidated[w] for w in words], axis=0)
    deriver = make_deriver(N_DIM, d_act, DERIV_SEED)
    return {w: phases_to_spikes(deriver(consolidated[w] - common))
            for w in words}


def run_one_seed(seed, smoke=False):
    """Build (or load cached) substrate; derive grounded symbols;
    build gamma-slot positions; per trial encode a K-tuple sequence
    and run BOTH readouts on the SAME C -- the ONLY change from the
    pre-registered mode-unification runner is the order-bearing
    decoder (parallel-population matching, no TPAM)."""
    print(f"\n--- seed {seed} ---", flush=True)
    k_vocab = SMOKE_K if smoke else K_VOCAB_TARGET
    n_trials = SMOKE_TRIALS if smoke else N_TRIALS
    loads = SMOKE_LOADS if smoke else LOADS

    acts, words, _patterns, _n_lang, _m_obs = _build_and_capture(
        seed, smoke, verbose=True)
    d_act = acts[words[0]].shape[1]
    consolidated = {w: acts[w][:k_vocab].mean(axis=0) for w in words}
    grounded = _ground_symbols(consolidated, words, d_act)

    positions = gamma_slot_positions(seed, N_GAMMA_SLOTS, N_DIM)
    net = ResonateFireFHRR(N_DIM, np.random.default_rng(seed))
    word_idx = {w: i for i, w in enumerate(words)}
    qrng = np.random.default_rng(seed + 1)

    per_load = {}
    for load in loads:
        assert load <= N_GAMMA_SLOTS, (
            f"load {load} exceeds gamma slots {N_GAMMA_SLOTS}")
        ob_ok = oi_ok = 0
        for _ in range(n_trials):
            items = list(qrng.choice(words, size=load, replace=False))
            items_idx = tuple(word_idx[w] for w in items)
            # Encode C = bundle_k bind(grounded[item_k], position_k)
            # on resonate-and-fire neurons. SAME C for both readouts.
            C = net.encode([(grounded[items[k]], positions[k])
                            for k in range(load)])
            # Cache per-slot unbinds -- BOTH readouts reuse them.
            unbinds = [net.query(C, positions[k]) for k in range(load)]
            # ORDER-BEARING via PARALLEL POPULATION MATCHING:
            # per-slot argmax of phase_similarity(unbinds[k],
            # grounded[w]) over the FULL vocabulary. Biology:
            # dendritic integration + lateral-inhibition WTA across
            # a population of neurons each tuned to one substrate-
            # derived concept. The "vocabulary" is the substrate's
            # own grounded symbols (NOT a hand-supplied engineered
            # table). True items NEVER index the decoder.
            recovered = []
            for k in range(load):
                scores = [phase_similarity(unbinds[k], grounded[w])
                          for w in words]
                recovered.append(int(np.argmax(scores)))
            if tuple(recovered) == items_idx:
                ob_ok += 1
            # ORDER-INVARIANT (unchanged from the mode-unification
            # runner): per-vocab-item marginal-sum of similarities
            # across slots; top-K sorted by index; compared to
            # encoded set.
            scores_oi = np.zeros(len(words))
            for k in range(load):
                for w_idx, w in enumerate(words):
                    scores_oi[w_idx] += phase_similarity(unbinds[k],
                                                          grounded[w])
            topK = sorted(int(i) for i in np.argsort(scores_oi)[-load:])
            if tuple(topK) == tuple(sorted(items_idx)):
                oi_ok += 1
        per_load[load] = {
            "order_bearing_accuracy": ob_ok / n_trials,
            "order_invariant_accuracy": oi_ok / n_trials,
            "n_trials": n_trials,
        }
        print(f"  L={load}: order-bearing="
              f"{per_load[load]['order_bearing_accuracy']:.4f}  "
              f"order-invariant="
              f"{per_load[load]['order_invariant_accuracy']:.4f}",
              flush=True)

    return {
        "seed": seed, "smoke": bool(smoke),
        "bridge": TEST_BRIDGE,
        "decoder_order_bearing": "parallel_population_matching",
        "decoder_order_invariant": "marginal_sum_phase_similarity",
        "k_vocab": k_vocab, "k_recog": SMOKE_K if smoke else K_RECOG,
        "n_concepts": len(words), "activity_dim": int(d_act),
        "per_load": {str(load): v for load, v in per_load.items()},
    }


def main():
    ap = argparse.ArgumentParser(
        description="Biologized spiking mode-unification with "
                    "PARALLEL-POPULATION-MATCHING order-bearing "
                    "decoder (biology-grounded alternative to TPAM "
                    "attractor; same K=16 PASS recipe).")
    ap.add_argument("--smoke", action="store_true",
                    help="reduced-scale grounding check (tiny bridge, "
                         "tiny vocab, few trials) -- toy numbers NOT "
                         "propagated as a result")
    args = ap.parse_args()
    smoke = bool(args.smoke)
    seeds = [42] if smoke else list(SEEDS)
    loads = SMOKE_LOADS if smoke else LOADS

    print("=== biologized spiking mode-unification "
          "(parallel-population-matching decoder) ===", flush=True)
    if smoke:
        print("  *** SMOKE MODE: tiny bridge + vocab + few trials -- "
              "toy numbers, NOT a result ***", flush=True)
    print(f"  ORACLE-ADJACENCY CAVEAT: the order-bearing decoder is "
          f"parallel population matching (feedforward similarity + "
          f"lateral-inhibition WTA) over the substrate's own derived "
          f"grounded symbols. Biology-grounded mechanism, distinct "
          f"from the TPAM attractor. See design doc for full framing.",
          flush=True)
    print(f"bridge={TEST_BRIDGE}; seeds={seeds}; loads={loads}; "
          f"N_gamma_slots={N_GAMMA_SLOTS}; bar={BAR}; "
          f"K_VOCAB={SMOKE_K if smoke else K_VOCAB_TARGET}; "
          f"decoder=parallel_population_matching",
          flush=True)

    seed_results = [run_one_seed(s, smoke=smoke) for s in seeds]

    print(f"\n=== MULTI-SEED AGGREGATE ===", flush=True)
    print(f"          order-bearing       order-invariant", flush=True)
    agg = {}
    ob_all_pass = True
    oi_all_pass = True
    for load in loads:
        ob = [r["per_load"][str(load)]["order_bearing_accuracy"]
              for r in seed_results]
        oi = [r["per_load"][str(load)]["order_invariant_accuracy"]
              for r in seed_results]
        ob_mean = float(np.mean(ob))
        oi_mean = float(np.mean(oi))
        agg[load] = {"order_bearing_mean": ob_mean,
                     "order_bearing_per_seed": ob,
                     "order_invariant_mean": oi_mean,
                     "order_invariant_per_seed": oi}
        if ob_mean < BAR:
            ob_all_pass = False
        if oi_mean < BAR:
            oi_all_pass = False
        print(f"  L={load}:  {ob_mean:.4f} "
              f"({'>=' if ob_mean >= BAR else '<'}{BAR})        "
              f"{oi_mean:.4f} ({'>=' if oi_mean >= BAR else '<'}{BAR})",
              flush=True)

    print(f"\n=== VERDICT ===", flush=True)
    if smoke:
        verdict = "SMOKE"
        print("  SMOKE run -- toy numbers, not propagated.", flush=True)
    elif ob_all_pass and oi_all_pass:
        verdict = "MODE_UNIFICATION_BIOLOGIZED_PASS_VIA_PARALLEL_MATCHING"
        print("  BOTH order-bearing AND order-invariant clear the "
              "frozen 0.80 bar multi-seed at every tested load. The "
              "biologized mode-unification both-readouts capability "
              "is realised on the project's substrate via parallel-"
              "population-matching identification (a biology-grounded "
              "alternative to the TPAM attractor). Subject to a fresh "
              "dedicated adversarial review before any capability "
              "claim. The oracle-adjacency caveat from the design doc "
              "applies.", flush=True)
    else:
        verdict = "NEGATIVE_PARALLEL_MATCHING_INSUFFICIENT"
        print("  Either readout misses the bar. Diagnostic predicted "
              "PASS; investigation needed for the divergence.",
              flush=True)

    out = {
        "seeds": seeds, "bridge": TEST_BRIDGE, "loads": loads,
        "bar": BAR, "n_gamma_slots": N_GAMMA_SLOTS,
        "decoder_order_bearing": "parallel_population_matching",
        "decoder_order_invariant": "marginal_sum_phase_similarity",
        "k_vocab": SMOKE_K if smoke else K_VOCAB_TARGET,
        "k_recog": SMOKE_K if smoke else K_RECOG,
        "n_trials": SMOKE_TRIALS if smoke else N_TRIALS,
        "n_train_events": SMOKE_TRAIN_EVENTS if smoke else N_TRAIN_EVENTS,
        "n_concepts_per_bridge": (SMOKE_VOCAB if smoke
                                   else N_CONCEPTS_PER_BRIDGE),
        "n_dim": N_DIM, "smoke": smoke,
        "per_seed": seed_results,
        "aggregate": {str(l): v for l, v in agg.items()},
        "verdict": verdict,
    }
    tag = "smoke" if smoke else "full"
    out_path = os.path.join(
        _HERE,
        f"biologized_spiking_mode_unification_parallel_matching_runner_{tag}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
