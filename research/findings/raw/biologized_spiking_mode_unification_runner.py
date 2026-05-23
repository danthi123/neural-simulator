"""Biologized spiking theta-gamma mode-unification runner.

Implements the catalog-documented Lisman-Idiart N.16 mechanism on the
project's validated substrate: order-bearing AND order-invariant
readouts from the SAME spiking-substrate encoded code, where the
encoding represents an ordered K-item sequence by binding each item's
grounded symbol to its gamma-slot position phasor and bundling on
resonate-and-fire neurons.

Reuse-by-import only:
- `train_substrate` from the trained-substrate runner.
- `capture_concept_activity` from the vocab-scaling decisive runner.
- `ResonateFireFHRR.encode/query`, `ResonateFireTPAM.settle_annealed`,
  the anneal constants -- all from the FHRR-biologization arc.
- `phases_to_spikes`, `phase_similarity` from the SpikingPhasorFHRR
  scaffold.
- `make_deriver` from the pattern-separation grounding probe.
- `gamma_slot_positions` from Task 1's helper module.
- `bridge_vocab_and_patterns` from the 160-ensemble helpers
  (returns the bridge's 32-word vocab + sparse K-of-N patterns).

Genuinely-new code in this runner:
1. The orchestration loop that builds + trains + captures one bridge,
   derives grounded symbols, builds gamma-slot positions, and runs
   trials.
2. Per-trial sequence encoding via
   `ResonateFireFHRR.encode([(grounded[item_k], position_k) for k])`.
3. The two readout decoders on the SAME encoded C:
   - ORDER-BEARING: for each slot k, attractor-settle
     `net.query(C, position_k)` against the vocabulary TPAM, find
     the argmax-overlap item, compare K-tuple.
   - ORDER-INVARIANT: for each candidate item w, score
     `sum_k phase_similarity(net.query(C, position_k), grounded[w])`,
     take top-K items sorted by index, compare set.
   Both readouts share the SAME encoded C and the SAME per-slot
   unbinds (cached in the loop so the order-invariant readout reuses
   the order-bearing readout's unbind work).

PRE-REGISTERED reading (fixed; never tuned):
- PASS: BOTH order-bearing AND order-invariant multi-seed-mean >=
  the frozen 0.80 bar at every load {2, 3, 5}. The biologized
  spiking implementation realises the Lisman-Idiart N.16 mechanism
  on the project's substrate.
- NEGATIVE_ORDER_BEARING_ONLY / NEGATIVE_ORDER_INVARIANT_ONLY /
  NEGATIVE_BOTH: same trichotomy as the algebra probe; each is an
  honest finding about which side of unification fails on the
  biologized substrate.

`--smoke` runs a reduced-scale grounding check (tiny bridge, 8-concept
vocab subset, few trials) -- toy numbers NOT propagated. Kill-safe
per-seed activity cache so a re-run resumes from the next uncached
seed. Plain ASCII; no protected/frozen/moat module modified; no
automatic differentiation.
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

# Validated substrate + training, byte-unchanged reuse.
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
# 160-ensemble helper that pulls a bridge's vocab + sparse patterns.
from research.findings.raw.vocabulary_scaling_160ensemble_helpers import (
    bridge_vocab_and_patterns,
)
# FHRR-biologization arc's resonate-and-fire layer.
from research.runners.resonate_fire_fhrr import (
    ResonateFireFHRR, ResonateFireTPAM,
    ANNEAL_THETA_LOW, ANNEAL_THETA_HIGH, ANNEAL_ITERS,
)
from research.runners.spiking_phasor_fhrr import (
    phases_to_spikes, phase_similarity,
)
from research.findings.raw.pattern_separation_grounding_probe import (
    make_deriver,
)
# Task 1's helper, re-exported here so Task 0's pin sees it.
from research.findings.raw.biologized_spiking_mode_unification_helpers import (
    gamma_slot_positions,
)

# =====================================================================
# Pre-registered constants. The K=16 PASS recipe (the recipe the
# 64-concept K=16 PASS used) is fixed; not tunable.
# =====================================================================
N_GAMMA_SLOTS = 7            # Lisman-Idiart biologically grounded value
N_CONCEPTS_PER_BRIDGE = 32   # matches the 160-ensemble per-bridge size
K_VOCAB_TARGET = 16          # K=16 PASS recipe (overrides module default 8)
M_OBS_FULL = 16              # cache 16 observations per concept
DERIV_SEED = 90909           # same deriver seed as the FHRR-biologization arc
TEST_BRIDGE = "bridgeA_nouns"

# Substrate sizing (the validated G.20 defaults the 64-concept thread used).
DEFAULT_N_LANG_INPUT = 8192
DEFAULT_N_SHARED_POOL = 2000
DEFAULT_N_SHARED_FS = 300
DEFAULT_PATTERN_SIZE = 100

# Smoke: tiny bridge, 8-concept vocab subset, few trials.
SMOKE_VOCAB = 8
SMOKE_N_LANG_INPUT = 512
SMOKE_N_SHARED_POOL = 512
SMOKE_N_SHARED_FS = 60
SMOKE_PATTERN_SIZE = 24
SMOKE_TRAIN_EVENTS = 10
SMOKE_LOADS = [2, 3]

CACHE_DIR = os.path.join(
    _HERE, "biologized_spiking_mode_unification_cache")


def _cache_path(seed, smoke):
    tag = "smoke" if smoke else "full"
    return os.path.join(CACHE_DIR, f"{tag}_seed{seed}.npz")


def _build_and_capture(seed, smoke, verbose):
    """Build the trained substrate for one seed and capture per-concept
    activity. Cached on disk; a re-run loads the cache and skips.
    Returns (acts, words, patterns, n_lang, m_obs)."""
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
    The biologized grounded-symbol pipeline byte-equivalent to
    `biologized_grounded_composition.run_one_seed`'s meancenter
    branch."""
    common = np.mean([consolidated[w] for w in words], axis=0)
    deriver = make_deriver(N_DIM, d_act, DERIV_SEED)
    return {w: phases_to_spikes(deriver(consolidated[w] - common))
            for w in words}


def run_one_seed(seed, smoke=False):
    """Build + train + capture for one seed; derive grounded symbols;
    build gamma-slot positions; per trial encode a K-tuple sequence
    and run BOTH readouts on the SAME C. Returns the per-load result
    dict."""
    print(f"\n--- seed {seed} ---", flush=True)
    k_vocab = SMOKE_K if smoke else K_VOCAB_TARGET
    n_trials = SMOKE_TRIALS if smoke else N_TRIALS
    loads = SMOKE_LOADS if smoke else LOADS

    acts, words, _patterns, _n_lang, _m_obs = _build_and_capture(
        seed, smoke, verbose=True)
    d_act = acts[words[0]].shape[1]
    consolidated = {w: acts[w][:k_vocab].mean(axis=0) for w in words}
    grounded = _ground_symbols(consolidated, words, d_act)

    # Build per-seed gamma-slot positions and the per-bridge TPAM
    # over the grounded vocabulary.
    positions = gamma_slot_positions(seed, N_GAMMA_SLOTS, N_DIM)
    net = ResonateFireFHRR(N_DIM, np.random.default_rng(seed))
    tpam = ResonateFireTPAM([grounded[w] for w in words])
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
            # on resonate-and-fire neurons.
            C = net.encode([(grounded[items[k]], positions[k])
                            for k in range(load)])
            # Cache per-slot unbinds -- both readouts reuse them.
            unbinds = [net.query(C, positions[k]) for k in range(load)]
            # ORDER-BEARING: per-slot attractor settle + argmax over
            # the vocab TPAM patterns.
            recovered = []
            for k in range(load):
                z, _ = tpam.settle_annealed(
                    unbinds[k], ANNEAL_THETA_LOW, ANNEAL_THETA_HIGH,
                    ANNEAL_ITERS, fast=True)
                overlaps = np.abs(tpam.s.conj().T @ z)
                recovered.append(int(np.argmax(overlaps)))
            if tuple(recovered) == items_idx:
                ob_ok += 1
            # ORDER-INVARIANT: per-candidate-item marginal-sum-of-
            # similarities across slots; top-K items sorted by index;
            # compared to the encoded set.
            scores = np.zeros(len(words))
            for k in range(load):
                for w_idx, w in enumerate(words):
                    scores[w_idx] += phase_similarity(unbinds[k],
                                                       grounded[w])
            topK = sorted(int(i) for i in np.argsort(scores)[-load:])
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
        "k_vocab": k_vocab, "k_recog": SMOKE_K if smoke else K_RECOG,
        "n_concepts": len(words), "activity_dim": int(d_act),
        "per_load": {str(load): v for load, v in per_load.items()},
    }


def main():
    ap = argparse.ArgumentParser(
        description="Biologized spiking theta-gamma mode-unification "
                    "on the project's validated substrate (1 bridge, "
                    "K=16 PASS recipe).")
    ap.add_argument("--smoke", action="store_true",
                    help="reduced-scale grounding check (tiny bridge, "
                         "tiny vocab, few trials) -- toy numbers NOT "
                         "propagated as a result")
    args = ap.parse_args()
    smoke = bool(args.smoke)
    seeds = [42] if smoke else list(SEEDS)
    loads = SMOKE_LOADS if smoke else LOADS

    print("=== biologized spiking theta-gamma mode-unification ===",
          flush=True)
    if smoke:
        print("  *** SMOKE MODE: tiny bridge + vocab + few trials -- "
              "toy numbers, NOT a result ***", flush=True)
    print(f"bridge={TEST_BRIDGE}; seeds={seeds}; loads={loads}; "
          f"N_gamma_slots={N_GAMMA_SLOTS}; bar={BAR}; "
          f"K_VOCAB={SMOKE_K if smoke else K_VOCAB_TARGET}; "
          f"K=16 PASS recipe", flush=True)

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
        print("  SMOKE run -- toy numbers, not propagated as a result.",
              flush=True)
    elif ob_all_pass and oi_all_pass:
        verdict = "MODE_UNIFICATION_BIOLOGIZED_PASS"
        print("  BOTH order-bearing AND order-invariant clear the "
              "frozen 0.80 bar multi-seed at every tested load on the "
              "biologized spiking substrate. The Lisman-Idiart N.16 "
              "mechanism is realised on the project's substrate. "
              "Subject to a fresh dedicated adversarial review before "
              "any capability claim.", flush=True)
    elif ob_all_pass and not oi_all_pass:
        verdict = "NEGATIVE_ORDER_BEARING_ONLY"
        print("  Order-bearing PASSes on the biologized substrate; "
              "order-invariant misses. The biologized per-position "
              "decoder works but the marginal-sum decoder does not.",
              flush=True)
    elif oi_all_pass and not ob_all_pass:
        verdict = "NEGATIVE_ORDER_INVARIANT_ONLY"
        print("  Order-invariant PASSes on the biologized substrate; "
              "order-bearing misses. The biologized marginal-sum "
              "decoder works but the per-position attractor settle "
              "does not.", flush=True)
    else:
        verdict = "NEGATIVE_BOTH"
        print("  Neither readout clears the frozen 0.80 bar on the "
              "biologized substrate. The algebra-PASS does not "
              "transfer; the failure mode sharpens which biological "
              "component would need refinement.", flush=True)

    out = {
        "seeds": seeds, "bridge": TEST_BRIDGE, "loads": loads,
        "bar": BAR, "n_gamma_slots": N_GAMMA_SLOTS,
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
        f"biologized_spiking_mode_unification_runner_{tag}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
