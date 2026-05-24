"""Cross-bridge biologized mode-unification probe -- OPTION 4 from the
(e) completion (autonomous parallel direction while awaiting owner
steer on (c) generative-replay architecture-integration choice).

Tests whether the parallel-matching biologized mode-unification
extends ACROSS bridge boundaries: encode a composite whose K items are
drawn UNIFORMLY from the union of all 5 bridges' 32-concept
vocabularies (160 concepts total), and decode per-slot via the
parallel-matching mechanism over the FULL 160-concept union.

GPU-BATCHED PATH (the only path; CuPy backend; falls back to numpy if
SIM_BACKEND=numpy or no GPU). The per-slot decoder stacks the 160
grounded symbols as a (V, N_dim) phase matrix and computes all V
similarities in one broadcast + mean operation per slot per trial,
replacing 160 scalar phase_similarity calls. ~5-20x faster than the
scalar Python-loop pattern the (b) and (e) parents used. Same scalar
contract: the BATCHED phase_similarity is verified byte-equivalent to
the reused scalar phase_similarity at startup before any
characterisation work; if not equivalent the runner refuses to run.

CRITICAL TECHNICAL DECISION (recorded honestly): per-bridge
_ground_symbols subtracts each bridge's OWN common mode (mean across
that bridge's 32 concepts). For a fair cross-bridge geometry, we
re-mean-centre GLOBALLY across all 160 concepts and re-derive the
grounded symbols via the SAME fixed-seed deriver pipeline. This is
more biology-faithful (cortical pooled inhibition normalises across
the whole cortical extent, not per-region) and yields a uniform
phasor space across bridges. The probe also runs a comparison
condition (per_bridge_mean) so the choice is characterised.

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

Reuses every 160-ensemble cache + parallel-matching primitives byte-
unchanged; no protected/frozen/moat module modified; no autograd;
no-confab moat must stay 7/7 green.
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
    phases_to_spikes, phase_similarity, spikes_to_phases, CYCLE_STEPS,
)
from sim.backend import get_backend, to_host, is_gpu_backend

BRIDGES = ["bridgeA_nouns", "bridgeB_verbs", "bridgeC_adj",
           "bridgeD_spatial", "bridgeE_functional"]
ENSEMBLE_CACHE_DIR = os.path.join(
    _HERE, "vocabulary_scaling_160ensemble_cache")
N_GAMMA_SLOTS = 7
OUT_JSON = os.path.join(
    _HERE, "cross_bridge_mode_unification_probe.json")


def build_vocab_phase_matrix(grounded, all_words, xp):
    """Stack all V grounded symbols as a (V, N_dim) phase matrix on the
    active backend. spikes_to_phases is the reused conversion (byte-
    unchanged); the stacking is the only new operation. Returns the
    matrix on the active backend (xp arrays)."""
    phases_list = [spikes_to_phases(grounded[bw], CYCLE_STEPS)
                   for bw in all_words]
    # Stack on host first (numpy), then move to backend.
    phase_matrix_host = np.stack(phases_list, axis=0)  # (V, N_dim)
    return xp.asarray(phase_matrix_host)


def batched_phase_similarity(unbind_spikes, vocab_phase_matrix, xp):
    """Vectorised FHRR similarity of one unbind against ALL V vocab
    symbols. Returns (V,) array on the active backend.

    Mathematically IDENTICAL to scalar phase_similarity(unbind, v)
    iterated for v in vocab: mean(cos(2*pi*(pu - pv))) along the
    N_dim axis. The broadcast is the only optimisation."""
    pu_host = spikes_to_phases(unbind_spikes, CYCLE_STEPS)  # (N_dim,) numpy
    pu = xp.asarray(pu_host)
    # vocab_phase_matrix: (V, N_dim); pu[None, :]: (1, N_dim) -> broadcast
    diffs = pu[None, :] - vocab_phase_matrix  # (V, N_dim)
    sims = xp.mean(xp.cos(2.0 * xp.pi * diffs), axis=1)  # (V,)
    return sims


def verify_batched_equivalent_to_scalar(grounded, all_words, xp,
                                          rng_seed=0):
    """Byte-equivalence check: build a phase matrix; compute batched
    similarities of one random unbind against ALL vocab; compare to
    scalar phase_similarity(unbind, v) for each v. Tolerance 1e-10
    (both paths are double-precision phase cosine). Fail-closed."""
    vocab_phase_matrix = build_vocab_phase_matrix(grounded, all_words, xp)
    rng = np.random.default_rng(rng_seed)
    # Probe vector: random spike pattern in the same integer space.
    probe = rng.integers(0, CYCLE_STEPS, size=N_DIM).astype(np.int64)
    sims_batched = to_host(batched_phase_similarity(
        probe, vocab_phase_matrix, xp))  # (V,)
    sims_scalar = np.array(
        [phase_similarity(probe, grounded[bw], CYCLE_STEPS)
         for bw in all_words])  # (V,)
    max_diff = float(np.max(np.abs(sims_batched - sims_scalar)))
    if max_diff > 1e-10:
        raise RuntimeError(
            f"Batched vs scalar phase_similarity max-diff {max_diff:.3e} "
            f"exceeds tolerance 1e-10 -- refusing to run.")
    return max_diff, vocab_phase_matrix


def _global_ground_symbols(seed):
    """Load all 5 bridges' caches for this seed; build the 160-
    concept union; re-mean-centre GLOBALLY across all 160 concepts;
    derive grounded symbols via the SAME fixed-seed deriver
    (DERIV_SEED=90909). Returns (all_words, grounded, d_act)."""
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
    common = np.mean([consolidated[bw] for bw in all_words], axis=0)
    deriver = make_deriver(N_DIM, d_act, DERIV_SEED)
    grounded = {bw: phases_to_spikes(deriver(consolidated[bw] - common))
                for bw in all_words}
    return all_words, grounded, d_act


def _per_bridge_ground_symbols(seed):
    """Comparison condition: per-bridge mean-centring (each bridge's
    own common mode; the (e) extension's choice)."""
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


def run_one_seed_one_condition(seed, condition, xp):
    if condition == "global_mean":
        all_words, grounded, d_act = _global_ground_symbols(seed)
    elif condition == "per_bridge_mean":
        all_words, grounded, d_act = _per_bridge_ground_symbols(seed)
    else:
        raise ValueError(f"unknown condition: {condition}")
    V = len(all_words)
    # Equivalence check at the start of every cell -- fail-closed.
    max_diff, vocab_phase_matrix = verify_batched_equivalent_to_scalar(
        grounded, all_words, xp, rng_seed=seed)
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
            # ORDER-BEARING: per-slot argmax over the 160-concept union
            # via BATCHED phase-similarity (mathematically identical to
            # scalar; ~V-fold speedup on GPU; verified at cell start).
            recovered = []
            scores_oi_gpu = xp.zeros(V)
            for k in range(load):
                sims_k = batched_phase_similarity(
                    unbinds[k], vocab_phase_matrix, xp)  # (V,) on xp
                recovered.append(int(xp.argmax(sims_k)))
                scores_oi_gpu = scores_oi_gpu + sims_k
            if tuple(recovered) == items_idx:
                ob_ok += 1
            # ORDER-INVARIANT: marginal-sum already accumulated above
            # via the same batched primitive; top-K via argsort.
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
    gpu = is_gpu_backend()
    print("=== cross-bridge biologized mode-unification probe "
          "(OPTION 4, BATCHED) ===", flush=True)
    print(f"backend={backend_name} (GPU={gpu}); seeds={list(SEEDS)}; "
          f"loads={LOADS}; bridges={BRIDGES}; K_VOCAB={K_VOCAB_TARGET}; "
          f"bar={BAR}; decoder=parallel_population_matching on "
          f"160-concept union (batched)", flush=True)
    print("Reuses 160-ensemble caches + parallel-matching primitives "
          "byte-unchanged; per-cell startup verifies batched ==  "
          "scalar phase_similarity within 1e-10 (fail-closed).",
          flush=True)

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
            ob_str = ", ".join(f"L{l}={per_load[l]['order_bearing_accuracy']:.3f}"
                                for l in LOADS)
            oi_str = ", ".join(f"L{l}={per_load[l]['order_invariant_accuracy']:.3f}"
                                for l in LOADS)
            print(f"  [seed={seed} V={V} diff={max_diff:.2e}] "
                  f"OB({ob_str}) | OI({oi_str})  "
                  f"({time.time()-t_seed:.1f}s)", flush=True)
    print(f"\nTotal wall-clock: {time.time()-t0:.1f}s "
          f"(backend={backend_name})", flush=True)

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
        "backend": backend_name, "gpu": gpu,
        "bridges": BRIDGES, "seeds": list(SEEDS), "loads": LOADS,
        "bar": BAR, "n_gamma_slots": N_GAMMA_SLOTS,
        "k_vocab": K_VOCAB_TARGET,
        "decoder_order_bearing": "parallel_population_matching_batched",
        "decoder_order_invariant": "marginal_sum_phase_similarity_batched",
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
