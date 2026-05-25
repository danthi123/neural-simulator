"""Direction 5 Tier 1 decoder-fix probe: threshold-to-top-K BEFORE FHRR
projection (binarize the captured shared_concept_pool activity to the K=100
most-active neurons per word, mimicking G.20 sparse exact 0/1 K-of-N
geometry that PASSED at pillar n=95).

Tests the hypothesis from the 2026-05-25 D5 pattern enrichment diagnostic
(commit 187d983): K-of-N pattern IS embedded in shared_concept_pool
activity at ~2.1x baseline (signal IS present) BUT the FHRR projection
(2000 -> 512) treats pattern + background equally and loses the signal-
to-noise. G.20 sparse cross-bridge works because activity is 0/1 K-of-N
exactly; D5 hybrid activity is K-of-N WITH NOISE (~0.11 background +
~0.23 pattern). Thresholding-to-top-K BEFORE projection RESTORES the
exact 0/1 geometry.

Operates on EXISTING D5 smoke cache (research/findings/raw/direction_5_cache/
activity_smoke_*.npz). No retraining. CPU-only. ~5-10 min wall.

Pre-registered: per-cell PASS/FAIL on the SAME frozen 0.80 bar
(direction_5_verdict module; thresholds UNCHANGED). The decoder-fix
probe is a diagnostic variant; it does NOT modify the existing D5
probe or the frozen verdict module. Result tag uses the SAME verdict
labels (DIRECTION_5_PASS / PARTIAL / NEGATIVE / VOID_MALFORMED).

If decoder-fix PASS:
- Major biology-translatable finding: cortical sparse-code extraction
  requires neuron-level selection (winner-take-all / sparse activation)
  BEFORE downstream projection. The substrate is fine; the decoder
  was the binding constraint.
- Opens path to pillar n=106 candidate (decoder-fix variant of D5).

If decoder-fix FAIL:
- Confirms substrate geometry difference is fundamental.
- Approach C (learned dedicated->shared projection) is needed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Reuse D5 primitives byte-unchanged via import (NOT modifying the
# existing probe).
from research.findings.raw.direction_5_cross_bridge_probe import (
    BRIDGES, LOADS, BAR_OB, BAR_OI, N_GAMMA_SLOTS, CACHE_DIR,
    _PER_BRIDGE_WORDS, _activity_cache_path, _load_per_bridge_activity,
    run_one_seed_cross_bridge,
)
from research.findings.raw.direction_5_vocab_spec import (
    DIRECTION_5_TOTAL,
)
from research.findings.raw.direction_5_verdict import (
    compute_verdict,
    _DIRECTION_5_OB_MIN,
    _DIRECTION_5_OI_MIN,
    _DIRECTION_5_LOADS,
    _DIRECTION_5_MIN_SEEDS,
)
from research.findings.raw.biologized_spiking_mode_unification_parallel_matching_runner import (
    DERIV_SEED,
)
from research.findings.raw.pattern_separation_grounding_probe import (
    make_deriver,
)
from research.findings.raw.cross_bridge_mode_unification_probe import (
    batched_phase_similarity, verify_batched_equivalent_to_scalar,
)
from research.findings.raw.biologized_spiking_mode_unification_helpers import (
    gamma_slot_positions,
)
from research.runners.resonate_fire_fhrr import ResonateFireFHRR
from research.runners.spiking_phasor_fhrr import phases_to_spikes
from sim.backend import get_backend, is_gpu_backend, to_host


# Pre-registered top-K threshold (mirrors the D5 sparse pattern K=100
# used at training time; the captured activity is thresholded to the
# same K to match G.20 sparse exact K-of-N geometry).
TOP_K_THRESHOLD: int = 100
N_DIM_LOCAL: int = 512
N_TRIALS_LOCAL: int = 200


def _consolidate_topK_binary(act: np.ndarray, K: int = TOP_K_THRESHOLD) -> np.ndarray:
    """Mean-pool across M_OBS then binarize to 0/1 at the K most-active
    neurons. Returns a length-d_act float32 vector with exactly K
    non-zero entries (each set to 1.0).

    This is the decoder-fix: matches G.20 sparse exact 0/1 K-of-N
    geometry (the pillar n=95 cross-bridge passing pattern). The
    diagnostic showed the K-of-N signal IS embedded in the captured
    activity at ~2.1x baseline; binarizing isolates the K most-active
    neurons (presumably the trained pattern) from the noise floor.
    """
    mean_act = act.mean(axis=0).astype(np.float32)
    d_act = mean_act.shape[0]
    if d_act <= K:
        # Degenerate: all neurons active; return original normalized.
        return (mean_act > 0).astype(np.float32)
    top_k_idx = np.argpartition(mean_act, -K)[-K:]
    binary = np.zeros(d_act, dtype=np.float32)
    binary[top_k_idx] = 1.0
    return binary


def derive_global_grounded_symbols_topK(
    per_bridge_activity: Dict[str, Dict[str, np.ndarray]],
    seed: int,
    K: int = TOP_K_THRESHOLD,
) -> Tuple[List[Tuple[str, str]], Dict[Tuple[str, str], np.ndarray]]:
    """Decoder-fix variant: threshold each word's activity to the top-K
    most-active neurons BEFORE the FHRR projection. Otherwise byte-
    identical to derive_global_grounded_symbols from the existing
    direction_5_cross_bridge_probe.

    The mean-centring step is preserved (each bridge's binary patterns
    are mean-centred across the 16 concepts before projection) so the
    cortical-pooled-inhibition normalisation analog still applies; the
    deriver projects d_act=2000 -> N_DIM=512 per bridge as before.
    """
    all_words: List[Tuple[str, str]] = []
    grounded: Dict[Tuple[str, str], np.ndarray] = {}

    for bridge in BRIDGES:
        acts = per_bridge_activity[bridge]
        words = _PER_BRIDGE_WORDS[bridge]
        consolidated_b = {
            w: _consolidate_topK_binary(acts[w], K=K) for w in words
        }
        d_act = next(iter(consolidated_b.values())).shape[0]
        common_b = np.mean([consolidated_b[w] for w in words], axis=0)
        deriver = make_deriver(N_DIM_LOCAL, d_act, DERIV_SEED)
        for w in words:
            key = (bridge, w)
            grounded[key] = phases_to_spikes(
                deriver(consolidated_b[w] - common_b)
            )
            all_words.append(key)
    return all_words, grounded


def run_one_seed_topK(
    seed: int,
    per_bridge_activity: Dict[str, Dict[str, np.ndarray]],
    xp,
    K: int = TOP_K_THRESHOLD,
    loads: Optional[List[int]] = None,
    n_trials: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[Dict[int, Dict[str, float]], int, float]:
    """One-seed cross-bridge probe using the top-K binarized
    consolidator. Otherwise mirrors run_one_seed_cross_bridge from the
    existing direction_5_cross_bridge_probe.
    """
    if loads is None:
        loads = list(_DIRECTION_5_LOADS)
    if n_trials is None:
        n_trials = N_TRIALS_LOCAL

    all_words, grounded = derive_global_grounded_symbols_topK(
        per_bridge_activity, seed, K=K,
    )
    V = len(all_words)

    max_diff, vocab_phase_matrix = verify_batched_equivalent_to_scalar(
        grounded, all_words, xp, rng_seed=seed,
    )
    if verbose:
        print(
            "  [seed " + str(seed) + " topK=" + str(K)
            + "] V=" + str(V)
            + " grounded ready; batched-vs-scalar max-diff="
            + ("%.2e" % max_diff),
            flush=True,
        )

    positions = gamma_slot_positions(seed, N_GAMMA_SLOTS, N_DIM_LOCAL)
    net = ResonateFireFHRR(N_DIM_LOCAL, np.random.default_rng(seed))
    qrng = np.random.default_rng(seed + 1)

    per_load: Dict[int, Dict[str, float]] = {}
    for load in loads:
        ob_ok = oi_ok = 0
        for _ in range(n_trials):
            items_idx = tuple(
                int(i) for i in
                qrng.choice(V, size=load, replace=False)
            )
            items = [all_words[i] for i in items_idx]
            # Use existing ResonateFireFHRR API: encode + query
            C = net.encode([
                (grounded[items[k]], positions[k]) for k in range(load)
            ])
            unbinds = [net.query(C, positions[k]) for k in range(load)]
            # ORDER-BEARING: per-slot argmax over the 80-concept union
            recovered = []
            scores_oi_xp = xp.zeros(V)
            for k in range(load):
                sims_k = batched_phase_similarity(
                    unbinds[k], vocab_phase_matrix, xp,
                )
                recovered.append(int(xp.argmax(sims_k)))
                scores_oi_xp = scores_oi_xp + sims_k
            if tuple(recovered) == items_idx:
                ob_ok += 1
            # ORDER-INVARIANT: marginal-sum + topK via argsort
            scores_oi_host = to_host(scores_oi_xp)
            topK = sorted(
                int(i) for i in np.argsort(scores_oi_host)[-load:]
            )
            if tuple(topK) == tuple(sorted(items_idx)):
                oi_ok += 1
        per_load[load] = {
            "order_bearing_accuracy": ob_ok / n_trials,
            "order_invariant_accuracy": oi_ok / n_trials,
            "n_trials": n_trials,
        }
        if verbose:
            print(
                "    L=" + str(load)
                + ": OB=" + ("%.3f" % per_load[load]["order_bearing_accuracy"])
                + " OI=" + ("%.3f" % per_load[load]["order_invariant_accuracy"]),
                flush=True,
            )
    return per_load, V, max_diff


def run_cross_bridge_probe_topK(
    seeds: Optional[List[int]] = None,
    tag: str = "smoke",
    K: int = TOP_K_THRESHOLD,
    cache_dir: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Multi-seed wrapper: run the decoder-fix probe across seeds; emit
    verdict via the frozen direction_5_verdict module.
    """
    if seeds is None:
        seeds = [42, 43, 44]
    if cache_dir is not None:
        global CACHE_DIR
        # NOTE: CACHE_DIR is imported from direction_5_cross_bridge_probe;
        # we cannot mutate the imported module's CACHE_DIR. The override
        # is informational only here.

    xp, backend_name = get_backend()
    gpu = is_gpu_backend()
    if verbose:
        print("=== Direction 5 Tier 1 decoder-fix probe (topK="
              + str(K) + ") ===", flush=True)
        print("  backend=" + backend_name + " (GPU=" + str(gpu) + ")",
              flush=True)
        print("  K=" + str(K)
              + " (top-K most-active neurons binarized before FHRR projection)",
              flush=True)
        print("  seeds=" + str(seeds) + "; loads="
              + str(list(_DIRECTION_5_LOADS)) + "; tag=" + tag,
              flush=True)

    t0 = time.time()
    seed_results: List[Dict[str, Any]] = []
    for seed in seeds:
        if verbose:
            print("\n--- seed " + str(seed) + " ---", flush=True)
        per_bridge_activity = {}
        for bridge in BRIDGES:
            acts = _load_per_bridge_activity(bridge, seed, tag)
            per_bridge_activity[bridge] = acts
            if verbose:
                first_w = next(iter(acts.keys()))
                first_shape = acts[first_w].shape
                print("  [seed " + str(seed) + "] loaded " + bridge
                      + " (M_OBS=" + str(first_shape[0])
                      + " d_act=" + str(first_shape[1])
                      + " V_bridge=" + str(len(acts)) + ")", flush=True)

        t_seed = time.time()
        per_load, V, max_diff = run_one_seed_topK(
            seed, per_bridge_activity, xp, K=K, verbose=verbose,
        )
        elapsed_seed = time.time() - t_seed

        seed_results.append({
            "seed": seed,
            "V": V,
            "per_load": {str(L): per_load[L] for L in per_load},
            "batched_vs_scalar_max_diff": float(max_diff),
            "elapsed_seconds": elapsed_seed,
        })
        if verbose:
            print("  [seed " + str(seed) + " done in "
                  + ("%.1fs" % elapsed_seed) + "]", flush=True)

    total_time = time.time() - t0

    # Aggregate multi-seed mean accuracy per load.
    agg: Dict[int, Dict[str, float]] = {}
    for L in list(_DIRECTION_5_LOADS):
        ob_vals = [s["per_load"][str(L)]["order_bearing_accuracy"]
                   for s in seed_results]
        oi_vals = [s["per_load"][str(L)]["order_invariant_accuracy"]
                   for s in seed_results]
        agg[L] = {
            "order_bearing_accuracy_mean": float(np.mean(ob_vals)),
            "order_invariant_accuracy_mean": float(np.mean(oi_vals)),
            "n_seeds": len(seed_results),
        }

    if verbose:
        print("\n=== MULTI-SEED AGGREGATE (topK=" + str(K) + ") ===",
              flush=True)
        header = "            "
        for L in agg:
            header += "L=" + str(L) + " OB    OI    "
        print(header, flush=True)
        line = "  multi-seed:  "
        for L in agg:
            line += ("%.3f" % agg[L]["order_bearing_accuracy_mean"])
            line += " "
            line += ("%.3f" % agg[L]["order_invariant_accuracy_mean"])
            line += "   "
        print(line, flush=True)

    # Verdict via frozen direction_5_verdict module.
    per_seed_for_verdict = []
    for s in seed_results:
        per_load_data = {}
        for L_str, cell in s["per_load"].items():
            per_load_data[int(L_str)] = {
                "OB": cell["order_bearing_accuracy"],
                "OI": cell["order_invariant_accuracy"],
            }
        per_seed_for_verdict.append(per_load_data)
    verdict = compute_verdict(per_seed_for_verdict)
    if verbose:
        print("\n=== VERDICT (frozen, pre-registered) ===", flush=True)
        print("  verdict: " + verdict, flush=True)
        print("  bar: OB>="
              + str(_DIRECTION_5_OB_MIN) + ", OI>="
              + str(_DIRECTION_5_OI_MIN)
              + ", loads=" + str(list(_DIRECTION_5_LOADS))
              + ", seeds_needed=" + str(_DIRECTION_5_MIN_SEEDS),
              flush=True)
        print("  wall: " + ("%.1fs" % total_time)
              + " (backend=" + backend_name + ")", flush=True)

    return {
        "backend": backend_name,
        "gpu": gpu,
        "seeds": seeds,
        "tag": tag,
        "decoder_fix": "top_K_binary",
        "top_K_threshold": K,
        "n_trials": N_TRIALS_LOCAL,
        "loads": list(_DIRECTION_5_LOADS),
        "bar_ob": _DIRECTION_5_OB_MIN,
        "bar_oi": _DIRECTION_5_OI_MIN,
        "min_seeds": _DIRECTION_5_MIN_SEEDS,
        "per_seed": seed_results,
        "aggregate": {str(L): v for L, v in agg.items()},
        "verdict": verdict,
        "wall_clock_seconds": total_time,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Direction 5 Tier 1 decoder-fix probe: "
                    "threshold-to-top-K BEFORE FHRR projection"
    )
    ap.add_argument("--smoke", action="store_true",
                    help="use 'smoke' tag activity caches (default)")
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="seeds; default [42, 43, 44]")
    ap.add_argument("--K", type=int, default=TOP_K_THRESHOLD,
                    help="top-K threshold (default 100; matches "
                         "G.20 sparse K-of-N pattern size)")
    ap.add_argument("--cache-dir", default=None,
                    help="override per-bridge activity cache directory")
    ap.add_argument("--out", default=None,
                    help="output JSON path")
    args = ap.parse_args()

    tag = "smoke"
    result = run_cross_bridge_probe_topK(
        seeds=args.seeds, tag=tag, K=args.K,
        cache_dir=args.cache_dir, verbose=True,
    )

    out_path = args.out or os.path.join(
        _HERE, "direction_5_cross_bridge_topK_" + tag + ".json",
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print("\nWrote " + out_path, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
