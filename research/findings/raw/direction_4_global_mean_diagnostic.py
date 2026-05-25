"""Direction 4 GLOBAL_MEAN centring diagnostic.

PURPOSE: characterise whether the Direction 4 NEGATIVE multi-seed
smoke verdict (commit 611027c; OB/OI essentially chance at every load
in {2,3,5}) is driven by the per_bridge_local mean-centring choice
inherited verbatim from the pillar n=95 cross-bridge probe, or by a
deeper substrate-geometry constraint.

Per-bridge-local centring subtracts each bridge's OWN 16-concept mean
from THAT bridge's 16 concepts before deriving phasors. Across the
80-concept cross-bridge union, this removes 5 distinct local biases.
The hypothesis being tested: those 5 distinct local biases create 5
misaligned phasor sub-spaces in the union, so cross-bridge composites
cannot decode their constituent concepts.

GLOBAL_MEAN centring (this module) subtracts ONE shared mean computed
across all 80 concepts (the full 80*M_OBS-row activity matrix) before
deriving phasors per concept. This is mathematically valid here
because all 5 bridges share d_act = n_per_pool * 16 (smoke: 1600;
production planned: 16384 per the spec) AND the deriver uses the
SAME fixed DERIV_SEED=90909 across bridges, so the random complex
projection W_re/W_im is byte-identical for every bridge. Subtracting
a shared mean before projecting through a shared W yields phasors
that live in a single common geometric reference instead of 5
independent ones.

DOES NOT modify:
- direction_4_cross_bridge_probe.py (frozen pillar n=95 byte-unchanged
  primitive; the recorded per_bridge_local result stays as published).
- direction_4_5bridge_smoke.json or direction_4_5bridge_smoke.log
  (the original NEGATIVE result is the system of record).
- direction_4_verdict.py (frozen thresholds; bar stays at 0.80).
- pattern_separation_grounding_probe.make_deriver (substrate primitive).

This module ONLY:
- Re-implements derive_grounded_symbols with global_mean centring.
- Re-runs the parallel-matching decoder + marginal-sum order-invariant
  decoder using the SAME pillar n=95 batched primitive (imported
  unmodified).
- Writes a SEPARATE JSON / log under research/findings/raw/.

If global_mean produces no signal either: bottleneck is the
bio_brain_regions substrate geometry, not the centring choice. That
is an honest diagnostic finding. Do NOT iterate centring variants
further (recipe-chasing is forbidden by the NEGATIVE chain).
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

# Reuse-by-import only. NONE of these are modified by this diagnostic.
from research.findings.raw.direction_4_vocab_spec import (
    DIRECTION_4_BRIDGE_A_WORDS,
    DIRECTION_4_BRIDGE_B_WORDS,
    DIRECTION_4_BRIDGE_C_WORDS,
    DIRECTION_4_BRIDGE_D_WORDS,
    DIRECTION_4_BRIDGE_E_WORDS,
    DIRECTION_4_TOTAL,
)
from research.findings.raw.direction_4_verdict import (
    compute_verdict,
    _DIRECTION_4_OB_MIN,
    _DIRECTION_4_OI_MIN,
    _DIRECTION_4_LOADS,
    _DIRECTION_4_MIN_SEEDS,
)
from research.findings.raw.vocabulary_scaling_run import N_DIM, N_TRIALS
from research.findings.raw.biologized_spiking_mode_unification_parallel_matching_runner import (  # noqa: E501
    DERIV_SEED,
)
from research.findings.raw.pattern_separation_grounding_probe import make_deriver
from research.findings.raw.biologized_spiking_mode_unification_helpers import (
    gamma_slot_positions,
)
from research.findings.raw.cross_bridge_mode_unification_probe import (
    batched_phase_similarity, verify_batched_equivalent_to_scalar,
)
from research.findings.raw.direction_4_cross_bridge_probe import (
    BRIDGES, CACHE_DIR as _DEFAULT_CACHE_DIR, _load_per_bridge_activity,
    _PER_BRIDGE_WORDS,
)
from research.runners.resonate_fire_fhrr import ResonateFireFHRR
from research.runners.spiking_phasor_fhrr import phases_to_spikes
from sim.backend import get_backend, is_gpu_backend, to_host


SEEDS = [42, 43, 44]
N_GAMMA_SLOTS = 7


# -----------------------------------------------------------------------
# OPTION A: global-mean centring across the concatenated 80*M_OBS row
# activity matrix. Mathematically valid here because d_act is uniform
# across bridges (n_per_pool * 16) AND the deriver uses a single fixed
# DERIV_SEED -> identical W_re/W_im across bridges.
# -----------------------------------------------------------------------
def derive_global_mean_grounded_symbols(
    per_bridge_activity: Dict[str, Dict[str, np.ndarray]],
    seed: int,  # unused by the deriver itself (DERIV_SEED is fixed)
    k_vocab_per_concept: Optional[int] = None,
) -> Tuple[
    List[Tuple[str, str]],
    Dict[Tuple[str, str], np.ndarray],
    Dict[str, Any],
]:
    """Build the 80-concept union; mean-centre GLOBALLY across all 80
    concepts (one shared mean across the full 80*M_OBS-row activity
    matrix); derive grounded phasor symbols.

    Args:
        per_bridge_activity: {bridge_name: {word: (M_OBS, d_act)}}.
        seed: not used by deriver (DERIV_SEED is fixed); kept for
              signature symmetry with the per_bridge_local variant.
        k_vocab_per_concept: if not None, use first K observations
                              per word for the consolidated mean.

    Returns:
        all_words: [(bridge, word), ...] in iteration-order union
                    (matches per_bridge_local variant exactly).
        grounded: {(bridge, word): phases_to_spikes(deriver(...))}.
        info: diagnostic stats (d_act_per_bridge, global_mean_norm,
              per_bridge_local_mean_norms, etc).
    """
    # Validate d_act uniformity (required precondition for Option A).
    d_acts: Dict[str, int] = {}
    for bridge in BRIDGES:
        acts = per_bridge_activity[bridge]
        first_w = next(iter(acts))
        d_acts[bridge] = int(acts[first_w].shape[1])
    if len(set(d_acts.values())) != 1:
        raise ValueError(
            "global_mean centring (Option A) requires uniform d_act "
            "across all 5 bridges. Observed: " + repr(d_acts)
        )
    d_act = next(iter(d_acts.values()))

    # Consolidate per (bridge, word): mean across M_OBS observations.
    all_words: List[Tuple[str, str]] = []
    consolidated: Dict[Tuple[str, str], np.ndarray] = {}
    for bridge in BRIDGES:
        acts = per_bridge_activity[bridge]
        words = _PER_BRIDGE_WORDS[bridge]
        for w in words:
            arr = acts[w]
            if k_vocab_per_concept is not None:
                arr = arr[:k_vocab_per_concept]
            consolidated[(bridge, w)] = arr.mean(axis=0)
            all_words.append((bridge, w))

    # GLOBAL MEAN: average of all 80 consolidated concept vectors.
    global_mean = np.mean(
        np.stack([consolidated[k] for k in all_words], axis=0),
        axis=0,
    )

    # Diagnostic stats: compare global_mean to per_bridge_local means.
    per_bridge_local_means: Dict[str, np.ndarray] = {}
    for bridge in BRIDGES:
        words = _PER_BRIDGE_WORDS[bridge]
        per_bridge_local_means[bridge] = np.mean(
            np.stack([consolidated[(bridge, w)] for w in words], axis=0),
            axis=0,
        )
    info = {
        "centring": "global_mean",
        "d_act_per_bridge": d_acts,
        "d_act_uniform": d_act,
        "global_mean_norm": float(np.linalg.norm(global_mean)),
        "per_bridge_local_mean_norms": {
            b: float(np.linalg.norm(per_bridge_local_means[b]))
            for b in BRIDGES
        },
        # cos(global_mean, local_mean) for each bridge: if ~1.0 the
        # local means are aligned with the global mean (centring choice
        # would be equivalent); if << 1.0 the local biases differ
        # systematically (the per_bridge_local choice removes those
        # systematic biases, but at the cost of misaligning the 5
        # phasor sub-spaces).
        "cos_global_to_local_mean": {
            b: float(
                np.dot(global_mean, per_bridge_local_means[b])
                / (
                    np.linalg.norm(global_mean)
                    * np.linalg.norm(per_bridge_local_means[b])
                    + 1e-12
                )
            )
            for b in BRIDGES
        },
        # Mean L2 distance between per-bridge local means and the
        # global mean (how dispersed the per-bridge biases are).
        "l2_dist_global_to_local_mean": {
            b: float(np.linalg.norm(
                per_bridge_local_means[b] - global_mean
            ))
            for b in BRIDGES
        },
    }

    # Derive phasors with a SINGLE shared deriver (d_act, DERIV_SEED)
    # -> identical W_re/W_im for every (bridge, word). This is the
    # essence of Option A's mathematical validity.
    deriver = make_deriver(N_DIM, d_act, DERIV_SEED)
    grounded: Dict[Tuple[str, str], np.ndarray] = {}
    for key in all_words:
        centred = consolidated[key] - global_mean
        grounded[key] = phases_to_spikes(deriver(centred))

    return all_words, grounded, info


# -----------------------------------------------------------------------
# Per-seed cross-bridge probe (parallel-matching decoder + marginal-sum
# order-invariant). Same decoder geometry as per_bridge_local variant;
# only difference: centring choice in derive_global_mean_grounded_symbols.
# -----------------------------------------------------------------------
def run_one_seed_global_mean(
    seed: int,
    per_bridge_activity: Dict[str, Dict[str, np.ndarray]],
    xp,
    loads: List[int],
    n_trials: int,
    k_vocab_per_concept: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[Dict[int, Dict[str, float]], int, float, Dict[str, Any]]:
    """Single-seed cross-bridge probe using global_mean centring."""
    all_words, grounded, centring_info = (
        derive_global_mean_grounded_symbols(
            per_bridge_activity, seed,
            k_vocab_per_concept=k_vocab_per_concept,
        )
    )
    V = len(all_words)

    # Same fail-closed primitive sanity check pillar n=95 uses.
    max_diff, vocab_phase_matrix = verify_batched_equivalent_to_scalar(
        grounded, all_words, xp, rng_seed=seed,
    )
    if verbose:
        print(
            "  [seed " + str(seed) + "] V=" + str(V)
            + " grounded ready (global_mean); batched-vs-scalar max-diff="
            + ("%.2e" % max_diff),
            flush=True,
        )

    positions = gamma_slot_positions(seed, N_GAMMA_SLOTS, N_DIM)
    net = ResonateFireFHRR(N_DIM, np.random.default_rng(seed))
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
            C = net.encode([
                (grounded[items[k]], positions[k]) for k in range(load)
            ])
            unbinds = [net.query(C, positions[k]) for k in range(load)]
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
    return per_load, V, max_diff, centring_info


def run_global_mean_diagnostic(
    seeds: Optional[List[int]] = None,
    loads: Optional[List[int]] = None,
    n_trials: Optional[int] = None,
    tag: str = "smoke",
    cache_dir: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Multi-seed cross-bridge probe driver with GLOBAL_MEAN centring."""
    if seeds is None:
        seeds = list(SEEDS)
    if loads is None:
        loads = list(_DIRECTION_4_LOADS)
    if n_trials is None:
        n_trials = N_TRIALS
    cache_dir = cache_dir or _DEFAULT_CACHE_DIR

    xp, backend_name = get_backend()
    gpu = is_gpu_backend()
    if verbose:
        print("=== Direction 4 GLOBAL_MEAN centring diagnostic ===",
              flush=True)
        print(
            "  backend=" + backend_name + " (GPU=" + str(gpu) + "); "
            "seeds=" + str(seeds) + "; loads=" + str(loads)
            + "; bridges=" + str(list(BRIDGES))
            + "; V_total=" + str(DIRECTION_4_TOTAL)
            + "; tag=" + tag,
            flush=True,
        )
        print(
            "  centring=global_mean (single shared mean across 80 "
            "concepts; Option A); decoder = parallel-matching batched "
            "(same primitive as per_bridge_local variant)",
            flush=True,
        )

    t0 = time.time()
    seed_results: List[Dict[str, Any]] = []
    for seed in seeds:
        if verbose:
            print("\n--- seed " + str(seed) + " ---", flush=True)
        t_seed = time.time()
        per_bridge_activity: Dict[str, Dict[str, np.ndarray]] = {}
        for bridge in BRIDGES:
            per_bridge_activity[bridge] = _load_per_bridge_activity(
                bridge, seed, tag,
            )
            if verbose:
                acts = per_bridge_activity[bridge]
                first_w = next(iter(acts))
                d_act = acts[first_w].shape[1]
                m_obs = acts[first_w].shape[0]
                print(
                    "  [seed " + str(seed) + "] loaded " + bridge
                    + " (M_OBS=" + str(m_obs)
                    + " d_act=" + str(d_act)
                    + " V_bridge=" + str(len(acts)) + ")",
                    flush=True,
                )
        per_load, V, max_diff, centring_info = run_one_seed_global_mean(
            seed=seed,
            per_bridge_activity=per_bridge_activity,
            xp=xp,
            loads=loads,
            n_trials=n_trials,
            verbose=verbose,
        )
        seed_results.append({
            "seed": seed, "V": V,
            "batched_vs_scalar_max_diff": max_diff,
            "centring_info": centring_info,
            "per_load": {str(l): v for l, v in per_load.items()},
            "verdict_entry": {
                ("L=" + str(l)): {
                    "OB": per_load[l]["order_bearing_accuracy"],
                    "OI": per_load[l]["order_invariant_accuracy"],
                }
                for l in loads
            },
        })
        if verbose:
            print(
                "  [seed " + str(seed) + " done in "
                + ("%.1f" % (time.time() - t_seed)) + "s]",
                flush=True,
            )
    total_time = time.time() - t0

    # Multi-seed aggregate
    agg: Dict[int, Dict[str, Any]] = {}
    cells_str_parts = []
    for load in loads:
        obs = [
            r["per_load"][str(load)]["order_bearing_accuracy"]
            for r in seed_results
        ]
        ois = [
            r["per_load"][str(load)]["order_invariant_accuracy"]
            for r in seed_results
        ]
        ob_m = float(np.mean(obs))
        oi_m = float(np.mean(ois))
        agg[load] = {
            "order_bearing_mean": ob_m,
            "order_bearing_per_seed": obs,
            "order_invariant_mean": oi_m,
            "order_invariant_per_seed": ois,
        }
        cells_str_parts.append(("%.3f" % ob_m) + " " + ("%.3f" % oi_m))

    # Load per_bridge_local result from the existing recorded smoke JSON
    # for side-by-side comparison. Read-only; never mutates that file.
    per_bridge_local_aggregate: Dict[str, Any] = {}
    smoke_p = os.path.join(_HERE, "direction_4_5bridge_smoke.json")
    if os.path.exists(smoke_p):
        try:
            with open(smoke_p, "r", encoding="utf-8") as f:
                smoke = json.load(f)
            for l in loads:
                key = str(l)
                if key in smoke.get("probe_result", {}).get("aggregate", {}):
                    a = smoke["probe_result"]["aggregate"][key]
                    per_bridge_local_aggregate[key] = {
                        "order_bearing_mean": a["order_bearing_mean"],
                        "order_invariant_mean": a["order_invariant_mean"],
                        "order_bearing_per_seed": a["order_bearing_per_seed"],
                        "order_invariant_per_seed": a["order_invariant_per_seed"],
                    }
        except Exception as e:  # noqa: BLE001
            per_bridge_local_aggregate = {"error": str(e)}

    if verbose:
        print("\n=== MULTI-SEED AGGREGATE (global_mean) ===", flush=True)
        loads_header = "   ".join(
            "L=" + str(l) + " OB    OI    " for l in loads
        )
        print("            " + loads_header, flush=True)
        print("  multi-seed:  " + "   ".join(cells_str_parts), flush=True)

        if per_bridge_local_aggregate and "error" not in per_bridge_local_aggregate:
            print("\n=== COMPARISON: per_bridge_local vs global_mean ===",
                  flush=True)
            print("  per_bridge_local result (from recorded smoke JSON):",
                  flush=True)
            for l in loads:
                a = per_bridge_local_aggregate[str(l)]
                print(
                    "    L=" + str(l)
                    + ": OB=" + ("%.3f" % a["order_bearing_mean"])
                    + " OI=" + ("%.3f" % a["order_invariant_mean"]),
                    flush=True,
                )
            print("  global_mean result (this diagnostic):", flush=True)
            for l in loads:
                a = agg[l]
                print(
                    "    L=" + str(l)
                    + ": OB=" + ("%.3f" % a["order_bearing_mean"])
                    + " OI=" + ("%.3f" % a["order_invariant_mean"]),
                    flush=True,
                )
            print("  delta (global_mean - per_bridge_local):", flush=True)
            for l in loads:
                a_g = agg[l]
                a_l = per_bridge_local_aggregate[str(l)]
                d_ob = (a_g["order_bearing_mean"]
                        - a_l["order_bearing_mean"])
                d_oi = (a_g["order_invariant_mean"]
                        - a_l["order_invariant_mean"])
                print(
                    "    L=" + str(l)
                    + ": dOB=" + ("%+.3f" % d_ob)
                    + " dOI=" + ("%+.3f" % d_oi),
                    flush=True,
                )

        print(
            "\nTotal wall-clock: " + ("%.1f" % total_time)
            + "s (backend=" + backend_name + ")",
            flush=True,
        )

    # Frozen verdict (same module, frozen thresholds).
    verdict_input = [r["verdict_entry"] for r in seed_results]
    verdict_g = compute_verdict(verdict_input)
    if verbose:
        print("\n=== VERDICT (frozen module; for diagnostic comparison) ===",
              flush=True)
        print("  global_mean verdict: " + verdict_g, flush=True)
        if per_bridge_local_aggregate and "error" not in per_bridge_local_aggregate:
            pbl_verdict = (
                "DIRECTION_4_NEGATIVE" if all(
                    a["order_bearing_mean"] < _DIRECTION_4_OB_MIN
                    and a["order_invariant_mean"] < _DIRECTION_4_OI_MIN
                    for a in per_bridge_local_aggregate.values()
                ) else "see direction_4_5bridge_smoke.json"
            )
            print("  per_bridge_local verdict (from recorded smoke): "
                  + pbl_verdict, flush=True)

    # Decide diagnostic interpretation tag.
    any_global_cell_above_threshold = any(
        agg[l]["order_bearing_mean"] > 0.20
        or agg[l]["order_invariant_mean"] > 0.20
        for l in loads
    )
    if any_global_cell_above_threshold:
        diagnostic_tag = "GLOBAL_MEAN_HELPS"
    else:
        diagnostic_tag = "GLOBAL_MEAN_DOES_NOT_HELP"

    return {
        "diagnostic": "direction_4_global_mean_centring",
        "diagnostic_tag": diagnostic_tag,
        "interpretation_threshold": 0.20,
        "interpretation_note": (
            "GLOBAL_MEAN_HELPS: any cell > 0.20 implies per_bridge_local "
            "centring was the constraint -> next-stage probe should use "
            "global_mean centring. GLOBAL_MEAN_DOES_NOT_HELP (all cells "
            "<= 0.20): bottleneck is bio_brain_regions substrate geometry, "
            "not the centring choice. Substantial redesign (e.g. sparse "
            "Kanerva-style coding per pillar n=95) would be needed."
        ),
        "backend": backend_name, "gpu": gpu,
        "tag": tag,
        "seeds": seeds, "loads": loads,
        "bar_ob": _DIRECTION_4_OB_MIN, "bar_oi": _DIRECTION_4_OI_MIN,
        "min_seeds": _DIRECTION_4_MIN_SEEDS,
        "V": DIRECTION_4_TOTAL,
        "n_bridges": len(BRIDGES),
        "bridges": list(BRIDGES),
        "decoder_order_bearing": "parallel_population_matching_batched",
        "decoder_order_invariant": "marginal_sum_phase_similarity_batched",
        "mean_centring": "global_mean",
        "centring_option": "A (single shared mean across 80*M_OBS rows)",
        "substrate": "bio_brain_regions_5bridge_ensemble_v14v16_recipe",
        "per_seed": seed_results,
        "aggregate_global_mean": {str(l): v for l, v in agg.items()},
        "aggregate_per_bridge_local_for_comparison": (
            per_bridge_local_aggregate
        ),
        "verdict_global_mean": verdict_g,
        "wall_clock_seconds": total_time,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Direction 4 GLOBAL_MEAN centring diagnostic "
                    "(operates on cached smoke activity; no retraining)",
    )
    ap.add_argument(
        "--tag", default="smoke",
        help="cache file tag ('smoke' or 'full'); default 'smoke'",
    )
    ap.add_argument(
        "--seeds", type=int, nargs="+", default=None,
        help="seeds to probe; default [42, 43, 44]",
    )
    ap.add_argument(
        "--cache-dir", default=None,
        help="override per-bridge activity cache directory",
    )
    ap.add_argument(
        "--out", default=None,
        help="output JSON path (default: side-by-side with this module)",
    )
    args = ap.parse_args()

    result = run_global_mean_diagnostic(
        seeds=args.seeds,
        tag=args.tag,
        cache_dir=args.cache_dir,
        verbose=True,
    )

    out_path = args.out or os.path.join(
        _HERE, "direction_4_global_mean_diagnostic_" + args.tag + ".json",
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print("\nWrote " + out_path, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
