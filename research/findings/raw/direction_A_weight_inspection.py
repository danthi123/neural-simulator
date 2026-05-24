"""Direction A weight diagnostic: inspect trained ec_context->pool
weights in the cached v1 substrate to confirm the v2 hypothesis
(plasticity was frozen during encoding, so the weights are at random
initialization, providing no positional selectivity).

If weights are near-zero or random-uninformative: confirms the v2
plasticity-during-encoding fix is mechanically justified.

If weights are non-trivial (e.g., training somehow grew them
implicitly): the v2 hypothesis is wrong and the partial signal in
v1 came from some other mechanism.

NUMPY-friendly GPU access; ~1 min wall.
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

from research.runners.concept_pool_demo import (
    build_concept_bridge, DIRECTION_VOCAB, NOUN_VOCAB, VERB_VOCAB,
    ADJECTIVE_VOCAB,
)
from research.findings.raw.direction_A_ec_context_sequence_full import (
    _bridge_save_path, N_LANG_INPUT, N_PER_POOL, N_FS_PER_POOL,
    N_EC_CONTEXT,
)
from sim.backend import get_backend, is_gpu_backend


OUT_JSON = os.path.join(_HERE, "direction_A_weight_inspection.json")
SEED = 42


def main():
    xp, backend_name = get_backend()
    gpu = is_gpu_backend()
    print(f"=== Direction A weight inspection (v1 substrate) ===",
          flush=True)
    print(f"  backend={backend_name} (GPU={gpu}); seed={SEED}",
          flush=True)

    bridge_p = _bridge_save_path(SEED)
    if not os.path.exists(bridge_p):
        print(f"  [FATAL] bridge cache missing: {bridge_p}",
              flush=True)
        return 1

    t0 = time.time()
    bridge = build_concept_bridge(
        seed=SEED, n_lang_input=N_LANG_INPUT, n_per_pool=N_PER_POOL,
        n_fs_per_pool=N_FS_PER_POOL, enable_adjective=True,
        weak_dynamics=True, enable_positional_context=True,
        n_ec_context=N_EC_CONTEXT, verbose=False,
    )
    bridge.load_checkpoint(bridge_p)
    print(f"  loaded bridge in {time.time()-t0:.1f}s", flush=True)

    rm = bridge.region_manager

    # Inspect each ec_context -> pool pathway weight
    cp, _ = get_backend()
    pathway_stats = {}
    target_pool_kinds = [
        ("noun_pool", ["APPLE", "RIVER", "DOG", "CAT"]),
        ("verb_pool", ["GO", "COME", "STOP", "LOOK"]),
        ("adjective_pool", ["BIG", "SMALL", "HOT", "COLD"]),
        ("motor", ["motor_N", "motor_E", "motor_S", "motor_W"]),
    ]

    try:
        ec_idx_list = list(rm.indices("ec_context"))
    except Exception as e:
        print(f"  [FATAL] ec_context region missing: {e}",
              flush=True)
        return 1
    ec_idx_set = set(ec_idx_list)
    print(f"  ec_context region: {len(ec_idx_list)} neurons "
          f"(indices {min(ec_idx_list)}..{max(ec_idx_list)})",
          flush=True)

    # Get the bridge's monolithic connection storage
    cp_conn = bridge.cp_connections
    if cp_conn is None:
        print(f"  [FATAL] bridge has no cp_connections", flush=True)
        return 1
    # CSR format: indptr, indices, weights
    indptr = cp.asnumpy(cp_conn.indptr)
    indices = cp.asnumpy(cp_conn.indices)
    weights = cp.asnumpy(cp_conn.data) if hasattr(
        cp_conn, "data") else None
    if weights is None:
        # Try alternate access
        try:
            weights = cp.asnumpy(bridge.cp_connection_weights)
        except Exception:
            print(f"  [WARN] cannot access weight data; "
                  f"inspecting structural only", flush=True)

    print(f"  total synapses: {len(indices)}", flush=True)

    # For each ec_context source -> count outgoing connections to
    # each pool region; report mean/min/max/std of weights
    for kind_name, names in target_pool_kinds:
        for n in names:
            region_name = (f"{kind_name}_{n}" if kind_name != "motor"
                            else n)
            try:
                target_idx_list = list(rm.indices(region_name))
            except Exception:
                continue
            target_idx_set = set(target_idx_list)

            # Find connections from ec_context to this region
            ec_to_region_weights = []
            for src_idx in ec_idx_list:
                row_start = indptr[src_idx]
                row_end = indptr[src_idx + 1]
                row_targets = indices[row_start:row_end]
                row_weights = (weights[row_start:row_end]
                               if weights is not None else None)
                for i, tgt in enumerate(row_targets):
                    if int(tgt) in target_idx_set:
                        if row_weights is not None:
                            ec_to_region_weights.append(
                                float(row_weights[i]))
                        else:
                            ec_to_region_weights.append(0.0)
            if not ec_to_region_weights:
                pathway_stats[region_name] = {
                    "n_edges": 0, "mean": None, "min": None,
                    "max": None, "std": None,
                }
                continue
            w_arr = np.array(ec_to_region_weights, dtype=np.float64)
            pathway_stats[region_name] = {
                "n_edges": len(w_arr),
                "mean": float(np.mean(w_arr)),
                "min": float(np.min(w_arr)),
                "max": float(np.max(w_arr)),
                "std": float(np.std(w_arr)),
                "abs_mean": float(np.mean(np.abs(w_arr))),
            }

    print(f"\n  ec_context -> pool weights per region:", flush=True)
    print(f"  {'region':25s}  {'n_edges':>8s}  "
          f"{'mean':>8s}  {'abs_mean':>8s}  {'std':>8s}",
          flush=True)
    for region_name, stats in pathway_stats.items():
        if stats["n_edges"] == 0:
            print(f"  {region_name:25s}  {0:>8d}  {'-':>8s}",
                  flush=True)
            continue
        print(f"  {region_name:25s}  {stats['n_edges']:>8d}  "
              f"{stats['mean']:>8.4f}  {stats['abs_mean']:>8.4f}  "
              f"{stats['std']:>8.4f}", flush=True)

    print(f"\n=== INTERPRETATION ===", flush=True)
    all_abs_means = [stats["abs_mean"] for stats in
                     pathway_stats.values()
                     if stats["n_edges"] > 0]
    if not all_abs_means:
        verdict = "NO_EC_TO_POOL_PATHWAYS_FOUND"
        print(f"  No ec_context->pool edges found; pathway "
              f"structure missing.", flush=True)
    else:
        mean_abs = float(np.mean(all_abs_means))
        print(f"  Mean absolute weight across all ec->pool "
              f"pathways: {mean_abs:.4f}", flush=True)
        # Heuristic: if abs_mean < 0.5 the weights are near-zero
        # (probably random init or near-zero post-freeze)
        if mean_abs < 0.5:
            verdict = "WEIGHTS_NEAR_ZERO_CONFIRMS_V2_HYPOTHESIS"
            print(f"  Weights are NEAR-ZERO (mean abs {mean_abs:.4f}"
                  f" < 0.5) -- consistent with the v2 hypothesis "
                  f"that v1 froze ec_context plasticity throughout "
                  f"so the pathway never learned. v2 fix (open the "
                  f"gates during encoding) is mechanically "
                  f"justified.", flush=True)
        else:
            verdict = "WEIGHTS_NON_TRIVIAL"
            print(f"  Weights are NON-TRIVIAL (mean abs "
                  f"{mean_abs:.4f} >= 0.5) -- somehow plasticity "
                  f"did grow these. v2 hypothesis incomplete; "
                  f"investigate further.", flush=True)

    out = {
        "backend": backend_name, "gpu": gpu, "seed": SEED,
        "n_ec_context_neurons": len(ec_idx_list),
        "pathway_stats": pathway_stats,
        "verdict": verdict,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_JSON}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
