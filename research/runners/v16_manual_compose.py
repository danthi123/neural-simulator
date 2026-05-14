"""Architecture validity test for v16: manually set verb_pool -> motor
weights to large values and check if composition emerges.

If this PASSes 3+/4, v16 architecture CAN do composition — the issue
is just that compose-training STDP doesn't grow weights enough.

If this FAILs even with manually-set weights, v16 architecture has
a fundamental issue (e.g., verb_pool activation isn't strong enough
to drive motor through direct pathway given other competing inputs).
"""
from __future__ import annotations
import argparse
import json

import numpy as np

import research.runners.concept_pool_demo as cpd
from research.runners.concept_compose_train import (
    measure_compose_inference, _WORD_TO_IDX, _WORD_TO_POOL,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--load-bridge", required=True,
                    help="v16 bridge (post Phase 1, untouched by compose)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-lang-input", type=int, default=2048)
    p.add_argument("--n-per-pool", type=int, default=200)
    p.add_argument("--n-fs-per-pool", type=int, default=24)
    p.add_argument("--compose-pairs", type=str,
                    default="go:north,come:south,stop:west,look:east")
    p.add_argument("--manual-weight", type=float, default=5.0,
                    help="Weight to install on verb_pool -> motor edges")
    p.add_argument("--sparsity", type=float, default=0.05)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    pairs = [tuple(s.strip().split(":")) for s in args.compose_pairs.split(",")]

    bridge = cpd.build_concept_bridge(
        seed=args.seed,
        n_lang_input=args.n_lang_input,
        n_per_pool=args.n_per_pool,
        n_fs_per_pool=args.n_fs_per_pool,
        enable_adjective=True,
        weak_dynamics=True,
        enable_direct_verb_to_motor=True,
        verbose=False,
    )
    bridge.load_checkpoint(args.load_bridge)

    rm = bridge.region_manager

    # For each (verb, motor) pair, install weight on all existing edges
    # verb_pool_X -> motor_Y in cp_connections
    print(f"[MANUAL] Installing weight={args.manual_weight} on "
          f"{len(pairs)} verb -> motor pathways")
    print()

    from sim.backend import to_host
    indptr = to_host(bridge.cp_connections.indptr)
    indices = to_host(bridge.cp_connections.indices)
    data = to_host(bridge.cp_connections.data)

    for verb_word, motor_word in pairs:
        verb_pool_name = _WORD_TO_POOL[verb_word]  # e.g., "verb_pool_GO"
        motor_pool_name = _WORD_TO_POOL[motor_word]  # e.g., "motor_N"
        verb_idx = list(rm.indices(verb_pool_name))
        motor_idx = np.array(list(rm.indices(motor_pool_name)), dtype=np.int64)

        # Vectorized: for each pre in verb_idx, find edges where post is in motor_idx
        all_edge_indices = []
        for pre in verb_idx:
            start, end = int(indptr[pre]), int(indptr[pre + 1])
            if end > start:
                row_cols = indices[start:end]
                mask = np.isin(row_cols, motor_idx)
                k_positions = np.where(mask)[0] + start
                all_edge_indices.append(k_positions)
        if all_edge_indices:
            edge_arr = np.concatenate(all_edge_indices)
            data[edge_arr] = args.manual_weight
            print(f"  {verb_pool_name} -> {motor_pool_name}: "
                  f"{len(edge_arr)} edges set to {args.manual_weight}",
                  flush=True)
        else:
            print(f"  {verb_pool_name} -> {motor_pool_name}: no edges found!",
                  flush=True)

    # Push back to GPU/backend
    from sim.backend import get_backend, get_sparse_module
    cp, _ = get_backend()
    csp = get_sparse_module()
    new_data = cp.asarray(data, dtype=cp.float32)
    new_indices = cp.asarray(indices, dtype=cp.int32)
    new_indptr = cp.asarray(indptr, dtype=cp.int32)
    new_csr = csp.csr_matrix(
        (new_data, new_indices, new_indptr),
        shape=bridge.cp_connections.shape,
        dtype=cp.float32,
    )
    bridge.cp_connections = new_csr

    # Test inference
    print()
    print(f"[TEST] Composition inference (drive verb alone, manual weights)")
    n_pass = 0
    results = []
    for verb, motor in pairs:
        r = measure_compose_inference(
            bridge, verb, motor,
            n_lang_input=args.n_lang_input,
            sparsity=args.sparsity,
            orthogonal_codes=True,
        )
        results.append(r)
        if r["passed"]:
            n_pass += 1
        marker = "PASS" if r["passed"] else "FAIL"
        print(f"  '{verb}' alone -> {r['expected_pool']:10s} "
              f"target={r['target_rate']:.2f}  "
              f"off={r['max_off']:.2f}/{r['max_off_pool']:10s}  "
              f"ratio={r['ratio']:.2f}x  [{marker}]")

    print()
    print(f"[VERDICT] {n_pass}/{len(pairs)} verbs drive trained motor with manual weights={args.manual_weight}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({
                "seed": args.seed,
                "load_bridge": args.load_bridge,
                "manual_weight": args.manual_weight,
                "results": results,
                "n_pass": n_pass,
                "n_total": len(pairs),
            }, f, indent=2)


if __name__ == "__main__":
    main()
