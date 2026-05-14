"""Compose concept readout via INCREMENT from baseline.

Measures each pool's firing during baseline (no drive) vs. during
cue drive. The DIFFERENCE is the "cue-evoked activation" — should
be cleaner signal than absolute firing rate.
"""
from __future__ import annotations
import argparse
import json
import numpy as np

import research.runners.concept_pool_demo as cpd
from research.runners.concept_compose_train import _WORD_TO_IDX, _WORD_TO_POOL
from research.runners.compose_concept_engram import encode_concept_pair
from research.runners.compose_concept_pool_readout import (
    measure_concept_pool_rates, _POOL_TO_WORD,
)
from sim.text_embeddings import orthogonal_drive_pattern


def measure_pool_rates_baseline(bridge, all_pool_names, stim_steps=100):
    """No drive — measure baseline pool firing."""
    from sim.backend import get_backend
    cp, _ = get_backend()
    rm = bridge.region_manager

    pool_arrs = {p: cp.asarray(list(rm.indices(p)), dtype=cp.int64)
                  for p in all_pool_names}
    spike_counts = {p: 0 for p in all_pool_names}

    bridge.cp_external_input_current[:] = 0.0
    bridge.clear_tag_drive()
    for _ in range(30):
        bridge._run_one_simulation_step()

    for _ in range(stim_steps):
        bridge._run_one_simulation_step()
        if hasattr(bridge, "cp_firing_states"):
            firing = bridge.cp_firing_states
            for p, arr in pool_arrs.items():
                spike_counts[p] += int(firing[arr].sum())

    rates = {p: spike_counts[p] / (stim_steps * len(pool_arrs[p]))
              for p in all_pool_names}
    return rates


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--load-bridge", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-lang-input", type=int, default=2048)
    p.add_argument("--n-per-pool", type=int, default=200)
    p.add_argument("--n-fs-per-pool", type=int, default=24)
    p.add_argument("--n-words-for-orthogonal", type=int, default=16)
    p.add_argument("--pairs", type=str,
                    default="apple:big,dog:small,cat:hot,river:cold,go:look,come:stop,big:hot,small:cold")
    p.add_argument("--encoding-steps", type=int, default=200)
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--drive-steps", type=int, default=100)
    p.add_argument("--sparsity", type=float, default=0.05)
    p.add_argument("--enable-cross-pool-concept-pathways", action="store_true",
                    help="v18/v19: build bridge with all-to-all plastic "
                    "pathways between concept pools (required to load v18+ "
                    "checkpoints).")
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    pairs = []
    for ps in args.pairs.split(","):
        a, b = ps.strip().split(":")
        if (a in _WORD_TO_IDX and b in _WORD_TO_IDX
            and _WORD_TO_IDX[a] < args.n_words_for_orthogonal
            and _WORD_TO_IDX[b] < args.n_words_for_orthogonal):
            pairs.append((a, b))

    bridge = cpd.build_concept_bridge(
        seed=args.seed,
        n_lang_input=args.n_lang_input,
        n_per_pool=args.n_per_pool,
        n_fs_per_pool=args.n_fs_per_pool,
        enable_adjective=True,
        weak_dynamics=True,
        enable_direct_verb_to_motor=True,
        enable_cross_pool_concept_pathways=args.enable_cross_pool_concept_pathways,
        verbose=False,
    )
    bridge.load_checkpoint(args.load_bridge)

    region_filter = []
    all_concept_pools = []
    for kind, names in [
        ("noun_pool", ["APPLE", "RIVER", "DOG", "CAT"]),
        ("verb_pool", ["GO", "COME", "STOP", "LOOK"]),
        ("adjective_pool", ["BIG", "SMALL", "HOT", "COLD"]),
    ]:
        for n in names:
            full = f"{kind}_{n}"
            try:
                bridge.region_manager.indices(full)
                region_filter.append(full)
                all_concept_pools.append(full)
            except Exception:
                pass

    # Encode pairs
    for a, b in pairs:
        encode_concept_pair(
            bridge, a, b, f"{a}_{b}",
            encoding_steps=args.encoding_steps,
            drive_pA=200.0, sparsity=args.sparsity,
            n_lang_input=args.n_lang_input,
            n_words_for_orthogonal=args.n_words_for_orthogonal,
            region_filter=region_filter, top_k=args.top_k,
            verbose=False,
        )

    # First measure BASELINE for all pools
    baseline = measure_pool_rates_baseline(bridge, all_concept_pools,
                                              stim_steps=args.drive_steps)

    print(f"  {'cue':10s} {'expect_b':10s} {'top-1 increment':16s} {'verdict':8s}")
    n_top1_pass = 0
    n_top3_pass = 0
    for a, b in pairs:
        rates = measure_concept_pool_rates(
            bridge, a, all_concept_pools,
            n_lang_input=args.n_lang_input,
            sparsity=args.sparsity,
            n_words_for_orthogonal=args.n_words_for_orthogonal,
            stim_steps=args.drive_steps,
        )
        # Compute increment from baseline
        increments = {p: rates[p] - baseline[p] for p in all_concept_pools}
        pool_a = _WORD_TO_POOL[a]
        pool_b = _WORD_TO_POOL[b]
        non_a_ranked = sorted(
            [(p, r) for p, r in increments.items() if p != pool_a],
            key=lambda kv: -kv[1])
        top1_pool = non_a_ranked[0][0]
        top1_word = _POOL_TO_WORD.get(top1_pool, top1_pool.split('_')[-1])
        top3_pools = [p for p, _ in non_a_ranked[:3]]
        top1_passed = (top1_pool == pool_b)
        top3_passed = pool_b in top3_pools
        if top1_passed:
            n_top1_pass += 1
        if top3_passed:
            n_top3_pass += 1
        verdict = "STRICT" if top1_passed else ("TOP3" if top3_passed else "FAIL")
        print(f"  {a:10s} {b:10s} {top1_word:16s} {verdict:8s}")

    print()
    print(f"[VERDICT increment]")
    print(f"  Strict top-1: {n_top1_pass}/{len(pairs)}")
    print(f"  Lenient top-3: {n_top3_pass}/{len(pairs)}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"seed": args.seed, "pairs": pairs,
                        "n_top1_pass": n_top1_pass,
                        "n_top3_pass": n_top3_pass}, f, indent=2)


if __name__ == "__main__":
    main()
