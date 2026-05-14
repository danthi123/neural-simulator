"""Concept chain inference test: train A<->B and B<->C, query A, check if C retrieved.

Tests transitive semantic memory. If "apple<->big" and "big<->hot" are
trained, can the system infer apple→hot via the intermediate "big"
association? This is a powerful test of the network's distributed
representation — the lang_input → multi-pool STDP and engram
co-firing should propagate associations.
"""
from __future__ import annotations
import argparse
import json

import research.runners.concept_pool_demo as cpd
from research.runners.concept_compose_train import _WORD_TO_IDX, _WORD_TO_POOL
from research.runners.compose_concept_engram import encode_concept_pair
from research.runners.compose_concept_pool_readout import (
    measure_concept_pool_rates, _POOL_TO_WORD,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--load-bridge", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-lang-input", type=int, default=2048)
    p.add_argument("--n-per-pool", type=int, default=200)
    p.add_argument("--n-fs-per-pool", type=int, default=24)
    p.add_argument("--n-words-for-orthogonal", type=int, default=16)
    # Chains: A:B:C means train (A,B) and (B,C), query A, expect C
    p.add_argument("--chains", type=str,
                    default="apple:big:hot,dog:small:cold,cat:hot:big,river:cold:small")
    p.add_argument("--encoding-steps", type=int, default=200)
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--drive-steps", type=int, default=100)
    p.add_argument("--sparsity", type=float, default=0.05)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    chains = []
    for c in args.chains.split(","):
        a, b, ce = c.strip().split(":")
        if all(w in _WORD_TO_IDX and _WORD_TO_IDX[w] < args.n_words_for_orthogonal
                for w in [a, b, ce]):
            chains.append((a, b, ce))

    print(f"=== compose_concept_chain_test (seed={args.seed}) ===")
    print(f"  Chains (A:B:C — train A<->B and B<->C, query A, expect C):")
    for c in chains:
        print(f"    {c[0]} -> {c[1]} -> {c[2]}")
    print()

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

    # Train the chain edges: for each (a, b, c), train (a, b) and (b, c)
    print("[ENCODE] Training chain edges A<->B and B<->C")
    encoded_pairs = set()
    for a, b, c in chains:
        for x, y in [(a, b), (b, c)]:
            if (x, y) in encoded_pairs or (y, x) in encoded_pairs:
                continue
            tag = f"{x}_{y}"
            encode_concept_pair(
                bridge, x, y, tag,
                encoding_steps=args.encoding_steps,
                drive_pA=200.0, sparsity=args.sparsity,
                n_lang_input=args.n_lang_input,
                n_words_for_orthogonal=args.n_words_for_orthogonal,
                region_filter=region_filter, top_k=args.top_k,
                verbose=False,
            )
            encoded_pairs.add((x, y))
            print(f"  trained: {x}<->{y}")

    # Test each chain
    print()
    print("[TEST] Query A, check if C (chained, indirect) appears in top-3")
    print(f"  {'A':8s} {'B (direct)':12s} {'C (chained)':12s} {'top-3 non-A':40s} {'C present?':10s}")
    n_chain_pass = 0
    n_direct_pass = 0
    results = []
    for a, b, c in chains:
        rates = measure_concept_pool_rates(
            bridge, a, all_concept_pools,
            n_lang_input=args.n_lang_input,
            sparsity=args.sparsity,
            n_words_for_orthogonal=args.n_words_for_orthogonal,
            stim_steps=args.drive_steps,
        )
        pool_a = _WORD_TO_POOL[a]
        pool_b = _WORD_TO_POOL[b]
        pool_c = _WORD_TO_POOL[c]
        non_a_ranked = sorted(
            [(p, r) for p, r in rates.items() if p != pool_a],
            key=lambda kv: -kv[1])
        top3_pools = [p for p, _ in non_a_ranked[:3]]
        top3_words = [_POOL_TO_WORD.get(p, p.split('_')[-1])
                       for p, _ in non_a_ranked[:3]]
        direct_passed = pool_b in top3_pools
        chain_passed = pool_c in top3_pools
        if direct_passed:
            n_direct_pass += 1
        if chain_passed:
            n_chain_pass += 1
        marker = "PASS" if chain_passed else "FAIL"
        top3_str = ", ".join(f"{w}={non_a_ranked[i][1]:.2f}"
                              for i, w in enumerate(top3_words))
        print(f"  {a:8s} {b:12s} {c:12s} [{top3_str:40s}] [{marker}]")
        results.append({
            "a": a, "b": b, "c": c,
            "top3_non_a": top3_words,
            "direct_pass": direct_passed,
            "chain_pass": chain_passed,
        })

    print()
    print(f"[VERDICT]")
    print(f"  Direct B in top-3 (A→B): {n_direct_pass}/{len(chains)}")
    print(f"  Chained C in top-3 (A→C via B): {n_chain_pass}/{len(chains)}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"seed": args.seed, "chains": chains,
                        "n_direct_pass": n_direct_pass,
                        "n_chain_pass": n_chain_pass,
                        "n_total": len(chains),
                        "results": results}, f, indent=2)


if __name__ == "__main__":
    main()
