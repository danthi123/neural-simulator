"""Strict anti-cheat for concept-concept: top-1 uniqueness.

Previous tests used "b in top-3 non-a" criterion (lenient). This
strict test checks if the trained associate is UNIQUELY the
top-firing non-cue pool — i.e., the system's "best guess" is correct.
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
    p.add_argument("--pairs", type=str,
                    default="apple:big,dog:small,cat:hot,river:cold,go:look,come:stop,big:hot,small:cold")
    p.add_argument("--encoding-steps", type=int, default=200)
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--drive-steps", type=int, default=100)
    p.add_argument("--sparsity", type=float, default=0.05)
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

    # Strict anti-cheat: top-1 of non-cue must be the trained b
    print(f"  {'cue':10s} {'expect_b':10s} {'top-1 non-a':12s} {'verdict':8s}")
    n_top1_pass = 0
    n_top3_pass = 0
    results = []
    for a, b in pairs:
        rates = measure_concept_pool_rates(
            bridge, a, all_concept_pools,
            n_lang_input=args.n_lang_input,
            sparsity=args.sparsity,
            n_words_for_orthogonal=args.n_words_for_orthogonal,
            stim_steps=args.drive_steps,
        )
        pool_a = _WORD_TO_POOL[a]
        pool_b = _WORD_TO_POOL[b]
        non_a_ranked = sorted(
            [(p, r) for p, r in rates.items() if p != pool_a],
            key=lambda kv: -kv[1])
        top1_pool, top1_rate = non_a_ranked[0]
        top1_word = _POOL_TO_WORD.get(top1_pool, top1_pool.split('_')[-1])
        top3_words = [_POOL_TO_WORD.get(p, p.split('_')[-1])
                       for p, _ in non_a_ranked[:3]]
        top1_passed = (top1_pool == pool_b)
        top3_passed = pool_b in [p for p, _ in non_a_ranked[:3]]
        if top1_passed:
            n_top1_pass += 1
        if top3_passed:
            n_top3_pass += 1
        verdict = "STRICT" if top1_passed else ("TOP3" if top3_passed else "FAIL")
        print(f"  {a:10s} {b:10s} {top1_word:12s} {verdict:8s}")
        results.append({
            "cue": a, "expect_b": b,
            "top1": top1_word, "top3": top3_words,
            "top1_pass": top1_passed, "top3_pass": top3_passed,
        })

    print()
    print(f"[VERDICT]")
    print(f"  Strict top-1 (b is the unique best guess): {n_top1_pass}/{len(pairs)}")
    print(f"  Lenient top-3 (b in top-3): {n_top3_pass}/{len(pairs)}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"seed": args.seed, "pairs": pairs,
                        "n_top1_pass": n_top1_pass,
                        "n_top3_pass": n_top3_pass,
                        "n_total": len(pairs),
                        "results": results}, f, indent=2)


if __name__ == "__main__":
    main()
