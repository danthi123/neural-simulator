"""Multi-seed transitive inference eval (corrected from retracted chain test).

Tests: train A↔B and B↔C. Query A. Does C appear in multitag retrieval?

The retracted compose_concept_chain_test used lang_output cosine with
the bug-corrupted architecture. This re-test uses the validated 90%
multitag mechanism + corrected architecture to honestly measure
whether transitive inference emerges.

The forensic finding (2026-05-14) showed that 2nd-degree neighbors
appear in multitag with smaller cosines than direct. Let's quantify
this multi-seed.
"""
from __future__ import annotations
import argparse
import json

import research.runners.concept_pool_demo as cpd
from research.runners.concept_compose_train import _WORD_TO_IDX
from research.runners.compose_concept_engram import (
    encode_concept_pair, lang_output_pattern_during_stim, cosine_to_word,
    _ALL_CONCEPTS,
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
    p.add_argument("--encoding-steps", type=int, default=500)
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--drive-steps", type=int, default=100)
    p.add_argument("--sparsity", type=float, default=0.05)
    p.add_argument("--balanced-teacher-pA", type=float, default=500.0)
    p.add_argument("--top-n", type=int, default=5,
                    help="Top-N for transitive search (use 5 for lenient)")
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    chains = []
    for c in args.chains.split(","):
        a, b, ce = c.strip().split(":")
        if all(w in _WORD_TO_IDX and _WORD_TO_IDX[w] < args.n_words_for_orthogonal
                for w in [a, b, ce]):
            chains.append((a, b, ce))

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
    region_filter = []
    for kind, names in [("noun_pool", ["APPLE", "RIVER", "DOG", "CAT"]),
                         ("verb_pool", ["GO", "COME", "STOP", "LOOK"]),
                         ("adjective_pool", ["BIG", "SMALL", "HOT", "COLD"])]:
        for n in names:
            try:
                rm.indices(f"{kind}_{n}")
                region_filter.append(f"{kind}_{n}")
            except Exception:
                pass

    valid_concepts = [w for w in _ALL_CONCEPTS
                       if _WORD_TO_IDX[w] < args.n_words_for_orthogonal]

    # Encode chain edges: for each (a, b, c), train (a, b) and (b, c)
    encoded_tags = []
    edges = set()
    for a, b, c in chains:
        edges.add((a, b))
        edges.add((b, c))
    for a, b in edges:
        tag = f"{a}_{b}"
        if tag in encoded_tags:
            continue
        encode_concept_pair(
            bridge, a, b, tag,
            encoding_steps=args.encoding_steps,
            drive_pA=200.0, sparsity=args.sparsity,
            n_lang_input=args.n_lang_input,
            n_words_for_orthogonal=args.n_words_for_orthogonal,
            region_filter=region_filter, top_k=args.top_k,
            balanced_teacher_pA=args.balanced_teacher_pA,
            verbose=False,
        )
        encoded_tags.append(tag)

    def multitag_score(cue):
        """Multitag scores for cue (max across matching tags)."""
        matching = [t for t in encoded_tags if cue in t.split("_")]
        if not matching:
            return {}
        scores = {}
        for tag in matching:
            pat, n_lo = lang_output_pattern_during_stim(
                bridge, tag, drive_pA=1500.0, stim_steps=args.drive_steps,
            )
            for w in valid_concepts:
                if w == cue:
                    continue
                s = cosine_to_word(
                    pat, w, n_lo,
                    n_words_for_orthogonal=args.n_words_for_orthogonal,
                    sparsity=args.sparsity,
                )
                if s > scores.get(w, -1):
                    scores[w] = s
        return scores

    print(f"\n=== transitive eval seed={args.seed} ===")
    print(f"  chains: {chains}")
    print(f"  encoded edges: {sorted(edges)}")
    print()

    results = []
    n_direct_pass = 0  # B in cue A's top-N
    n_transitive_pass = 0  # C in cue A's top-N (the indirect one)
    n_total = len(chains)

    for a, b, c in chains:
        scores = multitag_score(a)
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        top_n_words = [w for w, _ in ranked[:args.top_n]]
        b_in_top = b in top_n_words
        c_in_top = c in top_n_words  # transitive: C wasn't direct-trained with A
        if b_in_top:
            n_direct_pass += 1
        if c_in_top:
            n_transitive_pass += 1
        b_score = scores.get(b, 0)
        c_score = scores.get(c, 0)
        verdict = f"direct={'YES' if b_in_top else 'no'} transitive={'YES' if c_in_top else 'no'}"
        print(f"  A={a:6s} B={b:6s} C={c:6s} | "
              f"top-{args.top_n}: {top_n_words[:args.top_n]} | "
              f"B_score={b_score:.3f} C_score={c_score:.3f} | {verdict}")
        results.append({
            "A": a, "B": b, "C": c,
            "top_n": top_n_words,
            "B_score": b_score, "C_score": c_score,
            "direct_pass": b_in_top, "transitive_pass": c_in_top,
        })

    print()
    print(f"[VERDICT]")
    print(f"  Direct (B in cue A top-{args.top_n}): {n_direct_pass}/{n_total}")
    print(f"  Transitive (C in cue A top-{args.top_n}): {n_transitive_pass}/{n_total}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({
                "seed": args.seed,
                "chains": chains,
                "top_n": args.top_n,
                "n_direct_pass": n_direct_pass,
                "n_transitive_pass": n_transitive_pass,
                "n_total": n_total,
                "results": results,
            }, f, indent=2, default=str)


if __name__ == "__main__":
    main()
