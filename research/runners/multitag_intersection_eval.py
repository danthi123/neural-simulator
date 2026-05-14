"""Multi-seed eval for compositional intersection queries.

Tests the 'what is a AND b' capability:
- Encode N pairs in a clustered graph (apple, cat both bound to big,hot)
- For each cue pair (a, b), expected shared associates are the
  concepts bound to BOTH a and b
- Multitag retrieval for a and b separately, intersect, rank

Built 2026-05-14 to validate the compositional capability on top of
the 90% FULL multitag baseline.
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
    # Default: clustered graph (apple,cat,big,hot) and (dog,river,small,cold)
    p.add_argument("--pairs", type=str,
                    default="apple:big,apple:hot,cat:big,cat:hot,"
                            "dog:small,dog:cold,river:small,river:cold")
    # Intersection queries: each entry is a:b - returns expected shared associates
    # apple AND cat → shared: big, hot
    p.add_argument("--intersection-queries", type=str,
                    default="apple:cat,big:hot,dog:river,small:cold")
    p.add_argument("--encoding-steps", type=int, default=500)
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--drive-steps", type=int, default=100)
    p.add_argument("--sparsity", type=float, default=0.05)
    p.add_argument("--balanced-teacher-pA", type=float, default=500.0)
    p.add_argument("--top-n", type=int, default=3)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    pairs = []
    for ps in args.pairs.split(","):
        a, b = ps.strip().split(":")
        if a in _WORD_TO_IDX and b in _WORD_TO_IDX:
            pairs.append((a, b))

    intersection_queries = []
    for q in args.intersection_queries.split(","):
        a, b = q.strip().split(":")
        if a in _WORD_TO_IDX and b in _WORD_TO_IDX:
            intersection_queries.append((a, b))

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

    # Encode all pairs
    encoded_tags = []
    for a, b in pairs:
        tag = f"{a}_{b}"
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

    # Build cue→associates map from encoded pairs
    cue_to_associates = {}
    for tag in encoded_tags:
        a, b = tag.split("_")
        cue_to_associates.setdefault(a, set()).add(b)
        cue_to_associates.setdefault(b, set()).add(a)

    # For each intersection query, compute:
    # - expected: intersection of cue_to_associates[a] and cue_to_associates[b]
    # - actual: top-N intersection from multitag scores
    print(f"\n=== intersection eval seed={args.seed} ===")
    print(f"  encoded {len(encoded_tags)} pairs")
    print(f"  intersection queries: {intersection_queries}")
    print()

    def multitag_score(cue):
        """Compute per-word multitag score for cue (max across matching tags)."""
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

    results = []
    n_full_pass = 0
    n_partial_pass = 0
    n_total = 0

    for a, b in intersection_queries:
        expected_shared = cue_to_associates.get(a, set()) & cue_to_associates.get(b, set())
        if not expected_shared:
            print(f"  cue=({a:6s} AND {b:6s}) — no expected shared, skipping")
            continue
        n_total += 1
        scores_a = multitag_score(a)
        scores_b = multitag_score(b)
        # Intersection: words in both, ranked by min(score_a, score_b)
        shared = []
        for w in set(scores_a) & set(scores_b):
            shared.append((w, min(scores_a[w], scores_b[w])))
        shared.sort(key=lambda x: -x[1])
        top_n = [w for w, _ in shared[:args.top_n]]

        full_pass = all(e in top_n for e in expected_shared)
        partial_pass = any(e in top_n for e in expected_shared)
        if full_pass:
            n_full_pass += 1
        if partial_pass:
            n_partial_pass += 1

        verdict = "FULL" if full_pass else ("PARTIAL" if partial_pass else "FAIL")
        exp_str = "+".join(sorted(expected_shared))
        top_str = ",".join(top_n)
        print(f"  cue=({a:6s} AND {b:6s}) expected=[{exp_str:12s}] "
              f"top-{args.top_n}=[{top_str:20s}] {verdict}")
        results.append({
            "cue_a": a, "cue_b": b,
            "expected": sorted(expected_shared),
            "top_n": top_n,
            "full_pass": full_pass,
            "partial_pass": partial_pass,
        })

    print()
    print(f"[VERDICT]")
    print(f"  Full pass (all shared in top-{args.top_n}): {n_full_pass}/{n_total}")
    print(f"  Partial pass (any in top-{args.top_n}): {n_partial_pass}/{n_total}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({
                "seed": args.seed, "pairs": pairs,
                "intersection_queries": intersection_queries,
                "top_n": args.top_n,
                "n_full_pass": n_full_pass,
                "n_partial_pass": n_partial_pass,
                "n_total": n_total,
                "results": results,
            }, f, indent=2, default=str)


if __name__ == "__main__":
    main()
