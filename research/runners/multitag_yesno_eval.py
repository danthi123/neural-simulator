"""Multi-seed eval for yes/no binary questions.

Tests the chat REPL's 'is a b?' capability across seeds:
- Encode N true pairs (apple:big, dog:small, etc.)
- For each true pair, test 'is a b?' -> expect YES
- For each false pair (cross-cluster or wrong direction), test 'is a b?'
  -> expect NO

Reliability: tag existence check (deterministic) + stim-recall confidence
verification (87.5% per-tag).
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
    p.add_argument("--pairs", type=str,
                    default="apple:big,dog:small,cat:hot,river:cold,big:hot,small:cold")
    # Negative queries: pairs that were NOT trained (expect NO)
    p.add_argument("--negative-queries", type=str,
                    default="apple:cold,dog:hot,cat:cold,river:big,big:cold,small:hot")
    p.add_argument("--encoding-steps", type=int, default=500)
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--drive-steps", type=int, default=100)
    p.add_argument("--sparsity", type=float, default=0.05)
    p.add_argument("--balanced-teacher-pA", type=float, default=500.0)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    pairs = []
    for ps in args.pairs.split(","):
        a, b = ps.strip().split(":")
        if a in _WORD_TO_IDX and b in _WORD_TO_IDX:
            pairs.append((a, b))

    negative_queries = []
    for q in args.negative_queries.split(","):
        a, b = q.strip().split(":")
        if a in _WORD_TO_IDX and b in _WORD_TO_IDX:
            negative_queries.append((a, b))

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

    # Encode all true pairs
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

    def query_is_bound(a, b):
        """Return ('YES'|'PARTIAL'|'NO', confidence)."""
        tag1, tag2 = f"{a}_{b}", f"{b}_{a}"
        if tag1 in encoded_tags or tag2 in encoded_tags:
            actual_tag = tag1 if tag1 in encoded_tags else tag2
            pat, n_lo = lang_output_pattern_during_stim(
                bridge, actual_tag, drive_pA=1500.0, stim_steps=args.drive_steps,
            )
            scores = {w: cosine_to_word(
                pat, w, n_lo,
                n_words_for_orthogonal=args.n_words_for_orthogonal,
                sparsity=args.sparsity,
            ) for w in valid_concepts}
            ranked = sorted(scores.items(), key=lambda kv: -kv[1])
            top5_words = [w for w, _ in ranked[:5]]
            both_in_top5 = (a in top5_words) and (b in top5_words)
            return ("YES" if both_in_top5 else "PARTIAL"), {a: scores[a], b: scores[b]}
        return ("NO", {})

    print(f"\n=== yes/no eval seed={args.seed} ===")
    print(f"  encoded {len(encoded_tags)} pairs")
    print(f"  TRUE queries (expect YES): {pairs}")
    print(f"  NEGATIVE queries (expect NO): {negative_queries}")
    print()

    # Positive cases
    n_true_correct = 0
    pos_results = []
    for a, b in pairs:
        verdict, scores = query_is_bound(a, b)
        correct = (verdict == "YES")
        if correct:
            n_true_correct += 1
        print(f"  {a:6s} {b:6s} (expect YES) -> {verdict} {scores}")
        pos_results.append({"a": a, "b": b, "expected": "YES",
                             "actual": verdict, "scores": scores})

    print()
    n_neg_correct = 0
    neg_results = []
    for a, b in negative_queries:
        verdict, scores = query_is_bound(a, b)
        correct = (verdict == "NO")
        if correct:
            n_neg_correct += 1
        print(f"  {a:6s} {b:6s} (expect NO)  -> {verdict} {scores}")
        neg_results.append({"a": a, "b": b, "expected": "NO",
                             "actual": verdict, "scores": scores})

    print()
    print(f"[VERDICT]")
    print(f"  True positives: {n_true_correct}/{len(pairs)}")
    print(f"  True negatives: {n_neg_correct}/{len(negative_queries)}")
    total = len(pairs) + len(negative_queries)
    correct = n_true_correct + n_neg_correct
    print(f"  Accuracy: {correct}/{total} = {correct/total*100:.1f}%")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({
                "seed": args.seed,
                "pairs": pairs,
                "negative_queries": negative_queries,
                "n_true_correct": n_true_correct,
                "n_neg_correct": n_neg_correct,
                "n_total_true": len(pairs),
                "n_total_neg": len(negative_queries),
                "accuracy": correct / total if total else 0,
                "pos_results": pos_results,
                "neg_results": neg_results,
            }, f, indent=2, default=str)


if __name__ == "__main__":
    main()
