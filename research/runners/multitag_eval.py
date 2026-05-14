"""Multi-tag cue recall multi-seed evaluation.

For each seed, encode 8 concept-concept pairs, then for each cue
that appears in multiple tags, verify that ALL its trained associates
appear in the multitag retrieval top-N.

The multitag mechanism: for cue X, stim every engram containing X
and aggregate lang_output cosines. The 87.5% per-tag stim-recall
reliability lifts the cue-driven retrieval too.
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
                    default="apple:big,dog:small,cat:hot,river:cold,"
                            "big:hot,small:cold,apple:cat,dog:river")
    p.add_argument("--encoding-steps", type=int, default=500)
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--drive-steps", type=int, default=100)
    p.add_argument("--sparsity", type=float, default=0.05)
    p.add_argument("--balanced-teacher-pA", type=float, default=500.0)
    p.add_argument("--top-n", type=int, default=2,
                    help="Number of top associates required (default 2)")
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    pairs = []
    for ps in args.pairs.split(","):
        a, b = ps.strip().split(":")
        if a in _WORD_TO_IDX and b in _WORD_TO_IDX:
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

    # For each cue word that appears in multiple tags, test multitag recall
    cue_to_associates = {}
    for tag in encoded_tags:
        a, b = tag.split("_")
        cue_to_associates.setdefault(a, []).append(b)
        cue_to_associates.setdefault(b, []).append(a)

    print(f"\n=== multitag eval seed={args.seed} ===")
    print(f"  encoded {len(encoded_tags)} pairs: {encoded_tags}")
    print(f"  test cues (in >= 2 tags): "
          f"{[c for c, a in cue_to_associates.items() if len(a) >= 2]}")
    print()

    results = []
    n_full_pass = 0  # all associates in top-N
    n_partial_pass = 0  # at least one associate in top-N
    n_total = 0

    for cue, expected_associates in cue_to_associates.items():
        if len(expected_associates) < 2:
            continue  # only cues with multiple associates
        n_total += 1

        # Multi-tag: stim every tag containing cue, aggregate
        matching_tags = [t for t in encoded_tags
                         if cue in t.split("_")]
        associate_scores = {}  # word -> best score across matching tags
        associate_via_tag = {}
        for tag in matching_tags:
            pattern, n_lang_out = lang_output_pattern_during_stim(
                bridge, tag, drive_pA=1500.0, stim_steps=args.drive_steps,
            )
            for w in valid_concepts:
                if w == cue:
                    continue
                score = cosine_to_word(
                    pattern, w, n_lang_out,
                    n_words_for_orthogonal=args.n_words_for_orthogonal,
                    sparsity=args.sparsity,
                )
                if score > associate_scores.get(w, -1):
                    associate_scores[w] = score
                    associate_via_tag[w] = tag

        ranked = sorted(associate_scores.items(), key=lambda kv: -kv[1])
        top_n = [w for w, _ in ranked[:args.top_n]]

        full_pass = all(a in top_n for a in expected_associates)
        partial_pass = any(a in top_n for a in expected_associates)
        if full_pass:
            n_full_pass += 1
        if partial_pass:
            n_partial_pass += 1

        verdict = "FULL" if full_pass else ("PARTIAL" if partial_pass else "FAIL")
        expected_str = "+".join(expected_associates)
        top_str = ",".join(top_n)
        print(f"  cue={cue:8s} expected=[{expected_str:12s}] "
              f"top-{args.top_n}=[{top_str:18s}] {verdict}")
        results.append({
            "cue": cue,
            "expected": expected_associates,
            "top_n": top_n,
            "full_pass": full_pass,
            "partial_pass": partial_pass,
            "scores": dict(ranked[:5]),
        })

    print()
    print(f"[VERDICT]")
    print(f"  Full pass (all associates in top-{args.top_n}): {n_full_pass}/{n_total}")
    print(f"  Partial pass (any associate in top-{args.top_n}): {n_partial_pass}/{n_total}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({
                "seed": args.seed, "pairs": pairs,
                "top_n": args.top_n,
                "n_full_pass": n_full_pass,
                "n_partial_pass": n_partial_pass,
                "n_total": n_total,
                "results": results,
            }, f, indent=2, default=str)


if __name__ == "__main__":
    main()
