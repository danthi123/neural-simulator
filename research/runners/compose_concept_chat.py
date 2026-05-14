"""Concept-concept chat REPL — user types a concept, system replies with associated concept.

NO motor routing. Output is via lang_output cosine to concept words.
This is the real semantic-memory chat (vs motor-direction chat).

Usage:
  python -m research.runners.compose_concept_chat \\
    --load-bridge .../seed42_v16.simstate.h5 \\
    --seed 42 \\
    --pairs "apple:big,dog:small,cat:hot,river:cold,big:hot,small:cold" \\
    --scripted "apple,dog,big,small,hot,cat,go"

User input: any concept word.
System response: drives lang_input(word), reads lang_output, returns
top-3 concept words that come to mind.
"""
from __future__ import annotations
import argparse
import sys
import time
import numpy as np

import research.runners.concept_pool_demo as cpd
from research.runners.concept_compose_train import _WORD_TO_IDX, _WORD_TO_POOL
from research.runners.compose_concept_engram import (
    encode_concept_pair, lang_output_pattern_during_input,
    cosine_to_word, _ALL_CONCEPTS,
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
                            "big:hot,small:cold,apple:cat,dog:river",
                    help="Train these concept-concept associations")
    p.add_argument("--encoding-steps", type=int, default=200)
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--drive-steps", type=int, default=100)
    p.add_argument("--sparsity", type=float, default=0.05)
    p.add_argument("--scripted", type=str, default=None)
    args = p.parse_args()

    # Parse pairs
    pairs = []
    for ps in args.pairs.split(","):
        a, b = ps.strip().split(":")
        if a in _WORD_TO_IDX and b in _WORD_TO_IDX:
            pairs.append((a, b))

    print(f"Loading bridge: {args.load_bridge}", flush=True)
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

    # Freeze plasticity
    for g in [
        "language_input_to_motor", "language_input_to_verb_pool",
        "language_input_to_noun_pool", "language_input_to_adjective_pool",
        "motor_to_language_output", "verb_pool_to_language_output",
        "noun_pool_to_language_output", "adjective_pool_to_language_output",
        "motor_FS_to_motor", "verb_pool_FS_to_verb_pool",
        "noun_pool_FS_to_noun_pool", "adjective_pool_FS_to_adjective_pool",
    ]:
        try:
            bridge.set_plasticity_gate(g, 0.0)
        except Exception:
            pass

    # Region filter: concept pools only (no motor)
    rm = bridge.region_manager
    region_filter = []
    for kind, name in [("noun_pool", ["APPLE", "RIVER", "DOG", "CAT"]),
                        ("verb_pool", ["GO", "COME", "STOP", "LOOK"]),
                        ("adjective_pool", ["BIG", "SMALL", "HOT", "COLD"])]:
        for n in name:
            try:
                rm.indices(f"{kind}_{n}")
                region_filter.append(f"{kind}_{n}")
            except Exception:
                pass

    # Concepts that are in the bridge's vocab range
    valid_concepts = [w for w in _ALL_CONCEPTS
                       if _WORD_TO_IDX[w] < args.n_words_for_orthogonal]

    print(f"\nEncoding {len(pairs)} concept-concept associations...", flush=True)
    for a, b in pairs:
        tag = f"{a}_{b}"
        encode_concept_pair(
            bridge, a, b, tag,
            encoding_steps=args.encoding_steps,
            drive_pA=200.0, sparsity=args.sparsity,
            n_lang_input=args.n_lang_input,
            n_words_for_orthogonal=args.n_words_for_orthogonal,
            region_filter=region_filter, top_k=args.top_k,
            verbose=False,
        )
        print(f"  learned: '{a}' <-> '{b}'", flush=True)

    print()
    print("=" * 60)
    print("CONCEPT CHAT — type a concept word, system associates")
    print(f"Vocab: {valid_concepts}")
    print(f"Learned associations: {pairs}")
    print("Type 'quit' to exit")
    print("=" * 60, flush=True)

    def handle(word):
        if word not in _WORD_TO_IDX or _WORD_TO_IDX[word] >= args.n_words_for_orthogonal:
            return None
        t0 = time.time()
        pat, n_lo = lang_output_pattern_during_input(
            bridge, word,
            n_lang_input=args.n_lang_input, sparsity=args.sparsity,
            n_words_for_orthogonal=args.n_words_for_orthogonal,
            stim_steps=args.drive_steps,
        )
        scores = {w: cosine_to_word(
            pat, w, n_lo, n_words_for_orthogonal=args.n_words_for_orthogonal,
            sparsity=args.sparsity,
        ) for w in valid_concepts}
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        non_self = [(w, s) for w, s in ranked if w != word]
        return {
            "top_1_overall": ranked[0],
            "top_3_non_self": non_self[:3],
            "self_score": scores[word],
            "elapsed_s": time.time() - t0,
        }

    if args.scripted:
        inputs = [s.strip() for s in args.scripted.split(",") if s.strip()]
        for inp in inputs:
            print(f"\n> {inp}", flush=True)
            r = handle(inp)
            if r is None:
                print(f"  [unknown: '{inp}']", flush=True)
                continue
            top1, _ = r["top_1_overall"]
            associations = ", ".join(f"{w}={s:.2f}" for w, s in r["top_3_non_self"])
            print(f"  spelling: {top1} (cos={r['top_1_overall'][1]:.2f})", flush=True)
            print(f"  associated: [{associations}]", flush=True)
            print(f"  [{r['elapsed_s']:.1f}s]", flush=True)
    else:
        print("Ready.", flush=True)
        while True:
            try:
                line = input("> ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if line in ("quit", "exit", ""):
                break
            r = handle(line)
            if r is None:
                print(f"  [unknown word]")
                continue
            top1, _ = r["top_1_overall"]
            associations = ", ".join(f"{w}={s:.2f}" for w, s in r["top_3_non_self"])
            print(f"  spelling: {top1}")
            print(f"  associated: [{associations}]")

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
