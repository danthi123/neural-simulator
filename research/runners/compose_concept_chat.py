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
    lang_output_pattern_during_stim, cosine_to_word, _ALL_CONCEPTS,
)
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
                    default="apple:big,dog:small,cat:hot,river:cold,"
                            "big:hot,small:cold,apple:cat,dog:river",
                    help="Train these concept-concept associations")
    p.add_argument("--encoding-steps", type=int, default=500,
                    help="Encoding events per pair (default 500 for 87.5% "
                    "stim-recall recipe; was 200 before bug discovery)")
    p.add_argument("--balanced-teacher-pA", type=float, default=500.0,
                    help="Teacher current on both concept pools during "
                    "encoding (default 500 pA for 87.5% stim-recall)")
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--drive-steps", type=int, default=100)
    p.add_argument("--sparsity", type=float, default=0.05)
    p.add_argument("--scripted", type=str, default=None,
                    help="Comma-separated list of test inputs (skips "
                    "interactive). Cue mode for plain words, /stim <tag> "
                    "for stim-recall mode.")
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

    # IMPORTANT: do NOT freeze plasticity BEFORE encoding. Cross-pool
    # association weights (lang_input -> non-target pool) need active STDP
    # during engram encoding for the associative recall to work later.
    # We freeze gates AFTER encoding completes, before the chat loop.

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
    print(f"  recipe: {args.encoding_steps} events + teacher {args.balanced_teacher_pA} pA "
          f"(2026-05-14 validated 87.5% stim-recall multi-seed)", flush=True)
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
        print(f"  learned: '{a}' <-> '{b}' (tag: {tag})", flush=True)

    # Now freeze plasticity for inference stability (chat loop)
    for g in [
        "language_input_to_motor", "language_input_to_verb_pool",
        "language_input_to_noun_pool", "language_input_to_adjective_pool",
        "motor_to_language_output", "verb_pool_to_language_output",
        "noun_pool_to_language_output", "adjective_pool_to_language_output",
    ]:
        try:
            bridge.set_plasticity_gate(g, 0.0)
        except Exception:
            pass

    print()
    print("=" * 60)
    print("CONCEPT CHAT")
    print(f"Vocab: {valid_concepts}")
    print(f"Learned associations: {pairs}")
    print(f"Encoded tags: {encoded_tags}")
    print()
    print("Commands:")
    print("  <word>           Multi-tag recall: auto-stim all tags with word")
    print("                   (leverages 87.5% stim-recall per tag)")
    print("  /stim <tag>      Direct tag stim-recall (87.5% multi-seed)")
    print("  /cue <word>      Raw cue-pool firing rank (~28%; experimental)")
    print("  /tags            List all encoded engram tags")
    print("  quit             Exit")
    print("=" * 60, flush=True)

    # All concept pools (for pool-firing readout)
    all_concept_pools = [_WORD_TO_POOL[w] for w in valid_concepts]

    def handle(word):
        if word not in _WORD_TO_IDX or _WORD_TO_IDX[word] >= args.n_words_for_orthogonal:
            return None
        t0 = time.time()
        # Cue mode: drive lang_input alone, rank concept pools (27.5% multi-seed)
        rates = measure_concept_pool_rates(
            bridge, word, all_concept_pools,
            n_lang_input=args.n_lang_input,
            sparsity=args.sparsity,
            n_words_for_orthogonal=args.n_words_for_orthogonal,
            stim_steps=args.drive_steps,
        )
        pool_self = _WORD_TO_POOL[word]
        non_self_ranked = sorted(
            [(p, r) for p, r in rates.items() if p != pool_self],
            key=lambda kv: -kv[1])
        top3 = [(_POOL_TO_WORD.get(p, p.split('_')[-1]), r)
                 for p, r in non_self_ranked[:3]]
        return {
            "mode": "cue",
            "self_rate": rates[pool_self],
            "top3_non_self": top3,
            "elapsed_s": time.time() - t0,
        }

    def handle_multitag(cue_word):
        """Multi-tag aggregation: stim every engram tag containing this
        word and aggregate the lang_output cosines. This combines all
        learned associations for a single cue into a ranked list — the
        chat REPL equivalent of "what comes to mind when you hear X".

        Built 2026-05-14 to provide cue-driven retrieval at 87.5%-class
        reliability by leveraging stim-recall mechanism for each tag
        that contains the cue, rather than relying on weak cross-pool
        plastic weights.
        """
        if cue_word not in _WORD_TO_IDX:
            return None
        t0 = time.time()
        # Find all tags containing this cue word
        matching_tags = []
        for tag in encoded_tags:
            try:
                a_word, b_word = tag.split("_")
                if cue_word == a_word or cue_word == b_word:
                    other = b_word if cue_word == a_word else a_word
                    matching_tags.append((tag, other))
            except ValueError:
                pass
        if not matching_tags:
            return {"mode": "multitag", "cue": cue_word, "matches": [],
                     "associates": [], "elapsed_s": time.time() - t0}
        # For each matching tag, stim and read lang_output
        # Aggregate by averaging the cosine to each associate
        associate_scores = {}  # word → list of scores
        for tag, other_word in matching_tags:
            pattern, n_lang_out = lang_output_pattern_during_stim(
                bridge, tag, drive_pA=1500.0, stim_steps=args.drive_steps,
            )
            # Cosine to each vocab word in pool
            for w in valid_concepts:
                if w == cue_word:
                    continue  # skip self
                score = cosine_to_word(
                    pattern, w, n_lang_out,
                    n_words_for_orthogonal=args.n_words_for_orthogonal,
                    sparsity=args.sparsity,
                )
                associate_scores.setdefault(w, []).append((tag, other_word, score))
        # Rank associates: max score per associate (best matching tag)
        ranked = []
        for w, hits in associate_scores.items():
            best_score = max(h[2] for h in hits)
            best_tag = max(hits, key=lambda h: h[2])[0]
            n_hits = sum(1 for h in hits if h[2] > 0.1)
            ranked.append((w, best_score, best_tag, n_hits))
        ranked.sort(key=lambda x: -x[1])
        return {
            "mode": "multitag",
            "cue": cue_word,
            "matches": [t for t, _ in matching_tags],
            "associates": ranked[:5],
            "elapsed_s": time.time() - t0,
        }

    def handle_stim(tag_name):
        """Stim-recall: stimulate engram tag, read lang_output spelling.
        This is the 87.5% validated mode (2026-05-14)."""
        if tag_name not in encoded_tags:
            return None
        t0 = time.time()
        pattern, n_lang_out = lang_output_pattern_during_stim(
            bridge, tag_name, drive_pA=1500.0, stim_steps=args.drive_steps,
        )
        # Rank all 16 vocab words by cosine to lang_output pattern
        scores = {}
        for w in valid_concepts:
            scores[w] = cosine_to_word(
                pattern, w, n_lang_out,
                n_words_for_orthogonal=args.n_words_for_orthogonal,
                sparsity=args.sparsity,
            )
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        top5 = ranked[:5]
        # Expected: both A and B from "A_B" tag
        try:
            a_word, b_word = tag_name.split("_")
        except ValueError:
            a_word, b_word = None, None
        return {
            "mode": "stim",
            "tag": tag_name,
            "a_word": a_word,
            "b_word": b_word,
            "a_score": scores.get(a_word, 0.0) if a_word else 0.0,
            "b_score": scores.get(b_word, 0.0) if b_word else 0.0,
            "top5": top5,
            "elapsed_s": time.time() - t0,
        }

    def print_result(r):
        if r is None:
            print(f"  [unknown input]", flush=True)
            return
        if r["mode"] == "cue":
            associations = ", ".join(f"{w}={s:.2f}" for w, s in r["top3_non_self"])
            print(f"  [cue mode, ~28% multi-seed]", flush=True)
            print(f"  self: {r['self_rate']:.2f}", flush=True)
            print(f"  associates: [{associations}]", flush=True)
        elif r["mode"] == "multitag":
            if not r["matches"]:
                print(f"  [multitag] no engram tag contains '{r['cue']}'",
                      flush=True)
                return
            print(f"  [multitag, leverages 87.5% stim-recall per tag]",
                  flush=True)
            print(f"  cue: {r['cue']}", flush=True)
            print(f"  matched {len(r['matches'])} tag(s): {r['matches']}",
                  flush=True)
            print(f"  top-5 associates (best-tag cosine):", flush=True)
            for w, score, tag, n_hits in r["associates"]:
                marker = "***" if n_hits >= 2 else ("**" if n_hits >= 1 else "")
                print(f"    {w:8s} = {score:.3f} via {tag:20s} {marker}",
                      flush=True)
        elif r["mode"] == "stim":
            print(f"  [stim mode, 87.5% multi-seed] tag={r['tag']}", flush=True)
            print(f"  expected: {r['a_word']} + {r['b_word']}", flush=True)
            print(f"  a_score: {r['a_score']:.3f}   b_score: {r['b_score']:.3f}", flush=True)
            top5_str = ", ".join(f"{w}={s:.2f}" for w, s in r["top5"])
            print(f"  top-5 lang_output: [{top5_str}]", flush=True)
            both_in_top5 = (r["a_word"] in [w for w, _ in r["top5"]] and
                            r["b_word"] in [w for w, _ in r["top5"]])
            print(f"  verdict: {'PASS (both in top-5)' if both_in_top5 else 'PARTIAL/FAIL'}",
                  flush=True)
        print(f"  [{r['elapsed_s']:.1f}s]", flush=True)

    def dispatch(line):
        """Parse one chat line; return result dict or None for command."""
        line = line.strip().lower()
        if not line or line in ("quit", "exit"):
            return "EXIT"
        if line in ("/tags", "tags"):
            print(f"  tags: {encoded_tags}", flush=True)
            return None
        if line.startswith("/stim "):
            tag_arg = line[len("/stim "):].strip()
            r = handle_stim(tag_arg)
            print_result(r)
            return None
        if line.startswith("/cue "):
            word = line[len("/cue "):].strip()
            r = handle(word)
            print_result(r)
            return None
        # plain word -> multitag mode (the recommended cue retrieval)
        r = handle_multitag(line)
        print_result(r)
        return None

    if args.scripted:
        inputs = [s.strip() for s in args.scripted.split(",") if s.strip()]
        for inp in inputs:
            print(f"\n> {inp}", flush=True)
            if dispatch(inp) == "EXIT":
                break
    else:
        print("Ready.", flush=True)
        while True:
            try:
                line = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if dispatch(line) == "EXIT":
                break

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
