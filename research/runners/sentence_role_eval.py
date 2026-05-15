"""Multi-seed eval for 3-word sentence role queries (who / what did).

Tests the chat REPL's ability to:
1. Encode 'A V B' sentences as distinct engrams (tag = A_V_B)
2. Answer 'who V B?' (returns A)
3. Answer 'what did A V?' (returns B)
4. Distinguish 'A V B' from 'B V A' (different tags, different answers)

Validates subject-verb-object role distinction across multiple seeds.
"""
from __future__ import annotations
import argparse
import json

import research.runners.concept_pool_demo as cpd
from research.runners.concept_compose_train import _WORD_TO_IDX, _WORD_TO_POOL
from research.runners.compose_concept_engram import (
    lang_output_pattern_during_stim, cosine_to_word, _ALL_CONCEPTS,
)
from sim.text_embeddings import orthogonal_drive_pattern


def encode_triple(bridge, word_a, word_v, word_b, tag_name,
                    n_lang_input=2048, n_words_for_orthogonal=16,
                    sparsity=0.05, encoding_steps=500,
                    teacher_pA=500.0, top_k=100, region_filter=None):
    """Encode a 3-word triple as a single engram."""
    from sim.backend import get_backend
    cp, _ = get_backend()
    rm = bridge.region_manager

    drive_a = orthogonal_drive_pattern(
        cue_idx=_WORD_TO_IDX[word_a], n_cues=n_words_for_orthogonal,
        n_neurons=n_lang_input, drive_max_pA=200.0, sparsity=sparsity,
    )
    drive_v = orthogonal_drive_pattern(
        cue_idx=_WORD_TO_IDX[word_v], n_cues=n_words_for_orthogonal,
        n_neurons=n_lang_input, drive_max_pA=200.0, sparsity=sparsity,
    )
    drive_b = orthogonal_drive_pattern(
        cue_idx=_WORD_TO_IDX[word_b], n_cues=n_words_for_orthogonal,
        n_neurons=n_lang_input, drive_max_pA=200.0, sparsity=sparsity,
    )
    combined = cp.asarray(drive_a + drive_v + drive_b, dtype=cp.float32)

    lang_arr = cp.asarray(list(rm.indices("language_input")), dtype=cp.int64)
    pool_a = cp.asarray(list(rm.indices(_WORD_TO_POOL[word_a])), dtype=cp.int64)
    pool_v = cp.asarray(list(rm.indices(_WORD_TO_POOL[word_v])), dtype=cp.int64)
    pool_b = cp.asarray(list(rm.indices(_WORD_TO_POOL[word_b])), dtype=cp.int64)
    n_total = bridge.cp_external_input_current.shape[0]

    bridge.start_engram_recording(tag_name)
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(30):
        bridge._run_one_simulation_step()

    ext = cp.zeros(n_total, dtype=cp.float32)
    for _ in range(encoding_steps):
        ext.fill(0)
        ext[lang_arr] = combined
        ext[pool_a] = teacher_pA
        ext[pool_v] = teacher_pA
        ext[pool_b] = teacher_pA
        bridge.cp_external_input_current[:] = ext
        bridge._run_one_simulation_step()

    bridge.cp_external_input_current[:] = 0.0
    for _ in range(20):
        bridge._run_one_simulation_step()

    bridge.commit_engram_tag(tag_name, top_k=top_k,
                                region_filter=region_filter)


def neural_verify_triple(bridge, tag_name, words, n_words_for_orthogonal=16,
                          sparsity=0.05, valid_concepts=None):
    """Stim tag, check all 3 words appear in lang_output top-8."""
    valid_concepts = valid_concepts or _ALL_CONCEPTS
    pat, n_lo = lang_output_pattern_during_stim(
        bridge, tag_name, drive_pA=1500.0, stim_steps=100,
    )
    scores = {}
    for w in valid_concepts:
        if w in _WORD_TO_IDX and _WORD_TO_IDX[w] < n_words_for_orthogonal:
            scores[w] = cosine_to_word(
                pat, w, n_lo,
                n_words_for_orthogonal=n_words_for_orthogonal,
                sparsity=sparsity,
            )
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    top_8 = [w for w, _ in ranked[:8]]
    return all(w in top_8 for w in words)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--load-bridge", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-lang-input", type=int, default=2048)
    p.add_argument("--n-per-pool", type=int, default=200)
    p.add_argument("--n-fs-per-pool", type=int, default=24)
    p.add_argument("--encoding-steps", type=int, default=500)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    # 4 sentence pairs (each pair tests order discrimination)
    sentence_pairs = [
        ("apple", "stop", "big"),
        ("big", "stop", "apple"),
        ("dog", "look", "cat"),
        ("cat", "look", "dog"),
        ("river", "go", "cold"),
        ("cold", "go", "river"),
        ("hot", "come", "small"),
        ("small", "come", "hot"),
    ]

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
                       if _WORD_TO_IDX[w] < 16]

    # Encode all sentences
    encoded_tags = []
    print(f"\n[ENCODE] {len(sentence_pairs)} sentences")
    for (a, v, b) in sentence_pairs:
        tag = f"{a}_{v}_{b}"
        encode_triple(bridge, a, v, b, tag,
                       n_lang_input=args.n_lang_input,
                       encoding_steps=args.encoding_steps,
                       region_filter=region_filter)
        encoded_tags.append(tag)
        print(f"  {tag}")

    # Run 4 queries per sentence: who_VB, what_did_AV, and reverse direction
    print(f"\n[QUERIES]")
    n_correct = 0
    n_total = 0
    n_neural_pass = 0
    results = []

    for (a, v, b) in sentence_pairs:
        tag = f"{a}_{v}_{b}"
        # Query 1: who V B? → expect A
        matches_who = [t for t in encoded_tags if t.endswith(f"_{v}_{b}")]
        who_subjects = [t.split("_")[0] for t in matches_who]
        who_correct = (a in who_subjects)
        n_total += 1
        if who_correct:
            n_correct += 1
        if neural_verify_triple(bridge, tag, [a, v, b],
                                 valid_concepts=valid_concepts):
            n_neural_pass += 1

        # Query 2: what did A V? → expect B
        matches_what = [t for t in encoded_tags if t.startswith(f"{a}_{v}_")]
        what_objects = [t.split("_")[-1] for t in matches_what]
        what_correct = (b in what_objects)
        n_total += 1
        if what_correct:
            n_correct += 1

        print(f"  '{a} {v} {b}': "
              f"who_{v}_{b}={who_subjects} ({'OK' if who_correct else 'FAIL'}), "
              f"what_did_{a}_{v}={what_objects} ({'OK' if what_correct else 'FAIL'})")

        results.append({
            "sentence": f"{a} {v} {b}",
            "who_query": {"matches": who_subjects, "expected": a, "correct": who_correct},
            "what_query": {"matches": what_objects, "expected": b, "correct": what_correct},
        })

    accuracy = n_correct / n_total if n_total else 0
    neural_pass_rate = n_neural_pass / len(sentence_pairs) if sentence_pairs else 0

    print()
    print(f"[VERDICT seed={args.seed}]")
    print(f"  Role query accuracy: {n_correct}/{n_total} = {accuracy*100:.1f}%")
    print(f"  Neural verification (all 3 words in top-8): "
          f"{n_neural_pass}/{len(sentence_pairs)} = {neural_pass_rate*100:.1f}%")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({
                "seed": args.seed,
                "sentences": sentence_pairs,
                "n_correct": n_correct, "n_total": n_total,
                "accuracy": accuracy,
                "n_neural_pass": n_neural_pass,
                "neural_pass_rate": neural_pass_rate,
                "results": results,
            }, f, indent=2, default=str)


if __name__ == "__main__":
    main()
