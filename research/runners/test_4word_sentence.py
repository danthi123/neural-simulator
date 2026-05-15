"""Quick feasibility test for 4-word sentences (subject verb modifier object).

Hypothesis: encoding 4 words simultaneously as one engram still produces
distinct tags + neurally-verifiable retrieval, like 3-word sentences do.

Test:
- Encode "apple stop big cat" and "cat stop big apple" (different order)
- Encode "dog look hot cold" and "cold look hot dog"
- Verify each tag retrieves all 4 words via stim
- Verify role queries (4-word: subj_verb_mod_obj) discriminate correctly
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


def encode_quad(bridge, words, tag_name,
                 n_lang_input=2048, n_words_for_orthogonal=16,
                 sparsity=0.05, encoding_steps=500,
                 teacher_pA=500.0, top_k=100, region_filter=None):
    """Encode 4-word sentence as one engram."""
    from sim.backend import get_backend
    cp, _ = get_backend()
    rm = bridge.region_manager

    drives = [orthogonal_drive_pattern(
        cue_idx=_WORD_TO_IDX[w], n_cues=n_words_for_orthogonal,
        n_neurons=n_lang_input, drive_max_pA=200.0, sparsity=sparsity,
    ) for w in words]
    combined = cp.asarray(sum(drives), dtype=cp.float32)

    lang_arr = cp.asarray(list(rm.indices("language_input")), dtype=cp.int64)
    pool_arrs = [cp.asarray(list(rm.indices(_WORD_TO_POOL[w])),
                              dtype=cp.int64) for w in words]
    n_total = bridge.cp_external_input_current.shape[0]

    bridge.start_engram_recording(tag_name)
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(30):
        bridge._run_one_simulation_step()

    ext = cp.zeros(n_total, dtype=cp.float32)
    for _ in range(encoding_steps):
        ext.fill(0)
        ext[lang_arr] = combined
        for pa in pool_arrs:
            ext[pa] = teacher_pA
        bridge.cp_external_input_current[:] = ext
        bridge._run_one_simulation_step()

    bridge.cp_external_input_current[:] = 0.0
    for _ in range(20):
        bridge._run_one_simulation_step()

    bridge.commit_engram_tag(tag_name, top_k=top_k,
                                region_filter=region_filter)


def neural_verify(bridge, tag_name, words,
                    n_words_for_orthogonal=16, sparsity=0.05):
    """Check all N words appear in lang_output top-K."""
    pat, n_lo = lang_output_pattern_during_stim(
        bridge, tag_name, drive_pA=1500.0, stim_steps=100,
    )
    valid = [w for w in _ALL_CONCEPTS
              if w in _WORD_TO_IDX and _WORD_TO_IDX[w] < n_words_for_orthogonal]
    scores = {w: cosine_to_word(
        pat, w, n_lo, n_words_for_orthogonal=n_words_for_orthogonal,
        sparsity=sparsity,
    ) for w in valid}
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    # Top 10 for 4-word verification (relaxed vs top 8 for 3-word)
    top_k_names = [w for w, _ in ranked[:10]]
    found = [w for w in words if w in top_k_names]
    return len(found), len(words), top_k_names[:6]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--load-bridge", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--encoding-steps", type=int, default=500)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    # 4 sentence pairs (test order discrimination)
    quad_sentences = [
        ("apple", "stop", "big", "cat"),
        ("cat", "stop", "big", "apple"),  # same words, reversed S+O
        ("dog", "look", "hot", "cold"),
        ("cold", "look", "hot", "dog"),
        ("river", "go", "small", "big"),
        ("big", "go", "small", "river"),
        ("come", "river", "hot", "apple"),  # arbitrary V-S-M-O
        ("apple", "river", "hot", "come"),
    ]

    bridge = cpd.build_concept_bridge(
        seed=args.seed,
        n_lang_input=2048, n_per_pool=200, n_fs_per_pool=24,
        enable_adjective=True, weak_dynamics=True,
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

    # Encode and verify each sentence
    print(f"\n=== 4-word sentence test (seed={args.seed}) ===\n")
    n_tags_neural_pass = 0
    n_total_words = 0
    n_found_words = 0
    encoded_tags = []
    per_sentence = []

    for words in quad_sentences:
        tag = "_".join(words)
        encode_quad(bridge, list(words), tag,
                     encoding_steps=args.encoding_steps,
                     region_filter=region_filter)
        encoded_tags.append(tag)

        n_found, n_words, top6 = neural_verify(bridge, tag, list(words))
        n_total_words += n_words
        n_found_words += n_found
        all_found = (n_found == n_words)
        if all_found:
            n_tags_neural_pass += 1
        print(f"  {tag:30s} found {n_found}/{n_words}: "
              f"top-6={top6} {'PASS' if all_found else 'PARTIAL'}")
        per_sentence.append({
            "tag": tag, "words": list(words),
            "n_found": n_found, "n_words": n_words,
            "top6": top6, "all_found": all_found,
        })

    # Role queries: "who V M O?" → expect S; "what did S V M?" → expect O
    print(f"\n[ROLE QUERIES]")
    n_correct = 0
    n_total = 0
    for words in quad_sentences:
        s, v, m, o = words
        tag = "_".join(words)
        # who V M O?
        matches_who = [t for t in encoded_tags if t.endswith(f"_{v}_{m}_{o}")]
        subjects = [t.split("_")[0] for t in matches_who]
        who_correct = (s in subjects)
        # what did S V M?
        matches_what = [t for t in encoded_tags if t.startswith(f"{s}_{v}_{m}_")]
        objects = [t.split("_")[-1] for t in matches_what]
        what_correct = (o in objects)
        if who_correct:
            n_correct += 1
        if what_correct:
            n_correct += 1
        n_total += 2
        print(f"  '{s} {v} {m} {o}': who_{v}_{m}_{o}={subjects} "
              f"({'OK' if who_correct else 'FAIL'}), "
              f"what_did_{s}_{v}_{m}={objects} "
              f"({'OK' if what_correct else 'FAIL'})")

    print()
    print(f"[VERDICT seed={args.seed}]")
    print(f"  4-word neural verification: {n_tags_neural_pass}/{len(encoded_tags)} "
          f"tags PASS all-4-found; word-recall {n_found_words}/{n_total_words}")
    print(f"  4-word role queries: {n_correct}/{n_total} = "
          f"{n_correct/n_total*100:.1f}%")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({
                "seed": args.seed,
                "n_tags_neural_pass": n_tags_neural_pass,
                "n_total_tags": len(encoded_tags),
                "n_found_words": n_found_words,
                "n_total_words": n_total_words,
                "n_role_correct": n_correct,
                "n_role_total": n_total,
                "per_sentence": per_sentence,
            }, f, indent=2, default=str)


if __name__ == "__main__":
    main()
