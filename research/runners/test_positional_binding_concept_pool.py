"""Test positional binding on concept_pool architecture (catalog D.01/D.02/D.11).

Hypothesis: with ec_context region driving positional patterns, the engram
tag for (word, position) is DISTINCT from (word, other_position).

If this works, we can encode sentences as position-tagged engrams and
distinguish "alice ate apple" from "apple ate alice".

Test procedure:
1. Build concept-pool bridge with enable_positional_context=True
2. For each test word w:
   - Encode (w, pos=0): drive lang_input(w) + ec_context(pos_pattern_for_0)
     → tag "w_pos0"
   - Encode (w, pos=2): drive lang_input(w) + ec_context(pos_pattern_for_2)
     → tag "w_pos2"
3. Compare tag overlap (Jaccard index)
4. PASS if overlap < 30% (positions produce distinct ensembles)
"""
from __future__ import annotations
import argparse
import json
import numpy as np

import research.runners.concept_pool_demo as cpd
from sim.text_embeddings import orthogonal_drive_pattern, positional_drive_pattern


def encode_word_at_position(bridge, word_idx: int, position: int,
                              tag_name: str,
                              n_lang_input: int = 2048,
                              n_words_for_orthogonal: int = 16,
                              sparsity: float = 0.05,
                              n_ec_context: int = 200,
                              n_max_positions: int = 8,
                              positional_sparsity: float = 0.1,
                              encoding_steps: int = 200,
                              drive_pA: float = 200.0,
                              ec_drive_pA: float = 200.0,
                              region_filter=None,
                              top_k: int = 100,
                              teacher_pA: float = 500.0,
                              teacher_pool_name: str = None):
    """Encode (word, position) as an engram. Drives lang_input(word) +
    ec_context(position) simultaneously, optionally with teacher current
    on a target concept pool.
    """
    from sim.backend import get_backend
    cp, _ = get_backend()
    rm = bridge.region_manager

    lang_arr = cp.asarray(list(rm.indices("language_input")), dtype=cp.int64)
    ec_arr = cp.asarray(list(rm.indices("ec_context")), dtype=cp.int64)

    word_drive = orthogonal_drive_pattern(
        cue_idx=word_idx, n_cues=n_words_for_orthogonal,
        n_neurons=n_lang_input, drive_max_pA=drive_pA, sparsity=sparsity,
    )
    pos_drive = positional_drive_pattern(
        position=position, n_neurons=n_ec_context,
        drive_max_pA=ec_drive_pA, sparsity=positional_sparsity,
        n_max_positions=n_max_positions,
    )
    word_drive_gpu = cp.asarray(word_drive, dtype=cp.float32)
    pos_drive_gpu = cp.asarray(pos_drive, dtype=cp.float32)

    n_total = bridge.cp_external_input_current.shape[0]
    bridge.start_engram_recording(tag_name)
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(30):
        bridge._run_one_simulation_step()

    if teacher_pool_name:
        pool_arr = cp.asarray(list(rm.indices(teacher_pool_name)),
                                dtype=cp.int64)
    else:
        pool_arr = None

    ext = cp.zeros(n_total, dtype=cp.float32)
    for _ in range(encoding_steps):
        ext.fill(0)
        ext[lang_arr] = word_drive_gpu
        ext[ec_arr] = pos_drive_gpu
        if pool_arr is not None:
            ext[pool_arr] = teacher_pA
        bridge.cp_external_input_current[:] = ext
        bridge._run_one_simulation_step()

    bridge.cp_external_input_current[:] = 0.0
    for _ in range(20):
        bridge._run_one_simulation_step()

    stats = bridge.commit_engram_tag(tag_name, top_k=top_k,
                                       region_filter=region_filter)
    return stats


def jaccard_index(set_a, set_b):
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union > 0 else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-lang-input", type=int, default=2048)
    p.add_argument("--n-per-pool", type=int, default=200)
    p.add_argument("--n-fs-per-pool", type=int, default=24)
    p.add_argument("--n-words-for-orthogonal", type=int, default=16)
    p.add_argument("--sparsity", type=float, default=0.05)
    p.add_argument("--n-ec-context", type=int, default=200)
    p.add_argument("--n-max-positions", type=int, default=8)
    p.add_argument("--positional-sparsity", type=float, default=0.1)
    p.add_argument("--encoding-steps", type=int, default=200)
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    print(f"=== positional binding test (seed={args.seed}) ===")
    print(f"  Architecture: concept_pool + ec_context (200 neurons)")
    print(f"  positions: 8 max, sparsity {args.positional_sparsity}")
    print()

    bridge = cpd.build_concept_bridge(
        seed=args.seed,
        n_lang_input=args.n_lang_input,
        n_per_pool=args.n_per_pool,
        n_fs_per_pool=args.n_fs_per_pool,
        enable_adjective=True,
        weak_dynamics=True,
        enable_direct_verb_to_motor=True,
        enable_positional_context=True,
        n_ec_context=args.n_ec_context,
        verbose=False,
    )

    rm = bridge.region_manager
    region_filter = []
    for kind, names in [
        ("noun_pool", ["APPLE", "RIVER", "DOG", "CAT"]),
        ("verb_pool", ["GO", "COME", "STOP", "LOOK"]),
        ("adjective_pool", ["BIG", "SMALL", "HOT", "COLD"]),
    ]:
        for n in names:
            try:
                rm.indices(f"{kind}_{n}")
                region_filter.append(f"{kind}_{n}")
            except Exception:
                pass
    # CRITICAL: include ec_context so positional pattern is captured in tag.
    # Without this, all (w, pos*) tags for the same word have identical
    # concept-pool neurons and differ only in ec_context activity that's
    # not recorded.
    region_filter.append("ec_context")
    # NEW: also restrict to ec_context ONLY for the cleanest positional
    # discrimination. The engram becomes the positional code itself; recall
    # via stim re-injects the position code which fires the bound concept
    # pool via STDP-grown ec_context -> pool weights.
    region_filter_ec_only = ["ec_context"]

    # v16 vocab order
    _ALL_WORDS = [
        "north", "east", "south", "west",
        "apple", "river", "dog", "cat",
        "go", "come", "stop", "look",
        "big", "small", "hot", "cold",
    ]
    _WORD_TO_IDX = {w: i for i, w in enumerate(_ALL_WORDS)}

    # Test words and their target pools (teacher current biases the pool
    # to fire, so positional context modulates WITHIN-pool firing)
    word_to_pool = {
        "apple": "noun_pool_APPLE",
        "dog": "noun_pool_DOG",
        "cat": "noun_pool_CAT",
    }
    test_words = ["apple", "dog", "cat"]
    test_positions = [0, 2, 4]

    # Encode (word, position) for each combo
    print("[ENCODE] Encoding (word, position) combos WITH teacher current")
    tag_indices = {}  # tag_name -> set of neuron indices
    for word in test_words:
        for pos in test_positions:
            tag = f"{word}_pos{pos}"
            encode_word_at_position(
                bridge, word_idx=_WORD_TO_IDX[word], position=pos,
                tag_name=tag,
                n_lang_input=args.n_lang_input,
                n_words_for_orthogonal=args.n_words_for_orthogonal,
                sparsity=args.sparsity,
                n_ec_context=args.n_ec_context,
                n_max_positions=args.n_max_positions,
                positional_sparsity=args.positional_sparsity,
                encoding_steps=args.encoding_steps,
                region_filter=region_filter_ec_only,  # ec_context only!
                top_k=30,  # small top-K matching positional band size
                teacher_pool_name=word_to_pool[word],  # bias toward target
                teacher_pA=500.0,
                ec_drive_pA=500.0,
            )
            # Read back tag neurons
            indices = bridge.get_engram_tag_indices(tag)
            # Convert to set of ints
            from sim.backend import to_host
            tag_indices[tag] = set(to_host(indices).tolist())
            print(f"  {tag}: {len(tag_indices[tag])} neurons")

    print()
    print("[ANALYSIS] Pairwise Jaccard overlap")
    print(f"  {'tag_a':16s} {'tag_b':16s} {'jaccard':10s} {'verdict':10s}")

    results = []
    same_word_diff_pos_overlaps = []
    diff_word_same_pos_overlaps = []
    diff_word_diff_pos_overlaps = []

    for w1 in test_words:
        for p1 in test_positions:
            for w2 in test_words:
                for p2 in test_positions:
                    tag_a = f"{w1}_pos{p1}"
                    tag_b = f"{w2}_pos{p2}"
                    if tag_a >= tag_b:
                        continue  # only unique pairs
                    j = jaccard_index(tag_indices[tag_a], tag_indices[tag_b])
                    # Categorize
                    same_word = (w1 == w2)
                    same_pos = (p1 == p2)
                    if same_word and not same_pos:
                        same_word_diff_pos_overlaps.append(j)
                        cat = "SAME_W_DIFF_P"
                    elif not same_word and same_pos:
                        diff_word_same_pos_overlaps.append(j)
                        cat = "DIFF_W_SAME_P"
                    elif not same_word and not same_pos:
                        diff_word_diff_pos_overlaps.append(j)
                        cat = "DIFF_W_DIFF_P"
                    else:
                        cat = "SAME_W_SAME_P"  # shouldn't happen with unique pairs
                    print(f"  {tag_a:16s} {tag_b:16s} {j:.3f}      [{cat}]")
                    results.append({
                        "tag_a": tag_a, "tag_b": tag_b,
                        "jaccard": j, "category": cat,
                    })

    print()
    print(f"[VERDICT]")
    print(f"  Same word, different position (should be LOW):")
    if same_word_diff_pos_overlaps:
        print(f"    mean={np.mean(same_word_diff_pos_overlaps):.3f} "
              f"max={np.max(same_word_diff_pos_overlaps):.3f}")
    print(f"  Different word, same position (should be LOW):")
    if diff_word_same_pos_overlaps:
        print(f"    mean={np.mean(diff_word_same_pos_overlaps):.3f} "
              f"max={np.max(diff_word_same_pos_overlaps):.3f}")
    print(f"  Different word, different position (should be LOW):")
    if diff_word_diff_pos_overlaps:
        print(f"    mean={np.mean(diff_word_diff_pos_overlaps):.3f}")

    # PASS criterion: same_word_diff_pos overlap < 0.30 (positions
    # produce distinct ensembles even for the same word)
    same_w_diff_p_mean = (np.mean(same_word_diff_pos_overlaps)
                            if same_word_diff_pos_overlaps else 0.0)
    passed = same_w_diff_p_mean < 0.30
    print(f"\n  POSITIONAL BINDING TEST: "
          f"{'PASS' if passed else 'FAIL'} "
          f"(same-word-diff-pos mean = {same_w_diff_p_mean:.3f} < 0.30 threshold)")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({
                "seed": args.seed,
                "results": results,
                "same_word_diff_pos_mean": float(same_w_diff_p_mean),
                "diff_word_same_pos_mean": float(np.mean(diff_word_same_pos_overlaps)) if diff_word_same_pos_overlaps else 0.0,
                "diff_word_diff_pos_mean": float(np.mean(diff_word_diff_pos_overlaps)) if diff_word_diff_pos_overlaps else 0.0,
                "passed": bool(passed),
            }, f, indent=2)


if __name__ == "__main__":
    main()
