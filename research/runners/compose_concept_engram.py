"""Concept-concept engram composition — NO motor routing.

This is the architectural test for REAL semantic association memory:
encode engrams binding two CONCEPT words (e.g., apple+red, walk+fast,
dog+big) and verify retrieval via lang_output spelling, NOT motor.

Tests:
1. ENCODE: drive lang_input(concept_A) + lang_input(concept_B), tag
   the co-fired neurons (no motor teacher — let concepts compete
   equally for representation).
2. STIM-RECALL: stimulate tag, read lang_output cosine → which words
   does the system "say"? Should include BOTH concept_A and concept_B
   above off-target words.
3. ASSOCIATIVE-RECALL: drive lang_input(concept_A) ALONE; check if
   lang_output also produces signal for concept_B (semantic association).

A clean PASS means the system has stored "concept_A is related to
concept_B" as a recoverable memory, independent of motor actions.
"""
from __future__ import annotations
import argparse
import itertools
import json
import numpy as np

# Patch v17 vocab for extended-vocab compatibility
import research.runners.compose_engram_demo_v2  # noqa: F401

import research.runners.concept_pool_demo as cpd
from research.runners.concept_compose_train import _WORD_TO_IDX, _WORD_TO_POOL
from sim.text_embeddings import orthogonal_drive_pattern


# All concept words (any pool that's NOT motor)
_ALL_CONCEPTS = [
    "apple", "river", "dog", "cat", "tree", "bird", "sun", "moon",
    "go", "come", "stop", "look", "walk", "run", "eat", "sleep",
    "big", "small", "hot", "cold", "red", "blue", "fast", "slow",
]


def encode_concept_pair(bridge, word_a: str, word_b: str, tag_name: str,
                          encoding_steps: int = 200, drive_pA: float = 200.0,
                          sparsity: float = 0.03, n_lang_input: int = 4096,
                          n_words_for_orthogonal: int = 28,
                          region_filter=None, top_k: int = 100,
                          balanced_teacher_pA: float = 0.0,
                          verbose: bool = True):
    """Encode (word_a, word_b) engram — both concepts, no motor.

    With balanced_teacher_pA > 0, applies teacher current to BOTH
    concept pools (the pool for word_a AND word_b) during encoding.
    Analogous to motor_teacher but symmetric across both concepts —
    ensures both pools get strong representation in the top-K tag,
    not just whichever happens to fire stronger from lang_input drive.
    """
    from sim.backend import get_backend
    cp, _ = get_backend()
    rm = bridge.region_manager

    drive_a = orthogonal_drive_pattern(
        cue_idx=_WORD_TO_IDX[word_a], n_cues=n_words_for_orthogonal,
        n_neurons=n_lang_input, drive_max_pA=drive_pA, sparsity=sparsity,
    )
    drive_b = orthogonal_drive_pattern(
        cue_idx=_WORD_TO_IDX[word_b], n_cues=n_words_for_orthogonal,
        n_neurons=n_lang_input, drive_max_pA=drive_pA, sparsity=sparsity,
    )
    combined_gpu = cp.asarray(drive_a + drive_b, dtype=cp.float32)

    lang_arr_gpu = cp.asarray(
        list(rm.indices("language_input")), dtype=cp.int64)
    n_total = bridge.cp_external_input_current.shape[0]

    # Balanced teachers: drive both concept pools' neurons directly
    use_teacher = balanced_teacher_pA > 0.0
    if use_teacher:
        pool_a_idx = list(rm.indices(_WORD_TO_POOL[word_a]))
        pool_b_idx = list(rm.indices(_WORD_TO_POOL[word_b]))
        pool_a_arr_gpu = cp.asarray(pool_a_idx, dtype=cp.int64)
        pool_b_arr_gpu = cp.asarray(pool_b_idx, dtype=cp.int64)

    bridge.start_engram_recording(tag_name)
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(30):
        bridge._run_one_simulation_step()

    ext = cp.zeros(n_total, dtype=cp.float32)
    for _ in range(encoding_steps):
        ext.fill(0)
        ext[lang_arr_gpu] = combined_gpu
        if use_teacher:
            ext[pool_a_arr_gpu] = balanced_teacher_pA
            ext[pool_b_arr_gpu] = balanced_teacher_pA
        bridge.cp_external_input_current[:] = ext
        bridge._run_one_simulation_step()

    bridge.cp_external_input_current[:] = 0.0
    for _ in range(20):
        bridge._run_one_simulation_step()

    stats = bridge.commit_engram_tag(tag_name, top_k=top_k,
                                       region_filter=region_filter)
    if verbose:
        print(f"  [{tag_name}] tagged {stats['n_tagged']} neurons "
              f"({word_a} + {word_b})")
    return stats


def lang_output_pattern_during_stim(bridge, tag_name, drive_pA=1500.0,
                                       stim_steps=100):
    """Stimulate engram tag, accumulate lang_output spike pattern."""
    from sim.backend import get_backend, to_host
    cp, _ = get_backend()
    rm = bridge.region_manager
    lang_out_idx = list(rm.indices("language_output"))
    lang_out_arr = cp.asarray(lang_out_idx, dtype=cp.int64)
    n_lang_out = len(lang_out_idx)

    bridge.cp_external_input_current[:] = 0.0
    bridge.clear_tag_drive()
    for _ in range(30):
        bridge._run_one_simulation_step()

    bridge.stimulate_tag(tag_name, drive_pA=drive_pA, additive=False)
    pattern = cp.zeros(n_lang_out, dtype=cp.float32)
    for _ in range(stim_steps):
        bridge._run_one_simulation_step()
        if hasattr(bridge, "cp_firing_states"):
            firing = bridge.cp_firing_states
            pattern += firing[lang_out_arr].astype(cp.float32)

    bridge.clear_tag_drive(tag_name)
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(20):
        bridge._run_one_simulation_step()

    return to_host(pattern), n_lang_out


def lang_output_pattern_during_input(bridge, cue_word, n_lang_input=4096,
                                        sparsity=0.03,
                                        n_words_for_orthogonal=28,
                                        drive_pA=200.0, stim_steps=100):
    """Drive lang_input(cue_word) alone, accumulate lang_output spike pattern.

    Tests if the cue's natural pathway through the bridge fires
    associated concept neurons.
    """
    from sim.backend import get_backend, to_host
    cp, _ = get_backend()
    rm = bridge.region_manager
    lang_in_idx = list(rm.indices("language_input"))
    lang_in_arr = cp.asarray(lang_in_idx, dtype=cp.int64)
    lang_out_idx = list(rm.indices("language_output"))
    lang_out_arr = cp.asarray(lang_out_idx, dtype=cp.int64)
    n_lang_out = len(lang_out_idx)
    n_total = bridge.cp_external_input_current.shape[0]

    drive = orthogonal_drive_pattern(
        cue_idx=_WORD_TO_IDX[cue_word], n_cues=n_words_for_orthogonal,
        n_neurons=n_lang_input, drive_max_pA=drive_pA, sparsity=sparsity,
    )
    drive_gpu = cp.asarray(drive, dtype=cp.float32)

    bridge.cp_external_input_current[:] = 0.0
    bridge.clear_tag_drive()
    for _ in range(30):
        bridge._run_one_simulation_step()

    ext = cp.zeros(n_total, dtype=cp.float32)
    pattern = cp.zeros(n_lang_out, dtype=cp.float32)
    for _ in range(stim_steps):
        ext.fill(0)
        ext[lang_in_arr] = drive_gpu
        bridge.cp_external_input_current[:] = ext
        bridge._run_one_simulation_step()
        if hasattr(bridge, "cp_firing_states"):
            firing = bridge.cp_firing_states
            pattern += firing[lang_out_arr].astype(cp.float32)

    bridge.cp_external_input_current[:] = 0.0
    for _ in range(20):
        bridge._run_one_simulation_step()

    return to_host(pattern), n_lang_out


def cosine_to_word(pattern: np.ndarray, target_word: str,
                     n_lang_out: int,
                     n_words_for_orthogonal: int = 28,
                     sparsity: float = 0.03):
    """Cosine similarity of lang_output pattern to target word's spell-pattern."""
    target_pat = orthogonal_drive_pattern(
        cue_idx=_WORD_TO_IDX[target_word], n_cues=n_words_for_orthogonal,
        n_neurons=n_lang_out, drive_max_pA=1.0, sparsity=sparsity,
    )
    a = float(np.linalg.norm(pattern))
    b = float(np.linalg.norm(target_pat))
    if a == 0 or b == 0:
        return 0.0
    return float(np.dot(pattern, target_pat) / (a * b))


def main():
    p = argparse.ArgumentParser(description="Concept-concept compose (no motor)")
    p.add_argument("--load-bridge", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-lang-input", type=int, default=4096)
    p.add_argument("--n-per-pool", type=int, default=200)
    p.add_argument("--n-fs-per-pool", type=int, default=24)
    p.add_argument("--n-words-for-orthogonal", type=int, default=28)
    p.add_argument("--pairs", type=str,
                    default="apple:red,dog:big,tree:tall,walk:fast,sun:hot,moon:cold,bird:fly,river:cold",
                    help="word_a:word_b concept-concept pairs")
    p.add_argument("--encoding-steps", type=int, default=200)
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--recall-stim-pA", type=float, default=1500.0)
    p.add_argument("--recall-steps", type=int, default=100)
    p.add_argument("--sparsity", type=float, default=0.03)
    p.add_argument("--balanced-teacher-pA", type=float, default=0.0,
                    help="Drive both concept pools with teacher current "
                    "during encoding (analog of motor_teacher but on both "
                    "concept pools). Helps ensure balanced representation.")
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    # Parse + validate pairs (must be in concept vocab; reject unknown words)
    pairs = []
    for ps in args.pairs.split(","):
        a, b = ps.strip().split(":")
        if a not in _WORD_TO_IDX:
            print(f"WARN: '{a}' not in vocab — skipping pair {a}:{b}")
            continue
        if b not in _WORD_TO_IDX:
            print(f"WARN: '{b}' not in vocab — skipping pair {a}:{b}")
            continue
        if a in ["north", "east", "south", "west"] or b in ["north", "east", "south", "west"]:
            print(f"WARN: pair {a}:{b} contains a motor word — this runner is for non-motor")
        pairs.append((a, b))

    if not pairs:
        print("No valid pairs.")
        return

    print(f"=== compose_concept_engram (seed={args.seed}) ===")
    print(f"  Pairs: {pairs}")
    print(f"  Architecture: NO motor — pure concept-concept binding")
    print()

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

    # Region filter excludes motor — concept pools only
    region_filter = (
        [f"verb_pool_{v}" for v in
         ["GO", "COME", "STOP", "LOOK", "WALK", "RUN", "EAT", "SLEEP"]]
        + [f"noun_pool_{n}" for n in
            ["APPLE", "RIVER", "DOG", "CAT", "TREE", "BIRD", "SUN", "MOON"]]
        + [f"adjective_pool_{a}" for a in
            ["BIG", "SMALL", "HOT", "COLD", "RED", "BLUE", "FAST", "SLOW"]]
    )

    # ENCODE
    print("[ENCODE] Concept-concept engrams (NO motor teacher, NO motor in tag)")
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
            verbose=True,
        )

    # TEST 1: STIM-RECALL. Stimulate tag, read lang_output, rank words.
    print()
    print("[TEST 1: STIM-RECALL] Stimulate engram, read lang_output")
    print(f"  {'tag':24s} {'a_score':10s} {'b_score':10s} {'top-1':10s} {'top-3':40s}")
    results_stim = []
    for a, b in pairs:
        tag = f"{a}_{b}"
        pat, n_lo = lang_output_pattern_during_stim(
            bridge, tag, drive_pA=args.recall_stim_pA,
            stim_steps=args.recall_steps)
        # Cosine to all concept words IN THE BRIDGE'S VOCAB RANGE
        valid_words = [w for w in _ALL_CONCEPTS
                        if _WORD_TO_IDX[w] < args.n_words_for_orthogonal]
        scores = {w: cosine_to_word(
            pat, w, n_lo,
            n_words_for_orthogonal=args.n_words_for_orthogonal,
            sparsity=args.sparsity,
        ) for w in valid_words}
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        a_score = scores[a]
        b_score = scores[b]
        top1 = ranked[0][0]
        top3 = ", ".join(f"{w}={s:.2f}" for w, s in ranked[:3])
        # PASS = both a and b are in top-5 with above-noise scores
        in_top5 = [w for w, _ in ranked[:5]]
        passed = (a in in_top5) and (b in in_top5)
        marker = "PASS" if passed else "FAIL"
        print(f"  {tag:24s} {a_score:.3f}     {b_score:.3f}     {top1:10s} [{top3:40s}] [{marker}]")
        results_stim.append({
            "tag": tag, "a": a, "b": b,
            "a_score": float(a_score), "b_score": float(b_score),
            "top_1": top1, "in_top_5": in_top5, "passed": bool(passed),
        })
    n_stim_pass = sum(1 for r in results_stim if r["passed"])

    # TEST 2: ASSOCIATIVE-RECALL. Drive concept_a alone, see if concept_b appears in lang_output.
    print()
    print("[TEST 2: ASSOCIATIVE-RECALL] Drive lang_input(a) alone, check b in lang_output")
    print(f"  {'cue':10s} {'expect_b':10s} {'a_score':10s} {'b_score':10s} {'top-3':40s}")
    results_assoc = []
    for a, b in pairs:
        pat, n_lo = lang_output_pattern_during_input(
            bridge, a,
            n_lang_input=args.n_lang_input,
            sparsity=args.sparsity,
            n_words_for_orthogonal=args.n_words_for_orthogonal,
            stim_steps=args.recall_steps,
        )
        valid_words = [w for w in _ALL_CONCEPTS
                        if _WORD_TO_IDX[w] < args.n_words_for_orthogonal]
        scores = {w: cosine_to_word(
            pat, w, n_lo,
            n_words_for_orthogonal=args.n_words_for_orthogonal,
            sparsity=args.sparsity,
        ) for w in valid_words}
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        a_score = scores[a]
        b_score = scores[b]
        top3 = ", ".join(f"{w}={s:.2f}" for w, s in ranked[:3])
        # PASS for associative: b is in top-3 (excluding a itself if it dominates)
        non_a_ranked = [w for w, _ in ranked if w != a]
        passed = (b in non_a_ranked[:3])
        marker = "PASS" if passed else "FAIL"
        print(f"  {a:10s} {b:10s} {a_score:.3f}     {b_score:.3f}     [{top3:40s}] [{marker}]")
        results_assoc.append({
            "cue": a, "expect_b": b,
            "a_score": float(a_score), "b_score": float(b_score),
            "non_a_top3": non_a_ranked[:3], "passed": bool(passed),
        })
    n_assoc_pass = sum(1 for r in results_assoc if r["passed"])

    print()
    print(f"[VERDICT]")
    print(f"  Stim-recall (both concepts in top-5): {n_stim_pass}/{len(pairs)}")
    print(f"  Associative-recall (b in non-a top-3): {n_assoc_pass}/{len(pairs)}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({
                "seed": args.seed, "pairs": pairs,
                "n_stim_pass": n_stim_pass,
                "n_assoc_pass": n_assoc_pass,
                "n_total": len(pairs),
                "stim_results": results_stim,
                "assoc_results": results_assoc,
            }, f, indent=2, default=str)
        print(f"  [OUT] wrote {args.out}")


if __name__ == "__main__":
    main()
