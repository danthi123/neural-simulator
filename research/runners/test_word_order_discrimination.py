"""Test word order discrimination via positional encoding.

Hypothesis: encoding "apple big" creates STDP weights
ec_context(pos0) -> noun_pool_APPLE and ec_context(pos1) -> adjective_pool_BIG.
Encoding "big apple" creates the OPPOSITE binding.

Test: after encoding, drive ec_context(pos0) alone and measure which
concept pool fires strongest. The position-conditioned firing tells us
which word was at that position.

A different bridge per sentence (else 2nd encoding would overwrite 1st
ec_context->pool weights). Cleanest test.
"""
from __future__ import annotations
import argparse
import json
import numpy as np

import research.runners.concept_pool_demo as cpd
from sim.text_embeddings import orthogonal_drive_pattern, positional_drive_pattern


_WORD_TO_IDX = {
    "north": 0, "east": 1, "south": 2, "west": 3,
    "apple": 4, "river": 5, "dog": 6, "cat": 7,
    "go": 8, "come": 9, "stop": 10, "look": 11,
    "big": 12, "small": 13, "hot": 14, "cold": 15,
}
_WORD_TO_POOL = {
    "north": "motor_N", "east": "motor_E", "south": "motor_S", "west": "motor_W",
    "apple": "noun_pool_APPLE", "river": "noun_pool_RIVER",
    "dog": "noun_pool_DOG", "cat": "noun_pool_CAT",
    "go": "verb_pool_GO", "come": "verb_pool_COME",
    "stop": "verb_pool_STOP", "look": "verb_pool_LOOK",
    "big": "adjective_pool_BIG", "small": "adjective_pool_SMALL",
    "hot": "adjective_pool_HOT", "cold": "adjective_pool_COLD",
}


def encode_sentence(bridge, words: list, n_lang_input=2048,
                     n_words_for_orthogonal=16, sparsity=0.05,
                     n_ec_context=200, n_max_positions=8,
                     positional_sparsity=0.1,
                     encoding_steps_per_word=200,
                     drive_pA=200.0, ec_drive_pA=500.0,
                     teacher_pA=500.0):
    """Encode a sentence (list of words at sequential positions).

    For each (word, position) phase, drive lang_input(word) +
    ec_context(position) + teacher current on target pool. STDP grows
    ec_context_pos -> word_pool weights.
    """
    from sim.backend import get_backend
    cp, _ = get_backend()
    rm = bridge.region_manager
    lang_arr = cp.asarray(list(rm.indices("language_input")), dtype=cp.int64)
    ec_arr = cp.asarray(list(rm.indices("ec_context")), dtype=cp.int64)
    n_total = bridge.cp_external_input_current.shape[0]

    bridge.cp_external_input_current[:] = 0.0
    for _ in range(30):
        bridge._run_one_simulation_step()

    for position, word in enumerate(words):
        word_drive = orthogonal_drive_pattern(
            cue_idx=_WORD_TO_IDX[word], n_cues=n_words_for_orthogonal,
            n_neurons=n_lang_input, drive_max_pA=drive_pA, sparsity=sparsity,
        )
        pos_drive = positional_drive_pattern(
            position=position, n_neurons=n_ec_context,
            drive_max_pA=ec_drive_pA, sparsity=positional_sparsity,
            n_max_positions=n_max_positions,
        )
        pool_name = _WORD_TO_POOL[word]
        pool_arr = cp.asarray(list(rm.indices(pool_name)), dtype=cp.int64)

        word_drive_gpu = cp.asarray(word_drive, dtype=cp.float32)
        pos_drive_gpu = cp.asarray(pos_drive, dtype=cp.float32)

        ext = cp.zeros(n_total, dtype=cp.float32)
        for _ in range(encoding_steps_per_word):
            ext.fill(0)
            ext[lang_arr] = word_drive_gpu
            ext[ec_arr] = pos_drive_gpu
            ext[pool_arr] = teacher_pA
            bridge.cp_external_input_current[:] = ext
            bridge._run_one_simulation_step()

    bridge.cp_external_input_current[:] = 0.0
    for _ in range(20):
        bridge._run_one_simulation_step()


def query_position(bridge, position, all_pool_names,
                     n_ec_context=200, n_max_positions=8,
                     positional_sparsity=0.1, ec_drive_pA=500.0,
                     stim_steps=100):
    """Drive ec_context(position) alone, measure which pool fires most.
    Returns pool firing rates dict.
    """
    from sim.backend import get_backend
    cp, _ = get_backend()
    rm = bridge.region_manager
    ec_arr = cp.asarray(list(rm.indices("ec_context")), dtype=cp.int64)
    n_total = bridge.cp_external_input_current.shape[0]

    pool_arrs = {p: cp.asarray(list(rm.indices(p)), dtype=cp.int64)
                  for p in all_pool_names}
    spike_counts = {p: 0 for p in all_pool_names}

    pos_drive = positional_drive_pattern(
        position=position, n_neurons=n_ec_context,
        drive_max_pA=ec_drive_pA, sparsity=positional_sparsity,
        n_max_positions=n_max_positions,
    )
    pos_drive_gpu = cp.asarray(pos_drive, dtype=cp.float32)

    bridge.cp_external_input_current[:] = 0.0
    for _ in range(30):
        bridge._run_one_simulation_step()

    ext = cp.zeros(n_total, dtype=cp.float32)
    for _ in range(stim_steps):
        ext.fill(0)
        ext[ec_arr] = pos_drive_gpu
        bridge.cp_external_input_current[:] = ext
        bridge._run_one_simulation_step()
        if hasattr(bridge, "cp_firing_states"):
            firing = bridge.cp_firing_states
            for p, arr in pool_arrs.items():
                spike_counts[p] += int(firing[arr].sum())

    rates = {p: spike_counts[p] / (stim_steps * len(pool_arrs[p]))
              for p in all_pool_names}
    return rates


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-lang-input", type=int, default=2048)
    p.add_argument("--n-per-pool", type=int, default=200)
    p.add_argument("--n-fs-per-pool", type=int, default=24)
    p.add_argument("--sparsity", type=float, default=0.05)
    p.add_argument("--n-ec-context", type=int, default=200)
    p.add_argument("--n-max-positions", type=int, default=8)
    p.add_argument("--positional-sparsity", type=float, default=0.1)
    p.add_argument("--encoding-steps", type=int, default=400,
                    help="Steps per word position during encoding")
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    all_concept_pools = [
        "noun_pool_APPLE", "noun_pool_RIVER", "noun_pool_DOG", "noun_pool_CAT",
        "verb_pool_GO", "verb_pool_COME", "verb_pool_STOP", "verb_pool_LOOK",
        "adjective_pool_BIG", "adjective_pool_SMALL",
        "adjective_pool_HOT", "adjective_pool_COLD",
    ]

    # Build TWO independent bridges (one per sentence) so STDP doesn't overwrite
    print("=== Word Order Discrimination Test ===")
    print()
    sentences = [
        ["apple", "big"],   # apple@pos0, big@pos1
        ["big", "apple"],   # big@pos0, apple@pos1 (reversed)
    ]
    results = {}

    for sent_idx, words in enumerate(sentences):
        sent_str = " ".join(words)
        print(f"\n--- Sentence {sent_idx}: '{sent_str}' ---")
        # Fresh bridge each time. NOTE: using CANON dynamics (not weak)
        # because ec_context drive needs to bootstrap pool firing via
        # STDP-grown weights — weak pools don't sustain.
        bridge = cpd.build_concept_bridge(
            seed=args.seed,
            n_lang_input=args.n_lang_input,
            n_per_pool=args.n_per_pool,
            n_fs_per_pool=args.n_fs_per_pool,
            enable_adjective=True,
            weak_dynamics=False,  # canon dynamics for stronger pool firing
            enable_direct_verb_to_motor=True,
            enable_positional_context=True,
            n_ec_context=args.n_ec_context,
            verbose=False,
        )
        print(f"[ENCODE] '{sent_str}'", flush=True)
        encode_sentence(
            bridge, words,
            n_lang_input=args.n_lang_input,
            n_words_for_orthogonal=16,
            sparsity=args.sparsity,
            n_ec_context=args.n_ec_context,
            n_max_positions=args.n_max_positions,
            positional_sparsity=args.positional_sparsity,
            encoding_steps_per_word=args.encoding_steps,
        )

        # Test: query each position, see which pool fires
        print(f"[QUERY] Drive ec_context(pos), see which pool fires")
        sent_results = {}
        for position in range(len(words)):
            rates = query_position(
                bridge, position, all_concept_pools,
                n_ec_context=args.n_ec_context,
                n_max_positions=args.n_max_positions,
                positional_sparsity=args.positional_sparsity,
            )
            # Find top firing pool
            ranked = sorted(rates.items(), key=lambda kv: -kv[1])
            top3 = ranked[:3]
            expected_pool = _WORD_TO_POOL[words[position]]
            top_pool = ranked[0][0]
            correct = (top_pool == expected_pool)
            print(f"  pos={position} expected={words[position]:6s} "
                  f"({expected_pool:20s}) "
                  f"top: {', '.join(f'{p}={r:.2f}' for p, r in top3)} "
                  f"{'CORRECT' if correct else 'WRONG'}")
            sent_results[position] = {
                "expected_word": words[position],
                "expected_pool": expected_pool,
                "top3": top3,
                "correct": correct,
            }
        results[sent_str] = sent_results

    # Verdict: did both sentences get correct position-conditioned recall?
    correct_count = 0
    total = 0
    for sent_str, sent_results in results.items():
        for pos, r in sent_results.items():
            total += 1
            if r["correct"]:
                correct_count += 1

    print()
    print(f"[VERDICT]")
    print(f"  Position-conditioned recall: {correct_count}/{total} = "
          f"{correct_count/total*100:.1f}%")
    word_order_passed = correct_count == total
    if word_order_passed:
        print(f"  WORD ORDER DISCRIMINATION: PASS (both sentences fully decoded)")
    else:
        print(f"  WORD ORDER DISCRIMINATION: PARTIAL (some positions failed)")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({
                "seed": args.seed,
                "sentences": [{"words": s, "results": r}
                                for (s_str, r), s in zip(results.items(), sentences)
                                for _ in [None]],
                "correct_count": correct_count,
                "total": total,
                "accuracy": correct_count/total if total else 0,
                "passed": bool(word_order_passed),
            }, f, indent=2, default=str)


if __name__ == "__main__":
    main()
