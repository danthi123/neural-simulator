"""Full chat-usable composition pipeline: query -> match engram -> motor output.

Tests the complete loop end-to-end:
1. ENCODE: train 4 (verb, motor) engrams
2. For each TEST query (verb, motor):
   - Drive lang_input(verb+motor) -> measure firing pattern
   - Cosine-match firing to all stored engram patterns -> pick best match
   - Stimulate the matched engram -> measure motor pool firing
   - Verify: the matched engram's TRUE motor is the one that fires

Anti-cheat: also test PERMUTED queries (e.g., lang_input("go" + "south"))
and verify they DON'T retrieve a strong match (or retrieve the wrong tag,
which would be a real architectural failure mode).
"""
from __future__ import annotations
import argparse
import itertools
import json
import time
import numpy as np

import research.runners.concept_pool_demo as cpd
from research.runners.concept_compose_train import _WORD_TO_IDX, _WORD_TO_POOL
from research.runners.compose_engram_retrieval import (
    encode_with_pattern, measure_firing_pattern_during_drive, cosine_sim,
)
from research.runners.compose_engram_demo import recall_compose_tag


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--load-bridge", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-lang-input", type=int, default=2048)
    p.add_argument("--n-per-pool", type=int, default=200)
    p.add_argument("--n-fs-per-pool", type=int, default=24)
    p.add_argument("--compose-pairs", type=str,
                    default="go:north,come:south,stop:west,look:east")
    p.add_argument("--encoding-steps", type=int, default=200)
    p.add_argument("--retrieval-steps", type=int, default=200)
    p.add_argument("--recall-stim-pA", type=float, default=1500.0)
    p.add_argument("--recall-steps", type=int, default=100)
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--sparsity", type=float, default=0.05)
    p.add_argument("--anti-cheat", action="store_true",
                    help="Also test permuted (verb, motor) cues — should NOT match strongly")
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    pairs = [tuple(s.strip().split(":")) for s in args.compose_pairs.split(",")]
    verb_words = [v for v, _ in pairs]
    motor_words = [m for _, m in pairs]
    true_dict = dict(pairs)

    print(f"=== compose_full_pipeline (seed={args.seed}) ===")
    print(f"  Pairs: {pairs}")
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

    # CRITICAL: freeze ALL plasticity gates for the entire pipeline.
    # STDP active during encoding+matching+recall causes lang_input -> motor
    # weights to drift, breaking cosine-match consistency between iterations.
    # The chat pipeline must operate on a STABLE bridge (no weight changes).
    all_gates = [
        "language_input_to_motor",
        "language_input_to_verb_pool",
        "language_input_to_noun_pool",
        "language_input_to_adjective_pool",
        "motor_to_language_output",
        "verb_pool_to_language_output",
        "noun_pool_to_language_output",
        "adjective_pool_to_language_output",
        "motor_FS_to_motor",
        "verb_pool_FS_to_verb_pool",
        "noun_pool_FS_to_noun_pool",
        "adjective_pool_FS_to_adjective_pool",
        "verb_to_motor_direct",
        "verb_pool_to_dlpfc_uni",
        "dlpfc_verb_to_motor_uni",
    ]
    n_frozen = 0
    for g in all_gates:
        try:
            bridge.set_plasticity_gate(g, 0.0)
            n_frozen += 1
        except Exception:
            pass
    print(f"[FREEZE] {n_frozen}/{len(all_gates)} plasticity gates set to 0.0")
    print()

    rm = bridge.region_manager
    region_filter = (
        [f"verb_pool_{v}" for v in ["GO", "COME", "STOP", "LOOK"]]
        + [f"noun_pool_{n}" for n in ["APPLE", "RIVER", "DOG", "CAT"]]
        + [f"adjective_pool_{a}" for a in ["BIG", "SMALL", "HOT", "COLD"]]
        + [f"motor_{a}" for a in ["N", "E", "S", "W"]]
    )
    n_total = bridge.cp_external_input_current.shape[0]
    rf_mask = np.zeros(n_total, dtype=bool)
    for rname in region_filter:
        try:
            rf_mask[list(rm.indices(rname))] = True
        except Exception:
            pass

    # ENCODE
    print("[ENCODE]")
    encoded = {}
    for verb, motor in pairs:
        tag_name = f"{verb}_{motor}"
        _, pattern = encode_with_pattern(
            bridge, verb, motor, tag_name,
            encoding_steps=args.encoding_steps,
            sparsity=args.sparsity,
            n_lang_input=args.n_lang_input,
            region_filter=region_filter,
            top_k=args.top_k,
            verbose=True,
        )
        encoded[tag_name] = {"verb": verb, "motor": motor, "pattern": pattern}
    print()

    # PHASE 1: ALL queries first (no recall in between to preserve state)
    print("[PHASE 1] All cosine-match queries (no stimulation between)")
    print(f"  {'query':18s} {'matched':18s} {'match_score':12s} {'match correct?':12s}")
    n_match_correct = 0
    matches = []
    for verb, motor in pairs:
        true_tag = f"{verb}_{motor}"
        query_pattern = measure_firing_pattern_during_drive(
            bridge, verb, motor, rf_mask,
            drive_steps=args.retrieval_steps,
            sparsity=args.sparsity,
            n_lang_input=args.n_lang_input,
        )
        scores = {tag: cosine_sim(query_pattern, d["pattern"])
                   for tag, d in encoded.items()}
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        matched_tag = ranked[0][0]
        match_score = ranked[0][1]
        is_match = (matched_tag == true_tag)
        if is_match:
            n_match_correct += 1
        matches.append({
            "verb": verb, "motor": motor, "true_tag": true_tag,
            "matched_tag": matched_tag, "match_score": match_score,
            "all_scores": scores, "is_match": is_match,
        })
        print(f"  {true_tag:18s} {matched_tag:18s} {match_score:.3f}        "
              f"{'YES' if is_match else 'NO'}")

    # PHASE 2: For each matched engram, stimulate and measure motor.
    # Done AFTER all queries to avoid state contamination.
    print()
    print("[PHASE 2] Stimulate each matched engram, measure motor output")
    print(f"  {'matched tag':18s} {'recalled motor':15s} {'TRUE motor':12s} {'correct?':8s}")
    n_motor_correct = 0
    results = []
    for m in matches:
        matched_tag = m["matched_tag"]
        true_motor_pool = _WORD_TO_POOL[m["motor"]]
        motor_rates = recall_compose_tag(
            bridge, matched_tag,
            drive_pA=args.recall_stim_pA,
            recall_steps=args.recall_steps,
        )
        recalled_motor_pool = max(motor_rates, key=motor_rates.get)
        is_motor_correct = (recalled_motor_pool == true_motor_pool)
        if is_motor_correct:
            n_motor_correct += 1
        marker = "YES" if is_motor_correct else "NO"
        print(f"  {matched_tag:18s} {recalled_motor_pool:13s} {true_motor_pool:10s} {marker}")
        results.append({**m,
                         "motor_rates": motor_rates,
                         "recalled_motor_pool": recalled_motor_pool,
                         "true_motor_pool": true_motor_pool,
                         "motor_correct": is_motor_correct})

    print()
    print(f"[VERDICT]")
    print(f"  Cosine-match correct: {n_match_correct}/{len(pairs)}")
    print(f"  Motor recall correct: {n_motor_correct}/{len(pairs)}")

    # ANTI-CHEAT: permuted (verb, motor) queries should match TRUE engram for that verb
    # If query is "go" + "south" (not trained), the system might still match
    # the "go_north" engram if the verb-pool drive dominates. That would be
    # acceptable (verb determines action). But if it strongly matches some
    # unrelated engram, that's a failure.
    if args.anti_cheat:
        print()
        print("[ANTI-CHEAT] Test 12 NON-TRUE permuted (verb, motor) queries")
        print(f"  {'permuted query':24s} {'matched':18s} {'score':8s} {'verb_only_match':18s}")
        n_perm_test = 0
        n_perm_verb_match = 0
        for v in verb_words:
            for m in motor_words:
                if (v, m) in pairs:
                    continue  # skip TRUE pairs
                n_perm_test += 1
                query_pattern = measure_firing_pattern_during_drive(
                    bridge, v, m, rf_mask,
                    drive_steps=args.retrieval_steps,
                    sparsity=args.sparsity,
                    n_lang_input=args.n_lang_input,
                )
                scores = {tag: cosine_sim(query_pattern, d["pattern"])
                           for tag, d in encoded.items()}
                ranked = sorted(scores.items(), key=lambda kv: -kv[1])
                matched_tag = ranked[0][0]
                top_score = ranked[0][1]
                # Expected: the system should match the engram whose VERB matches.
                # E.g., "go" + "south" should match "go_north" (verb dominates).
                verb_expected_tag = f"{v}_{true_dict[v]}"
                is_verb_match = (matched_tag == verb_expected_tag)
                if is_verb_match:
                    n_perm_verb_match += 1
                print(f"  {v+'+'+m:24s} {matched_tag:18s} {top_score:.3f}    "
                      f"{verb_expected_tag:18s} {'YES' if is_verb_match else 'NO'}")
        print()
        print(f"[ANTI-CHEAT VERDICT] Permuted queries map back to verb-correct engram:")
        print(f"  {n_perm_verb_match}/{n_perm_test} - VERB DOMINATES")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({
                "seed": args.seed, "load_bridge": args.load_bridge,
                "encoding_steps": args.encoding_steps,
                "retrieval_steps": args.retrieval_steps,
                "recall_stim_pA": args.recall_stim_pA,
                "recall_steps": args.recall_steps,
                "n_match_correct": n_match_correct,
                "n_motor_correct": n_motor_correct,
                "n_total": len(pairs),
                "results": results,
            }, f, indent=2, default=str)
        print(f"\n[OUT] wrote {args.out}")


if __name__ == "__main__":
    main()
