"""Diagnostic: measure dlpfc_verb firing when verb word is driven on a v15 bridge.

Loads the v15c smoke bridge (seed 42), drives "go" word, and reports
dlpfc_verb activity. With weak dlpfc dynamics + zero verb_pool->dlpfc
weights, dlpfc_verb should be silent. This confirms v15-alone is
truly silent until compose training is implemented.

Run:
  python -m research.runners.v15_dlpfc_probe \
      --load-bridge research/findings/raw/g11_bg/concept_pool_demo/seed42_v15.simstate.h5
"""
from __future__ import annotations

import argparse
import json
import sys

# Match concept_pool_demo's import style
import research.runners.concept_pool_demo as cpd
from research.runners.text_minimal_isolation import vocab_to_drive_pattern, orthogonal_drive_pattern


def main():
    p = argparse.ArgumentParser(description="v15 dlpfc_verb probe")
    p.add_argument("--load-bridge", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-lang-input", type=int, default=2048)
    p.add_argument("--n-per-pool", type=int, default=200)
    p.add_argument("--n-fs-per-pool", type=int, default=24)
    p.add_argument("--orthogonal-codes", action="store_true")
    p.add_argument("--sparsity", type=float, default=0.05)
    p.add_argument("--n-words-orthogonal", type=int, default=16)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    # Build bridge with v15 architecture (must match training config)
    bridge = cpd.build_concept_bridge(
        seed=args.seed,
        n_lang_input=args.n_lang_input,
        n_per_pool=args.n_per_pool,
        n_fs_per_pool=args.n_fs_per_pool,
        enable_adjective=True,
        weak_dynamics=True,
        enable_dlpfc_verb_unidirectional=True,
        verbose=False,
    )

    print(f"[LOAD] {args.load_bridge}")
    bridge.load_checkpoint(args.load_bridge)

    # Check if dlpfc_verb exists
    region_names = [r["name"] for r in bridge.brain_regions_meta] if hasattr(bridge, "brain_regions_meta") else []
    rm = getattr(bridge, "region_manager", None)
    if rm is None:
        print("ERROR: bridge has no region_manager", file=sys.stderr)
        sys.exit(1)

    all_region_names = list(rm._region_idx_ranges.keys()) if hasattr(rm, "_region_idx_ranges") else []
    print(f"[INFO] regions present: {len(all_region_names)}")
    has_dlpfc = "dlpfc_verb" in all_region_names
    print(f"[INFO] dlpfc_verb present: {has_dlpfc}")
    if not has_dlpfc:
        print("ERROR: dlpfc_verb region not in bridge — was v15 enabled?",
              file=sys.stderr)
        sys.exit(1)

    # Test words: 1 verb (drives v15 path), 1 noun (control), 1 motor (control)
    test_words = [
        ("go",     "verb"),    # should activate verb_pool_GO
        ("apple",  "noun"),    # should activate noun_pool_APPLE; dlpfc silent
        ("north",  "motor"),   # should activate motor_N; dlpfc silent
    ]

    # Drive 100 steps per word, measure dlpfc_verb firing rate
    n_lang_input = args.n_lang_input
    if args.orthogonal_codes:
        # Match training's word ordering for orthogonal codes
        all_train_words = [
            "north", "east", "south", "west",
            "apple", "river", "dog", "cat",
            "go", "come", "stop", "look",
            "big", "small", "hot", "cold",
        ]
        word_to_idx = {w: i for i, w in enumerate(all_train_words)}

    pools_to_measure = [
        "dlpfc_verb",
        "verb_pool_GO", "verb_pool_COME", "verb_pool_STOP", "verb_pool_LOOK",
        "noun_pool_APPLE", "noun_pool_RIVER", "noun_pool_DOG", "noun_pool_CAT",
        "motor_N", "motor_E", "motor_S", "motor_W",
    ]

    results = {}
    print()
    print(f"{'word':10s} {'kind':6s} ", end="")
    for p_name in ["dlpfc_verb", "verb_pool_GO", "noun_pool_APPLE", "motor_N"]:
        print(f"{p_name:18s}", end="")
    print()
    print("-" * 90)

    for word, kind in test_words:
        # Generate drive pattern
        if args.orthogonal_codes:
            drive = orthogonal_drive_pattern(
                cue_idx=word_to_idx[word],
                n_cues=args.n_words_orthogonal,
                n_neurons=n_lang_input,
                sparsity=args.sparsity,
            )
        else:
            drive = vocab_to_drive_pattern(
                word, n_neurons=n_lang_input,
                sparsity=args.sparsity,
            )

        rates = cpd.measure_pool_firing(
            bridge, word, pools_to_measure, n_lang_input=n_lang_input,
            n_words_for_orthogonal=args.n_words_orthogonal,
            sparsity=args.sparsity,
            orthogonal_codes=args.orthogonal_codes,
        )
        results[word] = {"kind": kind, "rates": rates}

        print(f"{word:10s} {kind:6s} ", end="")
        for p_name in ["dlpfc_verb", "verb_pool_GO", "noun_pool_APPLE", "motor_N"]:
            print(f"{rates.get(p_name, 0):.3f}              ", end="")
        print()

    # Verdict: is dlpfc_verb activated only by verb words?
    dlpfc_verb_rate = results["go"]["rates"]["dlpfc_verb"]
    dlpfc_noun_rate = results["apple"]["rates"]["dlpfc_verb"]
    dlpfc_motor_rate = results["north"]["rates"]["dlpfc_verb"]

    print()
    print(f"[VERDICT] dlpfc_verb rates: verb={dlpfc_verb_rate:.3f}, "
          f"noun={dlpfc_noun_rate:.3f}, motor={dlpfc_motor_rate:.3f}")
    print(f"          v15 expected: verb >> noun, motor (selective verb activation)")
    print(f"          v15c-zero-init expected: ALL near 0 (silent until compose training)")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({
                "bridge": args.load_bridge,
                "seed": args.seed,
                "results": results,
                "verdict": {
                    "dlpfc_on_verb": dlpfc_verb_rate,
                    "dlpfc_on_noun": dlpfc_noun_rate,
                    "dlpfc_on_motor": dlpfc_motor_rate,
                    "verb_selective": (
                        dlpfc_verb_rate > 2 * max(dlpfc_noun_rate, dlpfc_motor_rate)
                        if max(dlpfc_noun_rate, dlpfc_motor_rate) > 0
                        else dlpfc_verb_rate > 0.1
                    ),
                },
            }, f, indent=2)
        print(f"[OUT] wrote {args.out}")


if __name__ == "__main__":
    main()
