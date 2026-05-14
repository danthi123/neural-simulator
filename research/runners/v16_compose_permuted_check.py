"""Anti-cheat: test if v16+compose-training compose-binding is real
vs structural noise at chance level.

Methodology:
- Use the 4 trained compose pairs: (go, north), (come, south),
  (stop, west), (look, east)
- For each seed, test all 24 = 4! permuted mappings of verb -> motor
- A REAL learning result: the TRUE mapping should have the highest
  PASS count among all permutations.
- A NOISE result: TRUE mapping has random rank among permutations,
  PASS rate ~chance for all permutations.

Reuses concept_speak_demo's helper machinery for driving + measuring.
"""
from __future__ import annotations
import argparse
import itertools
import json

import research.runners.concept_pool_demo as cpd
from research.runners.concept_compose_train import (
    measure_compose_inference, _WORD_TO_IDX, _WORD_TO_POOL,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--load-bridge", required=True,
                    help="Path to v16-composed bridge (post compose-train)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-lang-input", type=int, default=2048)
    p.add_argument("--n-per-pool", type=int, default=200)
    p.add_argument("--n-fs-per-pool", type=int, default=24)
    p.add_argument("--sparsity", type=float, default=0.05)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

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

    verb_words = ["go", "come", "stop", "look"]
    motor_words = ["north", "south", "west", "east"]
    true_mapping = list(zip(verb_words, motor_words))
    # The TRUE mapping that was trained:
    # go->north, come->south, stop->west, look->east

    print(f"[ANTI-CHEAT] Testing 24 permutations of verb->motor mapping")
    print(f"  True mapping: {true_mapping}")
    print()

    all_results = []
    # First, for each verb, measure firing across ALL motor pools
    # (only need to drive each verb once, not once per permutation)
    verb_firing = {}
    for verb in verb_words:
        rates = cpd.measure_pool_firing(
            bridge, verb,
            ["motor_N", "motor_E", "motor_S", "motor_W"],
            n_lang_input=args.n_lang_input,
            n_words_for_orthogonal=16,
            sparsity=args.sparsity,
            orthogonal_codes=True,
            word_to_idx=_WORD_TO_IDX,
        )
        verb_firing[verb] = rates
        print(f"  '{verb}' -> rates: motor_N={rates['motor_N']:.2f} "
              f"motor_E={rates['motor_E']:.2f} motor_S={rates['motor_S']:.2f} "
              f"motor_W={rates['motor_W']:.2f}")
    print()

    # Now evaluate each of 24 permutations
    perm_results = []
    for perm_idx, motor_perm in enumerate(itertools.permutations(motor_words)):
        mapping = list(zip(verb_words, motor_perm))
        n_pass = 0
        for verb, motor in mapping:
            pool = _WORD_TO_POOL[motor]
            rates = verb_firing[verb]
            target = rates[pool]
            off = max(v for k, v in rates.items() if k != pool)
            if target > off:
                n_pass += 1
        is_true = (motor_perm == tuple(motor_words))
        perm_results.append({
            "perm_idx": perm_idx,
            "mapping": mapping,
            "n_pass": n_pass,
            "is_true": is_true,
        })

    # Rank
    perm_results.sort(key=lambda r: -r["n_pass"])
    print(f"[RANKED] PERMUTATIONS by PASS count:")
    print(f"  {'rank':5s} {'mapping':40s} {'n_pass':8s} {'is_true':8s}")
    for rank, r in enumerate(perm_results[:10], start=1):
        m_str = ", ".join(f"{v}->{m}" for v, m in r["mapping"])
        is_true_str = "** TRUE **" if r["is_true"] else ""
        print(f"  {rank:5d} {m_str:40s} {r['n_pass']:8d} {is_true_str}")

    true_rank = next(i for i, r in enumerate(perm_results, start=1)
                      if r["is_true"])
    true_pass = next(r["n_pass"] for r in perm_results if r["is_true"])
    max_pass = perm_results[0]["n_pass"]

    print()
    print(f"[VERDICT] True mapping ranked {true_rank}/24 with {true_pass}/4 PASS")
    print(f"          Best permutation: {max_pass}/4 PASS")

    if true_rank == 1 and true_pass > max(r["n_pass"] for r in perm_results[1:]):
        print(f"          [GOOD] TRUE mapping is UNIQUELY best - real learning")
    elif true_pass >= 3:
        print(f"          [GOOD] TRUE mapping has >=3/4 PASS - likely real signal")
    elif true_rank <= 4:
        print(f"          [WARN] TRUE in top 4/24, but not uniquely best")
    else:
        print(f"          [FAIL] TRUE mapping ranked {true_rank}/24 - NOT distinguishable from chance")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({
                "seed": args.seed,
                "load_bridge": args.load_bridge,
                "true_mapping": true_mapping,
                "verb_firing": verb_firing,
                "all_permutations": perm_results,
                "true_rank": true_rank,
                "true_n_pass": true_pass,
                "max_n_pass": max_pass,
            }, f, indent=2)


if __name__ == "__main__":
    main()
