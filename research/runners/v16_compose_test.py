"""Quick inference test on the already-trained v16-composed bridge."""
from __future__ import annotations
import argparse
import json
import sys

import research.runners.concept_pool_demo as cpd
from research.runners.concept_compose_train import (
    measure_compose_inference, _WORD_TO_POOL, _WORD_TO_IDX,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--load-bridge", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-lang-input", type=int, default=2048)
    p.add_argument("--n-per-pool", type=int, default=200)
    p.add_argument("--n-fs-per-pool", type=int, default=24)
    p.add_argument("--compose-pairs", type=str,
                    default="go:north,come:south,stop:west,look:east")
    p.add_argument("--orthogonal-codes", action="store_true", default=True)
    p.add_argument("--sparsity", type=float, default=0.05)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    pairs = []
    for s in args.compose_pairs.split(","):
        v, m = s.strip().split(":")
        pairs.append((v, m))

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

    print(f"[TEST] Composition inference (drive verb alone) on {args.load_bridge}")
    print()
    n_pass = 0
    results = []
    for verb, motor in pairs:
        r = measure_compose_inference(
            bridge, verb, motor,
            n_lang_input=args.n_lang_input,
            sparsity=args.sparsity,
            orthogonal_codes=args.orthogonal_codes,
        )
        results.append(r)
        if r["passed"]:
            n_pass += 1
        marker = "PASS" if r["passed"] else "FAIL"
        print(f"  '{verb}' alone -> {r['expected_pool']:10s} "
              f"target={r['target_rate']:.2f}  "
              f"off={r['max_off']:.2f}/{r['max_off_pool']:10s}  "
              f"ratio={r['ratio']:.2f}x  [{marker}]")
        print(f"    all rates: {r['all_rates']}")

    print()
    print(f"[VERDICT] {n_pass}/{len(pairs)} verbs drive their trained motor pool")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"results": results, "n_pass": n_pass,
                        "n_total": len(pairs)}, f, indent=2)
        print(f"[OUT] wrote {args.out}")


if __name__ == "__main__":
    main()
