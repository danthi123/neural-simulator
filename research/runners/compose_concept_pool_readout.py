"""Concept-concept compose with POOL-FIRING readout (not lang_output spelling).

The previous concept-concept test cosine-matched lang_output spike
patterns to word spelling patterns. With multiple pools firing
simultaneously, the cosine gets halved (signal split between two
spellings).

This runner uses a simpler readout: rank concept pools by their
firing rate during recall. The top-firing pool is the "associated"
concept. Should give cleaner signal than lang_output cosine.
"""
from __future__ import annotations
import argparse
import json
import numpy as np

import research.runners.concept_pool_demo as cpd
from research.runners.concept_compose_train import _WORD_TO_IDX, _WORD_TO_POOL
from research.runners.compose_concept_engram import (
    encode_concept_pair, _ALL_CONCEPTS,
)
from sim.text_embeddings import orthogonal_drive_pattern


_POOL_TO_WORD = {v: k for k, v in _WORD_TO_POOL.items()}


def measure_concept_pool_rates(bridge, cue_word, all_pool_names,
                                  n_lang_input=2048, sparsity=0.05,
                                  n_words_for_orthogonal=16,
                                  drive_pA=200.0, stim_steps=100):
    """Drive lang_input(cue), measure spike rate per concept pool."""
    from sim.backend import get_backend
    cp, _ = get_backend()
    rm = bridge.region_manager

    pool_arrs = {p: cp.asarray(list(rm.indices(p)), dtype=cp.int64)
                  for p in all_pool_names}
    spike_counts = {p: 0 for p in all_pool_names}

    lang_arr_gpu = cp.asarray(
        list(rm.indices("language_input")), dtype=cp.int64)
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
    for _ in range(stim_steps):
        ext.fill(0)
        ext[lang_arr_gpu] = drive_gpu
        bridge.cp_external_input_current[:] = ext
        bridge._run_one_simulation_step()
        if hasattr(bridge, "cp_firing_states"):
            firing = bridge.cp_firing_states
            for p, arr in pool_arrs.items():
                spike_counts[p] += int(firing[arr].sum())

    bridge.cp_external_input_current[:] = 0.0
    for _ in range(20):
        bridge._run_one_simulation_step()

    rates = {p: spike_counts[p] / (stim_steps * len(pool_arrs[p]))
              for p in all_pool_names}
    return rates


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--load-bridge", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-lang-input", type=int, default=2048)
    p.add_argument("--n-per-pool", type=int, default=200)
    p.add_argument("--n-fs-per-pool", type=int, default=24)
    p.add_argument("--n-words-for-orthogonal", type=int, default=16)
    p.add_argument("--pairs", type=str,
                    default="apple:big,dog:small,cat:hot,river:cold,go:look,come:stop,big:hot,small:cold")
    p.add_argument("--encoding-steps", type=int, default=200)
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--drive-steps", type=int, default=100)
    p.add_argument("--sparsity", type=float, default=0.05)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    pairs = []
    for ps in args.pairs.split(","):
        a, b = ps.strip().split(":")
        if (a in _WORD_TO_IDX and b in _WORD_TO_IDX
            and _WORD_TO_IDX[a] < args.n_words_for_orthogonal
            and _WORD_TO_IDX[b] < args.n_words_for_orthogonal):
            pairs.append((a, b))

    print(f"=== compose_concept_pool_readout (seed={args.seed}) ===")
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

    region_filter = []
    all_concept_pools = []
    for kind, names in [
        ("noun_pool", ["APPLE", "RIVER", "DOG", "CAT"]),
        ("verb_pool", ["GO", "COME", "STOP", "LOOK"]),
        ("adjective_pool", ["BIG", "SMALL", "HOT", "COLD"]),
    ]:
        for n in names:
            full = f"{kind}_{n}"
            try:
                bridge.region_manager.indices(full)
                region_filter.append(full)
                all_concept_pools.append(full)
            except Exception:
                pass

    print("[ENCODE]")
    for a, b in pairs:
        tag = f"{a}_{b}"
        encode_concept_pair(
            bridge, a, b, tag,
            encoding_steps=args.encoding_steps,
            drive_pA=200.0, sparsity=args.sparsity,
            n_lang_input=args.n_lang_input,
            n_words_for_orthogonal=args.n_words_for_orthogonal,
            region_filter=region_filter, top_k=args.top_k,
            verbose=False,
        )

    print()
    print("[ASSOC-RECALL via pool firing] Drive lang_input(a), rank concept pools")
    print(f"  {'cue':10s} {'expect_b':10s} {'b_rate':8s} {'top-3 non-a pools':40s}")
    n_assoc_pass = 0
    results = []
    for a, b in pairs:
        rates = measure_concept_pool_rates(
            bridge, a, all_concept_pools,
            n_lang_input=args.n_lang_input,
            sparsity=args.sparsity,
            n_words_for_orthogonal=args.n_words_for_orthogonal,
            stim_steps=args.drive_steps,
        )
        # Exclude pool A (the one that fires from direct cue)
        pool_a = _WORD_TO_POOL[a]
        pool_b = _WORD_TO_POOL[b]
        b_rate = rates[pool_b]
        non_a_ranked = sorted(
            [(p, r) for p, r in rates.items() if p != pool_a],
            key=lambda kv: -kv[1])
        top3_non_a = non_a_ranked[:3]
        # PASS if pool_b is in top-3 non-a
        non_a_top3_pools = [p for p, _ in top3_non_a]
        passed = pool_b in non_a_top3_pools
        if passed:
            n_assoc_pass += 1
        marker = "PASS" if passed else "FAIL"
        top3_str = ", ".join(f"{_POOL_TO_WORD.get(p, p.split('_')[-1])}={r:.2f}"
                              for p, r in top3_non_a)
        print(f"  {a:10s} {b:10s} {b_rate:.3f}    [{top3_str:50s}] [{marker}]")
        results.append({
            "cue": a, "expect_b": b,
            "b_rate": float(b_rate),
            "top3_non_a": [(_POOL_TO_WORD.get(p, p), float(r))
                            for p, r in top3_non_a],
            "passed": bool(passed),
        })

    print()
    print(f"[VERDICT] Associative recall via pool firing: {n_assoc_pass}/{len(pairs)}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"seed": args.seed, "pairs": pairs,
                        "n_assoc_pass": n_assoc_pass, "n_total": len(pairs),
                        "results": results}, f, indent=2)
        print(f"  [OUT] wrote {args.out}")


if __name__ == "__main__":
    main()
