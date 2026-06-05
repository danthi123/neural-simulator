"""D cue-recall arc, REAL-substrate de-risk: does SWR offline-replay consolidation lift cue-direction associative
recall above the 27.5% floor on the v16 concept-pool architecture (WORKING pool->pool propagation + the real
baseline)? Encode N concept-concept pairs (cross-pool STDP), measure baseline cue-recall (drive a alone -> is b in
lang_output top-3?), apply SWR consolidation (offline replay: drive BOTH pools repeatedly with the cross_pool_concept
gate OPEN + STDP -> strengthen the directed a->b cross-pool pathway), re-measure. GATE: cue-recall lifts well above
27.5% multi-seed, SPECIFICALLY (a permuted/control pairing must NOT lift the TRUE cue-recall). McClelland CLS +
Buzsaki SWR: repeated sleep replay consolidates the association into a directed cortical pathway.
Design: docs/plans/2026-06-05-D-cue-recall-SWR-consolidation-design.md.
"""
import argparse

import numpy as np

import research.runners.concept_pool_demo as cpd
from research.runners.compose_concept_engram import (
    encode_concept_pair, lang_output_pattern_during_input, cosine_to_word,
    _WORD_TO_POOL,
)
from sim.backend import get_backend

VOCAB16 = ["north", "east", "south", "west", "apple", "river", "dog", "cat",
           "go", "come", "stop", "look", "big", "small", "hot", "cold"]
REGION_FILTER = ([f"verb_pool_{v}" for v in ["GO", "COME", "STOP", "LOOK"]]
                 + [f"noun_pool_{n}" for n in ["APPLE", "RIVER", "DOG", "CAT"]]
                 + [f"adjective_pool_{a}" for a in ["BIG", "SMALL", "HOT", "COLD"]])


def assoc_recall(bridge, pairs, n_lang_input, sparsity, n_orth=16):
    """For each (a,b): drive a alone -> lang_output -> is b in the top-3 (of non-a words)? (the 27.5% cue-recall)."""
    hits = 0
    rows = []
    for a, b in pairs:
        pat, n_lo = lang_output_pattern_during_input(bridge, a, n_lang_input=n_lang_input, sparsity=sparsity,
                                                     n_words_for_orthogonal=n_orth)
        scores = {w: cosine_to_word(pat, w, n_lo, n_words_for_orthogonal=n_orth, sparsity=sparsity)
                  for w in VOCAB16 if w != a}
        ranked = sorted(scores, key=scores.get, reverse=True)
        hit = b in ranked[:3]
        hits += int(hit)
        rows.append((a, b, hit, ranked[:3]))
    return hits, len(pairs), rows


def swr_consolidate(bridge, pairs, cycles, teacher_pA, replay_steps=12, quiet_steps=6):
    """Offline SWR replay: drive BOTH concept pools together (no lang_input), cross_pool_concept gate OPEN + STDP ->
    strengthen the directed a->b cross-pool pathway. Repeated cycles = the consolidation."""
    cp, _ = get_backend()
    rm = bridge.region_manager
    try:
        bridge.set_plasticity_gate("cross_pool_concept", 1.0)
    except KeyError:
        pass
    n_total = bridge.cp_external_input_current.shape[0]
    pool_idx = {w: cp.asarray(list(rm.indices(_WORD_TO_POOL[w])), dtype=cp.int64) for pr in pairs for w in pr}
    for _ in range(cycles):
        for a, b in pairs:
            ext = cp.zeros(n_total, dtype=cp.float32)
            ext[pool_idx[a]] = teacher_pA
            ext[pool_idx[b]] = teacher_pA
            bridge.cp_external_input_current[:] = ext
            for _ in range(replay_steps):
                bridge._run_one_simulation_step()
            bridge.cp_external_input_current[:] = 0.0
            for _ in range(quiet_steps):
                bridge._run_one_simulation_step()
    try:
        bridge.set_plasticity_gate("cross_pool_concept", 0.0)
    except KeyError:
        pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--load-bridge", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-lang-input", type=int, default=2048)
    p.add_argument("--n-per-pool", type=int, default=200)
    p.add_argument("--n-fs-per-pool", type=int, default=24)
    p.add_argument("--sparsity", type=float, default=0.05)
    p.add_argument("--pairs", type=str, default="apple:big,dog:small,cat:hot,river:cold")
    p.add_argument("--encoding-steps", type=int, default=200)
    p.add_argument("--teacher-pA", type=float, default=500.0)
    p.add_argument("--swr-cycles", type=int, default=40)
    p.add_argument("--swr-teacher-pA", type=float, default=600.0)
    p.add_argument("--permute", action="store_true",
                    help="ANTI-CHEAT: consolidate PERMUTED (wrong) pairings -- the TRUE cue-recall must NOT lift.")
    args = p.parse_args()

    pairs = [tuple(s.split(":")) for s in args.pairs.split(",")]
    # consolidation pairs: TRUE, or (anti-cheat) permuted -- rotate the b's so each a consolidates with the WRONG b.
    cons_pairs = pairs
    if args.permute:
        bs = [b for _, b in pairs]
        bs = bs[1:] + bs[:1]
        cons_pairs = [(a, b) for (a, _), b in zip(pairs, bs)]

    bridge = cpd.build_concept_bridge(
        seed=args.seed, n_lang_input=args.n_lang_input, n_per_pool=args.n_per_pool,
        n_fs_per_pool=args.n_fs_per_pool, enable_adjective=True, weak_dynamics=True,
        enable_direct_verb_to_motor=True, enable_cross_pool_concept_pathways=True, verbose=False)
    bridge.load_checkpoint(args.load_bridge)

    for a, b in pairs:
        encode_concept_pair(bridge, a, b, f"{a}_{b}", encoding_steps=args.encoding_steps,
                            sparsity=args.sparsity, n_lang_input=args.n_lang_input,
                            n_words_for_orthogonal=16, region_filter=REGION_FILTER,
                            balanced_teacher_pA=args.teacher_pA, verbose=False)

    base_hits, n, base_rows = assoc_recall(bridge, pairs, args.n_lang_input, args.sparsity)
    swr_consolidate(bridge, cons_pairs, args.swr_cycles, args.swr_teacher_pA)
    post_hits, _, post_rows = assoc_recall(bridge, pairs, args.n_lang_input, args.sparsity)   # always measure TRUE

    tag = "PERMUTED-control (TRUE must NOT lift)" if args.permute else "TRUE consolidation"
    print(f"=== D SWR v16 de-risk (seed={args.seed}, swr_cycles={args.swr_cycles}, {tag}) ===")
    print(f"cue-recall (TRUE b in top-3): BASELINE {base_hits}/{n}  ->  POST-SWR {post_hits}/{n}")
    for (a, b, h0, t0), (_, _, h1, t1) in zip(base_rows, post_rows):
        print(f"  {a}->{b}: base {'HIT' if h0 else 'miss'} {t0}  ->  post {'HIT' if h1 else 'miss'} {t1}")


if __name__ == "__main__":
    main()
