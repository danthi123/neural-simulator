"""Post-hoc engram capture-quality remediation (plan->implement->test).

Tests the fix hypothesis from
2026-05-16-G20-dynamical-signature-UNDER-RECALL: weak indices fail by
UNDER-RECALL (engram tag fails to reignite its own pattern). Lever:
re-capture under-recalling per-concept tags with BOOSTED teacher drive
+ longer window. This is ARTIFACT-SAFE -- it improves engram tags on
an EXISTING bridge; it does NOT touch generate_sparse_patterns, does
NOT retrain the pattern set, and writes to a NEW bridge path (the
validated 320 artifact is preserved). Reuses existing sparse
capture + the dynamical probe (DRY).

Protocol (controlled before/after on the same bridge):
  1. Probe self-recall for ALL 64 indices -> baseline self_cum.
  2. under-recallers := self_rank > 1 (own tag doesn't self-win).
  3. Re-capture each under-recaller's per-concept tag with boosted
     teacher pA + longer window (overwrites the weak tag).
  4. Re-probe -> did self_cum rise / self_rank reach 1?
Honest: reports flip rate + mean self_cum delta. YES => artifact-safe
fix for the main open weakness; NO => negative, narrows further.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

from research.runners.concept_pool_sparse_distributed import (
    build_sparse_pool_bridge, generate_sparse_patterns,
)
from research.runners.g20_multibridge import read_vocab_file
from research.runners.g20_dynamical_probe import probe_tag


def self_stats(traj_cum, idx):
    rank = int((traj_cum > traj_cum[idx]).sum()) + 1
    return float(traj_cum[idx]), rank


def recapture(bridge, word, idx, pat, shared, n_lang, sparsity,
               teacher_pA, steps):
    """Re-commit the per-concept tag with boosted drive."""
    from sim.backend import get_backend
    from sim.text_embeddings import orthogonal_drive_pattern
    cp, _ = get_backend()
    lang = cp.asarray(list(bridge.region_manager.indices(
        "language_input")), dtype=cp.int64)
    parr = cp.asarray([shared[k] for k in pat], dtype=cp.int64)
    drive = cp.asarray(orthogonal_drive_pattern(
        cue_idx=idx, n_cues=64, n_neurons=n_lang,
        drive_max_pA=200.0, sparsity=sparsity), dtype=cp.float32)
    ext = cp.zeros(bridge.cp_external_input_current.shape[0],
                   dtype=cp.float32)
    bridge.start_engram_recording(word)
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(20):
        bridge._run_one_simulation_step()
    for _ in range(steps):
        ext.fill(0)
        ext[lang] = drive
        ext[parr] = teacher_pA
        bridge.cp_external_input_current[:] = ext
        bridge._run_one_simulation_step()
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(10):
        bridge._run_one_simulation_step()
    bridge.commit_engram_tag(word, top_k=150,
                              region_filter=["shared_concept_pool"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bridge", default="research/findings/raw/g11_bg/"
                   "g20_sparse_bridges_320/bridgeA_nouns_sparse64.simstate.h5")
    p.add_argument("--vocab", default="research/findings/raw/g11_bg/"
                   "g20_bridgeA_nouns_vocab64.txt")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-concepts", type=int, default=64)
    p.add_argument("--sparsity", type=float, default=0.007)
    p.add_argument("--boost-teacher-pA", type=float, default=400.0,
                    help="vs training's 100 teacher-bias")
    p.add_argument("--boost-steps", type=int, default=250,
                    help="vs training's 100-step capture window")
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    vocab = read_vocab_file(args.vocab)
    pats = generate_sparse_patterns(args.n_concepts, 2000, 100,
                                     args.seed)
    bridge = build_sparse_pool_bridge(
        seed=args.seed, n_lang_input=8192, n_shared_pool=2000,
        n_lang_output=8192, verbose=False)
    bridge.load_checkpoint(args.bridge)
    from sim.backend import get_backend
    cp, _ = get_backend()
    shared = list(bridge.region_manager.indices("shared_concept_pool"))
    parrs = [cp.asarray([shared[k] for k in pat], dtype=cp.int64)
             for pat in pats]

    # 1. baseline probe ALL indices
    base = {}
    for i in range(args.n_concepts):
        traj = probe_tag(bridge, vocab[i], parrs, steps=80)
        sc, rk = self_stats(traj.sum(axis=0), i)
        base[i] = (sc, rk)
    under = sorted(i for i in base if base[i][1] > 1)
    med = float(np.median([base[i][0] for i in base]))
    print(f"baseline: {len(under)}/{args.n_concepts} under-recall "
          f"(self_rank>1); median self_cum={med:.0f}", flush=True)
    print(f"under-recall idxs: {under}", flush=True)

    # 2. re-capture under-recallers with boosted drive
    for i in under:
        recapture(bridge, vocab[i], i, pats[i], shared, 8192,
                  args.sparsity, args.boost_teacher_pA,
                  args.boost_steps)

    # 3. re-probe the remediated indices
    flipped, deltas, rows = 0, [], []
    for i in under:
        traj = probe_tag(bridge, vocab[i], parrs, steps=80)
        sc, rk = self_stats(traj.sum(axis=0), i)
        d = sc - base[i][0]
        deltas.append(d)
        if rk == 1:
            flipped += 1
        rows.append({"idx": i, "word": vocab[i],
                      "base_cum": base[i][0], "base_rank": base[i][1],
                      "post_cum": sc, "post_rank": rk, "delta": d})
        print(f"  idx{i:2d} {vocab[i]:10s}: cum {base[i][0]:.0f}"
              f"->{sc:.0f} (d={d:+.0f}) rank {base[i][1]}->{rk}"
              f"{'  FIXED' if rk==1 else ''}", flush=True)

    n = len(under)
    summary = {
        "n_under_recall": n,
        "n_fixed_to_self_rank1": flipped,
        "fix_rate": flipped / max(n, 1),
        "mean_self_cum_delta": float(np.mean(deltas)) if deltas else 0,
        "boost_teacher_pA": args.boost_teacher_pA,
        "boost_steps": args.boost_steps, "seed": args.seed,
    }
    print(f"\n=== RESULT (capture-quality remediation) ===", flush=True)
    print(f"  {flipped}/{n} under-recall idxs FIXED to self-rank-1 "
          f"({100*summary['fix_rate']:.1f}%)", flush=True)
    print(f"  mean self_cum delta: {summary['mean_self_cum_delta']:+.0f}",
          flush=True)
    print(f"  verdict: {'ARTIFACT-SAFE FIX WORKS' if summary['fix_rate']>=0.5 else 'INSUFFICIENT (negative/partial)'}",
          flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"summary": summary, "rows": rows},
                   open(args.out, "w"), indent=2)
        print(f"  -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
