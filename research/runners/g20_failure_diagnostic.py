"""Diagnostic tool: WHY do specific words fail in G.20 bridges?

For each failing word at a given seed, measures:
1. Target slice's firing rate during stim (slice_rate_target)
2. Winning off-target slice's firing rate (slice_rate_winner)
3. Ratio (target / winner) — low ratio = weak target slice
4. Pre-training topographic-prior weight sum on target slice
5. Post-training weight sum on target slice
6. Engram tag overlap with target slice (how much of the captured
   top-K=100 is in slice N vs off-slice)

Goal: identify whether failures are caused by:
- A. Weak random init (low pre-prior weights on target slice)
- B. Topographic prior insufficient (low post-prior weights even after boost)
- C. Engram tag pollution (top-K captures off-slice neurons)
- D. Cross-slice interference (off-slice winning despite trained target)

Reads pre-trained shared_pool bridge + concept_pool_demo_shared result JSON.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bridge", type=str, required=True,
                    help="Path to trained .simstate.h5")
    p.add_argument("--result-json", type=str, required=True,
                    help="Path to result JSON (has rank info)")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--n-lang-input", type=int, default=8192)
    p.add_argument("--n-shared-pool", type=int, default=1600)
    p.add_argument("--slice-size", type=int, default=50)
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    from research.runners.concept_pool_demo_shared import (
        build_shared_pool_bridge,
    )
    from sim.backend import get_backend
    cp, _ = get_backend()

    result = json.loads(Path(args.result_json).read_text())
    vocab = result["vocab"]
    n_concepts = len(vocab)

    # Identify failed words (rank > 5 OR rank > 1 with low ratio)
    failed = [r for r in result["results"] if r["rank"] > 1]
    print(f"Loading bridge + analyzing {len(failed)} non-top-1 words...",
          flush=True)

    bridge = build_shared_pool_bridge(
        seed=args.seed,
        n_lang_input=args.n_lang_input,
        n_shared_pool=args.n_shared_pool,
        n_shared_fs=200,
        n_lang_output=args.n_lang_input,
        verbose=False,
    )
    bridge.load_checkpoint(args.bridge)

    rm = bridge.region_manager
    shared_indices = list(rm.indices("shared_concept_pool"))
    lang_input_indices = list(rm.indices("language_input"))

    # Get connections matrix
    indptr = cp.asnumpy(bridge.cp_connections.indptr)
    indices = cp.asnumpy(bridge.cp_connections.indices)
    data = cp.asnumpy(bridge.cp_connections.data)

    diagnostics: List[Dict] = []

    for r in failed:
        word = r["word"]
        target_idx = r["target_idx"]
        target_slice_neurons = set(
            shared_indices[target_idx * args.slice_size:
                            (target_idx + 1) * args.slice_size])
        # Winner slice (max off-target firing during stim)
        slice_rates = r["slice_rates"]
        winner_idx = int(np.argmax([
            s if i != target_idx else -1
            for i, s in enumerate(slice_rates)]))

        # Get lang_input active neurons for this word (orthogonal code)
        from sim.text_embeddings import orthogonal_drive_pattern
        drive = orthogonal_drive_pattern(
            cue_idx=target_idx, n_cues=n_concepts,
            n_neurons=args.n_lang_input,
            drive_max_pA=1.0, sparsity=result.get("prior_stats", {}).get(
                "slice_size", 0.03) / 1000 if False else 0.03,
        )
        # Use the same sparsity as training; fallback to 0.03 default
        active_lang_local = np.where(drive > 0)[0]
        active_lang_global = set(lang_input_indices[i]
                                   for i in active_lang_local)

        # Sum weights from active_lang -> target_slice (post-training)
        target_weight_sum = 0.0
        n_target_edges = 0
        winner_slice_neurons = set(
            shared_indices[winner_idx * args.slice_size:
                            (winner_idx + 1) * args.slice_size])
        winner_weight_sum = 0.0
        n_winner_edges = 0
        n_rows = int(bridge.cp_connections.shape[0])
        for pre_global in active_lang_global:
            start = int(indptr[pre_global])
            end = int(indptr[pre_global + 1])
            for off in range(start, end):
                post = int(indices[off])
                w = float(data[off])
                if post in target_slice_neurons:
                    target_weight_sum += w
                    n_target_edges += 1
                elif post in winner_slice_neurons:
                    winner_weight_sum += w
                    n_winner_edges += 1

        # Engram tag overlap with target slice
        tag_indices = bridge.get_engram_tag_indices(word)
        if hasattr(tag_indices, 'get'):
            tag_indices = tag_indices.get()
        tag_set = set(int(t) for t in np.asarray(tag_indices))
        overlap_with_target = len(tag_set & target_slice_neurons)
        overlap_with_winner = len(tag_set & winner_slice_neurons)

        diag = {
            "word": word,
            "rank": r["rank"],
            "target_rate": r["target_rate"],
            "winner_word": vocab[winner_idx],
            "winner_rate": float(np.max([
                s if i != target_idx else -1
                for i, s in enumerate(slice_rates)])),
            "rate_ratio": (r["target_rate"] /
                            max(r["max_off_rate"], 0.01)),
            "weight_sum_target": target_weight_sum,
            "n_target_edges": n_target_edges,
            "weight_sum_winner": winner_weight_sum,
            "n_winner_edges": n_winner_edges,
            "weight_ratio": (target_weight_sum /
                              max(winner_weight_sum, 0.001)),
            "tag_in_target_slice": overlap_with_target,
            "tag_in_winner_slice": overlap_with_winner,
            "tag_size": len(tag_set),
        }
        diagnostics.append(diag)

    # Print summary
    print(f"\n{'word':12} {'rank':4} {'tgt_rate':10} {'winner':12} "
          f"{'win_rate':10} {'w_tgt':10} {'w_win':10} {'tag_tgt/k':10}")
    print("-" * 100)
    for d in sorted(diagnostics, key=lambda x: -x["rank"]):
        print(f"{d['word']:12} {d['rank']:4} {d['target_rate']:10.1f} "
              f"{d['winner_word']:12} {d['winner_rate']:10.1f} "
              f"{d['weight_sum_target']:10.1f} "
              f"{d['weight_sum_winner']:10.1f} "
              f"{d['tag_in_target_slice']:3}/{d['tag_size']:3}")

    # Classification
    print("\n=== Failure classification ===")
    # A. Weak target weights (post-prior weight sum < 50% of winner's)
    weak_target = [d for d in diagnostics
                    if d["weight_sum_target"] < 0.5 * d["weight_sum_winner"]]
    print(f"A. Weak target weights ({len(weak_target)}/{len(diagnostics)}): "
          f"prior didn't dominate random init")

    # B. Tag pollution (target_slice has < 50% of tag)
    polluted = [d for d in diagnostics
                 if d["tag_in_target_slice"] < d["tag_size"] * 0.5]
    print(f"B. Engram tag pollution ({len(polluted)}/{len(diagnostics)}): "
          f"tag captured mostly off-slice neurons")

    # C. Both A and B (compounding)
    both = [d for d in diagnostics if d in weak_target and d in polluted]
    print(f"   Both A+B (compounding): {len(both)}/{len(diagnostics)}")

    if args.out:
        Path(args.out).write_text(json.dumps({
            "bridge": args.bridge,
            "seed": args.seed,
            "n_concepts": n_concepts,
            "n_failed": len(failed),
            "diagnostics": diagnostics,
            "classification": {
                "weak_target_weights": len(weak_target),
                "tag_pollution": len(polluted),
                "both": len(both),
            },
        }, indent=2, default=str))
        print(f"\n[OUT] -> {args.out}")


if __name__ == "__main__":
    main()
