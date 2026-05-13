"""concept_weight_probe — inspect lang_input -> pool weights after training.

After concept_pool_demo trains a bridge (with --save-bridge), load the
checkpoint and report:
- Mean weight per (word, pool) — should be high for target, low for off-target
- Topographic ratio target/off-target — Tier 1 baseline is ~2.1x
- Per-pool activated-lang_input-neuron weight matrix

Diagnostic value: if Phase 1 cross-category isolation fails, this probe
tells us whether the failure is in:
- STDP not converging (weights look random)
- Topographic prior eroded (target close to off-target weights)
- Lang_input drive pattern overlap (different words activate overlapping
  lang_input neurons -> can't differentiate at pathway level)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from research.runners.concept_pool_demo import (
    DIRECTION_VOCAB, NOUN_VOCAB, VERB_VOCAB,
    NOUN_NAMES, VERB_NAMES, MOTOR_NAMES,
    build_concept_bridge,
)


def _to_host(arr):
    try:
        from sim.backend import get_backend
        cp, _ = get_backend()
        return cp.asnumpy(arr)
    except Exception:
        import numpy as np
        return np.asarray(arr)


def extract_pathway_weights(bridge, src_indices, dst_indices,
                              return_full_matrix: bool = False):
    """Extract weights from src_indices -> dst_indices in current bridge.

    Returns dict with:
      - "n_edges": count of edges between src and dst
      - "mean": mean weight
      - "std": std deviation
      - "min", "max": extremes
      - "full" (optional): NxM matrix
    """
    import numpy as np
    indptr = _to_host(bridge.cp_connections.indptr)
    indices = _to_host(bridge.cp_connections.indices)
    data = _to_host(bridge.cp_connections.data)
    src_set = set(src_indices)
    dst_set = set(dst_indices)
    weights: List[float] = []
    if return_full_matrix:
        # Map (src, dst) -> weight
        full = np.zeros((len(src_indices), len(dst_indices)), dtype=np.float32)
        src_to_row = {s: i for i, s in enumerate(src_indices)}
        dst_to_col = {d: i for i, d in enumerate(dst_indices)}
    for r in src_indices:
        if r not in src_set:
            continue
        start = int(indptr[r])
        end = int(indptr[r + 1])
        for off in range(start, end):
            dst = int(indices[off])
            if dst in dst_set:
                w = float(data[off])
                weights.append(w)
                if return_full_matrix:
                    full[src_to_row[r], dst_to_col[dst]] = w
    if not weights:
        return {"n_edges": 0, "mean": 0, "std": 0, "min": 0, "max": 0}
    arr = np.asarray(weights, dtype=np.float64)
    out = {
        "n_edges": len(weights),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }
    if return_full_matrix:
        out["full"] = full
    return out


def probe_word_to_pool_weights(bridge,
                                 n_lang_input: int = 4096,
                                 sparsity: float = 0.1) -> Dict:
    """For each word, measure mean weight to each pool from word's active lang_input neurons."""
    import numpy as np
    from sim.text_embeddings import vocab_to_drive_pattern

    rm = bridge.region_manager
    lang_input_indices = list(rm.indices("language_input"))

    all_pools = (
        [f"motor_{a}" for a in MOTOR_NAMES]
        + [f"noun_pool_{n}" for n in NOUN_NAMES]
        + [f"verb_pool_{v}" for v in VERB_NAMES]
    )
    pool_indices = {p: list(rm.indices(p)) for p in all_pools}

    all_words = list(DIRECTION_VOCAB) + list(NOUN_VOCAB) + list(VERB_VOCAB)
    word_to_target = {}
    for w, a in DIRECTION_VOCAB.items():
        word_to_target[w] = f"motor_{a}"
    for w, n in NOUN_VOCAB.items():
        word_to_target[w] = f"noun_pool_{n}"
    for w, v in VERB_VOCAB.items():
        word_to_target[w] = f"verb_pool_{v}"

    result = {}
    for word in all_words:
        drive = vocab_to_drive_pattern(
            word, n_neurons=n_lang_input, sparsity=sparsity,
        )
        local_active = np.where(drive > 0)[0]
        global_active = [lang_input_indices[i] for i in local_active]

        per_pool_weights = {}
        for pool in all_pools:
            stats = extract_pathway_weights(
                bridge, global_active, pool_indices[pool],
            )
            per_pool_weights[pool] = stats
        target = word_to_target[word]
        target_mean = per_pool_weights[target]["mean"]
        # off-target mean = max over OTHER pools (most concerning)
        max_off_pool = max(
            (p for p in all_pools if p != target),
            key=lambda p: per_pool_weights[p]["mean"],
        )
        max_off_mean = per_pool_weights[max_off_pool]["mean"]
        ratio = target_mean / max(max_off_mean, 0.001)
        result[word] = {
            "target_pool": target,
            "target_mean_weight": target_mean,
            "max_off_pool": max_off_pool,
            "max_off_mean_weight": max_off_mean,
            "target_to_max_off_ratio": ratio,
            "per_pool": {
                p: {"mean": stats["mean"], "n_edges": stats["n_edges"]}
                for p, stats in per_pool_weights.items()
            },
        }
    return result


def main():
    parser = argparse.ArgumentParser(description="Probe concept-pool weights.")
    parser.add_argument("--checkpoint", type=str, required=True,
                         help="Path to .simstate.h5 from --save-bridge")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-lang-input", type=int, default=4096)
    parser.add_argument("--n-per-pool", type=int, default=500)
    parser.add_argument("--n-fs-per-pool", type=int, default=60)
    parser.add_argument("--out", type=str, default=None,
                         help="Output JSON path")
    args = parser.parse_args()

    # Build bridge skeleton then load checkpoint
    print(f"[probe] building bridge (seed={args.seed})", flush=True)
    bridge = build_concept_bridge(
        seed=args.seed,
        n_lang_input=args.n_lang_input,
        n_per_pool=args.n_per_pool,
        n_fs_per_pool=args.n_fs_per_pool,
        verbose=False,
    )
    print(f"[probe] loading {args.checkpoint}", flush=True)
    bridge.load_checkpoint(args.checkpoint)

    print(f"[probe] measuring weights...", flush=True)
    result = probe_word_to_pool_weights(
        bridge, n_lang_input=args.n_lang_input,
    )

    print(f"\n[RESULTS] per-word target vs max-off-target weights:")
    print(f"{'word':10s} {'target_pool':22s} {'target_w':>10s} "
          f"{'max_off_pool':>22s} {'max_off_w':>10s} {'ratio':>8s}")
    print("-" * 90)
    for word, r in result.items():
        print(f"{word:10s} {r['target_pool']:22s} {r['target_mean_weight']:10.3f} "
              f"{r['max_off_pool']:>22s} {r['max_off_mean_weight']:10.3f} "
              f"{r['target_to_max_off_ratio']:8.2f}x")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2))
        print(f"\n[OUT] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
