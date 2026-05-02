"""Diagnostic: dump language pathway weights from a saved bridge checkpoint.

For each text-IO pathway:
  language_input -> cortex_{N,E,S,W}
  language_input -> motor_{N,E,S,W} (PFC-bypass)
  language_input -> dlpfc_wm
  IT -> language_output
  cortex_{N,E,S,W} -> language_output

Computes per-pathway:
  - n_synapses (CSR-counted)
  - weight mean / std / min / max
  - per-token-vs-target-pool weight bias: when language_input "north"
    pattern is active, what's the mean weight from those active sources
    to motor_N vs motor_E/S/W?

Identifies whether STDP differentiated "north" -> motor_N over the
other directions, even if the eval accuracy stayed at chance. A trained
network that learned the mapping should show:
  W(north_active, motor_N) > W(north_active, motor_{E,S,W})

If all four pools have similar mean weights for "north_active" sources,
the language pathway didn't learn discrimination.

Usage:
  python -m research.runners.text_weight_diagnostic <checkpoint.h5> \\
      [--out report.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=str)
    ap.add_argument("--out", type=str, default=None,
                    help="optional JSON output file (also pretty-prints)")
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        ap.error(f"checkpoint not found: {ckpt}")

    # Load bridge (uses same path as text_reeval.py)
    from research.runners.text_reeval import load_bridge
    bridge = load_bridge(str(ckpt))

    import cupy as cp

    # Get region indices
    rm = bridge.region_manager
    DIRS = ["N", "E", "S", "W"]
    DIR_TOKENS = {"N": "north", "E": "east", "S": "south", "W": "west"}

    region_indices = {}
    for r in ["language_input", "language_output", "cortex_it", "dlpfc_wm"]:
        try:
            region_indices[r] = list(rm.indices(r))
        except KeyError:
            print(f"  WARN: region '{r}' not found in bridge")
            region_indices[r] = []
    for d in DIRS:
        for prefix in ["cortex", "motor"]:
            name = f"{prefix}_{d}"
            try:
                region_indices[name] = list(rm.indices(name))
            except KeyError:
                region_indices[name] = []

    # Convert connection matrix to CPU for analysis
    # cp_connections is a CSR sparse matrix
    csr = bridge.cp_connections
    if csr is None:
        print("  ERROR: no cp_connections matrix")
        return
    indptr = cp.asnumpy(csr.indptr)
    indices = cp.asnumpy(csr.indices)
    data = cp.asnumpy(csr.data)
    print(f"Loaded {len(data)} synapses (csr shape {csr.shape})")

    # Helper: extract weights from src_indices to dst_indices
    def weights_between(src_ids, dst_ids):
        """Returns array of weights for synapses src->dst (any pre, any post)."""
        if not src_ids or not dst_ids:
            return np.array([], dtype=np.float32)
        src_set = set(src_ids)
        dst_set = set(dst_ids)
        weights = []
        for src in src_ids:
            row_start, row_end = indptr[src], indptr[src + 1]
            for j in range(row_start, row_end):
                if indices[j] in dst_set:
                    weights.append(data[j])
        return np.array(weights, dtype=np.float32)

    # Compute per-pathway stats
    pathways = []
    for d in DIRS:
        # language_input -> cortex_d
        w = weights_between(region_indices["language_input"],
                            region_indices[f"cortex_{d}"])
        pathways.append((f"lang_in -> cortex_{d}", w))
        # language_input -> motor_d (PFC bypass)
        w = weights_between(region_indices["language_input"],
                            region_indices[f"motor_{d}"])
        pathways.append((f"lang_in -> motor_{d}", w))
        # cortex_d -> language_output
        w = weights_between(region_indices[f"cortex_{d}"],
                            region_indices["language_output"])
        pathways.append((f"cortex_{d} -> lang_out", w))

    if region_indices.get("cortex_it") and region_indices.get("language_output"):
        w = weights_between(region_indices["cortex_it"],
                            region_indices["language_output"])
        pathways.append(("IT -> lang_out", w))

    if region_indices.get("language_input") and region_indices.get("dlpfc_wm"):
        w = weights_between(region_indices["language_input"],
                            region_indices["dlpfc_wm"])
        pathways.append(("lang_in -> dlpfc_wm", w))

    # Pretty-print + collect for JSON
    report = {
        "checkpoint": str(ckpt),
        "n_total_synapses": int(len(data)),
        "pathways": [],
    }
    print(f"\n{'pathway':<30} {'n_syn':>8} {'mean':>8} {'std':>8} "
          f"{'min':>8} {'max':>8}")
    print("-" * 80)
    for name, w in pathways:
        if w.size == 0:
            row = {"name": name, "n": 0, "mean": None, "std": None,
                   "min": None, "max": None}
            print(f"{name:<30} {0:>8} {'(empty)':>8}")
        else:
            row = {"name": name, "n": int(w.size),
                   "mean": float(w.mean()), "std": float(w.std()),
                   "min": float(w.min()), "max": float(w.max())}
            print(f"{name:<30} {row['n']:>8} {row['mean']:>8.3f} "
                  f"{row['std']:>8.3f} {row['min']:>8.3f} {row['max']:>8.3f}")
        report["pathways"].append(row)

    # Token-targeted analysis: for each direction word, compute the
    # mean weight from active language_input neurons (~26 active per token)
    # to each motor pool.  If "north" learned its mapping, we expect:
    #   W(north_active, motor_N) >> W(north_active, motor_E/S/W)
    print(f"\n{'='*80}")
    print("Token-targeted analysis (PFC-bypass: lang_in active for token -> motor_X)")
    print(f"{'='*80}")
    print(f"{'token':<8} {'active_neurons':<14} "
          f"{'->motor_N':>10} {'->motor_E':>10} "
          f"{'->motor_S':>10} {'->motor_W':>10}  diff_target")
    print("-" * 80)
    from sim.text_embeddings import vocab_to_drive_pattern
    n_lang_in = len(region_indices["language_input"])
    token_results = {}
    for token, target in [("north", "N"), ("east", "E"),
                          ("south", "S"), ("west", "W")]:
        drive = vocab_to_drive_pattern(token, n_neurons=n_lang_in,
                                       drive_max_pA=1.0, sparsity=0.1)
        # Active language_input neuron INDICES
        active_local = np.where(drive > 0)[0]
        active_global = [region_indices["language_input"][i]
                         for i in active_local.tolist()]
        means = {}
        for d in DIRS:
            w = weights_between(active_global, region_indices[f"motor_{d}"])
            means[d] = float(w.mean()) if w.size > 0 else 0.0
        # Differential: target mean - mean of others
        non_target = np.mean([means[d] for d in DIRS if d != target])
        diff_target = means[target] - non_target
        token_results[token] = {
            "active_neurons": len(active_global),
            "means": means,
            "target": target,
            "diff_target": diff_target,
        }
        diff_str = f"{diff_target:+.4f}"
        if diff_target > 0:
            diff_str += " (LEARNED!)"
        print(f"{token:<8} {len(active_global):<14} "
              f"{means['N']:>10.3f} {means['E']:>10.3f} "
              f"{means['S']:>10.3f} {means['W']:>10.3f}  {diff_str}")

    report["token_to_motor_analysis"] = token_results

    # Verdict: did STDP differentiate the mapping?
    n_learned = sum(1 for v in token_results.values() if v["diff_target"] > 0)
    print(f"\n{'='*80}")
    print(f"Verdict: {n_learned}/4 tokens have target-bias > 0")
    if n_learned == 4:
        print("  STRONG: all 4 tokens prefer their target motor pool")
    elif n_learned >= 2:
        print("  PARTIAL: some tokens learned; others did not")
    else:
        print("  CHANCE: weights essentially unchanged from random init")
    report["verdict"] = {
        "n_learned": n_learned,
        "summary": "STRONG" if n_learned == 4 else "PARTIAL" if n_learned >= 2 else "CHANCE",
    }

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2, default=str))
        print(f"\nReport saved: {args.out}")


if __name__ == "__main__":
    main()
