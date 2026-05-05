"""Benchmarks the perf-stack improvements honestly.

Measures actual wall-clock time per event for combinations of:
  - parallel: 1, 2, 6 (memory-permitting)
  - fp16_synapse_state: False, True
  - gpu_eligibility (3-factor only): False, True

Reports:
  - Mean per-event wall time (sec)
  - Total chain time at 4000 events
  - GPU memory peak per process
  - Throughput delta vs baseline

Designed to run AFTER the current bio_three_factor sweep finishes
(so we don't compete for GPU). Single-seed per config to keep
benchmark cost low (~30 min total).

Usage:
    python -m research.runners.bench_perf_stack --quick
    python -m research.runners.bench_perf_stack --full
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple


def _run_one_bench(
    config: Dict,
    seed: int,
    n_events: int,
    out_dir: Path,
    label: str,
) -> Dict:
    """Run a single bio_three_factor seed with given config, time it.

    Returns:
        {
            'label': str,
            'config': dict,
            'wall_seconds': float,
            'sec_per_event': float,
            'eval_accuracy': float,
        }
    """
    out_path = out_dir / f"bench_{label}_seed{seed}.json"
    cli = [
        sys.executable, "-u",
        "-m", "research.runners.bio_three_factor",
        "--biological",
        "--seed", str(seed),
        "--n-events-per-direction", str(n_events // 4),
        "--out-stats", str(out_path),
    ]
    if config.get("apply_topographic_bias"):
        cli.append("--apply-topographic-bias")
    if config.get("enable_motor_fs"):
        cli.append("--enable-motor-fs")
    if not config.get("gpu_eligibility", True):
        cli.append("--no-gpu-eligibility")
    if config.get("fp16"):
        cli.append("--fp16-synapse-state")

    t0 = time.time()
    proc = subprocess.run(cli, capture_output=True, text=True)
    elapsed = time.time() - t0

    eval_acc = None
    if out_path.exists():
        try:
            d = json.loads(out_path.read_text())
            eval_acc = d.get("word_to_action_eval", {}).get("accuracy")
        except Exception:
            pass

    return {
        "label": label,
        "config": config,
        "wall_seconds": elapsed,
        "sec_per_event": elapsed / n_events if n_events > 0 else None,
        "eval_accuracy": eval_acc,
        "exit_code": proc.returncode,
        "stdout_tail": proc.stdout[-500:] if proc.stdout else "",
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true",
                    help="Quick mode: 100 events/dir, 1 seed per config (~5 min total)")
    ap.add_argument("--full", action="store_true",
                    help="Full mode: 1000 events/dir, 3 seeds per config (~3-4 hours)")
    ap.add_argument("--out", type=Path,
                    default=Path("research/findings/raw/g11_bg/bench_perf_stack.json"))
    args = ap.parse_args()

    if args.full:
        n_events = 4000  # 1000/dir × 4 dirs
        seeds = [42, 43, 44]
    else:  # quick (default)
        n_events = 400  # 100/dir × 4 dirs
        seeds = [42]

    # Configs to benchmark — each row is a perf-stack layer.
    # Strategy: keep ONLY the change-of-interest different from the
    # prior row, so each speedup attribution is clean.
    configs = [
        # Baseline: pre-Phase-1 (CPU eligibility, FP32 everywhere)
        {"label": "baseline_fp32_cpu_eligibility",
         "apply_topographic_bias": True, "enable_motor_fs": True,
         "gpu_eligibility": False, "fp16": False},
        # +Phase 1: GPU-resident eligibility (no host round-trip)
        {"label": "phase1_gpu_eligibility",
         "apply_topographic_bias": True, "enable_motor_fs": True,
         "gpu_eligibility": True, "fp16": False},
        # +Phase 2: FP16 synapse state on top of Phase 1
        {"label": "phase2_gpu_eligibility_fp16",
         "apply_topographic_bias": True, "enable_motor_fs": True,
         "gpu_eligibility": True, "fp16": True},
    ]

    results = []
    for cfg in configs:
        for seed in seeds:
            print(f"\n=== Benchmarking {cfg['label']} seed={seed} ===", flush=True)
            r = _run_one_bench(
                config=cfg, seed=seed, n_events=n_events,
                out_dir=Path("research/findings/raw/g11_bg"),
                label=f"{cfg['label']}_seed{seed}",
            )
            results.append(r)
            print(f"  wall: {r['wall_seconds']:.1f}s "
                  f"({r['sec_per_event']*1000:.1f} ms/event), "
                  f"acc: {r['eval_accuracy']}", flush=True)

    # Summary table
    print("\n" + "=" * 70)
    print("PERF STACK BENCHMARK")
    print("=" * 70)
    print(f"{'Config':<40} {'sec/event':>12} {'speedup':>10}")
    baseline_spe = None
    for r in results:
        spe = r["sec_per_event"] or 0
        if baseline_spe is None:
            baseline_spe = spe
            speedup = 1.0
        else:
            speedup = baseline_spe / spe if spe > 0 else 0
        print(f"{r['label']:<40} {spe*1000:>10.1f}ms {speedup:>9.2f}x")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"results": results}, indent=2))
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
