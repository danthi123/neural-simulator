"""Performance benchmark harness for sim/bridge inner loop.

Runs a controlled bridge config for N simulation steps and reports:
- Wall clock breakdown (init / per-step / final)
- Steps per second
- Per-step latency (ms)
- VRAM usage at peak
- Optional: cProfile breakdown of Python-side hot paths
- Optional: nsys trace export for CUDA kernel inspection

Usage:
    # Quick smoke (1000 steps, default arch)
    python -m research.runners.perf_benchmark --steps 1000

    # Compare FP32 vs FP16 (run twice, compare wall-clock)
    python -m research.runners.perf_benchmark --steps 1000 --fp16

    # With Python profiling output
    python -m research.runners.perf_benchmark --steps 1000 --profile

    # Compare freeze-plasticity-during-reset variants
    python -m research.runners.perf_benchmark --steps 1000 --freeze-plasticity-during-reset

    # Larger arch (matches Tier 2.1 v4)
    python -m research.runners.perf_benchmark --steps 5000 --vocab-size 8 \\
        --n-lang-input 4096 --n-motor-per-action 1000

The harness uses chat_synonym_demo's train_chat_bridge to set up a
realistic bridge with biology-grounded plasticity active. Then it runs
a fixed number of bridge.runs_one_simulation_step() calls in a tight
loop while measuring wall-clock + GPU memory.

Output:
    perf_benchmark output:
      arch: lang=4096 motor=1000 motor_fs=120 (Tier 2.1 v4)
      config: fp16_synapse_state=False, freeze_plasticity_during_reset=False
      bridge init: 12.4s
      1000 steps: 18.7s (53.5 steps/sec, 18.7 ms/step)
      VRAM peak: 6234 MB

Use this harness to validate optimization claims (e.g., "FP16 is 1.3x
faster") with empirical data BEFORE deploying changes broadly.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
import json
import sys


def benchmark(
    n_steps: int,
    fp16_synapse_state: bool = False,
    freeze_plasticity_during_reset: bool = False,
    vocab_size: int = 8,
    n_lang_input: int = 4096,
    n_motor_per_action: int = 1000,
    n_motor_fs_per_action: int = 120,
    profile: bool = False,
    seed: int = 42,
    out: str = None,
):
    """Run a controlled bridge and measure inner-loop wall clock."""
    print(f"=== perf_benchmark ===")
    print(f"  n_steps: {n_steps}")
    print(f"  arch: lang={n_lang_input} motor={n_motor_per_action} "
          f"motor_fs={n_motor_fs_per_action}")
    print(f"  config: fp16_synapse_state={fp16_synapse_state}, "
          f"freeze_plasticity_during_reset={freeze_plasticity_during_reset}")
    print(f"  vocab_size: {vocab_size}")
    print(f"  seed: {seed}")
    print()

    # Build a realistic bridge using bio_three_factor scaffolding.
    # Run with n_events=2 (minimum) to set up arch + plasticity gates.
    import cupy as cp
    from research.runners.bio_three_factor import run_three_factor

    print("[init] Building bridge...", flush=True)
    t0 = time.time()
    bridge, _ = run_three_factor(
        seed=seed,
        n_events_per_direction=2,  # minimal training, just for arch setup
        n_lang_input=n_lang_input,
        n_motor_per_action=n_motor_per_action,
        n_motor_fs_per_action=n_motor_fs_per_action,
        biological=True,
        enable_motor_fs=True,
        enable_nmda=True,
        apply_topographic_bias=True,
        embodied_hebbian=True,
        synonym_mode=True,
        synonym_vocab_size=vocab_size,
        fp16_synapse_state=fp16_synapse_state,
        freeze_plasticity_during_reset=freeze_plasticity_during_reset,
        verbose=False,
    )
    t_init = time.time() - t0
    print(f"[init] Bridge ready ({t_init:.1f}s)", flush=True)

    # Enable bridge's built-in step profiler. It times 7 sections per step
    # (init/stp/syn/dyn/plast/homeo/final) and emits a [PROFILER] summary
    # line every 500 steps. Captured naturally in stdout.
    bridge.gpu_config.enable_step_profiler = True

    # Warm-up (avoid first-call JIT overhead)
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(50):
        bridge._run_one_simulation_step()
    cp.cuda.Stream.null.synchronize()
    # Reset profiler accumulators after warm-up so the warm-up doesn't
    # contaminate the timing.
    if hasattr(bridge, "_prof_accum"):
        bridge._prof_accum = None
        bridge._prof_count = 0

    # Actual benchmark loop
    print(f"\n[bench] Running {n_steps} simulation steps...", flush=True)
    t_start = time.time()
    if profile:
        import cProfile
        import pstats
        prof = cProfile.Profile()
        prof.enable()
    for i in range(n_steps):
        bridge._run_one_simulation_step()
        if (i + 1) % 500 == 0:
            cp.cuda.Stream.null.synchronize()
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            print(f"  step {i+1}/{n_steps}  "
                  f"({elapsed:.1f}s elapsed, {rate:.1f} steps/sec)",
                  flush=True)
    cp.cuda.Stream.null.synchronize()
    if profile:
        prof.disable()
    t_bench = time.time() - t_start

    # Memory + summary
    mempool = cp.get_default_memory_pool()
    vram_peak_mb = mempool.used_bytes() / (1024 * 1024)

    print(f"\n=== Results ===")
    print(f"  init:        {t_init:.1f}s")
    print(f"  {n_steps} steps:    {t_bench:.1f}s")
    print(f"  steps/sec:   {n_steps / t_bench:.1f}")
    print(f"  ms/step:     {1000 * t_bench / n_steps:.2f}")
    print(f"  VRAM peak:   {vram_peak_mb:.0f} MB")
    print(f"  config:      fp16={fp16_synapse_state}, "
          f"freeze_reset={freeze_plasticity_during_reset}")

    if profile:
        print(f"\n=== Top Python hot paths ===")
        stats = pstats.Stats(prof).sort_stats("cumulative")
        stats.print_stats(15)

    if out:
        result = {
            "n_steps": n_steps,
            "vocab_size": vocab_size,
            "n_lang_input": n_lang_input,
            "n_motor_per_action": n_motor_per_action,
            "n_motor_fs_per_action": n_motor_fs_per_action,
            "fp16_synapse_state": fp16_synapse_state,
            "freeze_plasticity_during_reset": freeze_plasticity_during_reset,
            "seed": seed,
            "t_init_sec": t_init,
            "t_bench_sec": t_bench,
            "steps_per_sec": n_steps / t_bench,
            "ms_per_step": 1000 * t_bench / n_steps,
            "vram_peak_mb": vram_peak_mb,
        }
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\n[OUT] {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=1000,
                    help="Number of inner-loop sim steps (default: 1000)")
    ap.add_argument("--fp16", action="store_true",
                    dest="fp16_synapse_state",
                    help="Enable fp16_synapse_state (FP16 eligibility traces)")
    ap.add_argument("--freeze-plasticity-during-reset", action="store_true",
                    help="Enable plasticity freeze during reset_steps")
    ap.add_argument("--vocab-size", type=int, default=8,
                    choices=[8, 12, 16, 24, 32, 48, 64, 96, 128, 256])
    ap.add_argument("--n-lang-input", type=int, default=4096)
    ap.add_argument("--n-motor-per-action", type=int, default=1000)
    ap.add_argument("--n-motor-fs-per-action", type=int, default=120)
    ap.add_argument("--profile", action="store_true",
                    help="Print top-15 Python hot paths via cProfile")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default=None,
                    help="Optional JSON output path for results")
    args = ap.parse_args()

    benchmark(
        n_steps=args.steps,
        fp16_synapse_state=args.fp16_synapse_state,
        freeze_plasticity_during_reset=args.freeze_plasticity_during_reset,
        vocab_size=args.vocab_size,
        n_lang_input=args.n_lang_input,
        n_motor_per_action=args.n_motor_per_action,
        n_motor_fs_per_action=args.n_motor_fs_per_action,
        profile=args.profile,
        seed=args.seed,
        out=args.out,
    )


if __name__ == "__main__":
    main()
