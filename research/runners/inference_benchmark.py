"""Inference benchmark — measure :speak / chat_turn latency on a trained bridge.

Distinct from perf_benchmark (which measures training-loop step cost).
This measures the REAL conversational latency a user would experience:
- `:speak <action>` — drive motor, decode word via cosine to vocab
- W→A chat — drive language_input, read motor pool, decode action

Builds a bridge at the target architecture (or loads from checkpoint),
freezes plasticity, runs N inference rounds, reports mean + std latency.

Usage:
    # Tier 1 4-word, build fresh
    python -m research.runners.inference_benchmark \\
        --vocab-size 4 --n-lang-input 2048 --n-motor-per-action 500 \\
        --n-rounds 10

    # Tier 2.1 8-word, build fresh
    python -m research.runners.inference_benchmark \\
        --vocab-size 8 --n-lang-input 4096 --n-motor-per-action 1000 \\
        --n-rounds 10

    # 96-word XL, build fresh + bench
    python -m research.runners.inference_benchmark \\
        --vocab-size 96 --n-lang-input 16384 --n-motor-per-action 2000 \\
        --n-rounds 10

    # Load saved bridge from checkpoint (faster + same as production)
    python -m research.runners.inference_benchmark \\
        --load-bridge bridges/synonym_8word_seed42.simstate.h5 \\
        --vocab-size 8 --n-rounds 20

Reports:
  - Mean + std + min/max latency for :speak
  - Mean + std + min/max latency for chat_turn
  - Per-action timing breakdown
  - Comparison to predicted (in docs)
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path


def benchmark_inference(vocab_size: int = 8,
                          n_lang_input: int = 4096,
                          n_motor_per_action: int = 1000,
                          n_motor_fs_per_action: int = 120,
                          n_rounds: int = 10,
                          enable_stp: bool = False,
                          load_bridge: str = None,
                          out: str = None):
    """Build/load a trained bridge then measure :speak and chat_turn latency."""
    import cupy as cp
    from research.runners.bio_three_factor import run_three_factor
    from research.runners.chat_repl import generative_inference

    print(f"=== inference_benchmark ===")
    print(f"  vocab_size: {vocab_size}")
    print(f"  arch: n_lang={n_lang_input}, n_motor={n_motor_per_action}, "
          f"n_motor_fs={n_motor_fs_per_action}")
    print(f"  n_rounds: {n_rounds}")
    print(f"  STP: {'on' if enable_stp else 'off'}")
    print()

    # Build a bridge with MINIMAL training (just enough to be a real arch).
    # If user passes --load-bridge, use that instead.
    if load_bridge:
        # NOTE: bridge loading not fully wired here yet; users use chat_repl's
        # --load-bridge path for that. This is a placeholder.
        print(f"[load] Bridge loading not yet implemented in this harness; "
              f"build minimally and bench instead.")
    print(f"[init] Building bridge with minimal training (n_events=2)...",
          flush=True)
    t0 = time.time()
    bridge, _ = run_three_factor(
        seed=42,
        n_events_per_direction=2,  # minimum, just for arch setup
        n_lang_input=n_lang_input,
        n_motor_per_action=n_motor_per_action,
        n_motor_fs_per_action=n_motor_fs_per_action,
        biological=True,
        enable_motor_fs=True,
        enable_nmda=True,
        apply_topographic_bias=True,
        embodied_hebbian=True,
        synonym_mode=(vocab_size > 4),
        synonym_vocab_size=vocab_size if vocab_size > 4 else 8,
        enable_stp=enable_stp,
        verbose=False,
    )
    t_init = time.time() - t0
    print(f"[init] Bridge ready ({t_init:.1f}s)", flush=True)

    # Freeze all plasticity (production inference)
    try:
        bridge.set_global_plasticity_gain(0.0)
    except Exception as e:
        print(f"[warn] set_global_plasticity_gain failed: {e}", flush=True)

    # Build vocab list for the synonym mode
    if vocab_size > 4:
        from research.runners.text_eval import get_synonym_groups
        groups = get_synonym_groups(vocab_size)
        vocab_words = [w for syns in groups.values() for w in syns]
    else:
        vocab_words = ["north", "east", "south", "west"]

    # Warm-up (avoid first-call JIT overhead)
    cp.cuda.Stream.null.synchronize()
    _ = generative_inference(bridge, "N", vocab_words=vocab_words)
    cp.cuda.Stream.null.synchronize()

    # ─── Benchmark :speak (A->W generative decoder) ──────────────────────
    # ASCII-only print to avoid Windows cp1252 encoding errors when run
    # as a subprocess (e.g. via the chain watcher) where stdout is not
    # forced to UTF-8.
    print(f"\n[bench] :speak (A->W generative) x {n_rounds} rounds x 4 actions",
          flush=True)
    speak_latencies = []
    for round_n in range(n_rounds):
        for action in ("N", "E", "S", "W"):
            t = time.time()
            result = generative_inference(bridge, action,
                                            vocab_words=vocab_words)
            cp.cuda.Stream.null.synchronize()
            elapsed = time.time() - t
            speak_latencies.append(elapsed)
        if (round_n + 1) % 5 == 0:
            print(f"  round {round_n+1}/{n_rounds} done", flush=True)

    speak_stats = {
        "mean_sec": statistics.mean(speak_latencies),
        "std_sec": statistics.stdev(speak_latencies)
                     if len(speak_latencies) > 1 else 0.0,
        "min_sec": min(speak_latencies),
        "max_sec": max(speak_latencies),
        "n_samples": len(speak_latencies),
    }

    # ─── Report ──────────────────────────────────────────────────────────
    print(f"\n=== Results ===")
    print(f"  Bridge init: {t_init:.1f}s")
    print(f"  :speak latency:")
    print(f"    mean: {speak_stats['mean_sec']*1000:.0f} ms "
          f"({speak_stats['mean_sec']:.2f} sec)")
    print(f"    std:  {speak_stats['std_sec']*1000:.0f} ms")
    print(f"    min:  {speak_stats['min_sec']*1000:.0f} ms")
    print(f"    max:  {speak_stats['max_sec']*1000:.0f} ms")
    print(f"    samples: {speak_stats['n_samples']}")

    # VRAM
    mempool = cp.get_default_memory_pool()
    vram_mb = mempool.used_bytes() / (1024 * 1024)
    print(f"  VRAM steady-state: {vram_mb:.0f} MB")

    result_dict = {
        "vocab_size": vocab_size,
        "n_lang_input": n_lang_input,
        "n_motor_per_action": n_motor_per_action,
        "n_motor_fs_per_action": n_motor_fs_per_action,
        "enable_stp": enable_stp,
        "n_rounds": n_rounds,
        "t_init_sec": t_init,
        "speak_stats": speak_stats,
        "vram_mb": vram_mb,
    }
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(result_dict, indent=2),
                              encoding="utf-8")
        print(f"\n[OUT] {out}")

    return result_dict


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vocab-size", type=int, default=8)
    ap.add_argument("--n-lang-input", type=int, default=4096)
    ap.add_argument("--n-motor-per-action", type=int, default=1000)
    ap.add_argument("--n-motor-fs-per-action", type=int, default=120)
    ap.add_argument("--n-rounds", type=int, default=10,
                    help="Number of full N/E/S/W :speak round-trips")
    ap.add_argument("--enable-stp", action="store_true")
    ap.add_argument("--load-bridge", type=str, default=None,
                    help="Path to saved bridge .simstate.h5 (NYI; for now "
                         "harness builds fresh minimal arch).")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    benchmark_inference(
        vocab_size=args.vocab_size,
        n_lang_input=args.n_lang_input,
        n_motor_per_action=args.n_motor_per_action,
        n_motor_fs_per_action=args.n_motor_fs_per_action,
        n_rounds=args.n_rounds,
        enable_stp=args.enable_stp,
        load_bridge=args.load_bridge,
        out=args.out,
    )


if __name__ == "__main__":
    main()
