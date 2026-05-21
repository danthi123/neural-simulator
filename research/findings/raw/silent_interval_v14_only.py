"""V14-only silent-interval persistence probe (Direction L).

Mirrors `silent_interval_persistence_probe.py` but uses the v14-only
substrate builder. Tests whether the silent-interval qualitative
patterns characterized on the unified substrate hold on v14-only.

The unified-substrate seed 42 (Direction E) showed MONOTONIC DECAY
across silent-interval lengths (6.7% at 1k -> 20% at 100k). Direction L
tests v14-only seed 42 800ev with the same protocol.

Reuse: tests test_one_checkpoint_v14 from v14_only_phase1_diagnostic.py;
silent-interval mechanic is the bridge's existing step function with
cp_external_input_current zeroed each step.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from importlib import util as _import_util
_v14_path = os.path.join(_HERE, "v14_only_phase1_diagnostic.py")
_spec = _import_util.spec_from_file_location("_v14", _v14_path)
_v14 = _import_util.module_from_spec(_spec)
_spec.loader.exec_module(_v14)
test_one_checkpoint_v14 = _v14.test_one_checkpoint_v14
_build_v14_only_bridge = _v14._build_v14_only_bridge

from research.runners.unified_per_regime_monitor_runner import (
    _phase1_cache_path,
    _freeze_phase1_gates,
)


def run_silent_interval_v14_and_save(
    seed: int, cache_dir: str, post_silence_cache_dir: str,
    n_silent_steps: int,
):
    print(f"\n=== V14-only silent interval at seed {seed}: {n_silent_steps} steps ===")
    print(f"  Source cache: {cache_dir}")

    bridge = _build_v14_only_bridge(int(seed), False)
    src_path = _phase1_cache_path(cache_dir, seed)
    print(f"  Loading {src_path}")
    bridge.load_checkpoint(str(src_path))
    _freeze_phase1_gates(bridge)

    bridge.cp_external_input_current[:] = 0.0

    print(f"  Running {n_silent_steps} silent steps...")
    t_start = time.time()
    for i in range(n_silent_steps):
        bridge.cp_external_input_current[:] = 0.0
        bridge._run_one_simulation_step()
        if (i + 1) % 1000 == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta = (n_silent_steps - i - 1) / rate
            print(f"    step {i+1}/{n_silent_steps} ({100.0*(i+1)/n_silent_steps:.1f}%); "
                  f"elapsed {elapsed:.1f}s; ETA {eta:.1f}s")
    print(f"  Silent interval complete; {time.time()-t_start:.1f}s wall-clock")

    Path(post_silence_cache_dir).mkdir(parents=True, exist_ok=True)
    dst_path = _phase1_cache_path(post_silence_cache_dir, seed)
    print(f"  Saving post-silence cache to {dst_path}")
    bridge.save_checkpoint(str(dst_path))
    return dst_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-silent-steps", type=int, default=5000)
    parser.add_argument("--ev", type=int, default=800)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    src_cache = f"research/findings/raw/v14_only_per_regime/phase1_{args.ev}ev"
    post_cache = f"research/findings/raw/v14_only_per_regime/phase1_{args.ev}ev_post_silence"

    pre = test_one_checkpoint_v14(
        args.seed, src_cache, f"PRE-silence v14-only {args.ev}ev seed {args.seed}"
    )

    run_silent_interval_v14_and_save(
        args.seed, src_cache, post_cache, args.n_silent_steps
    )

    post = test_one_checkpoint_v14(
        args.seed, post_cache,
        f"POST-silence v14-only {args.ev}ev seed {args.seed}"
    )

    pre_acc = pre["accuracy"]
    post_acc = post["accuracy"]
    fgt = 0.0 if pre_acc == 0 else 100.0 * (pre_acc - post_acc) / pre_acc

    print(f"\n=== DIRECTION L RESULT (v14-only {args.ev}ev seed {args.seed}) ===")
    print(f"  PRE  : {pre['n_correct']}/16 = {100.0*pre_acc:.1f}%")
    print(f"  POST : {post['n_correct']}/16 = {100.0*post_acc:.1f}%")
    print(f"  Forgetting %: {fgt:.1f}%")

    out = args.out or (
        f"research/findings/raw/silent_interval_v14_only_seed{args.seed}_{args.ev}ev_{args.n_silent_steps}.json"
    )
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "seed": args.seed,
            "ev_per_word": args.ev,
            "n_silent_steps": args.n_silent_steps,
            "src_cache": src_cache,
            "post_silence_cache": post_cache,
            "pre_n_correct": pre["n_correct"],
            "pre_accuracy": pre_acc,
            "post_n_correct": post["n_correct"],
            "post_accuracy": post_acc,
            "forgetting_pct": fgt,
            "pre_per_word": pre["per_word"],
            "post_per_word": post["per_word"],
        }, f, indent=2)
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
