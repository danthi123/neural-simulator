"""Memory persistence diagnostic: silent interval across training-event regimes.

Direction E (single-seed cheap-first; seed 42 across 4 existing caches).

For each of the 4 multi-seed cached substrates (200ev, 300ev, 400ev,
800ev), load the cache, run N silent steps (cp_external_input_current
zeroed each step; substrate's own dynamics + plasticity + homeostasis
proceed), save the post-silence state as a new cache, then run the
existing direct binding diagnostic against the post-silence cache.

Reports: pre-silence direct binding accuracy vs post-silence direct
binding accuracy per cache; forgetting % per cache.

Hypothesis (from CLS theory): the substrate's training-event regimes
are RETENTION regimes too. DIRECT-FAVORED (800ev) schema-consolidated
substrate should retain direct binding best. COMPOSITIONAL-FAVORED
(200ev) episodic-style substrate should show more forgetting.
SUB-OPTIMAL VALLEY (300ev) should show worst retention.
TRANSITIONAL (400ev) should be intermediate.

Reuse: test_one_checkpoint byte-unchanged via the wrapper pattern.
No new core code. Silent interval is just the bridge's existing
step function with cp_external_input_current set to zero each step.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Reuse byte-unchanged
from importlib import util as _import_util
_diag_path = os.path.join(_HERE, "direct_binding_phase1_comparison.py")
_spec = _import_util.spec_from_file_location("_db", _diag_path)
_db = _import_util.module_from_spec(_spec)
_spec.loader.exec_module(_db)
test_one_checkpoint = _db.test_one_checkpoint

import research.runners.unified_per_regime_monitor_runner as urr
from research.runners.unified_per_regime_monitor_runner import (
    _build_bridge_with_phase1_recipe,
    _phase1_cache_path,
    _freeze_phase1_gates,
)


def run_silent_interval_and_save(
    seed: int, cache_dir: str, post_silence_cache_dir: str,
    n_silent_steps: int,
):
    """Load cache, run N silent steps, save post-silence state.

    Silence: cp_external_input_current is zeroed each step.
    All other dynamics (synaptic transmission, neuron firing, STP,
    STDP, homeostasis, neuromodulator subsystem if any) proceed.
    """
    print(f"\n=== Silent interval at seed {seed}: {n_silent_steps} steps ===")
    print(f"  Source cache: {cache_dir}")

    bridge = _build_bridge_with_phase1_recipe(int(seed), False)
    src_path = _phase1_cache_path(cache_dir, seed)
    print(f"  Loading {src_path}")
    bridge.load_checkpoint(str(src_path))
    _freeze_phase1_gates(bridge)

    # Initialize external input to zero
    bridge.cp_external_input_current[:] = 0.0

    print(f"  Running {n_silent_steps} silent steps...")
    import time
    t_start = time.time()
    for i in range(n_silent_steps):
        # Ensure no driven input
        bridge.cp_external_input_current[:] = 0.0
        bridge._run_one_simulation_step()
        if (i + 1) % 1000 == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta = (n_silent_steps - i - 1) / rate
            print(f"    step {i+1}/{n_silent_steps} ({100.0*(i+1)/n_silent_steps:.1f}%); "
                  f"elapsed {elapsed:.1f}s; ETA {eta:.1f}s")
    print(f"  Silent interval complete; {time.time()-t_start:.1f}s wall-clock")

    # Save post-silence state
    Path(post_silence_cache_dir).mkdir(parents=True, exist_ok=True)
    dst_path = _phase1_cache_path(post_silence_cache_dir, seed)
    print(f"  Saving post-silence cache to {dst_path}")
    bridge.save_checkpoint(str(dst_path))
    return dst_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Single seed for cheap-first probe."
    )
    parser.add_argument(
        "--n-silent-steps", type=int, default=5000,
        help="Silent interval length in bridge steps (default 5000)."
    )
    parser.add_argument(
        "--ev-list", type=int, nargs="+", default=[200, 300, 400, 800],
        help="Training-event budgets to test (default 200 300 400 800)."
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="Output JSON path (default silent_interval_persistence_probe.json)."
    )
    args = parser.parse_args()

    results = []
    for ev in args.ev_list:
        if ev == 200:
            src_cache = "research/findings/raw/unified_per_regime/phase1"
        else:
            src_cache = f"research/findings/raw/unified_per_regime/phase1_{ev}ev"
        post_silence_cache = f"research/findings/raw/unified_per_regime/phase1_{ev}ev_post_silence"

        # Pre-silence direct binding (re-test on src cache for clean comparison)
        pre_result = test_one_checkpoint(
            args.seed, src_cache, f"PRE-silence {ev}ev seed {args.seed}"
        )

        # Silent interval + save
        run_silent_interval_and_save(
            args.seed, src_cache, post_silence_cache, args.n_silent_steps
        )

        # Post-silence direct binding
        post_result = test_one_checkpoint(
            args.seed, post_silence_cache, f"POST-silence {ev}ev seed {args.seed}"
        )

        # Forgetting metric
        pre_acc = pre_result["accuracy"]
        post_acc = post_result["accuracy"]
        forgetting_pct = 0.0 if pre_acc == 0 else 100.0 * (pre_acc - post_acc) / pre_acc

        results.append({
            "ev_per_word": ev,
            "src_cache": src_cache,
            "post_silence_cache": post_silence_cache,
            "n_silent_steps": args.n_silent_steps,
            "pre_n_correct": pre_result["n_correct"],
            "pre_accuracy": pre_acc,
            "post_n_correct": post_result["n_correct"],
            "post_accuracy": post_acc,
            "forgetting_pct": forgetting_pct,
        })

    print("\n=== MEMORY PERSISTENCE ACROSS TRAINING-EVENT REGIMES ===")
    print(f"  seed: {args.seed}; silent interval: {args.n_silent_steps} steps")
    print(f"  {'ev':>5} {'pre':>10} {'post':>10} {'forgetting%':>12}")
    for r in results:
        pre_pct = 100.0 * r["pre_accuracy"]
        post_pct = 100.0 * r["post_accuracy"]
        print(f"  {r['ev_per_word']:>5} {pre_pct:>9.1f}% {post_pct:>9.1f}% "
              f"{r['forgetting_pct']:>11.1f}%")

    out = args.out or "research/findings/raw/silent_interval_persistence_probe.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "seed": args.seed,
            "n_silent_steps": args.n_silent_steps,
            "ev_list": args.ev_list,
            "per_ev_results": results,
        }, f, indent=2)
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
