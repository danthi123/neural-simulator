"""Investigate: how many co-firing events are needed for reliable
new-vocab binding against a synonym-scale bridge?

Background (2026-05-11): live smoke against the bootstrapped `main`
lineage showed that BridgeMemory.store("favorite", "north", n_events=200)
binds the edges (motor_N teacher fires, STDP captures co-activity)
but RECALL still returns motor_S or motor_E because the random-init
weights for "favorite"'s neuron pattern dominate at 12K neurons.

This runner sweeps n_events ∈ {100, 200, 400, 800, 1200} on a forked
lineage and reports recall accuracy across multiple new-vocab keys.

Usage:
    python -m research.runners.investigate_n_events_curve \
        --base-lineage main --seed 42

Outputs JSON to research/findings/raw/g11_bg/n_events_curve.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


# Test keys: 4 made-up words, each bound to a different direction.
# Symmetry check: if bind is real, all 4 should recall correctly.
# If random-bias dominates, we'll see clustering on whatever direction
# the bridge's init biases toward.
TEST_BINDINGS = [
    ("apple", "north"),
    ("river", "east"),
    ("mountain", "south"),
    ("forest", "west"),
]


def run_curve(base_lineage: str, seed: int, out_path: Path,
                event_levels: list[int] | None = None,
                verbose: bool = True) -> dict:
    """Sweep n_events and measure recall accuracy on 4 made-up keys.

    For each n_events level:
      1. Fork the base lineage so each level starts from the same state
      2. Bind 4 new-vocab keys at that n_events level
      3. Recall each, check if top-1 motor matches the intended action

    Returns:
        {
          "base_lineage": str,
          "event_levels": [n_events, ...],
          "results": [
            {"n_events": int, "n_correct": int, "n_total": 4,
             "bind_seconds": float, "details": [{key, expected, got, ...}]},
            ...
          ]
        }
    """
    log = print if verbose else (lambda *a, **k: None)
    event_levels = event_levels or [100, 200, 400, 800, 1200]

    log("=" * 60)
    log(f"n_events curve sweep (base={base_lineage}, seed={seed})")
    log(f"Levels: {event_levels}")
    log("=" * 60)

    from sim.lineage import BridgeLineage
    from sim.bridge_memory import BridgeMemory

    # Verify base exists
    base = BridgeLineage(base_lineage)
    if not base.exists():
        raise RuntimeError(f"base lineage '{base_lineage}' does not exist")

    results = []
    for n_events in event_levels:
        log(f"\n[LEVEL {n_events}]")
        # Fork into a sweep-specific lineage so each level is independent
        fork_name = f"n_events_sweep_{n_events}"
        fork = BridgeLineage(fork_name)
        if fork.exists():
            log(f"  (fork {fork_name} exists; reusing for sweep)")
        else:
            log(f"  forking {base_lineage} -> {fork_name}")
            base.fork(fork_name)
        # Open BridgeMemory in synonym mode
        mem = BridgeMemory(
            lineage_name=fork_name, mode="synonym",
            auto_save=False, verbose=False,
        )
        mem._ensure_loaded()

        t_start = time.time()
        details = []
        for key, expected_value in TEST_BINDINGS:
            log(f"    bind('{key}', '{expected_value}', n_events={n_events})")
            bind_result = mem.store(key, expected_value, n_events=n_events)
            recall_result = mem.recall(key, top_k=4)
            top = recall_result[0] if recall_result else {}
            got_value = top.get("value", "")
            got_action = top.get("action", "")
            expected_action = {"north": "N", "east": "E",
                                "south": "S", "west": "W"}[expected_value]
            correct = (got_action == expected_action)
            log(f"      -> top={got_value} ({got_action}); "
                f"expected={expected_action}; "
                f"correct={correct}; "
                f"bind_time={bind_result['elapsed_seconds']:.1f}s")
            details.append({
                "key": key, "expected_value": expected_value,
                "expected_action": expected_action,
                "got_value": got_value, "got_action": got_action,
                "correct": correct,
                "confidence": top.get("confidence", 0.0),
                "raw_delta": top.get("raw_delta", 0),
                "bind_seconds": bind_result["elapsed_seconds"],
            })
        elapsed = time.time() - t_start
        n_correct = sum(1 for d in details if d["correct"])
        log(f"  -> {n_correct}/{len(details)} correct, {elapsed:.0f}s total")
        results.append({
            "n_events": n_events,
            "n_correct": n_correct,
            "n_total": len(details),
            "accuracy": n_correct / len(details) if details else 0.0,
            "total_seconds": elapsed,
            "details": details,
        })

    summary = {
        "base_lineage": base_lineage,
        "seed": seed,
        "test_bindings": TEST_BINDINGS,
        "event_levels": event_levels,
        "results": results,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, default=str),
                          encoding="utf-8")
    log(f"\n[OUT] {out_path}")
    log("=" * 60)
    log("Summary:")
    for r in results:
        log(f"  n_events={r['n_events']:>4}  "
            f"acc={r['accuracy']*100:>5.1f}% ({r['n_correct']}/{r['n_total']})  "
            f"({r['total_seconds']:.0f}s)")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-lineage", type=str, default="main",
                    help="Base lineage to fork from (default 'main')")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str,
                    default="research/findings/raw/g11_bg/n_events_curve.json")
    ap.add_argument("--levels", type=str, default="100,200,400,800",
                    help="Comma-separated n_events levels (default 100,200,400,800)")
    args = ap.parse_args()
    levels = [int(x.strip()) for x in args.levels.split(",") if x.strip()]
    run_curve(
        base_lineage=args.base_lineage,
        seed=args.seed,
        out_path=Path(args.out),
        event_levels=levels,
        verbose=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
