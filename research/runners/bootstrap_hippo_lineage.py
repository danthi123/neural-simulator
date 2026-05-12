"""Bootstrap a hippocampus-enabled lineage for BridgeMemory.consolidate().

Wraps consolidation_trainer.run_consolidation_training() and saves the
resulting bridge to a named lineage via BridgeLineage. Once saved, the
lineage can be opened via BridgeMemory(lineage_name=..., mode='synonym')
and the new Phase 3.2 consolidate() real-ops will actually run SWR
sleep cycles against it.

Usage:
    # Default: bootstrap `main_hippo` with Tier 1 config
    python -m research.runners.bootstrap_hippo_lineage \
        --lineage main_hippo --seed 42

    # Faster smoke (50 awake events per word, fewer SWR per cycle)
    python -m research.runners.bootstrap_hippo_lineage \
        --lineage hippo_smoke --seed 42 --n-awake 50 --n-swr 50

This is independent of the regular `main` lineage; BridgeMemory.consolidate
detects the hippocampus by looking for a 'ca3' region. Hippo-enabled
lineages have it; regular lineages don't.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def bootstrap_hippo_lineage(
    lineage_name: str = "main_hippo",
    seed: int = 42,
    n_awake_events_per_word: int = 200,
    n_sleep_swr_events: int = 200,
    consolidation_interval: int = 4,
    n_awake_per_word_map: dict | None = None,
    train_only: bool = True,
    verbose: bool = True,
) -> dict:
    """Train a hippocampus-enabled Tier 1 bridge and save to lineage.

    Args:
        lineage_name: target lineage (default 'main_hippo')
        seed: RNG seed (default 42)
        n_awake_events_per_word: encoding events per direction (4 dirs total)
        n_sleep_swr_events: SWR bursts per sleep cycle
        consolidation_interval: awake events between sleep cycles
        train_only: if True, skip eval; just train + save
        verbose: print progress

    Returns:
        dict with lineage_name, total training events, save path,
        elapsed seconds.
    """
    log = print if verbose else (lambda *a, **k: None)
    log("=" * 60)
    log(f"BOOTSTRAP HIPPO LINEAGE (name='{lineage_name}', seed={seed})")
    log("=" * 60)
    t0 = time.time()

    from research.runners.consolidation_trainer import (
        run_consolidation_training,
    )
    from sim.lineage import BridgeLineage

    bridge, training_stats = run_consolidation_training(
        seed=seed,
        n_awake_events_per_word=n_awake_events_per_word,
        n_sleep_swr_events=n_sleep_swr_events,
        consolidation_interval=consolidation_interval,
        n_awake_per_word_map=n_awake_per_word_map,
        verbose=verbose,
    )

    train_sec = time.time() - t0
    log(f"\n[TRAINING] complete ({train_sec:.0f}s)")

    # Save to lineage
    lineage = BridgeLineage(lineage_name)
    if lineage.exists():
        log(f"  (lineage '{lineage_name}' exists; will overwrite current)")
    arch = {
        "mode": "tier1_hippo",
        "n_lang_input": 2048,
        "n_motor_per_action": 500,
        "n_motor_fs_per_action": 60,
        "enable_hippocampus_consolidation": True,
        "n_awake_events_per_word": n_awake_events_per_word,
        "n_sleep_swr_events": n_sleep_swr_events,
        "consolidation_interval": consolidation_interval,
    }
    lineage.save(bridge, tier="tier1_hippo", arch=arch)
    log(f"  Saved to lineage '{lineage_name}'")

    # Record growth event (read-modify-write per BridgeLineage API)
    n_awake_total = n_awake_events_per_word * 4  # 4 primary directions
    n_sleep_cycles = max(1, n_awake_events_per_word // consolidation_interval)
    meta = lineage.read_metadata()
    meta.add_growth_event(
        kind="bootstrap_hippo",
        description=(
            f"Bootstrap hippocampus-enabled bridge: "
            f"{n_awake_total} awake events + {n_sleep_cycles} sleep cycles "
            f"({n_sleep_swr_events} SWR each), seed={seed}"
        ),
        seed=seed,
        n_awake_total=n_awake_total,
        n_sleep_cycles=n_sleep_cycles,
        train_seconds=train_sec,
    )
    lineage.write_metadata(meta)

    result = {
        "lineage_name": lineage_name,
        "seed": seed,
        "n_awake_total": n_awake_total,
        "n_sleep_cycles": n_sleep_cycles,
        "train_seconds": train_sec,
        "training_stats": training_stats,
    }
    log(f"\n[BOOTSTRAP] complete. Use:")
    log(f"  BridgeMemory(lineage_name='{lineage_name}', mode='synonym')")
    log(f"  -> mem.consolidate(n_sleep_cycles=3) will now run SWR replay")
    log("=" * 60)
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lineage", type=str, default="main_hippo")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-awake", type=int, default=200,
                    help="awake events per direction word (default 200)")
    ap.add_argument("--n-swr", type=int, default=200,
                    help="SWR bursts per sleep cycle (default 200)")
    ap.add_argument("--consolidation-interval", type=int, default=4)
    ap.add_argument("--n-awake-north", type=int, default=None,
                    help="Per-direction override for north (default uses --n-awake)")
    ap.add_argument("--n-awake-east", type=int, default=None,
                    help="Per-direction override for east")
    ap.add_argument("--n-awake-south", type=int, default=None,
                    help="Per-direction override for south")
    ap.add_argument("--n-awake-west", type=int, default=None,
                    help="Per-direction override for west")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    # Build per-direction map if any overrides given
    n_awake_per_word_map = None
    if any(x is not None for x in [args.n_awake_north, args.n_awake_east,
                                      args.n_awake_south, args.n_awake_west]):
        n_awake_per_word_map = {
            "north": args.n_awake_north if args.n_awake_north is not None else args.n_awake,
            "east":  args.n_awake_east  if args.n_awake_east  is not None else args.n_awake,
            "south": args.n_awake_south if args.n_awake_south is not None else args.n_awake,
            "west":  args.n_awake_west  if args.n_awake_west  is not None else args.n_awake,
        }

    result = bootstrap_hippo_lineage(
        lineage_name=args.lineage,
        seed=args.seed,
        n_awake_events_per_word=args.n_awake,
        n_sleep_swr_events=args.n_swr,
        consolidation_interval=args.consolidation_interval,
        n_awake_per_word_map=n_awake_per_word_map,
        train_only=True,
        verbose=True,
    )
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2, default=str),
                                     encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
