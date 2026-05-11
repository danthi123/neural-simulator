"""End-to-end BridgeMemory demo — Path 3 Phase 3.1.6.

Demonstrates the LLM-callable memory subsystem on a real bridge.
Wires store / recall / speak / stats together in a scripted flow
that an LLM could replicate via tool-use.

Usage:
    # NumPy backend (fast, CPU-only)
    SIM_BACKEND=numpy python -m research.runners.bridge_memory_demo \
        --seed 42 --lineage memory_demo --out demo_result.json

    # CuPy backend (production speed)
    python -m research.runners.bridge_memory_demo --seed 42

Output (sample):

  === BridgeMemory demo (lineage='memory_demo') ===
  [1/4] Building tier1 toy bridge under SIM_BACKEND=numpy...
        ✓ 208 neurons, 3438 synapses built in 0.9s
  [2/4] Bind: store("alice", "north") via embodied-Hebbian
        ✓ target_action=N, n_events=20, confidence=1.50, bound=True
  [3/4] Recall: mem.recall("alice")
        rank 1: north (action=N, conf=1.00, delta=317)
        rank 2: west  (action=W, conf=0.28, delta=88)
        ...
  [4/4] Speak: mem.speak("N")  # what word for motor_N?
        rank 1: alice (sim=0.85)
        rank 2: ...
  ✓ Lineage 'memory_demo' updated. Growth events: 4
  ✓ Demo complete.

This is the Path 3 distinguishing capability: a key-value memory backed
by a biology-grounded neural sim that continually learns + persists +
consolidates across sessions. Future Phase 3.2 work wires this to a
local LLM via tool-use.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def run_memory_demo(seed: int = 42,
                     lineage_name: str = "memory_demo",
                     lineage_root: Path | None = None,
                     out_path: Path | None = None,
                     verbose: bool = True) -> dict:
    """Run the end-to-end BridgeMemory demo.

    Args:
        seed: RNG seed for reproducibility
        lineage_name: where to save the demo's growth events
        lineage_root: optional override for lineage storage location
        out_path: optional path to write the result summary JSON
        verbose: print progress to stdout

    Returns:
        Dict with bind / recall / speak results + final stats.
    """
    log = print if verbose else (lambda *a, **kw: None)

    log("=" * 60)
    log(f"BridgeMemory demo (lineage='{lineage_name}', seed={seed})")
    log("=" * 60)

    # ── Step 1: build a tier1 toy bridge ─────────────────────────────
    log("[1/4] Building tier1 toy bridge...")
    t0 = time.time()
    from research.runners.bio_three_factor import run_three_factor
    bridge, _ = run_three_factor(
        seed=seed, n_events_per_direction=2, biological=True,
        n_lang_input=64, n_motor_per_action=16, n_motor_fs_per_action=4,
        enable_motor_fs=True, enable_nmda=False,
        apply_topographic_bias=True, embodied_hebbian=True,
        synonym_mode=False, verbose=False,
    )
    build_sec = time.time() - t0
    n_neurons = int(bridge.core_config.num_neurons)
    n_synapses = int(bridge.cp_connections.nnz)
    log(f"      OK {n_neurons} neurons, {n_synapses} synapses built in "
        f"{build_sec:.1f}s")

    # ── Step 2: open BridgeMemory + bind a fact ──────────────────────
    log("[2/4] BridgeMemory.store('alice', 'north')")
    from sim.bridge_memory import BridgeMemory
    from sim.lineage import BridgeLineage

    if lineage_root is None:
        lineage = BridgeLineage(lineage_name)
    else:
        lineage = BridgeLineage(lineage_name, root=lineage_root)
    # Save initial state so BridgeMemory can find it
    lineage.save(bridge, tier="tier1", arch={"mode": "tier1"})

    mem = BridgeMemory(
        lineage_name=lineage_name,
        mode="tier1",
        bridge=bridge,
        auto_save=True,
        verbose=False,
    )
    mem._lineage = lineage  # avoid re-loading

    bind_result = mem.store("alice", "north", n_events=20)
    log(f"      OK target_action={bind_result['target_action']} "
        f"n_events={bind_result['n_events_run']} "
        f"confidence={bind_result['confidence']:.2f} "
        f"bound={bind_result['bound_correctly']}")

    # ── Step 3: W->A recall ──────────────────────────────────────────
    log("[3/4] BridgeMemory.recall('alice')")
    recall_result = mem.recall("alice", top_k=4)
    for r in recall_result:
        log(f"      rank {r['rank']}: {r['value']:>8} "
            f"(action={r['action']}, conf={r['confidence']:.2f}, "
            f"delta={r['raw_delta']})")

    # ── Step 4: A->W speak ───────────────────────────────────────────
    log("[4/4] BridgeMemory.speak('N')  # what word goes with motor_N?")
    try:
        speak_result = mem.speak("N", top_k=4)
        for r in speak_result:
            log(f"      rank {r['rank']}: {r['word']:>10} "
                f"(sim={r['similarity']:.2f})")
    except Exception as e:
        log(f"      WARN speak failed: {e}")
        speak_result = []

    # ── Final stats ──────────────────────────────────────────────────
    stats = mem.stats()
    meta = lineage.read_metadata()
    n_growth = len(meta.growth_events)
    log("─" * 60)
    log(f"Lineage '{lineage_name}' growth events: {n_growth}")
    log(f"Bridge state: {stats['bridge_neurons']} neurons, "
        f"{stats['bridge_synapses']} synapses")
    log("Demo complete.")
    log("=" * 60)

    result = {
        "seed": seed,
        "lineage_name": lineage_name,
        "bridge_neurons": n_neurons,
        "bridge_synapses": n_synapses,
        "build_sec": build_sec,
        "bind_result": bind_result,
        "recall_top1": recall_result[0] if recall_result else None,
        "recall_all": recall_result,
        "speak_top1": speak_result[0] if speak_result else None,
        "speak_all": speak_result,
        "n_growth_events": n_growth,
        "final_stats": stats,
    }

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, default=str),
                              encoding="utf-8")
        log(f"\n[OUT] {out_path}")

    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lineage", type=str, default="memory_demo",
                    help="Lineage name to use (default 'memory_demo')")
    ap.add_argument("--out", type=str, default=None,
                    help="Optional JSON output path")
    args = ap.parse_args()

    run_memory_demo(
        seed=args.seed,
        lineage_name=args.lineage,
        out_path=Path(args.out) if args.out else None,
        verbose=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
