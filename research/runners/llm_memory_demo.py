"""End-to-end LLM-memory demo — Path 3 Phase 3.2 scaffold.

Demonstrates the LLM tool-use loop driving a real BridgeMemory, which
in turn drives a biology-grounded SimulationBridge.

Stack:
    MockLLM → LLMMemoryOrchestrator → BridgeMemory → SimulationBridge
                                          ↓
                                    BridgeLineage (persisted state)

The MockLLM is hand-coded (no external LLM dependency). It recognizes
a handful of natural-language patterns and emits the right tool calls
(memory_store / memory_recall / memory_speak). Phase 3.3 will swap in a
real local LLM (Phi-3-mini / Llama 3.2 / Qwen2.5) via tool-use; the
orchestrator interface stays unchanged.

Usage:
    # NumPy backend (CPU-only, fast for the toy)
    SIM_BACKEND=numpy python -m research.runners.llm_memory_demo \
        --seed 42 --lineage llm_demo --out demo_result.json

    # CuPy backend (production GPU)
    python -m research.runners.llm_memory_demo --seed 42

Output (sample):

    === LLM-memory demo (lineage='llm_demo') ===
    [1/3] Building tier1 toy bridge...
          OK 208 neurons, 3438 synapses
    [2/3] Wrapping in BridgeMemory + LLMMemoryOrchestrator (MockLLM)
    [3/3] Running scripted chat:

    USER:     Remember that my favorite is north.
    ASSISTANT: Got it. I'll remember: favorite = north.

    USER:     What's my favorite?
    ASSISTANT: Your answer is north.

    USER:     What word goes with east?
    ASSISTANT: No clear word found for that direction.

    USER:     Remember that my pet is east.
    ASSISTANT: Got it. I'll remember: pet = east.

    USER:     What word goes with east?
    ASSISTANT: The word for that direction is 'pet'.

    Conversation length: 15 messages
    Tool calls: 5 (store=2, recall=1, speak=2)
    Demo complete.

Note: at the toy scale (208 neurons, 2 train events / direction) the
science accuracy is low (recall often returns the wrong direction).
This is a PROTOCOL demo — it proves the LLM tool-use loop, BridgeMemory
dispatch, and persistence pipeline all work end-to-end. For real science
accuracy, use BridgeMemory with mode="synonym" or larger n_events.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


SCRIPT = [
    "Remember that my favorite is north.",
    "What's my favorite?",
    "What word goes with east?",
    "Remember that my pet is east.",
    "What word goes with east?",
]


def run_llm_demo(seed: int = 42,
                  lineage_name: str = "llm_demo",
                  lineage_root: Path | None = None,
                  out_path: Path | None = None,
                  script: list[str] | None = None,
                  verbose: bool = True) -> dict:
    """Run the end-to-end LLM-memory demo.

    Args:
        seed: RNG seed for reproducibility
        lineage_name: where to save the demo's growth events
        lineage_root: optional override for lineage storage location
        out_path: optional path to write the result summary JSON
        script: optional list of user messages (default: 5-turn demo)
        verbose: print progress to stdout

    Returns:
        Dict with bridge stats + per-turn results + conversation transcript.
    """
    log = print if verbose else (lambda *a, **kw: None)
    script = script or SCRIPT

    log("=" * 60)
    log(f"LLM-memory demo (lineage='{lineage_name}', seed={seed})")
    log("=" * 60)

    # ── Step 1: build a tier1 toy bridge ─────────────────────────────
    log("[1/3] Building tier1 toy bridge...")
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
    log(f"      OK {n_neurons} neurons, {n_synapses} synapses "
        f"({build_sec:.1f}s build)")

    # ── Step 2: wrap in BridgeMemory + orchestrator ──────────────────
    log("[2/3] Wrapping in BridgeMemory + LLMMemoryOrchestrator (MockLLM)")
    from sim.bridge_memory import BridgeMemory
    from sim.lineage import BridgeLineage
    from sim.llm_memory_orchestrator import LLMMemoryOrchestrator, MockLLM

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

    # Speed up store() — each store does n_events co-firing.
    # Tier-1 default n_events=50 is fine for the toy demo.
    llm = MockLLM()
    orch = LLMMemoryOrchestrator(memory=mem, llm_callable=llm,
                                  max_tool_iterations=5)

    # ── Step 3: run scripted chat ────────────────────────────────────
    log("[3/3] Running scripted chat:")
    log("")
    turns: list[dict] = []
    tool_counts = {"memory_store": 0, "memory_recall": 0, "memory_speak": 0}

    for user_input in script:
        log(f"USER:      {user_input}")
        turn_start_len = len(orch.conversation)
        t0 = time.time()
        response = orch.chat(user_input)
        elapsed = time.time() - t0

        # Count tool calls in the conversation slice for this turn
        slice_turns = orch.conversation[turn_start_len:]
        turn_tools = [t for t in slice_turns if t["role"] == "tool"]
        for tt in turn_tools:
            name = tt.get("name", "")
            if name in tool_counts:
                tool_counts[name] += 1

        log(f"ASSISTANT: {response}")
        if turn_tools:
            tools_str = ", ".join(t.get("name", "?") for t in turn_tools)
            log(f"  (tool: {tools_str}, {elapsed:.2f}s)")
        log("")

        turns.append({
            "user": user_input,
            "assistant": response,
            "tools_called": [t.get("name", "?") for t in turn_tools],
            "elapsed_seconds": elapsed,
        })

    # ── Final stats ──────────────────────────────────────────────────
    n_msgs = len(orch.conversation)
    total_tools = sum(tool_counts.values())
    meta = lineage.read_metadata()
    n_growth = len(meta.growth_events)

    log("-" * 60)
    log(f"Conversation length: {n_msgs} messages")
    log(f"Tool calls: {total_tools} (store={tool_counts['memory_store']}, "
        f"recall={tool_counts['memory_recall']}, "
        f"speak={tool_counts['memory_speak']})")
    log(f"Lineage growth events: {n_growth}")
    log(f"Bridge state: {n_neurons} neurons, {n_synapses} synapses")
    log("Demo complete.")
    log("=" * 60)

    result = {
        "seed": seed,
        "lineage_name": lineage_name,
        "bridge_neurons": n_neurons,
        "bridge_synapses": n_synapses,
        "build_seconds": build_sec,
        "n_turns": len(script),
        "n_messages": n_msgs,
        "tool_call_counts": tool_counts,
        "n_growth_events": n_growth,
        "turns": turns,
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
    ap.add_argument("--lineage", type=str, default="llm_demo",
                    help="Lineage name to use (default 'llm_demo')")
    ap.add_argument("--out", type=str, default=None,
                    help="Optional JSON output path")
    args = ap.parse_args()

    run_llm_demo(
        seed=args.seed,
        lineage_name=args.lineage,
        out_path=Path(args.out) if args.out else None,
        verbose=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
