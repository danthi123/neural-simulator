"""Test cumulative in-vivo novel-key binding.

The investigate_invivo_binding_fix runner forks main_hippo and tests
each binding INDEPENDENTLY. This runner tests CUMULATIVE binding:
bind A on main_hippo, then bind B on the result, then C, etc.
After each binding, recall ALL previous keys to check for
interference / catastrophic forgetting.

Hypothesis: V_SCHEMA's anchor reinforcement should preserve prior
bindings because each new key's anchor refresh strengthens the
existing schema. If apple→N then mountain→S both work cumulatively,
the sim has true in-vivo vocabulary growth.

Test sequence (uses target pools that worked at 2/4 baseline):
  1. apple → N
  2. mountain → S
  3. (optional) test recall of both after each
  4. Try adding cat → E (will probably fail per east anchor weakness)
  5. Try adding dog → W (same)

Usage:
    PYTHONIOENCODING=utf-8 python -m research.runners.test_cumulative_invivo_binding \
        --base-lineage main_hippo --seed 42 \
        --bindings apple:north,mountain:south,cat:east,dog:west \
        --out research/findings/raw/g11_bg/invivo_binding/cumulative.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional


def run_cumulative_binding(
    base_lineage: str,
    bindings: list[tuple[str, str]],
    seed: int = 42,
    n_events: int = 200,
    consolidate_between: bool = False,
    n_consolidation_cycles: int = 3,
    verbose: bool = True,
) -> dict:
    """Bind keys one after another on the same bridge. After each
    binding, recall ALL previously-bound keys to check for
    interference."""
    log = print if verbose else (lambda *a, **k: None)
    log("=" * 60)
    log(f"CUMULATIVE IN-VIVO BINDING")
    log(f"  base: {base_lineage}, seed={seed}, n_events={n_events}/binding")
    log(f"  bindings: {bindings}")
    log("=" * 60)

    from sim.lineage import BridgeLineage
    from sim.bridge_memory import BridgeMemory
    from research.runners.chat_repl import learn_word_pairing, chat_inference

    fork_name = f"cumulative_{seed}"
    base = BridgeLineage(base_lineage)
    if not base.exists():
        raise RuntimeError(f"base lineage '{base_lineage}' not found")
    fork = BridgeLineage(fork_name)
    if fork.exists():
        log(f"  removing prior fork {fork_name}")
        import shutil
        shutil.rmtree(fork.root)
    log(f"  forking {base_lineage} -> {fork_name}")
    base.fork(fork_name)

    mem = BridgeMemory(lineage_name=fork_name, mode="synonym",
                        auto_save=False, verbose=False)
    mem._ensure_loaded()

    binding_steps = []
    for step_idx, (key, value) in enumerate(bindings):
        target_action = mem._value_to_action(value)
        anchor_word = {"N": "north", "E": "east",
                        "S": "south", "W": "west"}[target_action]
        log(f"\n[STEP {step_idx+1}/{len(bindings)}] "
            f"bind '{key}' -> {target_action} (anchor: {anchor_word})")

        # V_SCHEMA training (same as variant_v_schema)
        M = 20
        n_batches = max(1, n_events // M)
        t0 = time.time()
        for _ in range(n_batches):
            learn_word_pairing(mem.bridge, word=key,
                                target_action=target_action,
                                n_events=M, verbose=False)
            learn_word_pairing(mem.bridge, word=anchor_word,
                                target_action=target_action,
                                n_events=2, verbose=False)
        train_sec = time.time() - t0
        log(f"  training done ({train_sec:.0f}s)")

        # Optional: consolidate via SWR sleep replay BEFORE testing
        # recall. Hypothesis: pushes hippocampal trace -> cortex so
        # the binding persists across subsequent bindings.
        consolidation_sec = 0.0
        if consolidate_between:
            log(f"  [CONSOLIDATE] running {n_consolidation_cycles} sleep cycles...")
            t1 = time.time()
            try:
                consol_result = mem.consolidate(
                    n_sleep_cycles=n_consolidation_cycles,
                )
                consolidation_sec = time.time() - t1
                log(f"  consolidation done ({consolidation_sec:.0f}s, "
                    f"hippocampus_enabled={consol_result.get('hippocampus_enabled')})")
            except Exception as e:
                log(f"  [WARN] consolidation failed: {e}")
                consolidation_sec = -1.0

        # Recall this key + ALL previous keys
        recalls = {}
        for prev_idx in range(step_idx + 1):
            prev_key, prev_value = bindings[prev_idx]
            prev_target = mem._value_to_action(prev_value)
            try:
                check = chat_inference(mem.bridge, prev_key)
                predicted = check.get("predicted_action", "?")
                confidence = float(check.get("confidence_ratio", 0.0))
                correct = (predicted == prev_target)
            except Exception as e:
                predicted = f"ERROR:{e}"
                confidence = 0.0
                correct = False
            recalls[prev_key] = {
                "expected": prev_target,
                "got": predicted,
                "confidence": confidence,
                "correct": correct,
            }
            marker = "OK" if correct else "X "
            log(f"    {marker} {prev_key} (target={prev_target}) "
                f"-> {predicted} (conf={confidence:.2f})")
        n_correct = sum(1 for r in recalls.values() if r["correct"])
        log(f"  After step {step_idx+1}: {n_correct}/{step_idx+1} bindings correct")
        binding_steps.append({
            "step": step_idx + 1,
            "new_key": key,
            "new_target": target_action,
            "train_seconds": train_sec,
            "consolidation_seconds": consolidation_sec,
            "recalls": recalls,
            "n_correct_so_far": n_correct,
        })

    # Final summary: how many of the original bindings survived?
    final_step = binding_steps[-1]
    final_correct = sum(1 for r in final_step["recalls"].values() if r["correct"])
    log("\n" + "=" * 60)
    log(f"FINAL: {final_correct}/{len(bindings)} bindings correct after all training")
    log("=" * 60)

    return {
        "base_lineage": base_lineage,
        "seed": seed,
        "n_events_per_binding": n_events,
        "bindings": [{"key": k, "value": v} for k, v in bindings],
        "binding_steps": binding_steps,
        "final_correct": final_correct,
        "total_bindings": len(bindings),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-lineage", type=str, default="main_hippo")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-events", type=int, default=200)
    ap.add_argument("--bindings", type=str,
                    default="apple:north,mountain:south,cat:east,dog:west",
                    help="comma-separated key:value pairs")
    ap.add_argument("--consolidate-between", action="store_true",
                    help="Run SWR sleep replay between each binding "
                         "(Phase 1.3 mechanism). Hypothesis: prevents "
                         "binding displacement by consolidating to cortex.")
    ap.add_argument("--n-consolidation-cycles", type=int, default=3)
    ap.add_argument("--out", type=str,
                    default="research/findings/raw/g11_bg/invivo_binding/cumulative.json")
    args = ap.parse_args()

    bindings = []
    for item in args.bindings.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"binding '{item}' missing ':' (use key:value)")
        k, v = item.split(":", 1)
        bindings.append((k.strip(), v.strip()))

    result = run_cumulative_binding(
        base_lineage=args.base_lineage,
        bindings=bindings,
        seed=args.seed,
        n_events=args.n_events,
        consolidate_between=args.consolidate_between,
        n_consolidation_cycles=args.n_consolidation_cycles,
        verbose=True,
    )

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps(result, indent=2, default=str),
            encoding="utf-8",
        )
        print(f"\n[OUT] {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
