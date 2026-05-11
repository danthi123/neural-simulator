"""Investigate in-vivo new-vocab binding strategies.

Followup to investigate_n_events_curve, which showed that more events
alone doesn't fix the recall failure for novel keys (e.g. "apple" ->
north binds the edges but recall still returns south/west). The recall
pathway's random-init weights for unseen embeddings dominate.

This runner tests four binding strategies on a forked lineage:

  V0 — Vanilla (control). Existing learn_word_pairing.

  V1 — Pre-bind anchoring. Before the bind loop, zero out
        cp_connections rows for the new word's active language_input
        neurons. STDP then builds the lang_input -> motor pathway from
        zero rather than fighting random-init weights.

  V2 — Curriculum anchoring. Interleave the new-word bind with brief
        re-trains of known anchor words. Hypothesis: the recall path
        is primed by recently-reinforced known patterns; new
        embeddings need to ride the same neural infrastructure.

  V3 — Recall-only fine-tune tail. After the standard bind loop, run
        additional events with ONLY drive_in (no language_output, no
        motor teacher). Lets STDP firm up the lang_input -> motor edges
        based on actual recall dynamics, not artificially-teachered
        co-firing.

Each variant is tested on a separate fork of the base lineage with 4
made-up keys (apple, river, mountain, forest) bound to N, E, S, W.

Saves results to research/findings/raw/g11_bg/invivo_binding_fix.json.

Usage:
    python -m research.runners.investigate_invivo_binding_fix \\
        --base-lineage main --n-events 200 --seed 42

This is part of the primary path: sim as standalone conversational
agent. New-vocab binding must work before scaling vocab.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable, Optional


TEST_BINDINGS = [
    ("apple", "north", "N"),
    ("river", "east", "E"),
    ("mountain", "south", "S"),
    ("forest", "west", "W"),
]

ANCHOR_WORDS = [
    ("north", "N"),
    ("east", "E"),
    ("south", "S"),
    ("west", "W"),
]


# ──────────────────────────────────────────────────────────────────────
# Variant implementations
# ──────────────────────────────────────────────────────────────────────


def variant_v0_vanilla(mem, key: str, value: str, n_events: int) -> dict:
    """Control: existing learn_word_pairing flow via mem.store."""
    return mem.store(key, value, n_events=n_events)


def variant_v1_anchored(mem, key: str, value: str, n_events: int) -> dict:
    """Pre-bind: zero out cp_connections weights for edges originating
    from the new key's active language_input neurons. Then bind."""
    from sim.text_embeddings import vocab_to_drive_pattern
    from sim.backend import to_host as _to_host
    import numpy as _np

    bridge = mem.bridge
    rm = bridge.region_manager
    lang_indices = list(rm.indices("language_input"))
    n_lang = len(lang_indices)

    # Find active language_input neurons for this key
    drive = vocab_to_drive_pattern(key, n_neurons=n_lang,
                                       drive_max_pA=200.0, sparsity=0.1)
    active_local = _np.where(drive > 0)[0]
    active_global = [lang_indices[i] for i in active_local]

    # Zero out outgoing rows in cp_connections (CSR pre->post layout)
    cp_conn = bridge.cp_connections
    indptr_host = _to_host(cp_conn.indptr)
    data = cp_conn.data
    n_zeroed = 0
    for src in active_global:
        start = int(indptr_host[src])
        end = int(indptr_host[src + 1])
        if end > start:
            data[start:end] = 0.0
            n_zeroed += (end - start)

    # Now run the standard bind loop
    result = mem.store(key, value, n_events=n_events)
    result["v1_n_zeroed_edges"] = n_zeroed
    return result


def variant_v2_curriculum(mem, key: str, value: str, n_events: int) -> dict:
    """Curriculum: interleave new-word bind with brief anchor re-trains.

    For every M new-word events, do 1 anchor-word event (anchor word
    matches the same target direction). Total work: n_events for the
    new word + ~n_events/M for anchors.

    Hypothesis: the recall path is primed by recently-reinforced
    known patterns; new embeddings should ride that infrastructure.
    """
    from research.runners.chat_repl import learn_word_pairing

    target_action = mem._value_to_action(value)
    anchor_word = {"N": "north", "E": "east",
                    "S": "south", "W": "west"}[target_action]

    # Interleave: alternate batches of (M new + 1 anchor)
    M = 20
    n_batches = max(1, n_events // M)
    actual_new_events = n_batches * M
    t0 = time.time()
    for batch in range(n_batches):
        learn_word_pairing(mem.bridge, word=key, target_action=target_action,
                            n_events=M, verbose=False)
        # Brief anchor refresh
        learn_word_pairing(mem.bridge, word=anchor_word, target_action=target_action,
                            n_events=2, verbose=False)
    elapsed = time.time() - t0

    # Best-effort recall for confidence stat
    from research.runners.chat_repl import chat_inference
    try:
        check = chat_inference(mem.bridge, key)
        confidence = float(check.get("confidence_ratio", 0.0))
        bound_correctly = (check.get("predicted_action") == target_action)
    except Exception:
        confidence = 0.0
        bound_correctly = False

    return {
        "key": key, "value": value, "target_action": target_action,
        "confidence": confidence, "bound_correctly": bound_correctly,
        "n_events_run": actual_new_events,
        "v2_n_anchor_events": n_batches * 2,
        "elapsed_seconds": elapsed,
    }


def variant_v3_recall_tail(mem, key: str, value: str, n_events: int) -> dict:
    """Standard bind + recall-only tail.

    Phase A: standard bind for 0.8 * n_events.
    Phase B: 0.2 * n_events events with ONLY drive_in (no
             language_output, no motor teacher). STDP fires based on
             whatever motor pool the recall pathway naturally
             activates — strengthening the actual recall dynamics.
    """
    from sim.text_embeddings import vocab_to_drive_pattern
    from sim.backend import get_backend
    cp, _ = get_backend()
    from research.runners.chat_repl import learn_word_pairing, chat_inference

    n_phase_a = int(0.8 * n_events)
    n_phase_b = n_events - n_phase_a
    target_action = mem._value_to_action(value)

    t0 = time.time()
    # Phase A: standard bind
    learn_word_pairing(mem.bridge, word=key, target_action=target_action,
                        n_events=n_phase_a, verbose=False)

    # Phase B: recall-only events (drive_in alone, plasticity gate open)
    bridge = mem.bridge
    rm = bridge.region_manager
    lang_indices = list(rm.indices("language_input"))
    n_lang = len(lang_indices)
    drive = vocab_to_drive_pattern(key, n_neurons=n_lang,
                                       drive_max_pA=200.0, sparsity=0.1)
    drive_gpu = cp.asarray(drive, dtype=cp.float32)
    lang_arr = cp.asarray(lang_indices, dtype=cp.int64)

    # Open language_input_to_motor gate
    try:
        bridge.set_plasticity_gate("language_input_to_motor", 1.0)
    except Exception:
        pass

    for _ in range(n_phase_b):
        bridge.cp_external_input_current[:] = 0.0
        # Reset window
        for _ in range(50):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
        # Drive language_input ONLY
        bridge.cp_external_input_current[lang_arr] = drive_gpu
        for _ in range(100):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1

    # Re-freeze gate
    try:
        bridge.set_plasticity_gate("language_input_to_motor", 0.0)
    except Exception:
        pass

    elapsed = time.time() - t0

    # Best-effort recall for confidence
    try:
        check = chat_inference(bridge, key)
        confidence = float(check.get("confidence_ratio", 0.0))
        bound_correctly = (check.get("predicted_action") == target_action)
    except Exception:
        confidence = 0.0
        bound_correctly = False

    return {
        "key": key, "value": value, "target_action": target_action,
        "confidence": confidence, "bound_correctly": bound_correctly,
        "n_events_run": n_events,
        "v3_phase_a_events": n_phase_a, "v3_phase_b_events": n_phase_b,
        "elapsed_seconds": elapsed,
    }


VARIANTS: dict[str, Callable] = {
    "v0_vanilla": variant_v0_vanilla,
    "v1_anchored": variant_v1_anchored,
    "v2_curriculum": variant_v2_curriculum,
    "v3_recall_tail": variant_v3_recall_tail,
}


# ──────────────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────────────


def run_variants(base_lineage: str, n_events: int = 200,
                  seed: int = 42,
                  variants: Optional[list[str]] = None,
                  out_path: Optional[Path] = None,
                  verbose: bool = True) -> dict:
    """Run all variants on independent forks of base_lineage."""
    log = print if verbose else (lambda *a, **k: None)
    variants = variants or list(VARIANTS.keys())

    log("=" * 60)
    log(f"INVIVO BINDING FIX investigation")
    log(f"  base: {base_lineage}, n_events={n_events}, seed={seed}")
    log(f"  variants: {variants}")
    log("=" * 60)

    from sim.lineage import BridgeLineage
    from sim.bridge_memory import BridgeMemory

    base = BridgeLineage(base_lineage)
    if not base.exists():
        raise RuntimeError(f"base lineage '{base_lineage}' not found")

    results = []
    for variant_name in variants:
        log(f"\n[VARIANT] {variant_name}")
        fork_name = f"invivo_fix_{variant_name}"
        fork = BridgeLineage(fork_name)
        if not fork.exists():
            log(f"  forking {base_lineage} -> {fork_name}")
            base.fork(fork_name)
        mem = BridgeMemory(lineage_name=fork_name, mode="synonym",
                            auto_save=False, verbose=False)
        mem._ensure_loaded()
        variant_fn = VARIANTS[variant_name]

        details = []
        t_start = time.time()
        for key, value, expected_action in TEST_BINDINGS:
            log(f"    bind('{key}', '{value}') with {variant_name}")
            bind_result = variant_fn(mem, key, value, n_events)
            recall = mem.recall(key, top_k=4)
            top = recall[0] if recall else {}
            got_action = top.get("action", "")
            correct = (got_action == expected_action)
            log(f"      -> top={top.get('value','')} ({got_action}); "
                f"expected={expected_action}; correct={correct}")
            details.append({
                "key": key, "expected_value": value,
                "expected_action": expected_action,
                "got_action": got_action,
                "got_value": top.get("value", ""),
                "correct": correct,
                "confidence": top.get("confidence", 0.0),
                "raw_delta": top.get("raw_delta", 0),
                "bind_result": bind_result,
            })
        elapsed = time.time() - t_start
        n_correct = sum(1 for d in details if d["correct"])
        log(f"  -> {n_correct}/{len(details)} correct, {elapsed:.0f}s total")
        results.append({
            "variant": variant_name,
            "n_correct": n_correct,
            "n_total": len(details),
            "accuracy": n_correct / len(details) if details else 0.0,
            "total_seconds": elapsed,
            "details": details,
        })

    summary = {
        "base_lineage": base_lineage,
        "n_events": n_events,
        "seed": seed,
        "test_bindings": TEST_BINDINGS,
        "variants_tested": variants,
        "results": results,
    }
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2, default=str),
                              encoding="utf-8")
        log(f"\n[OUT] {out_path}")

    log("=" * 60)
    log("Summary:")
    for r in results:
        log(f"  {r['variant']:>20}  "
            f"acc={r['accuracy']*100:>5.1f}% ({r['n_correct']}/{r['n_total']})  "
            f"({r['total_seconds']:.0f}s)")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-lineage", type=str, default="main")
    ap.add_argument("--n-events", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--variants", type=str,
                    default="v0_vanilla,v1_anchored,v2_curriculum,v3_recall_tail")
    ap.add_argument("--out", type=str,
                    default="research/findings/raw/g11_bg/invivo_binding_fix.json")
    args = ap.parse_args()
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    run_variants(
        base_lineage=args.base_lineage,
        n_events=args.n_events,
        seed=args.seed,
        variants=variants,
        out_path=Path(args.out),
        verbose=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
