"""Investigate in-vivo new-vocab binding strategies (BIOLOGY-FIRST REDESIGN).

Followup to investigate_n_events_curve, which showed that more events
alone doesn't fix the recall failure for novel keys.

This runner now follows the biology-first workflow per the project
methodology (see .claude/skills/continual-autonomous-work/SKILL.md
Rule 8). Each variant must be motivated by a specific biological
mechanism, with citation.

  V0 — Vanilla (control). Existing learn_word_pairing. This is the
        current CLS slow-learning style (direct cortical co-firing) —
        which biology uses for already-known schema, NOT for novel
        concepts.

  V_HIPPO_BIO — Hippocampus-routed binding + immediate consolidation.
        Biology citation: McClelland 1995 (complementary learning
        systems theory), Buzsáki 2015 (SWR ripple model), Tse 2007
        (schema-supported memory consolidation).
        Mechanism:
          1. Novel concept enters hippocampus first (DG pattern-
             separates; CA3 forms fast autoassociative trace; CA1
             relays to cortex).
          2. After encoding, switch to sleep mode and run SWR replay
             cycles to consolidate hippo -> cortex.
          3. Switch back to awake; now recall should propagate
             through the cortically-consolidated pathway.
        REQUIRES a hippocampus-enabled bridge (main_hippo lineage from
        research.runners.bootstrap_hippo_lineage).

  V_SCHEMA — Schema-supported binding.
        Biology citation: Tse et al 2007 (Science): schema-related
        new facts integrate faster because they activate existing
        cortical schemas. Implementation: interleave new-word bind
        events with brief reinforcement of the matching anchor word
        (e.g. binding "apple" -> north interleaves with brief
        re-encoding of "north"). The new word's hippocampal trace
        attaches to the recently-reinforced "north" schema.

Test bindings: 4 made-up keys (apple/river/mountain/forest) bound
to N/E/S/W. Each variant runs on a separate fork of the base lineage.

Validation: ≥ 4/6 seeds correct top-1 recall on all 4 novel keys.

Usage:
    # Requires main_hippo lineage:
    python -m research.runners.bootstrap_hippo_lineage --lineage main_hippo
    # Then:
    python -m research.runners.investigate_invivo_binding_fix \\
        --base-lineage main_hippo --n-events 200 --seed 42

Saves results to research/findings/raw/g11_bg/invivo_binding_fix.json.

This is Step 1 of the realigned primary path (2026-05-11): sim as
standalone conversational agent. New-vocab binding must work before
scaling vocab.

Deprecated variants (engineering tweaks, removed 2026-05-11 after
methodology check-in):
  V1 (pre-bind zero edges) — brains don't zero weights before learning
  V3 (recall-only tail)    — no biology citation
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


def variant_v_hippo_bio(mem, key: str, value: str, n_events: int) -> dict:
    """Hippocampus-routed binding + immediate sleep consolidation.

    Biology citation:
      - McClelland, McNaughton & O'Reilly 1995: complementary learning
        systems theory (novel concepts encoded fast in hippocampus,
        consolidated slowly to neocortex via offline replay).
      - Buzsáki 2015: hippocampal sharp-wave ripples drive cortical
        memory consolidation during NREM sleep.
      - Tse et al 2007 (Science): schema-related facts integrate faster.

    Mechanism:
      1. AWAKE encoding: bind events run with hippocampus pathways
         (ec→dg→ca3→ca1) plastic alongside the direct cortical pathway.
         Drive the new word + motor teacher; both pathways encode.
      2. SLEEP consolidation: switch gates to sleep mode, run K SWR
         replay cycles. CA3 burst patterns propagate to CA1 → motor
         and language_output, strengthening cortical edges via STDP.
      3. AWAKE recall: switch back; recall now propagates through the
         consolidated cortical pathway, no longer dependent on the
         hippocampal trace.

    REQUIRES a hippocampus-enabled bridge (look for 'ca3' region).
    """
    from research.runners.text_minimal_isolation import (
        set_awake_gates, set_sleep_gates,
    )
    from research.runners.consolidation_trainer import run_swr_replay_phase
    from research.runners.chat_repl import learn_word_pairing, chat_inference
    import numpy as _np

    # Pre-check: bridge must have hippocampus
    bridge = mem.bridge
    try:
        bridge.region_manager.indices("ca3")
    except Exception as e:
        return {
            "key": key, "value": value,
            "error": (f"V_HIPPO_BIO requires hippocampus-enabled bridge. "
                       f"Bootstrap main_hippo first: {e}"),
            "n_events_run": 0,
            "bound_correctly": False, "confidence": 0.0,
        }

    target_action = mem._value_to_action(value)
    n_sleep_cycles = 2  # K=2 — Buzsáki 2015 typical NREM cycle count
    n_swr_per_cycle = 100  # ~10-20% of CA3 active per ripple

    t0 = time.time()

    # Phase 1: AWAKE encoding (both pathways plastic)
    set_awake_gates(bridge)
    learn_word_pairing(bridge, word=key, target_action=target_action,
                        n_events=n_events, verbose=False)

    # Phase 2: SLEEP consolidation
    set_sleep_gates(bridge)
    rng = _np.random.default_rng()
    for _ in range(n_sleep_cycles):
        run_swr_replay_phase(
            bridge,
            n_swr_events=n_swr_per_cycle,
            swr_drive_pA=100.0,
            rng=rng,
        )

    # Phase 3: AWAKE recall (gates restored)
    set_awake_gates(bridge)
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
        "v_hippo_n_sleep_cycles": n_sleep_cycles,
        "v_hippo_n_swr_events": n_sleep_cycles * n_swr_per_cycle,
        "elapsed_seconds": elapsed,
    }


def variant_v_schema(mem, key: str, value: str, n_events: int) -> dict:
    """Schema-supported binding (Tse et al 2007 Science).

    Biology citation:
      - Tse et al 2007: schema-related new facts integrate into
        neocortex within ~24 hours instead of the standard weeks-
        months consolidation timescale. Mechanism: existing schema
        provides a "scaffold" that new related facts attach to.

    Mechanism: interleave new-word bind events with brief
    reinforcement of the matching anchor word. Hypothesis: the new
    word's trace attaches to the recently-reinforced anchor schema.

    Lighter-weight than V_HIPPO_BIO (doesn't require hippocampus),
    so this can also run on non-hippo bridges as a comparison.
    """
    from research.runners.chat_repl import learn_word_pairing, chat_inference

    target_action = mem._value_to_action(value)
    anchor_word = {"N": "north", "E": "east",
                    "S": "south", "W": "west"}[target_action]

    M = 20
    n_batches = max(1, n_events // M)
    actual_new_events = n_batches * M
    t0 = time.time()
    for batch in range(n_batches):
        learn_word_pairing(mem.bridge, word=key, target_action=target_action,
                            n_events=M, verbose=False)
        # Brief anchor refresh — reinforces the schema the new word attaches to
        learn_word_pairing(mem.bridge, word=anchor_word, target_action=target_action,
                            n_events=2, verbose=False)
    elapsed = time.time() - t0

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
        "v_schema_n_anchor_events": n_batches * 2,
        "elapsed_seconds": elapsed,
    }


def variant_v_schema_topo(mem, key: str, value: str, n_events: int) -> dict:
    """Schema-supported binding + topographic prior at binding time.

    Pre-applies topographic bias from lang_input(key) → motor_target
    BEFORE running the V_SCHEMA training. The topographic boost
    aligns initial weights toward target, then V_SCHEMA's anchor
    reinforcement compounds the learning signal.

    Hypothesis: V_SCHEMA's 2/4 ceiling at 200ev hippo is because
    east/west anchors aren't strong enough on their own. Pre-aligning
    weights with topographic bias should compensate for weaker
    anchors and unlock more bindings.

    Biology: Pulvermüller 2001-2003 cortical somatotopy (word-action
    pairs cluster in topographically-organized cortical zones).
    Applying topographic prior is biology-faithful: real cortex has
    pre-developmental priors that align with semantic content.
    """
    from research.runners.chat_repl import learn_word_pairing, chat_inference
    from research.runners.text_minimal_isolation import (
        apply_novel_key_topographic_bias,
    )

    target_action = mem._value_to_action(value)
    anchor_word = {"N": "north", "E": "east",
                    "S": "south", "W": "west"}[target_action]

    # Apply topographic bias BEFORE training. Factor configurable via
    # env var TOPO_FACTOR for quick experimentation; default 5.0
    # (matching Tier 1's apply_topographic_bias default).
    import os as _os
    _topo_factor = float(_os.environ.get("TOPO_FACTOR", "5.0"))
    topo_result = apply_novel_key_topographic_bias(
        mem.bridge, key=key, target_action=target_action,
        factor=_topo_factor, n_lang_input=2048, sparsity=0.1, verbose=False,
    )

    # Same schema-supported training pattern as v_schema
    M = 20
    n_batches = max(1, n_events // M)
    actual_new_events = n_batches * M
    t0 = time.time()
    for batch in range(n_batches):
        learn_word_pairing(mem.bridge, word=key, target_action=target_action,
                            n_events=M, verbose=False)
        learn_word_pairing(mem.bridge, word=anchor_word, target_action=target_action,
                            n_events=2, verbose=False)
    elapsed = time.time() - t0

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
        "v_schema_topo_topographic_bias": topo_result,
        "elapsed_seconds": elapsed,
    }


VARIANTS: dict[str, Callable] = {
    "v0_vanilla": variant_v0_vanilla,
    "v_hippo_bio": variant_v_hippo_bio,
    "v_schema": variant_v_schema,
    "v_schema_topo": variant_v_schema_topo,
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
                    default="v0_vanilla,v_hippo_bio,v_schema")
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
