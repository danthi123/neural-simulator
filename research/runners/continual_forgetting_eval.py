"""Phase 1.4 — Catastrophic forgetting eval suite.

THE foundational test for Path F's premise: does biology-grounded
continual learning preserve old knowledge when learning new?

Test design (sequential vocabulary expansion):
1. Phase A: train 4 primary direction words (north/east/south/west)
   to motor pools via embodied Hebbian (Tier 1 paradigm).
2. Eval after Phase A: measure W->A on ALL 4 primary words.
3. Phase B: train 4 NEW synonym words (up/right/down/left), no
   exposure to primaries. STDP at lang_input -> motor and motor ->
   lang_output pathways gets new patterns.
4. Eval after Phase B: measure W->A on:
   - Primaries (north/east/south/west) — RETENTION TEST
   - Synonyms (up/right/down/left) — NEW LEARNING TEST
5. (Optional) Phase C: sleep consolidation simulation.
6. Re-eval — does retention recover to baseline?

Pass criteria:
- After Phase B, primary retention ≥ 50% of original (catastrophic
  forgetting NOT happening)
- After Phase B, synonym new learning ≥ 30% accuracy (basic learning
  works)
- (If C runs) After sleep, primary retention ≥ 80% of original

Saves retention curves to JSON; produces matplotlib plot if available.
"""
from __future__ import annotations

import argparse
import json
import time
import sys
from pathlib import Path

import numpy as np

# Direction words -- Phase A vocab
PRIMARY_WORDS = ["north", "east", "south", "west"]
PRIMARY_TO_ACTION = {"north": "N", "east": "E", "south": "S", "west": "W"}

# Synonym words -- Phase B vocab (same actions, new word forms)
SYNONYM_WORDS = ["up", "right", "down", "left"]
SYNONYM_TO_ACTION = {"up": "N", "right": "E", "down": "S", "left": "W"}

ALL_WORDS = PRIMARY_WORDS + SYNONYM_WORDS


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--phase-a-events", type=int, default=200,
                    help="Events per word in Phase A (primary training)")
    ap.add_argument("--phase-b-events", type=int, default=200,
                    help="Events per word in Phase B (synonym training)")
    ap.add_argument("--enable-sleep-consolidation", action="store_true",
                    help="Phase C: SWR-style replay between phases")
    ap.add_argument("--n-eval-per-word", type=int, default=25)
    # Tier 1 standard arch (validated 5/6 + 6/6 in
    # 2026-05-06-Tier1-BREAKTHROUGH-bidirectional-binding.md;
    # mean W->A 33-45% across 6 seeds).
    # NOTE: Phase 1.4 v1 (smoke 2026-05-06 20:31 EDT) used scale-up
    # arch (4096/1000/120) and Phase A baseline collapsed to 14% --
    # below chance. Scale-up was for Tier 2.1 8-word synonyms; for
    # 4-word vocab in v2 we use standard arch matching Tier 1.
    ap.add_argument("--n-lang-input", type=int, default=2048,
                    help="Standard arch matching Tier 1 v2 (validated)")
    ap.add_argument("--n-motor-per-action", type=int, default=500)
    ap.add_argument("--n-motor-fs-per-action", type=int, default=60)
    ap.add_argument("--out-stats", type=str, default=None)
    args = ap.parse_args()

    # Reuse bio_three_factor's training infrastructure
    from research.runners.bio_three_factor import run_three_factor
    from research.runners.text_eval import (
        evaluate_word_to_action, EXTENDED_WORD_TO_ACTION,
    )

    print("=" * 60)
    print(f"PHASE 1.4 CATASTROPHIC FORGETTING EVAL (seed={args.seed})")
    print("=" * 60, flush=True)

    # Track retention curves
    retention = {
        "phase_a_events": args.phase_a_events,
        "phase_b_events": args.phase_b_events,
        "seed": args.seed,
        "checkpoints": [],  # list of {checkpoint_name, eval_results}
    }

    # ─── PHASE A: Primary training ───
    print(f"\n--- PHASE A: train {len(PRIMARY_WORDS)} primary words "
          f"({args.phase_a_events} events/word) ---", flush=True)
    t0 = time.time()
    bridge, _ = run_three_factor(
        seed=args.seed,
        n_events_per_direction=args.phase_a_events,
        n_lang_input=args.n_lang_input,
        n_motor_per_action=args.n_motor_per_action,
        n_motor_fs_per_action=args.n_motor_fs_per_action,
        biological=True,
        enable_motor_fs=True,
        apply_topographic_bias=True,
        embodied_hebbian=True,
        synonym_mode=False,  # Phase A uses primary words only
        verbose=False,
    )
    print(f"  Phase A training complete ({time.time()-t0:.0f}s)", flush=True)

    # ─── EVAL after Phase A ───
    print("\n--- EVAL after Phase A: primary words ---", flush=True)
    wa_a = evaluate_word_to_action(
        bridge, n_trials_per_word=args.n_eval_per_word,
        stim_steps_per_trial=100, n_reset_steps=50, token_sparsity=0.1,
        verbose=False,
    )
    print(f"  W->A on primary: {wa_a['accuracy']:.1%}", flush=True)
    print(f"  Confusion: {wa_a['confusion_matrix']}", flush=True)
    retention["checkpoints"].append({
        "name": "after_phase_a",
        "primary_wa": wa_a,
    })

    # ─── PHASE B: Synonym training ───
    # Continue training on same bridge with synonym vocab. The
    # bridge's STDP traces persist; new training adds onto existing
    # weights.
    print(f"\n--- PHASE B: train {len(SYNONYM_WORDS)} synonym words "
          f"({args.phase_b_events} events/word) ---", flush=True)
    print("  WARNING: this is the catastrophic forgetting test. "
          "We train ONLY synonyms, no primaries, and check if "
          "primary retention degrades.", flush=True)

    # Custom Phase B training — present synonyms only
    # We need to do this manually because run_three_factor doesn't
    # support "continue training on different vocab on existing bridge."
    import cupy as cp
    rng = np.random.default_rng(args.seed * 7 + 1)
    rm = bridge.region_manager
    lang_input_idx = list(rm.indices("language_input"))
    motor_idx = {a: list(rm.indices(f"motor_{a}")) for a in ["N","E","S","W"]}
    lang_output_idx = list(rm.indices("language_output"))

    # Build synonym event buffer
    synonym_buffer = []
    for word in SYNONYM_WORDS:
        action = SYNONYM_TO_ACTION[word]
        for _ in range(args.phase_b_events):
            synonym_buffer.append({"token": word, "action": action})
    rng.shuffle(synonym_buffer)

    # Open plasticity gates
    try:
        bridge.set_plasticity_gate("language_input_to_motor", 1.0)
        bridge.set_plasticity_gate("motor_to_language_output", 1.0)
    except Exception:
        pass

    from sim.text_embeddings import vocab_to_drive_pattern

    t0 = time.time()
    embodied_motor_teacher_pA = 300.0
    stim_steps_per_event = 50
    reset_steps = 50
    print(f"  Training {len(synonym_buffer)} synonym events...", flush=True)
    for ev_idx, event in enumerate(synonym_buffer):
        token = event["token"]
        target_action = event["action"]
        # Inter-trial reset
        bridge.cp_external_input_current[:] = 0.0
        for _ in range(reset_steps):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
        # Drive language_input + language_output (teachers)
        n_lang_in = len(lang_input_idx)
        drive = vocab_to_drive_pattern(token, n_neurons=n_lang_in,
                                        drive_max_pA=200.0, sparsity=0.1)
        bridge.cp_external_input_current[
            cp.asarray(lang_input_idx, dtype=cp.int64)
        ] = cp.asarray(drive, dtype=cp.float32)
        bridge.cp_external_input_current[
            cp.asarray(lang_output_idx, dtype=cp.int64)
        ] = cp.asarray(drive, dtype=cp.float32)
        # Motor teacher
        target_motor_arr = cp.asarray(motor_idx[target_action], dtype=cp.int64)
        bridge.cp_external_input_current[target_motor_arr] += float(
            embodied_motor_teacher_pA
        )
        # Forward propagate
        for _ in range(stim_steps_per_event):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
        if (ev_idx + 1) % 100 == 0:
            print(f"    [{ev_idx+1}/{len(synonym_buffer)} events, "
                  f"{time.time()-t0:.0f}s]", flush=True)

    # Freeze for eval
    try:
        bridge.set_plasticity_gate("language_input_to_motor", 0.0)
        bridge.set_plasticity_gate("motor_to_language_output", 0.0)
    except Exception:
        pass

    print(f"  Phase B training complete ({time.time()-t0:.0f}s)", flush=True)

    # ─── EVAL after Phase B ───
    print("\n--- EVAL after Phase B: primary words (RETENTION) ---", flush=True)
    wa_primary_after_b = evaluate_word_to_action(
        bridge, n_trials_per_word=args.n_eval_per_word,
        stim_steps_per_trial=100, n_reset_steps=50, token_sparsity=0.1,
        verbose=False,
    )
    print(f"  W->A on primary AFTER B: {wa_primary_after_b['accuracy']:.1%}",
          flush=True)
    print(f"    (was {wa_a['accuracy']:.1%} after Phase A)", flush=True)

    print("\n--- EVAL after Phase B: synonym words (NEW LEARNING) ---",
          flush=True)
    # Need synonym-aware eval. Use synonym_mode + vocab_size=8.
    wa_synonym = evaluate_word_to_action(
        bridge, n_trials_per_word=args.n_eval_per_word,
        stim_steps_per_trial=100, n_reset_steps=50, token_sparsity=0.1,
        synonym_mode=True, synonym_vocab_size=8,
        verbose=False,
    )
    print(f"  W->A on synonym vocab: {wa_synonym['accuracy']:.1%}", flush=True)

    retention["checkpoints"].append({
        "name": "after_phase_b",
        "primary_wa": wa_primary_after_b,
        "synonym_wa": wa_synonym,
    })

    # ─── Compute retention metrics ───
    primary_a_acc = wa_a["accuracy"]
    primary_b_acc = wa_primary_after_b["accuracy"]
    synonym_b_acc = wa_synonym["accuracy"]

    if primary_a_acc > 0:
        retention_pct = (primary_b_acc / primary_a_acc) * 100
    else:
        retention_pct = 0.0

    print("\n" + "=" * 60)
    print("CATASTROPHIC FORGETTING ASSESSMENT")
    print("=" * 60)
    print(f"  Primary accuracy after Phase A:  {primary_a_acc:.1%}")
    print(f"  Primary accuracy after Phase B:  {primary_b_acc:.1%}")
    print(f"  Retention ratio (post-A->post-B): {retention_pct:.0f}%")
    print(f"  Synonym new-learning accuracy:   {synonym_b_acc:.1%}")
    print()
    if retention_pct >= 80:
        print("  ✅ RETENTION GOOD (≥80%): biology-grounded continual "
              "learning preserves old patterns")
    elif retention_pct >= 50:
        print("  ⚠️  RETENTION MODERATE (50-80%): some drift, not "
              "catastrophic")
    else:
        print("  ❌ CATASTROPHIC FORGETTING (<50%): biology-grounded "
              "STDP alone insufficient for continual learning")
    print(f"  (Path F's premise requires retention ≥80% for continual learning claim)")
    print("=" * 60, flush=True)

    retention["metrics"] = {
        "primary_a_acc": primary_a_acc,
        "primary_b_acc": primary_b_acc,
        "synonym_b_acc": synonym_b_acc,
        "retention_pct": retention_pct,
    }

    if args.out_stats:
        Path(args.out_stats).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_stats).write_text(json.dumps(retention, indent=2,
                                                    default=str))
        print(f"\nSaved: {args.out_stats}", flush=True)


if __name__ == "__main__":
    main()
