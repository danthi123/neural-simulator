"""Phase 1.4 BRANCH A continual learning demo (chat-transcript form).

Shows the validated continual-learning capability end-to-end:

1. Train primary vocab (north/east/south/west) via Tier 1
   embodied Hebbian
2. Test binding -> baseline accuracy
3. Train SYNONYMS only (up/right/down/left) -- no primary
   exposure. Phase 1.4 paradigm.
4. Re-test PRIMARIES -> retention (validates no catastrophic
   forgetting)
5. Test synonyms -> new vocab binding

This is the user-friendly chat-transcript version of
continual_forgetting_eval.py, designed for a non-technical
observer to see the Phase 1.4 BRANCH A finding (5/6 PASS, mean
103% retention) in concrete terms.

Usage:
    python -m research.runners.chat_continual_demo \\
        --seed 43 --train-events 200 \\
        --transcript-out research/findings/2026-05-07-continual-demo.md
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List


PRIMARY_WORDS = ["north", "east", "south", "west"]
PRIMARY_TO_ACTION = {"north": "N", "east": "E", "south": "S", "west": "W"}
SYNONYM_WORDS = ["up", "right", "down", "left"]
SYNONYM_TO_ACTION = {"up": "N", "right": "E", "down": "S", "left": "W"}

ACTION_TO_PRIMARY = {a: w for w, a in PRIMARY_TO_ACTION.items()}
ACTION_TO_SYNONYM = {a: w for w, a in SYNONYM_TO_ACTION.items()}


def chat_test_word(bridge, word: str, vocab_action_map: dict,
                    stim_steps: int = 100, reset_steps: int = 50,
                    drive_pA: float = 200.0, sparsity: float = 0.1):
    """Single test turn using baseline-vs-driven delta methodology."""
    import cupy as cp
    import numpy as np
    from sim.text_embeddings import vocab_to_drive_pattern

    rm = bridge.region_manager
    lang_input_idx = list(rm.indices("language_input"))
    motor_idx = {a: list(rm.indices(f"motor_{a}"))
                 for a in ["N", "E", "S", "W"]}
    motor_arr = {a: cp.asarray(motor_idx[a], dtype=cp.int64)
                 for a in ["N", "E", "S", "W"]}
    n_lang_in = len(lang_input_idx)
    lang_input_arr = cp.asarray(lang_input_idx, dtype=cp.int64)

    # Phase A: baseline
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(reset_steps):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
    baseline = cp.zeros(4, dtype=cp.int32)
    for _ in range(stim_steps):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        fired = bridge.cp_firing_states
        for a_i, a in enumerate(["N", "E", "S", "W"]):
            baseline[a_i] += fired[motor_arr[a]].sum()

    # Phase B: driven
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(reset_steps):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
    drive = vocab_to_drive_pattern(
        word, n_neurons=n_lang_in,
        drive_max_pA=drive_pA, sparsity=sparsity,
    )
    bridge.cp_external_input_current[lang_input_arr] = \
        cp.asarray(drive, dtype=cp.float32)
    drive_counts = cp.zeros(4, dtype=cp.int32)
    for _ in range(stim_steps):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        fired = bridge.cp_firing_states
        for a_i, a in enumerate(["N", "E", "S", "W"]):
            drive_counts[a_i] += fired[motor_arr[a]].sum()

    bl = baseline.get()
    dr = drive_counts.get()
    delta = dr - bl
    predicted_idx = int(np.argmax(delta))
    predicted_action = ["N", "E", "S", "W"][predicted_idx]
    target_action = vocab_action_map.get(word)

    return {
        "word": word,
        "delta": {a: int(delta[i])
                   for i, a in enumerate(["N", "E", "S", "W"])},
        "predicted_action": predicted_action,
        "target_action": target_action,
        "correct": predicted_action == target_action,
    }


def run_continual_demo(
    seed: int = 42,
    n_events_per_word: int = 200,
    n_test_per_word: int = 5,
    n_lang_input: int = 2048,
    n_motor_per_action: int = 500,
    n_motor_fs_per_action: int = 60,
    transcript_out: str = None,
    verbose: bool = True,
):
    """Full continual-learning demo:
    1. Train primaries (Tier 1 embodied Hebbian)
    2. Test primary binding
    3. Train synonyms only (Phase 1.4 paradigm)
    4. Test primary retention + synonym new learning
    """
    from research.runners.bio_three_factor import run_three_factor
    import cupy as cp
    import numpy as np
    from sim.text_embeddings import vocab_to_drive_pattern
    # 2026-05-09: emit_progress for live frontend visibility
    from sim.progress import emit_progress

    transcript = []
    transcript.append({"type": "system",
                        "text": f"PHASE 1.4 BRANCH A continual learning demo "
                                f"(seed={seed})"})
    transcript.append({"type": "system",
                        "text": f"Training primaries: {PRIMARY_WORDS} via "
                                f"Tier 1 embodied Hebbian "
                                f"({n_events_per_word} events/word)..."})

    # 4 phases: train_primaries, test_primaries, train_synonyms, test_retention
    emit_progress("phase", current=0, total=4, phase="train_primaries",
                  unit="phases", label="chat_continual_demo")

    if verbose:
        print(f"[TRAINING PHASE A] Primaries: {PRIMARY_WORDS}")
    t0 = time.time()
    bridge, _ = run_three_factor(
        seed=seed,
        n_events_per_direction=n_events_per_word,
        n_lang_input=n_lang_input,
        n_motor_per_action=n_motor_per_action,
        n_motor_fs_per_action=n_motor_fs_per_action,
        biological=True,
        enable_motor_fs=True,
        enable_nmda=True,
        apply_topographic_bias=True,
        embodied_hebbian=True,
        synonym_mode=False,
        verbose=False,
    )
    elapsed = time.time() - t0
    if verbose:
        print(f"  done ({elapsed:.0f}s)")
    emit_progress("complete", current=1, total=4, phase="train_primaries",
                  unit="phases", label="chat_continual_demo",
                  wall_clock_s=int(elapsed))

    transcript.append({"type": "system",
                        "text": f"Phase A training complete "
                                f"({elapsed:.0f}s).\n\nTest primary binding:"})

    # Freeze plasticity for testing
    try:
        bridge.set_plasticity_gate("language_input_to_motor", 0.0)
        bridge.set_plasticity_gate("motor_to_language_output", 0.0)
    except Exception:
        pass

    if verbose:
        print(f"\n[TEST] Primary binding (post Phase A):")
    emit_progress("phase", current=1, total=4, phase="test_primaries",
                  unit="phases", label="chat_continual_demo")
    primary_results_post_a = []
    for word in PRIMARY_WORDS:
        for _ in range(n_test_per_word):
            r = chat_test_word(bridge, word, PRIMARY_TO_ACTION)
            primary_results_post_a.append(r)
            if verbose:
                marker = "[OK]" if r["correct"] else "[X]"
                d = r["delta"]
                pred_word = ACTION_TO_PRIMARY[r["predicted_action"]]
                print(f"  {marker} You: {word:<6} -> Sim: {pred_word:<6} "
                      f"(delta N{d['N']:+4d} E{d['E']:+4d} S{d['S']:+4d} "
                      f"W{d['W']:+4d})")
            transcript.append({"type": "test", "phase": "primary_post_a",
                                **r})

    primary_a_correct = sum(1 for r in primary_results_post_a if r["correct"])
    primary_a_acc = primary_a_correct / len(primary_results_post_a)
    transcript.append({"type": "result",
                        "text": f"Primary post-A: {primary_a_correct}/"
                                f"{len(primary_results_post_a)} = "
                                f"{primary_a_acc:.0%}"})

    # ─── Phase B: train synonyms ONLY ───
    transcript.append({"type": "system",
                        "text": f"\nNow training NEW synonyms: {SYNONYM_WORDS}"
                                f". NO primary exposure during Phase B.\n"
                                f"This is the Phase 1.4 catastrophic-"
                                f"forgetting test."})
    if verbose:
        print(f"\n[TRAINING PHASE B] Synonyms only (no primary exposure):")
        t0 = time.time()

    # Manual Phase B training (synonym-only, on existing bridge)
    rm = bridge.region_manager
    lang_input_idx = list(rm.indices("language_input"))
    motor_idx = {a: list(rm.indices(f"motor_{a}"))
                 for a in ["N", "E", "S", "W"]}
    lang_output_idx = list(rm.indices("language_output"))
    n_lang_in = len(lang_input_idx)
    lang_input_arr = cp.asarray(lang_input_idx, dtype=cp.int64)
    lang_output_arr = cp.asarray(lang_output_idx, dtype=cp.int64)
    motor_arr = {a: cp.asarray(motor_idx[a], dtype=cp.int64)
                 for a in ["N", "E", "S", "W"]}

    rng = np.random.default_rng(seed * 7 + 1)
    synonym_buffer = []
    for word in SYNONYM_WORDS:
        action = SYNONYM_TO_ACTION[word]
        for _ in range(n_events_per_word):
            synonym_buffer.append({"token": word, "action": action})
    rng.shuffle(synonym_buffer)

    try:
        bridge.set_plasticity_gate("language_input_to_motor", 1.0)
        bridge.set_plasticity_gate("motor_to_language_output", 1.0)
    except Exception:
        pass

    embodied_motor_teacher_pA = 300.0
    stim_steps_per_event = 50
    reset_steps_phase_b = 50
    for ev in synonym_buffer:
        bridge.cp_external_input_current[:] = 0.0
        for _ in range(reset_steps_phase_b):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
        d = vocab_to_drive_pattern(ev["token"], n_neurons=n_lang_in,
                                    drive_max_pA=200.0, sparsity=0.1)
        bridge.cp_external_input_current[lang_input_arr] = \
            cp.asarray(d, dtype=cp.float32)
        bridge.cp_external_input_current[lang_output_arr] = \
            cp.asarray(d, dtype=cp.float32)
        bridge.cp_external_input_current[motor_arr[ev["action"]]] += \
            float(embodied_motor_teacher_pA)
        for _ in range(stim_steps_per_event):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1

    # Freeze for testing
    try:
        bridge.set_plasticity_gate("language_input_to_motor", 0.0)
        bridge.set_plasticity_gate("motor_to_language_output", 0.0)
    except Exception:
        pass

    if verbose:
        elapsed = time.time() - t0
        print(f"  Phase B training done ({elapsed:.0f}s)\n")
    transcript.append({"type": "system",
                        "text": f"Phase B complete ({elapsed:.0f}s).\n\n"
                                f"Test PRIMARY retention "
                                f"(did synonym training erase primaries?):"})

    if verbose:
        print(f"[TEST] Primary retention (post Phase B):")
    primary_results_post_b = []
    for word in PRIMARY_WORDS:
        for _ in range(n_test_per_word):
            r = chat_test_word(bridge, word, PRIMARY_TO_ACTION)
            primary_results_post_b.append(r)
            if verbose:
                marker = "[OK]" if r["correct"] else "[X]"
                d = r["delta"]
                pred_word = ACTION_TO_PRIMARY[r["predicted_action"]]
                print(f"  {marker} You: {word:<6} -> Sim: {pred_word:<6} "
                      f"(delta N{d['N']:+4d} E{d['E']:+4d} S{d['S']:+4d} "
                      f"W{d['W']:+4d})")
            transcript.append({"type": "test", "phase": "primary_post_b",
                                **r})

    primary_b_correct = sum(1 for r in primary_results_post_b if r["correct"])
    primary_b_acc = primary_b_correct / len(primary_results_post_b)
    retention = (primary_b_acc / primary_a_acc) if primary_a_acc > 0 else 0
    transcript.append({"type": "result",
                        "text": f"Primary post-B: {primary_b_correct}/"
                                f"{len(primary_results_post_b)} = "
                                f"{primary_b_acc:.0%} "
                                f"(retention: {retention:.0%})"})

    if verbose:
        print(f"\n[TEST] Synonym new learning:")
    synonym_results = []
    for word in SYNONYM_WORDS:
        for _ in range(n_test_per_word):
            r = chat_test_word(bridge, word, SYNONYM_TO_ACTION)
            synonym_results.append(r)
            if verbose:
                marker = "[OK]" if r["correct"] else "[X]"
                d = r["delta"]
                # Synonym words; show target action's word
                target_pred_word = (ACTION_TO_PRIMARY[r["predicted_action"]]
                                    if r["predicted_action"] in ACTION_TO_PRIMARY
                                    else "?")
                print(f"  {marker} You: {word:<6} -> Sim: "
                      f"{target_pred_word:<6} "
                      f"(delta N{d['N']:+4d} E{d['E']:+4d} S{d['S']:+4d} "
                      f"W{d['W']:+4d})")
            transcript.append({"type": "test", "phase": "synonym",
                                **r})

    synonym_correct = sum(1 for r in synonym_results if r["correct"])
    synonym_acc = synonym_correct / len(synonym_results)
    transcript.append({"type": "result",
                        "text": f"Synonym new learning: {synonym_correct}/"
                                f"{len(synonym_results)} = "
                                f"{synonym_acc:.0%}"})

    # Verdict
    if retention >= 0.8:
        verdict = ("PASS (>= 80% retention) -- biology-grounded "
                   "continual learning preserves old knowledge!")
    elif retention >= 0.5:
        verdict = ("MODERATE (50-80% retention) -- some loss but "
                   "not catastrophic")
    else:
        verdict = "FAIL (<50% retention) -- catastrophic forgetting"

    transcript.append({"type": "summary",
                        "primary_a_acc": primary_a_acc,
                        "primary_b_acc": primary_b_acc,
                        "synonym_acc": synonym_acc,
                        "retention": retention,
                        "verdict": verdict})

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"DEMO SUMMARY (seed={seed})")
        print(f"{'=' * 60}")
        print(f"Primary post-A:    {primary_a_acc:.0%}")
        print(f"Primary post-B:    {primary_b_acc:.0%}")
        print(f"Retention ratio:   {retention:.0%}")
        print(f"Synonym learning:  {synonym_acc:.0%}")
        print(f"Verdict: {verdict}")

    if transcript_out:
        write_continual_md(transcript, transcript_out, seed=seed,
                            n_events=n_events_per_word)

    return {
        "seed": seed,
        "primary_a_acc": primary_a_acc,
        "primary_b_acc": primary_b_acc,
        "synonym_acc": synonym_acc,
        "retention": retention,
        "verdict": verdict,
    }


def write_continual_md(transcript, path, seed, n_events):
    md = []
    md.append("# Phase 1.4 BRANCH A continual learning demo "
              "(chat-transcript)\n\n")
    md.append(f"**Seed:** {seed}\n")
    md.append(f"**Training:** Tier 1 embodied Hebbian, {n_events} events/word\n\n")
    md.append("---\n\n## Conversation transcript\n\n```\n")

    for entry in transcript:
        if entry["type"] == "system":
            md.append(f"[SYSTEM] {entry['text']}\n\n")
        elif entry["type"] == "test":
            r = entry
            phase = entry["phase"]
            marker = "[OK]" if r["correct"] else "[X]"
            d = r["delta"]
            target_word = (ACTION_TO_PRIMARY[r["target_action"]]
                            if r["phase"] != "synonym"
                            else ACTION_TO_SYNONYM.get(r["target_action"], "?"))
            pred_word = (ACTION_TO_PRIMARY[r["predicted_action"]]
                         if r["predicted_action"] in ACTION_TO_PRIMARY
                         else "?")
            md.append(
                f"  {marker} You: {r['word']:<6} -> Sim: {pred_word:<6} "
                f"(delta N{d['N']:+4d} E{d['E']:+4d} S{d['S']:+4d} "
                f"W{d['W']:+4d})\n"
            )
        elif entry["type"] == "result":
            md.append(f"\n  >> {entry['text']}\n\n")
        elif entry["type"] == "summary":
            md.append(f"\n=== SUMMARY ===\n")
            md.append(f"Primary post-A:   {entry['primary_a_acc']:.0%}\n")
            md.append(f"Primary post-B:   {entry['primary_b_acc']:.0%}\n")
            md.append(f"Retention ratio:  {entry['retention']:.0%}\n")
            md.append(f"Synonym learning: {entry['synonym_acc']:.0%}\n")
            md.append(f"Verdict: {entry['verdict']}\n")

    md.append("```\n\n---\n\n## What this demonstrates\n\n")
    md.append("Phase 1.4 BRANCH A continual learning (validated 5/6 PASS, "
              "mean 103% retention across 6 seeds):\n\n")
    md.append("- Tier 1 binds primaries via embodied Hebbian (Phase A)\n")
    md.append("- Synonym-only training (Phase B, no primary exposure)\n")
    md.append("- Primary retention measured -- catastrophic forgetting test\n")
    md.append("- Synonym new-learning measured -- novel binding test\n\n")
    md.append("Pass criterion: retention >= 80% (>= 4/6 seeds in 6-seed "
              "validation).\n\n")
    md.append("Per master plan: this is THE foundational test for Path F's "
              "biology-grounded continual learning premise. Validated at "
              "6-seed (5/6 PASS, mean 103% retention).\n")

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(md), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-events", type=int, default=200)
    ap.add_argument("--n-test-per-word", type=int, default=5)
    ap.add_argument("--transcript-out", type=str, default=None)
    ap.add_argument("--out-stats", type=str, default=None)
    args = ap.parse_args()

    result = run_continual_demo(
        seed=args.seed,
        n_events_per_word=args.train_events,
        n_test_per_word=args.n_test_per_word,
        transcript_out=args.transcript_out,
        verbose=True,
    )

    if args.out_stats:
        Path(args.out_stats).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_stats).write_text(json.dumps(result, indent=2,
                                                    default=str))
        print(f"\nSaved stats: {args.out_stats}")


if __name__ == "__main__":
    main()
