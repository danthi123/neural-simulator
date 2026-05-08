"""Interactive chat REPL on biology-grounded foundation.

The first INTERACTIVE conversational artifact built on the validated
Phase 1.4 BRANCH A continual-learning + Tier 2.1 BREAKTHROUGH synonym
binding architectures. Supports two modes:

  --mode tier1           4-word vocab (north/east/south/west)
  --mode synonym         8-word vocab (synonyms: up/right/down/left also)

User types a direction word. Sim activates the corresponding motor pool
and responds with what it predicts. Quits on "quit", "exit", or EOF.

Per master plan section "For full Path F demo": "Accept Phase 1.4
BRANCH A as the primary continual-learning result + build conversational
demo on Phase 1.4 architecture using larger Tier 1/2.1 vocab."

This is the master plan's "build conversational demo on Phase 1.4
architecture" milestone — the interactive REPL that lets a user
actually talk to the sim.

Usage:
    # Tier 1 (4-word, ~6 min training):
    python -m research.runners.chat_repl --mode tier1 --seed 43 \\
        --train-events 200

    # Tier 2.1 synonym (8-word, ~20 min training):
    python -m research.runners.chat_repl --mode synonym --seed 42 \\
        --train-events 400

    # Then interactively:
    > north
    [TIER1 seed=43] sim hears 'north', activates motor_N (delta N+205, x2.1)
    > up
    [SYNONYM seed=42] sim hears 'up', activates motor_N (delta N+87, x1.7)
    > what
    [SYNONYM] 'what' is not in vocab; tracking deltas anyway:
              motor_N+12 motor_E+45 motor_S+8 motor_W-3
              best guess: motor_E (low confidence x1.4)
    > quit
    [DONE] 8 turns total.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


def _load_or_train_tier1(seed: int, n_train_events: int, verbose: bool):
    """Train Tier 1 architecture (4-word vocab) and return bridge."""
    from research.runners.bio_three_factor import run_three_factor

    if verbose:
        print(f"[TRAINING] Tier 1 architecture (seed={seed}, "
              f"n_events={n_train_events})", flush=True)
        t0 = time.time()

    bridge, _ = run_three_factor(
        seed=seed,
        n_events_per_direction=n_train_events,
        n_lang_input=2048,
        n_motor_per_action=500,
        n_motor_fs_per_action=60,
        biological=True,
        enable_motor_fs=True,
        enable_nmda=True,
        apply_topographic_bias=True,
        embodied_hebbian=True,
        synonym_mode=False,
        verbose=False,
    )

    if verbose:
        print(f"[TRAINING] complete ({time.time() - t0:.0f}s)", flush=True)

    try:
        bridge.set_plasticity_gate("language_input_to_motor", 0.0)
        bridge.set_plasticity_gate("motor_to_language_output", 0.0)
    except Exception:
        pass

    return bridge


def _load_or_train_synonym(seed: int, n_train_events: int, verbose: bool):
    """Train Tier 2.1 v4 scale-up synonym architecture (8-word vocab)."""
    from research.runners.bio_three_factor import run_three_factor

    if verbose:
        print(f"[TRAINING] Tier 2.1 v4 scale-up architecture "
              f"(seed={seed}, n_events={n_train_events})", flush=True)
        t0 = time.time()

    bridge, _ = run_three_factor(
        seed=seed,
        n_events_per_direction=n_train_events,
        n_lang_input=4096,
        n_motor_per_action=1000,
        n_motor_fs_per_action=120,
        biological=True,
        enable_motor_fs=True,
        enable_nmda=True,
        apply_topographic_bias=True,
        embodied_hebbian=True,
        synonym_mode=True,
        synonym_vocab_size=8,
        verbose=False,
    )

    if verbose:
        print(f"[TRAINING] complete ({time.time() - t0:.0f}s)", flush=True)

    try:
        bridge.set_plasticity_gate("language_input_to_motor", 0.0)
        bridge.set_plasticity_gate("motor_to_language_output", 0.0)
    except Exception:
        pass

    return bridge


def chat_inference(
    bridge,
    user_word: str,
    stim_steps: int = 100,
    reset_steps: int = 50,
    drive_pA: float = 200.0,
    sparsity: float = 0.1,
):
    """Run one chat turn with baseline-vs-driven delta methodology.

    Returns dict with delta_counts, predicted_action, predicted_direction,
    confidence_ratio.
    """
    import cupy as cp
    from sim.text_embeddings import vocab_to_drive_pattern

    rm = bridge.region_manager
    lang_input_idx = list(rm.indices("language_input"))
    motor_idx = {a: list(rm.indices(f"motor_{a}"))
                 for a in ["N", "E", "S", "W"]}
    motor_arr = {a: cp.asarray(motor_idx[a], dtype=cp.int64)
                 for a in ["N", "E", "S", "W"]}
    n_lang_in = len(lang_input_idx)
    lang_input_arr = cp.asarray(lang_input_idx, dtype=cp.int64)

    # Phase A: baseline (no input)
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

    # Phase B: driven (word input)
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(reset_steps):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
    drive = vocab_to_drive_pattern(
        user_word, n_neurons=n_lang_in,
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
    action_to_word = {"N": "north", "E": "east", "S": "south", "W": "west"}
    predicted_direction = action_to_word[predicted_action]

    sorted_delta = np.sort(delta)[::-1]
    if sorted_delta[1] > 0:
        confidence = float(sorted_delta[0] / sorted_delta[1])
    elif sorted_delta[0] > 0:
        confidence = float("inf")
    else:
        confidence = 1.0

    return {
        "user_word": user_word,
        "delta_counts": {a: int(delta[i])
                          for i, a in enumerate(["N", "E", "S", "W"])},
        "predicted_action": predicted_action,
        "predicted_direction": predicted_direction,
        "confidence_ratio": confidence,
    }


# ─── REPL ─────────────────────────────────────────────────────────────

VOCAB_TIER1 = {"north", "east", "south", "west"}
VOCAB_SYNONYM = {"north", "east", "south", "west",
                  "up", "right", "down", "left"}
WORD_TO_ACTION_SYNONYM = {
    "north": "N", "up": "N",
    "east": "E", "right": "E",
    "south": "S", "down": "S",
    "west": "W", "left": "W",
}


def _load_bridge_from_checkpoint(checkpoint_path: str, mode: str, seed: int,
                                   verbose: bool = True):
    """Load a previously-trained bridge from an HDF5 checkpoint.

    Reuses the standard build/init path then loads weights from disk.
    Per CLAUDE.md gotcha: save_checkpoint doesn't preserve firing
    thresholds, STP, eligibility -- but for inference (REPL chat),
    weights are what matter; dynamic state self-recovers in a few
    timesteps of free-running.
    """
    from sim.config import (
        CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig,
    )
    from sim.bridge import SimulationBridge

    if verbose:
        print(f"[LOAD] Reading bridge state from {checkpoint_path}",
              flush=True)
        t0 = time.time()

    # Re-build bridge with the same config (mode determines arch)
    if mode == "tier1":
        bridge = _load_or_train_tier1(seed, n_train_events=0, verbose=False)
    elif mode == "synonym":
        bridge = _load_or_train_synonym(seed, n_train_events=0, verbose=False)
    else:
        raise ValueError(f"unknown mode: {mode}")

    # Now overlay weights from the checkpoint.
    bridge.load_checkpoint(checkpoint_path)

    if verbose:
        print(f"[LOAD] complete ({time.time() - t0:.0f}s)", flush=True)

    # Re-freeze plasticity gates (load_checkpoint may have reset them)
    try:
        bridge.set_plasticity_gate("language_input_to_motor", 0.0)
        bridge.set_plasticity_gate("motor_to_language_output", 0.0)
    except Exception:
        pass

    return bridge


def _save_bridge_checkpoint(bridge, checkpoint_path: str, verbose: bool = True):
    """Save the trained bridge state for fast reload in future sessions."""
    from pathlib import Path

    if verbose:
        print(f"[SAVE] Writing bridge state to {checkpoint_path}",
              flush=True)
        t0 = time.time()

    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    bridge.save_checkpoint(checkpoint_path)

    if verbose:
        print(f"[SAVE] complete ({time.time() - t0:.0f}s)", flush=True)


def run_repl(mode: str, seed: int, n_train_events: int,
             transcript_out: str = None,
             load_bridge: str = None,
             save_bridge: str = None):
    """Train + interactive REPL loop.

    If load_bridge is given, skip training and load from checkpoint.
    If save_bridge is given (and we DID train), save the trained bridge
    for future use. Combined: training takes ~6-20 min depending on
    mode; checkpoint reload takes ~10-30 sec, making subsequent
    interactive sessions effectively instant.
    """
    print("=" * 60)
    print(f"BIOLOGY-GROUNDED CHAT REPL — mode={mode}, seed={seed}")
    print(f"Type a direction word; sim activates the motor pool.")
    print(f"Quit with 'quit', 'exit', or Ctrl-D.")
    print("=" * 60, flush=True)

    if load_bridge:
        bridge = _load_bridge_from_checkpoint(load_bridge, mode, seed,
                                                verbose=True)
        if mode == "tier1":
            vocab = VOCAB_TIER1
            mode_label = "TIER1"
        elif mode == "synonym":
            vocab = VOCAB_SYNONYM
            mode_label = "SYNONYM"
        else:
            raise ValueError(f"unknown mode: {mode}")
    elif mode == "tier1":
        bridge = _load_or_train_tier1(seed, n_train_events, verbose=True)
        vocab = VOCAB_TIER1
        mode_label = "TIER1"
        if save_bridge:
            _save_bridge_checkpoint(bridge, save_bridge, verbose=True)
    elif mode == "synonym":
        bridge = _load_or_train_synonym(seed, n_train_events, verbose=True)
        vocab = VOCAB_SYNONYM
        mode_label = "SYNONYM"
        if save_bridge:
            _save_bridge_checkpoint(bridge, save_bridge, verbose=True)
    else:
        raise ValueError(f"unknown mode: {mode}")

    print(f"\nReady. Vocab: {sorted(vocab)}")
    print(f"Type a word and press Enter.\n", flush=True)

    transcript = []
    n_turns = 0
    correct = 0

    try:
        while True:
            try:
                line = input("> ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\n[EOF]", flush=True)
                break

            if not line:
                continue
            if line in ("quit", "exit", "q"):
                print("[QUIT]", flush=True)
                break

            n_turns += 1
            result = chat_inference(bridge, line)
            d = result["delta_counts"]
            pred_action = result["predicted_action"]
            pred_word = result["predicted_direction"]
            conf = result["confidence_ratio"]

            in_vocab = line in vocab
            expected_action = (
                WORD_TO_ACTION_SYNONYM.get(line) if mode == "synonym"
                else {"north": "N", "east": "E",
                      "south": "S", "west": "W"}.get(line)
            )
            is_correct = (in_vocab and pred_action == expected_action)
            if is_correct:
                correct += 1

            if in_vocab:
                marker = "[OK]" if is_correct else "[X] "
                print(f"  {marker} [{mode_label} seed={seed}] sim hears "
                      f"'{line}', activates motor_{pred_action} "
                      f"(delta N{d['N']:+d} E{d['E']:+d} "
                      f"S{d['S']:+d} W{d['W']:+d}, x{conf:.1f})",
                      flush=True)
            else:
                print(f"  [?] '{line}' is not in vocab; tracking deltas "
                      f"anyway:", flush=True)
                print(f"      delta N{d['N']:+d} E{d['E']:+d} "
                      f"S{d['S']:+d} W{d['W']:+d}", flush=True)
                print(f"      best guess: motor_{pred_action} "
                      f"(low confidence x{conf:.1f})", flush=True)

            transcript.append({
                "turn": n_turns,
                "user_word": line,
                "in_vocab": in_vocab,
                "expected_action": expected_action,
                "predicted_action": pred_action,
                "confidence": conf,
                "delta": d,
                "correct": is_correct,
            })
    finally:
        print("\n" + "=" * 60)
        print(f"[DONE] {n_turns} turns total.")
        if n_turns > 0:
            in_vocab_turns = sum(1 for t in transcript if t["in_vocab"])
            if in_vocab_turns > 0:
                print(f"  In-vocab accuracy: {correct}/{in_vocab_turns} "
                      f"= {correct/in_vocab_turns:.1%}")
        print("=" * 60, flush=True)

        if transcript_out and transcript:
            Path(transcript_out).parent.mkdir(parents=True, exist_ok=True)
            md = []
            md.append(f"# Interactive REPL transcript (mode={mode}, "
                      f"seed={seed})\n\n")
            md.append(f"**Vocab:** {sorted(vocab)}  \n")
            md.append(f"**Training:** {n_train_events} events/word\n\n")
            md.append("## Conversation\n\n```\n")
            for t in transcript:
                marker = "[OK]" if t["correct"] else "[X] " if t["in_vocab"] else "[?] "
                d = t["delta"]
                md.append(
                    f"  {marker} You: {t['user_word']:<8} -> "
                    f"motor_{t['predicted_action']} "
                    f"(delta N{d['N']:+4d} E{d['E']:+4d} "
                    f"S{d['S']:+4d} W{d['W']:+4d}, x{t['confidence']:.1f})\n"
                )
            md.append("```\n")
            Path(transcript_out).write_text("".join(md), encoding="utf-8")
            print(f"  Transcript saved: {transcript_out}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["tier1", "synonym"], default="tier1",
                    help="Architecture mode (tier1=4-word, synonym=8-word)")
    ap.add_argument("--seed", type=int, default=43,
                    help="Random seed (43 is the documented best Tier 1 seed; "
                         "42 is best Tier 2.1 single-seed)")
    ap.add_argument("--train-events", type=int, default=None,
                    help="Events per word during training (default: "
                         "200 for tier1, 400 for synonym)")
    ap.add_argument("--transcript-out", type=str, default=None,
                    help="Save transcript to this markdown file at exit")
    ap.add_argument("--save-bridge", type=str, default=None,
                    help="Save the trained bridge state to this HDF5 path "
                         "after training. Future sessions can reload with "
                         "--load-bridge to skip ~6 min of training.")
    ap.add_argument("--load-bridge", type=str, default=None,
                    help="Load a previously-saved bridge state instead of "
                         "training. Skips the ~6 min training phase and "
                         "starts the REPL in ~10-30 sec. Per CLAUDE.md "
                         "save_checkpoint gotcha: doesn't preserve firing "
                         "thresholds / STP / eligibility -- but for "
                         "inference (REPL chat), weights are sufficient.")
    args = ap.parse_args()

    if args.train_events is None:
        args.train_events = 200 if args.mode == "tier1" else 400

    if args.load_bridge and args.save_bridge:
        ap.error("--load-bridge and --save-bridge are mutually exclusive "
                 "(saving overwrites a checkpoint that was just loaded)")

    run_repl(
        mode=args.mode,
        seed=args.seed,
        n_train_events=args.train_events,
        transcript_out=args.transcript_out,
        load_bridge=args.load_bridge,
        save_bridge=args.save_bridge,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
