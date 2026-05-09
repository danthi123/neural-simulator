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


def _load_or_train_synonym(seed: int, n_train_events: int, verbose: bool,
                             vocab_size: int = 8,
                             n_motor_per_action: int = 1000,
                             n_motor_fs_per_action: int = 120):
    """Train Tier 2.1 scale-up synonym architecture.

    vocab_size=8: validated 3/3 GO (n_motor=1000)
    vocab_size=12: PARTIAL at default n_motor=1000, GO at n_motor=2000
                   (capacity hypothesis; per 2026-05-08 finding)
    vocab_size=16: tested only at n_motor=2000 (master plan extension)
    """
    from research.runners.bio_three_factor import run_three_factor

    if verbose:
        print(f"[TRAINING] Tier 2.1 scale-up architecture "
              f"(seed={seed}, n_events={n_train_events}, "
              f"vocab={vocab_size}, n_motor={n_motor_per_action})",
              flush=True)
        t0 = time.time()

    bridge, _ = run_three_factor(
        seed=seed,
        n_events_per_direction=n_train_events,
        n_lang_input=4096,
        n_motor_per_action=n_motor_per_action,
        n_motor_fs_per_action=n_motor_fs_per_action,
        biological=True,
        enable_motor_fs=True,
        enable_nmda=True,
        apply_topographic_bias=True,
        embodied_hebbian=True,
        synonym_mode=True,
        synonym_vocab_size=vocab_size,
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


# ─── Online vocab learning (Track 3 scaffolding, 2026-05-09) ─────────


def _parse_learn_command(line: str):
    """Parse a 'learn <word> <action>' REPL command.

    Returns (word, action) on success, None on parse failure. Strips
    whitespace and lowercases word; uppercases action; validates action
    is one of N/E/S/W.

    Examples:
        'learn ahead N'         -> ('ahead', 'N')
        'learn ahead north'     -> ('ahead', 'N')   # word form ok
        'learn forward up'      -> ('forward', 'N') # synonyms accepted
        'learn  HELLO  e '      -> ('hello', 'E')   # trim + case
        'learn'                 -> None             # missing args
        'learn ahead'           -> None             # missing action
        'learn ahead nope'      -> None             # bad action
    """
    parts = line.strip().split()
    if len(parts) < 3 or parts[0].lower() != "learn":
        return None
    word = parts[1].strip().lower()
    action_raw = parts[2].strip().lower()
    # Accept N/E/S/W directly OR full direction names OR synonym words
    action_aliases = {
        "n": "N", "north": "N", "up": "N", "↑": "N",
        "e": "E", "east":  "E", "right": "E", "→": "E",
        "s": "S", "south": "S", "down": "S", "↓": "S",
        "w": "W", "west":  "W", "left": "W", "←": "W",
    }
    action = action_aliases.get(action_raw)
    if action is None:
        return None
    if not word:
        return None
    return (word, action)


def learn_word_pairing(bridge, word: str, target_action: str,
                       n_events: int = 50, stim_steps_per_event: int = 100,
                       reset_steps: int = 50, drive_pA: float = 200.0,
                       teacher_pA: float = 1500.0, sparsity: float = 0.1,
                       verbose: bool = True):
    """Online embodied-Hebbian binding of a NEW word to an existing motor pool.

    Runs ``n_events`` paired co-firing events on the already-trained bridge:
      - Drive language_input with the new word's drive pattern
      - Drive language_output with the same pattern (output teacher)
      - Drive motor_<target_action> with elevated current (action teacher)
      - Step the bridge so STDP fires on co-active synapses

    The bridge's plastic ``language_input_to_motor`` and (if present)
    ``motor_to_language_output`` gates are temporarily opened, then
    re-frozen on exit. This lets the existing population codes reach
    new bindings without disturbing inference-time stability.

    Args:
        bridge: trained SimulationBridge (post chat_repl init)
        word: new vocabulary word to bind
        target_action: one of "N", "E", "S", "W"
        n_events: number of paired events (50 is a moderate dose;
            empirically gives a ~detectable binding without dramatically
            shifting existing bindings on the same motor pool)
        stim_steps_per_event: forward-prop steps per event
        reset_steps: free-running steps between events to clear
            transient state
        drive_pA: peak drive on language input + output sites
        teacher_pA: motor-pool teacher current (must be high enough to
            drive motor_X spikes regardless of upstream)
        sparsity: fraction of language_input neurons activated by the
            word's drive pattern
        verbose: log progress every 10 events

    Returns:
        dict with summary stats (n_events_run, target_action, gates_opened)
    """
    import cupy as cp
    from sim.text_embeddings import vocab_to_drive_pattern

    if target_action not in ("N", "E", "S", "W"):
        raise ValueError(f"target_action must be N/E/S/W, got {target_action!r}")

    rm = bridge.region_manager
    lang_input_idx = list(rm.indices("language_input"))
    motor_idx = list(rm.indices(f"motor_{target_action}"))
    lang_input_arr = cp.asarray(lang_input_idx, dtype=cp.int64)
    motor_arr = cp.asarray(motor_idx, dtype=cp.int64)
    n_lang_in = len(lang_input_idx)

    # language_output is optional — only present if bridge was trained with
    # embodied_hebbian=True (which chat_repl always does, but defensive).
    try:
        lang_output_idx = list(rm.indices("language_output"))
        lang_output_arr = cp.asarray(lang_output_idx, dtype=cp.int64)
        n_lang_out = len(lang_output_idx)
        has_output = True
    except Exception:
        has_output = False
        n_lang_out = 0

    # Drive pattern for the new word — same scheme as inference path.
    drive_in = vocab_to_drive_pattern(
        word, n_neurons=n_lang_in,
        drive_max_pA=drive_pA, sparsity=sparsity,
    )
    drive_in_gpu = cp.asarray(drive_in, dtype=cp.float32)
    if has_output:
        drive_out = vocab_to_drive_pattern(
            word, n_neurons=n_lang_out,
            drive_max_pA=drive_pA, sparsity=sparsity,
        )
        drive_out_gpu = cp.asarray(drive_out, dtype=cp.float32)

    # Open plasticity gates for the duration of learning.
    gates_opened = []
    for gate_name in ("language_input_to_motor", "motor_to_language_output"):
        try:
            bridge.set_plasticity_gate(gate_name, 1.0)
            gates_opened.append(gate_name)
        except Exception:
            pass

    if verbose:
        print(f"[LEARN] '{word}' -> motor_{target_action} | "
              f"{n_events} events | gates open: {gates_opened}",
              flush=True)
        t0 = time.time()

    try:
        for ev in range(n_events):
            # Reset between events: zero drive, free-run to clear transients
            bridge.cp_external_input_current[:] = 0.0
            for _ in range(reset_steps):
                bridge._run_one_simulation_step()
                bridge.runtime_state.current_time_step += 1

            # Drive language_input + language_output + motor_TARGET
            bridge.cp_external_input_current[lang_input_arr] = drive_in_gpu
            if has_output:
                bridge.cp_external_input_current[lang_output_arr] = drive_out_gpu
            bridge.cp_external_input_current[motor_arr] += float(teacher_pA)

            # Forward-prop — STDP fires on plastic synapses with co-active pre+post
            for _ in range(stim_steps_per_event):
                bridge._run_one_simulation_step()
                bridge.runtime_state.current_time_step += 1

            if verbose and (ev + 1) % 10 == 0:
                print(f"  [LEARN] {ev + 1}/{n_events} events", flush=True)
    finally:
        # Re-freeze gates regardless of exception
        for gate_name in gates_opened:
            try:
                bridge.set_plasticity_gate(gate_name, 0.0)
            except Exception:
                pass

    if verbose:
        print(f"[LEARN] complete ({time.time() - t0:.0f}s)", flush=True)

    return {
        "word": word,
        "target_action": target_action,
        "n_events_run": n_events,
        "gates_opened": gates_opened,
    }


# ─── REPL ─────────────────────────────────────────────────────────────

VOCAB_TIER1 = {"north", "east", "south", "west"}
VOCAB_SYNONYM = {"north", "east", "south", "west",
                  "up", "right", "down", "left"}
VOCAB_SYNONYM_12 = VOCAB_SYNONYM | {"n", "e", "s", "w"}
VOCAB_SYNONYM_16 = VOCAB_SYNONYM_12 | {"↑", "→", "↓", "←"}

WORD_TO_ACTION_SYNONYM = {
    "north": "N", "up": "N",
    "east": "E", "right": "E",
    "south": "S", "down": "S",
    "west": "W", "left": "W",
}
WORD_TO_ACTION_SYNONYM_12 = {**WORD_TO_ACTION_SYNONYM,
    "n": "N", "e": "E", "s": "S", "w": "W"}
WORD_TO_ACTION_SYNONYM_16 = {**WORD_TO_ACTION_SYNONYM_12,
    "↑": "N", "→": "E", "↓": "S", "←": "W"}


def _vocab_for_mode(mode: str):
    """Return (vocab_set, word_to_action_dict) for a chat_repl mode."""
    if mode == "tier1":
        return VOCAB_TIER1, {"north": "N", "east": "E",
                              "south": "S", "west": "W"}
    if mode == "synonym":
        return VOCAB_SYNONYM, WORD_TO_ACTION_SYNONYM
    if mode == "synonym12":
        return VOCAB_SYNONYM_12, WORD_TO_ACTION_SYNONYM_12
    if mode == "synonym16":
        return VOCAB_SYNONYM_16, WORD_TO_ACTION_SYNONYM_16
    raise ValueError(f"unknown mode: {mode}")


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
             save_bridge: str = None,
             scripted_words: list = None,
             allow_learn: bool = False,
             learn_n_events: int = 50):
    """Train + interactive REPL loop.

    If load_bridge is given, skip training and load from checkpoint.
    If save_bridge is given (and we DID train), save the trained bridge
    for future use. Combined: training takes ~6-20 min depending on
    mode; checkpoint reload takes ~10-30 sec, making subsequent
    interactive sessions effectively instant.

    If scripted_words is given (a list of words), run those instead of
    interactive stdin -- useful for CI / regression tests / batch
    eval. Exits after processing the list.

    If ``allow_learn`` is True (per --learn), the REPL recognizes
    ``learn <word> <action>`` commands which run an online embodied-
    Hebbian binding session of ``learn_n_events`` paired events, then
    automatically test the new binding. Default OFF — learning during
    the REPL is opt-in because it can perturb existing bindings.
    """
    print("=" * 60)
    print(f"BIOLOGY-GROUNDED CHAT REPL — mode={mode}, seed={seed}")
    print(f"Type a direction word; sim activates the motor pool.")
    if allow_learn:
        print(f"Online learning: ON. Type 'learn <word> <action>' to bind a new word.")
    print(f"Quit with 'quit', 'exit', or Ctrl-D.")
    print("=" * 60, flush=True)

    vocab, _ = _vocab_for_mode(mode)
    mode_label = mode.upper()

    if load_bridge:
        bridge = _load_bridge_from_checkpoint(load_bridge, mode, seed,
                                                verbose=True)
    elif mode == "tier1":
        bridge = _load_or_train_tier1(seed, n_train_events, verbose=True)
        if save_bridge:
            _save_bridge_checkpoint(bridge, save_bridge, verbose=True)
    elif mode == "synonym":
        bridge = _load_or_train_synonym(seed, n_train_events, verbose=True,
                                          vocab_size=8)
        if save_bridge:
            _save_bridge_checkpoint(bridge, save_bridge, verbose=True)
    elif mode == "synonym12":
        # 12-word: capacity boundary at default arch -- use scaled (n_motor=2000)
        bridge = _load_or_train_synonym(seed, n_train_events, verbose=True,
                                          vocab_size=12,
                                          n_motor_per_action=2000,
                                          n_motor_fs_per_action=240)
        if save_bridge:
            _save_bridge_checkpoint(bridge, save_bridge, verbose=True)
    elif mode == "synonym16":
        # 16-word: only tested at scaled arch (n_motor=2000)
        bridge = _load_or_train_synonym(seed, n_train_events, verbose=True,
                                          vocab_size=16,
                                          n_motor_per_action=2000,
                                          n_motor_fs_per_action=240)
        if save_bridge:
            _save_bridge_checkpoint(bridge, save_bridge, verbose=True)
    else:
        raise ValueError(f"unknown mode: {mode}")

    print(f"\nReady. Vocab: {sorted(vocab)}")
    if scripted_words is None:
        print(f"Type a word and press Enter.\n", flush=True)
    else:
        print(f"[SCRIPTED] running {len(scripted_words)} predefined words.",
              flush=True)

    transcript = []
    n_turns = 0
    correct = 0
    scripted_iter = iter(scripted_words) if scripted_words else None

    try:
        while True:
            if scripted_iter is not None:
                try:
                    line = next(scripted_iter).strip().lower()
                    print(f"> {line}", flush=True)
                except StopIteration:
                    print("[SCRIPTED COMPLETE]", flush=True)
                    break
            else:
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

            # Online learn command (only when --learn was passed)
            if allow_learn and line.startswith("learn "):
                parsed = _parse_learn_command(line)
                if parsed is None:
                    print("  [?] usage: learn <word> <action>  "
                          "(action = N/E/S/W or north/east/south/west or "
                          "up/right/down/left)", flush=True)
                    continue
                new_word, target = parsed
                # Run the binding session, then auto-test the new word
                learn_word_pairing(bridge, new_word, target,
                                   n_events=learn_n_events, verbose=True)
                test_result = chat_inference(bridge, new_word)
                td = test_result["delta_counts"]
                pred_a = test_result["predicted_action"]
                conf = test_result["confidence_ratio"]
                bound_ok = (pred_a == target)
                marker = "[OK]" if bound_ok else "[X] "
                print(f"  {marker} [LEARN-TEST] '{new_word}' -> "
                      f"motor_{pred_a} (target motor_{target}) "
                      f"(delta N{td['N']:+d} E{td['E']:+d} "
                      f"S{td['S']:+d} W{td['W']:+d}, x{conf:.1f})",
                      flush=True)
                transcript.append({
                    "turn": n_turns + 1,
                    "user_word": f"learn {new_word} {target}",
                    "is_learn_command": True,
                    "learned_word": new_word,
                    "target_action": target,
                    "predicted_action": pred_a,
                    "confidence": conf,
                    "delta": td,
                    "bound_correctly": bound_ok,
                    "n_events_run": learn_n_events,
                })
                n_turns += 1
                continue

            n_turns += 1
            result = chat_inference(bridge, line)
            d = result["delta_counts"]
            pred_action = result["predicted_action"]
            pred_word = result["predicted_direction"]
            conf = result["confidence_ratio"]

            in_vocab = line in vocab
            _, word_to_action = _vocab_for_mode(mode)
            expected_action = word_to_action.get(line)
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
    ap.add_argument("--mode",
                    choices=["tier1", "synonym", "synonym12", "synonym16"],
                    default="tier1",
                    help="Architecture mode: "
                         "tier1=4-word (validated 5/6+6/6); "
                         "synonym=8-word (validated 5/6+6/6, 3/3 GO at "
                         "consolidation); "
                         "synonym12=12-word (PARTIAL at default arch; "
                         "REPL uses scaled n_motor=2000 per capacity "
                         "hypothesis); "
                         "synonym16=16-word (master plan extension, "
                         "Unicode arrows up/right/down/left as 4th synonym)")
    ap.add_argument("--seed", type=int, default=43,
                    help="Random seed (43 is the documented best Tier 1 seed; "
                         "42 is best Tier 2.1 single-seed)")
    ap.add_argument("--train-events", type=int, default=None,
                    help="Events per word during training (default: "
                         "200 for tier1, 400 for synonym, 200 for synonym12/16)")
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
    ap.add_argument("--scripted-words", type=str, default=None,
                    help="Comma-separated word list to process instead of "
                         "interactive stdin. Useful for CI / regression / "
                         "batch eval. Example: --scripted-words 'north,up,east,right'")
    ap.add_argument("--learn", action="store_true",
                    help="Enable online vocabulary learning. The REPL will "
                         "recognize 'learn <word> <action>' commands and "
                         "run an embodied-Hebbian binding session that adds "
                         "the new word to the existing motor pool, then "
                         "auto-tests the binding. Default OFF (learning "
                         "during chat is opt-in because new bindings can "
                         "perturb existing ones).")
    ap.add_argument("--learn-events", type=int, default=50,
                    help="Number of paired co-firing events per learn "
                         "command (default 50). Higher values give a "
                         "stronger binding but risk perturbing existing "
                         "vocab on the same motor pool.")
    args = ap.parse_args()

    if args.train_events is None:
        if args.mode == "tier1":
            args.train_events = 200
        elif args.mode == "synonym":
            args.train_events = 400  # Tier 2.1 BREAKTHROUGH validated config
        else:  # synonym12, synonym16
            args.train_events = 200  # Per consolidation_synonym medium

    if args.load_bridge and args.save_bridge:
        ap.error("--load-bridge and --save-bridge are mutually exclusive "
                 "(saving overwrites a checkpoint that was just loaded)")

    scripted_words = None
    if args.scripted_words:
        scripted_words = [w.strip() for w in args.scripted_words.split(",")
                          if w.strip()]
        if not scripted_words:
            ap.error("--scripted-words got an empty list")

    run_repl(
        mode=args.mode,
        seed=args.seed,
        n_train_events=args.train_events,
        transcript_out=args.transcript_out,
        load_bridge=args.load_bridge,
        save_bridge=args.save_bridge,
        scripted_words=scripted_words,
        allow_learn=args.learn,
        learn_n_events=args.learn_events,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
