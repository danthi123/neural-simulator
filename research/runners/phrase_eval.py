"""Phase 1.2 Tier 2.3 phrase eval suite.

Tests three conditions on a trained Tier 2.3 architecture:

1. PHRASE: drive 'go' (100ms) -> drive direction (100ms) ->
   motor pool prediction. Pass if argmax matches direction's action.
   Tests compositional binding.

2. DIRECTION-ONLY (Tier 1 compat): drive direction alone (200ms) ->
   motor pool prediction. Pass if argmax matches. Tests that
   Tier 1 binding is preserved during Tier 2.3 training.

3. VERB-ONLY (anti-action): drive 'go' alone (200ms) -> motor pool
   activity. Pass if motor pools STAY QUIET (max activity below
   chance threshold). Tests that 'go' alone doesn't trigger
   action without context.

Each condition uses 25 trials per direction (or 25 verb trials for
condition 3). Pass criteria match Tier 2.3 design:
  - Phrase >= 4/6 seeds correctly execute
  - Direction-only >= 4/6 seeds preserve Tier 1
  - Verb-only >= 4/6 seeds keep motor quiet

Reuses evaluate_word_to_action's reset/stim/readout machinery.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


DIRECTIONS = ["north", "east", "south", "west"]
DIRECTION_TO_ACTION = {"north": "N", "east": "E", "south": "S", "west": "W"}
ACTIONS = ["N", "E", "S", "W"]


def evaluate_phrase(
    bridge,
    n_trials_per_direction: int = 25,
    verb_drive_ms: int = 100,
    direction_drive_ms: int = 100,
    final_observe_ms: int = 50,
    reset_ms: int = 50,
    drive_pA: float = 200.0,
    sparsity: float = 0.1,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Evaluate phrase condition: 'go' + direction -> motor.

    Returns:
        {
            "accuracy": float in [0, 1],
            "n_correct": int,
            "n_total": int,
            "per_direction": {direction: accuracy},
            "confusion": {direction: {action: count}},
            "pass": bool (acc >= 0.5),
        }
    """
    import cupy as cp
    from sim.text_embeddings import vocab_to_drive_pattern

    rm = bridge.region_manager
    lang_input_idx = list(rm.indices("language_input"))
    motor_idx = {a: list(rm.indices(f"motor_{a}")) for a in ACTIONS}
    n_lang_in = len(lang_input_idx)
    lang_input_arr = cp.asarray(lang_input_idx, dtype=cp.int64)
    motor_arr = {a: cp.asarray(motor_idx[a], dtype=cp.int64)
                 for a in ACTIONS}

    def _drive_for(word: str):
        d = vocab_to_drive_pattern(word, n_neurons=n_lang_in,
                                    drive_max_pA=drive_pA, sparsity=sparsity)
        return cp.asarray(d, dtype=cp.float32)

    confusion = {d: {a: 0 for a in ACTIONS} for d in DIRECTIONS}
    n_correct = 0
    n_total = 0

    for direction in DIRECTIONS:
        target_action = DIRECTION_TO_ACTION[direction]
        for trial_idx in range(n_trials_per_direction):
            # Reset
            bridge.cp_external_input_current[:] = 0.0
            for _ in range(reset_ms):
                bridge._run_one_simulation_step()
                bridge.runtime_state.current_time_step += 1
            # Stage 1: drive verb only
            verb_drive = _drive_for("go")
            bridge.cp_external_input_current[lang_input_arr] = verb_drive
            for _ in range(verb_drive_ms):
                bridge._run_one_simulation_step()
                bridge.runtime_state.current_time_step += 1
            # Stage 2: drive direction; PFC bistability holds verb context.
            # Count motor pool spikes for the response. Only count during
            # the direction-drive window -- Stage 3 spikes are decay noise
            # that would dilute the per-direction signal.
            dir_drive = _drive_for(direction)
            bridge.cp_external_input_current[:] = 0.0
            bridge.cp_external_input_current[lang_input_arr] = dir_drive
            motor_spike_counts = cp.zeros(4, dtype=cp.int32)
            for _ in range(direction_drive_ms):
                bridge._run_one_simulation_step()
                bridge.runtime_state.current_time_step += 1
                fired = bridge.cp_firing_states
                for a_i, a in enumerate(ACTIONS):
                    motor_spike_counts[a_i] += fired[motor_arr[a]].sum()
            # Stage 3: settle (no input, no spike counting). Allows PFC
            # NMDA bistability to decay before next trial without
            # contaminating motor spike counts.
            bridge.cp_external_input_current[:] = 0.0
            for _ in range(final_observe_ms):
                bridge._run_one_simulation_step()
                bridge.runtime_state.current_time_step += 1
            counts = motor_spike_counts.get()
            predicted = ACTIONS[int(np.argmax(counts))]
            confusion[direction][predicted] += 1
            n_total += 1
            if predicted == target_action:
                n_correct += 1

    accuracy = n_correct / max(n_total, 1)
    per_direction = {
        d: confusion[d][DIRECTION_TO_ACTION[d]] / n_trials_per_direction
        for d in DIRECTIONS
    }
    return {
        "name": "phrase",
        "accuracy": accuracy,
        "n_correct": n_correct,
        "n_total": n_total,
        "per_direction": per_direction,
        "confusion": confusion,
        "pass": accuracy >= 0.5,  # 2x chance threshold
    }


def evaluate_direction_only(bridge, n_trials_per_direction: int = 25,
                              verbose: bool = True) -> Dict[str, Any]:
    """Reuses Tier 1 evaluate_word_to_action -- direction word alone -> motor.
    This tests that Tier 1 backward compat is preserved during Tier 2.3
    training.
    """
    from research.runners.text_eval import evaluate_word_to_action
    wa = evaluate_word_to_action(
        bridge, n_trials_per_word=n_trials_per_direction,
        stim_steps_per_trial=200, n_reset_steps=50, token_sparsity=0.1,
        verbose=verbose,
    )
    return {
        "name": "direction_only",
        "accuracy": wa["accuracy"],
        "confusion": wa.get("confusion_matrix", {}),
        "pass": wa["accuracy"] >= 0.30,  # Tier 1 baseline ~33-45%
    }


def evaluate_verb_only(
    bridge,
    n_trials: int = 25,
    drive_ms: int = 200,
    reset_ms: int = 50,
    drive_pA: float = 200.0,
    sparsity: float = 0.1,
    quiet_threshold_hz: float = 5.0,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Drive 'go' alone, measure max motor pool firing rate.
    Pass if motor pools stay BELOW quiet_threshold_hz.

    Biological rationale: 'go' alone shouldn't trigger any action.
    PFC verb context exists, but the action_gate neuromodulator
    operates on excitability_drive -- it makes motor pools MORE
    responsive to direction input but doesn't itself trigger firing
    above threshold.
    """
    import cupy as cp
    from sim.text_embeddings import vocab_to_drive_pattern

    rm = bridge.region_manager
    lang_input_idx = list(rm.indices("language_input"))
    motor_idx = {a: list(rm.indices(f"motor_{a}")) for a in ACTIONS}
    n_lang_in = len(lang_input_idx)
    lang_input_arr = cp.asarray(lang_input_idx, dtype=cp.int64)
    motor_arr = {a: cp.asarray(motor_idx[a], dtype=cp.int64)
                 for a in ACTIONS}

    def _drive_for(word: str):
        d = vocab_to_drive_pattern(word, n_neurons=n_lang_in,
                                    drive_max_pA=drive_pA, sparsity=sparsity)
        return cp.asarray(d, dtype=cp.float32)

    max_motor_rates_per_trial = []  # per-trial peak motor pool firing Hz
    for trial_idx in range(n_trials):
        # Reset
        bridge.cp_external_input_current[:] = 0.0
        for _ in range(reset_ms):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
        # Drive 'go' alone
        verb_drive = _drive_for("go")
        bridge.cp_external_input_current[lang_input_arr] = verb_drive
        motor_spike_counts = cp.zeros(4, dtype=cp.int32)
        for _ in range(drive_ms):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
            fired = bridge.cp_firing_states
            for a_i, a in enumerate(ACTIONS):
                motor_spike_counts[a_i] += fired[motor_arr[a]].sum()
        counts = motor_spike_counts.get()
        # Convert to firing rate Hz (counts / pool_size / drive_seconds)
        pool_sizes = {a: len(motor_idx[a]) for a in ACTIONS}
        drive_sec = drive_ms / 1000.0
        rates_hz = [counts[i] / pool_sizes[a] / drive_sec
                    for i, a in enumerate(ACTIONS)]
        max_motor_rates_per_trial.append(float(max(rates_hz)))

    mean_max_rate = float(np.mean(max_motor_rates_per_trial))
    pct_quiet = float(np.mean(
        [r < quiet_threshold_hz for r in max_motor_rates_per_trial]
    ))
    return {
        "name": "verb_only",
        "mean_max_motor_rate_hz": mean_max_rate,
        "pct_trials_below_threshold": pct_quiet,
        "quiet_threshold_hz": quiet_threshold_hz,
        "pass": pct_quiet >= 0.8,  # 80%+ trials should be quiet
    }


def main():
    """Standalone eval: load checkpoint, run all 3 conditions."""
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=str, required=True,
                    help="Path to .simstate.h5 from phrase_trainer")
    ap.add_argument("--n-trials-per-direction", type=int, default=25)
    ap.add_argument("--n-verb-only-trials", type=int, default=25)
    ap.add_argument("--out-stats", type=str, default=None)
    args = ap.parse_args()

    raise NotImplementedError(
        "Standalone phrase_eval CLI requires the same architecture-build "
        "machinery as phrase_trainer (so the bridge has all regions + "
        "neuromodulators when checkpoint is loaded). Recommend calling "
        "evaluate_phrase / evaluate_direction_only / evaluate_verb_only "
        "directly from phrase_trainer's run() after training, before "
        "freezing plasticity. CLI standalone mode TBD; for now use "
        "as a library."
    )


if __name__ == "__main__":
    main()
