"""Text evaluation runner — given a trained bridge, tests:
1. Image -> word: present a fresh gridworld image, read language_output,
   does the agent emit the correct cardinal direction?
2. Word -> action: drive language_input with a word, observe motor
   firing, does the agent take the correct action?

Reuses the bridge built by text_train.py (or loads a checkpoint).
The same training-time supervision regime is used, but WITHOUT clamping
the supervisor signal — we observe the agent's natural response.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ACTION_NAMES = ["N", "E", "S", "W"]
WORD_TO_ACTION = {"north": "N", "east": "E", "south": "S", "west": "W"}


def _direction_from_positions(agent_pos, goal_pos) -> str:
    ax, ay = agent_pos
    gx, gy = goal_pos
    dx, dy = gx - ax, gy - ay
    if abs(dx) >= abs(dy):
        return "east" if dx > 0 else ("west" if dx < 0 else "north")
    else:
        return "north" if dy > 0 else ("south" if dy < 0 else "east")


def evaluate_image_to_word(
    bridge,
    n_trials: int = 100,
    grid_size: int = 8,
    stim_steps_per_trial: int = 200,
    drive_pA: float = 200.0,
    seed: int = 1,
    verbose: bool = True,
):
    """Present fresh gridworld images; check if agent emits correct
    cardinal direction in language_output.

    Returns dict with accuracy, per-class breakdown, and confusion matrix.
    """
    import cupy as cp
    from sim.visual_cortex import (
        render_gridworld_to_image,
        image_to_retina_drive,
    )

    rng = np.random.default_rng(seed)

    retina_idx = cp.asarray(
        list(bridge.region_manager.indices("retina")), dtype=cp.int64
    )
    lang_output_idx = cp.asarray(
        list(bridge.region_manager.indices("language_output")), dtype=cp.int64
    )

    correct = 0
    confusion = {w: {w2: 0 for w2 in ["north", "east", "south", "west"]}
                 for w in ["north", "east", "south", "west"]}
    n_reset_steps = 100  # match training inter-trial reset

    # Pre-cache embedding vectors for the 4 cardinal direction tokens
    # so we can do baseline-subtracted nearest-token decoding.
    from sim.text_embeddings import embed
    DIRECTIONS = ["north", "east", "south", "west"]
    target_embeddings = np.stack([embed(t, dim=int(lang_output_idx.size))
                                    for t in DIRECTIONS])  # (4, n_lang_output)

    for trial in range(n_trials):
        # Random fresh image
        while True:
            ax, ay = rng.integers(0, grid_size, size=2)
            gx, gy = rng.integers(0, grid_size, size=2)
            if (ax, ay) != (gx, gy):
                break
        target_word = _direction_from_positions((ax, ay), (gx, gy))

        # ─── Phase A: BASELINE language_output activity (no input) ───
        bridge.cp_external_input_current[:] = 0.0
        bridge.core_config.current_reward_signal = 0.0
        for _ in range(n_reset_steps):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
        baseline_spikes = cp.zeros(int(lang_output_idx.size), dtype=cp.int32)
        for s in range(stim_steps_per_trial):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
            if s >= 60:
                baseline_spikes += bridge.cp_firing_states[lang_output_idx].astype(cp.int32)

        # ─── Phase B: image-driven measurement ───
        bridge.cp_external_input_current[:] = 0.0
        bridge.core_config.current_reward_signal = 0.0
        for _ in range(n_reset_steps):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1

        img = render_gridworld_to_image(
            agent_pos=(int(ax), int(ay)), goal_pos=(int(gx), int(gy)),
            grid_size=grid_size, image_size=32,
        )
        bridge.cp_external_input_current[retina_idx] = cp.asarray(
            image_to_retina_drive(img, drive_max_pA=drive_pA),
            dtype=cp.float32,
        )
        # NO supervisor clamp on language_output — we want the agent's
        # natural response.
        bridge.core_config.current_reward_signal = 0.0  # no reward at eval

        # Tally language_output spikes
        spike_counts = cp.zeros(int(lang_output_idx.size), dtype=cp.int32)
        for s in range(stim_steps_per_trial):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
            # Skip first 30ms (~60 sub-steps) onset, count rest
            if s >= 60:
                firing = bridge.cp_firing_states[lang_output_idx]
                spike_counts += firing.astype(cp.int32)

        # ─── Delta decoding: subtract baseline spikes, then cosine-match
        # to direction embeddings. Baseline-subtraction reveals the
        # image-driven RESPONSE rather than absolute activity. ───
        delta_spikes = (spike_counts - baseline_spikes).get().astype(np.float32)
        # Cosine similarity to each direction embedding
        sims = []
        for emb in target_embeddings:
            denom = np.linalg.norm(delta_spikes) * np.linalg.norm(emb)
            if denom < 1e-8:
                sims.append(0.0)
            else:
                sims.append(float(np.dot(delta_spikes, emb) / denom))
        predicted = DIRECTIONS[int(np.argmax(sims))]

        is_correct = predicted == target_word
        if is_correct:
            correct += 1
        confusion[target_word][predicted] += 1

        if verbose and (trial + 1) % 25 == 0:
            tag = "OK" if is_correct else "WRONG"
            print(f"  [eval I->W] {trial+1}/{n_trials}  target={target_word} "
                  f"got={predicted} {tag} "
                  f"acc-so-far={correct}/{trial+1}={100*correct/(trial+1):.1f}%",
                  flush=True)

    accuracy = correct / max(n_trials, 1)
    return {
        "n_trials": n_trials,
        "correct": correct,
        "accuracy": accuracy,
        "confusion_matrix": confusion,
    }


def evaluate_word_to_action(
    bridge,
    n_trials_per_word: int = 25,
    stim_steps_per_trial: int = 200,
    drive_pA: float = 200.0,
    verbose: bool = True,
):
    """Drive language_input with each direction word; observe which
    cortex_X has the highest firing rate. Did the agent learn the
    word-action mapping?"""
    import cupy as cp

    cortex_idx = {
        a: cp.asarray(list(bridge.region_manager.indices(f"cortex_{a}")),
                      dtype=cp.int64)
        for a in ACTION_NAMES
    }

    correct = 0
    total = 0
    confusion = {w: {a: 0 for a in ACTION_NAMES}
                 for w in ["north", "east", "south", "west"]}

    n_reset_steps = 100  # match training inter-trial reset
    for word in ["north", "east", "south", "west"]:
        target_action = WORD_TO_ACTION[word]
        for trial in range(n_trials_per_word):
            # ─── Phase A: BASELINE measurement ───
            # Reset, then run with NO input. Measure spontaneous cortex_X.
            # This subtracts cascade default bias (cortex_N 2x higher etc.)
            # so we measure the DELTA caused by language_input.
            bridge.cp_external_input_current[:] = 0.0
            bridge.core_config.current_reward_signal = 0.0
            for _ in range(n_reset_steps):
                bridge._run_one_simulation_step()
                bridge.runtime_state.current_time_step += 1
            baseline_counts = {a: 0 for a in ACTION_NAMES}
            for s in range(stim_steps_per_trial):
                bridge._run_one_simulation_step()
                bridge.runtime_state.current_time_step += 1
                if s >= 60:
                    firing = bridge.cp_firing_states
                    for a in ACTION_NAMES:
                        baseline_counts[a] += int(firing[cortex_idx[a]].sum().get())

            # ─── Phase B: LANGUAGE-DRIVEN measurement ───
            # Reset, drive language_input, measure cortex_X again
            bridge.cp_external_input_current[:] = 0.0
            bridge.core_config.current_reward_signal = 0.0
            for _ in range(n_reset_steps):
                bridge._run_one_simulation_step()
                bridge.runtime_state.current_time_step += 1
            bridge.set_token_drive(word, drive_pA=drive_pA, sparsity=0.1)

            spike_counts = {a: 0 for a in ACTION_NAMES}
            for s in range(stim_steps_per_trial):
                bridge._run_one_simulation_step()
                bridge.runtime_state.current_time_step += 1
                if s >= 60:
                    firing = bridge.cp_firing_states
                    for a in ACTION_NAMES:
                        spike_counts[a] += int(firing[cortex_idx[a]].sum().get())

            # ─── DELTA selection: which cortex_X had the largest INCREASE
            # over its own baseline? This subtracts the cascade's structural
            # N-bias and reveals language-driven differential response. ───
            delta_counts = {a: spike_counts[a] - baseline_counts[a]
                            for a in ACTION_NAMES}
            predicted = max(delta_counts, key=lambda a: delta_counts[a])
            confusion[word][predicted] += 1
            if predicted == target_action:
                correct += 1
            total += 1

        if verbose:
            print(f"  [eval W->A] word={word} target={target_action} "
                  f"baseline={baseline_counts} drive={spike_counts} "
                  f"delta={delta_counts}", flush=True)

    accuracy = correct / max(total, 1)
    return {
        "n_trials": total,
        "correct": correct,
        "accuracy": accuracy,
        "confusion_matrix": confusion,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-image-word", type=int, default=500)
    ap.add_argument("--n-word-action", type=int, default=500)
    ap.add_argument("--n-eval-image-word", type=int, default=100)
    ap.add_argument("--n-eval-word-action", type=int, default=25)
    ap.add_argument("--grid-size", type=int, default=8)
    ap.add_argument("--out-stats", type=str, default=None)
    args = ap.parse_args()

    from research.runners.text_train import run_text_training

    # Train
    print("=" * 60)
    print(f"TRAINING (seed={args.seed}, "
          f"{args.n_image_word} I->W + {args.n_word_action} W->A pairs)")
    print("=" * 60)
    bridge, train_stats = run_text_training(
        seed=args.seed,
        n_image_word_pairs=args.n_image_word,
        n_word_action_pairs=args.n_word_action,
        grid_size=args.grid_size,
        verbose=True,
    )

    # Evaluate
    print("\n" + "=" * 60)
    print(f"EVAL: image -> word ({args.n_eval_image_word} fresh trials)")
    print("=" * 60)
    iw_result = evaluate_image_to_word(
        bridge, n_trials=args.n_eval_image_word, grid_size=args.grid_size,
    )
    print(f"\n  Accuracy: {iw_result['correct']}/{iw_result['n_trials']} "
          f"= {iw_result['accuracy']:.1%}")
    print(f"  Confusion: {iw_result['confusion_matrix']}")

    print("\n" + "=" * 60)
    print(f"EVAL: word -> action ({args.n_eval_word_action} per word)")
    print("=" * 60)
    wa_result = evaluate_word_to_action(
        bridge, n_trials_per_word=args.n_eval_word_action,
    )
    print(f"\n  Accuracy: {wa_result['correct']}/{wa_result['n_trials']} "
          f"= {wa_result['accuracy']:.1%}")
    print(f"  Confusion: {wa_result['confusion_matrix']}")

    if args.out_stats:
        out = {
            "seed": args.seed,
            "n_image_word_train": args.n_image_word,
            "n_word_action_train": args.n_word_action,
            "image_to_word_eval": iw_result,
            "word_to_action_eval": wa_result,
            "training_stats": train_stats,
        }
        Path(args.out_stats).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_stats).write_text(json.dumps(out, indent=2, default=str))
        print(f"\n  Saved: {args.out_stats}")


if __name__ == "__main__":
    main()
