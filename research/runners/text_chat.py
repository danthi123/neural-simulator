"""Interactive text chat with the trained agent.

Usage:
    python -m research.runners.text_chat [--seed 42] [--n-train-pairs 500]

Trains an agent (via text_train.run_text_training) then drops into a
stdin/stdout REPL where you can:
  - Type a direction word ("north"/"east"/"south"/"west" or alias) →
    the agent's cortex_X firing rate decides which action it would take.
  - Type "show <ax> <ay> <gx> <gy>" → renders a gridworld at those
    coords, drives the retina, reads the language_output to see what
    word the agent emits.
  - Type "quit" / "q" / Ctrl+D to exit.

This is the v1 textual interaction loop. Vocabulary is bounded
(north/east/south/west + simple object words), single-token I/O.
v2 adds multi-token sequences and natural-English bridging via Claude.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np


WORD_TO_ACTION = {
    "north": "N", "n": "N", "up": "N",
    "east": "E", "e": "E", "right": "E",
    "south": "S", "s": "S", "down": "S",
    "west": "W", "w": "W", "left": "W",
}
ACTION_TO_WORD = {"N": "north", "E": "east", "S": "south", "W": "west"}


def repl_loop(bridge, grid_size: int = 8, stim_steps: int = 200,
              drive_pA: float = 200.0):
    """Interactive read-eval-print loop.

    Each user input results in one stim window (~100 ms simulated time)
    followed by a readout. State persists across exchanges (the bridge
    is the same instance), so prior context can influence later responses.
    """
    import cupy as cp
    from sim.visual_cortex import (
        render_gridworld_to_image,
        image_to_retina_drive,
    )

    retina_idx = cp.asarray(
        list(bridge.region_manager.indices("retina")), dtype=cp.int64
    )
    lang_output_idx = cp.asarray(
        list(bridge.region_manager.indices("language_output")), dtype=cp.int64
    )
    cortex_idx = {
        a: cp.asarray(list(bridge.region_manager.indices(f"cortex_{a}")),
                      dtype=cp.int64)
        for a in ["N", "E", "S", "W"]
    }

    print("\n" + "=" * 60)
    print("INTERACTIVE TEXT CHAT (v1)")
    print("=" * 60)
    print("Commands:")
    print("  <word>            - drive language_input with the word, observe action")
    print("                       (e.g. 'north', 'east', 'left'/'right')")
    print("  show AX AY GX GY  - drive retina with that gridworld, agent says word")
    print("                       (e.g. 'show 1 1 6 6')")
    print("  reset             - zero all current input")
    print("  quit / q / Ctrl+D - exit")
    print()

    while True:
        try:
            line = input("you> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n[exit]")
            return

        if not line:
            continue
        if line in ("quit", "q", "exit"):
            print("[exit]")
            return
        if line == "reset":
            bridge.cp_external_input_current[:] = 0.0
            bridge.core_config.current_reward_signal = 0.0
            print("agent> [state reset]")
            continue

        # Parse: 'show ax ay gx gy'
        if line.startswith("show "):
            parts = line.split()
            if len(parts) != 5:
                print("agent> [usage: show AX AY GX GY]")
                continue
            try:
                ax, ay, gx, gy = (int(p) for p in parts[1:5])
            except ValueError:
                print("agent> [coords must be integers]")
                continue
            if not (0 <= ax < grid_size and 0 <= ay < grid_size
                    and 0 <= gx < grid_size and 0 <= gy < grid_size):
                print(f"agent> [coords must be in 0..{grid_size-1}]")
                continue
            img = render_gridworld_to_image(
                agent_pos=(ax, ay), goal_pos=(gx, gy),
                grid_size=grid_size, image_size=32,
            )
            bridge.cp_external_input_current[:] = 0.0
            bridge.cp_external_input_current[retina_idx] = cp.asarray(
                image_to_retina_drive(img, drive_max_pA=drive_pA),
                dtype=cp.float32,
            )
            bridge.core_config.current_reward_signal = 0.0

            spike_counts = cp.zeros(int(lang_output_idx.size), dtype=cp.int32)
            for s in range(stim_steps):
                bridge._run_one_simulation_step()
                bridge.runtime_state.current_time_step += 1
                if s >= 60:  # skip 30ms onset
                    spike_counts += bridge.cp_firing_states[lang_output_idx].astype(cp.int32)
            top3 = bridge.read_language_output(
                spike_counts=spike_counts,
                n_steps=stim_steps - 60,
                top_k=3,
                vocab=["north", "east", "south", "west"],
            )
            print(f"agent> {top3[0]} (other guesses: {', '.join(top3[1:])})")
            continue

        # Otherwise: treat as direction word
        word = line.split()[0]
        if word not in WORD_TO_ACTION:
            print(f"agent> [unknown word '{word}'. Try: north/east/south/west "
                  f"or 'show A B G H']")
            continue

        target_action = WORD_TO_ACTION[word]
        bridge.cp_external_input_current[:] = 0.0
        bridge.set_token_drive(word, drive_pA=drive_pA, sparsity=0.1)
        bridge.core_config.current_reward_signal = 0.0

        spike_counts = {a: 0 for a in ["N", "E", "S", "W"]}
        for s in range(stim_steps):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
            if s >= 60:
                firing = bridge.cp_firing_states
                for a in ["N", "E", "S", "W"]:
                    spike_counts[a] += int(firing[cortex_idx[a]].sum().get())
        predicted = max(spike_counts, key=lambda a: spike_counts[a])
        word_guess = ACTION_TO_WORD[predicted]
        match = "(matches expected)" if predicted == target_action else "(differs from expected!)"
        print(f"agent> would go {word_guess} {match}  cortex_X spikes: {spike_counts}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-image-word", type=int, default=300,
                    help="image->word training pairs (smaller = faster bootstrap)")
    ap.add_argument("--n-word-action", type=int, default=300,
                    help="word->action training pairs")
    ap.add_argument("--grid-size", type=int, default=8)
    ap.add_argument("--skip-training", action="store_true",
                    help="don't train; useful if loading a checkpoint")
    args = ap.parse_args()

    from research.runners.text_train import run_text_training

    print("=" * 60)
    if args.skip_training:
        print("Building bridge WITHOUT training (skip-training=True)")
    else:
        print(f"TRAINING: seed={args.seed}, "
              f"{args.n_image_word} I->W + {args.n_word_action} W->A pairs")
    print("=" * 60)

    bridge, _ = run_text_training(
        seed=args.seed,
        n_image_word_pairs=0 if args.skip_training else args.n_image_word,
        n_word_action_pairs=0 if args.skip_training else args.n_word_action,
        grid_size=args.grid_size,
        verbose=True,
    )

    repl_loop(bridge, grid_size=args.grid_size)


if __name__ == "__main__":
    main()
