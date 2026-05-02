"""Eval after embodied text training.

After training via gridworld navigation (text_train_embodied.py), test:
1. Image -> word: does the agent emit the correct cardinal direction?
2. Word -> action: does the agent take the correct action when given a word?
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.runners.text_train_embodied import run_embodied_text_training
from research.runners.text_eval import evaluate_image_to_word, evaluate_word_to_action


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-episodes", type=int, default=50)
    ap.add_argument("--steps-per-episode", type=int, default=30)
    # 2026-05-02: Defaults bumped from 40/10 to 100/25 after analyzer
    # diagnosis revealed 13/40 = 32.5% has p=0.18 vs chance (NOT
    # significant). At n=100 + 100 (25 per word x 4 words), 35%+ is
    # p<0.05 — actually distinguishable from chance.
    ap.add_argument("--n-eval-image-word", type=int, default=100)
    ap.add_argument("--n-eval-word-action", type=int, default=25)
    ap.add_argument("--grid-size", type=int, default=8)
    ap.add_argument("--out-stats", type=str, default=None)
    # Overrides for embodied training drives (R3+R6 defaults: 200/200/150)
    ap.add_argument("--retina-drive-pA", type=float, default=200.0)
    ap.add_argument("--lang-input-drive-pA", type=float, default=200.0)
    ap.add_argument("--lang-output-coactive-pA", type=float, default=150.0)
    args = ap.parse_args()

    print("=" * 60)
    print(f"EMBODIED TRAINING (seed={args.seed}, "
          f"{args.n_episodes} ep x {args.steps_per_episode} steps)")
    print(f"  drives: retina={args.retina_drive_pA} "
          f"lang_in={args.lang_input_drive_pA} "
          f"lang_out_coact={args.lang_output_coactive_pA} pA")
    print("=" * 60)
    bridge, train_stats = run_embodied_text_training(
        seed=args.seed,
        n_episodes=args.n_episodes,
        steps_per_episode=args.steps_per_episode,
        grid_size=args.grid_size,
        retina_drive_pA=args.retina_drive_pA,
        lang_input_drive_pA=args.lang_input_drive_pA,
        lang_output_coactive_pA=args.lang_output_coactive_pA,
        verbose=True,
    )

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
            "regime": "embodied",
            "seed": args.seed,
            "n_episodes": args.n_episodes,
            "steps_per_episode": args.steps_per_episode,
            "training_stats": train_stats,
            "image_to_word_eval": iw_result,
            "word_to_action_eval": wa_result,
        }
        Path(args.out_stats).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_stats).write_text(json.dumps(out, indent=2, default=str))
        print(f"\n  Saved: {args.out_stats}")


if __name__ == "__main__":
    main()
