"""Eval after contrastive text training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.runners.text_train_contrastive import run_contrastive_training
from research.runners.text_eval import evaluate_image_to_word, evaluate_word_to_action


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-image-word", type=int, default=200)
    ap.add_argument("--n-word-action", type=int, default=200)
    ap.add_argument("--n-eval-image-word", type=int, default=40)
    ap.add_argument("--n-eval-word-action", type=int, default=10)
    ap.add_argument("--grid-size", type=int, default=8)
    ap.add_argument("--out-stats", type=str, default=None)
    args = ap.parse_args()

    print("=" * 60)
    print(f"CONTRASTIVE TRAINING (seed={args.seed})")
    print("=" * 60)
    bridge, train_stats = run_contrastive_training(
        seed=args.seed,
        n_image_word_pairs=args.n_image_word,
        n_word_action_pairs=args.n_word_action,
        grid_size=args.grid_size,
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
            "regime": "contrastive",
            "seed": args.seed,
            "training_stats": train_stats,
            "image_to_word_eval": iw_result,
            "word_to_action_eval": wa_result,
        }
        Path(args.out_stats).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_stats).write_text(json.dumps(out, indent=2, default=str))
        print(f"\n  Saved: {args.out_stats}")


if __name__ == "__main__":
    main()
