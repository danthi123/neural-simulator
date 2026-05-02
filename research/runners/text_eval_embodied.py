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
    # Stim window + reset overrides (validated config: stim=100, reset=100;
    # the 2026-05-01 regression doc shows reset=50 breaks language pathway
    # at scale via NMDA bleedover; do NOT use reset < 100 for >100 ep runs)
    ap.add_argument("--stim-steps-per-step", type=int, default=100,
                    help="sub-steps per env step during stim window (default 100 = 50ms)")
    ap.add_argument("--reset-steps", type=int, default=100,
                    help="inter-step reset sub-steps (default 100 = 50ms = 1 NMDA tau)")
    ap.add_argument("--enable-per-type-stp", action="store_true",
                    help="enable per-connection-type STP (Tier 1.5 revert; "
                    "default off for speed, but the 2026-05-02 partial-T1 "
                    "regression suggests it may matter for language pathway)")
    # Reward shaping (2026-05-02): adjust LTP/LTD asymmetry
    ap.add_argument("--correct-move-reward", type=float, default=1.0,
                    help="reward for moves that reduce Manhattan distance (default +1.0)")
    ap.add_argument("--wrong-move-reward", type=float, default=-0.5,
                    help="reward for moves that increase Manhattan distance (default -0.5; "
                    "set to 0 to eliminate negative LTP/LTD asymmetry that may cause "
                    "directional learning reversal as observed for 'south' in PID 39408)")
    # Eval-time drives (2026-05-02): the v2 reeval sweep showed that
    # increasing language_input drive at eval time from 200 to 500 pA
    # surfaced W->A signal: 32% at d500/r100 vs 27% at default d200.
    # Stronger drive helps language signal overcome cascade structural
    # noise during eval. These default to the original 200 pA for
    # backwards compatibility but can be raised for better readout.
    ap.add_argument("--eval-iw-drive-pA", type=float, default=200.0,
                    help="retina drive at I->W eval (default 200)")
    ap.add_argument("--eval-wa-drive-pA", type=float, default=200.0,
                    help="language_input drive at W->A eval (default 200; "
                    "v2 reeval sweep showed 500 surfaces signal hidden at 200)")
    # Architecture sizing (2026-05-02): bigger motor pools for more
    # discriminability per direction. 6-seed v2 result of 28.5% W→A
    # may be limited by 10-neuron pools' high variance.
    ap.add_argument("--n-motor-per-action", type=int, default=10,
                    help="motor neurons per direction (default 10; try 30 "
                    "for ~3x more spike-count discriminability)")
    ap.add_argument("--text-n-input-neurons", type=int, default=256,
                    help="language_input region size (default 256; try 512)")
    ap.add_argument("--text-n-output-neurons", type=int, default=256,
                    help="language_output region size (default 256; try 512)")
    # Auto-checkpoint after training so we can re-eval same bridge later
    # with different methodologies (e.g., compare interleaved vs block eval).
    ap.add_argument("--save-checkpoint", action="store_true",
                    help="save bridge state after training to <out-stats>.h5 "
                    "for later re-evaluation with text_reeval.py")
    ap.add_argument("--no-save-checkpoint", dest="save_checkpoint",
                    action="store_false")
    ap.set_defaults(save_checkpoint=True)
    args = ap.parse_args()

    print("=" * 60)
    print(f"EMBODIED TRAINING (seed={args.seed}, "
          f"{args.n_episodes} ep x {args.steps_per_episode} steps)")
    print(f"  drives: retina={args.retina_drive_pA} "
          f"lang_in={args.lang_input_drive_pA} "
          f"lang_out_coact={args.lang_output_coactive_pA} pA")
    print(f"  stim_steps={args.stim_steps_per_step} "
          f"reset_steps={args.reset_steps}")
    print("=" * 60)
    bridge, train_stats = run_embodied_text_training(
        seed=args.seed,
        n_episodes=args.n_episodes,
        steps_per_episode=args.steps_per_episode,
        grid_size=args.grid_size,
        stim_steps_per_step=args.stim_steps_per_step,
        reset_steps=args.reset_steps,
        enable_per_type_stp=args.enable_per_type_stp,
        retina_drive_pA=args.retina_drive_pA,
        lang_input_drive_pA=args.lang_input_drive_pA,
        lang_output_coactive_pA=args.lang_output_coactive_pA,
        correct_move_reward=args.correct_move_reward,
        wrong_move_reward=args.wrong_move_reward,
        n_motor_per_action=args.n_motor_per_action,
        text_n_input_neurons=args.text_n_input_neurons,
        text_n_output_neurons=args.text_n_output_neurons,
        verbose=True,
    )

    print("\n" + "=" * 60)
    print(f"EVAL: image -> word ({args.n_eval_image_word} fresh trials)")
    print("=" * 60)
    iw_result = evaluate_image_to_word(
        bridge, n_trials=args.n_eval_image_word, grid_size=args.grid_size,
        drive_pA=args.eval_iw_drive_pA,
    )
    print(f"\n  Accuracy: {iw_result['correct']}/{iw_result['n_trials']} "
          f"= {iw_result['accuracy']:.1%}")
    print(f"  Confusion: {iw_result['confusion_matrix']}")

    print("\n" + "=" * 60)
    print(f"EVAL: word -> action ({args.n_eval_word_action} per word)")
    print("=" * 60)
    wa_result = evaluate_word_to_action(
        bridge, n_trials_per_word=args.n_eval_word_action,
        drive_pA=args.eval_wa_drive_pA,
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
            "config": {
                "retina_drive_pA": args.retina_drive_pA,
                "lang_input_drive_pA": args.lang_input_drive_pA,
                "lang_output_coactive_pA": args.lang_output_coactive_pA,
                "stim_steps_per_step": args.stim_steps_per_step,
                "reset_steps": args.reset_steps,
            },
        }
        Path(args.out_stats).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_stats).write_text(json.dumps(out, indent=2, default=str))
        print(f"\n  Saved: {args.out_stats}")

        if args.save_checkpoint:
            ckpt_path = Path(args.out_stats).with_suffix(".simstate.h5")
            try:
                bridge.save_checkpoint(str(ckpt_path))
                print(f"  Saved checkpoint: {ckpt_path}")
            except Exception as e:
                print(f"  WARNING: checkpoint save failed: {e}")


if __name__ == "__main__":
    main()
