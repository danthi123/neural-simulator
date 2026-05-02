"""Re-evaluate a previously-trained text-IO bridge from a checkpoint.

Loads a bridge from <result>.simstate.h5 and runs the text I/O evals
without re-training. Useful for testing different eval methodologies
(interleaved vs block ordering, longer reset windows, more trials,
different drive strengths) on the SAME trained network.

Usage:
  # Re-eval with new methodology (interleaved, n=100, n=25)
  python -m research.runners.text_reeval \\
      research/findings/raw/g11_bg/text_eval_R3_R6_100ep_partialT1.simstate.h5 \\
      --out-stats research/findings/raw/g11_bg/text_reeval_R3R6_100ep_v2.json

  # Re-eval with legacy block ordering for direct comparison
  python -m research.runners.text_reeval ckpt.h5 --legacy-block-eval \\
      --out-stats reeval_legacy.json

  # Re-eval with stronger drive
  python -m research.runners.text_reeval ckpt.h5 \\
      --drive-pA 400 --out-stats reeval_drive400.json

The checkpoint must have been saved by text_eval_embodied.py with
--save-checkpoint (default ON as of 2026-05-02).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.runners.text_eval import (
    evaluate_image_to_word,
    evaluate_word_to_action,
    evaluate_word_to_action_LEGACY_BLOCK,
)


def load_bridge(checkpoint_path: str):
    """Reconstruct a SimulationBridge from a checkpoint.

    Replays the same builder used at training time (build_bg_brain_regions
    with text+visual cortex enabled), then loads the saved weights/state
    over it.
    """
    import cupy as cp  # noqa: F401  (load_checkpoint expects CuPy initialized)

    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.bridge import SimulationBridge
    from research.runners.g11_bg_runner import build_bg_brain_regions

    regions, pathways = build_bg_brain_regions(
        enable_striatal_fsis=True,
        enable_cluster_a_closed_loop=True,
        enable_cluster_e_topography=True,
        enable_pfc=True,
        pfc_enable_nmda=True,
        enable_visual_cortex=True,
        enable_text_io=True,
    )

    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = list(regions)
    cfg.region_pathways = list(pathways)
    cfg.dt_ms = 0.5
    cfg.enable_nmda = True
    cfg.nmda_ratio = 0.5
    cfg.enable_structural_plasticity = False
    cfg.enable_per_type_stp = False

    bridge = SimulationBridge(
        core_config=cfg,
        viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(),
        gpu_config=GPUConfig(),
    )
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge._initialize_simulation_data(called_from_playback_init=False)

    bridge.load_checkpoint(checkpoint_path)
    return bridge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=str, help="path to .simstate.h5")
    ap.add_argument("--n-eval-image-word", type=int, default=100)
    ap.add_argument("--n-eval-word-action", type=int, default=25)
    ap.add_argument("--grid-size", type=int, default=8)
    ap.add_argument("--drive-pA", type=float, default=200.0,
                    help="language_input drive at eval time (default 200)")
    ap.add_argument("--n-reset-steps", type=int, default=100,
                    help="inter-trial reset (default 100 = 50ms = 1 NMDA tau)")
    ap.add_argument("--legacy-block-eval", action="store_true",
                    help="use legacy block-ordered W->A eval for backward "
                    "comparison to pre-2026-05-02 results")
    ap.add_argument("--out-stats", type=str, default=None)
    ap.add_argument("--seed", type=int, default=1, help="eval-side rng seed")
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        ap.error(f"checkpoint not found: {ckpt}")

    print("=" * 60)
    print(f"RE-EVAL from checkpoint: {ckpt}")
    print(f"  drive_pA={args.drive_pA} n_reset={args.n_reset_steps} "
          f"interleave={'NO' if args.legacy_block_eval else 'YES'}")
    print("=" * 60)

    bridge = load_bridge(str(ckpt))

    print(f"\nEVAL: image -> word ({args.n_eval_image_word} fresh trials)")
    iw_result = evaluate_image_to_word(
        bridge, n_trials=args.n_eval_image_word, grid_size=args.grid_size,
        drive_pA=args.drive_pA, seed=args.seed,
    )
    print(f"  Accuracy: {iw_result['correct']}/{iw_result['n_trials']} "
          f"= {iw_result['accuracy']:.1%}")

    print(f"\nEVAL: word -> action ({args.n_eval_word_action} per word)")
    eval_func = (
        evaluate_word_to_action_LEGACY_BLOCK
        if args.legacy_block_eval
        else evaluate_word_to_action
    )
    wa_kwargs = {
        "n_trials_per_word": args.n_eval_word_action,
        "drive_pA": args.drive_pA,
    }
    if not args.legacy_block_eval:
        wa_kwargs["n_reset_steps"] = args.n_reset_steps
        wa_kwargs["seed"] = args.seed
    wa_result = eval_func(bridge, **wa_kwargs)
    print(f"  Accuracy: {wa_result['correct']}/{wa_result['n_trials']} "
          f"= {wa_result['accuracy']:.1%}")

    if args.out_stats:
        out = {
            "regime": "embodied_reeval",
            "checkpoint": str(ckpt),
            "image_to_word_eval": iw_result,
            "word_to_action_eval": wa_result,
            "eval_config": {
                "drive_pA": args.drive_pA,
                "n_reset_steps": args.n_reset_steps,
                "interleave_words": not args.legacy_block_eval,
                "n_eval_image_word": args.n_eval_image_word,
                "n_eval_word_action": args.n_eval_word_action,
                "seed": args.seed,
            },
        }
        Path(args.out_stats).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_stats).write_text(json.dumps(out, indent=2, default=str))
        print(f"\n  Saved: {args.out_stats}")


if __name__ == "__main__":
    main()
