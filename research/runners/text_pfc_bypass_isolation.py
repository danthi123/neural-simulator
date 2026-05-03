"""
PFC bypass isolation experiment (H4) — what's the upper bound for the
language → motor pathway WITHOUT cascade interference?

Background:
v2 baseline gives W→A 28.5% (n=6, p=0.027). All 9+ architectural
variations tested through 2026-05-02 hit a ceiling at this number.
SWR replay (2026-05-03) regressed it to ~22% across 3 seeds (and
4 expected by morning).

The hypothesis tested here: maybe the 28.5% W→A ceiling reflects a
fundamental architecture limit, not a tuning issue. The language→motor
direct pathway (Wernicke→arcuate→Broca→M1 analog, language_input →
motor_X via PFC bypass) is what the eval reads. If we train it in
ISOLATION — without the cascade firing, without retina input, without
the BG selecting actions — what's the maximum accuracy it can reach?

If H4 isolation gives 80%+: the cascade is interfering during full
training, and the architectural fix is to train the bypass first
(reverse curriculum). If isolation also gives ~28%: the architecture
itself is the bottleneck (sparse pattern coding, motor pool
discrimination, etc.) and we need bigger changes.

Design:
* Same v2 baseline brain region setup
* No Phase 1 (visuomotor)
* No Phase 2 (text I/O on cascade)
* "Phase 3"-style paired stimulation training on a synthetic balanced
  buffer of (token, action) pairs — N events per direction word
* Standard eval (W→A and I→W)

The paired-stim training procedure is identical to the existing
_run_swr_replay_phase (which is already a clean paired-stim driver
with R-STDP) — we just feed it a synthetic balanced buffer instead
of a buffer collected during cascade training.

Usage:
    python -m research.runners.text_pfc_bypass_isolation \\
        --seed 42 --n-events-per-direction 100 \\
        --out-stats research/findings/raw/g11_bg/text_eval_h4_seed42.json

Output JSON has the same shape as text_eval_embodied / curriculum so
the Language tab in the webapp surfaces it natively.
"""

import argparse
import json
import time
import numpy as np

# Reuse the curriculum runner's helpers; we want the SAME bridge setup
from research.runners.text_train_curriculum import (
    ACTION_NAMES,
    ACTION_DELTAS,
    _set_language_gates,
    _run_swr_replay_phase,
)


def run_pfc_bypass_isolation(
    seed: int = 42,
    n_events_per_direction: int = 100,
    stim_steps_per_step: int = 200,
    reset_steps: int = 100,
    lang_input_drive_pA: float = 200.0,
    lang_output_coactive_pA: float = 150.0,
    motor_replay_drive_pA: float = 50.0,
    n_motor_per_action: int = 10,
    text_n_input_neurons: int = 256,
    text_n_output_neurons: int = 256,
    enable_distributed_motor_pop: bool = False,
    n_motor_pop_per_subpool: int = 5,
    token_sparsity: float = 0.1,
    verbose: bool = True,
):
    """Run the H4 PFC bypass isolation experiment.

    Skips Phase 1 + Phase 2 entirely; runs paired-stim training on a
    synthetic balanced buffer of (token, action) pairs.

    Returns (bridge, training_stats) so caller can do eval.
    """
    import cupy as cp
    from sim.config import (
        CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig,
    )
    from sim.bridge import SimulationBridge
    from sim.visual_cortex import apply_v1_gabor_weights
    from research.runners.g11_bg_runner import build_bg_brain_regions

    rng = np.random.default_rng(seed)

    # Build same architecture as v2 baseline / curriculum runner
    if verbose:
        print("=" * 60)
        print(f"PFC BYPASS ISOLATION (seed={seed})")
        print(f"  {n_events_per_direction} paired-stim events per direction")
        print(f"  Total: {4 * n_events_per_direction} events")
        print(f"  Skipping Phase 1 + Phase 2 (no cascade training)")
        print("=" * 60, flush=True)

    # Match the curriculum runner's bridge setup EXACTLY so this is a
    # like-for-like architecture comparison. Critical: the v2 readout
    # pathway init weights (text_cortex_to_output_weight=0.5 + jitter)
    # are part of what makes text I/O work at all — without them STDP
    # can't grow the readout from 0.
    regions, pathways = build_bg_brain_regions(
        enable_striatal_fsis=True,
        enable_cluster_a_closed_loop=True,
        enable_cluster_e_topography=True,
        enable_pfc=True,
        pfc_enable_nmda=True,
        enable_visual_cortex=True,
        enable_text_io=True,
        n_motor_per_action=n_motor_per_action,
        text_n_input_neurons=text_n_input_neurons,
        text_n_output_neurons=text_n_output_neurons,
        text_cortex_to_output_weight=0.5,
        text_cortex_to_output_jitter=0.3,
        text_it_to_output_weight=0.5,
        text_it_to_output_jitter=0.3,
        enable_distributed_motor_pop=enable_distributed_motor_pop,
        n_motor_pop_per_subpool=n_motor_pop_per_subpool,
    )

    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = list(regions)
    cfg.region_pathways = list(pathways)
    cfg.dt_ms = 0.5
    cfg.seed = seed
    cfg.enable_nmda = True
    cfg.nmda_ratio = 0.5
    cfg.enable_structural_plasticity = False
    cfg.enable_per_type_stp = False
    # v2 critical fixes (2026-05-02)
    cfg.enable_hebbian_learning = False
    cfg.stdp_w_max = 5.0

    bridge = SimulationBridge(
        core_config=cfg,
        viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(),
        gpu_config=GPUConfig(),
    )
    bridge.runtime_state.max_delay_steps = int(
        cfg.max_synaptic_delay_ms / cfg.dt_ms
    )
    bridge._initialize_simulation_data(called_from_playback_init=False)

    apply_v1_gabor_weights(
        bridge,
        n_orientations=8, n_frequencies=2, n_positions_per_dim=8,
        retina_size=32, receptive_field_radius=4, weight_scale=10.0,
    )

    # Force language gates open from the start
    _set_language_gates(bridge, 1.0, verbose=verbose)

    # Build synthetic balanced experience buffer
    # Each event = (token, action, reward, correct_move) tuple just like
    # the curriculum runner's buffer entries.
    DIRECTIONS = ["north", "east", "south", "west"]
    DIRECTION_TO_ACTION = {"north": "N", "east": "E", "south": "S", "west": "W"}
    synthetic_buffer = []
    for direction in DIRECTIONS:
        action = DIRECTION_TO_ACTION[direction]
        for _ in range(n_events_per_direction):
            synthetic_buffer.append({
                "token": direction,
                "action": action,
                "reward": 1.0,  # all positive — we're teaching the mapping
                "correct_move": True,
            })
    # Shuffle so directions don't train in blocks
    rng.shuffle(synthetic_buffer)

    if verbose:
        print(f"\n[isolation] Synthetic buffer: {len(synthetic_buffer)} events "
              f"({n_events_per_direction}/direction, shuffled)", flush=True)

    # Run paired-stim training using the existing SWR replay function.
    # This is the same code path as Phase 3 of text_train_curriculum, just
    # fed a different buffer.
    t_start = time.time()
    n_replayed = _run_swr_replay_phase(
        bridge, cp, rng,
        experience_buffer=synthetic_buffer,
        n_replay_events=len(synthetic_buffer),
        stim_steps_per_step=stim_steps_per_step,
        reset_steps=reset_steps,
        lang_input_drive_pA=lang_input_drive_pA,
        lang_output_coactive_pA=lang_output_coactive_pA,
        motor_replay_drive_pA=motor_replay_drive_pA,
        only_correct_experiences=True,
        balanced_directions=False,  # buffer is already balanced by construction
        token_sparsity=token_sparsity,
        verbose=verbose,
    )
    elapsed = time.time() - t_start

    if verbose:
        print(f"\n[isolation] Training complete: {n_replayed} events "
              f"({elapsed:.0f}s)", flush=True)

    training_stats = [{
        "phase": 1,
        "regime": "pfc_bypass_isolation",
        "n_total_events": n_replayed,
        "n_per_direction": n_events_per_direction,
        "elapsed_seconds": elapsed,
    }]

    return bridge, training_stats


def main():
    ap = argparse.ArgumentParser(
        description="H4: PFC bypass isolation experiment. Tests upper bound of "
                    "language->motor direct pathway WITHOUT cascade interference."
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-events-per-direction", type=int, default=100,
                    help="Paired-stim events per direction word (default 100). "
                    "Total events = 4 × this value.")
    ap.add_argument("--stim-steps-per-step", type=int, default=200)
    ap.add_argument("--reset-steps", type=int, default=100)
    ap.add_argument("--n-eval-image-word", type=int, default=100)
    ap.add_argument("--n-eval-word-action", type=int, default=25)
    ap.add_argument("--out-stats", type=str, default=None)
    ap.add_argument("--lang-input-drive-pA", type=float, default=200.0)
    ap.add_argument("--lang-output-coactive-pA", type=float, default=150.0)
    ap.add_argument("--motor-replay-drive-pA", type=float, default=50.0)
    ap.add_argument("--n-motor-per-action", type=int, default=10)
    ap.add_argument("--text-n-input-neurons", type=int, default=256)
    ap.add_argument("--text-n-output-neurons", type=int, default=256)
    ap.add_argument("--enable-distributed-motor-pop", action="store_true",
                    default=False)
    ap.add_argument("--n-motor-pop-per-subpool", type=int, default=5)
    ap.add_argument("--token-sparsity", type=float, default=0.1,
                    help="fraction of language_input neurons activated per "
                    "word (default 0.1). Try 0.05 for orthogonal codes.")
    args = ap.parse_args()

    bridge, train_stats = run_pfc_bypass_isolation(
        seed=args.seed,
        n_events_per_direction=args.n_events_per_direction,
        stim_steps_per_step=args.stim_steps_per_step,
        reset_steps=args.reset_steps,
        lang_input_drive_pA=args.lang_input_drive_pA,
        lang_output_coactive_pA=args.lang_output_coactive_pA,
        motor_replay_drive_pA=args.motor_replay_drive_pA,
        n_motor_per_action=args.n_motor_per_action,
        text_n_input_neurons=args.text_n_input_neurons,
        text_n_output_neurons=args.text_n_output_neurons,
        enable_distributed_motor_pop=args.enable_distributed_motor_pop,
        n_motor_pop_per_subpool=args.n_motor_pop_per_subpool,
        token_sparsity=args.token_sparsity,
        verbose=True,
    )

    # Eval W→A and I→W using the same evaluators the curriculum runner uses
    from research.runners.text_eval import (
        evaluate_image_to_word, evaluate_word_to_action,
    )
    print("\n" + "=" * 60)
    print(f"EVAL: image -> word ({args.n_eval_image_word} fresh trials)")
    print("=" * 60, flush=True)
    iw_result = evaluate_image_to_word(
        bridge, n_trials=args.n_eval_image_word, grid_size=8,
    )
    print(f"\n  Accuracy: {iw_result['correct']}/{iw_result['n_trials']} "
          f"= {iw_result['accuracy']:.1%}")
    print(f"  Confusion: {iw_result['confusion_matrix']}", flush=True)

    print("\n" + "=" * 60)
    print(f"EVAL: word -> action ({args.n_eval_word_action} per word, "
          f"token_sparsity={args.token_sparsity})")
    print("=" * 60, flush=True)
    wa_result = evaluate_word_to_action(
        bridge, n_trials_per_word=args.n_eval_word_action,
        token_sparsity=args.token_sparsity,
    )
    print(f"\n  Accuracy: {wa_result['correct']}/{wa_result['n_trials']} "
          f"= {wa_result['accuracy']:.1%}")
    print(f"  Confusion: {wa_result['confusion_matrix']}", flush=True)

    if args.out_stats:
        out = {
            "regime": "pfc_bypass_isolation",
            "seed": args.seed,
            "n_events_per_direction": args.n_events_per_direction,
            "n_total_events": 4 * args.n_events_per_direction,
            "training_stats": train_stats,
            "image_to_word_eval": iw_result,
            "word_to_action_eval": wa_result,
            "config": {
                "lang_input_drive_pA": args.lang_input_drive_pA,
                "lang_output_coactive_pA": args.lang_output_coactive_pA,
                "motor_replay_drive_pA": args.motor_replay_drive_pA,
                "stim_steps_per_step": args.stim_steps_per_step,
                "reset_steps": args.reset_steps,
                "n_motor_per_action": args.n_motor_per_action,
                "text_n_input_neurons": args.text_n_input_neurons,
                "text_n_output_neurons": args.text_n_output_neurons,
                "token_sparsity": args.token_sparsity,
            },
        }
        from pathlib import Path
        Path(args.out_stats).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_stats).write_text(json.dumps(out, indent=2))
        print(f"\n  Saved: {args.out_stats}", flush=True)


if __name__ == "__main__":
    main()
