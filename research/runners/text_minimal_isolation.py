"""
Minimal language->motor isolation experiment.

Tests the decisive question: can the architecture learn word-action
mapping AT ALL when stripped of cascade interference?

Prior data (2026-05-03 autonomous overnight) shows 0/39 aligned across
all v2-architecture conditions. Pattern analysis shows misalignment is
seed-dependent (each random init creates its own private misalignment),
with a mild cascade-driven motor_E bias of ~3pp.

If THIS minimal architecture (NO cascade, NO PFC, NO retina, NO
visuomotor — just language_input -> motor_X with paired-stim training)
achieves aligned >= 4/6, the cascade IS the dominant interference.

If THIS also fails, the fundamental issue is deeper (plasticity dose,
soft-bound STDP, sparse-code overlap, or eval methodology).

Architecture:
  - language_input: 256 neurons (same as v2 baseline for fair compare)
  - motor_N, motor_E, motor_S, motor_W: 25 each (slightly larger than
    v2's 10 to reduce SNR noise; doesn't affect alignment if test
    works)
  - language_input -> motor_X pathways (4 plastic, all4 actions)
  - NO cluster_a, NO cluster_e, NO cortex_X cascade
  - NO retina, NO visual cortex, NO PFC
  - NO visuomotor pathways

Training:
  - paired-stim only (same _run_swr_replay_phase mechanism as H4)
  - synthetic balanced buffer: N events per direction, +1 reward
  - directly tests STDP's ability to differentiate words on a clean
    pathway

Eval:
  - same evaluate_word_to_action that everything else uses
  - 25 trials per word, interleaved
  - aligned ratio is the headline metric

Usage:
    python -m research.runners.text_minimal_isolation \\
        --seed 42 --n-events-per-direction 1000 \\
        --out-stats research/findings/raw/g11_bg/text_eval_minimal_iso_seed42.json
"""

import argparse
import json
import time
import numpy as np


def build_minimal_brain_regions(
    n_lang_input: int = 256,
    n_motor_per_action: int = 25,
    text_input_to_motor_density: float = 0.30,
    text_input_to_motor_weight: float = 3.0,
    text_input_to_motor_jitter: float = 0.5,
):
    """Build a minimal language->motor architecture for isolation tests.

    Returns (regions, pathways) tuple compatible with the brain-region
    framework.
    """
    from sim.regions import BrainRegion, RegionPathway
    from sim.enums import NeuronType

    ACTION_NAMES = ["N", "E", "S", "W"]

    regions = []
    pathways = []

    # Language input region (sparse code substrate)
    regions.append(BrainRegion(
        name="language_input",
        n_neurons=n_lang_input,
        exc_fraction=0.8,
        internal_density=0.05,
        exc_weight_mean=2.0, inh_weight_mean=4.0,
        weight_jitter=0.2, plastic_internal=True,
        izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name,
    ))

    # Motor pools — separate region per action
    for action in ACTION_NAMES:
        regions.append(BrainRegion(
            name=f"motor_{action}",
            n_neurons=n_motor_per_action,
            exc_fraction=1.0,  # purely excitatory motor pool
            internal_density=0.0,  # no internal recurrence
            exc_weight_mean=0.0, inh_weight_mean=0.0,
            weight_jitter=0.0, plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name,
        ))

    # language_input -> motor_X pathways (the ONE pathway being tested)
    for action in ACTION_NAMES:
        pathways.append(RegionPathway(
            from_region="language_input", to_region=f"motor_{action}",
            density=text_input_to_motor_density,
            weight_mean=text_input_to_motor_weight,
            weight_jitter=text_input_to_motor_jitter,
            plastic=True,
            plasticity_gate="language_input_to_motor",
        ))

    return regions, pathways


def run_minimal_isolation(
    seed: int = 42,
    n_events_per_direction: int = 1000,
    stim_steps_per_step: int = 100,
    reset_steps: int = 50,
    lang_input_drive_pA: float = 200.0,
    lang_output_coactive_pA: float = 0.0,  # no language_output region
    motor_replay_drive_pA: float = 50.0,
    n_motor_per_action: int = 25,
    n_lang_input: int = 256,
    token_sparsity: float = 0.1,
    dt_ms: float = 1.0,
    text_input_to_motor_weight: float = 3.0,
    text_input_to_motor_jitter: float = 0.5,
    stdp_w_max: float = 5.0,
    enable_hebbian: bool = False,
    verbose: bool = True,
):
    """Run the minimal isolation experiment. Returns (bridge, stats)."""
    import cupy as cp
    from sim.config import (
        CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig,
    )
    from sim.bridge import SimulationBridge

    rng = np.random.default_rng(seed)

    if verbose:
        print("=" * 60)
        print(f"MINIMAL LANGUAGE->MOTOR ISOLATION (seed={seed})")
        print(f"  n_lang_input={n_lang_input}, motor_per_action={n_motor_per_action}")
        print(f"  Total: {n_lang_input + 4*n_motor_per_action} neurons")
        print(f"  {n_events_per_direction} paired-stim events per direction")
        print(f"  dt={dt_ms}ms, stim={stim_steps_per_step}, reset={reset_steps}")
        print(f"  enable_hebbian={enable_hebbian}, stdp_w_max={stdp_w_max}")
        print("=" * 60, flush=True)

    regions, pathways = build_minimal_brain_regions(
        n_lang_input=n_lang_input,
        n_motor_per_action=n_motor_per_action,
        text_input_to_motor_weight=text_input_to_motor_weight,
        text_input_to_motor_jitter=text_input_to_motor_jitter,
    )

    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = list(regions)
    cfg.region_pathways = list(pathways)
    cfg.dt_ms = dt_ms
    cfg.seed = seed
    cfg.enable_nmda = False  # not needed for minimal arch
    cfg.enable_structural_plasticity = False
    cfg.enable_per_type_stp = False
    cfg.enable_hebbian_learning = enable_hebbian
    cfg.stdp_w_max = stdp_w_max

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

    # Build synthetic balanced experience buffer
    DIRECTIONS = ["north", "east", "south", "west"]
    DIRECTION_TO_ACTION = {"north": "N", "east": "E", "south": "S", "west": "W"}
    synthetic_buffer = []
    for direction in DIRECTIONS:
        action = DIRECTION_TO_ACTION[direction]
        for _ in range(n_events_per_direction):
            synthetic_buffer.append({
                "token": direction,
                "action": action,
                "reward": 1.0,
                "correct_move": True,
            })
    rng.shuffle(synthetic_buffer)

    if verbose:
        print(f"\n[minimal-iso] Synthetic buffer: {len(synthetic_buffer)} events "
              f"({n_events_per_direction}/dir, shuffled)", flush=True)

    # Training: paired-stim using same mechanism as H4 SWR replay.
    # Inline since we don't have language_output region (curriculum's
    # _run_swr_replay_phase requires it).
    from sim.text_embeddings import vocab_to_drive_pattern

    rm = bridge.region_manager
    lang_input_idx = cp.asarray(list(rm.indices("language_input")), dtype=cp.int64)
    motor_idx = {
        a: cp.asarray(list(rm.indices(f"motor_{a}")), dtype=cp.int64)
        for a in ["N", "E", "S", "W"]
    }
    n_lang = int(lang_input_idx.size)

    t_start = time.time()
    n_replays = 0
    for event_idx, event in enumerate(synthetic_buffer):
        token = event["token"]
        action = event["action"]
        reward = event["reward"]

        # Inter-trial reset
        bridge.cp_external_input_current[:] = 0.0
        bridge.core_config.current_reward_signal = 0.0
        for _ in range(reset_steps):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1

        # Drive language_input only (no language_output in minimal arch)
        in_drive = vocab_to_drive_pattern(
            token, n_neurons=n_lang,
            drive_max_pA=lang_input_drive_pA, sparsity=token_sparsity,
        )
        bridge.cp_external_input_current[lang_input_idx] = cp.asarray(
            in_drive, dtype=cp.float32,
        )
        # Drive motor pool (the "nudge" toward correct action)
        bridge.cp_external_input_current[motor_idx[action]] += motor_replay_drive_pA

        # Stim window
        for _ in range(stim_steps_per_step):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1

        # Apply reward
        bridge.core_config.current_reward_signal = float(reward)
        for _ in range(20):  # reward window
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1

        n_replays += 1

        if verbose and (event_idx + 1) % 250 == 0:
            elapsed = time.time() - t_start
            print(f"  [minimal-iso] {event_idx+1}/{len(synthetic_buffer)} events "
                  f"({elapsed:.0f}s)", flush=True)

    elapsed = time.time() - t_start
    if verbose:
        print(f"\n[minimal-iso] Training complete: {n_replays} events "
              f"({elapsed:.0f}s)", flush=True)

    training_stats = [{
        "phase": 1,
        "regime": "minimal_language_motor_isolation",
        "n_total_events": n_replays,
        "n_per_direction": n_events_per_direction,
        "elapsed_seconds": elapsed,
    }]

    return bridge, training_stats


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-events-per-direction", type=int, default=1000,
                    help="Paired-stim events per direction (default 1000)")
    ap.add_argument("--n-eval-word-action", type=int, default=25)
    ap.add_argument("--out-stats", type=str, default=None)
    ap.add_argument("--lang-input-drive-pA", type=float, default=200.0)
    ap.add_argument("--motor-replay-drive-pA", type=float, default=50.0)
    ap.add_argument("--n-motor-per-action", type=int, default=25)
    ap.add_argument("--n-lang-input", type=int, default=256)
    ap.add_argument("--stim-steps-per-step", type=int, default=100)
    ap.add_argument("--reset-steps", type=int, default=50)
    ap.add_argument("--token-sparsity", type=float, default=0.1)
    ap.add_argument("--dt-ms", type=float, default=1.0)
    ap.add_argument("--text-input-to-motor-weight", type=float, default=3.0)
    ap.add_argument("--text-input-to-motor-jitter", type=float, default=0.5)
    ap.add_argument("--stdp-w-max", type=float, default=5.0)
    ap.add_argument("--enable-hebbian", action="store_true", default=False)
    args = ap.parse_args()

    bridge, train_stats = run_minimal_isolation(
        seed=args.seed,
        n_events_per_direction=args.n_events_per_direction,
        stim_steps_per_step=args.stim_steps_per_step,
        reset_steps=args.reset_steps,
        lang_input_drive_pA=args.lang_input_drive_pA,
        motor_replay_drive_pA=args.motor_replay_drive_pA,
        n_motor_per_action=args.n_motor_per_action,
        n_lang_input=args.n_lang_input,
        token_sparsity=args.token_sparsity,
        dt_ms=args.dt_ms,
        text_input_to_motor_weight=args.text_input_to_motor_weight,
        text_input_to_motor_jitter=args.text_input_to_motor_jitter,
        stdp_w_max=args.stdp_w_max,
        enable_hebbian=args.enable_hebbian,
        verbose=True,
    )

    # Eval W->A only (no I->W since no visual cortex)
    from research.runners.text_eval import evaluate_word_to_action
    print("\n" + "=" * 60)
    print(f"EVAL: word -> action ({args.n_eval_word_action} per word, "
          f"token_sparsity={args.token_sparsity})")
    print("=" * 60, flush=True)
    wa_result = evaluate_word_to_action(
        bridge, n_trials_per_word=args.n_eval_word_action,
        stim_steps_per_trial=args.stim_steps_per_step,
        n_reset_steps=args.reset_steps,
        token_sparsity=args.token_sparsity,
    )
    print(f"\n  Accuracy: {wa_result['correct']}/{wa_result['n_trials']} "
          f"= {wa_result['accuracy']:.1%}")
    print(f"  Confusion: {wa_result['confusion_matrix']}", flush=True)

    if args.out_stats:
        out = {
            "regime": "minimal_language_motor_isolation",
            "seed": args.seed,
            "n_events_per_direction": args.n_events_per_direction,
            "n_total_events": 4 * args.n_events_per_direction,
            "training_stats": train_stats,
            "word_to_action_eval": wa_result,
            "config": {
                "n_lang_input": args.n_lang_input,
                "n_motor_per_action": args.n_motor_per_action,
                "lang_input_drive_pA": args.lang_input_drive_pA,
                "motor_replay_drive_pA": args.motor_replay_drive_pA,
                "stim_steps_per_step": args.stim_steps_per_step,
                "reset_steps": args.reset_steps,
                "token_sparsity": args.token_sparsity,
                "dt_ms": args.dt_ms,
                "text_input_to_motor_weight": args.text_input_to_motor_weight,
                "stdp_w_max": args.stdp_w_max,
                "enable_hebbian": args.enable_hebbian,
            },
        }
        from pathlib import Path
        Path(args.out_stats).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_stats).write_text(json.dumps(out, indent=2))
        print(f"\n  Saved: {args.out_stats}", flush=True)


if __name__ == "__main__":
    main()
