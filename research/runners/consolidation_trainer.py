"""Phase 1.3 -- Hippocampus -> cortex consolidation trainer.

Implements the awake/sleep training loop for biology-grounded
continual learning consolidation. Per design at
docs/plans/2026-05-06-Phase-1.3-consolidation-design.md.

Training arc per episode:
  1. AWAKE PHASE: train Tier 1 (or Tier 2.1 / 2.3) vocabulary as
     normal. Direct lang -> motor pathway plastic, hippocampal
     encoding pathways plastic, consolidation pathways FROZEN.

  2. SLEEP PHASE: input drive zeroed. Direct lang -> motor frozen.
     CA3 SWR-burst plasticity ON. CA1 -> motor / lang_output
     consolidation pathways ON. Drive sub-populations of CA3 with
     SWR-style bursts (~150Hz, 100ms windows). CA3 attractors
     activate -> drive CA1 -> CA1's plastic projections to motor +
     language_output get STDP updates -> cortex consolidates.

Eval modes (separate module phrase_eval.py / forgetting_eval):
  - Standard W->A (matches Tier 1 baseline)
  - Hippo-OFF (zero CA3 + CA1, retest -- proves cortex consolidated)
  - Sleep-recovery (Phase 1.4 + sleep cycle, retest retention)

Status (2026-05-06):
- Builder + gate helpers: DONE (text_minimal_isolation.py)
- This trainer: SKELETON (TODO: GPU validation, hippo-OFF eval)
- Smoke + 6-seed: TBD

Caveat: this code is UNTESTED on GPU. Do not run for production
until smoke validation lands.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def run_swr_replay_phase(
    bridge,
    n_swr_events: int = 200,
    burst_duration_ms: int = 100,
    inter_burst_ms: int = 50,
    swr_drive_pA: float = 100.0,
    rng: Optional[np.random.Generator] = None,
):
    """Drive CA3 with SWR-style bursts during sleep phase.

    Each SWR event:
    1. Pick random subset of CA3 neurons (~10-20% sparse pattern)
    2. Drive at swr_drive_pA for burst_duration_ms (~100ms)
       producing ~150Hz population firing in driven subset
    3. CA3 recurrent (ca3_swr_burst gate ON) sharpens the pattern
       via attractor dynamics
    4. Drive propagates through ca3 -> ca1 (Schaffer collaterals)
    5. CA1 fires patterns related to CA3 activity
    6. CA1 -> motor / lang_output STDP transfers patterns to cortex
    7. inter_burst_ms quiet between bursts

    Implements Buzsaki 2015 ripple model + McClelland 1995 CLS theory.
    """
    import cupy as cp
    if rng is None:
        rng = np.random.default_rng()

    rm = bridge.region_manager
    ca3_indices = list(rm.indices("ca3"))
    n_ca3 = len(ca3_indices)
    ca3_arr = cp.asarray(ca3_indices, dtype=cp.int64)

    sparsity = 0.15  # 15% of CA3 active per SWR event

    for event_idx in range(n_swr_events):
        # Pick random sparse pattern in CA3
        n_active = max(1, int(sparsity * n_ca3))
        active_local = rng.choice(n_ca3, size=n_active, replace=False)
        active_global = ca3_arr[cp.asarray(active_local, dtype=cp.int64)]
        # Drive burst
        bridge.cp_external_input_current[:] = 0.0
        bridge.cp_external_input_current[active_global] = float(swr_drive_pA)
        for _ in range(burst_duration_ms):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
        # Quiet inter-burst window
        bridge.cp_external_input_current[:] = 0.0
        for _ in range(inter_burst_ms):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1


def run_consolidation_training(
    seed: int = 42,
    n_awake_events_per_word: int = 200,
    n_sleep_swr_events: int = 200,
    consolidation_interval: int = 4,  # # awake events per word per
                                       # sleep cycle. 4 means 1 sleep
                                       # per 4 awake events per word
                                       # (i.e., 1 sleep per 200/4 = 50
                                       # full epochs).
    n_lang_input: int = 2048,
    n_motor_per_action: int = 500,
    n_motor_fs_per_action: int = 60,
    swr_drive_pA: float = 100.0,
    verbose: bool = True,
):
    """Train Tier 1 vocab with hippocampus consolidation interleaved.

    Awake phases: standard Tier 1 embodied Hebbian (cortex + hippo
    both plastic).
    Sleep phases: replay-driven cortex consolidation.

    Returns (bridge, training_stats).

    Caveat (2026-05-06): UNTESTED on GPU. Logic mirrors design but
    needs runtime verification before production use.
    """
    import cupy as cp
    from sim.config import (
        CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig,
    )
    from sim.bridge import SimulationBridge
    from sim.text_embeddings import vocab_to_drive_pattern
    from research.runners.text_minimal_isolation import (
        build_biological_brain_regions,
        apply_topographic_bias as _apply_topo,
        set_awake_gates, set_sleep_gates, freeze_all_gates,
    )

    PRIMARY_WORDS = ["north", "east", "south", "west"]
    PRIMARY_TO_ACTION = {
        "north": "N", "east": "E", "south": "S", "west": "W",
    }
    rng = np.random.default_rng(seed)

    if verbose:
        print("=" * 60)
        print(f"CONSOLIDATION TRAINER (Phase 1.3, seed={seed})")
        print(f"  Awake events/word: {n_awake_events_per_word}")
        print(f"  Sleep SWR events: {n_sleep_swr_events}")
        print(f"  Consolidation interval: every {consolidation_interval} "
              f"awake events/word")
        print("=" * 60, flush=True)

    # Build architecture: Tier 1 + hippocampus consolidation
    regions, pathways = build_biological_brain_regions(
        n_lang_input=n_lang_input,
        n_motor_per_action=n_motor_per_action,
        enable_motor_fs=True,
        n_motor_fs_per_action=n_motor_fs_per_action,
        enable_language_output=True,
        n_lang_output=n_lang_input,
        motor_to_language_output_weight=2.0,
        enable_hippocampus_consolidation=True,
    )

    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = list(regions)
    cfg.region_pathways = list(pathways)
    cfg.dt_ms = 1.0
    cfg.seed = seed
    cfg.enable_nmda = True
    cfg.enable_structural_plasticity = False
    cfg.enable_per_type_stp = False
    cfg.enable_hebbian_learning = False
    cfg.stdp_w_max = 5.0
    cfg.fast_spike_reset = True

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

    # Topographic prior on language_input -> motor (Tier 1 BREAKTHROUGH)
    _apply_topo(
        bridge,
        topographic_factor=1.5,
        off_target_factor=0.7,
        n_lang_input=n_lang_input,
        sparsity=0.1,
        apply_reciprocal=True,
        n_lang_output=n_lang_input,
        verbose=verbose,
    )

    # Build awake training buffer
    awake_buffer = []
    for word in PRIMARY_WORDS:
        for _ in range(n_awake_events_per_word):
            awake_buffer.append({
                "token": word,
                "action": PRIMARY_TO_ACTION[word],
            })
    rng.shuffle(awake_buffer)

    rm = bridge.region_manager
    lang_input_idx = list(rm.indices("language_input"))
    lang_output_idx = list(rm.indices("language_output"))
    motor_idx = {a: list(rm.indices(f"motor_{a}"))
                 for a in ["N", "E", "S", "W"]}
    n_lang_in = len(lang_input_idx)
    lang_input_arr = cp.asarray(lang_input_idx, dtype=cp.int64)
    lang_output_arr = cp.asarray(lang_output_idx, dtype=cp.int64)
    motor_arr = {a: cp.asarray(motor_idx[a], dtype=cp.int64)
                 for a in ["N", "E", "S", "W"]}

    def _drive_for(word: str):
        d = vocab_to_drive_pattern(
            word, n_neurons=n_lang_in,
            drive_max_pA=200.0, sparsity=0.1,
        )
        return cp.asarray(d, dtype=cp.float32)

    # Awake/sleep alternation. Sleep interval is per-word, so we
    # split the buffer into chunks of 4*consolidation_interval
    # awake events (1 chunk = consolidation_interval per word).
    awake_chunk_size = 4 * consolidation_interval
    n_chunks = max(1, len(awake_buffer) // awake_chunk_size)
    if verbose:
        print(f"  Total awake events: {len(awake_buffer)} in {n_chunks} "
              f"chunks of {awake_chunk_size}", flush=True)

    motor_teacher_pA = 300.0
    stim_steps_per_event = 50
    reset_steps = 50

    t0 = time.time()
    n_sleep_phases_run = 0
    for chunk_idx in range(n_chunks):
        # Awake phase
        set_awake_gates(bridge)
        chunk_start = chunk_idx * awake_chunk_size
        chunk_end = min(chunk_start + awake_chunk_size, len(awake_buffer))
        for ev_idx in range(chunk_start, chunk_end):
            ev = awake_buffer[ev_idx]
            # Inter-trial reset
            bridge.cp_external_input_current[:] = 0.0
            for _ in range(reset_steps):
                bridge._run_one_simulation_step()
                bridge.runtime_state.current_time_step += 1
            # Drive lang_input + lang_output (teacher) + motor target
            drive = _drive_for(ev["token"])
            bridge.cp_external_input_current[lang_input_arr] = drive
            bridge.cp_external_input_current[lang_output_arr] = drive
            bridge.cp_external_input_current[motor_arr[ev["action"]]] += \
                float(motor_teacher_pA)
            for _ in range(stim_steps_per_event):
                bridge._run_one_simulation_step()
                bridge.runtime_state.current_time_step += 1
        if verbose:
            print(f"  [chunk {chunk_idx+1}/{n_chunks} awake done "
                  f"({time.time()-t0:.0f}s)]", flush=True)

        # Sleep phase
        set_sleep_gates(bridge)
        run_swr_replay_phase(
            bridge,
            n_swr_events=n_sleep_swr_events,
            burst_duration_ms=100,
            inter_burst_ms=50,
            swr_drive_pA=swr_drive_pA,
            rng=rng,
        )
        n_sleep_phases_run += 1
        if verbose:
            print(f"  [sleep phase {n_sleep_phases_run} done "
                  f"({time.time()-t0:.0f}s)]", flush=True)

    # Freeze all plasticity for downstream eval
    freeze_all_gates(bridge)

    return bridge, {
        "n_awake_events": len(awake_buffer),
        "n_sleep_phases": n_sleep_phases_run,
        "n_total_swr_events":
            n_sleep_phases_run * n_sleep_swr_events,
        "wall_clock_seconds": time.time() - t0,
    }


def run_full(
    seed: int = 42,
    n_awake_events_per_word: int = 200,
    n_sleep_swr_events: int = 200,
    consolidation_interval: int = 4,
    n_lang_input: int = 2048,
    n_motor_per_action: int = 500,
    n_motor_fs_per_action: int = 60,
    n_test_per_word: int = 25,
    verbose: bool = True,
) -> Dict[str, Any]:
    """End-to-end Phase 1.3: train with awake/sleep alternation +
    run consolidation proof. Returns unified JSON-friendly dict.
    """
    from research.runners.consolidation_eval import (
        evaluate_consolidation_proof,
    )
    bridge, stats = run_consolidation_training(
        seed=seed,
        n_awake_events_per_word=n_awake_events_per_word,
        n_sleep_swr_events=n_sleep_swr_events,
        consolidation_interval=consolidation_interval,
        n_lang_input=n_lang_input,
        n_motor_per_action=n_motor_per_action,
        n_motor_fs_per_action=n_motor_fs_per_action,
        verbose=verbose,
    )
    if verbose:
        print("\n=== CONSOLIDATION PROOF ===", flush=True)
    result = evaluate_consolidation_proof(
        bridge, n_trials_per_word=n_test_per_word, verbose=verbose,
    )
    if verbose:
        passed = result.get("pass", False)
        print("\n" + "=" * 60)
        print(f"PHASE 1.3 SEED {seed}: "
              f"{'[OK] CONSOLIDATION CONFIRMED' if passed else '[X] FAIL'}")
        if "ratio" in result:
            print(f"  Pre-silence W->A: {result['pre_silence_acc']:.1%}")
            print(f"  Hippo-OFF W->A:   {result['hippo_off_acc']:.1%}")
            print(f"  Ratio:            {result['ratio']:.0%}")
        print("=" * 60, flush=True)
    return {
        "seed": seed,
        "stats": stats,
        "consolidation_proof": result,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-awake-events-per-word", type=int, default=200)
    ap.add_argument("--n-sleep-swr-events", type=int, default=200)
    ap.add_argument("--consolidation-interval", type=int, default=4)
    ap.add_argument("--n-lang-input", type=int, default=2048)
    ap.add_argument("--n-motor-per-action", type=int, default=500)
    ap.add_argument("--n-motor-fs-per-action", type=int, default=60)
    ap.add_argument("--n-test-per-word", type=int, default=25)
    ap.add_argument("--train-only", action="store_true",
                    help="Skip consolidation proof; output stats only")
    ap.add_argument("--out-stats", type=str, default=None)
    args = ap.parse_args()

    if args.train_only:
        bridge, stats = run_consolidation_training(
            seed=args.seed,
            n_awake_events_per_word=args.n_awake_events_per_word,
            n_sleep_swr_events=args.n_sleep_swr_events,
            consolidation_interval=args.consolidation_interval,
            n_lang_input=args.n_lang_input,
            n_motor_per_action=args.n_motor_per_action,
            n_motor_fs_per_action=args.n_motor_fs_per_action,
            verbose=True,
        )
        result = {"seed": args.seed, "stats": stats}
    else:
        result = run_full(
            seed=args.seed,
            n_awake_events_per_word=args.n_awake_events_per_word,
            n_sleep_swr_events=args.n_sleep_swr_events,
            consolidation_interval=args.consolidation_interval,
            n_lang_input=args.n_lang_input,
            n_motor_per_action=args.n_motor_per_action,
            n_motor_fs_per_action=args.n_motor_fs_per_action,
            n_test_per_word=args.n_test_per_word,
            verbose=True,
        )

    if args.out_stats:
        Path(args.out_stats).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_stats).write_text(json.dumps(
            result, indent=2, default=str
        ))
        print(f"Saved: {args.out_stats}", flush=True)


if __name__ == "__main__":
    main()
