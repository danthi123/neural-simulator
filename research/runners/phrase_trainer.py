"""Phase 1.2 Tier 2.3 -- Compositional 2-word phrase trainer.

ONLY trains; eval lives in phrase_eval.py. Builder, action_gate
neuromodulator, and tests are pre-staged in:
  - research/runners/text_minimal_isolation.py
    (build_biological_brain_regions(enable_dlpfc_verb=True),
     build_tier_2_3_action_gate())
  - tests/test_dlpfc_verb_builder.py (10 tests)

Per design at docs/plans/2026-05-06-Tier2.3-two-word-phrases-design.md.

Mechanism
---------
Vocabulary:
  - 1 verb: "go"
  - 4 directions: north, east, south, west
  - 4 phrases: "go north", "go east", "go south", "go west"

Sequential drive timing (per phrase trial):
  t = 0-100ms   : drive language_input["go"]
                  (NMDA bistability in dlpfc_verb holds verb context)
  t = 100-200ms : drive language_input["north"]
                  + motor_N teacher (action_gate boosted by PFC firing)
  t = 200-250ms : continue forward prop, observe motor selection

STDP at:
  - language_input["go"]    -> dlpfc_verb (verb binding)
  - language_input["north"] -> motor_N    (Tier 1 binding, reinforced)
  - dlpfc_verb activity      -> action_gate -> motor pools
    (excitability_drive, NOT plasticity per se -- mediates whether
     motor pools fire above threshold given direction word alone)

Training curriculum (per design):
  60% phrase trials   ("go" + direction -> motor target)
  30% direction-only  (just direction -> motor target; Tier 1 compat)
  10% verb-only       (just "go" -> no motor; PFC active alone)

Pass criteria (separate eval after training):
  - Phrase: >= 4/6 seeds correctly execute "go [dir]" -> motor target
  - Direction-only: >= 4/6 seeds keep Tier 1 W->A binding
  - Verb-only: >= 4/6 seeds keep motor pools quiet on bare "go"

Why this is harder than Tier 2.1
---------------------------------
Tier 2.3 introduces TEMPORAL COMPOSITIONALITY. The meaning of
"north" depends on whether "go" came first. The architecture
extends Tier 1 with PFC working memory acting as a context channel
that modulates motor selection (per Pulvermuller 2003, Wang 2002,
Goldman-Rakic 1995, Miller & Cohen 2001).

Status
------
This is the TRAINING runner only. Eval, smoke validation, and
6-seed run are TODO. Eval lives in a separate
research/runners/phrase_eval.py module that loads a checkpoint
and tests the 3 conditions independently.

Caveat
------
This runner is Phase 1.4 v3 follow-on work. Per master plan
decision tree, it is the natural next step ONLY if Phase 1.4 lands
in Branch A (>= 80% retention). If Phase 1.4 lands Branch B/C,
Phase 1.3 (consolidation) takes priority and this runner waits.

Usage
-----
After Phase 1.4 v3 6-seed validates Branch A:
    python -m research.runners.phrase_trainer --seed 42 \\
        --n-phrase-events 200 \\
        --n-direction-only-events 100 \\
        --n-verb-only-events 30 \\
        --out-checkpoint research/findings/raw/phrase/seed42.simstate.h5
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


VERBS = ["go"]  # Tier 2.3 first pass: 1 verb
DIRECTIONS = ["north", "east", "south", "west"]
DIRECTION_TO_ACTION = {"north": "N", "east": "E", "south": "S", "west": "W"}


def build_phrase_buffer(
    n_phrase_events: int,
    n_direction_only_events: int,
    n_verb_only_events: int,
    rng: np.random.Generator,
) -> List[Dict[str, Any]]:
    """Build the mixed-trial training buffer per design curriculum.

    Returns shuffled list of trial dicts with one of three types:
      - phrase:        verb + direction -> motor (60% per design)
      - direction_only: direction -> motor (30% per design; Tier 1 compat)
      - verb_only:      verb -> no motor (10%; anti-action test)
    """
    buffer = []
    # Phrase trials: 1 verb x 4 directions = 4 distinct phrases.
    # Distribute n_phrase_events across the 4 phrases evenly.
    per_phrase = n_phrase_events // len(DIRECTIONS)
    for direction in DIRECTIONS:
        for _ in range(per_phrase):
            buffer.append({
                "type": "phrase",
                "verb": VERBS[0],
                "direction": direction,
                "action": DIRECTION_TO_ACTION[direction],
            })
    # Direction-only trials: distribute across 4 directions
    per_dir_only = n_direction_only_events // len(DIRECTIONS)
    for direction in DIRECTIONS:
        for _ in range(per_dir_only):
            buffer.append({
                "type": "direction_only",
                "direction": direction,
                "action": DIRECTION_TO_ACTION[direction],
            })
    # Verb-only trials
    for _ in range(n_verb_only_events):
        buffer.append({
            "type": "verb_only",
            "verb": VERBS[0],
            "action": None,  # no motor target
        })
    rng.shuffle(buffer)
    return buffer


def run_phrase_training(
    seed: int = 42,
    n_phrase_events: int = 200,
    n_direction_only_events: int = 100,
    n_verb_only_events: int = 30,
    n_lang_input: int = 2048,
    n_motor_per_action: int = 500,
    n_motor_fs_per_action: int = 60,
    n_dlpfc_verb: int = 200,
    verb_drive_ms: int = 100,
    direction_drive_ms: int = 100,
    final_observe_ms: int = 50,
    reset_ms: int = 50,
    drive_pA: float = 200.0,
    motor_teacher_pA: float = 300.0,
    action_gate_drive_pA: float = 50.0,
    verbose: bool = True,
):
    """Train Tier 2.3 architecture with mixed phrase / direction-only /
    verb-only curriculum.

    Builds the architecture by extending Tier 1 BREAKTHROUGH config
    with dlpfc_verb region + lang->dlpfc_verb pathway + action_gate
    neuromodulator. Sequential drive timing implements the
    compositional binding mechanism.

    Returns (bridge, training_stats).
    """
    import cupy as cp
    from sim.config import (
        CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig,
    )
    from sim.bridge import SimulationBridge
    from sim.text_embeddings import vocab_to_drive_pattern
    from research.runners.text_minimal_isolation import (
        build_biological_brain_regions,
        build_tier_2_3_action_gate,
        apply_topographic_bias as _apply_topo,
    )

    rng = np.random.default_rng(seed)

    if verbose:
        print("=" * 60)
        print(f"PHRASE TRAINER (Tier 2.3, seed={seed})")
        print(f"  Buffer: {n_phrase_events} phrase + "
              f"{n_direction_only_events} direction-only + "
              f"{n_verb_only_events} verb-only")
        print(f"  Arch: lang={n_lang_input}, motor={n_motor_per_action}, "
              f"motor_FS={n_motor_fs_per_action}, dlpfc_verb={n_dlpfc_verb}")
        print("=" * 60, flush=True)

    # Build architecture: Tier 1 base + Tier 2.3 PFC verb pool
    regions, pathways = build_biological_brain_regions(
        n_lang_input=n_lang_input,
        n_motor_per_action=n_motor_per_action,
        enable_motor_fs=True,
        n_motor_fs_per_action=n_motor_fs_per_action,
        enable_language_output=True,    # Tier 1 bidirectional binding
        n_lang_output=n_lang_input,
        motor_to_language_output_weight=2.0,
        enable_dlpfc_verb=True,         # Tier 2.3 verb pool
        n_dlpfc_verb=n_dlpfc_verb,
    )

    # Build cfg with NMDA + neuromodulator subsystem on
    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = list(regions)
    cfg.region_pathways = list(pathways)
    cfg.dt_ms = 1.0
    cfg.seed = seed
    cfg.enable_nmda = True              # critical for Tier 1 baseline + PFC
    cfg.enable_structural_plasticity = False
    cfg.enable_per_type_stp = False
    cfg.enable_hebbian_learning = False
    cfg.stdp_w_max = 5.0
    cfg.fast_spike_reset = True
    cfg.enable_neuromodulator_subsystem = True
    cfg.neuromodulators = [
        build_tier_2_3_action_gate(drive_pA=action_gate_drive_pA),
    ]

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

    # Topographic prior (Tier 1 BREAKTHROUGH config)
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

    # Wire neuromodulator group registrations (action_gate needs to know
    # which neuron indices belong to each motor_X group)
    rm = bridge.region_manager
    if hasattr(bridge, "neuromodulator_manager") and \
       bridge.neuromodulator_manager is not None:
        bridge.neuromodulator_manager.set_group_indices({
            "motor_N": list(rm.indices("motor_N")),
            "motor_E": list(rm.indices("motor_E")),
            "motor_S": list(rm.indices("motor_S")),
            "motor_W": list(rm.indices("motor_W")),
        })

    # Open all relevant plasticity gates for training
    for gate in (
        "language_input_to_motor",
        "motor_to_language_output",
        "language_input_to_dlpfc_verb",
    ):
        try:
            bridge.set_plasticity_gate(gate, 1.0)
        except Exception:
            pass

    # Build training buffer
    buffer = build_phrase_buffer(
        n_phrase_events=n_phrase_events,
        n_direction_only_events=n_direction_only_events,
        n_verb_only_events=n_verb_only_events,
        rng=rng,
    )

    if verbose:
        print(f"  Total buffer: {len(buffer)} trials\n", flush=True)

    # Pre-compute index arrays
    lang_input_idx = list(rm.indices("language_input"))
    lang_output_idx = list(rm.indices("language_output"))
    motor_idx = {a: list(rm.indices(f"motor_{a}"))
                 for a in ["N", "E", "S", "W"]}
    n_lang_in = len(lang_input_idx)
    lang_input_arr = cp.asarray(lang_input_idx, dtype=cp.int64)
    lang_output_arr = cp.asarray(lang_output_idx, dtype=cp.int64)
    motor_arr = {a: cp.asarray(motor_idx[a], dtype=cp.int64)
                 for a in ["N", "E", "S", "W"]}

    def _drive_for(word: str) -> cp.ndarray:
        """Encode word as sparse drive pattern (CuPy float32)."""
        d = vocab_to_drive_pattern(word, n_neurons=n_lang_in,
                                    drive_max_pA=drive_pA, sparsity=0.1)
        return cp.asarray(d, dtype=cp.float32)

    stats = {"per_type_count": {"phrase": 0, "direction_only": 0,
                                 "verb_only": 0}}
    t0 = time.time()
    for i, trial in enumerate(buffer):
        # Inter-trial reset
        bridge.cp_external_input_current[:] = 0.0
        for _ in range(reset_ms):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1

        ttype = trial["type"]
        stats["per_type_count"][ttype] += 1

        # === PHRASE: verb (100ms) -> direction + motor teacher (100ms) ===
        if ttype == "phrase":
            # Stage 1: drive verb only
            verb_drive = _drive_for(trial["verb"])
            bridge.cp_external_input_current[lang_input_arr] = verb_drive
            for _ in range(verb_drive_ms):
                bridge._run_one_simulation_step()
                bridge.runtime_state.current_time_step += 1
            # Stage 2: drive direction + motor teacher.
            # PFC keeps verb context active via NMDA bistability.
            dir_drive = _drive_for(trial["direction"])
            bridge.cp_external_input_current[:] = 0.0
            bridge.cp_external_input_current[lang_input_arr] = dir_drive
            bridge.cp_external_input_current[lang_output_arr] = dir_drive
            bridge.cp_external_input_current[motor_arr[trial["action"]]] += \
                float(motor_teacher_pA)
            for _ in range(direction_drive_ms):
                bridge._run_one_simulation_step()
                bridge.runtime_state.current_time_step += 1
            # Stage 3: observe (final_observe_ms)
            bridge.cp_external_input_current[:] = 0.0
            for _ in range(final_observe_ms):
                bridge._run_one_simulation_step()
                bridge.runtime_state.current_time_step += 1

        # === DIRECTION-ONLY: standard Tier 1 paradigm ===
        elif ttype == "direction_only":
            dir_drive = _drive_for(trial["direction"])
            bridge.cp_external_input_current[lang_input_arr] = dir_drive
            bridge.cp_external_input_current[lang_output_arr] = dir_drive
            bridge.cp_external_input_current[motor_arr[trial["action"]]] += \
                float(motor_teacher_pA)
            for _ in range(verb_drive_ms + direction_drive_ms):
                bridge._run_one_simulation_step()
                bridge.runtime_state.current_time_step += 1

        # === VERB-ONLY: drive verb, no motor target. PFC fires alone. ===
        elif ttype == "verb_only":
            verb_drive = _drive_for(trial["verb"])
            bridge.cp_external_input_current[lang_input_arr] = verb_drive
            for _ in range(verb_drive_ms + direction_drive_ms):
                bridge._run_one_simulation_step()
                bridge.runtime_state.current_time_step += 1

        if verbose and (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(buffer)} trials, {elapsed:.0f}s]",
                  flush=True)

    # Freeze plasticity for downstream eval
    for gate in (
        "language_input_to_motor",
        "motor_to_language_output",
        "language_input_to_dlpfc_verb",
    ):
        try:
            bridge.set_plasticity_gate(gate, 0.0)
        except Exception:
            pass

    if verbose:
        print(f"\n  Training complete ({time.time()-t0:.0f}s)", flush=True)
        print(f"  Per-type counts: {stats['per_type_count']}", flush=True)

    return bridge, stats


def run_full(
    seed: int = 42,
    n_phrase_events: int = 200,
    n_direction_only_events: int = 100,
    n_verb_only_events: int = 30,
    n_lang_input: int = 2048,
    n_motor_per_action: int = 500,
    n_motor_fs_per_action: int = 60,
    n_dlpfc_verb: int = 200,
    action_gate_drive_pA: float = 50.0,
    verb_drive_ms: int = 100,
    direction_drive_ms: int = 100,
    n_test_per_direction: int = 25,
    n_verb_only_test: int = 25,
    verbose: bool = True,
) -> Dict[str, Any]:
    """End-to-end: train + run all 3 test conditions. Returns unified
    JSON-friendly dict suitable for committing as 6-seed input."""
    from research.runners.phrase_eval import (
        evaluate_phrase, evaluate_direction_only, evaluate_verb_only,
    )

    bridge, stats = run_phrase_training(
        seed=seed,
        n_phrase_events=n_phrase_events,
        n_direction_only_events=n_direction_only_events,
        n_verb_only_events=n_verb_only_events,
        n_lang_input=n_lang_input,
        n_motor_per_action=n_motor_per_action,
        n_motor_fs_per_action=n_motor_fs_per_action,
        n_dlpfc_verb=n_dlpfc_verb,
        action_gate_drive_pA=action_gate_drive_pA,
        verb_drive_ms=verb_drive_ms,
        direction_drive_ms=direction_drive_ms,
        verbose=verbose,
    )

    if verbose:
        print("\n=== PHRASE TEST ===", flush=True)
    res_phrase = evaluate_phrase(
        bridge, n_trials_per_direction=n_test_per_direction,
        verbose=verbose,
    )
    if verbose:
        print(f"  Phrase acc: {res_phrase['accuracy']:.1%}, "
              f"pass={res_phrase['pass']}", flush=True)
        print(f"  Per-direction: {res_phrase['per_direction']}", flush=True)

    if verbose:
        print("\n=== DIRECTION-ONLY TEST (Tier 1 compat) ===", flush=True)
    res_dir = evaluate_direction_only(
        bridge, n_trials_per_direction=n_test_per_direction,
        verbose=verbose,
    )
    if verbose:
        print(f"  Direction-only acc: {res_dir['accuracy']:.1%}, "
              f"pass={res_dir['pass']}", flush=True)

    if verbose:
        print("\n=== VERB-ONLY TEST (anti-action) ===", flush=True)
    res_verb = evaluate_verb_only(
        bridge, n_trials=n_verb_only_test, verbose=verbose,
    )
    if verbose:
        print(f"  Verb-only mean max rate: "
              f"{res_verb['mean_max_motor_rate_hz']:.1f} Hz, "
              f"% quiet: {res_verb['pct_trials_below_threshold']:.0%}, "
              f"pass={res_verb['pass']}", flush=True)

    all_pass = (res_phrase["pass"] and res_dir["pass"]
                and res_verb["pass"])
    if verbose:
        print("\n" + "=" * 60)
        print(f"TIER 2.3 SEED {seed}: "
              f"{'[OK] ALL 3 PASS' if all_pass else '[X] FAIL'}")
        print("=" * 60, flush=True)

    return {
        "seed": seed,
        "stats": stats,
        "phrase": res_phrase,
        "direction_only": res_dir,
        "verb_only": res_verb,
        "all_pass": all_pass,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-phrase-events", type=int, default=200)
    ap.add_argument("--n-direction-only-events", type=int, default=100)
    ap.add_argument("--n-verb-only-events", type=int, default=30)
    ap.add_argument("--n-lang-input", type=int, default=2048)
    ap.add_argument("--n-motor-per-action", type=int, default=500)
    ap.add_argument("--n-motor-fs-per-action", type=int, default=60)
    ap.add_argument("--n-dlpfc-verb", type=int, default=200)
    ap.add_argument("--n-test-per-direction", type=int, default=25)
    ap.add_argument("--n-verb-only-test", type=int, default=25)
    # Tier 2.3 tuning parameters (per Sec 7 of design)
    ap.add_argument("--action-gate-drive-pA", type=float, default=50.0,
                    help="Per-motor-pool current boost when PFC verb "
                         "context is active. 50pA default; try 0/10/100/200 "
                         "for tuning.")
    ap.add_argument("--verb-drive-ms", type=int, default=100,
                    help="Stage 1 verb drive duration. Longer values give "
                         "PFC more time to establish NMDA bistability.")
    ap.add_argument("--direction-drive-ms", type=int, default=100,
                    help="Stage 2 direction drive duration.")
    ap.add_argument("--train-only", action="store_true",
                    help="Skip post-train tests; output stats only")
    ap.add_argument("--out-stats", type=str, default=None,
                    help="JSON output path for training+test stats")
    args = ap.parse_args()

    if args.train_only:
        bridge, stats = run_phrase_training(
            seed=args.seed,
            n_phrase_events=args.n_phrase_events,
            n_direction_only_events=args.n_direction_only_events,
            n_verb_only_events=args.n_verb_only_events,
            n_lang_input=args.n_lang_input,
            n_motor_per_action=args.n_motor_per_action,
            n_motor_fs_per_action=args.n_motor_fs_per_action,
            n_dlpfc_verb=args.n_dlpfc_verb,
            action_gate_drive_pA=args.action_gate_drive_pA,
            verb_drive_ms=args.verb_drive_ms,
            direction_drive_ms=args.direction_drive_ms,
            verbose=True,
        )
        result = {"seed": args.seed, "stats": stats}
    else:
        result = run_full(
            seed=args.seed,
            n_phrase_events=args.n_phrase_events,
            n_direction_only_events=args.n_direction_only_events,
            n_verb_only_events=args.n_verb_only_events,
            n_lang_input=args.n_lang_input,
            n_motor_per_action=args.n_motor_per_action,
            n_motor_fs_per_action=args.n_motor_fs_per_action,
            n_dlpfc_verb=args.n_dlpfc_verb,
            action_gate_drive_pA=args.action_gate_drive_pA,
            verb_drive_ms=args.verb_drive_ms,
            direction_drive_ms=args.direction_drive_ms,
            n_test_per_direction=args.n_test_per_direction,
            n_verb_only_test=args.n_verb_only_test,
            verbose=True,
        )

    if args.out_stats:
        Path(args.out_stats).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_stats).write_text(json.dumps(
            result, indent=2, default=str
        ))
        print(f"\nSaved stats: {args.out_stats}", flush=True)


if __name__ == "__main__":
    main()
