"""Phase 1.3 + Tier 2.1 combined: hippocampus consolidation at synonym scale.

Validates that McClelland 1995 / Buzsaki 2013 complementary learning
systems theory holds for the Tier 2.1 8-word synonym vocab, not just
Tier 1's 4-word primary vocab.

Design: docs/plans/2026-05-07-Phase1.3-Tier2.1-combined-design.md

Setup:
- Architecture: Tier 2.1 v4 scale-up (n_lang=4096, n_motor=1000,
  n_motor_fs=120) + hippocampus_consolidation enabled
- Vocab: 8 words ({north,up}, {east,right}, {south,down}, {west,left})
- Awake/sleep alternation per Phase 1.3 protocol

Eval:
- Pre-silence W->A on all 8 synonyms (baseline)
- Hippo-OFF W->A on all 8 synonyms (cortex-alone retention)
- Pass criterion: ratio >= 50% (matches Phase 1.3 threshold)

Usage:
    python -m research.runners.consolidation_synonym_trainer \\
        --seed 42 --n-awake-events-per-word 400 \\
        --n-sleep-swr-events 200 --consolidation-interval 4 \\
        --n-test-per-word 25 \\
        --out-stats research/findings/raw/g11_bg/consol_syn_seed42.json

Wall-clock (corrected 2026-05-07 EDT after empirical observation):
- --smoke         ~21 min/seed (50 events/word x 4 words, 50 SWR
                  events/cycle, 12 chunks total; eval 10 trials/word)
- --medium        ~80 min/seed (200 events/word, 100 SWR, 50 chunks)
- default (full)  ~6.5 HOURS/seed (400 events/word, 200 SWR, 100
                  chunks; each chunk + sleep ~3.7 min at Tier 2.1
                  v4 scale-up arch with NMDA enabled)

Original design plan estimate of 30-45 min for full was wrong by
5-9x. The miscalculation: I scaled wall-clock with awake events
linearly but didn't account for SWR events per chunk (50 -> 200 =
4x more) AND chunk count (12 -> 100 = 8x more). Total work ~32x
more than smoke. Multi-seed full (3 seeds) = ~19 hours; killed
mid-seed-42 on first attempt.

Recommendation:
- --smoke   for runner validation (~21 min)
- --medium  for quick multi-seed (~4 hrs / 3 seeds)
- default   only for overnight or multi-day validation runs

Smoke validated 2026-05-07: seed 42 retention 111% overall (hippo-OFF
higher than pre-silence). See
research/findings/2026-05-07-consolidation-synonym-smoke-seed42.md.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np


# Synonym groups — must match research.runners.text_eval.get_synonym_groups(8)
SYNONYM_GROUPS = {
    "N": ["north", "up"],
    "E": ["east", "right"],
    "S": ["south", "down"],
    "W": ["west", "left"],
}


def run_consolidation_synonym_training(
    seed: int = 42,
    n_awake_events_per_word: int = 400,
    n_sleep_swr_events: int = 200,
    consolidation_interval: int = 4,
    n_lang_input: int = 4096,
    n_motor_per_action: int = 1000,
    n_motor_fs_per_action: int = 120,
    swr_drive_pA: float = 100.0,
    smoke: bool = False,
    medium: bool = False,
    verbose: bool = True,
):
    """Train Tier 2.1 synonym vocab with hippocampus consolidation.

    Each awake trial picks a random synonym for the target action
    (matching bio_three_factor synonym_mode behavior). Both
    primary and synonym words bind to the same motor pool via
    embodied Hebbian co-firing.

    Sleep phases run SWR replay to consolidate language_input ->
    cortex -> motor patterns.

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
        apply_topographic_bias as _apply_topo,
        set_awake_gates, set_sleep_gates, freeze_all_gates,
    )
    from research.runners.consolidation_trainer import run_swr_replay_phase

    if smoke:
        # Reduced for fast smoke validation (~21 min/seed empirically;
        # initially documented "~5 min" but Tier 2.1 v4 arch with NMDA
        # is heavier than the Tier 1 smoke equivalent)
        n_awake_events_per_word = min(n_awake_events_per_word, 50)
        n_sleep_swr_events = min(n_sleep_swr_events, 50)
    elif medium:
        # Medium mode (added 2026-05-07): a feasible multi-seed config
        # between smoke (~21 min) and default full (~6.5 HOURS). 200
        # events/word + 100 SWR events/cycle + 50 chunks = ~80 min/seed,
        # so 3-seed validation completes in ~4 hrs. Useful when you want
        # more training than smoke but can't justify a multi-day full
        # run.
        n_awake_events_per_word = min(n_awake_events_per_word, 200)
        n_sleep_swr_events = min(n_sleep_swr_events, 100)

    rng = np.random.default_rng(seed)

    if verbose:
        print("=" * 60)
        print(f"CONSOLIDATION SYNONYM TRAINER (Phase 1.3 + Tier 2.1, "
              f"seed={seed})")
        print(f"  Vocab: {SYNONYM_GROUPS}")
        print(f"  Awake events/word: {n_awake_events_per_word}")
        print(f"  Sleep SWR events: {n_sleep_swr_events}")
        print(f"  Consolidation interval: every {consolidation_interval} "
              f"awake events/word")
        print(f"  Architecture: n_lang={n_lang_input}, "
              f"n_motor={n_motor_per_action}, "
              f"n_motor_fs={n_motor_fs_per_action}")
        if smoke:
            print("  *** SMOKE MODE *** (reduced events for fast validation)")
        print("=" * 60, flush=True)

    # Build architecture: Tier 2.1 scale-up + hippocampus
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

    # Topographic prior on language_input -> motor (Tier 2.1 BREAKTHROUGH)
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

    # Build awake training buffer with synonym mode
    # Each trial picks a random synonym for the target action.
    awake_buffer = []
    for action, synonyms in SYNONYM_GROUPS.items():
        for _ in range(n_awake_events_per_word):
            token = synonyms[rng.integers(0, len(synonyms))]
            awake_buffer.append({"token": token, "action": action})
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
            bridge.cp_external_input_current[:] = 0.0
            for _ in range(reset_steps):
                bridge._run_one_simulation_step()
                bridge.runtime_state.current_time_step += 1
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

        # Sleep phase: SWR replay to consolidate
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

    freeze_all_gates(bridge)

    return bridge, {
        "n_awake_events": len(awake_buffer),
        "n_sleep_phases": n_sleep_phases_run,
        "n_total_swr_events": n_sleep_phases_run * n_sleep_swr_events,
        "wall_clock_seconds": time.time() - t0,
        "vocab_size": 8,
        "synonym_groups": SYNONYM_GROUPS,
    }


def run_full(
    seed: int = 42,
    n_awake_events_per_word: int = 400,
    n_sleep_swr_events: int = 200,
    consolidation_interval: int = 4,
    n_lang_input: int = 4096,
    n_motor_per_action: int = 1000,
    n_motor_fs_per_action: int = 120,
    n_test_per_word: int = 25,
    smoke: bool = False,
    medium: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """End-to-end: train + pre-silence eval + hippo-OFF eval.

    Returns unified JSON-friendly dict with retention ratio for both
    overall accuracy and primary-vs-synonym split.
    """
    from research.runners.consolidation_eval import evaluate_with_hippo_off
    from research.runners.text_eval import evaluate_word_to_action

    bridge, stats = run_consolidation_synonym_training(
        seed=seed,
        n_awake_events_per_word=n_awake_events_per_word,
        n_sleep_swr_events=n_sleep_swr_events,
        consolidation_interval=consolidation_interval,
        n_lang_input=n_lang_input,
        n_motor_per_action=n_motor_per_action,
        n_motor_fs_per_action=n_motor_fs_per_action,
        smoke=smoke,
        medium=medium,
        verbose=verbose,
    )

    if verbose:
        print("\n=== PRE-SILENCE EVAL (synonym mode) ===", flush=True)

    # Pre-silence: hippo present, eval on all 8 synonyms
    pre_eval = evaluate_word_to_action(
        bridge,
        n_trials_per_word=n_test_per_word,
        stim_steps_per_trial=100,
        n_reset_steps=50,
        token_sparsity=0.1,
        synonym_mode=True,
        synonym_vocab_size=8,
        verbose=verbose,
    )
    pre_acc = pre_eval["accuracy"]

    # Split primary vs synonym accuracy from confusion matrix.
    # evaluate_word_to_action returns confusion_matrix[word][action] = count;
    # per-word accuracy = confusion[word][correct_action] / sum(confusion[word]).
    primary_words = ["north", "east", "south", "west"]
    synonym_words = ["up", "right", "down", "left"]
    word_to_action = {"north": "N", "up": "N", "east": "E", "right": "E",
                       "south": "S", "down": "S", "west": "W", "left": "W"}

    def _per_word_acc(eval_result, word):
        cm = eval_result.get("confusion_matrix", {}).get(word, {})
        total = sum(cm.values())
        if total == 0:
            return 0.0
        return cm.get(word_to_action[word], 0) / total

    pre_per_word = {w: _per_word_acc(pre_eval, w)
                    for w in primary_words + synonym_words}
    pre_primary = float(np.mean([pre_per_word[w] for w in primary_words]))
    pre_synonym = float(np.mean([pre_per_word[w] for w in synonym_words]))

    if verbose:
        print(f"  Pre-silence overall: {pre_acc:.1%}")
        print(f"  Pre-silence primary: {pre_primary:.1%}")
        print(f"  Pre-silence synonym: {pre_synonym:.1%}")
        print("\n=== HIPPO-OFF EVAL (synonym mode) ===", flush=True)

    # Hippo-OFF: silence hippocampus, eval on all 8 synonyms
    # We need a synonym-aware version; manually inline the silence wrapper
    # because evaluate_with_hippo_off doesn't expose synonym_mode.
    import cupy as cp
    rm = bridge.region_manager
    HIPPO_REGIONS = ["ec", "dg", "dg_pv_basket", "ca3", "ca1"]
    hippo_indices = []
    for region_name in HIPPO_REGIONS:
        try:
            idx = rm.indices(region_name)
            if idx is not None:
                hippo_indices.extend(list(idx))
        except Exception:
            pass

    if not hippo_indices:
        if verbose:
            print("  WARN: no hippocampus regions; hippo-OFF eval skipped")
        post_eval = {"accuracy": 0.0, "per_word_accuracy": {}}
    else:
        hippo_arr = cp.asarray(hippo_indices, dtype=cp.int64)
        original_step = bridge._run_one_simulation_step
        silence_pA = -200.0

        def silenced_step():
            bridge.cp_external_input_current[hippo_arr] = float(silence_pA)
            return original_step()

        bridge._run_one_simulation_step = silenced_step
        try:
            post_eval = evaluate_word_to_action(
                bridge,
                n_trials_per_word=n_test_per_word,
                stim_steps_per_trial=100,
                n_reset_steps=50,
                token_sparsity=0.1,
                synonym_mode=True,
                synonym_vocab_size=8,
                verbose=verbose,
            )
        finally:
            bridge._run_one_simulation_step = original_step
            bridge.cp_external_input_current[hippo_arr] = 0.0

    post_acc = post_eval["accuracy"]
    post_per_word = {w: _per_word_acc(post_eval, w)
                     for w in primary_words + synonym_words}
    post_primary = float(np.mean([post_per_word[w] for w in primary_words]))
    post_synonym = float(np.mean([post_per_word[w] for w in synonym_words]))

    overall_ratio = (post_acc / pre_acc) if pre_acc > 0 else 0.0
    primary_ratio = (post_primary / pre_primary) if pre_primary > 0 else 0.0
    synonym_ratio = (post_synonym / pre_synonym) if pre_synonym > 0 else 0.0

    # Pass criteria from design plan:
    #   primary retention >= 80%, synonym retention >= 60%
    primary_pass = primary_ratio >= 0.80
    synonym_pass = synonym_ratio >= 0.60
    if primary_pass and synonym_pass:
        verdict = "GO"
    elif primary_pass:
        verdict = "PARTIAL (primary consolidates, synonym does not)"
    else:
        verdict = "NO-GO (architectural insight: Tier 2.1 sub-pop binding " \
                  "doesn't survive consolidation)"

    if verbose:
        print(f"\n  Hippo-OFF overall: {post_acc:.1%}")
        print(f"  Hippo-OFF primary: {post_primary:.1%}")
        print(f"  Hippo-OFF synonym: {post_synonym:.1%}")
        print(f"  Retention overall: {overall_ratio:.0%}")
        print(f"  Retention primary: {primary_ratio:.0%} "
              f"({'PASS' if primary_pass else 'FAIL'})")
        print(f"  Retention synonym: {synonym_ratio:.0%} "
              f"({'PASS' if synonym_pass else 'FAIL'})")
        print("\n" + "=" * 60)
        print(f"PHASE 1.3 + Tier 2.1 SEED {seed}: {verdict}")
        print("=" * 60, flush=True)

    return {
        "seed": seed,
        "smoke": smoke,
        "stats": stats,
        "pre_silence": {
            "accuracy": pre_acc,
            "primary_acc": pre_primary,
            "synonym_acc": pre_synonym,
            "per_word": pre_per_word,
        },
        "hippo_off": {
            "accuracy": post_acc,
            "primary_acc": post_primary,
            "synonym_acc": post_synonym,
            "per_word": post_per_word,
        },
        "retention": {
            "overall": overall_ratio,
            "primary": primary_ratio,
            "synonym": synonym_ratio,
            "primary_pass": primary_pass,
            "synonym_pass": synonym_pass,
        },
        "verdict": verdict,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-awake-events-per-word", type=int, default=400)
    ap.add_argument("--n-sleep-swr-events", type=int, default=200)
    ap.add_argument("--consolidation-interval", type=int, default=4)
    ap.add_argument("--n-lang-input", type=int, default=4096)
    ap.add_argument("--n-motor-per-action", type=int, default=1000)
    ap.add_argument("--n-motor-fs-per-action", type=int, default=120)
    ap.add_argument("--n-test-per-word", type=int, default=25)
    ap.add_argument("--smoke", action="store_true",
                    help="Smoke mode: 50 events/word, 50 SWR events, "
                         "12 chunks (~21 min/seed)")
    ap.add_argument("--medium", action="store_true",
                    help="Medium mode: 200 events/word, 100 SWR events, "
                         "50 chunks (~80 min/seed). Feasible for 3-seed "
                         "validation in ~4 hrs vs default's ~19 hrs")
    ap.add_argument("--out-stats", type=str, default=None)
    args = ap.parse_args()

    if args.smoke and args.medium:
        ap.error("--smoke and --medium are mutually exclusive")

    result = run_full(
        seed=args.seed,
        n_awake_events_per_word=args.n_awake_events_per_word,
        n_sleep_swr_events=args.n_sleep_swr_events,
        consolidation_interval=args.consolidation_interval,
        n_lang_input=args.n_lang_input,
        n_motor_per_action=args.n_motor_per_action,
        n_motor_fs_per_action=args.n_motor_fs_per_action,
        n_test_per_word=args.n_test_per_word,
        smoke=args.smoke,
        medium=args.medium,
        verbose=True,
    )

    if args.out_stats:
        Path(args.out_stats).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_stats).write_text(
            json.dumps(result, indent=2, default=str)
        )
        print(f"Saved: {args.out_stats}", flush=True)


if __name__ == "__main__":
    main()
