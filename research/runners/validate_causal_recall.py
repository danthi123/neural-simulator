"""Liu 2012-style causal recall test (P2 validation).

Catalog D.14 (engram cells, Tonegawa) + T1.C behavioral validation.

Tonegawa et al's optogenetic memory paradigm:
  1. Train context-A → reward; tag the active engram ensemble
  2. Place mouse in context-B (no actual training association)
  3. Stimulate the tagged ensemble optogenetically
  4. Observe reward-conditioned behavior in context-B
     (the mouse "remembers" being rewarded because the engram
     ensemble was reactivated)

In our sim:
  1. Train hippo+cortex: drive lang_input(word) → trains
     ca1→motor pathway via STDP; tag the active CA3 ensemble
     as "word_X"
  2. Test causal recall: stimulate ONLY the CA3 engram tag
     (no lang_input drive) → measure motor pool activity
  3. PASS criterion: motor pool corresponding to the trained
     direction fires MORE than other motor pools when only the
     CA3 engram tag is stimulated.

This is the engram-tagging API's behavioral validation per
catalog D.14 + roadmap T1.C.

Usage:
    python -m research.runners.validate_causal_recall \\
        --seed 42 --out research/findings/raw/g11_bg/causal_seed42.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np


def measure_motor_response(bridge, n_steps: int = 100):
    """Measure firing rate in each motor_X pool over n_steps.
    Returns {N, E, S, W: spike_count}."""
    from sim.backend import get_backend, to_host
    cp, _ = get_backend()
    rm = bridge.region_manager
    motor_indices = {}
    motor_counts = {}
    for action in "NESW":
        try:
            mi = list(rm.indices(f"motor_{action}"))
            motor_indices[action] = cp.asarray(mi, dtype=cp.int64)
            motor_counts[action] = cp.zeros(len(mi), dtype=cp.float32)
        except Exception:
            return None
    for _ in range(n_steps):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        for action in "NESW":
            fired = bridge.cp_firing_states[motor_indices[action]]
            motor_counts[action] += fired.astype(cp.float32)
    return {a: float(to_host(motor_counts[a]).sum()) for a in "NESW"}


def run_causal_recall(
    seed: int = 42,
    n_lang_input: int = 1024,
    n_motor_per_action: int = 32,
    n_motor_fs_per_action: int = 8,
    train_events: int = 200,
    drive_pA: float = 200.0,
    teacher_pA: float = 1500.0,
    out_path: Optional[Path] = None,
    verbose: bool = True,
):
    """Encode word→motor binding via hippo path, tag the CA3 ensemble,
    then test causal recall by stimulating only the tag.
    """
    log = print if verbose else (lambda *a, **k: None)
    log("=" * 60)
    log(f"Liu 2012-style causal recall test (seed={seed})")
    log("=" * 60)

    from sim.config import (CoreSimConfig, RuntimeState, GPUConfig,
                              VisualizationConfig)
    from sim.bridge import SimulationBridge
    from research.runners.text_minimal_isolation import (
        build_biological_brain_regions,
    )
    from sim.backend import get_backend, to_host
    from sim.text_embeddings import vocab_to_drive_pattern
    cp, _ = get_backend()

    t0 = time.time()
    regions, pathways = build_biological_brain_regions(
        n_lang_input=n_lang_input,
        n_motor_per_action=n_motor_per_action,
        n_motor_fs_per_action=n_motor_fs_per_action,
        enable_motor_fs=True, enable_language_output=True,
        n_lang_output=n_lang_input,
        enable_hippocampus_consolidation=True,
        n_ec=200, n_dg=800, n_dg_pv_basket=240,
        n_ca3=400, n_ca1=200,
        ca3_recurrent_weight=5.0,
    )
    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = list(regions)
    cfg.region_pathways = list(pathways)
    cfg.dt_ms = 1.0
    cfg.seed = seed
    cfg.enable_nmda = True
    cfg.fast_spike_reset = True
    cfg.stdp_w_max = 10.0
    cfg.enable_hebbian_learning = False

    bridge = SimulationBridge(
        core_config=cfg, viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(), gpu_config=GPUConfig(),
    )
    bridge.runtime_state.max_delay_steps = int(
        cfg.max_synaptic_delay_ms / cfg.dt_ms
    )
    bridge._initialize_simulation_data(called_from_playback_init=False)
    log(f"Built in {time.time() - t0:.1f}s; {cfg.num_neurons} neurons")

    # Encode "north" → motor_N via paired co-firing
    # (lang_input + motor_N teacher)
    target_action = "N"
    word_drive = vocab_to_drive_pattern(
        "north", n_neurons=n_lang_input, drive_max_pA=drive_pA,
        sparsity=0.1,
    )
    word_active_local = np.where(word_drive > 0)[0]
    rm = bridge.region_manager
    lang_idx = list(rm.indices("language_input"))
    word_global = np.array([lang_idx[i] for i in word_active_local],
                             dtype=np.int64)
    word_arr = cp.asarray(word_global, dtype=cp.int64)
    motor_idx = list(rm.indices(f"motor_{target_action}"))
    motor_arr = cp.asarray(motor_idx, dtype=cp.int64)

    # Open all hippo + lang->motor plasticity
    for g in ("lang_to_ec", "ec_to_dg", "dg_to_ca3", "ca3_to_ca1",
              "ca3_swr_burst", "ec_to_ca1",
              "ca1_to_motor", "language_input_to_motor"):
        try:
            bridge.set_plasticity_gate(g, 1.0)
        except Exception:
            pass

    log(f"\nTraining 'north' -> motor_N for {train_events} events...")
    for ev in range(train_events):
        bridge.cp_external_input_current[:] = 0.0
        for _ in range(30):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
        bridge.cp_external_input_current[word_arr] = float(drive_pA)
        bridge.cp_external_input_current[motor_arr] += float(teacher_pA)
        for _ in range(100):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1

    # Close all plasticity gates
    for g in ("lang_to_ec", "ec_to_dg", "dg_to_ca3", "ca3_to_ca1",
              "ca3_swr_burst", "ec_to_ca1",
              "ca1_to_motor", "language_input_to_motor"):
        try:
            bridge.set_plasticity_gate(g, 0.0)
        except Exception:
            pass

    log("  training complete")

    # Tag the CA3 ensemble fired during a final fresh drive of 'north'
    log("\nTagging CA3 ensemble for 'north' (recall pass only)...")
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(50):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
    bridge.start_engram_recording("north_engram")
    bridge.cp_external_input_current[word_arr] = float(drive_pA)
    for _ in range(100):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
    bridge.cp_external_input_current[:] = 0.0
    stats = bridge.commit_engram_tag(
        "north_engram", top_k=50, region_filter=["ca3"],
    )
    log(f"  tagged {stats['n_tagged']} CA3 neurons")

    # Baseline: motor response with NO drive
    log("\n[BASELINE] Measuring motor response with no drive...")
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(50):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
    baseline_motor = measure_motor_response(bridge, n_steps=100)
    log(f"  baseline motor spikes: {baseline_motor}")

    # Word-driven recall: drive 'north' through lang_input
    log("\n[WORD-DRIVEN] Drive lang_input('north'), measure motor...")
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(50):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
    bridge.cp_external_input_current[word_arr] = float(drive_pA)
    word_motor = measure_motor_response(bridge, n_steps=100)
    bridge.cp_external_input_current[:] = 0.0
    log(f"  word-driven motor: {word_motor}")

    # Causal recall: drive ONLY the CA3 engram tag (no lang_input)
    log("\n[CAUSAL] Drive only CA3 engram tag, measure motor...")
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(50):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
    bridge.stimulate_tag("north_engram", drive_pA=drive_pA)
    causal_motor = measure_motor_response(bridge, n_steps=100)
    bridge.cp_external_input_current[:] = 0.0
    log(f"  causal-recall motor: {causal_motor}")

    # Analysis
    target = target_action
    target_baseline = baseline_motor[target]
    target_word = word_motor[target]
    target_causal = causal_motor[target]
    # Other motor pools (non-target)
    other_baseline = sum(baseline_motor[a] for a in "NESW" if a != target) / 3
    other_word = sum(word_motor[a] for a in "NESW" if a != target) / 3
    other_causal = sum(causal_motor[a] for a in "NESW" if a != target) / 3

    log("\n" + "=" * 60)
    log(f"Target action: motor_{target}")
    log(f"Target spike counts: baseline={target_baseline:.0f}, "
        f"word={target_word:.0f}, causal={target_causal:.0f}")
    log(f"Mean other-pool counts: baseline={other_baseline:.0f}, "
        f"word={other_word:.0f}, causal={other_causal:.0f}")

    # PASS criteria for Liu 2012-style causal recall:
    # - Word-driven recall: motor_target > motor_others (training worked)
    # - Causal recall: motor_target > motor_others (engram tag drives
    #   downstream pathway)
    # - Causal selectivity: ratio target/other > 1.5 (significant
    #   preference for the trained target)
    pass_word = target_word > other_word * 1.3
    causal_ratio = (target_causal / max(other_causal, 1e-6))
    pass_causal = causal_ratio > 1.5

    log(f"\nWord-driven selectivity: target/other = "
        f"{target_word / max(other_word, 1e-6):.2f}x "
        f"{'PASS' if pass_word else 'FAIL'}")
    log(f"Causal-recall selectivity: target/other = {causal_ratio:.2f}x "
        f"{'PASS' if pass_causal else 'FAIL'}")
    log(f"OVERALL: {'PASS' if (pass_word and pass_causal) else 'FAIL'}")
    log("=" * 60)

    result = {
        "seed": seed,
        "target_action": target,
        "train_events": train_events,
        "n_engram_neurons": stats["n_tagged"],
        "baseline_motor": baseline_motor,
        "word_driven_motor": word_motor,
        "causal_recall_motor": causal_motor,
        "target_baseline": target_baseline,
        "target_word": target_word,
        "target_causal": target_causal,
        "other_baseline": other_baseline,
        "other_word": other_word,
        "other_causal": other_causal,
        "pass_word_driven": pass_word,
        "pass_causal_recall": pass_causal,
        "causal_recall_ratio": causal_ratio,
        "overall_passed": pass_word and pass_causal,
        "total_seconds": time.time() - t0,
    }

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, default=str),
                              encoding="utf-8")
        log(f"\n[OUT] {out_path}")

    return result


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-events", type=int, default=200)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    run_causal_recall(
        seed=args.seed,
        train_events=args.train_events,
        out_path=Path(args.out) if args.out else None,
        verbose=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
