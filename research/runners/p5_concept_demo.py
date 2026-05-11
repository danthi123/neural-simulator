"""P5 ventral semantic concept-recognition demo (iter W architecture).

The first user-facing demonstration of P5's 6/6 multi-seed
COMPREHENSION PASS (iter W breakthrough). Shows the sim can
distinguish concepts via per-concept wernicke pools.

Architecture (iter W, validated 6/6 at multi-seed):
- Path A multi-pool wernicke (2 pools, 100 neurons each)
- Cross-pool PV-FS lateral inhibition
- 400 training events per concept
- Mirror of Tier 1 motor pool pattern at semantic level

Demo flow:
1. Build iter W bridge
2. Train "apple" and "river" comprehension
3. User types "apple" or "river" → sim drives lang_input → measures
   which wernicke_pool fires most → reports recognized concept

Usage:
    python -m research.runners.p5_concept_demo --seed 42

Compared to chat_demo (motor binding for direction words), this
demo shows the sim handling NON-MOTOR abstract concepts via the
catalog G.11/G.13 ventral semantic stream.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import numpy as np


def measure_region_spikes(bridge, region_name: str, n_steps: int = 100):
    from sim.backend import get_backend, to_host
    cp, _ = get_backend()
    rm = bridge.region_manager
    indices = list(rm.indices(region_name))
    arr = cp.asarray(indices, dtype=cp.int64)
    counts = cp.zeros(len(indices), dtype=cp.float32)
    for _ in range(n_steps):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        fired = bridge.cp_firing_states[arr]
        counts += fired.astype(cp.float32)
    return to_host(counts)


def build_iterW_bridge(seed: int = 42, n_lang_input: int = 1024,
                        n_train_events: int = 400,
                        n_replay_cycles: int = 40,
                        verbose: bool = True):
    """Build + train iter W architecture (Path A multi-pool wernicke
    + 400 events) for 2-concept apple/river comprehension."""
    log = print if verbose else (lambda *a, **k: None)
    from sim.config import (CoreSimConfig, RuntimeState, GPUConfig,
                              VisualizationConfig)
    from sim.bridge import SimulationBridge
    from research.runners.text_minimal_isolation import (
        build_biological_brain_regions,
    )
    from research.runners.consolidation_trainer import (
        run_concept_replay_phase,
    )
    from sim.text_embeddings import vocab_to_drive_pattern
    from sim.backend import get_backend
    cp, _ = get_backend()

    t0 = time.time()
    log("Building iter W (Path A multi-pool wernicke) bridge...")
    regions, pathways = build_biological_brain_regions(
        n_lang_input=n_lang_input,
        n_motor_per_action=16, n_motor_fs_per_action=4,
        enable_motor_fs=True, enable_language_output=True,
        n_lang_output=n_lang_input,
        enable_hippocampus_consolidation=True,
        enable_ventral_semantic=True,
        enable_multi_pool_wernicke=True,
        n_wernicke_pools=2,
        n_per_wernicke_pool=100,
        n_per_wernicke_pool_fs=12,
        n_semantic_cortex=500, n_wernicke=200,
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
    log(f"  Built {cfg.num_neurons} neurons, "
        f"{int(bridge.cp_connections.nnz)} synapses "
        f"({time.time() - t0:.1f}s)")

    # Encode 2 concepts
    rm = bridge.region_manager
    lang_idx = list(rm.indices("language_input"))
    concept_arrays = {}
    for concept in ["apple", "river"]:
        drive = vocab_to_drive_pattern(
            concept, n_neurons=n_lang_input,
            drive_max_pA=200.0, sparsity=0.1,
        )
        arr = cp.asarray(
            [lang_idx[i] for i in np.where(drive > 0)[0]],
            dtype=cp.int64,
        )
        concept_arrays[concept] = arr

    pool_names = ["wernicke_pool_0", "wernicke_pool_1"]
    HIPPO_GATES = (
        "lang_to_ec", "ec_to_dg", "dg_to_ca3", "ca3_to_ca1", "ec_to_ca1",
    )
    VENTRAL_GATES = tuple(
        [f"lang_to_{p}" for p in pool_names]
        + [f"{p}_to_semantic" for p in pool_names]
        + ["ca1_to_semantic"]
    )
    PRODUCTION_GATES = tuple(
        [f"semantic_to_{p}" for p in pool_names]
        + [f"{p}_to_lang_out" for p in pool_names]
        + ["ca1_to_lang_out"]
    )
    REPLAY_GATES = ("ca3_swr_burst",)
    encode_gates = (
        HIPPO_GATES + REPLAY_GATES + VENTRAL_GATES + PRODUCTION_GATES
    )

    def encode_concept(name, drive_arr):
        for g in encode_gates:
            try:
                bridge.set_plasticity_gate(g, 1.0)
            except Exception:
                pass
        for _ in range(n_train_events):
            bridge.cp_external_input_current[:] = 0.0
            for _ in range(30):
                bridge._run_one_simulation_step()
                bridge.runtime_state.current_time_step += 1
            bridge.cp_external_input_current[drive_arr] = 200.0
            for _ in range(100):
                bridge._run_one_simulation_step()
                bridge.runtime_state.current_time_step += 1
        for g in encode_gates:
            try:
                bridge.set_plasticity_gate(g, 0.0)
            except Exception:
                pass
        bridge.cp_external_input_current[:] = 0.0
        for _ in range(30):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
        bridge.start_engram_recording(name)
        bridge.cp_external_input_current[drive_arr] = 200.0
        for _ in range(100):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
        bridge.cp_external_input_current[:] = 0.0
        return bridge.commit_engram_tag(
            name, top_k=50, region_filter=["ca3"]
        )

    log(f"Training apple ({n_train_events} events)...")
    encode_concept("apple", concept_arrays["apple"])
    log(f"Training river ({n_train_events} events)...")
    encode_concept("river", concept_arrays["river"])

    log(f"Replay phase ({n_replay_cycles} cycles each concept)...")
    replay_phase_gates = (
        "ca3_swr_burst", "ca1_to_semantic", "ca3_to_ca1",
        "ca1_to_lang_out",
        "semantic_to_wernicke_pool_0", "semantic_to_wernicke_pool_1",
        "wernicke_pool_0_to_lang_out", "wernicke_pool_1_to_lang_out",
    )
    for g in replay_phase_gates:
        try:
            bridge.set_plasticity_gate(g, 1.0)
        except Exception:
            pass
    run_concept_replay_phase(
        bridge, tag_names=["apple", "river"],
        n_replays_per_tag=n_replay_cycles,
        burst_duration_ms=50, inter_burst_ms=20,
        drive_pA=150.0,
    )
    for g in replay_phase_gates:
        try:
            bridge.set_plasticity_gate(g, 0.0)
        except Exception:
            pass

    log(f"Training complete ({time.time() - t0:.0f}s total)")
    return bridge, concept_arrays


def recognize_concept(bridge, concept_arrays, word: str,
                       drive_steps: int = 100, verbose: bool = True):
    """Drive lang_input(word), measure wernicke_pool firing, return
    recognized concept (the pool that fired most)."""
    log = print if verbose else (lambda *a, **k: None)
    from sim.backend import get_backend
    cp, _ = get_backend()
    if word not in concept_arrays:
        log(f"  '{word}' is not in the trained vocabulary "
            f"({list(concept_arrays.keys())})")
        return None
    drive_arr = concept_arrays[word]
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(30):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
    bridge.cp_external_input_current[drive_arr] = 200.0
    pool_0_spikes = measure_region_spikes(
        bridge, "wernicke_pool_0", n_steps=drive_steps,
    )
    bridge.cp_external_input_current[:] = 0.0
    # Reset before second drive
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(30):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
    bridge.cp_external_input_current[drive_arr] = 200.0
    pool_1_spikes = measure_region_spikes(
        bridge, "wernicke_pool_1", n_steps=drive_steps,
    )
    bridge.cp_external_input_current[:] = 0.0

    pool_0_total = float(pool_0_spikes.sum())
    pool_1_total = float(pool_1_spikes.sum())
    log(f"  pool_0 (apple) spikes: {pool_0_total:.0f}")
    log(f"  pool_1 (river) spikes: {pool_1_total:.0f}")

    if pool_0_total > pool_1_total:
        recognized = "apple"
        confidence = pool_0_total / (pool_0_total + pool_1_total)
    else:
        recognized = "river"
        confidence = pool_1_total / (pool_0_total + pool_1_total)
    log(f"  -> recognized: '{recognized}' (confidence {confidence:.1%})")
    return recognized, confidence


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-train-events", type=int, default=400)
    ap.add_argument("--n-replay-cycles", type=int, default=40)
    ap.add_argument("--test-words", type=str,
                    default="apple,river,apple,river",
                    help="Comma-separated test sequence")
    args = ap.parse_args()

    bridge, concept_arrays = build_iterW_bridge(
        seed=args.seed,
        n_train_events=args.n_train_events,
        n_replay_cycles=args.n_replay_cycles,
    )

    test_words = [w.strip() for w in args.test_words.split(",") if w.strip()]
    print("\n" + "=" * 60)
    print("P5 concept-recognition demo (iter W architecture)")
    print("=" * 60)
    n_correct = 0
    for word in test_words:
        print(f"\nUser drives lang_input('{word}')...")
        result = recognize_concept(bridge, concept_arrays, word)
        if result is not None:
            recognized, _ = result
            correct = (recognized == word)
            print(f"  expected: '{word}', got: '{recognized}' — "
                  f"{'CORRECT' if correct else 'WRONG'}")
            if correct:
                n_correct += 1
    print(f"\n{'=' * 60}")
    print(f"Accuracy: {n_correct}/{len(test_words)} "
          f"({100*n_correct/max(len(test_words),1):.0f}%)")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
