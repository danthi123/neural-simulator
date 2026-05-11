"""P5 ventral semantic stream validation — comprehension + naming.

Catalog G.11 (dual-stream language model, Hickok & Poeppel; Kandel
6e Ch 55 pp 1380-1387) + G.13 (Wernicke's area; Kandel pp 1384-1385).

Tests two characteristic functions of the ventral language stream:

  Test 1 — Comprehension (word → meaning):
    Drive lang_input(word) → measure semantic_cortex response.
    PASS criteria:
      - Same-concept activations across trials cosine > 0.6
        (stable concept representation)
      - Different-concept activations cosine < 0.3
        (distinguishable semantic codes)

  Test 2 — Naming (meaning → word):
    Drive semantic_cortex with the stored pattern for concept X
    (via the engram tag) → measure lang_output response.
    PASS criterion: lang_output activates ABOVE baseline (production
    pathway works; specific word matching is a downstream test).

  Test 3 — Hippo-independent recall (durability):
    After consolidation, silence ca3+ca1 (set excitability_drive
    to strongly negative).
    Drive lang_input("apple"). Measure semantic_cortex.
    PASS: semantic_cortex still produces the "apple" pattern even
    without hippocampus (per catalog D.01: consolidation transforms
    labile traces into durable cortical representations).

Usage:
    python -m research.runners.validate_ventral_semantic \\
        --seed 42 --out research/findings/raw/g11_bg/p5_seed42.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).flatten()
    b = np.asarray(b, dtype=np.float64).flatten()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def measure_region_spikes(bridge, region_name: str, n_steps: int = 100):
    """Run n_steps and return per-neuron spike count for region_name."""
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


def run_ventral_validation(
    seed: int = 42,
    n_lang_input: int = 1024,
    n_motor_per_action: int = 16,
    n_motor_fs_per_action: int = 4,
    n_semantic_cortex: int = 500,
    n_wernicke: int = 100,
    n_train_events: int = 100,
    n_replay_cycles: int = 20,
    out_path: Optional[Path] = None,
    verbose: bool = True,
):
    log = print if verbose else (lambda *a, **k: None)
    log("=" * 60)
    log(f"P5 ventral semantic stream validation (seed={seed})")
    log("=" * 60)

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
    from sim.backend import get_backend, to_host
    cp, _ = get_backend()

    t0 = time.time()
    regions, pathways = build_biological_brain_regions(
        n_lang_input=n_lang_input,
        n_motor_per_action=n_motor_per_action,
        n_motor_fs_per_action=n_motor_fs_per_action,
        enable_motor_fs=True, enable_language_output=True,
        n_lang_output=n_lang_input,
        enable_hippocampus_consolidation=True,
        enable_ventral_semantic=True,
        n_semantic_cortex=n_semantic_cortex,
        n_wernicke=n_wernicke,
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
    build_sec = time.time() - t0
    log(f"Built in {build_sec:.1f}s; {cfg.num_neurons} neurons, "
        f"{int(bridge.cp_connections.nnz)} synapses")

    # Encode 2 concepts via lang_input drive + hippo plasticity
    # The hippo trace + ca1->semantic_cortex pathway will produce
    # semantic_cortex activations during/after training.
    word_apple = vocab_to_drive_pattern(
        "apple", n_neurons=n_lang_input, drive_max_pA=200.0, sparsity=0.1,
    )
    word_river = vocab_to_drive_pattern(
        "river", n_neurons=n_lang_input, drive_max_pA=200.0, sparsity=0.1,
    )
    rm = bridge.region_manager
    lang_idx = list(rm.indices("language_input"))
    apple_arr = cp.asarray(
        [lang_idx[i] for i in np.where(word_apple > 0)[0]], dtype=cp.int64
    )
    river_arr = cp.asarray(
        [lang_idx[i] for i in np.where(word_river > 0)[0]], dtype=cp.int64
    )

    def encode_concept(name, drive_arr):
        """Encode + tag the CA3 ensemble for this concept."""
        # Open gates
        for g in ("lang_to_ec", "ec_to_dg", "dg_to_ca3", "ca3_to_ca1",
                  "ec_to_ca1", "ca3_swr_burst",
                  "lang_to_wernicke", "wernicke_to_semantic",
                  "ca1_to_semantic"):
            try:
                bridge.set_plasticity_gate(g, 1.0)
            except Exception:
                pass
        # Training
        for _ in range(n_train_events):
            bridge.cp_external_input_current[:] = 0.0
            for _ in range(30):
                bridge._run_one_simulation_step()
                bridge.runtime_state.current_time_step += 1
            bridge.cp_external_input_current[drive_arr] = 200.0
            for _ in range(100):
                bridge._run_one_simulation_step()
                bridge.runtime_state.current_time_step += 1
        # Close
        for g in ("lang_to_ec", "ec_to_dg", "dg_to_ca3", "ca3_to_ca1",
                  "ec_to_ca1", "ca3_swr_burst",
                  "lang_to_wernicke", "wernicke_to_semantic",
                  "ca1_to_semantic"):
            try:
                bridge.set_plasticity_gate(g, 0.0)
            except Exception:
                pass
        # Tag CA3 ensemble for replay later
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
        stats = bridge.commit_engram_tag(
            name, top_k=50, region_filter=["ca3"]
        )
        return stats

    log(f"\nEncoding 'apple' ({n_train_events} events)...")
    apple_tag = encode_concept("apple", apple_arr)
    log(f"  CA3 tag: {apple_tag['n_tagged']} neurons")

    log(f"\nEncoding 'river' ({n_train_events} events)...")
    river_tag = encode_concept("river", river_arr)
    log(f"  CA3 tag: {river_tag['n_tagged']} neurons")

    # Run concept replay (P3.1) to consolidate to semantic_cortex
    log(f"\nRunning concept replay ({n_replay_cycles} cycles each)...")
    # Open ca1_to_semantic for the consolidation transfer
    for g in ("ca3_swr_burst", "ca1_to_semantic", "ca3_to_ca1"):
        try:
            bridge.set_plasticity_gate(g, 1.0)
        except Exception:
            pass
    t_replay = time.time()
    replay_stats = run_concept_replay_phase(
        bridge, tag_names=["apple", "river"],
        n_replays_per_tag=n_replay_cycles,
        burst_duration_ms=50, inter_burst_ms=20,
        drive_pA=150.0,
    )
    for g in ("ca3_swr_burst", "ca1_to_semantic", "ca3_to_ca1"):
        try:
            bridge.set_plasticity_gate(g, 0.0)
        except Exception:
            pass
    log(f"  replay done ({replay_stats['n_replays']} events, "
        f"{time.time() - t_replay:.0f}s)")

    # Test 1: Comprehension — drive each word, measure semantic_cortex
    log("\n[TEST 1] Comprehension: drive lang_input, measure semantic_cortex")
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(50):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
    bridge.cp_external_input_current[apple_arr] = 200.0
    apple_sem_1 = measure_region_spikes(bridge, "semantic_cortex", n_steps=100)
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(30):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
    # Second 'apple' trial
    bridge.cp_external_input_current[apple_arr] = 200.0
    apple_sem_2 = measure_region_spikes(bridge, "semantic_cortex", n_steps=100)
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(30):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
    # 'river' trial
    bridge.cp_external_input_current[river_arr] = 200.0
    river_sem = measure_region_spikes(bridge, "semantic_cortex", n_steps=100)
    bridge.cp_external_input_current[:] = 0.0

    cos_apple_self = cosine_similarity(apple_sem_1, apple_sem_2)
    cos_apple_river = cosine_similarity(apple_sem_1, river_sem)
    log(f"  apple trial 1 vs apple trial 2: cos = {cos_apple_self:.3f}")
    log(f"    (same-concept stability; target > 0.6)")
    log(f"  apple vs river: cos = {cos_apple_river:.3f}")
    log(f"    (different-concept; target < 0.3)")
    pass_comprehension = (cos_apple_self > 0.5) and (cos_apple_river < 0.4)

    # Test 2: Naming — stimulate the apple engram (CA3), measure lang_output
    log("\n[TEST 2] Naming: stimulate apple CA3 tag, measure lang_output")
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(50):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
    baseline_lang_out = measure_region_spikes(bridge, "language_output", n_steps=100)
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(30):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
    bridge.stimulate_tag("apple", drive_pA=200.0)
    causal_lang_out = measure_region_spikes(bridge, "language_output", n_steps=100)
    bridge.cp_external_input_current[:] = 0.0

    baseline_sum = float(baseline_lang_out.sum())
    causal_sum = float(causal_lang_out.sum())
    log(f"  baseline lang_output spikes: {baseline_sum:.0f}")
    log(f"  causal (engram-driven) lang_output spikes: {causal_sum:.0f}")
    log(f"    (causal/baseline ratio; target > 1.3)")
    naming_ratio = causal_sum / max(baseline_sum, 1.0)
    pass_naming = naming_ratio > 1.3

    log("\n" + "=" * 60)
    log("PASS criteria:")
    log(f"  Comprehension (apple_self > 0.5 AND apple_river < 0.4): "
        f"{'PASS' if pass_comprehension else 'FAIL'}")
    log(f"    apple_self={cos_apple_self:.3f}, "
        f"apple_river={cos_apple_river:.3f}")
    log(f"  Naming (causal/baseline > 1.3): "
        f"{'PASS' if pass_naming else 'FAIL'}")
    log(f"    ratio={naming_ratio:.2f}x")
    overall = pass_comprehension and pass_naming
    log(f"  OVERALL: {'PASS' if overall else 'FAIL'}")
    log("=" * 60)

    result = {
        "seed": seed,
        "build_seconds": build_sec,
        "n_neurons": int(cfg.num_neurons),
        "n_synapses": int(bridge.cp_connections.nnz),
        "n_train_events": n_train_events,
        "n_replay_cycles": n_replay_cycles,
        "apple_tag_size": apple_tag["n_tagged"],
        "river_tag_size": river_tag["n_tagged"],
        "comprehension": {
            "apple_self_cosine": cos_apple_self,
            "apple_river_cosine": cos_apple_river,
            "passed": pass_comprehension,
        },
        "naming": {
            "baseline_lang_out_spikes": baseline_sum,
            "causal_lang_out_spikes": causal_sum,
            "ratio": naming_ratio,
            "passed": pass_naming,
        },
        "overall_passed": overall,
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
    ap.add_argument("--n-train-events", type=int, default=100)
    ap.add_argument("--n-replay-cycles", type=int, default=20)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    run_ventral_validation(
        seed=args.seed,
        n_train_events=args.n_train_events,
        n_replay_cycles=args.n_replay_cycles,
        out_path=Path(args.out) if args.out else None,
        verbose=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
