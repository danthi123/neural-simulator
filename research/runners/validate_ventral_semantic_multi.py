"""P5 ventral semantic stream multi-concept validation.

Extends iter W (Path A multi-pool wernicke + 400 events) from 2 to N
concepts. Tests architectural scalability of the breakthrough.

The 2-concept iter W achieved 6/6 multi-seed COMPREHENSION PASS.
This variant tests N=4 concepts at the same recipe with N pools.

Catalog G.11 + G.13 (Hickok & Poeppel dual-stream, Wernicke's area).

Usage:
    python -m research.runners.validate_ventral_semantic_multi \\
        --seed 42 --concepts apple,river,alice,table \\
        --n-train-events 400 --n-replay-cycles 40 \\
        --out research/findings/raw/g11_bg/p5_multi_seed42.json

Reuses the same multi-pool wernicke architecture as iter W:
- N wernicke pools (one per concept), 100 neurons each
- N FS pools (12 PV-FS each)
- Cross-pool FS inhibition (each pool's FS inhibits OTHER pools)
- 400 training events per concept (iter W sweet spot)

PASS criteria (per concept):
- COMP self_cosine > mean(cross_cosines_to_other_concepts)
- biology-faithful: margin > 0.03 AND ratio > 1.3
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional, List

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).flatten()
    b = np.asarray(b, dtype=np.float64).flatten()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def index_cosine(a_idx, b_idx, n_total):
    if len(a_idx) == 0 or len(b_idx) == 0:
        return 0.0
    s_a = set(int(x) for x in a_idx)
    s_b = set(int(x) for x in b_idx)
    overlap = len(s_a & s_b)
    return float(overlap / (np.sqrt(len(s_a)) * np.sqrt(len(s_b))))


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


def run_multi_concept_validation(
    concepts: List[str],
    seed: int = 42,
    n_lang_input: int = 1024,
    n_train_events: int = 400,
    n_replay_cycles: int = 40,
    n_per_wernicke_pool: int = 100,
    n_per_wernicke_pool_fs: int = 12,
    out_path: Optional[Path] = None,
    verbose: bool = True,
):
    log = print if verbose else (lambda *a, **k: None)
    n_concepts = len(concepts)
    log("=" * 60)
    log(f"P5 multi-concept validation (seed={seed}, "
        f"N={n_concepts} concepts: {concepts})")
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
        n_motor_per_action=16,
        n_motor_fs_per_action=4,
        enable_motor_fs=True, enable_language_output=True,
        n_lang_output=n_lang_input,
        enable_hippocampus_consolidation=True,
        enable_ventral_semantic=True,
        enable_multi_pool_wernicke=True,
        n_wernicke_pools=n_concepts,
        n_per_wernicke_pool=n_per_wernicke_pool,
        n_per_wernicke_pool_fs=n_per_wernicke_pool_fs,
        n_semantic_cortex=500, n_wernicke=200,  # n_wernicke unused
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

    # Drive patterns for each concept
    rm = bridge.region_manager
    lang_idx = list(rm.indices("language_input"))
    concept_arrays = {}
    for concept in concepts:
        drive = vocab_to_drive_pattern(
            concept, n_neurons=n_lang_input,
            drive_max_pA=200.0, sparsity=0.1,
        )
        arr = cp.asarray(
            [lang_idx[i] for i in np.where(drive > 0)[0]],
            dtype=cp.int64,
        )
        concept_arrays[concept] = arr

    # Gates for multi-pool path
    pool_names = [f"wernicke_pool_{i}" for i in range(n_concepts)]
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
        # Tag CA3 ensemble
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

    log(f"\nEncoding {n_concepts} concepts × {n_train_events} events...")
    tags = {}
    for concept in concepts:
        log(f"  encoding '{concept}'...")
        tags[concept] = encode_concept(concept, concept_arrays[concept])
        log(f"    CA3 tag: {tags[concept]['n_tagged']} neurons")

    # Concept replay
    log(f"\nConcept replay ({n_replay_cycles} cycles/concept)...")
    replay_phase_gates = (
        "ca3_swr_burst", "ca1_to_semantic", "ca3_to_ca1",
        "ca1_to_lang_out",
    ) + tuple(
        [f"semantic_to_{p}" for p in pool_names]
        + [f"{p}_to_lang_out" for p in pool_names]
    )
    for g in replay_phase_gates:
        try:
            bridge.set_plasticity_gate(g, 1.0)
        except Exception:
            pass
    t_replay = time.time()
    run_concept_replay_phase(
        bridge, tag_names=concepts,
        n_replays_per_tag=n_replay_cycles,
        burst_duration_ms=50, inter_burst_ms=20,
        drive_pA=150.0,
    )
    for g in replay_phase_gates:
        try:
            bridge.set_plasticity_gate(g, 0.0)
        except Exception:
            pass
    log(f"  replay done ({time.time() - t_replay:.0f}s)")

    # Test 1: COMP — tag semantic_cortex for each concept,
    # then measure reactivation
    log("\n[TEST 1] Comprehension matrix (pairwise cosines)")
    drive_steps = 100

    def drive_and_tag_semantic(name, drive_arr):
        bridge.cp_external_input_current[:] = 0.0
        for _ in range(50):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
        bridge.start_engram_recording(name)
        bridge.cp_external_input_current[drive_arr] = 200.0
        for _ in range(drive_steps):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
        bridge.cp_external_input_current[:] = 0.0
        return bridge.commit_engram_tag(
            name, top_k=50, region_filter=["semantic_cortex"],
        )

    def measure_semantic_response_indices(drive_arr):
        bridge.cp_external_input_current[:] = 0.0
        for _ in range(30):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
        bridge.cp_external_input_current[drive_arr] = 200.0
        spike_counts = measure_region_spikes(
            bridge, "semantic_cortex", n_steps=drive_steps,
        )
        bridge.cp_external_input_current[:] = 0.0
        return np.where(spike_counts > 0)[0]

    # Tag semantic_cortex ensembles for all concepts
    sem_tags_global = {}
    for concept in concepts:
        tag_name = f"{concept}_semantic"
        drive_and_tag_semantic(tag_name, concept_arrays[concept])
        sem_tags_global[concept] = to_host(
            bridge.get_engram_tag_indices(tag_name)
        )
        log(f"  {concept} sem tag: {len(sem_tags_global[concept])} neurons")

    # Measure reactivation for each concept
    sem_cortex_indices = list(rm.indices("semantic_cortex"))
    reactivations = {}
    for concept in concepts:
        local = measure_semantic_response_indices(concept_arrays[concept])
        glob = np.array(
            [sem_cortex_indices[i] for i in local
             if i < len(sem_cortex_indices)],
            dtype=np.int64,
        )
        reactivations[concept] = glob

    n_total = int(cfg.num_neurons)
    comp_matrix = {}
    log("\n  Comprehension cosine matrix (rows=reactivation, cols=tag):")
    log("            " + "  ".join(f"{c[:6]:>6s}" for c in concepts))
    for c_react in concepts:
        row = []
        for c_tag in concepts:
            cos = index_cosine(reactivations[c_react],
                                sem_tags_global[c_tag], n_total)
            row.append(cos)
            comp_matrix[f"{c_react}_react_vs_{c_tag}_tag"] = cos
        log(f"  {c_react[:8]:>8s}  " + "  ".join(f"{v:.3f} " for v in row))

    # Per-concept COMP PASS: self > all cross
    comp_results = {}
    for c in concepts:
        self_cos = comp_matrix[f"{c}_react_vs_{c}_tag"]
        cross_coses = [comp_matrix[f"{c}_react_vs_{other}_tag"]
                        for other in concepts if other != c]
        mean_cross = sum(cross_coses) / len(cross_coses)
        max_cross = max(cross_coses)
        margin_mean = self_cos - mean_cross
        margin_max = self_cos - max_cross
        ratio = self_cos / max(mean_cross, 0.01)
        comp_results[c] = {
            "self_cosine": self_cos,
            "mean_cross_cosine": mean_cross,
            "max_cross_cosine": max_cross,
            "margin_mean": margin_mean,
            "margin_max": margin_max,
            "ratio_self_to_mean_cross": ratio,
            "pass_strict": margin_max > 0,  # self > all cross
            "pass_biology": margin_mean > 0.03 and ratio > 1.3,
        }

    n_comp_pass_strict = sum(1 for c in concepts
                              if comp_results[c]["pass_strict"])
    n_comp_pass_bio = sum(1 for c in concepts
                          if comp_results[c]["pass_biology"])
    log(f"\n  COMP PASS strict (self > max cross): "
        f"{n_comp_pass_strict}/{n_concepts}")
    log(f"  COMP PASS biology-faithful (margin > 0.03, ratio > 1.3): "
        f"{n_comp_pass_bio}/{n_concepts}")

    log("\n" + "=" * 60)
    overall_pass = (n_comp_pass_bio >= n_concepts * 0.75)
    log(f"  OVERALL ({n_comp_pass_bio}/{n_concepts} biology-faithful): "
        f"{'PASS' if overall_pass else 'PARTIAL'}")
    log("=" * 60)

    result = {
        "seed": seed,
        "concepts": concepts,
        "n_concepts": n_concepts,
        "build_seconds": build_sec,
        "n_neurons": int(cfg.num_neurons),
        "n_synapses": int(bridge.cp_connections.nnz),
        "n_train_events": n_train_events,
        "n_replay_cycles": n_replay_cycles,
        "comp_matrix": comp_matrix,
        "comp_results": comp_results,
        "n_comp_pass_strict": n_comp_pass_strict,
        "n_comp_pass_biology": n_comp_pass_bio,
        "overall_passed": overall_pass,
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
    ap.add_argument("--concepts", type=str,
                    default="apple,river,alice,table",
                    help="Comma-separated concept list")
    ap.add_argument("--n-train-events", type=int, default=400)
    ap.add_argument("--n-replay-cycles", type=int, default=40)
    ap.add_argument("--n-per-wernicke-pool", type=int, default=100)
    ap.add_argument("--n-per-wernicke-pool-fs", type=int, default=12)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    concepts = [c.strip() for c in args.concepts.split(",") if c.strip()]
    run_multi_concept_validation(
        concepts=concepts,
        seed=args.seed,
        n_train_events=args.n_train_events,
        n_replay_cycles=args.n_replay_cycles,
        n_per_wernicke_pool=args.n_per_wernicke_pool,
        n_per_wernicke_pool_fs=args.n_per_wernicke_pool_fs,
        out_path=Path(args.out) if args.out else None,
        verbose=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
