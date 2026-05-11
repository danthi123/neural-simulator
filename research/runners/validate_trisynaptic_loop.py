"""Validate hippocampal trisynaptic loop (P1 of realigned plan v3).

Catalog entries:
  D.03 Trisynaptic pathway — EC layer II → DG → CA3 → CA1 (indirect)
       Kandel 6e Ch 54 pp 1340–1342, Fig 54-1.
  D.12 Pattern separation — DG sparsifies overlapping inputs
       Kandel 6e Ch 54 pp 1357–1360.
  D.13 Pattern completion — CA3 recurrents reconstruct full pattern
       Kandel 6e Ch 54 pp 1342, 1360–1361. Marr 1971 autoassociator.

Roadmap T1.A — Hippocampal trisynaptic loop.

The existing `build_biological_brain_regions(
enable_hippocampus_consolidation=True)` builder in
text_minimal_isolation.py already wires the trisynaptic structure
(EC → DG → CA3 → CA1 with EC → CA1 direct bypass and CA3 recurrent
attractor). What's NEVER been done: validate that this circuit
exhibits the two characteristic functional properties:

  Test 1 — Pattern separation (D.12):
    Present 2 highly similar EC input patterns (cosine ~0.9).
    Measure cosine similarity of DG output ensembles.
    PASS criterion: DG cosine < 0.5 (orthogonalized by >40%).
    Per Kandel: DG should achieve "expansion recoding" via sparse
    coding (~2-5% active) + strong feedforward inhibition (PV basket
    cells).

  Test 2 — Pattern completion (D.13):
    Train CA3 by co-firing a full pattern (P) via mossy fiber drive
    + recurrent plasticity ON for N events.
    Present a partial cue (P_partial = first 50% of P's drive).
    Measure CA3 output: does it converge to the full P?
    PASS criterion: cosine(CA3_output_partial, CA3_output_full) > 0.7.

Optional Test 3 — CA1 readout integrates CA3 + direct EC:
    Drive EC and CA3 separately; measure CA1 response.
    PASS criterion: CA1 fires with both pathways open and either
    pathway alone produces partial response (linear-ish summation).

Usage:
    SIM_BACKEND=cupy python -m research.runners.validate_trisynaptic_loop \\
        --seed 42 --out research/findings/raw/g11_bg/trisynaptic_seed42.json

    # Multi-seed
    for s in 42 43 44; do
        python -m research.runners.validate_trisynaptic_loop \\
            --seed $s --out research/findings/raw/g11_bg/trisynaptic_seed${s}.json
    done

Wall-clock budget: ~3-5 min per seed (much smaller than full Tier 1
training; no language binding needed, just pattern recall).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two firing-rate vectors (host arrays)."""
    a = np.asarray(a, dtype=np.float64).flatten()
    b = np.asarray(b, dtype=np.float64).flatten()
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def measure_region_response(
    bridge,
    region_name: str,
    drive_indices,
    drive_pA: float,
    drive_region: str = "language_input",
    n_steps: int = 100,
    reset_steps: int = 50,
):
    """Drive `drive_region` at `drive_indices` with `drive_pA`, run for
    n_steps, return per-neuron firing-rate vector of `region_name`."""
    from sim.backend import get_backend, to_host
    cp, _ = get_backend()

    rm = bridge.region_manager
    drive_arr = cp.asarray(drive_indices, dtype=cp.int64)
    region_indices = list(rm.indices(region_name))
    region_arr = cp.asarray(region_indices, dtype=cp.int64)

    # Reset transients
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(reset_steps):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1

    # Capture firing counts via cp_firing_states
    # We'll sample by accumulating during the drive window
    drive_values = cp.full(len(drive_indices), drive_pA, dtype=cp.float32)
    spike_counts = cp.zeros(len(region_indices), dtype=cp.float32)

    bridge.cp_external_input_current[drive_arr] = drive_values
    for _ in range(n_steps):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        # Track spikes in the readout region
        fired = bridge.cp_firing_states[region_arr]
        spike_counts += fired.astype(cp.float32)

    bridge.cp_external_input_current[drive_arr] = 0.0
    return to_host(spike_counts)


def build_drive_pattern(n_neurons: int, sparsity: float, seed: int) -> np.ndarray:
    """Random sparse activation pattern: pick `sparsity * n_neurons`
    neurons to be active. Returns indices."""
    rng = np.random.default_rng(seed)
    n_active = max(1, int(round(sparsity * n_neurons)))
    indices = rng.choice(n_neurons, size=n_active, replace=False)
    return indices


def overlap_drive_patterns(
    n_neurons: int,
    sparsity: float,
    seed: int,
    overlap_frac: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return two drive patterns with `overlap_frac` of indices shared.

    overlap_frac=1.0 → identical patterns
    overlap_frac=0.0 → completely disjoint
    overlap_frac=0.8 → 80% shared (high similarity for D.12 test)
    """
    n_active = max(1, int(round(sparsity * n_neurons)))
    n_shared = int(round(overlap_frac * n_active))
    n_unique = n_active - n_shared

    rng_a = np.random.default_rng(seed)
    rng_b = np.random.default_rng(seed + 1)
    # Shared core
    shared = rng_a.choice(n_neurons, size=n_shared, replace=False)
    # Pattern A: shared + unique_a
    remaining_a = np.setdiff1d(np.arange(n_neurons), shared)
    unique_a = rng_a.choice(remaining_a, size=n_unique, replace=False) \
        if n_unique > 0 else np.array([], dtype=np.int64)
    # Pattern B: shared + unique_b
    remaining_b = np.setdiff1d(remaining_a, unique_a)
    unique_b = rng_b.choice(remaining_b, size=n_unique, replace=False) \
        if n_unique > 0 else np.array([], dtype=np.int64)

    a = np.concatenate([shared, unique_a]) if n_unique > 0 else shared
    b = np.concatenate([shared, unique_b]) if n_unique > 0 else shared
    return a.astype(np.int64), b.astype(np.int64)


# ──────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────


def test_pattern_separation(
    bridge,
    n_lang_input: int,
    n_dg: int,
    seed: int = 42,
    overlap_frac: float = 0.8,
    drive_pA: float = 200.0,
    verbose: bool = True,
):
    """D.12 — Pattern separation test.

    Present 2 EC inputs with 80% overlap; measure cosine of DG outputs.
    PASS: DG cosine < 0.5 (significant orthogonalization).
    """
    log = print if verbose else (lambda *a, **k: None)
    log("\n[D.12] Pattern separation test")
    log(f"  Generating 2 EC patterns with overlap_frac={overlap_frac}")

    drive_a, drive_b = overlap_drive_patterns(
        n_neurons=n_lang_input, sparsity=0.1,
        seed=seed, overlap_frac=overlap_frac,
    )

    # Sanity: input cosine should be near overlap_frac
    in_vec_a = np.zeros(n_lang_input)
    in_vec_a[drive_a] = 1.0
    in_vec_b = np.zeros(n_lang_input)
    in_vec_b[drive_b] = 1.0
    input_cos = cosine_similarity(in_vec_a, in_vec_b)
    log(f"  Input cosine: {input_cos:.3f} (expected ~{overlap_frac})")

    # Measure DG response to each pattern
    dg_a = measure_region_response(
        bridge, "dg", drive_a, drive_pA=drive_pA,
        drive_region="language_input", n_steps=100,
    )
    dg_b = measure_region_response(
        bridge, "dg", drive_b, drive_pA=drive_pA,
        drive_region="language_input", n_steps=100,
    )
    dg_cos = cosine_similarity(dg_a, dg_b)
    log(f"  DG output cosine: {dg_cos:.3f}")

    # Sparsity check
    dg_sparsity_a = float(np.mean(dg_a > 0))
    dg_sparsity_b = float(np.mean(dg_b > 0))
    log(f"  DG sparsity: A={dg_sparsity_a:.3f}, B={dg_sparsity_b:.3f} "
        f"(target 0.02–0.05)")

    passed = dg_cos < 0.5
    log(f"  PASS criterion (DG cos < 0.5): {'PASS' if passed else 'FAIL'}")

    return {
        "test": "pattern_separation_D12",
        "input_cosine": input_cos,
        "dg_cosine": dg_cos,
        "dg_sparsity_a": dg_sparsity_a,
        "dg_sparsity_b": dg_sparsity_b,
        "orthogonalization": input_cos - dg_cos,
        "passed": passed,
        "drive_a_size": int(len(drive_a)),
        "drive_b_size": int(len(drive_b)),
    }


def test_pattern_completion(
    bridge,
    n_lang_input: int,
    n_ca3: int,
    seed: int = 42,
    train_events: int = 30,
    partial_frac: float = 0.5,
    drive_pA: float = 200.0,
    verbose: bool = True,
):
    """D.13 — Pattern completion test.

    1. Pick a full EC drive pattern P.
    2. Co-fire P + open ca3_swr_burst gate to strengthen CA3 recurrents
       via STDP (autoassociator learning).
    3. Present a partial cue P_partial (first partial_frac of P's indices).
    4. Measure CA3 output; compare to full-cue CA3 output.
    PASS: cosine(CA3_partial, CA3_full) > 0.7.
    """
    log = print if verbose else (lambda *a, **k: None)
    log("\n[D.13] Pattern completion test")
    log(f"  Training CA3 attractor over {train_events} events")

    drive_full = build_drive_pattern(
        n_neurons=n_lang_input, sparsity=0.1, seed=seed,
    )
    n_full = len(drive_full)
    n_partial = max(1, int(round(partial_frac * n_full)))
    drive_partial = drive_full[:n_partial]
    log(f"  Full drive: {n_full} neurons; partial: {n_partial} "
        f"({partial_frac*100:.0f}%)")

    # Open ca3 recurrent plasticity for training
    try:
        bridge.set_plasticity_gate("ca3_swr_burst", 1.0)
        bridge.set_plasticity_gate("dg_to_ca3", 1.0)
        bridge.set_plasticity_gate("ec_to_dg", 1.0)
        bridge.set_plasticity_gate("lang_to_ec", 1.0)
    except Exception:
        pass

    # Training: present full pattern N times, let STDP strengthen CA3
    # recurrents.
    from sim.backend import get_backend
    cp, _ = get_backend()
    drive_arr_full = cp.asarray(drive_full, dtype=cp.int64)

    for ev in range(train_events):
        bridge.cp_external_input_current[:] = 0.0
        # Reset
        for _ in range(30):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
        # Drive
        bridge.cp_external_input_current[drive_arr_full] = float(drive_pA)
        for _ in range(100):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1

    # Close training gates
    try:
        bridge.set_plasticity_gate("ca3_swr_burst", 0.0)
        bridge.set_plasticity_gate("dg_to_ca3", 0.0)
        bridge.set_plasticity_gate("ec_to_dg", 0.0)
        bridge.set_plasticity_gate("lang_to_ec", 0.0)
    except Exception:
        pass

    log("  Training complete; measuring recall")

    # Recall with full cue
    ca3_full = measure_region_response(
        bridge, "ca3", drive_full, drive_pA=drive_pA,
        drive_region="language_input", n_steps=100,
    )
    # Recall with partial cue
    ca3_partial = measure_region_response(
        bridge, "ca3", drive_partial, drive_pA=drive_pA,
        drive_region="language_input", n_steps=100,
    )

    cos = cosine_similarity(ca3_full, ca3_partial)
    n_active_full = float(np.mean(ca3_full > 0))
    n_active_partial = float(np.mean(ca3_partial > 0))
    log(f"  CA3 full cue active: {n_active_full:.3f}; "
        f"partial cue active: {n_active_partial:.3f}")
    log(f"  CA3 cosine(partial vs full): {cos:.3f}")

    passed = cos > 0.7
    log(f"  PASS criterion (cosine > 0.7): {'PASS' if passed else 'FAIL'}")

    return {
        "test": "pattern_completion_D13",
        "train_events": train_events,
        "partial_frac": partial_frac,
        "ca3_cosine_partial_vs_full": cos,
        "ca3_active_full": n_active_full,
        "ca3_active_partial": n_active_partial,
        "passed": passed,
    }


# ──────────────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────────────


def run_validation(
    seed: int = 42,
    n_lang_input: int = 2048,
    n_dg: int = 800,
    n_ca3: int = 400,
    n_ca1: int = 200,
    n_ec: int = 200,
    n_dg_pv_basket: int = 240,  # 30% of DG, biology-grounded FFi ratio
    ca3_recurrent_density: float = 0.30,
    ca3_recurrent_weight: float = 1.5,
    overlap_frac: float = 0.8,
    train_events: int = 30,
    out_path: Optional[Path] = None,
    verbose: bool = True,
) -> dict:
    """Build a hippocampus-enabled bridge and run D.12 + D.13 tests."""
    log = print if verbose else (lambda *a, **k: None)
    log("=" * 60)
    log(f"P1 — Trisynaptic loop validation (seed={seed})")
    log("=" * 60)

    t0 = time.time()

    # Build bridge using existing biological brain regions builder
    from sim.config import (
        CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig,
    )
    from sim.bridge import SimulationBridge
    from research.runners.text_minimal_isolation import (
        build_biological_brain_regions,
    )

    log(f"\nBuilding hippocampus-enabled bridge:")
    log(f"  n_lang_input={n_lang_input}, n_ec={n_ec}")
    log(f"  n_dg={n_dg}, n_dg_pv_basket={n_dg_pv_basket}")
    log(f"  n_ca3={n_ca3}, n_ca1={n_ca1}")

    regions, pathways = build_biological_brain_regions(
        n_lang_input=n_lang_input,
        n_motor_per_action=16,  # minimal motor (we don't use these here)
        n_motor_fs_per_action=4,
        enable_motor_fs=True,
        enable_language_output=True,
        n_lang_output=n_lang_input,
        enable_hippocampus_consolidation=True,
        n_ec=n_ec,
        n_dg=n_dg,
        n_dg_pv_basket=n_dg_pv_basket,
        n_ca3=n_ca3,
        n_ca1=n_ca1,
        ca3_recurrent_density=ca3_recurrent_density,
        ca3_recurrent_weight=ca3_recurrent_weight,
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
    cfg.stdp_w_max = 10.0  # higher cap for autoassociator strengthening
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
    build_sec = time.time() - t0
    log(f"  Built in {build_sec:.1f}s; {cfg.num_neurons} neurons, "
        f"{int(bridge.cp_connections.nnz)} synapses")

    # Run tests
    results = {
        "seed": seed,
        "build_seconds": build_sec,
        "n_neurons": int(cfg.num_neurons),
        "n_synapses": int(bridge.cp_connections.nnz),
        "n_lang_input": n_lang_input,
        "n_ec": n_ec, "n_dg": n_dg, "n_dg_pv_basket": n_dg_pv_basket,
        "n_ca3": n_ca3, "n_ca1": n_ca1,
        "overlap_frac": overlap_frac,
        "train_events": train_events,
        "tests": [],
    }

    t1 = time.time()
    sep_result = test_pattern_separation(
        bridge, n_lang_input=n_lang_input, n_dg=n_dg,
        seed=seed, overlap_frac=overlap_frac, verbose=verbose,
    )
    sep_result["elapsed_seconds"] = time.time() - t1
    results["tests"].append(sep_result)

    t2 = time.time()
    comp_result = test_pattern_completion(
        bridge, n_lang_input=n_lang_input, n_ca3=n_ca3,
        seed=seed, train_events=train_events, verbose=verbose,
    )
    comp_result["elapsed_seconds"] = time.time() - t2
    results["tests"].append(comp_result)

    n_passed = sum(1 for t in results["tests"] if t["passed"])
    n_total = len(results["tests"])
    results["n_passed"] = n_passed
    results["n_total"] = n_total
    results["all_passed"] = (n_passed == n_total)
    results["total_seconds"] = time.time() - t0

    log("=" * 60)
    log(f"Summary: {n_passed}/{n_total} tests PASS "
        f"({results['total_seconds']:.0f}s total)")
    log("=" * 60)

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2, default=str),
                              encoding="utf-8")
        log(f"\n[OUT] {out_path}")

    return results


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-lang-input", type=int, default=2048)
    ap.add_argument("--n-ec", type=int, default=200)
    ap.add_argument("--n-dg", type=int, default=800)
    ap.add_argument("--n-dg-pv-basket", type=int, default=240)
    ap.add_argument("--n-ca3", type=int, default=400)
    ap.add_argument("--n-ca1", type=int, default=200)
    ap.add_argument("--overlap-frac", type=float, default=0.8)
    ap.add_argument("--train-events", type=int, default=30)
    ap.add_argument("--ca3-recurrent-density", type=float, default=0.30)
    ap.add_argument("--ca3-recurrent-weight", type=float, default=1.5)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    run_validation(
        seed=args.seed,
        n_lang_input=args.n_lang_input,
        n_ec=args.n_ec,
        n_dg=args.n_dg,
        n_dg_pv_basket=args.n_dg_pv_basket,
        n_ca3=args.n_ca3,
        n_ca1=args.n_ca1,
        ca3_recurrent_density=args.ca3_recurrent_density,
        ca3_recurrent_weight=args.ca3_recurrent_weight,
        overlap_frac=args.overlap_frac,
        train_events=args.train_events,
        out_path=Path(args.out) if args.out else None,
        verbose=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
