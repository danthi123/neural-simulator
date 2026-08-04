#!/usr/bin/env python3
"""Hash the preregistered V14 Stage-A default-off compatibility cells."""

from __future__ import annotations

import hashlib
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "numpy")
SOURCE_ROOT = Path(
    os.environ.get("SIM_SOURCE_ROOT", Path(__file__).resolve().parents[1])
).resolve()
sys.path.insert(0, str(SOURCE_ROOT))

import numpy as np

from sim.backend import to_host
from sim.bridge import SimulationBridge
from sim.config import CoreSimConfig, GPUConfig, RuntimeState, VisualizationConfig
from sim.enums import NeuronModel


SEEDS = (193883, 261805, 768106, 929013, 736887, 366590)


def _bridge(seed: int) -> SimulationBridge:
    config = CoreSimConfig(
        num_neurons=16,
        connections_per_neuron=0,
        seed=seed,
        neuron_model_type=NeuronModel.HODGKIN_HUXLEY.name,
        default_neuron_type_hh="HH_EXCITATORY_DEFAULT_LEGACY",
        dt_ms=0.05,
        enable_parameter_heterogeneity=False,
        enable_conductance_noise=False,
        enable_hebbian_learning=False,
        enable_short_term_plasticity=False,
        enable_structural_plasticity=False,
        enable_ou_process=False,
        hh_external_drive_scale=0.0,
    )
    bridge = SimulationBridge(
        core_config=config,
        viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(),
        gpu_config=GPUConfig(enable_profiling=False),
    )
    bridge._initialize_simulation_data()
    if not bridge.is_initialized:
        raise RuntimeError("bridge initialization failed")
    return bridge


def _hash_arrays(arrays) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        host = np.ascontiguousarray(to_host(array))
        digest.update(str(host.dtype).encode("ascii"))
        digest.update(np.asarray(host.shape, dtype=np.int64).tobytes())
        digest.update(host.tobytes())
    return digest.hexdigest()


def _advance(bridge: SimulationBridge, steps: int):
    raster = []
    for _ in range(steps):
        bridge._run_one_simulation_step()
        raster.append(to_host(bridge.cp_firing_states).copy())
        bridge.runtime_state.current_time_step += 1
        bridge.runtime_state.current_time_ms += bridge.core_config.dt_ms
    return np.stack(raster)


def run_seed(seed: int) -> dict:
    bridge = _bridge(seed)
    raster = _advance(bridge, 100)
    trajectory_hash = _hash_arrays((
        raster,
        bridge.cp_membrane_potential_v,
        bridge.cp_gating_variable_m,
        bridge.cp_gating_variable_h,
        bridge.cp_gating_variable_n,
        bridge.cp_connections.data,
    ))
    with tempfile.TemporaryDirectory(prefix="v14-stageA-") as directory:
        checkpoint = Path(directory) / "default-off.simstate.h5"
        if not bridge.save_checkpoint(str(checkpoint)):
            raise RuntimeError("checkpoint save failed")
        uninterrupted = _advance(bridge, 25)
        uninterrupted_hash = _hash_arrays((
            uninterrupted,
            bridge.cp_membrane_potential_v,
            bridge.cp_gating_variable_m,
            bridge.cp_gating_variable_h,
            bridge.cp_gating_variable_n,
        ))
        restored = _bridge(seed)
        if not restored.load_checkpoint(str(checkpoint)):
            raise RuntimeError("checkpoint load failed")
        continued = _advance(restored, 25)
        continuation_hash = _hash_arrays((
            continued,
            restored.cp_membrane_potential_v,
            restored.cp_gating_variable_m,
            restored.cp_gating_variable_h,
            restored.cp_gating_variable_n,
        ))
    bundle_allocated = any(
        getattr(bridge, name, None) is not None
        for name in (
            "cp_snr_g_nalcn_max", "cp_snr_g_nap_max", "cp_snr_g_ca_max",
            "cp_snr_g_sk_max", "cp_snr_g_h_max",
            "cp_snr_nap_activation", "cp_snr_nap_inactivation",
            "cp_snr_ca_activation", "cp_snr_ca_inactivation",
            "cp_snr_calcium", "cp_snr_sk_activation", "cp_snr_h_activation",
        )
    )
    return {
        "seed": seed,
        "trajectory_sha256": trajectory_hash,
        "uninterrupted_sha256": uninterrupted_hash,
        "continued_sha256": continuation_hash,
        "checkpoint_exact": uninterrupted_hash == continuation_hash,
        "bundle_allocated": bundle_allocated,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    cells = [run_seed(seed) for seed in SEEDS]
    result = {
        "schema": "v14-stageA-default-off-compatibility-v1",
        "label": args.label,
        "runner": str(Path(__file__).resolve()),
        "argv": sys.argv,
        "seeds": list(SEEDS),
        "source_root": str(SOURCE_ROOT),
        "source_revision": subprocess.check_output(
            ["git", "-C", str(SOURCE_ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_dirty": bool(subprocess.check_output(
            ["git", "-C", str(SOURCE_ROOT), "status", "--porcelain"], text=True
        ).strip()),
        "backend": os.environ.get("SIM_BACKEND", "auto"),
        "cells": cells,
        "all_checkpoint_exact": all(cell["checkpoint_exact"] for cell in cells),
        "all_bundle_unallocated": not any(cell["bundle_allocated"] for cell in cells),
    }
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
