"""Preregistered Gate B v13 Stage-0 tonic-output substrate executor."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile
import time

import h5py
import numpy as np

from research.runners._vocal_action_selector_gate import (
    _indices,
    _set_equal_tonic_current,
    build_selector_bridge,
    selector_config,
)
from sim import (
    CoreSimConfig,
    GPUConfig,
    NeuronModel,
    RuntimeState,
    SimulationBridge,
    VisualizationConfig,
)
from sim.backend import get_backend, is_gpu_backend, synchronize, to_host
from sim.enums import NeuronType
from sim.regions import BrainRegion, RegionPathway
from tools.lab import assert_backend, project_cost
from tools.verdict import Verdict


ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = ROOT / "research/specs/v13_tonic_output_substrate.json"
RUNNER_PATH = Path(__file__).resolve()
COMPATIBILITY_ROOT = ROOT / "research/findings/raw/v13_deterministic_compatibility"
COMPATIBILITY_CORRECTION_PATH = COMPATIBILITY_ROOT / "comparison-baseline-vs-candidate.json"
COMPATIBILITY_RUNNER_PATH = ROOT / "research/runners/_v13_deterministic_compatibility.py"
COMPATIBILITY_SPEC_PATH = ROOT / "research/specs/v13_tonic_output_deterministic_compatibility.json"
DETERMINISTIC_PATCH_ID = "18bd23624a3247cb0f205795081b7a540c15ed89"
PARTITIONS = {
    "audit_only": [314159],
    "compatibility": [271828],
    "calibration": [1013],
    "replication": [1019],
    "held_out": [1021],
    "reserved_for_stage1": [1031],
}
LADDER_PA = [75, 100, 125, 150, 175]
HETEROGENEITY = {
    "izh_a_val": {
        "type": "lognormal",
        "mean_log": -2.995732273553991,
        "sigma_log": 0.15,
    },
    "izh_b_val": {"type": "gaussian", "mean": 2.0, "std": 0.3},
    "izh_d_val": {"type": "gaussian", "mean": 25.0, "std": 3.75},
    "izh_C_val": {"type": "gaussian", "mean": 60.0, "std": 9.0},
}
CRITERIA = {
    "minimum_bin_rate_hz": 40.0,
    "maximum_bin_rate_hz": 80.0,
    "all_cells_fire": True,
    "maximum_same_step_fraction": 0.25,
    "minimum_distinct_first_spike_steps": 8,
    "minimum_first_spike_span_ms": 8.0,
    "external_current_exact_zero": True,
    "intrinsic_vector_immutable": True,
}
DEFAULT_HASHES = {
    "numpy": {
        "raster": "4bfec7fa4c4865db6e31dced73d3d1385820682cf062439a547190853ef3c79d",
        "v": "2e848ed2673a192408f118d66fcded3cf9a21719ae909ef4190eab9fc76ff54b",
        "u": "cd1b254e7482a26b6a3054777edd411d7a7deff30d3bb1207ae2e474da7b7313",
        "g_e": "33e31475a067f8ac34cc85462b2db8386191d2101eda91421f06d61c80c29b3a",
        "g_i": "65b329f4d6992523da618e1ce43aca126ca0ceb1d339bfa8af6c07e21dc81890",
    },
    "cupy": {
        "raster": "690867e2c44ac456ee1f3a0cb8db9addeef8448753170b587561767c6e51ec2b",
        "v": "d1706d17f1a1136a57672546fb643e10f991476c32098ae1906b3b3ec88683df",
        "u": "f319dbcfcb1d09f983ad86ddf912484820d3fce94a2b93e135d73b0219c96317",
        "g_e": "11fc8612831e72007d6540dc997d2159d833448d1a3d73b5a29b90f267ba29bc",
        "g_i": "90b3c1c3825eba353c19bbf18017254a498007bdb8cf2cbab4e59acacd61f305",
    },
}
WEIGHT_HASH = "a9021bcda62b216e67ff1c14c46b011b8590056352514e8672234613b6704b82"
EXTERNAL_HASH = "4b36942def742bbd214715a7d3e387fb111051ff213935eff1b346e346c2c551"


def _hash(value) -> str:
    array = np.ascontiguousarray(np.asarray(to_host(value)))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def _source_identity() -> dict[str, str]:
    paths = (
        RUNNER_PATH,
        ROOT / "sim/bridge.py",
        ROOT / "sim/regions.py",
        ROOT / "sim/kernels.py",
        SPEC_PATH,
    )
    return {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def _earned_verdict(label: str, go: bool, requirements: dict[str, bool]) -> dict:
    verdict = Verdict(label)
    for name, measured in requirements.items():
        verdict.require(name, bool(measured), expect=True)
    decided = verdict.decide(go=bool(go), verbose=False)
    return {
        "verdict_status": decided["status"],
        "preconditions": decided["preconditions"],
        "disabled_processes": decided["disabled_processes"],
        "undefined_reasons": decided["undefined_reasons"],
        "go": decided["go"],
    }


def _outcome(prefix: str, verdict: dict) -> str:
    status = verdict["verdict_status"]
    if status == "UNDEFINED":
        return f"{prefix}_UNDEFINED"
    return f"{prefix}_GO" if verdict["go"] else f"{prefix}_NO_GO"


def _artifact_verdict_is_earned(artifact: dict) -> bool:
    preconditions = artifact.get("preconditions")
    return bool(
        isinstance(preconditions, list)
        and preconditions
        and all(item.get("ok") is True for item in preconditions)
        and not artifact.get("undefined_reasons")
        and artifact.get("verdict_status") in ("GO", "NO-GO")
    )


def _canonical_artifact_digest(artifact: dict) -> str:
    payload = json.dumps(artifact, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _load_compatibility_correction(path: Path) -> dict:
    path = path.resolve()
    if path != COMPATIBILITY_CORRECTION_PATH.resolve():
        raise ValueError(
            f"compatibility correction must be {COMPATIBILITY_CORRECTION_PATH}"
        )
    artifact = json.loads(path.read_text())
    expected_identity = {
        "runner_sha256": hashlib.sha256(COMPATIBILITY_RUNNER_PATH.read_bytes()).hexdigest(),
        "spec_sha256": hashlib.sha256(COMPATIBILITY_SPEC_PATH.read_bytes()).hexdigest(),
    }
    valid = (
        artifact.get("schema") == "v13-deterministic-compatibility-comparison-v1"
        and artifact.get("stage") == "cross_twin_compare"
        and artifact.get("outcome") == "DETERMINISTIC_COMPATIBILITY_GO"
        and artifact.get("go") is True
        and artifact.get("verdict_status") == "GO"
        and _artifact_verdict_is_earned(artifact)
        and artifact.get("acceptance_checks") == {
            "all_seven_hashes_exact_across_twins": True,
            "topology_exact_across_twins": True,
        }
        and artifact.get("deterministic_patch_id") == DETERMINISTIC_PATCH_ID
        and artifact.get("executor_identity") == expected_identity
    )
    if not valid:
        raise ValueError("compatibility correction is missing, stale, or not an earned exact GO")

    bundles = {
        "baseline": COMPATIBILITY_ROOT / "bundle-baseline_8994_plus_deterministic_patch.json",
        "candidate": COMPATIBILITY_ROOT / "bundle-candidate_v13.json",
    }
    for label, bundle_path in bundles.items():
        bundle = json.loads(bundle_path.read_text())
        if (
            _canonical_artifact_digest(bundle) != artifact[f"{label}_bundle_sha256"]
            or bundle.get("go") is not True
            or not _artifact_verdict_is_earned(bundle)
        ):
            raise ValueError(f"{label} compatibility bundle is missing or changed")

    candidate = json.loads(bundles["candidate"].read_text())
    candidate_sha = candidate.get("source_identity", {}).get("git_sha")
    if not candidate_sha:
        raise ValueError("candidate compatibility bundle has no source revision")
    critical_paths = (
        "sim/bridge.py", "sim/regions.py", "sim/kernels.py",
        "research/runners/_vocal_action_selector_gate.py",
        "research/specs/v13_tonic_output_substrate.json",
    )
    changed = subprocess.run(
        ["git", "diff", "--name-only", f"{candidate_sha}..HEAD", "--", *critical_paths],
        cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout.splitlines()
    if changed:
        raise ValueError(f"compatibility-covered simulator inputs changed: {changed}")

    baseline_relative = str(bundles["baseline"].relative_to(ROOT))
    committed_baseline = subprocess.run(
        ["git", "show", f"{candidate_sha}:{baseline_relative}"],
        cwd=ROOT, check=True, capture_output=True,
    ).stdout
    if committed_baseline != bundles["baseline"].read_bytes():
        raise ValueError("candidate source did not contain the sealed baseline bundle")

    baseline_contract = json.loads(bundles["baseline"].read_text()).get(
        "source_twin_contract", {}
    )
    candidate_contract = candidate.get("source_twin_contract", {})
    baseline_cells = [
        json.loads(item.read_text())
        for item in sorted(COMPATIBILITY_ROOT.glob("cell-baseline_8994_plus_deterministic_patch-*.json"))
        if not item.name.endswith(".prov.json")
    ]
    candidate_cells = [
        json.loads(item.read_text())
        for item in sorted(COMPATIBILITY_ROOT.glob("cell-candidate_v13-*.json"))
        if not item.name.endswith(".prov.json")
    ]
    intrinsic_states_valid = (
        len(baseline_cells) == 36
        and len(candidate_cells) == 36
        and baseline_contract.get("contract_valid") is True
        and baseline_contract.get("brain_region_has_intrinsic_field") is False
        and candidate_contract.get("contract_valid") is True
        and candidate_contract.get("brain_region_has_intrinsic_field") is True
        and all(
            item.get("intrinsic_default_state", {}).get("bridge_state")
            == "attribute_absent" for item in baseline_cells
        )
        and all(
            item.get("intrinsic_default_state", {}).get("bridge_value_is_none")
            is True for item in candidate_cells
        )
    )
    if not intrinsic_states_valid:
        raise ValueError("source twins do not carry the required feature-absent/default-None states")
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": _canonical_artifact_digest(artifact),
        "candidate_source_sha": candidate_sha,
        "baseline_bundle_present_in_candidate_source": True,
        "twin_intrinsic_states_valid": True,
        "deterministic_patch_id": DETERMINISTIC_PATCH_ID,
        "outcome": artifact["outcome"],
    }


def _assert_source_sealed() -> None:
    relative = [str(path.relative_to(ROOT)) for path in (
        RUNNER_PATH, ROOT / "sim/bridge.py", ROOT / "sim/regions.py",
        ROOT / "sim/kernels.py", SPEC_PATH,
    )]
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", *relative], cwd=ROOT,
        capture_output=True, text=True,
    )
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", *relative], cwd=ROOT,
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    if tracked.returncode != 0 or dirty:
        raise RuntimeError(
            "held-out execution requires committed, clean runner/spec/substrate source"
        )


def _backend_info() -> dict:
    requested = os.environ.get("SIM_BACKEND")
    if requested not in ("numpy", "cupy"):
        raise ValueError("SIM_BACKEND must be explicitly set to numpy or cupy")
    assert_backend(requested, note="Gate B v13 Stage 0")
    xp, actual = get_backend()
    if actual != requested:
        raise RuntimeError(f"requested {requested}, resolved {actual}")
    result = {
        "backend": actual,
        "device": "CPU (NumPy backend)",
        "host": platform.node(),
    }
    if actual == "cupy":
        props = xp.cuda.runtime.getDeviceProperties(0)
        name = props["name"]
        result["device"] = (
            name.decode("utf-8", errors="replace")
            if isinstance(name, bytes) else str(name)
        )
    return result


def load_locked_spec() -> dict:
    spec = json.loads(SPEC_PATH.read_text())
    checks = {
        "id": spec.get("id") == "gateB-v13-stage0-tonic-output-substrate",
        "status": spec.get("status") == "preregistered",
        "partitions": spec.get("partitions") == PARTITIONS,
        "ladder": spec["calibration"].get("intrinsic_current_ladder_pA") == LADDER_PA,
        "criteria": spec["calibration"].get("criteria") == CRITERIA,
        "heterogeneity": (
            spec["population"].get("heterogeneity_distributions") == HETEROGENEITY
        ),
        "backends": spec.get("backends") == ["numpy", "cupy"],
        "field": spec["implementation"].get("region_field") == "intrinsic_current_pA",
    }
    if not all(checks.values()):
        raise ValueError(f"runner/spec disagreement: {checks}")
    return spec


def _read_only_config(seed: int, regions, pathways=(), *, step_mode="normal"):
    regions = list(regions)
    cfg = CoreSimConfig(
        num_neurons=sum(int(region.n_neurons) for region in regions),
        neuron_model_type=NeuronModel.IZHIKEVICH.name,
        neural_profile_name="GENERIC_UNSTRUCTURED",
        seed=int(seed),
        dt_ms=1.0,
        enable_brain_region_framework=True,
        brain_regions=regions,
        region_pathways=list(pathways),
        enable_parameter_heterogeneity=False,
        heterogeneity_seed=int(seed),
        heterogeneity_distributions=json.loads(json.dumps(HETEROGENEITY)),
        enable_ou_process=False,
        enable_conductance_noise=False,
        enable_hebbian_learning=False,
        enable_short_term_plasticity=False,
        enable_homeostasis=False,
        enable_stdp=False,
        enable_inhibitory_stdp=False,
        enable_structural_plasticity=False,
        enable_reward_modulation=False,
        enable_neuromodulator_subsystem=False,
        enable_nmda=False,
        enable_nmda_recurrent=False,
        enable_gabab=False,
        enable_coincidence_detection=False,
        enable_graded_dendritic_plateau=False,
        read_only_fast_step=True,
        fast_spike_reset=True,
        enable_step_megakernel=(step_mode == "v1"),
        enable_step_cudagraph=False,
        enable_step_megakernel_v2=(step_mode == "v2"),
    )
    return cfg


def _gpi_region(n: int, current_pA: float, *, name="gpi_snr"):
    return BrainRegion(
        name=name,
        n_neurons=int(n),
        exc_fraction=0.0,
        internal_density=0.1,
        exc_weight_mean=0.0,
        inh_weight_mean=0.0,
        weight_jitter=0.0,
        plastic_internal=False,
        izh_neuron_type=NeuronType.IZH2007_GPI_OUTPUT.name,
        intrinsic_current_pA=float(current_pA),
        enable_nmda=False,
        enable_homeostasis=False,
        enable_heterogeneity=True,
    )


def _new_bridge(cfg) -> SimulationBridge:
    bridge = SimulationBridge(
        core_config=cfg,
        viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(),
        gpu_config=GPUConfig(enable_profiling=False),
    )
    bridge.strict_step_errors = True
    bridge._initialize_simulation_data()
    if not bridge.is_initialized:
        raise RuntimeError("bridge initialization failed")
    return bridge


def _zero_construction_edges(bridge) -> None:
    """Undo the wiring layer's 0.01 floor for declared zero-weight edges."""
    xp, _ = get_backend()
    bridge.cp_connections.data[:] = xp.where(
        bridge.cp_connections.data <= xp.float32(0.011),
        xp.float32(0.0), bridge.cp_connections.data,
    )


def build_tonic_bridge(seed: int, current_pA: float, *, n=40, step_mode="normal"):
    cfg = _read_only_config(
        seed, [_gpi_region(n, current_pA)], step_mode=step_mode
    )
    bridge = _new_bridge(cfg)
    _zero_construction_edges(bridge)
    if bridge.cp_connections is None or np.any(to_host(bridge.cp_connections.data) != 0):
        raise AssertionError("tonic arm must have only zero-weight internal wiring")
    return bridge


def _population_audit(bridge) -> dict:
    arrays = {
        "C": np.asarray(to_host(bridge.cp_izh_C), dtype=np.float64),
        "a": np.asarray(to_host(bridge.cp_izh_a), dtype=np.float64),
        "b": np.asarray(to_host(bridge.cp_izh_b), dtype=np.float64),
        "d": np.asarray(to_host(bridge.cp_izh_d_increment), dtype=np.float64),
    }
    fixed = {
        "k": np.asarray(to_host(bridge.cp_izh_k)),
        "vr": np.asarray(to_host(bridge.cp_izh_vr)),
        "vt": np.asarray(to_host(bridge.cp_izh_vt)),
        "vpeak": np.asarray(to_host(bridge.cp_izh_vpeak)),
    }
    centers = {"C": 60.0, "a": 0.05, "b": 2.0, "d": 25.0}
    cortical = {
        "C": float(bridge.core_config.izh_C_val),
        "a": float(bridge.core_config.izh_a_val),
        "b": float(bridge.core_config.izh_b_val),
        "d": float(bridge.core_config.izh_d_val),
    }
    means = {name: float(values.mean()) for name, values in arrays.items()}
    checks = {
        "k_exact": bool(np.all(fixed["k"] == np.float32(1.0))),
        "vr_exact": bool(np.all(fixed["vr"] == np.float32(-65.0))),
        "vt_exact": bool(np.all(fixed["vt"] == np.float32(-50.0))),
        "vpeak_exact": bool(np.all(fixed["vpeak"] == np.float32(25.0))),
        "heterogeneous_means_are_gpi_centered": all(
            abs(means[name] - centers[name]) < abs(means[name] - cortical[name])
            for name in centers
        ),
        "global_heterogeneity_off": not bridge.core_config.enable_parameter_heterogeneity,
        "gpi_mask_covers_population": bool(np.all(to_host(bridge.cp_heterogeneity_neuron_mask))),
    }
    return {"checks": checks, "pass": all(checks.values()), "means": means}


def _run_steps(bridge, steps: int, *, drive=None) -> dict:
    n = int(bridge.core_config.num_neurons)
    raster = np.zeros((int(steps), n), dtype=bool)
    external_zero = True
    intrinsic_before = _hash(bridge.cp_intrinsic_current_pA) if bridge.cp_intrinsic_current_pA is not None else None
    weight_before = _hash(bridge.cp_connections.data)
    for step in range(int(steps)):
        if drive is not None:
            drive(step, bridge)
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_ms += bridge.core_config.dt_ms
        raster[step] = np.asarray(to_host(bridge.cp_firing_states), dtype=bool)
        external_zero = external_zero and bool(
            np.all(np.asarray(to_host(bridge.cp_external_input_current)) == 0.0)
        )
    states = {
        name: np.asarray(to_host(value)).copy()
        for name, value in {
            "v": bridge.cp_membrane_potential_v,
            "u": bridge.cp_recovery_variable_u,
            "g_e": bridge.cp_conductance_g_e,
            "g_i": bridge.cp_conductance_g_i,
        }.items()
    }
    return {
        "raster": raster,
        "states": states,
        "external_zero": external_zero,
        "intrinsic_hash_before": intrinsic_before,
        "intrinsic_hash_after": (
            _hash(bridge.cp_intrinsic_current_pA)
            if bridge.cp_intrinsic_current_pA is not None else None
        ),
        "weight_hash_before": weight_before,
        "weight_hash_after": _hash(bridge.cp_connections.data),
    }


def _physiology_metrics(run: dict, *, n=40, bin_steps=100) -> dict:
    raster = run["raster"]
    bin_rates = [
        float(raster[start:start + bin_steps].sum() / n / (bin_steps / 1000.0))
        for start in range(0, raster.shape[0], bin_steps)
    ]
    per_cell = raster.sum(axis=0)
    first = np.asarray([
        int(np.flatnonzero(raster[:, cell])[0]) if per_cell[cell] else -1
        for cell in range(n)
    ])
    valid_first = first[first >= 0]
    metrics = {
        "bin_rates_hz": bin_rates,
        "population_rate_hz": float(raster.sum() / n / (raster.shape[0] / 1000.0)),
        "cells_firing": int(np.count_nonzero(per_cell)),
        "max_same_step_fraction": float(raster.sum(axis=1).max(initial=0) / n),
        "distinct_first_spike_steps": int(np.unique(valid_first).size),
        "first_spike_span_ms": float(valid_first.max() - valid_first.min()) if valid_first.size else 0.0,
        "total_spikes": int(raster.sum()),
    }
    checks = {
        "all_bins_in_rate_range": bool(
            bin_rates
            and min(bin_rates) >= CRITERIA["minimum_bin_rate_hz"]
            and max(bin_rates) <= CRITERIA["maximum_bin_rate_hz"]
        ),
        "all_cells_fire": metrics["cells_firing"] == n,
        "same_step_fraction": metrics["max_same_step_fraction"] <= CRITERIA["maximum_same_step_fraction"],
        "distinct_first_spikes": metrics["distinct_first_spike_steps"] >= CRITERIA["minimum_distinct_first_spike_steps"],
        "first_spike_span": metrics["first_spike_span_ms"] >= CRITERIA["minimum_first_spike_span_ms"],
        "external_exact_zero": bool(run["external_zero"]),
        "intrinsic_immutable": run["intrinsic_hash_before"] == run["intrinsic_hash_after"],
        "weights_immutable": run["weight_hash_before"] == run["weight_hash_after"],
        "finite_state": all(np.all(np.isfinite(value)) for value in run["states"].values()),
    }
    return {"metrics": metrics, "checks": checks, "pass": all(checks.values())}


def run_calibration(compatibility_path: Path) -> dict:
    spec = load_locked_spec()
    compatibility = _load_compatibility_correction(compatibility_path)
    backend = _backend_info()
    started = time.perf_counter()
    rows = []
    for index, current in enumerate(LADDER_PA, start=1):
        unit_started = time.perf_counter()
        bridge = build_tonic_bridge(PARTITIONS["calibration"][0], current)
        audit = _population_audit(bridge)
        run = _run_steps(bridge, spec["calibration"]["steps"])
        physiology = _physiology_metrics(run)
        rows.append({
            "current_pA": current,
            "audit": audit,
            "physiology": physiology,
            "raster_hash": _hash(run["raster"]),
            "state_hashes": {name: _hash(value) for name, value in run["states"].items()},
            "pass": bool(audit["pass"] and physiology["pass"]),
        })
        project_cost(
            "v13 tonic calibration", index, len(LADDER_PA),
            time.perf_counter() - unit_started,
        )
        bridge.clear_simulation_state_and_gpu_memory()
    return {
        "probe": "gateB_v13_tonic_output_calibration",
        "stage": "calibration_backend",
        "seed": PARTITIONS["calibration"][0],
        "backend": backend["backend"],
        "device": backend["device"],
        "backend_info": backend,
        "source_sha": _git_sha(),
        "source_identity": _source_identity(),
        "spec_sha256": hashlib.sha256(SPEC_PATH.read_bytes()).hexdigest(),
        "compatibility_correction": compatibility,
        "rows": rows,
        "passing_currents_pA": [row["current_pA"] for row in rows if row["pass"]],
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def merge_calibration(numpy_path: Path, cupy_path: Path) -> dict:
    artifacts = [json.loads(path.read_text()) for path in (numpy_path, cupy_path)]
    by_backend = {item["backend"]: item for item in artifacts}
    if set(by_backend) != {"numpy", "cupy"}:
        raise ValueError(f"expected numpy and cupy artifacts, got {sorted(by_backend)}")
    for item in artifacts:
        if item["seed"] != PARTITIONS["calibration"][0]:
            raise ValueError("calibration artifact used wrong seed")
        if [row["current_pA"] for row in item["rows"]] != LADDER_PA:
            raise ValueError("calibration artifact changed the locked ladder")
    identities = [item.get("source_identity") for item in artifacts]
    if not identities[0] or identities[0] != identities[1]:
        raise ValueError("calibration backends did not use identical sealed sources")
    compatibility = [item.get("compatibility_correction") for item in artifacts]
    expected_compatibility = _load_compatibility_correction(
        COMPATIBILITY_CORRECTION_PATH
    )
    if compatibility != [expected_compatibility, expected_compatibility]:
        raise ValueError("calibration artifacts lack the earned compatibility correction")
    common = [
        current for current in LADDER_PA
        if all(current in by_backend[name]["passing_currents_pA"] for name in ("numpy", "cupy"))
    ]
    selected = min(common) if common else None
    verdict = _earned_verdict(
        "Gate B v13 cross-backend calibration",
        selected is not None,
        {
            "numpy and cupy artifacts present": set(by_backend) == {"numpy", "cupy"},
            "locked calibration seed used": all(
                item["seed"] == PARTITIONS["calibration"][0] for item in artifacts
            ),
            "locked ladder used": all(
                [row["current_pA"] for row in item["rows"]] == LADDER_PA
                for item in artifacts
            ),
            "identical sealed source identities": bool(
                identities[0] and identities[0] == identities[1]
            ),
            "earned compatibility correction bound": compatibility
            == [expected_compatibility, expected_compatibility],
        },
    )
    return {
        "probe": "gateB_v13_tonic_output_calibration",
        "stage": "calibration_cross_backend",
        "backend": "cross_backend",
        "device": "numpy_and_cupy",
        "seed": PARTITIONS["calibration"][0],
        "source_shas": {name: by_backend[name]["source_sha"] for name in by_backend},
        "source_identity": identities[0],
        "compatibility_correction": expected_compatibility,
        "artifacts": {"numpy": str(numpy_path), "cupy": str(cupy_path)},
        "common_passing_currents_pA": common,
        "selected_current_pA": selected,
        **verdict,
        "outcome": _outcome("CALIBRATION", verdict),
        "calibration_go": verdict["go"],
    }


def _load_selection(path: Path) -> tuple[dict, float]:
    selection = json.loads(path.read_text())
    if selection.get("stage") != "calibration_cross_backend":
        raise ValueError("selection artifact is not a cross-backend calibration")
    if not selection.get("calibration_go"):
        raise ValueError("calibration did not open replication")
    if not _artifact_verdict_is_earned(selection):
        raise ValueError("calibration selection has no earned precondition block")
    current = selection.get("selected_current_pA")
    if current not in LADDER_PA:
        raise ValueError("selection artifact contains an unlocked current")
    if selection.get("source_identity") != _source_identity():
        raise ValueError("selection source identity differs from executable source")
    if selection.get("compatibility_correction") != _load_compatibility_correction(
        COMPATIBILITY_CORRECTION_PATH
    ):
        raise ValueError("selection is not bound to the earned compatibility correction")
    return selection, float(current)


def _source_region():
    return BrainRegion(
        name="inhibitory_source",
        n_neurons=20,
        exc_fraction=0.0,
        internal_density=0.1,
        exc_weight_mean=0.0,
        inh_weight_mean=0.0,
        weight_jitter=0.0,
        plastic_internal=False,
        izh_neuron_type=NeuronType.IZH2007_FS_CORTICAL_INTERNEURON.name,
        enable_nmda=False,
        enable_homeostasis=False,
        enable_heterogeneity=False,
    )


def build_inhibitory_bridge(seed: int, current_pA: float):
    regions = [_source_region(), _gpi_region(40, current_pA)]
    pathway = RegionPathway(
        from_region="inhibitory_source",
        to_region="gpi_snr",
        density=1.0,
        weight_mean=8.0,
        weight_jitter=0.0,
        plastic=False,
        receptor="gaba_a",
    )
    bridge = _new_bridge(_read_only_config(seed, regions, [pathway]))
    _zero_construction_edges(bridge)
    return bridge


def run_inhibitory_response(seed: int, current_pA: float) -> dict:
    rows = {}
    for arm in ("source_on", "source_off"):
        bridge = build_inhibitory_bridge(seed, current_pA)
        xp, _ = get_backend()
        source = xp.asarray(_indices(bridge, "inhibitory_source"))
        target = xp.asarray(_indices(bridge, "gpi_snr"))
        initial_weights = _hash(bridge.cp_connections.data)
        initial_intrinsic = _hash(bridge.cp_intrinsic_current_pA)
        raster = np.zeros((1200, 60), dtype=bool)
        target_gaba_max = np.zeros(1200, dtype=np.float64)
        target_external_zero = True
        for step in range(1200):
            bridge.cp_external_input_current[:] = xp.float32(0.0)
            if arm == "source_on" and 500 <= step < 700:
                bridge.cp_external_input_current[source] = xp.float32(1000.0)
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_ms += bridge.core_config.dt_ms
            raster[step] = np.asarray(to_host(bridge.cp_firing_states), dtype=bool)
            target_gaba_max[step] = float(np.max(np.asarray(
                to_host(bridge.cp_conductance_g_i[target]), dtype=np.float64
            ), initial=0.0))
            target_external_zero = target_external_zero and bool(np.all(
                np.asarray(to_host(bridge.cp_external_input_current[target])) == 0.0
            ))
        source_raster = raster[:, :20]
        target_raster = raster[:, 20:]
        rates = [
            float(target_raster[start:start + 100].sum() / 40 / 0.1)
            for start in range(0, 1200, 100)
        ]
        rows[arm] = {
            "source_spikes_by_phase": {
                "baseline": int(source_raster[:500].sum()),
                "inhibition": int(source_raster[500:700].sum()),
                "release": int(source_raster[700:].sum()),
            },
            "target_rates_hz_100ms": rates,
            "target_spikes_by_phase": {
                "baseline": int(target_raster[:500].sum()),
                "inhibition": int(target_raster[500:700].sum()),
                "release": int(target_raster[700:].sum()),
            },
            "target_gaba_max_by_step": target_gaba_max.tolist(),
            "target_external_zero": target_external_zero,
            "initial_weight_hash": initial_weights,
            "final_weight_hash": _hash(bridge.cp_connections.data),
            "initial_intrinsic_hash": initial_intrinsic,
            "final_intrinsic_hash": _hash(bridge.cp_intrinsic_current_pA),
            "raster_hash": _hash(raster),
        }
        bridge.clear_simulation_state_and_gpu_memory()

    on, off = rows["source_on"], rows["source_off"]
    baseline_rate = float(np.mean(on["target_rates_hz_100ms"][:5]))
    inhibition_rate = float(on["target_spikes_by_phase"]["inhibition"] / 40 / 0.2)
    off_inhibition_rate = float(off["target_spikes_by_phase"]["inhibition"] / 40 / 0.2)
    release_rates = on["target_rates_hz_100ms"][7:]
    checks = {
        "baseline_bins_in_range": all(40.0 <= rate <= 80.0 for rate in on["target_rates_hz_100ms"][:5]),
        "source_phase_specific": (
            on["source_spikes_by_phase"]["baseline"] == 0
            and on["source_spikes_by_phase"]["inhibition"] > 0
            and on["source_spikes_by_phase"]["release"] == 0
        ),
        "inhibition_at_most_ten_percent": inhibition_rate <= 0.10 * baseline_rate,
        "inhibition_below_source_off": inhibition_rate < off_inhibition_rate,
        "recovered_by_second_release_bin": len(release_rates) >= 2 and 40.0 <= release_rates[1] <= 80.0,
        "release_remains_in_range": len(release_rates) >= 2 and all(40.0 <= rate <= 80.0 for rate in release_rates[1:]),
        "rebound_bounded": all(rate <= 1.5 * baseline_rate for rate in release_rates),
        "gaba_follows_source": (
            max(on["target_gaba_max_by_step"][:501]) == 0.0
            and max(on["target_gaba_max_by_step"][501:]) > 0.0
        ),
        "target_external_zero": on["target_external_zero"] and off["target_external_zero"],
        "weights_immutable": all(
            row["initial_weight_hash"] == row["final_weight_hash"] for row in rows.values()
        ),
        "intrinsic_immutable": all(
            row["initial_intrinsic_hash"] == row["final_intrinsic_hash"] for row in rows.values()
        ),
    }
    return {
        "baseline_rate_hz": baseline_rate,
        "inhibition_rate_hz": inhibition_rate,
        "source_off_inhibition_rate_hz": off_inhibition_rate,
        "checks": checks,
        "pass": all(checks.values()),
        "arms": rows,
    }


def run_checkpoint_gate(seed: int, current_pA: float) -> dict:
    bridge = build_tonic_bridge(seed, current_pA)
    _run_steps(bridge, 300)
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint = Path(temp_dir) / "tonic.simstate.h5"
        if not bridge.save_checkpoint(str(checkpoint)):
            raise RuntimeError("checkpoint save failed")
        uninterrupted = _run_steps(bridge, 500)
        restored = SimulationBridge(
            core_config=CoreSimConfig(num_neurons=1),
            viz_config=VisualizationConfig(), runtime_state=RuntimeState(),
            gpu_config=GPUConfig(enable_profiling=False),
        )
        if not restored.load_checkpoint(str(checkpoint)):
            raise RuntimeError("checkpoint restore failed")
        resumed = _run_steps(restored, 500)
        with h5py.File(checkpoint, "r+") as h5f:
            del h5f["cp_intrinsic_current_pA"]
        old = SimulationBridge(
            core_config=CoreSimConfig(num_neurons=1),
            viz_config=VisualizationConfig(), runtime_state=RuntimeState(),
            gpu_config=GPUConfig(enable_profiling=False),
        )
        old_loaded = old.load_checkpoint(str(checkpoint))
        old_is_none = old.cp_intrinsic_current_pA is None
    hashes = lambda run: {
        "raster": _hash(run["raster"]),
        **{name: _hash(value) for name, value in run["states"].items()},
        "intrinsic": run["intrinsic_hash_after"],
        "weights": run["weight_hash_after"],
    }
    left, right = hashes(uninterrupted), hashes(resumed)
    checks = {
        "exact_continuation": left == right,
        "old_checkpoint_loads": bool(old_loaded),
        "old_checkpoint_intrinsic_is_none": bool(old_is_none),
    }
    return {"checks": checks, "pass": all(checks.values()), "uninterrupted": left, "resumed": right}


def run_replication(selection_path: Path, *, held_out=False) -> dict:
    selection, current = _load_selection(selection_path)
    if held_out:
        _assert_source_sealed()
    seed = PARTITIONS["held_out" if held_out else "replication"][0]
    backend = _backend_info()
    started = time.perf_counter()
    intact_bridge = build_tonic_bridge(seed, current)
    audit = _population_audit(intact_bridge)
    intact_run = _run_steps(intact_bridge, 1000)
    intact = _physiology_metrics(intact_run)
    intact_bridge.clear_simulation_state_and_gpu_memory()
    lesion_bridge = build_tonic_bridge(seed, 0.0)
    lesion_run = _run_steps(lesion_bridge, 1000)
    lesion = {
        "total_spikes": int(lesion_run["raster"].sum()),
        "external_zero": lesion_run["external_zero"],
        "pass": bool(lesion_run["raster"].sum() == 0 and lesion_run["external_zero"]),
        "raster_hash": _hash(lesion_run["raster"]),
    }
    lesion_bridge.clear_simulation_state_and_gpu_memory()
    inhibitory = run_inhibitory_response(seed, current)
    checkpoint = None if held_out else run_checkpoint_gate(seed, current)
    checks = {
        "population_audit": audit["pass"],
        "intact_physiology": intact["pass"],
        "intrinsic_lesion": lesion["pass"],
        "inhibitory_response": inhibitory["pass"],
        "checkpoint": True if checkpoint is None else checkpoint["pass"],
    }
    stage = "held_out" if held_out else "replication"
    verdict = _earned_verdict(
        f"Gate B v13 {stage}",
        all(checks.values()),
        {
            "selection verdict earned": _artifact_verdict_is_earned(selection),
            "selected current is locked": current in LADDER_PA,
            "complete gate set measured": set(checks) == {
                "population_audit", "intact_physiology", "intrinsic_lesion",
                "inhibitory_response", "checkpoint",
            },
            "held-out omits no required checkpoint": not held_out or checkpoint is None,
        },
    )
    return {
        "probe": "gateB_v13_tonic_output",
        "stage": stage,
        "seed": seed,
        "selected_current_pA": current,
        "backend": backend["backend"],
        "device": backend["device"],
        "backend_info": backend,
        "source_sha": _git_sha(),
        "source_identity": _source_identity(),
        "selection_artifact": str(selection_path),
        "selection": selection,
        "audit": audit,
        "intact": intact,
        "intact_hashes": {
            "raster": _hash(intact_run["raster"]),
            **{name: _hash(value) for name, value in intact_run["states"].items()},
        },
        "lesion": lesion,
        "inhibitory_response": inhibitory,
        "checkpoint": checkpoint,
        "checks": checks,
        **verdict,
        "outcome": _outcome(stage.upper(), verdict),
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def run_compatibility() -> dict:
    spec = load_locked_spec()
    backend = _backend_info()
    started = time.perf_counter()
    config = selector_config("v2")
    bridge = build_selector_bridge(
        PARTITIONS["compatibility"][0],
        config,
        commit_enable_nmda=False,
        core_config_updates={"enable_nmda": False},
    )
    if bridge.cp_intrinsic_current_pA is not None:
        raise AssertionError("default selector unexpectedly allocated intrinsic current")
    xp, _ = get_backend()
    raster = np.zeros((300, bridge.core_config.num_neurons), dtype=bool)
    for step in range(300):
        _set_equal_tonic_current(bridge, config)
        bridge.cp_external_input_current[
            xp.asarray(_indices(bridge, "practice_arousal"))
        ] = xp.float32(250.0)
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_ms += bridge.core_config.dt_ms
        raster[step] = np.asarray(to_host(bridge.cp_firing_states), dtype=bool)
    actual = {
        "raster": _hash(raster),
        "v": _hash(bridge.cp_membrane_potential_v),
        "u": _hash(bridge.cp_recovery_variable_u),
        "g_e": _hash(bridge.cp_conductance_g_e),
        "g_i": _hash(bridge.cp_conductance_g_i),
        "weights": _hash(bridge.cp_connections.data),
        "external": _hash(bridge.cp_external_input_current),
    }
    expected = {
        **DEFAULT_HASHES[backend["backend"]],
        "weights": WEIGHT_HASH,
        "external": EXTERNAL_HASH,
    }
    checks = {
        name: actual[name] == expected[name] for name in expected
    }
    checks["intrinsic_is_none"] = bridge.cp_intrinsic_current_pA is None
    verdict = _earned_verdict(
        "Gate B v13 default-off compatibility",
        all(checks.values()),
        {
            "locked spec loaded": spec["partitions"] == PARTITIONS,
            "explicit supported backend": backend["backend"] in DEFAULT_HASHES,
            "locked compatibility seed used": bridge.core_config.seed
            == PARTITIONS["compatibility"][0],
            "complete 300-step raster captured": raster.shape
            == (300, bridge.core_config.num_neurons),
            "complete expected hash set measured": set(actual) == set(expected),
        },
    )
    return {
        "probe": "gateB_v13_default_off_compatibility",
        "stage": "compatibility",
        "seed": PARTITIONS["compatibility"][0],
        "backend": backend["backend"],
        "device": backend["device"],
        "source_sha": _git_sha(),
        "source_identity": _source_identity(),
        "intrinsic_is_none": bridge.cp_intrinsic_current_pA is None,
        "actual_hashes": actual,
        "expected_hashes": expected,
        "checks": checks,
        **verdict,
        "outcome": _outcome("COMPATIBILITY", verdict),
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def _performance_bridge(*, active: bool, step_mode: str):
    regions = [
        _gpi_region(40, 100.0 if active else 0.0, name="active_gpi"),
        BrainRegion(
            name="control_gpi", n_neurons=560, exc_fraction=0.0,
            internal_density=0.1, exc_weight_mean=0.0, inh_weight_mean=0.0,
            weight_jitter=0.0, plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_GPI_OUTPUT.name,
            intrinsic_current_pA=0.0, enable_nmda=False,
            enable_homeostasis=False, enable_heterogeneity=False,
        ),
    ]
    bridge = _new_bridge(_read_only_config(
        PARTITIONS["audit_only"][0], regions, step_mode=step_mode
    ))
    _zero_construction_edges(bridge)
    return bridge


def _legacy_performance_bridge():
    regions = [
        BrainRegion(
            name="gpi_a", n_neurons=40, exc_fraction=0.0,
            internal_density=0.1, exc_weight_mean=0.0, inh_weight_mean=0.0,
            weight_jitter=0.0, plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_GPI_OUTPUT.name,
            enable_nmda=False, enable_homeostasis=False,
            enable_heterogeneity=True,
        ),
        BrainRegion(
            name="gpi_b", n_neurons=560, exc_fraction=0.0,
            internal_density=0.1, exc_weight_mean=0.0, inh_weight_mean=0.0,
            weight_jitter=0.0, plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_GPI_OUTPUT.name,
            enable_nmda=False, enable_homeostasis=False,
            enable_heterogeneity=False,
        ),
    ]
    bridge = _new_bridge(_read_only_config(
        PARTITIONS["audit_only"][0], regions, step_mode="normal"
    ))
    _zero_construction_edges(bridge)
    return bridge


def run_legacy_performance_baseline() -> dict:
    backend = _backend_info()
    if backend["backend"] != "cupy" or "3090" not in backend["device"]:
        raise ValueError("legacy performance baseline requires the RTX 3090 CuPy lane")
    times = []
    for index in range(3):
        bridge = _legacy_performance_bridge()
        for _ in range(500):
            bridge._run_one_simulation_step()
        synchronize()
        started = time.perf_counter()
        for _ in range(20000):
            bridge._run_one_simulation_step()
        synchronize()
        times.append(float(time.perf_counter() - started))
        project_cost("v13 legacy performance baseline", index + 1, 3, times[-1])
        bridge.clear_simulation_state_and_gpu_memory()
    return {
        "probe": "gateB_v13_legacy_default_performance",
        "stage": "legacy_performance_baseline",
        "seed": PARTITIONS["audit_only"][0],
        "backend": backend["backend"],
        "device": backend["device"],
        "source_sha": _git_sha(),
        "wall_seconds": times,
        "median_seconds": float(np.median(times)),
        "median_step_seconds": float(np.median(times) / 20000.0),
        "outcome": "BASELINE_RECORDED",
        "elapsed_seconds": float(sum(times)),
    }


def _benchmark_cell(*, active: bool, step_mode: str, repetitions=3) -> dict:
    times = []
    feature_bytes = []
    pool_bytes = []
    dispatch = []
    for _ in range(int(repetitions)):
        bridge = _performance_bridge(active=active, step_mode=step_mode)
        dispatch.append(bool(bridge._step_megakernel_can_dispatch()))
        for _ in range(500):
            bridge._run_one_simulation_step()
        synchronize()
        started = time.perf_counter()
        for _ in range(20000):
            bridge._run_one_simulation_step()
        synchronize()
        times.append(float(time.perf_counter() - started))
        feature_bytes.append(
            0 if bridge.cp_intrinsic_current_pA is None
            else int(bridge.cp_intrinsic_current_pA.nbytes)
        )
        xp, _ = get_backend()
        pool_bytes.append(
            int(xp.get_default_memory_pool().used_bytes()) if is_gpu_backend() else 0
        )
        bridge.clear_simulation_state_and_gpu_memory()
    return {
        "active": bool(active),
        "step_mode": step_mode,
        "wall_seconds": times,
        "median_seconds": float(np.median(times)),
        "median_step_seconds": float(np.median(times) / 20000.0),
        "feature_bytes": feature_bytes,
        "memory_pool_used_bytes": pool_bytes,
        "megakernel_dispatch": dispatch,
    }


def run_performance(old_baseline_path: Path | None = None) -> dict:
    load_locked_spec()
    backend = _backend_info()
    if backend["backend"] != "cupy":
        raise ValueError("performance gate must run on CuPy/RTX 3090")
    if "3090" not in backend["device"]:
        raise ValueError(f"performance gate requires RTX 3090, got {backend['device']}")
    started = time.perf_counter()
    cells = {}
    for mode in ("normal", "v1", "v2"):
        cells[f"{mode}_default"] = _benchmark_cell(active=False, step_mode=mode)
        cells[f"{mode}_active"] = _benchmark_cell(active=True, step_mode=mode)
    old = json.loads(old_baseline_path.read_text()) if old_baseline_path else None
    old_median = None if old is None else float(old["median_seconds"])
    ratios = {
        "default_vs_old": (
            None if old_median is None
            else cells["normal_default"]["median_seconds"] / old_median
        ),
        "normal_active": cells["normal_active"]["median_seconds"] / cells["normal_default"]["median_seconds"],
        "v1_active": cells["v1_active"]["median_seconds"] / cells["v1_default"]["median_seconds"],
        "v2_active": cells["v2_active"]["median_seconds"] / cells["v2_default"]["median_seconds"],
    }
    checks = {
        "old_baseline_supplied": old_median is not None,
        "default_off_ratio": ratios["default_vs_old"] is not None and ratios["default_vs_old"] <= 1.02,
        "normal_active_ratio": ratios["normal_active"] <= 1.10,
        "v1_active_ratio": ratios["v1_active"] <= 1.10,
        "v2_active_ratio": ratios["v2_active"] <= 1.10,
        "feature_storage": all(
            all(value <= 4 * 600 for value in row["feature_bytes"])
            for row in cells.values()
        ),
        "default_does_not_allocate": all(
            value == 0 for name, row in cells.items() if name.endswith("_default")
            for value in row["feature_bytes"]
        ),
        "v1_dispatches": all(cells["v1_active"]["megakernel_dispatch"]),
        "v2_dispatches": all(cells["v2_active"]["megakernel_dispatch"]),
    }
    verdict = _earned_verdict(
        "Gate B v13 performance",
        all(checks.values()),
        {
            "RTX 3090 CuPy lane": backend["backend"] == "cupy"
            and "3090" in backend["device"],
            "all six benchmark cells measured": len(cells) == 6,
            "three repetitions per cell": all(
                len(row["wall_seconds"]) == 3 for row in cells.values()
            ),
            "legacy baseline supplied": old is not None,
        },
    )
    return {
        "probe": "gateB_v13_tonic_output_performance",
        "stage": "performance",
        "seed": PARTITIONS["audit_only"][0],
        "backend": backend["backend"],
        "device": backend["device"],
        "source_sha": _git_sha(),
        "source_identity": _source_identity(),
        "old_baseline_artifact": str(old_baseline_path) if old_baseline_path else None,
        "old_baseline": old,
        "cells": cells,
        "ratios": ratios,
        "checks": checks,
        **verdict,
        "outcome": _outcome("PERFORMANCE", verdict),
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def merge_final(paths: list[Path]) -> dict:
    if len(paths) != 6:
        raise ValueError("final merge requires 6 artifacts")
    artifacts = [json.loads(path.read_text()) for path in paths]
    if not all(_artifact_verdict_is_earned(artifact) for artifact in artifacts):
        raise ValueError("final merge refuses an artifact without an earned verdict")
    by_stage = {}
    for path, artifact in zip(paths, artifacts):
        by_stage.setdefault(artifact.get("stage"), []).append((path, artifact))
    expected_counts = {
        "cross_twin_compare": 1,
        "replication": 2,
        "held_out": 2,
        "performance": 1,
    }
    if {stage: len(by_stage.get(stage, [])) for stage in expected_counts} != expected_counts:
        raise ValueError("final artifacts do not contain the required stage/backend matrix")
    for stage in ("replication", "held_out"):
        backends = {artifact["backend"] for _, artifact in by_stage[stage]}
        if backends != {"numpy", "cupy"}:
            raise ValueError(f"{stage} requires NumPy and CuPy artifacts")
    selected = {
        artifact.get("selected_current_pA")
        for stage in ("replication", "held_out")
        for _, artifact in by_stage[stage]
    }
    if len(selected) != 1 or next(iter(selected)) not in LADDER_PA:
        raise ValueError("replication and held-out artifacts disagree on selection")
    checks = {
        "compatibility": (
            by_stage["cross_twin_compare"][0][1]
            == json.loads(COMPATIBILITY_CORRECTION_PATH.read_text())
            and _load_compatibility_correction(COMPATIBILITY_CORRECTION_PATH)["outcome"]
            == "DETERMINISTIC_COMPATIBILITY_GO"
        ),
        "replication": all(item.get("go") for _, item in by_stage["replication"]),
        "held_out": all(item.get("go") for _, item in by_stage["held_out"]),
        "performance": bool(by_stage["performance"][0][1].get("go")),
    }
    verdict = _earned_verdict(
        "Gate B v13 Stage 0 final",
        all(checks.values()),
        {
            "six earned input verdicts": all(
                _artifact_verdict_is_earned(artifact) for artifact in artifacts
            ),
            "required stage matrix complete": all(
                len(by_stage.get(stage, [])) == count
                for stage, count in expected_counts.items()
            ),
            "single locked selected current": len(selected) == 1
            and next(iter(selected)) in LADDER_PA,
        },
    )
    return {
        "probe": "gateB_v13_tonic_output",
        "stage": "final_cross_backend",
        "backend": "cross_backend",
        "device": "numpy_cupy_rtx3090",
        "selected_current_pA": next(iter(selected)),
        "artifacts": {str(path): artifact.get("outcome") for path, artifact in zip(paths, artifacts)},
        "checks": checks,
        **verdict,
        "outcome": _outcome("TONIC_OUTPUT", verdict),
    }


def self_check() -> dict:
    spec = load_locked_spec()
    bridge = build_tonic_bridge(7, 100.0)
    audit = _population_audit(bridge)
    run = _run_steps(bridge, 50)
    checks = {
        "spec_locked": spec["partitions"] == PARTITIONS,
        "population_audit": audit["pass"],
        "intrinsic_allocated": bridge.cp_intrinsic_current_pA is not None,
        "external_zero": run["external_zero"],
        "intrinsic_immutable": run["intrinsic_hash_before"] == run["intrinsic_hash_after"],
    }
    verdict = _earned_verdict(
        "Gate B v13 runner self-check",
        all(checks.values()),
        {
            "unregistered engineering seed": bridge.core_config.seed == 7,
            "locked spec available": spec["partitions"] == PARTITIONS,
            "all self-checks measured": len(checks) == 5,
        },
    )
    return {
        "probe": "gateB_v13_tonic_output_runner_self_check",
        "stage": "self_check",
        "seed": 7,
        "backend": _backend_info()["backend"],
        "device": _backend_info()["device"],
        "checks": checks,
        **verdict,
        "outcome": _outcome("SELF_CHECK", verdict),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    stages = parser.add_mutually_exclusive_group(required=True)
    stages.add_argument("--calibration", action="store_true")
    stages.add_argument("--merge-calibration", nargs=2, metavar=("NUMPY", "CUPY"))
    stages.add_argument("--replication", metavar="SELECTION")
    stages.add_argument("--held-out", metavar="SELECTION")
    stages.add_argument("--compatibility", action="store_true")
    stages.add_argument("--performance", action="store_true")
    stages.add_argument("--legacy-performance-baseline", action="store_true")
    stages.add_argument("--merge-final", nargs=6, metavar=(
        "COMPATIBILITY_GO", "REPL_NUMPY", "REPL_CUPY",
        "HELD_CUPY", "HELD_NUMPY", "PERFORMANCE",
    ))
    stages.add_argument("--self-check", action="store_true")
    parser.add_argument("--old-baseline")
    parser.add_argument("--compatibility-correction")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    if args.calibration:
        if not args.compatibility_correction:
            parser.error("--calibration requires --compatibility-correction")
        result = run_calibration(Path(args.compatibility_correction))
    elif args.merge_calibration:
        result = merge_calibration(
            Path(args.merge_calibration[0]), Path(args.merge_calibration[1])
        )
    elif args.replication:
        result = run_replication(Path(args.replication), held_out=False)
    elif args.held_out:
        result = run_replication(Path(args.held_out), held_out=True)
    elif args.compatibility:
        result = run_compatibility()
    elif args.performance:
        result = run_performance(
            Path(args.old_baseline) if args.old_baseline else None
        )
    elif args.legacy_performance_baseline:
        result = run_legacy_performance_baseline()
    elif args.merge_final:
        result = merge_final([Path(value) for value in args.merge_final])
    else:
        result = self_check()

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {output}")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "stage": result["stage"], "outcome": result.get("outcome"),
        "backend": result.get("backend"), "output": str(output),
    }, indent=2))
    return 0 if result.get("go", result.get("calibration_go", True)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
