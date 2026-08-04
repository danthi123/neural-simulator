"""Sealed cross-backend state-transplant diagnostic for V13 inhibition."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import socket
import subprocess
import sys
from typing import Any

import numpy as np

from research.runners._vocal_action_credit_gate_v13_tonic_output import (
    build_inhibitory_bridge,
)
from sim.backend import get_backend, get_sparse_module, synchronize, to_host
from tools.lab import assert_backend


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPEC_PATH = ROOT / "research/specs/v13_backend_state_transplant.json"
SCHEMA_BUNDLE = "v13-backend-state-transplant-bundle-v1"
SCHEMA_RUN = "v13-backend-state-transplant-run-v1"
SCHEMA_COMPARISON = "v13-backend-state-transplant-comparison-v1"
SCHEMA_AGGREGATE = "v13-backend-state-transplant-aggregate-v1"
BACKENDS = ("numpy", "cupy")
MODES = ("default", "deterministic_transpose_matvec")
TRAJECTORIES = ("v", "u", "g_e", "g_i", "spikes")
CONTINUOUS_TRAJECTORIES = ("v", "u", "g_e", "g_i")
RTOL = 1e-6
ATOL = 1e-6

LOCKED_SEED = 7_606_856
SOURCE_ANCHOR_SHA = "b3d57494b7dd7d99d5e91088489da44d89a85bf3"
LOCKED_NETWORK = {
    "source_region": {"name": "inhibitory_source", "n_neurons": 20},
    "target_region": {"name": "gpi_snr", "n_neurons": 40},
    "target_intrinsic_current_pA": 100.0,
    "pathway": {
        "density": 1.0,
        "weight_mean": 8.0,
        "weight_jitter": 0.0,
        "plastic": False,
        "receptor": "gaba_a",
    },
}
NETWORK_CHECK_NAMES = frozenset({
    "source_region_identity",
    "target_region_identity",
    "total_neuron_count",
    "region_config_counts",
    "target_region_intrinsic_field",
    "intrinsic_vector_exact",
    "target_external_current_exact",
    "single_pathway",
    "pathway_regions",
    "pathway_density",
    "pathway_weight_mean",
    "pathway_weight_jitter",
    "pathway_plastic",
    "pathway_receptor",
    "full_source_target_topology",
    "nonzero_edge_count",
    "nonzero_weights_exact",
})
FORBIDDEN_V13_SEEDS = frozenset({
    1013, 1019, 1021, 1031, 271828, 271829, 271831, 271837, 271843,
    271849, 271853, 314159,
})
SOURCE_PATHS = (
    "sim/backend.py",
    "sim/bridge.py",
    "sim/kernels.py",
    "sim/regions.py",
    "research/runners/_vocal_action_credit_gate_v13_tonic_output.py",
    "research/runners/_v13_backend_state_transplant.py",
)
BUNDLE_ARRAY_ATTRIBUTES = {
    "C": "cp_izh_C",
    "a": "cp_izh_a",
    "b": "cp_izh_b",
    "d": "cp_izh_d_increment",
    "k": "cp_izh_k",
    "vr": "cp_izh_vr",
    "vt": "cp_izh_vt",
    "vpeak": "cp_izh_vpeak",
    "v": "cp_membrane_potential_v",
    "u": "cp_recovery_variable_u",
    "g_e": "cp_conductance_g_e",
    "g_i": "cp_conductance_g_i",
    "intrinsic_current": "cp_intrinsic_current_pA",
    "external_current": "cp_external_input_current",
}
CSR_NAMES = ("csr_data", "csr_indices", "csr_indptr")
PARAMETER_ARRAY_NAMES = frozenset({
    "cp_izh_C", "cp_izh_a", "cp_izh_b", "cp_izh_d_increment",
    "cp_izh_k", "cp_izh_vr", "cp_izh_vt", "cp_izh_vpeak",
    "cp_izh_c_reset", "cp_intrinsic_current_pA", "cp_neuron_type_ids",
    "cp_traits", "cp_heterogeneity_neuron_mask",
    "cp_syn_reversal_potential_i_per_neuron",
})
RUNTIME_DISABLED_FLAGS = (
    "enable_parameter_heterogeneity",
    "enable_ou_process",
    "enable_conductance_noise",
    "enable_hebbian_learning",
    "enable_branchless_plasticity",
    "enable_short_term_plasticity",
    "enable_homeostasis",
    "enable_stdp",
    "enable_inhibitory_stdp",
    "enable_structural_plasticity",
    "enable_reward_modulation",
    "enable_neuromodulator_subsystem",
    "enable_nmda",
    "enable_nmda_recurrent",
    "enable_gabab",
    "enable_coincidence_detection",
    "enable_graded_dendritic_plateau",
    "enable_step_megakernel",
    "enable_step_cudagraph",
    "enable_step_megakernel_v2",
)
RUNTIME_ENABLED_FLAGS = (
    "read_only_fast_step",
    "fast_spike_reset",
)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hash_array(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(to_host(value)))
    return _sha256(array.tobytes(order="C"))


def _artifact_digest(artifact: dict) -> str:
    payload = dict(artifact)
    payload.pop("artifact_sha256", None)
    return _sha256(_canonical(payload))


def _seal_artifact(payload: dict) -> dict:
    artifact = dict(payload)
    artifact["artifact_sha256"] = _artifact_digest(artifact)
    return artifact


def _validate_artifact_digest(artifact: dict, label: str) -> None:
    if artifact.get("artifact_sha256") != _artifact_digest(artifact):
        raise ValueError(f"{label} artifact digest mismatch")


def _source_identity() -> dict[str, str]:
    return {path: _sha256((ROOT / path).read_bytes()) for path in SOURCE_PATHS}


def _source_manifest() -> dict[str, Any]:
    files = _source_identity()
    return {
        "schema": "v13-backend-state-transplant-source-manifest-v1",
        "files": files,
        "sha256": _sha256(_canonical(files)),
    }


def _execution_environment() -> dict[str, str]:
    xp, backend = get_backend()
    return {
        "backend": backend,
        "backend_library_version": str(getattr(xp, "__version__", "unknown")),
        "hostname": socket.gethostname(),
        "machine": platform.machine(),
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "python": sys.version,
        "sim_backend_env": os.environ.get("SIM_BACKEND", ""),
    }


def _environment_record_valid(record: Any, backend: str) -> bool:
    required = {
        "backend", "backend_library_version", "hostname", "machine",
        "numpy_version", "platform", "python", "sim_backend_env",
    }
    return bool(
        isinstance(record, dict)
        and set(record) == required
        and record.get("backend") == backend
        and record.get("sim_backend_env") == backend
        and all(isinstance(record[name], str) and record[name] for name in required - {"sim_backend_env"})
    )


def _trajectory_step_hashes(value: np.ndarray) -> list[str]:
    return [_sha256(np.ascontiguousarray(row).tobytes(order="C")) for row in value]


def _expected_runtime_config(spec: dict, mode: str) -> dict:
    values = {name: False for name in RUNTIME_DISABLED_FLAGS}
    values.update({name: True for name in RUNTIME_ENABLED_FLAGS})
    values.update({
        "seed": spec["seed"],
        "heterogeneity_seed": spec["seed"],
        "dt_ms": spec["steps"]["dt_ms"],
        "neuron_model_type": "IZHIKEVICH",
        "deterministic_transpose_matvec": mode == "deterministic_transpose_matvec",
    })
    return values


def _comparison_tolerance(spec: dict) -> dict:
    locked = spec["comparison_tolerance"]
    return {
        "rtol": float(locked["continuous_rtol"]),
        "atol": float(locked["continuous_atol"]),
        "spikes": locked["spikes"],
    }


def _runtime_contract(bridge, spec: dict, mode: str) -> dict:
    expected = _expected_runtime_config(spec, mode)
    observed = {name: getattr(bridge.core_config, name, None) for name in expected}
    side_channels = {
        "experiment_inactive": not bool(
            getattr(getattr(bridge, "experiment_engine", None), "is_experiment_running", False)
        ),
        "data_bus_inactive": getattr(bridge, "data_bus", None) is None,
        "synapse_store_inactive": getattr(bridge, "synapse_store", None) is None,
        "recording_inactive": not bool(getattr(bridge, "recording_file_handle", None)),
        "engram_recordings_inactive": not bool(getattr(bridge, "_engram_recordings", None)),
        "gate_couplings_inactive": not bool(getattr(bridge, "_gate_couplings", None)),
        "step_profiler_inactive": not bool(
            getattr(getattr(bridge, "gpu_config", None), "enable_step_profiler", False)
        ),
    }
    random_flags_disabled = all(
        observed[name] is False for name in RUNTIME_DISABLED_FLAGS
    )
    contract = {
        "mode": mode,
        "expected_core_config": expected,
        "observed_core_config": observed,
        "core_config_exact": observed == expected,
        "runtime_random_processes_disabled": random_flags_disabled,
        "runtime_side_channels": side_channels,
        "runtime_side_channels_inactive": all(side_channels.values()),
    }
    contract["contract_valid"] = all((
        contract["core_config_exact"],
        contract["runtime_random_processes_disabled"],
        contract["runtime_side_channels_inactive"],
    ))
    return contract


def _require_runtime_contract(bridge, spec: dict, mode: str) -> dict:
    contract = _runtime_contract(bridge, spec, mode)
    if not contract["contract_valid"]:
        raise ValueError(f"runtime feature contract mismatch: {contract}")
    return contract


def _runtime_contract_record_valid(record: Any, spec: dict, mode: str) -> bool:
    expected = _expected_runtime_config(spec, mode)
    side_channel_names = {
        "experiment_inactive", "data_bus_inactive", "synapse_store_inactive",
        "recording_inactive", "engram_recordings_inactive",
        "gate_couplings_inactive", "step_profiler_inactive",
    }
    return bool(
        isinstance(record, dict)
        and record.get("mode") == mode
        and record.get("expected_core_config") == expected
        and record.get("observed_core_config") == expected
        and record.get("core_config_exact") is True
        and record.get("runtime_random_processes_disabled") is True
        and isinstance(record.get("runtime_side_channels"), dict)
        and set(record["runtime_side_channels"]) == side_channel_names
        and all(value is True for value in record["runtime_side_channels"].values())
        and record.get("runtime_side_channels_inactive") is True
        and record.get("contract_valid") is True
    )


def _validate_sha256(value: str, label: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")


def _anchor_is_ancestor(anchor: str) -> bool:
    return subprocess.run(
        ["git", "merge-base", "--is-ancestor", anchor, "HEAD"],
        cwd=ROOT, capture_output=True,
    ).returncode == 0


def load_locked_spec(path: Path, expected_sha256: str) -> tuple[dict, str]:
    """Load exactly the committed v13_backend_state_transplant.json contract."""
    _validate_sha256(expected_sha256, "spec SHA-256")
    raw = path.read_bytes()
    actual_sha256 = _sha256(raw)
    if actual_sha256 != expected_sha256:
        raise ValueError("locked spec digest mismatch")
    try:
        spec = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("locked spec is not valid JSON") from exc
    if not isinstance(spec, dict):
        raise ValueError("locked spec must be a JSON object")

    expected_keys = {
        "schema_version", "status", "mechanism", "promotion_value",
        "source_anchor_sha", "seed", "seed_derivation", "forbidden_seeds",
        "origins", "execution_backends", "modes", "expected_matrix_cells",
        "network", "steps", "stimulus", "required_bundle_arrays",
        "required_trajectory_arrays", "required_state_scope",
        "comparison_tolerance", "verdict",
    }
    steps = spec.get("steps")
    checks = {
        "fields": set(spec) == expected_keys,
        "schema_version": spec.get("schema_version") == 1,
        "status": spec.get("status") == "locked",
        "mechanism": spec.get("mechanism") == "gateB-v13-backend-state-transplant-diagnostic",
        "promotion_value": spec.get("promotion_value") == "none",
        "source_anchor": spec.get("source_anchor_sha") == SOURCE_ANCHOR_SHA,
        "source_anchor_present": _anchor_is_ancestor(SOURCE_ANCHOR_SHA),
        "seed": spec.get("seed") == LOCKED_SEED,
        "seed_is_not_formal": spec.get("seed") not in FORBIDDEN_V13_SEEDS,
        "seed_derivation": spec.get("seed_derivation") == {
            "material": (
                "V13_BACKEND_STATE_TRANSPLANT_V1|"
                f"{SOURCE_ANCHOR_SHA}|role=paired_origin"
            ),
            "sha256_prefix_12": "b6388351e7c8",
            "formula": "2000000 + (prefix_integer mod 7000000)",
        },
        "forbidden_seeds": spec.get("forbidden_seeds") == [1013, 1019, 1021, 1031],
        "origins": spec.get("origins") == list(BACKENDS),
        "execution_backends": spec.get("execution_backends") == list(BACKENDS),
        "modes": spec.get("modes") == list(MODES),
        "expected_matrix_cells": spec.get("expected_matrix_cells") == 8,
        "network": spec.get("network") == LOCKED_NETWORK,
        "steps": steps == {
            "baseline": 500, "inhibition": 200, "release": 500, "dt_ms": 1.0,
        },
        "stimulus": spec.get("stimulus") == {
            "source_current_pA": 1000.0,
            "target_external_current_pA": 0.0,
        },
        "required_bundle_arrays": spec.get("required_bundle_arrays") == [
            "C", "a", "b", "d", "k", "vr", "vt", "vpeak",
            "v", "u", "g_e", "g_i", "intrinsic_current", "external_current",
            "csr_data", "csr_indices", "csr_indptr",
        ],
        "required_trajectory_arrays": spec.get("required_trajectory_arrays") == list(TRAJECTORIES),
        "required_state_scope": spec.get("required_state_scope") == "all_allocated_cp_ndarrays",
        "comparison_tolerance": spec.get("comparison_tolerance") == {
            "continuous_rtol": RTOL,
            "continuous_atol": ATOL,
            "spikes": "exact",
        },
        "verdict": spec.get("verdict") == "DIAGNOSTIC_ONLY",
    }
    if not all(checks.values()):
        raise ValueError(f"locked spec disagreement: {checks}")
    return spec, actual_sha256


def _encode_array(value: Any) -> dict:
    array = np.ascontiguousarray(np.asarray(to_host(value)))
    if array.dtype.hasobject:
        raise ValueError("object arrays are not valid diagnostic state")
    raw = array.tobytes(order="C")
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "sha256": _sha256(raw),
        "data_base64": base64.b64encode(raw).decode("ascii"),
    }


def _decode_array(record: dict, label: str) -> np.ndarray:
    if not isinstance(record, dict) or set(record) != {
        "dtype", "shape", "sha256", "data_base64",
    }:
        raise ValueError(f"{label} has an invalid array record")
    try:
        dtype = np.dtype(record["dtype"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} has an invalid dtype") from exc
    if dtype.hasobject:
        raise ValueError(f"{label} cannot contain objects")
    shape = record["shape"]
    if not isinstance(shape, list) or not all(type(item) is int and item >= 0 for item in shape):
        raise ValueError(f"{label} has an invalid shape")
    try:
        raw = base64.b64decode(record["data_base64"], validate=True)
    except Exception as exc:
        raise ValueError(f"{label} has invalid base64 data") from exc
    count = int(np.prod(shape, dtype=np.int64)) if shape else 1
    if len(raw) != count * dtype.itemsize or _sha256(raw) != record["sha256"]:
        raise ValueError(f"{label} array digest or byte count mismatch")
    return np.frombuffer(raw, dtype=dtype).copy().reshape(shape)


def _is_sparse(value: Any) -> bool:
    return all(hasattr(value, name) for name in ("data", "indices", "indptr"))


def _allocated_cp_arrays(bridge) -> dict[str, Any]:
    """Capture every allocated direct ``cp_*`` ndarray, regardless of shape."""
    arrays = {}
    for name, value in vars(bridge).items():
        if not name.startswith("cp_") or value is None or name == "cp_connections":
            continue
        if _is_sparse(value):
            raise ValueError(f"allocated sparse state {name} is outside the sealed CSR contract")
        if hasattr(value, "shape") and hasattr(value, "dtype"):
            encoded = _encode_array(value)
            arrays[name] = value
            if encoded["dtype"] == "|O":
                raise ValueError(f"allocated dynamic array {name} is not serializable")
    missing = set(BUNDLE_ARRAY_ATTRIBUTES.values()) - set(arrays)
    if missing:
        raise ValueError(f"bridge is missing required bundle arrays: {sorted(missing)}")
    return arrays


def _region_identity(bridge, name: str) -> dict:
    indices = [int(item) for item in bridge.region_manager.indices(name)]
    return {"name": name, "indices": indices, "n_neurons": len(indices)}


def _validate_constructed_network(bridge, spec: dict) -> dict:
    network = spec["network"]
    source_spec = network["source_region"]
    target_spec = network["target_region"]
    source = _region_identity(bridge, source_spec["name"])
    target = _region_identity(bridge, target_spec["name"])
    region_configs = {
        region.name: region for region in getattr(bridge.core_config, "brain_regions", [])
    }
    source_cfg = region_configs.get(source_spec["name"])
    target_cfg = region_configs.get(target_spec["name"])
    pathways = list(getattr(bridge.core_config, "region_pathways", []))
    pathway = pathways[0] if len(pathways) == 1 else None

    data = np.asarray(to_host(bridge.cp_connections.data))
    indices = np.asarray(to_host(bridge.cp_connections.indices), dtype=np.int64)
    indptr = np.asarray(to_host(bridge.cp_connections.indptr), dtype=np.int64)
    nonzero_edges = []
    for pre in range(int(bridge.core_config.num_neurons)):
        for offset in range(int(indptr[pre]), int(indptr[pre + 1])):
            weight = float(data[offset])
            if weight != 0.0:
                nonzero_edges.append((pre, int(indices[offset]), weight))
    expected_pairs = {
        (pre, post)
        for pre in source["indices"] for post in target["indices"]
    }
    observed_pairs = {(pre, post) for pre, post, _ in nonzero_edges}
    observed_weights = [weight for _, _, weight in nonzero_edges]

    intrinsic = np.asarray(to_host(bridge.cp_intrinsic_current_pA))
    external = np.asarray(to_host(bridge.cp_external_input_current))
    pathway_spec = network["pathway"]
    checks = {
        "source_region_identity": source == {
            "name": source_spec["name"],
            "indices": list(range(source_spec["n_neurons"])),
            "n_neurons": source_spec["n_neurons"],
        },
        "target_region_identity": target == {
            "name": target_spec["name"],
            "indices": list(range(
                source_spec["n_neurons"],
                source_spec["n_neurons"] + target_spec["n_neurons"],
            )),
            "n_neurons": target_spec["n_neurons"],
        },
        "total_neuron_count": int(bridge.core_config.num_neurons)
        == source_spec["n_neurons"] + target_spec["n_neurons"],
        "region_config_counts": (
            source_cfg is not None
            and target_cfg is not None
            and int(source_cfg.n_neurons) == source_spec["n_neurons"]
            and int(target_cfg.n_neurons) == target_spec["n_neurons"]
        ),
        "target_region_intrinsic_field": (
            target_cfg is not None
            and float(target_cfg.intrinsic_current_pA)
            == network["target_intrinsic_current_pA"]
        ),
        "intrinsic_vector_exact": bool(
            np.all(intrinsic[source["indices"]] == 0.0)
            and np.all(
                intrinsic[target["indices"]]
                == np.float32(network["target_intrinsic_current_pA"])
            )
        ),
        "target_external_current_exact": bool(np.all(
            external[target["indices"]]
            == np.float32(spec["stimulus"]["target_external_current_pA"])
        )),
        "single_pathway": pathway is not None,
        "pathway_regions": (
            pathway is not None
            and pathway.from_region == source_spec["name"]
            and pathway.to_region == target_spec["name"]
        ),
        "pathway_density": pathway is not None
        and float(pathway.density) == pathway_spec["density"],
        "pathway_weight_mean": pathway is not None
        and float(pathway.weight_mean) == pathway_spec["weight_mean"],
        "pathway_weight_jitter": pathway is not None
        and float(pathway.weight_jitter) == pathway_spec["weight_jitter"],
        "pathway_plastic": pathway is not None
        and bool(pathway.plastic) is pathway_spec["plastic"],
        "pathway_receptor": pathway is not None
        and pathway.receptor == pathway_spec["receptor"],
        "full_source_target_topology": observed_pairs == expected_pairs,
        "nonzero_edge_count": len(nonzero_edges)
        == source_spec["n_neurons"] * target_spec["n_neurons"],
        "nonzero_weights_exact": bool(observed_weights)
        and all(weight == pathway_spec["weight_mean"] for weight in observed_weights),
    }
    result = {
        "network": network,
        "checks": checks,
        "all_exact": all(bool(value) for value in checks.values()),
        "observed_nonzero_edge_count": len(nonzero_edges),
        "observed_nonzero_weight_values": sorted(set(observed_weights)),
    }
    if set(checks) != NETWORK_CHECK_NAMES:
        raise AssertionError("network validator does not cover the locked check set")
    if not result["all_exact"]:
        raise ValueError(f"constructed network differs from locked spec: {result}")
    return result


def _network_validation_record_valid(record: Any, spec: dict) -> bool:
    expected_edge_count = (
        spec["network"]["source_region"]["n_neurons"]
        * spec["network"]["target_region"]["n_neurons"]
    )
    return bool(
        isinstance(record, dict)
        and record.get("network") == spec["network"]
        and isinstance(record.get("checks"), dict)
        and set(record["checks"]) == NETWORK_CHECK_NAMES
        and all(value is True for value in record["checks"].values())
        and record.get("all_exact") is True
        and record.get("observed_nonzero_edge_count") == expected_edge_count
        and record.get("observed_nonzero_weight_values")
        == [spec["network"]["pathway"]["weight_mean"]]
    )


def _validate_regions(regions: dict, neuron_count: int) -> None:
    source = regions.get("source", {})
    target = regions.get("target", {})
    if source.get("name") != "inhibitory_source" or target.get("name") != "gpi_snr":
        raise ValueError("bundle region names mismatch")
    source_indices = source.get("indices")
    target_indices = target.get("indices")
    if not all(isinstance(items, list) for items in (source_indices, target_indices)):
        raise ValueError("bundle region indices are missing")
    combined = source_indices + target_indices
    if (
        source.get("n_neurons") != len(source_indices)
        or target.get("n_neurons") != len(target_indices)
        or any(type(item) is not int for item in combined)
        or len(combined) != len(set(combined))
        or sorted(combined) != list(range(neuron_count))
    ):
        raise ValueError("bundle region identity mismatch")


def _required_bundle_hashes(cp_arrays: dict[str, dict], csr: dict) -> dict[str, str]:
    hashes = {
        alias: cp_arrays[attr]["sha256"]
        for alias, attr in BUNDLE_ARRAY_ATTRIBUTES.items()
    }
    hashes.update({
        "csr_data": csr["data"]["sha256"],
        "csr_indices": csr["indices"]["sha256"],
        "csr_indptr": csr["indptr"]["sha256"],
    })
    return hashes


def _capture_bundle_payload(bridge, spec: dict, spec_sha256: str,
                            origin_backend: str) -> dict:
    arrays = _allocated_cp_arrays(bridge)
    if bridge.cp_connections is None or not _is_sparse(bridge.cp_connections):
        raise ValueError("bridge has no CSR connections")
    encoded_arrays = {name: _encode_array(arrays[name]) for name in sorted(arrays)}
    csr = {
        "shape": list(bridge.cp_connections.shape),
        "data": _encode_array(bridge.cp_connections.data),
        "indices": _encode_array(bridge.cp_connections.indices),
        "indptr": _encode_array(bridge.cp_connections.indptr),
    }
    return {
        "schema": SCHEMA_BUNDLE,
        "verdict": spec["verdict"],
        "spec_sha256": spec_sha256,
        "source_identity": _source_identity(),
        "source_manifest": _source_manifest(),
        "execution_environment": _execution_environment(),
        "source_anchor_sha": spec["source_anchor_sha"],
        "seed": spec["seed"],
        "origin": origin_backend,
        "network": spec["network"],
        "network_validation": _validate_constructed_network(bridge, spec),
        "required_state_scope": spec["required_state_scope"],
        "initialization_disclosure": {
            "backend_native_bridge_initialized": True,
            "initialization_may_have_used_rng": True,
            "sealed_state_captured_after_initialization": True,
            "claim_of_no_rng_call": False,
        },
        "runtime_config_contract": _require_runtime_contract(
            bridge, spec, "default"
        ),
        "regions": {
            "source": _region_identity(bridge, "inhibitory_source"),
            "target": _region_identity(bridge, "gpi_snr"),
        },
        "cp_arrays": encoded_arrays,
        "connections_csr": csr,
        "required_bundle_array_sha256": _required_bundle_hashes(encoded_arrays, csr),
    }


def _write_new_json(path: Path, artifact: dict) -> None:
    _assert_output_available(path)
    data = json.dumps(artifact, sort_keys=True, indent=2) + "\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _assert_output_available(path: Path) -> None:
    if path.suffix != ".json":
        raise ValueError("output artifact must use a .json suffix")
    if not path.parent.is_dir():
        raise ValueError("output parent directory does not exist")
    if os.path.lexists(path):
        raise FileExistsError(f"output artifact already exists: {path}")


def _assert_backend(expected: str) -> None:
    if expected not in BACKENDS:
        raise ValueError(f"unsupported backend: {expected}")
    assert_backend(expected, note="V13 backend-state transplant diagnostic")
    _, actual = get_backend()
    if actual != expected:
        raise AssertionError(f"backend module mismatch: expected {expected}, got {actual}")


def _build_bridge_from_spec(spec: dict):
    network = spec["network"]
    return build_inhibitory_bridge(
        spec["seed"], float(network["target_intrinsic_current_pA"])
    )


def create_bundle(spec_path: Path, spec_sha256: str, origin: str, out: Path) -> dict:
    _assert_output_available(out)
    spec, locked_digest = load_locked_spec(spec_path, spec_sha256)
    if spec["seed"] in FORBIDDEN_V13_SEEDS:
        raise ValueError("formal V13 seeds are forbidden in this diagnostic")
    if origin not in spec["origins"]:
        raise ValueError("bundle origin is not allowed by the locked spec")
    _assert_backend(origin)
    bridge = _build_bridge_from_spec(spec)
    try:
        artifact = _seal_artifact(
            _capture_bundle_payload(bridge, spec, locked_digest, origin)
        )
        _write_new_json(out, artifact)
        return artifact
    finally:
        bridge.clear_simulation_state_and_gpu_memory()


def _validate_csr(csr: dict, neuron_count: int) -> None:
    if not isinstance(csr, dict) or set(csr) != {"shape", "data", "indices", "indptr"}:
        raise ValueError("bundle CSR record mismatch")
    if csr["shape"] != [neuron_count, neuron_count]:
        raise ValueError("bundle CSR shape mismatch")
    data = _decode_array(csr["data"], "CSR data")
    indices = _decode_array(csr["indices"], "CSR indices")
    indptr = _decode_array(csr["indptr"], "CSR indptr")
    if data.ndim != 1 or indices.shape != data.shape or indptr.shape != (neuron_count + 1,):
        raise ValueError("bundle CSR arrays are inconsistent")
    if indices.dtype.kind not in "iu" or indptr.dtype.kind not in "iu":
        raise ValueError("bundle CSR indices must be integers")
    if indptr[0] != 0 or indptr[-1] != data.size or np.any(np.diff(indptr) < 0):
        raise ValueError("bundle CSR indptr is invalid")
    if np.any(indices < 0) or np.any(indices >= neuron_count):
        raise ValueError("bundle CSR column index is invalid")


def _load_bundle(path: Path, spec: dict, spec_sha256: str) -> dict:
    try:
        artifact = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError("bundle is not valid JSON") from exc
    if not isinstance(artifact, dict) or artifact.get("schema") != SCHEMA_BUNDLE:
        raise ValueError("bundle schema mismatch")
    _validate_artifact_digest(artifact, "bundle")
    checks = {
        "verdict": artifact.get("verdict") == spec["verdict"],
        "spec": artifact.get("spec_sha256") == spec_sha256,
        "source": artifact.get("source_identity") == _source_identity(),
        "source_manifest": artifact.get("source_manifest") == _source_manifest(),
        "execution_environment": _environment_record_valid(
            artifact.get("execution_environment"), artifact.get("origin")
        ),
        "anchor": artifact.get("source_anchor_sha") == spec["source_anchor_sha"],
        "seed": artifact.get("seed") == spec["seed"] and artifact.get("seed") not in FORBIDDEN_V13_SEEDS,
        "origin": artifact.get("origin") in spec["origins"],
        "network": artifact.get("network") == spec["network"],
        "network_validation": _network_validation_record_valid(
            artifact.get("network_validation"), spec
        ),
        "required_state_scope": artifact.get("required_state_scope")
        == spec["required_state_scope"],
        "initialization_disclosed": artifact.get("initialization_disclosure") == {
            "backend_native_bridge_initialized": True,
            "initialization_may_have_used_rng": True,
            "sealed_state_captured_after_initialization": True,
            "claim_of_no_rng_call": False,
        },
        "runtime_contract": _runtime_contract_record_valid(
            artifact.get("runtime_config_contract"), spec, "default"
        ),
    }
    if not all(checks.values()):
        raise ValueError(f"bundle contract mismatch: {checks}")
    cp_arrays = artifact.get("cp_arrays")
    if not isinstance(cp_arrays, dict) or not cp_arrays:
        raise ValueError("bundle has no sealed cp arrays")
    for name, record in cp_arrays.items():
        if not name.startswith("cp_"):
            raise ValueError(f"bundle array name is not cp-prefixed: {name}")
        _decode_array(record, f"bundle array {name}")
    if not set(BUNDLE_ARRAY_ATTRIBUTES.values()).issubset(cp_arrays):
        raise ValueError("bundle is missing required cp arrays")
    neuron_count = int(_decode_array(cp_arrays["cp_firing_states"], "firing states").size)
    _validate_regions(artifact.get("regions", {}), neuron_count)
    _validate_csr(artifact.get("connections_csr"), neuron_count)
    expected_hashes = _required_bundle_hashes(cp_arrays, artifact["connections_csr"])
    if artifact.get("required_bundle_array_sha256") != expected_hashes:
        raise ValueError("required bundle-array digest map mismatch")
    if list(expected_hashes) != spec["required_bundle_arrays"]:
        raise ValueError("bundle does not implement the locked required-array order")
    return artifact


def _restore_bundle(bridge, bundle: dict) -> dict:
    xp, _ = get_backend()
    current = _allocated_cp_arrays(bridge)
    encoded = bundle["cp_arrays"]
    if set(current) != set(encoded):
        missing = sorted(set(encoded) - set(current))
        extra = sorted(set(current) - set(encoded))
        raise ValueError(f"allocated cp-array set differs from bundle: missing={missing}, extra={extra}")
    array_checks = {}
    for name in sorted(current):
        host = _decode_array(encoded[name], f"bundle array {name}")
        target = current[name]
        if tuple(target.shape) != host.shape or np.dtype(target.dtype).str != host.dtype.str:
            raise ValueError(f"allocated cp array {name} shape or dtype differs from bundle")
        target[...] = xp.asarray(host)
        array_checks[name] = _hash_array(target) == encoded[name]["sha256"]

    csr = bundle["connections_csr"]
    data = _decode_array(csr["data"], "CSR data")
    indices = _decode_array(csr["indices"], "CSR indices")
    indptr = _decode_array(csr["indptr"], "CSR indptr")
    bridge.cp_connections = get_sparse_module().csr_matrix(
        (xp.asarray(data), xp.asarray(indices), xp.asarray(indptr)),
        shape=tuple(csr["shape"]),
    )
    bridge._cached_inhibitory_mask = None
    bridge._cached_coo_matrix = None
    bridge.runtime_state.current_time_ms = 0.0
    bridge.runtime_state.current_time_step = 0
    csr_checks = {
        name: _hash_array(getattr(bridge.cp_connections, name)) == csr[name]["sha256"]
        for name in ("data", "indices", "indptr")
    }
    verification = {
        "cp_array_set_exact": set(current) == set(encoded),
        "cp_array_hashes_exact": all(array_checks.values()),
        "csr_hashes_exact": all(csr_checks.values()),
        "cp_array_checks": array_checks,
        "csr_checks": csr_checks,
    }
    verification["all_exact"] = all((
        verification["cp_array_set_exact"],
        verification["cp_array_hashes_exact"],
        verification["csr_hashes_exact"],
    ))
    if not verification["all_exact"]:
        raise ValueError("state restoration was not exact before first step")
    return verification


def _phase_bounds(spec: dict) -> dict[str, tuple[int, int]]:
    baseline = int(spec["steps"]["baseline"])
    inhibition = int(spec["steps"]["inhibition"])
    release = int(spec["steps"]["release"])
    return {
        "baseline": (0, baseline),
        "inhibition": (baseline, baseline + inhibition),
        "release": (baseline + inhibition, baseline + inhibition + release),
    }


def _phase_metrics(spikes: np.ndarray, target_indices: list[int], spec: dict) -> tuple[dict, dict]:
    rates, counts = {}, {}
    dt_ms = float(spec["steps"]["dt_ms"])
    target_count = len(target_indices)
    for phase, (start, stop) in _phase_bounds(spec).items():
        count = int(spikes[start:stop, target_indices].sum())
        counts[phase] = count
        rates[phase] = float(count / target_count / ((stop - start) * dt_ms / 1000.0))
    ratio = None if rates["baseline"] == 0.0 else float(rates["inhibition"] / rates["baseline"])
    rates["suppression_ratio"] = ratio
    return rates, counts


def execute_bundle(spec_path: Path, spec_sha256: str, bundle_path: Path,
                   backend: str, mode: str, out: Path) -> dict:
    _assert_output_available(out)
    spec, locked_digest = load_locked_spec(spec_path, spec_sha256)
    if backend not in spec["execution_backends"] or mode not in spec["modes"]:
        raise ValueError("execution backend or mode is outside the locked spec")
    bundle = _load_bundle(bundle_path, spec, locked_digest)
    _assert_backend(backend)
    bridge = _build_bridge_from_spec(spec)
    try:
        initialization_network = _validate_constructed_network(bridge, spec)
        initialization_contract = _require_runtime_contract(bridge, spec, "default")
        restore = _restore_bundle(bridge, bundle)
        restored_network = _validate_constructed_network(bridge, spec)
        requested_flag = mode == "deterministic_transpose_matvec"
        bridge.core_config.deterministic_transpose_matvec = requested_flag
        readback_flag = bool(bridge.core_config.deterministic_transpose_matvec)
        if readback_flag != requested_flag:
            raise ValueError("deterministic_transpose_matvec mode did not read back exactly")
        runtime_contract = _require_runtime_contract(bridge, spec, mode)

        xp, _ = get_backend()
        source_indices = bundle["regions"]["source"]["indices"]
        target_indices = bundle["regions"]["target"]["indices"]
        source = xp.asarray(source_indices, dtype=xp.int64)
        target = xp.asarray(target_indices, dtype=xp.int64)
        bounds = _phase_bounds(spec)
        total_steps = bounds["release"][1]
        n = int(bridge.core_config.num_neurons)
        trajectories = {
            "v": np.empty((total_steps, n), dtype=np.float32),
            "u": np.empty((total_steps, n), dtype=np.float32),
            "g_e": np.empty((total_steps, n), dtype=np.float32),
            "g_i": np.empty((total_steps, n), dtype=np.float32),
            "spikes": np.empty((total_steps, n), dtype=np.bool_),
        }
        source_spikes = np.empty((total_steps, len(source_indices)), dtype=np.bool_)
        target_external = np.empty((total_steps, len(target_indices)), dtype=np.float32)
        initial_weight_hash = _hash_array(bridge.cp_connections.data)
        initial_intrinsic_hash = _hash_array(bridge.cp_intrinsic_current_pA)
        inhibition_start, inhibition_stop = bounds["inhibition"]
        source_current = xp.float32(spec["stimulus"]["source_current_pA"])
        target_current = xp.float32(spec["stimulus"]["target_external_current_pA"])

        for step in range(total_steps):
            bridge.cp_external_input_current[:] = target_current
            if inhibition_start <= step < inhibition_stop:
                bridge.cp_external_input_current[source] = source_current
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_ms += bridge.core_config.dt_ms
            bridge.runtime_state.current_time_step += 1
            trajectories["v"][step] = np.asarray(to_host(bridge.cp_membrane_potential_v))
            trajectories["u"][step] = np.asarray(to_host(bridge.cp_recovery_variable_u))
            trajectories["g_e"][step] = np.asarray(to_host(bridge.cp_conductance_g_e))
            trajectories["g_i"][step] = np.asarray(to_host(bridge.cp_conductance_g_i))
            trajectories["spikes"][step] = np.asarray(to_host(bridge.cp_firing_states))
            source_spikes[step] = trajectories["spikes"][step, source_indices]
            target_external[step] = np.asarray(to_host(bridge.cp_external_input_current[target]))
        synchronize()

        final_weight_hash = _hash_array(bridge.cp_connections.data)
        final_intrinsic_hash = _hash_array(bridge.cp_intrinsic_current_pA)
        rates, counts = _phase_metrics(trajectories["spikes"], target_indices, spec)
        source_counts = {
            phase: int(source_spikes[start:stop].sum())
            for phase, (start, stop) in bounds.items()
        }
        source_schedule_valid = (
            source_counts["baseline"] == 0
            and source_counts["inhibition"] > 0
            and source_counts["release"] == 0
        )
        validations = {
            "pre_step_restore_exact": restore["all_exact"],
            "mode_flag_exact": readback_flag == requested_flag,
            "source_spike_schedule_valid": source_schedule_valid,
            "target_external_current_zero": bool(np.all(target_external == 0.0)),
            "weights_immutable": initial_weight_hash == final_weight_hash,
            "intrinsic_current_immutable": initial_intrinsic_hash == final_intrinsic_hash,
            "finite_continuous_trajectories": all(
                np.all(np.isfinite(trajectories[name])) for name in CONTINUOUS_TRAJECTORIES
            ),
            "baseline_rate_defined": rates["baseline"] > 0.0,
        }
        encoded = {name: _encode_array(trajectories[name]) for name in TRAJECTORIES}
        artifact = _seal_artifact({
            "schema": SCHEMA_RUN,
            "verdict": spec["verdict"],
            "spec_sha256": locked_digest,
            "source_identity": _source_identity(),
            "source_manifest": _source_manifest(),
            "execution_environment": _execution_environment(),
            "source_anchor_sha": spec["source_anchor_sha"],
            "bundle_artifact_sha256": bundle["artifact_sha256"],
            "bundle_file_sha256": _sha256(bundle_path.read_bytes()),
            "seed": spec["seed"],
            "origin": bundle["origin"],
            "execution_backend": backend,
            "mode": mode,
            "deterministic_transpose_matvec_requested": requested_flag,
            "deterministic_transpose_matvec_readback": readback_flag,
            "initialization_disclosure": {
                "backend_native_bridge_initialized_before_restore": True,
                "initialization_may_have_used_rng": True,
                "claim_of_no_rng_call": False,
                "initialization_runtime_contract": initialization_contract,
                "all_allocated_cp_ndarrays_overwritten_exactly": restore[
                    "cp_array_hashes_exact"
                ] and restore["cp_array_set_exact"],
                "csr_overwritten_exactly": restore["csr_hashes_exact"],
                "no_resampled_array_state_survived_restore": restore["all_exact"],
            },
            "runtime_config_contract": runtime_contract,
            "regions": bundle["regions"],
            "network": spec["network"],
            "network_validation": {
                "before_restore": initialization_network,
                "after_restore": restored_network,
            },
            "required_state_scope": spec["required_state_scope"],
            "steps": spec["steps"],
            "stimulus": spec["stimulus"],
            "tolerance": _comparison_tolerance(spec),
            "pre_step_restore_verification": restore,
            "trajectories": encoded,
            "trajectory_sha256": {name: encoded[name]["sha256"] for name in TRAJECTORIES},
            "trajectory_step_sha256": {
                name: _trajectory_step_hashes(trajectories[name]) for name in TRAJECTORIES
            },
            "audit_arrays": {
                "source_spikes": _encode_array(source_spikes),
                "target_external_current": _encode_array(target_external),
            },
            "source_spike_counts_by_phase": source_counts,
            "target_spike_counts_by_phase": counts,
            "target_rates_hz_by_phase": rates,
            "initial_weight_sha256": initial_weight_hash,
            "final_weight_sha256": final_weight_hash,
            "initial_intrinsic_sha256": initial_intrinsic_hash,
            "final_intrinsic_sha256": final_intrinsic_hash,
            "validations": validations,
            "instrument_valid": all(validations.values()),
        })
        _write_new_json(out, artifact)
        return artifact
    finally:
        bridge.clear_simulation_state_and_gpu_memory()


def _load_run(path: Path, spec: dict, spec_sha256: str) -> dict:
    try:
        artifact = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError("run artifact is not valid JSON") from exc
    if not isinstance(artifact, dict) or artifact.get("schema") != SCHEMA_RUN:
        raise ValueError("run artifact schema mismatch")
    _validate_artifact_digest(artifact, "run")
    checks = {
        "verdict": artifact.get("verdict") == spec["verdict"],
        "spec": artifact.get("spec_sha256") == spec_sha256,
        "source": artifact.get("source_identity") == _source_identity(),
        "source_manifest": artifact.get("source_manifest") == _source_manifest(),
        "execution_environment": _environment_record_valid(
            artifact.get("execution_environment"), artifact.get("execution_backend")
        ),
        "anchor": artifact.get("source_anchor_sha") == spec["source_anchor_sha"],
        "seed": artifact.get("seed") == spec["seed"] and artifact.get("seed") not in FORBIDDEN_V13_SEEDS,
        "origin": artifact.get("origin") in spec["origins"],
        "backend": artifact.get("execution_backend") in spec["execution_backends"],
        "mode": artifact.get("mode") in spec["modes"],
        "network": artifact.get("network") == spec["network"],
        "network_validation": (
            _network_validation_record_valid(
                artifact.get("network_validation", {}).get("before_restore"), spec
            )
            and _network_validation_record_valid(
                artifact.get("network_validation", {}).get("after_restore"), spec
            )
        ),
        "required_state_scope": artifact.get("required_state_scope")
        == spec["required_state_scope"],
        "steps": artifact.get("steps") == spec["steps"],
        "stimulus": artifact.get("stimulus") == spec["stimulus"],
        "tolerance": artifact.get("tolerance") == _comparison_tolerance(spec),
        "mode_flag": (
            artifact.get("deterministic_transpose_matvec_requested")
            == (artifact.get("mode") == "deterministic_transpose_matvec")
            and artifact.get("deterministic_transpose_matvec_readback")
            == (artifact.get("mode") == "deterministic_transpose_matvec")
        ),
        "restore": artifact.get("pre_step_restore_verification", {}).get("all_exact") is True,
        "initialization_disclosed": (
            artifact.get("initialization_disclosure", {}).get(
                "backend_native_bridge_initialized_before_restore"
            ) is True
            and artifact.get("initialization_disclosure", {}).get(
                "initialization_may_have_used_rng"
            ) is True
            and artifact.get("initialization_disclosure", {}).get("claim_of_no_rng_call") is False
            and artifact.get("initialization_disclosure", {}).get(
                "all_allocated_cp_ndarrays_overwritten_exactly"
            ) is True
            and artifact.get("initialization_disclosure", {}).get(
                "csr_overwritten_exactly"
            ) is True
            and artifact.get("initialization_disclosure", {}).get(
                "no_resampled_array_state_survived_restore"
            ) is True
            and _runtime_contract_record_valid(
                artifact.get("initialization_disclosure", {}).get(
                    "initialization_runtime_contract"
                ),
                spec,
                "default",
            )
        ),
        "runtime_contract": _runtime_contract_record_valid(
            artifact.get("runtime_config_contract"), spec, artifact.get("mode")
        ),
        "instrument": artifact.get("instrument_valid") is True,
    }
    if not all(checks.values()):
        raise ValueError(f"run contract mismatch: {checks}")
    _validate_sha256(artifact.get("bundle_artifact_sha256"), "bundle artifact SHA-256")
    _validate_sha256(artifact.get("bundle_file_sha256"), "bundle file SHA-256")
    records = artifact.get("trajectories")
    if not isinstance(records, dict) or set(records) != set(TRAJECTORIES):
        raise ValueError("run trajectory set mismatch")
    total_steps = sum(int(spec["steps"][name]) for name in ("baseline", "inhibition", "release"))
    regions = artifact.get("regions", {})
    neuron_count = len(regions.get("source", {}).get("indices", [])) + len(
        regions.get("target", {}).get("indices", [])
    )
    _validate_regions(regions, neuron_count)
    arrays = {}
    for name in TRAJECTORIES:
        arrays[name] = _decode_array(records[name], f"trajectory {name}")
        if arrays[name].shape != (total_steps, neuron_count):
            raise ValueError(f"run trajectory {name} shape mismatch")
        if artifact.get("trajectory_sha256", {}).get(name) != records[name]["sha256"]:
            raise ValueError(f"run trajectory {name} hash mismatch")
        if artifact.get("trajectory_step_sha256", {}).get(name) != _trajectory_step_hashes(
            arrays[name]
        ):
            raise ValueError(f"run trajectory {name} per-step hashes mismatch")
    if arrays["spikes"].dtype != np.bool_:
        raise ValueError("spike trajectory must be boolean")

    source_indices = regions["source"]["indices"]
    target_indices = regions["target"]["indices"]
    source_spikes = _decode_array(
        artifact.get("audit_arrays", {}).get("source_spikes"), "source spikes"
    )
    target_external = _decode_array(
        artifact.get("audit_arrays", {}).get("target_external_current"),
        "target external current",
    )
    if (
        source_spikes.shape != (total_steps, len(source_indices))
        or not np.array_equal(source_spikes, arrays["spikes"][:, source_indices])
    ):
        raise ValueError("source spike schedule does not match full raster")
    if target_external.shape != (total_steps, len(target_indices)) or np.any(target_external != 0.0):
        raise ValueError("target external-current zero audit failed")
    rates, counts = _phase_metrics(arrays["spikes"], target_indices, spec)
    if artifact.get("target_rates_hz_by_phase") != rates or artifact.get("target_spike_counts_by_phase") != counts:
        raise ValueError("target phase metrics mismatch")
    source_counts = {
        phase: int(source_spikes[start:stop].sum())
        for phase, (start, stop) in _phase_bounds(spec).items()
    }
    if artifact.get("source_spike_counts_by_phase") != source_counts:
        raise ValueError("source spike phase metrics mismatch")
    source_schedule_valid = (
        source_counts["baseline"] == 0
        and source_counts["inhibition"] > 0
        and source_counts["release"] == 0
    )
    for label in (
        "initial_weight_sha256", "final_weight_sha256",
        "initial_intrinsic_sha256", "final_intrinsic_sha256",
    ):
        _validate_sha256(artifact.get(label), label)
    expected_validations = {
        "pre_step_restore_exact": True,
        "mode_flag_exact": artifact["deterministic_transpose_matvec_readback"]
        == artifact["deterministic_transpose_matvec_requested"],
        "source_spike_schedule_valid": source_schedule_valid,
        "target_external_current_zero": bool(np.all(target_external == 0.0)),
        "weights_immutable": artifact["initial_weight_sha256"] == artifact["final_weight_sha256"],
        "intrinsic_current_immutable": (
            artifact["initial_intrinsic_sha256"] == artifact["final_intrinsic_sha256"]
        ),
        "finite_continuous_trajectories": all(
            np.all(np.isfinite(arrays[name])) for name in CONTINUOUS_TRAJECTORIES
        ),
        "baseline_rate_defined": rates["baseline"] > 0.0,
    }
    if (
        artifact.get("validations") != expected_validations
        or artifact.get("instrument_valid") != all(expected_validations.values())
    ):
        raise ValueError("run validation block does not match sealed measurements")
    return artifact


def _first_mask_divergence(mask: np.ndarray) -> dict | None:
    by_step = np.any(mask, axis=tuple(range(1, mask.ndim)))
    steps = np.flatnonzero(by_step)
    if not steps.size:
        return None
    step = int(steps[0])
    indices = np.flatnonzero(mask[step]).astype(int).tolist()
    return {"step": step, "neuron_indices": indices, "differing_neuron_count": len(indices)}


def _first_byte_divergence(left: np.ndarray, right: np.ndarray) -> dict | None:
    if left.shape != right.shape or left.dtype != right.dtype:
        raise ValueError("array shape or dtype mismatch")
    left_bytes = np.ascontiguousarray(left).view(np.uint8).reshape(left.shape + (left.dtype.itemsize,))
    right_bytes = np.ascontiguousarray(right).view(np.uint8).reshape(right.shape + (right.dtype.itemsize,))
    return _first_mask_divergence(np.any(left_bytes != right_bytes, axis=-1))


def _first_tolerance_divergence(left: np.ndarray, right: np.ndarray,
                                tolerance: dict) -> dict | None:
    if left.shape != right.shape or left.dtype != right.dtype:
        raise ValueError("array shape or dtype mismatch")
    close = np.isclose(
        left, right,
        rtol=float(tolerance["rtol"]),
        atol=float(tolerance["atol"]),
        equal_nan=True,
    )
    return _first_mask_divergence(~close)


def _pair_comparison(left: dict, right: dict, tolerance: dict) -> dict:
    if {left["execution_backend"], right["execution_backend"]} != set(BACKENDS):
        raise ValueError("comparison requires one NumPy and one CuPy run")
    equal_fields = (
        "bundle_artifact_sha256", "bundle_file_sha256", "seed", "origin",
        "mode", "steps", "stimulus", "source_identity", "spec_sha256",
    )
    mismatched = [name for name in equal_fields if left.get(name) != right.get(name)]
    if mismatched:
        raise ValueError(f"run artifacts do not share the same sealed bundle and mode: {mismatched}")

    byte_divergence, tolerance_divergence = {}, {}
    hashes_equal = {}
    for name in TRAJECTORIES:
        left_array = _decode_array(left["trajectories"][name], f"left {name}")
        right_array = _decode_array(right["trajectories"][name], f"right {name}")
        byte_divergence[name] = _first_byte_divergence(left_array, right_array)
        tolerance_divergence[name] = (
            _first_tolerance_divergence(left_array, right_array, tolerance)
            if name in CONTINUOUS_TRAJECTORIES else None
        )
        hashes_equal[name] = left["trajectory_sha256"][name] == right["trajectory_sha256"][name]

    byte_rows = [(name, row) for name, row in byte_divergence.items() if row is not None]
    tolerance_rows = [(name, row) for name, row in tolerance_divergence.items() if row is not None]
    first_byte = min(byte_rows, key=lambda item: (item[1]["step"], item[0])) if byte_rows else None
    first_tolerance = min(
        tolerance_rows, key=lambda item: (item[1]["step"], item[0])
    ) if tolerance_rows else None
    return {
        "origin": left["origin"],
        "mode": left["mode"],
        "bundle_artifact_sha256": left["bundle_artifact_sha256"],
        "run_artifacts": {
            left["execution_backend"]: left["artifact_sha256"],
            right["execution_backend"]: right["artifact_sha256"],
        },
        "tolerance": tolerance,
        "trajectory_hashes_equal": hashes_equal,
        "first_byte_divergence_by_trajectory": byte_divergence,
        "first_tolerance_divergence_by_trajectory": tolerance_divergence,
        "first_byte_exact_divergence": (
            None if first_byte is None else {"trajectory": first_byte[0], **first_byte[1]}
        ),
        "first_tolerance_divergence": (
            None if first_tolerance is None
            else {"trajectory": first_tolerance[0], **first_tolerance[1]}
        ),
        "spikes_exact": byte_divergence["spikes"] is None,
        "continuous_within_tolerance": not tolerance_rows,
    }


def compare_runs(spec_path: Path, spec_sha256: str, left_path: Path,
                 right_path: Path, out: Path) -> dict:
    _assert_output_available(out)
    spec, locked_digest = load_locked_spec(spec_path, spec_sha256)
    left = _load_run(left_path, spec, locked_digest)
    right = _load_run(right_path, spec, locked_digest)
    pair = _pair_comparison(left, right, _comparison_tolerance(spec))
    artifact = _seal_artifact({
        "schema": SCHEMA_COMPARISON,
        "verdict": spec["verdict"],
        "spec_sha256": locked_digest,
        "source_identity": _source_identity(),
        **pair,
    })
    _write_new_json(out, artifact)
    return artifact


def _compare_bundle_arrays(left: dict, right: dict, tolerance_contract: dict) -> dict:
    left_arrays, right_arrays = left["cp_arrays"], right["cp_arrays"]
    if set(left_arrays) != set(right_arrays):
        raise ValueError("origin bundles have different full cp-array sets")
    parameter_names = sorted(
        name for name in left_arrays
        if name.startswith("cp_izh_") or name in PARAMETER_ARRAY_NAMES
    )
    parameter_comparison = {}
    for name in parameter_names:
        a = _decode_array(left_arrays[name], f"numpy-origin {name}")
        b = _decode_array(right_arrays[name], f"cupy-origin {name}")
        byte = _first_byte_divergence(a.reshape(1, -1), b.reshape(1, -1))
        first_tolerance = None
        if a.dtype.kind in "fc":
            first_tolerance = _first_tolerance_divergence(
                a.reshape(1, -1), b.reshape(1, -1), tolerance_contract
            )
        parameter_comparison[name] = {
            "byte_exact": byte is None,
            "within_tolerance": first_tolerance is None if a.dtype.kind in "fc" else None,
            "first_byte_difference": byte,
            "first_tolerance_difference": first_tolerance,
        }
    left_csr, right_csr = left["connections_csr"], right["connections_csr"]
    topology = {
        name: left_csr[name]["sha256"] == right_csr[name]["sha256"]
        for name in ("indices", "indptr")
    }
    return {
        "full_cp_array_names": sorted(left_arrays),
        "parameter_arrays": parameter_comparison,
        "topology_exact": all(topology.values()) and left_csr["shape"] == right_csr["shape"],
        "topology_checks": topology,
        "weight_data_exact": left_csr["data"]["sha256"] == right_csr["data"]["sha256"],
    }


def aggregate_matrix(spec_path: Path, spec_sha256: str, bundle_paths: list[Path],
                     run_paths: list[Path], out: Path) -> dict:
    _assert_output_available(out)
    spec, locked_digest = load_locked_spec(spec_path, spec_sha256)
    if len(bundle_paths) != 2:
        raise ValueError("aggregate requires exactly two origin bundles")
    bundles = [_load_bundle(path, spec, locked_digest) for path in bundle_paths]
    by_origin = {bundle["origin"]: bundle for bundle in bundles}
    bundle_file_sha256 = {
        bundle["origin"]: _sha256(path.read_bytes())
        for path, bundle in zip(bundle_paths, bundles)
    }
    if len(by_origin) != 2 or set(by_origin) != set(BACKENDS):
        raise ValueError("aggregate requires one unique NumPy-origin and one unique CuPy-origin bundle")
    if len({bundle["artifact_sha256"] for bundle in bundles}) != 2:
        raise ValueError("aggregate origin bundles must be distinct artifacts")
    for field in ("seed", "source_identity", "spec_sha256", "source_anchor_sha"):
        if bundles[0].get(field) != bundles[1].get(field):
            raise ValueError(f"origin bundle {field} mismatch")

    if len(run_paths) != spec["expected_matrix_cells"]:
        raise ValueError(
            f"aggregate requires exactly {spec['expected_matrix_cells']} run artifacts"
        )
    runs = [_load_run(path, spec, locked_digest) for path in run_paths]
    matrix = {}
    for run in runs:
        key = (run["origin"], run["execution_backend"], run["mode"])
        if key in matrix:
            raise ValueError(f"duplicate aggregate matrix cell: {key}")
        bundle = by_origin.get(run["origin"])
        if (
            bundle is None
            or run["bundle_artifact_sha256"] != bundle["artifact_sha256"]
            or run["bundle_file_sha256"] != bundle_file_sha256[run["origin"]]
        ):
            raise ValueError("run artifact is bound to the wrong origin bundle")
        matrix[key] = run
    expected = {
        (origin, backend, mode)
        for origin in BACKENDS for backend in BACKENDS for mode in MODES
    }
    if len(expected) != spec["expected_matrix_cells"]:
        raise ValueError("locked expected_matrix_cells disagrees with origin/backend/mode product")
    if set(matrix) != expected:
        missing = sorted(expected - set(matrix))
        extra = sorted(set(matrix) - expected)
        raise ValueError(f"aggregate matrix cells mismatch: missing={missing}, extra={extra}")

    comparisons = {}
    for origin in BACKENDS:
        for mode in MODES:
            key = f"{origin}:{mode}"
            comparisons[key] = _pair_comparison(
                matrix[(origin, "numpy", mode)], matrix[(origin, "cupy", mode)],
                _comparison_tolerance(spec),
            )
    artifact = _seal_artifact({
        "schema": SCHEMA_AGGREGATE,
        "verdict": spec["verdict"],
        "spec_sha256": locked_digest,
        "source_identity": _source_identity(),
        "seed": spec["seed"],
        "tolerance": _comparison_tolerance(spec),
        "bundles": {origin: by_origin[origin]["artifact_sha256"] for origin in BACKENDS},
        "matrix_cell_count": len(matrix),
        "matrix_cells": [
            {"origin": key[0], "execution_backend": key[1], "mode": key[2],
             "artifact_sha256": matrix[key]["artifact_sha256"]}
            for key in sorted(matrix)
        ],
        "within_origin_mode_comparisons": comparisons,
        "origin_bundle_comparison": _compare_bundle_arrays(
            by_origin["numpy"], by_origin["cupy"], _comparison_tolerance(spec)
        ),
        "matrix_complete": True,
        "outcome": "DIAGNOSTIC_MATRIX_COMPLETE",
    })
    _write_new_json(out, artifact)
    return artifact


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC_PATH)
    parser.add_argument("--spec-sha256", required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--create-bundle", action="store_true")
    modes.add_argument("--run", action="store_true")
    modes.add_argument("--compare", action="store_true")
    modes.add_argument("--aggregate", action="store_true")
    parser.add_argument("--origin", choices=BACKENDS)
    parser.add_argument("--backend", choices=BACKENDS)
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--left", type=Path)
    parser.add_argument("--right", type=Path)
    parser.add_argument("--bundles", nargs="*", type=Path)
    parser.add_argument("--runs", nargs="*", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.create_bundle:
        if args.origin is None or any(value is not None for value in (
            args.backend, args.mode, args.bundle, args.left, args.right, args.bundles, args.runs,
        )):
            raise SystemExit("--create-bundle requires --origin only")
        artifact = create_bundle(args.spec, args.spec_sha256, args.origin, args.out)
    elif args.run:
        if args.backend is None or args.mode is None or args.bundle is None or any(
            value is not None for value in (args.origin, args.left, args.right, args.bundles, args.runs)
        ):
            raise SystemExit("--run requires --backend, --mode, and --bundle only")
        artifact = execute_bundle(
            args.spec, args.spec_sha256, args.bundle, args.backend, args.mode, args.out
        )
    elif args.compare:
        if args.left is None or args.right is None or any(value is not None for value in (
            args.origin, args.backend, args.mode, args.bundle, args.bundles, args.runs,
        )):
            raise SystemExit("--compare requires --left and --right only")
        artifact = compare_runs(
            args.spec, args.spec_sha256, args.left, args.right, args.out
        )
    else:
        if args.bundles is None or args.runs is None or any(value is not None for value in (
            args.origin, args.backend, args.mode, args.bundle, args.left, args.right,
        )):
            raise SystemExit("--aggregate requires --bundles and --runs only")
        artifact = aggregate_matrix(
            args.spec, args.spec_sha256, args.bundles, args.runs, args.out
        )
    print(json.dumps({
        "artifact": str(args.out),
        "artifact_sha256": artifact["artifact_sha256"],
        "schema": artifact["schema"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
