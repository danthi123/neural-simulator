#!/usr/bin/env python3
"""Authenticated project-operational companion physiology observations.

This runner records raw compact traces for the Stage B NaP phased-voltage and
HCN current-family assays.  It deliberately computes no physiology scores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

if os.environ.get("SIM_BACKEND") not in {None, "numpy"}:
    raise RuntimeError("V14 Stage B companion runner requires SIM_BACKEND=numpy")
os.environ["SIM_BACKEND"] = "numpy"

import numpy as np

from sim.backend import get_backend, to_host
from sim.bridge import SimulationBridge
from sim.config import GPUConfig, RuntimeState, VisualizationConfig
from sim.snr_packet_runtime import (
    RuntimeSNrPacketBinding,
    load_runtime_snr_packet_bindings,
    runtime_binding_manifest_bytes,
)
from tools.compact_trace import CompactTraceError, save_compact_trace
from tools.lab import before_after
from research.runners.v14_stageB_physiology import (
    _binding_provenance,
    _build_config,
    _candidate_echo,
    _canonical_bytes,
    _load_candidate_release,
    _load_parameter_document,
)


OUTPUT_SCHEMA = "v14-snr-stageB-companion-physiology-v1"
PROTOCOL_SCHEMA = "v14-snr-stageB-intrinsic-protocol-v3"
DT_MS = 0.05
NAP_BASELINE_S = 2.0
NAP_POST_LESION_S = 1.0
HCN_BASELINE_S = 0.25
HCN_STEP_S = 1.0
HCN_CURRENT_FAMILY_PA = (0.0, -20.0, -40.0, -60.0, -80.0, -100.0, -120.0)


class StageBCompanionRunnerError(ValueError):
    """Raised before a companion observation can be truthfully published."""


def _digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(value: Any, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise StageBCompanionRunnerError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _bound_file(
    root: Path, path_value: str | Path, digest_value: str, context: str,
) -> tuple[Path, dict[str, str], dict[str, Any]]:
    digest = _sha256(digest_value, f"{context} sha256")
    supplied = Path(path_value).expanduser()
    path = supplied.resolve()
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise StageBCompanionRunnerError(f"{context} escapes repository_root") from exc
    if path.is_symlink() or not path.is_file():
        raise StageBCompanionRunnerError(f"{context} must be a regular file")
    raw = path.read_bytes()
    if _digest_bytes(raw) != digest:
        raise StageBCompanionRunnerError(f"{context} digest does not match")
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise StageBCompanionRunnerError(f"{context} is not valid JSON") from exc
    if not isinstance(document, dict):
        raise StageBCompanionRunnerError(f"{context} must contain a JSON object")
    binding = {"path": PurePosixPath(*relative.parts).as_posix(), "sha256": digest}
    return path, binding, document


def _load_inputs(
    root: Path,
    parameter_document_path: str | Path,
    parameter_document_sha256: str,
    protocol_spec_path: str | Path,
    protocol_spec_sha256: str,
    causal_gate_path: str | Path,
    causal_gate_sha256: str,
    assay: str,
) -> tuple[
    dict[str, Any], dict[str, str], dict[str, Any], dict[str, str], dict[str, str]
]:
    parameter_path, parameter_binding, raw_parameter = _bound_file(
        root, parameter_document_path, parameter_document_sha256, "parameter document"
    )
    if parameter_path.read_bytes() != _canonical_bytes(raw_parameter):
        raise StageBCompanionRunnerError("parameter document must be canonical JSON")
    try:
        parameter = _load_parameter_document(parameter_path.read_text(encoding="ascii"))
    except (UnicodeError, ValueError) as exc:
        raise StageBCompanionRunnerError(f"parameter document does not authenticate: {exc}") from exc
    expected_arm = "nap_lesion" if assay == "nap" else "hcn_baseline_lesion"
    if parameter["arm"] != expected_arm:
        raise StageBCompanionRunnerError(
            f"{assay} companion requires parameter-document arm {expected_arm!r}"
        )

    protocol_path, protocol_binding, protocol = _bound_file(
        root, protocol_spec_path, protocol_spec_sha256, "protocol spec"
    )
    causal_path, causal_binding, causal_gate = _bound_file(
        root, causal_gate_path, causal_gate_sha256, "causal gate"
    )
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        raise StageBCompanionRunnerError("protocol spec has the wrong schema")
    arms = protocol.get("arms")
    arm = arms.get(expected_arm) if isinstance(arms, Mapping) else None
    if not isinstance(arm, Mapping):
        raise StageBCompanionRunnerError("protocol spec does not define the companion arm")
    authority = protocol.get("causal_gate_authority")
    if (
        not isinstance(authority, Mapping)
        or authority.get("path") != causal_binding["path"]
    ):
        raise StageBCompanionRunnerError("protocol spec names a different causal gate")
    if causal_gate.get("authorized_analysis_protocol") != protocol_binding:
        raise StageBCompanionRunnerError(
            "causal gate does not authorize the exact supplied protocol"
        )
    execution = protocol.get("execution")
    if not isinstance(execution, Mapping) or execution.get("dt_ms") != DT_MS:
        raise StageBCompanionRunnerError("protocol spec changed the companion timestep")
    _validate_assay_contract(arm, assay)
    return parameter, parameter_binding, protocol, protocol_binding, causal_binding


def _validate_assay_contract(arm: Mapping[str, Any], assay: str) -> None:
    if assay == "nap":
        contract = arm.get("mean_voltage_change")
        schedule = contract.get("phase_schedule") if isinstance(contract, Mapping) else None
        intervention = contract.get("intervention") if isinstance(contract, Mapping) else None
        if schedule != {
            "intact_baseline_duration_s": 2.0,
            "lesion_onset_s": 2.0,
            "post_lesion_duration_s": 1.0,
            "total_duration_s": 3.0,
        }:
            raise StageBCompanionRunnerError("protocol spec changed the NaP phase schedule")
        if not isinstance(intervention, Mapping) or intervention.get(
            "nap_conductance_fraction_after_onset"
        ) != 0.0 or intervention.get("lesion_onset_sample_s") != 2.0:
            raise StageBCompanionRunnerError("protocol spec changed the NaP intervention")
        if contract.get("same_cell_requirement") != (
            "one continuously simulated cell; do not substitute independently initialized intact and lesion traces"
        ):
            raise StageBCompanionRunnerError("protocol spec changed the NaP same-cell requirement")
        return

    contract = arm.get("hyperpolarized_input_resistance")
    schedule = contract.get("phase_schedule") if isinstance(contract, Mapping) else None
    conditions = contract.get("conditions") if isinstance(contract, Mapping) else None
    if list(contract.get("current_family_pA", ())) != list(HCN_CURRENT_FAMILY_PA):
        raise StageBCompanionRunnerError("protocol spec changed the HCN current family")
    if schedule != {
        "baseline_duration_s": 0.25,
        "current_step_duration_s": 1.0,
        "current_step_onset_s": 0.25,
        "steady_state_window_relative_to_step_s": [0.9, 1.0],
        "total_duration_s": 1.25,
    }:
        raise StageBCompanionRunnerError("protocol spec changed the HCN phase schedule")
    ttx = conditions.get("shared_ttx_equivalent") if isinstance(conditions, Mapping) else None
    intact = conditions.get("intact_hcn") if isinstance(conditions, Mapping) else None
    lesion = conditions.get("hcn_complete_lesion") if isinstance(conditions, Mapping) else None
    if (
        not isinstance(ttx, Mapping)
        or ttx.get("fast_na_conductance_fraction_of_candidate") != 0.0
        or ttx.get("nap_conductance_fraction_of_candidate") != 0.0
        or not isinstance(intact, Mapping)
        or intact.get("g_hcn_fraction_of_candidate") != 1.0
        or not isinstance(lesion, Mapping)
        or lesion.get("g_hcn_fraction_of_candidate") != 0.0
    ):
        raise StageBCompanionRunnerError("protocol spec changed the HCN conductance conditions")


def _initialize_bridge(root: Path, references: Mapping[str, str]):
    config = _build_config(references)
    bindings = load_runtime_snr_packet_bindings(config, source_root=root)
    binding = bindings.get("snr")
    if binding is None or len(bindings) != 1:
        raise StageBCompanionRunnerError("authenticated references did not produce one SNr binding")
    if (
        binding.packet_path != references["packet_path"]
        or binding.packet_file_sha256 != references["packet_sha256"]
        or binding.authority_policy_sha256 != references["policy_sha256"]
    ):
        raise StageBCompanionRunnerError("runtime binding does not match packet references")
    bridge = SimulationBridge(
        core_config=config,
        viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(),
        gpu_config=GPUConfig(enable_profiling=False),
        simulation_source_root=str(root),
    )
    bridge._initialize_simulation_data()
    if not bridge.is_initialized or set(bridge.snr_packet_bindings) != {"snr"}:
        bridge.clear_simulation_state_and_gpu_memory()
        raise StageBCompanionRunnerError("authenticated SNr bridge initialization failed")
    return config, bindings, binding, bridge


def _scalar_array(bridge: Any, field: str, *, nonnegative: bool = False) -> np.ndarray:
    value = getattr(bridge, field, None)
    if value is None:
        raise StageBCompanionRunnerError(f"runtime field {field} was not initialized")
    host = np.asarray(to_host(value), dtype=np.float64)
    if host.shape != (1,) or not np.all(np.isfinite(host)):
        raise StageBCompanionRunnerError(f"runtime field {field} has an invalid value")
    if nonnegative and np.any(host < 0.0):
        raise StageBCompanionRunnerError(f"runtime field {field} is negative")
    return host


def _zero_conductance(bridge: Any, field: str) -> dict[str, Any]:
    values = getattr(bridge, field)
    before, after, _ = before_after(
        f"{field} complete lesion",
        lambda: float(_scalar_array(bridge, field, nonnegative=True)[0]),
        lambda: values.__setitem__(Ellipsis, 0.0),
    )
    if after != 0.0:
        raise StageBCompanionRunnerError(f"complete lesion did not zero {field}")
    return {
        "runtime_conductance_field": field,
        "conductance_density_unit": "mS/cm^2",
        "before": [before],
        "after": [after],
    }


def _step_trace(bridge: Any, steps: int) -> tuple[np.ndarray, np.ndarray]:
    voltages = np.empty(steps, dtype=np.dtype("<f8"))
    spikes = np.empty(steps, dtype=np.dtype("|b1"))
    for index in range(steps):
        bridge._run_one_simulation_step()
        voltage = _scalar_array(bridge, "cp_membrane_potential_v")
        firing = np.asarray(to_host(bridge.cp_firing_states), dtype=bool)
        if firing.shape != (1,):
            raise StageBCompanionRunnerError("SNr bridge changed the single-cell spike shape")
        voltages[index] = voltage[0]
        spikes[index] = firing[0]
    return voltages, spikes


def _trace_binding(
    root: Path, archive: Path, time_s: np.ndarray, voltage_mV: np.ndarray, spikes: np.ndarray,
) -> dict[str, Any]:
    try:
        relative = archive.relative_to(root).as_posix()
    except ValueError as exc:
        raise StageBCompanionRunnerError("compact trace output must be inside repository_root") from exc
    try:
        digest = save_compact_trace(archive, time_s, voltage_mV, spikes)
    except (CompactTraceError, OSError, TypeError, ValueError) as exc:
        raise StageBCompanionRunnerError(f"could not write compact trace: {exc}") from exc
    return {
        "path": relative,
        "sha256": digest,
        "sample_count": int(time_s.size),
        "sample_interval_s": DT_MS / 1000.0,
        "sample_semantics": "post-update state at the declared time",
        "time_unit": "s",
        "voltage_unit": "mV",
    }


def _base_output(
    assay: str,
    parameter: Mapping[str, Any],
    parameter_binding: Mapping[str, str],
    protocol_binding: Mapping[str, str],
    causal_binding: Mapping[str, str],
    release: Mapping[str, Any],
    bindings: Mapping[str, RuntimeSNrPacketBinding],
    root: Path,
) -> dict[str, Any]:
    binding = bindings["snr"]
    return {
        "schema": OUTPUT_SCHEMA,
        "process_status": "completed",
        "assay": assay,
        "backend": "numpy",
        "device": "cpu",
        "scientific_verdict": None,
        "adaptive_candidate": _candidate_echo(parameter),
        "contracts": {
            "parameter_document": dict(parameter_binding),
            "protocol_spec": dict(protocol_binding),
            "causal_gate": dict(causal_binding),
        },
        "provenance": {
            "runner": "research/runners/v14_stageB_companion_physiology.py",
            "repository_root": str(root),
            "runtime_binding_manifest_sha256": _digest_bytes(
                runtime_binding_manifest_bytes(bindings)
            ),
            "bindings": [_binding_provenance(binding)],
            "candidate_release": {
                "path": parameter["references"]["release_path"],
                "sha256": parameter["references"]["release_sha256"],
                "candidate_sha256": release["candidate"]["sha256"],
            },
        },
    }


def _publish(destination: Path, result: Mapping[str, Any], archives: Sequence[Path]) -> None:
    created = False
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as handle:
            created = True
            handle.write(_canonical_bytes(result))
    except Exception:
        if created:
            destination.unlink(missing_ok=True)
        for archive in archives:
            archive.unlink(missing_ok=True)
        raise


def run_nap_companion(
    parameter_document_path: str | Path,
    parameter_document_sha256: str,
    protocol_spec_path: str | Path,
    protocol_spec_sha256: str,
    causal_gate_path: str | Path,
    causal_gate_sha256: str,
    output: str | Path,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    """Record one continuous 2 s intact plus 1 s NaP-lesion trace."""

    root = Path(repository_root).expanduser().resolve(strict=True)
    destination = Path(output).expanduser().resolve()
    if destination.exists():
        raise StageBCompanionRunnerError("refusing to replace an existing observation")
    backend, backend_name = get_backend()
    if backend_name != "numpy" or backend is not np:
        raise StageBCompanionRunnerError("companion runner did not acquire NumPy")
    parameter, parameter_binding, _, protocol_binding, causal_binding = _load_inputs(
        root, parameter_document_path, parameter_document_sha256,
        protocol_spec_path, protocol_spec_sha256,
        causal_gate_path, causal_gate_sha256, "nap",
    )
    release = _load_candidate_release(root, parameter["references"], parameter)
    config, bindings, _, bridge = _initialize_bridge(root, parameter["references"])
    archive = destination.with_name(f"{destination.stem}.nap.trace.zip")
    try:
        total_steps = int(round(
            (NAP_BASELINE_S + NAP_POST_LESION_S) * 1000.0 / config.dt_ms
        ))
        baseline_steps = int(round(NAP_BASELINE_S * 1000.0 / config.dt_ms)) - 1
        post_steps = total_steps - baseline_steps
        before_v, before_s = _step_trace(bridge, baseline_steps)
        intervention = _zero_conductance(bridge, "cp_snr_g_nap_max")
        after_v, after_s = _step_trace(bridge, post_steps)
    finally:
        bridge.clear_simulation_state_and_gpu_memory()
    voltage = np.concatenate((before_v, after_v))
    spikes = np.concatenate((before_s, after_s))
    time_s = np.arange(1, voltage.size + 1, dtype=np.dtype("<f8")) * (config.dt_ms / 1000.0)
    trace = _trace_binding(root, archive, time_s, voltage, spikes)
    result = _base_output(
        "nap_same_cell_phased_voltage", parameter, parameter_binding,
        protocol_binding, causal_binding, release, bindings, root,
    )
    result["observation"] = {
        "kind": "same_cell_phased_voltage_spike_trace",
        "compact_trace": trace,
        "phase_schedule": {
            "intact_baseline_s": [0.0, NAP_BASELINE_S],
            "post_lesion_s": [NAP_BASELINE_S, NAP_BASELINE_S + NAP_POST_LESION_S],
        },
        "runtime_intervention": {
            "kind": "complete_intrinsic_current_lesion",
            "operation": "set_conductance_density_to_zero_between_post_update_samples",
            "target": "nap",
            "timestamp_s": NAP_BASELINE_S,
            "lesion_onset_sample_index": baseline_steps,
            "lesion_onset_sample_number": baseline_steps + 1,
            "last_intact_sample_s": NAP_BASELINE_S - config.dt_ms / 1000.0,
            "first_lesion_sample_s": NAP_BASELINE_S,
            **intervention,
        },
    }
    _publish(destination, result, [archive])
    return result


def _current_token(current_pA: float) -> str:
    return "zero" if current_pA == 0.0 else f"minus-{abs(int(current_pA))}pa"


def run_hcn_companion(
    parameter_document_path: str | Path,
    parameter_document_sha256: str,
    protocol_spec_path: str | Path,
    protocol_spec_sha256: str,
    causal_gate_path: str | Path,
    causal_gate_sha256: str,
    output: str | Path,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    """Record fresh intact/lesion HCN trials across the filed current family."""

    root = Path(repository_root).expanduser().resolve(strict=True)
    destination = Path(output).expanduser().resolve()
    if destination.exists():
        raise StageBCompanionRunnerError("refusing to replace an existing observation")
    backend, backend_name = get_backend()
    if backend_name != "numpy" or backend is not np:
        raise StageBCompanionRunnerError("companion runner did not acquire NumPy")
    parameter, parameter_binding, _, protocol_binding, causal_binding = _load_inputs(
        root, parameter_document_path, parameter_document_sha256,
        protocol_spec_path, protocol_spec_sha256,
        causal_gate_path, causal_gate_sha256, "hcn",
    )
    release = _load_candidate_release(root, parameter["references"], parameter)
    trials: list[dict[str, Any]] = []
    archives: list[Path] = []
    manifest_digests: set[str] = set()
    first_bindings = None
    try:
        for condition in ("intact_hcn", "hcn_complete_lesion"):
            for current_pA in HCN_CURRENT_FAMILY_PA:
                config, bindings, binding, bridge = _initialize_bridge(root, parameter["references"])
                if first_bindings is None:
                    first_bindings = bindings
                manifest_digests.add(_digest_bytes(runtime_binding_manifest_bytes(bindings)))
                area_um2 = float(binding.runtime_parameters.geometry.membrane_area_um2)
                if not math.isfinite(area_um2) or area_um2 <= 0.0:
                    bridge.clear_simulation_state_and_gpu_memory()
                    raise StageBCompanionRunnerError("candidate membrane area is invalid")
                interventions = [
                    {"target": "fast_na", **_zero_conductance(bridge, "cp_hh_g_Na_max")},
                    {"target": "nap", **_zero_conductance(bridge, "cp_snr_g_nap_max")},
                ]
                if condition == "hcn_complete_lesion":
                    interventions.append({
                        "target": "hcn", **_zero_conductance(bridge, "cp_snr_g_h_max")
                    })
                else:
                    intact_hcn = _scalar_array(bridge, "cp_snr_g_h_max", nonnegative=True)
                    interventions.append({
                        "target": "hcn",
                        "runtime_conductance_field": "cp_snr_g_h_max",
                        "conductance_density_unit": "mS/cm^2",
                        "before": [float(intact_hcn[0])],
                        "after": [float(intact_hcn[0])],
                    })
                total_steps = int(round(
                    (HCN_BASELINE_S + HCN_STEP_S) * 1000.0 / config.dt_ms
                ))
                baseline_steps = int(round(HCN_BASELINE_S * 1000.0 / config.dt_ms)) - 1
                step_steps = total_steps - baseline_steps
                try:
                    bridge.cp_external_input_current[...] = 0.0
                    base_v, base_s = _step_trace(bridge, baseline_steps)
                    bridge_current_numeric = current_pA * 1.0e8 / area_um2
                    bridge.cp_external_input_current[...] = bridge_current_numeric
                    step_v, step_s = _step_trace(bridge, step_steps)
                finally:
                    bridge.clear_simulation_state_and_gpu_memory()
                voltage = np.concatenate((base_v, step_v))
                spikes = np.concatenate((base_s, step_s))
                time_s = np.arange(1, voltage.size + 1, dtype=np.dtype("<f8")) * (
                    config.dt_ms / 1000.0
                )
                archive = destination.with_name(
                    f"{destination.stem}.hcn-{condition}-{_current_token(current_pA)}.trace.zip"
                )
                trace = _trace_binding(root, archive, time_s, voltage, spikes)
                archives.append(archive)
                trials.append({
                    "condition": condition,
                    "current_pA": current_pA,
                    "membrane_area_um2": area_um2,
                    "current_density_uA_per_cm2": 100.0 * current_pA / area_um2,
                    "bridge_external_current_numeric": bridge_current_numeric,
                    "current_units": {
                        "whole_cell": "pA",
                        "membrane_area": "um^2",
                        "density_equivalent": "uA/cm^2",
                        "bridge_external_current": "cp_external_input_current numeric; HH kernel scales by 1e-6",
                    },
                    "baseline_s": [0.0, HCN_BASELINE_S],
                    "current_step_s": [HCN_BASELINE_S, HCN_BASELINE_S + HCN_STEP_S],
                    "current_step_onset_sample_index": baseline_steps,
                    "current_step_onset_sample_number": baseline_steps + 1,
                    "last_baseline_sample_s": HCN_BASELINE_S - config.dt_ms / 1000.0,
                    "first_current_step_sample_s": HCN_BASELINE_S,
                    "interventions": interventions,
                    "compact_trace": trace,
                    "provenance": {
                        "fresh_bridge": True,
                        "runtime_binding_manifest_sha256": _digest_bytes(
                            runtime_binding_manifest_bytes(bindings)
                        ),
                        "binding": _binding_provenance(binding),
                    },
                })
    except Exception:
        for archive in archives:
            archive.unlink(missing_ok=True)
        raise
    if first_bindings is None or len(manifest_digests) != 1:
        for archive in archives:
            archive.unlink(missing_ok=True)
        raise StageBCompanionRunnerError("fresh HCN trials changed runtime packet identity")
    result = _base_output(
        "hcn_hyperpolarized_current_family", parameter, parameter_binding,
        protocol_binding, causal_binding, release, first_bindings, root,
    )
    result["observation"] = {
        "kind": "independent_fresh_bridge_current_family",
        "current_family_pA": list(HCN_CURRENT_FAMILY_PA),
        "conditions": ["intact_hcn", "hcn_complete_lesion"],
        "trial_count": len(trials),
        "trials": trials,
    }
    _publish(destination, result, archives)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assay", choices=("nap", "hcn"))
    parser.add_argument("--parameter-document-path", required=True)
    parser.add_argument("--parameter-document-sha256", required=True)
    parser.add_argument("--protocol-spec-path", required=True)
    parser.add_argument("--protocol-spec-sha256", required=True)
    parser.add_argument("--causal-gate-path", required=True)
    parser.add_argument("--causal-gate-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repository-root", required=True)
    args = parser.parse_args(argv)
    runner = run_nap_companion if args.assay == "nap" else run_hcn_companion
    try:
        result = runner(
            args.parameter_document_path,
            args.parameter_document_sha256,
            args.protocol_spec_path,
            args.protocol_spec_sha256,
            args.causal_gate_path,
            args.causal_gate_sha256,
            args.output,
            repository_root=args.repository_root,
        )
    except (OSError, StageBCompanionRunnerError, ValueError, TypeError) as exc:
        parser.exit(2, f"Stage B companion runner infrastructure failure: {exc}\n")
    print(_canonical_bytes(result).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
