#!/usr/bin/env python3
"""Authenticate and analyze the preregistered Stage B fast-channel clamps."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from scipy.optimize import curve_fit


ROOT = Path(__file__).resolve().parents[1]
SPEC_SCHEMA = "v14-snr-stageB-fast-channel-clamp-execution-v1"
OBSERVATION_SCHEMA = "v14-snr-stageB-fast-channel-clamp-observation-v1"
OUTPUT_SCHEMA = "v14-snr-stageB-fast-channel-clamp-analysis-v1"


class FastChannelClampAnalysisError(ValueError):
    """Raised when clamp evidence is incomplete or unauthenticated."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256(value: Any, context: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise FastChannelClampAnalysisError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _inside_file(root: Path, value: Any, context: str) -> tuple[str, Path]:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise FastChannelClampAnalysisError(f"{context} path is invalid")
    relative = PurePosixPath(value)
    if relative.is_absolute() or str(relative) != value or any(p in {"", ".", ".."} for p in relative.parts):
        raise FastChannelClampAnalysisError(f"{context} path is not canonical")
    unresolved = root.joinpath(*relative.parts)
    if unresolved.is_symlink():
        raise FastChannelClampAnalysisError(f"{context} must not be a symbolic link")
    path = unresolved.resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise FastChannelClampAnalysisError(f"{context} escapes repository root") from exc
    if not path.is_file():
        raise FastChannelClampAnalysisError(f"{context} must be a regular file")
    return value, path


def _load_json(path: Path, context: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise FastChannelClampAnalysisError(f"{context} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise FastChannelClampAnalysisError(f"{context} must contain an object")
    return value


def _load_bound_json(root: Path, binding: Mapping[str, Any], context: str) -> tuple[dict[str, str], dict[str, Any]]:
    if not isinstance(binding, Mapping) or set(binding) != {"path", "sha256"}:
        raise FastChannelClampAnalysisError(f"{context} binding is invalid")
    relative, path = _inside_file(root, binding["path"], context)
    observed = _file_digest(path)
    if observed != _sha256(binding["sha256"], f"{context} sha256"):
        raise FastChannelClampAnalysisError(f"{context} digest does not match")
    return {"path": relative, "sha256": observed}, _load_json(path, context)


def _load_observation(root: Path, binding: Mapping[str, Any], backend: str) -> tuple[dict[str, str], dict[str, Any]]:
    relative, path = _inside_file(root, binding["path"], f"{backend} observation")
    observed_file_sha = _file_digest(path)
    expected_file_sha = binding.get("sha256")
    if expected_file_sha is not None and observed_file_sha != _sha256(expected_file_sha, f"{backend} file sha256"):
        raise FastChannelClampAnalysisError(f"{backend} observation digest does not match")
    document = _load_json(path, f"{backend} observation")
    body = {key: value for key, value in document.items() if key != "sha256"}
    if document.get("sha256") != _digest(body):
        raise FastChannelClampAnalysisError(f"{backend} observation self digest is invalid")
    sidecar_path = Path(str(path) + ".prov.json")
    sidecar = _load_json(sidecar_path, f"{backend} provenance sidecar")
    expected_argv = [
        str(root / "research/runners/v14_stageB_fast_channel_clamp.py"),
        "--spec", document["execution_spec"]["path"],
        "--spec-sha256", document["execution_spec"]["sha256"],
        "--backend", backend,
        "--out", relative,
    ]
    if (
        sidecar.get("runner") != "research/runners/v14_stageB_fast_channel_clamp.py"
        or sidecar.get("argv") != expected_argv
        or sidecar.get("artifact") != relative
        or sidecar.get("sim_backend") != backend
        or sidecar.get("sim_backend_requested") != backend
        or not sidecar.get("run_id")
    ):
        raise FastChannelClampAnalysisError(f"{backend} provenance sidecar is invalid")
    return {
        "path": relative,
        "sha256": observed_file_sha,
        "self_sha256": document["sha256"],
        "provenance_path": sidecar_path.relative_to(root).as_posix(),
        "provenance_sha256": _file_digest(sidecar_path),
        "run_id": sidecar["run_id"],
    }, document


def _validate_observation(document: Mapping[str, Any], spec: Mapping[str, Any]) -> None:
    assays = document.get("assays")
    execution = document.get("execution")
    if (
        document.get("sample_interval_ms") != spec["numeric_contract"]["clamp_sample_interval_ms"]
        or document.get("analysis_status") != "raw_unanalyzed"
        or not isinstance(assays, Mapping)
        or set(assays) != {
            "fast_na_activation", "fast_na_composite_zero", "fast_na_deactivation",
            "fast_na_inactivation", "fast_na_recovery", "kv3_activation",
            "kv3_deactivation", "kv3_inactivation", "kv3_rise",
        }
        or not isinstance(execution, Mapping)
        or execution.get("segment_launch_count") != 26
        or execution.get("per_time_step_host_loop") is not False
        or execution.get("host_transfer_boundary") != "final_serialization_only"
    ):
        raise FastChannelClampAnalysisError("observation shape or execution boundary is invalid")
    commands = spec["commands"]
    exact = {
        "fast_na_activation": ("command_voltage_mv", commands["fast_na_activation"]["test_mV"]),
        "fast_na_inactivation": ("prepulse_voltage_mv", commands["fast_na_inactivation"]["prepulse_mV"]),
        "fast_na_recovery": ("recovery_duration_ms", commands["fast_na_recovery"]["recovery_duration_ms"]),
        "fast_na_deactivation": ("command_voltage_mv", commands["fast_na_deactivation"]["test_mV"]),
        "kv3_activation": ("command_voltage_mv", commands["kv3_activation"]["test_mV"]),
        "kv3_inactivation": ("prepulse_voltage_mv", commands["kv3_inactivation"]["prepulse_mV"]),
        "kv3_deactivation": ("command_voltage_mv", commands["kv3_deactivation"]["test_mV"]),
    }
    for assay_name, (field, expected) in exact.items():
        observed = np.asarray(assays[assay_name].get(field), dtype=np.float64)
        expected_array = np.asarray(expected, dtype=np.float64)
        if observed.shape != expected_array.shape or not np.allclose(
            observed, expected_array, rtol=0.0, atol=1e-6
        ):
            raise FastChannelClampAnalysisError(f"{assay_name} command ladder changed")
    for assay_name, assay in assays.items():
        elapsed = np.asarray(assay.get("elapsed_ms"), dtype=np.float64)
        if elapsed.ndim != 1 or not elapsed.size or not np.all(np.isfinite(elapsed)) or not np.all(np.diff(elapsed) > 0):
            raise FastChannelClampAnalysisError(f"{assay_name} elapsed-time grid is invalid")
        if not np.isclose(elapsed[0], document["sample_interval_ms"], rtol=0.0, atol=1e-12):
            raise FastChannelClampAnalysisError(f"{assay_name} first sample changed")
        for key, value in assay.items():
            if key in {"command"} or not isinstance(value, list):
                continue
            array = np.asarray(value)
            if np.issubdtype(array.dtype, np.number) and not np.all(np.isfinite(array)):
                raise FastChannelClampAnalysisError(f"{assay_name}.{key} contains non-finite values")


def _boltzmann_activation(voltage: np.ndarray, vhalf: float, slope: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(voltage - vhalf) / slope))


def _boltzmann_inactivation(voltage: np.ndarray, vhalf: float, slope: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp((voltage - vhalf) / slope))


def _fit_boltzmann(voltage: Any, values: Any, *, activation: bool) -> tuple[float, float]:
    function = _boltzmann_activation if activation else _boltzmann_inactivation
    fitted, _ = curve_fit(
        function, np.asarray(voltage, dtype=np.float64), np.asarray(values, dtype=np.float64),
        p0=(-20.0, 8.0), bounds=([-150.0, 0.01], [100.0, 100.0]), maxfev=50_000,
    )
    return float(fitted[0]), float(fitted[1])


def _crossing(time: np.ndarray, values: np.ndarray, level: float, *, rising: bool, start: int, stop: int) -> float:
    for index in range(max(1, start), stop):
        left, right = values[index - 1], values[index]
        crossed = left <= level <= right if rising else left >= level >= right
        if crossed and right != left:
            fraction = (level - left) / (right - left)
            return float(time[index - 1] + fraction * (time[index] - time[index - 1]))
    raise FastChannelClampAnalysisError(f"trace does not cross {level:g}")


def _rise_and_decay(assay: Mapping[str, Any], low: float, high: float) -> tuple[float, float | None]:
    time = np.asarray(assay["elapsed_ms"], dtype=np.float64)
    trace = np.asarray(assay["trace_normalized_absolute_current"], dtype=np.float64)
    if trace.shape != (1, time.size):
        raise FastChannelClampAnalysisError("rise/decay trace shape is invalid")
    values = trace[0]
    peak = int(np.argmax(values))
    normalized = values / values[peak]
    low_time = _crossing(time, normalized, low, rising=True, start=0, stop=peak + 1)
    high_time = _crossing(time, normalized, high, rising=True, start=0, stop=peak + 1)
    rise = high_time - low_time
    if low != 0.1 or high != 0.9:
        return rise, None
    decay_high = _crossing(time, normalized, high, rising=False, start=peak + 1, stop=time.size)
    decay_low = _crossing(time, normalized, low, rising=False, start=peak + 1, stop=time.size)
    return rise, decay_low - decay_high


def _fit_recovery(durations: Any, values: Any) -> tuple[float, float, float]:
    def model(time: np.ndarray, fast_fraction: float, fast_tau: float, slow_tau: float) -> np.ndarray:
        return 1.0 - fast_fraction * np.exp(-time / fast_tau) - (1.0 - fast_fraction) * np.exp(-time / slow_tau)

    fitted, _ = curve_fit(
        model, np.asarray(durations, dtype=np.float64), np.asarray(values, dtype=np.float64),
        p0=(0.526, 0.59, 35.1), bounds=([0.0, 0.001, 0.01], [1.0, 100.0, 1000.0]),
        maxfev=100_000,
    )
    fraction, first, second = map(float, fitted)
    return (1.0 - fraction, second, first) if first > second else (fraction, first, second)


def _fit_tail(assay: Mapping[str, Any], channel: str, command_mv: float) -> float:
    voltages = list(map(float, assay["command_voltage_mv"]))
    try:
        row = voltages.index(float(command_mv))
    except ValueError as exc:
        raise FastChannelClampAnalysisError(f"deactivation command {command_mv:g} mV is absent") from exc
    time = np.asarray(assay["elapsed_ms"], dtype=np.float64)[1:]
    values = np.asarray(assay[channel], dtype=np.float64)[row, 1:]
    if values.shape != time.shape:
        raise FastChannelClampAnalysisError("deactivation trace shape is invalid")

    def model(t: np.ndarray, asymptote: float, amplitude: float, tau: float) -> np.ndarray:
        return asymptote + amplitude * np.exp(-t / tau)

    scale = max(1.0, float(np.max(np.abs(values))))
    fitted, _ = curve_fit(
        model, time, values, p0=(float(values[-1]), float(values[0] - values[-1]), 1.0),
        bounds=([-10.0 * scale, -10.0 * scale, 0.00001], [10.0 * scale, 10.0 * scale, 1000.0]),
        maxfev=100_000,
    )
    return float(fitted[2])


def _metrics(document: Mapping[str, Any]) -> dict[str, float]:
    assays = document["assays"]
    na_av, na_as = _fit_boltzmann(assays["fast_na_activation"]["command_voltage_mv"], assays["fast_na_activation"]["ladder_normalized_peak_conductance"], activation=True)
    na_iv, na_is = _fit_boltzmann(assays["fast_na_inactivation"]["prepulse_voltage_mv"], assays["fast_na_inactivation"]["ladder_normalized_peak_test_current"], activation=False)
    kv_av, kv_as = _fit_boltzmann(assays["kv3_activation"]["command_voltage_mv"], assays["kv3_activation"]["ladder_normalized_peak_conductance"], activation=True)
    kv_iv, kv_is = _fit_boltzmann(assays["kv3_inactivation"]["prepulse_voltage_mv"], assays["kv3_inactivation"]["ladder_normalized_peak_test_current"], activation=False)
    na_rise, na_decay = _rise_and_decay(assays["fast_na_composite_zero"], 0.1, 0.9)
    kv_rise, _ = _rise_and_decay(assays["kv3_rise"], 0.2, 0.8)
    fraction, fast_tau, slow_tau = _fit_recovery(assays["fast_na_recovery"]["recovery_duration_ms"], assays["fast_na_recovery"]["ladder_normalized_peak_test_current"])
    return {
        "fast_na.activation_vhalf_mV": na_av, "fast_na.activation_slope_mV": na_as,
        "fast_na.inactivation_vhalf_mV": na_iv, "fast_na.inactivation_slope_mV": na_is,
        "fast_na.activation_10_90_at_0_mV_ms": na_rise,
        "fast_na.inactivation_10_90_at_0_mV_ms": float(na_decay),
        "fast_na.recovery_fast_tau_ms": fast_tau, "fast_na.recovery_slow_tau_ms": slow_tau,
        "fast_na.recovery_fast_fraction": fraction,
        "fast_na.deactivation_at_minus_40_mV_ms": _fit_tail(assays["fast_na_deactivation"], "normalized_na_current", -40.0),
        "kv3_like.activation_vhalf_mV": kv_av, "kv3_like.activation_slope_mV": kv_as,
        "kv3_like.inactivation_vhalf_mV": kv_iv, "kv3_like.inactivation_slope_mV": kv_is,
        "kv3_like.rise_20_80_at_plus_40_mV_ms": kv_rise,
        "kv3_like.deactivation_at_minus_60_mV_ms": _fit_tail(assays["kv3_deactivation"], "normalized_kv3_current", -60.0),
        "kv3_like.deactivation_at_minus_50_mV_ms": _fit_tail(assays["kv3_deactivation"], "normalized_kv3_current", -50.0),
        "kv3_like.deactivation_at_minus_40_mV_ms": _fit_tail(assays["kv3_deactivation"], "normalized_kv3_current", -40.0),
    }


def _targets(protocol: Mapping[str, Any]) -> dict[str, Mapping[str, float]]:
    transfers = protocol["source_transfers"]
    result = {}
    for family, prefix in (("fast_sodium", "fast_na"), ("kv3_like", "kv3_like")):
        constraints = transfers[family]["constraints"]
        for name, target in constraints.items():
            if isinstance(target, Mapping) and set(target) == {"mean", "sem"} and f"{prefix}.{name}" in _METRIC_NAMES:
                result[f"{prefix}.{name}"] = target
    for target in transfers["kv3_like"]["constraints"]["deactivation_tau_ms"]:
        voltage = int(abs(float(target["command_mV"])))
        result[f"kv3_like.deactivation_at_minus_{voltage}_mV_ms"] = target
    return result


_METRIC_NAMES = frozenset({
    "fast_na.activation_vhalf_mV", "fast_na.activation_slope_mV", "fast_na.inactivation_vhalf_mV",
    "fast_na.inactivation_slope_mV", "fast_na.activation_10_90_at_0_mV_ms",
    "fast_na.inactivation_10_90_at_0_mV_ms", "fast_na.recovery_fast_tau_ms",
    "fast_na.recovery_slow_tau_ms", "fast_na.recovery_fast_fraction",
    "fast_na.deactivation_at_minus_40_mV_ms", "kv3_like.activation_vhalf_mV",
    "kv3_like.activation_slope_mV", "kv3_like.inactivation_vhalf_mV",
    "kv3_like.inactivation_slope_mV", "kv3_like.rise_20_80_at_plus_40_mV_ms",
    "kv3_like.deactivation_at_minus_60_mV_ms", "kv3_like.deactivation_at_minus_50_mV_ms",
    "kv3_like.deactivation_at_minus_40_mV_ms",
})


def analyze(spec_path: str | Path, spec_sha256: str, *, repository_root: str | Path = ROOT) -> dict[str, Any]:
    root = Path(repository_root).expanduser().resolve(strict=True)
    supplied = Path(spec_path).expanduser()
    path = (supplied if supplied.is_absolute() else root / supplied).resolve()
    relative = path.relative_to(root).as_posix()
    _, path = _inside_file(root, relative, "execution spec")
    observed_spec_sha = _file_digest(path)
    if observed_spec_sha != _sha256(spec_sha256, "execution spec sha256"):
        raise FastChannelClampAnalysisError("execution spec digest does not match")
    spec = _load_json(path, "execution spec")
    spec_binding = {"path": relative, "sha256": observed_spec_sha}
    if (
        spec.get("schema") != SPEC_SCHEMA
        or spec.get("status") != "preregistered_not_executed"
        or spec.get("scientific_verdict") is not None
        or spec.get("candidate_calibration_allowed") is not False
        or spec.get("conductance_scale_fitting_allowed") is not False
        or spec.get("whole_cell_promotion_allowed") is not False
        or spec.get("analysis_output") != "research/findings/raw/v14_snr_stageB_fast_channel_clamp_analysis_v1.json"
    ):
        raise FastChannelClampAnalysisError("execution spec identity is invalid")
    protocol_binding, protocol = _load_bound_json(root, spec["protocol"], "protocol")
    implementation = spec.get("implementation")
    if not isinstance(implementation, Mapping) or set(implementation) != {
        "parameter_module", "production_kernel_module", "runner"
    }:
        raise FastChannelClampAnalysisError("implementation binding set is invalid")
    for name, binding in implementation.items():
        if not isinstance(binding, Mapping) or set(binding) != {"path", "sha256"}:
            raise FastChannelClampAnalysisError(f"{name} binding is invalid")
        _, implementation_path = _inside_file(root, binding["path"], name)
        if _file_digest(implementation_path) != _sha256(binding["sha256"], f"{name} sha256"):
            raise FastChannelClampAnalysisError(f"{name} digest does not match")
    matrix = spec.get("execution_matrix")
    if (
        not isinstance(matrix, list) or len(matrix) != 2
        or {row.get("backend") for row in matrix if isinstance(row, Mapping)} != {"numpy", "cupy"}
    ):
        raise FastChannelClampAnalysisError("execution matrix is invalid")
    documents = {}
    bindings = {}
    for matrix_row in matrix:
        backend = matrix_row["backend"]
        bindings[backend], documents[backend] = _load_observation(
            root, {"path": matrix_row["output"]}, backend
        )
        document = documents[backend]
        if (
            document.get("schema") != OBSERVATION_SCHEMA or document.get("backend") != backend
            or document.get("device") != matrix_row["device"] or document.get("dtype") != "float32"
            or document.get("execution_spec") != spec_binding
            or document.get("protocol") != protocol_binding or document.get("implementation") != spec["implementation"]
            or document.get("analysis_status") != "raw_unanalyzed" or document.get("scientific_verdict") is not None
            or document.get("candidate_calibration_allowed") is not False
        ):
            raise FastChannelClampAnalysisError(f"{backend} observation identity or authority is invalid")
        _validate_observation(document, spec)
    reference = _metrics(documents["numpy"])
    accelerated = _metrics(documents["cupy"])
    targets = _targets(protocol)
    if set(reference) != set(targets) or set(accelerated) != set(targets):
        raise FastChannelClampAnalysisError("metric and source-target sets differ")
    gates = []
    for name in sorted(targets):
        target = targets[name]
        lower = float(target["mean"] - 2.0 * target["sem"])
        upper = float(target["mean"] + 2.0 * target["sem"])
        value = reference[name]
        gates.append({"metric": name, "value": value, "source_mean": float(target["mean"]), "source_sem": float(target["sem"]), "lower_2sem": lower, "upper_2sem": upper, "passed": lower <= value <= upper})
    parity = {
        "status": "NOT_ESTABLISHED_NO_PREREGISTERED_TOLERANCE",
        "promoting": False,
        "metric_absolute_differences": {name: abs(reference[name] - accelerated[name]) for name in sorted(reference)},
        "reason": "the sealed execution contract requires parity but declares no numerical tolerance",
    }
    source_passed = all(row["passed"] for row in gates)
    verdict = "INCONCLUSIVE_INVALID_EVIDENCE" if source_passed else "STAGE1_STRUCTURAL_NO_GO"
    output = {
        "schema": OUTPUT_SCHEMA,
        "execution_spec": {"path": relative, "sha256": spec_sha256},
        "inputs": bindings,
        "analysis_contract": spec["analysis_contract"],
        "numpy_reference_metrics": reference,
        "cupy_descriptive_metrics": accelerated,
        "source_transfer_gates": gates,
        "source_transfer_status": "GO" if source_passed else "STRUCTURAL_NO_GO",
        "backend_parity": parity,
        "scientific_verdict": verdict,
        "candidate_calibration_allowed": False,
        "whole_cell_promotion_allowed": False,
        "failed_metric_count": sum(not row["passed"] for row in gates),
    }
    output["sha256"] = _digest(output)
    return output


def _write_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FastChannelClampAnalysisError(f"refusing to overwrite existing output: {path}")
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--spec-sha256", required=True)
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args(argv)
    result = analyze(arguments.spec, arguments.spec_sha256)
    expected = result["execution_spec"]["path"]
    spec = _load_json(ROOT / expected, "execution spec")
    output = Path(arguments.out)
    output_relative = (output if output.is_absolute() else ROOT / output).resolve().relative_to(ROOT).as_posix()
    if output_relative != spec["analysis_output"]:
        raise FastChannelClampAnalysisError("output path does not match sealed analysis output")
    _write_new(ROOT / output_relative, result)
    print(json.dumps({"output": output_relative, "sha256": result["sha256"], "verdict": result["scientific_verdict"], "failed_metric_count": result["failed_metric_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
