"""Execute source-command Stage B fast-channel voltage-clamp assays.

The runner is deliberately verdict-free.  It authenticates an execution
document, advances complete voltage-by-time grids with exact fixed-voltage
Rush-Larsen updates, and writes raw curves for a separate analyzer.  There is
no per-time-step Python loop and no host transfer before final serialization.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from sim.snr_structural_successor import (
    DEFAULT_FAST_CHANNEL_PARAMETERS,
    EVIDENCE_CLASSES,
    PROTOCOL_PATH,
    PROTOCOL_SHA256,
    UNITS,
)
from tools import lab


ROOT = Path(__file__).resolve().parents[2]
EXECUTION_SPEC_PATH = (
    ROOT / "research/specs/v14_snr_stageB_fast_channel_clamp_execution_v1.json"
)
EXECUTION_SCHEMA = "v14-snr-stageB-fast-channel-clamp-execution-v1"
OUTPUT_SCHEMA = "v14-snr-stageB-fast-channel-clamp-observation-v1"
ASSAY_NAMES = (
    "fast_na_activation",
    "fast_na_inactivation",
    "fast_na_composite_zero",
    "fast_na_recovery",
    "fast_na_deactivation",
    "kv3_activation",
    "kv3_inactivation",
    "kv3_rise",
    "kv3_deactivation",
)
RECOVERY_LADDER_V2_MS = (
    0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0
)


class FastChannelClampError(RuntimeError):
    """Raised when a clamp execution cannot preserve its declared contract."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _semantic_digest(document: Mapping[str, Any]) -> str:
    body = {key: value for key, value in document.items() if key != "sha256"}
    return hashlib.sha256(_canonical_json(body)).hexdigest()


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FastChannelClampError(message)


def _numbers(
    value: Any,
    name: str,
    *,
    exact: Sequence[float] | None = None,
    nonnegative: bool = False,
    strictly_increasing: bool = False,
) -> tuple[float, ...]:
    _require(isinstance(value, list) and value, f"{name} must be a nonempty list")
    result: list[float] = []
    for item in value:
        _require(
            not isinstance(item, bool) and isinstance(item, (int, float))
            and math.isfinite(float(item)),
            f"{name} must contain only finite numbers",
        )
        result.append(float(item))
    if nonnegative:
        _require(all(item >= 0.0 for item in result), f"{name} must be nonnegative")
    if strictly_increasing:
        _require(
            all(right > left for left, right in zip(result, result[1:])),
            f"{name} must be strictly increasing",
        )
    if exact is not None:
        _require(tuple(result) == tuple(float(item) for item in exact), f"{name} changed")
    return tuple(result)


def _scalar(mapping: Mapping[str, Any], key: str, expected: float) -> None:
    value = mapping.get(key)
    _require(
        not isinstance(value, bool) and isinstance(value, (int, float))
        and math.isfinite(float(value)) and float(value) == float(expected),
        f"{key} must equal the source-command value {expected}",
    )


def _validate_commands(commands: Any) -> None:
    expected_names = {
        "fast_na_activation", "fast_na_inactivation", "fast_na_recovery",
        "fast_na_deactivation", "kv3_activation", "kv3_inactivation",
        "kv3_deactivation",
    }
    _require(isinstance(commands, Mapping), "commands must be an object")
    _require(set(commands) == expected_names, "command set is incomplete or unexpected")
    for name in expected_names:
        _require(isinstance(commands[name], Mapping), f"{name} must be an object")

    a = commands["fast_na_activation"]
    _scalar(a, "hold_mV", -100.0)
    _scalar(a, "test_duration_ms", 20.0)
    _numbers(a.get("test_mV"), "fast_na_activation.test_mV", exact=range(-80, 31, 5))

    a = commands["fast_na_inactivation"]
    _scalar(a, "hold_mV", -100.0)
    _scalar(a, "prepulse_duration_ms", 50.0)
    _scalar(a, "test_mV", 0.0)
    _scalar(a, "test_duration_ms", 20.0)
    _numbers(a.get("prepulse_mV"), "fast_na_inactivation.prepulse_mV",
             exact=range(-120, -19, 10))

    a = commands["fast_na_recovery"]
    for key, expected in (
        ("hold_mV", -90.0), ("recovery_prepulse_mV", -120.0),
        ("recovery_prepulse_duration_ms", 50.0), ("inactivation_mV", 0.0),
        ("inactivation_duration_ms", 300.0), ("recovery_mV", -120.0),
        ("test_mV", 0.0), ("test_duration_ms", 20.0),
    ):
        _scalar(a, key, expected)
    _numbers(a.get("recovery_duration_ms"), "fast_na_recovery.recovery_duration_ms",
             exact=RECOVERY_LADDER_V2_MS)
    _require(
        a.get("duration_evidence") == "project_operational_sampling_not_source_reported",
        "recovery duration evidence label changed",
    )

    a = commands["fast_na_deactivation"]
    for key, expected in (
        ("hold_mV", -90.0), ("prepulse_mV", -120.0),
        ("prepulse_duration_ms", 50.0), ("activation_mV", 0.0),
        ("activation_duration_ms", 0.2), ("test_duration_ms", 50.0),
    ):
        _scalar(a, key, expected)
    _numbers(a.get("test_mV"), "fast_na_deactivation.test_mV",
             exact=range(-100, -19, 10))

    a = commands["kv3_activation"]
    _scalar(a, "hold_mV", -100.0)
    _scalar(a, "test_duration_ms", 100.0)
    _numbers(a.get("test_mV"), "kv3_activation.test_mV",
             exact=range(-80, 51, 10))

    a = commands["kv3_inactivation"]
    for key, expected in (
        ("hold_mV", -90.0), ("prepulse_duration_ms", 10_000.0),
        ("test_mV", 50.0), ("test_duration_ms", 100.0),
    ):
        _scalar(a, key, expected)
    _numbers(a.get("prepulse_mV"), "kv3_inactivation.prepulse_mV",
             exact=range(-110, 1, 10))

    a = commands["kv3_deactivation"]
    for key, expected in (
        ("hold_mV", -90.0), ("activation_mV", 20.0),
        ("activation_duration_ms", 100.0),
    ):
        _scalar(a, key, expected)
    _scalar(a, "test_duration_ms", 50.0)
    _numbers(a.get("test_mV"), "kv3_deactivation.test_mV",
             exact=(-30, -40, -50, -60, -70))


def load_execution_spec(
    path: Path | str = EXECUTION_SPEC_PATH,
    *,
    expected_file_sha256: str,
    repository_root: Path | str = ROOT,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Authenticate and validate one immutable execution declaration."""

    root = Path(repository_root).resolve()
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise FastChannelClampError("execution spec must be inside the repository") from exc
    _require(candidate.is_file() and not candidate.is_symlink(), "execution spec is not a regular file")
    _require(
        isinstance(expected_file_sha256, str) and len(expected_file_sha256) == 64,
        "expected execution-spec SHA-256 is required",
    )
    file_sha = _file_digest(candidate)
    _require(file_sha == expected_file_sha256, "execution spec file SHA-256 mismatch")
    try:
        document = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FastChannelClampError(f"cannot read execution spec: {exc}") from exc
    _require(isinstance(document, dict), "execution spec must be a JSON object")
    _require(document.get("schema") == EXECUTION_SCHEMA, "execution spec schema mismatch")
    _require(
        document.get("status") == "preregistered_not_executed",
        "execution spec is not preregistered",
    )
    _require(document.get("scientific_verdict") is None, "execution spec must be verdict-free")
    _require(document.get("candidate_calibration_allowed") is False, "candidate calibration is not forbidden")
    _require(document.get("conductance_scale_fitting_allowed") is False, "conductance fitting is not forbidden")
    _require(document.get("whole_cell_promotion_allowed") is False, "whole-cell promotion is not forbidden")
    protocol = document.get("protocol")
    _require(
        protocol == {"path": PROTOCOL_PATH, "sha256": PROTOCOL_SHA256},
        "structural protocol binding mismatch",
    )
    protocol_file = (root / PROTOCOL_PATH).resolve()
    _require(
        protocol_file.is_file() and _file_digest(protocol_file) == PROTOCOL_SHA256,
        "structural protocol file authentication failed",
    )
    numeric = document.get("numeric_contract")
    _require(isinstance(numeric, Mapping), "numeric contract is missing")
    _require(numeric.get("state_dtype") == "float32", "numeric dtype must be float32")
    interval = numeric.get("clamp_sample_interval_ms")
    _require(
        not isinstance(interval, bool) and isinstance(interval, (int, float))
        and math.isfinite(float(interval)) and float(interval) > 0.0,
        "sample interval must be finite and positive",
    )
    _require(float(numeric.get("conductance_scale", math.nan)) == 1.0, "conductance scale changed")
    _require(float(numeric.get("sodium_reversal_mV", math.nan)) == 50.0, "sodium reversal changed")
    _require(float(numeric.get("potassium_reversal_mV", math.nan)) == -90.0, "potassium reversal changed")
    _validate_commands(document.get("commands"))

    implementation = document.get("implementation")
    _require(isinstance(implementation, Mapping), "implementation bindings are missing")
    _require(
        set(implementation) == {"parameter_module", "production_kernel_module", "runner"},
        "implementation binding set changed",
    )
    for label, reference in implementation.items():
        _require(isinstance(reference, Mapping), f"{label} binding must be an object")
        relative_path = reference.get("path")
        digest = reference.get("sha256")
        _require(
            isinstance(relative_path, str) and relative_path
            and isinstance(digest, str) and len(digest) == 64
            and digest == digest.lower(),
            f"{label} binding is incomplete",
        )
        bound_path = (root / relative_path).resolve()
        try:
            bound_path.relative_to(root)
        except ValueError as exc:
            raise FastChannelClampError(f"{label} path escapes repository") from exc
        _require(bound_path.is_file() and not bound_path.is_symlink(), f"{label} file is unavailable")
        _require(_file_digest(bound_path) == digest, f"{label} SHA-256 mismatch")

    matrix = document.get("execution_matrix")
    _require(isinstance(matrix, list) and len(matrix) == 2, "execution matrix is invalid")
    matrix_by_backend = {
        row.get("backend"): row for row in matrix if isinstance(row, Mapping)
    }
    _require(set(matrix_by_backend) == {"numpy", "cupy"}, "execution matrix backend set changed")
    _require(matrix_by_backend["numpy"].get("device") == "cpu", "NumPy device changed")
    _require(matrix_by_backend["cupy"].get("device") == "cuda:0", "CuPy device changed")
    relative = candidate.relative_to(root).as_posix()
    return document, {"path": relative, "sha256": file_sha}


class _ClampRuntime:
    def __init__(self, backend: str):
        from sim.backend import get_backend
        from sim.kernels import fused_snr_fast_channel_clamp_update

        self.xp, actual = get_backend()
        _require(actual == backend, "kernel backend changed after backend assertion")
        self.backend = backend
        self.kernel = fused_snr_fast_channel_clamp_update
        self.parameters = DEFAULT_FAST_CHANNEL_PARAMETERS
        self.segment_launches = 0

    def array(self, value: Any):
        return self.xp.asarray(value, dtype=self.xp.float32)

    def zeros(self, count: int):
        return tuple(self.xp.zeros(count, dtype=self.xp.float32) for _ in range(5))

    def advance(self, voltage: Any, states: tuple[Any, ...], dt: Any):
        p = self.parameters
        self.segment_launches += 1
        return self.kernel(
            self.array(voltage), *states, self.array(dt),
            self.xp.float32(p.na_activation_half_mv),
            self.xp.float32(p.na_activation_slope_mv),
            self.xp.float32(p.na_inactivation_half_mv),
            self.xp.float32(p.na_inactivation_slope_mv),
            self.xp.float32(p.na_activation_gate_tau_at_zero_ms),
            self.xp.float32(p.na_deactivation_gate_tau_at_minus_40_ms),
            self.xp.float32(p.na_recovery_fast_tau_ms),
            self.xp.float32(p.na_recovery_slow_tau_ms),
            self.xp.float32(p.na_inactivation_gate_tau_at_zero_ms),
            self.xp.float32(p.na_recovery_fast_fraction),
            self.xp.float32(p.kv3_activation_half_mv),
            self.xp.float32(p.kv3_activation_slope_mv),
            self.xp.float32(p.kv3_inactivation_half_mv),
            self.xp.float32(p.kv3_inactivation_slope_mv),
            self.xp.float32(p.kv3_activation_gate_tau_at_plus_40_ms),
            *(self.xp.float32(value) for value in p.kv3_deactivation_gate_taus_ms),
            self.xp.float32(p.kv3_inactivation_tau_prior_ms),
            self.xp.float32(p.na_reversal_mv),
            self.xp.float32(p.potassium_reversal_mv),
        )

    def equilibrate(self, voltage: Any):
        voltage = self.array(voltage)
        return self.advance(voltage, self.zeros(int(voltage.size)), self.xp.full_like(voltage, 1.0e6))[:5]

    def trace(self, voltage: Any, states: tuple[Any, ...], times: Sequence[float]):
        voltage = self.array(voltage)
        times_array = self.array(times)
        count = int(voltage.size)
        width = int(times_array.size)
        command = self.xp.repeat(voltage, width)
        elapsed = self.xp.tile(times_array, count)
        expanded = tuple(self.xp.repeat(state, width) for state in states)
        result = self.advance(command, expanded, elapsed)
        return tuple(value.reshape(count, width) for value in result)


def _safe_normalize(runtime: _ClampRuntime, values: Any, *, axis: int | None = None):
    xp = runtime.xp
    scale = xp.max(xp.abs(values), axis=axis, keepdims=axis is not None)
    return xp.where(scale > xp.float32(0.0), values / scale, xp.zeros_like(values))


def _curve_payload(
    voltage: Any, times: Sequence[float], result: tuple[Any, ...], channel: str,
) -> dict[str, Any]:
    _require(channel in {"na", "kv3"}, "curve channel is invalid")
    current = result[5] if channel == "na" else result[6]
    return {
        "command_voltage_mv": voltage,
        "elapsed_ms": list(times),
        f"normalized_{channel}_current": current,
    }


def _filed_times(duration_ms: float, sample_interval_ms: float) -> list[float]:
    count = round(float(duration_ms) / float(sample_interval_ms))
    _require(
        count >= 1
        and math.isclose(count * sample_interval_ms, duration_ms, rel_tol=0.0, abs_tol=1e-9),
        "command duration is not an integral number of samples",
    )
    # The execution contract defines the first sample as one interval after
    # the transition. Every elapsed time is evaluated independently from the
    # same segment-entry state by one vectorized exact update.
    return [sample_interval_ms * index for index in range(1, count + 1)]


def _normalize_commands(spec: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    commands = spec["commands"]
    interval = float(spec["numeric_contract"]["clamp_sample_interval_ms"])
    na_activation = commands["fast_na_activation"]
    kv3_activation = commands["kv3_activation"]
    return {
        "fast_na_activation": {
            "hold_mv": na_activation["hold_mV"],
            "test_voltages_mv": na_activation["test_mV"],
            "sample_times_ms": _filed_times(na_activation["test_duration_ms"], interval),
        },
        "fast_na_inactivation": {
            "hold_mv": commands["fast_na_inactivation"]["hold_mV"],
            "prepulse_voltages_mv": commands["fast_na_inactivation"]["prepulse_mV"],
            "prepulse_duration_ms": commands["fast_na_inactivation"]["prepulse_duration_ms"],
            "test_mv": commands["fast_na_inactivation"]["test_mV"],
            "test_sample_times_ms": _filed_times(
                commands["fast_na_inactivation"]["test_duration_ms"], interval
            ),
        },
        "fast_na_composite_zero": {
            "hold_mv": na_activation["hold_mV"],
            "test_mv": 0.0,
            "sample_times_ms": _filed_times(na_activation["test_duration_ms"], interval),
        },
        "fast_na_recovery": {
            "hold_mv": commands["fast_na_recovery"]["hold_mV"],
            "conditioning_mv": commands["fast_na_recovery"]["recovery_prepulse_mV"],
            "conditioning_duration_ms": commands["fast_na_recovery"]["recovery_prepulse_duration_ms"],
            "inactivation_mv": commands["fast_na_recovery"]["inactivation_mV"],
            "inactivation_duration_ms": commands["fast_na_recovery"]["inactivation_duration_ms"],
            "recovery_mv": commands["fast_na_recovery"]["recovery_mV"],
            "recovery_durations_ms": commands["fast_na_recovery"]["recovery_duration_ms"],
            "test_mv": commands["fast_na_recovery"]["test_mV"],
            "test_sample_times_ms": _filed_times(
                commands["fast_na_recovery"]["test_duration_ms"], interval
            ),
        },
        "fast_na_deactivation": {
            "hold_mv": commands["fast_na_deactivation"]["hold_mV"],
            "conditioning_mv": commands["fast_na_deactivation"]["prepulse_mV"],
            "conditioning_duration_ms": commands["fast_na_deactivation"]["prepulse_duration_ms"],
            "activation_mv": commands["fast_na_deactivation"]["activation_mV"],
            "activation_duration_ms": commands["fast_na_deactivation"]["activation_duration_ms"],
            "tail_voltages_mv": commands["fast_na_deactivation"]["test_mV"],
            "sample_times_ms": _filed_times(
                commands["fast_na_deactivation"]["test_duration_ms"], interval
            ),
        },
        "kv3_activation": {
            "hold_mv": kv3_activation["hold_mV"],
            "test_voltages_mv": kv3_activation["test_mV"],
            "sample_times_ms": _filed_times(kv3_activation["test_duration_ms"], interval),
        },
        "kv3_inactivation": {
            "hold_mv": commands["kv3_inactivation"]["hold_mV"],
            "prepulse_voltages_mv": commands["kv3_inactivation"]["prepulse_mV"],
            "prepulse_duration_ms": commands["kv3_inactivation"]["prepulse_duration_ms"],
            "test_mv": commands["kv3_inactivation"]["test_mV"],
            "test_sample_times_ms": _filed_times(
                commands["kv3_inactivation"]["test_duration_ms"], interval
            ),
        },
        "kv3_rise": {
            "hold_mv": kv3_activation["hold_mV"],
            "test_mv": 40.0,
            "sample_times_ms": _filed_times(kv3_activation["test_duration_ms"], interval),
        },
        "kv3_deactivation": {
            "hold_mv": commands["kv3_deactivation"]["hold_mV"],
            "activation_mv": commands["kv3_deactivation"]["activation_mV"],
            "activation_duration_ms": commands["kv3_deactivation"]["activation_duration_ms"],
            "tail_voltages_mv": commands["kv3_deactivation"]["test_mV"],
            "sample_times_ms": _filed_times(
                commands["kv3_deactivation"]["test_duration_ms"], interval
            ),
        },
    }


def _run_assays(runtime: _ClampRuntime, assays: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    xp = runtime.xp
    p = runtime.parameters
    output: dict[str, Any] = {}

    a = assays["fast_na_activation"]
    voltages = runtime.array(a["test_voltages_mv"])
    states = runtime.equilibrate(xp.full_like(voltages, a["hold_mv"]))
    result = runtime.trace(voltages, states, a["sample_times_ms"])
    conductance = result[5] / (voltages[:, None] - xp.float32(p.na_reversal_mv))
    peak = xp.max(xp.abs(conductance), axis=1)
    output["fast_na_activation"] = {
        **_curve_payload(voltages, a["sample_times_ms"], result, "na"),
        "raw_normalized_conductance": conductance,
        "peak_normalized_conductance": peak,
        "ladder_normalized_peak_conductance": _safe_normalize(runtime, peak),
    }

    a = assays["fast_na_inactivation"]
    prepulse = runtime.array(a["prepulse_voltages_mv"])
    states = runtime.equilibrate(xp.full_like(prepulse, a["hold_mv"]))
    states = runtime.advance(prepulse, states, xp.full_like(prepulse, a["prepulse_duration_ms"]))[:5]
    test_voltage = xp.full_like(prepulse, a["test_mv"])
    result = runtime.trace(test_voltage, states, a["test_sample_times_ms"])
    peak = xp.max(xp.abs(result[5]), axis=1)
    output["fast_na_inactivation"] = {
        "prepulse_voltage_mv": prepulse,
        **_curve_payload(test_voltage, a["test_sample_times_ms"], result, "na"),
        "peak_absolute_test_current": peak,
        "ladder_normalized_peak_test_current": _safe_normalize(runtime, peak),
    }

    a = assays["fast_na_composite_zero"]
    voltage = runtime.array([a["test_mv"]])
    states = runtime.equilibrate(runtime.array([a["hold_mv"]]))
    result = runtime.trace(voltage, states, a["sample_times_ms"])
    output["fast_na_composite_zero"] = {
        **_curve_payload(voltage, a["sample_times_ms"], result, "na"),
        "trace_normalized_absolute_current": _safe_normalize(runtime, xp.abs(result[5])),
    }

    a = assays["fast_na_recovery"]
    durations = runtime.array(a["recovery_durations_ms"])
    states = runtime.equilibrate(xp.full_like(durations, a["hold_mv"]))
    states = runtime.advance(
        xp.full_like(durations, a["conditioning_mv"]), states,
        xp.full_like(durations, a["conditioning_duration_ms"]),
    )[:5]
    states = runtime.advance(
        xp.full_like(durations, a["inactivation_mv"]), states,
        xp.full_like(durations, a["inactivation_duration_ms"]),
    )[:5]
    states = runtime.advance(xp.full_like(durations, a["recovery_mv"]), states, durations)[:5]
    test_voltage = xp.full_like(durations, a["test_mv"])
    result = runtime.trace(test_voltage, states, a["test_sample_times_ms"])
    peak = xp.max(xp.abs(result[5]), axis=1)
    output["fast_na_recovery"] = {
        "recovery_duration_ms": durations,
        **_curve_payload(test_voltage, a["test_sample_times_ms"], result, "na"),
        "peak_absolute_test_current": peak,
        "ladder_normalized_peak_test_current": _safe_normalize(runtime, peak),
    }

    a = assays["fast_na_deactivation"]
    voltages = runtime.array(a["tail_voltages_mv"])
    states = runtime.equilibrate(xp.full_like(voltages, a["hold_mv"]))
    states = runtime.advance(
        xp.full_like(voltages, a["conditioning_mv"]), states,
        xp.full_like(voltages, a["conditioning_duration_ms"]),
    )[:5]
    states = runtime.advance(
        xp.full_like(voltages, a["activation_mv"]), states,
        xp.full_like(voltages, a["activation_duration_ms"]),
    )[:5]
    result = runtime.trace(voltages, states, a["sample_times_ms"])
    output["fast_na_deactivation"] = _curve_payload(
        voltages, a["sample_times_ms"], result, "na"
    )

    a = assays["kv3_activation"]
    voltages = runtime.array(a["test_voltages_mv"])
    states = runtime.equilibrate(xp.full_like(voltages, a["hold_mv"]))
    result = runtime.trace(voltages, states, a["sample_times_ms"])
    conductance = result[6] / (voltages[:, None] - xp.float32(p.potassium_reversal_mv))
    peak = xp.max(xp.abs(conductance), axis=1)
    output["kv3_activation"] = {
        **_curve_payload(voltages, a["sample_times_ms"], result, "kv3"),
        "raw_normalized_conductance": conductance,
        "peak_normalized_conductance": peak,
        "ladder_normalized_peak_conductance": _safe_normalize(runtime, peak),
    }

    a = assays["kv3_inactivation"]
    prepulse = runtime.array(a["prepulse_voltages_mv"])
    states = runtime.equilibrate(xp.full_like(prepulse, a["hold_mv"]))
    states = runtime.advance(prepulse, states, xp.full_like(prepulse, a["prepulse_duration_ms"]))[:5]
    test_voltage = xp.full_like(prepulse, a["test_mv"])
    result = runtime.trace(test_voltage, states, a["test_sample_times_ms"])
    peak = xp.max(xp.abs(result[6]), axis=1)
    output["kv3_inactivation"] = {
        "prepulse_voltage_mv": prepulse,
        **_curve_payload(test_voltage, a["test_sample_times_ms"], result, "kv3"),
        "peak_absolute_test_current": peak,
        "ladder_normalized_peak_test_current": _safe_normalize(runtime, peak),
    }

    a = assays["kv3_rise"]
    voltage = runtime.array([a["test_mv"]])
    states = runtime.equilibrate(runtime.array([a["hold_mv"]]))
    result = runtime.trace(voltage, states, a["sample_times_ms"])
    output["kv3_rise"] = {
        **_curve_payload(voltage, a["sample_times_ms"], result, "kv3"),
        "trace_normalized_absolute_current": _safe_normalize(runtime, xp.abs(result[6])),
    }

    a = assays["kv3_deactivation"]
    voltages = runtime.array(a["tail_voltages_mv"])
    states = runtime.equilibrate(xp.full_like(voltages, a["hold_mv"]))
    states = runtime.advance(
        xp.full_like(voltages, a["activation_mv"]), states,
        xp.full_like(voltages, a["activation_duration_ms"]),
    )[:5]
    result = runtime.trace(voltages, states, a["sample_times_ms"])
    output["kv3_deactivation"] = _curve_payload(
        voltages, a["sample_times_ms"], result, "kv3"
    )
    for name, payload in output.items():
        payload["command"] = dict(assays[name])
    return output


def _host_value(runtime: _ClampRuntime, value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _host_value(runtime, item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_host_value(runtime, item) for item in value]
    if isinstance(value, list):
        return [_host_value(runtime, item) for item in value]
    if isinstance(value, runtime.xp.ndarray):
        if runtime.backend == "cupy":
            value = runtime.xp.asnumpy(value)
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return value


def run(
    *,
    spec_path: Path | str,
    spec_sha256: str,
    output_path: Path | str,
    backend: str,
    repository_root: Path | str = ROOT,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    spec, binding = load_execution_spec(
        spec_path, expected_file_sha256=spec_sha256, repository_root=root
    )
    matrix = {
        row["backend"]: row for row in spec["execution_matrix"]
    }
    _require(backend in matrix, "CLI backend is absent from execution matrix")
    execution = matrix[backend]
    output = Path(output_path)
    if not output.is_absolute():
        output = root / output
    output = output.resolve()
    try:
        relative_output = output.relative_to(root).as_posix()
    except ValueError as exc:
        raise FastChannelClampError("output must be inside the repository") from exc
    _require(relative_output == execution.get("output"), "output path differs from execution spec")
    _require(not output.exists(), "refusing to overwrite existing output")

    lab.assert_backend(backend, note="V14 Stage B fast-channel source-command clamp")
    runtime = _ClampRuntime(backend)
    normalized_commands = _normalize_commands(spec)
    device_curves = _run_assays(runtime, normalized_commands)
    document: dict[str, Any] = {
        "schema": OUTPUT_SCHEMA,
        "execution_spec": binding,
        "protocol": spec["protocol"],
        "backend": backend,
        "device": execution["device"],
        "dtype": "float32",
        "sample_interval_ms": float(spec["numeric_contract"]["clamp_sample_interval_ms"]),
        "implementation": spec["implementation"],
        "assays": _host_value(runtime, device_curves),
        "model_prior_boundaries": {
            **dict(EVIDENCE_CLASSES),
            "activation_gate_roots": "equation_derived_not_measured_gate_curves",
            "recovery_duration_ladder": "project_operational_sampling_not_source_reported",
            "ideal_current_timing": "operational_transfer_instrument_filter_unresolved",
            "junction_potential": "source_command_voltages_uncorrected",
        },
        "candidate_calibration_allowed": False,
        "scientific_verdict": None,
        "analysis_status": "raw_unanalyzed",
        "units": dict(UNITS),
        "parameters": asdict(DEFAULT_FAST_CHANNEL_PARAMETERS),
        "execution": {
            "integration": "exact_rush_larsen_fixed_voltage",
            "per_time_step_host_loop": False,
            "segment_launch_count": runtime.segment_launches,
            "host_transfer_boundary": "final_serialization_only",
        },
    }
    document["sha256"] = _semantic_digest(document)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("x", encoding="ascii") as handle:
            json.dump(document, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
            handle.write("\n")
    except FileExistsError as exc:
        raise FastChannelClampError("refusing to overwrite existing output") from exc
    return document


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=EXECUTION_SPEC_PATH)
    parser.add_argument("--spec-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--backend", choices=("numpy", "cupy"), required=True)
    args = parser.parse_args(argv)
    run(
        spec_path=args.spec,
        spec_sha256=args.spec_sha256,
        output_path=args.out,
        backend=args.backend,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
