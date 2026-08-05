"""Run one sealed source-model comparator under the Stage B clamp commands."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from sim import kv3_source_models as kv3
from sim import sodium_source_models as sodium
from tools import lab


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "v14-snr-stageB-source-model-transfer-v1"
OUTPUT_SCHEMA = "v14-snr-stageB-source-model-observation-v1"


class SourceModelTransferError(RuntimeError):
    """Raised when a source-model transfer leaves its sealed boundary."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SourceModelTransferError(message)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")


def _digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _semantic_digest(value: Mapping[str, Any]) -> str:
    return _digest_bytes(_canonical_bytes({key: item for key, item in value.items() if key != "sha256"}))


def _inside_file(root: Path, value: Any, context: str) -> tuple[str, Path]:
    _require(isinstance(value, str) and value and "\\" not in value and "\x00" not in value,
             f"{context} path is invalid")
    relative = PurePosixPath(value)
    _require(
        not relative.is_absolute() and str(relative) == value
        and all(part not in {"", ".", ".."} for part in relative.parts),
        f"{context} path is not canonical",
    )
    candidate = root.joinpath(*relative.parts)
    _require(not candidate.is_symlink(), f"{context} must not be a symbolic link")
    path = candidate.resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise SourceModelTransferError(f"{context} escapes repository") from exc
    _require(path.is_file(), f"{context} must be a regular file")
    return value, path


def _load_json(path: Path, context: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SourceModelTransferError(f"{context} is not valid JSON") from exc
    _require(isinstance(value, dict), f"{context} must contain an object")
    return value


def _load_binding(root: Path, binding: Any, context: str) -> tuple[dict[str, str], Path]:
    _require(isinstance(binding, Mapping) and set(binding) == {"path", "sha256"},
             f"{context} binding is invalid")
    relative, path = _inside_file(root, binding["path"], context)
    observed = _digest_bytes(path.read_bytes())
    _require(observed == binding["sha256"], f"{context} digest mismatch")
    return {"path": relative, "sha256": observed}, path


def _validate_models(spec: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    models = spec.get("models")
    _require(isinstance(models, list) and len(models) == 4, "exactly four source models are required")
    by_id = {row.get("model_id"): row for row in models if isinstance(row, Mapping)}
    _require(set(by_id) == {
        sodium.KHALIQ_RAMAN_13_STATE, sodium.BALBI_NAV16_SIX_STATE,
        kv3.LABRO_2015, kv3.DESAI_2008_CONTROL,
    }, "source model set changed")
    for model_id, row in by_id.items():
        _require(row.get("calibration_allowed") is False, f"{model_id} permits calibration")
        _require(row.get("conductance_fitting_allowed") is False, f"{model_id} permits conductance fitting")
        _require(isinstance(row.get("model_fingerprint"), str) and len(row["model_fingerprint"]) == 64,
                 f"{model_id} fingerprint is invalid")
        conditions = row.get("conditions")
        _require(isinstance(conditions, list) and conditions, f"{model_id} has no conditions")
        ids = [condition.get("condition_id") for condition in conditions if isinstance(condition, Mapping)]
        _require(len(ids) == len(conditions) and len(set(ids)) == len(ids),
                 f"{model_id} condition identities are invalid")
    expected_conditions = {
        sodium.KHALIQ_RAMAN_13_STATE: [("graph_stationary", None)],
        sodium.BALBI_NAV16_SIX_STATE: [("source_22c", 22.0)],
        kv3.LABRO_2015: [("room_20c", 20.0), ("room_22p5c", 22.5), ("room_25c", 25.0)],
        kv3.DESAI_2008_CONTROL: [("no_temperature", None)],
    }
    expected_unavailable = {
        sodium.KHALIQ_RAMAN_13_STATE: [],
        sodium.BALBI_NAV16_SIX_STATE: [],
        kv3.LABRO_2015: ["kv3_inactivation"],
        kv3.DESAI_2008_CONTROL: [],
    }
    for model_id, expected in expected_conditions.items():
        observed = [
            (condition.get("condition_id"), condition.get("temperature_c"))
            for condition in by_id[model_id]["conditions"]
        ]
        _require(observed == expected, f"{model_id} sealed conditions changed")
        _require(
            by_id[model_id].get("unavailable_assays") == expected_unavailable[model_id],
            f"{model_id} unavailable assay boundary changed",
        )
    return by_id


def load_spec(
    path: str | Path, expected_sha256: str, repository_root: str | Path = ROOT,
) -> tuple[dict[str, Any], dict[str, str]]:
    root = Path(repository_root).resolve()
    supplied = Path(path)
    candidate = (supplied if supplied.is_absolute() else root / supplied).resolve()
    try:
        relative = candidate.relative_to(root).as_posix()
    except ValueError as exc:
        raise SourceModelTransferError("execution spec escapes repository") from exc
    _require(candidate.is_file() and not candidate.is_symlink(), "execution spec is unavailable")
    observed = _digest_bytes(candidate.read_bytes())
    _require(observed == expected_sha256, "execution spec digest mismatch")
    spec = _load_json(candidate, "execution spec")
    _require(spec.get("schema") == SCHEMA, "execution spec schema mismatch")
    _require(spec.get("status") == "preregistered_not_executed", "execution spec is not prospective")
    _require(spec.get("scientific_verdict") is None, "execution spec contains a verdict")
    _require(spec.get("candidate_calibration_allowed") is False, "calibration is not forbidden")
    _require(spec.get("hybridization_allowed") is False, "hybridization is not forbidden")
    _require(spec.get("stage2_integration_allowed") is False, "Stage 2 integration is not forbidden")

    command_binding, command_path = _load_binding(root, spec.get("command_authority"), "command authority")
    commands = _load_json(command_path, "command authority")
    _require(commands.get("schema") == "v14-snr-stageB-fast-channel-clamp-execution-v1",
             "command authority schema changed")
    _require(commands.get("numeric_contract", {}).get("clamp_sample_interval_ms") == 0.005,
             "command sample interval changed")
    _require(set(commands.get("commands", {})) == {
        "fast_na_activation", "fast_na_inactivation", "fast_na_recovery",
        "fast_na_deactivation", "kv3_activation", "kv3_inactivation", "kv3_deactivation",
    }, "command authority assay set changed")
    spec["_commands"] = commands
    spec["_command_binding"] = command_binding

    for label in ("target_authority", "research_gate", "khaliq_initialization_correction"):
        _load_binding(root, spec.get(label), label.replace("_", " "))
    analysis_contract = spec.get("analysis_contract")
    _require(isinstance(analysis_contract, Mapping), "analysis contract is invalid")
    _load_binding(
        root, analysis_contract.get("analyzer_authority"), "analyzer authority"
    )
    _load_binding(
        root, analysis_contract.get("estimator_authority"), "estimator authority"
    )
    parity = analysis_contract.get("cpu_gpu_parity")
    _require(
        isinstance(parity, Mapping)
        and parity.get("rtol") == 5e-8
        and parity.get("atol") == 5e-10
        and parity.get("pointwise") is True,
        "CPU/GPU parity contract changed",
    )
    _require(analysis_contract.get("compensation_allowed") is False,
             "analysis permits compensation")
    _require(analysis_contract.get("stage2_verdict_allowed") is False,
             "analysis permits a Stage 2 verdict")

    implementation = spec.get("implementation")
    _require(isinstance(implementation, Mapping) and set(implementation) == {
        "sodium_models", "kv3_models", "runner"
    }, "implementation binding set changed")
    for label, binding in implementation.items():
        authenticated, loaded_path = _load_binding(root, binding, label)
        _require(authenticated == binding, f"{label} binding changed")
        if label == "runner":
            _require(loaded_path == Path(__file__).resolve(), "loaded runner path differs from binding")
        elif label == "sodium_models":
            _require(loaded_path == Path(sodium.__file__).resolve(), "loaded sodium module differs from binding")
        else:
            _require(loaded_path == Path(kv3.__file__).resolve(), "loaded Kv3 module differs from binding")

    models = _validate_models(spec)
    matrix = spec.get("execution_matrix")
    _require(isinstance(matrix, list) and len(matrix) == 12, "execution matrix must contain 12 jobs")
    identities = []
    outputs = []
    for row in matrix:
        _require(isinstance(row, Mapping), "execution row is invalid")
        identity = (row.get("model_id"), row.get("condition_id"), row.get("backend"))
        identities.append(identity)
        outputs.append(row.get("output"))
        _require(row.get("backend") in {"numpy", "cupy"}, "execution backend is invalid")
        _require(row.get("authority") == (
            "cpu_reference" if row.get("backend") == "numpy" else "gpu_parity_only"
        ), "execution authority is invalid")
    expected_identities = {
        (model_id, condition["condition_id"], backend)
        for model_id, model in models.items()
        for condition in model["conditions"]
        for backend in ("numpy", "cupy")
    }
    _require(len(set(identities)) == len(matrix), "execution matrix duplicates a job")
    _require(set(identities) == expected_identities, "execution matrix is not the exact sealed matrix")
    _require(len(set(outputs)) == len(matrix), "execution matrix duplicates an output")
    return spec, {"path": relative, "sha256": observed}


def _filed_times(duration_ms: float, interval_ms: float) -> list[float]:
    count = round(duration_ms / interval_ms)
    _require(count > 0 and math.isclose(count * interval_ms, duration_ms, abs_tol=1e-9),
             "command duration is not an integral sample count")
    return [interval_ms * index for index in range(1, count + 1)]


class _Runtime:
    def __init__(self, backend: str, model_id: str, temperature_c: float | None):
        from sim.backend import get_backend

        self.xp, actual = get_backend()
        _require(actual == backend, "backend changed after assertion")
        self.backend = backend
        self.model_id = model_id
        self.temperature_c = temperature_c
        self.family = "sodium" if model_id in sodium.MODEL_METADATA else "kv3"
        self.segment_operations = 0

    def array(self, value: Any):
        return self.xp.asarray(value, dtype=self.xp.float64)

    def equilibrium(self, voltage: Any):
        if self.family == "sodium":
            return sodium.equilibrium(self.model_id, voltage, self.temperature_c, self.xp)
        return kv3.equilibrium(self.model_id, voltage, self.temperature_c, self.xp)

    def advance(self, voltage: Any, states: Any, duration: Any):
        self.segment_operations += 1
        if self.family == "sodium":
            return sodium.advance(
                self.model_id, voltage, states, duration, self.temperature_c, self.xp
            )
        return kv3.advance(
            self.model_id, states, voltage, duration, self.temperature_c, self.xp
        )

    def open_probability(self, states: Any):
        if self.family == "sodium":
            return sodium.open_probability(self.model_id, states, self.xp)
        return kv3.open_probability(self.model_id, states, self.xp)

    def trace(self, voltage: Any, states: Any, times: Sequence[float]):
        voltage = self.array(voltage)
        times_array = self.array(times)
        if self.family == "sodium":
            self.segment_operations += 1
            return sodium.trace(
                self.model_id, voltage, states, times_array, self.temperature_c, self.xp
            )
        state_width = states.shape[-1]
        leading = voltage.shape
        expanded_state = self.xp.broadcast_to(
            states[..., None, :], leading + (times_array.size, state_width)
        )
        expanded_voltage = self.xp.broadcast_to(voltage[..., None], leading + (times_array.size,))
        expanded_time = self.xp.broadcast_to(times_array, leading + (times_array.size,))
        return self.advance(expanded_voltage, expanded_state, expanded_time)


def _safe_normalize(runtime: _Runtime, values: Any, axis: int | None = None):
    xp = runtime.xp
    scale = xp.max(xp.abs(values), axis=axis, keepdims=axis is not None)
    return xp.where(scale > 0.0, values / scale, xp.zeros_like(values))


def _current(runtime: _Runtime, voltage: Any, states: Any, reversal_mv: float):
    return runtime.open_probability(states) * (voltage - runtime.xp.float64(reversal_mv))


def _curve(runtime: _Runtime, voltage: Any, states: Any, times: Sequence[float], reversal_mv: float):
    traced = runtime.trace(voltage, states, times)
    voltage_grid = runtime.xp.broadcast_to(voltage[..., None], traced.shape[:-1])
    current = _current(runtime, voltage_grid, traced, reversal_mv)
    return traced, current


def _sodium_assays(runtime: _Runtime, commands: Mapping[str, Any], interval: float) -> dict[str, Any]:
    xp = runtime.xp
    reversal = 50.0
    output: dict[str, Any] = {}

    command = commands["fast_na_activation"]
    voltage = runtime.array(command["test_mV"])
    state = runtime.equilibrium(xp.full_like(voltage, command["hold_mV"]))
    times = _filed_times(command["test_duration_ms"], interval)
    traced, current = _curve(runtime, voltage, state, times, reversal)
    conductance = current / (voltage[:, None] - reversal)
    peak = xp.max(xp.abs(conductance), axis=1)
    output["fast_na_activation"] = {
        "command_voltage_mv": voltage, "elapsed_ms": times,
        "normalized_na_current": current, "raw_normalized_conductance": conductance,
        "peak_normalized_conductance": peak,
        "ladder_normalized_peak_conductance": _safe_normalize(runtime, peak),
    }

    command = commands["fast_na_inactivation"]
    prepulse = runtime.array(command["prepulse_mV"])
    state = runtime.equilibrium(xp.full_like(prepulse, command["hold_mV"]))
    state = runtime.advance(prepulse, state, command["prepulse_duration_ms"])
    test_voltage = xp.full_like(prepulse, command["test_mV"])
    times = _filed_times(command["test_duration_ms"], interval)
    traced, current = _curve(runtime, test_voltage, state, times, reversal)
    peak = xp.max(xp.abs(current), axis=1)
    output["fast_na_inactivation"] = {
        "prepulse_voltage_mv": prepulse, "command_voltage_mv": test_voltage,
        "elapsed_ms": times, "normalized_na_current": current,
        "peak_absolute_test_current": peak,
        "ladder_normalized_peak_test_current": _safe_normalize(runtime, peak),
    }

    activation = commands["fast_na_activation"]
    voltage = runtime.array([0.0])
    state = runtime.equilibrium(runtime.array([activation["hold_mV"]]))
    times = _filed_times(activation["test_duration_ms"], interval)
    traced, current = _curve(runtime, voltage, state, times, reversal)
    output["fast_na_composite_zero"] = {
        "command_voltage_mv": voltage, "elapsed_ms": times,
        "normalized_na_current": current,
        "trace_normalized_absolute_current": _safe_normalize(runtime, xp.abs(current)),
    }

    command = commands["fast_na_recovery"]
    durations = runtime.array(command["recovery_duration_ms"])
    state = runtime.equilibrium(xp.full_like(durations, command["hold_mV"]))
    state = runtime.advance(xp.full_like(durations, command["recovery_prepulse_mV"]), state,
                            command["recovery_prepulse_duration_ms"])
    state = runtime.advance(xp.full_like(durations, command["inactivation_mV"]), state,
                            command["inactivation_duration_ms"])
    state = runtime.advance(xp.full_like(durations, command["recovery_mV"]), state, durations)
    test_voltage = xp.full_like(durations, command["test_mV"])
    times = _filed_times(command["test_duration_ms"], interval)
    traced, current = _curve(runtime, test_voltage, state, times, reversal)
    peak = xp.max(xp.abs(current), axis=1)
    output["fast_na_recovery"] = {
        "recovery_duration_ms": durations, "command_voltage_mv": test_voltage,
        "elapsed_ms": times, "normalized_na_current": current,
        "peak_absolute_test_current": peak,
        "ladder_normalized_peak_test_current": _safe_normalize(runtime, peak),
    }

    command = commands["fast_na_deactivation"]
    voltage = runtime.array(command["test_mV"])
    state = runtime.equilibrium(xp.full_like(voltage, command["hold_mV"]))
    state = runtime.advance(xp.full_like(voltage, command["prepulse_mV"]), state,
                            command["prepulse_duration_ms"])
    state = runtime.advance(xp.full_like(voltage, command["activation_mV"]), state,
                            command["activation_duration_ms"])
    times = _filed_times(command["test_duration_ms"], interval)
    traced, current = _curve(runtime, voltage, state, times, reversal)
    output["fast_na_deactivation"] = {
        "command_voltage_mv": voltage, "elapsed_ms": times,
        "normalized_na_current": current,
    }
    return output


def _kv3_assays(
    runtime: _Runtime, commands: Mapping[str, Any], interval: float, *, include_inactivation: bool,
) -> dict[str, Any]:
    xp = runtime.xp
    reversal = -90.0
    output: dict[str, Any] = {}
    command = commands["kv3_activation"]
    voltage = runtime.array(command["test_mV"])
    state = runtime.equilibrium(xp.full_like(voltage, command["hold_mV"]))
    times = _filed_times(command["test_duration_ms"], interval)
    traced, current = _curve(runtime, voltage, state, times, reversal)
    conductance = current / (voltage[:, None] - reversal)
    peak = xp.max(xp.abs(conductance), axis=1)
    output["kv3_activation"] = {
        "command_voltage_mv": voltage, "elapsed_ms": times,
        "normalized_kv3_current": current, "raw_normalized_conductance": conductance,
        "peak_normalized_conductance": peak,
        "ladder_normalized_peak_conductance": _safe_normalize(runtime, peak),
    }

    voltage = runtime.array([40.0])
    state = runtime.equilibrium(runtime.array([command["hold_mV"]]))
    traced, current = _curve(runtime, voltage, state, times, reversal)
    output["kv3_rise"] = {
        "command_voltage_mv": voltage, "elapsed_ms": times,
        "normalized_kv3_current": current,
        "trace_normalized_absolute_current": _safe_normalize(runtime, xp.abs(current)),
    }

    if include_inactivation:
        command = commands["kv3_inactivation"]
        prepulse = runtime.array(command["prepulse_mV"])
        state = runtime.equilibrium(xp.full_like(prepulse, command["hold_mV"]))
        state = runtime.advance(prepulse, state, command["prepulse_duration_ms"])
        test_voltage = xp.full_like(prepulse, command["test_mV"])
        test_times = _filed_times(command["test_duration_ms"], interval)
        traced, current = _curve(runtime, test_voltage, state, test_times, reversal)
        peak = xp.max(xp.abs(current), axis=1)
        output["kv3_inactivation"] = {
            "prepulse_voltage_mv": prepulse, "command_voltage_mv": test_voltage,
            "elapsed_ms": test_times, "normalized_kv3_current": current,
            "peak_absolute_test_current": peak,
            "ladder_normalized_peak_test_current": _safe_normalize(runtime, peak),
        }

    command = commands["kv3_deactivation"]
    voltage = runtime.array(command["test_mV"])
    state = runtime.equilibrium(xp.full_like(voltage, command["hold_mV"]))
    state = runtime.advance(xp.full_like(voltage, command["activation_mV"]), state,
                            command["activation_duration_ms"])
    tail_times = _filed_times(command["test_duration_ms"], interval)
    traced, current = _curve(runtime, voltage, state, tail_times, reversal)
    output["kv3_deactivation"] = {
        "command_voltage_mv": voltage, "elapsed_ms": tail_times,
        "normalized_kv3_current": current,
    }
    return output


def _host(runtime: _Runtime, value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _host(runtime, item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_host(runtime, item) for item in value]
    if isinstance(value, runtime.xp.ndarray):
        if runtime.backend == "cupy":
            value = runtime.xp.asnumpy(value)
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return value


def run(
    *, spec_path: str | Path, spec_sha256: str, model_id: str, condition_id: str,
    backend: str, output_path: str | Path, repository_root: str | Path = ROOT,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    spec, binding = load_spec(spec_path, spec_sha256, root)
    models = _validate_models(spec)
    _require(model_id in models, "model is absent from the sealed contract")
    model = models[model_id]
    conditions = {row["condition_id"]: row for row in model["conditions"]}
    _require(condition_id in conditions, "condition is absent from the sealed model")
    condition = conditions[condition_id]
    jobs = [row for row in spec["execution_matrix"] if (
        row["model_id"], row["condition_id"], row["backend"]
    ) == (model_id, condition_id, backend)]
    _require(len(jobs) == 1, "execution job is not uniquely sealed")
    job = jobs[0]
    output = Path(output_path)
    output = (output if output.is_absolute() else root / output).resolve()
    expected_output = (root / job["output"]).resolve()
    _require(output == expected_output, "output differs from the sealed job")
    _require(not output.exists(), "refusing to overwrite source-model evidence")

    lab.assert_backend(backend, note=f"Stage B source-model transfer: {model_id}/{condition_id}")
    runtime = _Runtime(backend, model_id, condition.get("temperature_c"))
    commands = spec["_commands"]["commands"]
    interval = float(spec["_commands"]["numeric_contract"]["clamp_sample_interval_ms"])
    if runtime.family == "sodium":
        assays = _sodium_assays(runtime, commands, interval)
        metadata = dict(sodium.MODEL_METADATA[model_id])
    else:
        assays = _kv3_assays(
            runtime, commands, interval,
            include_inactivation=model_id == kv3.DESAI_2008_CONTROL,
        )
        metadata = dict(kv3.MODEL_METADATA[model_id])

    document: dict[str, Any] = {
        "schema": OUTPUT_SCHEMA,
        "execution_spec": binding,
        "command_authority": spec["_command_binding"],
        "implementation": dict(spec["implementation"]),
        "model_id": model_id,
        "condition_id": condition_id,
        "family": runtime.family,
        "model_fingerprint": model["model_fingerprint"],
        "model_metadata": _host(runtime, metadata),
        "condition": dict(condition),
        "backend": backend,
        "device": job["device"],
        "authority": job["authority"],
        "dtype": "float64",
        "sample_interval_ms": interval,
        "assays": _host(runtime, assays),
        "unavailable_assays": list(model["unavailable_assays"]),
        "candidate_calibration_allowed": False,
        "conductance_fitting_allowed": False,
        "scientific_verdict": None,
        "analysis_status": "raw_unanalyzed",
        "execution": {
            "fixed_voltage_solver": model["fixed_voltage_solver"],
            "per_time_step_host_loop": False,
            "segment_operation_count": runtime.segment_operations,
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
        raise SourceModelTransferError("refusing to overwrite source-model evidence") from exc
    return document


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--spec-sha256", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--condition-id", required=True)
    parser.add_argument("--backend", choices=("numpy", "cupy"), required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    result = run(
        spec_path=args.spec, spec_sha256=args.spec_sha256, model_id=args.model_id,
        condition_id=args.condition_id, backend=args.backend, output_path=args.out,
    )
    print(json.dumps({
        "output": args.out, "sha256": result["sha256"], "model_id": args.model_id,
        "condition_id": args.condition_id, "backend": args.backend,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
