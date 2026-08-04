"""Create-only strict-arithmetic replay of sealed V13 transplant state."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import re
from typing import Any, Iterator

import numpy as np

from research.runners import _v13_backend_state_transplant as transplant
from sim.backend import get_backend, synchronize, to_host
from sim.config import CoreSimConfig
from tools import execution_receipt
from tools.lab import assert_backend


ROOT = Path(__file__).resolve().parents[2]
SPEC_RELATIVE_PATH = Path(
    "research/specs/v13_backend_neutral_izh_arithmetic_replay_diagnostic.json"
)
SPEC_PATH = ROOT / SPEC_RELATIVE_PATH
SPEC_SHA256 = "c8f42fee5d8d2dc044cf05ed7676c06ef7c20e093613a06facce311512686d2d"
BACKENDS = ("numpy", "cupy")
TRAJECTORIES = ("v", "u", "spikes")
SCHEMA_CELL = "v13-backend-neutral-izh-arithmetic-replay-cell-v1"
SCHEMA_COMPARISON = "v13-backend-neutral-izh-arithmetic-replay-comparison-v1"
PROMOTION_VALUE = "none"
TOTAL_STEPS = 1_200
AUTHORITY_SOURCE_PATHS = (
    "research/__init__.py",
    "research/runners/__init__.py",
    "research/runners/_vocal_action_credit_gate_v13_tonic_output.py",
    "research/runners/_v13_backend_state_transplant.py",
    "research/runners/_v13_backend_neutral_izh_arithmetic_replay.py",
    "research/findings/2026-08-04-neural-vocal-credit-gateB-v13-backend-"
    "arithmetic-correction-DIAGNOSTIC-PREREGISTRATION.md",
    "research/findings/2026-08-04-neural-vocal-credit-gateB-v13-backend-neutral-"
    "izh-arithmetic-replay-DIAGNOSTIC-PREREGISTRATION.md",
    "research/specs/v13_backend_arithmetic_localizer.json",
    "research/specs/v13_backend_neutral_izh_arithmetic_replay_diagnostic.json",
    "research/specs/v13_backend_state_transplant.json",
    "research/findings/raw/v13_backend_state_transplant/aggregate.json",
    "research/findings/raw/v13_backend_state_transplant/bundle-numpy.json",
    "research/findings/raw/v13_backend_state_transplant/receipt-aggregate.json",
    "research/findings/raw/v13_backend_state_transplant/receipt-bundle-numpy.json",
    "research/findings/raw/v13_backend_state_transplant/source-manifest.sha256",
    "tools/execution_receipt.py",
    "tools/lab.py",
    "tools/v13_backend_neutral_izh_initialization_evidence.py",
    "tools/v13_backend_neutral_izh_arithmetic_replay_evidence.py",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class ReplayProtocol:
    """Immutable names and authorities for one replay protocol version."""

    spec_relative_path: Path
    spec_sha256: str
    spec_id: str
    output_directory: str
    diagnostic_schema: str
    runner_module: str
    cell_schema: str
    comparison_schema: str
    authority_source_paths: tuple[str, ...]
    enforce_output_directory: bool


V1_PROTOCOL = ReplayProtocol(
    spec_relative_path=SPEC_RELATIVE_PATH,
    spec_sha256=SPEC_SHA256,
    spec_id="gateB-v13-backend-neutral-izh-arithmetic-replay-diagnostic-v1",
    output_directory=(
        "research/findings/raw/v13_backend_neutral_izh_arithmetic_replay_diagnostic"
    ),
    diagnostic_schema="v13-backend-neutral-izh-arithmetic-replay-spec-v1",
    runner_module="research.runners._v13_backend_neutral_izh_arithmetic_replay",
    cell_schema=SCHEMA_CELL,
    comparison_schema=SCHEMA_COMPARISON,
    authority_source_paths=AUTHORITY_SOURCE_PATHS,
    enforce_output_directory=False,
)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")


def _digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest_file(path: Path) -> str:
    return _digest_bytes(path.read_bytes())


def _artifact_digest(value: dict[str, Any]) -> str:
    return _digest_bytes(_canonical({key: item for key, item in value.items() if key != "sha256"}))


def _seal(value: dict[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["sha256"] = _artifact_digest(result)
    return result


def _write_new_json(path: Path, value: dict[str, Any]) -> None:
    if path.suffix != ".json":
        raise ValueError("output artifact must use a .json suffix")
    if not path.parent.is_dir():
        raise ValueError("output parent directory does not exist")
    if os.path.lexists(path):
        raise FileExistsError(f"output artifact already exists: {path}")
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("ascii")
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o644
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _require_protocol_output(
    path: Path, protocol: ReplayProtocol, label: str,
) -> None:
    if not protocol.enforce_output_directory:
        return
    root = ROOT.resolve(strict=True)
    expected = (root / protocol.output_directory).resolve(strict=False)
    resolved = path.resolve(strict=False)
    try:
        relative = resolved.relative_to(expected)
    except ValueError as exc:
        raise ValueError(
            f"{label} must be inside {protocol.output_directory}"
        ) from exc
    if not relative.parts:
        raise ValueError(f"{label} must name a file inside the output directory")


def source_paths(
    root: Path | None = None, *, protocol: ReplayProtocol = V1_PROTOCOL,
) -> tuple[str, ...]:
    """Return every simulator Python input and exact replay authority."""
    resolved = (ROOT if root is None else root).resolve(strict=True)
    simulator = (
        path.relative_to(resolved).as_posix()
        for path in (resolved / "sim").rglob("*.py")
        if path.is_file()
    )
    return tuple(sorted(set(simulator).union(protocol.authority_source_paths)))


def load_locked_spec(
    path: Path | None = None, expected_sha256: str | None = None, *,
    protocol: ReplayProtocol = V1_PROTOCOL,
) -> dict[str, Any]:
    path = ROOT / protocol.spec_relative_path if path is None else path
    expected_sha256 = protocol.spec_sha256 if expected_sha256 is None else expected_sha256
    if _SHA256.fullmatch(expected_sha256) is None:
        raise ValueError("spec SHA-256 must be a lowercase digest")
    raw = path.read_bytes()
    if _digest_bytes(raw) != expected_sha256:
        raise ValueError("locked spec digest mismatch")
    try:
        spec = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("locked spec is not valid JSON") from exc
    replay = spec.get("replay", {})
    execution = spec.get("execution", {})
    rng = spec.get("rng", {})
    acceptance = spec.get("acceptance", {})
    checks = {
        "schema": spec.get("schema") == "sim-experiment-spec-v0",
        "id": spec.get("id") == protocol.spec_id,
        "status": spec.get("status") == "preregistered_not_executed",
        "device": spec.get("device") == "not_applicable_non_executed_protocol",
        "promotion": spec.get("promotion_value") == PROMOTION_VALUE,
        "diagnostic": spec.get("diagnostic_only") is True,
        "verdict": spec.get("scientific_verdict") is None,
        "backends": execution.get("backends") == list(BACKENDS),
        "flag": execution.get("config_flag") == "backend_neutral_izh_arithmetic",
        "flag_value": execution.get("config_flag_value") is True,
        "default": execution.get("default_value") is False,
        "mode": execution.get("mode") == "default",
        "trajectories": execution.get("trajectory_arrays") == list(TRAJECTORIES),
        "steps": replay.get("total_steps") == TOTAL_STEPS
        and sum(replay.get(name, -TOTAL_STEPS) for name in (
            "baseline_steps", "inhibition_steps", "release_steps"
        )) == TOTAL_STEPS,
        "rng": rng.get("measured_replay_allowed") is False
        and rng.get("measured_replay_seed") is None
        and rng.get("allocation_state_must_be_fully_overwritten") is True,
        "exact": all(acceptance.get(name) is True for name in (
            "all_1200_v_rows_byte_exact", "all_1200_u_rows_byte_exact",
            "all_1200_spike_rows_byte_exact", "no_rng_call_during_measured_replay",
            "no_tolerance_fallback", "source_and_receipt_binding_required",
        )),
        "output": spec.get("output_directory") == protocol.output_directory,
        "diagnostic_schema": spec.get("diagnostic_schema")
        == protocol.diagnostic_schema,
    }
    if not all(checks.values()):
        raise ValueError(f"runner/spec disagreement: {checks}")
    return spec


def _source_snapshot(
    manifest: Path, revision: str, *, protocol: ReplayProtocol = V1_PROTOCOL,
) -> dict[str, Any]:
    if _REVISION.fullmatch(revision) is None:
        raise ValueError("source revision must be a full lowercase Git SHA")
    try:
        relative = manifest.resolve(strict=True).relative_to(ROOT.resolve(strict=True)).as_posix()
    except (OSError, ValueError) as exc:
        raise ValueError("source manifest must be inside the repository") from exc
    snapshot = execution_receipt.verify_source_manifest(ROOT, relative)
    kind = execution_receipt._source_revision(ROOT, revision, snapshot["manifest_sha256"])
    if set(snapshot["files"]) != set(source_paths(protocol=protocol)):
        raise ValueError("source manifest does not contain the exact replay source set")
    return {
        "file_count": snapshot["file_count"],
        "git_sha": revision,
        "kind": kind,
        "manifest": snapshot["manifest"],
        "manifest_sha256": snapshot["manifest_sha256"],
        "tree_sha256": snapshot["tree_sha256"],
    }


def _read_bound_json(record: dict[str, Any], label: str) -> tuple[dict[str, Any], Path]:
    path = ROOT / record["path"]
    if _digest_file(path) != record["file_sha256"]:
        raise ValueError(f"completed {label} file digest mismatch")
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"completed {label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"completed {label} must be a JSON object")
    return value, path


def _validate_historical_receipt(
    record: dict[str, Any], artifact_path: str, artifact_sha256: str,
    source_revision: str, source_manifest: dict[str, Any], label: str,
) -> dict[str, Any]:
    path = ROOT / record["receipt_path"]
    if _digest_file(path) != record["receipt_file_sha256"]:
        raise ValueError(f"completed {label} receipt digest mismatch")
    try:
        receipt = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"completed {label} receipt is not valid JSON") from exc
    receipt_source = receipt.get("source", {})
    checks = {
        "schema": receipt.get("schema") == "sim-execution-receipt-v1",
        "status": receipt.get("status") == "success" and receipt.get("exit_code") == 0,
        "artifact": receipt.get("artifact", {}).get("path") == artifact_path
        and receipt.get("artifact", {}).get("sha256") == artifact_sha256,
        "source": receipt_source.get("git_sha") == source_revision
        and receipt_source.get("kind") == "git"
        and receipt_source.get("manifest") == source_manifest["path"]
        and receipt_source.get("manifest_sha256") == source_manifest["file_sha256"]
        and receipt_source.get("tree_sha256") == source_manifest["tree_sha256"]
        and receipt_source.get("file_count") == 6,
    }
    if not all(checks.values()):
        raise ValueError(f"completed {label} receipt contract mismatch: {checks}")
    return receipt


def load_completed_input(spec: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the immutable transplant bundle without applying current-source checks."""
    completed = spec["completed_input"]
    source_manifest_record = completed["source_manifest"]
    source_manifest_path = ROOT / source_manifest_record["path"]
    if _digest_file(source_manifest_path) != source_manifest_record["file_sha256"]:
        raise ValueError("completed transplant source manifest digest mismatch")
    transplant_spec_record = completed["transplant_spec"]
    transplant_spec_path = ROOT / transplant_spec_record["path"]
    if _digest_file(transplant_spec_path) != transplant_spec_record["file_sha256"]:
        raise ValueError("completed transplant spec digest mismatch")
    transplant_spec = json.loads(transplant_spec_path.read_text(encoding="ascii"))

    bundle_record = completed["bundle"]
    bundle, _ = _read_bound_json(bundle_record, "bundle")
    transplant._validate_artifact_digest(bundle, "bundle")
    checks = {
        "schema": bundle.get("schema") == transplant.SCHEMA_BUNDLE,
        "artifact": bundle.get("artifact_sha256") == bundle_record["artifact_sha256"],
        "spec": bundle.get("spec_sha256") == transplant_spec_record["file_sha256"],
        "origin": bundle.get("origin") == bundle_record["origin"] == "numpy",
        "seed": bundle.get("seed") == transplant_spec.get("seed"),
        "network": bundle.get("network") == transplant_spec.get("network"),
    }
    if not all(checks.values()):
        raise ValueError(f"completed bundle contract mismatch: {checks}")
    cp_arrays = bundle.get("cp_arrays")
    if not isinstance(cp_arrays, dict) or not cp_arrays:
        raise ValueError("completed bundle has no sealed arrays")
    for name, value in cp_arrays.items():
        transplant._decode_array(value, f"bundle array {name}")
    neuron_count = int(
        transplant._decode_array(cp_arrays["cp_firing_states"], "firing states").size
    )
    transplant._validate_regions(bundle.get("regions", {}), neuron_count)
    transplant._validate_csr(bundle.get("connections_csr"), neuron_count)
    expected_hashes = transplant._required_bundle_hashes(
        cp_arrays, bundle["connections_csr"]
    )
    if bundle.get("required_bundle_array_sha256") != expected_hashes:
        raise ValueError("completed bundle array binding mismatch")
    _validate_historical_receipt(
        bundle_record, bundle_record["path"], bundle_record["file_sha256"],
        completed["source_revision"], source_manifest_record, "bundle",
    )

    aggregate_record = completed["aggregate"]
    aggregate, _ = _read_bound_json(aggregate_record, "aggregate")
    transplant._validate_artifact_digest(aggregate, "aggregate")
    aggregate_checks = {
        "schema": aggregate.get("schema") == transplant.SCHEMA_AGGREGATE,
        "artifact": aggregate.get("artifact_sha256") == aggregate_record["artifact_sha256"],
        "spec": aggregate.get("spec_sha256") == transplant_spec_record["file_sha256"],
        "bundle": aggregate.get("bundles", {}).get("numpy")
        == bundle_record["artifact_sha256"],
        "complete": aggregate.get("matrix_complete") is True,
        "seed": aggregate.get("seed") == transplant_spec.get("seed"),
    }
    if not all(aggregate_checks.values()):
        raise ValueError(f"completed aggregate contract mismatch: {aggregate_checks}")
    _validate_historical_receipt(
        aggregate_record, aggregate_record["path"], aggregate_record["file_sha256"],
        completed["source_revision"], source_manifest_record, "aggregate",
    )

    replay = spec["replay"]
    schedule_checks = {
        "baseline": transplant_spec.get("steps", {}).get("baseline")
        == replay["baseline_steps"],
        "inhibition": transplant_spec.get("steps", {}).get("inhibition")
        == replay["inhibition_steps"],
        "release": transplant_spec.get("steps", {}).get("release")
        == replay["release_steps"],
        "dt": transplant_spec.get("steps", {}).get("dt_ms") == replay["dt_ms"],
        "source_current": transplant_spec.get("stimulus", {}).get("source_current_pA")
        == replay["source_current_pA"],
        "target_current": transplant_spec.get("stimulus", {}).get(
            "target_external_current_pA"
        ) == replay["target_external_current_pA"],
    }
    if not all(schedule_checks.values()):
        raise ValueError(f"replay schedule differs from completed transplant: {schedule_checks}")
    return bundle, transplant_spec


@contextmanager
def _forbid_rng_calls() -> Iterator[dict[str, Any]]:
    """Fail on global RNG calls during the measured replay boundary."""
    xp, _ = get_backend()
    modules = [np.random, random]
    backend_random = getattr(xp, "random", None)
    if backend_random is not None and backend_random is not np.random:
        modules.append(backend_random)
    names = (
        "beta", "choice", "default_rng", "getrandbits", "integers", "lognormal",
        "normal", "rand", "randint", "randn", "random", "random_sample",
        "randrange", "sample", "shuf" + "fle", "uniform",
    )
    originals: list[tuple[Any, str, Any]] = []
    audit = {"active": True, "calls_observed": 0, "guarded_apis": []}

    def blocked(*_args, **_kwargs):
        audit["calls_observed"] += 1
        raise RuntimeError("RNG call forbidden during measured V13 replay")

    try:
        for module in modules:
            module_name = getattr(module, "__name__", type(module).__name__)
            for name in names:
                if hasattr(module, name):
                    original = getattr(module, name)
                    try:
                        setattr(module, name, blocked)
                    except (AttributeError, TypeError):
                        continue
                    originals.append((module, name, original))
                    audit["guarded_apis"].append(f"{module_name}.{name}")
        yield audit
    finally:
        for module, name, original in reversed(originals):
            setattr(module, name, original)
        audit["active"] = False


def _runtime_contract(bridge, transplant_spec: dict[str, Any]) -> dict[str, Any]:
    inherited = transplant._require_runtime_contract(bridge, transplant_spec, "default")
    strict = getattr(bridge.core_config, "backend_neutral_izh_arithmetic", None)
    default = CoreSimConfig().backend_neutral_izh_arithmetic
    contract = {
        "transplant_runtime_contract": inherited,
        "backend_neutral_izh_arithmetic": strict,
        "declared_default": default,
        "strict_flag_exact": strict is True,
        "default_off": default is False,
        "step_megakernel_disabled": bridge.core_config.enable_step_megakernel is False,
    }
    contract["valid"] = all((
        inherited["contract_valid"], contract["strict_flag_exact"],
        contract["default_off"], contract["step_megakernel_disabled"],
    ))
    if not contract["valid"]:
        raise ValueError(f"strict replay runtime contract mismatch: {contract}")
    return contract


def run_cell(
    *, backend: str, out: Path, source_manifest: Path, source_revision: str,
    spec_path: Path | None = None, spec_sha256: str | None = None,
    protocol: ReplayProtocol = V1_PROTOCOL,
) -> dict[str, Any]:
    if backend not in BACKENDS:
        raise ValueError("backend must be numpy or cupy")
    _require_protocol_output(out, protocol, "cell output")
    spec_sha256 = protocol.spec_sha256 if spec_sha256 is None else spec_sha256
    spec = load_locked_spec(spec_path, spec_sha256, protocol=protocol)
    source = _source_snapshot(source_manifest, source_revision, protocol=protocol)
    assert_backend(backend, note="V13 strict-arithmetic matched-state replay")
    _, actual_backend = get_backend()
    if actual_backend != backend or os.environ.get("SIM_BACKEND") != backend:
        raise ValueError("active backend does not match the requested replay cell")
    if os.path.lexists(out):
        raise FileExistsError(f"output artifact already exists: {out}")
    bundle, transplant_spec = load_completed_input(spec)

    bridge = transplant._build_bridge_from_spec(transplant_spec)
    try:
        construction_contract = transplant._require_runtime_contract(
            bridge, transplant_spec, "default"
        )
        restore = transplant._restore_bundle(bridge, bundle)
        bridge.core_config.backend_neutral_izh_arithmetic = True
        runtime_contract = _runtime_contract(bridge, transplant_spec)
        if not restore["all_exact"]:
            raise ValueError("sealed state was not restored exactly")

        xp, _ = get_backend()
        source_indices = bundle["regions"]["source"]["indices"]
        source_index_array = xp.asarray(source_indices, dtype=xp.int64)
        replay = spec["replay"]
        inhibition_start = replay["baseline_steps"]
        inhibition_stop = inhibition_start + replay["inhibition_steps"]
        source_current = xp.float32(replay["source_current_pA"])
        target_current = xp.float32(replay["target_external_current_pA"])
        neuron_count = int(bridge.core_config.num_neurons)
        trajectories = {
            "v": np.empty((TOTAL_STEPS, neuron_count), dtype=np.float32),
            "u": np.empty((TOTAL_STEPS, neuron_count), dtype=np.float32),
            "spikes": np.empty((TOTAL_STEPS, neuron_count), dtype=np.bool_),
        }
        initial_weight_hash = transplant._hash_array(bridge.cp_connections.data)
        initial_intrinsic_hash = transplant._hash_array(bridge.cp_intrinsic_current_pA)

        with _forbid_rng_calls() as rng_audit:
            for step in range(TOTAL_STEPS):
                bridge.cp_external_input_current[:] = target_current
                if inhibition_start <= step < inhibition_stop:
                    bridge.cp_external_input_current[source_index_array] = source_current
                bridge._run_one_simulation_step()
                bridge.runtime_state.current_time_ms += bridge.core_config.dt_ms
                bridge.runtime_state.current_time_step += 1
                trajectories["v"][step] = np.asarray(
                    to_host(bridge.cp_membrane_potential_v)
                )
                trajectories["u"][step] = np.asarray(to_host(bridge.cp_recovery_variable_u))
                trajectories["spikes"][step] = np.asarray(to_host(bridge.cp_firing_states))
            synchronize()

        validations = {
            "pre_step_restore_exact": restore["all_exact"],
            "runtime_contract_valid": runtime_contract["valid"],
            "rng_calls_during_measured_replay": rng_audit["calls_observed"],
            "rng_guard_released_after_replay": rng_audit["active"] is False,
            "steps_exact": bridge.runtime_state.current_time_step == TOTAL_STEPS,
            "time_exact": bridge.runtime_state.current_time_ms
            == TOTAL_STEPS * replay["dt_ms"],
            "finite_v_u": bool(np.isfinite(trajectories["v"]).all())
            and bool(np.isfinite(trajectories["u"]).all()),
            "weights_immutable": initial_weight_hash
            == transplant._hash_array(bridge.cp_connections.data),
            "intrinsic_current_immutable": initial_intrinsic_hash
            == transplant._hash_array(bridge.cp_intrinsic_current_pA),
        }
        instrument_valid = all((
            validations["pre_step_restore_exact"], validations["runtime_contract_valid"],
            validations["rng_calls_during_measured_replay"] == 0,
            validations["rng_guard_released_after_replay"], validations["steps_exact"],
            validations["time_exact"], validations["finite_v_u"],
            validations["weights_immutable"], validations["intrinsic_current_immutable"],
        ))
        if not instrument_valid:
            raise ValueError(f"replay instrument contract failed: {validations}")
        encoded = {
            name: transplant._encode_array(trajectories[name]) for name in TRAJECTORIES
        }
        artifact = _seal({
            "schema": protocol.cell_schema,
            "promotion_value": PROMOTION_VALUE,
            "diagnostic_only": True,
            "scientific_verdict": None,
            "backend": backend,
            "spec_sha256": spec_sha256,
            "source": source,
            "completed_input": {
                "bundle_path": spec["completed_input"]["bundle"]["path"],
                "bundle_file_sha256": spec["completed_input"]["bundle"]["file_sha256"],
                "bundle_artifact_sha256": bundle["artifact_sha256"],
                "aggregate_file_sha256": spec["completed_input"]["aggregate"][
                    "file_sha256"
                ],
                "historical_source_revision": spec["completed_input"]["source_revision"],
            },
            "allocation_disclosure": {
                "bridge_allocated_before_restore": True,
                "allocation_may_have_used_rng": True,
                "all_allocated_cp_arrays_overwritten_exactly": restore[
                    "cp_array_hashes_exact"
                ] and restore["cp_array_set_exact"],
                "csr_overwritten_exactly": restore["csr_hashes_exact"],
                "no_allocation_state_survived_restore": restore["all_exact"],
            },
            "measured_replay_rng": {
                "seed": None,
                "allowed": False,
                "boundary": spec["rng"]["replay_boundary"],
                "calls_observed": rng_audit["calls_observed"],
                "guarded_apis": sorted(rng_audit["guarded_apis"]),
            },
            "runtime_contract": runtime_contract,
            "pre_step_restore_verification": restore,
            "schedule": replay,
            "simulation_steps_executed": TOTAL_STEPS,
            "trajectories": encoded,
            "trajectory_sha256": {
                name: encoded[name]["sha256"] for name in TRAJECTORIES
            },
            "trajectory_step_sha256": {
                name: transplant._trajectory_step_hashes(trajectories[name])
                for name in TRAJECTORIES
            },
            "validations": validations,
            "instrument_valid": True,
        })
        _write_new_json(out, artifact)
        return artifact
    finally:
        bridge.clear_simulation_state_and_gpu_memory()


def _expected_cell_argv(
    *, root: Path, artifact: Path, backend: str, source: dict[str, Any],
    python: str, protocol: ReplayProtocol = V1_PROTOCOL,
) -> list[str]:
    return [
        python, "-m", protocol.runner_module,
        "--spec", str((root / protocol.spec_relative_path).resolve()),
        "--spec-sha256", protocol.spec_sha256,
        "--run", "--backend", backend,
        "--source-manifest", str((root / source["manifest"]).resolve()),
        "--source-revision", source["git_sha"],
        "--out", str(artifact.resolve()),
    ]


def _load_cell(
    path: Path, receipt_path: Path, backend: str, *,
    protocol: ReplayProtocol = V1_PROTOCOL,
) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {backend} replay cell") from exc
    if not isinstance(value, dict) or value.get("schema") != protocol.cell_schema:
        raise ValueError(f"invalid {backend} replay cell schema")
    if value.get("sha256") != _artifact_digest(value):
        raise ValueError(f"invalid {backend} replay cell digest")
    checks = {
        "promotion": value.get("promotion_value") == PROMOTION_VALUE,
        "diagnostic": value.get("diagnostic_only") is True,
        "verdict": value.get("scientific_verdict") is None,
        "backend": value.get("backend") == backend,
        "spec": value.get("spec_sha256") == protocol.spec_sha256,
        "steps": value.get("simulation_steps_executed") == TOTAL_STEPS,
        "instrument": value.get("instrument_valid") is True,
        "rng": value.get("measured_replay_rng", {}).get("allowed") is False
        and value.get("measured_replay_rng", {}).get("seed") is None
        and value.get("measured_replay_rng", {}).get("calls_observed") == 0,
        "trajectories": isinstance(value.get("trajectories"), dict)
        and set(value["trajectories"]) == set(TRAJECTORIES),
    }
    if not all(checks.values()):
        raise ValueError(f"{backend} replay cell contract mismatch: {checks}")
    arrays: dict[str, np.ndarray] = {}
    for name in TRAJECTORIES:
        arrays[name] = transplant._decode_array(value["trajectories"][name], name)
        expected_dtype = np.bool_ if name == "spikes" else np.float32
        if arrays[name].shape != (TOTAL_STEPS, 60) or arrays[name].dtype != expected_dtype:
            raise ValueError(f"invalid {backend} {name} trajectory shape or dtype")
        if value.get("trajectory_sha256", {}).get(name) != value["trajectories"][name][
            "sha256"
        ]:
            raise ValueError(f"invalid {backend} {name} trajectory digest")
        if value.get("trajectory_step_sha256", {}).get(name) != (
            transplant._trajectory_step_hashes(arrays[name])
        ):
            raise ValueError(f"invalid {backend} {name} per-step digest")
    try:
        receipt = execution_receipt.verify_receipt(ROOT, receipt_path.relative_to(ROOT))
    except (ValueError, execution_receipt.ReceiptError) as exc:
        raise ValueError(f"invalid {backend} replay receipt") from exc
    argv = receipt.get("argv")
    if (
        not isinstance(argv, list)
        or not argv
        or not Path(argv[0]).is_absolute()
        or argv != _expected_cell_argv(
            root=ROOT, artifact=path, backend=backend, source=receipt["source"],
            python=argv[0], protocol=protocol,
        )
    ):
        raise ValueError(f"{backend} replay receipt command differs from the frozen run command")
    if (
        receipt["artifact"]["path"] != path.relative_to(ROOT).as_posix()
        or receipt["artifact"]["sha256"] != _digest_file(path)
        or receipt["env_allowlist"] != {"SIM_BACKEND": backend}
        or receipt["source"] != value.get("source")
    ):
        raise ValueError(f"{backend} replay receipt does not bind its cell")
    value["_decoded_arrays"] = arrays
    value["_receipt_path"] = receipt_path.relative_to(ROOT).as_posix()
    value["_file_sha256"] = _digest_file(path)
    return value


def _first_difference(left: np.ndarray, right: np.ndarray) -> dict[str, Any] | None:
    if left.shape != right.shape:
        return {"kind": "shape", "left": list(left.shape), "right": list(right.shape)}
    if left.dtype != right.dtype:
        return {"kind": "dtype", "left": left.dtype.str, "right": right.dtype.str}
    unequal = np.argwhere(left != right)
    if unequal.size == 0:
        return None
    step, cell = (int(item) for item in unequal[0])
    left_value = left[step, cell]
    right_value = right[step, cell]
    return {
        "kind": "value", "step_index": step, "step_number": step + 1, "cell": cell,
        "numpy": bool(left_value) if left.dtype == np.bool_ else float(left_value),
        "cupy": bool(right_value) if right.dtype == np.bool_ else float(right_value),
    }


def compare_cells(
    *, numpy_artifact: Path, numpy_receipt: Path, cupy_artifact: Path,
    cupy_receipt: Path, out: Path, spec_path: Path | None = None,
    spec_sha256: str | None = None, protocol: ReplayProtocol = V1_PROTOCOL,
) -> dict[str, Any]:
    spec_sha256 = protocol.spec_sha256 if spec_sha256 is None else spec_sha256
    for path, label in (
        (numpy_artifact, "NumPy artifact"),
        (numpy_receipt, "NumPy receipt"),
        (cupy_artifact, "CuPy artifact"),
        (cupy_receipt, "CuPy receipt"),
        (out, "comparison output"),
    ):
        _require_protocol_output(path, protocol, label)
    load_locked_spec(spec_path, spec_sha256, protocol=protocol)
    if os.path.lexists(out):
        raise FileExistsError(f"output artifact already exists: {out}")
    cells = {
        "numpy": _load_cell(
            numpy_artifact, numpy_receipt, "numpy", protocol=protocol,
        ),
        "cupy": _load_cell(
            cupy_artifact, cupy_receipt, "cupy", protocol=protocol,
        ),
    }
    if cells["numpy"]["source"] != cells["cupy"]["source"]:
        raise ValueError("replay cells use different source snapshots")
    if cells["numpy"]["completed_input"] != cells["cupy"]["completed_input"]:
        raise ValueError("replay cells use different sealed input")
    comparisons: dict[str, Any] = {}
    for name in TRAJECTORIES:
        left = cells["numpy"]["_decoded_arrays"][name]
        right = cells["cupy"]["_decoded_arrays"][name]
        first = _first_difference(left, right)
        rows = [
            left_hash == right_hash
            for left_hash, right_hash in zip(
                cells["numpy"]["trajectory_step_sha256"][name],
                cells["cupy"]["trajectory_step_sha256"][name], strict=True,
            )
        ]
        comparisons[name] = {
            "shape_exact": left.shape == right.shape,
            "dtype_exact": left.dtype == right.dtype,
            "bytes_exact": left.tobytes(order="C") == right.tobytes(order="C"),
            "all_1200_rows_exact": len(rows) == TOTAL_STEPS and all(rows),
            "first_difference": first,
        }
        comparisons[name]["exact"] = all((
            comparisons[name]["shape_exact"], comparisons[name]["dtype_exact"],
            comparisons[name]["bytes_exact"], comparisons[name]["all_1200_rows_exact"],
            first is None,
        ))
    passed = all(row["exact"] for row in comparisons.values())
    artifact = _seal({
        "schema": protocol.comparison_schema,
        "promotion_value": PROMOTION_VALUE,
        "diagnostic_only": True,
        "scientific_verdict": None,
        "spec_sha256": spec_sha256,
        "source": cells["numpy"]["source"],
        "completed_input": cells["numpy"]["completed_input"],
        "simulation_steps_compared": {"numpy": TOTAL_STEPS, "cupy": TOTAL_STEPS},
        "cell_artifacts": {
            backend: {
                "path": artifact.relative_to(ROOT).as_posix(),
                "file_sha256": cells[backend]["_file_sha256"],
                "receipt_path": receipt.relative_to(ROOT).as_posix(),
                "artifact_sha256": cells[backend]["sha256"],
            }
            for backend, artifact, receipt in (
                ("numpy", numpy_artifact, numpy_receipt),
                ("cupy", cupy_artifact, cupy_receipt),
            )
        },
        "trajectory_comparisons": comparisons,
        "all_required_trajectories_exact": passed,
        "outcome": "DIAGNOSTIC_PASS" if passed else "DIAGNOSTIC_FAIL",
    })
    _write_new_json(out, artifact)
    return artifact


def _parser(protocol: ReplayProtocol = V1_PROTOCOL) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--run", action="store_true")
    action.add_argument("--compare", action="store_true")
    parser.add_argument("--backend", choices=BACKENDS)
    parser.add_argument(
        "--spec", type=Path, default=ROOT / protocol.spec_relative_path,
    )
    parser.add_argument("--spec-sha256", default=protocol.spec_sha256)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--source-revision")
    parser.add_argument("--numpy-artifact", type=Path)
    parser.add_argument("--numpy-receipt", type=Path)
    parser.add_argument("--cupy-artifact", type=Path)
    parser.add_argument("--cupy-receipt", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(
    argv: list[str] | None = None, *, protocol: ReplayProtocol = V1_PROTOCOL,
) -> int:
    args = _parser(protocol).parse_args(argv)
    if args.run:
        if args.backend is None or args.source_manifest is None or args.source_revision is None:
            raise SystemExit("--run requires --backend, --source-manifest, and --source-revision")
        if any(value is not None for value in (
            args.numpy_artifact, args.numpy_receipt, args.cupy_artifact, args.cupy_receipt,
        )):
            raise SystemExit("--run does not accept comparison inputs")
        result = run_cell(
            backend=args.backend, out=args.out, source_manifest=args.source_manifest,
            source_revision=args.source_revision, spec_path=args.spec,
            spec_sha256=args.spec_sha256, protocol=protocol,
        )
    else:
        required = (
            args.numpy_artifact, args.numpy_receipt, args.cupy_artifact, args.cupy_receipt,
        )
        if args.backend is not None or args.source_manifest is not None or (
            args.source_revision is not None
        ) or any(value is None for value in required):
            raise SystemExit("--compare requires exactly both artifacts and receipts")
        result = compare_cells(
            numpy_artifact=args.numpy_artifact, numpy_receipt=args.numpy_receipt,
            cupy_artifact=args.cupy_artifact, cupy_receipt=args.cupy_receipt,
            out=args.out, spec_path=args.spec, spec_sha256=args.spec_sha256,
            protocol=protocol,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
