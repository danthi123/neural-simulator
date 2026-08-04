"""Create-only, zero-step backend-neutral Izh initialization diagnostic."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any

import numpy as np

from research.runners import _vocal_action_credit_gate_v13_tonic_output as tonic
from sim.backend import get_backend, synchronize, to_host
from sim.regions import RegionPathway
from tools import execution_receipt
from tools.lab import assert_backend


ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = ROOT / "research/specs/v13_backend_neutral_izh_initialization_diagnostic.json"
SPEC_SHA256 = "a3be9d0d0c0b99bce7e6f14f54a8e6b2ce937e62d72ac05cc680a9a8a7ed3ef4"
LOCKED_SEED = 6_556_023
BACKENDS = ("numpy", "cupy")
SCHEMA_CELL = "v13-backend-neutral-izh-initialization-cell-v1"
SCHEMA_COMPARISON = "v13-backend-neutral-izh-initialization-comparison-v1"
PROMOTION_VALUE = "none"
ARRAYS = (
    "cp_traits",
    "cp_neuron_type_ids",
    "cp_neuron_firing_thresholds",
    "cp_neuron_positions_3d",
    "cp_izh_C",
    "cp_izh_k",
    "cp_izh_vr",
    "cp_izh_vt",
    "cp_izh_vpeak",
    "cp_izh_a",
    "cp_izh_b",
    "cp_izh_c_reset",
    "cp_izh_d_increment",
    "cp_membrane_potential_v",
    "cp_recovery_variable_u",
)
AUTHORITY_SOURCE_PATHS = (
    "research/__init__.py",
    "research/runners/__init__.py",
    "research/runners/_vocal_action_selector_gate.py",
    "research/runners/_vocal_action_credit_gate_v13_tonic_output.py",
    "research/runners/_v13_backend_neutral_izh_initialization_diagnostic.py",
    "research/findings/2026-08-04-neural-vocal-credit-gateB-v13-backend-neutral-izh-initialization-correction-DIAGNOSTIC-PREREGISTRATION.md",
    "research/specs/v13_backend_neutral_izh_initialization_diagnostic.json",
    "research/specs/v13_tonic_output_substrate.json",
    "tools/execution_receipt.py",
    "tools/lab.py",
    "tools/verdict.py",
    "tools/v13_backend_neutral_izh_initialization_evidence.py",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")


def _digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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


def source_paths(root: Path | None = None) -> tuple[str, ...]:
    """Return all local simulator Python plus the exact diagnostic authorities."""
    if root is None:
        root = ROOT
    resolved = root.resolve(strict=True)
    sim_root = resolved / "sim"
    simulator = (
        path.relative_to(resolved).as_posix()
        for path in sim_root.rglob("*.py")
        if path.is_file()
    )
    return tuple(sorted(set(simulator).union(AUTHORITY_SOURCE_PATHS)))


def load_locked_spec(path: Path = SPEC_PATH, expected_sha256: str = SPEC_SHA256) -> dict[str, Any]:
    if _SHA256.fullmatch(expected_sha256) is None:
        raise ValueError("spec SHA-256 must be a lowercase digest")
    raw = path.read_bytes()
    if _digest_bytes(raw) != expected_sha256:
        raise ValueError("locked spec digest mismatch")
    try:
        spec = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("locked spec is not valid JSON") from exc
    checks = {
        "schema": spec.get("schema") == "sim-experiment-spec-v0",
        "id": spec.get("id") == "gateB-v13-backend-neutral-izh-initialization-diagnostic-v1",
        "status": spec.get("status") == "preregistered_not_executed",
        "promotion": spec.get("promotion_value") == PROMOTION_VALUE,
        "seed": spec.get("diagnostic_partition", {}).get("seed") == LOCKED_SEED,
        "seed_role": spec.get("diagnostic_partition", {}).get("role") == "paired_initialization_only",
        "backends": spec.get("paired_backends") == list(BACKENDS),
        "arrays": spec.get("required_exact_arrays") == list(ARRAYS),
        "output": spec.get("output_directory")
        == "research/findings/raw/v13_backend_neutral_izh_initialization_diagnostic",
        "flag": spec.get("correction", {}).get("config_flag")
        == "backend_neutral_izh_initialization",
        "steps": spec.get("acceptance", {}).get("no_simulation_steps") is True,
        "verdict": spec.get("acceptance", {}).get("no_scientific_verdict") is True,
    }
    if not all(checks.values()):
        raise ValueError(f"runner/spec disagreement: {checks}")
    return spec


def _source_snapshot(manifest: Path, revision: str) -> dict[str, Any]:
    if _REVISION.fullmatch(revision) is None:
        raise ValueError("source revision must be a full lowercase Git SHA")
    try:
        relative = manifest.resolve(strict=True).relative_to(ROOT.resolve(strict=True)).as_posix()
    except (OSError, ValueError) as exc:
        raise ValueError("source manifest must be inside the repository") from exc
    snapshot = execution_receipt.verify_source_manifest(ROOT, relative)
    kind = execution_receipt._source_revision(ROOT, revision, snapshot["manifest_sha256"])
    expected = set(source_paths())
    if set(snapshot["files"]) != expected:
        raise ValueError("source manifest does not contain the exact diagnostic source set")
    return {
        "file_count": snapshot["file_count"],
        "git_sha": revision,
        "kind": kind,
        "manifest": snapshot["manifest"],
        "manifest_sha256": snapshot["manifest_sha256"],
        "tree_sha256": snapshot["tree_sha256"],
    }


def _config_contract(config) -> dict[str, Any]:
    complete = config.to_dict()
    return {
        "sha256": _digest_bytes(_canonical(complete)),
        "complete": complete,
        "locked_summary": _summary_from_complete(complete),
    }


def _summary_from_complete(complete: dict[str, Any]) -> dict[str, Any]:
    regions = complete.get("brain_regions")
    if not isinstance(regions, list) or len(regions) != 2 or not all(
        isinstance(region, dict) for region in regions
    ):
        raise ValueError("complete config does not contain the frozen two-region population")
    return {
        "backend_neutral_izh_initialization": complete.get(
            "backend_neutral_izh_initialization"
        ),
        "brain_regions": [
            {
                "name": region.get("name"),
                "n_neurons": region.get("n_neurons"),
                "izh_neuron_type": region.get("izh_neuron_type"),
                "intrinsic_current_pA": region.get("intrinsic_current_pA"),
                "enable_heterogeneity": region.get("enable_heterogeneity"),
            }
            for region in regions
        ],
        "heterogeneity_seed": complete.get("heterogeneity_seed"),
        "neuron_model_type": complete.get("neuron_model_type"),
        "num_neurons": complete.get("num_neurons"),
        "seed": complete.get("seed"),
        "simulation_steps": 0,
    }


def _expected_config_summary() -> dict[str, Any]:
    return {
        "backend_neutral_izh_initialization": True,
        "brain_regions": [
            {
                "name": "inhibitory_source", "n_neurons": 20,
                "izh_neuron_type": tonic.NeuronType.IZH2007_FS_CORTICAL_INTERNEURON.name,
                "intrinsic_current_pA": 0.0, "enable_heterogeneity": False,
            },
            {
                "name": "gpi_snr", "n_neurons": 40,
                "izh_neuron_type": tonic.NeuronType.IZH2007_GPI_OUTPUT.name,
                "intrinsic_current_pA": 100.0, "enable_heterogeneity": True,
            },
        ],
        "heterogeneity_seed": LOCKED_SEED,
        "neuron_model_type": tonic.NeuronModel.IZHIKEVICH.name,
        "num_neurons": 60,
        "seed": LOCKED_SEED,
        "simulation_steps": 0,
    }


def _build_bridge():
    regions = [tonic._source_region(), tonic._gpi_region(40, 100.0)]
    pathway = RegionPathway(
        from_region="inhibitory_source",
        to_region="gpi_snr",
        density=1.0,
        weight_mean=8.0,
        weight_jitter=0.0,
        plastic=False,
        receptor="gaba_a",
    )
    config = tonic._read_only_config(LOCKED_SEED, regions, [pathway])
    config.backend_neutral_izh_initialization = True
    bridge = tonic._new_bridge(config)
    tonic._zero_construction_edges(bridge)
    return bridge


def _encode_array(value: Any) -> dict[str, Any]:
    array = np.ascontiguousarray(np.asarray(to_host(value)))
    payload = array.tobytes(order="C")
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "bytes_base64": base64.b64encode(payload).decode("ascii"),
        "bytes_sha256": _digest_bytes(payload),
    }


def _validate_array_record(record: Any, name: str) -> bytes:
    if not isinstance(record, dict) or set(record) != {
        "dtype", "shape", "bytes_base64", "bytes_sha256"
    }:
        raise ValueError(f"invalid array record: {name}")
    try:
        dtype = np.dtype(record["dtype"])
        shape = tuple(record["shape"])
        payload = base64.b64decode(record["bytes_base64"], validate=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid encoded array: {name}") from exc
    if any(type(item) is not int or item < 0 for item in shape):
        raise ValueError(f"invalid array shape: {name}")
    expected_size = int(dtype.itemsize * np.prod(shape, dtype=np.int64))
    if len(payload) != expected_size or _digest_bytes(payload) != record["bytes_sha256"]:
        raise ValueError(f"array bytes do not match metadata: {name}")
    return payload


def capture(
    *, backend: str, out: Path, source_manifest: Path, source_revision: str,
    spec_path: Path = SPEC_PATH, spec_sha256: str = SPEC_SHA256,
) -> dict[str, Any]:
    if backend not in BACKENDS:
        raise ValueError("backend must be numpy or cupy")
    load_locked_spec(spec_path, spec_sha256)
    source = _source_snapshot(source_manifest, source_revision)
    assert_backend(backend, note="V13 backend-neutral initialization diagnostic")
    _, actual_backend = get_backend()
    if actual_backend != backend or os.environ.get("SIM_BACKEND") != backend:
        raise ValueError("active backend does not match the requested diagnostic cell")
    if os.path.lexists(out):
        raise FileExistsError(f"output artifact already exists: {out}")

    bridge = _build_bridge()
    try:
        before = {
            "current_time_ms": float(bridge.runtime_state.current_time_ms),
            "current_time_step": int(bridge.runtime_state.current_time_step),
        }
        if before != {"current_time_ms": 0.0, "current_time_step": 0}:
            raise ValueError("bridge was not captured at zero simulation steps")
        arrays: dict[str, Any] = {}
        for name in ARRAYS:
            value = getattr(bridge, name, None)
            if value is None:
                raise ValueError(f"required initialization array is missing: {name}")
            arrays[name] = _encode_array(value)
        synchronize()
        after = {
            "current_time_ms": float(bridge.runtime_state.current_time_ms),
            "current_time_step": int(bridge.runtime_state.current_time_step),
        }
        if after != before:
            raise ValueError("simulation state advanced during initialization capture")
        contract = _config_contract(bridge.core_config)
        summary = contract["locked_summary"]
        if summary != _expected_config_summary():
            raise ValueError("constructed configuration differs from the frozen contract")
        artifact = _seal({
            "schema": SCHEMA_CELL,
            "promotion_value": PROMOTION_VALUE,
            "diagnostic_only": True,
            "backend": backend,
            "seed": LOCKED_SEED,
            "spec_sha256": spec_sha256,
            "source": source,
            "config": contract,
            "runtime_state_before_capture": before,
            "runtime_state_after_capture": after,
            "simulation_steps_executed": 0,
            "arrays": arrays,
        })
        _write_new_json(out, artifact)
        return artifact
    finally:
        bridge.clear_simulation_state_and_gpu_memory()


def _load_cell(path: Path, expected_backend: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {expected_backend} cell artifact") from exc
    if not isinstance(value, dict) or value.get("schema") != SCHEMA_CELL:
        raise ValueError(f"invalid {expected_backend} cell schema")
    if value.get("sha256") != _artifact_digest(value):
        raise ValueError(f"invalid {expected_backend} cell digest")
    checks = {
        "promotion": value.get("promotion_value") == PROMOTION_VALUE,
        "diagnostic": value.get("diagnostic_only") is True,
        "backend": value.get("backend") == expected_backend,
        "seed": value.get("seed") == LOCKED_SEED,
        "spec": value.get("spec_sha256") == SPEC_SHA256,
        "steps": value.get("simulation_steps_executed") == 0,
        "before": value.get("runtime_state_before_capture")
        == {"current_time_ms": 0.0, "current_time_step": 0},
        "after": value.get("runtime_state_after_capture")
        == {"current_time_ms": 0.0, "current_time_step": 0},
        "arrays": isinstance(value.get("arrays"), dict)
        and set(value["arrays"]) == set(ARRAYS),
    }
    if not all(checks.values()):
        raise ValueError(f"{expected_backend} cell contract mismatch: {checks}")
    source = value.get("source")
    if (
        not isinstance(source, dict)
        or set(source) != {
            "file_count", "git_sha", "kind", "manifest", "manifest_sha256",
            "tree_sha256",
        }
        or _REVISION.fullmatch(source.get("git_sha", "")) is None
        or source.get("kind") not in {"git", "git_archive"}
        or type(source.get("file_count")) is not int
        or source["file_count"] != len(source_paths())
        or any(_SHA256.fullmatch(source.get(name, "")) is None for name in (
            "manifest_sha256", "tree_sha256",
        ))
    ):
        raise ValueError(f"invalid {expected_backend} cell source identity")
    config = value.get("config")
    if (
        not isinstance(config, dict)
        or set(config) != {"sha256", "complete", "locked_summary"}
        or not isinstance(config.get("complete"), dict)
        or config.get("sha256") != _digest_bytes(_canonical(config["complete"]))
        or config.get("locked_summary") != _summary_from_complete(config["complete"])
        or config.get("locked_summary") != _expected_config_summary()
    ):
        raise ValueError(f"invalid {expected_backend} cell config identity")
    for name in ARRAYS:
        _validate_array_record(value["arrays"][name], name)
    return value


def _receipt_for_cell(path: Path, artifact_path: Path, cell: dict[str, Any], backend: str) -> dict[str, Any]:
    try:
        relative_receipt = path.resolve(strict=True).relative_to(ROOT.resolve(strict=True)).as_posix()
        relative_artifact = artifact_path.resolve(strict=True).relative_to(ROOT.resolve(strict=True)).as_posix()
    except (OSError, ValueError) as exc:
        raise ValueError("cell artifacts and receipts must be inside the repository") from exc
    receipt = execution_receipt.verify_receipt(ROOT, relative_receipt)
    if receipt["artifact"]["path"] != relative_artifact:
        raise ValueError(f"{backend} receipt names a different artifact")
    if receipt["artifact"]["sha256"] != _digest_bytes(artifact_path.read_bytes()):
        raise ValueError(f"{backend} receipt artifact digest mismatch")
    if receipt["env_allowlist"] != {"SIM_BACKEND": backend}:
        raise ValueError(f"{backend} receipt environment mismatch")
    source = cell.get("source")
    receipt_source = receipt.get("source", {})
    for key in ("git_sha", "kind", "manifest", "manifest_sha256", "tree_sha256", "file_count"):
        if source.get(key) != receipt_source.get(key):
            raise ValueError(f"{backend} cell source differs from its receipt")
    argv = receipt.get("argv")
    if not isinstance(argv, list) or not argv or not Path(argv[0]).is_absolute():
        raise ValueError(f"{backend} receipt command is invalid")
    expected_argv = [
        argv[0], "-m",
        "research.runners._v13_backend_neutral_izh_initialization_diagnostic",
        "--spec", str(SPEC_PATH.resolve()),
        "--spec-sha256", SPEC_SHA256,
        "--capture", "--backend", backend,
        "--source-manifest", str((ROOT / source["manifest"]).resolve()),
        "--source-revision", source["git_sha"],
        "--out", str(artifact_path.resolve()),
    ]
    if argv != expected_argv:
        raise ValueError(f"{backend} receipt command differs from the frozen capture command")
    return receipt


def compare(
    *, numpy_artifact: Path, numpy_receipt: Path, cupy_artifact: Path,
    cupy_receipt: Path, out: Path,
) -> dict[str, Any]:
    if os.path.lexists(out):
        raise FileExistsError(f"output artifact already exists: {out}")
    numpy_cell = _load_cell(numpy_artifact, "numpy")
    cupy_cell = _load_cell(cupy_artifact, "cupy")
    receipts = {
        "numpy": _receipt_for_cell(numpy_receipt, numpy_artifact, numpy_cell, "numpy"),
        "cupy": _receipt_for_cell(cupy_receipt, cupy_artifact, cupy_cell, "cupy"),
    }
    if numpy_cell["source"] != cupy_cell["source"]:
        raise ValueError("backend cells used different source identities")
    if numpy_cell["config"] != cupy_cell["config"]:
        raise ValueError("backend cells used different configuration identities")

    comparisons: dict[str, Any] = {}
    for name in ARRAYS:
        left = numpy_cell["arrays"][name]
        right = cupy_cell["arrays"][name]
        dtype_equal = left["dtype"] == right["dtype"]
        shape_equal = left["shape"] == right["shape"]
        bytes_equal = _validate_array_record(left, name) == _validate_array_record(right, name)
        comparisons[name] = {
            "dtype_equal": dtype_equal,
            "shape_equal": shape_equal,
            "bytes_equal": bytes_equal,
            "exact": dtype_equal and shape_equal and bytes_equal,
        }
    passed = all(record["exact"] for record in comparisons.values())
    artifact = _seal({
        "schema": SCHEMA_COMPARISON,
        "promotion_value": PROMOTION_VALUE,
        "diagnostic_only": True,
        "scientific_verdict": None,
        "seed": LOCKED_SEED,
        "spec_sha256": SPEC_SHA256,
        "source": numpy_cell["source"],
        "config_sha256": numpy_cell["config"]["sha256"],
        "simulation_steps_executed": {"numpy": 0, "cupy": 0},
        "cell_artifacts": {
            backend: {
                "path": str(path.resolve().relative_to(ROOT.resolve())),
                "sha256": _digest_bytes(path.read_bytes()),
                "receipt_path": str(receipt_path.resolve().relative_to(ROOT.resolve())),
                "receipt_sha256": _digest_bytes(receipt_path.read_bytes()),
                "host": receipts[backend]["host"],
                "device": receipts[backend]["device"],
            }
            for backend, path, receipt_path in (
                ("numpy", numpy_artifact, numpy_receipt),
                ("cupy", cupy_artifact, cupy_receipt),
            )
        },
        "array_comparisons": comparisons,
        "all_required_arrays_exact": passed,
        "outcome": "DIAGNOSTIC_PASS" if passed else "DIAGNOSTIC_FAIL",
    })
    _write_new_json(out, artifact)
    return artifact


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=SPEC_PATH)
    parser.add_argument("--spec-sha256", default=SPEC_SHA256)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--capture", action="store_true")
    modes.add_argument("--compare", action="store_true")
    parser.add_argument("--backend", choices=BACKENDS)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--source-revision")
    parser.add_argument("--numpy-artifact", type=Path)
    parser.add_argument("--numpy-receipt", type=Path)
    parser.add_argument("--cupy-artifact", type=Path)
    parser.add_argument("--cupy-receipt", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.capture:
        if args.backend is None or args.source_manifest is None or args.source_revision is None:
            raise SystemExit("--capture requires --backend, --source-manifest, and --source-revision")
        if any(value is not None for value in (
            args.numpy_artifact, args.numpy_receipt, args.cupy_artifact, args.cupy_receipt,
        )):
            raise SystemExit("--capture does not accept comparison inputs")
        artifact = capture(
            backend=args.backend, out=args.out, source_manifest=args.source_manifest,
            source_revision=args.source_revision, spec_path=args.spec,
            spec_sha256=args.spec_sha256,
        )
    else:
        required = (
            args.numpy_artifact, args.numpy_receipt, args.cupy_artifact, args.cupy_receipt,
        )
        if any(value is None for value in required) or any(value is not None for value in (
            args.backend, args.source_manifest, args.source_revision,
        )):
            raise SystemExit("--compare requires exactly both backend artifacts and receipts")
        artifact = compare(
            numpy_artifact=args.numpy_artifact, numpy_receipt=args.numpy_receipt,
            cupy_artifact=args.cupy_artifact, cupy_receipt=args.cupy_receipt,
            out=args.out,
        )
    print(json.dumps({
        "artifact": str(args.out), "schema": artifact["schema"],
        "sha256": artifact["sha256"], "promotion_value": PROMOTION_VALUE,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
