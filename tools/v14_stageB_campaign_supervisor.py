#!/usr/bin/env python3
"""Run one resumable, fail-closed step of the V14 Stage B GPU campaign.

This is deliberately a supervisor, not a research controller.  It consumes an
already sealed campaign, executes at most one existing declaration per call,
and delegates simulation and final engineering triage to the established Stage
B tools.  It does not select candidates, change specifications, or make a
scientific decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim.snr_executable_packet import canonical_bytes
from tools.compact_trace import CompactTraceError, load_compact_trace
from tools.v14_stageB_campaign import (
    CAMPAIGN_SCHEMA,
    GPU_BATCH_RECEIPT_SCHEMA,
    PHASED_CAMPAIGN_SCHEMA,
    PHASED_GPU_BATCH_RECEIPT_SCHEMA,
    _digest,
    _digest_bytes,
    _load_bound_json,
    run_gpu_batch,
)
from tools.v14_stageB_gpu_triage import (
    StageBGPUTriageError,
    triage_gpu_campaign,
)


ROOT = Path(__file__).resolve().parents[1]
SUPERVISOR_SCHEMA = "v14-snr-stageB-campaign-supervisor-v1"
STATE_FILENAME = "supervisor-state.json"
TRIAGE_FILENAME = "triage.json"
ARM_ORDER = (
    "nap_lesion",
    "intact_autonomous",
    "cav2_2_lesion",
    "sk_lesion",
    "hcn_baseline_lesion",
)
_ARM_SET = frozenset(ARM_ORDER)
_RECEIPT_SCHEMAS = frozenset((GPU_BATCH_RECEIPT_SCHEMA, PHASED_GPU_BATCH_RECEIPT_SCHEMA))
_SHA256_HEX = frozenset("0123456789abcdef")
# This is the interpreter named by docs/AUTONOMOUS-EXECUTION.md.  A canonical
# checkout may instead provide its own equivalent .venv below repository_root.
DOCUMENTED_INTERPRETER = Path("/home/dant123/Projects/sim/.venv/bin/python")


class StageBCampaignSupervisorError(ValueError):
    """Raised when a campaign cannot be resumed without risking false evidence."""


def _sha(value: Any, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_HEX for character in value)
    ):
        raise StageBCampaignSupervisorError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _relative_path(root: Path, value: str | Path, context: str, *, require_file: bool = False) -> tuple[str, Path]:
    supplied = Path(value).expanduser()
    path = (supplied if supplied.is_absolute() else root / supplied).resolve()
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise StageBCampaignSupervisorError(f"{context} must be inside repository_root") from exc
    pure = PurePosixPath(relative.as_posix())
    if not pure.parts or any(part in {"", ".", ".."} for part in pure.parts):
        raise StageBCampaignSupervisorError(f"{context} path is not canonical")
    original = supplied if supplied.is_absolute() else root / supplied
    if original.is_symlink():
        raise StageBCampaignSupervisorError(f"{context} cannot be a symlink")
    if require_file and (not path.is_file() or path.is_symlink()):
        raise StageBCampaignSupervisorError(f"{context} must be a regular file")
    return pure.as_posix(), path


def _root(repository_root: str | Path) -> Path:
    root = Path(repository_root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise StageBCampaignSupervisorError("repository_root must be a directory")
    return root


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> str:
    """Publish a mutable state document without exposing a half-written file."""
    raw = canonical_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise StageBCampaignSupervisorError("supervisor state cannot be a symlink")
    with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return _digest_bytes(raw)


def _load_self_digested(path: Path, context: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise StageBCampaignSupervisorError(f"{context} must be a regular file")
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StageBCampaignSupervisorError(f"{context} is unreadable or invalid JSON") from exc
    if not isinstance(value, dict):
        raise StageBCampaignSupervisorError(f"{context} must contain an object")
    if canonical_bytes(value) != raw:
        raise StageBCampaignSupervisorError(f"{context} is not canonical JSON")
    digest = _sha(value.get("sha256"), f"{context} self digest")
    if digest != _digest({key: item for key, item in value.items() if key != "sha256"}):
        raise StageBCampaignSupervisorError(f"{context} self digest is invalid")
    return value


def _sanctioned_environments(root: Path) -> tuple[Path, ...]:
    candidates = [DOCUMENTED_INTERPRETER, root / ".venv" / "bin" / "python"]
    result: list[Path] = []
    for candidate in candidates:
        if candidate.exists():
            environment = candidate.parent.parent.resolve()
            if environment not in result:
                result.append(environment)
    return tuple(result)


def preflight_gpu(*, repository_root: str | Path = ROOT) -> dict[str, Any]:
    """Require the sanctioned project interpreter and a usable CUDA device."""
    root = _root(repository_root)
    executable = Path(sys.executable).expanduser()
    environment = Path(sys.prefix).expanduser().resolve()
    allowed = _sanctioned_environments(root)
    if not allowed or environment not in allowed:
        rendered = ", ".join(str(path) for path in allowed) or "none found"
        raise StageBCampaignSupervisorError(
            f"wrong interpreter environment: {environment} is not one of {rendered}"
        )
    try:
        import cupy as cp

        count = int(cp.cuda.runtime.getDeviceCount())
        if count < 1:
            raise RuntimeError("CuPy reports no CUDA devices")
        device = cp.cuda.Device(0)
        device.use()
        probe = cp.asarray([1], dtype=cp.int32)
        probe.item()
        cp.cuda.Stream.null.synchronize()
        properties = cp.cuda.runtime.getDeviceProperties(0)
        name = properties.get("name", b"unknown") if isinstance(properties, dict) else "unknown"
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="replace")
    except Exception as exc:  # normalize driver, import, and runtime failures
        raise StageBCampaignSupervisorError(f"CuPy GPU preflight failed: {exc}") from exc
    return {
        "interpreter": str(executable),
        "environment_prefix": str(environment),
        "python_version": platform.python_version(),
        "cupy_version": str(cp.__version__),
        "cuda_device_count": count,
        "cuda_device_0": str(name),
    }


def _campaign(root: Path, campaign_path: str | Path, campaign_sha256: str) -> tuple[dict[str, str], dict[str, Any], list[dict[str, Any]]]:
    try:
        reference, document = _load_bound_json(root, campaign_path, campaign_sha256, "campaign manifest")
    except (OSError, TypeError, ValueError) as exc:
        if isinstance(exc, StageBCampaignSupervisorError):
            raise
        raise StageBCampaignSupervisorError(str(exc)) from exc
    body = {key: value for key, value in document.items() if key != "sha256"}
    if (
        document.get("schema") not in {CAMPAIGN_SCHEMA, PHASED_CAMPAIGN_SCHEMA}
        or document.get("sha256") != _digest(body)
        or document.get("status") != "materialized-not-executed"
        or document.get("engineering_screening_only") is not True
        or document.get("scientific_verdict") is not None
        or document.get("numpy_confirmation_required") is not True
        or document.get("arm_count") != len(ARM_ORDER)
        or document.get("candidate_count") != 512
    ):
        raise StageBCampaignSupervisorError("campaign is not an executable sealed V14 Stage B screen")
    declarations = document.get("declarations")
    if not isinstance(declarations, list) or len(declarations) != document.get("batch_count"):
        raise StageBCampaignSupervisorError("campaign declarations are incomplete")
    checked: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for declaration in declarations:
        if not isinstance(declaration, Mapping) or set(declaration) != {
            "arm", "batch_index", "candidate_count", "path", "sha256", "declaration_sha256"
        }:
            raise StageBCampaignSupervisorError("campaign declaration identity is invalid")
        arm = declaration["arm"]
        index = declaration["batch_index"]
        if arm not in _ARM_SET or isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise StageBCampaignSupervisorError("campaign declaration has an invalid arm or batch index")
        key = (arm, index)
        if key in seen:
            raise StageBCampaignSupervisorError("campaign contains duplicate declarations")
        seen.add(key)
        try:
            declaration_reference, declaration_document = _load_bound_json(
                root, declaration["path"], declaration["sha256"], "batch declaration"
            )
        except (OSError, TypeError, ValueError) as exc:
            raise StageBCampaignSupervisorError(str(exc)) from exc
        declaration_body = {key: value for key, value in declaration_document.items() if key != "sha256"}
        candidates = declaration_document.get("candidates")
        if (
            declaration_document.get("sha256") != declaration["declaration_sha256"]
            or declaration_document.get("arm") != arm
            or not isinstance(candidates, list)
            or len(candidates) != declaration["candidate_count"]
            or declaration["candidate_count"] < 1
            or declaration_document.get("schema") is None
            or declaration_document.get("analysis_protocol") is None
            or declaration_document.get("sha256") != _digest(declaration_body)
        ):
            raise StageBCampaignSupervisorError("batch declaration is not self-consistent")
        expected = {}
        for item in candidates:
            if not isinstance(item, Mapping) or not isinstance(item.get("candidate_id"), str):
                raise StageBCampaignSupervisorError("batch declaration contains an invalid candidate")
            candidate_id = item["candidate_id"]
            candidate_sha = item.get("candidate_sha256")
            if candidate_id in expected or not isinstance(candidate_sha, str):
                raise StageBCampaignSupervisorError("batch declaration contains duplicate candidate identity")
            _sha(candidate_sha, "candidate digest")
            expected[candidate_id] = candidate_sha
        checked.append({
            "arm": arm,
            "batch_index": index,
            "candidate_count": declaration["candidate_count"],
            "path": declaration_reference["path"],
            "sha256": declaration["sha256"],
            "declaration_sha256": declaration["declaration_sha256"],
            "expected_candidates": expected,
        })
    if {item["arm"] for item in checked} != _ARM_SET:
        raise StageBCampaignSupervisorError("campaign does not contain exactly the five authorized arms")
    # Each arm must have contiguous, deterministic batch indices.  Candidate
    # selection is never performed here; this only authenticates the sealed rows.
    for arm in ARM_ORDER:
        indices = sorted(item["batch_index"] for item in checked if item["arm"] == arm)
        if indices != list(range(len(indices))):
            raise StageBCampaignSupervisorError(f"{arm} declarations are not contiguous")
    checked.sort(key=lambda item: (ARM_ORDER.index(item["arm"]), item["batch_index"]))
    return reference, document, checked


def _binding(declaration: Mapping[str, Any], *, output_root: str, receipt_sha256: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "arm": declaration["arm"],
        "batch_index": declaration["batch_index"],
        "declaration": {
            "path": declaration["path"],
            "sha256": declaration["sha256"],
            "declaration_sha256": declaration["declaration_sha256"],
        },
    }
    if receipt_sha256 is not None:
        result["receipt"] = {
            "path": f"{output_root}/{declaration['arm']}/batch-{declaration['batch_index']:03d}/receipt.json",
            "sha256": receipt_sha256,
        }
    return result


def _receipt_error(message: str) -> StageBCampaignSupervisorError:
    return StageBCampaignSupervisorError(f"GPU receipt is invalid: {message}")


def _validate_receipt(
    *, root: Path, output: Path, campaign_reference: Mapping[str, str], campaign: Mapping[str, Any], declaration: Mapping[str, Any]
) -> tuple[dict[str, Any], str]:
    batch_dir = output / declaration["arm"] / f"batch-{declaration['batch_index']:03d}"
    receipt_path = batch_dir / "receipt.json"
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise _receipt_error("receipt.json is missing")
    try:
        raw = receipt_path.read_bytes()
        receipt = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _receipt_error(f"receipt.json is unreadable: {exc}") from exc
    if not isinstance(receipt, dict):
        raise _receipt_error("receipt.json must contain an object")
    try:
        if canonical_bytes(receipt) != raw:
            raise _receipt_error("receipt.json is not canonical JSON")
    except (TypeError, ValueError) as exc:
        raise _receipt_error("receipt.json is not canonical JSON") from exc
    receipt_digest = _sha(receipt.get("sha256"), "receipt self digest")
    if _digest({key: value for key, value in receipt.items() if key != "sha256"}) != receipt_digest:
        raise _receipt_error("receipt self digest does not match")
    expected_declaration = {
        "path": declaration["path"],
        "sha256": declaration["sha256"],
        "declaration_sha256": declaration["declaration_sha256"],
    }
    expected_schema = PHASED_GPU_BATCH_RECEIPT_SCHEMA if campaign.get("schema") == PHASED_CAMPAIGN_SCHEMA else GPU_BATCH_RECEIPT_SCHEMA
    expected_receipt_keys = {
        "schema", "process_status", "engineering_screening_only", "scientific_verdict",
        "numpy_confirmation_required", "campaign", "declaration", "arm", "batch_index",
        "execution", "provenance", "traces", "sha256",
    }
    if set(receipt) != expected_receipt_keys:
        raise _receipt_error("receipt schema has unexpected fields")
    if (
        receipt.get("schema") != expected_schema
        or receipt.get("process_status") != "completed"
        or receipt.get("engineering_screening_only") is not True
        or receipt.get("scientific_verdict") is not None
        or receipt.get("numpy_confirmation_required") is not True
        or receipt.get("campaign") != dict(campaign_reference)
        or receipt.get("declaration") != expected_declaration
        or receipt.get("arm") != declaration["arm"]
        or receipt.get("batch_index") != declaration["batch_index"]
        or not isinstance(receipt.get("traces"), list)
    ):
        raise _receipt_error("receipt identity or scientific boundary changed")
    expected_candidates = declaration["expected_candidates"]
    observed: dict[str, str] = {}
    trace_paths: set[Path] = set()
    for trace in receipt["traces"]:
        if not isinstance(trace, Mapping):
            raise _receipt_error("trace entry is not an object")
        expected_trace_keys = {
            "candidate_id", "candidate_sha256", "termination", "compact_trace",
        }
        if campaign.get("schema") == PHASED_CAMPAIGN_SCHEMA:
            expected_trace_keys.add("runtime_intervention")
        if set(trace) != expected_trace_keys:
            raise _receipt_error("trace schema has unexpected fields")
        candidate_id = trace.get("candidate_id")
        candidate_sha = trace.get("candidate_sha256")
        compact = trace.get("compact_trace")
        if candidate_id not in expected_candidates or candidate_id in observed or candidate_sha != expected_candidates[candidate_id] or not isinstance(compact, Mapping):
            raise _receipt_error("trace candidate identity does not match declaration")
        archive_value = compact.get("path")
        try:
            archive_relative, archive = _relative_path(root, archive_value, "compact trace", require_file=True)
        except (OSError, TypeError, ValueError) as exc:
            raise _receipt_error(str(exc)) from exc
        if archive.parent != batch_dir.resolve() or archive.name == "receipt.json" or archive in trace_paths:
            raise _receipt_error("compact trace is outside its batch output")
        expected_archive_sha = _sha(compact.get("sha256"), "compact trace digest")
        try:
            arrays = load_compact_trace(archive, expected_sha256=expected_archive_sha)
        except (CompactTraceError, OSError, TypeError, ValueError) as exc:
            raise _receipt_error(f"compact trace authentication failed: {exc}") from exc
        sample_count = compact.get("sample_count")
        if set(compact) != {"path", "sha256", "sample_count"}:
            raise _receipt_error("compact trace binding has unexpected fields")
        if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count != len(arrays["time"]):
            raise _receipt_error("compact trace sample count does not match archive")
        observed[candidate_id] = candidate_sha
        trace_paths.add(archive)
        # Keep the normalized path check explicit so a platform-specific path
        # cannot be smuggled into a repository-relative receipt.
        if archive_relative != archive.relative_to(root).as_posix():
            raise _receipt_error("compact trace path is not canonical")
    if observed != expected_candidates:
        raise _receipt_error("receipt does not exactly cover its declaration")
    if not batch_dir.is_dir() or batch_dir.is_symlink():
        raise _receipt_error("batch output is not a regular directory")
    expected_files = {receipt_path, *trace_paths}
    children = list(batch_dir.iterdir())
    actual_files = {item for item in children if item.is_file()}
    if any(item.is_symlink() or not item.is_file() for item in children) or actual_files != expected_files:
        raise _receipt_error("batch output contains partial or unexpected files")
    return receipt, _digest_bytes(raw)


def _discover(
    *, root: Path, output: Path, campaign_reference: Mapping[str, str], campaign: Mapping[str, Any], declarations: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return valid receipt bindings and declarations still missing receipts."""
    if output.exists() and (output.is_symlink() or not output.is_dir()):
        raise StageBCampaignSupervisorError("output_root is not a regular directory")
    receipts: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    expected_dirs = {arm for arm in ARM_ORDER}
    if output.exists():
        allowed_logs = {
            f"{item['arm']}-batch-{item['batch_index']:03d}.run.log"
            for item in declarations
        }
        for child in output.iterdir():
            if child.name in {STATE_FILENAME, TRIAGE_FILENAME}:
                if not child.is_file() or child.is_symlink():
                    raise StageBCampaignSupervisorError("output root contains a corrupt control file")
                continue
            if child.name in allowed_logs:
                if not child.is_file() or child.is_symlink():
                    raise StageBCampaignSupervisorError("output root contains a corrupt run log")
                continue
            if child.name not in expected_dirs or not child.is_dir() or child.is_symlink():
                raise StageBCampaignSupervisorError("output root contains an unexpected or partial output")
        triage_path = output / TRIAGE_FILENAME
        if triage_path.exists():
            _load_self_digested(triage_path, "triage output")
    for declaration in declarations:
        batch_dir = output / declaration["arm"] / f"batch-{declaration['batch_index']:03d}"
        receipt_path = batch_dir / "receipt.json"
        if receipt_path.exists() or batch_dir.exists():
            if not receipt_path.is_file() or receipt_path.is_symlink():
                raise StageBCampaignSupervisorError("partial batch output has no valid receipt")
            receipt, digest = _validate_receipt(
                root=root,
                output=output,
                campaign_reference=campaign_reference,
                campaign=campaign,
                declaration=declaration,
            )
            receipts.append(_binding(declaration, output_root=output.relative_to(root).as_posix(), receipt_sha256=digest))
        else:
            missing.append(dict(declaration))
    for arm in ARM_ORDER:
        arm_dir = output / arm
        if not arm_dir.exists():
            continue
        expected_batch_dirs = {
            f"batch-{item['batch_index']:03d}"
            for item in declarations
            if item["arm"] == arm
        }
        actual_batch_dirs = {child.name for child in arm_dir.iterdir()}
        if not actual_batch_dirs.issubset(expected_batch_dirs):
            raise StageBCampaignSupervisorError("output contains an unexpected or partial batch directory")
    return receipts, missing


def _state_body(
    *, campaign_reference: Mapping[str, str], output_relative: str, environment: Mapping[str, Any] | None,
    completed: Sequence[Mapping[str, Any]], in_flight: Mapping[str, Any] | None, status: str,
    triage: Mapping[str, Any] | None,
) -> dict[str, Any]:
    body = {
        "schema": SUPERVISOR_SCHEMA,
        "campaign": dict(campaign_reference),
        "output_root": output_relative,
        "environment": dict(environment) if environment is not None else None,
        "status": status,
        "completed": [dict(item) for item in completed],
        "in_flight": dict(in_flight) if in_flight is not None else None,
        "triage": dict(triage) if triage is not None else None,
    }
    return {**body, "sha256": _digest(body)}


def _validate_state(
    state: Mapping[str, Any], *, campaign_reference: Mapping[str, str], output_relative: str,
) -> dict[str, Any]:
    expected = {"schema", "campaign", "output_root", "environment", "status", "completed", "in_flight", "triage", "sha256"}
    if set(state) != expected or state.get("schema") != SUPERVISOR_SCHEMA:
        raise StageBCampaignSupervisorError("supervisor state schema is invalid")
    if state.get("campaign") != dict(campaign_reference) or state.get("output_root") != output_relative:
        raise StageBCampaignSupervisorError("supervisor state is bound to a different campaign or output root")
    if state.get("status") not in {"ready", "in-flight", "complete"}:
        raise StageBCampaignSupervisorError("supervisor state status is invalid")
    if state.get("environment") is not None and not isinstance(state.get("environment"), Mapping):
        raise StageBCampaignSupervisorError("supervisor state environment is invalid")
    if not isinstance(state.get("completed"), list) or not all(isinstance(item, Mapping) for item in state["completed"]):
        raise StageBCampaignSupervisorError("supervisor state completed list is invalid")
    if state.get("in_flight") is not None and not isinstance(state.get("in_flight"), Mapping):
        raise StageBCampaignSupervisorError("supervisor state in-flight record is invalid")
    if state.get("triage") is not None and not isinstance(state.get("triage"), Mapping):
        raise StageBCampaignSupervisorError("supervisor state triage record is invalid")
    digest = _sha(state.get("sha256"), "supervisor state self digest")
    if digest != _digest({key: value for key, value in state.items() if key != "sha256"}):
        raise StageBCampaignSupervisorError("supervisor state self digest is invalid")
    return dict(state)


def _triage_output(
    *, root: Path, output: Path, campaign_path: str | Path, campaign_sha256: str,
) -> dict[str, Any]:
    triage_path = output / TRIAGE_FILENAME
    if triage_path.exists():
        if triage_path.is_symlink() or not triage_path.is_file():
            raise StageBCampaignSupervisorError("triage output is corrupt")
        existing = _load_self_digested(triage_path, "triage output")
    try:
        result = triage_gpu_campaign(
            campaign_path,
            campaign_sha256,
            output,
            repository_root=root,
        )
    except (OSError, TypeError, ValueError, StageBGPUTriageError) as exc:
        raise StageBCampaignSupervisorError(f"strict Stage B triage failed: {exc}") from exc
    if triage_path.exists():
        if existing != result:
            raise StageBCampaignSupervisorError("existing triage output changed identity")
    else:
        _write_json_atomic(triage_path, result)
    return result


def _status_report(
    *, campaign_reference: Mapping[str, str], output_relative: str, receipts: Sequence[Mapping[str, Any]], missing: Sequence[Mapping[str, Any]], state: Mapping[str, Any] | None, environment: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    next_item = missing[0] if missing else None
    return {
        "schema": SUPERVISOR_SCHEMA,
        "mode": "status",
        "campaign": dict(campaign_reference),
        "output_root": output_relative,
        "state_present": state is not None,
        "state_status": state.get("status") if state is not None else None,
        "completed_count": len(receipts),
        "remaining_count": len(missing),
        "next": ({"arm": next_item["arm"], "batch_index": next_item["batch_index"]} if next_item else None),
        "arm_order": list(ARM_ORDER),
        "preflight": environment,
        "engineering_screening_only": True,
        "scientific_selection_performed": False,
    }


def supervise_campaign(
    campaign_path: str | Path,
    campaign_sha256: str,
    output_root: str | Path,
    *,
    repository_root: str | Path = ROOT,
    chunk_steps: int = 4096,
    dry_run: bool = False,
    status: bool = False,
) -> dict[str, Any]:
    """Inspect or advance one exact Stage B campaign by at most one GPU batch."""
    if dry_run and status:
        raise StageBCampaignSupervisorError("choose only one read-only mode")
    root = _root(repository_root)
    campaign_reference, campaign, declarations = _campaign(root, campaign_path, campaign_sha256)
    output_relative, output = _relative_path(root, output_root, "output_root")
    if output == root:
        raise StageBCampaignSupervisorError("output_root must not be repository_root itself")
    state_path = output / STATE_FILENAME
    state: dict[str, Any] | None = None
    if state_path.exists():
        state = _validate_state(
            _load_self_digested(state_path, "supervisor state"),
            campaign_reference=campaign_reference,
            output_relative=output_relative,
        )
    receipts, missing = _discover(
        root=root,
        output=output,
        campaign_reference=campaign_reference,
        campaign=campaign,
        declarations=declarations,
    )
    if state is not None:
        completed = state.get("completed", [])
        inflight = state.get("in_flight")
        if inflight is not None:
            matching_declaration = next(
                (
                    item for item in declarations
                    if item["arm"] == inflight.get("arm")
                    and item["batch_index"] == inflight.get("batch_index")
                ),
                None,
            )
            if matching_declaration is None or dict(inflight) != _binding(
                matching_declaration, output_root=output_relative
            ):
                raise StageBCampaignSupervisorError("supervisor state in-flight identity is invalid")
        def record_key(item: Mapping[str, Any], context: str) -> tuple[str, int]:
            arm = item.get("arm")
            index = item.get("batch_index")
            if arm not in _ARM_SET or isinstance(index, bool) or not isinstance(index, int) or index < 0:
                raise StageBCampaignSupervisorError(f"{context} identity is invalid")
            return arm, index

        state_records = {record_key(item, "supervisor state completed record"): item for item in completed}
        if len(state_records) != len(completed):
            raise StageBCampaignSupervisorError("supervisor state completed records are duplicated")
        receipt_records = {record_key(item, "discovered receipt"): item for item in receipts}
        if not set(state_records).issubset(receipt_records):
            raise StageBCampaignSupervisorError("supervisor state refers to a removed receipt")
        if any(state_records[key] != receipt_records[key] for key in state_records):
            raise StageBCampaignSupervisorError("supervisor state receipt binding changed")
    if status or dry_run:
        return _status_report(
            campaign_reference=campaign_reference,
            output_relative=output_relative,
            receipts=receipts,
            missing=missing,
            state=state,
        )
    environment = preflight_gpu(repository_root=root)
    if state is not None and state.get("environment") is not None and state["environment"] != environment:
        raise StageBCampaignSupervisorError("execution environment changed since supervisor state was created")
    if not output.exists():
        output.mkdir(parents=True)
    completed = receipts
    if state is None:
        state = _state_body(
            campaign_reference=campaign_reference,
            output_relative=output_relative,
            environment=environment,
            completed=completed,
            in_flight=None,
            status="ready",
            triage=None,
        )
        _write_json_atomic(state_path, state)
    else:
        state = dict(state)
        if state.get("in_flight") is not None and any(
            item["arm"] == state["in_flight"].get("arm") and item["batch_index"] == state["in_flight"].get("batch_index")
            for item in receipts
        ):
            state = _state_body(
                campaign_reference=campaign_reference,
                output_relative=output_relative,
                environment=environment,
                completed=completed,
                in_flight=None,
                status="ready",
                triage=state.get("triage"),
            )
            _write_json_atomic(state_path, state)
    if missing:
        target = missing[0]
        target_binding = _binding(target, output_root=output_relative)
        state = _state_body(
            campaign_reference=campaign_reference,
            output_relative=output_relative,
            environment=environment,
            completed=completed,
            in_flight=target_binding,
            status="in-flight",
            triage=None,
        )
        _write_json_atomic(state_path, state)
        batch_output = output / target["arm"] / f"batch-{target['batch_index']:03d}"
        old_backend = os.environ.get("SIM_BACKEND")
        os.environ["SIM_BACKEND"] = "cupy"
        try:
            try:
                run_gpu_batch(
                    campaign_path,
                    campaign_sha256,
                    target["arm"],
                    target["batch_index"],
                    batch_output,
                    repository_root=root,
                    chunk_steps=chunk_steps,
                )
            except (OSError, TypeError, ValueError) as exc:
                raise StageBCampaignSupervisorError(f"GPU batch execution failed: {exc}") from exc
        finally:
            if old_backend is None:
                os.environ.pop("SIM_BACKEND", None)
            else:
                os.environ["SIM_BACKEND"] = old_backend
        receipts, missing = _discover(
            root=root,
            output=output,
            campaign_reference=campaign_reference,
            campaign=campaign,
            declarations=declarations,
        )
        if not any(item["arm"] == target["arm"] and item["batch_index"] == target["batch_index"] for item in receipts):
            raise StageBCampaignSupervisorError("GPU runner returned without publishing a valid receipt")
    if missing:
        state = _state_body(
            campaign_reference=campaign_reference,
            output_relative=output_relative,
            environment=environment,
            completed=receipts,
            in_flight=None,
            status="ready",
            triage=None,
        )
        _write_json_atomic(state_path, state)
        return {
            **_status_report(
                campaign_reference=campaign_reference,
                output_relative=output_relative,
                receipts=receipts,
                missing=missing,
                state=state,
                environment=environment,
            ),
            "mode": "execute",
            "executed": True,
        }
    triage = _triage_output(
        root=root,
        output=output,
        campaign_path=campaign_path,
        campaign_sha256=campaign_sha256,
    )
    triage_reference = {
        "path": (output / TRIAGE_FILENAME).relative_to(root).as_posix(),
        "sha256": _digest_bytes((output / TRIAGE_FILENAME).read_bytes()),
    }
    state = _state_body(
        campaign_reference=campaign_reference,
        output_relative=output_relative,
        environment=environment,
        completed=receipts,
        in_flight=None,
        status="complete",
        triage=triage_reference,
    )
    _write_json_atomic(state_path, state)
    return {
        **_status_report(
            campaign_reference=campaign_reference,
            output_relative=output_relative,
            receipts=receipts,
            missing=[],
            state=state,
            environment=environment,
        ),
        "mode": "execute",
        "executed": False,
        "triage": triage_reference,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--campaign-sha256", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--repository-root", default=str(ROOT))
    parser.add_argument("--chunk-steps", type=int, default=4096)
    parser.add_argument("--dry-run", action="store_true", help="read-only next-step report")
    parser.add_argument("--status", action="store_true", help="read-only campaign status report")
    args = parser.parse_args(argv)
    try:
        result = supervise_campaign(
            args.campaign,
            args.campaign_sha256,
            args.output_root,
            repository_root=args.repository_root,
            chunk_steps=args.chunk_steps,
            dry_run=args.dry_run,
            status=args.status,
        )
    except (OSError, TypeError, ValueError) as exc:
        parser.exit(2, f"Stage B campaign supervisor failure: {exc}\n")
    print(canonical_bytes(result).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
