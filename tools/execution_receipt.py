#!/usr/bin/env python3
"""Run one command and create a fail-closed receipt for its output artifact."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import stat
import subprocess
import time
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

try:
    from tools.stable_json_evidence import (
        StableJsonEvidenceError,
        read_stable_json_evidence,
    )
except ModuleNotFoundError:  # Direct ``python tools/...`` invocation.
    from stable_json_evidence import (  # type: ignore[no-redef]
        StableJsonEvidenceError,
        read_stable_json_evidence,
    )


SCHEMA = "sim-execution-receipt-v1"
SCHEMA_V2 = "sim-execution-receipt-v2"
PROVENANCE_SCHEMA_V2 = "sim-run-provenance-v2"
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^[0-9a-fA-F]{7,64}$")
_RUN_ID = re.compile(r"^[0-9a-f]{64}$")
_PRIVATE_PROVENANCE_ENV = {
    "SIM_PROVENANCE_V2",
    "SIM_PROVENANCE_RUN_ID",
    "SIM_PROVENANCE_SOURCE_KIND",
    "SIM_PROVENANCE_SOURCE_MANIFEST_SHA256",
}


class ReceiptError(ValueError):
    """Raised when an execution cannot earn a success receipt."""


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReceiptError(f"{field} must be a non-empty string")
    return value


def _safe_relative_path(root: Path, value: str | Path, field: str) -> tuple[str, Path]:
    try:
        text = os.fspath(value)
    except TypeError as exc:
        raise ReceiptError(f"{field} must be a path string") from exc
    if not isinstance(text, str):
        raise ReceiptError(f"{field} must be a path string")
    relative = PurePosixPath(text)
    if (
        not text
        or relative.is_absolute()
        or not relative.name
        or "." in relative.parts
        or ".." in relative.parts
    ):
        raise ReceiptError(f"{field} must be a safe repository-relative path: {text!r}")
    root_resolved = root.resolve(strict=True)
    candidate = root_resolved.joinpath(*relative.parts)
    current = root_resolved
    for part in relative.parts:
        current = current / part
        if not os.path.lexists(current):
            continue
        try:
            mode = current.lstat().st_mode
        except OSError as exc:
            raise ReceiptError(f"cannot inspect {field}: {current}: {exc}") from exc
        if stat.S_ISLNK(mode):
            raise ReceiptError(f"{field} cannot contain a symlink: {text!r}")
    try:
        resolved = candidate.resolve(strict=False)
        resolved.relative_to(root_resolved)
    except (OSError, ValueError) as exc:
        raise ReceiptError(f"{field} escapes execution root: {text!r}") from exc
    return relative.as_posix(), candidate


def _file_state(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _hash_regular_file(path: Path, label: str) -> tuple[str, int, tuple[int, ...]]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ReceiptError(f"cannot open {label} {path}: {exc}") from exc
    try:
        with os.fdopen(descriptor, "rb") as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise ReceiptError(f"{label} is not a regular file: {path}")
            digest = hashlib.sha256()
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
            after = os.fstat(handle.fileno())
    except OSError as exc:
        raise ReceiptError(f"cannot hash {label} {path}: {exc}") from exc
    if _file_state(before) != _file_state(after):
        raise ReceiptError(f"{label} changed while hashing: {path}")
    try:
        named = path.lstat()
    except OSError as exc:
        raise ReceiptError(f"{label} disappeared after hashing: {path}") from exc
    if stat.S_ISLNK(named.st_mode) or _file_state(named) != _file_state(after):
        raise ReceiptError(f"{label} changed while hashing: {path}")
    return digest.hexdigest(), after.st_size, _file_state(after)


def _parse_manifest(data: bytes, manifest: Path) -> dict[str, str]:
    try:
        lines = data.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ReceiptError(f"source manifest is not UTF-8: {manifest}") from exc
    if not lines:
        raise ReceiptError(f"source manifest is empty: {manifest}")
    entries: dict[str, str] = {}
    for line_number, line in enumerate(lines, 1):
        digest, separator, relative_text = line.partition("  ")
        relative = PurePosixPath(relative_text)
        if (
            not separator
            or not _SHA256.fullmatch(digest)
            or relative.is_absolute()
            or not relative.name
            or "." in relative.parts
            or ".." in relative.parts
        ):
            raise ReceiptError(
                f"source manifest has an invalid entry on line {line_number}: {manifest}"
            )
        normalized = relative.as_posix()
        if normalized in entries:
            raise ReceiptError(f"source manifest has a duplicate entry: {normalized}")
        entries[normalized] = digest
    return entries


def verify_source_manifest(root: Path, manifest_path: str | Path) -> dict[str, Any]:
    """Verify every file named by a checksum manifest and return a source snapshot."""
    root = root.resolve(strict=True)
    relative_manifest, manifest = _safe_relative_path(root, manifest_path, "source manifest")
    manifest_hash, manifest_size, manifest_state = _hash_regular_file(
        manifest, "source manifest"
    )
    try:
        manifest_bytes = manifest.read_bytes()
    except OSError as exc:
        raise ReceiptError(f"cannot read source manifest {manifest}: {exc}") from exc
    if hashlib.sha256(manifest_bytes).hexdigest() != manifest_hash:
        raise ReceiptError(f"source manifest changed while reading: {manifest}")
    entries = _parse_manifest(manifest_bytes, manifest)
    files: dict[str, dict[str, Any]] = {}
    tree_digest = hashlib.sha256()
    for relative, expected in sorted(entries.items()):
        normalized, source = _safe_relative_path(root, relative, "source manifest entry")
        actual, size, file_state = _hash_regular_file(source, "source file")
        if actual != expected:
            raise ReceiptError(f"source digest mismatch: {normalized}")
        files[normalized] = {"sha256": actual, "size_bytes": size, "state": file_state}
        tree_digest.update(f"{actual}  {normalized}\n".encode("utf-8"))
    return {
        "manifest": relative_manifest,
        "manifest_sha256": manifest_hash,
        "manifest_size_bytes": manifest_size,
        "manifest_state": manifest_state,
        "tree_sha256": tree_digest.hexdigest(),
        "file_count": len(files),
        "files": files,
    }


def _source_revision(root: Path, expected_git_sha: str, manifest_sha256: str) -> str:
    if not _GIT_SHA.fullmatch(expected_git_sha):
        raise ReceiptError("git SHA must contain 7-64 hexadecimal characters")
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        revision_path = root / ".source_revision"
        try:
            values = dict(
                line.partition("=")[::2]
                for line in revision_path.read_text(encoding="utf-8").splitlines()
                if "=" in line
            )
        except OSError as exc:
            raise ReceiptError(
                "cannot verify Git identity from Git or .source_revision"
            ) from exc
        if values.get("source_kind") != "git_archive":
            raise ReceiptError(".source_revision does not identify a Git archive")
        if values.get("source_manifest_sha256") != manifest_sha256:
            raise ReceiptError("source manifest digest does not match .source_revision")
        actual = values.get("git_sha", "")
        kind = "git_archive"
    else:
        actual = completed.stdout.strip()
        kind = "git"
    if actual.lower() != expected_git_sha.lower():
        raise ReceiptError(
            f"Git identity mismatch: expected {expected_git_sha}, found {actual or 'missing'}"
        )
    return kind


def _environment(names: Sequence[str], environ: Mapping[str, str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for name in names:
        if not _ENV_NAME.fullmatch(name):
            raise ReceiptError(f"invalid environment allowlist name: {name!r}")
        if name in result:
            raise ReceiptError(f"duplicate environment allowlist name: {name}")
        if name not in environ:
            raise ReceiptError(f"allowlisted environment variable is not set: {name}")
        value = environ[name]
        if not isinstance(value, str):
            raise ReceiptError(f"allowlisted environment variable is not text: {name}")
        result[name] = value
    return result


def _refuse_existing(path: Path, label: str) -> None:
    if os.path.lexists(path):
        raise ReceiptError(f"refusing existing {label}: {path}")


def _write_receipt(path: Path, receipt: dict[str, Any]) -> None:
    payload = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("ascii")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o644)
    except FileExistsError as exc:
        raise ReceiptError(f"refusing existing receipt: {path}") from exc
    except OSError as exc:
        raise ReceiptError(f"cannot create receipt {path}: {exc}") from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def _provenance_path(
    root: Path, artifact_relative: str, artifact: Path
) -> tuple[str, Path]:
    relative = artifact_relative + ".prov.json"
    normalized, path = _safe_relative_path(root, relative, "provenance sidecar path")
    if path != Path(os.fspath(artifact) + ".prov.json"):
        raise ReceiptError("provenance sidecar path is not adjacent to artifact")
    return normalized, path


def _read_provenance_v2(path: Path) -> tuple[dict[str, Any], str]:
    try:
        evidence = read_stable_json_evidence(path, require_object=True)
    except StableJsonEvidenceError as exc:
        raise ReceiptError(f"invalid provenance sidecar: {exc}") from exc
    return evidence.value, evidence.file_sha256


def _validate_provenance_v2(
    value: Mapping[str, Any],
    *,
    artifact_relative: str,
    git_sha: str,
    source_kind: str,
    source_manifest_sha256: str,
    run_id: str,
    receipt_started_utc_ns: int,
    receipt_ended_utc_ns: int,
    environment: Mapping[str, str],
) -> tuple[int, int]:
    required_fields = {
        "artifact",
        "ended_utc_ns",
        "env",
        "git_dirty",
        "git_sha",
        "run_id",
        "schema",
        "sim_backend",
        "sim_backend_cupy_importable",
        "sim_backend_requested",
        "source_kind",
        "source_manifest_exit_error",
        "source_manifest_sha256",
        "source_manifest_start_error",
        "source_manifest_verified_at_exit",
        "source_manifest_verified_at_start",
        "started_utc_ns",
    }
    missing = required_fields.difference(value)
    if missing:
        raise ReceiptError(f"provenance sidecar is missing fields: {sorted(missing)}")
    if value.get("schema") != PROVENANCE_SCHEMA_V2:
        raise ReceiptError("provenance sidecar schema is invalid")
    if not _RUN_ID.fullmatch(run_id) or value.get("run_id") != run_id:
        raise ReceiptError("provenance sidecar run ID does not match receipt")
    if value.get("artifact") != artifact_relative:
        raise ReceiptError("provenance sidecar artifact path does not match receipt")
    if value.get("git_sha") != git_sha:
        raise ReceiptError("provenance sidecar Git SHA does not match receipt")
    if value.get("source_kind") != source_kind:
        raise ReceiptError("provenance sidecar source kind does not match receipt")
    if value.get("source_manifest_sha256") != source_manifest_sha256:
        raise ReceiptError("provenance sidecar source manifest does not match receipt")
    if type(value.get("git_dirty")) is not bool:
        raise ReceiptError("provenance sidecar git_dirty must be Boolean")

    started = value.get("started_utc_ns")
    ended = value.get("ended_utc_ns")
    if (
        type(started) is not int
        or type(ended) is not int
        or not receipt_started_utc_ns <= started <= ended <= receipt_ended_utc_ns
    ):
        raise ReceiptError("provenance sidecar timestamps are outside receipt interval")

    if "SIM_BACKEND" not in environment:
        raise ReceiptError("provenance v2 requires SIM_BACKEND in the environment allowlist")
    backend = environment["SIM_BACKEND"]
    if backend not in {"numpy", "cupy"}:
        raise ReceiptError("provenance sidecar backend is unsupported")
    sidecar_environment = value.get("env")
    if not isinstance(sidecar_environment, dict):
        raise ReceiptError("provenance sidecar environment is invalid")
    if any(name in sidecar_environment for name in _PRIVATE_PROVENANCE_ENV):
        raise ReceiptError("provenance sidecar exposes private provenance environment")
    if sidecar_environment.get("SIM_BACKEND") != backend:
        raise ReceiptError("provenance sidecar environment backend does not match receipt")
    if value.get("sim_backend_requested") != backend or value.get("sim_backend") != backend:
        raise ReceiptError("provenance sidecar backend does not match receipt")
    if type(value.get("sim_backend_cupy_importable")) is not bool:
        raise ReceiptError("provenance sidecar CuPy availability must be Boolean")
    if backend == "cupy" and value["sim_backend_cupy_importable"] is not True:
        raise ReceiptError("provenance sidecar does not confirm CuPy availability")

    verification = (
        value.get("source_manifest_verified_at_start"),
        value.get("source_manifest_start_error"),
        value.get("source_manifest_verified_at_exit"),
        value.get("source_manifest_exit_error"),
    )
    if source_kind == "git_archive":
        if verification != (True, None, True, None):
            raise ReceiptError("archive provenance source verification is invalid")
    elif source_kind == "git":
        if verification != (None, None, None, None):
            raise ReceiptError("Git provenance archive verification fields must be null")
    else:
        raise ReceiptError("provenance sidecar source kind is invalid")
    return started, ended


def run_and_receipt(
    *,
    root: Path,
    artifact_path: str | Path,
    receipt_path: str | Path,
    source_manifest: str | Path,
    git_sha: str,
    host: str,
    device: str,
    argv: Sequence[str],
    env_allowlist: Sequence[str] = (),
    environ: Mapping[str, str] | None = None,
    provenance_v2: bool = False,
) -> dict[str, Any]:
    """Execute argv and write a success receipt only after all checks pass."""
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ReceiptError(f"execution root is not a directory: {root}")
    if not argv or any(not isinstance(item, str) or not item for item in argv):
        raise ReceiptError("argv must be a non-empty sequence of non-empty strings")
    host = _required_text(host, "host")
    device = _required_text(device, "device")
    artifact_relative, artifact = _safe_relative_path(root, artifact_path, "artifact path")
    receipt_relative, receipt_file = _safe_relative_path(root, receipt_path, "receipt path")
    manifest_relative, _ = _safe_relative_path(root, source_manifest, "source manifest")
    provenance_relative = None
    provenance_file = None
    if provenance_v2:
        provenance_relative, provenance_file = _provenance_path(
            root, artifact_relative, artifact
        )
    distinct_paths = {artifact_relative, receipt_relative, manifest_relative}
    if provenance_relative is not None:
        distinct_paths.add(provenance_relative)
    if len(distinct_paths) != (4 if provenance_v2 else 3):
        raise ReceiptError("artifact, receipt, and source manifest paths must be distinct")
    outputs = [(artifact, "artifact"), (receipt_file, "receipt")]
    if provenance_file is not None:
        outputs.append((provenance_file, "provenance sidecar"))
    for path, label in outputs:
        if not path.parent.is_dir():
            raise ReceiptError(f"{label} parent directory does not exist: {path.parent}")
        _refuse_existing(path, label)

    environment = _environment(env_allowlist, os.environ if environ is None else environ)
    if provenance_v2 and "SIM_BACKEND" not in environment:
        raise ReceiptError("provenance v2 requires SIM_BACKEND in the environment allowlist")
    if provenance_v2 and _PRIVATE_PROVENANCE_ENV.intersection(environment):
        raise ReceiptError("private provenance variables cannot be allowlisted")
    if provenance_v2 and len(git_sha) != 40:
        raise ReceiptError("provenance v2 requires a full Git SHA")
    before = verify_source_manifest(root, manifest_relative)
    source_kind = _source_revision(root, git_sha, before["manifest_sha256"])
    normalized_git_sha = git_sha.lower()
    run_id = secrets.token_hex(32) if provenance_v2 else None
    child_environment = dict(environment)
    if provenance_v2:
        assert run_id is not None
        child_environment.update(
            {
                "SIM_PROVENANCE_V2": "1",
                "SIM_PROVENANCE_RUN_ID": run_id,
                "SIM_PROVENANCE_SOURCE_KIND": source_kind,
                "SIM_PROVENANCE_SOURCE_MANIFEST_SHA256": before["manifest_sha256"],
            }
        )

    started_utc_ns = time.time_ns()
    started_monotonic_ns = time.monotonic_ns()
    try:
        completed = subprocess.run(list(argv), cwd=root, env=child_environment, check=False)
    except OSError as exc:
        raise ReceiptError(f"cannot launch command: {exc}") from exc
    ended_monotonic_ns = time.monotonic_ns()
    ended_utc_ns = time.time_ns()
    if completed.returncode != 0:
        raise ReceiptError(f"command exited nonzero: {completed.returncode}")

    provenance = None
    if provenance_v2:
        assert provenance_file is not None
        assert provenance_relative is not None
        assert run_id is not None
        sidecar, sidecar_sha256 = _read_provenance_v2(provenance_file)
        sidecar_started, sidecar_ended = _validate_provenance_v2(
            sidecar,
            artifact_relative=artifact_relative,
            git_sha=normalized_git_sha,
            source_kind=source_kind,
            source_manifest_sha256=before["manifest_sha256"],
            run_id=run_id,
            receipt_started_utc_ns=started_utc_ns,
            receipt_ended_utc_ns=ended_utc_ns,
            environment=environment,
        )
        provenance = {
            "path": provenance_relative,
            "sha256": sidecar_sha256,
            "run_id": run_id,
            "started_utc_ns": sidecar_started,
            "ended_utc_ns": sidecar_ended,
        }

    _safe_relative_path(root, artifact_relative, "artifact path")
    artifact_sha256, artifact_size, _ = _hash_regular_file(artifact, "artifact")
    after = verify_source_manifest(root, manifest_relative)
    after_kind = _source_revision(root, git_sha, after["manifest_sha256"])
    if after != before or after_kind != source_kind:
        raise ReceiptError("source drift detected during command execution")
    receipt_relative_after, receipt_file_after = _safe_relative_path(
        root, receipt_relative, "receipt path"
    )
    if receipt_relative_after != receipt_relative or receipt_file_after != receipt_file:
        raise ReceiptError("receipt path changed during command execution")
    final_artifact_sha256, final_artifact_size, _ = _hash_regular_file(artifact, "artifact")
    if (final_artifact_sha256, final_artifact_size) != (artifact_sha256, artifact_size):
        raise ReceiptError("artifact changed after command completion")
    if provenance is not None:
        assert provenance_file is not None
        _, final_sidecar_sha256 = _read_provenance_v2(provenance_file)
        if final_sidecar_sha256 != provenance["sha256"]:
            raise ReceiptError("provenance sidecar changed after command completion")
    _refuse_existing(receipt_file, "receipt")

    receipt = {
        "argv": list(argv),
        "artifact": {
            "path": artifact_relative,
            "sha256": artifact_sha256,
            "size_bytes": artifact_size,
        },
        "device": device,
        "duration_monotonic_ns": ended_monotonic_ns - started_monotonic_ns,
        "ended_utc_ns": ended_utc_ns,
        "env_allowlist": environment,
        "execution_root": ".",
        "exit_code": completed.returncode,
        "host": host,
        "schema": SCHEMA_V2 if provenance_v2 else SCHEMA,
        "source": {
            "file_count": before["file_count"],
            "git_sha": normalized_git_sha,
            "kind": source_kind,
            "manifest": before["manifest"],
            "manifest_sha256": before["manifest_sha256"],
            "tree_sha256": before["tree_sha256"],
        },
        "started_utc_ns": started_utc_ns,
        "status": "success",
    }
    if provenance is not None:
        receipt["provenance"] = provenance
    _write_receipt(receipt_file, receipt)
    return receipt


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ReceiptError(
            f"invalid {label} fields: expected {sorted(expected)}, found {sorted(value)}"
        )


def verify_receipt(root: Path, receipt_path: str | Path) -> dict[str, Any]:
    """Validate a receipt and re-hash its artifact and current source manifest."""
    root = root.resolve(strict=True)
    _, path = _safe_relative_path(root, receipt_path, "receipt path")
    _, _, receipt_state = _hash_regular_file(path, "receipt")
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReceiptError(f"receipt is not valid JSON: {path}") from exc
    try:
        current_state = _file_state(path.lstat())
    except OSError as exc:
        raise ReceiptError(f"receipt disappeared while reading: {path}") from exc
    if current_state != receipt_state:
        raise ReceiptError(f"receipt changed while reading: {path}")
    if not isinstance(value, dict):
        raise ReceiptError("receipt must be a JSON object")
    schema = value.get("schema")
    receipt_fields = {
        "argv", "artifact", "device", "duration_monotonic_ns", "ended_utc_ns",
        "env_allowlist", "execution_root", "exit_code", "host", "schema", "source",
        "started_utc_ns", "status",
    }
    if schema == SCHEMA_V2:
        receipt_fields.add("provenance")
    _exact_keys(
        value,
        receipt_fields,
        "receipt",
    )
    if schema not in {SCHEMA, SCHEMA_V2} or value["status"] != "success":
        raise ReceiptError("receipt schema or status is invalid")
    if value["execution_root"] != "." or value["exit_code"] != 0:
        raise ReceiptError("receipt execution root or exit code is invalid")
    if (
        type(value["started_utc_ns"]) is not int
        or type(value["ended_utc_ns"]) is not int
        or value["ended_utc_ns"] < value["started_utc_ns"]
        or type(value["duration_monotonic_ns"]) is not int
        or value["duration_monotonic_ns"] < 0
    ):
        raise ReceiptError("receipt timestamps are invalid")
    _required_text(value["host"], "receipt host")
    _required_text(value["device"], "receipt device")
    if (
        not isinstance(value["argv"], list)
        or not value["argv"]
        or any(not isinstance(item, str) or not item for item in value["argv"])
    ):
        raise ReceiptError("receipt argv is invalid")
    if not isinstance(value["env_allowlist"], dict):
        raise ReceiptError("receipt environment allowlist is invalid")
    _environment(list(value["env_allowlist"]), value["env_allowlist"])

    artifact = value["artifact"]
    if not isinstance(artifact, dict):
        raise ReceiptError("receipt artifact is invalid")
    _exact_keys(artifact, {"path", "sha256", "size_bytes"}, "artifact")
    if not isinstance(artifact["sha256"], str) or not _SHA256.fullmatch(artifact["sha256"]):
        raise ReceiptError("receipt artifact SHA-256 is invalid")
    if type(artifact["size_bytes"]) is not int or artifact["size_bytes"] < 0:
        raise ReceiptError("receipt artifact size is invalid")
    _, artifact_path = _safe_relative_path(root, artifact["path"], "artifact path")
    actual_hash, actual_size, _ = _hash_regular_file(artifact_path, "artifact")
    if (actual_hash, actual_size) != (artifact["sha256"], artifact["size_bytes"]):
        raise ReceiptError("artifact does not match receipt")

    source = value["source"]
    if not isinstance(source, dict):
        raise ReceiptError("receipt source is invalid")
    _exact_keys(
        source,
        {"file_count", "git_sha", "kind", "manifest", "manifest_sha256", "tree_sha256"},
        "source",
    )
    if source["kind"] not in {"git", "git_archive"}:
        raise ReceiptError("receipt source kind is invalid")
    if not isinstance(source["git_sha"], str) or not _GIT_SHA.fullmatch(source["git_sha"]):
        raise ReceiptError("receipt Git SHA is invalid")
    if schema == SCHEMA_V2 and len(source["git_sha"]) != 40:
        raise ReceiptError("provenance v2 receipt requires a full Git SHA")
    for field in ("manifest_sha256", "tree_sha256"):
        if not isinstance(source[field], str) or not _SHA256.fullmatch(source[field]):
            raise ReceiptError(f"receipt source {field} is invalid")
    if type(source["file_count"]) is not int or source["file_count"] < 1:
        raise ReceiptError("receipt source file count is invalid")
    current = verify_source_manifest(root, source["manifest"])
    if (
        current["manifest_sha256"] != source["manifest_sha256"]
        or current["tree_sha256"] != source["tree_sha256"]
        or current["file_count"] != source["file_count"]
    ):
        raise ReceiptError("source manifest does not match receipt")
    if _source_revision(root, source["git_sha"], current["manifest_sha256"]) != source["kind"]:
        raise ReceiptError("source identity does not match receipt")

    if schema == SCHEMA_V2:
        provenance = value["provenance"]
        if not isinstance(provenance, dict):
            raise ReceiptError("receipt provenance is invalid")
        _exact_keys(
            provenance,
            {"path", "sha256", "run_id", "started_utc_ns", "ended_utc_ns"},
            "provenance",
        )
        if not isinstance(provenance["sha256"], str) or not _SHA256.fullmatch(
            provenance["sha256"]
        ):
            raise ReceiptError("receipt provenance SHA-256 is invalid")
        run_id = _required_text(provenance["run_id"], "receipt provenance run ID")
        if not _RUN_ID.fullmatch(run_id):
            raise ReceiptError("receipt provenance run ID is invalid")
        expected_provenance_relative, expected_provenance_path = _provenance_path(
            root, artifact["path"], artifact_path
        )
        if provenance["path"] != expected_provenance_relative:
            raise ReceiptError("receipt provenance path is not adjacent to artifact")
        sidecar, sidecar_sha256 = _read_provenance_v2(expected_provenance_path)
        if sidecar_sha256 != provenance["sha256"]:
            raise ReceiptError("provenance sidecar does not match receipt")
        sidecar_started, sidecar_ended = _validate_provenance_v2(
            sidecar,
            artifact_relative=artifact["path"],
            git_sha=source["git_sha"],
            source_kind=source["kind"],
            source_manifest_sha256=source["manifest_sha256"],
            run_id=run_id,
            receipt_started_utc_ns=value["started_utc_ns"],
            receipt_ended_utc_ns=value["ended_utc_ns"],
            environment=value["env_allowlist"],
        )
        if (
            provenance["started_utc_ns"] != sidecar_started
            or provenance["ended_utc_ns"] != sidecar_ended
        ):
            raise ReceiptError("receipt provenance timestamps do not match sidecar")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="subcommand", required=True)
    run = subparsers.add_parser("run", help="run a command and create a success receipt")
    run.add_argument("--root", type=Path, required=True)
    run.add_argument("--artifact", required=True)
    run.add_argument("--receipt", required=True)
    run.add_argument("--source-manifest", required=True)
    run.add_argument("--git-sha", required=True)
    run.add_argument("--host", required=True)
    run.add_argument("--device", required=True)
    run.add_argument("--env", action="append", default=[], dest="env_allowlist")
    run.add_argument("--provenance-v2", action="store_true")
    run.add_argument("command", nargs=argparse.REMAINDER)

    verify = subparsers.add_parser("verify", help="verify a receipt and its bound artifact")
    verify.add_argument("--root", type=Path, required=True)
    verify.add_argument("--receipt", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.subcommand == "run":
            command = args.command[1:] if args.command[:1] == ["--"] else args.command
            result = run_and_receipt(
                root=args.root,
                artifact_path=args.artifact,
                receipt_path=args.receipt,
                source_manifest=args.source_manifest,
                git_sha=args.git_sha,
                host=args.host,
                device=args.device,
                argv=command,
                env_allowlist=args.env_allowlist,
                provenance_v2=args.provenance_v2,
            )
            print(f"WROTE {args.receipt} ({result['artifact']['sha256']})")
        else:
            result = verify_receipt(args.root, args.receipt)
            print(f"VERIFIED {args.receipt} ({result['artifact']['sha256']})")
    except ReceiptError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
