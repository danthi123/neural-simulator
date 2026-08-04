#!/usr/bin/env python3
"""Build and verify the frozen V13 legacy scientific source package.

The builder reads Git objects only. It never imports or executes scientific
code, creates command envelopes, or reads experiment seed configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import socket
import stat
import subprocess
import sys
import time
from typing import Any, Mapping
import re
import zlib


ROOT = Path(__file__).resolve().parents[1]
BASE_REVISION = "8994b5102b39a8a6aa6abdeb9fde02579b7db6a8"
CANDIDATE_REVISION = "d091fa6692bdf8115c8073af6fd31fc9626921a8"
OVERLAY_PATH = "research/runners/_vocal_action_credit_gate_v13_tonic_output.py"
MANIFEST_NAME = ".source_manifest.sha256"
LOCK_NAME = ".legacy_source_package.json"
LOCK_SCHEMA = "v13-legacy-scientific-execution-package-v2"
RUN_DIRECTORY = "_run"
IDENTITY_DIRECTORY = ".source_identity.git"
IDENTITY_REFS_DIRECTORY = f"{IDENTITY_DIRECTORY}/refs"
IDENTITY_HEAD = f"{IDENTITY_DIRECTORY}/HEAD"
IDENTITY_CONFIG = f"{IDENTITY_DIRECTORY}/config"
IDENTITY_OBJECT = (
    f"{IDENTITY_DIRECTORY}/objects/{BASE_REVISION[:2]}/{BASE_REVISION[2:]}"
)
TRANSFER_MANIFEST_NAME = "legacy_transfer_manifest.json"
TRANSFER_SCHEMA = "v13-legacy-artifact-transfer-v1"
LEGACY_OUTPUT_NAME = "legacy_performance_baseline.json"
EXECUTION_RECEIPT_NAME = "legacy_execution_receipt.json"
EXECUTION_RECEIPT_SCHEMA = "v13-legacy-execution-receipt-v1"
RUNNER_MODULE = "research.runners._vocal_action_credit_gate_v13_tonic_output"
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_TRANSFER_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

# Import closure established by the independent legacy-package audit. The
# overlay is intentionally absent: it is sourced only from CANDIDATE_REVISION.
BASE_PATHS = (
    "experiment/__init__.py",
    "experiment/engine.py",
    "experiment/groups.py",
    "experiment/presets.py",
    "experiment/readout.py",
    "experiment/stimulus.py",
    "experiment/training.py",
    "research/__init__.py",
    "research/runners/__init__.py",
    "research/runners/_vocal_action_selector_gate.py",
    "sim/__init__.py",
    "sim/backend.py",
    "sim/bridge.py",
    "sim/config.py",
    "sim/connectivity.py",
    "sim/enums.py",
    "sim/kernels.py",
    "sim/neuromodulators.py",
    "sim/profiles.py",
    "sim/regions.py",
    "sim/synapse_storage.py",
    "sim/text_embeddings.py",
    "tools/lab.py",
    "tools/verdict.py",
)
EXPECTED_OVERLAYS = {OVERLAY_PATH: CANDIDATE_REVISION}
SOURCE_PATHS = tuple(sorted((*BASE_PATHS, OVERLAY_PATH)))


class PackageError(RuntimeError):
    """Raised when the package cannot be constructed or verified exactly."""


def _canonical_digest(value: Mapping[str, Any]) -> str:
    body = dict(value)
    body.pop("sha256", None)
    payload = json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_loose_commit(root: Path) -> bytes:
    content = bytes(_git(root, "cat-file", "commit", BASE_REVISION, binary=True))
    uncompressed = f"commit {len(content)}\0".encode("ascii") + content
    if hashlib.sha1(uncompressed).hexdigest() != BASE_REVISION:
        raise PackageError("legacy commit object does not match the required identity")
    return zlib.compress(uncompressed)


def _safe_path(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or not path.name
        or "." in path.parts
        or ".." in path.parts
        or path.as_posix() != value
    ):
        raise PackageError(f"unsafe package path: {value!r}")
    return value


def _git(root: Path, *args: str, binary: bool = False) -> str | bytes:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        capture_output=True,
        text=not binary,
        check=False,
    )
    if result.returncode != 0:
        error = (
            result.stderr.decode("utf-8", "replace")
            if binary
            else result.stderr
        )
        raise PackageError(f"git {' '.join(args)} failed: {error.strip()}")
    return result.stdout


def _resolve_exact_commit(root: Path, revision: str) -> str:
    resolved = str(_git(root, "rev-parse", f"{revision}^{{commit}}")).strip()
    if resolved != revision:
        raise PackageError(
            f"revision identity mismatch: expected {revision}, found {resolved}"
        )
    return resolved


def _commit_tree(root: Path, revision: str) -> str:
    tree = str(_git(root, "rev-parse", f"{revision}^{{tree}}")).strip()
    if not tree:
        raise PackageError(f"revision has no tree: {revision}")
    return tree


def _git_entry(root: Path, revision: str, relative: str) -> dict[str, str]:
    relative = _safe_path(relative)
    raw = bytes(_git(root, "ls-tree", "-z", revision, "--", relative, binary=True))
    records = [record for record in raw.split(b"\0") if record]
    if len(records) != 1:
        raise PackageError(
            f"expected one Git entry for {relative} at {revision}, found {len(records)}"
        )
    try:
        metadata, encoded_path = records[0].split(b"\t", 1)
        mode, kind, object_id = metadata.decode("ascii").split(" ")
        actual_path = encoded_path.decode("utf-8")
    except (UnicodeDecodeError, ValueError) as exc:
        raise PackageError(f"invalid Git tree entry for {relative}") from exc
    if actual_path != relative:
        raise PackageError(f"Git returned an unexpected path for {relative}: {actual_path}")
    if kind != "blob" or mode not in {"100644", "100755"}:
        raise PackageError(
            f"source must be a regular Git blob: {relative} ({mode} {kind})"
        )
    return {"path": relative, "mode": mode, "git_blob": object_id}


def _blob(root: Path, object_id: str) -> bytes:
    return bytes(_git(root, "cat-file", "blob", object_id, binary=True))


def _validate_overlay_policy(overlays: Mapping[str, str] | None) -> None:
    supplied = EXPECTED_OVERLAYS if overlays is None else dict(overlays)
    if supplied != EXPECTED_OVERLAYS:
        raise PackageError(
            "overlay policy must contain only the audited V13 runner overlay "
            f"from {CANDIDATE_REVISION}"
        )


def _inventory(
    root: Path, overlays: Mapping[str, str] | None = None,
) -> tuple[list[dict[str, Any]], str, str]:
    _validate_overlay_policy(overlays)
    base_revision = _resolve_exact_commit(root, BASE_REVISION)
    candidate_revision = _resolve_exact_commit(root, CANDIDATE_REVISION)
    base_tree = _commit_tree(root, base_revision)
    candidate_tree = _commit_tree(root, candidate_revision)

    records: list[dict[str, Any]] = []
    for relative in BASE_PATHS:
        entry = _git_entry(root, base_revision, relative)
        data = _blob(root, entry["git_blob"])
        records.append({
            **entry,
            "source_revision": base_revision,
            "sha256": _sha256(data),
            "data": data,
        })

    overlay = _git_entry(root, candidate_revision, OVERLAY_PATH)
    overlay_data = _blob(root, overlay["git_blob"])
    records.append({
        **overlay,
        "source_revision": candidate_revision,
        "sha256": _sha256(overlay_data),
        "data": overlay_data,
    })
    records.sort(key=lambda item: item["path"])
    if [item["path"] for item in records] != list(SOURCE_PATHS):
        raise PackageError("constructed source set differs from the audited closure")
    if sum(item["source_revision"] == candidate_revision for item in records) != 1:
        raise PackageError("package must contain exactly one candidate overlay")
    return records, base_tree, candidate_tree


def _reject_symlink_ancestors(path: Path) -> None:
    absolute = path.absolute()
    ancestors = list(reversed(absolute.parents))
    for ancestor in ancestors:
        try:
            info = ancestor.lstat()
        except FileNotFoundError:
            raise PackageError(f"output parent does not exist: {ancestor}")
        if stat.S_ISLNK(info.st_mode):
            raise PackageError(f"output path has a symlink ancestor: {ancestor}")
        if not stat.S_ISDIR(info.st_mode):
            raise PackageError(f"output ancestor is not a directory: {ancestor}")


def _write_exclusive(path: Path, data: bytes, mode: int = 0o644) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, mode)
    with os.fdopen(descriptor, "wb") as handle:
        os.fchmod(handle.fileno(), mode)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _package_file_set_digest(paths: list[str]) -> str:
    return _sha256("".join(f"{path}\n" for path in sorted(paths)).encode("utf-8"))


def _lock(
    records: list[dict[str, Any]],
    *,
    base_tree: str,
    candidate_tree: str,
    manifest: bytes,
    identity_files: Mapping[str, bytes],
) -> dict[str, Any]:
    files = [
        {key: record[key] for key in (
            "path", "mode", "git_blob", "sha256", "source_revision",
        )}
        for record in records
    ]
    exact_paths = [*SOURCE_PATHS, *identity_files, MANIFEST_NAME, LOCK_NAME]
    value: dict[str, Any] = {
        "schema": LOCK_SCHEMA,
        "status": "frozen",
        "base": {"revision": BASE_REVISION, "tree": base_tree},
        "candidate": {"revision": CANDIDATE_REVISION, "tree": candidate_tree},
        "overlay": {
            "path": OVERLAY_PATH,
            "source_revision": CANDIDATE_REVISION,
            "sha256": next(
                item["sha256"] for item in records if item["path"] == OVERLAY_PATH
            ),
        },
        "source_manifest": {
            "path": MANIFEST_NAME,
            "sha256": _sha256(manifest),
            "file_count": len(records),
        },
        "execution": {
            "runner": OVERLAY_PATH,
            "runner_module": RUNNER_MODULE,
            "python_flags": ["-B"],
            "environment": {
                "PYTHONDONTWRITEBYTECODE": "1",
                "SIM_BACKEND": "cupy",
                "SIM_NO_PROVENANCE": "1",
            },
            "working_directory": ".",
            "output_directory": RUN_DIRECTORY,
            "source_identity": {
                "kind": "package_git_object",
                "reported_revision": BASE_REVISION,
                "git_directory": IDENTITY_DIRECTORY,
                "files": [
                    {
                        "path": path,
                        "sha256": _sha256(data),
                    }
                    for path, data in sorted(identity_files.items())
                ],
            },
        },
        "package_file_set": {
            "file_count": len(exact_paths),
            "paths_sha256": _package_file_set_digest(exact_paths),
        },
        "files": files,
    }
    value["sha256"] = _canonical_digest(value)
    return value


def build_package(
    *,
    root: Path,
    output: Path,
    overlays: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Materialize the audited package without importing or executing it."""
    root = root.resolve(strict=True)
    output = output.expanduser().absolute()
    _reject_symlink_ancestors(output)
    if os.path.lexists(output):
        raise PackageError(f"refusing existing output: {output}")

    records, base_tree, candidate_tree = _inventory(root, overlays)
    identity_files = {
        IDENTITY_HEAD: f"{BASE_REVISION}\n".encode("ascii"),
        IDENTITY_CONFIG: (
            b"[core]\n\trepositoryformatversion = 0\n\tbare = false\n"
        ),
        IDENTITY_OBJECT: _git_loose_commit(root),
    }
    created = False
    try:
        os.mkdir(output, 0o755)
        created = True
        for record in records:
            destination = output.joinpath(*PurePosixPath(record["path"]).parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.parent.is_symlink():
                raise PackageError(f"refusing symlinked package directory: {destination.parent}")
            mode = 0o755 if record["mode"] == "100755" else 0o644
            _write_exclusive(destination, record["data"], mode)

        for relative, data in identity_files.items():
            destination = output.joinpath(*PurePosixPath(relative).parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            _write_exclusive(destination, data)

        (output / IDENTITY_REFS_DIRECTORY).mkdir()

        run_directory = output / RUN_DIRECTORY
        run_directory.mkdir(mode=0o700)

        manifest = "".join(
            f"{record['sha256']}  {record['path']}\n" for record in records
        ).encode("utf-8")
        _write_exclusive(output / MANIFEST_NAME, manifest)
        lock = _lock(
            records,
            base_tree=base_tree,
            candidate_tree=candidate_tree,
            manifest=manifest,
            identity_files=identity_files,
        )
        lock_bytes = (json.dumps(lock, indent=2, sort_keys=True) + "\n").encode("ascii")
        _write_exclusive(output / LOCK_NAME, lock_bytes)
        verified = verify_package(output)
    except Exception:
        if created:
            shutil.rmtree(output)
        raise
    return verified


def _regular_files(package: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for directory, names, filenames in os.walk(package, followlinks=False):
        directory_path = Path(directory)
        directory_info = directory_path.lstat()
        if stat.S_ISLNK(directory_info.st_mode) or not stat.S_ISDIR(directory_info.st_mode):
            raise PackageError(f"package contains a non-directory: {directory_path}")
        relative_directory = directory_path.relative_to(package).as_posix()
        if relative_directory == ".":
            names[:] = [name for name in names if name != RUN_DIRECTORY]
        for name in names:
            child = directory_path / name
            info = child.lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                raise PackageError(f"package contains a symlink or non-directory: {child}")
            if name == "__pycache__":
                raise PackageError(f"package contains a bytecode cache directory: {child}")
        for name in filenames:
            child = directory_path / name
            info = child.lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
                raise PackageError(f"package contains a symlink or non-regular file: {child}")
            relative = child.relative_to(package).as_posix()
            if child.suffix in {".pyc", ".pyo"}:
                raise PackageError(f"package contains Python bytecode: {child}")
            found[_safe_path(relative)] = child
    return found


def _runtime_files(package: Path) -> dict[str, Path]:
    runtime = package / RUN_DIRECTORY
    try:
        info = runtime.lstat()
    except FileNotFoundError as exc:
        raise PackageError("package runtime directory is missing") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise PackageError("package runtime path is not a real directory")

    found: dict[str, Path] = {}
    for directory, names, filenames in os.walk(runtime, followlinks=False):
        directory_path = Path(directory)
        for name in names:
            child = directory_path / name
            child_info = child.lstat()
            if (
                name == "__pycache__"
                or stat.S_ISLNK(child_info.st_mode)
                or not stat.S_ISDIR(child_info.st_mode)
            ):
                raise PackageError(f"invalid runtime directory: {child}")
        for name in filenames:
            child = directory_path / name
            child_info = child.lstat()
            if stat.S_ISLNK(child_info.st_mode) or not stat.S_ISREG(child_info.st_mode):
                raise PackageError(f"invalid runtime file: {child}")
            if child.suffix in {".pyc", ".pyo"}:
                raise PackageError(f"runtime contains Python bytecode: {child}")
            found[child.relative_to(runtime).as_posix()] = child
    return found


def verify_package(package: Path) -> dict[str, Any]:
    """Verify package bytes, modes, provenance, self-digest, and exact file set."""
    package = package.expanduser().absolute()
    try:
        root_info = package.lstat()
    except FileNotFoundError as exc:
        raise PackageError(f"package does not exist: {package}") from exc
    if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
        raise PackageError(f"package root is not a real directory: {package}")

    files_on_disk = _regular_files(package)
    lock_path = package / LOCK_NAME
    manifest_path = package / MANIFEST_NAME
    try:
        lock = json.loads(lock_path.read_text(encoding="ascii"))
        manifest = manifest_path.read_bytes()
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PackageError(f"cannot read package metadata: {exc}") from exc

    expected_lock_fields = {
        "schema", "status", "base", "candidate", "overlay", "source_manifest",
        "execution", "package_file_set", "files", "sha256",
    }
    if not isinstance(lock, dict) or set(lock) != expected_lock_fields:
        raise PackageError("package lock has missing or extra fields")
    if lock["schema"] != LOCK_SCHEMA or lock["status"] != "frozen":
        raise PackageError("package lock schema or status is invalid")
    if lock.get("sha256") != _canonical_digest(lock):
        raise PackageError("package lock self-digest is invalid")
    if lock["base"].get("revision") != BASE_REVISION:
        raise PackageError("package lock has the wrong base revision")
    if lock["candidate"].get("revision") != CANDIDATE_REVISION:
        raise PackageError("package lock has the wrong candidate revision")
    if not _HEX40.fullmatch(str(lock["base"].get("tree", ""))):
        raise PackageError("package lock has an invalid base tree")
    if not _HEX40.fullmatch(str(lock["candidate"].get("tree", ""))):
        raise PackageError("package lock has an invalid candidate tree")
    if lock["overlay"].get("path") != OVERLAY_PATH or lock["overlay"].get(
        "source_revision"
    ) != CANDIDATE_REVISION:
        raise PackageError("package lock has an unexpected overlay")

    execution = lock.get("execution")
    if not isinstance(execution, dict) or set(execution) != {
        "runner", "runner_module", "python_flags", "environment", "working_directory",
        "output_directory", "source_identity",
    }:
        raise PackageError("package execution contract is invalid")
    if execution != {
        "runner": OVERLAY_PATH,
        "runner_module": RUNNER_MODULE,
        "python_flags": ["-B"],
        "environment": {
            "PYTHONDONTWRITEBYTECODE": "1",
            "SIM_BACKEND": "cupy",
            "SIM_NO_PROVENANCE": "1",
        },
        "working_directory": ".",
        "output_directory": RUN_DIRECTORY,
        "source_identity": execution["source_identity"],
    }:
        raise PackageError("package execution controls differ from the frozen contract")
    identity = execution["source_identity"]
    identity_paths = [IDENTITY_HEAD, IDENTITY_CONFIG, IDENTITY_OBJECT]
    if not isinstance(identity, dict) or identity != {
        "kind": "package_git_object",
        "reported_revision": BASE_REVISION,
        "git_directory": IDENTITY_DIRECTORY,
        "files": identity.get("files"),
    }:
        raise PackageError("package source identity is invalid")
    identity_records = identity.get("files")
    if (
        not isinstance(identity_records, list)
        or [item.get("path") for item in identity_records if isinstance(item, dict)]
        != identity_paths
        or any(
            not isinstance(item, dict)
            or set(item) != {"path", "sha256"}
            or not _HEX64.fullmatch(str(item["sha256"]))
            for item in identity_records
        )
    ):
        raise PackageError("package source identity files are invalid")

    source_records = lock.get("files")
    if not isinstance(source_records, list):
        raise PackageError("package lock files must be a list")
    record_fields = {"path", "mode", "git_blob", "sha256", "source_revision"}
    if any(not isinstance(item, dict) or set(item) != record_fields for item in source_records):
        raise PackageError("package lock contains an invalid source record")
    if any(
        item["mode"] not in {"100644", "100755"}
        or not _HEX40.fullmatch(str(item["git_blob"]))
        or not _HEX64.fullmatch(str(item["sha256"]))
        for item in source_records
    ):
        raise PackageError("package lock contains invalid source identity fields")
    record_paths = [item["path"] for item in source_records]
    if record_paths != list(SOURCE_PATHS) or len(record_paths) != len(set(record_paths)):
        raise PackageError("package source set differs from the audited closure")
    overlay_records = [
        item for item in source_records if item["source_revision"] == CANDIDATE_REVISION
    ]
    if len(overlay_records) != 1 or overlay_records[0]["path"] != OVERLAY_PATH:
        raise PackageError("package contains an unexpected candidate overlay")
    if any(
        item["source_revision"] != BASE_REVISION
        for item in source_records
        if item["path"] != OVERLAY_PATH
    ):
        raise PackageError("package contains a non-base scientific source")

    manifest_lines = manifest.decode("utf-8").splitlines()
    expected_lines = [f"{item['sha256']}  {item['path']}" for item in source_records]
    if manifest_lines != expected_lines:
        raise PackageError("source manifest differs from the locked source records")
    manifest_binding = lock["source_manifest"]
    if manifest_binding != {
        "path": MANIFEST_NAME,
        "sha256": _sha256(manifest),
        "file_count": len(source_records),
    }:
        raise PackageError("source manifest binding is invalid")

    expected_paths = {
        *SOURCE_PATHS, *identity_paths, MANIFEST_NAME, LOCK_NAME,
    }
    if set(files_on_disk) != expected_paths:
        missing = sorted(expected_paths - set(files_on_disk))
        extra = sorted(set(files_on_disk) - expected_paths)
        raise PackageError(f"package file set differs: missing={missing}, extra={extra}")
    file_set = lock["package_file_set"]
    if file_set != {
        "file_count": len(expected_paths),
        "paths_sha256": _package_file_set_digest(list(expected_paths)),
    }:
        raise PackageError("package exact-file-set binding is invalid")

    for item in source_records:
        path = files_on_disk[item["path"]]
        data = path.read_bytes()
        if _sha256(data) != item["sha256"]:
            raise PackageError(f"source digest mismatch: {item['path']}")
        expected_mode = 0o755 if item["mode"] == "100755" else 0o644
        if stat.S_IMODE(path.stat().st_mode) != expected_mode:
            raise PackageError(f"source mode mismatch: {item['path']}")
    if lock["overlay"].get("sha256") != overlay_records[0]["sha256"]:
        raise PackageError("overlay digest binding is invalid")

    for item in identity_records:
        path = files_on_disk[item["path"]]
        if _sha256(path.read_bytes()) != item["sha256"]:
            raise PackageError(f"source identity digest mismatch: {item['path']}")
        if stat.S_IMODE(path.stat().st_mode) != 0o644:
            raise PackageError(f"source identity mode mismatch: {item['path']}")
    refs = package / IDENTITY_REFS_DIRECTORY
    if not refs.is_dir() or refs.is_symlink() or any(refs.iterdir()):
        raise PackageError("package source identity refs directory is invalid")
    _runtime_files(package)

    identity_result = subprocess.run(
        ["git", "rev-parse", "HEAD^{commit}"], cwd=package,
        env=_execution_environment(package), capture_output=True, text=True, check=False,
    )
    if identity_result.returncode != 0 or identity_result.stdout.strip() != BASE_REVISION:
        raise PackageError("package-owned Git identity cannot report the legacy revision")

    return {
        "schema": LOCK_SCHEMA,
        "status": "verified",
        "base_revision": BASE_REVISION,
        "candidate_revision": CANDIDATE_REVISION,
        "source_file_count": len(source_records),
        "package_file_count": len(expected_paths),
        "manifest_sha256": _sha256(manifest),
        "lock_sha256": lock["sha256"],
        "reported_source_revision": BASE_REVISION,
    }


def _execution_environment(package: Path) -> dict[str, str]:
    environment = os.environ.copy()
    for name in tuple(environment):
        if name.startswith(("SIM_", "GIT_")):
            environment.pop(name)
    environment.update({
        "PYTHONDONTWRITEBYTECODE": "1",
        "SIM_BACKEND": "cupy",
        "SIM_NO_PROVENANCE": "1",
        "GIT_DIR": str(package / IDENTITY_DIRECTORY),
        "GIT_WORK_TREE": str(package),
        "GIT_CEILING_DIRECTORIES": str(package.parent),
        "GIT_CONFIG_NOSYSTEM": "1",
    })
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    return environment


def _validate_legacy_artifact(data: bytes, label: str) -> dict[str, Any]:
    try:
        artifact = json.loads(data)
    except json.JSONDecodeError as exc:
        raise PackageError(f"{label} is not JSON") from exc
    if (
        not isinstance(artifact, dict)
        or artifact.get("stage") != "legacy_performance_baseline"
        or artifact.get("source_sha") != BASE_REVISION
        or artifact.get("outcome") != "BASELINE_RECORDED"
        or artifact.get("backend") != "cupy"
        or "3090" not in str(artifact.get("device", ""))
        or not isinstance(artifact.get("median_seconds"), (int, float))
        or isinstance(artifact.get("median_seconds"), bool)
        or artifact["median_seconds"] <= 0
    ):
        raise PackageError(
            f"{label} has the wrong source, stage, backend, device, or outcome"
        )
    return artifact


def _receipt_environment() -> dict[str, str]:
    return {
        "PYTHONDONTWRITEBYTECODE": "1",
        "SIM_BACKEND": "cupy",
        "SIM_NO_PROVENANCE": "1",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_IDENTITY": IDENTITY_DIRECTORY,
        "PYTHONPATH": "absent",
        "PYTHONHOME": "absent",
    }


def _validate_execution_receipt(
    data: bytes,
    *,
    package: Path,
    verified: Mapping[str, Any],
    artifact_bytes: bytes,
) -> dict[str, Any]:
    try:
        receipt = json.loads(data)
    except json.JSONDecodeError as exc:
        raise PackageError("legacy execution receipt is not JSON") from exc
    required = {
        "schema", "status", "package", "command", "working_directory",
        "environment_controls", "host", "device", "timing", "exit_code",
        "artifact", "sha256",
    }
    if not isinstance(receipt, dict) or set(receipt) != required:
        raise PackageError("legacy execution receipt has missing or extra fields")
    if (
        receipt["schema"] != EXECUTION_RECEIPT_SCHEMA
        or receipt["status"] != "completed"
        or receipt["sha256"] != _canonical_digest(receipt)
    ):
        raise PackageError("legacy execution receipt identity is invalid")
    lock_bytes = (package / LOCK_NAME).read_bytes()
    manifest_bytes = (package / MANIFEST_NAME).read_bytes()
    if receipt["package"] != {
        "lock_path": LOCK_NAME,
        "lock_file_sha256": _sha256(lock_bytes),
        "lock_sha256": verified["lock_sha256"],
        "source_manifest_path": MANIFEST_NAME,
        "source_manifest_sha256": _sha256(manifest_bytes),
        "base_revision": BASE_REVISION,
        "overlay_revision": CANDIDATE_REVISION,
    }:
        raise PackageError("legacy execution receipt package binding is invalid")
    expected_suffix = [
        "-B", "-m", RUNNER_MODULE, "--legacy-performance-baseline", "--out",
        str(package / RUN_DIRECTORY / LEGACY_OUTPUT_NAME),
    ]
    command = receipt["command"]
    if (
        not isinstance(command, list)
        or len(command) != len(expected_suffix) + 1
        or not isinstance(command[0], str)
        or not command[0]
        or command[1:] != expected_suffix
    ):
        raise PackageError("legacy execution receipt command is invalid")
    if receipt["working_directory"] != ".":
        raise PackageError("legacy execution receipt working directory is invalid")
    if receipt["environment_controls"] != _receipt_environment():
        raise PackageError("legacy execution receipt environment is invalid")
    timing = receipt["timing"]
    if (
        not isinstance(timing, dict)
        or set(timing) != {"started_utc_ns", "ended_utc_ns", "duration_ns"}
        or any(isinstance(timing[key], bool) or not isinstance(timing[key], int) for key in timing)
        or timing["started_utc_ns"] <= 0
        or timing["ended_utc_ns"] < timing["started_utc_ns"]
        or timing["duration_ns"] < 0
    ):
        raise PackageError("legacy execution receipt timing is invalid")
    artifact = _validate_legacy_artifact(artifact_bytes, "legacy execution artifact")
    if receipt["exit_code"] != 0 or receipt["device"] != artifact["device"]:
        raise PackageError("legacy execution receipt result is invalid")
    if not isinstance(receipt["host"], str) or not receipt["host"]:
        raise PackageError("legacy execution receipt host is invalid")
    if receipt["artifact"] != {
        "path": f"{RUN_DIRECTORY}/{LEGACY_OUTPUT_NAME}",
        "sha256": _sha256(artifact_bytes),
        "size_bytes": len(artifact_bytes),
    }:
        raise PackageError("legacy execution receipt artifact binding is invalid")
    return receipt


def probe_execution_package(package: Path, *, python: str = sys.executable) -> dict[str, Any]:
    """Exercise the execution controls without importing or running the simulator."""
    package = package.expanduser().absolute()
    verified = verify_package(package)
    probe = (
        "import json,os,subprocess,sys;"
        "print(json.dumps({'bytecode':sys.dont_write_bytecode,"
        "'revision':subprocess.run(['git','rev-parse','HEAD^{commit}'],"
        "check=True,capture_output=True,text=True).stdout.strip(),"
        "'cwd':os.getcwd()}))"
    )
    completed = subprocess.run(
        [python, "-B", "-c", probe], cwd=package,
        env=_execution_environment(package), capture_output=True, text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise PackageError(f"execution-package probe failed: {completed.stderr.strip()}")
    try:
        observation = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise PackageError("execution-package probe returned invalid JSON") from exc
    if observation != {
        "bytecode": True,
        "revision": BASE_REVISION,
        "cwd": str(package),
    }:
        raise PackageError("execution-package probe did not enforce its controls")
    runner_probe = subprocess.run(
        [python, "-B", "-m", RUNNER_MODULE, "--help"], cwd=package,
        env=_execution_environment(package), capture_output=True, text=True,
        check=False,
    )
    if runner_probe.returncode != 0 or "--legacy-performance-baseline" not in (
        runner_probe.stdout
    ):
        raise PackageError(
            "frozen runner cannot start under the execution-package controls: "
            f"{runner_probe.stderr.strip()}"
        )
    if _runtime_files(package):
        raise PackageError("seed-free execution-package probe changed the runtime area")
    verify_package(package)
    return {
        "schema": LOCK_SCHEMA,
        "status": "execution_controls_verified",
        "lock_sha256": verified["lock_sha256"],
        "manifest_sha256": verified["manifest_sha256"],
        "reported_source_revision": BASE_REVISION,
        "python_no_bytecode": True,
        "runtime_file_count": 0,
    }


def execute_legacy_baseline(
    package: Path, *, python: str = sys.executable,
) -> dict[str, Any]:
    """Run only the frozen legacy baseline, with output isolated under ``_run``."""
    package = package.expanduser().absolute()
    verified = verify_package(package)
    if _runtime_files(package):
        raise PackageError("legacy execution requires an empty runtime directory")
    output = package / RUN_DIRECTORY / LEGACY_OUTPUT_NAME
    command = [
        python, "-B", "-m", RUNNER_MODULE, "--legacy-performance-baseline",
        "--out", str(output),
    ]
    started_utc_ns = time.time_ns()
    started_monotonic_ns = time.monotonic_ns()
    completed = subprocess.run(
        command, cwd=package, env=_execution_environment(package),
        capture_output=True, text=True, check=False,
    )
    ended_monotonic_ns = time.monotonic_ns()
    ended_utc_ns = time.time_ns()
    verify_package(package)
    if completed.returncode != 0:
        raise PackageError(
            "legacy baseline execution failed under the package contract: "
            f"{completed.stderr.strip()}"
        )
    runtime = _runtime_files(package)
    if set(runtime) != {LEGACY_OUTPUT_NAME}:
        raise PackageError("legacy execution produced files outside its one allowed artifact")
    artifact_bytes = runtime[LEGACY_OUTPUT_NAME].read_bytes()
    artifact = _validate_legacy_artifact(artifact_bytes, "legacy execution artifact")
    lock_bytes = (package / LOCK_NAME).read_bytes()
    manifest_bytes = (package / MANIFEST_NAME).read_bytes()
    receipt: dict[str, Any] = {
        "schema": EXECUTION_RECEIPT_SCHEMA,
        "status": "completed",
        "package": {
            "lock_path": LOCK_NAME,
            "lock_file_sha256": _sha256(lock_bytes),
            "lock_sha256": verified["lock_sha256"],
            "source_manifest_path": MANIFEST_NAME,
            "source_manifest_sha256": _sha256(manifest_bytes),
            "base_revision": BASE_REVISION,
            "overlay_revision": CANDIDATE_REVISION,
        },
        "command": command,
        "working_directory": ".",
        "environment_controls": _receipt_environment(),
        "host": socket.gethostname(),
        "device": artifact["device"],
        "timing": {
            "started_utc_ns": started_utc_ns,
            "ended_utc_ns": ended_utc_ns,
            "duration_ns": ended_monotonic_ns - started_monotonic_ns,
        },
        "exit_code": completed.returncode,
        "artifact": {
            "path": f"{RUN_DIRECTORY}/{LEGACY_OUTPUT_NAME}",
            "sha256": _sha256(artifact_bytes),
            "size_bytes": len(artifact_bytes),
        },
    }
    receipt["sha256"] = _canonical_digest(receipt)
    receipt_bytes = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("ascii")
    _write_exclusive(package / RUN_DIRECTORY / EXECUTION_RECEIPT_NAME, receipt_bytes)
    runtime = _runtime_files(package)
    if set(runtime) != {LEGACY_OUTPUT_NAME, EXECUTION_RECEIPT_NAME}:
        raise PackageError("legacy execution produced files outside its allowed evidence set")
    _validate_execution_receipt(
        receipt_bytes, package=package, verified=verified, artifact_bytes=artifact_bytes,
    )
    return {
        "schema": LOCK_SCHEMA,
        "status": "legacy_baseline_recorded",
        "lock_sha256": verified["lock_sha256"],
        "manifest_sha256": verified["manifest_sha256"],
        "reported_source_revision": BASE_REVISION,
        "artifact": {
            "path": f"{RUN_DIRECTORY}/{LEGACY_OUTPUT_NAME}",
            "sha256": _sha256(artifact_bytes),
            "size_bytes": len(artifact_bytes),
        },
        "execution_receipt": {
            "path": f"{RUN_DIRECTORY}/{EXECUTION_RECEIPT_NAME}",
            "sha256": _sha256(receipt_bytes),
            "canonical_sha256": receipt["sha256"],
            "size_bytes": len(receipt_bytes),
        },
    }


def transfer_legacy_artifact(
    package: Path, *, candidate_evidence: Path, transfer_name: str,
) -> dict[str, Any]:
    """Create a collision-proof evidence transfer containing artifact and manifest."""
    if not _TRANSFER_NAME.fullmatch(transfer_name) or transfer_name in {".", ".."}:
        raise PackageError(f"invalid transfer name: {transfer_name!r}")
    package = package.expanduser().absolute()
    verified = verify_package(package)
    runtime = _runtime_files(package)
    if set(runtime) != {LEGACY_OUTPUT_NAME, EXECUTION_RECEIPT_NAME}:
        raise PackageError("transfer requires exactly one artifact and execution receipt")
    artifact_path = runtime[LEGACY_OUTPUT_NAME]
    artifact_bytes = artifact_path.read_bytes()
    artifact = _validate_legacy_artifact(artifact_bytes, "legacy transfer artifact")
    receipt_bytes = runtime[EXECUTION_RECEIPT_NAME].read_bytes()
    receipt = _validate_execution_receipt(
        receipt_bytes, package=package, verified=verified, artifact_bytes=artifact_bytes,
    )

    candidate_evidence = candidate_evidence.expanduser().absolute()
    _reject_symlink_ancestors(candidate_evidence / transfer_name)
    try:
        evidence_info = candidate_evidence.lstat()
    except FileNotFoundError as exc:
        raise PackageError("candidate evidence root does not exist") from exc
    if stat.S_ISLNK(evidence_info.st_mode) or not stat.S_ISDIR(evidence_info.st_mode):
        raise PackageError("candidate evidence root is not a real directory")
    destination = candidate_evidence / transfer_name
    if os.path.lexists(destination):
        raise PackageError(f"refusing existing transfer destination: {destination}")

    lock_bytes = (package / LOCK_NAME).read_bytes()
    manifest_bytes = (package / MANIFEST_NAME).read_bytes()
    if _sha256(manifest_bytes) != verified["manifest_sha256"]:
        raise PackageError("source manifest changed before evidence transfer")
    transferred_relative = f"{transfer_name}/{LEGACY_OUTPUT_NAME}"
    value: dict[str, Any] = {
        "schema": TRANSFER_SCHEMA,
        "status": "transferred",
        "backend": artifact["backend"],
        "device": artifact["device"],
        "package": {
            "lock_path": LOCK_NAME,
            "lock_file_sha256": _sha256(lock_bytes),
            "lock_sha256": verified["lock_sha256"],
            "source_manifest_path": MANIFEST_NAME,
            "source_manifest_sha256": _sha256(manifest_bytes),
            "reported_source_revision": BASE_REVISION,
        },
        "artifact": {
            "package_path": f"{RUN_DIRECTORY}/{LEGACY_OUTPUT_NAME}",
            "candidate_evidence_path": transferred_relative,
            "sha256": _sha256(artifact_bytes),
            "size_bytes": len(artifact_bytes),
        },
        "execution_receipt": {
            "package_path": f"{RUN_DIRECTORY}/{EXECUTION_RECEIPT_NAME}",
            "candidate_evidence_path": f"{transfer_name}/{EXECUTION_RECEIPT_NAME}",
            "sha256": _sha256(receipt_bytes),
            "canonical_sha256": receipt["sha256"],
            "size_bytes": len(receipt_bytes),
        },
    }
    value["sha256"] = _canonical_digest(value)
    transfer_bytes = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("ascii")

    created = False
    try:
        os.mkdir(destination, 0o755)
        created = True
        _write_exclusive(destination / LEGACY_OUTPUT_NAME, artifact_bytes)
        _write_exclusive(destination / EXECUTION_RECEIPT_NAME, receipt_bytes)
        _write_exclusive(destination / TRANSFER_MANIFEST_NAME, transfer_bytes)
    except Exception:
        if created:
            shutil.rmtree(destination)
        raise
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--package", type=Path, required=True)
    probe = commands.add_parser("probe-execution")
    probe.add_argument("--package", type=Path, required=True)
    execute = commands.add_parser("execute-legacy-baseline")
    execute.add_argument("--package", type=Path, required=True)
    transfer = commands.add_parser("transfer")
    transfer.add_argument("--package", type=Path, required=True)
    transfer.add_argument("--candidate-evidence", type=Path, required=True)
    transfer.add_argument("--transfer-name", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "build":
            result = build_package(root=args.root, output=args.output)
        elif args.command == "verify":
            result = verify_package(args.package)
        elif args.command == "probe-execution":
            result = probe_execution_package(args.package)
        elif args.command == "execute-legacy-baseline":
            result = execute_legacy_baseline(args.package)
        else:
            result = transfer_legacy_artifact(
                args.package, candidate_evidence=args.candidate_evidence,
                transfer_name=args.transfer_name,
            )
    except (PackageError, OSError, ValueError, TypeError) as exc:
        print(f"v13-legacy-source-package: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
