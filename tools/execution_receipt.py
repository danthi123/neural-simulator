#!/usr/bin/env python3
"""Run one command and create a fail-closed receipt for its output artifact."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import time
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


SCHEMA = "sim-execution-receipt-v1"
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^[0-9a-fA-F]{7,64}$")


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
    candidate = root.joinpath(*relative.parts)
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
    if len({artifact_relative, receipt_relative, manifest_relative}) != 3:
        raise ReceiptError("artifact, receipt, and source manifest paths must be distinct")
    for path, label in ((artifact, "artifact"), (receipt_file, "receipt")):
        if not path.parent.is_dir():
            raise ReceiptError(f"{label} parent directory does not exist: {path.parent}")
        _refuse_existing(path, label)

    environment = _environment(env_allowlist, os.environ if environ is None else environ)
    before = verify_source_manifest(root, manifest_relative)
    source_kind = _source_revision(root, git_sha, before["manifest_sha256"])

    started_utc_ns = time.time_ns()
    started_monotonic_ns = time.monotonic_ns()
    try:
        completed = subprocess.run(list(argv), cwd=root, env=environment, check=False)
    except OSError as exc:
        raise ReceiptError(f"cannot launch command: {exc}") from exc
    ended_monotonic_ns = time.monotonic_ns()
    ended_utc_ns = time.time_ns()
    if completed.returncode != 0:
        raise ReceiptError(f"command exited nonzero: {completed.returncode}")

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
        "schema": SCHEMA,
        "source": {
            "file_count": before["file_count"],
            "git_sha": git_sha.lower(),
            "kind": source_kind,
            "manifest": before["manifest"],
            "manifest_sha256": before["manifest_sha256"],
            "tree_sha256": before["tree_sha256"],
        },
        "started_utc_ns": started_utc_ns,
        "status": "success",
    }
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
    _exact_keys(
        value,
        {
            "argv", "artifact", "device", "duration_monotonic_ns", "ended_utc_ns",
            "env_allowlist", "execution_root", "exit_code", "host", "schema", "source",
            "started_utc_ns", "status",
        },
        "receipt",
    )
    if value["schema"] != SCHEMA or value["status"] != "success":
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
