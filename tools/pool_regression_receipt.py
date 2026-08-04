#!/usr/bin/env python3
"""Run the fixed seedless pool regression and seal a portable receipt."""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import socket
import stat
import subprocess
import time
from typing import Any, Mapping, Sequence

try:
    from tools import execution_receipt
    from tools.pool.provisioning import ancestry_attestation
except ModuleNotFoundError:  # Direct ``python tools/...`` invocation.
    import execution_receipt  # type: ignore[no-redef]
    from pool.provisioning import ancestry_attestation  # type: ignore[no-redef]


SCHEMA = "sim-pool-regression-receipt-v1"
MANIFEST_PATH = ".source_manifest.sha256"
BUNDLE_PATH = "tools/pool_regression_bundle_v1.sh"
_REVISION = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SCIENTIFIC_SEED_OUTPUT = re.compile(
    rb"(?i)(?<![a-z0-9])(?:scientific[-_ ]*)?seed(?:s)?\b"
)
_RECEIPT_FIELDS = {
    "command", "duration_monotonic_ns", "ended_utc", "ended_utc_ns",
    "environment", "exit_code", "expected_revision", "schema", "sha256",
    "source", "started_utc", "started_utc_ns", "status", "stderr", "stdout",
}


class PoolReceiptError(RuntimeError):
    """Raised when a regression receipt cannot be produced without ambiguity."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PoolReceiptError(message)


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")


def _self_digest(value: Mapping[str, Any]) -> str:
    body = dict(value)
    body.pop("sha256", None)
    return hashlib.sha256(_canonical_bytes(body)).hexdigest()


def _hash_regular(path: Path, label: str) -> tuple[str, int]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise PoolReceiptError(f"cannot open {label}: {path}") from exc
    try:
        with os.fdopen(descriptor, "rb") as stream:
            before = os.fstat(stream.fileno())
            _require(stat.S_ISREG(before.st_mode), f"{label} is not a regular file: {path}")
            digest = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
            after = os.fstat(stream.fileno())
    except OSError as exc:
        raise PoolReceiptError(f"cannot read {label}: {path}") from exc
    state = lambda item: (
        item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns, item.st_ctime_ns,
    )
    try:
        named = path.lstat()
    except OSError as exc:
        raise PoolReceiptError(f"{label} disappeared while hashing: {path}") from exc
    _require(
        not stat.S_ISLNK(named.st_mode)
        and state(before) == state(after) == state(named),
        f"{label} changed while hashing: {path}",
    )
    return digest.hexdigest(), after.st_size


def _root_shadowing_files(root: Path) -> list[str]:
    shadows: list[str] = []
    try:
        children = list(root.iterdir())
    except OSError as exc:
        raise PoolReceiptError(f"cannot inspect deployed root: {root}") from exc
    for child in children:
        name = child.name
        try:
            mode = child.lstat().st_mode
        except OSError as exc:
            raise PoolReceiptError(f"cannot inspect root-level entry: {child}") from exc
        if name == "__pycache__" and stat.S_ISDIR(mode):
            shadows.append(name)
        elif name.lower().endswith((".py", ".pyc", ".pyo")):
            shadows.append(name)
    return sorted(shadows)


def _verify_deployment(root: Path, expected_revision: str) -> tuple[dict[str, Any], dict[str, Any]]:
    shadows = _root_shadowing_files(root)
    _require(not shadows, f"root-level Python import-shadowing files are forbidden: {shadows}")
    try:
        source = execution_receipt.verify_source_manifest(root, MANIFEST_PATH)
        ancestry = ancestry_attestation.require_source_ancestor(
            root, expected_revision, expected_head=expected_revision,
        )
    except (execution_receipt.ReceiptError, ancestry_attestation.AncestryError) as exc:
        raise PoolReceiptError(f"deployed source verification failed: {exc}") from exc
    _require(ancestry.get("kind") == "git_archive", "deployed root must be an isolated Git archive")
    _require(
        ancestry.get("source_manifest_sha256") == source["manifest_sha256"],
        "source manifest and ancestry binding disagree",
    )
    _require(BUNDLE_PATH in source["files"], "source manifest does not bind the fixed regression bundle")
    bundle_digest, bundle_size = _hash_regular(root / BUNDLE_PATH, "regression bundle")
    bound_bundle = source["files"][BUNDLE_PATH]
    _require(
        bundle_digest == bound_bundle["sha256"] and bundle_size == bound_bundle["size_bytes"],
        "regression bundle does not match the source manifest",
    )
    return source, ancestry


def _command_environment() -> dict[str, str]:
    return {
        "HOME": "/nonexistent",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin",
        "TMPDIR": "/tmp",
        "TZ": "UTC",
    }


def _environment_metadata(root: Path) -> dict[str, Any]:
    deployed_python = root / ".venv/bin/python"
    _require(deployed_python.exists(), f"missing deployed interpreter: {deployed_python}")
    try:
        resolved_python = deployed_python.resolve(strict=True)
    except OSError as exc:
        raise PoolReceiptError(f"cannot resolve deployed interpreter: {deployed_python}") from exc
    python_digest, python_size = _hash_regular(resolved_python, "deployed interpreter")

    metadata_code = (
        "import importlib.metadata as m,json,platform,sys;"
        "d=sorted((x.metadata.get('Name') or x.name,x.version) for x in m.distributions());"
        "print(json.dumps({'implementation':platform.python_implementation(),"
        "'packages':d,'version':platform.python_version()},sort_keys=True,separators=(',',':')))"
    )
    try:
        completed = subprocess.run(
            [str(deployed_python), "-I", "-c", metadata_code],
            cwd=root,
            env=_command_environment(),
            capture_output=True,
            check=False,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise PoolReceiptError("cannot inspect deployed Python environment") from exc
    _require(completed.returncode == 0, "deployed Python environment inspection failed")
    _require(not completed.stderr, "deployed Python environment inspection wrote to stderr")
    try:
        python_metadata = json.loads(completed.stdout.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PoolReceiptError("deployed Python environment metadata is invalid") from exc
    _require(
        isinstance(python_metadata, dict)
        and set(python_metadata) == {"implementation", "packages", "version"},
        "deployed Python environment metadata has invalid fields",
    )
    uname = platform.uname()
    return {
        "command_environment": _command_environment(),
        "cpu_count": os.cpu_count(),
        "hostname": socket.gethostname(),
        "kernel": {
            "machine": uname.machine,
            "release": uname.release,
            "system": uname.system,
            "version": uname.version,
        },
        "python": {
            **python_metadata,
            "executable_path": ".venv/bin/python",
            "resolved_executable": str(resolved_python),
            "sha256": python_digest,
            "size_bytes": python_size,
        },
    }


def _stream_record(payload: bytes) -> dict[str, Any]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        return {
            "base64": base64.b64encode(payload).decode("ascii"),
            "encoding": "base64",
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }
    return {
        "encoding": "utf-8",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
        "text": text,
    }


def _receipt_target(root: Path, receipt_path: Path) -> Path:
    target = Path(os.path.abspath(receipt_path))
    _require(target.parent.is_dir(), f"receipt parent does not exist: {target.parent}")
    try:
        resolved_parent = target.parent.resolve(strict=True)
        resolved_parent.relative_to(root)
    except ValueError:
        pass
    except OSError as exc:
        raise PoolReceiptError(f"cannot resolve receipt parent: {target.parent}") from exc
    else:
        raise PoolReceiptError("receipt must be written outside the deployed root")
    _require(not os.path.lexists(target), f"refusing existing receipt: {target}")
    return target


def _write_create_only(path: Path, value: Mapping[str, Any]) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("ascii")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o644)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise PoolReceiptError(f"refusing existing receipt: {path}") from exc
    except OSError as exc:
        try:
            path.unlink()
        except OSError:
            pass
        raise PoolReceiptError(f"cannot create receipt: {path}") from exc


def collect_receipt(*, root: Path, expected_revision: str, receipt_path: Path) -> dict[str, Any]:
    """Run the fixed no-argument regression and write its external receipt."""
    _require(
        isinstance(expected_revision, str) and _REVISION.fullmatch(expected_revision) is not None,
        "expected revision must be a full lowercase Git commit ID",
    )
    try:
        root = root.resolve(strict=True)
    except OSError as exc:
        raise PoolReceiptError(f"deployed root does not exist: {root}") from exc
    _require(root.is_dir(), f"deployed root is not a directory: {root}")
    _require(
        root.name == expected_revision,
        "deployed revision mismatch: root is not isolated under the expected revision",
    )
    target = _receipt_target(root, receipt_path)
    before_source, before_ancestry = _verify_deployment(root, expected_revision)
    environment = _environment_metadata(root)
    command = ["bash", BUNDLE_PATH]

    started_utc_ns = time.time_ns()
    started_monotonic_ns = time.monotonic_ns()
    try:
        completed = subprocess.run(
            command,
            cwd=root,
            env=_command_environment(),
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        raise PoolReceiptError("cannot launch fixed pool regression bundle") from exc
    ended_monotonic_ns = time.monotonic_ns()
    ended_utc_ns = time.time_ns()

    if _SCIENTIFIC_SEED_OUTPUT.search(completed.stdout + b"\n" + completed.stderr):
        raise PoolReceiptError("regression output violated the seedless receipt contract")
    after_source, after_ancestry = _verify_deployment(root, expected_revision)
    _require(before_source == after_source, "source manifest state drifted during regression")
    _require(before_ancestry == after_ancestry, "source ancestry state drifted during regression")
    _require(not os.path.lexists(target), f"refusing existing receipt: {target}")

    source = {
        "ancestry_sha256": before_ancestry["source_ancestry_sha256"],
        "file_count": before_source["file_count"],
        "git_sha": expected_revision,
        "kind": before_ancestry["kind"],
        "manifest": before_source["manifest"],
        "manifest_sha256": before_source["manifest_sha256"],
        "tree_sha256": before_source["tree_sha256"],
    }
    value: dict[str, Any] = {
        "command": {"argv": command, "cwd": ".", "scientific_arguments": []},
        "duration_monotonic_ns": ended_monotonic_ns - started_monotonic_ns,
        "ended_utc": datetime.fromtimestamp(ended_utc_ns / 1_000_000_000, timezone.utc).isoformat(),
        "ended_utc_ns": ended_utc_ns,
        "environment": environment,
        "exit_code": completed.returncode,
        "expected_revision": expected_revision,
        "schema": SCHEMA,
        "source": source,
        "started_utc": datetime.fromtimestamp(started_utc_ns / 1_000_000_000, timezone.utc).isoformat(),
        "started_utc_ns": started_utc_ns,
        "status": "passed" if completed.returncode == 0 else "failed",
        "stderr": _stream_record(completed.stderr),
        "stdout": _stream_record(completed.stdout),
    }
    value["sha256"] = _self_digest(value)
    _write_create_only(target, value)
    return value


def verify_receipt(path: Path) -> dict[str, Any]:
    """Verify the exact fields and self-digest of a stored pool receipt."""
    digest, _ = _hash_regular(path, "pool regression receipt")
    del digest
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PoolReceiptError(f"pool regression receipt is invalid JSON: {path}") from exc
    _require(isinstance(value, dict), "pool regression receipt must be an object")
    _require(set(value) == _RECEIPT_FIELDS, "pool regression receipt fields are invalid")
    _require(value.get("schema") == SCHEMA, "pool regression receipt schema is invalid")
    _require(
        isinstance(value.get("sha256"), str) and _SHA256.fullmatch(value["sha256"]) is not None,
        "pool regression receipt digest is invalid",
    )
    _require(value.get("sha256") == _self_digest(value), "pool regression receipt self-digest is invalid")
    _require(
        value.get("status") in {"passed", "failed"}
        and type(value.get("exit_code")) is int
        and (value["status"] == "passed") == (value["exit_code"] == 0),
        "pool regression receipt status is invalid",
    )
    _require(
        value.get("command")
        == {"argv": ["bash", BUNDLE_PATH], "cwd": ".", "scientific_arguments": []},
        "pool regression receipt command is invalid",
    )
    _require(
        isinstance(value.get("expected_revision"), str)
        and _REVISION.fullmatch(value["expected_revision"]) is not None,
        "pool regression receipt revision is invalid",
    )
    source = value.get("source")
    _require(
        isinstance(source, dict)
        and set(source) == {
            "ancestry_sha256", "file_count", "git_sha", "kind", "manifest",
            "manifest_sha256", "tree_sha256",
        }
        and source.get("git_sha") == value["expected_revision"]
        and source.get("kind") == "git_archive"
        and source.get("manifest") == MANIFEST_PATH
        and type(source.get("file_count")) is int
        and source["file_count"] > 0
        and all(
            isinstance(source.get(field), str)
            and _SHA256.fullmatch(source[field]) is not None
            for field in ("ancestry_sha256", "manifest_sha256", "tree_sha256")
        ),
        "pool regression receipt source binding is invalid",
    )
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--receipt", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        value = collect_receipt(
            root=args.root,
            expected_revision=args.expected_revision,
            receipt_path=args.receipt,
        )
    except PoolReceiptError as exc:
        parser.error(str(exc))
    print(f"WROTE {args.receipt} ({value['sha256']})")
    return 0 if value["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
