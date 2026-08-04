#!/usr/bin/env python3
"""Seal V13 Stage-0 evidence, including runner provenance, as one manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys
from typing import Any, Mapping, Sequence

try:
    from tools import execution_receipt
    from tools import v13_stage0_controller as controller
except ModuleNotFoundError:  # Direct ``python tools/...`` invocation.
    import execution_receipt  # type: ignore[no-redef]
    import v13_stage0_controller as controller  # type: ignore[no-redef]


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_V1 = "v13-stage0-artifact-manifest-v1"
SCHEMA_V2 = "v13-stage0-artifact-manifest-v2"
SCHEMA = SCHEMA_V2
MANIFEST_FIELDS_V1 = frozenset((
    "schema", "kind", "config_sha256", "source_revision", "artifact",
    "command_envelope", "execution_receipt", "sha256",
))
MANIFEST_FIELDS_V2 = frozenset((*MANIFEST_FIELDS_V1, "provenance_sidecar"))
REFERENCE_FIELDS = frozenset(("path", "sha256"))
KINDS = tuple(sorted(controller.MANIFEST_ACTIONS))
_ACTION_BY_KIND = controller.MANIFEST_ACTIONS
_HEX64 = frozenset("0123456789abcdef")


class ManifestError(ValueError):
    """Raised when evidence cannot earn a V13 Stage-0 artifact manifest."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ManifestError(message)


def _safe_repo_path(
    root: Path,
    value: str | Path,
    label: str,
    *,
    must_exist: bool = False,
) -> tuple[str, Path]:
    try:
        text = os.fspath(value)
    except TypeError as exc:
        raise ManifestError(f"{label} must be a repository-relative path") from exc
    _require(isinstance(text, str) and bool(text),
             f"{label} must be a repository-relative path")
    relative = PurePosixPath(text)
    _require(
        not relative.is_absolute()
        and bool(relative.name)
        and "." not in relative.parts
        and ".." not in relative.parts,
        f"{label} must be a safe repository-relative path",
    )
    normalized = relative.as_posix()
    candidate = root.joinpath(*relative.parts)
    try:
        resolved = candidate.resolve(strict=must_exist)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise ManifestError(f"{label} escapes or is missing from the repository") from exc
    return normalized, candidate


def _hash_regular_file(path: Path, label: str) -> str:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ManifestError(f"cannot open {label}: {path}: {exc}") from exc
    try:
        with os.fdopen(descriptor, "rb") as handle:
            before = os.fstat(handle.fileno())
            _require(stat.S_ISREG(before.st_mode), f"{label} is not a regular file: {path}")
            digest = hashlib.sha256()
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
            after = os.fstat(handle.fileno())
    except OSError as exc:
        raise ManifestError(f"cannot hash {label}: {path}: {exc}") from exc
    _require(
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
        f"{label} changed while hashing: {path}",
    )
    try:
        named = path.lstat()
    except OSError as exc:
        raise ManifestError(f"{label} disappeared after hashing: {path}") from exc
    _require(not stat.S_ISLNK(named.st_mode), f"{label} cannot be a symlink: {path}")
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot read {label}: {path}: {exc}") from exc
    _require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def _is_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and not (set(value) - _HEX64)
    )


def _validate_manifest_shape(
    manifest: dict[str, Any],
    *,
    allow_legacy_v1: bool = False,
    process_correction_version: int | None = None,
) -> None:
    """Validate a manifest without promoting or rewriting its evidence."""
    schema = manifest.get("schema")
    if schema == SCHEMA_V1:
        _require(
            allow_legacy_v1
            and (process_correction_version is None or process_correction_version < 2),
            "artifact manifest v1 cannot represent process-correction-v2 evidence",
        )
        expected_fields = MANIFEST_FIELDS_V1
    elif schema == SCHEMA_V2:
        expected_fields = MANIFEST_FIELDS_V2
    else:
        raise ManifestError("artifact manifest schema is invalid")
    _require(set(manifest) == expected_fields,
             "artifact manifest has missing or extra fields")
    _require(_is_digest(manifest.get("sha256")),
             "artifact manifest sha256 is invalid")
    _require(
        manifest["sha256"] == controller._canonical_digest(manifest),
        "artifact manifest self-digest is invalid",
    )
    for field in ("artifact", "command_envelope"):
        reference = manifest.get(field)
        _require(
            isinstance(reference, dict) and set(reference) == REFERENCE_FIELDS,
            f"artifact manifest {field} reference is invalid",
        )
        _require(isinstance(reference.get("path"), str) and reference["path"],
                 f"artifact manifest {field} path is invalid")
        _require(_is_digest(reference.get("sha256")),
                 f"artifact manifest {field} sha256 is invalid")
    receipt = manifest.get("execution_receipt")
    _require(
        isinstance(receipt, dict)
        and set(receipt) == controller.MANIFEST_RECEIPT_REFERENCE_FIELDS,
        "artifact manifest execution receipt reference is invalid",
    )
    _require(isinstance(receipt.get("path"), str) and receipt["path"],
             "artifact manifest execution receipt path is invalid")
    _require(_is_digest(receipt.get("sha256")),
             "artifact manifest execution receipt sha256 is invalid")
    _require(isinstance(receipt.get("host"), str) and receipt["host"].strip(),
             "artifact manifest execution receipt host is invalid")
    _require(isinstance(receipt.get("device"), str) and receipt["device"].strip(),
             "artifact manifest execution receipt device is invalid")
    started = receipt.get("started_utc_ns")
    ended = receipt.get("ended_utc_ns")
    _require(type(started) is int and type(ended) is int and started <= ended,
             "artifact manifest execution receipt timestamps are invalid")
    if schema == SCHEMA_V2:
        reference = manifest.get("provenance_sidecar")
        _require(
            isinstance(reference, dict) and set(reference) == REFERENCE_FIELDS,
            "artifact manifest provenance sidecar reference is invalid",
        )
        _require(isinstance(reference.get("path"), str) and reference["path"],
                 "artifact manifest provenance sidecar path is invalid")
        _require(_is_digest(reference.get("sha256")),
                 "artifact manifest provenance sidecar sha256 is invalid")


def load_manifest_read_only(
    path: str | Path,
    *,
    root: Path = ROOT,
    allow_legacy_v1: bool = False,
    process_correction_version: int | None = None,
) -> dict[str, Any]:
    """Read and structurally verify v1/v2 manifests without promoting evidence.

    Historical v1 requires an explicit read-only opt-in and is always rejected for
    a process-correction-v2 chain.
    """
    root = root.resolve(strict=True)
    _, manifest_path = _safe_repo_path(
        root, path, "artifact manifest path", must_exist=True
    )
    digest = _hash_regular_file(manifest_path, "artifact manifest")
    manifest = _load_json(manifest_path, "artifact manifest")
    _require(
        _hash_regular_file(manifest_path, "artifact manifest") == digest,
        "artifact manifest changed while being validated",
    )
    _validate_manifest_shape(
        manifest,
        allow_legacy_v1=allow_legacy_v1,
        process_correction_version=process_correction_version,
    )
    return manifest


def _expected_source(config: Mapping[str, Any], kind: str) -> str:
    return controller._expected_manifest_source(dict(config), kind)


def _expected_env(kind: str) -> dict[str, str]:
    return controller._expected_manifest_env(kind)


def _expected_argv(
    *, config: Mapping[str, Any], kind: str, root: Path, output: Path,
) -> list[str]:
    return controller._expected_manifest_argv(
        config=dict(config), kind=kind, root=root, output=output
    )


def _validate_envelope(
    envelope: dict[str, Any], *, envelope_path: Path, config_path: Path,
    config: dict[str, Any], kind: str, root: Path,
) -> tuple[Path, Path]:
    expected_fields = set(controller.COMMAND_FIELDS)
    if kind == "final_stage0":
        expected_fields.add("expected_result")
    _require(set(envelope) == expected_fields,
             "command envelope has missing or extra fields")
    _require(envelope.get("schema") == controller.COMMAND_SCHEMA,
             "command envelope schema is invalid")
    _require(envelope.get("action") == _ACTION_BY_KIND[kind],
             "command envelope action does not match manifest kind")
    _require(envelope.get("correction_id") == config["correction_id"],
             "command envelope correction ID differs from config")
    config_ref = envelope.get("config")
    _require(isinstance(config_ref, dict) and set(config_ref) == {"path", "sha256"},
             "command envelope config reference is invalid")
    _require(config_ref.get("sha256") == config["sha256"],
             "command envelope config digest differs from frozen config")
    try:
        envelope_config_path = Path(config_ref.get("path", "")).resolve(strict=True)
    except (OSError, TypeError) as exc:
        raise ManifestError("command envelope config path is invalid") from exc
    _require(envelope_config_path == config_path.resolve(),
             "command envelope names a different config path")

    expected_source = _expected_source(config, kind)
    _require(envelope.get("source_revision") == expected_source,
             "command envelope source revision is invalid for this kind")
    _require(envelope.get("execution") == "not_executed",
             "command envelope execution marker is invalid")
    _require(isinstance(envelope.get("prerequisites"), list),
             "command envelope prerequisites must be a list")

    try:
        cwd = Path(envelope.get("cwd", "")).resolve(strict=True)
    except (OSError, TypeError) as exc:
        raise ManifestError("command envelope cwd is invalid") from exc
    _require(cwd.is_dir(), "command envelope cwd is not a directory")
    if kind == "performance_baseline":
        _require(controller._git_head(cwd) == expected_source,
                 "performance baseline cwd is not the frozen legacy source")
        legacy = config["legacy_performance"]
        _, runner = controller._repo_path(cwd, legacy["runner_path"], "legacy runner")
        _require(runner.is_file() and controller._file_digest(runner) == legacy["runner_sha256"],
                 "performance baseline cwd has the wrong legacy runner")
    else:
        _require(cwd == root, "candidate command envelope cwd must equal repository root")

    artifact_relative = config["artifacts"][kind]
    _, output = _safe_repo_path(root, artifact_relative, f"{kind} artifact", must_exist=True)
    output = output.resolve(strict=True)
    _require(envelope.get("output") == str(output),
             "command envelope output is not the canonical artifact destination")
    expected_argv = _expected_argv(config=config, kind=kind, root=root, output=output)
    _require(envelope.get("argv") == expected_argv,
             "command envelope argv differs from the frozen command")
    _require(envelope.get("env") == _expected_env(kind),
             "command envelope environment differs from the frozen command")
    if kind == "final_stage0":
        _require(envelope.get("expected_result") == {
            "stage": "final_cross_backend", "outcome": "TONIC_OUTPUT_GO", "go": True,
        }, "command envelope expected result is invalid")
    _require(envelope_path.is_file(), "command envelope is missing")
    return cwd, output


def _artifact_backend(artifact: Mapping[str, Any]) -> str | None:
    backend = artifact.get("backend")
    if isinstance(backend, str) and backend:
        return backend
    backend_info = artifact.get("backend_info")
    if isinstance(backend_info, dict):
        backend = backend_info.get("backend")
        if isinstance(backend, str) and backend:
            return backend
    return None


def _artifact_source_revisions(artifact: Mapping[str, Any]) -> set[str]:
    revisions: set[str] = set()
    source = artifact.get("source_sha")
    if isinstance(source, str) and source:
        revisions.add(source.lower())
    sources = artifact.get("source_shas")
    if isinstance(sources, dict):
        revisions.update(
            value.lower() for value in sources.values()
            if isinstance(value, str) and value
        )
    return revisions


def _validate_sidecar(
    *,
    root: Path,
    cwd: Path,
    artifact_path: Path,
    artifact: dict[str, Any],
    envelope: dict[str, Any],
    receipt: dict[str, Any],
    kind: str,
) -> tuple[str, str]:
    sidecar_path = Path(f"{artifact_path}.prov.json")
    try:
        sidecar_relative = sidecar_path.relative_to(root).as_posix()
    except ValueError as exc:
        raise ManifestError("provenance sidecar is outside the repository") from exc
    _, sidecar_file = _safe_repo_path(
        root, sidecar_relative, "provenance sidecar", must_exist=True
    )
    sidecar_digest = _hash_regular_file(sidecar_file, "provenance sidecar")
    sidecar = _load_json(sidecar_file, "provenance sidecar")

    sidecar_artifact_relative, sidecar_artifact = _safe_repo_path(
        root, sidecar.get("artifact", ""), "provenance sidecar artifact", must_exist=True
    )
    del sidecar_artifact_relative
    _require(sidecar_artifact.resolve(strict=True) == artifact_path,
             "provenance sidecar artifact path differs from canonical artifact")

    runner_relative = controller.RUNNER_MODULE.replace(".", "/") + ".py"
    _require(sidecar.get("runner") == runner_relative,
             "provenance sidecar runner differs from frozen runner")
    sidecar_argv = sidecar.get("argv")
    _require(
        isinstance(sidecar_argv, list)
        and all(isinstance(item, str) for item in sidecar_argv)
        and bool(sidecar_argv),
        "provenance sidecar argv is invalid",
    )
    expected_argv = envelope["argv"]
    _require(
        len(expected_argv) >= 3
        and expected_argv[1:3] == ["-m", controller.RUNNER_MODULE],
        "command envelope does not invoke the frozen runner module",
    )
    runner_input = Path(sidecar_argv[0])
    runner_path = runner_input if runner_input.is_absolute() else cwd / runner_input
    try:
        runner_path = runner_path.resolve(strict=True)
        expected_runner = (cwd / runner_relative).resolve(strict=True)
    except OSError as exc:
        raise ManifestError("provenance sidecar argv runner path is invalid") from exc
    _require(runner_path == expected_runner,
             "provenance sidecar argv names a different runner")
    _require(sidecar_argv[1:] == expected_argv[3:],
             "provenance sidecar argv differs from command envelope")
    _require(receipt.get("argv") == expected_argv,
             "provenance sidecar cannot bind a receipt with different argv")

    expected_source = envelope["source_revision"].lower()
    sidecar_source = sidecar.get("git_sha")
    _require(
        isinstance(sidecar_source, str)
        and 7 <= len(sidecar_source) <= 40
        and expected_source.startswith(sidecar_source.lower()),
        "provenance sidecar source revision differs from command envelope",
    )
    _require(receipt.get("source", {}).get("git_sha") == expected_source,
             "provenance sidecar cannot bind a receipt from another source revision")
    artifact_sources = _artifact_source_revisions(artifact)
    _require(artifact_sources or kind == "final_stage0",
             "artifact lacks a source revision binding")
    for source in artifact_sources:
        _require(source == expected_source,
                 "artifact source revision differs from provenance sidecar")

    expected_backend = envelope["env"].get("SIM_BACKEND")
    receipt_backend = receipt.get("env_allowlist", {}).get("SIM_BACKEND")
    sidecar_env = sidecar.get("env")
    _require(isinstance(sidecar_env, dict),
             "provenance sidecar environment is invalid")
    sidecar_env_backend = sidecar_env.get("SIM_BACKEND")
    sidecar_backend = sidecar.get("sim_backend")
    sidecar_requested = sidecar.get("sim_backend_requested")
    _require(sidecar_backend in {"numpy", "cupy"},
             "provenance sidecar backend is invalid")
    _require(isinstance(sidecar_requested, str) and sidecar_requested,
             "provenance sidecar requested backend is invalid")
    for value, label in (
        (receipt_backend, "execution receipt"),
        (sidecar_env_backend, "provenance sidecar environment"),
        (sidecar_backend, "provenance sidecar backend"),
        (sidecar_requested, "provenance sidecar requested backend"),
    ):
        if expected_backend is not None:
            _require(value == expected_backend,
                     f"{label} differs from command envelope backend")
    artifact_backend = _artifact_backend(artifact)
    _require(artifact_backend is not None, "artifact lacks a backend binding")
    if expected_backend is not None:
        _require(artifact_backend == expected_backend,
                 "artifact backend differs from provenance sidecar")
    elif artifact_backend is not None:
        _require(artifact_backend == "cross_backend",
                 f"{kind} artifact has an unexpected unsealed backend")

    _require(
        _hash_regular_file(sidecar_file, "provenance sidecar") == sidecar_digest,
        "provenance sidecar changed while being validated",
    )
    return sidecar_relative, sidecar_digest


def _write_create_only(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("ascii")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o644)
    except FileExistsError as exc:
        raise ManifestError(f"refusing to overwrite manifest: {path}") from exc
    except OSError as exc:
        raise ManifestError(f"cannot create manifest: {path}: {exc}") from exc
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


def create_manifest(
    *, root: Path, config_path: str | Path, envelope_path: str | Path,
    receipt_path: str | Path, kind: str, emit: str | Path,
) -> dict[str, Any]:
    """Validate one completed command and emit its controller-consumable manifest."""
    root = root.resolve(strict=True)
    _require(root.is_dir(), "repository root is not a directory")
    _require(kind in KINDS, f"unsupported Stage-0 artifact kind: {kind}")
    config_relative, config_file = _safe_repo_path(
        root, config_path, "config path", must_exist=True
    )
    envelope_relative, envelope_file = _safe_repo_path(
        root, envelope_path, "command envelope path", must_exist=True
    )
    emit_relative, emit_file = _safe_repo_path(root, emit, "manifest output path")
    _require(len({config_relative, envelope_relative, emit_relative}) == 3,
             "config, envelope, and manifest paths must be distinct")

    try:
        config = controller.load_config(config_file, root=root)
    except controller.ControllerError as exc:
        raise ManifestError(f"frozen controller config is invalid: {exc}") from exc
    envelope_digest = _hash_regular_file(envelope_file, "command envelope")
    envelope = _load_json(envelope_file, "command envelope")
    _require(
        _hash_regular_file(envelope_file, "command envelope") == envelope_digest,
        "command envelope changed while being validated",
    )
    cwd, artifact_path = _validate_envelope(
        envelope,
        envelope_path=envelope_file,
        config_path=config_file,
        config=config,
        kind=kind,
        root=root,
    )

    receipt_relative, receipt_file = _safe_repo_path(
        cwd, receipt_path, "execution receipt path", must_exist=True
    )
    receipt_digest = _hash_regular_file(receipt_file, "execution receipt")
    try:
        receipt = execution_receipt.verify_receipt(cwd, receipt_relative)
    except execution_receipt.ReceiptError as exc:
        raise ManifestError(f"execution receipt is invalid: {exc}") from exc
    _require(
        _hash_regular_file(receipt_file, "execution receipt") == receipt_digest,
        "execution receipt changed while being validated",
    )

    _require(receipt.get("argv") == envelope["argv"],
             "execution receipt argv differs from command envelope")
    _require(receipt.get("env_allowlist") == envelope["env"],
             "execution receipt environment differs from command envelope")
    _require(receipt.get("source", {}).get("git_sha") == envelope["source_revision"],
             "execution receipt source revision differs from command envelope")
    if kind != "performance_baseline":
        try:
            controller._require_candidate_receipt_source(
                receipt.get("source"), config=config, label="execution receipt"
            )
        except controller.ControllerError as exc:
            raise ManifestError(str(exc)) from exc
    _require(isinstance(receipt.get("host"), str) and bool(receipt["host"].strip()),
             "execution receipt lacks an explicit host")
    _require(isinstance(receipt.get("device"), str) and bool(receipt["device"].strip()),
             "execution receipt lacks an explicit device")
    started = receipt.get("started_utc_ns")
    ended = receipt.get("ended_utc_ns")
    _require(type(started) is int and type(ended) is int and started <= ended,
             "execution receipt timestamps are not ordered")

    receipt_artifact = receipt.get("artifact")
    _require(isinstance(receipt_artifact, dict), "execution receipt artifact is invalid")
    receipt_artifact_relative, receipt_artifact_path = _safe_repo_path(
        cwd, receipt_artifact.get("path", ""), "receipt artifact path", must_exist=True
    )
    del receipt_artifact_relative
    _require(receipt_artifact_path.resolve(strict=True) == artifact_path,
             "execution receipt artifact path differs from canonical destination")
    artifact_digest = _hash_regular_file(artifact_path, "Stage-0 artifact")
    _require(receipt_artifact.get("sha256") == artifact_digest,
             "execution receipt artifact digest differs from canonical artifact")
    artifact = _load_json(artifact_path, "Stage-0 artifact")
    _require(
        _hash_regular_file(artifact_path, "Stage-0 artifact") == artifact_digest,
        "Stage-0 artifact changed while being validated",
    )
    sidecar_relative, sidecar_digest = _validate_sidecar(
        root=root,
        cwd=cwd,
        artifact_path=artifact_path,
        artifact=artifact,
        envelope=envelope,
        receipt=receipt,
        kind=kind,
    )

    manifest = {
        "schema": SCHEMA,
        "kind": kind,
        "config_sha256": config["sha256"],
        "source_revision": _expected_source(config, kind),
        "artifact": {
            "path": config["artifacts"][kind],
            "sha256": artifact_digest,
        },
        "command_envelope": {
            "path": envelope_relative,
            "sha256": envelope_digest,
        },
        "provenance_sidecar": {
            "path": sidecar_relative,
            "sha256": sidecar_digest,
        },
        "execution_receipt": {
            "path": receipt_relative,
            "sha256": receipt_digest,
            "host": receipt["host"],
            "device": receipt["device"],
            "started_utc_ns": started,
            "ended_utc_ns": ended,
        },
    }
    manifest["sha256"] = controller._canonical_digest(manifest)
    _validate_manifest_shape(manifest, process_correction_version=2)
    _write_create_only(emit_file, manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--config", required=True)
    parser.add_argument("--envelope", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--kind", choices=KINDS, required=True)
    parser.add_argument("--emit", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = create_manifest(
            root=args.root,
            config_path=args.config,
            envelope_path=args.envelope,
            receipt_path=args.receipt,
            kind=args.kind,
            emit=args.emit,
        )
    except ManifestError as exc:
        print(f"v13-stage0-manifest: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "artifact": manifest["artifact"]["path"],
        "kind": manifest["kind"],
        "manifest": args.emit,
        "sha256": manifest["sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
