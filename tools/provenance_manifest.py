#!/usr/bin/env python3
"""Create a deterministic, non-ledger provenance manifest for experiment artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence


SCHEMA = "sim-provenance-manifest-v1"
SIDECAR_SUFFIX = ".prov.json"


class ManifestError(ValueError):
    """Raised when provenance evidence is incomplete or contradictory."""


def _same_file_state(before: os.stat_result, after: os.stat_result) -> bool:
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    return all(getattr(before, field) == getattr(after, field) for field in fields)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
            after = os.fstat(handle.fileno())
    except OSError as exc:
        raise ManifestError(f"cannot hash {path}: {exc}") from exc
    if not _same_file_state(before, after):
        raise ManifestError(f"file changed while hashing: {path}")
    return digest.hexdigest()


def _read_bytes(path: Path) -> bytes:
    try:
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            data = handle.read()
            after = os.fstat(handle.fileno())
    except OSError as exc:
        raise ManifestError(f"cannot read {path}: {exc}") from exc
    if not _same_file_state(before, after):
        raise ManifestError(f"file changed while reading: {path}")
    return data


def _load_object(data: bytes, path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestError(f"malformed {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ManifestError(f"malformed {label} {path}: expected a JSON object")
    return value


def _required_text(value: Any, field: str, sidecar: Path) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"malformed sidecar {sidecar}: {field} must be a non-empty string")
    return value


def _first_text(record: dict[str, Any], names: Sequence[str]) -> str | None:
    for name in names:
        value = record.get(name)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _repository_root(path: Path) -> Path | None:
    for candidate in (path, *path.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _validate_declared_path(
    artifact: Path,
    artifact_root: Path,
    declared: Any,
    sidecar: Path,
) -> None:
    text = _required_text(declared, "artifact", sidecar)
    declared_path = Path(text).expanduser()
    bases = [artifact_root, Path.cwd().resolve()]
    repository_root = _repository_root(artifact_root)
    if repository_root is not None:
        bases.append(repository_root)

    if declared_path.is_absolute():
        candidates = [declared_path.resolve()]
    else:
        candidates = [(base / declared_path).resolve() for base in bases]
    if artifact.resolve() not in candidates:
        raise ManifestError(
            f"artifact path mismatch in {sidecar}: declared {text!r}, selected {artifact}"
        )


def _source(sidecar_data: dict[str, Any], sidecar: Path) -> Any:
    explicit = sidecar_data.get("source")
    if isinstance(explicit, str) and explicit.strip():
        return explicit
    if isinstance(explicit, dict) and explicit:
        return explicit
    revision = _first_text(sidecar_data, ("git_sha", "source_commit", "source_revision"))
    if revision is None:
        raise ManifestError(f"malformed sidecar {sidecar}: missing source identity")
    source: dict[str, Any] = {"git_sha": revision}
    optional = {
        "dirty": sidecar_data.get("git_dirty"),
        "kind": sidecar_data.get("source_kind"),
        "manifest_sha256": sidecar_data.get("source_manifest_sha256"),
    }
    for key, value in optional.items():
        if value is not None:
            source[key] = value
    return source


def _command(sidecar_data: dict[str, Any], sidecar: Path) -> str | list[str]:
    value = sidecar_data.get("command", sidecar_data.get("argv"))
    if isinstance(value, str) and value.strip():
        return value
    if (
        isinstance(value, list)
        and value
        and all(isinstance(item, str) and item for item in value)
    ):
        return value
    raise ManifestError(f"malformed sidecar {sidecar}: command/argv is missing or invalid")


def _timestamp(sidecar_data: dict[str, Any], sidecar: Path) -> str:
    value = _first_text(sidecar_data, ("timestamp", "started", "started_utc"))
    if value is None:
        raise ManifestError(f"malformed sidecar {sidecar}: missing timestamp")
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ManifestError(f"malformed sidecar {sidecar}: invalid timestamp {value!r}") from exc
    return value


def _artifact_metadata(data: bytes) -> dict[str, Any]:
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _device(sidecar_data: dict[str, Any], artifact_data: dict[str, Any], sidecar: Path) -> str:
    value = _first_text(sidecar_data, ("device",))
    if value is None:
        value = _first_text(artifact_data, ("device",))
    if value is None and isinstance(artifact_data.get("runtime"), dict):
        value = _first_text(artifact_data["runtime"], ("device",))
    if value is None:
        raise ManifestError(f"malformed provenance for {sidecar}: missing device identity")
    return value


def _record(artifact: Path, artifact_root: Path) -> dict[str, Any]:
    sidecar = Path(str(artifact) + SIDECAR_SUFFIX)
    if not sidecar.is_file():
        raise ManifestError(f"missing provenance sidecar for {artifact}: expected {sidecar}")
    sidecar_bytes = _read_bytes(sidecar)
    sidecar_data = _load_object(sidecar_bytes, sidecar, "sidecar")
    _validate_declared_path(artifact, artifact_root, sidecar_data.get("artifact"), sidecar)

    backend = _first_text(sidecar_data, ("backend", "sim_backend"))
    if backend is None:
        raise ManifestError(f"malformed sidecar {sidecar}: missing backend")
    artifact_data: dict[str, Any] = {}
    if _first_text(sidecar_data, ("device",)) is None:
        artifact_bytes = _read_bytes(artifact)
        artifact_sha256 = hashlib.sha256(artifact_bytes).hexdigest()
        artifact_data = _artifact_metadata(artifact_bytes)
    else:
        artifact_sha256 = _sha256(artifact)
    relative_artifact = artifact.relative_to(artifact_root).as_posix()
    relative_sidecar = sidecar.relative_to(artifact_root).as_posix()
    return {
        "artifact": relative_artifact,
        "artifact_sha256": artifact_sha256,
        "backend": backend,
        "command": _command(sidecar_data, sidecar),
        "device": _device(sidecar_data, artifact_data, sidecar),
        "run_id": _required_text(sidecar_data.get("run_id"), "run_id", sidecar),
        "sidecar": relative_sidecar,
        "sidecar_sha256": hashlib.sha256(sidecar_bytes).hexdigest(),
        "source": _source(sidecar_data, sidecar),
        "timestamp": _timestamp(sidecar_data, sidecar),
    }


def select_artifacts(
    artifact_root: Path,
    artifacts: Sequence[str] = (),
    patterns: Sequence[str] = (),
    output: Path | None = None,
) -> list[Path]:
    root = artifact_root.resolve()
    if not root.is_dir():
        raise ManifestError(f"artifact directory does not exist: {artifact_root}")

    selected: list[Path] = []
    if artifacts or patterns:
        selected.extend(root / name for name in artifacts)
        for pattern in patterns:
            if not pattern:
                raise ManifestError("artifact glob must not be empty")
            matches = sorted(
                path for path in root.glob(pattern) if not path.name.endswith(SIDECAR_SUFFIX)
            )
            if not matches:
                raise ManifestError(f"artifact glob matched no artifact files: {pattern!r}")
            selected.extend(matches)
    else:
        selected.extend(path for path in root.rglob("*") if path.is_file())

    output_resolved = output.resolve() if output is not None else None
    result: list[Path] = []
    seen: set[Path] = set()
    for candidate in selected:
        resolved = candidate.resolve()
        if output_resolved is not None and resolved == output_resolved:
            continue
        if not _is_within(resolved, root):
            raise ManifestError(f"selected artifact escapes artifact directory: {candidate}")
        if not resolved.is_file():
            raise ManifestError(f"selected artifact is not a file: {candidate}")
        if resolved.name.endswith(SIDECAR_SUFFIX):
            if artifacts or patterns:
                raise ManifestError(
                    f"selected path is a provenance sidecar, not an artifact: {candidate}"
                )
            continue
        if resolved in seen:
            raise ManifestError(f"duplicate artifact record selected: {resolved.relative_to(root)}")
        seen.add(resolved)
        result.append(resolved)

    if not result:
        raise ManifestError(f"no artifacts selected under {artifact_root}")
    return sorted(result, key=lambda path: path.relative_to(root).as_posix())


def build_manifest(artifact_root: Path, selected: Sequence[Path]) -> dict[str, Any]:
    root = artifact_root.resolve()
    records = [_record(path.resolve(), root) for path in selected]
    identities: set[tuple[str, str]] = set()
    for record in records:
        identity = (record["run_id"], record["artifact"])
        if identity in identities:
            raise ManifestError(
                f"duplicate provenance record: run_id={identity[0]} artifact={identity[1]}"
            )
        identities.add(identity)
    return {
        "artifact_root": ".",
        "record_count": len(records),
        "records": records,
        "schema": SCHEMA,
    }


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    payload = (
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as exc:
        raise ManifestError(f"refusing to overwrite existing manifest: {path}") from exc
    except OSError as exc:
        raise ManifestError(f"cannot create manifest {path}: {exc}") from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
    except Exception:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_dir", type=Path, help="directory containing experiment artifacts")
    parser.add_argument(
        "--output", type=Path, required=True, help="new manifest path (must not exist)"
    )
    parser.add_argument(
        "--artifact",
        action="append",
        default=[],
        help="artifact path relative to ARTIFACT_DIR; repeat to select multiple files",
    )
    parser.add_argument(
        "--glob",
        dest="patterns",
        action="append",
        default=[],
        help="glob relative to ARTIFACT_DIR; repeat to select multiple groups",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        selected = select_artifacts(args.artifact_dir, args.artifact, args.patterns, args.output)
        manifest = build_manifest(args.artifact_dir, selected)
        write_manifest(args.output, manifest)
    except ManifestError as exc:
        parser.error(str(exc))
    print(f"WROTE {args.output} ({manifest['record_count']} records)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
