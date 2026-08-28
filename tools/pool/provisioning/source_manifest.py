#!/usr/bin/env python3
"""Create and verify complete immutable-source manifests for pool archives."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path, PurePosixPath
from typing import Sequence


CONTROL_FILES = frozenset({
    ".source_manifest.sha256", ".source_revision", ".pool_environment.json",
})
IGNORED_PARTS = frozenset({".git", "__pycache__"})
RUNTIME_PREFIXES = (
    PurePosixPath("research/experiment-runtime"),
    PurePosixPath("research/findings/raw"),
    PurePosixPath("data"),                 # runtime data/corpus/models — gitignored, needed by runners, NOT source
)
# Runtime output accretions on a live pool node (dispatch logs, job status) — never source.
RUNTIME_SUFFIXES = frozenset({".out", ".log"})


class SourceManifestError(ValueError):
    """Raised when an archive source tree is incomplete, widened, or modified."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ignored(relative: PurePosixPath) -> bool:
    if relative.as_posix() in CONTROL_FILES:
        return True
    if any(part in IGNORED_PARTS for part in relative.parts):
        return True
    if relative.parts and relative.parts[0].startswith(".venv"):
        return True
    if relative.suffix in RUNTIME_SUFFIXES:
        return True
    return any(relative == prefix or prefix in relative.parents for prefix in RUNTIME_PREFIXES)


def enumerate_source_files(root: str | Path) -> dict[str, Path]:
    """Return every immutable source file; reject source symlinks and special files."""

    root_path = Path(root).expanduser().resolve(strict=True)
    files: dict[str, Path] = {}
    for directory, names, filenames in os.walk(root_path, topdown=True, followlinks=False):
        current = Path(directory)
        relative_directory = PurePosixPath(current.relative_to(root_path).as_posix())
        kept = []
        for name in names:
            relative = relative_directory / name
            path = current / name
            if _ignored(relative):
                continue
            if path.is_symlink():
                raise SourceManifestError(f"source tree contains a symbolic-link directory: {relative}")
            kept.append(name)
        names[:] = kept
        for name in filenames:
            path = current / name
            relative = PurePosixPath(path.relative_to(root_path).as_posix())
            if _ignored(relative):
                continue
            if path.is_symlink() or not path.is_file():
                raise SourceManifestError(f"source tree contains a non-regular file: {relative}")
            files[relative.as_posix()] = path
    if not files:
        raise SourceManifestError("source tree contains no immutable files")
    return dict(sorted(files.items()))


def manifest_bytes(root: str | Path) -> bytes:
    return "".join(
        f"{_sha256(path)}  {relative}\n"
        for relative, path in enumerate_source_files(root).items()
    ).encode("ascii")


def write_manifest(root: str | Path, output: str | Path) -> str:
    payload = manifest_bytes(root)
    destination = Path(output).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise SourceManifestError("refusing to replace source manifest")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(payload).hexdigest()


def parse_manifest(payload: bytes) -> dict[str, str]:
    try:
        lines = payload.decode("ascii").splitlines()
    except UnicodeDecodeError as exc:
        raise SourceManifestError("source manifest must be ASCII") from exc
    if not lines:
        raise SourceManifestError("source manifest is empty")
    entries: dict[str, str] = {}
    for number, line in enumerate(lines, 1):
        digest, separator, path_text = line.partition("  ")
        path = PurePosixPath(path_text)
        if (
            not separator or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or path.is_absolute() or not path.name
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise SourceManifestError(f"source manifest has an invalid row {number}")
        normalized = path.as_posix()
        if normalized in entries:
            raise SourceManifestError(f"source manifest duplicates {normalized}")
        entries[normalized] = digest
    return entries


def verify_manifest(
    root: str | Path, manifest: str | Path, expected_sha256: str,
) -> dict[str, str | int | bool]:
    root_path = Path(root).expanduser().resolve(strict=True)
    manifest_path = Path(manifest).expanduser().resolve(strict=True)
    payload = manifest_path.read_bytes()
    actual_manifest_sha = hashlib.sha256(payload).hexdigest()
    if actual_manifest_sha != expected_sha256:
        raise SourceManifestError("source manifest file digest does not match")
    expected = parse_manifest(payload)
    actual = enumerate_source_files(root_path)
    if set(expected) != set(actual):
        missing = sorted(set(expected) - set(actual))[:5]
        extra = sorted(set(actual) - set(expected))[:5]
        raise SourceManifestError(
            f"source file set differs from manifest; missing={missing}, extra={extra}"
        )
    for relative, expected_digest in expected.items():
        if _sha256(actual[relative]) != expected_digest:
            raise SourceManifestError(f"source digest mismatch: {relative}")
    return {
        "source_manifest_verified": True,
        "source_manifest_sha256": actual_manifest_sha,
        "source_file_count": len(expected),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    create = sub.add_parser("create")
    create.add_argument("--root", required=True)
    create.add_argument("--output", required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--root", required=True)
    verify.add_argument("--manifest", required=True)
    verify.add_argument("--expected-sha256", required=True)
    args = parser.parse_args(argv)
    try:
        if args.action == "create":
            print(write_manifest(args.root, args.output))
        else:
            print(verify_manifest(args.root, args.manifest, args.expected_sha256))
    except (OSError, SourceManifestError, ValueError) as exc:
        parser.exit(2, f"source manifest failure: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
