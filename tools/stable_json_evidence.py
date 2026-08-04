"""Read JSON evidence and bind parsing to one stable exact-byte snapshot."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any


CANONICALIZATION = "python-json-sort-keys-compact-separators-utf8-v1"


class StableJsonEvidenceError(ValueError):
    """Raised when a path cannot provide stable JSON evidence."""


@dataclass(frozen=True)
class StableJsonEvidence:
    """A parsed JSON value and digests derived from its single byte snapshot."""

    path: Path
    value: Any
    raw_bytes: bytes
    file_sha256: str
    canonical_json_sha256: str
    canonicalization: str = CANONICALIZATION


def _file_state(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


class _NonStandardJsonConstant(ValueError):
    pass


def _reject_nonstandard_constant(value: str) -> None:
    raise _NonStandardJsonConstant(value)


def _parse_json(raw_bytes: bytes) -> Any:
    return json.loads(raw_bytes, parse_constant=_reject_nonstandard_constant)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def read_stable_json_evidence(
    path: str | os.PathLike[str],
    *,
    require_object: bool = False,
) -> StableJsonEvidence:
    """Read, parse, and digest a JSON file while rejecting path or file mutation."""

    evidence_path = Path(path)
    if not hasattr(os, "O_NOFOLLOW"):
        raise StableJsonEvidenceError("O_NOFOLLOW is unavailable on this platform")

    flags = os.O_RDONLY | os.O_NOFOLLOW
    flags |= getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(evidence_path, flags)
    except OSError as exc:
        raise StableJsonEvidenceError(
            f"cannot open JSON evidence without following symlinks: {evidence_path}: {exc}"
        ) from exc

    try:
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise StableJsonEvidenceError(
                    f"JSON evidence is not a regular file: {evidence_path}"
                )

            with os.fdopen(descriptor, "rb", closefd=False) as handle:
                raw_bytes = handle.read()

            try:
                value = _parse_json(raw_bytes)
            except (json.JSONDecodeError, UnicodeDecodeError, _NonStandardJsonConstant) as exc:
                raise StableJsonEvidenceError(
                    f"JSON evidence is invalid: {evidence_path}: {exc}"
                ) from exc

            if require_object and not isinstance(value, dict):
                raise StableJsonEvidenceError(
                    f"JSON evidence must be an object: {evidence_path}"
                )

            file_sha256 = hashlib.sha256(raw_bytes).hexdigest()
            canonical_json_sha256 = hashlib.sha256(
                _canonical_json_bytes(value)
            ).hexdigest()
            after = os.fstat(descriptor)
        except OSError as exc:
            raise StableJsonEvidenceError(
                f"cannot read stable JSON evidence: {evidence_path}: {exc}"
            ) from exc

        if _file_state(before) != _file_state(after):
            raise StableJsonEvidenceError(
                f"JSON evidence changed while being parsed: {evidence_path}"
            )

        try:
            named = evidence_path.lstat()
        except OSError as exc:
            raise StableJsonEvidenceError(
                f"JSON evidence disappeared or changed after parsing: {evidence_path}: {exc}"
            ) from exc
        if stat.S_ISLNK(named.st_mode) or not stat.S_ISREG(named.st_mode):
            raise StableJsonEvidenceError(
                f"JSON evidence pathname is not the opened regular file: {evidence_path}"
            )
        if _file_state(named) != _file_state(after):
            raise StableJsonEvidenceError(
                f"JSON evidence pathname changed while being parsed: {evidence_path}"
            )

        return StableJsonEvidence(
            path=evidence_path,
            value=value,
            raw_bytes=raw_bytes,
            file_sha256=file_sha256,
            canonical_json_sha256=canonical_json_sha256,
        )
    finally:
        os.close(descriptor)
