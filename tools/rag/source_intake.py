#!/usr/bin/env python3
"""Durably register externally found sources in the canonical RAG catalog."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any

try:
    from .rag_paths import resolve_paths
except ImportError:  # direct script execution
    from rag_paths import resolve_paths


ARCHIVABLE_LICENSES = {"open-access", "public-domain", "permission-granted"}
LICENSE_STATUSES = ARCHIVABLE_LICENSES | {"metadata-only"}
SAFE_NAME_RE = re.compile(r"[^a-z0-9]+")


class SourceIntakeError(RuntimeError):
    pass


def catalog_path(repo: Path) -> Path:
    override = os.environ.get("SIM_CATALOG")
    if override:
        return Path(override).expanduser().resolve()
    return resolve_paths(repo).catalog


def _slug(value: str) -> str:
    text = SAFE_NAME_RE.sub("-", value.lower()).strip("-")
    return text[:64] or "source"


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _append_ledger(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        existing: list[dict[str, Any]] = []
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    existing.append(json.loads(line))
        if not any(item.get("intake_id") == record["intake_id"] for item in existing):
            existing.append(record)
        payload = "".join(json.dumps(item, sort_keys=True) + "\n" for item in existing)
        _atomic_write(path, payload)


def _render_record(record: dict[str, Any]) -> str:
    archived = record.get("archived_path") or "Not archived; metadata and evidence excerpt only."
    return (
        f"# External source intake: {record['citation']}\n\n"
        f"- Intake ID: `{record['intake_id']}`\n"
        f"- Source URL: {record['url']}\n"
        f"- Source kind: `{record['kind']}`\n"
        f"- License status: `{record['license_status']}`\n"
        f"- Accessed: {record['accessed_at']}\n"
        f"- Questions: {', '.join(record['questions'])}\n"
        f"- Search query: {record['query']}\n"
        f"- Exact locator: {record['locator']}\n"
        f"- Archived copy: {archived}\n\n"
        "## Evidence relevant to the research gate\n\n"
        f"{record['evidence']}\n\n"
        "This record preserves discovery provenance. It is not a substitute for reading the cited source.\n"
    )


def register_source(
    repo: Path,
    *,
    citation: str,
    url: str,
    kind: str,
    license_status: str,
    accessed_at: str,
    questions: list[str],
    query: str,
    locator: str,
    evidence: str,
    local_file: str | None = None,
) -> dict[str, Any]:
    """Create an idempotent catalog record and optionally archive a permitted local copy."""
    if license_status not in LICENSE_STATUSES:
        raise SourceIntakeError(f"unknown license status: {license_status}")
    if local_file and license_status not in ARCHIVABLE_LICENSES:
        raise SourceIntakeError(
            "a local copy requires open-access, public-domain, or permission-granted licensing"
        )
    identity = hashlib.sha256(f"{url}\n{citation}".encode("utf-8")).hexdigest()[:16]
    intake_id = f"source-{identity}"
    catalog = catalog_path(repo)
    catalog.mkdir(parents=True, exist_ok=True)
    record_path = catalog / f"{intake_id}-{_slug(citation)}.md"
    archived_path: Path | None = None
    if local_file:
        source = Path(local_file).expanduser().resolve()
        if not source.is_file():
            raise SourceIntakeError(f"local source copy does not exist: {source}")
        archive_dir = catalog / "textbooks" / "external-intake"
        archive_dir.mkdir(parents=True, exist_ok=True)
        suffix = source.suffix.lower() or ".bin"
        archived_path = archive_dir / f"{intake_id}{suffix}"
        if archived_path.exists():
            if hashlib.sha256(archived_path.read_bytes()).digest() != hashlib.sha256(source.read_bytes()).digest():
                raise SourceIntakeError(f"intake ID collision at {archived_path}")
        else:
            temporary = archived_path.with_name(f".{archived_path.name}.{os.getpid()}.tmp")
            shutil.copyfile(source, temporary)
            os.replace(temporary, archived_path)

    record: dict[str, Any] = {
        "schema": 1,
        "intake_id": intake_id,
        "citation": citation,
        "url": url,
        "kind": kind,
        "license_status": license_status,
        "accessed_at": accessed_at,
        "questions": questions,
        "query": query,
        "locator": locator,
        "evidence": evidence,
        "record_path": str(record_path),
        "archived_path": str(archived_path) if archived_path else None,
    }
    if archived_path:
        record["archive_sha256"] = hashlib.sha256(archived_path.read_bytes()).hexdigest()
    _atomic_write(record_path, _render_record(record))
    _append_ledger(catalog / "source-intake.jsonl", record)
    return record
