#!/usr/bin/env python3
"""Durably register externally found sources in the canonical RAG catalog."""
from __future__ import annotations

import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any
from urllib.parse import urlparse

try:
    from .rag_paths import resolve_paths
except ImportError:  # direct script execution
    from rag_paths import resolve_paths


ARCHIVABLE_LICENSES = {"open-access", "public-domain", "permission-granted"}
LICENSE_STATUSES = ARCHIVABLE_LICENSES | {"metadata-only"}
SOURCE_KINDS = {"peer-reviewed-primary", "primary-preprint", "review", "secondary"}
SAFE_NAME_RE = re.compile(r"[^a-z0-9]+")
PARAMETER_CLAIM_FIELDS = (
    "id",
    "units",
    "condition",
    "species",
    "preparation",
    "uncertainty",
    "locator",
    "limitations",
)


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


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SourceIntakeError(f"{field} must be a non-empty string")
    return value.strip()


def _validate_parameter_claims(claims: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Preserve explicit reviews; this intake layer never changes claim status."""
    if claims is None:
        return []
    if not isinstance(claims, list):
        raise SourceIntakeError("parameter_claims must be a list")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for claim in claims:
        if not isinstance(claim, dict):
            raise SourceIntakeError("parameter_claims entries must be objects")
        item = dict(claim)
        for field in PARAMETER_CLAIM_FIELDS:
            item[field] = _required_text(item.get(field), f"parameter_claims.{field}")
        if item["id"] in seen:
            raise SourceIntakeError(f"duplicate parameter claim id: {item['id']}")
        seen.add(item["id"])
        if item.get("value") is None:
            raise SourceIntakeError("parameter_claims.value is required")
        source_ids = item.get("source_ids")
        if not isinstance(source_ids, list) or not source_ids or not all(
            isinstance(source_id, str) and source_id.strip() for source_id in source_ids
        ):
            raise SourceIntakeError("parameter_claims.source_ids must contain source IDs")
        item["source_ids"] = [source_id.strip() for source_id in source_ids]
        if item.get("status") != "accepted":
            raise SourceIntakeError("only explicitly accepted parameter claims may be intaken")
        review = item.get("review")
        if not isinstance(review, dict) or review.get("decision") != "approved":
            raise SourceIntakeError("accepted parameter claims require an approving review")
        item["review"] = dict(review)
        item["review"]["reviewer"] = _required_text(
            review.get("reviewer"), "parameter_claims.review.reviewer"
        )
        item["review"]["reviewed_at"] = _required_text(
            review.get("reviewed_at"), "parameter_claims.review.reviewed_at"
        )
        try:
            dt.date.fromisoformat(item["review"]["reviewed_at"])
        except ValueError as exc:
            raise SourceIntakeError(
                "parameter_claims.review.reviewed_at must be YYYY-MM-DD"
            ) from exc
        normalized.append(item)
    return normalized


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
        matching = [item for item in existing if item.get("intake_id") == record["intake_id"]]
        if matching and matching[0] != record:
            raise SourceIntakeError(f"intake ID collision in ledger: {record['intake_id']}")
        if not matching:
            existing.append(record)
        payload = "".join(json.dumps(item, sort_keys=True) + "\n" for item in existing)
        _atomic_write(path, payload)


def _render_record(record: dict[str, Any]) -> str:
    archived = record.get("archived_path") or "Not archived; metadata and evidence excerpt only."
    text = (
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
    )
    provenance = record.get("packet_provenance")
    if provenance:
        text += (
            "## Reviewed packet provenance\n\n"
            f"- Packet path: `{provenance['packet_path']}`\n"
            f"- Packet SHA-256: `{provenance['packet_sha256']}`\n"
            f"- Packet version: `{provenance['packet_version']}`\n"
            f"- Packet question: `{provenance['question_id']}`\n\n"
        )
    claims = record.get("parameter_claims", [])
    if claims:
        text += "## Explicitly reviewed parameter claims\n\n"
        for claim in claims:
            value = (
                claim["value"]
                if isinstance(claim["value"], str)
                else json.dumps(claim["value"], sort_keys=True)
            )
            text += (
                f"### {claim['id']}\n\n"
                f"- Value: {value} {claim['units']}\n"
                f"- Condition: {claim['condition']}\n"
                f"- Species: {claim['species']}\n"
                f"- Preparation: {claim['preparation']}\n"
                f"- Exact claim locator: {claim['locator']}\n"
                f"- Uncertainty: {claim['uncertainty']}\n"
                f"- Limitations: {claim['limitations']}\n"
                f"- Review: explicitly approved by {claim['review']['reviewer']} "
                f"on {claim['review']['reviewed_at']}\n\n"
            )
    return text + (
        "This record preserves discovery provenance and prior review decisions. "
        "Catalog intake does not independently accept a scientific claim and is not a substitute "
        "for reading the cited source.\n"
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
    parameter_claims: list[dict[str, Any]] | None = None,
    packet_provenance: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Create an idempotent catalog record and optionally archive a permitted local copy."""
    citation = _required_text(citation, "citation")
    url = _required_text(url, "url")
    parsed_url = urlparse(url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise SourceIntakeError("url must be an http(s) URL")
    kind = _required_text(kind, "kind")
    if kind not in SOURCE_KINDS:
        raise SourceIntakeError(f"unknown source kind: {kind}")
    license_status = _required_text(license_status, "license_status")
    accessed_at = _required_text(accessed_at, "accessed_at")
    query = _required_text(query, "query")
    locator = _required_text(locator, "locator")
    evidence = _required_text(evidence, "evidence")
    if not isinstance(questions, list) or not questions:
        raise SourceIntakeError("questions must contain at least one question ID")
    questions = [_required_text(question, "questions") for question in questions]
    claims = _validate_parameter_claims(parameter_claims)
    provenance: dict[str, str] | None = None
    if packet_provenance is not None:
        if not isinstance(packet_provenance, dict):
            raise SourceIntakeError("packet_provenance must be an object")
        provenance = {
            field: _required_text(packet_provenance.get(field), f"packet_provenance.{field}")
            for field in ("packet_path", "packet_sha256", "packet_version", "question_id")
        }
        if not re.fullmatch(r"[0-9a-f]{64}", provenance["packet_sha256"]):
            raise SourceIntakeError("packet_provenance.packet_sha256 must be a SHA-256 hex digest")
    if license_status not in LICENSE_STATUSES:
        raise SourceIntakeError(f"unknown license status: {license_status}")
    if local_file and license_status not in ARCHIVABLE_LICENSES:
        raise SourceIntakeError(
            "a local copy requires open-access, public-domain, or permission-granted licensing"
        )
    identity_material = f"{url}\n{citation}"
    if provenance:
        identity_material += f"\n{provenance['packet_sha256']}"
    identity = hashlib.sha256(identity_material.encode("utf-8")).hexdigest()[:16]
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
        "schema": 2,
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
        "parameter_claims": claims,
        "packet_provenance": provenance,
    }
    if archived_path:
        record["archive_sha256"] = hashlib.sha256(archived_path.read_bytes()).hexdigest()
    _atomic_write(record_path, _render_record(record))
    _append_ledger(catalog / "source-intake.jsonl", record)
    return record
