#!/usr/bin/env python3
"""Validate bounded research packets before evidence enters project workflows.

This module is deliberately standard-library only.  It records discovery and
structured claims, but it does not search, judge biology, or publish claims to
the catalog.  A claim starts as ``pending_review`` and can become ``accepted``
only through :func:`accept_claim`.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping
from urllib.parse import urlparse


PACKET_VERSION = "research-packet-v1"
CLAIM_STATUSES = {"pending_review", "accepted", "rejected"}
PRIMARY_KINDS = {"peer-reviewed-primary", "primary-preprint"}
DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$", re.IGNORECASE)
SEARCH_DATE_FORMAT = "%Y-%m-%d"


class ResearchPacketError(ValueError):
    """Raised when a packet cannot be trusted as structured research input."""


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ResearchPacketError(f"{field} must be a non-empty string")
    return value.strip()


def _list(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise ResearchPacketError(f"{field} must be a list")
    return value


def _ids(items: list[Mapping[str, Any]], field: str) -> set[str]:
    result: set[str] = set()
    for item in items:
        if not isinstance(item, Mapping):
            raise ResearchPacketError(f"{field} entries must be objects")
        identifier = _text(item.get("id"), f"{field}.id")
        if identifier in result:
            raise ResearchPacketError(f"duplicate {field} id: {identifier}")
        result.add(identifier)
    return result


def _date(value: Any, field: str) -> dt.date:
    try:
        return dt.date.fromisoformat(_text(value, field))
    except ValueError as exc:
        raise ResearchPacketError(f"{field} must be YYYY-MM-DD") from exc


def _url(value: Any, field: str) -> str:
    result = _text(value, field)
    parsed = urlparse(result)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ResearchPacketError(f"{field} must be an http(s) URL")
    return result


def _validate_question(question: Any) -> None:
    if not isinstance(question, Mapping):
        raise ResearchPacketError("question must be an object")
    _text(question.get("id"), "question.id")
    text = _text(question.get("text"), "question.text")
    if len(text.split()) < 5:
        raise ResearchPacketError("question.text must state a precise, multi-word question")
    _text(question.get("kind"), "question.kind")
    _text(question.get("target"), "question.target")
    _text(question.get("requested_measurement"), "question.requested_measurement")


def _validate_prior_work(items: Any) -> set[str]:
    entries = _list(items, "prior_work_matches")
    result = _ids(entries, "prior_work_matches")
    for item in entries:
        _text(item.get("reference"), "prior_work_matches.reference")
        _text(item.get("relationship"), "prior_work_matches.relationship")
        _text(item.get("summary"), "prior_work_matches.summary")
        if item.get("status") not in {"failed", "related", "unresolved", "superseded"}:
            raise ResearchPacketError("prior_work_matches.status is not a supported state")
    return result


def _validate_searches(items: Any) -> set[str]:
    entries = _list(items, "online_searches")
    result = _ids(entries, "online_searches")
    for item in entries:
        databases = [_text(v, "online_searches.databases") for v in _list(item.get("databases"), "online_searches.databases")]
        queries = [_text(v, "online_searches.query_variants") for v in _list(item.get("query_variants"), "online_searches.query_variants")]
        urls = [_url(v, "online_searches.urls") for v in _list(item.get("urls"), "online_searches.urls")]
        if not databases or not queries or not urls:
            raise ResearchPacketError("each online search needs databases, query variants, and result URLs")
        date_from = _date(item.get("date_from"), "online_searches.date_from")
        date_to = _date(item.get("date_to"), "online_searches.date_to")
        if date_from > date_to:
            raise ResearchPacketError("online_searches.date_from cannot be after date_to")
        _text(item.get("outcome"), "online_searches.outcome")
        if item.get("claim_absence"):
            if len(set(databases)) < 2 or len(set(queries)) < 2 or len(urls) < 2:
                raise ResearchPacketError(
                    "claim_absence searches require two databases, two query variants, and two URLs"
                )
    return result


def _validate_sources(items: Any, search_ids: set[str]) -> set[str]:
    entries = _list(items, "sources")
    result = _ids(entries, "sources")
    for item in entries:
        _text(item.get("citation"), "sources.citation")
        _url(item.get("url"), "sources.url")
        kind = _text(item.get("kind"), "sources.kind")
        if kind not in PRIMARY_KINDS | {"review", "secondary"}:
            raise ResearchPacketError(f"unsupported source kind: {kind}")
        search_id = _text(item.get("search_id"), "sources.search_id")
        if search_id not in search_ids:
            raise ResearchPacketError(f"source refers to unknown search: {search_id}")
        _text(item.get("locator"), "sources.locator")
        _text(item.get("evidence"), "sources.evidence")
        doi = item.get("doi")
        if doi is not None and (not isinstance(doi, str) or not DOI_RE.fullmatch(doi.strip())):
            raise ResearchPacketError("sources.doi must be a normalized DOI")
        discovery = item.get("discovery")
        if discovery is not None:
            if not isinstance(discovery, Mapping):
                raise ResearchPacketError("sources.discovery must be an object")
            _text(discovery.get("provider"), "sources.discovery.provider")
            _text(discovery.get("provider_record_id"), "sources.discovery.provider_record_id")
            _url(discovery.get("search_url"), "sources.discovery.search_url")
            query_ids = [_text(v, "sources.discovery.query_ids") for v in _list(
                discovery.get("query_ids"), "sources.discovery.query_ids"
            )]
            if not query_ids:
                raise ResearchPacketError("sources.discovery.query_ids cannot be empty")
            records = discovery.get("records")
            if records is not None:
                for record in _list(records, "sources.discovery.records"):
                    if not isinstance(record, Mapping):
                        raise ResearchPacketError("sources.discovery.records entries must be objects")
                    _text(record.get("query_id"), "sources.discovery.records.query_id")
                    _text(record.get("provider"), "sources.discovery.records.provider")
                    _text(record.get("provider_record_id"), "sources.discovery.records.provider_record_id")
                    _url(record.get("search_url"), "sources.discovery.records.search_url")
                    _text(record.get("exact_locator"), "sources.discovery.records.exact_locator")
        if item.get("license_status") not in {
            "open-access", "public-domain", "permission-granted", "metadata-only"
        }:
            raise ResearchPacketError("sources.license_status is invalid")
    return result


def _validate_claims(items: Any, sources: Mapping[str, Mapping[str, Any]], *, require_review: bool) -> None:
    entries = _list(items, "claims")
    _ids(entries, "claims")
    for item in entries:
        _text(item.get("id"), "claims.id")
        source_refs = [_text(v, "claims.source_ids") for v in _list(item.get("source_ids"), "claims.source_ids")]
        if not source_refs or any(ref not in sources for ref in source_refs):
            raise ResearchPacketError("each claim must cite at least one known source")
        if item.get("value") is None:
            raise ResearchPacketError("claims.value is required, including for qualitative values")
        for field in ("units", "condition", "species", "preparation", "uncertainty", "locator", "limitations"):
            _text(item.get(field), f"claims.{field}")
        status = item.get("status", "pending_review")
        if status not in CLAIM_STATUSES:
            raise ResearchPacketError(f"claims.status is invalid: {status}")
        review = item.get("review")
        if status == "accepted":
            if not isinstance(review, Mapping) or review.get("decision") != "approved":
                raise ResearchPacketError("accepted claims require an approving review")
            if not any(sources[ref].get("kind") in PRIMARY_KINDS for ref in source_refs):
                raise ResearchPacketError("accepted claims require at least one primary source")
            _text(review.get("reviewer"), "claims.review.reviewer")
            _date(review.get("reviewed_at"), "claims.review.reviewed_at")
        elif require_review and review is not None:
            if not isinstance(review, Mapping) or review.get("decision") not in {"approved", "rejected"}:
                raise ResearchPacketError("claim review must have an approved or rejected decision")


def validate_packet(packet: Mapping[str, Any], *, require_review: bool = True) -> dict[str, Any]:
    """Return a deep copy of a valid packet or fail closed with an exception."""
    if not isinstance(packet, Mapping):
        raise ResearchPacketError("packet must be an object")
    if packet.get("packet_version") != PACKET_VERSION:
        raise ResearchPacketError("unsupported or missing packet_version")
    if packet.get("review_required") is not True:
        raise ResearchPacketError("review_required must be true")
    _validate_question(packet.get("question"))
    _validate_prior_work(packet.get("prior_work_matches"))
    search_ids = _validate_searches(packet.get("online_searches"))
    source_ids = _validate_sources(packet.get("sources"), search_ids)
    source_entries = {item["id"]: item for item in packet["sources"]}
    _validate_claims(packet.get("claims"), source_entries, require_review=require_review)
    _text(packet.get("created_at"), "created_at")
    return copy.deepcopy(dict(packet))


def create_packet(
    *,
    question: Mapping[str, Any],
    prior_work_matches: list[Mapping[str, Any]],
    online_searches: list[Mapping[str, Any]],
    sources: list[Mapping[str, Any]],
    claims: list[Mapping[str, Any]],
    created_at: str | None = None,
) -> dict[str, Any]:
    """Construct and validate a packet; all claims start pending review by default."""
    normalized_claims = []
    for claim in claims:
        item = dict(claim)
        item.setdefault("status", "pending_review")
        item.setdefault("review", None)
        normalized_claims.append(item)
    packet = {
        "packet_version": PACKET_VERSION,
        "review_required": True,
        "created_at": created_at or dt.date.today().isoformat(),
        "question": dict(question),
        "prior_work_matches": [dict(item) for item in prior_work_matches],
        "online_searches": [dict(item) for item in online_searches],
        "sources": [dict(item) for item in sources],
        "claims": normalized_claims,
    }
    return validate_packet(packet)


def accept_claim(
    packet: Mapping[str, Any], claim_id: str, *, reviewer: str, reviewed_at: str, notes: str = ""
) -> dict[str, Any]:
    """Approve one claim explicitly, then revalidate the complete packet."""
    result = validate_packet(packet)
    _text(claim_id, "claim_id")
    _text(reviewer, "reviewer")
    _date(reviewed_at, "reviewed_at")
    for claim in result["claims"]:
        if claim["id"] == claim_id:
            if claim.get("status") == "rejected":
                raise ResearchPacketError("a rejected claim cannot be accepted without a new packet")
            claim["status"] = "accepted"
            claim["review"] = {
                "decision": "approved",
                "reviewer": reviewer.strip(),
                "reviewed_at": reviewed_at.strip(),
                "notes": notes.strip(),
            }
            return validate_packet(result)
    raise ResearchPacketError(f"unknown claim id: {claim_id}")


def save_packet(path: Path, packet: Mapping[str, Any]) -> None:
    """Validate and atomically save a packet as JSON."""
    validated = validate_packet(packet)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(validated, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def load_packet(path: Path) -> dict[str, Any]:
    path = Path(path)
    try:
        packet = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResearchPacketError(f"cannot read packet: {path}") from exc
    return validate_packet(packet)


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("packet", type=Path)
    accept_parser = subparsers.add_parser("accept")
    accept_parser.add_argument("packet", type=Path)
    accept_parser.add_argument("claim_id")
    accept_parser.add_argument("--reviewer", required=True)
    accept_parser.add_argument("--reviewed-at", required=True)
    accept_parser.add_argument("--notes", default="")
    args = parser.parse_args()
    try:
        if args.command == "validate":
            load_packet(args.packet)
        else:
            packet = accept_claim(
                load_packet(args.packet), args.claim_id, reviewer=args.reviewer,
                reviewed_at=args.reviewed_at, notes=args.notes,
            )
            save_packet(args.packet, packet)
    except ResearchPacketError as exc:
        parser.error(str(exc))
    print(f"{args.command}: valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
