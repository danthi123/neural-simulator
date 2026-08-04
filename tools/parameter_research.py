#!/usr/bin/env python3
"""Plan and preserve bounded searches for missing biological parameters.

Discovery is deliberately adapter-driven: this tool can consume fixtures or a
separate scholarly API client without making network behavior part of the
evidence boundary.  Candidate claims always remain pending until the existing
research-packet review workflow explicitly accepts them.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit

try:
    from tools import research_packet
except ModuleNotFoundError:  # direct script execution
    import research_packet


STATE_VERSION = "parameter-research-v1"
GAP_FIELDS = ("parameter", "value", "units", "species", "preparation")
SOURCE_KINDS = {"peer-reviewed-primary", "primary-preprint", "review", "secondary"}
LICENSE_STATUSES = {"open-access", "public-domain", "permission-granted", "metadata-only"}
DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$", re.IGNORECASE)


class ParameterResearchError(RuntimeError):
    """Raised when discovery state is incomplete or internally inconsistent."""


DiscoveryAdapter = Callable[[Mapping[str, Any]], Mapping[str, Any]]
LocalSearchAdapter = Callable[[str, str], Mapping[str, Any]]


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ParameterResearchError(f"{field} must be a non-empty string")
    return value.strip()


def _optional_text(value: Any, field: str) -> str | None:
    if value is None:
        return None
    return _text(value, field)


def _date(value: Any, field: str) -> str:
    result = _text(value, field)
    try:
        dt.date.fromisoformat(result)
    except ValueError as exc:
        raise ParameterResearchError(f"{field} must be YYYY-MM-DD") from exc
    return result


def _url(value: Any, field: str) -> str:
    result = _text(value, field)
    parsed = urlsplit(result)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ParameterResearchError(f"{field} must be an http(s) URL")
    return result


def normalize_doi(value: str | None) -> str | None:
    if value is None:
        return None
    result = value.strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if result.startswith(prefix):
            result = result[len(prefix):]
            break
    if not DOI_RE.fullmatch(result):
        raise ParameterResearchError("candidate.doi is not a valid DOI")
    return result.rstrip(".,;)")


def normalize_url(value: str) -> str:
    parsed = urlsplit(_url(value, "candidate.url"))
    host = parsed.netloc.lower()
    path = parsed.path.rstrip("/") or "/"
    return urlunsplit((parsed.scheme.lower(), host, path, parsed.query, ""))


def _tokens(*values: str | None) -> str:
    return " ".join(value.strip() for value in values if value and value.strip())


def derive_query_variants(question: Mapping[str, Any], gap: Mapping[str, Any]) -> list[str]:
    """Derive deterministic search variants from known and explicitly missing fields."""
    target = _text(question.get("target"), "question.target")
    requested = _text(question.get("requested_measurement"), "question.requested_measurement")
    missing = gap.get("missing_fields")
    if not isinstance(missing, list) or not missing:
        raise ParameterResearchError("gap.missing_fields must be a non-empty list")
    missing = [_text(item, "gap.missing_fields") for item in missing]
    unknown = sorted(set(missing) - set(GAP_FIELDS))
    if unknown:
        raise ParameterResearchError(f"unsupported missing fields: {', '.join(unknown)}")
    known: dict[str, str | None] = {}
    for field in GAP_FIELDS:
        known[field] = _optional_text(gap.get(field), f"gap.{field}")
        if field in missing and known[field] is not None:
            raise ParameterResearchError(f"gap.{field} cannot be supplied and missing")

    context = _tokens(target, known["parameter"], known["species"], known["preparation"])
    variants = [
        _tokens(context, requested, "quantitative measurement"),
        _tokens(context, known["units"], "methods results parameter table"),
    ]
    expansions = {
        "parameter": _tokens(target, requested, known["species"], known["preparation"], "mechanism parameter"),
        "value": _tokens(context, "mean range variance dose response kinetics conductance density"),
        "units": _tokens(context, "measurement units methods"),
        "species": _tokens(target, known["parameter"], requested, "species comparative"),
        "preparation": _tokens(target, known["parameter"], known["species"], "in vivo in vitro slice recording preparation"),
    }
    variants.extend(expansions[field] for field in missing)
    return list(dict.fromkeys(value for value in variants if value))


def _default_local_search(repo: Path) -> LocalSearchAdapter:
    def run(query: str, purpose: str) -> Mapping[str, Any]:
        if purpose == "project_rag":
            search_query = query
            corpus = "all"
        else:
            search_query = f"{query} NO-GO failed boundary"
            corpus = "finding"
        completed = subprocess.run(
            ["bash", "tools/rag/search.sh", search_query, "5", "--corpus", corpus],
            cwd=repo,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        return {
            "status": "complete" if completed.returncode == 0 else "unavailable",
            "query": search_query,
            "summary": (completed.stdout or "").strip()[:20_000] or "No output.",
            "returncode": completed.returncode,
            "corpus": corpus,
        }
    return run


def create_plan(
    *,
    question: Mapping[str, Any],
    gaps: Sequence[Mapping[str, Any]],
    local_search: LocalSearchAdapter,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Create a search plan only after local evidence and prior failures are checked."""
    research_packet._validate_question(question)
    if not isinstance(gaps, Sequence) or isinstance(gaps, (str, bytes)) or not gaps:
        raise ParameterResearchError("gaps must contain at least one parameter gap")
    gap_records: list[dict[str, Any]] = []
    queries: list[dict[str, Any]] = []
    local_checks: list[dict[str, Any]] = []
    seen_gap_ids: set[str] = set()
    for raw_gap in gaps:
        if not isinstance(raw_gap, Mapping):
            raise ParameterResearchError("gap entries must be objects")
        gap = dict(raw_gap)
        gap_id = _text(gap.get("id"), "gap.id")
        if gap_id in seen_gap_ids:
            raise ParameterResearchError(f"duplicate gap id: {gap_id}")
        seen_gap_ids.add(gap_id)
        variants = derive_query_variants(question, gap)
        gap["missing_fields"] = list(dict.fromkeys(gap["missing_fields"]))
        gap_records.append(gap)
        for index, query in enumerate(variants, 1):
            query_id = f"{gap_id}-Q{index}"
            queries.append({"id": query_id, "gap_id": gap_id, "text": query, "status": "planned"})
            for purpose in ("project_rag", "prior_failures"):
                result = dict(local_search(query, purpose))
                status = result.get("status")
                if status not in {"complete", "unavailable"}:
                    raise ParameterResearchError("local search status must be complete or unavailable")
                local_checks.append({
                    "id": f"LC{len(local_checks) + 1}",
                    "query_id": query_id,
                    "purpose": purpose,
                    "status": status,
                    "query": _text(result.get("query", query), "local_check.query"),
                    "summary": _text(result.get("summary"), "local_check.summary"),
                    "returncode": result.get("returncode"),
                })
    state = {
        "state_version": STATE_VERSION,
        "created_at": created_at or dt.date.today().isoformat(),
        "updated_at": created_at or dt.date.today().isoformat(),
        "question": dict(question),
        "gaps": gap_records,
        "local_checks": local_checks,
        "queries": queries,
        "searches": [],
        "candidates": [],
    }
    return validate_state(state)


def validate_state(state: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(state, Mapping) or state.get("state_version") != STATE_VERSION:
        raise ParameterResearchError("unsupported or missing state_version")
    research_packet._validate_question(state.get("question"))
    _date(state.get("created_at"), "created_at")
    _date(state.get("updated_at"), "updated_at")
    gaps = state.get("gaps")
    queries = state.get("queries")
    checks = state.get("local_checks")
    searches = state.get("searches")
    candidates = state.get("candidates")
    if not all(isinstance(value, list) for value in (gaps, queries, checks, searches, candidates)):
        raise ParameterResearchError("state collections must be lists")
    gap_ids = {_text(gap.get("id"), "gap.id") for gap in gaps}
    if len(gap_ids) != len(gaps):
        raise ParameterResearchError("duplicate gap id")
    query_ids = {_text(query.get("id"), "query.id") for query in queries}
    if len(query_ids) != len(queries):
        raise ParameterResearchError("duplicate query id")
    for query in queries:
        if query.get("gap_id") not in gap_ids:
            raise ParameterResearchError("query refers to an unknown gap")
        _text(query.get("text"), "query.text")
    required_checks = {(query_id, purpose) for query_id in query_ids for purpose in ("project_rag", "prior_failures")}
    actual_checks: set[tuple[str, str]] = set()
    for check in checks:
        key = (check.get("query_id"), check.get("purpose"))
        if key in actual_checks or key not in required_checks:
            raise ParameterResearchError("local checks are duplicate or refer to an unknown query/purpose")
        actual_checks.add(key)
        if check.get("status") not in {"complete", "unavailable"}:
            raise ParameterResearchError("local search status must be complete or unavailable")
        _text(check.get("summary"), "local_check.summary")
    if actual_checks != required_checks:
        raise ParameterResearchError("every query requires project-RAG and prior-failure checks")
    if any(check["status"] != "complete" for check in checks) and (searches or candidates):
        raise ParameterResearchError("external discovery is blocked while a local check is unavailable")
    search_ids: set[str] = set()
    searched_queries: set[str] = set()
    for search in searches:
        search_id = _text(search.get("id"), "search.id")
        if search_id in search_ids:
            raise ParameterResearchError("duplicate search id")
        search_ids.add(search_id)
        query_id = _text(search.get("query_id"), "search.query_id")
        if query_id not in query_ids or query_id in searched_queries:
            raise ParameterResearchError("search query is unknown or already recorded")
        searched_queries.add(query_id)
        _text(search.get("database"), "search.database")
        _url(search.get("search_url"), "search.search_url")
        _date(search.get("searched_at"), "search.searched_at")
        _text(search.get("outcome"), "search.outcome")
        if not isinstance(search.get("candidate_ids"), list):
            raise ParameterResearchError("search.candidate_ids must be a list")
    identities: set[tuple[str, str]] = set()
    candidate_ids: set[str] = set()
    for candidate in candidates:
        candidate_id = _text(candidate.get("id"), "candidate.id")
        if candidate_id in candidate_ids:
            raise ParameterResearchError("duplicate candidate id")
        candidate_ids.add(candidate_id)
        keys = _candidate_keys(candidate)
        if keys & identities:
            raise ParameterResearchError("duplicate candidate DOI/URL identity")
        identities.update(keys)
        for field in ("title", "citation", "exact_locator", "evidence", "provider", "provider_record_id"):
            _text(candidate.get(field), f"candidate.{field}")
        _url(candidate.get("search_url"), "candidate.search_url")
        kind = _text(candidate.get("kind"), "candidate.kind")
        if kind not in SOURCE_KINDS:
            raise ParameterResearchError(f"unsupported source kind: {kind}")
        if candidate.get("license_status") not in LICENSE_STATUSES:
            raise ParameterResearchError("candidate.license_status is invalid")
        if candidate.get("status") != "pending_review":
            raise ParameterResearchError("candidate must remain pending_review")
        candidate_queries = candidate.get("query_ids")
        if not isinstance(candidate_queries, list) or not candidate_queries or any(
            query_id not in query_ids for query_id in candidate_queries
        ):
            raise ParameterResearchError("candidate.query_ids must refer to planned queries")
        claims = candidate.get("claims")
        if not isinstance(claims, list):
            raise ParameterResearchError("candidate.claims must be a list")
        for claim in claims:
            if not isinstance(claim, Mapping):
                raise ParameterResearchError("candidate claims must be objects")
            if claim.get("status") != "pending_review" or claim.get("review") is not None:
                raise ParameterResearchError("discovery claims must remain pending_review")
            for field in ("id", "units", "condition", "species", "preparation", "uncertainty", "locator", "limitations"):
                _text(claim.get(field), f"claim.{field}")
            if claim.get("value") is None:
                raise ParameterResearchError("claim.value is required")
        discovery_records = candidate.get("discovery_records")
        if not isinstance(discovery_records, list) or not discovery_records:
            raise ParameterResearchError("candidate.discovery_records must be a non-empty list")
        for record in discovery_records:
            if not isinstance(record, Mapping):
                raise ParameterResearchError("candidate.discovery_records entries must be objects")
            if record.get("query_id") not in candidate_queries:
                raise ParameterResearchError("candidate discovery refers to an unknown candidate query")
            for field in ("provider", "provider_record_id", "exact_locator"):
                _text(record.get(field), f"candidate.discovery_records.{field}")
            _url(record.get("search_url"), "candidate.discovery_records.search_url")
    for search in searches:
        if any(candidate_id not in candidate_ids for candidate_id in search["candidate_ids"]):
            raise ParameterResearchError("search refers to an unknown candidate")
    for query in queries:
        expected = "searched" if query["id"] in searched_queries else "planned"
        if query.get("status") != expected:
            raise ParameterResearchError("query status does not match durable search records")
    return json.loads(json.dumps(state))


def _candidate_keys(candidate: Mapping[str, Any]) -> set[tuple[str, str]]:
    doi = normalize_doi(candidate.get("doi"))
    keys = {("url", normalize_url(candidate.get("url")))}
    if doi:
        keys.add(("doi", doi))
    return keys


def add_discovery_results(
    state: Mapping[str, Any], *, adapter: DiscoveryAdapter, searched_at: str | None = None
) -> dict[str, Any]:
    """Run planned queries through an adapter and durably normalize candidate metadata."""
    result = validate_state(state)
    if any(check["status"] != "complete" for check in result["local_checks"]):
        raise ParameterResearchError("external discovery requires successful local checks first")
    searched_at = searched_at or dt.date.today().isoformat()
    _date(searched_at, "searched_at")
    identities = {
        key: candidate["id"]
        for candidate in result["candidates"]
        for key in _candidate_keys(candidate)
    }
    for query in result["queries"]:
        if query.get("status") == "searched":
            continue
        response = adapter(dict(query))
        if not isinstance(response, Mapping):
            raise ParameterResearchError("discovery adapter must return an object")
        provider = _text(response.get("provider"), "discovery.provider")
        search_url = _url(response.get("search_url"), "discovery.search_url")
        records = response.get("candidates")
        if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
            raise ParameterResearchError("discovery.candidates must be a sequence")
        search_id = f"S{len(result['searches']) + 1}"
        candidate_ids: list[str] = []
        for raw in records:
            if not isinstance(raw, Mapping):
                raise ParameterResearchError("discovery candidates must be objects")
            item = dict(raw)
            title = _text(item.get("title"), "candidate.title")
            citation = _text(item.get("citation"), "candidate.citation")
            provider_record_id = _text(item.get("provider_record_id"), "candidate.provider_record_id")
            exact_locator = _text(item.get("exact_locator"), "candidate.exact_locator")
            evidence = _text(item.get("evidence"), "candidate.evidence")
            kind = _text(item.get("kind"), "candidate.kind")
            if kind not in SOURCE_KINDS:
                raise ParameterResearchError(f"unsupported source kind: {kind}")
            license_status = _text(item.get("license_status", "metadata-only"), "candidate.license_status")
            if license_status not in LICENSE_STATUSES:
                raise ParameterResearchError("candidate.license_status is invalid")
            raw_claims = item.get("claims", [])
            if not isinstance(raw_claims, list):
                raise ParameterResearchError("candidate.claims must be a list")
            normalized_claims: list[dict[str, Any]] = []
            for claim in raw_claims:
                if not isinstance(claim, Mapping):
                    raise ParameterResearchError("candidate claims must be objects")
                claim = dict(claim)
                if claim.get("status", "pending_review") != "pending_review" or claim.get("review") not in (None, {}):
                    raise ParameterResearchError("discovery claims must remain pending_review")
                claim["status"] = "pending_review"
                claim["review"] = None
                normalized_claims.append(claim)
            doi = normalize_doi(item.get("doi"))
            url = _url(item.get("url"), "candidate.url")
            keys = {("url", normalize_url(url))}
            if doi:
                keys.add(("doi", doi))
            matching_ids = {identities[key] for key in keys if key in identities}
            if len(matching_ids) > 1:
                raise ParameterResearchError("candidate identifiers conflict with multiple sources")
            if matching_ids:
                existing = next(c for c in result["candidates"] if c["id"] == next(iter(matching_ids)))
                if doi and existing.get("doi") and doi != existing["doi"]:
                    raise ParameterResearchError("same candidate URL was returned with conflicting DOIs")
                if doi and not existing.get("doi"):
                    existing["doi"] = doi
                if query["id"] not in existing["query_ids"]:
                    existing["query_ids"].append(query["id"])
                discovery_record = {
                    "query_id": query["id"],
                    "provider": provider,
                    "provider_record_id": provider_record_id,
                    "search_url": search_url,
                    "exact_locator": exact_locator,
                }
                if discovery_record not in existing["discovery_records"]:
                    existing["discovery_records"].append(discovery_record)
                known_claims = {claim["id"]: claim for claim in existing["claims"]}
                for claim in normalized_claims:
                    claim_id = _text(claim.get("id"), "claim.id")
                    if claim_id in known_claims and claim != known_claims[claim_id]:
                        raise ParameterResearchError(f"conflicting duplicate claim id: {claim_id}")
                    if claim_id not in known_claims:
                        existing["claims"].append(claim)
                identities.update({key: existing["id"] for key in keys})
                candidate_ids.append(existing["id"])
                continue
            candidate_id = f"SRC{len(result['candidates']) + 1}"
            candidate = {
                "id": candidate_id,
                "title": title,
                "citation": citation,
                "authors": [_text(author, "candidate.authors") for author in item.get("authors", [])],
                "year": item.get("year"),
                "doi": doi,
                "url": url,
                "kind": kind,
                "license_status": license_status,
                "exact_locator": exact_locator,
                "evidence": evidence,
                "provider": provider,
                "provider_record_id": provider_record_id,
                "search_url": search_url,
                "query_ids": [query["id"]],
                "status": "pending_review",
                "claims": normalized_claims,
                "discovery_records": [{
                    "query_id": query["id"],
                    "provider": provider,
                    "provider_record_id": provider_record_id,
                    "search_url": search_url,
                    "exact_locator": exact_locator,
                }],
            }
            result["candidates"].append(candidate)
            identities.update({key: candidate_id for key in keys})
            candidate_ids.append(candidate_id)
        result["searches"].append({
            "id": search_id,
            "query_id": query["id"],
            "database": provider,
            "search_url": search_url,
            "searched_at": searched_at,
            "candidate_ids": list(dict.fromkeys(candidate_ids)),
            "outcome": f"{len(set(candidate_ids))} unique candidate source(s) retained.",
        })
        query["status"] = "searched"
    result["updated_at"] = searched_at
    return validate_state(result)


def export_packet(state: Mapping[str, Any]) -> dict[str, Any]:
    """Export discovered sources and proposed claims into the existing packet boundary."""
    state = validate_state(state)
    if not state["searches"]:
        raise ParameterResearchError("no completed external searches are available")
    search_by_query = {search["query_id"]: search for search in state["searches"]}
    online_searches = []
    for search in state["searches"]:
        query = next(item for item in state["queries"] if item["id"] == search["query_id"])
        online_searches.append({
            "id": search["id"],
            "databases": [search["database"]],
            "query_variants": [query["text"]],
            "date_from": "1900-01-01",
            "date_to": search["searched_at"],
            "urls": [search["search_url"]],
            "outcome": search["outcome"],
        })
    sources: list[dict[str, Any]] = []
    claims: list[dict[str, Any]] = []
    claim_ids: set[str] = set()
    for candidate in state["candidates"]:
        search = search_by_query[candidate["query_ids"][0]]
        sources.append({
            "id": candidate["id"],
            "citation": candidate["citation"],
            "url": candidate["url"],
            "doi": candidate["doi"],
            "kind": candidate["kind"],
            "search_id": search["id"],
            "locator": candidate["exact_locator"],
            "evidence": candidate["evidence"],
            "license_status": candidate["license_status"],
            "discovery": {
                "provider": candidate["provider"],
                "provider_record_id": candidate["provider_record_id"],
                "search_url": candidate["search_url"],
                "query_ids": candidate["query_ids"],
                "records": candidate["discovery_records"],
            },
        })
        for raw_claim in candidate["claims"]:
            claim = dict(raw_claim)
            claim_id = _text(claim.get("id"), "claim.id")
            if claim_id in claim_ids:
                raise ParameterResearchError(f"duplicate claim id: {claim_id}")
            claim_ids.add(claim_id)
            claim["source_ids"] = [candidate["id"]]
            claim["status"] = "pending_review"
            claim["review"] = None
            claims.append(claim)
    if not sources or not claims:
        raise ParameterResearchError("packet export requires candidates with proposed parameter claims")
    prior = []
    for check in state["local_checks"]:
        if check["purpose"] == "prior_failures":
            prior.append({
                "id": f"F{len(prior) + 1}",
                "reference": f"project RAG check {check['id']}",
                "relationship": f"Prior-failure search for {check['query_id']}",
                "status": "unresolved",
                "summary": check["summary"],
            })
    return research_packet.create_packet(
        question=state["question"], prior_work_matches=prior,
        online_searches=online_searches, sources=sources, claims=claims,
        created_at=state["updated_at"],
    )


def save_state(path: Path, state: Mapping[str, Any]) -> None:
    validated = validate_state(state)
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


def load_state(path: Path) -> dict[str, Any]:
    try:
        return validate_state(json.loads(Path(path).read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as exc:
        raise ParameterResearchError(f"cannot read parameter-research state: {path}") from exc


def _fixture_adapter(records: Sequence[Mapping[str, Any]]) -> DiscoveryAdapter:
    indexed = {record.get("query_id"): record for record in records}
    def discover(query: Mapping[str, Any]) -> Mapping[str, Any]:
        record = indexed.get(query["id"])
        if record is None:
            raise ParameterResearchError(f"fixture has no response for query {query['id']}")
        return record
    return discover


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--spec", required=True, type=Path)
    plan.add_argument("--output", required=True, type=Path)
    plan.add_argument("--repo", type=Path, default=Path.cwd())
    discover = sub.add_parser("import-results")
    discover.add_argument("--state", required=True, type=Path)
    discover.add_argument("--results", required=True, type=Path)
    export = sub.add_parser("export-packet")
    export.add_argument("--state", required=True, type=Path)
    export.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        if args.command == "plan":
            spec = json.loads(args.spec.read_text(encoding="utf-8"))
            state = create_plan(
                question=spec["question"], gaps=spec["gaps"],
                local_search=_default_local_search(args.repo.resolve()),
            )
            save_state(args.output, state)
        elif args.command == "import-results":
            records = json.loads(args.results.read_text(encoding="utf-8"))
            save_state(args.state, add_discovery_results(load_state(args.state), adapter=_fixture_adapter(records)))
        else:
            research_packet.save_packet(args.output, export_packet(load_state(args.state)))
    except (KeyError, OSError, json.JSONDecodeError, ParameterResearchError, research_packet.ResearchPacketError) as exc:
        parser.error(str(exc))
    print(f"{args.command}: valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
