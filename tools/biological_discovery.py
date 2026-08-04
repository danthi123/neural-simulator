#!/usr/bin/env python3
"""Discover biological evidence candidates for a blocked simulation experiment.

This layer searches metadata APIs and creates a review packet.  It deliberately
does not treat abstracts as parameter evidence and does not register sources in
the local catalog; reviewed candidates can later enter research_escalation.py
and tools/rag/source_intake.py.
"""
from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import datetime as dt
import hashlib
import html
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any, Protocol
from urllib.parse import urlencode
from urllib.request import Request, urlopen


SCHEMA = "biological-discovery-packet-v1"
WALL_SCHEMA = "biological-discovery-wall-v1"
CANONICALIZATION = "json-sort-keys-utf8-compact-v1"
USER_AGENT = "sim-biological-discovery/1.0 (metadata research; no full-text harvesting)"
PROVIDERS = ("europe_pmc", "crossref", "openalex")
PRIMARY_TYPES = {
    "article",
    "clinical-trial",
    "dissertation",
    "journal-article",
    "posted-content",
    "preprint",
    "proceedings-article",
}
QUANTITATIVE_TERMS = {
    "amplitude", "conductance", "connectivity", "current", "density", "dose",
    "electrophysiology", "frequency", "kinetics", "latency", "measurement",
    "morphometry", "patch clamp", "probability", "rate", "recording", "response",
    "strength", "threshold", "time constant", "tracing", "voltage", "weight",
}
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "does", "for",
    "from", "how", "in", "is", "it", "of", "on", "or", "that", "the", "to",
    "what", "when", "where", "which", "with",
}


class DiscoveryError(RuntimeError):
    """Raised when discovery cannot produce a complete, reviewable packet."""


class JsonHttpClient(Protocol):
    def get_json(self, url: str, *, timeout: float) -> Mapping[str, Any]: ...


class UrllibJsonClient:
    """Small standard-library HTTP client; tests inject an offline replacement."""

    def get_json(self, url: str, *, timeout: float) -> Mapping[str, Any]:
        request = Request(url, headers={"Accept": "application/json", "User-Agent": USER_AGENT})
        try:
            with urlopen(request, timeout=timeout) as response:
                payload = response.read()
        except Exception as exc:  # network and HTTP failures are one fail-closed boundary
            raise DiscoveryError(f"request failed for {url}: {exc}") from exc
        try:
            value = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DiscoveryError(f"provider returned invalid JSON for {url}: {exc}") from exc
        if not isinstance(value, Mapping):
            raise DiscoveryError(f"provider returned a non-object payload for {url}")
        return value


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _text(value: Any) -> str:
    if isinstance(value, str):
        without_tags = re.sub(r"<[^>]+>", " ", html.unescape(value))
        return re.sub(r"\s+", " ", without_tags).strip()
    return ""


def _list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _tokens(value: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+(?:[-'][a-z0-9]+)?", value.lower())
    return {word for word in words if len(word) > 1 and word not in STOPWORDS}


def _phrase_hits(text: str, phrases: set[str]) -> list[str]:
    lowered = text.lower()
    return sorted(phrase for phrase in phrases if phrase and phrase.lower() in lowered)


def _flatten_context(value: Any) -> list[str]:
    if isinstance(value, Mapping):
        result: list[str] = []
        for key in sorted(value):
            result.extend(_flatten_context(value[key]))
        return result
    if isinstance(value, list):
        result = []
        for item in value:
            result.extend(_flatten_context(item))
        return result
    text = _text(value)
    return [text] if text else []


def _normalize_questions(value: Any, prefix: str) -> list[dict[str, str]]:
    field = "parameter_questions" if prefix == "P" else "wiring_questions"
    if not isinstance(value, list) or not value:
        raise DiscoveryError(f"{field} must be a non-empty list")
    result: list[dict[str, str]] = []
    for index, item in enumerate(value, start=1):
        if isinstance(item, str):
            question_id, question = f"{prefix}{index}", _text(item)
        elif isinstance(item, Mapping):
            question_id = _text(item.get("id")) or f"{prefix}{index}"
            question = _text(item.get("text"))
        else:
            raise DiscoveryError(f"{field}[{index - 1}] is not a string or object")
        if not question:
            raise DiscoveryError(f"{field}[{index - 1}] has no text")
        result.append({"id": question_id, "kind": "parameter" if prefix == "P" else "wiring", "text": question})
    if len({item["id"] for item in result}) != len(result):
        raise DiscoveryError(f"{field} IDs must be unique")
    return result


def validate_wall(wall: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize the structured research wall."""
    if wall.get("schema") not in (None, WALL_SCHEMA):
        raise DiscoveryError(f"unsupported wall schema: {wall.get('schema')!r}")
    wall_id = _text(wall.get("wall_id"))
    blocked = _text(wall.get("blocked_experiment"))
    reason = _text(wall.get("wall_reason"))
    preparation = wall.get("preparation")
    if not wall_id or not blocked or not reason:
        raise DiscoveryError("wall_id, blocked_experiment, and wall_reason are required")
    if not isinstance(preparation, Mapping) or not _flatten_context(preparation):
        raise DiscoveryError("preparation must be a non-empty object")
    parameter = _normalize_questions(wall.get("parameter_questions"), "P")
    wiring = _normalize_questions(wall.get("wiring_questions"), "W")
    if len({item["id"] for item in parameter + wiring}) != len(parameter + wiring):
        raise DiscoveryError("question IDs must be unique across parameter_questions and wiring_questions")
    mechanisms = sorted({_text(item) for item in _list(wall.get("mechanisms")) if _text(item)})
    return {
        "schema": WALL_SCHEMA,
        "wall_id": wall_id,
        "blocked_experiment": blocked,
        "wall_reason": reason,
        "preparation": dict(preparation),
        "mechanisms": mechanisms,
        "questions": parameter + wiring,
        "prior_attempts": [_text(item) for item in _list(wall.get("prior_attempts")) if _text(item)],
    }


def generate_query_variants(wall: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Create deterministic broad, quantitative, and methods-oriented queries."""
    normalized = validate_wall(wall) if "questions" not in wall else dict(wall)
    preparation = " ".join(_flatten_context(normalized["preparation"]))
    mechanisms = " ".join(normalized.get("mechanisms", []))
    variants: list[dict[str, Any]] = []
    seen: set[str] = set()
    for question in normalized["questions"]:
        stems = [
            ("preparation", f"{question['text']} {preparation} {mechanisms}"),
            (
                "quantitative-methods",
                f"{question['text']} {preparation} {mechanisms} quantitative measurement methods electrophysiology tracing",
            ),
        ]
        for purpose, raw in stems:
            query = _text(raw)
            key = query.casefold()
            if key in seen:
                continue
            seen.add(key)
            variants.append({
                "id": f"Q{len(variants) + 1}",
                "question_ids": [question["id"]],
                "purpose": purpose,
                "query": query,
            })
    if len(variants) < 2:
        raise DiscoveryError("query generation produced fewer than two variants")
    return variants


def _url(base: str, params: Mapping[str, str]) -> str:
    return f"{base}?{urlencode(params)}"


def _doi(value: Any) -> str | None:
    text = _text(value).lower()
    text = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", text)
    return text.removeprefix("doi:").strip() or None


def _external_id(value: Any, prefix: str) -> str | None:
    text = _text(value)
    if not text:
        return None
    tail = text.rstrip("/").rsplit("/", 1)[-1].upper()
    if prefix.upper() == "PMC":
        return tail if tail.startswith("PMC") else f"PMC{tail}"
    return tail.removeprefix(f"{prefix.upper()}:") or None


def _year_from_parts(value: Any) -> int | None:
    try:
        year = int(value["date-parts"][0][0])
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    return year if 1600 <= year <= 2200 else None


def _abstract_from_inverted(value: Any) -> str:
    if not isinstance(value, Mapping):
        return ""
    positions: list[tuple[int, str]] = []
    for word, indexes in value.items():
        if not isinstance(word, str) or not isinstance(indexes, list):
            continue
        for index in indexes:
            if isinstance(index, int):
                positions.append((index, word))
    return " ".join(word for _, word in sorted(positions))


def _authors_crossref(item: Mapping[str, Any]) -> list[str]:
    authors = []
    for author in _list(item.get("author")):
        if isinstance(author, Mapping):
            name = _text(" ".join(filter(None, [_text(author.get("given")), _text(author.get("family"))])))
            if name:
                authors.append(name)
    return authors


def _authors_openalex(item: Mapping[str, Any]) -> list[str]:
    authors = []
    for authorship in _list(item.get("authorships")):
        if isinstance(authorship, Mapping) and isinstance(authorship.get("author"), Mapping):
            name = _text(authorship["author"].get("display_name"))
            if name:
                authors.append(name)
    return authors


def _full_text(url: str, provider: str, basis: str, license_name: str | None = None) -> dict[str, Any]:
    return {
        "url": url,
        "provider": provider,
        "access_basis": basis,
        "license": license_name,
        "downloaded": False,
    }


def _europe_pmc(payload: Mapping[str, Any], query: dict[str, Any], request_url: str) -> list[dict[str, Any]]:
    result_list = payload.get("resultList")
    if not isinstance(result_list, Mapping) or not isinstance(result_list.get("result"), list):
        raise DiscoveryError("Europe PMC response is missing resultList.result")
    results = result_list["result"]
    records = []
    for item in _list(results):
        if not isinstance(item, Mapping):
            continue
        pmid, pmcid, doi = _text(item.get("pmid")) or None, _text(item.get("pmcid")) or None, _doi(item.get("doi"))
        links = []
        if pmcid and str(item.get("isOpenAccess", "")).upper() == "Y":
            links.append(_full_text(f"https://europepmc.org/articles/{pmcid}", "europe_pmc", "pmc-open-access"))
        full_text_list = item.get("fullTextUrlList", {})
        if str(item.get("isOpenAccess", "")).upper() == "Y" and isinstance(full_text_list, Mapping):
            for link in _list(full_text_list.get("fullTextUrl")):
                if isinstance(link, Mapping) and _text(link.get("url")).startswith("http"):
                    links.append(_full_text(_text(link["url"]), "europe_pmc", "api-marked-open-access"))
        article_url = (
            f"https://europepmc.org/article/MED/{pmid}" if pmid else
            f"https://doi.org/{doi}" if doi else request_url
        )
        records.append({
            "title": _text(item.get("title")), "authors": [_text(item.get("authorString"))] if _text(item.get("authorString")) else [],
            "year": int(item["pubYear"]) if str(item.get("pubYear", "")).isdigit() else None,
            "journal": _text(item.get("journalTitle")), "publication_type": _text(item.get("pubType")).lower(),
            "doi": doi, "pmid": pmid, "pmcid": pmcid, "abstract": _text(item.get("abstractText")),
            "article_url": article_url, "lawful_full_text_links": links,
            "origin": {"provider": "europe_pmc", "query_id": query["id"], "request_url": request_url},
        })
    return records


def _crossref(payload: Mapping[str, Any], query: dict[str, Any], request_url: str) -> list[dict[str, Any]]:
    message = payload.get("message")
    if not isinstance(message, Mapping) or not isinstance(message.get("items"), list):
        raise DiscoveryError("Crossref response is missing message.items")
    items = message["items"]
    records = []
    for item in _list(items):
        if not isinstance(item, Mapping):
            continue
        licenses = [entry for entry in _list(item.get("license")) if isinstance(entry, Mapping) and _text(entry.get("URL"))]
        links = []
        if licenses:
            license_url = _text(licenses[0].get("URL"))
            for link in _list(item.get("link")):
                if isinstance(link, Mapping) and _text(link.get("URL")).startswith("http"):
                    links.append(_full_text(_text(link["URL"]), "crossref", "publisher-link-with-license", license_url))
        titles = _list(item.get("title"))
        containers = _list(item.get("container-title"))
        doi = _doi(item.get("DOI"))
        records.append({
            "title": _text(titles[0]) if titles else "", "authors": _authors_crossref(item),
            "year": _year_from_parts(item.get("published")) or _year_from_parts(item.get("published-print")),
            "journal": _text(containers[0]) if containers else "", "publication_type": _text(item.get("type")).lower(),
            "doi": doi, "pmid": None, "pmcid": None, "abstract": _text(item.get("abstract")),
            "article_url": _text(item.get("URL")) or (f"https://doi.org/{doi}" if doi else request_url),
            "lawful_full_text_links": links,
            "origin": {"provider": "crossref", "query_id": query["id"], "request_url": request_url},
        })
    return records


def _openalex(payload: Mapping[str, Any], query: dict[str, Any], request_url: str) -> list[dict[str, Any]]:
    if not isinstance(payload.get("results"), list):
        raise DiscoveryError("OpenAlex response is missing results")
    records = []
    for item in _list(payload.get("results")):
        if not isinstance(item, Mapping):
            continue
        ids = item.get("ids") if isinstance(item.get("ids"), Mapping) else {}
        doi = _doi(ids.get("doi") or item.get("doi"))
        pmid, pmcid = _external_id(ids.get("pmid"), "PMID"), _external_id(ids.get("pmcid"), "PMC")
        location = item.get("best_oa_location") if isinstance(item.get("best_oa_location"), Mapping) else {}
        oa = item.get("open_access") if isinstance(item.get("open_access"), Mapping) else {}
        links = []
        if oa.get("is_oa") is True:
            for candidate in (location.get("pdf_url"), location.get("landing_page_url"), oa.get("oa_url")):
                link = _text(candidate)
                if link.startswith("http"):
                    links.append(_full_text(link, "openalex", "openalex-marked-open-access", _text(location.get("license")) or None))
        primary = item.get("primary_location") if isinstance(item.get("primary_location"), Mapping) else {}
        source = primary.get("source") if isinstance(primary.get("source"), Mapping) else {}
        records.append({
            "title": _text(item.get("title") or item.get("display_name")), "authors": _authors_openalex(item),
            "year": item.get("publication_year") if isinstance(item.get("publication_year"), int) else None,
            "journal": _text(source.get("display_name")), "publication_type": _text(item.get("type")).lower(),
            "doi": doi, "pmid": pmid, "pmcid": pmcid, "abstract": _abstract_from_inverted(item.get("abstract_inverted_index")),
            "article_url": _text(primary.get("landing_page_url") or item.get("id")) or (f"https://doi.org/{doi}" if doi else request_url),
            "lawful_full_text_links": links,
            "origin": {"provider": "openalex", "query_id": query["id"], "request_url": request_url},
        })
    return records


PROVIDER_CONFIG: dict[str, tuple[str, Callable[[dict[str, Any]], dict[str, str]], Callable[..., list[dict[str, Any]]]]] = {
    "europe_pmc": (
        "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
        lambda query: {"query": query["query"], "format": "json", "pageSize": "25", "resultType": "core"},
        _europe_pmc,
    ),
    "crossref": (
        "https://api.crossref.org/works",
        lambda query: {"query.bibliographic": query["query"], "rows": "25"},
        _crossref,
    ),
    "openalex": (
        "https://api.openalex.org/works",
        lambda query: {"search": query["query"], "per-page": "25"},
        _openalex,
    ),
}


def _identities(record: Mapping[str, Any]) -> set[str]:
    identities = {
        f"{field}:{_text(record.get(field)).lower()}"
        for field in ("doi", "pmid", "pmcid") if _text(record.get(field))
    }
    title = re.sub(r"[^a-z0-9]+", " ", _text(record.get("title")).lower()).strip()
    if title:
        identities.add(f"title:{title}:{record.get('year') or 'unknown'}")
    return identities


def _merge_records(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    parents = list(range(len(records)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    owner: dict[str, int] = {}
    for index, record in enumerate(records):
        for identity in _identities(record):
            if identity in owner:
                union(index, owner[identity])
            else:
                owner[identity] = index

    components: dict[int, list[dict[str, Any]]] = {}
    for index, record in enumerate(records):
        components.setdefault(find(index), []).append(record)

    merged: list[dict[str, Any]] = []
    for component in components.values():
        current = {**component[0], "origins": []}
        abstract_origin = component[0]["origin"] if component[0].get("abstract") else None
        current.pop("origin", None)
        current["authors"] = []
        current["lawful_full_text_links"] = []
        for incoming in component:
            for field in ("title", "abstract"):
                if len(_text(incoming.get(field))) > len(_text(current.get(field))):
                    current[field] = incoming[field]
                    if field == "abstract":
                        abstract_origin = incoming["origin"]
            for field in ("year", "journal", "publication_type", "doi", "pmid", "pmcid", "article_url"):
                if not current.get(field) and incoming.get(field):
                    current[field] = incoming[field]
            current["authors"] = sorted(set(current["authors"]) | set(incoming.get("authors", [])))
            current["lawful_full_text_links"] = _unique_dicts(
                current["lawful_full_text_links"] + incoming.get("lawful_full_text_links", []), "url"
            )
            current["origins"].append(incoming["origin"])
        current["abstract_origin"] = abstract_origin
        merged.append(current)
    return merged


def _unique_dicts(values: Sequence[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    result, seen = [], set()
    for value in values:
        key = value.get(field)
        if key and key not in seen:
            seen.add(key)
            result.append(value)
    return result


def _snippet(text: str, needles: set[str], width: int = 360) -> str:
    clean = _text(text)
    if not clean:
        return ""
    lowered = clean.lower()
    starts = [lowered.find(needle.lower()) for needle in needles if lowered.find(needle.lower()) >= 0]
    center = min(starts) if starts else 0
    start = max(0, center - width // 3)
    end = min(len(clean), start + width)
    prefix, suffix = ("..." if start else ""), ("..." if end < len(clean) else "")
    return prefix + clean[start:end].strip() + suffix


def _rank(record: dict[str, Any], wall: Mapping[str, Any]) -> dict[str, Any]:
    preparation = {item.lower() for item in _flatten_context(wall["preparation"])}
    mechanisms = {item.lower() for item in wall.get("mechanisms", [])}
    question_tokens = set().union(*(_tokens(item["text"]) for item in wall["questions"]))
    text = " ".join([record.get("title", ""), record.get("abstract", ""), record.get("journal", "")])
    text_tokens = _tokens(text)
    prep_hits = _phrase_hits(text, preparation)
    mechanism_hits = _phrase_hits(text, mechanisms)
    quantitative_hits = _phrase_hits(text, QUANTITATIVE_TERMS)
    question_hits = sorted(question_tokens & text_tokens)
    review_markers = ("review", "meta-analysis", "meta analysis")
    publication_type = record.get("publication_type", "").lower()
    primary = (
        any(kind == publication_type or kind in publication_type for kind in PRIMARY_TYPES)
        and not any(marker in f"{record.get('title', '')} {publication_type}".lower() for marker in review_markers)
    )
    score = min(30, 4 * len(prep_hits)) + min(30, 5 * len(mechanism_hits))
    score += min(20, 2 * len(quantitative_hits)) + min(10, len(question_hits))
    score += 8 if primary else 0
    score += 2 if record.get("lawful_full_text_links") else 0
    if primary and (prep_hits or mechanism_hits) and quantitative_hits:
        strength = "primary-quantitative-candidate"
    elif primary:
        strength = "primary-source-candidate"
    else:
        strength = "context-only"
    record["rank"] = {
        "score": score,
        "preparation_hits": prep_hits,
        "mechanism_hits": mechanism_hits,
        "quantitative_method_hits": quantitative_hits,
        "question_term_hits": question_hits,
    }
    record["evidence_strength"] = strength
    record["primary_source_status"] = "candidate-requires-review" if primary else "not-established-as-primary"
    snippet = _snippet(record.get("abstract", ""), preparation | mechanisms | QUANTITATIVE_TERMS)
    abstract_origin = record.pop("abstract_origin", None)
    record["extraction_candidates"] = ([{
        "source": "abstract-metadata",
        "locator": "abstract",
        "snippet": snippet,
        "metadata_provider": abstract_origin["provider"] if abstract_origin else None,
        "metadata_request_url": abstract_origin["request_url"] if abstract_origin else None,
        "article_url": record.get("article_url"),
        "review_status": "required",
        "may_resolve_question_ids": sorted({qid for origin in record["origins"] for qid in origin.get("question_ids", [])}),
        "exact_parameter_claim": False,
        "warning": "This excerpt is a discovery lead, not verified parameter or wiring evidence.",
    }] if snippet else [])
    record["unresolved_fields"] = [
        "exact_parameter_values_and_units",
        "full_text_methods_or_results_locator",
        "preparation_match",
        "primary_experimental_status",
        "wiring_direction_and_geometry",
    ]
    record["origins"] = sorted(record["origins"], key=lambda item: (item["provider"], item["query_id"]))
    return record


def discover(
    wall: Mapping[str, Any],
    *,
    client: JsonHttpClient,
    now: Callable[[], str] = utc_now,
    timeout: float = 20.0,
) -> dict[str, Any]:
    """Search every provider and return a complete discovery packet in memory."""
    if timeout <= 0:
        raise DiscoveryError("timeout must be positive")
    normalized = validate_wall(wall)
    queries = generate_query_variants(normalized)
    requests: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    for provider in PROVIDERS:
        base, params_for, parser = PROVIDER_CONFIG[provider]
        for query in queries:
            request_url = _url(base, params_for(query))
            requested_at = now()
            try:
                payload = client.get_json(request_url, timeout=timeout)
                parsed = parser(payload, query, request_url)
            except Exception as exc:
                errors.append(f"{provider}/{query['id']}: {exc}")
                requests.append({
                    "provider": provider, "query_id": query["id"], "query": query["query"],
                    "requested_at": requested_at, "source_url": request_url, "status": "failed",
                    "completed_at": now(), "error": str(exc),
                })
                continue
            for record in parsed:
                record["origin"]["question_ids"] = query["question_ids"]
            records.extend(parsed)
            requests.append({
                "provider": provider, "query_id": query["id"], "query": query["query"],
                "requested_at": requested_at, "source_url": request_url, "status": "complete",
                "completed_at": now(), "result_count": len(parsed),
            })
    if errors:
        raise DiscoveryError("partial provider failure; no packet may be emitted: " + " | ".join(errors))
    ranked = [_rank(record, normalized) for record in _merge_records(records) if record.get("title")]
    ranked.sort(key=lambda item: (-item["rank"]["score"], -(item.get("year") or 0), item["title"].casefold()))
    for index, record in enumerate(ranked, start=1):
        record["candidate_id"] = f"C{index}"
    packet: dict[str, Any] = {
        "schema": SCHEMA,
        "created_at": now(),
        "status": "complete-review-required",
        "wall": normalized,
        "queries": queries,
        "requests": requests,
        "providers_required": list(PROVIDERS),
        "candidates": ranked,
        "unresolved_questions": [
            {"id": question["id"], "text": question["text"], "status": "unresolved-pending-full-text-review"}
            for question in normalized["questions"]
        ],
        "review_contract": {
            "automatic_parameter_claims_allowed": False,
            "abstracts_are_discovery_evidence_only": True,
            "source_intake_required": True,
            "next_step": "Review the primary full text, preserve an exact page/figure/table/methods locator, then use research_escalation.py record-source and source_intake.",
        },
    }
    packet["canonicalization"] = CANONICALIZATION
    packet["packet_sha256_scope"] = "canonical packet JSON excluding packet_sha256"
    digest_payload = json.dumps(packet, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    packet["packet_sha256"] = hashlib.sha256(digest_payload).hexdigest()
    return packet


def write_packet_create_only(path: Path, packet: Mapping[str, Any]) -> None:
    """Publish a packet without replacing any existing destination."""
    path = path.expanduser().absolute()
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        if not os.path.lexists(current):
            continue
        try:
            mode = current.lstat().st_mode
        except OSError as exc:
            raise DiscoveryError(f"cannot inspect discovery packet path: {current}: {exc}") from exc
        if stat.S_ISLNK(mode):
            raise DiscoveryError(f"discovery packet path cannot contain a symlink: {current}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise DiscoveryError(f"refusing to overwrite existing discovery packet: {path}")
    payload = json.dumps(packet, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise DiscoveryError(f"refusing to overwrite existing discovery packet: {path}") from exc
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _load_wall(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DiscoveryError(f"cannot read wall {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise DiscoveryError("wall JSON must be an object")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wall", type=Path, required=True, help="structured wall JSON")
    parser.add_argument("--output", type=Path, required=True, help="new discovery packet JSON")
    parser.add_argument("--timeout", type=float, default=20.0, help="per-request timeout in seconds")
    args = parser.parse_args(argv)
    try:
        if args.output.exists() or args.output.is_symlink():
            raise DiscoveryError(f"refusing to overwrite existing discovery packet: {args.output}")
        packet = discover(_load_wall(args.wall), client=UrllibJsonClient(), timeout=args.timeout)
        write_packet_create_only(args.output, packet)
    except DiscoveryError as exc:
        parser.error(str(exc))
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
