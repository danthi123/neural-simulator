#!/usr/bin/env python3
"""No-key scholarly metadata discovery through OpenAlex and Crossref.

The output is discovery metadata, not scientific evidence.  In particular,
full-text URLs are leads that still need retrieval and review; this module does
not create exact locators, evidence excerpts, or quantitative claims.
"""
from __future__ import annotations

from dataclasses import dataclass
import email.utils
import json
import time
from typing import Any, Callable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen


SCHEMA_VERSION = "scholarly-discovery-v1"
TRANSIENT_STATUSES = {429, 500, 502, 503, 504}


class ScholarlyDiscoveryError(RuntimeError):
    """Raised when a query or provider response cannot be trusted."""


@dataclass(frozen=True)
class HttpResponse:
    status: int
    headers: Mapping[str, str]
    body: bytes


Transport = Callable[[str, Mapping[str, str], float], HttpResponse]
Sleeper = Callable[[float], None]


def _default_transport(url: str, headers: Mapping[str, str], timeout: float) -> HttpResponse:
    request = Request(url, headers=dict(headers), method="GET")
    try:
        with urlopen(request, timeout=timeout) as response:
            return HttpResponse(response.status, dict(response.headers.items()), response.read())
    except HTTPError as exc:
        return HttpResponse(exc.code, dict(exc.headers.items()), exc.read())
    except (URLError, TimeoutError, OSError) as exc:
        raise ScholarlyDiscoveryError(f"request failed for {url}: {exc}") from exc


def _text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value = " ".join(value.split())
    return value or None


def _http_url(value: Any) -> str | None:
    value = _text(value)
    if value is None:
        return None
    parsed = urlsplit(value)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        return None
    path = parsed.path.rstrip("/") or "/"
    return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), path, parsed.query, ""))


def _doi(value: Any) -> str | None:
    value = _text(value)
    if value is None:
        return None
    lowered = value.lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if lowered.startswith(prefix):
            lowered = lowered[len(prefix):]
            break
    lowered = lowered.rstrip(".,;)")
    if not lowered.startswith("10.") or "/" not in lowered or any(char.isspace() for char in lowered):
        return None
    return lowered


def _string_list(values: Any) -> list[str]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return []
    return list(dict.fromkeys(value for item in values if (value := _text(item)) is not None))


def _url_list(values: Any) -> list[str]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return []
    return sorted(set(value for item in values if (value := _http_url(item)) is not None))


def _retry_delay(value: str | None, *, now: Callable[[], float], cap: float) -> float:
    if value is None:
        return 0.0
    try:
        delay = float(value)
    except ValueError:
        try:
            parsed = email.utils.parsedate_to_datetime(value)
        except (TypeError, ValueError, OverflowError):
            return 0.0
        if parsed is None:
            return 0.0
        delay = parsed.timestamp() - now()
    return min(cap, max(0.0, delay))


class ScholarlyDiscoveryClient:
    """Fetch and normalize public scholarly metadata with bounded retries."""

    def __init__(
        self,
        *,
        user_agent: str,
        timeout: float = 15.0,
        max_retries: int = 2,
        retry_after_cap: float = 5.0,
        per_provider: int = 25,
        transport: Transport = _default_transport,
        sleep: Sleeper = time.sleep,
        now: Callable[[], float] = time.time,
    ) -> None:
        if not _text(user_agent) or "/" not in user_agent:
            raise ValueError("user_agent must identify the application, for example sim-research/1.0")
        if timeout <= 0 or max_retries < 0 or retry_after_cap < 0 or not 1 <= per_provider <= 100:
            raise ValueError("invalid timeout, retry, cap, or result limit")
        self.user_agent = user_agent
        self.timeout = float(timeout)
        self.max_retries = max_retries
        self.retry_after_cap = float(retry_after_cap)
        self.per_provider = per_provider
        self.transport = transport
        self.sleep = sleep
        self.now = now

    def discover(self, planned_query: Mapping[str, Any]) -> dict[str, Any]:
        """Return deterministic metadata candidates for one planned query."""
        if not isinstance(planned_query, Mapping):
            raise ScholarlyDiscoveryError("planned_query must be an object")
        query_id = _text(planned_query.get("id"))
        query_text = _text(planned_query.get("text"))
        if query_id is None or query_text is None:
            raise ScholarlyDiscoveryError("planned_query requires non-empty id and text")

        requests = (
            ("openalex", self._openalex_url(query_text), self._normalize_openalex),
            ("crossref", self._crossref_url(query_text), self._normalize_crossref),
        )
        provider_searches: list[dict[str, Any]] = []
        records: list[dict[str, Any]] = []
        for provider, search_url, normalizer in requests:
            payload, attempts = self._get_json(search_url)
            provider_records = normalizer(payload)
            provider_searches.append({
                "provider": provider,
                "search_url": search_url,
                "attempts": attempts,
                "metadata_records": len(provider_records),
            })
            records.extend(provider_records)

        return {
            "schema_version": SCHEMA_VERSION,
            "query": {"id": query_id, "text": query_text},
            "provider_searches": provider_searches,
            "candidates": _deduplicate(records),
            "evidence_boundary": {
                "content_status": "metadata_only",
                "full_text_retrieved": False,
                "full_text_urls_are_leads_only": True,
            },
        }

    def _openalex_url(self, query: str) -> str:
        return "https://api.openalex.org/works?" + urlencode({
            "search": query,
            "per-page": self.per_provider,
            "select": "id,doi,title,authorships,publication_year,type,primary_location,best_oa_location,open_access",
        })

    def _crossref_url(self, query: str) -> str:
        return "https://api.crossref.org/works?" + urlencode({
            "query.bibliographic": query,
            "rows": self.per_provider,
            "select": "DOI,URL,title,author,published,type,link,resource",
        })

    def _get_json(self, url: str) -> tuple[Mapping[str, Any], int]:
        headers = {"Accept": "application/json", "User-Agent": self.user_agent}
        for attempt in range(1, self.max_retries + 2):
            response = self.transport(url, headers, self.timeout)
            if 200 <= response.status < 300:
                try:
                    payload = json.loads(response.body.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ScholarlyDiscoveryError(f"provider returned invalid JSON for {url}") from exc
                if not isinstance(payload, Mapping):
                    raise ScholarlyDiscoveryError(f"provider returned a non-object response for {url}")
                return payload, attempt
            if response.status not in TRANSIENT_STATUSES or attempt > self.max_retries:
                raise ScholarlyDiscoveryError(
                    f"provider request failed with HTTP {response.status} after {attempt} attempt(s): {url}"
                )
            retry_after = next(
                (value for key, value in response.headers.items() if key.lower() == "retry-after"),
                None,
            )
            self.sleep(_retry_delay(retry_after, now=self.now, cap=self.retry_after_cap))
        raise AssertionError("bounded retry loop exhausted unexpectedly")

    @staticmethod
    def _normalize_openalex(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
        items = payload.get("results")
        if not isinstance(items, list):
            raise ScholarlyDiscoveryError("OpenAlex response is missing results")
        records: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, Mapping):
                continue
            provider_id = _text(item.get("id"))
            title = _text(item.get("title"))
            if provider_id is None or title is None:
                continue
            primary = item.get("primary_location") if isinstance(item.get("primary_location"), Mapping) else {}
            best_oa = item.get("best_oa_location") if isinstance(item.get("best_oa_location"), Mapping) else {}
            open_access = item.get("open_access") if isinstance(item.get("open_access"), Mapping) else {}
            landing = _http_url(primary.get("landing_page_url"))
            doi = _doi(item.get("doi"))
            canonical = landing or (f"https://doi.org/{quote(doi, safe='/')}" if doi else _http_url(provider_id))
            authors: list[str] = []
            for authorship in item.get("authorships", []) if isinstance(item.get("authorships"), list) else []:
                author = authorship.get("author") if isinstance(authorship, Mapping) else None
                name = _text(author.get("display_name")) if isinstance(author, Mapping) else None
                if name:
                    authors.append(name)
            leads = _url_list([
                primary.get("pdf_url"),
                best_oa.get("pdf_url"),
                best_oa.get("landing_page_url"),
                open_access.get("oa_url"),
            ])
            records.append(_record(
                provider="openalex",
                provider_id=provider_id,
                doi=doi,
                canonical_url=canonical,
                full_text_url_leads=leads,
                title=title,
                authors=authors,
                year=item.get("publication_year"),
                work_type=item.get("type"),
            ))
        return records

    @staticmethod
    def _normalize_crossref(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
        message = payload.get("message")
        items = message.get("items") if isinstance(message, Mapping) else None
        if not isinstance(items, list):
            raise ScholarlyDiscoveryError("Crossref response is missing message.items")
        records: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, Mapping):
                continue
            doi = _doi(item.get("DOI"))
            provider_id = doi or _http_url(item.get("URL"))
            titles = _string_list(item.get("title"))
            if provider_id is None or not titles:
                continue
            authors: list[str] = []
            for author in item.get("author", []) if isinstance(item.get("author"), list) else []:
                if not isinstance(author, Mapping):
                    continue
                name = _text(author.get("name")) or _text(" ".join(
                    part for part in (_text(author.get("given")), _text(author.get("family"))) if part
                ))
                if name:
                    authors.append(name)
            year = None
            published = item.get("published")
            date_parts = published.get("date-parts") if isinstance(published, Mapping) else None
            if isinstance(date_parts, list) and date_parts and isinstance(date_parts[0], list) and date_parts[0]:
                year = date_parts[0][0]
            leads = []
            for link in item.get("link", []) if isinstance(item.get("link"), list) else []:
                if isinstance(link, Mapping):
                    leads.append(link.get("URL"))
            records.append(_record(
                provider="crossref",
                provider_id=provider_id,
                doi=doi,
                canonical_url=_http_url(item.get("URL")) or (f"https://doi.org/{quote(doi, safe='/')}" if doi else None),
                full_text_url_leads=_url_list(leads),
                title=titles[0],
                authors=authors,
                year=year,
                work_type=item.get("type"),
            ))
        return records


def _record(
    *,
    provider: str,
    provider_id: str,
    doi: str | None,
    canonical_url: str | None,
    full_text_url_leads: Sequence[str],
    title: str,
    authors: Sequence[str],
    year: Any,
    work_type: Any,
) -> dict[str, Any]:
    normalized_year = year if isinstance(year, int) and 1000 <= year <= 9999 else None
    return {
        "record_kind": "metadata_candidate",
        "content_status": "metadata_only",
        "provider_records": [{"provider": provider, "provider_id": provider_id}],
        "doi": doi,
        "canonical_url": canonical_url,
        "full_text_url_leads": sorted(set(full_text_url_leads)),
        "title": title,
        "authors": list(dict.fromkeys(authors)),
        "year": normalized_year,
        "type": _text(work_type),
    }


def _deduplicate(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for raw in records:
        record = dict(raw)
        match = next((item for item in merged if _same_work(item, record)), None)
        if match is None:
            merged.append(record)
            continue
        match["provider_records"] = sorted(
            {tuple(sorted(item.items())) for item in match["provider_records"] + record["provider_records"]}
        )
        match["provider_records"] = [dict(items) for items in match["provider_records"]]
        match["full_text_url_leads"] = sorted(set(
            match["full_text_url_leads"] + record["full_text_url_leads"]
        ))
        match["authors"] = list(dict.fromkeys(match["authors"] + record["authors"]))
        for field in ("doi", "canonical_url", "year", "type"):
            if match[field] is None:
                match[field] = record[field]
    return sorted(merged, key=lambda item: (
        item["doi"] or "",
        item["canonical_url"] or "",
        item["title"].casefold(),
    ))


def _same_work(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return bool(
        (left.get("doi") and left.get("doi") == right.get("doi"))
        or (
            left.get("canonical_url")
            and left.get("canonical_url") == right.get("canonical_url")
        )
    )


def discover(planned_query: Mapping[str, Any], **client_options: Any) -> dict[str, Any]:
    """Convenience entry point for adapter-style integration."""
    return ScholarlyDiscoveryClient(**client_options).discover(planned_query)
