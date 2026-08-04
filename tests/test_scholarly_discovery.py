from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.scholarly_discovery import (  # noqa: E402
    HttpResponse,
    ScholarlyDiscoveryClient,
    ScholarlyDiscoveryError,
)


OPENALEX = {
    "results": [{
        "id": "https://openalex.org/W123",
        "doi": "https://doi.org/10.1000/ABC",
        "title": "  A quantitative neural study  ",
        "authorships": [
            {"author": {"display_name": "Ada Lovelace"}},
            {"author": {"display_name": "Grace Hopper"}},
        ],
        "publication_year": 2024,
        "type": "article",
        "primary_location": {
            "landing_page_url": "https://doi.org/10.1000/ABC",
            "pdf_url": "https://repository.example/paper.pdf",
        },
        "best_oa_location": {
            "landing_page_url": "https://repository.example/item/7",
            "pdf_url": "https://repository.example/paper.pdf",
        },
        "open_access": {"is_oa": True},
    }]
}

CROSSREF = {
    "message": {"items": [{
        "DOI": "10.1000/abc",
        "URL": "https://doi.org/10.1000/abc",
        "title": ["A quantitative neural study"],
        "author": [{"given": "Ada", "family": "Lovelace"}],
        "published": {"date-parts": [[2024, 2, 3]]},
        "type": "journal-article",
        "link": [{"URL": "https://publisher.example/full.pdf", "content-type": "application/pdf"}],
        "resource": {"primary": {"URL": "https://publisher.example/article"}},
    }]}
}


class FixtureTransport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, url, headers, timeout):
        self.calls.append((url, dict(headers), timeout))
        response = self.responses.pop(0)
        if isinstance(response, HttpResponse):
            return response
        return HttpResponse(200, {}, json.dumps(response).encode())


def client(transport, **options):
    return ScholarlyDiscoveryClient(
        user_agent="sim-research/1.0 (contact: research@example.test)",
        transport=transport,
        sleep=options.pop("sleep", lambda _: None),
        **options,
    )


def test_discovers_deduplicated_metadata_and_preserves_evidence_boundary():
    transport = FixtureTransport([OPENALEX, CROSSREF])

    result = client(transport).discover({"id": "gap-Q1", "text": "SNr calcium channel density"})

    assert result["schema_version"] == "scholarly-discovery-v1"
    assert result["query"] == {"id": "gap-Q1", "text": "SNr calcium channel density"}
    assert [item["provider"] for item in result["provider_searches"]] == ["openalex", "crossref"]
    assert len(result["candidates"]) == 1
    candidate = result["candidates"][0]
    assert candidate == {
        "record_kind": "metadata_candidate",
        "content_status": "metadata_only",
        "provider_records": [
            {"provider": "crossref", "provider_id": "10.1000/abc"},
            {"provider": "openalex", "provider_id": "https://openalex.org/W123"},
        ],
        "doi": "10.1000/abc",
        "canonical_url": "https://doi.org/10.1000/ABC",
        "full_text_url_leads": [
            "https://publisher.example/full.pdf",
            "https://repository.example/item/7",
            "https://repository.example/paper.pdf",
        ],
        "title": "A quantitative neural study",
        "authors": ["Ada Lovelace", "Grace Hopper"],
        "year": 2024,
        "type": "article",
    }
    assert result["evidence_boundary"] == {
        "content_status": "metadata_only",
        "full_text_retrieved": False,
        "full_text_urls_are_leads_only": True,
    }
    serialized = json.dumps(result, sort_keys=True)
    for forbidden in ('"exact_locator"', '"evidence"', '"quantitative_claim'):
        assert forbidden not in serialized

    assert all(call[1]["User-Agent"].startswith("sim-research/") for call in transport.calls)
    assert all(call[1]["Accept"] == "application/json" for call in transport.calls)
    assert all(call[2] == 15.0 for call in transport.calls)
    assert "search=SNr+calcium+channel+density" in transport.calls[0][0]
    assert "query.bibliographic=SNr+calcium+channel+density" in transport.calls[1][0]


def test_retry_is_bounded_and_honors_capped_retry_after():
    sleeps = []
    transport = FixtureTransport([
        HttpResponse(429, {"Retry-After": "99"}, b"rate limited"),
        OPENALEX,
        CROSSREF,
    ])

    result = client(transport, max_retries=1, retry_after_cap=2.5, sleep=sleeps.append).discover(
        {"id": "Q1", "text": "ion conductance"}
    )

    assert sleeps == [2.5]
    assert result["provider_searches"][0]["attempts"] == 2
    assert len(transport.calls) == 3


def test_malformed_retry_after_is_treated_as_zero_delay():
    sleeps = []
    transport = FixtureTransport([
        HttpResponse(500, {"retry-after": "not a date"}, b"temporary"),
        OPENALEX,
        CROSSREF,
    ])

    client(transport, max_retries=1, sleep=sleeps.append).discover({"id": "Q1", "text": "ion conductance"})

    assert sleeps == [0.0]


def test_retry_exhaustion_and_nontransient_errors_fail_closed():
    transient = FixtureTransport([
        HttpResponse(503, {}, b"unavailable"),
        HttpResponse(503, {}, b"unavailable"),
    ])
    with pytest.raises(ScholarlyDiscoveryError, match=r"HTTP 503 after 2 attempt"):
        client(transient, max_retries=1).discover({"id": "Q1", "text": "channel kinetics"})
    assert len(transient.calls) == 2

    permanent = FixtureTransport([HttpResponse(400, {}, b"bad query")])
    with pytest.raises(ScholarlyDiscoveryError, match=r"HTTP 400 after 1 attempt"):
        client(permanent, max_retries=3).discover({"id": "Q1", "text": "channel kinetics"})
    assert len(permanent.calls) == 1


def test_provider_record_without_doi_is_retained_and_url_deduplicated():
    openalex = {"results": [{
        "id": "https://openalex.org/W9",
        "doi": None,
        "title": "No DOI work",
        "authorships": [],
        "publication_year": 2020,
        "type": "preprint",
        "primary_location": {"landing_page_url": "https://example.test/work/9/"},
        "best_oa_location": None,
    }]}
    crossref = {"message": {"items": [{
        "URL": "https://example.test/work/9",
        "title": ["No DOI work"],
        "author": [],
        "published": {"date-parts": [[2020]]},
        "type": "posted-content",
        "link": [],
    }]}}

    result = client(FixtureTransport([openalex, crossref])).discover({"id": "Q9", "text": "rare result"})

    assert len(result["candidates"]) == 1
    assert result["candidates"][0]["doi"] is None
    assert len(result["candidates"][0]["provider_records"]) == 2


@pytest.mark.parametrize(
    "query",
    [None, {}, {"id": "Q1"}, {"text": "something"}, {"id": " ", "text": "something"}],
)
def test_invalid_planned_query_is_rejected_before_network(query):
    transport = FixtureTransport([])
    with pytest.raises(ScholarlyDiscoveryError, match="planned_query"):
        client(transport).discover(query)
    assert transport.calls == []


def test_malformed_provider_payload_and_invalid_json_fail_closed():
    with pytest.raises(ScholarlyDiscoveryError, match="OpenAlex response"):
        client(FixtureTransport([{"unexpected": []}])).discover({"id": "Q", "text": "valid query"})

    invalid_json = FixtureTransport([HttpResponse(200, {}, b"not-json")])
    with pytest.raises(ScholarlyDiscoveryError, match="invalid JSON"):
        client(invalid_json).discover({"id": "Q", "text": "valid query"})


def test_result_is_deterministic_and_contains_only_json_primitives():
    first = client(FixtureTransport([OPENALEX, CROSSREF])).discover({"id": "Q", "text": "same query"})
    second = client(FixtureTransport([OPENALEX, CROSSREF])).discover({"id": "Q", "text": "same query"})

    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert json.loads(json.dumps(first)) == first
