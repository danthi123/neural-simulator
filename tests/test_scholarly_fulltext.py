from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.scholarly_fulltext import (  # noqa: E402
    HttpResponse,
    ScholarlyFulltextError,
    ScholarlyFulltextRetriever,
    locate_parameter_passages,
    _receipt_sha256,
)


LEAD = {
    "record_kind": "metadata_candidate",
    "content_status": "metadata_only",
    "doi": "10.1000/example",
    "canonical_url": "https://doi.org/10.1000/example",
    "title": "Example physiology",
    "full_text_url_leads": ["https://repository.example/paper.txt"],
}


class FixtureTransport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, url, headers, timeout, max_bytes):
        self.calls.append((url, dict(headers), timeout, max_bytes))
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def response(body=b"conductance was 12 nS\n", *, mime="text/plain", final_url=None, status=200):
    return HttpResponse(
        status=status,
        headers={"Content-Type": mime},
        body=body,
        final_url=final_url or "https://repository.example/paper.txt",
    )


def retriever(transport, **options):
    return ScholarlyFulltextRetriever(
        user_agent="sim-research/1.0 (contact: research@example.test)",
        transport=transport,
        **options,
    )


def test_retrieves_declared_lead_persists_digest_and_candidate_locator(tmp_path):
    transport = FixtureTransport([response()])

    receipt = retriever(transport).retrieve(
        LEAD,
        store=tmp_path,
        parameter_terms=["conductance"],
        unit_patterns=[r"\bnS\b"],
        context_lines=0,
    )

    assert receipt["schema_version"] == "scholarly-fulltext-v1"
    retrieval = receipt["retrieval"]
    content = tmp_path / retrieval["content_path"]
    assert content.read_bytes() == b"conductance was 12 nS\n"
    assert content.name.startswith(retrieval["content_sha256"])
    assert receipt["candidate_locators"] == [{
        "record_kind": "candidate_parameter_locator",
        "locator": "lines 1-1",
        "page": None,
        "line_start": 1,
        "line_end": 1,
        "matched_parameter_terms": ["conductance"],
        "matched_unit_patterns": [r"\bnS\b"],
        "passage": "conductance was 12 nS",
        "claim_status": "not_a_claim",
        "review_status": "pending_review",
    }]
    assert receipt["evidence_boundary"]["accepted_claims"] is False
    assert receipt["evidence_boundary"]["interpreted_parameter_values"] is False
    assert transport.calls == [(
        "https://repository.example/paper.txt",
        {
            "Accept": "application/pdf, application/xml, text/xml, text/html, text/plain",
            "User-Agent": "sim-research/1.0 (contact: research@example.test)",
        },
        20.0,
        32 * 1024 * 1024,
    )]


def test_form_feed_gives_exact_pdf_page_and_page_local_lines():
    text = "intro\nno result\fheading\nSK conductance\nwas 8 nS\nclosing\fappendix"

    passages = locate_parameter_passages(
        text,
        kind="pdf",
        parameter_terms=["SK conductance"],
        unit_patterns=[r"\bnS\b"],
        context_lines=1,
    )

    assert len(passages) == 1
    assert passages[0]["locator"] == "page 2, lines 1-3"
    assert passages[0]["page"] == 2
    assert passages[0]["passage"] == "heading\nSK conductance\nwas 8 nS"


@pytest.mark.parametrize(
    ("mime", "body", "message"),
    [
        ("application/zip", b"PK\x03\x04payload", "archive or executable"),
        ("application/octet-stream", b"arbitrary", "unsupported MIME"),
        ("application/pdf", b"not actually pdf", "does not match"),
        ("text/plain", b"%PDF-1.7", "does not match"),
        ("text/plain", b"\x7fELFpayload", "archive or executable"),
    ],
)
def test_rejects_unsupported_mime_and_dangerous_or_mismatched_content(tmp_path, mime, body, message):
    transport = FixtureTransport([response(body, mime=mime)])
    with pytest.raises(ScholarlyFulltextError, match=message):
        retriever(transport).retrieve(
            LEAD,
            store=tmp_path,
            parameter_terms=["conductance"],
            unit_patterns=["nS"],
        )
    assert not list((tmp_path / "receipts").glob("*.json"))


def test_size_limit_and_non_http_redirect_fail_closed(tmp_path):
    oversized = FixtureTransport([response(b"12345")])
    with pytest.raises(ScholarlyFulltextError, match="exceeds max_bytes"):
        retriever(oversized, max_bytes=4).retrieve(
            LEAD, store=tmp_path / "size", parameter_terms=["x"], unit_patterns=["nS"]
        )

    redirected = FixtureTransport([response(final_url="file:///etc/passwd")])
    with pytest.raises(ScholarlyFulltextError, match="redirect target"):
        retriever(redirected).retrieve(
            LEAD, store=tmp_path / "redirect", parameter_terms=["x"], unit_patterns=["nS"]
        )


@pytest.mark.parametrize(
    "bad_url",
    ["file:///tmp/paper", "../paper.pdf", "https://user:secret@example.test/paper", ""],
)
def test_only_declared_safe_http_urls_are_attempted(tmp_path, bad_url):
    lead = {**LEAD, "full_text_url_leads": [bad_url]}
    transport = FixtureTransport([])
    with pytest.raises(ScholarlyFulltextError, match="safe HTTP"):
        retriever(transport).retrieve(
            lead, store=tmp_path, parameter_terms=["x"], unit_patterns=["nS"]
        )
    assert transport.calls == []


def test_resumes_from_atomic_receipt_without_network_or_reconversion(tmp_path):
    conversions = []

    def convert(body, kind, timeout):
        conversions.append((body, kind, timeout))
        return "conductance 5 nS"

    first_transport = FixtureTransport([response(b"source")])
    first = retriever(first_transport, converter=convert).retrieve(
        LEAD, store=tmp_path, parameter_terms=["conductance"], unit_patterns=["nS"]
    )
    receipt_files = list((tmp_path / "receipts").glob("*.json"))
    assert len(receipt_files) == 1
    assert not list((tmp_path / "receipts").glob("*.tmp"))
    assert json.loads(receipt_files[0].read_text()) == first

    second_transport = FixtureTransport([])
    second = retriever(second_transport, converter=convert).retrieve(
        LEAD, store=tmp_path, parameter_terms=["conductance"], unit_patterns=["nS"]
    )
    assert second == first
    assert second_transport.calls == []
    assert len(conversions) == 1


def test_same_content_from_different_leads_is_deduplicated(tmp_path):
    body = b"conductance 5 nS"
    first = retriever(FixtureTransport([response(body)])).retrieve(
        LEAD, store=tmp_path, parameter_terms=["conductance"], unit_patterns=["nS"]
    )
    other = {
        **LEAD,
        "doi": "10.1000/other",
        "full_text_url_leads": ["https://other.example/full.txt"],
    }
    second = retriever(FixtureTransport([
        response(body, final_url="https://other.example/full.txt")
    ])).retrieve(
        other, store=tmp_path, parameter_terms=["conductance"], unit_patterns=["nS"]
    )

    assert first["retrieval"]["content_path"] == second["retrieval"]["content_path"]
    assert len(list((tmp_path / "content").iterdir())) == 1
    assert len(list((tmp_path / "receipts").iterdir())) == 2


def test_corrupt_resume_and_receipt_path_escape_fail_closed(tmp_path):
    receipt = retriever(FixtureTransport([response()])).retrieve(
        LEAD, store=tmp_path, parameter_terms=["conductance"], unit_patterns=["nS"]
    )
    content_path = tmp_path / receipt["retrieval"]["content_path"]
    content_path.write_bytes(b"tampered")
    with pytest.raises(ScholarlyFulltextError, match="missing or corrupt"):
        retriever(FixtureTransport([])).retrieve(
            LEAD, store=tmp_path, parameter_terms=["conductance"], unit_patterns=["nS"]
        )

    content_path.write_bytes(b"conductance was 12 nS\n")
    receipt_path = next((tmp_path / "receipts").glob("*.json"))
    stored = json.loads(receipt_path.read_text())
    stored["retrieval"]["content_path"] = "../../outside"
    stored["sha256"] = _receipt_sha256(stored)
    receipt_path.write_text(json.dumps(stored))
    with pytest.raises(ScholarlyFulltextError, match="escapes store"):
        retriever(FixtureTransport([])).retrieve(
            LEAD, store=tmp_path, parameter_terms=["conductance"], unit_patterns=["nS"]
        )


def test_resume_rejects_tampered_locator_even_when_content_is_intact(tmp_path):
    receipt = retriever(FixtureTransport([response()])).retrieve(
        LEAD, store=tmp_path, parameter_terms=["conductance"], unit_patterns=["nS"]
    )
    receipt_path = next((tmp_path / "receipts").glob("*.json"))
    stored = json.loads(receipt_path.read_text())
    stored["candidate_locators"][0]["passage"] = "fabricated 999 nS"
    receipt_path.write_text(json.dumps(stored))

    with pytest.raises(ScholarlyFulltextError, match="receipt digest"):
        retriever(FixtureTransport([])).retrieve(
            LEAD, store=tmp_path, parameter_terms=["conductance"], unit_patterns=["nS"]
        )
    assert receipt["sha256"] != _receipt_sha256(stored)


def test_converter_is_injected_and_document_kinds_are_distinguished(tmp_path):
    cases = [
        ("application/pdf", b"%PDF-1.7 fixture", "pdf"),
        ("application/xml", b"<?xml version='1.0'?><p>x</p>", "xml"),
        ("text/html; charset=utf-8", b"<html><p>x</p></html>", "html"),
        ("text/plain", b"x", "plain"),
    ]
    observed = []
    for index, (mime, body, expected) in enumerate(cases):
        lead = {
            **LEAD,
            "doi": f"10.1000/{index}",
            "full_text_url_leads": [f"https://example.test/{index}"],
        }
        transport = FixtureTransport([
            response(body, mime=mime, final_url=f"https://example.test/{index}")
        ])

        def convert(content, kind, timeout):
            observed.append((content, kind, timeout))
            return "parameter 1 nS"

        receipt = retriever(transport, converter=convert).retrieve(
            lead,
            store=tmp_path,
            parameter_terms=["parameter"],
            unit_patterns=["nS"],
        )
        assert receipt["retrieval"]["document_kind"] == expected
    assert [item[1] for item in observed] == ["pdf", "xml", "html", "plain"]


def test_failed_first_declared_lead_falls_through_without_trying_undeclared_urls(tmp_path):
    lead = {
        **LEAD,
        "full_text_url_leads": [
            "https://repository.example/blocked",
            "https://repository.example/open.txt",
        ],
    }
    transport = FixtureTransport([
        response(b"denied", status=403),
        response(
            b"conductance 2 nS",
            final_url="https://repository.example/open.txt",
        ),
    ])
    receipt = retriever(transport).retrieve(
        lead, store=tmp_path, parameter_terms=["conductance"], unit_patterns=["nS"]
    )
    assert [call[0] for call in transport.calls] == lead["full_text_url_leads"]
    assert receipt["retrieval"]["declared_url"] == lead["full_text_url_leads"][1]


def test_output_has_no_claim_or_interpreted_value_fields(tmp_path):
    receipt = retriever(FixtureTransport([response()])).retrieve(
        LEAD, store=tmp_path, parameter_terms=["conductance"], unit_patterns=["nS"]
    )
    serialized = json.dumps(receipt, sort_keys=True)
    for forbidden in ('"accepted_claim"', '"parameter_value"', '"quantitative_claim"', '"evidence_excerpt"'):
        assert forbidden not in serialized
    assert '"claim_status": "not_a_claim"' in serialized
