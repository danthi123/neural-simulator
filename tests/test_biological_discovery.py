from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

from tools import biological_discovery as discovery


def wall() -> dict:
    return {
        "schema": discovery.WALL_SCHEMA,
        "wall_id": "gpi-tonic-output",
        "blocked_experiment": "GPi output cells do not sustain autonomous tonic firing.",
        "wall_reason": "Two bounded current and conductance scans remained trial-bound.",
        "preparation": {
            "species": "mouse",
            "brain_region": "globus pallidus internus",
            "cell_type": "GPi projection neuron",
            "state": "adult ex vivo slice",
            "recording": "whole-cell patch clamp",
        },
        "mechanisms": ["HCN current", "persistent sodium current"],
        "parameter_questions": [{"id": "P1", "text": "What autonomous firing rate and HCN conductance are measured?"}],
        "wiring_questions": [{"id": "W1", "text": "Which inhibitory inputs set tonic firing and where do they terminate?"}],
        "prior_attempts": ["Current sweep did not generalize."],
    }


class FakeClient:
    def __init__(self, *, fail_provider: str | None = None) -> None:
        self.fail_provider = fail_provider
        self.urls: list[str] = []

    def get_json(self, url: str, *, timeout: float):
        self.urls.append(url)
        assert timeout == 7.0
        host = urlparse(url).netloc
        query = parse_qs(urlparse(url).query)
        assert any(key in query for key in ("query", "query.bibliographic", "search"))
        failing_host = {
            "europepmc": "ebi.ac.uk",
            "crossref": "crossref",
            "openalex": "openalex",
        }.get(self.fail_provider or "")
        if failing_host and failing_host in host:
            raise TimeoutError("mock timeout")
        if "ebi.ac.uk" in host:
            return {
                "resultList": {"result": [{
                    "title": "HCN conductance supports autonomous firing in mouse GPi neurons",
                    "authorString": "A. Example, B. Example",
                    "pubYear": "2024", "journalTitle": "Neural Methods", "pubType": "article",
                    "doi": "10.1000/shared", "pmid": "123", "pmcid": "PMC456",
                    "abstractText": "Whole-cell patch clamp measurement found HCN conductance and tonic firing rate responses in adult mouse GPi neurons.",
                    "isOpenAccess": "Y",
                    "fullTextUrlList": {"fullTextUrl": [{"url": "https://europepmc.org/articles/PMC456?pdf=render"}]},
                }]}
            }
        if "crossref" in host:
            return {"message": {"items": [{
                "title": ["HCN conductance supports autonomous firing in mouse GPi neurons"],
                "author": [{"given": "A.", "family": "Example"}],
                "published": {"date-parts": [[2024]]}, "container-title": ["Neural Methods"],
                "type": "journal-article", "DOI": "https://doi.org/10.1000/SHARED",
                "URL": "https://doi.org/10.1000/shared",
                "abstract": "Electrophysiology and quantitative measurement of HCN conductance in mouse globus pallidus internus.",
                "license": [{"URL": "https://creativecommons.org/licenses/by/4.0/"}],
                "link": [{"URL": "https://publisher.test/shared.pdf", "content-type": "application/pdf"}],
            }, {
                "title": ["A qualitative review of basal ganglia"], "published": {"date-parts": [[2025]]},
                "type": "journal-article", "DOI": "10.1000/review", "URL": "https://doi.org/10.1000/review",
            }]}}
        if "openalex" in host:
            return {"results": [{
                "id": "https://openalex.org/W1", "title": "HCN conductance supports autonomous firing in mouse GPi neurons",
                "publication_year": 2024, "type": "article", "doi": "https://doi.org/10.1000/shared",
                "ids": {"pmid": "https://pubmed.ncbi.nlm.nih.gov/123", "pmcid": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC456"},
                "authorships": [{"author": {"display_name": "A. Example"}}],
                "abstract_inverted_index": {"Quantitative": [0], "patch": [1], "clamp": [2], "recording": [3], "mouse": [4], "HCN": [5], "current": [6]},
                "primary_location": {"landing_page_url": "https://doi.org/10.1000/shared", "source": {"display_name": "Neural Methods"}},
                "open_access": {"is_oa": True, "oa_url": "https://repository.test/shared"},
                "best_oa_location": {"pdf_url": "https://repository.test/shared.pdf", "license": "cc-by"},
            }]}
        raise AssertionError(url)


def test_query_generation_uses_questions_preparation_and_multiple_purposes():
    variants = discovery.generate_query_variants(wall())
    assert len(variants) == 4
    assert {item["purpose"] for item in variants} == {"preparation", "quantitative-methods"}
    assert {tuple(item["question_ids"]) for item in variants} == {("P1",), ("W1",)}
    assert all("mouse" in item["query"] and "whole-cell patch clamp" in item["query"] for item in variants)
    assert any("quantitative measurement" in item["query"] for item in variants)


def test_discovery_queries_all_providers_deduplicates_and_ranks():
    client = FakeClient()
    ticks = iter(f"2026-08-04T06:{30 + second // 60:02d}:{second % 60:02d}+00:00" for second in range(60))
    packet = discovery.discover(wall(), client=client, now=lambda: next(ticks), timeout=7.0)

    assert packet["status"] == "complete-review-required"
    assert len(client.urls) == 12
    assert {record["provider"] for record in packet["requests"]} == set(discovery.PROVIDERS)
    assert all(record["status"] == "complete" and record["source_url"].startswith("https://") for record in packet["requests"])
    assert len(packet["candidates"]) == 2
    top = packet["candidates"][0]
    assert top["doi"] == "10.1000/shared"
    assert {origin["provider"] for origin in top["origins"]} == set(discovery.PROVIDERS)
    assert top["evidence_strength"] == "primary-quantitative-candidate"
    assert top["rank"]["score"] > packet["candidates"][1]["rank"]["score"]
    assert packet["candidates"][1]["evidence_strength"] == "context-only"
    assert {link["provider"] for link in top["lawful_full_text_links"]} == set(discovery.PROVIDERS)
    assert all(link["downloaded"] is False and link["access_basis"] for link in top["lawful_full_text_links"])


def test_abstract_candidates_preserve_locator_and_never_claim_exact_parameters():
    packet = discovery.discover(wall(), client=FakeClient(), now=lambda: "2026-08-04T06:30:00+00:00", timeout=7.0)
    candidate = packet["candidates"][0]
    extraction = candidate["extraction_candidates"][0]
    assert extraction["locator"] == "abstract"
    assert "HCN conductance" in extraction["snippet"]
    assert extraction["metadata_provider"] in discovery.PROVIDERS
    assert extraction["metadata_request_url"].startswith("https://")
    assert extraction["article_url"].startswith("https://")
    assert extraction["review_status"] == "required"
    assert extraction["exact_parameter_claim"] is False
    assert packet["review_contract"]["automatic_parameter_claims_allowed"] is False
    assert packet["review_contract"]["source_intake_required"] is True
    assert all(item["status"] == "unresolved-pending-full-text-review" for item in packet["unresolved_questions"])


@pytest.mark.parametrize("provider", ["europepmc", "crossref", "openalex"])
def test_partial_api_failure_fails_closed(provider):
    client = FakeClient(fail_provider=provider)
    with pytest.raises(discovery.DiscoveryError, match="partial provider failure"):
        discovery.discover(wall(), client=client, now=lambda: "2026-08-04T06:30:00+00:00", timeout=7.0)


def test_partial_failure_never_creates_packet(tmp_path):
    output = tmp_path / "packet.json"
    with pytest.raises(discovery.DiscoveryError):
        packet = discovery.discover(
            wall(), client=FakeClient(fail_provider="crossref"),
            now=lambda: "2026-08-04T06:30:00+00:00", timeout=7.0,
        )
        discovery.write_packet_create_only(output, packet)
    assert not output.exists()


def test_packet_writer_is_create_only(tmp_path):
    output = tmp_path / "packet.json"
    packet = {"schema": discovery.SCHEMA, "status": "test"}
    discovery.write_packet_create_only(output, packet)
    assert json.loads(output.read_text()) == packet
    original = output.read_bytes()
    with pytest.raises(discovery.DiscoveryError, match="overwrite|cannot contain a symlink"):
        discovery.write_packet_create_only(output, {"replacement": True})
    assert output.read_bytes() == original


def test_packet_writer_rejects_symlink_destination(tmp_path):
    target = tmp_path / "target.json"
    target.write_text("unchanged")
    output = tmp_path / "packet.json"
    output.symlink_to(target)
    with pytest.raises(discovery.DiscoveryError, match="overwrite|cannot contain a symlink"):
        discovery.write_packet_create_only(output, {"replacement": True})
    assert target.read_text() == "unchanged"


def test_packet_writer_rejects_broken_symlink_destination(tmp_path):
    output = tmp_path / "packet.json"
    output.symlink_to(tmp_path / "missing.json")
    with pytest.raises(discovery.DiscoveryError, match="overwrite|cannot contain a symlink"):
        discovery.write_packet_create_only(output, {"replacement": True})
    assert output.is_symlink()


def test_packet_writer_rejects_symlinked_parent_directory(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)

    with pytest.raises(discovery.DiscoveryError, match="cannot contain a symlink"):
        discovery.write_packet_create_only(linked / "packet.json", {"value": 1})
    assert not (real / "packet.json").exists()


def test_deduplication_merges_records_bridged_by_different_identifiers():
    records = [
        {"title": "One", "year": 2024, "doi": "10.1/x", "pmid": None, "pmcid": None,
         "authors": [], "abstract": "", "lawful_full_text_links": [], "origin": {"provider": "a"}},
        {"title": "Two", "year": 2024, "doi": None, "pmid": "123", "pmcid": None,
         "authors": [], "abstract": "", "lawful_full_text_links": [], "origin": {"provider": "b"}},
        {"title": "Bridge", "year": 2024, "doi": "10.1/x", "pmid": "123", "pmcid": None,
         "authors": [], "abstract": "", "lawful_full_text_links": [], "origin": {"provider": "c"}},
    ]
    merged = discovery._merge_records(records)
    assert len(merged) == 1
    assert {origin["provider"] for origin in merged[0]["origins"]} == {"a", "b", "c"}


@pytest.mark.parametrize(
    "payload_provider",
    [
        ("ebi.ac.uk", {}),
        ("crossref", {"message": {}}),
        ("openalex", {"results": {}}),
    ],
)
def test_malformed_provider_payload_fails_closed(payload_provider):
    marker, malformed = payload_provider

    class MalformedClient(FakeClient):
        def get_json(self, url: str, *, timeout: float):
            if marker in url:
                return malformed
            return super().get_json(url, timeout=timeout)

    with pytest.raises(discovery.DiscoveryError, match="partial provider failure"):
        discovery.discover(wall(), client=MalformedClient(), now=lambda: "2026-08-04T06:30:00+00:00", timeout=7.0)


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda item: item.pop("wall_id"), "wall_id"),
        (lambda item: item.update(preparation={}), "preparation"),
        (lambda item: item.update(parameter_questions=[]), "parameter_questions"),
        (lambda item: item.update(wiring_questions=[]), "wiring_questions"),
    ],
)
def test_invalid_wall_fails_before_http(mutation, message):
    value = wall()
    mutation(value)
    client = FakeClient()
    with pytest.raises(discovery.DiscoveryError, match=message):
        discovery.discover(value, client=client, timeout=7.0)
    assert client.urls == []


def test_no_results_is_complete_but_keeps_questions_unresolved():
    class EmptyClient:
        def get_json(self, url: str, *, timeout: float):
            if "ebi.ac.uk" in url:
                return {"resultList": {"result": []}}
            if "crossref" in url:
                return {"message": {"items": []}}
            return {"results": []}

    packet = discovery.discover(wall(), client=EmptyClient(), now=lambda: "2026-08-04T06:30:00+00:00")
    assert packet["candidates"] == []
    assert len(packet["unresolved_questions"]) == 2
    assert packet["status"] == "complete-review-required"
