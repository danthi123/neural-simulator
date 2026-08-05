from __future__ import annotations

import copy
from hashlib import sha256
import json
from pathlib import Path

import pytest

from tools import parameter_research as pr
from tools.research_packet import validate_packet


QUESTION = {
    "id": "P1",
    "kind": "parameter",
    "target": "adult substantia nigra pars reticulata pacemaking",
    "requested_measurement": "NALCN conductance density and autonomous firing rate",
    "text": "What NALCN conductance density supports autonomous firing in adult SNr neurons?",
}
GAPS = [{
    "id": "G1",
    "parameter": "NALCN maximal conductance density",
    "value": None,
    "units": None,
    "species": "mouse",
    "preparation": "acute slice whole-cell recording",
    "missing_fields": ["value", "units"],
}]


def _local(query: str, purpose: str) -> dict:
    return {"status": "complete", "query": query, "summary": f"No resolving {purpose} match."}


def _candidate(query_id: str, **overrides) -> dict:
    item = {
        "query_id": query_id,
        "provider_record_id": "W123",
        "title": "NALCN supports autonomous firing",
        "citation": "Example et al. (2024), NALCN supports autonomous firing",
        "authors": ["A. Example"],
        "year": 2024,
        "doi": "https://doi.org/10.1234/EXAMPLE.1",
        "url": "https://doi.org/10.1234/example.1",
        "kind": "peer-reviewed-primary",
        "license_status": "metadata-only",
        "exact_locator": "Methods, Table 2, row NALCN",
        "evidence": "Table 2 reports a fitted maximal conductance density.",
        "claims": [{
            "id": "C1", "value": {"minimum": 0.01, "maximum": 0.03},
            "units": "mS/cm^2", "condition": "fast synaptic transmission blocked",
            "species": "mouse", "preparation": "acute slice whole-cell recording",
            "uncertainty": "reported fitted range", "locator": "Table 2, row NALCN",
            "limitations": "adult mouse preparation only",
        }],
    }
    item.update(overrides)
    return item


def _response(*candidates) -> dict:
    return {
        "provider": "OpenAlex fixture",
        "search_url": "https://api.openalex.org/works?search=nalcn",
        "candidates": list(candidates),
    }


def _metadata_response(query: dict, *, doi: str = "10.1234/example.1") -> dict:
    return {
        "schema_version": "scholarly-discovery-v1",
        "query": {"id": query["id"], "text": query["text"]},
        "provider_searches": [{
            "provider": "openalex", "search_url": "https://api.openalex.org/works?search=nalcn",
            "attempts": 1, "metadata_records": 1,
        }],
        "candidates": [{
            "record_kind": "metadata_candidate", "content_status": "metadata_only",
            "provider_records": [{"provider": "openalex", "provider_id": "W123"}],
            "doi": doi, "canonical_url": f"https://doi.org/{doi}",
            "full_text_url_leads": ["https://repository.test/paper.pdf"],
            "title": "NALCN supports autonomous firing", "authors": ["A. Example"],
            "year": 2024, "type": "article",
        }],
        "evidence_boundary": {
            "content_status": "metadata_only", "full_text_retrieved": False,
            "full_text_urls_are_leads_only": True,
        },
    }


def _known_lead(query_ids, **overrides) -> dict:
    lead = {
        "title": "NALCN supports autonomous firing",
        "doi": "https://doi.org/10.1234/EXAMPLE.1",
        "canonical_url": "https://doi.org/10.1234/example.1",
        "full_text_url_leads": ["https://repository.test/paper.pdf"],
        "authors": ["A. Example"],
        "year": 2024,
        "type": "article",
        "provider_records": [{"provider": "openalex", "provider_id": "W123"}],
        "query_ids": list(query_ids),
    }
    lead.update(overrides)
    return lead


def _fulltext_receipt(lead: dict, store: Path, *, request_sha: str = "a" * 64) -> dict:
    content = b"NALCN conductance was fitted in mS/cm^2.\n"
    digest = sha256(content).hexdigest()
    content_path = store / "content" / f"{digest}.txt"
    content_path.parent.mkdir(parents=True, exist_ok=True)
    content_path.write_bytes(content)
    receipt = {
        "schema_version": "scholarly-fulltext-v1",
        "request_sha256": request_sha,
        "source": {
            "doi": lead.get("doi"),
            "canonical_url": lead.get("canonical_url"),
            "title": lead.get("title"),
        },
        "retrieval": {
            "declared_url": lead["full_text_url_leads"][0],
            "final_url": lead["full_text_url_leads"][0],
            "mime_type": "text/plain",
            "document_kind": "plain",
            "byte_count": len(content),
            "content_sha256": digest,
            "content_path": f"content/{digest}.txt",
        },
        "candidate_locators": [{
            "record_kind": "candidate_parameter_locator",
            "locator": "lines 12-14",
            "page": None,
            "line_start": 12,
            "line_end": 14,
            "matched_parameter_terms": ["NALCN conductance"],
            "matched_unit_patterns": [r"mS/cm\^2"],
            "passage": "NALCN conductance was fitted in mS/cm^2.",
            "claim_status": "not_a_claim",
            "review_status": "pending_review",
        }],
        "evidence_boundary": {
            "content_retrieved": True,
            "locators_are_candidates_only": True,
            "accepted_claims": False,
            "interpreted_parameter_values": False,
            "review_status": "pending_review",
        },
    }
    receipt["sha256"] = pr.scholarly_fulltext._receipt_sha256(receipt)
    return receipt


def test_plan_derives_field_specific_queries_and_checks_rag_and_failures_first():
    calls = []
    def local(query, purpose):
        calls.append((query, purpose))
        return _local(query, purpose)

    state = pr.create_plan(question=QUESTION, gaps=GAPS, local_search=local, created_at="2026-08-04")
    texts = [query["text"] for query in state["queries"]]
    assert any("mean range variance" in text for text in texts)
    assert any("measurement units" in text for text in texts)
    assert len(calls) == len(texts) * 2
    assert {purpose for _, purpose in calls} == {"project_rag", "prior_failures"}


def test_default_local_search_uses_full_catalog_then_findings(monkeypatch, tmp_path: Path):
    commands = []

    class Completed:
        returncode = 0
        stdout = "result"

    def run(command, **_kwargs):
        commands.append(command)
        return Completed()

    monkeypatch.setattr(pr.subprocess, "run", run)
    search = pr._default_local_search(tmp_path)
    assert search("SNr conductance", "project_rag")["corpus"] == "all"
    assert search("SNr conductance", "prior_failures")["corpus"] == "finding"
    assert commands[0][-1] == "all"
    assert commands[1][-1] == "finding"


def test_unavailable_local_retrieval_blocks_external_discovery():
    state = pr.create_plan(
        question=QUESTION, gaps=GAPS,
        local_search=lambda q, p: {"status": "unavailable", "query": q, "summary": "RAG down"},
        created_at="2026-08-04",
    )
    with pytest.raises(pr.ParameterResearchError, match="successful local checks"):
        pr.add_discovery_results(state, adapter=lambda query: _response(), searched_at="2026-08-04")


def test_live_metadata_is_checkpointed_per_query_and_never_exported_as_a_claim():
    state = pr.create_plan(question=QUESTION, gaps=GAPS, local_search=_local, created_at="2026-08-04")
    checkpoints = []
    discovered = pr.add_scholarly_metadata(
        state,
        discoverer=lambda query: _metadata_response(query),
        checkpoint=lambda value: checkpoints.append(copy.deepcopy(value)),
        searched_at="2026-08-04",
    )
    assert len(checkpoints) == len(state["queries"])
    assert len(discovered["metadata_searches"]) == len(state["queries"])
    assert len(discovered["metadata_leads"]) == 1
    lead = discovered["metadata_leads"][0]
    assert lead["content_status"] == "metadata_only"
    assert set(lead["query_ids"]) == {query["id"] for query in state["queries"]}
    assert not {"exact_locator", "evidence", "claims"}.intersection(lead)
    with pytest.raises(pr.ParameterResearchError, match="no completed external searches"):
        pr.export_packet(discovered)


def test_live_metadata_resumes_after_an_interrupted_remote_query():
    state = pr.create_plan(question=QUESTION, gaps=GAPS, local_search=_local, created_at="2026-08-04")
    saved = []

    def interrupted(query):
        if query["id"] == state["queries"][1]["id"]:
            raise RuntimeError("network interrupted")
        return _metadata_response(query)

    with pytest.raises(RuntimeError, match="network interrupted"):
        pr.add_scholarly_metadata(
            state, discoverer=interrupted,
            checkpoint=lambda value: saved.append(copy.deepcopy(value)),
            searched_at="2026-08-04",
        )
    assert len(saved) == 1
    calls = []
    resumed = pr.add_scholarly_metadata(
        saved[-1], discoverer=lambda query: calls.append(query["id"]) or _metadata_response(query),
        searched_at="2026-08-04",
    )
    assert state["queries"][0]["id"] not in calls
    assert len(resumed["metadata_searches"]) == len(state["queries"])


def test_live_metadata_rejects_claims_smuggled_into_provider_output():
    state = pr.create_plan(question=QUESTION, gaps=GAPS, local_search=_local, created_at="2026-08-04")

    def bad(query):
        response = _metadata_response(query)
        response["candidates"][0]["claims"] = [{"value": 0.1}]
        return response

    with pytest.raises(pr.ParameterResearchError, match="cannot contain evidence or claims"):
        pr.add_scholarly_metadata(state, discoverer=bad, searched_at="2026-08-04")


def test_known_metadata_leads_import_as_metadata_only_without_mutating_input_state():
    state = pr.create_plan(question=QUESTION, gaps=GAPS, local_search=_local, created_at="2026-08-04")
    original = copy.deepcopy(state)
    query_ids = [state["queries"][0]["id"], state["queries"][1]["id"]]

    result = pr.import_known_metadata_leads(
        state, [_known_lead(query_ids)], imported_at="2026-08-04"
    )

    assert state == original
    assert len(result["metadata_leads"]) == 1
    lead = result["metadata_leads"][0]
    assert lead["doi"] == "10.1234/example.1"
    assert lead["canonical_url"] == "https://doi.org/10.1234/example.1"
    assert lead["query_ids"] == query_ids
    assert lead["fulltext_retrieval_ids"] == []
    assert not {"claims", "exact_locator", "evidence"}.intersection(lead)
    assert result["metadata_searches"] == []
    assert result["fulltext_retrievals"] == []


def test_known_metadata_leads_deduplicate_by_doi_and_merge_nonconflicting_metadata():
    state = pr.create_plan(question=QUESTION, gaps=GAPS, local_search=_local, created_at="2026-08-04")
    first, second = state["queries"][0]["id"], state["queries"][1]["id"]
    state = pr.import_known_metadata_leads(state, [_known_lead([first])], imported_at="2026-08-04")
    result = pr.import_known_metadata_leads(state, [_known_lead(
        [second], canonical_url="https://publisher.test/article/", authors=["B. Researcher"],
        provider_records=[{"provider": "crossref", "provider_id": "10.1234/example.1"}],
        full_text_url_leads=["https://repository.test/paper.pdf", "https://publisher.test/article.pdf"],
    )], imported_at="2026-08-04")

    assert len(result["metadata_leads"]) == 1
    lead = result["metadata_leads"][0]
    assert set(lead["query_ids"]) == {first, second}
    assert {record["provider"] for record in lead["provider_records"]} == {"openalex", "crossref"}
    assert lead["full_text_url_leads"] == [
        "https://publisher.test/article.pdf", "https://repository.test/paper.pdf"
    ]


def test_known_metadata_leads_reject_doi_disagreement_and_conflicting_duplicate():
    state = pr.create_plan(question=QUESTION, gaps=GAPS, local_search=_local, created_at="2026-08-04")
    query_id = state["queries"][0]["id"]
    state = pr.import_known_metadata_leads(state, [_known_lead([query_id])], imported_at="2026-08-04")

    with pytest.raises(pr.ParameterResearchError, match="DOI disagreement"):
        pr.import_known_metadata_leads(
            state, [_known_lead([query_id], canonical_url="https://doi.org/10.9999/other")],
            imported_at="2026-08-04",
        )
    with pytest.raises(pr.ParameterResearchError, match="conflicting title"):
        pr.import_known_metadata_leads(
            state, [_known_lead([query_id], title="A different paper")], imported_at="2026-08-04"
        )


@pytest.mark.parametrize("mutator, message", [
    (lambda lead, query_id: lead.update(query_ids=["UNKNOWN"]), "unknown query"),
    (lambda lead, query_id: lead.update(full_text_url_leads=["file:///tmp/paper.pdf"]), "http"),
    (lambda lead, query_id: lead.update(claims=[]), "widened"),
    (lambda lead, query_id: lead.pop("provider_records"), "provider_records"),
    (lambda lead, query_id: lead.pop("full_text_url_leads"), "full_text_url_leads"),
])
def test_known_metadata_leads_reject_malformed_or_widened_records(mutator, message):
    state = pr.create_plan(question=QUESTION, gaps=GAPS, local_search=_local, created_at="2026-08-04")
    query_id = state["queries"][0]["id"]
    lead = _known_lead([query_id])
    mutator(lead, query_id)
    with pytest.raises(pr.ParameterResearchError, match=message):
        pr.import_known_metadata_leads(state, [lead], imported_at="2026-08-04")


def test_import_known_leads_cli_writes_one_validated_atomic_state(tmp_path: Path, monkeypatch):
    state_path = tmp_path / "state.json"
    leads_path = tmp_path / "known-leads.json"
    state = pr.create_plan(question=QUESTION, gaps=GAPS, local_search=_local, created_at="2026-08-04")
    query_id = state["queries"][0]["id"]
    pr.save_state(state_path, state)
    leads_path.write_text(json.dumps([_known_lead([query_id])]), encoding="utf-8")

    monkeypatch.setattr("sys.argv", [
        "parameter_research.py", "import-known-leads", "--state", str(state_path),
        "--leads", str(leads_path), "--imported-at", "2026-08-04",
    ])
    assert pr._main() == 0
    saved = pr.load_state(state_path)
    assert len(saved["metadata_leads"]) == 1
    assert saved["updated_at"] == "2026-08-04"


def test_import_known_leads_cli_does_not_checkpoint_invalid_batch(tmp_path: Path, monkeypatch):
    state_path = tmp_path / "state.json"
    leads_path = tmp_path / "known-leads.json"
    state = pr.create_plan(question=QUESTION, gaps=GAPS, local_search=_local, created_at="2026-08-04")
    pr.save_state(state_path, state)
    original_bytes = state_path.read_bytes()
    leads_path.write_text(json.dumps([_known_lead(["UNKNOWN"])]), encoding="utf-8")

    monkeypatch.setattr("sys.argv", [
        "parameter_research.py", "import-known-leads", "--state", str(state_path),
        "--leads", str(leads_path), "--imported-at", "2026-08-04",
    ])
    with pytest.raises(SystemExit):
        pr._main()
    assert state_path.read_bytes() == original_bytes


def test_fulltext_receipt_and_locators_are_linked_but_remain_nonclaims(tmp_path: Path):
    state = pr.create_plan(question=QUESTION, gaps=GAPS, local_search=_local, created_at="2026-08-04")
    state = pr.add_scholarly_metadata(
        state, discoverer=lambda query: _metadata_response(query), searched_at="2026-08-04"
    )
    checkpoints = []
    result = pr.add_fulltext_retrievals(
        state,
        retrieve=lambda lead: _fulltext_receipt(lead, tmp_path),
        store=tmp_path,
        checkpoint=lambda value: checkpoints.append(copy.deepcopy(value)),
        retrieved_at="2026-08-04",
    )

    assert len(checkpoints) == 1
    assert result["metadata_leads"][0]["content_status"] == "metadata_only"
    assert result["metadata_leads"][0]["fulltext_retrieval_ids"] == ["FT1"]
    retrieval = result["fulltext_retrievals"][0]
    assert retrieval["lead_id"] == result["metadata_leads"][0]["id"]
    assert retrieval["receipt"]["candidate_locators"][0]["claim_status"] == "not_a_claim"
    assert retrieval["receipt"]["evidence_boundary"]["accepted_claims"] is False
    with pytest.raises(pr.ParameterResearchError, match="no completed external searches"):
        pr.export_packet(result)

    resumed = pr.add_fulltext_retrievals(
        result, retrieve=lambda lead: _fulltext_receipt(lead, tmp_path),
        store=tmp_path,
    )
    assert resumed == result


def test_fulltext_state_rejects_tampered_boundary_and_wrong_lead_source(tmp_path: Path):
    state = pr.create_plan(question=QUESTION, gaps=GAPS, local_search=_local, created_at="2026-08-04")
    state = pr.add_scholarly_metadata(
        state, discoverer=lambda query: _metadata_response(query), searched_at="2026-08-04"
    )
    result = pr.add_fulltext_retrievals(
        state, retrieve=lambda lead: _fulltext_receipt(lead, tmp_path), store=tmp_path,
        retrieved_at="2026-08-04",
    )

    tampered = copy.deepcopy(result)
    receipt = tampered["fulltext_retrievals"][0]["receipt"]
    receipt["evidence_boundary"]["accepted_claims"] = True
    receipt["sha256"] = pr.scholarly_fulltext._receipt_sha256(receipt)
    with pytest.raises(pr.ParameterResearchError, match="pending-review evidence boundary"):
        pr.validate_state(tampered)

    wrong_source = copy.deepcopy(result)
    receipt = wrong_source["fulltext_retrievals"][0]["receipt"]
    receipt["source"]["doi"] = "10.9999/wrong"
    receipt["sha256"] = pr.scholarly_fulltext._receipt_sha256(receipt)
    with pytest.raises(pr.ParameterResearchError, match="DOI does not match"):
        pr.validate_state(wrong_source)

    relative_content = result["fulltext_retrievals"][0]["receipt"]["retrieval"]["content_path"]
    content_path = tmp_path / relative_content
    content_path.write_bytes(b"corrupt")
    with pytest.raises(pr.ParameterResearchError, match="content is missing or corrupt"):
        pr.validate_state(result)


def test_fulltext_source_binding_is_checked_before_request_dedup(tmp_path: Path):
    state = pr.create_plan(question=QUESTION, gaps=GAPS, local_search=_local, created_at="2026-08-04")
    state = pr.add_scholarly_metadata(
        state, discoverer=lambda query: _metadata_response(query), searched_at="2026-08-04"
    )
    second = copy.deepcopy(state["metadata_leads"][0])
    second.update({
        "id": "LEAD2",
        "doi": "10.1234/example.2",
        "canonical_url": "https://doi.org/10.1234/example.2",
        "title": "Second physiology paper",
        "full_text_url_leads": ["https://example.test/second.txt"],
    })
    state["metadata_leads"].append(second)
    state = pr.validate_state(state)
    first_receipt = _fulltext_receipt(state["metadata_leads"][0], tmp_path)
    state = pr.add_fulltext_retrievals(
        state, retrieve=lambda lead: first_receipt, store=tmp_path,
        lead_ids=["LEAD1"], retrieved_at="2026-08-04",
    )

    with pytest.raises(pr.ParameterResearchError, match="DOI does not match"):
        pr.add_fulltext_retrievals(
            state, retrieve=lambda lead: first_receipt, store=tmp_path,
            lead_ids=["LEAD2"], retrieved_at="2026-08-04",
        )


def test_malformed_fulltext_retrieval_refs_raise_parameter_research_error(tmp_path: Path):
    state = pr.create_plan(question=QUESTION, gaps=GAPS, local_search=_local, created_at="2026-08-04")
    state = pr.add_scholarly_metadata(
        state, discoverer=lambda query: _metadata_response(query), searched_at="2026-08-04"
    )
    state["metadata_leads"][0]["fulltext_retrieval_ids"] = [{}]
    state_path = tmp_path / "malformed-state.json"
    state_path.write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(pr.ParameterResearchError, match="references must be strings"):
        pr.load_state(state_path)


def test_state_without_fulltext_fields_is_safely_normalized():
    state = pr.create_plan(question=QUESTION, gaps=GAPS, local_search=_local, created_at="2026-08-04")
    state = pr.add_scholarly_metadata(
        state, discoverer=lambda query: _metadata_response(query), searched_at="2026-08-04"
    )
    state.pop("fulltext_retrievals")
    for lead in state["metadata_leads"]:
        lead.pop("fulltext_retrieval_ids")

    normalized = pr.validate_state(state)

    assert normalized["fulltext_retrievals"] == []
    assert normalized["metadata_leads"][0]["fulltext_retrieval_ids"] == []


def test_retrieve_fulltext_cli_checkpoints_linked_receipt(tmp_path: Path, monkeypatch):
    state_path = tmp_path / "state.json"
    state = pr.create_plan(question=QUESTION, gaps=GAPS, local_search=_local, created_at="2026-08-04")
    state = pr.add_scholarly_metadata(
        state, discoverer=lambda query: _metadata_response(query), searched_at="2026-08-04"
    )
    pr.save_state(state_path, state)
    calls = []

    class FakeRetriever:
        def __init__(self, **options):
            calls.append(("init", options))

        def retrieve(self, lead, **options):
            calls.append(("retrieve", lead["title"], options))
            return _fulltext_receipt(lead, Path(options["store"]))

    monkeypatch.setattr(pr.scholarly_fulltext, "ScholarlyFulltextRetriever", FakeRetriever)
    monkeypatch.setattr("sys.argv", [
        "parameter_research.py", "retrieve-fulltext",
        "--state", str(state_path), "--store", str(tmp_path / "store"),
        "--lead-id", "LEAD1", "--parameter-term", "NALCN conductance",
        "--unit-pattern", r"mS/cm\^2",
    ])
    assert pr._main() == 0

    saved = pr.load_state(state_path)
    assert saved["metadata_leads"][0]["fulltext_retrieval_ids"] == ["FT1"]
    assert calls[1][2]["parameter_terms"] == ["NALCN conductance"]
    assert calls[1][2]["unit_patterns"] == [r"mS/cm\^2"]


def test_discovery_is_resumable_deduplicated_by_doi_and_preserves_exact_metadata(tmp_path: Path):
    state = pr.create_plan(question=QUESTION, gaps=GAPS, local_search=_local, created_at="2026-08-04")
    first_id, second_id = state["queries"][0]["id"], state["queries"][1]["id"]
    def adapter(query):
        if query["id"] == first_id:
            return _response(_candidate(first_id))
        if query["id"] == second_id:
            return _response(_candidate(second_id, provider_record_id="W456", url="https://publisher.test/article"))
        return _response()

    discovered = pr.add_discovery_results(state, adapter=adapter, searched_at="2026-08-04")
    assert len(discovered["candidates"]) == 1
    candidate = discovered["candidates"][0]
    assert candidate["doi"] == "10.1234/example.1"
    assert candidate["exact_locator"] == "Methods, Table 2, row NALCN"
    assert set(candidate["query_ids"]) >= {first_id, second_id}
    assert {record["provider_record_id"] for record in candidate["discovery_records"]} == {"W123", "W456"}
    path = tmp_path / "resume.json"
    pr.save_state(path, discovered)
    resumed = pr.add_discovery_results(pr.load_state(path), adapter=lambda query: pytest.fail("already searched"))
    assert resumed == discovered


def test_url_deduplication_and_pending_review_boundary():
    state = pr.create_plan(question=QUESTION, gaps=GAPS, local_search=_local, created_at="2026-08-04")
    def adapter(query):
        return _response(_candidate(query["id"], doi=None, url="https://example.org/paper/#abstract"))
    discovered = pr.add_discovery_results(state, adapter=adapter, searched_at="2026-08-04")
    assert len(discovered["candidates"]) == 1
    packet = pr.export_packet(discovered)
    validate_packet(packet)
    assert packet["claims"][0]["status"] == "pending_review"
    assert packet["claims"][0]["review"] is None
    assert packet["sources"][0]["discovery"]["provider_record_id"] == "W123"


def test_url_deduplication_adds_doi_from_a_later_provider_record():
    state = pr.create_plan(question=QUESTION, gaps=GAPS, local_search=_local, created_at="2026-08-04")
    first_id, second_id = state["queries"][0]["id"], state["queries"][1]["id"]
    def adapter(query):
        if query["id"] == first_id:
            return _response(_candidate(first_id, doi=None, url="https://example.org/paper"))
        if query["id"] == second_id:
            return _response(_candidate(second_id, doi="10.1234/example.1", url="https://example.org/paper/"))
        return _response()
    discovered = pr.add_discovery_results(state, adapter=adapter, searched_at="2026-08-04")
    assert len(discovered["candidates"]) == 1
    assert discovered["candidates"][0]["doi"] == "10.1234/example.1"


def test_discovery_rejects_preapproved_claims_and_missing_exact_locator():
    state = pr.create_plan(question=QUESTION, gaps=GAPS, local_search=_local, created_at="2026-08-04")
    query_id = state["queries"][0]["id"]
    approved = _candidate(query_id)
    approved["claims"][0]["status"] = "accepted"
    with pytest.raises(pr.ParameterResearchError, match="pending_review"):
        pr.add_discovery_results(state, adapter=lambda query: _response(approved), searched_at="2026-08-04")
    with pytest.raises(pr.ParameterResearchError, match="exact_locator"):
        pr.add_discovery_results(
            state, adapter=lambda query: _response(_candidate(query_id, exact_locator="")), searched_at="2026-08-04"
        )


def test_state_validation_detects_missing_prior_check_and_packet_requires_claims():
    state = pr.create_plan(question=QUESTION, gaps=GAPS, local_search=_local, created_at="2026-08-04")
    broken = copy.deepcopy(state)
    broken["local_checks"].pop()
    with pytest.raises(pr.ParameterResearchError, match="every query"):
        pr.validate_state(broken)
    empty = pr.add_discovery_results(state, adapter=lambda query: _response(), searched_at="2026-08-04")
    with pytest.raises(pr.ParameterResearchError, match="candidates"):
        pr.export_packet(empty)


def test_cli_fixture_workflow_produces_valid_resumable_packet(tmp_path: Path, monkeypatch):
    spec = tmp_path / "spec.json"
    state_path = tmp_path / "state.json"
    results = tmp_path / "results.json"
    packet = tmp_path / "packet.json"
    spec.write_text(json.dumps({"question": QUESTION, "gaps": GAPS}))
    monkeypatch.setattr(pr, "_default_local_search", lambda repo: _local)
    monkeypatch.setattr("sys.argv", ["parameter_research.py", "plan", "--spec", str(spec), "--output", str(state_path)])
    assert pr._main() == 0
    query_id = pr.load_state(state_path)["queries"][0]["id"]
    query_ids = [query["id"] for query in pr.load_state(state_path)["queries"]]
    fixture = []
    for item in query_ids:
        response = _response(_candidate(item)) if item == query_id else _response()
        fixture.append({"query_id": item, **response})
    results.write_text(json.dumps(fixture))
    monkeypatch.setattr("sys.argv", ["parameter_research.py", "import-results", "--state", str(state_path), "--results", str(results)])
    assert pr._main() == 0
    monkeypatch.setattr("sys.argv", ["parameter_research.py", "export-packet", "--state", str(state_path), "--output", str(packet)])
    assert pr._main() == 0
    assert validate_packet(json.loads(packet.read_text()))["claims"][0]["status"] == "pending_review"
