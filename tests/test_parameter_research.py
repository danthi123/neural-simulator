from __future__ import annotations

import copy
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
