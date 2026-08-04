from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.rag import source_intake


def _claim(status: str = "accepted") -> dict:
    return {
        "id": "C1",
        "source_ids": ["SRC1"],
        "value": {"minimum": 75, "maximum": 100},
        "units": "Hz",
        "condition": "without fast synaptic input",
        "species": "macaque",
        "preparation": "in vivo cell-attached recording",
        "uncertainty": "reported range",
        "locator": "Results, Figure 2",
        "limitations": "preparation-specific",
        "status": status,
        "review": {
            "decision": "approved" if status == "accepted" else "pending",
            "reviewer": "biology-review",
            "reviewed_at": "2026-08-04",
        },
    }


def _register(tmp_path: Path, monkeypatch, **overrides):
    catalog = tmp_path / "catalog"
    monkeypatch.setenv("SIM_CATALOG", str(catalog))
    kwargs = {
        "citation": "Example et al. (2026)",
        "url": "https://doi.org/10.0000/example",
        "kind": "peer-reviewed-primary",
        "license_status": "metadata-only",
        "accessed_at": "2026-08-04T12:00:00+00:00",
        "questions": ["P1"],
        "query": "GPi spontaneous firing rate",
        "locator": "Results, Figure 2",
        "evidence": "Cell-attached recordings report the baseline rate.",
        "parameter_claims": [_claim()],
        "packet_provenance": {
            "packet_path": "/tmp/reviewed.json",
            "packet_sha256": "a" * 64,
            "packet_version": "research-packet-v1",
            "question_id": "P1",
        },
    }
    kwargs.update(overrides)
    return source_intake.register_source(tmp_path, **kwargs)


def test_structured_reviewed_claim_is_durable_and_traceable(tmp_path, monkeypatch):
    record = _register(tmp_path, monkeypatch)
    durable = json.loads((tmp_path / "catalog/source-intake.jsonl").read_text().strip())
    assert durable == record
    assert durable["parameter_claims"][0]["value"] == {"minimum": 75, "maximum": 100}
    assert durable["parameter_claims"][0]["locator"] == "Results, Figure 2"
    assert durable["packet_provenance"]["packet_sha256"] == "a" * 64
    rendered = Path(record["record_path"]).read_text(encoding="utf-8")
    assert "Catalog intake does not independently accept a scientific claim" in rendered


def test_repeating_identical_intake_is_idempotent(tmp_path, monkeypatch):
    first = _register(tmp_path, monkeypatch)
    second = _register(tmp_path, monkeypatch)
    assert second == first
    assert len((tmp_path / "catalog/source-intake.jsonl").read_text().splitlines()) == 1


@pytest.mark.parametrize("status", ["pending_review", "rejected"])
def test_unaccepted_claims_fail_before_catalog_creation(tmp_path, monkeypatch, status):
    with pytest.raises(source_intake.SourceIntakeError, match="explicitly accepted"):
        _register(tmp_path, monkeypatch, parameter_claims=[_claim(status)])
    assert not (tmp_path / "catalog").exists()


def test_malformed_locator_or_packet_digest_fails_closed(tmp_path, monkeypatch):
    claim = _claim()
    claim["locator"] = ""
    with pytest.raises(source_intake.SourceIntakeError, match="locator"):
        _register(tmp_path, monkeypatch, parameter_claims=[claim])
    with pytest.raises(source_intake.SourceIntakeError, match="SHA-256"):
        _register(
            tmp_path,
            monkeypatch,
            packet_provenance={
                "packet_path": "/tmp/reviewed.json",
                "packet_sha256": "bad",
                "packet_version": "research-packet-v1",
                "question_id": "P1",
            },
        )
    assert not (tmp_path / "catalog").exists()


def test_unknown_source_kind_and_malformed_review_date_fail_closed(tmp_path, monkeypatch):
    with pytest.raises(source_intake.SourceIntakeError, match="source kind"):
        _register(tmp_path, monkeypatch, kind="blog")
    claim = _claim()
    claim["review"]["reviewed_at"] = "yesterday"
    with pytest.raises(source_intake.SourceIntakeError, match="YYYY-MM-DD"):
        _register(tmp_path, monkeypatch, parameter_claims=[claim])
    assert not (tmp_path / "catalog").exists()
