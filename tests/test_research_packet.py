from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tools.research_packet import (
    ResearchPacketError,
    accept_claim,
    create_packet,
    load_packet,
    save_packet,
    validate_packet,
)


def _packet() -> dict:
    return create_packet(
        question={
            "id": "P1",
            "kind": "parameter",
            "target": "GPi autonomous output",
            "requested_measurement": "baseline firing rate and variability",
            "text": "What baseline firing rate is measured in GPi output neurons without fast synaptic input?",
        },
        prior_work_matches=[
            {
                "id": "F1",
                "reference": "research/findings/old-tonic-output-NO-GO.md",
                "relationship": "same wall; host current was a confound",
                "status": "failed",
                "summary": "The prior attempt did not establish autonomous output.",
            }
        ],
        online_searches=[
            {
                "id": "S1",
                "databases": ["PubMed", "Crossref"],
                "query_variants": ["GPi spontaneous firing rate", "entopeduncular nucleus synaptic blockade"],
                "date_from": "1900-01-01",
                "date_to": "2026-08-04",
                "urls": ["https://pubmed.ncbi.nlm.nih.gov/?term=gpi", "https://search.crossref.org/?q=gpi"],
                "outcome": "Found preparation-matched primary measurements.",
            }
        ],
        sources=[
            {
                "id": "SRC1",
                "citation": "Example et al. (2026), GPi output",
                "url": "https://doi.org/10.0000/example",
                "kind": "peer-reviewed-primary",
                "search_id": "S1",
                "locator": "Results, Figure 2",
                "evidence": "Cell-attached recordings report the baseline rate.",
                "license_status": "metadata-only",
            }
        ],
        claims=[
            {
                "id": "C1",
                "source_ids": ["SRC1"],
                "value": "75-100",
                "units": "Hz",
                "condition": "without fast synaptic input",
                "species": "macaque",
                "preparation": "in vivo cell-attached recording",
                "uncertainty": "reported range; species and preparation dependent",
                "locator": "Results, Figure 2",
                "limitations": "Not a universal value for all GPi preparations.",
            }
        ],
    )


def test_new_claims_are_pending_and_acceptance_requires_explicit_review():
    packet = _packet()
    assert packet["review_required"] is True
    assert packet["claims"][0]["status"] == "pending_review"
    accepted = accept_claim(packet, "C1", reviewer="biology-review", reviewed_at="2026-08-04", notes="Conditions retained.")
    assert accepted["claims"][0]["status"] == "accepted"
    assert accepted["claims"][0]["review"]["reviewer"] == "biology-review"


def test_validation_fails_closed_for_dangling_and_unreviewed_acceptance():
    dangling = _packet()
    dangling["claims"][0]["source_ids"] = ["missing"]
    with pytest.raises(ResearchPacketError, match="known source"):
        validate_packet(dangling)

    forged = _packet()
    forged["claims"][0]["status"] = "accepted"
    with pytest.raises(ResearchPacketError, match="approving review"):
        validate_packet(forged)


def test_absence_search_requires_strict_search_protocol():
    packet = _packet()
    packet["online_searches"][0]["claim_absence"] = True
    packet["online_searches"][0]["databases"] = ["PubMed"]
    with pytest.raises(ResearchPacketError, match="two databases"):
        validate_packet(packet)


def test_non_primary_source_cannot_be_accepted_as_biological_evidence():
    packet = _packet()
    packet["sources"][0]["kind"] = "review"
    accepted = copy.deepcopy(packet)
    accepted["claims"][0]["status"] = "accepted"
    accepted["claims"][0]["review"] = {
        "decision": "approved", "reviewer": "reviewer", "reviewed_at": "2026-08-04"
    }
    # Validation permits discovery claims to remain pending, but acceptance must
    # never silently turn a review citation into primary evidence.
    validate_packet(packet)
    with pytest.raises(ResearchPacketError, match="primary"):
        accept_claim(packet, "C1", reviewer="reviewer", reviewed_at="2026-08-04")


def test_save_and_load_use_validated_atomic_json(tmp_path: Path):
    path = tmp_path / "packet.json"
    save_packet(path, _packet())
    loaded = load_packet(path)
    assert loaded["packet_version"] == "research-packet-v1"
    assert json.loads(path.read_text(encoding="utf-8"))["claims"][0]["status"] == "pending_review"
