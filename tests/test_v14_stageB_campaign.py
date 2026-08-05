"""Tests for exact Stage B campaign materialization."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import tools.v14_stageB_campaign as campaign
from sim.snr_executable_packet import canonical_bytes


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "research/specs/v14_snr_stageB_sobol_candidates.json"
PROTOCOL = ROOT / "research/specs/v14_snr_stageB_intrinsic_protocol.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_materializes_exact_release_and_batch_graph(tmp_path: Path, monkeypatch) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    for relative in (
        "research/specs/v14_snr_stageB_sobol_candidates.json",
        "research/specs/v14_snr_stageB_packet_template.json",
        "research/specs/v14_snr_stageB_intrinsic_protocol.json",
    ):
        target = repository / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())

    # Keep this unit test small while exercising the same exact-count checks.
    manifest_path = repository / "research/specs/v14_snr_stageB_sobol_candidates.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["candidates"] = manifest["candidates"][:4]
    manifest["design"]["exact_count"] = 4
    manifest["sha256"] = campaign._digest({
        key: value for key, value in manifest.items() if key != "sha256"
    })
    manifest_path.write_bytes(canonical_bytes(manifest))
    monkeypatch.setattr(campaign, "EXACT_SCREEN_COUNT", 4)

    protocol = repository / "research/specs/v14_snr_stageB_intrinsic_protocol.json"
    result = campaign.materialize_campaign(
        manifest_path,
        _sha(manifest_path),
        protocol,
        _sha(protocol),
        repository / "runtime/campaign",
        repository_root=repository,
        batch_size=3,
        workers=2,
    )

    assert result["candidate_count"] == 4
    assert result["arm_count"] == 5
    assert result["batch_count"] == 10
    assert result["status"] == "materialized-not-executed"
    assert result["scientific_verdict"] is None
    assert result["numpy_confirmation_required"] is True
    for declaration in result["declarations"]:
        path = repository / declaration["path"]
        assert _sha(path) == declaration["sha256"]
        document = json.loads(path.read_bytes())
        assert document["sha256"] == declaration["declaration_sha256"]
        assert 1 <= len(document["candidates"]) <= 3
        for candidate in document["candidates"]:
            for key in ("release", "packet", "policy"):
                artifact = repository / candidate[key]["path"]
                assert _sha(artifact) == candidate[key]["sha256"]


def test_rejects_digest_mismatch_and_existing_destination(tmp_path: Path) -> None:
    with pytest.raises(campaign.StageBCampaignError, match="digest does not match"):
        campaign.materialize_campaign(
            MANIFEST,
            "0" * 64,
            PROTOCOL,
            _sha(PROTOCOL),
            ROOT / "research/findings/raw/never-created-campaign",
            repository_root=ROOT,
        )

    existing = ROOT / "research/findings/raw"
    with pytest.raises(campaign.StageBCampaignError, match="must not already exist"):
        campaign.materialize_campaign(
            MANIFEST,
            _sha(MANIFEST),
            PROTOCOL,
            _sha(PROTOCOL),
            existing,
            repository_root=ROOT,
        )
