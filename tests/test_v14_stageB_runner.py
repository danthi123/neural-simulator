"""Focused checks for the real, readiness-only V14 Stage B packet runner."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from research.runners.v14_stageB_physiology import (
    OUTPUT_SCHEMA,
    StageBPhysiologyRunnerError,
    run_readiness_intact,
)
from sim.snr_executable_packet import AUTHORITY_POLICY_SCHEMA_VERSION, canonical_bytes
from tests.test_snr_executable_packet import _authority_policy, _packet


def _write_authenticated_artifacts(root: Path) -> dict[str, str]:
    """Create a sealed test packet through the production authentication path."""

    # Packet leaf artifacts are deliberately relative to the packet file's
    # directory, so create the test source/adjudication siblings there too.
    packet = _packet(root / "packets", state="SEALED")
    policy = _authority_policy(packet)
    policy_document = {
        "schema_version": AUTHORITY_POLICY_SCHEMA_VERSION,
        "policy_id": policy.policy_id,
        "trusted_claims": [
            {
                "authority": claim.authority.value,
                "artifact_sha256": claim.artifact_sha256,
                "claim_sha256": claim.claim_sha256,
            }
            for claim in sorted(policy.trusted_claims)
        ],
        "trusted_adjudication_receipts": sorted(policy.trusted_adjudication_receipts),
    }
    policy_path = root / "packets/policy.json"
    packet_path = root / "packets/snr.json"
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    policy_path.write_bytes(canonical_bytes(policy_document))
    packet_path.write_bytes(canonical_bytes(packet))
    return {
        "snr_executable_packet_path": "packets/snr.json",
        "snr_executable_packet_sha256": hashlib.sha256(packet_path.read_bytes()).hexdigest(),
        "snr_authority_policy_path": "packets/policy.json",
        "snr_authority_policy_sha256": hashlib.sha256(policy_path.read_bytes()).hexdigest(),
    }


def _parameter_document(references: dict[str, str], *, arm: str = "intact_autonomous") -> str:
    candidate = {
        "schema": "sim-adaptive-candidate-v1",
        "candidate_id": "packet-backed-readiness-intact",
        "parameters": references,
    }
    digest = hashlib.sha256(canonical_bytes(candidate)).hexdigest()
    return json.dumps(
        {
            "schema": "sim-adaptive-run-parameters-v1",
            "candidate_id": candidate["candidate_id"],
            "candidate_sha256": digest,
            "candidate_parameters": references,
            "arm": arm,
            "arm_parameters": {},
            "effective_parameters": references,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def test_readiness_runner_executes_real_authenticated_packet_and_writes_uncropped_trace(tmp_path):
    references = _write_authenticated_artifacts(tmp_path)
    parameter_document = _parameter_document(references)
    output = tmp_path / "output" / "raw.json"

    result = run_readiness_intact(parameter_document, output, repository_root=tmp_path)

    persisted = json.loads(output.read_text(encoding="ascii"))
    assert persisted == result
    assert result["schema"] == OUTPUT_SCHEMA
    assert result["process_status"] == "completed"
    assert result["backend"] == "numpy"
    assert result["device"] == "cpu"
    assert result["readiness_only"] == {
        "enabled": True,
        "reserved_seed_count": 0,
        "scientific_seed": None,
        "engine_seed": 0,
        "engine_seed_effect": "none; connectivity, heterogeneity, noise, and plasticity are disabled",
    }
    assert result["adaptive_candidate"] == {
        "candidate_id": "packet-backed-readiness-intact",
        "candidate_sha256": json.loads(parameter_document)["candidate_sha256"],
        "effective_parameters": references,
    }
    raw = result["raw_observation"]
    assert raw["uncropped"] is True
    assert len(raw["time_s"]) == 20
    assert raw["time_s"][0] == pytest.approx(0.00005)
    assert raw["recording_start_s"] == raw["time_s"][0]
    assert raw["sample_semantics"] == "post-update state at the declared time"
    assert len(raw["voltage_mV"]) == len(raw["spike_states"]) == 20
    assert all(len(row) == 1 for row in raw["voltage_mV"])
    assert all(len(row) == 1 for row in raw["spike_states"])
    assert all(isinstance(row[0], float) for row in raw["voltage_mV"])
    binding = result["provenance"]["bindings"]
    assert binding[0]["packet_path"] == references["snr_executable_packet_path"]
    assert binding[0]["packet_file_sha256"] == references["snr_executable_packet_sha256"]
    assert binding[0]["authority_policy_sha256"] == references["snr_authority_policy_sha256"]
    assert len(result["provenance"]["runtime_binding_manifest_sha256"]) == 64


def test_readiness_runner_rejects_unsupported_arm_without_creating_raw_artifact(tmp_path):
    references = _write_authenticated_artifacts(tmp_path)
    output = tmp_path / "output" / "raw.json"

    with pytest.raises(StageBPhysiologyRunnerError, match="unsupported Stage B arm"):
        run_readiness_intact(
            _parameter_document(references, arm="nalcn_lesion"), output, repository_root=tmp_path
        )

    assert not output.exists()


def test_readiness_runner_fails_closed_when_sealed_packet_digest_is_wrong(tmp_path):
    references = _write_authenticated_artifacts(tmp_path)
    references["snr_executable_packet_sha256"] = "0" * 64
    output = tmp_path / "output" / "raw.json"

    with pytest.raises(Exception, match="digest mismatch"):
        run_readiness_intact(_parameter_document(references), output, repository_root=tmp_path)

    assert not output.exists()


def test_readiness_runner_rejects_parameter_documents_that_do_not_echo_candidate_exactly(tmp_path):
    references = _write_authenticated_artifacts(tmp_path)
    document = json.loads(_parameter_document(references))
    document["effective_parameters"] = {**references, "unexpected": "mutation"}
    output = tmp_path / "output" / "raw.json"

    with pytest.raises(StageBPhysiologyRunnerError, match="does not exactly merge"):
        run_readiness_intact(json.dumps(document), output, repository_root=tmp_path)

    assert not output.exists()
