"""Focused checks for the real, readiness-only V14 Stage B packet runner."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from research.runners.v14_stageB_physiology import (
    OUTPUT_SCHEMA,
    READINESS_ARMS,
    StageBPhysiologyRunnerError,
    run_readiness_arm,
    run_readiness_intact,
)
from sim.snr_executable_packet import canonical_bytes
from tests.test_v14_stageB_packet_compiler import _candidate, _template
from tools.v14_stageB_packet_compiler import compile_candidate
from tools.v14_stageB_packet_verifier import verify_candidate


def _write_authenticated_artifacts(root: Path) -> tuple[dict[str, float], dict[str, str]]:
    """Compile and verify one candidate through the production authentication path."""

    candidate = _candidate()
    candidate["candidate_id"] = "packet-backed-readiness-intact"
    template_path = root / "template.json"
    candidate_path = root / "candidate-input.json"
    template_path.write_bytes(canonical_bytes(_template()))
    candidate_path.write_bytes(canonical_bytes(candidate))
    template_sha = hashlib.sha256(template_path.read_bytes()).hexdigest()
    candidate_sha = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
    output = root / "packets"
    compile_candidate(
        template_path, template_sha, candidate_path, candidate_sha, output,
        repository_root=root,
    )
    release = verify_candidate(template_path, template_sha, output, repository_root=root)
    packet_path = output / "packet.sealed.json"
    policy_path = output / "authority-policy.json"
    release_path = output / "candidate-release.json"
    references = {
        "snr_candidate_release_path": "packets/candidate-release.json",
        "snr_candidate_release_sha256": release["candidate_release_sha256"],
        "snr_executable_packet_path": "packets/packet.sealed.json",
        "snr_executable_packet_sha256": hashlib.sha256(packet_path.read_bytes()).hexdigest(),
        "snr_authority_policy_path": "packets/authority-policy.json",
        "snr_authority_policy_sha256": hashlib.sha256(policy_path.read_bytes()).hexdigest(),
    }
    assert references["snr_candidate_release_sha256"] == hashlib.sha256(release_path.read_bytes()).hexdigest()
    return candidate["parameters"], references


def _parameter_document(
    candidate_parameters: dict[str, float], references: dict[str, str],
    *, arm: str = "intact_autonomous",
) -> str:
    candidate = {
        "schema": "sim-adaptive-candidate-v1",
        "candidate_id": "packet-backed-readiness-intact",
        "parameters": candidate_parameters,
    }
    digest = hashlib.sha256(canonical_bytes(candidate)).hexdigest()
    return json.dumps(
        {
            "schema": "sim-adaptive-run-parameters-v1",
            "candidate_id": candidate["candidate_id"],
            "candidate_sha256": digest,
            "candidate_parameters": candidate_parameters,
            "arm": arm,
            "arm_parameters": references,
            "effective_parameters": {**candidate_parameters, **references},
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def test_readiness_runner_executes_real_authenticated_packet_and_writes_uncropped_trace(tmp_path):
    candidate_parameters, references = _write_authenticated_artifacts(tmp_path)
    parameter_document = _parameter_document(candidate_parameters, references)
    output = tmp_path / "output" / "raw.json"

    result = run_readiness_intact(parameter_document, output, repository_root=tmp_path)

    persisted = json.loads(output.read_text(encoding="ascii"))
    assert persisted == result
    assert result["schema"] == OUTPUT_SCHEMA
    assert result["process_status"] == "completed"
    assert result["backend"] == "numpy"
    assert result["device"] == "cpu"
    assert result["runtime_intervention"] == {
        "kind": "none",
        "operation": "authenticated_packet_intact",
        "target": None,
        "runtime_conductance_field": None,
        "conductance_density_unit": "mS/cm^2",
        "before": None,
        "after": None,
    }
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
        "effective_parameters": {**candidate_parameters, **references},
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
    assert result["provenance"]["candidate_release"]["sha256"] == references["snr_candidate_release_sha256"]
    assert len(result["provenance"]["runtime_binding_manifest_sha256"]) == 64


@pytest.mark.parametrize(
    ("arm", "target", "runtime_field"),
    [
        ("nap_lesion", "nap", "cp_snr_g_nap_max"),
        ("cav2_2_lesion", "cav2.2", "cp_snr_g_ca_max"),
        ("sk_lesion", "sk", "cp_snr_g_sk_max"),
        ("hcn_baseline_lesion", "hcn", "cp_snr_g_h_max"),
    ],
)
def test_readiness_runner_executes_authenticated_complete_intrinsic_current_lesions(
    tmp_path, arm, target, runtime_field,
):
    candidate_parameters, references = _write_authenticated_artifacts(tmp_path)
    parameter_document = _parameter_document(candidate_parameters, references, arm=arm)
    output = tmp_path / "output" / f"{arm}.json"

    result = run_readiness_arm(parameter_document, output, repository_root=tmp_path)

    assert json.loads(output.read_text(encoding="ascii")) == result
    assert result["arm"] == arm
    assert result["adaptive_candidate"] == {
        "candidate_id": "packet-backed-readiness-intact",
        "candidate_sha256": json.loads(parameter_document)["candidate_sha256"],
        "effective_parameters": {**candidate_parameters, **references},
    }
    intervention = result["runtime_intervention"]
    assert intervention["kind"] == "complete_intrinsic_current_lesion"
    assert intervention["operation"] == (
        "set_conductance_density_to_zero_after_authenticated_packet_initialization"
    )
    assert intervention["target"] == target
    assert intervention["runtime_conductance_field"] == runtime_field
    assert intervention["conductance_density_unit"] == "mS/cm^2"
    assert intervention["before"][0] > 0.0
    assert intervention["after"] == [0.0]
    assert result["readiness_only"]["scientific_seed"] is None
    assert "scientific_verdict" not in result
    assert result["provenance"]["candidate_release"]["sha256"] == references[
        "snr_candidate_release_sha256"
    ]
    binding = result["provenance"]["bindings"][0]
    assert binding["packet_path"] == references["snr_executable_packet_path"]
    assert binding["packet_file_sha256"] == references["snr_executable_packet_sha256"]
    assert binding["authority_policy_sha256"] == references[
        "snr_authority_policy_sha256"
    ]


@pytest.mark.parametrize("arm", ["nalcn_lesion", "hcn_hyperpolarization_lesion"])
def test_readiness_runner_rejects_unsupported_arm_without_creating_raw_artifact(tmp_path, arm):
    candidate_parameters, references = _write_authenticated_artifacts(tmp_path)
    output = tmp_path / "output" / "raw.json"

    with pytest.raises(StageBPhysiologyRunnerError, match="unsupported Stage B arm"):
        run_readiness_arm(
            _parameter_document(candidate_parameters, references, arm=arm),
            output, repository_root=tmp_path
        )

    assert not output.exists()


def test_intact_compatibility_entry_point_rejects_lesion_arm(tmp_path):
    candidate_parameters, references = _write_authenticated_artifacts(tmp_path)

    with pytest.raises(StageBPhysiologyRunnerError, match="requires arm 'intact_autonomous'"):
        run_readiness_intact(
            _parameter_document(candidate_parameters, references, arm="nap_lesion"),
            tmp_path / "raw.json",
            repository_root=tmp_path,
        )

    assert READINESS_ARMS == {
        "intact_autonomous", "nap_lesion", "cav2_2_lesion", "sk_lesion",
        "hcn_baseline_lesion",
    }


def test_readiness_runner_fails_closed_when_sealed_packet_digest_is_wrong(tmp_path):
    candidate_parameters, references = _write_authenticated_artifacts(tmp_path)
    references["snr_executable_packet_sha256"] = "0" * 64
    output = tmp_path / "output" / "raw.json"

    with pytest.raises(StageBPhysiologyRunnerError, match="does not bind the packet and policy"):
        run_readiness_intact(
            _parameter_document(candidate_parameters, references), output,
            repository_root=tmp_path,
        )

    assert not output.exists()


def test_readiness_runner_rejects_parameter_documents_that_do_not_echo_candidate_exactly(tmp_path):
    candidate_parameters, references = _write_authenticated_artifacts(tmp_path)
    document = json.loads(_parameter_document(candidate_parameters, references))
    document["effective_parameters"] = {
        **candidate_parameters, **references, "unexpected": "mutation"
    }
    output = tmp_path / "output" / "raw.json"

    with pytest.raises(StageBPhysiologyRunnerError, match="does not exactly merge"):
        run_readiness_intact(json.dumps(document), output, repository_root=tmp_path)

    assert not output.exists()


def test_readiness_runner_rejects_release_from_another_candidate(tmp_path):
    candidate_parameters, references = _write_authenticated_artifacts(tmp_path)
    document = json.loads(_parameter_document(candidate_parameters, references))
    document["candidate_id"] = "wrong-candidate"
    candidate = {
        "schema": "sim-adaptive-candidate-v1",
        "candidate_id": document["candidate_id"],
        "parameters": candidate_parameters,
    }
    document["candidate_sha256"] = hashlib.sha256(canonical_bytes(candidate)).hexdigest()

    with pytest.raises(StageBPhysiologyRunnerError, match="does not bind the adaptive candidate"):
        run_readiness_intact(
            json.dumps(document, sort_keys=True, separators=(",", ":")),
            tmp_path / "raw.json",
            repository_root=tmp_path,
        )


def test_readiness_runner_rejects_widened_candidate_release(tmp_path):
    candidate_parameters, references = _write_authenticated_artifacts(tmp_path)
    release_path = tmp_path / references["snr_candidate_release_path"]
    release = json.loads(release_path.read_text())
    release["artifacts"]["unreviewed"] = "0" * 64
    release_path.write_bytes(canonical_bytes(release))
    references["snr_candidate_release_sha256"] = hashlib.sha256(
        release_path.read_bytes()
    ).hexdigest()

    with pytest.raises(StageBPhysiologyRunnerError, match="invalid artifact bindings"):
        run_readiness_intact(
            _parameter_document(candidate_parameters, references),
            tmp_path / "raw.json",
            repository_root=tmp_path,
        )
