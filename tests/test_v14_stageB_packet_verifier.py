import hashlib
import json

import pytest

from sim.snr_executable_packet import PacketError, canonical_bytes, load_authority_policy_file, load_packet_file, materialize_packet
from tests.test_v14_stageB_packet_compiler import _candidate, _template
from tools.v14_stageB_packet_compiler import compile_candidate
from tools.v14_stageB_packet_verifier import StageBPacketVerifierError, verify_candidate


def _write(path, document):
    raw = canonical_bytes(document)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _compiled(tmp_path, candidate=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    template_path = tmp_path / "template.json"
    candidate_path = tmp_path / "candidate-input.json"
    template_sha = _write(template_path, _template())
    _write(candidate_path, _candidate() if candidate is None else candidate)
    output = tmp_path / "compiled"
    compile_candidate(template_path, template_sha, candidate_path, hashlib.sha256(candidate_path.read_bytes()).hexdigest(), output, repository_root=tmp_path)
    return template_path, template_sha, output


def test_verifier_issues_one_69_claim_policy_and_materializable_sealed_packet(tmp_path):
    template_path, template_sha, output = _compiled(tmp_path)
    result = verify_candidate(template_path, template_sha, output, repository_root=tmp_path)
    assert {path.name for path in output.iterdir()} == {
        "candidate.json", "compilation-request.json", "evidence-claims.json", "packet.structural.json",
        "authority-claims.json", "packet.artifacts-verified.json", "adjudication.json",
        "authority-policy.json", "packet.sealed.json", "candidate-release.json",
    }
    policy = json.loads((output / "authority-policy.json").read_text())
    assert len(policy["trusted_claims"]) == 69
    assert len(policy["trusted_adjudication_receipts"]) == 1
    release = json.loads((output / "candidate-release.json").read_text())
    assert "never measurements" in release["fitted_value_status"]
    assert set(release["artifacts"]) == {
        "compilation_request_sha256", "evidence_claims_sha256", "authority_claims_sha256",
        "structural_packet_sha256", "artifacts_verified_packet_sha256", "adjudication_sha256",
        "authority_policy_sha256", "sealed_packet_sha256", "materialized_sha256",
    }
    loaded_policy = load_authority_policy_file("authority-policy.json", artifact_root=output, expected_sha256=result["policy_sha256"])
    packet = load_packet_file("packet.sealed.json", artifact_root=output, expected_sha256=result["packet_sha256"], authority_policy=loaded_policy)
    materialized = materialize_packet(packet, packet.validation_receipt)
    assert len(materialized.groups) == 9
    assert sum(len(group) for group in materialized.groups.values()) == 69


@pytest.mark.parametrize("artifact", ["candidate.json", "compilation-request.json", "evidence-claims.json", "packet.structural.json"])
def test_verifier_rejects_each_tampered_compiler_layer_before_outputs(tmp_path, artifact):
    template_path, template_sha, output = _compiled(tmp_path)
    document = json.loads((output / artifact).read_text())
    if artifact == "candidate.json":
        document["candidate_id"] = "tampered"
    elif artifact == "compilation-request.json":
        document["compiler_authority"] = "some"
    elif artifact == "evidence-claims.json":
        document["candidate_sha256"] = "0" * 64
    else:
        document["packet_id"] = "tampered"
    _write(output / artifact, document)
    with pytest.raises(StageBPacketVerifierError):
        verify_candidate(template_path, template_sha, output, repository_root=tmp_path)
    assert {path.name for path in output.iterdir()} == {"candidate.json", "compilation-request.json", "evidence-claims.json", "packet.structural.json"}


def test_verifier_rejects_template_tamper_and_preexisting_outputs_before_scientific_artifacts(tmp_path):
    template_path, template_sha, output = _compiled(tmp_path)
    changed = _template()
    changed["template_id"] = "tampered-template"
    _write(template_path, changed)
    with pytest.raises(StageBPacketVerifierError, match="digest"):
        verify_candidate(template_path, template_sha, output, repository_root=tmp_path)
    assert len(list(output.iterdir())) == 4

    template_path, template_sha, output = _compiled(tmp_path / "second")
    (output / "authority-policy.json").write_bytes(b"{}")
    with pytest.raises(StageBPacketVerifierError, match="exactly the four"):
        verify_candidate(template_path, template_sha, output, repository_root=tmp_path / "second")
    assert not (output / "candidate-release.json").exists()


def test_verifier_candidate_isolation_releases_distinct_packets(tmp_path):
    first_template, first_sha, first_output = _compiled(tmp_path / "first")
    second = _candidate(g_nalcn=0.03)
    second["candidate_id"] = "candidate-b"
    second_template, second_sha, second_output = _compiled(tmp_path / "second", second)
    first = verify_candidate(first_template, first_sha, first_output, repository_root=tmp_path / "first")
    second_result = verify_candidate(second_template, second_sha, second_output, repository_root=tmp_path / "second")
    assert first["packet_sha256"] != second_result["packet_sha256"]
    assert first["policy_sha256"] != second_result["policy_sha256"]


def test_verifier_agrees_with_compiler_on_exponent_decimal(tmp_path):
    template = _template()
    template["parameter_leaves"]["fast_hh"]["sodium_conductance_density"]["value"] = "1e-7"
    template_path = tmp_path / "template.json"
    candidate_path = tmp_path / "candidate.json"
    template_sha = _write(template_path, template)
    candidate_sha = _write(candidate_path, _candidate())
    output = tmp_path / "compiled"
    compile_candidate(
        template_path, template_sha, candidate_path, candidate_sha, output,
        repository_root=tmp_path,
    )
    verify_candidate(template_path, template_sha, output, repository_root=tmp_path)
    sealed = json.loads((output / "packet.sealed.json").read_text())
    assert sealed["groups"]["fast_hh"]["sodium_conductance_density"]["value"] == "1e-7"


def test_verifier_removes_partial_authority_outputs_after_late_failure(tmp_path, monkeypatch):
    template_path, template_sha, output = _compiled(tmp_path)

    def fail_materialization(*args, **kwargs):
        raise PacketError("injected late failure")

    monkeypatch.setattr("tools.v14_stageB_packet_verifier.materialize_packet", fail_materialization)
    with pytest.raises(StageBPacketVerifierError, match="did not load and materialize"):
        verify_candidate(template_path, template_sha, output, repository_root=tmp_path)
    assert {path.name for path in output.iterdir()} == {
        "candidate.json", "compilation-request.json", "evidence-claims.json", "packet.structural.json"
    }


def test_verifier_rejects_symlinked_compiler_artifact(tmp_path):
    template_path, template_sha, output = _compiled(tmp_path)
    external = tmp_path / "external.json"
    (output / "candidate.json").replace(external)
    (output / "candidate.json").symlink_to(external)
    with pytest.raises(StageBPacketVerifierError, match="regular, non-symlink"):
        verify_candidate(template_path, template_sha, output, repository_root=tmp_path)
