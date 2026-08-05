import copy
import hashlib
import json
from pathlib import Path

import pytest

from sim.snr_executable_packet import PARAMETER_SCHEMA, canonical_bytes, load_packet
from tests.test_snr_executable_packet import _valid_value
from tools.v14_stageB_packet_compiler import (
    StageBPacketCompilerError,
    compile_candidate,
    compile_documents,
)


NOT_REPORTED = {"kind": "not_reported", "lower": None, "upper": None, "unit": None}
SEARCHED = {
    ("nalcn", "conductance_density"): "g_nalcn",
    ("nap", "conductance_density"): "g_nap",
}


def _template():
    groups = {}
    for group, schema in PARAMETER_SCHEMA.items():
        groups[group] = {}
        for parameter, units in schema.items():
            unit = next(iter(units))
            key = SEARCHED.get((group, parameter))
            if key:
                groups[group][parameter] = {
                    "mode": "searched", "candidate_key": key, "unit": unit,
                    "bounds": {"low": "0.01", "high": "2"}, "transform": "log",
                    "uncertainty": NOT_REPORTED, "evidence": "derived",
                    "authority": "project_decision",
                }
            else:
                groups[group][parameter] = {
                    "mode": "fixed", "value": _valid_value(group, parameter), "unit": unit,
                    "uncertainty": NOT_REPORTED, "evidence": "model_prior",
                    "authority": "model_source",
                }
    return {"schema": "v14-snr-stageB-packet-template-v1", "template_id": "test-template", "parameter_leaves": groups}


def _candidate(**parameters):
    values = {"g_nalcn": 0.02, "g_nap": 0.2}
    values.update(parameters)
    return {"schema": "sim-adaptive-candidate-v1", "candidate_id": "candidate-a", "parameters": values}


def _write(path: Path, value):
    raw = canonical_bytes(value)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def test_compiler_materializes_exact_69_leaf_structural_packet_without_authority_outputs(tmp_path):
    template = _template()
    result = compile_documents(template, _candidate(), template_sha256="a" * 64)
    assert set(result) == {
        "candidate.json", "compilation-request.json", "evidence-claims.json",
        "packet.structural.json", "expected_authority_claims",
    }
    packet = result["packet.structural.json"]
    assert packet["state"] == "STRUCTURAL" and packet["adjudication"] is None
    assert sum(len(leaves) for leaves in packet["groups"].values()) == 69
    assert packet["groups"]["nalcn"]["conductance_density"]["value"] == "0.02"
    assert packet["groups"]["nap"]["conductance_density"]["value"] == "0.2"
    loaded = load_packet(packet, artifact_root=tmp_path)
    assert loaded.packet_id == packet["packet_id"]
    request = result["compilation-request.json"]
    assert request["compiler_authority"] == "none"
    assert request["expected_authority_claims_sha256"] == hashlib.sha256(
        canonical_bytes(result["expected_authority_claims"])
    ).hexdigest()


def test_file_compiler_writes_only_four_write_once_artifacts(tmp_path):
    template_path = tmp_path / "template.json"
    candidate_path = tmp_path / "candidate-input.json"
    template_sha = _write(template_path, _template())
    candidate_sha = _write(candidate_path, _candidate())
    output = tmp_path / "compiled"
    request = compile_candidate(
        template_path, template_sha, candidate_path, candidate_sha, output,
        repository_root=tmp_path,
    )
    assert {path.name for path in output.iterdir()} == {
        "candidate.json", "compilation-request.json", "evidence-claims.json", "packet.structural.json"
    }
    assert request == json.loads((output / "compilation-request.json").read_text())
    with pytest.raises(StageBPacketCompilerError, match="must not already exist"):
        compile_candidate(
            template_path, template_sha, candidate_path, candidate_sha, output,
            repository_root=tmp_path,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value["parameters"].pop("g_nalcn"), "missing or duplicated"),
        (lambda value: value["parameters"].update({"extra": 1}), "unfiled parameters"),
        (lambda value: value["parameters"].update({"g_nalcn": True}), "finite JSON number"),
        (lambda value: value["parameters"].update({"g_nalcn": 3.0}), "outside sealed bounds"),
    ],
)
def test_compiler_rejects_missing_extra_boolean_and_out_of_range_candidates(mutation, message):
    candidate = _candidate()
    mutation(candidate)
    with pytest.raises(StageBPacketCompilerError, match=message):
        compile_documents(_template(), candidate, template_sha256="a" * 64)


def test_compiler_rejects_incomplete_template_and_self_authorized_search_leaf():
    incomplete = _template()
    del incomplete["parameter_leaves"]["sk"]["hill_coefficient"]
    with pytest.raises(StageBPacketCompilerError, match="incomplete or widened"):
        compile_documents(incomplete, _candidate(), template_sha256="a" * 64)

    self_authorized = _template()
    self_authorized["parameter_leaves"]["nalcn"]["conductance_density"]["authority"] = "primary_source"
    with pytest.raises(StageBPacketCompilerError, match="derived/project_decision"):
        compile_documents(self_authorized, _candidate(), template_sha256="a" * 64)


def test_pinned_inputs_reject_digest_mismatch_and_noncanonical_bytes(tmp_path):
    template_path = tmp_path / "template.json"
    candidate_path = tmp_path / "candidate.json"
    template_sha = _write(template_path, _template())
    candidate_path.write_text(json.dumps(_candidate(), default=str, indent=2), encoding="ascii")
    candidate_sha = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
    with pytest.raises(StageBPacketCompilerError, match="bytes are not canonical"):
        compile_candidate(
            template_path, template_sha, candidate_path, candidate_sha, tmp_path / "out",
            repository_root=tmp_path,
        )
    with pytest.raises(StageBPacketCompilerError, match="digest does not match"):
        compile_candidate(
            template_path, "0" * 64, template_path, template_sha, tmp_path / "other",
            repository_root=tmp_path,
        )


def test_file_compiler_rejects_output_outside_repository_root(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    template_path = root / "template.json"
    candidate_path = root / "candidate.json"
    template_sha = _write(template_path, _template())
    candidate_sha = _write(candidate_path, _candidate())
    with pytest.raises(StageBPacketCompilerError, match="output must be inside"):
        compile_candidate(
            template_path, template_sha, candidate_path, candidate_sha, tmp_path / "outside",
            repository_root=root,
        )
