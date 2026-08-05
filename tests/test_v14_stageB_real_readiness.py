"""Tests for the seed-free two-candidate real readiness orchestrator."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from research.runners.v14_stageB_physiology import OUTPUT_SCHEMA
from sim.snr_executable_packet import canonical_bytes
from tests.test_v14_stageB_packet_compiler import _candidate, _template
from tools.v14_stageB_real_readiness import (
    RECEIPT_SCHEMA,
    StageBRealReadinessError,
    run_real_readiness,
)


def _write(path: Path, document: dict) -> str:
    raw = canonical_bytes(document)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _inputs(tmp_path: Path) -> tuple[Path, str, Path, str, Path, str]:
    template = tmp_path / "template.json"
    candidate_a = tmp_path / "candidate-a.json"
    candidate_b = tmp_path / "candidate-b.json"
    template_sha = _write(template, _template())
    first = _candidate(g_nalcn=0.02)
    first["candidate_id"] = "real-readiness-a"
    second = _candidate(g_nalcn=0.03)
    second["candidate_id"] = "real-readiness-b"
    first_sha = _write(candidate_a, first)
    second_sha = _write(candidate_b, second)
    return template, template_sha, candidate_a, first_sha, candidate_b, second_sha


def test_direct_cli_launcher_can_import_repository_modules():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / "tools/v14_stageB_real_readiness.py"), "--help"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "two-candidate" in result.stdout


def test_two_candidates_compile_verify_run_real_traces_and_write_one_receipt(tmp_path):
    template, template_sha, candidate_a, candidate_a_sha, candidate_b, candidate_b_sha = _inputs(tmp_path)
    output = tmp_path / "readiness"

    receipt = run_real_readiness(
        template, template_sha, candidate_a, candidate_a_sha,
        candidate_b, candidate_b_sha, output, repository_root=tmp_path,
    )

    assert receipt["schema"] == RECEIPT_SCHEMA
    assert receipt["process_status"] == "completed"
    assert receipt["scientific_verdict"] is None
    assert receipt["backend"] == "numpy"
    assert receipt["device"] == "cpu"
    assert receipt["provenance"]["runner"] == "tools/v14_stageB_real_readiness.py"
    assert receipt["provenance"]["source_revision"] == "non-git-test-fixture"
    assert receipt["readiness_only"]["scored"] is False
    assert receipt["candidate_count"] == 2
    assert len(receipt["candidates"]) == 2
    assert (output / "readiness-receipt.json").read_bytes() == canonical_bytes(receipt)
    candidate_shas = {candidate_a_sha, candidate_b_sha}
    assert {item["candidate_sha256"] for item in receipt["candidates"]} == candidate_shas
    for item in receipt["candidates"]:
        candidate_dir = output / item["candidate_sha256"]
        assert candidate_dir.is_dir()
        assert (candidate_dir / "candidate-release.json").is_file()
        assert (candidate_dir / "packet.sealed.json").is_file()
        assert (candidate_dir / "authority-policy.json").is_file()
        assert (candidate_dir / "adaptive-parameters.json").is_file()
        for artifact in candidate_dir.glob("*.json"):
            if artifact.name.endswith(".prov.json"):
                continue
            sidecar = artifact.with_name(f"{artifact.name}.prov.json")
            assert sidecar.is_file()
            provenance = json.loads(sidecar.read_text())
            assert provenance["backend"] == "numpy"
            assert provenance["candidate_sha256"] == item["candidate_sha256"]
        adaptive = json.loads((candidate_dir / "adaptive-parameters.json").read_text())
        assert set(adaptive) == {
            "schema", "candidate_id", "candidate_sha256", "candidate_parameters",
            "arm", "arm_parameters", "effective_parameters",
        }
        assert adaptive["arm"] == "intact_autonomous"
        assert all(isinstance(value, (int, float)) and not isinstance(value, bool)
                   for value in adaptive["candidate_parameters"].values())
        assert set(adaptive["arm_parameters"]) == {
            "snr_candidate_release_path", "snr_candidate_release_sha256",
            "snr_executable_packet_path", "snr_executable_packet_sha256",
            "snr_authority_policy_path", "snr_authority_policy_sha256",
        }
        observation = json.loads((candidate_dir / "raw-observation.json").read_text())
        assert observation["schema"] == OUTPUT_SCHEMA
        assert observation["process_status"] == "completed"
        assert observation["readiness_only"]["scientific_seed"] is None
        assert observation["adaptive_candidate"]["candidate_sha256"] == item["candidate_sha256"]
        assert observation["provenance"]["candidate_release"]["sha256"] == item["release"]["sha256"]


def test_rejects_duplicate_candidates_before_creating_output(tmp_path):
    template, template_sha, candidate_a, candidate_a_sha, _, _ = _inputs(tmp_path)
    output = tmp_path / "readiness"

    with pytest.raises(StageBRealReadinessError, match="distinct pinned digests"):
        run_real_readiness(
            template, template_sha, candidate_a, candidate_a_sha,
            candidate_a, candidate_a_sha, output, repository_root=tmp_path,
        )
    assert not output.exists()


def test_rejects_candidate_digest_tamper_without_creating_output(tmp_path):
    template, template_sha, candidate_a, candidate_a_sha, candidate_b, candidate_b_sha = _inputs(tmp_path)
    tampered = json.loads(candidate_a.read_text())
    tampered["parameters"]["g_nalcn"] = 0.021
    candidate_a.write_bytes(canonical_bytes(tampered))
    output = tmp_path / "readiness"

    with pytest.raises(StageBRealReadinessError, match="digest does not match"):
        run_real_readiness(
            template, template_sha, candidate_a, candidate_a_sha,
            candidate_b, candidate_b_sha, output, repository_root=tmp_path,
        )
    assert not output.exists()


@pytest.mark.parametrize("mutation", ["release", "trace"])
def test_rejects_runner_release_or_trace_identity_mismatch_and_leaves_no_receipt(tmp_path, monkeypatch, mutation):
    template, template_sha, candidate_a, candidate_a_sha, candidate_b, candidate_b_sha = _inputs(tmp_path)
    output = tmp_path / "readiness"
    original = __import__(
        "tools.v14_stageB_real_readiness", fromlist=["run_readiness_intact"]
    ).run_readiness_intact

    def mismatching_runner(*args, **kwargs):
        result = original(*args, **kwargs)
        if mutation == "release":
            result["provenance"]["candidate_release"]["sha256"] = "0" * 64
        else:
            result["raw_observation"]["voltage_mV"][0][0] += 1.0
        return result

    monkeypatch.setattr(
        "tools.v14_stageB_real_readiness.run_readiness_intact", mismatching_runner
    )
    with pytest.raises(StageBRealReadinessError, match="(identity echo|returned trace)"):
        run_real_readiness(
            template, template_sha, candidate_a, candidate_a_sha,
            candidate_b, candidate_b_sha, output, repository_root=tmp_path,
        )
    assert not (output / "readiness-receipt.json").exists()
    assert not output.exists()


def test_rejects_partial_or_broadened_candidate_declarations(tmp_path):
    template, template_sha, candidate_a, candidate_a_sha, candidate_b, candidate_b_sha = _inputs(tmp_path)
    candidate = json.loads(candidate_a.read_text())
    candidate["parameters"]["snr_executable_packet_path"] = 1.0
    candidate_a_sha = _write(candidate_a, candidate)
    output = tmp_path / "readiness"

    with pytest.raises(StageBRealReadinessError, match="reserved parameter declaration"):
        run_real_readiness(
            template, template_sha, candidate_a, candidate_a_sha,
            candidate_b, candidate_b_sha, output, repository_root=tmp_path,
        )
    assert not output.exists()

    incomplete = json.loads(candidate_a.read_text())
    del incomplete["parameters"]["snr_executable_packet_path"]
    del incomplete["parameters"]["g_nap"]
    incomplete_sha = _write(candidate_a, incomplete)
    with pytest.raises(StageBRealReadinessError, match="missing or duplicated"):
        run_real_readiness(
            template, template_sha, candidate_a, incomplete_sha,
            candidate_b, candidate_b_sha, output, repository_root=tmp_path,
        )
    assert not output.exists()


def test_rejects_candidate_path_outside_repository_root(tmp_path):
    template, template_sha, _, _, candidate_b, candidate_b_sha = _inputs(tmp_path)
    outside = tmp_path.parent / "outside-real-readiness-candidate.json"
    outside_sha = _write(outside, _candidate(g_nalcn=0.025))
    output = tmp_path / "readiness"

    try:
        with pytest.raises(StageBRealReadinessError, match="inside repository_root"):
            run_real_readiness(
                template, template_sha, outside, outside_sha,
                candidate_b, candidate_b_sha, output, repository_root=tmp_path,
            )
        assert not output.exists()
    finally:
        outside.unlink(missing_ok=True)


def test_rejects_existing_output_and_seed_bearing_candidate(tmp_path):
    template, template_sha, candidate_a, candidate_a_sha, candidate_b, candidate_b_sha = _inputs(tmp_path)
    seeded = json.loads(candidate_b.read_text())
    seeded["seed"] = 17
    candidate_b_sha = _write(candidate_b, seeded)
    output = tmp_path / "readiness"
    output.mkdir()

    with pytest.raises(StageBRealReadinessError, match="must not already exist"):
        run_real_readiness(
            template, template_sha, candidate_a, candidate_a_sha,
            candidate_b, candidate_b_sha, output, repository_root=tmp_path,
        )
    output.rmdir()
    with pytest.raises(StageBRealReadinessError, match="contains seed data"):
        run_real_readiness(
            template, template_sha, candidate_a, candidate_a_sha,
            candidate_b, candidate_b_sha, output, repository_root=tmp_path,
        )
    assert not output.exists()
