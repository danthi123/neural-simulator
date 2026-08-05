"""Focused tests for the one-candidate intrinsic-lesion readiness controller."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from sim.snr_executable_packet import canonical_bytes
from tests.test_v14_stageB_packet_compiler import _candidate, _template
from tools.v14_stageB_intrinsic_readiness import (
    ARMS,
    RECEIPT_SCHEMA,
    StageBIntrinsicReadinessError,
    run_intrinsic_readiness,
)


ROOT = Path(__file__).resolve().parents[1]
CAUSAL_GATE = Path("research/specs/v14_snr_stageB_causal_gates.json")
ANALYSIS_PROTOCOL = Path("research/specs/v14_snr_stageB_intrinsic_protocol.json")


def _write(path: Path, value: dict) -> str:
    raw = canonical_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _inputs(root: Path) -> tuple[Path, str, Path, str, Path, str]:
    template = root / "template.json"
    candidate = root / "candidate.json"
    causal_gate = root / CAUSAL_GATE
    target_packet = root / "research/specs/v14_snr_stageB_target_packet.json"
    source_target_packet = ROOT / "research/specs/v14_snr_stageB_target_packet.json"
    candidate_document = _candidate()
    candidate_document["candidate_id"] = "intrinsic-readiness-one"
    target_packet.parent.mkdir(parents=True, exist_ok=True)
    target_packet.write_bytes(source_target_packet.read_bytes())
    return (
        template, _write(template, _template()),
        candidate, _write(candidate, candidate_document),
        causal_gate, _write(causal_gate, json.loads((ROOT / CAUSAL_GATE).read_text())),
    )


def _assert_sidecars(root: Path) -> None:
    for artifact in root.rglob("*.json"):
        if artifact.name.endswith(".prov.json"):
            continue
        sidecar_path = artifact.with_name(f"{artifact.name}.prov.json")
        assert sidecar_path.is_file(), artifact
        sidecar = json.loads(sidecar_path.read_text())
        assert sidecar["artifact"] == artifact.relative_to(root.parent).as_posix()


def test_one_candidate_runs_all_five_arms_scores_and_writes_one_receipt(tmp_path):
    template, template_sha, candidate, candidate_sha, causal_gate, gate_sha = _inputs(tmp_path)
    output = tmp_path / "readiness"

    receipt = run_intrinsic_readiness(
        template, template_sha, candidate, candidate_sha,
        causal_gate, gate_sha, output, repository_root=tmp_path,
        execution_argv=["test-v14-stageB-intrinsic-readiness"],
    )

    assert receipt["schema"] == RECEIPT_SCHEMA
    assert receipt["process_status"] == "completed"
    assert receipt["scientific_verdict"] is None
    assert receipt["candidate_count"] == 1
    assert receipt["readiness_only"]["scorer_invoked"] is True
    assert receipt["readiness_only"]["scientific_scoring"] is False
    assert receipt["score"]["readiness_contract_result"] == "UNAVAILABLE"
    assert receipt["score"]["all_intrinsic_lesion_gates_passed"] is None
    assert set(receipt["arms"]) == set(ARMS)
    assert len(list(output.glob("readiness-receipt.json"))) == 1

    candidate_dir = output / candidate_sha
    assert (candidate_dir / "authentication" / "candidate-release.json").is_file()
    for arm in ARMS:
        raw = candidate_dir / "arms" / arm / "raw-observation.json"
        assert raw.is_file()
        assert json.loads(raw.read_text())["arm"] == arm
        assert receipt["arms"][arm]["sha256"] == hashlib.sha256(raw.read_bytes()).hexdigest()
    scorer_input = candidate_dir / "intrinsic-lesion-observations.json"
    score = candidate_dir / "intrinsic-lesion-score.json"
    scorer_bindings = json.loads(scorer_input.read_text())["runner_observations"]
    assert scorer_bindings.keys() == set(ARMS)
    assert all(set(binding) == {"path", "sha256"} for binding in scorer_bindings.values())
    assert json.loads(score.read_text())["scientific_verdict"] is None
    assert json.loads((output / "readiness-receipt.json").read_bytes()) == receipt
    _assert_sidecars(output)


def test_production_protocol_is_forwarded_to_all_arms_and_bound_in_receipt(
    tmp_path, monkeypatch,
):
    from sim.bridge import SimulationBridge

    template, template_sha, candidate, candidate_sha, causal_gate, gate_sha = _inputs(tmp_path)
    protocol = tmp_path / ANALYSIS_PROTOCOL
    protocol.parent.mkdir(parents=True, exist_ok=True)
    protocol.write_bytes((ROOT / ANALYSIS_PROTOCOL).read_bytes())
    protocol_sha = hashlib.sha256(protocol.read_bytes()).hexdigest()

    def synthetic_spike_step(bridge):
        bridge.cp_membrane_potential_v[:] = -55.0
        bridge.cp_firing_states[:] = True

    monkeypatch.setattr(SimulationBridge, "_run_one_simulation_step", synthetic_spike_step)
    output = tmp_path / "production-readiness"
    receipt = run_intrinsic_readiness(
        template, template_sha, candidate, candidate_sha,
        causal_gate, gate_sha, output, repository_root=tmp_path,
        analysis_protocol_path=protocol,
        analysis_protocol_sha256=protocol_sha,
        execution_argv=["test-v14-stageB-production-readiness"],
    )

    assert receipt["analysis_protocol"] == {
        "path": ANALYSIS_PROTOCOL.as_posix(), "sha256": protocol_sha,
    }
    for arm, arm_receipt in receipt["arms"].items():
        raw = json.loads((tmp_path / arm_receipt["path"]).read_text())
        assert raw["raw_observation"]["analysis_protocol"]["binding"] == receipt[
            "analysis_protocol"
        ]
        expected_samples = 20_000 if arm == "nap_lesion" else 101
        assert arm_receipt["trace_samples"] == expected_samples


def test_failure_cleans_partial_output_and_refuses_overwrite(tmp_path, monkeypatch):
    template, template_sha, candidate, candidate_sha, causal_gate, gate_sha = _inputs(tmp_path)
    output = tmp_path / "readiness"

    def fail(*args, **kwargs):
        raise RuntimeError("test arm failure")

    monkeypatch.setattr("tools.v14_stageB_intrinsic_readiness.run_readiness_arm", fail)
    with pytest.raises(StageBIntrinsicReadinessError, match="intrinsic readiness failed"):
        run_intrinsic_readiness(
            template, template_sha, candidate, candidate_sha,
            causal_gate, gate_sha, output, repository_root=tmp_path,
        )
    assert not output.exists()

    output.mkdir()
    with pytest.raises(StageBIntrinsicReadinessError, match="new child"):
        run_intrinsic_readiness(
            template, template_sha, candidate, candidate_sha,
            causal_gate, gate_sha, output, repository_root=tmp_path,
        )


def test_dirty_git_source_is_rejected_before_output_creation(tmp_path):
    template, template_sha, candidate, candidate_sha, causal_gate, gate_sha = _inputs(tmp_path)
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Stage B Test"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=tmp_path, check=True)
    (tmp_path / "dirty.txt").write_text("dirty\n")
    with pytest.raises(StageBIntrinsicReadinessError, match="clean committed"):
        run_intrinsic_readiness(
            template, template_sha, candidate, candidate_sha,
            causal_gate, gate_sha, tmp_path / "readiness", repository_root=tmp_path,
        )


def test_rejects_pinned_candidate_digest_tamper_without_output(tmp_path):
    template, template_sha, candidate, candidate_sha, causal_gate, gate_sha = _inputs(tmp_path)
    document = json.loads(candidate.read_text())
    document["parameters"]["g_nalcn"] += 0.001
    candidate.write_bytes(canonical_bytes(document))
    output = tmp_path / "readiness"

    with pytest.raises(StageBIntrinsicReadinessError, match="digest does not match"):
        run_intrinsic_readiness(
            template, template_sha, candidate, candidate_sha,
            causal_gate, gate_sha, output, repository_root=tmp_path,
        )
    assert not output.exists()


def test_runner_identity_mismatch_cleans_all_partial_output(tmp_path, monkeypatch):
    template, template_sha, candidate, candidate_sha, causal_gate, gate_sha = _inputs(tmp_path)
    output = tmp_path / "readiness"
    original = __import__(
        "tools.v14_stageB_intrinsic_readiness", fromlist=["run_readiness_arm"]
    ).run_readiness_arm

    def mismatching_runner(*args, **kwargs):
        result = original(*args, **kwargs)
        result["adaptive_candidate"]["candidate_sha256"] = "0" * 64
        return result

    monkeypatch.setattr(
        "tools.v14_stageB_intrinsic_readiness.run_readiness_arm", mismatching_runner
    )
    with pytest.raises(StageBIntrinsicReadinessError, match="candidate identity"):
        run_intrinsic_readiness(
            template, template_sha, candidate, candidate_sha,
            causal_gate, gate_sha, output, repository_root=tmp_path,
        )
    assert not output.exists()


def test_rejects_scorer_scientific_verdict_and_cleans_output(tmp_path, monkeypatch):
    template, template_sha, candidate, candidate_sha, causal_gate, gate_sha = _inputs(tmp_path)
    output = tmp_path / "readiness"
    original = __import__(
        "tools.v14_stageB_intrinsic_readiness",
        fromlist=["score_intrinsic_lesion_observations"],
    ).score_intrinsic_lesion_observations

    def verdict_scorer(*args, **kwargs):
        result = original(*args, **kwargs)
        result["scientific_verdict"] = "GO"
        return result

    monkeypatch.setattr(
        "tools.v14_stageB_intrinsic_readiness.score_intrinsic_lesion_observations",
        verdict_scorer,
    )
    with pytest.raises(StageBIntrinsicReadinessError, match="scientific verdict"):
        run_intrinsic_readiness(
            template, template_sha, candidate, candidate_sha,
            causal_gate, gate_sha, output, repository_root=tmp_path,
        )
    assert not output.exists()
