"""Focused tests for the one-candidate intrinsic-lesion readiness controller."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from sim.snr_executable_packet import canonical_bytes
from tests.test_v14_stageB_packet_compiler import _candidate, _template
from tools.compact_trace import save_compact_trace
from tools.v14_stageB_intrinsic_readiness import (
    ARMS,
    RECEIPT_SCHEMA,
    StageBIntrinsicReadinessError,
    _source_identity,
    run_intrinsic_readiness,
)


ROOT = Path(__file__).resolve().parents[1]
CAUSAL_GATE = Path("research/specs/v14_snr_stageB_causal_gates.json")
ANALYSIS_PROTOCOL = Path("research/specs/v14_snr_stageB_intrinsic_protocol.json")
CAUSAL_GATE_V2 = Path("research/specs/v14_snr_stageB_causal_gates_v2.json")
ANALYSIS_PROTOCOL_V2 = Path("research/specs/v14_snr_stageB_intrinsic_protocol_v2.json")
CAUSAL_GATE_V3 = Path("research/specs/v14_snr_stageB_causal_gates_v3.json")
ANALYSIS_PROTOCOL_V3 = Path("research/specs/v14_snr_stageB_intrinsic_protocol_v3.json")


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


def _copy_contract(root: Path, causal_path: Path, protocol_path: Path) -> tuple[Path, str, Path, str]:
    protocol = root / protocol_path
    protocol.parent.mkdir(parents=True, exist_ok=True)
    protocol.write_bytes((ROOT / protocol_path).read_bytes())
    protocol_sha = hashlib.sha256(protocol.read_bytes()).hexdigest()
    causal = root / causal_path
    causal.write_bytes((ROOT / causal_path).read_bytes())
    return causal, hashlib.sha256(causal.read_bytes()).hexdigest(), protocol, protocol_sha


def _v3_contract(root: Path) -> tuple[Path, str, Path, str]:
    protocol = root / ANALYSIS_PROTOCOL_V3
    protocol.parent.mkdir(parents=True, exist_ok=True)
    protocol.write_bytes((ROOT / ANALYSIS_PROTOCOL_V3).read_bytes())
    protocol_sha = hashlib.sha256(protocol.read_bytes()).hexdigest()
    causal_document = json.loads((ROOT / CAUSAL_GATE_V3).read_text())
    causal_document["authorized_analysis_protocol"] = {
        "path": ANALYSIS_PROTOCOL_V3.as_posix(),
        "sha256": protocol_sha,
    }
    causal = root / CAUSAL_GATE_V3
    causal_sha = _write(causal, causal_document)
    return causal, causal_sha, protocol, protocol_sha


def _synthetic_spike_step(bridge) -> None:
    bridge.cp_membrane_potential_v[:] = -55.0
    bridge.cp_firing_states[:] = True


def _mock_companion_result(root: Path, assay: str, output: Path, candidate_sha: str) -> dict:
    def trace(name: str) -> dict[str, str]:
        archive = output.parent / f"{name}.ct"
        digest = save_compact_trace(
            archive,
            np.asarray([0.0, 0.05], dtype=np.float64),
            np.asarray([-60.0, -61.0], dtype=np.float64),
            np.asarray([False, False], dtype=bool),
        )
        return {"path": archive.relative_to(root).as_posix(), "sha256": digest}

    if assay == "nap":
        assay_name = "nap_same_cell_phased_voltage"
        observation = {"compact_trace": trace("nap-trace")}
    else:
        assay_name = "hcn_hyperpolarized_current_family"
        observation = {
            "trials": [
                {"compact_trace": trace(f"hcn-trace-{index}")}
                for index in range(14)
            ]
        }
    result = {
        "schema": "v14-snr-stageB-companion-physiology-v1",
        "process_status": "completed",
        "assay": assay_name,
        "adaptive_candidate": {"candidate_sha256": candidate_sha},
        "observation": observation,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_bytes(result))
    return result


def _isolate_protocol_controller(monkeypatch) -> None:
    module = __import__(
        "tools.v14_stageB_intrinsic_readiness", fromlist=["run_readiness_arm"]
    )
    original = module.run_readiness_arm

    def arm_runner(parameter_text, output, **kwargs):
        result = original(
            parameter_text,
            output,
            repository_root=kwargs["repository_root"],
            analysis_protocol_path=None,
            analysis_protocol_sha256=None,
            compact_trace=False,
        )
        raw = result["raw_observation"]
        archive = Path(output).with_name("raw-observation.ct")
        digest = save_compact_trace(
            archive,
            np.asarray(raw.pop("time_s"), dtype=np.float64).reshape(-1),
            np.asarray(raw.pop("voltage_mV"), dtype=np.float64).reshape(-1),
            np.asarray(raw.pop("spike_states"), dtype=bool).reshape(-1),
        )
        protocol = Path(kwargs["analysis_protocol_path"])
        root = Path(kwargs["repository_root"])
        raw["analysis_protocol"] = {
            "binding": {
                "path": protocol.relative_to(root).as_posix(),
                "sha256": kwargs["analysis_protocol_sha256"],
            },
            "termination": {"controller_test_fixture": True},
        }
        raw["compact_trace"] = {
            "path": archive.relative_to(root).as_posix(),
            "sha256": digest,
        }
        Path(output).write_bytes(canonical_bytes(result))
        return result

    monkeypatch.setattr(module, "run_readiness_arm", arm_runner)


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


def test_v3_invokes_bound_companions_and_provenance_includes_their_traces(
    tmp_path, monkeypatch,
):
    from sim.bridge import SimulationBridge

    template, template_sha, candidate, candidate_sha, _, _ = _inputs(tmp_path)
    causal, causal_sha, protocol, protocol_sha = _v3_contract(tmp_path)
    calls = []
    scored = []

    def companion(assay, *args, **kwargs):
        calls.append((assay, args, kwargs))
        return _mock_companion_result(tmp_path, assay, Path(args[6]), candidate_sha)

    def scorer(document, *, root):
        scored.append((document, root))
        return {
            "scientific_verdict": None,
            "readiness_contract_result": "UNAVAILABLE",
            "all_intrinsic_lesion_gates_passed": None,
        }

    monkeypatch.setattr(SimulationBridge, "_run_one_simulation_step", _synthetic_spike_step)
    _isolate_protocol_controller(monkeypatch)
    monkeypatch.setattr("tools.v14_stageB_intrinsic_readiness.run_companion_assay", companion)
    monkeypatch.setattr(
        "tools.v14_stageB_intrinsic_readiness.score_intrinsic_lesion_observations", scorer,
    )
    output = tmp_path / "v3-readiness"
    receipt = run_intrinsic_readiness(
        template, template_sha, candidate, candidate_sha,
        causal, causal_sha, output, repository_root=tmp_path,
        analysis_protocol_path=protocol,
        analysis_protocol_sha256=protocol_sha,
        execution_argv=["test-v14-stageB-v3-readiness"],
    )

    assert [call[0] for call in calls] == ["nap", "hcn"]
    expected_arms = ["nap_lesion", "hcn_baseline_lesion"]
    for (_, args, kwargs), arm in zip(calls, expected_arms):
        parameter_path, parameter_sha = Path(args[0]), args[1]
        assert parameter_path == (
            output / candidate_sha / "arms" / arm / "adaptive-parameters.json"
        )
        assert hashlib.sha256(parameter_path.read_bytes()).hexdigest() == parameter_sha
        assert (Path(args[2]), args[3]) == (protocol, protocol_sha)
        assert (Path(args[4]), args[5]) == (causal, causal_sha)
        assert Path(args[6]) == output / candidate_sha / "companions" / (
            "nap-observation.json" if arm == "nap_lesion" else "hcn-observation.json"
        )
        assert kwargs == {"repository_root": tmp_path}

    scorer_input_path = output / candidate_sha / "intrinsic-lesion-observations.json"
    scorer_input = json.loads(scorer_input_path.read_text())
    assert scorer_input["schema"] == "v14-snr-stageB-intrinsic-lesion-observations-v2"
    assert set(scorer_input["companion_observations"]) == {"nap", "hcn"}
    for assay, binding in scorer_input["companion_observations"].items():
        artifact = tmp_path / binding["path"]
        assert artifact.name == f"{assay}-observation.json"
        assert hashlib.sha256(artifact.read_bytes()).hexdigest() == binding["sha256"]
    assert scored == [(scorer_input, tmp_path)]

    sidecar = json.loads(scorer_input_path.with_name(
        "intrinsic-lesion-observations.json.prov.json"
    ).read_text())
    companion_traces = [
        item for item in sidecar["compact_trace_artifacts"]
        if item["arm"].startswith("companion_")
    ]
    assert len(companion_traces) == 15
    assert [item["arm"] for item in companion_traces].count("companion_nap") == 1
    assert [item["arm"] for item in companion_traces].count("companion_hcn") == 14
    assert receipt["compact_trace_artifacts"] == sidecar["compact_trace_artifacts"]


@pytest.mark.parametrize(
    ("causal_path", "protocol_path"),
    [(CAUSAL_GATE, ANALYSIS_PROTOCOL), (CAUSAL_GATE_V2, ANALYSIS_PROTOCOL_V2)],
)
def test_v1_v2_protocols_do_not_invoke_companions(
    tmp_path, monkeypatch, causal_path, protocol_path,
):
    from sim.bridge import SimulationBridge

    template, template_sha, candidate, candidate_sha, _, _ = _inputs(tmp_path)
    causal, causal_sha, protocol, protocol_sha = _copy_contract(
        tmp_path, causal_path, protocol_path,
    )

    def unexpected_companion(*args, **kwargs):
        raise AssertionError("historical protocols must not invoke V3 companions")

    monkeypatch.setattr(SimulationBridge, "_run_one_simulation_step", _synthetic_spike_step)
    _isolate_protocol_controller(monkeypatch)
    monkeypatch.setattr(
        "tools.v14_stageB_intrinsic_readiness.run_companion_assay", unexpected_companion,
    )
    monkeypatch.setattr(
        "tools.v14_stageB_intrinsic_readiness.score_intrinsic_lesion_observations",
        lambda document, *, root: {
            "scientific_verdict": None,
            "readiness_contract_result": "UNAVAILABLE",
            "all_intrinsic_lesion_gates_passed": None,
        },
    )
    output = tmp_path / f"{protocol_path.stem}-readiness"
    run_intrinsic_readiness(
        template, template_sha, candidate, candidate_sha,
        causal, causal_sha, output, repository_root=tmp_path,
        analysis_protocol_path=protocol,
        analysis_protocol_sha256=protocol_sha,
    )

    scorer_input = json.loads(
        (output / candidate_sha / "intrinsic-lesion-observations.json").read_text()
    )
    assert scorer_input["schema"] == "v14-snr-stageB-intrinsic-lesion-observations-v1"
    assert "companion_observations" not in scorer_input


def test_v3_rejects_companion_result_that_differs_from_persisted_artifact(
    tmp_path, monkeypatch,
):
    from sim.bridge import SimulationBridge

    template, template_sha, candidate, candidate_sha, _, _ = _inputs(tmp_path)
    causal, causal_sha, protocol, protocol_sha = _v3_contract(tmp_path)

    def tampered_companion(assay, *args, **kwargs):
        output = Path(args[6])
        result = _mock_companion_result(tmp_path, assay, output, candidate_sha)
        persisted = dict(result)
        persisted["process_status"] = "tampered"
        output.write_bytes(canonical_bytes(persisted))
        return result

    monkeypatch.setattr(SimulationBridge, "_run_one_simulation_step", _synthetic_spike_step)
    _isolate_protocol_controller(monkeypatch)
    monkeypatch.setattr(
        "tools.v14_stageB_intrinsic_readiness.run_companion_assay", tampered_companion,
    )
    output = tmp_path / "tampered-v3-readiness"
    with pytest.raises(StageBIntrinsicReadinessError, match="invalid artifact"):
        run_intrinsic_readiness(
            template, template_sha, candidate, candidate_sha,
            causal, causal_sha, output, repository_root=tmp_path,
            analysis_protocol_path=protocol,
            analysis_protocol_sha256=protocol_sha,
        )
    assert not output.exists()


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


def _archive_attestation(root: Path) -> dict:
    ancestry = root / ".source_ancestry.json"
    source = root / "tools/example.py"
    source.parent.mkdir(parents=True, exist_ok=True)
    ancestry.write_bytes(b'{"schema":"test-ancestry"}')
    source.write_bytes(b"VALUE = 1\n")
    rows = []
    for path in (ancestry, source):
        rows.append(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  "
            f"{path.relative_to(root).as_posix()}\n"
        )
    manifest = root / ".source_manifest.sha256"
    manifest.write_text("".join(sorted(rows)), encoding="ascii")
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    ancestry_sha = hashlib.sha256(ancestry.read_bytes()).hexdigest()
    revision = "a" * 40
    (root / ".source_revision").write_text(
        "\n".join(
            (
                f"git_sha={revision}",
                "source_kind=git_archive",
                f"source_manifest_sha256={manifest_sha}",
                f"source_ancestry_sha256={ancestry_sha}",
                "excluded_worktree_paths=0",
                "created_utc=2026-08-05T00:00:00Z",
            )
        ) + "\n",
        encoding="ascii",
    )
    return {
        "kind": "git_archive",
        "revision": revision,
        "source_manifest_sha256": manifest_sha,
        "source_ancestry_sha256": ancestry_sha,
        "authoritative": True,
    }


def test_verified_archive_source_is_authoritative_without_git(tmp_path):
    expected = _archive_attestation(tmp_path)
    assert _source_identity(tmp_path, require_authoritative=True) == expected


def test_archive_source_tamper_and_missing_attestation_fail_closed(tmp_path):
    _archive_attestation(tmp_path)
    (tmp_path / "tools/example.py").write_bytes(b"VALUE = 2\n")
    with pytest.raises(StageBIntrinsicReadinessError, match="source digest mismatch"):
        _source_identity(tmp_path, require_authoritative=True)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(StageBIntrinsicReadinessError, match="requires a clean Git checkout"):
        _source_identity(empty, require_authoritative=True)


def test_nested_outer_git_checkout_does_not_override_archive_identity(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    archive = tmp_path / "archive"
    archive.mkdir()
    expected = _archive_attestation(archive)
    assert _source_identity(archive, require_authoritative=True) == expected


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
