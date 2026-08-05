"""Focused checks for authenticated V14 Stage B companion raw traces."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import research.runners.v14_stageB_companion_physiology as companion
from research.runners.v14_stageB_companion_physiology import (
    HCN_CURRENT_FAMILY_PA,
    PROTOCOL_SCHEMA,
    StageBCompanionRunnerError,
    run_hcn_companion,
    run_nap_companion,
)
from sim.snr_executable_packet import canonical_bytes
from tests.test_v14_stageB_packet_compiler import _candidate, _template
from tests.test_v14_stageB_runner import _parameter_document
from tools.compact_trace import load_compact_trace
from tools.v14_stageB_packet_compiler import compile_candidate
from tools.v14_stageB_packet_verifier import verify_candidate


def _write_authenticated_artifacts(root: Path) -> tuple[dict[str, float], dict[str, str]]:
    template = _template()
    template["parameter_leaves"]["geometry"]["membrane_area"]["value"] = "2000"
    candidate = _candidate()
    candidate["candidate_id"] = "packet-backed-readiness-intact"
    template_path = root / "template.json"
    candidate_path = root / "candidate-input.json"
    template_path.write_bytes(canonical_bytes(template))
    candidate_path.write_bytes(canonical_bytes(candidate))
    template_sha = hashlib.sha256(template_path.read_bytes()).hexdigest()
    candidate_sha = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
    packet_dir = root / "packets"
    compile_candidate(
        template_path, template_sha, candidate_path, candidate_sha, packet_dir,
        repository_root=root,
    )
    release = verify_candidate(template_path, template_sha, packet_dir, repository_root=root)
    return candidate["parameters"], {
        "snr_candidate_release_path": "packets/candidate-release.json",
        "snr_candidate_release_sha256": release["candidate_release_sha256"],
        "snr_executable_packet_path": "packets/packet.sealed.json",
        "snr_executable_packet_sha256": hashlib.sha256(
            (packet_dir / "packet.sealed.json").read_bytes()
        ).hexdigest(),
        "snr_authority_policy_path": "packets/authority-policy.json",
        "snr_authority_policy_sha256": hashlib.sha256(
            (packet_dir / "authority-policy.json").read_bytes()
        ).hexdigest(),
    }


def _protocol_v3() -> dict:
    return {
        "schema": PROTOCOL_SCHEMA,
        "causal_gate_authority": {"path": "causal-gate-v3.json"},
        "execution": {"dt_ms": 0.05},
        "arms": {
            "nap_lesion": {
                "mean_voltage_change": {
                    "phase_schedule": {
                        "intact_baseline_duration_s": 2.0,
                        "lesion_onset_s": 2.0,
                        "post_lesion_duration_s": 1.0,
                        "total_duration_s": 3.0,
                    },
                    "intervention": {
                        "nap_conductance_fraction_after_onset": 0.0,
                        "lesion_onset_sample_s": 2.0,
                    },
                    "same_cell_requirement": (
                        "one continuously simulated cell; do not substitute independently "
                        "initialized intact and lesion traces"
                    ),
                }
            },
            "hcn_baseline_lesion": {
                "hyperpolarized_input_resistance": {
                    "current_family_pA": list(HCN_CURRENT_FAMILY_PA),
                    "phase_schedule": {
                        "baseline_duration_s": 0.25,
                        "current_step_duration_s": 1.0,
                        "current_step_onset_s": 0.25,
                        "steady_state_window_relative_to_step_s": [0.9, 1.0],
                        "total_duration_s": 1.25,
                    },
                    "conditions": {
                        "shared_ttx_equivalent": {
                            "fast_na_conductance_fraction_of_candidate": 0.0,
                            "nap_conductance_fraction_of_candidate": 0.0,
                        },
                        "intact_hcn": {"g_hcn_fraction_of_candidate": 1.0},
                        "hcn_complete_lesion": {"g_hcn_fraction_of_candidate": 0.0},
                    },
                }
            },
        },
    }


@pytest.fixture
def authenticated_case(tmp_path):
    candidate, references = _write_authenticated_artifacts(tmp_path)
    protocol_path = tmp_path / "protocol-v3.json"
    protocol_path.write_bytes(canonical_bytes(_protocol_v3()))
    protocol_sha = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    causal_path = tmp_path / "causal-gate-v3.json"
    causal_path.write_bytes(canonical_bytes({
        "authorized_analysis_protocol": {
            "path": "protocol-v3.json", "sha256": protocol_sha,
        }
    }))
    documents = {}
    for assay, arm in (("nap", "nap_lesion"), ("hcn", "hcn_baseline_lesion")):
        path = tmp_path / f"{assay}-parameters.json"
        path.write_bytes(_parameter_document(candidate, references, arm=arm).encode("ascii"))
        documents[assay] = (path, hashlib.sha256(path.read_bytes()).hexdigest())
    return {
        "root": tmp_path,
        "references": references,
        "protocol": (protocol_path, protocol_sha),
        "causal_gate": (causal_path, hashlib.sha256(causal_path.read_bytes()).hexdigest()),
        "documents": documents,
    }


def _fast_steps(monkeypatch, observations):
    def step_trace(bridge, count):
        observations.append({
            "bridge": id(bridge),
            "count": count,
            "nap": float(bridge.cp_snr_g_nap_max[0]),
            "fast_na": float(bridge.cp_hh_g_Na_max[0]),
            "hcn": float(bridge.cp_snr_g_h_max[0]),
            "current": float(bridge.cp_external_input_current[0]),
        })
        voltage = -67.0 if bridge.cp_snr_g_nap_max[0] == 0.0 else -55.0
        return np.full(count, voltage, dtype="<f8"), np.zeros(count, dtype="|b1")

    monkeypatch.setattr(companion, "_step_trace", step_trace)


def test_nap_same_cell_orders_intervention_and_preserves_exact_bindings(
    authenticated_case, monkeypatch,
):
    case = authenticated_case
    observations = []
    _fast_steps(monkeypatch, observations)
    parameter_path, parameter_sha = case["documents"]["nap"]
    protocol_path, protocol_sha = case["protocol"]
    causal_path, causal_sha = case["causal_gate"]
    output = case["root"] / "raw" / "nap.json"

    result = run_nap_companion(
        parameter_path, parameter_sha, protocol_path, protocol_sha,
        causal_path, causal_sha, output,
        repository_root=case["root"],
    )

    assert len(observations) == 2
    assert observations[0]["bridge"] == observations[1]["bridge"]
    assert observations[0]["count"] == 39999
    assert observations[1]["count"] == 20001
    assert observations[0]["nap"] > 0.0 and observations[1]["nap"] == 0.0
    intervention = result["observation"]["runtime_intervention"]
    assert intervention["timestamp_s"] == 2.0
    assert intervention["lesion_onset_sample_index"] == 39999
    assert intervention["lesion_onset_sample_number"] == 40000
    assert intervention["last_intact_sample_s"] == pytest.approx(1.99995)
    assert intervention["first_lesion_sample_s"] == 2.0
    assert intervention["before"][0] > 0.0 and intervention["after"] == [0.0]
    assert result["contracts"] == {
        "parameter_document": {"path": "nap-parameters.json", "sha256": parameter_sha},
        "protocol_spec": {"path": "protocol-v3.json", "sha256": protocol_sha},
        "causal_gate": {"path": "causal-gate-v3.json", "sha256": causal_sha},
    }
    binding = result["provenance"]["bindings"][0]
    assert binding["packet_file_sha256"] == case["references"]["snr_executable_packet_sha256"]
    assert binding["authority_policy_sha256"] == case["references"]["snr_authority_policy_sha256"]
    assert output.read_bytes() == canonical_bytes(result)
    trace = result["observation"]["compact_trace"]
    arrays = load_compact_trace(
        case["root"] / trace["path"], expected_sha256=trace["sha256"]
    )
    assert arrays["time"].size == 60000
    assert arrays["time"][39998] == pytest.approx(1.99995)
    assert arrays["time"][39999] == pytest.approx(2.0)
    assert arrays["voltage"][39998] == -55.0
    assert arrays["voltage"][39999] == -67.0


def test_hcn_family_converts_units_and_initializes_every_trial_independently(
    authenticated_case, monkeypatch,
):
    case = authenticated_case
    observations = []
    bridges = []
    original_initialize = companion._initialize_bridge

    def tracked_initialize(*args, **kwargs):
        initialized = original_initialize(*args, **kwargs)
        bridges.append(initialized[3])
        return initialized

    monkeypatch.setattr(companion, "_initialize_bridge", tracked_initialize)
    _fast_steps(monkeypatch, observations)
    parameter_path, parameter_sha = case["documents"]["hcn"]
    protocol_path, protocol_sha = case["protocol"]
    causal_path, causal_sha = case["causal_gate"]
    output = case["root"] / "raw" / "hcn.json"

    result = run_hcn_companion(
        parameter_path, parameter_sha, protocol_path, protocol_sha,
        causal_path, causal_sha, output,
        repository_root=case["root"],
    )

    trials = result["observation"]["trials"]
    assert len(bridges) == len({id(bridge) for bridge in bridges}) == 14
    assert len(trials) == 14 and len(observations) == 28
    for baseline, current in zip(observations[::2], observations[1::2]):
        assert baseline["bridge"] == current["bridge"]
        assert baseline["count"] == 4999 and current["count"] == 20001
        assert baseline["current"] == 0.0
        assert baseline["fast_na"] == baseline["nap"] == 0.0
    assert all(row["hcn"] > 0.0 for row in observations[:14])
    assert all(row["hcn"] == 0.0 for row in observations[14:])
    minus_120 = next(
        trial for trial in trials
        if trial["condition"] == "intact_hcn" and trial["current_pA"] == -120.0
    )
    assert minus_120["membrane_area_um2"] == 2000.0
    assert minus_120["current_density_uA_per_cm2"] == -6.0
    assert minus_120["bridge_external_current_numeric"] == -6_000_000.0
    assert minus_120["current_units"] == {
        "whole_cell": "pA",
        "membrane_area": "um^2",
        "density_equivalent": "uA/cm^2",
        "bridge_external_current": "cp_external_input_current numeric; HH kernel scales by 1e-6",
    }
    assert minus_120["current_step_onset_sample_index"] == 4999
    assert minus_120["current_step_onset_sample_number"] == 5000
    assert minus_120["last_baseline_sample_s"] == pytest.approx(0.24995)
    assert minus_120["first_current_step_sample_s"] == 0.25
    trace = minus_120["compact_trace"]
    arrays = load_compact_trace(
        case["root"] / trace["path"], expected_sha256=trace["sha256"]
    )
    assert arrays["time"][4998] == pytest.approx(0.24995)
    assert arrays["time"][4999] == pytest.approx(0.25)
    assert result["scientific_verdict"] is None
    assert output.read_bytes() == canonical_bytes(result)


def test_digest_tampering_fails_closed_and_formatted_protocol_is_authenticated(
    authenticated_case, monkeypatch,
):
    case = authenticated_case
    parameter_path, parameter_sha = case["documents"]["nap"]
    protocol_path, protocol_sha = case["protocol"]
    causal_path, causal_sha = case["causal_gate"]
    output = case["root"] / "raw" / "tampered.json"
    protocol = json.loads(protocol_path.read_bytes())
    protocol["arms"]["nap_lesion"]["mean_voltage_change"]["phase_schedule"][
        "lesion_onset_s"
    ] = 1.5
    protocol_path.write_bytes(canonical_bytes(protocol))

    with pytest.raises(StageBCompanionRunnerError, match="digest does not match"):
        run_nap_companion(
            parameter_path, parameter_sha, protocol_path, protocol_sha,
            causal_path, causal_sha, output,
            repository_root=case["root"],
        )
    assert not output.exists()

    protocol_path.write_text(json.dumps(_protocol_v3(), indent=2), encoding="ascii")
    pretty_sha = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    causal_path.write_bytes(canonical_bytes({
        "authorized_analysis_protocol": {
            "path": "protocol-v3.json", "sha256": pretty_sha,
        }
    }))
    causal_sha = hashlib.sha256(causal_path.read_bytes()).hexdigest()
    _fast_steps(monkeypatch, [])
    result = run_nap_companion(
        parameter_path, parameter_sha, protocol_path, pretty_sha,
        causal_path, causal_sha, output,
        repository_root=case["root"],
    )
    assert result["contracts"]["protocol_spec"]["sha256"] == pretty_sha


def test_outputs_are_create_only(authenticated_case, monkeypatch):
    case = authenticated_case
    observations = []
    _fast_steps(monkeypatch, observations)
    parameter_path, parameter_sha = case["documents"]["nap"]
    protocol_path, protocol_sha = case["protocol"]
    causal_path, causal_sha = case["causal_gate"]
    output = case["root"] / "raw" / "create-only.json"
    run_nap_companion(
        parameter_path, parameter_sha, protocol_path, protocol_sha,
        causal_path, causal_sha, output,
        repository_root=case["root"],
    )
    original = output.read_bytes()
    with pytest.raises(StageBCompanionRunnerError, match="refusing to replace"):
        run_nap_companion(
            parameter_path, parameter_sha, protocol_path, protocol_sha,
            causal_path, causal_sha, output,
            repository_root=case["root"],
        )
    assert output.read_bytes() == original


def test_wrong_one_way_causal_authorization_fails_closed(authenticated_case):
    case = authenticated_case
    parameter_path, parameter_sha = case["documents"]["nap"]
    protocol_path, protocol_sha = case["protocol"]
    causal_path, _ = case["causal_gate"]
    causal_path.write_bytes(canonical_bytes({
        "authorized_analysis_protocol": {
            "path": "protocol-v3.json",
            "sha256": "0" * 64,
        }
    }))
    causal_sha = hashlib.sha256(causal_path.read_bytes()).hexdigest()
    output = case["root"] / "raw" / "wrong-authority.json"

    with pytest.raises(StageBCompanionRunnerError, match="does not authorize"):
        run_nap_companion(
            parameter_path, parameter_sha, protocol_path, protocol_sha,
            causal_path, causal_sha, output, repository_root=case["root"],
        )

    assert not output.exists()
