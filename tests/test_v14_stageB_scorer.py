import copy
import hashlib
import json
from pathlib import Path

import pytest
import numpy as np

from tools.v14_stageB_scorer import (
    INTRINSIC_LESION_RESULT_SCHEMA,
    StageBScorerError,
    _score_intrinsic_hard_gate,
    _main,
    score_intrinsic_lesion_observations,
    score_raw_observation_file,
    score_raw_observations,
)
from tools.compact_trace import save_compact_trace


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_RELATIVE = Path("research/fixtures/v14_snr_stageB_scorer_fixtures.json")
CAUSAL_GATE_RELATIVE = Path("research/specs/v14_snr_stageB_causal_gates.json")
ANALYSIS_PROTOCOL_RELATIVE = Path("research/specs/v14_snr_stageB_intrinsic_protocol.json")
CAUSAL_GATE_V2_RELATIVE = Path("research/specs/v14_snr_stageB_causal_gates_v2.json")
ANALYSIS_PROTOCOL_V2_RELATIVE = Path("research/specs/v14_snr_stageB_intrinsic_protocol_v2.json")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _conductance(fixture_id: str, peak: float) -> dict:
    return {
        "fixture_id": fixture_id,
        "raw": {
            "kind": "conductance_trace",
            "time_s": [0.0, 0.001, 0.002, 0.003, 0.004, 0.005],
            "conductance_nS": [0.0, 0.0, peak / 2.0, peak, peak / 4.0, 0.0],
            "time_unit": "s",
            "conductance_unit": "nS",
            "sample_interval_s": 0.001,
            "recording_start_s": 0.0,
            "burn_in_start_s": 0.0,
            "burn_in_end_s": 0.002,
            "window_start_s": 0.002,
            "window_end_s": 0.005,
        },
    }


def _document() -> dict:
    fixture_path = ROOT / FIXTURE_RELATIVE
    return {
        "schema": "v14-snr-stageB-raw-observations-v1",
        "adaptive_candidate": {
            "candidate_id": "candidate-a",
            "candidate_sha256": "a" * 64,
            "effective_parameters": {"snr_g_nalcn_max": 0.01},
        },
        "fixture_packet": {"path": FIXTURE_RELATIVE.as_posix(), "sha256": _digest(fixture_path)},
        "observations": [
            {
                "fixture_id": "adult-autonomous-rate-observed-range",
                "raw": {
                    "kind": "spike_train",
                    "spike_times_s": [0.11, 0.16, 0.21, 0.26, 0.31, 0.36, 0.41, 0.46],
                    "time_unit": "s",
                    "sample_interval_s": 0.001,
                    "recording_start_s": 0.0,
                    "recording_end_s": 0.6,
                    "burn_in_start_s": 0.0,
                    "burn_in_end_s": 0.1,
                    "window_start_s": 0.1,
                    "window_end_s": 0.5,
                },
            },
            {
                "fixture_id": "nalcn-lesion-ratio-4mM-model-derived",
                "raw": {
                    "kind": "paired_spike_rate_ratio",
                    "intact": {
                        "spike_times_s": [0.11 + 0.025 * index for index in range(12)],
                        "time_unit": "s", "sample_interval_s": 0.001,
                        "recording_start_s": 0.0, "recording_end_s": 0.6,
                        "burn_in_start_s": 0.0, "burn_in_end_s": 0.1,
                        "window_start_s": 0.1, "window_end_s": 0.5,
                    },
                    "lesion": {
                        "spike_times_s": [0.11 + 0.05 * index for index in range(7)],
                        "time_unit": "s", "sample_interval_s": 0.001,
                        "recording_start_s": 0.0, "recording_end_s": 0.6,
                        "burn_in_start_s": 0.0, "burn_in_end_s": 0.1,
                        "window_start_s": 0.1, "window_end_s": 0.5,
                    },
                },
            },
            _conductance("direct-pathway-unitary-peak-observed-range", 1.0),
            _conductance("pallidonigral-unitary-peak-observed-range", 5.0),
            _conductance("pallidonigral-barrage-peak-selected-range", 2.0),
        ],
    }


_INTRINSIC_ARMS = {
    "intact_autonomous": None,
    "nap_lesion": ("nap", "cp_snr_g_nap_max"),
    "cav2_2_lesion": ("cav2.2", "cp_snr_g_ca_max"),
    "sk_lesion": ("sk", "cp_snr_g_sk_max"),
    "hcn_baseline_lesion": ("hcn", "cp_snr_g_h_max"),
}


def _copy_causal_contract(root: Path) -> None:
    for relative in (
        CAUSAL_GATE_RELATIVE,
        Path("research/specs/v14_snr_stageB_target_packet.json"),
        ANALYSIS_PROTOCOL_RELATIVE,
    ):
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / relative).read_bytes())


def _spike_states(times: list[float], spikes: set[float]) -> list[list[bool]]:
    return [[any(abs(time - spike) < 1e-9 for spike in spikes)] for time in times]


def _event_spike_steps(intervals: tuple[int, ...]) -> list[int]:
    steps = [20]
    while len(steps) < 101:
        steps.append(steps[-1] + intervals[(len(steps) - 1) % len(intervals)])
    return steps


def _runner_artifact(arm: str) -> dict:
    dt = 0.00005
    if arm == "nap_lesion":
        n_steps = 20_000
        spike_steps: list[int] = []
        termination = {
            "mode": "fixed_duration", "reason": "fixed_duration_complete",
            "steps_executed": n_steps, "spikes_observed": 0,
            "target_spike_count": None, "maximum_steps": n_steps,
            "timeout_is_physiology_failure": False,
        }
    else:
        intervals = {
            "intact_autonomous": (20,),
            "cav2_2_lesion": (10, 20),
            "sk_lesion": (10, 20),
            "hcn_baseline_lesion": (21,),
        }[arm]
        spike_steps = _event_spike_steps(intervals)
        n_steps = spike_steps[-1]
        termination = {
            "mode": "event_count_or_timeout", "reason": "target_spike_count_reached",
            "steps_executed": n_steps, "spikes_observed": 101,
            "target_spike_count": 101, "maximum_steps": 400_000,
            "timeout_is_physiology_failure": False,
        }
    times = [(index + 1) * dt for index in range(n_steps)]
    spike_indices = set(spike_steps)
    baseline_voltage = {
        "intact_autonomous": -70.0,
        "nap_lesion": -70.0,
        "cav2_2_lesion": -65.0,
        "sk_lesion": -65.0,
        "hcn_baseline_lesion": -65.0,
    }[arm]
    voltages = [baseline_voltage] * n_steps
    lesion = _INTRINSIC_ARMS[arm]
    if lesion is None:
        intervention = {
            "kind": "none",
            "operation": "authenticated_packet_intact",
            "target": None,
            "runtime_conductance_field": None,
            "conductance_density_unit": "mS/cm^2",
            "before": None,
            "after": None,
        }
    else:
        target, field = lesion
        intervention = {
            "kind": "complete_intrinsic_current_lesion",
            "operation": "set_conductance_density_to_zero_after_authenticated_packet_initialization",
            "target": target,
            "runtime_conductance_field": field,
            "conductance_density_unit": "mS/cm^2",
            "before": [0.1],
            "after": [0.0],
        }
    return {
        "schema": "v14-snr-stageB-physiology-observation-v1",
        "process_status": "completed",
        "readiness_only": {
            "enabled": True,
            "reserved_seed_count": 0,
            "scientific_seed": None,
            "engine_seed": 0,
            "engine_seed_effect": "none; connectivity, heterogeneity, noise, and plasticity are disabled",
        },
        "backend": "numpy",
        "device": "cpu",
        "arm": arm,
        "runtime_intervention": intervention,
        "adaptive_candidate": {
            "candidate_id": "candidate-a",
            "candidate_sha256": "a" * 64,
            "effective_parameters": {"snr_g_nap_max": 0.01},
        },
        "raw_observation": {
            "kind": "packet_voltage_spike_trace",
            "time_unit": "s",
            "voltage_unit": "mV",
            "sample_interval_s": dt,
            "recording_start_s": dt,
            "recording_end_s": (n_steps + 1) * dt,
            "uncropped": True,
            "time_s": times,
            "sample_semantics": "post-update state at the declared time",
            "voltage_mV": [[value] for value in voltages],
            "spike_states": [[index + 1 in spike_indices] for index in range(n_steps)],
            "analysis_protocol": {
                "binding": {
                    "path": ANALYSIS_PROTOCOL_RELATIVE.as_posix(),
                    "sha256": _digest(ROOT / ANALYSIS_PROTOCOL_RELATIVE),
                },
                "termination": termination,
            },
        },
        "provenance": {
            "runner": "research/runners/v14_stageB_physiology.py",
            "candidate_release": {
                "path": "candidate-release.json",
                "sha256": "b" * 64,
                "candidate_sha256": "a" * 64,
            },
            "bindings": [{
                "packet_sha256": "c" * 64,
                "authority_policy_sha256": "d" * 64,
            }],
        },
    }


def _write_runner_artifact(root: Path, document: dict, arm: str) -> dict[str, str]:
    path = root / "artifacts" / f"{arm}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, sort_keys=True, separators=(",", ":")), encoding="ascii")
    return {"path": path.relative_to(root).as_posix(), "sha256": _digest(path)}


def _intrinsic_document(root: Path) -> dict:
    _copy_causal_contract(root)
    declarations = {
        arm: _write_runner_artifact(root, _runner_artifact(arm), arm)
        for arm in _INTRINSIC_ARMS
    }
    gate_path = root / CAUSAL_GATE_RELATIVE
    return {
        "schema": "v14-snr-stageB-intrinsic-lesion-observations-v1",
        "readiness_only": {
            "enabled": True,
            "reserved_seed_count": 0,
            "scientific_seed": None,
        },
        "causal_gate_packet": {
            "path": CAUSAL_GATE_RELATIVE.as_posix(),
            "sha256": _digest(gate_path),
        },
        "runner_observations": declarations,
    }


def _intrinsic_v2_document(root: Path) -> dict:
    _copy_causal_contract(root)
    for relative in (CAUSAL_GATE_V2_RELATIVE, ANALYSIS_PROTOCOL_V2_RELATIVE):
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / relative).read_bytes())
    declarations = {}
    for arm in _INTRINSIC_ARMS:
        artifact = _runner_artifact(arm)
        artifact["raw_observation"]["analysis_protocol"]["binding"] = {
            "path": ANALYSIS_PROTOCOL_V2_RELATIVE.as_posix(),
            "sha256": _digest(ROOT / ANALYSIS_PROTOCOL_V2_RELATIVE),
        }
        declarations[arm] = _write_runner_artifact(root, artifact, arm)
    gate_path = root / CAUSAL_GATE_V2_RELATIVE
    return {
        "schema": "v14-snr-stageB-intrinsic-lesion-observations-v1",
        "readiness_only": {
            "enabled": True,
            "reserved_seed_count": 0,
            "scientific_seed": None,
        },
        "causal_gate_packet": {
            "path": CAUSAL_GATE_V2_RELATIVE.as_posix(),
            "sha256": _digest(gate_path),
        },
        "runner_observations": declarations,
    }
def _rewrite_arm(root: Path, document: dict, arm: str, mutate) -> None:
    declaration = document["runner_observations"][arm]
    path = root / declaration["path"]
    artifact = json.loads(path.read_text(encoding="ascii"))
    mutate(artifact)
    document["runner_observations"][arm] = _write_runner_artifact(root, artifact, arm)


def _replace_with_compact_trace(root: Path, document: dict, arm: str) -> Path:
    declaration = document["runner_observations"][arm]
    path = root / declaration["path"]
    artifact = json.loads(path.read_text(encoding="ascii"))
    raw = artifact["raw_observation"]
    archive = path.with_name(f"{path.stem}.trace.zip")
    digest = save_compact_trace(
        archive,
        np.asarray(raw.pop("time_s"), dtype=np.dtype("<f8")),
        np.asarray([row[0] for row in raw.pop("voltage_mV")], dtype=np.dtype("<f8")),
        np.asarray([row[0] for row in raw.pop("spike_states")], dtype=np.dtype("|b1")),
    )
    raw["compact_trace"] = {
        "path": archive.relative_to(root).as_posix(),
        "sha256": digest,
    }
    document["runner_observations"][arm] = _write_runner_artifact(root, artifact, arm)
    return archive


def _set_event_intervals(artifact: dict, intervals: tuple[int, ...]) -> None:
    raw = artifact["raw_observation"]
    dt = raw["sample_interval_s"]
    spike_steps = _event_spike_steps(intervals)
    n_steps = spike_steps[-1]
    spike_indices = set(spike_steps)
    raw["time_s"] = [(index + 1) * dt for index in range(n_steps)]
    raw["recording_end_s"] = (n_steps + 1) * dt
    raw["voltage_mV"] = [[-65.0] for _ in range(n_steps)]
    raw["spike_states"] = [[index + 1 in spike_indices] for index in range(n_steps)]
    raw["analysis_protocol"]["termination"].update({
        "steps_executed": n_steps,
        "spikes_observed": 101,
        "reason": "target_spike_count_reached",
    })


def _intrinsic_results(result: dict) -> dict[str, dict]:
    return {item["gate_id"]: item for item in result["results"]}


def test_intrinsic_scorer_recomputes_raw_traces_and_reports_missing_protocol_arms(tmp_path):
    document = _intrinsic_document(tmp_path)

    result = score_intrinsic_lesion_observations(document, root=tmp_path)

    assert result["schema"] == INTRINSIC_LESION_RESULT_SCHEMA
    assert result["scientific_verdict"] is None
    assert result["source_equivalence_claimed"] is False
    assert result["readiness_contract_result"] == "UNAVAILABLE"
    assert result["all_intrinsic_lesion_gates_passed"] is None
    by_gate = _intrinsic_results(result)
    nap = {item["metric"]: item for item in by_gate["nap-complete-lesion"]["hard_gates"]}
    assert nap["spike_count"]["observed"] == 0.0
    assert nap["spike_count"]["window_s"] == 1.0
    assert nap["mean_membrane_voltage_change_mV"]["status"] == "unavailable"
    cav = {item["metric"]: item for item in by_gate["cav2.2-complete-lesion"]["hard_gates"]}
    assert "firing_rate_hz" not in cav
    assert cav["isi_cv"]["status"] == "scored" and cav["isi_cv"]["passed"]
    assert cav["medium_ahp_depth_mV"]["status"] == "unavailable"
    sk = {item["metric"]: item for item in by_gate["sk-complete-lesion"]["hard_gates"]}
    assert sk["isi_cv"]["status"] == "scored" and sk["isi_cv"]["passed"]
    assert sk["medium_ahp_depth_mV"]["status"] == "unavailable"
    assert sk["depolarization_block_count"]["status"] == "unavailable"
    assert "12-cell" in sk["depolarization_block_count"]["reason"]
    hcn = {item["metric"]: item for item in by_gate["hcn-complete-lesion"]["hard_gates"]}
    assert hcn["hyperpolarized_input_resistance_MOhm"]["status"] == "unavailable"
    assert "current-step" in hcn["hyperpolarized_input_resistance_MOhm"]["reason"]
    assert all(
        hard_gate["source_equivalence_claimed"] is False
        for gate in result["results"] for hard_gate in gate["hard_gates"]
    )


def test_intrinsic_v2_scorer_resolves_total_ahp_direction_without_source_equivalence(tmp_path):
    result = score_intrinsic_lesion_observations(
        _intrinsic_v2_document(tmp_path), root=tmp_path
    )
    by_gate = _intrinsic_results(result)
    for gate_id in ("cav2.2-complete-lesion", "sk-complete-lesion"):
        gates = {item["metric"]: item for item in by_gate[gate_id]["hard_gates"]}
        nadir = gates["median_interspike_voltage_nadir_mV"]
        assert nadir["status"] == "scored"
        assert nadir["passed"] is True
        assert nadir["source_equivalence_claimed"] is False
        assert nadir["observed"] == {"intact": -70.0, "lesion": -65.0}
    assert result["readiness_contract_result"] == "UNAVAILABLE"


def test_intrinsic_scorer_compact_trace_metrics_match_inline_traces(tmp_path):
    document = _intrinsic_document(tmp_path)
    inline = score_intrinsic_lesion_observations(copy.deepcopy(document), root=tmp_path)
    for arm in _INTRINSIC_ARMS:
        _replace_with_compact_trace(tmp_path, document, arm)

    compact = score_intrinsic_lesion_observations(document, root=tmp_path)

    assert compact["readiness_contract_result"] == inline["readiness_contract_result"]
    assert compact["all_intrinsic_lesion_gates_passed"] == inline["all_intrinsic_lesion_gates_passed"]
    assert compact["results"] == inline["results"]


def test_intrinsic_scorer_rejects_tampered_or_path_escaping_compact_trace(tmp_path):
    document = _intrinsic_document(tmp_path)
    archive = _replace_with_compact_trace(tmp_path, document, "intact_autonomous")
    archive.write_bytes(archive.read_bytes() + b"tamper")
    with pytest.raises(StageBScorerError, match="compact trace is invalid"):
        score_intrinsic_lesion_observations(document, root=tmp_path)

    archive.unlink()
    document = _intrinsic_document(tmp_path)
    _replace_with_compact_trace(tmp_path, document, "intact_autonomous")
    _rewrite_arm(
        tmp_path, document, "intact_autonomous",
        lambda artifact: artifact["raw_observation"]["compact_trace"].update(
            {"path": "../outside.trace.zip"}
        ),
    )
    with pytest.raises(StageBScorerError, match="repository-relative"):
        score_intrinsic_lesion_observations(document, root=tmp_path)


def test_intrinsic_scorer_rejects_caller_supplied_aggregate_measurements(tmp_path):
    document = _intrinsic_document(tmp_path)
    document["measurements"] = {"nap-complete-lesion": {"spike_count": 0}}
    with pytest.raises(StageBScorerError, match="invalid shape"):
        score_intrinsic_lesion_observations(document, root=tmp_path)

    document = _intrinsic_document(tmp_path)
    _rewrite_arm(
        tmp_path,
        document,
        "nap_lesion",
        lambda artifact: artifact["raw_observation"].update(
            {"claimed_metrics": {"spike_count": 0, "mean_membrane_voltage_change_mV": -100.0}}
        ),
    )
    with pytest.raises(StageBScorerError, match="raw trace has an invalid shape"):
        score_intrinsic_lesion_observations(document, root=tmp_path)


def test_intrinsic_scorer_fails_closed_without_sealed_analysis_protocol(tmp_path):
    document = _intrinsic_document(tmp_path)
    for arm in _INTRINSIC_ARMS:
        _rewrite_arm(
            tmp_path, document, arm,
            lambda artifact: artifact["raw_observation"].pop("analysis_protocol"),
        )

    result = score_intrinsic_lesion_observations(document, root=tmp_path)

    assert result["readiness_contract_result"] == "UNAVAILABLE"
    assert result["all_intrinsic_lesion_gates_passed"] is None
    assert all(
        hard_gate["passed"] is None
        for gate in result["results"] for hard_gate in gate["hard_gates"]
    )


def test_event_count_timeout_is_unavailable_never_a_physiology_failure():
    trace = {
        "status": "recomputed",
        "analysis_protocol": {"termination": {"reason": "maximum_duration_reached"}},
    }
    result = _score_intrinsic_hard_gate(
        "hcn-complete-lesion",
        {
            "metric": "lesion_spike_count", "operator": "greater_than",
            "evidence_class": "source_reported_direction", "value": 0,
        },
        trace,
        trace,
    )
    assert result["status"] == "unavailable"
    assert result["passed"] is None
    assert "operational timeout" in result["reason"]

    for gate_id, metric, reason in (
        ("sk-complete-lesion", "depolarization_block_count", "12-cell"),
        ("hcn-complete-lesion", "hyperpolarized_input_resistance_MOhm", "current-step"),
        ("cav2.2-complete-lesion", "medium_ahp_depth_mV", "medium-AHP"),
    ):
        unavailable = _score_intrinsic_hard_gate(
            gate_id,
            {
                "metric": metric, "operator": "lesion_greater_than_intact",
                "evidence_class": "source_reported_direction",
            },
            trace,
            trace,
        )
        assert unavailable["status"] == "unavailable"
        assert reason in unavailable["reason"]


def test_intrinsic_scorer_accepts_production_runner_artifacts_as_unavailable(tmp_path):
    from research.runners.v14_stageB_physiology import run_readiness_arm
    from tests.test_v14_stageB_runner import _parameter_document, _write_authenticated_artifacts

    _copy_causal_contract(tmp_path)
    candidate_parameters, references = _write_authenticated_artifacts(tmp_path)
    declarations = {}
    for arm in _INTRINSIC_ARMS:
        path = tmp_path / "artifacts" / f"{arm}.json"
        run_readiness_arm(
            _parameter_document(candidate_parameters, references, arm=arm),
            path,
            repository_root=tmp_path,
        )
        declarations[arm] = {
            "path": path.relative_to(tmp_path).as_posix(),
            "sha256": _digest(path),
        }
    document = {
        "schema": "v14-snr-stageB-intrinsic-lesion-observations-v1",
        "readiness_only": {
            "enabled": True,
            "reserved_seed_count": 0,
            "scientific_seed": None,
        },
        "causal_gate_packet": {
            "path": CAUSAL_GATE_RELATIVE.as_posix(),
            "sha256": _digest(tmp_path / CAUSAL_GATE_RELATIVE),
        },
        "runner_observations": declarations,
    }

    result = score_intrinsic_lesion_observations(document, root=tmp_path)

    assert result["readiness_contract_result"] == "UNAVAILABLE"
    assert result["scientific_verdict"] is None
    assert result["adaptive_candidate"]["candidate_id"] == "packet-backed-readiness-intact"


@pytest.mark.parametrize(
    ("arm", "metric", "mutation"),
    [
        (
            "cav2_2_lesion",
            "isi_cv",
            lambda artifact: _set_event_intervals(artifact, (20,)),
        ),
        (
            "sk_lesion",
            "isi_cv",
            lambda artifact: _set_event_intervals(artifact, (20,)),
        ),
    ],
)
def test_intrinsic_scorer_rejects_wrong_sign_raw_trace_controls(
    tmp_path, arm, metric, mutation,
):
    document = _intrinsic_document(tmp_path)
    _rewrite_arm(tmp_path, document, arm, mutation)

    result = score_intrinsic_lesion_observations(document, root=tmp_path)

    gate_id = next(gate for gate, gate_arm in {
        "nap-complete-lesion": "nap_lesion",
        "cav2.2-complete-lesion": "cav2_2_lesion",
        "sk-complete-lesion": "sk_lesion",
        "hcn-complete-lesion": "hcn_baseline_lesion",
    }.items() if gate_arm == arm)
    hard_gate = next(
        item for item in _intrinsic_results(result)[gate_id]["hard_gates"]
        if item["metric"] == metric
    )
    assert hard_gate["status"] == "scored"
    assert hard_gate["passed"] is False
    assert result["readiness_contract_result"] == "FAIL"


@pytest.mark.parametrize("identity", ["candidate", "protocol", "release"])
def test_intrinsic_scorer_rejects_mismatched_runner_identities(tmp_path, identity):
    document = _intrinsic_document(tmp_path)

    def mutate(artifact):
        if identity == "candidate":
            artifact["adaptive_candidate"]["candidate_id"] = "other-candidate"
        elif identity == "protocol":
            artifact["raw_observation"]["analysis_protocol"]["binding"]["sha256"] = "e" * 64
        else:
            artifact["provenance"]["candidate_release"]["sha256"] = "e" * 64

    _rewrite_arm(tmp_path, document, "nap_lesion", mutate)
    with pytest.raises(StageBScorerError, match="candidate/protocol/release identity"):
        score_intrinsic_lesion_observations(document, root=tmp_path)


def test_intrinsic_scorer_rejects_runner_artifact_digest_tampering(tmp_path):
    document = _intrinsic_document(tmp_path)
    document["runner_observations"]["nap_lesion"]["sha256"] = "0" * 64
    with pytest.raises(StageBScorerError, match="runner observation nap_lesion digest does not match"):
        score_intrinsic_lesion_observations(document, root=tmp_path)


def test_raw_scorer_recomputes_all_bounded_fixtures_and_preserves_boundary():
    result = score_raw_observations(_document(), root=ROOT)
    assert result["process_status"] == "completed"
    assert result["scientific_verdict"] == "GO"
    assert result["all_bounded_fixtures_passed"] is True
    assert result["adaptive_candidate"] == _document()["adaptive_candidate"]
    assert result["unscored_boundaries"] == ["hcn-baseline-non-significance-boundary"]
    by_id = {item["fixture_id"]: item for item in result["results"]}
    rate = by_id["adult-autonomous-rate-observed-range"]
    assert rate["value"] == 20.0
    assert rate["raw_metrics"]["spike_count"] == 8
    ratio = by_id["nalcn-lesion-ratio-4mM-model-derived"]
    assert ratio["value"] == pytest.approx(7 / 12)
    assert ratio["interval_provenance"] == "model-derived"
    assert ratio["raw_metrics"]["persistent_lesion_firing"] is True


def test_out_of_band_trace_is_valid_scientific_failure_not_scorer_error():
    document = _document()
    document["observations"][0]["raw"]["spike_times_s"] = [0.2]
    result = score_raw_observations(document, root=ROOT)
    assert result["process_status"] == "completed"
    assert result["scientific_verdict"] == "NO_GO"
    assert result["all_bounded_fixtures_passed"] is False
    failed = [item for item in result["results"] if item["passed"] is False]
    assert [item["fixture_id"] for item in failed] == ["adult-autonomous-rate-observed-range"]


def test_scorer_rejects_hidden_burn_in_instead_of_trusting_claimed_metrics():
    document = _document()
    raw = document["observations"][0]["raw"]
    raw["recording_start_s"] = -0.1
    raw["claimed_firing_rate_hz"] = 20.0
    with pytest.raises(StageBScorerError, match="invalid:.*burn_in_start_s"):
        score_raw_observations(document, root=ROOT)


def test_scorer_rejects_missing_duplicate_and_unknown_fixtures():
    missing = _document()
    missing["observations"].pop()
    with pytest.raises(StageBScorerError, match="do not cover every bounded fixture"):
        score_raw_observations(missing, root=ROOT)

    duplicate = _document()
    duplicate["observations"].append(copy.deepcopy(duplicate["observations"][0]))
    with pytest.raises(StageBScorerError, match="unknown or duplicate"):
        score_raw_observations(duplicate, root=ROOT)


def test_ratio_requires_matched_protocols_and_zero_firing_is_scientific_no_go():
    mismatch = _document()
    ratio_raw = mismatch["observations"][1]["raw"]
    ratio_raw["lesion"]["window_end_s"] = 0.4
    with pytest.raises(StageBScorerError, match="protocols must match exactly"):
        score_raw_observations(mismatch, root=ROOT)

    silent = _document()
    silent["observations"][1]["raw"]["lesion"]["spike_times_s"] = []
    result = score_raw_observations(silent, root=ROOT)
    assert result["scientific_verdict"] == "NO_GO"
    ratio = next(item for item in result["results"] if item["fixture_id"].startswith("nalcn"))
    assert ratio["raw_metrics"]["persistent_lesion_firing"] is False


def test_scorer_rejects_fixture_digest_tampering(tmp_path: Path):
    document = _document()
    document["fixture_packet"]["sha256"] = "0" * 64
    with pytest.raises(StageBScorerError, match="fixture packet digest does not match"):
        score_raw_observations(document, root=ROOT)


def test_scorer_rejects_malformed_candidate_echo():
    document = _document()
    document["adaptive_candidate"]["candidate_sha256"] = "not-a-digest"
    with pytest.raises(StageBScorerError, match="adaptive_candidate is malformed"):
        score_raw_observations(document, root=ROOT)


def test_file_boundary_writes_completed_no_go_but_no_infrastructure_result(tmp_path: Path):
    valid_input = tmp_path / "valid.json"
    valid_output = tmp_path / "valid-score.json"
    document = _document()
    document["observations"][0]["raw"]["spike_times_s"] = [0.2]
    valid_input.write_text(json.dumps(document), encoding="utf-8")
    result = score_raw_observation_file(valid_input, valid_output, root=ROOT)
    assert result["scientific_verdict"] == "NO_GO"
    assert json.loads(valid_output.read_text())["process_status"] == "completed"

    invalid_input = tmp_path / "invalid.json"
    invalid_output = tmp_path / "invalid-score.json"
    document["fixture_packet"]["sha256"] = "0" * 64
    invalid_input.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(StageBScorerError, match="digest does not match"):
        score_raw_observation_file(invalid_input, invalid_output, root=ROOT)
    assert not invalid_output.exists()


def test_cli_exits_nonzero_without_result_on_infrastructure_failure(tmp_path: Path):
    source = tmp_path / "bad.json"
    output = tmp_path / "score.json"
    source.write_text("not JSON", encoding="utf-8")
    with pytest.raises(SystemExit) as raised:
        _main(["--input", str(source), "--output", str(output), "--root", str(ROOT)])
    assert raised.value.code == 2
    assert not output.exists()
