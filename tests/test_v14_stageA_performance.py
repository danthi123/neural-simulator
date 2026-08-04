import json
from pathlib import Path

import pytest

from tools import v14_stageA_performance as performance


def test_fixed_contract_matches_preregistration():
    spec = json.loads(performance.SPEC_PATH.read_text())
    assert performance.WARMUP_STEPS == spec["performance"]["warmup_steps"] == 500
    assert performance.TIMED_STEPS == spec["performance"]["measured_steps"] == 20000
    assert performance.REPETITIONS == spec["performance"]["repetitions"] == 3
    assert performance.NUM_NEURONS == spec["performance"]["num_neurons"] == 600
    assert performance.CONSTRUCTION_RNG_SEED == spec["performance"]["construction_rng_seed"] == 0
    assert performance.PROCESS_ORDER_SEED == spec["performance"]["process_order_seed"] == 20260804
    assert performance.DEFAULT_RATIO_MAX == spec["performance"]["default_off_ratio_max"]
    assert performance.ACTIVE_RATIO_MAX == spec["performance"]["active_ratio_max"]
    assert performance.ACTIVE_BYTES_PER_NEURON == 48


def test_run_plan_is_deterministic_complete_and_seed_free():
    assert performance.build_run_plan() == performance.build_run_plan()
    plan = performance.build_run_plan()
    assert len(plan) == 9
    assert [job["sequence"] for job in plan] == list(range(1, 10))
    assert {(job["cell"], job["rep"]) for job in plan} == {
        (cell, rep)
        for cell in performance.CELL_DEFINITIONS
        for rep in (1, 2, 3)
    }
    assert all("seed" not in job for job in plan)


def _rows(candidate_default=10.0, control_default=10.0, active=12.0):
    durations = {
        "candidate-default": candidate_default,
        "prechange-control-default": control_default,
        "candidate-active": active,
    }
    return [
        {
            "cell": cell,
            "status": "completed",
            "timing": {
                "host_seconds": duration + offset,
                "cuda_event_seconds": duration + offset - 1.0,
            },
            "structural": {"passed": True},
        }
        for cell, duration in durations.items()
        for offset in (-1.0, 0.0, 1.0)
    ]


def test_summary_reports_matching_host_and_cuda_median_ratios():
    summary = performance.summarize(_rows())
    ratios = summary["ratios_against_matching_medians"]
    assert ratios["default_host"] == pytest.approx(1.0)
    assert ratios["default_cuda_event"] == pytest.approx(1.0)
    assert ratios["active_host"] == pytest.approx(1.2)
    assert ratios["active_cuda_event"] == pytest.approx(11.0 / 9.0)
    assert summary["performance_status"] == "GO"
    assert summary["physiology_verdict"] is None
    assert summary["promotion_effect"] == "none"


def test_summary_preserves_fixed_failure_thresholds():
    summary = performance.summarize(_rows(candidate_default=10.3, control_default=10.0))
    assert summary["performance_status"] == "NO_GO"
    assert summary["fixed_thresholds"] == {
        "default_off_ratio_max": 1.02,
        "active_ratio_max": 1.25,
    }


def test_worker_subprocess_receives_explicit_source_root(monkeypatch, tmp_path):
    candidate = tmp_path / "candidate"
    control = tmp_path / "control"
    candidate.mkdir()
    control.mkdir()
    captured = {}

    class Completed:
        returncode = 0
        stderr = ""
        stdout = json.dumps({"status": "completed", "cell": "candidate-default"})

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        return Completed()

    monkeypatch.setattr(performance.subprocess, "run", fake_run)
    job = {
        "cell": "candidate-default",
        "source": "candidate",
        "active": False,
        "sequence": 1,
        "rep": 1,
    }
    result = performance.run_worker(job, candidate_root=candidate, control_root=control)
    assert captured["env"]["SIM_SOURCE_ROOT"] == str(candidate.resolve())
    assert captured["env"]["SIM_BACKEND"] == "cupy"
    assert captured["cwd"] == candidate
    assert "--source-root" in captured["command"]
    assert result["sequence"] == 1


def test_atomic_json_replaces_destination_without_temp_residue(tmp_path: Path):
    output = tmp_path / "result.json"
    output.write_text('{"old": true}\n')
    performance.write_json_atomic(output, {"new": True})
    assert json.loads(output.read_text()) == {"new": True}
    assert list(tmp_path.glob(f".{output.name}.*.tmp")) == []


def test_controller_has_no_lease_argument():
    destinations = {action.dest for action in performance._parser()._actions}
    assert "lease_path" not in destinations
    assert "scientific_seed" not in destinations
