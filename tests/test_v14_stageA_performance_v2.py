import inspect
import json

import pytest

from tools import v14_stageA_performance_v2 as performance


def test_protocol_is_fixed_long_blocked_and_seed_free():
    assert performance.OUTER_PAIRS == 3
    assert performance.WARMUP_STEPS == 5000
    assert performance.TIMING_BLOCKS == 5
    assert performance.STEPS_PER_BLOCK == 3000
    assert performance.DEFAULT_RATIO_MAX == 1.02
    assert performance.ACTIVE_RATIO_MAX == 1.25
    assert performance.DIRECT_OUTPUT_RATIO_MAX == 0.85
    assert performance.CONSTRUCTION_RNG_SEED == 0
    assert performance.MAX_INNER_BLOCK_RELATIVE_RANGE == 0.10
    assert performance.MAX_PAIRED_RATIO_RELATIVE_RANGE == 0.05


def test_committed_protocol_spec_matches_runtime_constants():
    spec = json.loads(performance.PROTOCOL_SPEC_PATH.read_text())
    assert spec["status"] == "preregistered-not-executed"
    assert spec["scientific_seeds"] == []
    assert spec["protocol"]["outer_pairs"] == performance.OUTER_PAIRS
    assert spec["protocol"]["warmup_steps"] == performance.WARMUP_STEPS
    assert spec["protocol"]["timing_blocks"] == performance.TIMING_BLOCKS
    assert spec["protocol"]["steps_per_block"] == performance.STEPS_PER_BLOCK
    assert spec["fixed_thresholds"]["default_off_ratio_max"] == performance.DEFAULT_RATIO_MAX
    assert spec["fixed_thresholds"]["active_ratio_max"] == performance.ACTIVE_RATIO_MAX
    assert spec["fixed_thresholds"]["direct_output_ratio_max"] == performance.DIRECT_OUTPUT_RATIO_MAX


def test_plan_uses_adjacent_deterministic_default_ab_ba_pairs():
    assert performance.build_run_plan() == performance.build_run_plan()
    plan = performance.build_run_plan()
    assert len(plan) == 12
    assert [job["sequence"] for job in plan] == list(range(1, 13))
    assert all("seed" not in job for job in plan)

    expected = {
        1: ("AB", ["candidate-default", "prechange-control-default"]),
        2: ("BA", ["prechange-control-default", "candidate-default"]),
        3: ("AB", ["candidate-default", "prechange-control-default"]),
    }
    for outer_pair, (order, cells) in expected.items():
        defaults = [
            job for job in plan
            if job["outer_pair"] == outer_pair and job["phase"] == "default_pair"
        ]
        assert [job["cell"] for job in defaults] == cells
        assert [job["pair_order"] for job in defaults] == [order, order]
        assert defaults[1]["sequence"] == defaults[0]["sequence"] + 1


def test_source_snapshot_tracks_governed_source_boundary(tmp_path, monkeypatch):
    tracked = (
        "sim/backend.py", "sim/bridge.py", "sim/config.py", "sim/kernels.py",
        "sim/regions.py",
    )
    for relative in tracked:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
    monkeypatch.setattr(performance, "_git", lambda *args: "clean")
    assert tuple(performance.source_snapshot(tmp_path)["files"]) == tracked


def test_worker_enforces_strict_errors_and_telemetry_boundary():
    bridge_source = inspect.getsource(performance._build_bridge)
    worker_source = inspect.getsource(performance.worker)
    assert "bridge.strict_step_errors = True" in bridge_source
    assert worker_source.index("telemetry_before = _nvidia_smi()") < worker_source.index(
        "for _ in range(WARMUP_STEPS)"
    )
    assert worker_source.index("telemetry_after = _nvidia_smi()") > worker_source.index(
        "for block_index in range(1, TIMING_BLOCKS + 1)"
    )
    uninterrupted = worker_source[
        worker_source.index("for _ in range(WARMUP_STEPS)"):
        worker_source.index("telemetry_after = _nvidia_smi()")
    ]
    assert "_nvidia_smi" not in uninterrupted


def _rows(
    *, default_ratio=1.0, active_ratio=0.8, direct_ratio=0.8,
    block_relative_range=0.02, pair_ratio_offsets=(0.0, 0.0, 0.0),
):
    rows = []
    for outer_pair, offset in enumerate(pair_ratio_offsets, 1):
        control = 10.0
        candidate = control * (default_ratio + offset)
        active = candidate * active_ratio
        unfused = active / direct_ratio
        values = {
            "candidate-default": candidate,
            "prechange-control-default": control,
            "candidate-active": active,
            "candidate-active-unfused": unfused,
        }
        for cell, duration in values.items():
            rows.append(
                {
                    "outer_pair": outer_pair,
                    "cell": cell,
                    "status": "completed",
                    "structural": {"passed": True},
                    "timing": {
                        "total_host_seconds": duration,
                        "total_cuda_event_seconds": duration,
                        "host_block_relative_range": block_relative_range,
                        "cuda_block_relative_range": block_relative_range,
                    },
                }
            )
    return rows


def test_summary_uses_median_of_three_paired_ratios_and_can_go():
    summary = performance.summarize(
        _rows(pair_ratio_offsets=(-0.005, 0.0, 0.005))
    )
    assert summary["median_paired_ratios"]["default_host"] == pytest.approx(1.0)
    assert summary["median_paired_ratios"]["active_host"] == pytest.approx(0.8)
    assert summary["median_paired_ratios"]["direct_output_host"] == pytest.approx(0.8)
    assert summary["dispersion"]["passed"] is True
    assert summary["infrastructure_valid"] is True
    assert summary["performance_status"] == "GO"
    assert summary["physiology_verdict"] is None
    assert summary["promotion_effect"] == "none"


def test_threshold_failure_is_scientific_no_go_when_infrastructure_valid():
    summary = performance.summarize(_rows(default_ratio=1.03))
    assert summary["infrastructure_valid"] is True
    assert summary["performance_status"] == "NO_GO"


@pytest.mark.parametrize(
    ("rows", "failed_boundary"),
    [
        (_rows(block_relative_range=0.101), "observation_blocks"),
        (_rows(pair_ratio_offsets=(-0.03, 0.0, 0.03)), "paired_ratio_relative_ranges"),
    ],
)
def test_excessive_dispersion_is_infrastructure_invalid(rows, failed_boundary):
    summary = performance.summarize(rows)
    assert summary["dispersion"]["passed"] is False
    assert summary["infrastructure_valid"] is False
    assert summary["performance_status"] == "infrastructure_invalid"
    assert summary["dispersion"][failed_boundary]


def test_incomplete_matrix_remains_pending():
    summary = performance.summarize(_rows()[:-1])
    assert summary["performance_status"] == "pending"
    assert summary["infrastructure_valid"] is False


def test_worker_subprocess_gets_explicit_source_root(monkeypatch, tmp_path):
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
    job = performance.build_run_plan()[0]
    performance.run_worker(job, candidate_root=candidate, control_root=control)
    assert captured["env"]["SIM_SOURCE_ROOT"] == str(candidate.resolve())
    assert captured["env"]["SIM_BACKEND"] == "cupy"
    assert captured["cwd"] == candidate
    assert "--source-root" in captured["command"]


def test_dirty_source_precondition_stops_before_dispatch(monkeypatch, tmp_path):
    snapshots = iter(
        [
            {"root": "candidate", "revision": "abc", "status_porcelain": " M sim/bridge.py"},
            {"root": "control", "revision": "def", "status_porcelain": ""},
        ]
    )
    monkeypatch.setattr(performance, "source_snapshot", lambda root: next(snapshots))
    monkeypatch.setattr(
        performance, "run_worker", lambda *args, **kwargs: pytest.fail("dispatched worker")
    )
    output = tmp_path / "receipt.json"
    result = performance.run_matrix(
        candidate_root=tmp_path, control_root=tmp_path, output=output
    )
    assert result["status"] == "infrastructure_invalid"
    assert result["summary"]["performance_status"] == "infrastructure_invalid"
    assert json.loads(output.read_text())["results"] == []


def test_controller_owns_neither_gpu_lease_nor_scientific_seed():
    destinations = {action.dest for action in performance._parser()._actions}
    assert "lease_path" not in destinations
    assert "scientific_seed" not in destinations
