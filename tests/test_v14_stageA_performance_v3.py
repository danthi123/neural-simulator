import inspect
import json
import os

import pytest

from tools import v14_stageA_performance_v3 as performance


def test_protocol_constants_include_audited_host_controls():
    assert performance.OUTER_PAIRS == 3
    assert performance.WARMUP_STEPS == 5000
    assert performance.REWARM_STEPS == 500
    assert performance.TIMING_BLOCKS == 5
    assert performance.STEPS_PER_BLOCK == 3000
    assert performance.DEFAULT_RATIO_MAX == 1.02
    assert performance.ACTIVE_RATIO_MAX == 1.25
    assert performance.DIRECT_OUTPUT_RATIO_MAX == 0.85
    assert performance.MAX_INNER_BLOCK_RELATIVE_RANGE == 0.10
    assert performance.MAX_PAIRED_RATIO_RELATIVE_RANGE == 0.05
    assert performance.WORKER_CPU == 10
    assert performance.CONTROLLER_CPU == 16
    assert performance.REQUIRED_NICE == -4
    assert performance.REQUIRED_GPU_CLOCK_MHZ == 1800
    assert performance.GPU_CLOCK_TOLERANCE_MHZ == 15
    assert performance.REQUIRED_MEMORY_CLOCK_MHZ == 9751
    assert performance.REQUIRED_POWER_LIMIT_W == 300


def test_protocol_spec_matches_runtime_contract():
    spec = json.loads(performance.PROTOCOL_SPEC_PATH.read_text())
    assert spec["status"] == "preregistered-not-executed"
    assert spec["scientific_seeds"] == []
    assert spec["measurement"]["outer_pairs"] == performance.OUTER_PAIRS
    assert spec["measurement"]["warmup_steps"] == performance.WARMUP_STEPS
    assert spec["measurement"]["fixed_rewarm_steps"] == performance.REWARM_STEPS
    assert spec["measurement"]["timing_blocks"] == performance.TIMING_BLOCKS
    assert spec["host_controls"]["worker_cpu"] == performance.WORKER_CPU
    assert spec["host_controls"]["controller_cpu"] == performance.CONTROLLER_CPU
    assert spec["host_controls"]["gpu_clock_mhz"] == performance.REQUIRED_GPU_CLOCK_MHZ
    assert spec["host_controls"]["memory_clock_mhz"] == performance.REQUIRED_MEMORY_CLOCK_MHZ
    assert spec["fixed_thresholds"]["default_off_ratio_max"] == performance.DEFAULT_RATIO_MAX
    assert spec["fixed_thresholds"]["active_ratio_max"] == performance.ACTIVE_RATIO_MAX
    assert spec["fixed_thresholds"]["direct_output_ratio_max"] == performance.DIRECT_OUTPUT_RATIO_MAX


def test_plan_is_reversible_four_cell_chain():
    plan = performance.build_run_plan()
    forward = [
        "candidate-active-unfused",
        "candidate-active",
        "candidate-default",
        "prechange-control-default",
    ]
    assert len(plan) == 12
    assert [job["sequence"] for job in plan] == list(range(1, 13))
    assert [job["cell"] for job in plan[:4]] == forward
    assert [job["cell"] for job in plan[4:8]] == list(reversed(forward))
    assert [job["cell"] for job in plan[8:]] == forward
    assert all("seed" not in job for job in plan)


def test_source_snapshot_tracks_v2_governed_boundary(tmp_path, monkeypatch):
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


def test_worker_is_persistent_strict_and_keeps_timing_uninterrupted():
    persistent_source = inspect.getsource(performance.persistent_worker)
    bridge_source = inspect.getsource(performance._build_bridge)
    observation_source = inspect.getsource(performance.run_observation)
    assert "for line in sys.stdin" in persistent_source
    assert 'command.get("command") == "shutdown"' in persistent_source
    assert "bridge.strict_step_errors = True" in bridge_source
    timed_window = observation_source[
        observation_source.index("for block_index in range"):
        observation_source.index("host_values =")
    ]
    assert "gpu_telemetry" not in timed_window
    assert "for _ in range(REWARM_STEPS)" in observation_source
    assert "scientific_seeds" in observation_source


def test_persistent_worker_launch_is_source_isolated_and_pinned(monkeypatch, tmp_path):
    captured = {}

    class FakeStream:
        def close(self):
            pass

    class FakeProcess:
        pid = 44
        stdin = FakeStream()
        stdout = FakeStream()

        def poll(self):
            return 0

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        return FakeProcess()

    monkeypatch.setattr(performance.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        performance.PersistentWorker,
        "_read_response",
        lambda self, **kwargs: {"status": "ready"},
    )
    worker = performance.PersistentWorker(source_root=tmp_path, source_kind="candidate")
    try:
        assert captured["cwd"] == tmp_path.resolve()
        assert captured["env"]["SIM_SOURCE_ROOT"] == str(tmp_path.resolve())
        assert captured["env"]["SIM_BACKEND"] == "cupy"
        assert "--persistent-worker" in captured["command"]
        affinity = set()
        monkeypatch.setattr(os, "sched_setaffinity", lambda pid, cpus: affinity.update(cpus))
        captured["preexec_fn"]()
        assert affinity == {10}
    finally:
        worker._stderr.close()


def _good_gpu_record(*, sm="1800", memory="9751", power="300.00"):
    return {
        "gpu": {
            "returncode": 0,
            "rows": [{"clocks.sm": sm, "clocks.mem": memory, "power.limit": power}],
        },
        "compute_processes": {"returncode": 0, "rows": []},
        "process_monitor": {"returncode": 0, "stdout": "# inventory"},
    }


@pytest.mark.parametrize(
    ("changes", "passed"),
    [
        ({}, True),
        ({"sm": "1815"}, True),
        ({"sm": "1815.1"}, False),
        ({"memory": "9750.999"}, False),
        ({"power": "299.999"}, False),
    ],
)
def test_gpu_control_validation_is_fail_closed(changes, passed):
    assert performance.validate_gpu_telemetry(_good_gpu_record(**changes))["passed"] is passed


def test_idle_gpu_precondition_validates_static_controls_only():
    idle = _good_gpu_record(sm="210", memory="405")
    assert performance.validate_gpu_telemetry(
        idle, require_active_clocks=False
    )["passed"] is True


def test_gpu_telemetry_records_process_inventory(monkeypatch):
    calls = []

    def fake_capture(command, **kwargs):
        calls.append(command)
        if "--query-gpu=" in command[1]:
            return {
                "command": command, "returncode": 0, "stderr": "",
                "stdout": "0, uuid, P0, 1800, 9751, 300, 100, 50, 90, 10, 24576",
            }
        return {"command": command, "returncode": 0, "stderr": "", "stdout": ""}

    monkeypatch.setattr(performance, "_command_capture", fake_capture)
    record = performance.gpu_telemetry()
    assert len(calls) == 3
    assert record["gpu"]["rows"][0]["clocks.sm"] == "1800"
    assert "compute_processes" in record
    assert "process_monitor" in record


def test_cpu_controls_require_governor_epp_profile_and_turbo(monkeypatch):
    def fake_read(path):
        path = str(path)
        if path.endswith("scaling_governor"):
            return "performance"
        if path.endswith("energy_performance_preference"):
            return "performance"
        if path.endswith("no_turbo"):
            return "0"
        if path.endswith("/online"):
            return "1"
        return None

    monkeypatch.setattr(performance, "_read_text", fake_read)
    monkeypatch.setattr(
        performance, "_power_profile", lambda: {"source": "test", "value": "performance"}
    )
    assert performance.cpu_controls(10)["passed"] is True
    monkeypatch.setattr(
        performance, "_power_profile", lambda: {"source": "test", "value": "balanced"}
    )
    assert performance.cpu_controls(10)["passed"] is False


def test_blocking_workload_scan_finds_indexer_and_research_runner(monkeypatch, tmp_path):
    proc = tmp_path / "proc"
    for pid, command in {
        "101": "python tools/rag/update_indexes.py",
        "102": "python -m research.runners.v14",
        "103": "python harmless.py",
    }.items():
        directory = proc / pid
        directory.mkdir(parents=True)
        (directory / "cmdline").write_bytes(command.replace(" ", "\0").encode())
    original_path = performance.Path

    def fake_path(value):
        return proc if value == "/proc" else original_path(value)

    monkeypatch.setattr(performance, "Path", fake_path)
    monkeypatch.setattr(performance.os, "getpid", lambda: 999)
    assert [row["pid"] for row in performance.active_blocking_workloads()] == [101, 102]


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
            rows.append({
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
            })
    return rows


def test_summary_preserves_v2_aggregation_and_thresholds():
    summary = performance.summarize(_rows(pair_ratio_offsets=(-0.005, 0.0, 0.005)))
    assert summary["median_paired_ratios"]["default_host"] == pytest.approx(1.0)
    assert summary["median_paired_ratios"]["active_host"] == pytest.approx(0.8)
    assert summary["median_paired_ratios"]["direct_output_host"] == pytest.approx(0.8)
    assert summary["infrastructure_valid"] is True
    assert summary["performance_status"] == "GO"
    assert summary["physiology_verdict"] is None
    assert summary["promotion_effect"] == "none"


def test_dispersion_or_threshold_failure_preserves_v2_semantics():
    assert performance.summarize(_rows(default_ratio=1.03))["performance_status"] == "NO_GO"
    invalid = performance.summarize(_rows(block_relative_range=0.101))
    assert invalid["infrastructure_valid"] is False
    assert invalid["performance_status"] == "infrastructure_invalid"


def test_preconditions_require_env_and_all_controls(monkeypatch):
    monkeypatch.setenv(performance.GPU_CLOCK_ENV, "1800")
    monkeypatch.setattr(performance, "cpu_controls", lambda cpu: {"passed": True})
    monkeypatch.setattr(performance, "active_blocking_workloads", lambda: [])
    monkeypatch.setattr(performance, "gpu_telemetry", _good_gpu_record)
    scheduling = {
        "affinity": [16], "nice": -4, "scheduler": performance.REQUIRED_SCHEDULER,
    }
    sources = {
        "candidate": {"revision": "abc", "status_porcelain": ""},
        "control": {"revision": "def", "status_porcelain": ""},
    }
    checks, _ = performance._preconditions(
        source_roots=sources, controller_state=scheduling
    )
    assert all(check["ok"] for check in checks)
    monkeypatch.setenv(performance.GPU_CLOCK_ENV, "1500")
    checks, _ = performance._preconditions(
        source_roots=sources, controller_state=scheduling
    )
    assert next(check for check in checks if check["name"] == "external GPU clock declaration")["ok"] is False


def test_failed_precondition_writes_receipt_without_workers(monkeypatch, tmp_path):
    snapshots = iter([
        {"revision": "abc", "status_porcelain": "dirty"},
        {"revision": "def", "status_porcelain": ""},
    ])
    monkeypatch.setattr(performance, "source_snapshot", lambda root: next(snapshots))
    monkeypatch.setattr(performance, "pin_current_process", lambda cpu: {})
    monkeypatch.setattr(
        performance, "_preconditions", lambda **kwargs: ([{"name": "failed", "ok": False}], None)
    )
    monkeypatch.setattr(
        performance, "PersistentWorker", lambda **kwargs: pytest.fail("worker launched")
    )
    output = tmp_path / "receipt.json"
    result = performance.run_matrix(
        candidate_root=tmp_path, control_root=tmp_path, output=output
    )
    assert result["status"] == "infrastructure_invalid"
    assert json.loads(output.read_text())["results"] == []


def test_controller_has_no_lease_seed_or_state_mutation_cli():
    destinations = {action.dest for action in performance._parser()._actions}
    assert "lease_path" not in destinations
    assert "scientific_seed" not in destinations
    source = inspect.getsource(performance)
    assert '["sudo"' not in source
    assert "nvidia-smi -lgc" not in source
