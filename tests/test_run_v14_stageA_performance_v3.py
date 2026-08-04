from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import run_v14_stageA_performance_v3 as runner


def test_contract_uses_reversible_fixed_environment():
    assert runner.CONTROLLER_CPU == 16
    assert runner.GPU_CLOCK_MHZ == 1800
    assert runner.MEMORY_CLOCK_MHZ == 9751


def test_active_project_workload_fails_before_mutation(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "HARNESS", tmp_path / "harness.py")
    monkeypatch.setattr(runner, "HOST_HEAVY_LEASE", tmp_path / "host.lock")
    runner.HARNESS.write_text("", encoding="utf-8")
    monkeypatch.setattr(runner, "_project_workloads", lambda: ["123 rag update"])
    with pytest.raises(SystemExit, match="project workload already active"):
        runner.main([
            "--prechange-control-root", str(tmp_path),
            "--out", str(tmp_path / "out.json"),
        ])


def test_gpu_clocks_and_profile_are_restored(monkeypatch, tmp_path):
    harness = tmp_path / "harness.py"
    harness.write_text("", encoding="utf-8")
    monkeypatch.setattr(runner, "HARNESS", harness)
    monkeypatch.setattr(runner, "HOST_HEAVY_LEASE", tmp_path / "host.lock")
    monkeypatch.setattr(runner, "_project_workloads", lambda: [])
    monkeypatch.setattr(runner, "_output", lambda command: "balanced")
    monkeypatch.setattr(runner.os, "sched_getaffinity", lambda pid: {0, 1})
    monkeypatch.setattr(runner.os, "sched_setaffinity", lambda pid, cpus: None)
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[:2] == ["sudo", "-n"] and command[-1] == "true":
            return SimpleNamespace(returncode=0)
        if command[0] == runner.sys.executable:
            return SimpleNamespace(returncode=0)
        return SimpleNamespace(returncode=0)

    affinity_values = iter(({0, 1}, {runner.CONTROLLER_CPU}))
    monkeypatch.setattr(runner.os, "sched_getaffinity", lambda pid: next(affinity_values))
    monkeypatch.setattr(runner, "_run", fake_run)

    assert runner.main([
        "--prechange-control-root", str(tmp_path),
        "--out", str(tmp_path / "out.json"),
    ]) == 0
    joined = [" ".join(command) for command in calls]
    assert any("-lgc 1800,1800" in command for command in joined)
    assert any("-lmc 9751,9751" in command for command in joined)
    assert any(command.endswith("-rgc") for command in joined)
    assert any(command.endswith("-rmc") for command in joined)
    assert ["powerprofilesctl", "set", "balanced"] in calls
