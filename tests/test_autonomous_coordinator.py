import json
import os
import subprocess
from pathlib import Path
import sys

from tools.autonomous_coordinator import (
    SCHEMA,
    _default_board,
    board_warnings,
    load_board,
    save_board,
)


def _quiet_resources():
    return {
        "cpu": {"logical_cores": 20, "load_1_5_15": [0.2, 0.2, 0.2]},
        "gpu": {"available": True, "utilization_percent": 0},
        "queues": {"gpu_pending": 0, "gpu_running": 0, "pool_pending": 0, "pool_running": 0},
    }


def test_default_board_has_persistent_next_actions_and_recovery_actions():
    board = _default_board()
    assert board["schema"] == SCHEMA
    assert board["lanes"]["v13-performance-diagnostic"]["status"] == "blocked"
    assert board["lanes"]["v13-performance-diagnostic"]["recovery_action"]
    assert all(str(lane.get("next_action", "")).strip() for lane in board["lanes"].values())


def test_ready_disjoint_agent_lanes_raise_dispatch_warning():
    board = _default_board()
    warnings = board_warnings(board, _quiet_resources())
    assert any(warning.startswith("AGENT-DISPATCH-REQUIRED") for warning in warnings)

    board["agents"]["agent-1"] = {
        "status": "running",
        "task": "bounded audit",
        "write_set": ["docs/example.md"],
        "started_at": board["updated_at"],
        "heartbeat_at": board["updated_at"],
    }
    warnings = board_warnings(board, _quiet_resources())
    assert not any(warning.startswith("AGENT-DISPATCH-REQUIRED") for warning in warnings)


def test_running_lane_without_fresh_heartbeat_is_loud():
    board = _default_board()
    board["lanes"]["rag-freshness"]["heartbeat_at"] = "2020-01-01T00:00:00+00:00"
    warnings = board_warnings(board, _quiet_resources())
    assert any(warning.startswith("LANE-STALE: rag-freshness") for warning in warnings)


def test_workboard_round_trips_atomically(tmp_path: Path):
    path = tmp_path / "workboard.json"
    board = _default_board()
    save_board(board, path)
    loaded = load_board(path)
    assert loaded["schema"] == SCHEMA
    assert loaded["lanes"].keys() == board["lanes"].keys()
    assert json.loads(path.read_text(encoding="utf-8"))["schema"] == SCHEMA


def test_cli_init_and_json_status_use_overridable_state(tmp_path: Path):
    board = tmp_path / "workboard.json"
    status = tmp_path / "status.json"
    heartbeats = tmp_path / "heartbeats.jsonl"
    env = {
        **os.environ,
        "SIM_WORKBOARD": str(board),
        "SIM_COORDINATOR_STATUS": str(status),
        "SIM_COORDINATOR_HEARTBEATS": str(heartbeats),
    }
    root = Path(__file__).resolve().parents[1]
    init = subprocess.run(
        [sys.executable, "tools/autonomous_coordinator.py", "init"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "created" in init.stdout
    report = subprocess.run(
        [sys.executable, "tools/autonomous_coordinator.py", "status", "--json"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    parsed = json.loads(report.stdout)
    assert parsed["schema"] == "sim-autonomous-snapshot-v1"
    assert status.is_file()
    assert heartbeats.is_file()

    subprocess.run(
        [sys.executable, "tools/autonomous_coordinator.py", "register-agent",
         "--id", "agent-1", "--task", "bounded task", "--write-set", "tools/example.py"],
        cwd=root,
        env=env,
        check=True,
    )
    subprocess.run(
        [sys.executable, "tools/autonomous_coordinator.py", "finish-agent",
         "--id", "agent-1", "--status", "completed", "--result", "tests passed"],
        cwd=root,
        env=env,
        check=True,
    )
    assert json.loads(board.read_text(encoding="utf-8"))["agents"]["agent-1"]["status"] == "completed"
