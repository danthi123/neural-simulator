#!/usr/bin/env python3
"""Persistent workboard and resource check for autonomous project sessions.

The conversation is not the source of truth for active work. This controller
keeps a small, human-readable board on disk and combines it with live process,
queue, CPU, and GPU observations. It does not make scientific decisions or
launch experiments; it makes forgotten work and unused lanes visible.

Typical session loop:

    python tools/autonomous_coordinator.py init
    python tools/autonomous_coordinator.py status
    python tools/autonomous_coordinator.py update-lane --id ... --status running --next-action ...
    python tools/autonomous_coordinator.py register-agent --id ... --task ... --write-set ...
    python tools/autonomous_coordinator.py snapshot
    python tools/autonomous_coordinator.py handoff

The same commands are safe from a linked worktree. Queue observations resolve
through the Git common checkout, matching the persistent dispatchers.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = ROOT / "research" / "coordination"
WORKBOARD = Path(os.environ.get("SIM_WORKBOARD", STATE_DIR / "workboard.json"))
STATUS_FILE = Path(os.environ.get("SIM_COORDINATOR_STATUS", STATE_DIR / "status.json"))
HEARTBEAT_FILE = Path(os.environ.get("SIM_COORDINATOR_HEARTBEATS", STATE_DIR / "heartbeats.jsonl"))
SCHEMA = "sim-autonomous-workboard-v1"
DEFAULT_HEARTBEAT_SECONDS = 900
STALE_AGENT_SECONDS = 3600


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def _epoch(value: Any) -> float | None:
    if not isinstance(value, str):
        return None
    try:
        return dt.datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.remove(temporary)
        except OSError:
            pass


def _default_board() -> dict[str, Any]:
    now = _now()
    return {
        "schema": SCHEMA,
        "updated_at": now,
        "mission": (
            "Advance a grounded, integrated, continuously learning mind on a shared fully spiking substrate; "
            "preserve biology, honest gates, scaffold burn-down, consumer hardware, and efficient development."
        ),
        "session_rules": {
            "source_of_truth": "research/coordination/workboard.json",
            "heartbeat_seconds": DEFAULT_HEARTBEAT_SECONDS,
            "controller_runs_experiments": True,
            "agents_build_or_audit_but_do_not_own_long_sweeps": True,
            "disjoint_work_must_run_concurrently": True,
            "gpu_requires_shared_lease_and_empty_running_queue_claim": True,
            "every_lane_has_a_next_action": True,
            "every_blocker_has_a_recovery_action": True,
        },
        "lanes": {
            "rag-freshness": {
                "status": "completed",
                "priority": 1,
                "resource": "local_cpu",
                "delegation": "controller",
                "write_set": ["tools/rag/update_indexes.py", "tests/test_rag_soma_manifest.py"],
                "next_action": "Keep the CPU-only refresh path current; rerun it after source or indexer changes and verify the labeled Llama gate.",
                "heartbeat_at": now,
                "owner": "controller",
            },
            "v13-performance-diagnostic": {
                "status": "blocked",
                "priority": 1,
                "resource": "local_gpu",
                "delegation": "controller",
                "write_set": ["research/specs/v13_stage0_performance_diagnostic_v9.json", "tools/v13_stage0_performance_diagnostic_v9.py"],
                "blocker": "Canonical GPU queue is occupied by the eprop K=16 job.",
                "recovery_action": "Recheck queue and shared lease; run the preregistered four-cell diagnostic when both are clear.",
                "next_action": "Do not consume GPU until readiness becomes true; then execute V9 controller.",
                "heartbeat_at": now,
                "owner": "controller",
            },
            "deep-research-packets": {
                "status": "ready",
                "priority": 2,
                "resource": "agent_cpu_network",
                "delegation": "agent",
                "write_set": ["tools/research_escalation.py", "tools/rag/source_intake.py", "tests/test_research_escalation.py"],
                "next_action": "Add structured parameter claims, prior-failure links, and a reviewable research-packet path around the existing escalation gate.",
                "heartbeat_at": now,
                "owner": "controller",
            },
            "experiment-controller": {
                "status": "ready",
                "priority": 2,
                "resource": "local_cpu_plus_pool",
                "delegation": "agent",
                "write_set": ["tools/experiment_controller.py", "tests/test_experiment_controller.py"],
                "next_action": "Implement a dry-run-only screen-to-confirmation controller around adaptive_experiment and the existing sealed job APIs.",
                "heartbeat_at": now,
                "owner": "controller",
            },
            "local-model-offload": {
                "status": "ready",
                "priority": 3,
                "resource": "local_gpu_gated",
                "delegation": "controller",
                "write_set": ["config/local_model_offload.json", "tools/local_model_offload.py"],
                "next_action": "Keep the service stopped during GPU experiments; use it only for bounded documentation/catalog triage when the lease is free.",
                "heartbeat_at": now,
                "owner": "controller",
            },
        },
        "agents": {},
        "events": [],
    }


def load_board(path: Path = WORKBOARD) -> dict[str, Any]:
    if not path.exists():
        return _default_board()
    try:
        board = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read workboard {path}: {exc}") from exc
    if not isinstance(board, dict) or board.get("schema") != SCHEMA:
        raise ValueError(f"workboard has unsupported schema: {path}")
    if not isinstance(board.get("lanes"), dict) or not isinstance(board.get("agents"), dict):
        raise ValueError("workboard must contain object-valued lanes and agents")
    return board


def save_board(board: dict[str, Any], path: Path = WORKBOARD) -> None:
    board["updated_at"] = _now()
    _atomic_json(path, board)


def _event(board: dict[str, Any], kind: str, subject: str, detail: str) -> None:
    events = board.setdefault("events", [])
    events.append({"at": _now(), "kind": kind, "subject": subject, "detail": detail})
    del events[:-100]


def _run(args: list[str], *, timeout: int = 10) -> str:
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip()


def shared_root() -> Path:
    override = os.environ.get("SIM_QUEUE_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    common = _run(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"])
    if common:
        return Path(common).parent.resolve()
    return ROOT


def _queue_count(path: Path) -> int:
    try:
        return sum(1 for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
                   if line.strip() and not line.lstrip().startswith("#"))
    except OSError:
        return 0


def _processes() -> list[dict[str, Any]]:
    rows = []
    output = _run(["ps", "-eo", "pid=,etimes=,pcpu=,args="], timeout=15)
    for line in output.splitlines():
        fields = line.strip().split(None, 3)
        if len(fields) != 4 or not fields[0].isdigit():
            continue
        args = fields[3]
        if not any(token in args for token in ("research.runners", "update_indexes.py", "llama-server", "autonomous_coordinator")):
            continue
        try:
            rows.append({"pid": int(fields[0]), "age_seconds": int(fields[1]), "cpu_percent": float(fields[2]), "args": args[:400]})
        except ValueError:
            continue
    return rows


def _gpu() -> dict[str, Any]:
    output = _run([
        "nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
        "--format=csv,noheader,nounits",
    ])
    if not output:
        return {"available": False}
    values = [item.strip() for item in output.splitlines()[0].split(",")]
    if len(values) != 5:
        return {"available": True, "raw": output, "parsed": False}
    names = ("utilization_percent", "memory_used_mib", "memory_total_mib", "power_watts", "temperature_c")
    parsed: dict[str, Any] = {"available": True, "parsed": True}
    for name, value in zip(names, values):
        try:
            parsed[name] = float(value)
        except ValueError:
            parsed[name] = value
    return parsed


def collect_resources() -> dict[str, Any]:
    queue = shared_root() / "research" / "queue"
    try:
        load = list(os.getloadavg())
    except OSError:
        load = []
    return {
        "observed_at": _now(),
        "host": socket.gethostname(),
        "cpu": {"logical_cores": os.cpu_count() or 0, "load_1_5_15": load},
        "gpu": _gpu(),
        "queues": {
            "root": str(queue),
            "gpu_pending": _queue_count(queue / "gpu.queue"),
            "gpu_running": _queue_count(queue / "gpu.queue.running"),
            "pool_pending": _queue_count(queue / "pool.queue"),
            "pool_running": _queue_count(queue / "pool.queue.running"),
        },
        "processes": _processes(),
    }


def board_warnings(board: dict[str, Any], resources: dict[str, Any] | None = None) -> list[str]:
    resources = resources or collect_resources()
    now = time.time()
    warnings: list[str] = []
    lanes = board.get("lanes", {})
    agents = board.get("agents", {})
    ready_agent_lanes = [
        (lane_id, lane) for lane_id, lane in lanes.items()
        if lane.get("status") in {"ready", "planned"} and lane.get("delegation") == "agent"
    ]
    missing_agent_lanes = []
    for lane_id, lane in ready_agent_lanes:
        agent_id = lane.get("agent_id")
        agent = agents.get(agent_id) if isinstance(agent_id, str) else None
        if not isinstance(agent, dict) or agent.get("status") != "running":
            missing_agent_lanes.append(lane_id)
    if missing_agent_lanes:
        names = ", ".join(missing_agent_lanes[:4])
        warnings.append(f"AGENT-DISPATCH-REQUIRED: ready disjoint lanes have no active assigned agent ({names})")
    for lane_id, lane in lanes.items():
        if lane.get("status") == "running":
            if not str(lane.get("next_action", "")).strip():
                warnings.append(f"LANE-MISSING-NEXT-ACTION: {lane_id}")
            heartbeat = _epoch(lane.get("heartbeat_at"))
            timeout = float(lane.get("heartbeat_timeout_seconds", DEFAULT_HEARTBEAT_SECONDS * 2))
            if heartbeat is None or now - heartbeat > timeout:
                warnings.append(f"LANE-STALE: {lane_id} has no heartbeat within {int(timeout)}s")
        if lane.get("status") == "blocked" and not str(lane.get("recovery_action", "")).strip():
            warnings.append(f"BLOCKED-WITHOUT-RECOVERY: {lane_id}")
    for agent_id, agent in agents.items():
        if agent.get("status") == "running":
            heartbeat = _epoch(agent.get("heartbeat_at") or agent.get("started_at"))
            if heartbeat is None or now - heartbeat > STALE_AGENT_SECONDS:
                warnings.append(f"AGENT-STALE: {agent_id}")
            if not str(agent.get("write_set", "")).strip() and not agent.get("read_only", False):
                warnings.append(f"AGENT-MISSING-WRITE-SET: {agent_id}")
    cpu = resources.get("cpu", {})
    load = (cpu.get("load_1_5_15") or [0])[0]
    cores = cpu.get("logical_cores") or 0
    if missing_agent_lanes and cores and isinstance(load, (int, float)) and load < max(1.0, cores * 0.35):
        warnings.append("CPU-CAPACITY-AVAILABLE: ready CPU-compatible lanes exist while host load is low")
    return warnings


def next_actions(board: dict[str, Any]) -> list[dict[str, Any]]:
    actions = []
    for lane_id, lane in board.get("lanes", {}).items():
        if lane.get("status") in {"ready", "planned", "running", "blocked"}:
            action = {
                "lane": lane_id,
                "status": lane.get("status"),
                "priority": lane.get("priority", 99),
                "next_action": lane.get("next_action", ""),
            }
            if lane.get("status") == "blocked":
                action["recovery_action"] = lane.get("recovery_action", "")
            actions.append(action)
    return sorted(actions, key=lambda item: (item["priority"], item["lane"]))


def snapshot(board: dict[str, Any], *, persist: bool = True) -> dict[str, Any]:
    resources = collect_resources()
    warnings = board_warnings(board, resources)
    result = {
        "schema": "sim-autonomous-snapshot-v1",
        "observed_at": resources["observed_at"],
        "workboard": str(WORKBOARD),
        "mission": board.get("mission", ""),
        "resources": resources,
        "warnings": warnings,
        "next_actions": next_actions(board),
        "agents": board.get("agents", {}),
        "lanes": board.get("lanes", {}),
    }
    if persist:
        _atomic_json(STATUS_FILE, result)
        HEARTBEAT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with HEARTBEAT_FILE.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"observed_at": result["observed_at"], "warnings": warnings,
                                     "resources": resources}, sort_keys=True) + "\n")
    return result


def _print_status(result: dict[str, Any], *, as_json: bool = False) -> None:
    if as_json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    resources = result["resources"]
    cpu = resources["cpu"]
    queues = resources["queues"]
    gpu = resources["gpu"]
    print(f"AUTONOMOUS STATUS {result['observed_at']} host={resources['host']}")
    print(f"CPU cores={cpu['logical_cores']} load={cpu['load_1_5_15']}")
    print(f"GPU available={gpu.get('available')} util={gpu.get('utilization_percent', '?')}% "
          f"memory={gpu.get('memory_used_mib', '?')}/{gpu.get('memory_total_mib', '?')}MiB")
    print("queues=" + " ".join(f"{key}={value}" for key, value in queues.items() if key != "root"))
    print("LANES")
    for action in result["next_actions"]:
        print(f"  [{action['status']}] p{action['priority']} {action['lane']}: {action['next_action']}")
    print("AGENTS")
    running = [key for key, value in result["agents"].items() if value.get("status") == "running"]
    print("  " + (", ".join(running) if running else "none"))
    if result["warnings"]:
        print("WARNINGS")
        for warning in result["warnings"]:
            print(f"  ! {warning}")
    else:
        print("WARNINGS\n  none")


def _lane(args: argparse.Namespace, board: dict[str, Any]) -> None:
    lane = board.setdefault("lanes", {}).setdefault(args.id, {})
    for name in ("status", "resource", "delegation", "next_action", "owner", "agent_id", "blocker", "recovery_action"):
        value = getattr(args, name, None)
        if value is not None:
            lane[name] = value
    if args.priority is not None:
        lane["priority"] = args.priority
    if args.write_set is not None:
        lane["write_set"] = args.write_set
    lane["heartbeat_at"] = _now()
    _event(board, "lane-update", args.id, lane.get("status", "updated"))


def _agent(args: argparse.Namespace, board: dict[str, Any], *, finish: bool = False) -> None:
    agent = board.setdefault("agents", {}).setdefault(args.id, {})
    if args.task is not None:
        agent["task"] = args.task
    if args.status is not None:
        agent["status"] = args.status
    if args.model is not None:
        agent["model"] = args.model
    if args.write_set is not None:
        agent["write_set"] = args.write_set
    if args.next_action is not None:
        agent["next_action"] = args.next_action
    result = getattr(args, "result", None)
    if result is not None:
        agent["result"] = result
    agent.setdefault("started_at", _now())
    agent["heartbeat_at"] = _now()
    if finish:
        agent["completed_at"] = _now()
    _event(board, "agent-finish" if finish else "agent-register", args.id, agent.get("status", "updated"))


def handoff(board: dict[str, Any], result: dict[str, Any]) -> Path:
    path = STATE_DIR / "HANDOFF.md"
    lines = [
        "# Autonomous Workboard Handoff",
        "",
        "This file is generated from `research/coordination/workboard.json` and live resource checks.",
        "Read it before continuing; the conversation is not the work ledger.",
        "",
        f"- Observed: `{result['observed_at']}`",
        f"- Host: `{result['resources']['host']}`",
        f"- Workboard: `{WORKBOARD}`",
        "",
        "## Next Actions",
        "",
    ]
    for action in result["next_actions"]:
        lines.append(f"- **{action['lane']}** (`{action['status']}`, priority {action['priority']}): {action['next_action']}")
        if action.get("recovery_action"):
            lines.append(f"  Recovery: {action['recovery_action']}")
    lines.extend(["", "## Agents", ""])
    if result["agents"]:
        for agent_id, agent in sorted(result["agents"].items()):
            lines.append(f"- `{agent_id}`: `{agent.get('status', 'unknown')}` - {agent.get('task', '')}")
            if agent.get("result"):
                lines.append(f"  Result: {agent['result']}")
    else:
        lines.append("- None registered.")
    lines.extend(["", "## Resource Snapshot", "", "```json", json.dumps(result["resources"], indent=2, sort_keys=True), "```", ""])
    lines.extend(["## Warnings", ""])
    lines.extend(f"- {warning}" for warning in result["warnings"] or ["none"])
    lines.extend(["", "## Operating Rule", "", "A running lane must have a heartbeat and a concrete next action. "
                  "A ready disjoint agent lane must be dispatched or explicitly blocked with a recovery action.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("init", help="create the persistent workboard if absent")
    status = sub.add_parser("status", help="show the board plus live resources")
    status.add_argument("--json", action="store_true")
    sub.add_parser("snapshot", help="write live status and append a heartbeat")
    sub.add_parser("handoff", help="write a resumable Markdown handoff from the board")
    check = sub.add_parser("check", help="exit 1 when work or resources need attention")
    check.add_argument("--json", action="store_true")
    nxt = sub.add_parser("next", help="print ordered persisted next actions")
    nxt.add_argument("--json", action="store_true")

    lane = sub.add_parser("update-lane", help="create or update one persistent lane")
    lane.add_argument("--id", required=True)
    lane.add_argument("--status", choices=("planned", "ready", "running", "blocked", "completed"))
    lane.add_argument("--priority", type=int)
    lane.add_argument("--resource")
    lane.add_argument("--delegation", choices=("controller", "agent", "local-model"))
    lane.add_argument("--next-action")
    lane.add_argument("--owner")
    lane.add_argument("--agent-id")
    lane.add_argument("--blocker")
    lane.add_argument("--recovery-action")
    lane.add_argument("--write-set", nargs="+")

    agent = sub.add_parser("register-agent", help="record a dispatched agent")
    agent.add_argument("--id", required=True)
    agent.add_argument("--task", required=True)
    agent.add_argument("--status", default="running", choices=("running", "completed", "blocked"))
    agent.add_argument("--model")
    agent.add_argument("--next-action")
    agent.add_argument("--write-set", nargs="*", default=[])
    finish = sub.add_parser("finish-agent", help="record an agent result")
    finish.add_argument("--id", required=True)
    finish.add_argument("--status", required=True, choices=("completed", "blocked"))
    finish.add_argument("--result", required=True)
    finish.add_argument("--task")
    finish.add_argument("--model")
    finish.add_argument("--next-action")
    finish.add_argument("--write-set", nargs="*")
    beat = sub.add_parser("heartbeat", help="refresh a lane or agent heartbeat")
    beat.add_argument("--lane")
    beat.add_argument("--agent")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "init":
        if WORKBOARD.exists():
            print(f"workboard exists: {WORKBOARD}")
        else:
            save_board(_default_board())
            print(f"created: {WORKBOARD}")
        return 0

    board = load_board()
    if args.command == "update-lane":
        _lane(args, board)
        save_board(board)
        return 0
    if args.command == "register-agent":
        _agent(args, board)
        save_board(board)
        return 0
    if args.command == "finish-agent":
        _agent(args, board, finish=True)
        save_board(board)
        return 0
    if args.command == "heartbeat":
        now = _now()
        if args.lane:
            board["lanes"].setdefault(args.lane, {})["heartbeat_at"] = now
        if args.agent:
            board["agents"].setdefault(args.agent, {})["heartbeat_at"] = now
        if not args.lane and not args.agent:
            raise SystemExit("heartbeat needs --lane or --agent")
        save_board(board)
        return 0

    result = snapshot(board, persist=True)
    if args.command == "handoff":
        path = handoff(board, result)
        print(path)
        return 1 if result["warnings"] else 0
    if args.command == "next":
        if args.json:
            print(json.dumps(result["next_actions"], indent=2, sort_keys=True))
        else:
            for item in result["next_actions"]:
                print(f"[{item['status']}] p{item['priority']} {item['lane']}: {item['next_action']}")
        return 0
    if args.command == "check":
        _print_status(result, as_json=args.json)
        return 1 if result["warnings"] else 0
    _print_status(result, as_json=getattr(args, "json", False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
