"""FastAPI server for the neural simulator research dashboard.

Phase 1 + 2 + 2.5 surface:
- GET /                       → dashboard HTML
- GET /api/findings           → list of all finding markdowns
- GET /api/findings/{name}    → raw markdown body
- GET /api/runs               → list completed runs (JSON files)
- GET /api/runs/{name}        → run detail (parsed JSON)
- POST /api/runs/launch       → kick off a new runner subprocess
- GET /api/runs/launch        → list active in-flight runs (Phase 2.5)
- GET /api/runs/launch/{id}   → poll status of a launched run
- WS  /ws/runs/{id}           → stream stdout lines + parsed progress events

Run:
    uvicorn webapp.server:app --reload --port 8765
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parent.parent
FINDINGS_DIR = REPO_ROOT / "research" / "findings"
RAW_RUNS_DIR = REPO_ROOT / "research" / "findings" / "raw" / "g11_bg"
STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Neural Simulator Research Dashboard")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ─────────────────────────────────────────────────────────────────────────
# Run launcher state — in-memory, single-process
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class ProgressEvent:
    """A parsed progress line from a runner's stdout — Phase 2.5 live mode."""
    step: int
    total: int
    pos: tuple[int, int]
    goal: tuple[int, int]
    recent_dist: float
    timestamp: float


@dataclass
class LaunchedRun:
    run_id: str
    cmd: list[str]
    started_at: float
    proc: subprocess.Popen[bytes] | None = None
    returncode: int | None = None
    stdout_lines: list[str] = field(default_factory=list)
    progress_events: list[ProgressEvent] = field(default_factory=list)
    out_path: str | None = None
    # If interactive launcher mode was used, this is the path to the JSON
    # file the runner polls every trial for runtime control. Webapp writes
    # to this file via POST /api/runs/launch/{id}/control.
    control_file: str | None = None


launched_runs: dict[str, LaunchedRun] = {}


# Match runner progress lines like:
#   [g11 seed=42] step 800/1800  pos=(6,1)  goal=(1,6)  recent_dist=7.58  actions=...
# The actions tail is optional and not parsed.
_PROGRESS_RE = re.compile(
    r"step\s+(\d+)/(\d+)\s+pos=\((-?\d+),(-?\d+)\)\s+goal=\((-?\d+),(-?\d+)\)"
    r"\s+recent_dist=([\d.]+)"
)


def _try_parse_progress(line: str, now: float) -> ProgressEvent | None:
    m = _PROGRESS_RE.search(line)
    if not m:
        return None
    step, total, x, y, gx, gy, rd = m.groups()
    try:
        return ProgressEvent(
            step=int(step), total=int(total),
            pos=(int(x), int(y)),
            goal=(int(gx), int(gy)),
            recent_dist=float(rd),
            timestamp=now,
        )
    except (TypeError, ValueError):
        return None


# ─────────────────────────────────────────────────────────────────────────
# Findings (markdown files)
# ─────────────────────────────────────────────────────────────────────────


@app.get("/api/findings")
def list_findings() -> JSONResponse:
    files = sorted(FINDINGS_DIR.glob("*.md"), reverse=True)
    out = []
    for f in files:
        out.append({
            "name": f.name,
            "size_bytes": f.stat().st_size,
            "modified_unix": f.stat().st_mtime,
        })
    return JSONResponse({"findings": out, "count": len(out)})


@app.get("/api/findings/{name}")
def get_finding(name: str) -> PlainTextResponse:
    if "/" in name or "\\" in name or ".." in name:
        raise HTTPException(400, "invalid finding name")
    f = FINDINGS_DIR / name
    if not f.is_file():
        raise HTTPException(404, "not found")
    return PlainTextResponse(f.read_text(encoding="utf-8"))


# ─────────────────────────────────────────────────────────────────────────
# Runs (raw output JSONs from the runner)
# ─────────────────────────────────────────────────────────────────────────


@app.get("/api/runs")
def list_runs() -> JSONResponse:
    """List completed runs with summary (finalQ, sum, seed)."""
    files = sorted(RAW_RUNS_DIR.glob("*.json"), reverse=True)
    out = []
    for f in files:
        try:
            data = json.loads(f.read_text())
        except Exception:
            continue
        # Compute summary metrics. The runner records phase metrics under
        # `final_quarter_mean_distance` (the canonical "finalQ" — mean
        # Manhattan distance during the last quarter of each phase).
        phase_stats = data.get("phase_stats") or []
        final_qs = []
        for ps in phase_stats:
            v = ps.get("final_quarter_mean_distance")
            if v is None:
                v = ps.get("finalQ")  # legacy fallback
            if v is not None:
                final_qs.append(v)
        summary = {
            "name": f.name,
            "size_bytes": f.stat().st_size,
            "modified_unix": f.stat().st_mtime,
            "seed": data.get("seed"),
            "n_steps": data.get("n_steps"),
            "n_phases": len(phase_stats),
            "final_qs": final_qs,
            "sum_finalQ": sum(final_qs) if final_qs else None,
            "config_flags": data.get("config_flags", []),
        }
        out.append(summary)
    return JSONResponse({"runs": out, "count": len(out)})


# NOTE: this catch-all run-by-name route must come AFTER all the specific
# `/api/runs/launch*` routes below — otherwise FastAPI matches by registration
# order and `/api/runs/launch` would be treated as a run name lookup.
def get_run_impl(name: str) -> JSONResponse:
    if "/" in name or "\\" in name or ".." in name:
        raise HTTPException(400, "invalid run name")
    f = RAW_RUNS_DIR / name
    if not f.is_file():
        raise HTTPException(404, "not found")
    try:
        data = json.loads(f.read_text())
    except Exception as e:
        raise HTTPException(500, f"failed to parse: {e}")
    return JSONResponse(data)


# ─────────────────────────────────────────────────────────────────────────
# Run launcher (spawn subprocess, stream stdout)
# ─────────────────────────────────────────────────────────────────────────


PRESETS: dict[str, list[str]] = {
    "flagship": [
        "--moving-goal",
        "--hippocampus", "--learned-perception", "--pfc",
        "--beacon-perception", "--beacon-replaces-goal",
        "--cue-reflex", "--cue-reflex-replaces-heuristic",
        "--landmarks", "--landmarks-replace-place",
        "--sensed-reward",
        "--adaptive-da", "--adaptive-da-ema-decay-negative", "0.7",
        "--curriculum", "--curriculum-warmup-steps", "600",
        "--n-steps", "1800",
    ],
    # Interactive presets — webapp wires --interactive-control-file
    # automatically. World-tab Live mode lets the user click in the grid
    # to teleport the goal, pause/resume, inject rewards.
    "interactive_flagship": [
        "--moving-goal",
        "--hippocampus", "--learned-perception", "--pfc",
        "--beacon-perception", "--beacon-replaces-goal",
        "--cue-reflex", "--cue-reflex-replaces-heuristic",
        "--landmarks", "--landmarks-replace-place",
        "--sensed-reward",
        "--adaptive-da", "--adaptive-da-ema-decay-negative", "0.7",
        "--curriculum", "--curriculum-warmup-steps", "600",
        # Long n_steps so the user has time to interact.
        "--n-steps", "3600",
    ],
    "interactive_baseline": [
        "--moving-goal",
        "--n-steps", "3600",
    ],
    "flagship_with_cheat5": [
        "--moving-goal",
        "--hippocampus", "--learned-perception", "--pfc",
        "--beacon-perception", "--beacon-replaces-goal",
        "--cue-reflex", "--cue-reflex-replaces-heuristic",
        "--landmarks", "--landmarks-replace-place",
        "--sensed-reward",
        "--bg-cross-projections", "--bg-cross-thaw-step", "1200", "--bg-cross-phase3-gain", "0.5",
        "--adaptive-da", "--adaptive-da-ema-decay-negative", "0.7",
        "--curriculum", "--curriculum-warmup-steps", "600",
        "--n-steps", "1800",
    ],
    "perception_only": [
        "--moving-goal",
        "--hippocampus", "--learned-perception", "--pfc",
        "--beacon-perception", "--beacon-replaces-goal",
        "--cue-reflex", "--cue-reflex-replaces-heuristic",
        "--landmarks", "--landmarks-replace-place",
        "--adaptive-da", "--adaptive-da-ema-decay-negative", "0.7",
        "--curriculum", "--curriculum-warmup-steps", "600",
        "--n-steps", "1800",
    ],
    "baseline": [
        "--moving-goal",
        "--n-steps", "1800",
    ],
    "smoke": [
        "--moving-goal",
        "--n-steps", "100",
    ],
}


class LaunchRequest(BaseModel):
    preset: str = "flagship"
    seed: int = 42
    extra_args: list[str] = []
    out_filename: str | None = None  # if None, generated from preset+seed


RUNTIME_DIR = Path(__file__).resolve().parent / "runtime"
RUNTIME_DIR.mkdir(parents=True, exist_ok=True)


@app.post("/api/runs/launch")
def launch_run(req: LaunchRequest) -> JSONResponse:
    if req.preset not in PRESETS:
        raise HTTPException(400, f"unknown preset; valid: {list(PRESETS)}")
    run_id = uuid.uuid4().hex[:12]
    out_filename = req.out_filename or f"g11_seed{req.seed}_{req.preset}_{run_id[:6]}.json"
    out_path = str(RAW_RUNS_DIR / out_filename)

    extras = list(req.extra_args)
    control_file: str | None = None
    if req.preset.startswith("interactive_"):
        control_file = str(RUNTIME_DIR / f"control_{run_id}.json")
        Path(control_file).write_text("{}")
        extras.extend(["--interactive-control-file", control_file])

    cmd = [
        sys.executable, "-m", "research.runners.g11_bg_runner",
        *PRESETS[req.preset],
        "--seed", str(req.seed),
        "--out", out_path,
        *extras,
    ]

    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )

    run = LaunchedRun(
        run_id=run_id, cmd=cmd, started_at=time.time(),
        proc=proc, out_path=out_path, control_file=control_file,
    )
    launched_runs[run_id] = run

    asyncio.create_task(_drain_stdout(run))
    return JSONResponse({
        "run_id": run_id,
        "cmd": cmd,
        "out_path": out_path,
        "control_file": control_file,
        "interactive": control_file is not None,
        "ws_url": f"/ws/runs/{run_id}",
    })


class ControlUpdate(BaseModel):
    """Body for POST /api/runs/launch/{run_id}/control. All fields optional;
    the runner reads the file fresh on each trial, so partial updates work
    by leaving fields unset (undefined → not changed)."""
    paused: bool | None = None
    goal: list[int] | None = None
    inject_reward: float | None = None


@app.post("/api/runs/launch/{run_id}/control")
def update_run_control(run_id: str, body: ControlUpdate) -> JSONResponse:
    """Write control state to the run's interactive control file. The runner
    polls this file at the start of each trial and applies whatever's there.
    Only available for runs launched with an `interactive_*` preset."""
    run = launched_runs.get(run_id)
    if not run:
        raise HTTPException(404, "unknown run_id")
    if not run.control_file:
        raise HTTPException(
            400,
            "this run was not launched in interactive mode (use an "
            "interactive_* preset to enable runtime control)"
        )
    # Merge into existing state (lets the user "pause" without clobbering goal)
    try:
        existing = json.loads(Path(run.control_file).read_text() or "{}")
    except (json.JSONDecodeError, OSError):
        existing = {}
    update = body.model_dump(exclude_unset=True)
    existing.update(update)
    try:
        Path(run.control_file).write_text(json.dumps(existing))
    except OSError as e:
        raise HTTPException(500, f"failed to write control file: {e}")
    return JSONResponse({"control_file": run.control_file, "state": existing})


@app.get("/api/runs/launch/{run_id}/control")
def get_run_control(run_id: str) -> JSONResponse:
    """Read the current interactive control state for a run."""
    run = launched_runs.get(run_id)
    if not run:
        raise HTTPException(404, "unknown run_id")
    if not run.control_file:
        return JSONResponse({"interactive": False, "state": None})
    try:
        state = json.loads(Path(run.control_file).read_text() or "{}")
    except (json.JSONDecodeError, OSError):
        state = {}
    return JSONResponse({
        "interactive": True,
        "control_file": run.control_file,
        "state": state,
    })


async def _drain_stdout(run: LaunchedRun) -> None:
    """Background task: drain subprocess stdout, save lines for tailers,
    and parse runner progress lines into structured events for live-mode
    consumers."""
    if run.proc is None or run.proc.stdout is None:
        return
    loop = asyncio.get_event_loop()
    while True:
        line = await loop.run_in_executor(None, run.proc.stdout.readline)
        if not line:
            break
        text = line.decode(errors="replace").rstrip("\r\n")
        run.stdout_lines.append(text)
        ev = _try_parse_progress(text, time.time())
        if ev is not None:
            run.progress_events.append(ev)
    run.returncode = run.proc.wait()


@app.get("/api/runs/launch")
def list_active_launches() -> JSONResponse:
    """Phase 2.5: list all in-flight (or recently-completed) runs known
    to this server process. Lets the World tab discover runs to follow."""
    out = []
    for run in launched_runs.values():
        is_running = run.proc is not None and run.proc.poll() is None
        latest = run.progress_events[-1] if run.progress_events else None
        out.append({
            "run_id": run.run_id,
            "running": is_running,
            "returncode": run.returncode,
            "started_at": run.started_at,
            "elapsed_sec": time.time() - run.started_at,
            "out_path": run.out_path,
            "interactive": run.control_file is not None,
            "latest_progress": _progress_to_json(latest) if latest else None,
        })
    out.sort(key=lambda r: r["started_at"], reverse=True)
    return JSONResponse({"runs": out, "count": len(out)})


@app.get("/api/runs/launch/{run_id}")
def launch_status(run_id: str) -> JSONResponse:
    run = launched_runs.get(run_id)
    if not run:
        raise HTTPException(404, "unknown run_id")
    is_running = run.proc is not None and run.proc.poll() is None
    return JSONResponse({
        "run_id": run.run_id,
        "running": is_running,
        "returncode": run.returncode,
        "started_at": run.started_at,
        "elapsed_sec": time.time() - run.started_at,
        "stdout_line_count": len(run.stdout_lines),
        "tail": run.stdout_lines[-20:],
        "progress_events": [_progress_to_json(p) for p in run.progress_events],
        "out_path": run.out_path,
    })


def _progress_to_json(p: ProgressEvent) -> dict[str, Any]:
    return {
        "step": p.step, "total": p.total,
        "pos": list(p.pos), "goal": list(p.goal),
        "recent_dist": p.recent_dist,
        "timestamp": p.timestamp,
    }


@app.websocket("/ws/runs/{run_id}")
async def ws_run_stdout(websocket: WebSocket, run_id: str) -> None:
    """Streams stdout lines AND parsed progress events as they arrive.
    On connect, replays any progress events buffered so far so a new
    subscriber sees the full trajectory the agent has covered."""
    await websocket.accept()
    run = launched_runs.get(run_id)
    if not run:
        await websocket.send_json({"error": "unknown run_id"})
        await websocket.close()
        return

    # Replay buffered progress events so a late-joining client gets the
    # full trajectory the agent has already covered.
    for p in run.progress_events:
        await websocket.send_json({"type": "progress", **_progress_to_json(p)})

    last_stdout = 0
    last_progress = len(run.progress_events)
    try:
        while True:
            # Stream new stdout lines
            cur_stdout = len(run.stdout_lines)
            if cur_stdout > last_stdout:
                for line in run.stdout_lines[last_stdout:cur_stdout]:
                    await websocket.send_json({"type": "stdout", "line": line})
                last_stdout = cur_stdout
            # Stream new progress events
            cur_progress = len(run.progress_events)
            if cur_progress > last_progress:
                for p in run.progress_events[last_progress:cur_progress]:
                    await websocket.send_json({"type": "progress", **_progress_to_json(p)})
                last_progress = cur_progress
            if run.proc is not None and run.proc.poll() is not None:
                await websocket.send_json({
                    "type": "done", "returncode": run.returncode,
                    "out_path": run.out_path,
                })
                break
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        return


# ─────────────────────────────────────────────────────────────────────────
# Project info
# ─────────────────────────────────────────────────────────────────────────


@app.get("/api/info")
def info() -> JSONResponse:
    return JSONResponse({
        "repo_root": str(REPO_ROOT),
        "findings_dir": str(FINDINGS_DIR),
        "raw_runs_dir": str(RAW_RUNS_DIR),
        "presets": list(PRESETS.keys()),
        "phase": "1 (research dashboard, runner launcher, findings browser)",
    })


# ─────────────────────────────────────────────────────────────────────────
# Frontend
# ─────────────────────────────────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
def index() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))


# Register the catch-all run-by-name route LAST so /api/runs/launch* take
# precedence (FastAPI matches by registration order).
@app.get("/api/runs/{name}")
def get_run(name: str) -> JSONResponse:
    return get_run_impl(name)


# Auto-group runs by filename suffix and aggregate per-experiment.
# `g11_seed42_v3lateral.json` → experiment "v3lateral".
# Used by the Experiments tab; mirrors the frontend's detectExperiment helper
# so users can hit /api/experiments and see the same grouping the UI shows.
_EXP_SUFFIX_RE = re.compile(r"^g11_seed\d+(?:_(.+))?\.json$")


def _detect_experiment(name: str) -> str:
    m = _EXP_SUFFIX_RE.match(name)
    if not m:
        return "(other)"
    return m.group(1) or "default"


@app.get("/api/experiments")
def list_experiments() -> JSONResponse:
    """Group runs by detected experiment, return per-experiment aggregates."""
    files = sorted(RAW_RUNS_DIR.glob("*.json"), reverse=True)
    by_exp: dict[str, list[dict[str, Any]]] = {}
    for f in files:
        try:
            data = json.loads(f.read_text())
        except Exception:
            continue
        phase_stats = data.get("phase_stats") or []
        final_qs = []
        for ps in phase_stats:
            v = ps.get("final_quarter_mean_distance") or ps.get("finalQ")
            if v is not None:
                final_qs.append(v)
        sum_q = sum(final_qs) if final_qs else None
        exp = _detect_experiment(f.name)
        by_exp.setdefault(exp, []).append({
            "name": f.name,
            "seed": data.get("seed"),
            "n_steps": data.get("n_steps"),
            "sum_finalQ": sum_q,
            "modified_unix": f.stat().st_mtime,
        })

    out = []
    for exp, runs in by_exp.items():
        sums = [r["sum_finalQ"] for r in runs if r["sum_finalQ"] is not None]
        if sums:
            mean_v = sum(sums) / len(sums)
            var = sum((x - mean_v) ** 2 for x in sums) / max(1, len(sums) - 1)
            std_v = var ** 0.5 if len(sums) > 1 else None
            min_v = min(sums)
            max_v = max(sums)
        else:
            mean_v = std_v = min_v = max_v = None
        out.append({
            "experiment": exp,
            "n_seeds": len(runs),
            "n_complete": len(sums),
            "mean_sum": mean_v,
            "std_sum": std_v,
            "min_sum": min_v,
            "max_sum": max_v,
            "runs": sorted(runs, key=lambda r: r["seed"] or 0),
        })

    out.sort(key=lambda r: (r["mean_sum"] is None, r["mean_sum"] or 0))
    return JSONResponse({"experiments": out, "count": len(out)})
