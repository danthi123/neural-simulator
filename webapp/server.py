"""FastAPI server for the neural simulator research dashboard.

Phase 1 surface:
- GET /                       → dashboard HTML
- GET /api/findings           → list of all finding markdowns
- GET /api/findings/{name}    → raw markdown body
- GET /api/runs               → list completed runs (JSON files)
- GET /api/runs/{name}        → run detail (parsed JSON)
- POST /api/runs/launch       → kick off a new runner subprocess
- GET /api/runs/launch/{id}   → poll status of a launched run
- WS  /ws/runs/{id}           → stream stdout lines as they arrive

Run:
    uvicorn webapp.server:app --reload --port 8765
"""
from __future__ import annotations

import asyncio
import json
import os
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
class LaunchedRun:
    run_id: str
    cmd: list[str]
    started_at: float
    proc: subprocess.Popen[bytes] | None = None
    returncode: int | None = None
    stdout_lines: list[str] = field(default_factory=list)
    out_path: str | None = None


launched_runs: dict[str, LaunchedRun] = {}


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
        # Compute summary metrics
        phase_stats = data.get("phase_stats") or []
        final_qs = [ps.get("finalQ") for ps in phase_stats if "finalQ" in ps]
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


@app.get("/api/runs/{name}")
def get_run(name: str) -> JSONResponse:
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


@app.post("/api/runs/launch")
def launch_run(req: LaunchRequest) -> JSONResponse:
    if req.preset not in PRESETS:
        raise HTTPException(400, f"unknown preset; valid: {list(PRESETS)}")
    run_id = uuid.uuid4().hex[:12]
    out_filename = req.out_filename or f"g11_seed{req.seed}_{req.preset}_{run_id[:6]}.json"
    out_path = str(RAW_RUNS_DIR / out_filename)

    cmd = [
        sys.executable, "-m", "research.runners.g11_bg_runner",
        *PRESETS[req.preset],
        "--seed", str(req.seed),
        "--out", out_path,
        *req.extra_args,
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
        proc=proc, out_path=out_path,
    )
    launched_runs[run_id] = run

    asyncio.create_task(_drain_stdout(run))
    return JSONResponse({
        "run_id": run_id,
        "cmd": cmd,
        "out_path": out_path,
        "ws_url": f"/ws/runs/{run_id}",
    })


async def _drain_stdout(run: LaunchedRun) -> None:
    """Background task: drain subprocess stdout, save lines for tailers."""
    if run.proc is None or run.proc.stdout is None:
        return
    loop = asyncio.get_event_loop()
    while True:
        line = await loop.run_in_executor(None, run.proc.stdout.readline)
        if not line:
            break
        text = line.decode(errors="replace").rstrip("\r\n")
        run.stdout_lines.append(text)
    run.returncode = run.proc.wait()


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
        "out_path": run.out_path,
    })


@app.websocket("/ws/runs/{run_id}")
async def ws_run_stdout(websocket: WebSocket, run_id: str) -> None:
    await websocket.accept()
    run = launched_runs.get(run_id)
    if not run:
        await websocket.send_json({"error": "unknown run_id"})
        await websocket.close()
        return

    last_sent = 0
    try:
        while True:
            cur = len(run.stdout_lines)
            if cur > last_sent:
                for line in run.stdout_lines[last_sent:cur]:
                    await websocket.send_json({"type": "stdout", "line": line})
                last_sent = cur
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
