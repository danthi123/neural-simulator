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

class NoCacheStaticFiles(StaticFiles):
    """Disable browser caching of /static/* in dev so JS/CSS edits are
    picked up on the next reload without a manual hard-refresh. The
    research dashboard is single-user dev-only, so no perf cost matters."""

    def file_response(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        resp = super().file_response(*args, **kwargs)
        resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["Expires"] = "0"
        return resp


app = FastAPI(title="Neural Simulator Research Dashboard")
app.mount("/static", NoCacheStaticFiles(directory=str(STATIC_DIR)), name="static")


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
    # Set when the subprocess exits (or, for orphan recovery, when we
    # detect the orphan PID is dead). Used to freeze elapsed_sec for done
    # runs so the live picker stops ticking once a run completes.
    finished_at: float | None = None
    stdout_lines: list[str] = field(default_factory=list)
    progress_events: list[ProgressEvent] = field(default_factory=list)
    out_path: str | None = None
    # If interactive launcher mode was used, this is the path to the JSON
    # file the runner polls every trial for runtime control. Webapp writes
    # to this file via POST /api/runs/launch/{id}/control.
    control_file: str | None = None
    # Path to the per-run log file holding the runner's stdout/stderr. We
    # write subprocess output here (instead of in-memory pipe) so the run
    # survives the webapp being restarted/reloaded — a broken pipe on
    # restart was killing every in-flight run before this change.
    log_file: str | None = None
    # PID of the detached subprocess. Stored even when proc=None (e.g.
    # after webapp restart we recover by reading the sidecar) so the kill
    # endpoint can still terminate orphan runs.
    pid: int | None = None
    # Position in the log file we've already drained (for the tail loop).
    log_pos: int = 0


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
    # Exclude sidecar `.cmd.json` files which are launcher metadata, not runs
    files = sorted(
        (f for f in RAW_RUNS_DIR.glob("*.json") if not f.name.endswith(".cmd.json")),
        reverse=True,
    )
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
    # Interactive presets — webapp wires --interactive-control-file +
    # per-step progress prints + a small throttle so the dashboard's
    # live mode can animate per-step (not jumping every 100). World-tab
    # Live mode lets the user click in the grid to teleport the goal,
    # pause/resume, inject rewards.
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
        "--progress-print-interval", "1",
        "--trial-sleep-ms", "30",  # ~33 steps/sec — human-watchable
    ],
    "interactive_baseline": [
        "--moving-goal",
        "--n-steps", "3600",
        "--progress-print-interval", "1",
        "--trial-sleep-ms", "30",
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

# Trash directory for soft-deleted runs. Files moved here keep their
# names (with timestamp suffix to avoid collisions). Restore moves them
# back to RAW_RUNS_DIR; "Empty trash" deletes permanently.
TRASH_DIR = RAW_RUNS_DIR / ".trash"
TRASH_DIR.mkdir(parents=True, exist_ok=True)


@app.post("/api/runs/launch")
async def launch_run(req: LaunchRequest) -> JSONResponse:
    """Spin up a runner subprocess. Must be `async def` because we use
    `asyncio.create_task` for the stdout drainer — a sync def gets
    dispatched to a worker thread with no running event loop, which
    raises RuntimeError when trying to schedule the drainer."""
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

    # Inject a sensible --progress-print-interval default if none is set
    # in the preset or the user's extras.
    #   interactive_*  -> 1   (per-step, for live-mode animation while attached)
    #   everything else -> 20 (every 20 steps; smoothes the live chart but
    #                          avoids the per-step CPU<->GPU sync overhead
    #                          identified in the throughput investigation)
    base_extras = list(extras) + list(PRESETS[req.preset])
    if not any(a == "--progress-print-interval" for a in base_extras):
        default_ppi = "1" if req.preset.startswith("interactive_") else "20"
        extras.extend(["--progress-print-interval", default_ppi])

    cmd = [
        sys.executable, "-m", "research.runners.g11_bg_runner",
        *PRESETS[req.preset],
        "--seed", str(req.seed),
        "--out", out_path,
        *extras,
    ]

    # Per-run log file. Subprocess writes stdout/stderr here directly
    # (NOT through a pipe) so the run survives webapp restarts. The drain
    # task tails this file; on startup we scan sidecars to recover orphan
    # runs after the webapp was reloaded.
    log_file = str(RUNTIME_DIR / f"run_{run_id}.log")
    log_handle = open(log_file, "wb")

    # Force UTF-8 stdout. Windows defaults to cp1252 which crashes on
    # Unicode chars (em-dash, arrows) in runner prints.
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    # Detach the subprocess so it survives webapp restart. On Unix
    # `start_new_session=True` is enough (setsid). On Windows we need
    # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP — start_new_session
    # alone only sets CREATE_NEW_PROCESS_GROUP and the child still dies
    # when the console session goes (verified empirically 2026-04-28).
    popen_kwargs = dict(
        cwd=str(REPO_ROOT),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        env=env,
    )
    if sys.platform == "win32":
        popen_kwargs["creationflags"] = (
            subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        )
    else:
        popen_kwargs["start_new_session"] = True
    proc = subprocess.Popen(cmd, **popen_kwargs)
    log_handle.close()  # child has its own handle now

    # Sidecar `.cmd.json` next to the run output. Records EVERYTHING needed
    # to recover this run after a webapp restart: preset, seed, extras,
    # cmd, pid, log_file, control_file, run_id. The Re-run button uses
    # preset+seed+extras; the orphan-recovery uses pid+log_file+run_id.
    cleanable_extras = [
        a for a in req.extra_args
        if not a.startswith("--interactive-control-file")
    ]
    sidecar_path = Path(out_path).with_suffix(".cmd.json")
    try:
        sidecar_path.write_text(json.dumps({
            "run_id": run_id,
            "preset": req.preset,
            "seed": req.seed,
            "extra_args": cleanable_extras,
            "cmd": cmd,
            "pid": proc.pid,
            "log_file": log_file,
            "control_file": control_file,
            "out_path": out_path,
            "started_at": time.time(),
        }, indent=2))
    except OSError:
        pass

    run = LaunchedRun(
        run_id=run_id, cmd=cmd, started_at=time.time(),
        proc=proc, out_path=out_path, control_file=control_file,
        log_file=log_file, pid=proc.pid,
    )
    launched_runs[run_id] = run

    asyncio.create_task(_drain_log(run))
    return JSONResponse({
        "run_id": run_id,
        "cmd": cmd,
        "out_path": out_path,
        "log_file": log_file,
        "control_file": control_file,
        "interactive": control_file is not None,
        "pid": proc.pid,
        "ws_url": f"/ws/runs/{run_id}",
    })


@app.get("/api/runs/{name}/sidecar")
def get_run_sidecar(name: str) -> JSONResponse:
    """Return the launcher-recorded sidecar for a given run, if it exists.
    Used by the Re-run button to prefill the launcher form."""
    if "/" in name or "\\" in name or ".." in name:
        raise HTTPException(400, "invalid run name")
    sidecar_path = (RAW_RUNS_DIR / name).with_suffix(".cmd.json")
    if not sidecar_path.is_file():
        raise HTTPException(404, "no sidecar — run was not launched via webapp")
    try:
        return JSONResponse(json.loads(sidecar_path.read_text()))
    except (OSError, json.JSONDecodeError) as e:
        raise HTTPException(500, f"sidecar read failed: {e}")


@app.post("/api/runs/launch/{run_id}/kill")
def kill_run(run_id: str) -> JSONResponse:
    """Terminate an in-flight run. Sends SIGTERM, escalates to SIGKILL after
    5s if still alive. Works for both natively-spawned runs (uses run.proc)
    and orphan runs recovered from sidecars after webapp restart (uses
    run.pid + os.kill)."""
    import signal as _signal
    run = launched_runs.get(run_id)
    if not run:
        raise HTTPException(404, "unknown run_id")

    # Native-spawned runs: use the Popen handle.
    if run.proc is not None:
        if run.proc.poll() is not None:
            return JSONResponse({
                "run_id": run_id, "status": "already_done",
                "returncode": run.returncode,
            })
        try:
            run.proc.terminate()
            try:
                run.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                run.proc.kill()
                run.proc.wait(timeout=2)
            run.returncode = run.proc.returncode
            if run.finished_at is None:
                run.finished_at = time.time()
            return JSONResponse({
                "run_id": run_id, "status": "killed",
                "returncode": run.returncode,
            })
        except Exception as e:
            raise HTTPException(500, f"kill failed: {e}")

    # Recovered orphan: only have pid. Use os.kill (cross-platform; on
    # Windows SIGTERM maps to TerminateProcess).
    if run.pid is None or not _process_alive(run.pid):
        return JSONResponse({
            "run_id": run_id, "status": "already_done",
            "returncode": run.returncode,
        })
    try:
        os.kill(run.pid, _signal.SIGTERM)
        # Brief grace period; then check
        for _ in range(50):
            if not _process_alive(run.pid):
                break
            time.sleep(0.1)
        if _process_alive(run.pid):
            try:
                os.kill(run.pid, _signal.SIGKILL)
            except (OSError, AttributeError):
                # Windows doesn't have SIGKILL; signal.SIGTERM already does TerminateProcess.
                pass
        run.returncode = -15  # convention for SIGTERM-killed
        if run.finished_at is None:
            run.finished_at = time.time()
        return JSONResponse({
            "run_id": run_id, "status": "killed (orphan)",
            "returncode": run.returncode,
        })
    except Exception as e:
        raise HTTPException(500, f"kill failed: {e}")


# ─────────────────────────────────────────────────────────────────────────
# Orphan-run recovery on server startup
# ─────────────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def recover_orphan_runs() -> None:
    """Scan sidecar `.cmd.json` files for runs whose pid is still alive.
    Rebuild LaunchedRun entries for them so the dashboard's Live picker
    can attach. Survives uvicorn --reload restarts.

    Sidecars whose pid is dead are ignored (run already completed).
    Sidecars whose log file is missing are skipped."""
    for sidecar_path in RAW_RUNS_DIR.glob("*.cmd.json"):
        try:
            sidecar = json.loads(sidecar_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        run_id = sidecar.get("run_id")
        if not run_id or run_id in launched_runs:
            continue
        pid = sidecar.get("pid")
        if not pid or not _process_alive(pid):
            continue
        log_file = sidecar.get("log_file")
        if not log_file or not Path(log_file).exists():
            continue
        # Reconstruct the LaunchedRun. We don't have the Popen handle
        # (proc=None) but we have everything else needed for tail + kill.
        run = LaunchedRun(
            run_id=run_id,
            cmd=sidecar.get("cmd", []),
            started_at=sidecar.get("started_at", time.time()),
            proc=None,  # orphan; can't direct-control
            out_path=sidecar.get("out_path"),
            control_file=sidecar.get("control_file"),
            log_file=log_file,
            pid=pid,
        )
        launched_runs[run_id] = run
        # Resume tailing the log file from current end (not start) so
        # we don't replay the entire history. New progress events will
        # arrive as the runner continues.
        try:
            run.log_pos = Path(log_file).stat().st_size
        except OSError:
            run.log_pos = 0
        # Replay progress events by also reading from start? We need them
        # for the WebSocket replay-on-attach. Compromise: re-read from
        # start NOW to seed progress_events, then continue tailing.
        try:
            new_pos, lines = _read_new_lines(log_file, 0)
            for line in lines:
                run.stdout_lines.append(line)
                ev = _try_parse_progress(line, time.time())
                if ev is not None:
                    run.progress_events.append(ev)
            run.log_pos = new_pos
        except Exception:
            pass
        asyncio.create_task(_drain_log(run))
        print(f"[webapp] recovered orphan run {run_id} (pid={pid})", flush=True)


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


def _read_new_lines(path: str, pos: int) -> tuple[int, list[str]]:
    """Read new complete lines from `path` starting at byte offset `pos`.
    Returns (new_pos, [line, ...]). Returns (pos, []) if nothing new or
    only a partial trailing line. Tail-friendly: never reads past the last
    newline so partial writes aren't surfaced as corrupt lines."""
    try:
        with open(path, "rb") as f:
            f.seek(pos)
            data = f.read()
    except FileNotFoundError:
        return pos, []
    if not data:
        return pos, []
    # Only consume up to the last newline so partial trailing lines wait.
    last_nl = data.rfind(b"\n")
    if last_nl < 0:
        return pos, []
    consumed = data[: last_nl + 1]
    new_pos = pos + len(consumed)
    text = consumed.decode("utf-8", errors="replace")
    lines = [ln for ln in text.split("\n") if ln.strip()]
    return new_pos, lines


def _process_alive(pid: int | None) -> bool:
    """Check whether a pid is still running.

    On Linux/Mac we use os.kill(pid, 0). On Windows os.kill can raise
    SystemError ("WinError 87 — the parameter is incorrect") for stale
    or recycled pids in a way that propagates through bare except
    handlers and breaks uvicorn --reload. Use psutil.pid_exists if
    available, which handles the bad-pid case cleanly. Fall back to
    os.kill otherwise."""
    if pid is None or pid <= 0:
        return False
    try:
        import psutil  # type: ignore
        return psutil.pid_exists(int(pid))
    except ImportError:
        pass
    try:
        os.kill(int(pid), 0)
        return True
    except Exception:
        return False


async def _drain_log(run: LaunchedRun) -> None:
    """Background task: tail the run's log file, surfacing complete lines
    for stdout consumers and parsing progress events for live-mode.

    Survives webapp restart: works whether or not run.proc is set (None
    when we recovered the run from a sidecar after restart). Stops when
    the process exits AND we've drained any final buffered lines."""
    if not run.log_file:
        return
    loop = asyncio.get_event_loop()
    quiet_iters = 0
    while True:
        new_pos, lines = await loop.run_in_executor(
            None, _read_new_lines, run.log_file, run.log_pos,
        )
        if lines:
            run.log_pos = new_pos
            quiet_iters = 0
            for line in lines:
                run.stdout_lines.append(line)
                ev = _try_parse_progress(line, time.time())
                if ev is not None:
                    run.progress_events.append(ev)
        else:
            quiet_iters += 1

        # Termination check: prefer the proc handle when we have it.
        if run.proc is not None:
            rc = run.proc.poll()
            if rc is not None:
                # Drain one more time for any tail bytes after exit.
                final_pos, final_lines = await loop.run_in_executor(
                    None, _read_new_lines, run.log_file, run.log_pos,
                )
                if final_lines:
                    run.log_pos = final_pos
                    for line in final_lines:
                        run.stdout_lines.append(line)
                        ev = _try_parse_progress(line, time.time())
                        if ev is not None:
                            run.progress_events.append(ev)
                run.returncode = rc
                if run.finished_at is None:
                    run.finished_at = time.time()
                break
        else:
            # Recovered orphan: no proc handle. Stop draining when the pid
            # is gone AND the log has been quiet for a few iterations.
            if not _process_alive(run.pid) and quiet_iters > 5:
                run.returncode = 0  # we don't know the actual rc post-orphan
                if run.finished_at is None:
                    # Best-effort: use the log file's mtime when available
                    # (closer to actual completion than time.time() now).
                    try:
                        run.finished_at = Path(run.log_file).stat().st_mtime
                    except (OSError, TypeError):
                        run.finished_at = time.time()
                break

        await asyncio.sleep(0.2)


@app.get("/api/runs/launch")
def list_active_launches() -> JSONResponse:
    """Phase 2.5: list all in-flight (or recently-completed) runs known
    to this server process. Lets the World tab discover runs to follow."""
    out = []
    for run in launched_runs.values():
        is_running = run.proc is not None and run.proc.poll() is None
        latest = run.progress_events[-1] if run.progress_events else None
        # Freeze elapsed_sec for done runs so the live picker stops ticking
        # once a run completes. Belt-and-suspenders: if drain_log somehow
        # missed setting finished_at (task crashed, race on shutdown, etc.)
        # but the run is no longer running, set it lazily here. Use the log
        # file's mtime when available — closer to actual completion than now.
        if not is_running and run.finished_at is None:
            try:
                run.finished_at = Path(run.log_file).stat().st_mtime if run.log_file else time.time()
            except (OSError, TypeError):
                run.finished_at = time.time()
        end_time = run.finished_at if run.finished_at is not None else time.time()
        out.append({
            "run_id": run.run_id,
            "running": is_running,
            "returncode": run.returncode,
            "started_at": run.started_at,
            "elapsed_sec": end_time - run.started_at,
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
    # Belt-and-suspenders: lazy-set finished_at if drain_log missed it.
    # See list_active_launches comment.
    if not is_running and run.finished_at is None:
        try:
            run.finished_at = Path(run.log_file).stat().st_mtime if run.log_file else time.time()
        except (OSError, TypeError):
            run.finished_at = time.time()
    end_time = run.finished_at if run.finished_at is not None else time.time()
    # Surface the launch command list so the frontend can detect feature
    # flags (e.g. --landmarks) for live runs without re-parsing the sidecar.
    return JSONResponse({
        "run_id": run.run_id,
        "running": is_running,
        "returncode": run.returncode,
        "started_at": run.started_at,
        "elapsed_sec": end_time - run.started_at,
        "stdout_line_count": len(run.stdout_lines),
        "tail": run.stdout_lines[-20:],
        "progress_events": [_progress_to_json(p) for p in run.progress_events],
        "out_path": run.out_path,
        "cmd": list(run.cmd),
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


# ─────────────────────────────────────────────────────────────────────────
# Soft-delete (trash) system. Lets users tidy up the runs list without
# losing data. Trashed runs go into RAW_RUNS_DIR/.trash/ with a timestamp
# suffix (so re-trashing a previously-restored file doesn't collide).
# UI surfaces individual + bulk delete, and a Trash view with restore +
# permanent purge.
# ─────────────────────────────────────────────────────────────────────────


def _trash_paths_for(name: str) -> list[Path]:
    """All files associated with the run named `name`: the run JSON, its
    sidecar `.cmd.json` (if any), and any associated logs/control files.
    We move them all together so restore brings everything back."""
    base = RAW_RUNS_DIR / name
    out = [base]
    sidecar = base.with_suffix(".cmd.json")
    if sidecar.exists():
        out.append(sidecar)
    return out


class TrashRequest(BaseModel):
    names: list[str]


@app.post("/api/runs/trash")
def trash_runs(req: TrashRequest) -> JSONResponse:
    """Soft-delete runs by moving them (and their sidecars) into the trash
    directory. Returns counts + per-run statuses."""
    trashed: list[str] = []
    skipped: list[dict[str, str]] = []
    ts = time.strftime("%Y%m%d_%H%M%S")
    for name in req.names:
        if "/" in name or "\\" in name or ".." in name or name.startswith("."):
            skipped.append({"name": name, "reason": "invalid name"})
            continue
        srcs = [p for p in _trash_paths_for(name) if p.exists()]
        if not srcs:
            skipped.append({"name": name, "reason": "not found"})
            continue
        try:
            for src in srcs:
                # Tag the trashed file with the timestamp to avoid collision
                # if the same name is re-trashed later.
                dst = TRASH_DIR / f"{src.name}.{ts}"
                src.rename(dst)
            trashed.append(name)
        except OSError as e:
            skipped.append({"name": name, "reason": str(e)})
    return JSONResponse({
        "trashed": trashed, "skipped": skipped,
        "n_trashed": len(trashed),
    })


@app.get("/api/runs/trash/list")
def list_trashed() -> JSONResponse:
    """List runs currently in the trash (only the *.json runs, not their
    associated sidecars). Each entry shows when it was trashed."""
    out = []
    for f in sorted(TRASH_DIR.glob("*.json.*"), reverse=True):
        # filename pattern: foo.json.YYYYmmdd_HHMMSS
        if ".cmd.json" in f.name:
            continue  # only show the run JSON, not its sidecar
        # Recover the original name + timestamp by splitting on the last "."
        idx = f.name.rfind(".")
        original_name = f.name[:idx]
        ts = f.name[idx + 1:]
        try:
            data = json.loads(f.read_text())
        except Exception:
            data = {}
        out.append({
            "trash_filename": f.name,
            "original_name": original_name,
            "trashed_at": ts,
            "size_bytes": f.stat().st_size,
            "seed": data.get("seed"),
            "n_steps": data.get("n_steps"),
        })
    return JSONResponse({"trashed": out, "count": len(out)})


class RestoreRequest(BaseModel):
    trash_filenames: list[str]


@app.post("/api/runs/trash/restore")
def restore_trashed(req: RestoreRequest) -> JSONResponse:
    """Move runs out of trash back to their original location. If a file
    of the same original name already exists, restore is skipped for that
    item (don't clobber)."""
    restored: list[str] = []
    skipped: list[dict[str, str]] = []
    for trash_name in req.trash_filenames:
        if "/" in trash_name or "\\" in trash_name or ".." in trash_name:
            skipped.append({"name": trash_name, "reason": "invalid name"})
            continue
        src = TRASH_DIR / trash_name
        if not src.exists():
            skipped.append({"name": trash_name, "reason": "not in trash"})
            continue
        # Restore the run JSON and any matching sidecar (look for cmd.json
        # variant with the same timestamp suffix).
        idx = trash_name.rfind(".")
        original_name = trash_name[:idx]
        ts = trash_name[idx + 1:]
        # Find the sidecar with same timestamp, if any
        candidates = [src]
        # Sidecar name pattern: foo.cmd.json.YYYYmmdd_HHMMSS
        if original_name.endswith(".json"):
            sidecar_basename = original_name[:-5] + ".cmd.json"
            sidecar_trash = TRASH_DIR / f"{sidecar_basename}.{ts}"
            if sidecar_trash.exists():
                candidates.append(sidecar_trash)
        try:
            for c in candidates:
                # Strip the trailing timestamp to recover original name
                recovered_name = c.name.rsplit(".", 1)[0]
                dst = RAW_RUNS_DIR / recovered_name
                if dst.exists():
                    skipped.append({
                        "name": trash_name,
                        "reason": f"target {recovered_name} already exists",
                    })
                    continue
                c.rename(dst)
            restored.append(trash_name)
        except OSError as e:
            skipped.append({"name": trash_name, "reason": str(e)})
    return JSONResponse({
        "restored": restored, "skipped": skipped,
        "n_restored": len(restored),
    })


class PurgeRequest(BaseModel):
    trash_filenames: list[str] | None = None  # None = purge all


@app.post("/api/runs/trash/purge")
def purge_trashed(req: PurgeRequest) -> JSONResponse:
    """Permanently delete trashed runs. If trash_filenames is None, empties
    the entire trash. Otherwise deletes only the specified files."""
    targets: list[Path]
    if req.trash_filenames is None:
        targets = list(TRASH_DIR.iterdir())
    else:
        targets = []
        for n in req.trash_filenames:
            if "/" in n or "\\" in n or ".." in n:
                continue
            p = TRASH_DIR / n
            if p.exists():
                targets.append(p)
            # Also pick up matching sidecar
            idx = n.rfind(".")
            original = n[:idx]
            ts = n[idx + 1:]
            if original.endswith(".json"):
                sb = original[:-5] + ".cmd.json"
                sp = TRASH_DIR / f"{sb}.{ts}"
                if sp.exists():
                    targets.append(sp)
    purged = []
    for p in targets:
        try:
            p.unlink()
            purged.append(p.name)
        except OSError:
            pass
    return JSONResponse({"purged": purged, "n_purged": len(purged)})


# A quick "trash all incomplete runs" convenience endpoint. "Incomplete" =
# the JSON has no phase_stats with finalQ data (i.e., the runner died
# before the trial loop finished writing phase_stats).
@app.post("/api/runs/trash/incomplete")
def trash_incomplete() -> JSONResponse:
    """Mass-trash all runs that don't have a complete phase_stats."""
    files = sorted(
        (f for f in RAW_RUNS_DIR.glob("*.json") if not f.name.endswith(".cmd.json")),
    )
    incomplete: list[str] = []
    for f in files:
        try:
            data = json.loads(f.read_text())
        except Exception:
            incomplete.append(f.name)
            continue
        ps = data.get("phase_stats") or []
        if not ps:
            incomplete.append(f.name)
            continue
        last = ps[-1]
        if last.get("final_quarter_mean_distance") is None and last.get("finalQ") is None:
            incomplete.append(f.name)
    if not incomplete:
        return JSONResponse({"trashed": [], "skipped": [], "n_trashed": 0})
    return trash_runs(TrashRequest(names=incomplete))


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
    files = sorted(
        (f for f in RAW_RUNS_DIR.glob("*.json") if not f.name.endswith(".cmd.json")),
        reverse=True,
    )
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
