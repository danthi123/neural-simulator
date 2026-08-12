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
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
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


# ─── Default to the cupy (GPU) backend when a CUDA GPU is present ───────────
# A bare `uvicorn webapp.server:app` would otherwise auto-detect a backend per
# `sim.backend` defaults, but the chat path's qwen renderer + any on-bridge
# spiking work is ~20× slower on the numpy-CPU backend. So: if a CUDA GPU is
# actually detectable, `setdefault` SIM_BACKEND=cupy BEFORE any sim/cupy import
# happens (all of those are lazy, inside endpoint handlers + _build_chat_brain,
# so this module-load-time set wins the first `get_backend()` resolution).
# Guarded both ways: `setdefault` never clobbers an explicit SIM_BACKEND
# (so `SIM_BACKEND=numpy uvicorn ...` stays numpy for the CPU-portable path),
# and we only set cupy when a GPU is genuinely found (else leave the env unset
# so `sim.backend` auto-detects → numpy on a GPU-less host). Detection is
# defensive: a probe failure leaves the env untouched (numpy fallback).
def _cuda_gpu_present() -> bool:
    """True iff a CUDA GPU is actually usable. Tries cupy's runtime first
    (cheapest, matches the production backend), then torch as a fallback.
    Any import/probe error → False (conservative: never force cupy on a host
    where the GPU can't be confirmed)."""
    try:
        import cupy  # noqa: PLC0415
        if cupy.cuda.runtime.getDeviceCount() > 0:
            return True
    except Exception:
        pass
    try:
        import torch  # noqa: PLC0415
        if torch.cuda.is_available():
            return True
    except Exception:
        pass
    return False


if "SIM_BACKEND" not in os.environ and _cuda_gpu_present():
    os.environ.setdefault("SIM_BACKEND", "cupy")
    print("[webapp] CUDA GPU detected -> defaulting SIM_BACKEND=cupy "
          "(set SIM_BACKEND=numpy to override for the CPU path)", flush=True)

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
    # Per-step action + reward (added 2026-04-29 for live-mode HUD).
    # action is the index 0-3 (NESW); -1 means "not parsed from this line"
    # (older runners or progress lines that predate the action= field).
    action: int = -1
    reward: float = 0.0


@dataclass
class ActivityFrame:
    """A parsed [ACTIVITY] {json} line from a runner's stdout — the live
    per-region brain-activity frame (frontend-revamp Phase 1, 2026-06-08).

    Tiny by construction (~30 region floats + ~30 flux floats), emitted by the
    runner on a throttle (every N steps). The server ring-buffers the most
    recent frames and re-broadcasts the LATEST one per WS client (latest-wins
    coalescing), so a slow browser never backs up the stream — and the sim,
    which only writes a fire-and-forget stdout line, is never in that loop."""
    t: float                       # sim time (ms) for this frame
    regions: dict[str, float]      # region_name -> mean firing fraction [0,1]
    flux: dict[str, float]         # pathway_name -> flux [0,1]
    timestamp: float               # server wall-clock receive time
    step: int = -1                 # runner step at emit (if present)
    seq: int = 0                   # monotonic per-run sequence (for coalescing)


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
    # Live brain-activity frames (frontend-revamp Phase 1, 2026-06-08). A bounded
    # ring buffer — 600 frames = 60 s at 10 Hz — so the activity stream never
    # grows without bound even on long runs. `activity_seq` is a monotonic
    # counter used by the WS handler for latest-wins coalescing (each client
    # only ever sends the freshest frame and tracks the seq it last sent).
    activity_frames: deque[ActivityFrame] = field(
        default_factory=lambda: deque(maxlen=600))
    activity_seq: int = 0


launched_runs: dict[str, LaunchedRun] = {}


# Match runner progress lines like:
#   [g11 seed=42] step 800/1800  pos=(6,1)  goal=(1,6)  recent_dist=7.58  actions=...
# The actions tail is optional and not parsed.
_PROGRESS_RE = re.compile(
    r"step\s+(\d+)/(\d+)\s+pos=\((-?\d+),(-?\d+)\)\s+goal=\((-?\d+),(-?\d+)\)"
    r"\s+recent_dist=([\d.]+)"
)
# Optional per-step action + reward fields, appended after recent_dist.
# Format: " action=N reward=+1.00" (action letter NESW, reward signed float).
# Both fields are optional for backward compat with older runner versions.
_PROGRESS_ACTION_RE = re.compile(r"action=([NESW?])\s+reward=([-+]?[\d.]+)")
_ACTION_LETTER_TO_IDX = {"N": 0, "E": 1, "S": 2, "W": 3}

# Live brain-activity channel (frontend-revamp Phase 1). Sibling of
# _PROGRESS_RE — same wire format (one line, JSON after a fixed prefix), new
# prefix. Mirrors sim/progress.py ACTIVITY_LINE_RE; kept inline here so the
# server has no import-time dependency on sim.progress (matching how
# _PROGRESS_RE is defined locally).
_ACTIVITY_RE = re.compile(r"\[ACTIVITY\]\s+(\{.*\})")


def _try_parse_activity(line: str, now: float, seq: int) -> ActivityFrame | None:
    """Parse one log line as an [ACTIVITY] frame, or return None.

    `seq` is the run's monotonic counter at the time of parse — stamped onto
    the frame so WS clients can coalesce to the latest (latest-wins)."""
    m = _ACTIVITY_RE.search(line)
    if not m:
        return None
    try:
        payload = json.loads(m.group(1))
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    regions = payload.get("regions")
    if not isinstance(regions, dict):
        return None
    flux = payload.get("flux")
    if not isinstance(flux, dict):
        flux = {}
    try:
        return ActivityFrame(
            t=float(payload.get("t", 0.0)),
            regions={str(k): float(v) for k, v in regions.items()},
            flux={str(k): float(v) for k, v in flux.items()},
            timestamp=now,
            step=int(payload.get("step", -1)),
            seq=seq,
        )
    except (TypeError, ValueError):
        return None


def _try_parse_progress(line: str, now: float) -> ProgressEvent | None:
    m = _PROGRESS_RE.search(line)
    if not m:
        return None
    step, total, x, y, gx, gy, rd = m.groups()
    try:
        ev = ProgressEvent(
            step=int(step), total=int(total),
            pos=(int(x), int(y)),
            goal=(int(gx), int(gy)),
            recent_dist=float(rd),
            timestamp=now,
        )
    except (TypeError, ValueError):
        return None
    # Try to parse the optional action + reward fields. Older runners
    # don't emit these — leave the dataclass defaults (action=-1, reward=0.0).
    am = _PROGRESS_ACTION_RE.search(line)
    if am:
        action_letter, reward_str = am.groups()
        ev.action = _ACTION_LETTER_TO_IDX.get(action_letter, -1)
        try:
            ev.reward = float(reward_str)
        except ValueError:
            pass
    return ev


# ─────────────────────────────────────────────────────────────────────────
# Findings (markdown files)
# ─────────────────────────────────────────────────────────────────────────


_DATE_PREFIX_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})-")


@app.get("/api/findings")
def list_findings() -> JSONResponse:
    """List findings markdowns. Sorted by parsed YYYY-MM-DD date prefix
    descending (newest first). Findings without a date prefix
    (INDEX.md, PHASE_B_QUICK_REFERENCE.md, etc.) are flagged as
    `is_reference=true` and sorted to the BOTTOM of the list — they're
    not chronological session findings, they're reference docs.

    Each entry includes:
      * `date`: parsed ISO date or null
      * `is_reference`: true if no date prefix found
      * `is_recent`: true if mtime is within last 3 days (UI highlight)
    """
    files = list(FINDINGS_DIR.glob("*.md"))
    now = time.time()
    out = []
    for f in files:
        m = _DATE_PREFIX_RE.match(f.name)
        date_iso = None
        if m:
            try:
                date_iso = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
            except Exception:
                pass
        mtime = f.stat().st_mtime
        out.append({
            "name": f.name,
            "size_bytes": f.stat().st_size,
            "modified_unix": mtime,
            "date": date_iso,
            "is_reference": date_iso is None,
            "is_recent": (now - mtime) < (3 * 24 * 3600),
        })
    # Sort: dated entries first (most recent date first), undated entries
    # last (alphabetically). Tiebreaker for same-date: mtime descending so
    # the most-recently-edited file wins.
    def sort_key(e):
        if e["date"]:
            # Negative date_str so sorted ASC gives newest first;
            # but Python tuples sort lexicographically, so just invert.
            # Use is_reference=False (0) before True (1) so undated go last.
            return (0, _negate_iso_date(e["date"]), -e["modified_unix"])
        else:
            return (1, e["name"].lower())
    out.sort(key=sort_key)
    return JSONResponse({"findings": out, "count": len(out)})


def _negate_iso_date(iso: str) -> str:
    """Return a string that sorts ASC inversely to the ISO date — so
    "2026-05-03" sorts before "2026-05-02". Used to put newest first
    while keeping the sort key a simple ascending comparison.
    Implementation: subtract each digit from 9.
    """
    return "".join(str(9 - int(c)) if c.isdigit() else c for c in iso)


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
    # R1 (2026-06-20) -- side-by-side SHOWCASE of the validated robust multi-cue Competition-Model parser:
    # it comprehends IMPERFECT / non-canonical English (object-fronted, scrambled, dropped function words)
    # where the default ORDER-ONLY parser inverts the roles, with the no-confab moat held in BOTH. Scripted +
    # deterministic; the runner takes only --seed + --out (both injected by the launcher), so no extra args.
    # See research/findings/2026-06-20-R1-imperfect-english-demo.md.
    "imperfect_english_demo": [],
    "flagship": [
        "--moving-goal",
        "--enable-place-goal-readout", "--learned-perception", "--enable-dlpfc-wm",
        "--beacon-perception", "--beacon-replaces-goal",
        "--cue-reflex", "--cue-reflex-replaces-heuristic",
        "--enable-landmark-sensor", "--landmarks-replace-place",
        "--sensed-reward",
        "--adaptive-da", "--adaptive-da-ema-decay-negative", "0.7",
        "--curriculum", "--curriculum-warmup-steps", "600",
        "--n-steps", "1800",
    ],
    # Biology-grounded minimal flagship (R-pass + Cluster B + Cluster A + E).
    # 2026-04-29 deterministic single-goal n=6 result: 3.31 +/- 0.74, beats
    # documented full flagship 4.08 +/- 0.49 by 19%. No --hippocampus,
    # no --learned-perception, no --sensed-reward, no curriculum.
    # See research/findings/2026-04-29-overnight-FINAL.md.
    "biology_AE": [
        "--moving-goal",
        "--enable-msn-lateral-inhibition",
        "--enable-d1-d2-asymmetry", "--enable-striatal-pv-fsi",
        "--enable-cluster-a-closed-loop", "--enable-cluster-e-topography",
        "--n-steps", "1800",
    ],
    # Same but with --goal-schedule multi for the harder 4-phase task.
    "biology_AE_multi": [
        "--moving-goal", "--goal-schedule", "multi",
        "--enable-msn-lateral-inhibition",
        "--enable-d1-d2-asymmetry", "--enable-striatal-pv-fsi",
        "--enable-cluster-a-closed-loop", "--enable-cluster-e-topography",
        "--n-steps", "1800",
    ],
    # Biology-grounded + --sensed-reward (gradient-based reward instead of
    # distance-based heuristic reward). Tests how low we can go with one cheat.
    "biology_AE_sensed": [
        "--moving-goal",
        "--enable-msn-lateral-inhibition",
        "--enable-d1-d2-asymmetry", "--enable-striatal-pv-fsi",
        "--enable-cluster-a-closed-loop", "--enable-cluster-e-topography",
        "--sensed-reward",
        "--n-steps", "1800",
    ],
    # Plain biology baseline (R-pass + Cluster B only) for fair tier-4 comparison
    # against biology_AE. Single-goal default schedule.
    "biology_baseline": [
        "--moving-goal",
        "--enable-msn-lateral-inhibition",
        "--enable-d1-d2-asymmetry", "--enable-striatal-pv-fsi",
        "--n-steps", "1800",
    ],
    # Interactive presets — webapp wires --interactive-control-file +
    # per-step progress prints + a small throttle so the dashboard's
    # live mode can animate per-step (not jumping every 100). World-tab
    # Live mode lets the user click in the grid to teleport the goal,
    # pause/resume, inject rewards.
    "interactive_flagship": [
        "--moving-goal",
        "--enable-place-goal-readout", "--learned-perception", "--enable-dlpfc-wm",
        "--beacon-perception", "--beacon-replaces-goal",
        "--cue-reflex", "--cue-reflex-replaces-heuristic",
        "--enable-landmark-sensor", "--landmarks-replace-place",
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
        "--enable-place-goal-readout", "--learned-perception", "--enable-dlpfc-wm",
        "--beacon-perception", "--beacon-replaces-goal",
        "--cue-reflex", "--cue-reflex-replaces-heuristic",
        "--enable-landmark-sensor", "--landmarks-replace-place",
        "--sensed-reward",
        "--enable-corticostriatal-cross", "--bg-cross-thaw-step", "1200", "--bg-cross-phase3-gain", "0.5",
        "--adaptive-da", "--adaptive-da-ema-decay-negative", "0.7",
        "--curriculum", "--curriculum-warmup-steps", "600",
        "--n-steps", "1800",
    ],
    "perception_only": [
        "--moving-goal",
        "--enable-place-goal-readout", "--learned-perception", "--enable-dlpfc-wm",
        "--beacon-perception", "--beacon-replaces-goal",
        "--cue-reflex", "--cue-reflex-replaces-heuristic",
        "--enable-landmark-sensor", "--landmarks-replace-place",
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
    # ─── 2026-05-01 NEW PRESETS ───────────────────────────────────────
    # G v2.5 flagship: NMDA on PFC + cortex_X + motor_X gives perfect
    # 2.00 ± 0.00 on cheat-5 multi-goal det at 8x8/16x16/24x24 (with
    # heuristic). 60% improvement over A+E single-pool baseline (5.02).
    # See research/findings/2026-05-01-cluster-g-nmda-breakthrough.md.
    "flagship_g_v25": [
        "--moving-goal", "--goal-schedule", "multi", "--deterministic",
        "--enable-msn-lateral-inhibition",
        "--enable-d1-d2-asymmetry", "--enable-striatal-pv-fsi",
        "--enable-cluster-a-closed-loop", "--enable-cluster-e-topography",
        "--heuristic-single-pool",
        "--enable-dlpfc-wm", "--enable-pfc-nmda",
        "--n-steps", "1800",
    ],
    # K v2 visual cortex (Cluster K v2): perception-only at 16x16,
    # 2.87 ± 0.19 (n=6) — biology-grounded, NO heuristic. 5.4× better
    # than perception arc (15.47). Closes 4 of 5 original cheats.
    # See research/findings/2026-05-01-cluster-k-v2-breakthrough.md.
    "flagship_k_v2_visual": [
        "--moving-goal", "--goal-schedule", "multi", "--deterministic",
        "--enable-msn-lateral-inhibition",
        "--enable-d1-d2-asymmetry", "--enable-striatal-pv-fsi",
        "--enable-cluster-a-closed-loop", "--enable-cluster-e-topography",
        "--enable-dlpfc-wm", "--enable-pfc-nmda",
        "--enable-visual-cortex", "--visual-cortex-action-warmup-steps", "600",
        "--grid-size", "16",
        "--n-steps", "1800",
    ],
    # K v2 visual cortex at 24x24: confirms grid invariance.
    # 2.87 ± 0.22 (n=3) on 9× larger grid.
    "flagship_k_v2_24x24": [
        "--moving-goal", "--goal-schedule", "multi", "--deterministic",
        "--enable-msn-lateral-inhibition",
        "--enable-d1-d2-asymmetry", "--enable-striatal-pv-fsi",
        "--enable-cluster-a-closed-loop", "--enable-cluster-e-topography",
        "--enable-dlpfc-wm", "--enable-pfc-nmda",
        "--enable-visual-cortex", "--visual-cortex-action-warmup-steps", "600",
        "--grid-size", "24",
        "--n-steps", "1800",
    ],
    # Interactive K v2 visual cortex — for live-mode dashboard demos.
    "interactive_k_v2_visual": [
        "--moving-goal", "--goal-schedule", "multi", "--deterministic",
        "--enable-msn-lateral-inhibition",
        "--enable-d1-d2-asymmetry", "--enable-striatal-pv-fsi",
        "--enable-cluster-a-closed-loop", "--enable-cluster-e-topography",
        "--enable-dlpfc-wm", "--enable-pfc-nmda",
        "--enable-visual-cortex", "--visual-cortex-action-warmup-steps", "600",
        "--grid-size", "16",
        "--n-steps", "3600",
        "--progress-print-interval", "1",
        "--trial-sleep-ms", "30",
    ],
    # Rank 2 generalization (frontend-revamp live-monitor + 6-seed verdict,
    # 2026-06-08): biologized flagship + an innate superior-colliculus orienting
    # reflex that TEACHES a learned-from-vision (dx,dy)->action map, then weans
    # off, tested on 3 NEW non-corner goals (the generalize2 schedule). 6000
    # steps. Used for the live-monitored Rank 2 extension (seeds 100/101/102).
    "rank2_generalize2": [
        "--moving-goal", "--goal-schedule", "generalize2", "--deterministic",
        "--enable-msn-lateral-inhibition",
        "--enable-d1-d2-asymmetry", "--enable-striatal-pv-fsi",
        "--enable-cluster-a-closed-loop", "--enable-cluster-e-topography",
        "--enable-dlpfc-wm", "--enable-pfc-nmda",
        "--enable-visual-cortex", "--visual-cortex-action-warmup-steps", "600",
        "--genuine-thal-disinhibition", "--genuine-gpi-tonic-pa", "1300",
        "--genuine-thal-tonic-pa", "750",
        "--readout-source", "spiking_wta", "--urgency-max-pa", "180",
        "--heuristic-strength", "0", "--sc-orienting-reflex",
        "--sc-reflex-wean-start", "2000", "--sc-reflex-wean-steps", "1000",
        "--learned-perception", "--learned-perception-from-vision",
        "--grid-size", "8", "--n-steps", "6000",
    ],
    # ─── 2026-05-02 Text I/O presets (use research.runners.text_eval_embodied) ──
    # text_io_v2_baseline — current best W→A breakthrough.
    # 28.5% W→A across 6 seeds (n=600, p=0.027 vs chance). Hebbian off,
    # stdp_w_max=5, readout init=0.5. See:
    # research/findings/2026-05-02-text-io-formal-writeup.md.
    "text_io_v2_baseline": [
        "--n-episodes", "100", "--steps-per-episode", "30",
        "--stim-steps-per-step", "200", "--reset-steps", "100",
    ],
    # text_io_v2_smoke — 10-episode smoke test for verifying the runner
    # without committing to ~50 minutes per seed.
    "text_io_v2_smoke": [
        "--n-episodes", "10", "--steps-per-episode", "30",
        "--stim-steps-per-step", "200", "--reset-steps", "100",
    ],
    # text_io_distributed_motor_pop — Pulvermüller G.20 architecture.
    # 8 motor sub-pools at 45° intervals with cosine-tuned thal pathways
    # and population-vector decoding. Tests whether labeled-line motor
    # pools were the bottleneck for the 28.5% W→A ceiling. Result so far
    # (seed 42): I→W 22%, W→A 28% — within v2 range.
    "text_io_distributed_motor_pop": [
        "--n-episodes", "100", "--steps-per-episode", "30",
        "--stim-steps-per-step", "200", "--reset-steps", "100",
        "--enable-distributed-motor-pop",
        "--n-motor-pop-per-subpool", "5",
    ],
    # text_io_motor_cross_coupling — 90°-adjacency cross-coupling tested
    # 2026-05-02. Result: 29% I→W, 22% W→A. NEGATIVE on W→A direction.
    # Kept as a documented baseline.
    "text_io_motor_cross_coupling": [
        "--n-episodes", "100", "--steps-per-episode", "30",
        "--stim-steps-per-step", "200", "--reset-steps", "100",
        "--enable-motor-cross-coupling",
        "--motor-cross-coupling-weight", "0.5",
        "--motor-cross-coupling-density", "0.3",
    ],
    # ─── 2026-05-06/07: Phase 1.3, 1.4, 1.5, Tier 2.3, chat demos ────
    # Phase 1.4 catastrophic forgetting eval. Validated 6-seed
    # (5/6 PASS, mean 103% retention) -- BRANCH A.
    # See research/findings/2026-05-07-Phase-1.4-v3-6seed-FINAL.md.
    "phase_1_4_forgetting": [
        "--phase-a-events", "200", "--phase-b-events", "200",
        "--n-eval-per-word", "25",
    ],
    # Phase 1.3 hippocampus consolidation (McClelland/Buzsaki CLS).
    # Validated 3/3 PASS, mean 96% hippo-OFF retention.
    # See research/findings/2026-05-07-Phase-1.3-3seed-CONFIRMED.md.
    "phase_1_3_consolidation": [
        "--n-awake-events-per-word", "100", "--n-sleep-swr-events", "100",
        "--consolidation-interval", "4", "--n-test-per-word", "25",
    ],
    # Phase 1.5 unified continual-learning eval suite. 4 benchmarks
    # (sequential_expansion + retention_over_time + interference +
    # long_tail). Smoke at default Tier 1 arch: 2/4 PASS, aggregate 0.62.
    # interference + long_tail need 8-word vocab capacity (per
    # 2026-05-08 capacity hypothesis findings) -- Tier 1 default arch
    # (2048/500/60) overflows with 8 words. Use phase_1_5_unified_scaled
    # for proper validation.
    "phase_1_5_unified": [
        "--benchmarks", "sequential_expansion", "retention_over_time",
        "interference", "long_tail",
        "--events-per-word", "200", "--n-eval-per-word", "25",
        "--silence-steps", "5000",
    ],
    # Phase 1.5 at Tier 2.1 v4 scale-up arch (4096/1000/120). Per 2026-05-08
    # capacity hypothesis: 8-word vocab benchmarks (interference, long_tail)
    # need scale-up arch to pass. Default arch only supports 4-word cleanly.
    # Wall clock ~160 min/seed (4 benchmarks at scaled arch sequentially).
    # Master plan named milestone: Phase 1.5 unified eval suite.
    "phase_1_5_unified_scaled": [
        "--benchmarks", "sequential_expansion", "retention_over_time",
        "interference", "long_tail",
        "--events-per-word", "200", "--n-eval-per-word", "25",
        "--silence-steps", "5000",
        "--n-lang-input", "4096",
        "--n-motor-per-action", "1000",
        "--n-motor-fs-per-action", "120",
    ],
    # 2026-05-09 follow-up: same suite at Tier 2.1's VALIDATED training
    # dose (400 events/word, matching the BREAKTHROUGH config) instead of
    # the 200 that was failing on interference (0.34 score at seed 42).
    # Hypothesis (research/findings/2026-05-09-Phase-1.5-interference-
    # undertraining-hypothesis.md): the interference benchmark needs at
    # least the validated event count to lift above 0.5. Wall clock
    # ~280-320 min/seed (2x the events doubles training time roughly).
    "phase_1_5_unified_scaled_v400": [
        "--benchmarks", "sequential_expansion", "retention_over_time",
        "interference", "long_tail",
        "--events-per-word", "400", "--n-eval-per-word", "25",
        "--silence-steps", "5000",
        "--n-lang-input", "4096",
        "--n-motor-per-action", "1000",
        "--n-motor-fs-per-action", "120",
    ],
    # Even more targeted: just the interference benchmark at 400 events/
    # word, for fast hypothesis confirmation (~70 min single seed vs the
    # full ~5 hr suite).
    "phase_1_5_interference_only_v400": [
        "--benchmarks", "interference",
        "--events-per-word", "400", "--n-eval-per-word", "25",
        "--silence-steps", "5000",
        "--n-lang-input", "4096",
        "--n-motor-per-action", "1000",
        "--n-motor-fs-per-action", "120",
    ],
    # 2026-05-09 (post-v400-REFUTED): the v400 hypothesis was tested
    # and produced essentially no change (0.340 -> 0.345). Failure is
    # architectural, not dose-bound. Next hypothesis: same n_motor=2000
    # scale-up that lifted 12-word vocab from 2/3 PARTIAL to 3/3 GO at
    # the capacity hypothesis arc (commit ffbac1c). Tests whether
    # interleaved 8-word training has a per-word capacity bottleneck
    # at the standard 1000 motor neurons. ~80-100 min single seed at
    # 2x neurons (slower per step due to bigger motor matrices).
    "phase_1_5_interference_only_n_motor_2000": [
        "--benchmarks", "interference",
        "--events-per-word", "400", "--n-eval-per-word", "25",
        "--silence-steps", "5000",
        "--n-lang-input", "4096",
        "--n-motor-per-action", "2000",
        "--n-motor-fs-per-action", "240",
    ],
    # 2026-05-09 follow-up: relaxed long_tail few-shot dose. Per seed
    # 42 FINAL finding (long_tail at 17%, rare-word binding fails at
    # 10 events): test whether rare-word binding works at 50 events
    # (rare-ratio=4, ~13s of speech) AND with motor teacher matching
    # the standard 1500 pA. Both knobs may be needed.
    "phase_1_5_long_tail_relaxed": [
        "--benchmarks", "long_tail",
        "--events-per-word", "200", "--n-eval-per-word", "25",
        "--long-tail-rare-ratio", "4",
        "--long-tail-rare-teacher-pA", "1500",
        "--silence-steps", "5000",
        "--n-lang-input", "4096",
        "--n-motor-per-action", "1000",
        "--n-motor-fs-per-action", "120",
    ],
    # 2026-05-09 Outcome-A combined preset (per
    # docs/plans/2026-05-09-Phase-1.5-decision-tree.md). Pre-staged for
    # the case where BOTH hypothesis tests pass:
    #   - n_motor=2000 fixes interference (capacity rule extends)
    #   - long_tail rare-ratio=4 + teacher=1500 fixes few-shot
    # Combines all 4 benchmarks at the validated dose (events_per_word=400)
    # AND the scaled-up arch (n_motor=2000) AND the relaxed long_tail
    # config. If single-seed smoke at this preset clears the master
    # plan threshold (>=0.70 mean), 3-seed multi-seed becomes the
    # Phase 1.5 milestone validation run. Wall clock estimate ~5-6 hr
    # single seed (4 benchmarks × ~1-1.5 hr each at 2x neurons + 2x
    # events).
    "phase_1_5_unified_combined": [
        "--benchmarks", "sequential_expansion", "retention_over_time",
        "interference", "long_tail",
        "--events-per-word", "400", "--n-eval-per-word", "25",
        "--silence-steps", "5000",
        "--n-lang-input", "4096",
        "--n-motor-per-action", "2000",
        "--n-motor-fs-per-action", "240",
        "--long-tail-rare-ratio", "4",
        "--long-tail-rare-teacher-pA", "1500",
    ],
    # Tier 2.3 PFC verb pool + compositional 2-word phrase trainer.
    # Architecture-limited at 41% phrase composition (sweep confirmed
    # action_gate is inert at default config).
    "tier_2_3_phrases": [
        "--n-phrase-events", "200", "--n-direction-only-events", "100",
        "--n-verb-only-events", "30", "--n-test-per-direction", "25",
    ],
    # Chat demo: 4-direction Tier 1 with chat-transcript output.
    # ~6 min single seed. Accuracy ~33-45%/seed (Phase 1.4 baseline).
    "chat_demo": [
        "--train-events", "200",
    ],
    # Chat-transcript Phase 1.4 BRANCH A continual learning demo.
    # Trains primaries + synonyms; tests retention.
    "chat_continual_demo": [
        "--train-events", "200", "--n-test-per-word", "10",
    ],
    # Tier 2.1 8-word synonym chat demo. Both "north" and "up" map to
    # motor_N. Built on Tier 2.1 BREAKTHROUGH 2026-05-06 (W->A 5/6 +
    # A->W 6/6 aligned, A->W mean 63.7%). Uses v4 scale-up arch
    # (n_lang=4096, n_motor=1000). ~15-20 min single seed RTX 3090
    # (Tier 2.1 v4 is significantly heavier than Tier 1).
    # See research/findings/2026-05-06-Tier2.1-BREAKTHROUGH-synonym-binding-via-scale.md.
    "chat_synonym_demo": [
        "--train-events", "400",
    ],
    # Track 3 online vocab learning demo (2026-05-09). Trains Tier 1
    # primaries, runs baseline accuracy test, learns 2 NEW words via
    # embodied-Hebbian co-firing (chat_repl.learn_word_pairing), tests
    # the new bindings, then re-tests primaries to check for catastrophic
    # forgetting. Verdict: GO if binding_rate >= 50% AND retention >= 80%.
    # Wall clock ~9-10 min single seed (6 min train + 2 learns of 50
    # events + 4-direction × 2-round tests + 2 binding tests).
    # See chat_repl.py learn_word_pairing for the learning primitive.
    "chat_learn_demo": [
        "--train-events", "200",
        "--learn-events", "50",
        "--new-words", "ahead:N,back:S",
    ],
    # Track 3 generative decoder demo (2026-05-09). Inverse of chat_demo:
    # tests A→W direction via :speak / generative_inference. Trains
    # Tier 1 bridge, regression-checks W→A as baseline, then drives
    # motor_<action> for each of NESW + reads language_output to decode
    # to a vocab word via cosine similarity. Validates the layer 4
    # primitive shipped in commit a675fa1. Verdict: GO if A→W >= 50%
    # AND W→A regression intact (>= 25% chance baseline). Tier 1
    # BREAKTHROUGH validated A→W mean 45-63% (6/6 aligned). Wall clock
    # ~10 min single seed.
    "chat_speak_demo": [
        "--train-events", "200",
    ],
    # Track 3 layer 4 :speak synonym variant (Tier 2.1 8-word A→W).
    # Trains the Tier 2.1 v4 scale-up arch (n_lang_input=4096,
    # n_motor=1000, n_motor_fs=120) on the 8-word synonym vocab,
    # then exercises generative_inference for each of N/E/S/W
    # ranking against ALL 8 words. Headline: any-synonym A→W >= 50%
    # (PASS if motor_X drives produces ANY of {primary, synonym}).
    # Validates the production-side analog of Tier 2.1 v4's W→A
    # reception (which was 5/6 aligned at A→W mean 63.7% in the
    # Tier 2.1 BREAKTHROUGH paper). Wall clock ~10-13 min single
    # seed (Tier 2.1 v4 is bigger than Tier 1 so a touch slower).
    "chat_speak_synonym_demo": [
        "--train-events", "400",
        "--n-lang-input", "4096",
        "--n-motor-per-action", "1000",
        "--n-motor-fs-per-action", "120",
    ],
    # Phase 1.3 + Tier 2.1 combined consolidation test. Trains synonym
    # vocab with hippocampus, alternates awake/sleep, then tests
    # whether cortex retains both primary AND synonym words after
    # hippocampus lesion. Validates CLS theory at synonym scale.
    # WALL-CLOCK CORRECTED 2026-05-07:
    #   --smoke:  ~21 min/seed (12 chunks, 50+50 events)
    #   --medium: ~80 min/seed (50 chunks, 200+100 events)
    #   default:  ~6.5 HOURS/seed (100 chunks, 400+200 events) -- only
    #             use for overnight or multi-day runs
    # Design: docs/plans/2026-05-07-Phase1.3-Tier2.1-combined-design.md
    "consolidation_synonym": [
        "--n-awake-events-per-word", "400",
        "--n-sleep-swr-events", "200",
        "--consolidation-interval", "4",
        "--n-test-per-word", "25",
    ],
    # Smoke mode: ~21 min/seed for runner validation.
    "consolidation_synonym_smoke": [
        "--smoke",
        "--n-test-per-word", "10",
    ],
    # Medium mode: ~80 min/seed -- feasible for 3-seed multi-seed in
    # ~4 hrs vs default's ~19 hrs. Recommended for first-pass multi-seed
    # validation.
    "consolidation_synonym_medium": [
        "--medium",
        "--n-test-per-word", "20",
    ],
    # Anti-cheat variant of medium: also 10x stronger hippo silencing +
    # zero ca1->cortex pathway weights at eval time. Tests whether the
    # 2026-05-08 "synonym retention >100%" finding is real cortex retention
    # or eval-time noise artifact from imperfect hippo silencing.
    # Same wall-clock as medium (~80 min/seed); training identical, only
    # eval differs. CONFIRMED hypothesis B 2026-05-08: identical to
    # non-strict result, cortex truly retains.
    "consolidation_synonym_medium_strict": [
        "--medium",
        "--strict-silence",
        "--n-test-per-word", "20",
    ],
    # 16-word vocab extension at scaled arch (n_motor=2000). Adds Unicode
    # arrows (↑→↓←) as 4th synonym per action. Tests whether the 2026-05-08
    # capacity hypothesis (bigger motor pools enable more sub-pops per
    # action) extends to 4 sub-pops per motor_X. Per master plan section
    # "Larger Tier 2.1 vocab (16-30 words)". Wall clock similar to
    # 12word_scaled (~3.5 hr/seed at scaled arch with 1600 vocab events).
    "consolidation_synonym_16word_scaled_medium": [
        "--medium",
        "--vocab-size", "16",
        "--n-motor-per-action", "2000",
        "--n-motor-fs-per-action", "240",
        "--n-test-per-word", "20",
    ],
    # Cheap 16-word smoke (~30-40 min/seed). Same scaled arch as the
    # medium variant but with --smoke chunking (50+50 events, 12
    # chunks). Use as a fast capacity-rule probe to predict whether the
    # full 16word_scaled_medium will pass before committing 3.5 hrs.
    # If smoke shows reasonable retention (>= 50% primary), proceed to
    # medium; if smoke fails outright, capacity boundary is at 4
    # sub-pops/motor_X with current scaled arch and we need to scale
    # further.
    "consolidation_synonym_16word_scaled_smoke": [
        "--smoke",
        "--vocab-size", "16",
        "--n-motor-per-action", "2000",
        "--n-motor-fs-per-action", "240",
        "--n-test-per-word", "10",
    ],
    # 2026-05-10: find-the-ceiling vocab tiers per user directive
    # ("start very high on the scale to test for failure"). Each is a
    # smoke run; if PASS at smoke, we know the architecture handles that
    # tier and can run medium for actual validation. If OOM, we know the
    # ceiling on 24 GB 3090.
    #
    # Estimated VRAM @ smoke (extrapolated from 16-word smoke ~7 GB):
    #   24-word @ n_motor=2000: ~8 GB    (likely PASS)
    #   32-word @ n_motor=3000: ~12 GB   (likely PASS)
    #   48-word @ n_motor=4000: ~17 GB   (likely PASS, tight)
    #   64-word @ n_motor=6000: ~28 GB   (likely OOM on 24 GB)
    "consolidation_synonym_24word_scaled_smoke": [
        "--smoke",
        "--vocab-size", "24",
        "--n-motor-per-action", "2000",
        "--n-motor-fs-per-action", "240",
        "--n-test-per-word", "10",
    ],
    "consolidation_synonym_32word_scaled_smoke": [
        "--smoke",
        "--vocab-size", "32",
        "--n-motor-per-action", "3000",
        "--n-motor-fs-per-action", "360",
        "--n-test-per-word", "10",
    ],
    "consolidation_synonym_48word_scaled_smoke": [
        "--smoke",
        "--vocab-size", "48",
        "--n-motor-per-action", "4000",
        "--n-motor-fs-per-action", "480",
        "--n-test-per-word", "10",
    ],
    "consolidation_synonym_64word_scaled_smoke": [
        "--smoke",
        "--vocab-size", "64",
        "--n-motor-per-action", "6000",
        "--n-motor-fs-per-action", "720",
        "--n-test-per-word", "10",
    ],
    # 2026-05-10 (continued): if 64-word fits, push higher to find true
    # encoding-wall ceiling. 96+ uses numbered variants (north_05, ...)
    # added to base 64-word vocab. Architecture stays at n_motor=6000
    # (validated to fit at 64-word at 16 GB VRAM); the failure mode at
    # higher vocab will be lang_input encoding collision (10% sparse
    # over 4096 = 410 active per word; 96 words × 410 = 39K vs 4096
    # capacity = 10× overlap), not VRAM OOM.
    "consolidation_synonym_96word_scaled_smoke": [
        "--smoke",
        "--vocab-size", "96",
        "--n-motor-per-action", "6000",
        "--n-motor-fs-per-action", "720",
        "--n-test-per-word", "5",
    ],
    "consolidation_synonym_128word_scaled_smoke": [
        "--smoke",
        "--vocab-size", "128",
        "--n-motor-per-action", "6000",
        "--n-motor-fs-per-action", "720",
        "--n-test-per-word", "5",
    ],
    "consolidation_synonym_256word_scaled_smoke": [
        "--smoke",
        "--vocab-size", "256",
        "--n-motor-per-action", "6000",
        "--n-motor-fs-per-action", "720",
        "--n-test-per-word", "3",
    ],
    # 2026-05-10: axis-decoupling experiments — per user observation that
    # the find-the-ceiling test scaled motor only, leaving encoding
    # (n_lang_input=4096) potentially the actual bottleneck. Test the
    # other two axes:
    #
    # (A) Encoding-scale at moderate motor: bigger lang_input, smaller
    #     motor, 64-word. Tests if encoding density alone fixes binding.
    "consolidation_synonym_64word_encoding_scale_smoke": [
        "--smoke",
        "--vocab-size", "64",
        "--n-lang-input", "8192",
        "--n-motor-per-action", "2000",
        "--n-motor-fs-per-action", "240",
        "--n-test-per-word", "5",
    ],
    # (B) Lang-balanced at 64-word: aspect ratio matched to encoding load
    #     (n_lang/n_motor ≈ 2). Predicted PASS at lower wall-clock than
    #     n_motor=6000 variant.
    "consolidation_synonym_64word_lang_balanced_smoke": [
        "--smoke",
        "--vocab-size", "64",
        "--n-lang-input", "8192",
        "--n-motor-per-action", "4000",
        "--n-motor-fs-per-action", "480",
        "--n-test-per-word", "5",
    ],
    # (C) Big-encoding at 256-word: scale encoding 4× (16384) to handle
    #     extreme vocab. Tests true encoding ceiling. Predicted ~24 GB
    #     VRAM (right at 3090 limit).
    "consolidation_synonym_256word_big_encoding_smoke": [
        "--smoke",
        "--vocab-size", "256",
        "--n-lang-input", "16384",
        "--n-motor-per-action", "2000",
        "--n-motor-fs-per-action", "240",
        "--n-test-per-word", "3",
    ],
    # (D) Big-motor at 16-word: motor capacity excess test. Does
    #     n_motor=12000 with only 16 words yield smoother / faster
    #     convergence than n_motor=2000? Tests whether the 333-floor
    #     rule has diminishing returns above some threshold.
    "consolidation_synonym_16word_big_motor_smoke": [
        "--smoke",
        "--vocab-size", "16",
        "--n-lang-input", "4096",
        "--n-motor-per-action", "12000",
        "--n-motor-fs-per-action", "1440",
        "--n-test-per-word", "10",
    ],
    # 12-word vocab extension (adds n/e/s/w abbreviations as 3rd synonym
    # per action). Tests whether CLS theory still holds at richer vocabulary
    # scale than the 8-word validated result. Same architecture as 8-word;
    # eval covers all 12 words. Wall-clock similar to 8-word medium
    # (~115 min/seed), since training event count scales linearly with
    # vocab_size but consolidation_interval keeps chunk count.
    "consolidation_synonym_12word_medium": [
        "--medium",
        "--vocab-size", "12",
        "--n-test-per-word", "20",
    ],
    # 12-word vocab + SCALED-UP motor pools (n_motor=2000 vs default 1000).
    # Per Tier 2.1 BREAKTHROUGH: "bigger motor pools give STDP enough room
    # for functional sub-populations". 8-word had 2 sub-pops per motor_X;
    # 12-word needs 3, may exceed 1000-neuron capacity. This preset doubles
    # n_motor + n_motor_fs to test whether scale-up lifts primary retention.
    # ~22K neurons, ~28M synapses, ~5 GB GPU. Wall clock ~150-170 min/seed
    # at medium config (~1.5x slower per step due to bigger motor matrices).
    # Per user 2026-05-08 EDT: "Feel free to scale up the sim if that's
    # needed as well, we have headroom for more neurons/synapses."
    "consolidation_synonym_12word_scaled_medium": [
        "--medium",
        "--vocab-size", "12",
        "--n-motor-per-action", "2000",
        "--n-motor-fs-per-action", "240",
        "--n-test-per-word", "20",
    ],
    # ─── 2026-05-13 concept pool architecture (diversity beyond 4 motors) ─
    # User mandate 2026-05-12: focus on concepts/composition/diversity.
    # Adds dedicated noun pools (APPLE/RIVER/DOG/CAT) + verb pools (GO/COME)
    # alongside existing motor pools (N/E/S/W). 10 distinct output categories
    # = 2.5x diversity over Tier 1's 4-motor ceiling. Each pool follows the
    # proven Tier 1 recipe (500 neurons + paired teacher + FS cross-inhibition
    # within kind + reciprocal lang_output).
    #
    # Phase 1: validate cross-category isolation (typing "apple" -> noun_APPLE,
    #          NOT motor_N or verb_GO). Tier 1 recipe at full scale.
    # Wall clock: ~15-25 min/seed.
    # v7 production recipe (2026-05-13): 6/12 PASS at single seed,
    # mean 12.75/16 across seeds 42-45 (std 1.50). Trains in ~17 min/seed.
    #
    # v14 production recipe (orthogonal drive codes + 16 pools):
    # weak-concept-dynamics (iter AA recipe), interleaved (Tier 1 pattern),
    # topographic 3.0/0.3 (target-priority 10x ratio), enable-adjective
    # (16-pool arch = 4 motor + 4 noun + 4 verb + 4 adjective),
    # orthogonal-codes (non-overlapping word patterns — fixes hash-based
    # ~10% pairwise overlap), sparsity 0.05.
    "concept_pool_demo": [
        "--n-train-events", "200",
        "--n-lang-input", "2048",
        "--n-per-pool", "200",
        "--n-fs-per-pool", "24",
        "--weak-concept-dynamics",
        "--interleaved",
        "--topographic-factor", "3.0",
        "--off-target-factor", "0.3",
        "--enable-adjective",
        "--orthogonal-codes",
        "--sparsity", "0.05",
    ],
    # Multi-bridge set wrappers (set2-5) use IDENTICAL args to concept_pool_demo;
    # only their vocab tables differ. Each trains a 12-concept-word v16 bridge.
    "concept_pool_demo_set2": [
        "--n-train-events", "200",
        "--n-lang-input", "2048",
        "--n-per-pool", "200",
        "--n-fs-per-pool", "24",
        "--weak-concept-dynamics",
        "--interleaved",
        "--topographic-factor", "3.0",
        "--off-target-factor", "0.3",
        "--enable-adjective",
        "--orthogonal-codes",
        "--sparsity", "0.05",
    ],
    "concept_pool_demo_set3": [
        "--n-train-events", "200",
        "--n-lang-input", "2048",
        "--n-per-pool", "200",
        "--n-fs-per-pool", "24",
        "--weak-concept-dynamics",
        "--interleaved",
        "--topographic-factor", "3.0",
        "--off-target-factor", "0.3",
        "--enable-adjective",
        "--orthogonal-codes",
        "--sparsity", "0.05",
    ],
    "concept_pool_demo_set4": [
        "--n-train-events", "200",
        "--n-lang-input", "2048",
        "--n-per-pool", "200",
        "--n-fs-per-pool", "24",
        "--weak-concept-dynamics",
        "--interleaved",
        "--topographic-factor", "3.0",
        "--off-target-factor", "0.3",
        "--enable-adjective",
        "--orthogonal-codes",
        "--sparsity", "0.05",
    ],
    "concept_pool_demo_set5": [
        "--n-train-events", "200",
        "--n-lang-input", "2048",
        "--n-per-pool", "200",
        "--n-fs-per-pool", "24",
        "--weak-concept-dynamics",
        "--interleaved",
        "--topographic-factor", "3.0",
        "--off-target-factor", "0.3",
        "--enable-adjective",
        "--orthogonal-codes",
        "--sparsity", "0.05",
    ],
    # End-to-end demo across all 5 bridges. Requires all 5 trained at the
    # requested seed. Use --friendly extra arg for natural-language output.
    "multibridge_60word_demo": [
        "--friendly",
    ],
    # 2026-05-15 catalog G.20 BREAKTHROUGH: distributed encoding in
    # shared pool. 32 concepts / 1600 neurons / 81.2% top-1 (4.2x
    # per-neuron efficiency vs v16). Multi-seed 74% mean.
    "g20_shared_pool_n32": [
        "--n-concepts", "32",
        "--n-train-events", "400",
        "--n-lang-input", "8192",
        "--n-shared-pool", "1600",
        "--slice-size", "50",
        "--top-k", "100",
        "--topographic-factor", "10.0",
        "--off-target-factor", "0.1",
        "--sparsity", "0.03",
    ],
    # G.20 capacity-stretch tier (60 concepts; 56.7% top-1 single seed).
    "g20_shared_pool_n60": [
        "--n-concepts", "60",
        "--n-train-events", "400",
        "--n-lang-input", "8192",
        "--n-shared-pool", "3200",
        "--slice-size", "50",
        "--top-k", "100",
        "--topographic-factor", "10.0",
        "--off-target-factor", "0.1",
        "--sparsity", "0.015",
    ],
    # 2026-05-14 validated semantic memory recipes (after architecture-mismatch
    # bug retraction). Both require a pre-trained v16 concept-pool bridge
    # passed via extra_args (--load-bridge ...). The launcher passes --seed
    # automatically; the user provides the bridge path in extra_args.
    #
    # engram_stim_recall: Tonegawa-style engram tagging. Encode N pairs of
    # concepts; stim the tag → both bound concepts reactivate.
    # Multi-seed result: 87.5% stim-recall, 27.5% assoc-recall.
    "engram_stim_recall": [
        "--n-lang-input", "2048",
        "--n-per-pool", "200",
        "--n-fs-per-pool", "24",
        "--n-words-for-orthogonal", "16",
        "--encoding-steps", "500",
        "--sparsity", "0.05",
        "--balanced-teacher-pA", "500.0",
        "--pairs", "apple:big,dog:small,cat:hot,river:cold,go:look,come:stop,big:hot,small:cold",
    ],
    # multitag_cue_recall: user-friendly cue retrieval. For cue X, find all
    # engram tags containing X, stim each, aggregate. Multi-seed: 90% FULL
    # (all associates in top-2 of 15) / 100% PARTIAL. This is the validated
    # concept-concept conversational capability.
    "multitag_cue_recall": [
        "--n-lang-input", "2048",
        "--n-per-pool", "200",
        "--n-fs-per-pool", "24",
        "--n-words-for-orthogonal", "16",
        "--encoding-steps", "500",
        "--sparsity", "0.05",
        "--balanced-teacher-pA", "500.0",
        "--top-n", "2",
        "--pairs", "apple:big,dog:small,cat:hot,river:cold,big:hot,small:cold,apple:cat,dog:river",
    ],
    # Phase 2: composition test - do multiple pools fire together for
    # phrases like "go north"? Tests NMDA sequential + co-fire merging.
    # Same v14 recipe arch. Compose passes co-fire ~2/6, sequential 0-1/6
    # (weak dynamics trade off vs Phase 1 isolation). Sequential is the
    # remaining open frontier — see docs/plans/2026-05-13-sequential-
    # composition-design-note.md for v15 plan.
    "concept_compose_demo": [
        "--n-train-events", "200",
        "--n-lang-input", "2048",
        "--n-per-pool", "200",
        "--n-fs-per-pool", "24",
        "--weak-concept-dynamics",
        "--topographic-factor", "3.0",
        "--off-target-factor", "0.3",
        "--enable-adjective",
        "--orthogonal-codes",
        "--sparsity", "0.05",
    ],
    # Phase 3: A->W readout. Drive each of 16 pools, cosine-rank the
    # network's "spoken" word against all 16 trained words. v14 result:
    # 16/16 = 100% unanimous across seeds 42-45 (64/64 = 100% across 4
    # seeds). CRITICAL: orthogonal-codes + sparsity 0.05 must match
    # training (the reference patterns must use the same drive codes).
    "concept_speak_demo": [
        "--n-train-events", "200",
        "--n-lang-input", "2048",
        "--n-per-pool", "200",
        "--n-fs-per-pool", "24",
        "--weak-concept-dynamics",
        "--topographic-factor", "3.0",
        "--off-target-factor", "0.3",
        "--enable-adjective",
        "--orthogonal-codes",
        "--sparsity", "0.05",
    ],
    # NOTE: phase_2_1_abc / phase_2_2_shakespeare presets exist only on
    # the path-f-hybrid branch (cortex_pretraining.py is not on main).
    # Run from that branch:
    #     git checkout path-f-hybrid
    #     python -m research.runners.cortex_pretraining --task abc ...
    # Not exposed via this launcher to avoid silent module-not-found
    # failures on main.
}


# Per-preset runner module override. If a preset is missing here, the
# launcher uses research.runners.g11_bg_runner (the navigation flagship).
# Text I/O presets dispatch to text_eval_embodied which has its own
# CLI surface (--out-stats instead of --out, no --moving-goal etc).
PRESET_RUNNERS: dict[str, str] = {
    "text_io_v2_baseline":           "research.runners.text_eval_embodied",
    "text_io_v2_smoke":              "research.runners.text_eval_embodied",
    "text_io_distributed_motor_pop": "research.runners.text_eval_embodied",
    "text_io_motor_cross_coupling":  "research.runners.text_eval_embodied",
    # 2026-05-06/07 Phase 1.3, 1.4, 1.5, Tier 2.3, chat demos
    "phase_1_4_forgetting":   "research.runners.continual_forgetting_eval",
    "phase_1_3_consolidation": "research.runners.consolidation_trainer",
    "phase_1_5_unified":                  "research.runners.continual_eval_suite",
    "phase_1_5_unified_scaled":           "research.runners.continual_eval_suite",
    "phase_1_5_unified_scaled_v400":             "research.runners.continual_eval_suite",
    "phase_1_5_interference_only_v400":          "research.runners.continual_eval_suite",
    "phase_1_5_interference_only_n_motor_2000":  "research.runners.continual_eval_suite",
    "phase_1_5_long_tail_relaxed":               "research.runners.continual_eval_suite",
    "phase_1_5_unified_combined":                "research.runners.continual_eval_suite",
    "tier_2_3_phrases":       "research.runners.phrase_trainer",
    "chat_demo":              "research.runners.chat_demo",
    "chat_continual_demo":    "research.runners.chat_continual_demo",
    "chat_synonym_demo":      "research.runners.chat_synonym_demo",
    "chat_learn_demo":        "research.runners.chat_learn_demo",
    "chat_speak_demo":        "research.runners.chat_speak_demo",
    "chat_speak_synonym_demo": "research.runners.chat_speak_synonym_demo",
    "consolidation_synonym":                       "research.runners.consolidation_synonym_trainer",
    "consolidation_synonym_smoke":                 "research.runners.consolidation_synonym_trainer",
    "consolidation_synonym_medium":                "research.runners.consolidation_synonym_trainer",
    "consolidation_synonym_medium_strict":         "research.runners.consolidation_synonym_trainer",
    "consolidation_synonym_12word_medium":         "research.runners.consolidation_synonym_trainer",
    "consolidation_synonym_12word_scaled_medium":  "research.runners.consolidation_synonym_trainer",
    "consolidation_synonym_16word_scaled_medium":  "research.runners.consolidation_synonym_trainer",
    "consolidation_synonym_16word_scaled_smoke":   "research.runners.consolidation_synonym_trainer",
    "consolidation_synonym_24word_scaled_smoke":   "research.runners.consolidation_synonym_trainer",
    "consolidation_synonym_32word_scaled_smoke":   "research.runners.consolidation_synonym_trainer",
    "consolidation_synonym_48word_scaled_smoke":   "research.runners.consolidation_synonym_trainer",
    "consolidation_synonym_64word_scaled_smoke":   "research.runners.consolidation_synonym_trainer",
    "consolidation_synonym_96word_scaled_smoke":   "research.runners.consolidation_synonym_trainer",
    "consolidation_synonym_128word_scaled_smoke":  "research.runners.consolidation_synonym_trainer",
    "consolidation_synonym_256word_scaled_smoke":  "research.runners.consolidation_synonym_trainer",
    "consolidation_synonym_64word_encoding_scale_smoke": "research.runners.consolidation_synonym_trainer",
    "consolidation_synonym_64word_lang_balanced_smoke":  "research.runners.consolidation_synonym_trainer",
    "consolidation_synonym_256word_big_encoding_smoke":  "research.runners.consolidation_synonym_trainer",
    "consolidation_synonym_16word_big_motor_smoke":      "research.runners.consolidation_synonym_trainer",
    # Concept pool architecture (2026-05-13)
    "concept_pool_demo":     "research.runners.concept_pool_demo",
    "concept_compose_demo":  "research.runners.concept_compose_demo",
    "concept_speak_demo":    "research.runners.concept_speak_demo",
    # Multi-bridge vocab expansion: each set wrapper trains a new v16 bridge
    # with DIFFERENT vocab. Stack: set2 = 12 new words, set3 = 12, etc.
    "concept_pool_demo_set2": "research.runners.concept_pool_demo_set2",
    "concept_pool_demo_set3": "research.runners.concept_pool_demo_set3",
    "concept_pool_demo_set4": "research.runners.concept_pool_demo_set4",
    "concept_pool_demo_set5": "research.runners.concept_pool_demo_set5",
    # 60-word multi-bridge end-to-end demo (2026-05-15). Requires all 5
    # bridges trained: seed${N}_v16/_set2/_set3/_set4/_set5.simstate.h5.
    "multibridge_60word_demo": "research.runners.multibridge_60word_demo",
    # G.20 distributed-encoding BREAKTHROUGH (catalog G.20, 2026-05-15)
    "g20_shared_pool_n32": "research.runners.concept_pool_demo_shared",
    "g20_shared_pool_n60": "research.runners.concept_pool_demo_shared",
    # Semantic memory + cue retrieval (2026-05-14 corrected after bug fix)
    "engram_stim_recall":    "research.runners.compose_concept_engram",
    "multitag_cue_recall":   "research.runners.multitag_eval",
    # R1 robust-comprehension showcase (2026-06-20): multi-cue vs order-only parser on imperfect English.
    "imperfect_english_demo": "research.runners.imperfect_english_demo",
    # phase_2_* presets removed -- cortex_pretraining lives on path-f-hybrid only.
}


# Per-preset output-flag override. The text_eval_embodied runner writes
# its stats JSON via --out-stats (g11_bg_runner uses --out for its trial
# JSON). Defaults to "--out".
PRESET_OUTPUT_FLAG: dict[str, str] = {
    "text_io_v2_baseline":           "--out-stats",
    "text_io_v2_smoke":              "--out-stats",
    "text_io_distributed_motor_pop": "--out-stats",
    "text_io_motor_cross_coupling":  "--out-stats",
    # 2026-05-06/07 Phase 1.3-1.5, Tier 2.3, chat demos all use --out-stats
    "phase_1_4_forgetting":   "--out-stats",
    "phase_1_3_consolidation": "--out-stats",
    "phase_1_5_unified":                  "--out-stats",
    "phase_1_5_unified_scaled":           "--out-stats",
    "phase_1_5_unified_scaled_v400":             "--out-stats",
    "phase_1_5_interference_only_v400":          "--out-stats",
    "phase_1_5_interference_only_n_motor_2000":  "--out-stats",
    "phase_1_5_long_tail_relaxed":               "--out-stats",
    "phase_1_5_unified_combined":                "--out-stats",
    "tier_2_3_phrases":       "--out-stats",
    "chat_demo":              "--out-stats",
    "chat_continual_demo":    "--out-stats",
    "chat_synonym_demo":      "--out-stats",
    "chat_learn_demo":        "--out-stats",
    "chat_speak_demo":        "--out-stats",
    "chat_speak_synonym_demo": "--out-stats",
    "consolidation_synonym":                       "--out-stats",
    "consolidation_synonym_smoke":                 "--out-stats",
    "consolidation_synonym_medium":                "--out-stats",
    "consolidation_synonym_medium_strict":         "--out-stats",
    "consolidation_synonym_12word_medium":         "--out-stats",
    "consolidation_synonym_12word_scaled_medium":  "--out-stats",
    "consolidation_synonym_16word_scaled_medium":  "--out-stats",
    "consolidation_synonym_16word_scaled_smoke":   "--out-stats",
    "consolidation_synonym_24word_scaled_smoke":   "--out-stats",
    "consolidation_synonym_32word_scaled_smoke":   "--out-stats",
    "consolidation_synonym_48word_scaled_smoke":   "--out-stats",
    "consolidation_synonym_64word_scaled_smoke":   "--out-stats",
    "consolidation_synonym_96word_scaled_smoke":   "--out-stats",
    "consolidation_synonym_128word_scaled_smoke":  "--out-stats",
    "consolidation_synonym_256word_scaled_smoke":  "--out-stats",
    "consolidation_synonym_64word_encoding_scale_smoke": "--out-stats",
    "consolidation_synonym_64word_lang_balanced_smoke":  "--out-stats",
    "consolidation_synonym_256word_big_encoding_smoke":  "--out-stats",
    "consolidation_synonym_16word_big_motor_smoke":      "--out-stats",
    # phase_2_* presets removed -- see PRESETS dict comment above.
}


class LaunchRequest(BaseModel):
    preset: str = "flagship"
    seed: int = 42
    extra_args: list[str] = []
    out_filename: str | None = None  # if None, generated from preset+seed
    # When True, set CUBLAS_WORKSPACE_CONFIG=:4096:8 in the subprocess env
    # so cuBLAS picks deterministic algorithms. Tightens seed-to-seed noise
    # floor at the cost of ~10-30% slowdown. Required for the cluster
    # comparisons that don't otherwise rise above the run-to-run noise.
    deterministic: bool = False
    # When True, append --emit-activity so the run streams live per-region
    # brain activity (frontend-revamp Phase 1). The frontend sets this ONLY
    # for launches from the Brain/Environment cockpit screens — NEVER for
    # multi-seed / science launches — so determinism-sensitive runs stay
    # byte-identical (the runner flag is itself default-off). Only injected
    # for live-mode-capable (g11_bg_runner navigation) presets.
    emit_activity: bool = False


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
    # Resolve which runner module + output flag this preset uses.
    # Defaults to g11_bg_runner / --out for navigation presets; text I/O
    # presets dispatch to text_eval_embodied / --out-stats.
    runner_module = PRESET_RUNNERS.get(req.preset, "research.runners.g11_bg_runner")
    output_flag = PRESET_OUTPUT_FLAG.get(req.preset, "--out")
    is_text_io = req.preset.startswith("text_io_")
    # Live-mode flags (--interactive-control-file, --progress-print-interval)
    # are only accepted by g11_bg_runner. Any preset that overrides
    # PRESET_RUNNERS away from g11_bg_runner can't take them — those runners
    # would reject the flags with "unrecognized arguments". Smoke test
    # 2026-05-07 caught chat_demo / chat_continual_demo / Phase 1.3-1.5 /
    # Tier 2.3 all failing this way.
    supports_live_mode = (req.preset not in PRESET_RUNNERS) and not is_text_io

    out_filename = req.out_filename or (
        f"text_io_seed{req.seed}_{req.preset}_{run_id[:6]}.json" if is_text_io
        else f"g11_seed{req.seed}_{req.preset}_{run_id[:6]}.json"
    )
    out_path = str(RAW_RUNS_DIR / out_filename)

    extras = list(req.extra_args)
    # Always create a control file so EVERY run can be paused/resumed.
    # Previously gated to interactive_* presets only, but the pause
    # capability is generally useful (free GPU for other work without
    # losing run progress). Goal-override / inject-reward fields still
    # only really matter for interactive_* runs but are harmless extras
    # for non-interactive runs.
    # Only g11_bg_runner accepts --interactive-control-file. Skip it for
    # any preset routed to a different runner (text I/O, chat demos,
    # phase 1.3/1.4/1.5 evals, tier 2.3 phrase trainer, etc.).
    control_file = None
    if supports_live_mode:
        control_file = str(RUNTIME_DIR / f"control_{run_id}.json")
        Path(control_file).write_text("{}")
        extras.extend(["--interactive-control-file", control_file])

    # Inject a sensible --progress-print-interval default if none is set
    # in the preset or the user's extras. Only g11_bg_runner has this
    # flag; skip it for text I/O / chat / phase / tier runners.
    #   interactive_*  -> 1   (per-step, for live-mode animation while attached)
    #   everything else -> 20 (every 20 steps; smoothes the live chart but
    #                          avoids the per-step CPU<->GPU sync overhead
    #                          identified in the throughput investigation)
    base_extras = list(extras) + list(PRESETS[req.preset])
    if supports_live_mode and not any(a == "--progress-print-interval" for a in base_extras):
        default_ppi = "1" if req.preset.startswith("interactive_") else "20"
        extras.extend(["--progress-print-interval", default_ppi])

    # Live brain-activity streaming (frontend-revamp Phase 1): only when the
    # frontend explicitly asks (Brain/Environment launch) AND the preset routes
    # to g11_bg_runner (the only runner that accepts --emit-activity). Never
    # auto-added to science/multi-seed launches, preserving determinism.
    if req.emit_activity and supports_live_mode \
            and not any(a == "--emit-activity" for a in base_extras):
        extras.append("--emit-activity")

    cmd = [
        sys.executable, "-m", runner_module,
        *PRESETS[req.preset],
        "--seed", str(req.seed),
        output_flag, out_path,
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
    if req.deterministic:
        # Must be set BEFORE the subprocess imports cupy/cuBLAS; setting it
        # here in the env propagates to the child correctly.
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

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
            "deterministic": req.deterministic,
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

def _scan_for_orphans() -> int:
    """Scan sidecar `.cmd.json` files for runs whose pid is still alive
    and not yet in launched_runs. Reconstruct LaunchedRun entries for
    them so the dashboard's Live picker can list (and pause/kill) them.

    Two flavors of sidecar are accepted:
    1. log_file is set (webapp-launched runs) — tail the log to seed
       progress_events and continue draining.
    2. log_file is None (raw-spawned runs that emit a sidecar via
       --emit-webapp-sidecar) — list-only: no log, no progress events,
       but pause + kill still work via control_file + pid.

    Returns the number of newly-recovered runs (for logging).

    Performance: skips sidecars older than 24h (their PIDs would be
    stale — not worth the psutil.Process query cost). Sorts newest
    first so the freshest in-flight runs are picked up first.

    Opt-out: set WEBAPP_NO_ORPHAN_RECOVERY=1 to disable sidecar recovery
    entirely. Needed when many historical sidecars exist alongside
    concurrent CLI/pool runs — PIDs get reused, so a stale sidecar can
    match an unrelated live process (manufacturing a phantom run, and a
    kill on that phantom would terminate the unrelated real process).
    With recovery off the dashboard tracks only this-session API launches.
    """
    if os.environ.get("WEBAPP_NO_ORPHAN_RECOVERY") == "1" \
            or (RUNTIME_DIR / ".no_orphan_recovery").exists():
        return 0
    new_count = 0
    cutoff = time.time() - 86400.0  # 24h
    # 2026-06-08: scan BOTH g11_bg/ (the default --out location) AND its
    # parent research/findings/raw/ (custom --out locations — e.g. pool / CLI
    # batches that pass an explicit --out) so a run with ANY --out path is
    # auto-detected, not just default-named ones. Sidecars are de-duped by
    # path and bounded by the 24h cutoff below.
    _scan_dirs = {RAW_RUNS_DIR, RAW_RUNS_DIR.parent}
    _all_sidecars: list[Path] = []
    for _d in _scan_dirs:
        try:
            _all_sidecars.extend(_d.glob("*.cmd.json"))
        except OSError:
            pass
    sidecars = sorted(
        set(_all_sidecars),
        key=lambda p: p.stat().st_mtime if p.exists() else 0,
        reverse=True,
    )
    for sidecar_path in sidecars:
        # Skip ancient sidecars to keep scan latency bounded (was hanging
        # the periodic 30s scan when hundreds of overnight files accumulated)
        try:
            if sidecar_path.stat().st_mtime < cutoff:
                continue
        except OSError:
            continue
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
        # 2026-06-08: PID-dedup. If this pid is already owned by a tracked
        # run, skip — otherwise a second sidecar recording the same (possibly
        # reused) pid double-tracks one live process, and a kill on the
        # phantom entry would terminate the unrelated real run. This is the
        # bug that killed a live Rank 2 run on 2026-06-08.
        if any(r.pid == pid for r in list(launched_runs.values())):
            continue
        # Defend against PID reuse: a sidecar's pid must belong to a
        # python process whose create_time matches the sidecar's
        # started_at. Without this check, recycled PIDs (Firefox, Claude,
        # etc.) get listed as live runs every server restart.
        if not _orphan_pid_belongs_to_runner(pid, sidecar.get("started_at")):
            continue
        log_file = sidecar.get("log_file")
        # log_file=None is now legal (raw-spawned runners). Skip only if
        # log_file is set but the file is missing (corrupt/moved sidecar).
        if log_file and not Path(log_file).exists():
            continue
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
        if log_file:
            # Webapp-launched orphan: resume from log start to seed progress.
            try:
                new_pos, lines = _read_new_lines(log_file, 0)
                for line in lines:
                    run.stdout_lines.append(line)
                    ev = _try_parse_progress(line, time.time())
                    if ev is not None:
                        run.progress_events.append(ev)
                    # Also seed activity frames so a run adopted mid-stream has
                    # its prior brain activity available to replay + scrub (the
                    # WS replays this history on connect). Without this, only
                    # post-adoption frames would be scrubbable.
                    af = _try_parse_activity(line, time.time(), run.activity_seq)
                    if af is not None:
                        run.activity_seq += 1
                        run.activity_frames.append(af)
                run.log_pos = new_pos
            except Exception:
                run.log_pos = 0
            asyncio.create_task(_drain_log(run))
        # else: raw-spawned (no log). Skip drain_log; entry is still
        # listable + killable + pausable.
        new_count += 1
        flavor = "raw-spawned" if not log_file else "webapp-launched"
        print(f"[webapp] discovered {flavor} run {run_id} (pid={pid})", flush=True)
    return new_count


@app.on_event("startup")
async def recover_orphan_runs() -> None:
    """Initial orphan scan at server start. Periodic re-scan is in
    `_periodic_orphan_scan` below."""
    n = _scan_for_orphans()
    if n:
        print(f"[webapp] startup: recovered {n} orphan runs", flush=True)


@app.on_event("startup")
async def _periodic_orphan_scan() -> None:
    """Re-scan for orphan sidecars every 5s. Picks up runs spawned via
    raw `python -m research.runners.<runner> --emit-webapp-sidecar` (the
    DEFAULT for g11_bg_runner) without requiring a server restart, so any
    CLI-started run is auto-detected, drained, and (if it passes
    --emit-activity) live-streamed to the 3D brain — not just runs launched
    through the dashboard.

    5s (was 30s): a full nav brain takes ~20-30s to build before it emits
    its first [ACTIVITY] frame, so a 5s scan reliably ADOPTS the run (starts
    its _drain_log tailer) BEFORE streaming begins — no missed frames and the
    run appears in the Live picker within a few seconds of launch. The scan is
    cheap (sidecars older than 24h are skipped; only live recent ones cost a
    psutil check), so the higher cadence is safe."""
    async def _loop():
        while True:
            await asyncio.sleep(5)
            try:
                _scan_for_orphans()
            except Exception as e:
                print(f"[webapp] periodic orphan scan failed: {e}", flush=True)
    asyncio.create_task(_loop())


@app.on_event("startup")
async def _warm_chat_brain() -> None:
    """Pre-build the off-bridge Qwen renderer (+ the default ChatBrain) ONCE at
    startup so the FIRST `/api/brain-chat` turn doesn't pay the ~58s Qwen-0.5B
    model load (the SK brain-load is already fixed; the renderer model load is
    the remaining first-turn cost — one-time per webapp session; warm turns are
    ~1.7s). See AUTONOMOUS_STATE CYCLE 521 + `_console_live_debug_fixes.json`.

    Runs in a BACKGROUND DAEMON THREAD so uvicorn reports startup-complete
    promptly (the dashboard + every other endpoint are usable immediately while
    the model loads). Idempotent + guarded:
      - only warms the qwen model when the default renderer actually resolves to
        'qwen' (a GPU host on the cupy backend); on a GPU-less / stub-renderer
        host this is a no-op (the stub builds instantly, no model download);
      - any failure (no GPU, qwen/torch unavailable, model not downloaded) is
        swallowed — the chat endpoint then builds on the first turn / falls back
        to the stub renderer exactly as before. Warming NEVER blocks boot and
        NEVER changes the chat answer (only the first-turn latency).

    The warm builds the DEFAULT cache key `(session='default', brain='tiny-demo',
    renderer=<default>)`, so a default first turn hits the warm cache outright;
    and because the qwen renderer is a PROCESS-WIDE shared singleton
    (`_get_warm_qwen_renderer`), even a first turn that picks a DIFFERENT brain
    reuses the already-loaded model (only its small brain build remains)."""
    renderer = _default_brain_renderer()
    if renderer != "qwen":
        # No GPU / cupy not selected → the stub renderer is instant; nothing to
        # warm. (Building the stub ChatBrain here would just duplicate the
        # cheap first-turn build, so skip — keep startup lean.)
        print(f"[webapp] startup: chat renderer is {renderer!r} (GPU-free) — "
              "no Qwen model to warm", flush=True)
        return

    def _warm() -> None:
        try:
            import time as _t
            t0 = _t.time()
            print("[webapp] startup: warming the off-bridge Qwen-0.5B renderer "
                  "(one-time model load; the first chat turn will be fast)...",
                  flush=True)
            # Build the DEFAULT ChatBrain (default brain + the resolved qwen
            # renderer). This constructs the shared warm QwenRenderer (the heavy
            # model load) AND caches the default ChatBrain so the default first
            # turn is instant. Cache key MUST mirror brain_chat()'s lookup.
            default_brain = "tiny-demo"   # == BrainChatRequest.brain default
            chat, source = _build_chat_brain(default_brain, renderer)
            chat._brain_chat_source = source  # type: ignore[attr-defined]
            cache_key = ("default", default_brain, renderer)
            _BRAIN_CHATS.setdefault(cache_key, chat)
            # AFFECT (Gate-B): pre-build the co-resident spiking affect organ so the first turn's mood read is
            # fast. Best-effort + guarded (default-ON; BRAIN_AFFECT=0 skips it); a failure never blocks chat.
            try:
                from research.runners.affect_production_organ import affect_enabled
                if affect_enabled():
                    _get_affect_organ().ensure_built()
                    print("[webapp] startup: affect organ (co-resident graded-affect ladder) WARM", flush=True)
            except Exception as _ae:
                print(f"[webapp] startup: affect organ warm skipped ({type(_ae).__name__}: {_ae})", flush=True)
            # COMPREHENSION (Gate-B, D4): pre-build the co-resident spiking comprehension monitor so the first
            # turn's role-binding read is fast. Best-effort + guarded (default-ON; BRAIN_COMPREHENSION_GATE=0 skips).
            try:
                from research.runners.comprehension_production_organ import comprehension_enabled
                if comprehension_enabled():
                    _get_comprehension_organ().ensure_built()
                    print("[webapp] startup: comprehension monitor (co-resident SpikingRoleCompetition) WARM",
                          flush=True)
            except Exception as _ce:
                print(f"[webapp] startup: comprehension monitor warm skipped ({type(_ce).__name__}: {_ce})",
                      flush=True)
            dt = round(_t.time() - t0, 1)
            print(f"[webapp] startup: Qwen renderer WARM in {dt}s "
                  f"(default ChatBrain cached as {cache_key!r}); "
                  "first chat turn is now fast", flush=True)
        except Exception as e:   # no GPU / model missing / qwen unavailable
            print(f"[webapp] startup: Qwen warm skipped ({type(e).__name__}: {e}) "
                  "— the chat endpoint will build on the first turn / use the "
                  "stub renderer", flush=True)

    # Daemon thread: never blocks process exit, never blocks uvicorn boot.
    _threading.Thread(target=_warm, name="qwen-warm", daemon=True).start()


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


def _orphan_pid_belongs_to_runner(pid: int | None, sidecar_started_at: float | None) -> bool:
    """Validate that a sidecar's PID is plausibly the same process the
    sidecar was written for, not a recycled PID belonging to an unrelated
    application. Two checks:

    1. Process name starts with "python" (case-insensitive). Filters out
       PIDs reused by Firefox, Claude, etc.
    2. Process create_time is within ±10 seconds of the sidecar's
       started_at. Filters out the rare case of one Python process
       inheriting a recycled PID from another Python process.

    Falls open (returns True) if psutil is unavailable or the process
    can't be inspected — the strict check is best-effort, not a hard
    gate. _process_alive must already be true before calling this.
    """
    if pid is None:
        return False
    try:
        import psutil  # type: ignore
    except ImportError:
        return True  # No psutil → keep prior behavior
    try:
        proc = psutil.Process(int(pid))
        name = (proc.name() or "").lower()
        if not name.startswith("python"):
            return False
        if sidecar_started_at is not None:
            create_time = proc.create_time()
            if abs(create_time - float(sidecar_started_at)) > 10.0:
                return False
        return True
    except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError):
        # No process at that pid (race) or can't inspect — be permissive
        return True


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
                # Live brain-activity frame (frontend-revamp Phase 1). Parsed on
                # the same tail loop; appended to a bounded ring (latest-wins on
                # the WS). Cheap — only matches lines with the [ACTIVITY] prefix.
                af = _try_parse_activity(line, time.time(), run.activity_seq)
                if af is not None:
                    run.activity_seq += 1
                    run.activity_frames.append(af)
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
                        af = _try_parse_activity(line, time.time(), run.activity_seq)
                        if af is not None:
                            run.activity_seq += 1
                            run.activity_frames.append(af)
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


def _is_run_alive(run) -> bool:
    """True if the launched run is still active. Handles both freshly-spawned
    runs (proc handle, use poll()) and orphan-recovered runs (proc=None,
    only the PID survives — check OS process state)."""
    if run.proc is not None:
        return run.proc.poll() is None
    if run.pid is not None:
        return _process_alive(run.pid)
    return False


@app.get("/api/runs/launch")
def list_active_launches() -> JSONResponse:
    """Phase 2.5: list all in-flight (or recently-completed) runs known
    to this server process. Lets the World tab discover runs to follow."""
    out = []
    for run in list(launched_runs.values()):  # snapshot — bg orphan-scan mutates concurrently
        is_running = _is_run_alive(run)
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
        # `attachable` = the run streams telemetry to the webapp via its
        # log file. Webapp-launched runs always do; raw-spawned runs that
        # emitted a sidecar don't (log_file=None) and clicking them in
        # the picker would just open an empty live viewer. Frontend uses
        # this to disable the click-to-attach gesture for batch runs.
        # `interactive` = supports full per-trial control (goal override,
        # reward injection, pause). `pause_only` = only pause is supported
        # (replicated runner). Mutually exclusive in practice.
        attachable = run.log_file is not None
        pause_only = run.control_file is not None and not attachable
        interactive = run.control_file is not None and attachable
        out.append({
            "run_id": run.run_id,
            "running": is_running,
            "returncode": run.returncode,
            "started_at": run.started_at,
            "elapsed_sec": end_time - run.started_at,
            "out_path": run.out_path,
            "interactive": interactive,
            "pause_only": pause_only,
            "attachable": attachable,
            "latest_progress": _progress_to_json(latest) if latest else None,
        })
    out.sort(key=lambda r: r["started_at"], reverse=True)
    return JSONResponse({"runs": out, "count": len(out)})


@app.get("/api/runs/launch/{run_id}")
def launch_status(run_id: str) -> JSONResponse:
    run = launched_runs.get(run_id)
    if not run:
        raise HTTPException(404, "unknown run_id")
    is_running = _is_run_alive(run)
    # Belt-and-suspenders: lazy-set finished_at if drain_log missed it.
    # See list_active_launches comment.
    if not is_running and run.finished_at is None:
        try:
            run.finished_at = Path(run.log_file).stat().st_mtime if run.log_file else time.time()
        except (OSError, TypeError):
            run.finished_at = time.time()
    end_time = run.finished_at if run.finished_at is not None else time.time()
    # Surface the launch command list so the frontend can detect feature
    # flags (e.g. --enable-landmark-sensor / legacy --landmarks) for live
    # runs without re-parsing the sidecar.
    return JSONResponse({
        "run_id": run.run_id,
        "running": is_running,
        "returncode": run.returncode,
        "started_at": run.started_at,
        "elapsed_sec": end_time - run.started_at,
        "stdout_line_count": len(run.stdout_lines),
        "tail": run.stdout_lines[-20:],
        "progress_events": [_progress_to_json(p) for p in run.progress_events],
        # Latest live brain-activity frame (frontend-revamp Phase 1), if any.
        # WS is the live channel; this is a convenience for non-WS pollers + a
        # cheap signal of whether the run is streaming activity.
        "latest_activity": (
            _activity_to_json(run.activity_frames[-1]) if run.activity_frames else None
        ),
        "activity_frame_count": run.activity_seq,
        "out_path": run.out_path,
        "cmd": list(run.cmd),
    })


# ─── Static brain-region map (frontend-revamp Phase 1, 2026-06-08) ──────────
# The Brain tab needs the region+pathway graph (names, families/colors, neuron
# counts, 3D layout coords, pathway from/to/transmitter) to build the scene
# BEFORE any activity arrives. Phase 1 returns the hardcoded NAVIGATION region
# map — the same `brain3d_layout.json` the renderer already loads — so the
# server and renderer agree on the graph. Phase 2 will derive it per-run from
# sim/regions.py for non-navigation architectures.
_REGION_MAP_CACHE: dict[str, Any] | None = None


def _load_nav_region_map() -> dict[str, Any]:
    """Load (and cache) the navigation region map from the static layout file.

    Returns a dict with `regions`, `pathways`, and `family_colors` keys, plus
    the underlying counts. The layout file is the single source of truth shared
    with brain3d.js."""
    global _REGION_MAP_CACHE
    if _REGION_MAP_CACHE is not None:
        return _REGION_MAP_CACHE
    layout_path = STATIC_DIR / "brain3d_layout.json"
    try:
        raw = json.loads(layout_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raw = {}
    regions = raw.get("regions", {}) or {}
    pathways = raw.get("pathways", []) or []
    family_colors = {
        k: v for k, v in (raw.get("_family_colors", {}) or {}).items()
        if not k.startswith("_")
    }
    _REGION_MAP_CACHE = {
        "family": "navigation",
        "regions": regions,
        "pathways": pathways,
        "family_colors": family_colors,
        "n_regions": len(regions),
        "n_pathways": len(pathways),
    }
    return _REGION_MAP_CACHE


@app.get("/api/runs/{run_id}/region-map")
def run_region_map(run_id: str) -> JSONResponse:
    """Static region+pathway graph for a run's Brain-tab scene.

    Phase 1: returns the navigation region map (hardcoded to match
    brain3d.js's built-in layout) regardless of run_id, so the Brain tab can
    build the scene before activity arrives. The run_id is echoed back; an
    unknown run_id is NOT an error here (the map is a static layout, useful
    even for a run the server hasn't registered)."""
    region_map = _load_nav_region_map()
    run = launched_runs.get(run_id)
    return JSONResponse({
        "run_id": run_id,
        "known_run": run is not None,
        **region_map,
    })


def _progress_to_json(p: ProgressEvent) -> dict[str, Any]:
    return {
        "step": p.step, "total": p.total,
        "pos": list(p.pos), "goal": list(p.goal),
        "recent_dist": p.recent_dist,
        "timestamp": p.timestamp,
        "action": p.action,
        "reward": p.reward,
    }


def _activity_to_json(a: ActivityFrame) -> dict[str, Any]:
    return {
        "t": a.t,
        "regions": a.regions,
        "flux": a.flux,
        "step": a.step,
        "seq": a.seq,
        "timestamp": a.timestamp,
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

    # Replay the buffered activity HISTORY (up to the 600-frame ring) in order
    # so a late-joining client can immediately scrub BACK through the run's
    # recent brain activity — not just see the current frame. The client buffers
    # each frame into its own history (liveActivityHistory); the live loop below
    # then coalesces to the freshest so a slow client never backs the stream up.
    # Snapshot the deque first (the _drain_log task appends concurrently).
    last_activity_seq = -1
    for af in list(run.activity_frames):
        await websocket.send_json({"type": "activity", **_activity_to_json(af)})
        last_activity_seq = af.seq

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
            # Stream the LATEST activity frame only (coalesce / drop stale).
            # We never iterate the backlog: if the browser fell behind while
            # many frames arrived, it skips straight to the freshest one, so a
            # slow client can never back the stream up. The sim is never in
            # this loop — it already fire-and-forgot the line to its log.
            if run.activity_frames:
                latest = run.activity_frames[-1]
                if latest.seq > last_activity_seq:
                    await websocket.send_json({"type": "activity", **_activity_to_json(latest)})
                    last_activity_seq = latest.seq
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


# ─── Saved-bridges library (2026-05-10) ───────────────────────────────────
# Lists `.simstate.h5` checkpoint files saved by chat_repl --save-bridge,
# along with their sidecar metadata. Lets users browse trained bridges
# via the webapp instead of dropping to the CLI to remember paths.
#
# Bridges live under `bridges/` at the repo root (created on first save).
# Each bridge has:
#   - <name>.simstate.h5 — HDF5 weights + region indices
#   - <name>.simstate.h5.meta.json — sidecar metadata (mode, seed,
#     n_train_events, n_neurons, n_synapses, saved_at)

BRIDGES_DIR = REPO_ROOT / "bridges"


@app.get("/api/bridges")
def list_bridges() -> JSONResponse:
    """List all saved bridges with sidecar metadata.

    Returns: {"bridges": [...], "directory": str}
    Each entry: {name, path, size_mb, modified_at, metadata?}
    """
    bridges = []
    if BRIDGES_DIR.exists():
        for h5_path in sorted(BRIDGES_DIR.glob("*.simstate.h5")):
            try:
                stat = h5_path.stat()
                entry = {
                    "name": h5_path.stem.replace(".simstate", ""),
                    "path": str(h5_path.relative_to(REPO_ROOT)),
                    "size_mb": round(stat.st_size / (1024 * 1024), 1),
                    "modified_at": datetime.fromtimestamp(
                        stat.st_mtime
                    ).isoformat(),
                }
                # Pick up sidecar metadata if present
                sidecar = h5_path.with_name(h5_path.name + ".meta.json")
                if sidecar.exists():
                    try:
                        entry["metadata"] = json.loads(
                            sidecar.read_text(encoding="utf-8")
                        )
                    except (json.JSONDecodeError, OSError) as e:
                        entry["metadata_error"] = str(e)
                bridges.append(entry)
            except OSError as e:
                bridges.append({
                    "name": h5_path.stem,
                    "error": str(e),
                })
    return JSONResponse({
        "bridges": bridges,
        "directory": str(BRIDGES_DIR.relative_to(REPO_ROOT)) \
                       if BRIDGES_DIR.exists() else None,
        "n_bridges": len(bridges),
    })


@app.get("/api/bridges/{name}")
def get_bridge(name: str) -> JSONResponse:
    """Get full metadata for one bridge by name."""
    h5_path = BRIDGES_DIR / f"{name}.simstate.h5"
    if not h5_path.exists():
        raise HTTPException(404, f"bridge not found: {name}")
    sidecar = h5_path.with_name(h5_path.name + ".meta.json")
    metadata = {}
    if sidecar.exists():
        try:
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    stat = h5_path.stat()
    return JSONResponse({
        "name": name,
        "path": str(h5_path.relative_to(REPO_ROOT)),
        "size_mb": round(stat.st_size / (1024 * 1024), 1),
        "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "metadata": metadata,
    })


# ─── Bridge Lineage Manager endpoints (2026-05-10) ──────────────────────
# Persistent training lineages live under bridges/lineage/<name>/. These
# endpoints expose them to the webapp's future Lineages tab.

@app.get("/api/lineages")
def list_lineages() -> JSONResponse:
    """List all known lineages with summary metadata.

    Returns: {"lineages": [...], "directory": str}
    Each entry: {name, tier, vocab_size, cumulative_events, n_snapshots,
                 parent_lineage?, last_updated_at, arch?}
    """
    from sim.lineage import BridgeLineage, LINEAGE_ROOT
    root = REPO_ROOT / LINEAGE_ROOT
    entries = []
    if root.exists():
        for L in BridgeLineage.list_all(root=root):
            try:
                meta = L.read_metadata()
                n_history = len(L.list_history())
                # Get size of current state
                current_size_mb = None
                try:
                    current_size_mb = round(
                        L.current_path.stat().st_size / (1024 * 1024), 1
                    )
                except OSError:
                    pass
                entries.append({
                    "name": L.name,
                    "tier": meta.current_tier,
                    "vocab_size": len(meta.vocab),
                    "vocab_preview": meta.vocab[:8]
                                      if len(meta.vocab) > 8 else meta.vocab,
                    "cumulative_events": meta.cumulative_training_events,
                    "n_snapshots": n_history,
                    "parent_lineage": meta.parent_lineage,
                    "branched_at": meta.branched_at,
                    "created_at": meta.created_at,
                    "last_updated_at": meta.last_updated_at,
                    "tags": meta.tags,
                    "arch": meta.arch,
                    "size_mb": current_size_mb,
                })
            except Exception as e:
                entries.append({"name": L.name, "error": str(e)})
    return JSONResponse({
        "lineages": entries,
        "directory": str(root.relative_to(REPO_ROOT))
                       if root.exists() else None,
        "n_lineages": len(entries),
    })


@app.get("/api/lineages/{name}")
def get_lineage(name: str) -> JSONResponse:
    """Full metadata for one lineage including growth events + accuracy history."""
    from sim.lineage import BridgeLineage, LINEAGE_ROOT
    root = REPO_ROOT / LINEAGE_ROOT
    lineage = BridgeLineage(name, root=root)
    if not lineage.exists():
        raise HTTPException(404, f"lineage not found: {name}")
    meta = lineage.read_metadata()
    snapshots = lineage.list_history()
    snap_entries = []
    for snap in snapshots:
        snap_id = snap.name.replace("-checkpoint.simstate.h5", "")
        try:
            size_mb = round(snap.stat().st_size / (1024 * 1024), 1)
        except OSError:
            size_mb = None
        snap_entries.append({
            "snapshot_id": snap_id,
            "size_mb": size_mb,
        })
    current_size_mb = None
    try:
        current_size_mb = round(
            lineage.current_path.stat().st_size / (1024 * 1024), 1
        )
    except OSError:
        pass
    return JSONResponse({
        "name": name,
        "metadata": meta.to_dict(),
        "snapshots": snap_entries,
        "current_size_mb": current_size_mb,
    })


# ─── Synapse tiering endpoints (Phase 3 Strategy B, 2026-05-11) ─────────
# Inspect TieredSynapseStore state for lineages that have exported per-
# pathway shards. Useful for verifying activity tracking, eviction
# behavior, and total disk usage.

@app.get("/api/synapse-tiering/{name}")
def get_synapse_tiering(name: str) -> JSONResponse:
    """Return per-pathway shard inventory + size for a lineage.

    Returns:
        {
          "lineage_name": "main",
          "shards_dir": "bridges/lineage/main/shards" (or null),
          "n_pathways": 24,
          "total_size_mb": 12.3,
          "shards": [
            {"name": "language_input_to_motor_N", "size_mb": 0.5, "exists": true},
            ...
          ]
        }

    404 if the lineage doesn't exist.
    200 with empty shards list if lineage exists but has no shards
    (user hasn't run export_shards yet).
    """
    from sim.lineage import BridgeLineage, LINEAGE_ROOT
    root = REPO_ROOT / LINEAGE_ROOT
    lineage = BridgeLineage(name, root=root)
    if not lineage.exists():
        raise HTTPException(404, f"lineage not found: {name}")

    shard_root = lineage.root / "shards"
    if not shard_root.exists():
        return JSONResponse({
            "lineage_name": name,
            "shards_dir": None,
            "n_pathways": 0,
            "total_size_mb": 0.0,
            "shards": [],
        })

    pathway_names = lineage.list_shards()
    entries = []
    total_size = 0
    for pw in pathway_names:
        path = shard_root / f"{pw}.npz"
        if not path.exists():
            entries.append({"name": pw, "exists": False, "size_mb": None})
            continue
        try:
            sz = path.stat().st_size
            total_size += sz
            entries.append({
                "name": pw,
                "exists": True,
                "size_mb": round(sz / (1024 * 1024), 3),
            })
        except OSError as e:
            entries.append({
                "name": pw, "exists": False, "size_mb": None,
                "error": str(e),
            })

    return JSONResponse({
        "lineage_name": name,
        "shards_dir": str(shard_root.relative_to(REPO_ROOT)),
        "n_pathways": len(pathway_names),
        "total_size_mb": round(total_size / (1024 * 1024), 3),
        "shards": entries,
    })


# ─── BridgeMemory stats endpoint (Path 3, 2026-05-11) ──────────────────
# Returns memory subsystem state for a lineage: how many bindings,
# consolidations, forgets have been recorded via growth events.

@app.get("/api/bridge-memory/{name}")
def get_bridge_memory(name: str) -> JSONResponse:
    """Aggregate BridgeMemory state for a lineage.

    Counts growth events of kinds memory_bind / memory_forget /
    memory_consolidate / manual_save and returns a summary.

    Returns:
        {
          "lineage_name": "main",
          "n_bindings": 12,      # count of memory_bind events
          "n_forgets": 1,
          "n_consolidations": 3,
          "last_consolidation_at": "2026-05-11T...",
          "bindings": [
            {"key": "alice", "value": "north", "at": "...",
             "target_action": "N", "confidence": 0.91},
            ...
          ],
          "current_tier": "8-word",
          "vocab_size": 8,
          "cumulative_training_events": 400
        }

    404 if lineage doesn't exist.
    """
    from sim.lineage import BridgeLineage, LINEAGE_ROOT
    root = REPO_ROOT / LINEAGE_ROOT
    lineage = BridgeLineage(name, root=root)
    if not lineage.exists():
        raise HTTPException(404, f"lineage not found: {name}")
    meta = lineage.read_metadata()
    bindings = []
    n_forgets = 0
    n_consolidations = 0
    last_consolidation_at = None
    for e in meta.growth_events:
        kind = e.get("kind", "")
        if kind == "memory_bind":
            md = e.get("metadata", {})
            bindings.append({
                "key": md.get("key", ""),
                "value": md.get("value", ""),
                "target_action": md.get("target_action", ""),
                "confidence": md.get("confidence", 0.0),
                "at": e.get("at", ""),
            })
        elif kind == "memory_forget":
            n_forgets += 1
        elif kind == "memory_consolidate":
            n_consolidations += 1
            last_consolidation_at = e.get("at", "")
    return JSONResponse({
        "lineage_name": name,
        "n_bindings": len(bindings),
        "n_forgets": n_forgets,
        "n_consolidations": n_consolidations,
        "last_consolidation_at": last_consolidation_at,
        "bindings": bindings[-50:],  # last 50 only to bound response size
        "current_tier": meta.current_tier,
        "vocab_size": len(meta.vocab),
        "cumulative_training_events": meta.cumulative_training_events,
    })


# ─── Path 3 Phase 3.2: LLM-driven chat (2026-05-11) ──────────────────
# Drives a tool-use loop between a MockLLM and a real BridgeMemory.
# Subsequent turns reuse the in-process orchestrator (and its bridge)
# to avoid the ~1.5s rebuild cost per turn. Each lineage caches its own
# orchestrator. To swap in a real LLM later, set llm_callable to the
# (Phi-3 / Llama / Qwen) callable; orchestrator interface unchanged.

_LLM_ORCHESTRATORS: dict[tuple[str, str], object] = {}


class LLMChatRequest(BaseModel):
    """One LLM-driven chat turn against a BridgeMemory."""
    lineage: str = "main"
    mode: str = "tier1"  # tier1 / synonym / synonym12 / synonym16
    message: str
    # If True and an orchestrator is cached for this (lineage, mode),
    # reset it before processing the message (useful for "start fresh").
    reset_conversation: bool = False


@app.post("/api/llm-chat")
def llm_chat(req: LLMChatRequest) -> JSONResponse:
    """One LLM-driven turn against a BridgeMemory.

    First call per (lineage, mode) loads the lineage and trains/loads a
    bridge — can take 30-60s. Subsequent turns are fast (~1-2s for
    store, ~30ms for recall/speak).

    Returns:
        {
          "lineage_name": str,
          "mode": str,
          "response": str,                # the assistant's final message
          "tool_calls": [
            {"name": "memory_store", "args": {...}, "result_summary": "..."},
            ...
          ],
          "conversation_length": int,     # total messages in transcript
          "n_turns": int,                 # how many user turns so far
        }

    404 if lineage doesn't exist (use POST /api/lineages first or the
    chat_repl CLI to create one).
    """
    from sim.lineage import BridgeLineage, LINEAGE_ROOT
    root = REPO_ROOT / LINEAGE_ROOT
    lineage = BridgeLineage(req.lineage, root=root)
    if not lineage.exists():
        raise HTTPException(
            404,
            f"lineage '{req.lineage}' not found. Run "
            f"`python -m research.runners.chat_repl --mode {req.mode} "
            f"--lineage {req.lineage}` first to create it, or use the "
            f"llm_memory_demo runner.",
        )

    cache_key = (req.lineage, req.mode)
    if req.reset_conversation and cache_key in _LLM_ORCHESTRATORS:
        del _LLM_ORCHESTRATORS[cache_key]

    orch = _LLM_ORCHESTRATORS.get(cache_key)
    if orch is None:
        from sim.bridge_memory import BridgeMemory
        from sim.llm_memory_orchestrator import (
            LLMMemoryOrchestrator, MockLLM,
        )
        # SIM_LLM_BACKEND env var picks the adapter:
        #   unset/'mock' -> MockLLM (default, zero external deps)
        #   'ollama'     -> OllamaLLM(model=$SIM_LLM_MODEL or 'llama3.2:3b')
        # See sim/llm_adapters.py + Phase 3.3 design doc.
        llm_backend = os.environ.get("SIM_LLM_BACKEND", "mock").lower()
        if llm_backend == "ollama":
            try:
                from sim.llm_adapters import OllamaLLM
                model = os.environ.get("SIM_LLM_MODEL", "llama3.2:3b")
                llm_callable = OllamaLLM(model=model, verbose=False)
            except ImportError as e:
                # Fall back to MockLLM if openai isn't installed; the
                # 'connection failure' is preferable to a 500.
                llm_callable = MockLLM()
        else:
            llm_callable = MockLLM()

        mem = BridgeMemory(
            lineage_name=req.lineage,
            mode=req.mode,
            auto_save=True,
            verbose=False,
        )
        # Force lineage load — _ensure_loaded triggers bridge load/train
        mem._lineage = lineage
        mem._ensure_loaded()
        orch = LLMMemoryOrchestrator(
            memory=mem,
            llm_callable=llm_callable,
            max_tool_iterations=5,
        )
        _LLM_ORCHESTRATORS[cache_key] = orch

    # Track tool calls for this turn by recording the conversation
    # length pre/post.
    pre_len = len(orch.conversation)
    response = orch.chat(req.message)
    turn_slice = orch.conversation[pre_len:]

    tool_call_summaries = []
    for t in turn_slice:
        if t.get("role") == "tool":
            content = t.get("content")
            # Build a short result summary (avoid serializing huge dicts)
            if isinstance(content, dict):
                if "error" in content:
                    summary = f"error: {content['error']}"
                elif "target_action" in content:
                    summary = (f"bound -> motor_{content['target_action']} "
                               f"(conf={content.get('confidence', 0):.2f})")
                elif "key" in content:
                    summary = f"forget('{content['key']}')"
                else:
                    summary = str(content)[:100]
            elif isinstance(content, list) and content:
                # recall/speak return list of {value/word, ...}
                top = content[0]
                if "value" in top:
                    summary = (f"top: {top.get('value', '?')} "
                                f"(conf={top.get('confidence', 0):.2f})")
                elif "word" in top:
                    summary = (f"top: '{top.get('word', '?')}' "
                                f"(sim={top.get('similarity', 0):.2f})")
                else:
                    summary = str(top)[:100]
            else:
                summary = str(content)[:100]
            tool_call_summaries.append({
                "name": t.get("name", "?"),
                "result_summary": summary,
            })

    n_turns = sum(1 for m in orch.conversation if m.get("role") == "user")
    return JSONResponse({
        "lineage_name": req.lineage,
        "mode": req.mode,
        "response": response,
        "tool_calls": tool_call_summaries,
        "conversation_length": len(orch.conversation),
        "n_turns": n_turns,
    })


@app.get("/api/llm-chat/{name}/transcript")
def llm_chat_transcript(name: str, mode: str = "tier1") -> JSONResponse:
    """Return the cached conversation transcript for a lineage+mode.

    Useful for refreshing the frontend chat panel after a reload. 404
    if no orchestrator has been instantiated for (name, mode) yet.
    """
    cache_key = (name, mode)
    orch = _LLM_ORCHESTRATORS.get(cache_key)
    if orch is None:
        raise HTTPException(
            404,
            f"no LLM chat session active for ({name}, {mode}). "
            f"POST /api/llm-chat to start one.",
        )
    # Filter conversation to user/assistant only for the UI (tool
    # turns are internal). Each entry: {"role", "content"}.
    visible = []
    for m in orch.conversation:
        if m.get("role") in ("user", "assistant"):
            visible.append({
                "role": m["role"],
                "content": m.get("content", ""),
            })
    return JSONResponse({
        "lineage_name": name,
        "mode": mode,
        "messages": visible,
        "n_turns": sum(1 for m in visible if m["role"] == "user"),
        "total_messages": len(orch.conversation),
    })


@app.post("/api/llm-chat/{name}/reset")
def llm_chat_reset(name: str, mode: str = "tier1") -> JSONResponse:
    """Clear the cached orchestrator for (name, mode).

    Frees the bridge + conversation. Next POST /api/llm-chat will
    reload from lineage.
    """
    cache_key = (name, mode)
    existed = cache_key in _LLM_ORCHESTRATORS
    _LLM_ORCHESTRATORS.pop(cache_key, None)
    return JSONResponse({"reset": existed, "lineage_name": name, "mode": mode})


# ─── Brain chat — the INTERACT centerpiece (2026-06-23) ──────────────────
# Talk to a DEVELOPED brain (the real conversational agent), NOT the
# deprecated MockLLM↔BridgeMemory pattern-matcher above. Backed by
# `research/runners/brain_chat_tui.ChatBrain` (reuse-by-import, NO sim/
# edit): GATE (spiking recall + the no-confab MOAT) → CONSTRAIN+VERIFY
# fluent render → (answer, abstained).
#
# Like _LLM_ORCHESTRATORS, the ChatBrain is session-cached + kept WARM:
# the first load (and, with the qwen renderer, the model warm-up) is
# expensive and must be paid once per session. The cache key is
# (session, brain, renderer) so switching brain/renderer rebuilds.
#
# GPU policy: the qwen renderer needs cupy + CUDA torch. On a GPU-less
# host (or when another build owns the GPU) the endpoint defaults to the
# `stub`/`raw` renderer — the moat + recall are CPU; only fluent surface
# form needs the GPU. So the console works out of the box, GPU-light.

_BRAIN_CHATS: dict[tuple[str, str, str], object] = {}
# Parallel cache of the RICH composer (the multi-sentence grounded path)
# keyed identically to _BRAIN_CHATS. The RichAnswerComposer WRAPS the warm
# ChatBrain (reuse-by-import, NO sim/ edit): on rich=True one turn becomes a
# SUBSTANTIVE multi-sentence reply (direct recall + multi-hop chain +
# elaboration), each sentence VERIFY-checked against the brain so the
# no-confab moat EXTENDS to multi-sentence. Built lazily on first rich turn
# (it carries discourse-thread state so 'tell me more' walks forward).
_BRAIN_RICH: dict[tuple[str, str, str], object] = {}

# ─── AFFECT / EMOTION (Gate-B, 2026-08-12): the brain's live MOOD colors the production turn ─────────────────
# The mood is a persistent spiking STATE read NEURALLY off the co-resident graded-affect ladder
# (`research.runners.affect_production_organ`, reuse-by-import, NO sim/ edit). It colors WHAT the brain volunteers
# (forthcomingness: how many gate-matched, moat-verified facts) AND HOW it phrases them (the fluent mouth's PROSE
# is warmer/curter). Default-ON; `BRAIN_AFFECT=0` -> the byte-identical oracle. The moat/recall/abstain paths are
# UNCHANGED (affect only colors an ALREADY-matched answer; it can never manufacture a fact or flip an abstain).
# Per-session mood STATE (EMA of the appraised valence/arousal), keyed identically to the ChatBrain cache. Held
# host-side; the NEURAL read (the ladder differential) is the load-bearing spiking part. Cleared on reset.
_SESSION_MOOD: dict[tuple[str, str, str], dict] = {}
_MOOD_EMA_DECAY = 0.4   # a strong induction turn dominates; a neutral turn (0 affective hits) HOLDS the prior mood


def _get_affect_organ():
    """The process-shared spiking affect organ (built once, ~1s on the process backend)."""
    from research.runners.affect_production_organ import get_organ
    return get_organ(seed=42)


def _get_comprehension_organ():
    """The process-shared spiking comprehension monitor (built once; the co-resident SpikingRoleCompetition)."""
    from research.runners.comprehension_production_organ import get_organ
    return get_organ(seed=42)


def _brain_vocab(chat) -> set:
    """The set of words the brain KNOWS (agents ∪ actions ∪ patients of its stored facts), lowercased. Used to
    scope the comprehension monitor: a real word the brain knows but that is not in the toy cue lexicon is OUT of
    the monitor's competence (passed through), while a genuinely-unknown token is OOV (judged)."""
    try:
        vocab = set()
        for s in (getattr(chat, "agents_set", None), getattr(chat, "actions_set", None),
                  getattr(chat, "patients_set", None)):
            if s:
                vocab |= {str(w).lower() for w in s}
        return vocab
    except Exception:
        return set()


def _update_session_mood(cache_key: tuple, appr: dict) -> dict:
    """EMA-update this session's mood from the message appraisal. A message with NO strongly-affective words
    (n_hits==0) HOLDS the prior mood (cross-turn persistence); an affective message moves it."""
    m = _SESSION_MOOD.get(cache_key) or {"valence": 0.0, "arousal": 0.0}
    if int(appr.get("n_hits", 0)) > 0:
        d = _MOOD_EMA_DECAY
        m = {"valence": d * m["valence"] + (1.0 - d) * float(appr["valence"]),
             "arousal": d * m["arousal"] + (1.0 - d) * float(appr["arousal"])}
    _SESSION_MOOD[cache_key] = m
    return m


def _default_brain_renderer() -> str:
    """Pick the out-of-box renderer: `qwen` only when a CUDA GPU is
    actually available AND the cupy backend is selected; else the GPU-free
    `stub`. Keeps the console GPU-light by default (another build may own
    the GPU; the chat endpoint must not contend for it unasked)."""
    backend = os.environ.get("SIM_BACKEND", "").lower()
    if backend and backend != "cupy":
        return "stub"
    try:
        import torch  # noqa: PLC0415
        if torch.cuda.is_available():
            return "qwen"
    except Exception:
        pass
    return "stub"


def _brain_rich_default() -> bool:
    """The production default for /api/brain-chat when a request OMITS `rich`.

    Production default is the FLUENT multi-sentence path (the RichAnswerComposer:
    multi-fact recall + multi-hop chain + elaboration, each sentence VERIFY-checked
    so the no-confab moat extends to multi-sentence). This flips the historical
    single-SVO default (2026-08-12, production-integration: the integrated
    production turn must be fluent, not a one-line template).

    ESCAPE / kill-switch: ``BRAIN_RICH=0`` (or false/no/off) in the environment
    forces the OLD single-SVO path globally for callers that don't specify `rich`.
    A per-request ``rich=False`` still overrides this to the single-SVO path, and
    ``rich=True`` still overrides to the fluent path — this only sets the default
    when the field is absent. NOTE: fluent PROSE additionally needs the qwen mouth
    (SIM_BACKEND=cupy + CUDA torch); GPU-free hosts get the multi-sentence TEMPLATE
    stub, which is still multi-sentence + moat-gated but not prose."""
    v = os.environ.get("BRAIN_RICH")
    if v is None:
        return True
    return v.strip().lower() not in ("0", "false", "no", "off", "")


def _pin_bridge_backend() -> None:
    """Re-assert the global ``sim.backend`` cache to whatever backend
    ``sim.bridge`` bound at import (``sim.bridge._backend_name``).

    WHY (the live single-fact bug, 2026-06-24): ``sim.bridge`` binds its
    module-level ``cp`` ONCE at import (cupy on a GPU box), so EVERY bridge
    state array (``cp_external_input_current``, ``cp_rf_spike_step`` …) is cupy.
    But ``get_backend()`` is a STICKY PROCESS-GLOBAL cache that any numpy-CPU
    code path elsewhere in this long-running server can flip to numpy (e.g. an
    imported runner calling ``get_backend("numpy")``). Once flipped, the chat
    path breaks pervasively: a numpy ``cur`` written into a cupy
    ``cp_external_input_current[:]`` raises cupy's "non-scalar numpy.ndarray
    cannot be used for fill", and ``sim.backend.to_host`` becomes a no-op
    passthrough that hands a cupy array to ``np.asarray`` ("Implicit conversion
    to a NumPy array is not allowed"). Pinning the cache back to the bridge's
    ACTUAL backend at the start of each chat call makes the global consistent
    with the cupy bridge state — fixing all those mismatches at the source
    rather than patching every call site. No-op when already consistent / on a
    genuine numpy server. NO ``sim/`` edit (reads a public-ish module attr)."""
    try:
        import sim.bridge as _sb  # noqa: PLC0415
        want = getattr(_sb, "_backend_name", None)
        if not want:
            return
        # CRITICAL: ``get_backend(None)`` re-reads ``SIM_BACKEND`` on EVERY call and
        # the env ALWAYS wins over the cache (sim/backend.py). So if the launch env
        # has ``SIM_BACKEND=numpy`` while the bridge bound cupy (the live server's
        # state), re-asserting the cache alone is undone by the next bare
        # ``get_backend()``. Align the ENV to the bridge's real backend so every
        # ``get_backend()`` / ``to_host`` in the chat path agrees with the cupy
        # bridge arrays. (We only ever STRENGTHEN to the bridge's actual backend;
        # if they already agree this is a no-op.)
        if os.environ.get("SIM_BACKEND", "").lower() != want:
            os.environ["SIM_BACKEND"] = want
        from sim.backend import get_backend as _gb  # noqa: PLC0415
        _xp, _cur = _gb()
        if _cur != want:
            _gb(want)   # re-assert the cache too (e.g. "cupy")
    except Exception:
        pass  # never let backend-pinning crash a chat turn


# A PROCESS-WIDE warm QwenRenderer singleton. The off-bridge Qwen-0.5B model
# load (from_pretrained → GPU + the spiking-op calibration) is the ~58s one-time
# cost; a fresh `QwenRenderer()` re-pays it every time. Building it ONCE and
# reusing the SAME instance across every (session, brain) ChatBrain means only
# the FIRST construction (the startup warm, below) pays the model load — every
# later brain that uses the qwen renderer reuses the already-loaded model. The
# renderer is stateless across `render_svo` calls (it re-prompts per fact), so a
# shared instance is safe to reuse for any brain/session. Guarded by a lock so a
# concurrent first-turn + startup-warm build the model only once.
import threading as _threading

_WARM_QWEN_RENDERER: object | None = None
_WARM_QWEN_LOCK = _threading.Lock()


def _get_warm_qwen_renderer():
    """Return the shared warm QwenRenderer, building it ONCE (the ~58s model
    load) under a lock. Reused by every ChatBrain that uses the qwen renderer
    so the model loads a single time per server process."""
    global _WARM_QWEN_RENDERER
    if _WARM_QWEN_RENDERER is not None:
        return _WARM_QWEN_RENDERER
    with _WARM_QWEN_LOCK:
        if _WARM_QWEN_RENDERER is None:   # double-checked: another thread may have built it
            from research.runners.brain_chat_tui import QwenRenderer
            _WARM_QWEN_RENDERER = QwenRenderer(seed=42)   # the heavy model load (paid once)
        return _WARM_QWEN_RENDERER


def _build_chat_brain(brain: str, renderer: str):
    """Construct a `ChatBrain` for the given brain source + renderer.

    brain: 'tiny-demo' (GPU-free fallback), 'self-knowledge' (the learned
        self-facts codes), or a developed-brain bundle DIRECTORY path
        (brain.json + grounded_codes.npz + facts.json).
    renderer: 'qwen' (off-bridge Qwen, GPU) / 'stub' (template-stub,
        GPU-free) / 'raw' (the brain's own raw triples, no LLM).
    """
    # Make the global backend cache consistent with the bridge's actual arrays
    # BEFORE building any bridge in the chat brain (the cupy/numpy-flip bug).
    _pin_bridge_backend()
    from research.runners.brain_chat_tui import (
        ChatBrain, StubRenderer,
        _build_tiny_demo, _load_self_knowledge, DEFAULT_SELF_ALIASES,
        _SK_CODES, _SK_CURRICULUM,
    )
    from research.runners.developed_brain_io import (
        is_developed_brain_bundle, load_developed_brain,
    )

    # A developed-brain bundle id from /api/brains is a repo-RELATIVE path
    # (e.g. 'bridges/developed/develop_gpu/brain'). The server may be launched
    # from any CWD, so resolve a relative bundle path against REPO_ROOT before
    # the is_developed_brain_bundle() check (an absolute path is used as-is).
    def _resolve_bundle(b: str) -> str:
        p = Path(b)
        if not p.is_absolute():
            cand = REPO_ROOT / b
            if (cand / "brain.json").exists():
                return str(cand)
        return b

    # --- load the brain (mirrors brain_chat_tui.load_brain precedence) ---
    if brain in ("", "tiny-demo", "tiny", "demo"):
        # BRAIN_COMPOSER_KIND (default 'rf' = the numpy fast-path recall): set 'onebrain' for the GENUINELY-SPIKING
        # recall (resonate-and-fire per query, runtime new-word LEARN via vocab_headroom). The onebrain build is
        # ~180s (speed secondary); it is the brain-based-only recall the mission requires. Env-gated so the flip to
        # spiking-by-default is one setting, and it can be verified via the real endpoint before becoming the default.
        _ck = os.environ.get("BRAIN_COMPOSER_KIND", "onebrain")
        agent, aliases, _n = _build_tiny_demo(42, use_multiturn=True,
                                              enable_neural_render=False, composer_kind=_ck)
        source = "tiny-demo"
    elif brain in ("self-knowledge", "self_knowledge", "self"):
        agent, aliases, _n = _load_self_knowledge(
            _SK_CODES, _SK_CURRICULUM, 42, True, False)
        source = "self-knowledge"
    elif is_developed_brain_bundle(_resolve_bundle(brain)):
        bundle = _resolve_bundle(brain)
        agent, manifest = load_developed_brain(bundle, use_multiturn=True,
                                               enable_neural_render=False)
        aliases = set(manifest.get("self_aliases") or []) | set(DEFAULT_SELF_ALIASES)
        source = f"developed-brain:{brain}"
    else:
        raise HTTPException(
            400,
            f"brain {brain!r} is neither 'tiny-demo'/'self-knowledge' nor a "
            f"developed-brain bundle directory (needs a brain.json manifest). "
            f"Save one with developed_brain_io.save_developed_brain.",
        )

    # --- the fluent renderer (GPU-free unless qwen explicitly requested) ---
    rname = (renderer or "stub").lower()
    if rname == "raw":
        rend = None
    elif rname == "qwen":
        rend = _get_warm_qwen_renderer()    # shared warm model (loaded once); needs cupy + CUDA torch
    else:
        rend = StubRenderer()               # GPU-free default
    return ChatBrain(agent, self_aliases=aliases, renderer=rend), source


def _get_rich_composer(cache_key: tuple, chat):
    """Build (once) + cache a RichAnswerComposer wrapping the warm ChatBrain.

    The composer is keyed identically to the ChatBrain so each session/brain/
    renderer gets its own multi-sentence discourse thread (so 'tell me more'
    elaborates forward within that conversation).

    PLANNER (2026-08-12): the webapp now uses the NEURAL dlPFC planner
    (``neural_planner=True`` below) — speed is SECONDARY (mission non-negotiable),
    so both the elaboration CONTENT and the relevance-by-latency ORDERING run on
    the substrate. It builds + steps a per-topic ``SimulationBridge`` ON THE GPU
    every turn (the earlier host-planner default was chosen only for latency and
    has been retired). The direct gate (the no-confab MOAT), the role-chase chain
    hop, and the per-sentence VERIFY are unchanged by the planner choice (the
    planner only steers WHICH grounded facts to bring up + in what order), so the
    moat is identical either way. This is what the DEFAULT ``/api/brain-chat``
    turn now runs (``_brain_rich_default`` is ON) — a fluent MULTI-SENTENCE reply.
    """
    rich = _BRAIN_RICH.get(cache_key)
    if rich is None:
        from research.runners.rich_answer_composer import RichAnswerComposer
        rich = RichAnswerComposer(
            chat, max_sentences=4, neural_planner=True,    # SPIKING dlPFC ordering (relevance_by_latency) — speed is
                                                           # SECONDARY (mission non-negotiable); both content (elaborate)
                                                           # AND ordering are now on the substrate. Was False for latency.
            planner_seed=getattr(getattr(chat, "inner", None), "seed", 42),
        )
        _BRAIN_RICH[cache_key] = rich
    return rich


# ─── /api/brains — LIST the brains the console can talk to (B2, 2026-06-24) ─
# The Interact-tab brain-selector dropdown is populated from this. It scans
# `bridges/developed/*` for developed-brain BUNDLES (any dir with a brain.json
# manifest — the `{brain, day_N}` layout the develop loop's save_bundle_root
# writes) + always includes the two built-in sources the chat endpoint loads
# (the GPU-free 'tiny-demo' + the learned 'self-knowledge' codes). Each entry's
# `id` is EXACTLY what /api/brain-chat's `brain` field expects (so picking a
# brain in the dropdown needs no further resolution server-side — the chat
# endpoint's existing precedence resolves 'tiny-demo'/'self-knowledge'/a bundle
# DIR path verbatim). Reuse-by-import, NO sim/ edit.

DEVELOPED_DIR = REPO_ROOT / "bridges" / "developed"
# the self-knowledge codes + curriculum (mirrors brain_chat_tui._SK_CODES/_SK_CURRICULUM)
_SK_CODES_PATH = REPO_ROOT / "research" / "findings" / "raw" / "_self_knowledge_grounded_codes.json"
_SK_CURRICULUM_PATH = REPO_ROOT / "research" / "findings" / "raw" / "_curriculum_self_knowledge.json"


def _read_json_safe(path: Path):
    """Read a JSON file, returning None on any error (missing / malformed)."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError, ValueError):
        return None


def _scan_developed_bundles() -> list[dict]:
    """Find every developed-brain BUNDLE under bridges/developed/ (a dir with a
    brain.json manifest). The develop loop writes `bridges/developed/<lineage>/
    {brain, day_<N>}`, so we look at depth 1 (a bundle directly under
    developed/) AND depth 2 (a bundle under a lineage subdir). Each bundle's
    `id` is its path RELATIVE to the repo root (forward slashes) — exactly what
    /api/brain-chat resolves via is_developed_brain_bundle(brain)."""
    out: list[dict] = []
    if not DEVELOPED_DIR.exists():
        return out
    seen: set[str] = set()

    def _add(d: Path):
        manifest_path = d / "brain.json"
        if not manifest_path.exists():
            return
        rel = d.resolve()
        key = str(rel)
        if key in seen:
            return
        seen.add(key)
        m = _read_json_safe(manifest_path) or {}
        meta = m.get("metadata") or {}
        # the day: per-day bundles carry metadata.day; the final brain carries
        # metadata.final_day; otherwise unknown.
        day = meta.get("day", meta.get("final_day"))
        try:
            id_path = d.resolve().relative_to(REPO_ROOT).as_posix()
        except ValueError:
            id_path = str(d.resolve())
        # a readable label: <lineage>/<bundle> (e.g. develop_gpu/day_3) + day tag
        try:
            label_rel = d.resolve().relative_to(DEVELOPED_DIR).as_posix()
        except ValueError:
            label_rel = d.name
        label = label_rel
        if day is not None:
            label = f"{label_rel} (day {day})"
        out.append({
            "id": id_path,
            "label": label,
            "kind": "developed-bundle",
            "path": id_path,
            "day": day,
            "n_facts": m.get("n_facts"),
            "vocab_size": len(m.get("vocab") or []),
            "n_grounded_codes": m.get("n_grounded_codes"),
            "composer_kind": m.get("composer_kind"),
            "seed": m.get("seed"),
            "provenance": meta.get("provenance"),
            "loadable": True,
        })

    # depth 1 (a bundle directly under developed/) + depth 2 (under a lineage)
    for child in sorted(DEVELOPED_DIR.iterdir()):
        if not child.is_dir():
            continue
        _add(child)
        for grandchild in sorted(child.iterdir()):
            if grandchild.is_dir():
                _add(grandchild)
    # sort: final 'brain' bundles first within a lineage, then day_N ascending,
    # then everything else — stable + readable in the dropdown.
    def _sort_key(e):
        name = e["id"].rsplit("/", 1)[-1]
        if name == "brain":
            return (0, e["id"])
        if name.startswith("day_"):
            try:
                return (1, int(name[4:]))
            except ValueError:
                return (2, e["id"])
        return (3, e["id"])
    out.sort(key=_sort_key)
    return out


@app.get("/api/brains")
def list_brains() -> JSONResponse:
    """List the brains the Interact tab can talk to (the dropdown source).

    Returns: {"brains": [...], "n_brains": int, "developed_root": str|null,
              "default": str}
    Each entry: {id, label, kind, loadable, n_facts?, vocab_size?, day?, ...}.
      - kind 'builtin'           → 'tiny-demo' (always loadable, code-only) and
                                   'self-knowledge' (loadable iff the learned
                                   codes JSON exists).
      - kind 'developed-bundle'  → a save_developed_brain bundle under
                                   bridges/developed/ (id = repo-relative path).

    The `id` of every loadable entry is EXACTLY what POST /api/brain-chat's
    `brain` field expects, so the dropdown sends it verbatim.
    """
    brains: list[dict] = []

    # --- built-in: the GPU-free tiny-demo (always available; code-only) ---
    brains.append({
        "id": "tiny-demo",
        "label": "Tiny demo (GPU-free)",
        "kind": "builtin",
        "n_facts": 5,
        "vocab_size": 8,
        "loadable": True,
        "note": "A handful of built-in self + object facts; no GPU. Good for a quick smoke.",
    })

    # --- built-in: the learned self-knowledge brain (loadable iff codes exist) ---
    sk_codes = _read_json_safe(_SK_CODES_PATH)
    sk_cur = _read_json_safe(_SK_CURRICULUM_PATH)
    sk_loadable = sk_codes is not None and sk_cur is not None
    sk_entry = {
        "id": "self-knowledge",
        "label": "Self-knowledge (learned codes)",
        "kind": "builtin",
        "loadable": bool(sk_loadable),
    }
    if sk_cur is not None:
        n_facts = len(sk_cur.get("facts") or []) + len(sk_cur.get("attribute_facts") or [])
        sk_entry["n_facts"] = n_facts
    if sk_codes is not None:
        sk_entry["n_grounded_codes"] = len(sk_codes.get("grounded_codes") or {})
        sk_entry["seed"] = sk_codes.get("seed")
        sk_entry["n_days"] = sk_codes.get("n_days")
        if sk_codes.get("learn_fidelity_mean") is not None:
            sk_entry["learn_fidelity_mean"] = round(float(sk_codes["learn_fidelity_mean"]), 4)
    if not sk_loadable:
        missing = []
        if sk_codes is None:
            missing.append("_self_knowledge_grounded_codes.json")
        if sk_cur is None:
            missing.append("_curriculum_self_knowledge.json")
        sk_entry["note"] = ("not yet developed — run _self_knowledge_demo to produce "
                            + " + ".join(missing) + " (it answers on seed codes meanwhile).")
        # the chat endpoint still loads 'self-knowledge' (falls back to seed
        # codes when the file is absent), so it remains selectable.
        sk_entry["loadable"] = True
    brains.append(sk_entry)

    # --- developed-brain bundles under bridges/developed/ ---
    brains.extend(_scan_developed_bundles())

    return JSONResponse({
        "brains": brains,
        "n_brains": len(brains),
        "developed_root": (str(DEVELOPED_DIR.relative_to(REPO_ROOT).as_posix())
                           if DEVELOPED_DIR.exists() else None),
        "default": "tiny-demo",
    })


class BrainChatRequest(BaseModel):
    """One conversational turn against a developed brain."""
    session: str = "default"
    message: str
    # 'tiny-demo' / 'self-knowledge' / a developed-brain bundle dir path.
    brain: str = "tiny-demo"
    # 'qwen' (GPU) / 'stub' (GPU-free) / 'raw'; None -> auto (GPU-light).
    renderer: str | None = None
    # RICH-answer mode: when True, the turn returns a SUBSTANTIVE
    # multi-sentence GROUNDED reply (direct recall + multi-hop chain +
    # elaboration), each sentence VERIFY-checked against the brain so the
    # no-confab moat EXTENDS to multi-sentence. Routed through a
    # session-cached RichAnswerComposer wrapping the warm ChatBrain. A bare
    # 'tell me more' / 'why?' follow-up elaborates the held topic further.
    # rich=False keeps the single-fact verified answer (the host path).
    #
    # DEFAULT (2026-08-12, production-integration): the field is now TRI-STATE
    # (None = unspecified). When a caller OMITS `rich`, the turn takes the
    # production default from `_brain_rich_default()` — the FLUENT multi-sentence
    # path (BRAIN_RICH env, default ON). An EXPLICIT `rich=False` in the body
    # still forces the old single-SVO path byte-identically (the per-request
    # escape); `BRAIN_RICH=0` in the environment forces it globally (the
    # production kill-switch). Explicit `rich=True` always takes the fluent path.
    rich: bool | None = None
    # If True, drop the cached ChatBrain for this (session, brain,
    # renderer) before answering (rebuilds — for 'start fresh').
    reset: bool = False


@app.post("/api/brain-chat")
def brain_chat(req: BrainChatRequest) -> JSONResponse:
    """One turn talking to a DEVELOPED brain (the INTERACT centerpiece).

    First call per (session, brain, renderer) builds + warms a ChatBrain
    (a few seconds for the tiny-demo; longer for a real bundle or the qwen
    warm-up). Subsequent turns reuse the warm cache.

    Returns:
        {
          "answer": str,            # the verified fluent answer, or
                                    # "I don't know about that." on abstain
          "abstained": bool,        # the no-confab MOAT fired (the brain
                                    # was never taught this) — shown distinctly
          "recalled_svo": [a, v, p] | null,   # the stored fact the gate hit
          "verified": bool,         # the render re-parsed back to the fact
          "renderer": str,          # which renderer produced the surface form
          "brain": str, "source": str,
          # B3 per-turn "brain activity" (read-only; null for the rate composer
          # or when the matcher abstained before any spiking query):
          "activity": {
            "roles": [{"role": "agent", "word": "dog", "confidence": 0.97,
                       "cue": true, "asserted": "dog"}, ...],  # decoded chips
            "matched_fact_index": int | null,  # which engram block answered
                                               # (null = abstained -> moat fired)
            "n_facts_scanned": int,            # how many stored facts scanned
            "abstained": bool,
            "rf": {"n_readout_neurons": int|null,
                   "frac_fired": float|null,   # cp_rf_fired.mean() over readouts
                   "mean_magnitude": float|null},  # mean |Z| (recovery strength)
            "composer": "rf" | "onebrain",
          } | null,
          # rich=True only (the multi-sentence grounded path):
          "rich": bool,             # whether the rich path produced this turn
          "n_sentences": int,       # number of brain-sourced sentences
          "supporting_facts": [[a, v, p], ...],   # the verified SVOs behind it
          "followup": bool,         # the turn was a 'tell me more'/'why?'
        }
    """
    renderer = (req.renderer or _default_brain_renderer()).lower()
    # A numpy-CPU code path elsewhere in this long-running server can have flipped
    # the global backend cache away from the bridge's cupy arrays between turns;
    # re-assert it here so gate()/render()/the rich path run consistently (the
    # cupy/numpy-flip bug, 2026-06-24).
    _pin_bridge_backend()
    cache_key = (req.session, req.brain, renderer)
    if req.reset:
        _BRAIN_CHATS.pop(cache_key, None)
        _BRAIN_RICH.pop(cache_key, None)   # drop the rich discourse thread too
        _SESSION_MOOD.pop(cache_key, None)  # clear the mood STATE (reset-between-topics)

    chat = _BRAIN_CHATS.get(cache_key)
    source = None
    if chat is None:
        try:
            chat, source = _build_chat_brain(req.brain, renderer)
        except HTTPException:
            raise
        except Exception as e:  # a missing model / bad bundle -> a clean 400
            raise HTTPException(
                400, f"failed to load brain {req.brain!r} with renderer "
                     f"{renderer!r}: {type(e).__name__}: {e}")
        chat._brain_chat_source = source  # type: ignore[attr-defined]
        _BRAIN_CHATS[cache_key] = chat
    source = getattr(chat, "_brain_chat_source", source)

    # B3 per-turn "brain activity": flip the composer's READ-ONLY trace flag ON (default-off in the composer; a
    # post-construction attribute flip only GATES the read-only `last_trace` recording, so it stays answer-identical +
    # the no-confab moat is unchanged). After the gate runs the spiking recall (`what_does` -> composer.query_patient),
    # the composer's `last_trace` holds what the brain DID this turn (decoded role chips + match confidence, the matched
    # engram block + how many scanned, a scalar RF firing/|Z| gauge). The endpoint reads it and attaches it as
    # `activity`. Guarded for the rate composer / any composer without `last_trace` (-> activity stays None).
    _composer = getattr(getattr(chat, "inner", None), "composer", None)
    if _composer is not None and hasattr(_composer, "last_trace"):
        try:
            _composer.trace = True   # read-only recording gate (default off = byte-identical for every other caller)
            # Reset the per-turn record BEFORE the gate runs: a turn that abstains
            # *before* the composer scan (e.g. a parse-fail on unknown words) never
            # touches last_trace, so without this it would return the STALE trace from
            # a prior turn on the same cached composer. Cleared -> such a turn -> None.
            _composer.last_trace = None
        except Exception:
            pass

    def _read_activity():
        """Read the composer's per-turn trace AFTER the gate (read-only). None when the composer doesn't trace (the
        rate composer / an external composer) or nothing was recorded (e.g. the matcher abstained before any query)."""
        c = getattr(getattr(chat, "inner", None), "composer", None)
        return getattr(c, "last_trace", None) if c is not None else None

    msg = (req.message or "").strip()
    if not msg:
        raise HTTPException(422, "message must be non-empty")

    rname = chat.renderer.name if getattr(chat, "renderer", None) is not None else "raw brain triples"

    # ── AFFECT / EMOTION coloring (Gate-B, 2026-08-12) ──────────────────────────────────────────────────────
    # Read the brain's live MOOD off the co-resident spiking graded-affect ladder and prepare (a) a CONTENT plan
    # (forthcomingness — how many gate-matched facts to volunteer) and (b) a prose MANNER template (warmer/curter
    # phrasing). Default-ON; `BRAIN_AFFECT=0` -> fully skipped (byte-identical oracle). The moat/recall/abstain
    # paths below are UNCHANGED — affect only colors an already-matched answer. `affect_info` is attached to the
    # response for the debug trace (the tone TOKEN lives here ONLY, never on the user-facing surface).
    affect_info = None
    affect_plan = None
    try:
        import research.runners.affect_production_organ as _AO
        _affect_on = _AO.affect_enabled()
    except Exception:
        _AO = None
        _affect_on = False
    if _affect_on:
        try:
            organ = _get_affect_organ()
            appr = _AO.appraise_text(msg)
            mood = _update_session_mood(cache_key, appr)
            lesion = _AO.affect_lesioned()
            read = organ.read_differential(mood["valence"], lesion=lesion)
            diff = float(read["differential"])
            level = _AO.tone_level(diff)
            affect_plan = _AO.content_plan(level)
            manner_tmpl = _AO.manner_template_for(level)
            tol = _AO.SA.LADDER_NEUTRAL_TOL
            affect_info = {
                "on": True, "lesioned": bool(lesion), "composer": "onebrain",
                "differential": diff,
                "valence_sign": ("+" if diff > tol else ("-" if diff < -tol else "0")),
                "tone_level": int(level),
                # DEBUG readout ONLY — the tone TOKEN is a trace field, never prepended to the answer surface.
                "tone_token": _AO.SA._graded_tone_token(level),
                "forthcomingness": affect_plan,
                "manner_template": manner_tmpl,
                "appraisal_valence": float(mood["valence"]), "appraisal_arousal": float(mood["arousal"]),
                "appraisal_hits": appr.get("words", []),
                "pos_rate": float(read["pos_rate"]), "neg_rate": float(read["neg_rate"]),
            }
            # install the mood-conditioned renderer (once per ChatBrain) + set THIS turn's manner template.
            base_r = getattr(chat, "renderer", None)
            if base_r is not None and not isinstance(base_r, _AO.MoodConditionedRenderer):
                chat.renderer = _AO.MoodConditionedRenderer(base_r)
            wrapped = getattr(chat, "renderer", None)
            if isinstance(wrapped, _AO.MoodConditionedRenderer):
                wrapped.manner = manner_tmpl
        except Exception as _e:  # never let affect crash a turn — degrade to the un-colored answer
            affect_info = {"on": True, "error": f"{type(_e).__name__}: {_e}"}
            affect_plan = None

    # HONEST inner-state READ-OUT (Wire-2): 'how do you feel' -> answered by the live valence differential (a
    # functional read, never a phenomenal claim). Gated on an explicit feel-query so it never hijacks a recall turn.
    if _affect_on and affect_info is not None and "error" not in affect_info and _AO.is_feel_query(msg):
        _m = _SESSION_MOOD.get(cache_key) or {"valence": 0.0, "arousal": 0.0}
        return JSONResponse({
            "answer": _AO.feel_readout(affect_info["differential"], _m["valence"], _m["arousal"]),
            "abstained": False, "recalled_svo": None, "verified": True,
            "renderer": rname, "brain": req.brain, "source": source,
            "rich": False, "activity": None, "affect": affect_info, "inner_state_readout": True,
        })

    # ── COMPREHENSION MEASUREMENT gate (Gate-B, D4, 2026-08-12) ─────────────────────────────────────────────
    # BEFORE acting on an incoming TRANSITIVE ASSERTION, read a genuinely-SPIKING signal of whether the brain's
    # role-binding RESOLVED (the co-resident `SpikingRoleCompetition` sel-pool margin off cp_firing_states,
    # reuse-by-import from `comprehension_production_organ`, 6/6-GO D4 de-risk). On a LOW margin (OOV / content-
    # ambiguous input the substrate could not comprehend), the brain honestly ABSTAINS ("my role-binding didn't
    # resolve — I didn't follow that") instead of silently ingesting it — this STRENGTHENS the no-confab moat.
    # SCOPE (non-regressive by construction): fires ONLY on a competent 3-content-token transitive (fully cue-
    # covered OR genuinely OOV); questions / self-queries / anaphora / open-ended / real-but-untabled vocab are
    # OUT OF SCOPE -> byte-identical, unchanged. GUARD: never abstains on a (agent,action) the brain KNOWS
    # (`what_does` truthy) -> a known fact is honored (and D2 surprise, below, checks the patient instead).
    # Default-ON; `BRAIN_COMPREHENSION_GATE=0` -> fully skipped (byte-identical oracle).
    comprehension_info = None
    try:
        import research.runners.comprehension_production_organ as _CO
        _comp_on = _CO.comprehension_enabled()
    except Exception:
        _CO = None
        _comp_on = False
    if _comp_on:
        try:
            corg = _get_comprehension_organ()
            cj = corg.judge(msg, brain_vocab=_brain_vocab(chat), lesion=_CO.comprehension_lesioned())
            if cj is not None:
                comprehension_info = dict(cj)
                a_c, v_c, p_c = cj["svo"]
                # the brain KNOWS this (agent,action)? -> a spiking recall exists -> honor it (never abstain on a
                # known fact); the mismatch of the asserted patient is D2's job, not comprehension's.
                try:
                    known = bool(chat.inner.what_does(a_c, v_c))
                except Exception:
                    known = False
                comprehension_info["known_binding"] = bool(known)
                if (not cj["comprehended"]) and (not known):
                    # LOW margin + not a known binding -> the brain did not comprehend the roles -> honest abstain.
                    comprehension_info["abstained"] = True
                    return JSONResponse({
                        "answer": _CO.didnt_follow_message(cj["svo"]),
                        "abstained": True, "recalled_svo": None, "verified": False,
                        "renderer": rname, "brain": req.brain, "source": source,
                        "rich": False, "activity": None, "affect": affect_info,
                        "comprehension": comprehension_info, "not_understood": True,
                    })
        except Exception as _ce:  # never let the comprehension read crash a turn — degrade to the normal answer
            comprehension_info = {"on": True, "error": f"{type(_ce).__name__}: {_ce}"}

    # ── RICH path: a SUBSTANTIVE multi-sentence grounded reply ──────────
    # The RichAnswerComposer does its own GATE (direct recall + the moat) +
    # multi-hop chain + elaboration, VERIFY-checks each sentence, and carries
    # the discourse thread (so a 'tell me more' follow-up elaborates forward).
    #
    # `rich` is TRI-STATE (None = the caller omitted it): resolve the effective
    # mode from the production default (`_brain_rich_default()`, fluent-ON, env
    # BRAIN_RICH) UNLESS the request set it explicitly. So the DEFAULT turn is
    # fluent multi-sentence; an explicit `rich=False` escapes to the byte-
    # identical single-SVO path below; `rich=True` forces fluent.
    use_rich = req.rich if req.rich is not None else _brain_rich_default()
    if use_rich:
        try:
            rich = _get_rich_composer(cache_key, chat)
            # AFFECT content coloring (the genuine WHAT): set THIS turn's forthcomingness (how many gate-matched
            # facts to volunteer + elaboration depth) from the mood plan; restore after so the cache is unchanged.
            _saved_plan = None
            if affect_plan is not None:
                _saved_plan = (rich.max_sentences, rich.max_elaborations)
                rich.max_sentences = int(affect_plan["max_sentences"])
                rich.max_elaborations = int(affect_plan["max_elaborations"])
            try:
                r = rich.answer(msg)
            finally:
                if _saved_plan is not None:
                    rich.max_sentences, rich.max_elaborations = _saved_plan
        except Exception as e:
            raise HTTPException(500, f"rich chat turn failed: {type(e).__name__}: {e}")
        facts = [list(f) for f in r.get("facts", [])]
        # OPEN-ENDED GENERATION (#3E): a generated HYPOTHESIS turn returns NO supporting facts (a guess is not a
        # recalled fact) but carries `hypothesis`/`hypothesis_svo`. Surface those so the client can render the
        # guess distinctly + so the SVO the fluent prose asserts is checkable, WITHOUT reporting it as recalled.
        is_hyp = bool(r.get("hypothesis"))
        resp = {
            "answer": r["answer"],
            "abstained": bool(r["abstained"]),
            # the direct recall (the first supporting fact) is the gate hit,
            # surfaced for parity with the single-fact path's recalled_svo.
            # A hypothesis has no recalled fact -> null (it is a guess, not knowledge).
            "recalled_svo": facts[0] if facts else None,
            # every kept sentence is gate-sourced + verify-checked, so the
            # multi-sentence reply is verified-by-construction (unless it
            # abstained, in which case there is nothing to verify).
            "verified": (not r["abstained"]) and bool(facts),
            "renderer": rname,
            "brain": req.brain,
            "source": source,
            "rich": True,
            "n_sentences": int(r.get("n_sentences", 0)),
            "supporting_facts": facts,
            "followup": bool(r.get("followup", False)),
            # B3: what the brain DID this turn (the LAST spiking recall the rich gate ran), or null. The rich path
            # plans over multiple supporting facts; last_trace reflects the most recent query (the direct recall).
            "activity": _read_activity(),
            # AFFECT (Gate-B): the live mood that colored this turn's forthcomingness + prose manner (debug trace;
            # the tone token lives here, never on the surface). null when affect is disabled (BRAIN_AFFECT=0).
            "affect": affect_info,
            # COMPREHENSION (Gate-B, D4): the spiking role-binding read for an in-scope transitive (margin/threshold/
            # comprehended); null when out of scope or disabled (BRAIN_COMPREHENSION_GATE=0). A pass-through here.
            "comprehension": comprehension_info,
        }
        if is_hyp:
            # additive markers (present ONLY on a generated-hypothesis turn): the guess flag, the (a,v,p) the fluent
            # prose asserts (VERIFY re-parse target), and whether the fluent mouth (True) or the raw template
            # fallback (False) produced the surface. A non-hypothesis rich turn is byte-identical (no extra keys).
            resp["hypothesis"] = True
            resp["hypothesis_svo"] = list(r.get("hypothesis_svo") or [])
            resp["fluent_hypothesis"] = bool(r.get("fluent_hypothesis"))
        return JSONResponse(resp)

    # ── single-fact path (rich=False): GATE -> CONSTRAIN+VERIFY render ──
    # Peek the GATE so we can report the recalled fact (exactly what the TUI
    # smoke records), then render. gate() returns None on the moat.
    try:
        gate_svo = chat.gate(msg)
        if gate_svo is None:
            answer, abstained, verified = "I don't know about that.", True, False
        else:
            answer = chat.render(gate_svo)
            abstained = False
            # 'verified' = the render did NOT fall back to the raw triple
            # (ChatBrain marks an unverified render with this suffix).
            verified = "[unverified render" not in answer
    except Exception as e:
        raise HTTPException(500, f"chat turn failed: {type(e).__name__}: {e}")

    return JSONResponse({
        "answer": answer,
        "abstained": abstained,
        "recalled_svo": list(gate_svo) if gate_svo is not None else None,
        "verified": verified,
        "renderer": rname,
        "brain": req.brain,
        "source": source,
        "rich": False,
        # B3: what the brain DID this turn -- the decoded role chips + match confidence, which engram block answered
        # (or null -> abstained), and a scalar RF firing/|Z| gauge. Read-only of the spiking recall the gate already
        # ran (composer.query_patient); null for the rate composer / when the matcher abstained before any query.
        "activity": _read_activity(),
        # AFFECT (Gate-B): the live mood that colored this turn's prose manner (debug trace; single-fact path has
        # no forthcomingness lever, so only manner applies here). null when affect is disabled (BRAIN_AFFECT=0).
        "affect": affect_info,
        # COMPREHENSION (Gate-B, D4): the spiking role-binding read for an in-scope transitive; null out of scope.
        "comprehension": comprehension_info,
    })


class BrainChatResetRequest(BaseModel):
    """Reset a brain-chat session (no `message` required, unlike a turn)."""
    session: str = "default"
    brain: str = "tiny-demo"
    renderer: str | None = None


@app.post("/api/brain-chat/reset")
def brain_chat_reset(req: BrainChatResetRequest) -> JSONResponse:
    """Drop the cached ChatBrain for (session, brain, renderer) and clear
    its discourse buffer (a fresh conversation). Idempotent — reports
    whether a session existed."""
    renderer = (req.renderer or _default_brain_renderer()).lower()
    cache_key = (req.session, req.brain, renderer)
    existed = cache_key in _BRAIN_CHATS
    _BRAIN_CHATS.pop(cache_key, None)
    _BRAIN_RICH.pop(cache_key, None)   # drop the rich discourse thread too
    _SESSION_MOOD.pop(cache_key, None)  # clear the mood STATE (reset-between-topics)
    return JSONResponse({"reset": existed, "session": req.session,
                         "brain": req.brain, "renderer": renderer})


# ─── In-flight detached-run monitor (2026-05-01) ────────────────────────
# Detached runs (launched via PowerShell Start-Process to survive Claude
# restart) write a *.pid + *.log file under research/findings/raw/g11_bg/.
# This endpoint scans for those, tails the log, and reports progress.

import re as _re
# Embodied training progress markers. Matches BOTH the old format
# (`[ep N/M] correct_moves=K/L=X.X%`) emitted by text_eval_embodied
# and the curriculum format (`[P2 ep N/M] correct_moves=K/L=X.X%`)
# emitted by text_train_curriculum (the SWR-replay-aware runner).
_DETACHED_PROGRESS_RE = _re.compile(
    r"\[(?:P\d\s+)?ep\s+(\d+)/(\d+)\]\s+correct_moves=(\d+)/(\d+)=([\d.]+)%"
)
# Curriculum phase marker (line like "PHASE 2: Text I/O training - 100 episodes").
# Captured separately so the inflight panel can show which phase is
# currently running.
_CURRICULUM_PHASE_RE = _re.compile(
    # Phase 1/2 say "episodes", Phase 3 SWR says "events".
    r"^={3,}\s*PHASE\s+(\d+):\s+(.+?)\s+(\d+)\s+(?:episodes|events)",
    _re.MULTILINE,
)
# Phase 3 SWR replay progress. Logged as "[P3 SWR] 100/500 replayed"
# by text_train_curriculum's _run_swr_replay_phase. Older format
# `[swr ev 100/500]` is also accepted for backward compatibility.
_SWR_PROGRESS_RE = _re.compile(
    r"\[(?:P\d\s+)?[Ss][Ww][Rr](?:\s+ev)?\]?\s+(\d+)/(\d+)"
)
# Paired-stim runners (text_pfc_bypass_isolation, text_minimal_isolation)
# log progress as "[isolation] 100/400 events" or "[minimal-iso] 250/4000 events".
# Matches either format; reuses the swr_replay panel kind for display.
_PAIRED_STIM_PROGRESS_RE = _re.compile(
    r"\[(?:isolation|minimal-iso)\]\s+(\d+)/(\d+)\s+events"
)
_GENERIC_STEP_RE = _re.compile(
    r"step\s+(\d+)/(\d+)\s+pos=\((-?\d+),(-?\d+)\)\s+goal=\((-?\d+),(-?\d+)\)"
)
# continual_eval_suite progress markers (added 2026-05-09 after user reported
# "0% · no progress markers yet" for Phase 1.5 multi-seed).
# Start marker:  "--- Running benchmark: NAME ---"
# End marker:    "  [OK] NAME: score=0.87 pass=True (1911s)"  (or "[X]" for fail)
_CES_BENCH_START_RE = _re.compile(r"---\s*Running benchmark:\s*(\w+)\s*---")
_CES_BENCH_END_RE = _re.compile(
    r"\[(OK|X)\]\s+(\w+):\s+score=([-\d.]+)\s+pass=(True|False)\s+\((\d+)s\)"
)


def _check_pid_alive(pid: int) -> bool:
    """Return True if PID is currently a running process (Windows)."""
    try:
        # Windows: tasklist, fast and reliable
        import subprocess
        r = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            capture_output=True, text=True, timeout=2,
        )
        return f'"{pid}"' in r.stdout or str(pid) in r.stdout
    except Exception:
        return False


def _parse_log_progress(log_path: Path) -> dict | None:
    """Tail the log file, find the most recent progress marker, return its
    parsed values + estimated completion."""
    if not log_path.exists():
        return None
    try:
        size = log_path.stat().st_size
        # Read last 32KB to find the latest progress line — multi-benchmark
        # runners (continual_eval_suite) print >8KB between completion
        # markers, so the smaller window misses earlier completed
        # benchmarks. 32KB is enough for a 4-benchmark suite even with
        # long inter-benchmark output.
        with log_path.open("rb") as f:
            f.seek(max(0, size - 32 * 1024))
            tail_bytes = f.read()
        tail = tail_bytes.decode("utf-8", errors="ignore")
    except Exception:
        return None

    # Try in priority order: structured [PROGRESS] events first; then
    # legacy formats (SWR > paired-stim > embodied > step). Always take
    # the LAST match in the tail (most recent progress).

    # 0. Universal structured progress (sim.progress.emit_progress).
    #    Future-proof: any runner using emit_progress() shows up here
    #    without needing webapp changes.
    try:
        from sim.progress import parse_last_progress
        evt = parse_last_progress(tail)
        if evt is not None:
            # Webapp uses kind/current/total/phase/extras as-is.
            # Add `kind`-aware aliases for legacy frontend compat.
            out = dict(evt)
            # Backwards-compat aliases for older frontend code paths
            if "current" in evt and "total" in evt:
                out["fraction"] = evt["current"] / max(1, evt["total"])
            if evt.get("kind") == "training":
                out["episode"] = evt.get("current")
                out["episodes_total"] = evt.get("total")
                out["correct_pct"] = evt.get("correct_pct")
                out["correct_moves"] = evt.get("correct_moves")
                out["n_steps"] = evt.get("n_steps")
            elif evt.get("kind") == "replay":
                out["ev"] = evt.get("current")
                out["ev_total"] = evt.get("total")
            elif evt.get("kind") == "eval":
                out["trial"] = evt.get("current")
                out["trials_total"] = evt.get("total")
            return out
    except Exception:
        pass  # fall through to legacy parsing

    # 1. Curriculum phase header — informational, not a progress %.
    last_phase = None
    for m in _CURRICULUM_PHASE_RE.finditer(tail):
        last_phase = m
    phase_info = None
    if last_phase:
        phase_num, phase_label, phase_total = last_phase.groups()
        phase_info = {
            "phase_num": int(phase_num),
            "phase_label": phase_label.strip(),
            "phase_episodes_total": int(phase_total),
        }

    # 2. Phase 3 SWR replay progress (highest priority — it's the latest
    #    phase, and the embodied episode markers from Phase 2 will still
    #    be in the tail buffer).
    last_swr = None
    for m in _SWR_PROGRESS_RE.finditer(tail):
        last_swr = m
    if last_swr:
        ev, ev_total = (int(g) for g in last_swr.groups())
        out = {
            "kind": "swr_replay",
            "ev": ev,
            "ev_total": ev_total,
            "fraction": ev / max(1, ev_total),
        }
        if phase_info:
            out.update(phase_info)
        return out

    # 2b. Paired-stim isolation progress (H4 PFC bypass, minimal isolation).
    #     Log format: "[isolation] N/M events" or "[minimal-iso] N/M events".
    last_paired = None
    for m in _PAIRED_STIM_PROGRESS_RE.finditer(tail):
        last_paired = m
    if last_paired:
        ev, ev_total = (int(g) for g in last_paired.groups())
        out = {
            "kind": "paired_stim",
            "ev": ev,
            "ev_total": ev_total,
            "fraction": ev / max(1, ev_total),
        }
        if phase_info:
            out.update(phase_info)
        return out

    # 3. Embodied episode markers (Phase 2 of curriculum, or text_eval_embodied).
    last_ep = None
    for m in _DETACHED_PROGRESS_RE.finditer(tail):
        last_ep = m
    if last_ep:
        ep_done, ep_total, correct, n_steps, accuracy = last_ep.groups()
        out = {
            "kind": "embodied_episode",
            "episode": int(ep_done),
            "episodes_total": int(ep_total),
            "fraction": int(ep_done) / max(1, int(ep_total)),
            "n_steps": int(n_steps),
            "correct_moves": int(correct),
            "correct_pct": float(accuracy),
        }
        if phase_info:
            out.update(phase_info)
        return out

    # 3b. continual_eval_suite — multi-benchmark suite. Counts completion
    # markers ("[OK] NAME: ... pass=...") and finds the latest "Running
    # benchmark: NAME" line. Reports fraction = completed/started (under-
    # estimates while a benchmark is in flight; jumps to 1.0 on final
    # completion). Added 2026-05-09 after user reported "0% · no progress
    # markers yet" for Phase 1.5 multi-seed runs.
    bench_starts = list(_CES_BENCH_START_RE.finditer(tail))
    bench_ends = list(_CES_BENCH_END_RE.finditer(tail))
    if bench_starts:
        latest_start = bench_starts[-1]
        current_bench = latest_start.group(1)
        # Completed = number of [OK]/[X] end markers we've seen.
        # Started = number of "Running benchmark:" markers we've seen
        # (>=completed since the start always precedes the end).
        n_completed = len(bench_ends)
        n_started = len(bench_starts)
        # Per-benchmark scores from end markers, surfaced for the panel.
        completed_results = [
            {
                "name": m.group(2),
                "score": float(m.group(3)),
                "pass": (m.group(4) == "True"),
                "wall_clock_s": int(m.group(5)),
            }
            for m in bench_ends
        ]
        # Estimated total — Phase 1.5 default is 4 benchmarks, but accept
        # any value if more starts have been seen.
        n_total_est = max(4, n_started)
        # If we've seen N completions and the (N+1)th start, the in-flight
        # benchmark is N+1. Treat it as "halfway done" so the bar moves
        # rather than sitting flat between completions.
        if n_started > n_completed:
            fraction = (n_completed + 0.5) / max(1, n_total_est)
        else:
            fraction = n_completed / max(1, n_total_est)
        out = {
            "kind": "continual_eval",
            "current_benchmark": current_bench,
            "n_completed": n_completed,
            "n_started": n_started,
            "n_total_est": n_total_est,
            "fraction": min(1.0, fraction),
            "completed_results": completed_results,
        }
        return out

    # 4. Generic per-step format from g11_bg_runner.
    last_step = None
    for m in _GENERIC_STEP_RE.finditer(tail):
        last_step = m
    if last_step:
        step, total, x, y, gx, gy = (int(g) for g in last_step.groups())
        return {
            "kind": "step",
            "step": step,
            "total": total,
            "fraction": step / max(1, total),
            "pos": [x, y],
            "goal": [gx, gy],
        }

    return None


@app.get("/api/runs/launch/log/{log_name}", response_class=PlainTextResponse)
def get_log_tail(log_name: str) -> str:
    """Serve the last ~32KB of an in-flight run's log file. Powers the
    Brain tab's "Watch logs" pane. Two log sources are supported:

    1. Detached run logs at research/findings/raw/g11_bg/{name}.log
       (created by PowerShell Start-Process launches).
    2. Webapp-launched run logs at webapp/runtime/run_{run_id}.log
       (created via /api/runs/launch).

    Path-traversal guarded — name must not contain '/', '\\', or '..',
    and it must end with '.log'.
    """
    if "/" in log_name or "\\" in log_name or ".." in log_name:
        raise HTTPException(400, "invalid log name")
    if not log_name.endswith(".log"):
        raise HTTPException(400, "log files must end in .log")
    candidates = [
        RAW_RUNS_DIR / log_name,
        RUNTIME_DIR / log_name,
    ]
    f = next((c for c in candidates if c.is_file()), None)
    if f is None:
        raise HTTPException(404, "log not found")
    # Read the last 32KB (or the whole thing if smaller).
    try:
        size = f.stat().st_size
        with f.open("rb") as fh:
            fh.seek(max(0, size - 32 * 1024))
            tail = fh.read()
        return tail.decode("utf-8", errors="replace")
    except Exception as e:
        raise HTTPException(500, f"failed to read log: {e}")


@app.get("/api/inflight")
def list_inflight_runs() -> JSONResponse:
    """List in-flight detached training runs (Start-Process launches with
    *.pid + *.log files).

    2026-05-03: filters out *.master.pid orchestrator-shepherd pids
    written by multi-seed runner scripts. They aren't training runs;
    they're bookkeeping for the run-coordinator process. Their actual
    work (the training subprocess) registers its own *.pid file.
    """
    inflight = []
    # Two flavors of pid file:
    # 1. {name}.pid — single training/eval run (per-seed-per-condition)
    # 2. {name}.sweep.pid — experiment_runner sweep orchestrator (covers
    #    multiple child runs). Surfaces sweep-level progress for the
    #    user instead of just the active child's per-trial cycle.
    # 3. {name}.master.pid — legacy orchestrator-shepherd; still filtered.
    for pid_file in sorted(RAW_RUNS_DIR.glob("*.pid")):
        # Skip legacy orchestrator-shepherd pids (not training runs)
        if pid_file.name.endswith(".master.pid"):
            continue
        try:
            pid = int(pid_file.read_text().strip())
        except Exception:
            continue
        is_sweep = pid_file.name.endswith(".sweep.pid")
        # Sweep pids use {name}.sweep.log; per-run pids use {name}.log
        if is_sweep:
            log_file = pid_file.with_suffix(".log")  # name.sweep.pid -> name.sweep.log
        else:
            log_file = pid_file.with_suffix(".log")
        result_json = pid_file.with_name(
            pid_file.stem.replace(".pid", "")
        ).with_suffix(".json")
        # Try multiple naming conventions for the result file
        candidate_results = [
            RAW_RUNS_DIR / f"text_eval_{pid_file.stem}.json",
            RAW_RUNS_DIR / f"{pid_file.stem}.json",
        ]
        result_path = next((p for p in candidate_results if p.exists()), None)

        # A run whose result file exists is finished — never report it alive.
        # _check_pid_alive is fooled by OS PID reuse (a recycled pid resolves to
        # an unrelated live process), which would otherwise resurrect a completed
        # run into the live panels. result_path is the authoritative done-signal.
        alive = _check_pid_alive(pid) and result_path is None
        progress = _parse_log_progress(log_file)
        log_mtime = log_file.stat().st_mtime if log_file.exists() else None
        log_size = log_file.stat().st_size if log_file.exists() else 0

        inflight.append({
            "name": pid_file.stem,
            "pid": pid,
            "alive": alive,
            "log_file": log_file.name if log_file.exists() else None,
            "log_size_kb": round(log_size / 1024, 1),
            "log_mtime": log_mtime,
            "progress": progress,
            "result_file": result_path.name if result_path else None,
            "completed": result_path is not None,
            "is_sweep": is_sweep,
        })
    # 2026-05-09: also include webapp-launched runs (POST /api/runs/launch)
    # in the same response so the live panels (Home in-flight + Runs tab
    # "Live runs") surface BOTH detached PID-file launches AND webapp-
    # launched runs. Without this, a user who launches via the dashboard
    # sees no live indicator anywhere — the run is reachable only via
    # /api/runs/launch, which doesn't feed the panels.
    seen_pids = {r.get("pid") for r in inflight if r.get("pid") is not None}
    # 2026-06-08: snapshot the dict — the background orphan-scan mutates
    # launched_runs concurrently, so iterating it live raised
    # "dictionary changed size during iteration" (intermittent /api/inflight
    # 500s -> the dashboard's "Failed to load in-flight runs" / stuck "Loading").
    for run_id, run in list(launched_runs.items()):
        # Dedup: if a webapp launch also created a PID file (rare),
        # skip the launched-runs entry to avoid double-rendering.
        if run.pid is not None and run.pid in seen_pids:
            continue
        # Determine alive: prefer subprocess.poll() result; fall back to
        # PID alive check for orphan-recovery cases where proc=None.
        if run.proc is not None:
            running = (run.proc.poll() is None)
        elif run.pid is not None:
            running = _check_pid_alive(run.pid)
        else:
            running = False
        # Build inflight-shaped entry. Reuse _parse_log_progress so the
        # frontend gets the same progress shape as detached runs.
        log_path = Path(run.log_file) if run.log_file else None
        progress = _parse_log_progress(log_path) if log_path else None
        log_size = (log_path.stat().st_size
                    if (log_path and log_path.exists()) else 0)
        log_mtime = (log_path.stat().st_mtime
                     if (log_path and log_path.exists()) else None)
        out_path = Path(run.out_path) if run.out_path else None
        result_exists = bool(out_path and out_path.exists())
        # A run whose result file exists is finished — never report it alive.
        # Otherwise OS PID reuse (run.pid recycled to an unrelated live process)
        # flips this back to alive=True / completed=False, resurrecting a done run
        # into the live picker (the '_biofix_neural_s44 lingers' bug). The result
        # file is the authoritative done-signal (runner writes --out once, at end).
        running = running and not result_exists
        # Use the output JSON's basename (without .json) as the display
        # name so the live panel shows e.g. "g11_seed42_phase_1_5_..."
        # rather than the opaque run_id hex.
        if out_path:
            display_name = out_path.stem
        else:
            display_name = f"webapp-{run_id[:8]}"
        inflight.append({
            "name": display_name,
            "pid": run.pid if run.pid is not None else 0,
            "alive": running,
            "log_file": log_path.name if log_path and log_path.exists() else None,
            "log_size_kb": round(log_size / 1024, 1),
            "log_mtime": log_mtime,
            "progress": progress,
            "result_file": out_path.name if out_path and result_exists else None,
            "completed": result_exists and not running,
            "is_sweep": False,
            # Mark the source so the frontend can offer launch-specific
            # actions (kill via /api/runs/launch/{id}/kill, attach,
            # control). Detached runs leave this field absent.
            "source": "webapp_launch",
            "run_id": run_id,
        })

    # Sort sweep entries first so the dashboard shows sweep-level
    # progress at the top rather than the cycling per-child cards.
    inflight.sort(key=lambda r: (not r.get("is_sweep"), r.get("name", "")))
    return JSONResponse({"inflight": inflight, "count": len(inflight)})


# ─────────────────────────────────────────────────────────────────────────
# Frontend
# ─────────────────────────────────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
def index() -> FileResponse:
    # Disable caching of the root HTML so users always get the latest
    # version pointing at the latest JS/CSS. /static/* assets are
    # already no-cache via NoCacheStaticFiles, but the root HTML was
    # being cached by the browser separately.
    return FileResponse(
        str(STATIC_DIR / "index.html"),
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


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
# `clusterG_Gfv2nmda_seed100.json` → experiment "clusterG_Gfv2nmda".
# `k_v2_stress_16x16_seed42.json` → experiment "k_v2_stress_16x16".
# `text_eval_R5_delta_seed42.json` → experiment "text_eval_R5_delta".
# Used by the Experiments tab; mirrors the frontend's detectExperiment helper
# so users can hit /api/experiments and see the same grouping the UI shows.
_EXP_SUFFIX_RE = re.compile(r"^g11_seed\d+(?:_(.+))?\.json$")
# Extended pattern (2026-05-01): runs whose seed appears at the END of
# the filename (modern naming convention for cluster / stress / text-io
# runners). The leading group (group 1) is the experiment name.
_EXP_PREFIX_RE = re.compile(r"^(.+?)_seed\d+(?:_[a-f0-9]{6})?\.json$")


def _detect_experiment(name: str) -> str:
    # Try legacy g11_seed-prefix style first.
    m = _EXP_SUFFIX_RE.match(name)
    if m:
        return m.group(1) or "default"
    # Then modern *_seed42-suffix style (e.g. clusterG_Gfv2nmda_seed100,
    # k_v2_stress_16x16_seed42, text_eval_R5_delta_seed42).
    m = _EXP_PREFIX_RE.match(name)
    if m:
        return m.group(1)
    # Special case: pure descriptive smoke files (e.g. clusterF_smoke.json,
    # text_eval_smoke.json). Recognized via suffix.
    if name.endswith("_smoke.json") or name.endswith("_test.json"):
        return name[:-5]
    return "(other)"


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


# ─────────────────────────────────────────────────────────────────────────
# Text I/O endpoints (2026-05-02)
#
# Text I/O experiments record different metrics than navigation runs:
# image_to_word_eval / word_to_action_eval blocks with confusion matrices,
# rather than per-phase finalQ Manhattan distances. The Runs/Experiments
# tabs filter these out via sum_finalQ heuristic; text I/O needs its own
# surface.
# ─────────────────────────────────────────────────────────────────────────


def _is_text_io_run(name: str) -> bool:
    """Filename heuristic for text I/O runs. Used by both list endpoints."""
    return name.startswith("text_eval_") or name.startswith("text_io_")


@app.get("/api/text_io_runs")
def list_text_io_runs() -> JSONResponse:
    """List text I/O run JSONs (text_eval_* / text_io_*) with W→A and I→W
    accuracies + confusion matrices. Powers the Language tab.
    """
    files = sorted(
        (f for f in RAW_RUNS_DIR.glob("*.json")
         if not f.name.endswith(".cmd.json") and _is_text_io_run(f.name)),
        reverse=True,
    )
    out = []
    # 2026-05-03: collect per-direction values across all runs for the
    # aggregate breakdown displayed in the Language tab. Each entry is
    # the accuracy for {word: predicted_action == word's letter} on
    # that run.
    per_dir_w2a = {"north": [], "east": [], "south": [], "west": []}
    per_dir_i2w = {"north": [], "east": [], "south": [], "west": []}
    for f in files:
        try:
            data = json.loads(f.read_text())
        except Exception:
            continue
        iw = data.get("image_to_word_eval") or {}
        wa = data.get("word_to_action_eval") or {}
        # Many older text_eval_*.json files predate the embodied training
        # stats block. Be defensive.
        ts = data.get("training_stats") or []
        correct_move_rate = None
        if ts and isinstance(ts, list):
            v = ts[0].get("correct_move_rate")
            if v is not None:
                correct_move_rate = float(v)
        # Per-direction accuracy for this run
        run_per_dir_w2a = {}
        wa_cm = wa.get("confusion_matrix") or {}
        for word in ("north", "east", "south", "west"):
            row = wa_cm.get(word, {}) or {}
            target = word[0].upper()
            total = sum(int(v) for v in row.values())
            correct = int(row.get(target, 0))
            acc = correct / total if total > 0 else None
            run_per_dir_w2a[word] = acc
            if acc is not None:
                per_dir_w2a[word].append(acc)
        run_per_dir_i2w = {}
        iw_cm = iw.get("confusion_matrix") or {}
        for word in ("north", "east", "south", "west"):
            row = iw_cm.get(word, {}) or {}
            total = sum(int(v) for v in row.values())
            correct = int(row.get(word, 0))
            acc = correct / total if total > 0 else None
            run_per_dir_i2w[word] = acc
            if acc is not None:
                per_dir_i2w[word].append(acc)
        out.append({
            "name": f.name,
            "size_bytes": f.stat().st_size,
            "modified_unix": f.stat().st_mtime,
            "regime": data.get("regime"),
            "seed": data.get("seed"),
            "n_episodes": data.get("n_episodes"),
            "steps_per_episode": data.get("steps_per_episode"),
            "i2w_accuracy": iw.get("accuracy"),
            "i2w_correct": iw.get("correct"),
            "i2w_n_trials": iw.get("n_trials"),
            "i2w_per_direction": run_per_dir_i2w,
            "w2a_accuracy": wa.get("accuracy"),
            "w2a_correct": wa.get("correct"),
            "w2a_n_trials": wa.get("n_trials"),
            "w2a_per_direction": run_per_dir_w2a,
            "correct_move_rate": correct_move_rate,
        })
    # Aggregate stats across the entire collection.
    agg = {"n_runs": len(out)}
    for key in ("i2w_accuracy", "w2a_accuracy"):
        vals = [r[key] for r in out if r[key] is not None]
        if vals:
            mean_v = sum(vals) / len(vals)
            var = sum((v - mean_v) ** 2 for v in vals) / max(1, len(vals) - 1)
            agg[f"{key}_mean"] = mean_v
            agg[f"{key}_std"] = var ** 0.5 if len(vals) > 1 else None
            agg[f"{key}_n"] = len(vals)
    # Per-direction aggregates
    agg["w2a_per_direction_mean"] = {
        d: (sum(vals) / len(vals)) if vals else None
        for d, vals in per_dir_w2a.items()
    }
    agg["i2w_per_direction_mean"] = {
        d: (sum(vals) / len(vals)) if vals else None
        for d, vals in per_dir_i2w.items()
    }
    return JSONResponse({"runs": out, "count": len(out), "aggregate": agg})


PLANS_DIR = REPO_ROOT / "docs" / "plans"


@app.get("/api/plans")
def list_plans() -> JSONResponse:
    """List all design / architecture decision record markdowns from
    docs/plans/. Powers the Plans tab. Same shape as /api/findings.
    """
    if not PLANS_DIR.is_dir():
        return JSONResponse({"plans": [], "count": 0})
    plans = []
    for f in sorted(PLANS_DIR.glob("*.md"), reverse=True):
        plans.append({
            "name": f.name,
            "size_bytes": f.stat().st_size,
            "modified_unix": f.stat().st_mtime,
        })
    return JSONResponse({"plans": plans, "count": len(plans)})


@app.get("/api/plans/{name}", response_class=PlainTextResponse)
def get_plan(name: str) -> str:
    """Return raw markdown body for a plan doc. Same path-traversal guard
    as /api/findings/{name}."""
    if "/" in name or "\\" in name or ".." in name:
        raise HTTPException(400, "invalid name")
    f = PLANS_DIR / name
    if not f.is_file():
        raise HTTPException(404, "not found")
    return f.read_text(encoding="utf-8")


@app.get("/api/current_state", response_class=PlainTextResponse)
def get_current_state() -> str:
    """Serve docs/CURRENT-STATE.md as plain text. Used by the About tab.
    Returns 404 if the file is missing."""
    path = REPO_ROOT / "docs" / "CURRENT-STATE.md"
    if not path.is_file():
        raise HTTPException(404, "docs/CURRENT-STATE.md not found")
    return path.read_text(encoding="utf-8")


@app.get("/api/biology", response_class=PlainTextResponse)
def get_biology() -> str:
    """Serve docs/biology.md as plain text."""
    path = REPO_ROOT / "docs" / "biology.md"
    if not path.is_file():
        raise HTTPException(404, "docs/biology.md not found")
    return path.read_text(encoding="utf-8")


@app.get("/api/frontend_guide", response_class=PlainTextResponse)
def get_frontend_guide() -> str:
    """Serve docs/webapp-frontend-guide.md as plain text."""
    path = REPO_ROOT / "docs" / "webapp-frontend-guide.md"
    if not path.is_file():
        raise HTTPException(404, "docs/webapp-frontend-guide.md not found")
    return path.read_text(encoding="utf-8")


@app.get("/api/readme", response_class=PlainTextResponse)
def get_readme() -> str:
    """Serve README.md as plain text."""
    path = REPO_ROOT / "README.md"
    if not path.is_file():
        raise HTTPException(404, "README.md not found")
    return path.read_text(encoding="utf-8")


# NOTE: the /api/capability-status endpoint (+ webapp/capability_status.json
# + the renderCapabilityStatus frontend) was RETIRED 2026-06-23 with the
# INTERACT-first console reframe: the webapp is a functional console
# (launch/manage · visualize · INTERACT), not a capability/milestone
# dashboard. The capability snapshot is no longer surfaced in the UI.


@app.get("/api/text_io_runs/{name}")
def get_text_io_run_detail(name: str) -> JSONResponse:
    """Full text I/O run JSON, with the same path-traversal guard as
    /api/runs/{name}.
    """
    if "/" in name or "\\" in name or ".." in name:
        raise HTTPException(400, "invalid name")
    f = RAW_RUNS_DIR / name
    if not f.is_file():
        raise HTTPException(404, "not found")
    if not _is_text_io_run(name):
        raise HTTPException(400, "not a text I/O run")
    try:
        data = json.loads(f.read_text())
    except Exception as e:
        raise HTTPException(500, f"failed to parse: {e}")
    return JSONResponse(data)


@app.get("/api/chain")
def get_chain_status() -> JSONResponse:
    """Return current autonomous decision-chain status as JSON.

    Reuses morning_briefing._chain_status() so the webapp and CLI tool
    are always in sync. Useful for the dashboard's chain-status panel.
    """
    try:
        from research.runners.morning_briefing import _chain_status
        cs = _chain_status()
    except Exception as e:
        return JSONResponse({"stage": "error", "error": str(e)}, status_code=500)

    # Augment with waiter log tail for inline preview
    try:
        from research.runners.morning_briefing import _waiter_status
        cs["waiter_log_tail"] = _waiter_status()
    except Exception:
        cs["waiter_log_tail"] = None

    # Add aggregator verdicts for all relevant configs
    verdicts = {}
    try:
        from research.result_aggregator import (
            BUILTIN_CONFIGS, AggregateConfig, ResultSet,
        )
        for cfg_name in ("biology", "minimum_biology", "sanity_check"):
            try:
                conds = BUILTIN_CONFIGS[cfg_name]["conditions"]
                seeds = BUILTIN_CONFIGS[cfg_name].get(
                    "seeds", [42, 43, 44, 100, 101, 102]
                )
                cfg = AggregateConfig(conditions=conds, seeds=seeds)
                rs = ResultSet.load(cfg)
                v = rs.verdict()
                # Strip markdown asterisks for compact display
                v = v.replace("**", "")
                verdicts[cfg_name] = v
            except Exception:
                verdicts[cfg_name] = None
    except Exception:
        verdicts = {}
    cs["aggregator_verdicts"] = verdicts

    return JSONResponse(cs)
