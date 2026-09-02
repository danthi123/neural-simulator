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
async def _continuous_state_tick() -> None:
    """CONTINUOUS-STATE ENGINE (2026-08-19 reframe): the brain keeps FEELING between turns, not just per-request.

    An always-on background tick — when a chat session goes idle, its felt mood RELAXES toward baseline and the
    spiking affect ladder is re-read, so the state keeps evolving with no input ('unplug the conversation and it's
    still changing'). Default-OFF behind `BRAIN_CONTINUOUS`; when off this loop is inert (byte-identical to today).
    Host code is only the clock; every mood read reuses the existing spiking affect organ. See
    webapp/continuous_engine.py. v1 = the mood seed; next rungs (self-init wander, idle consolidation) are own tasks.

    PRODUCTION CONCURRENCY (2026-08-20): the tick's self-initiation CA3 wander is ~55s on the cupy substrate (its full
    4000-step operating point). So the tick is run OFF the event loop in a thread executor — a synchronous call here
    would freeze every chat request for the whole wander — and an in-flight guard skips a new tick while the previous
    one is still running (with a 55s wander and IDLE_SEC=20s, ticks would otherwise pile up). CuPy work from one
    background thread serializes on the GPU with any concurrent chat turn (no correctness race — the affect organ's
    read is a short forward read; the per-session self-init organ is only touched by the tick)."""
    from webapp import continuous_engine as _CE

    _inflight = {"on": False}  # mutable holder: at most one heavy tick runs at a time (no per-20s pile-up)

    async def _loop():
        while True:
            await asyncio.sleep(_CE.IDLE_SEC)
            if not _CE.continuous_enabled():
                continue
            if _inflight["on"]:
                continue  # previous tick (possibly a ~55s cupy wander) still running -> don't stack another
            _inflight["on"] = True
            try:
                loop = asyncio.get_event_loop()
                n = await loop.run_in_executor(
                    None,
                    lambda: _CE.tick_idle_sessions(_SESSION_MOOD, _get_affect_organ,
                                                   selfinit_getter=_get_selfinit_organ,
                                                   episodic_getter=_get_episodic_organ_existing,
                                                   chat_getter=_get_chat_existing),
                )
                if n:
                    print("[webapp] continuous tick: evolved %d idle session(s)" % n, flush=True)
            except Exception as e:
                print(f"[webapp] continuous tick failed: {type(e).__name__}: {e}", flush=True)
            finally:
                _inflight["on"] = False
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
            # ONE-BRAIN SINGLE-POOL merge (opt-in, `BRAIN_ONEBRAIN_SINGLE_POOL`, default-OFF): when ON, the four
            # core organs below (surprise/world-model/metacog/pragmatic) resolve their `shared=` to ONE
            # merge_organs pool (retiring the two production pools for the turn) via their get_organ() singletons.
            # This is an OBSERVABILITY log only (the substantive wiring lives in each organ's get_organ()); OFF ->
            # the exact current two-pool path. De-risked by 2026-09-02-onebrain-twopool-merge-organ-read-GO.md.
            try:
                from research.runners.onebrain_single_pool_production import single_pool_enabled
                if single_pool_enabled():
                    print("[webapp] startup: ONE-BRAIN SINGLE-POOL merge ACTIVE — surprise + world-model + metacog "
                          "+ pragmatic co-inhabit ONE merge_organs pool (BRAIN_ONEBRAIN_SINGLE_POOL=1)", flush=True)
            except Exception as _spe:
                print(f"[webapp] startup: single-pool flag check skipped ({type(_spe).__name__}: {_spe})", flush=True)
            # SURPRISE (Gate-B, D2): pre-build the co-resident spiking expectation-violation circuit so the first
            # assertion's mismatch read is fast. Best-effort + guarded (default-ON; BRAIN_SURPRISE=0 skips it).
            try:
                from research.runners.surprise_production_organ import surprise_enabled
                if surprise_enabled():
                    _get_surprise_organ().ensure_built()
                    print("[webapp] startup: surprise organ (co-resident predictive-coding mismatch unit) WARM",
                          flush=True)
            except Exception as _se:
                print(f"[webapp] startup: surprise organ warm skipped ({type(_se).__name__}: {_se})", flush=True)
            # METACOG (Gate-B, E1): pre-build the co-resident spiking balance-of-evidence confidence monitor so the
            # first turn's confidence read is fast. Best-effort + guarded (default-ON; BRAIN_METACOG=0 skips it).
            try:
                from research.runners.metacog_production_organ import metacog_enabled
                if metacog_enabled():
                    _get_metacog_organ().ensure_built()
                    print("[webapp] startup: metacog organ (co-resident balance-of-evidence monitor) WARM", flush=True)
            except Exception as _me:
                print(f"[webapp] startup: metacog organ warm skipped ({type(_me).__name__}: {_me})", flush=True)
            # WORLD-MODEL (Gate-B, E2): pre-build the co-resident spiking affective forward model so the first
            # expectation query / violation read is fast. Best-effort + guarded (default-ON; BRAIN_WORLDMODEL=0 skips).
            try:
                from research.runners.worldmodel_production_organ import worldmodel_enabled
                if worldmodel_enabled():
                    _get_worldmodel_organ().ensure_built()
                    print("[webapp] startup: world-model organ (co-resident valence forward model) WARM", flush=True)
            except Exception as _we:
                print(f"[webapp] startup: world-model organ warm skipped ({type(_we).__name__}: {_we})", flush=True)
            # PRAGMATIC (Task-#12, D): pre-build the co-resident spiking W4 graded scalar-implicature listener-belief
            # so the first scalar-quantity turn's pragmatic read is fast. Best-effort + guarded (default-ON;
            # BRAIN_PRAGMATIC=0 skips it).
            try:
                from research.runners.pragmatic_production_organ import pragmatic_enabled
                if pragmatic_enabled():
                    _get_pragmatic_organ().ensure_built()
                    print("[webapp] startup: pragmatic organ (co-resident W4 graded scalar-implicature belief) WARM",
                          flush=True)
            except Exception as _pe:
                print(f"[webapp] startup: pragmatic organ warm skipped ({type(_pe).__name__}: {_pe})", flush=True)
            # VALUE-CHOICE (RANK-1 value-critic GO wire-in; DEFAULT-ON 2026-08-26): pre-build the LEARNED spiking
            # striosome_value critic + the value-WTA so the first >=2-competing turn does not stall on the ~4-min
            # value-train. Best-effort + guarded; BRAIN_VALUE_CHOICE=0 -> nothing built (byte-identical escape).
            try:
                from research.runners.value_choice_production_organ import (
                    value_choice_enabled as _vc_enabled, get_value_choice_organ as _get_vc_organ)
                if _vc_enabled():
                    _get_vc_organ(seed=42, untrained=False).ensure_built()
                    print("[webapp] startup: value-choice organ (learned striosome_value critic + spiking WTA) WARM",
                          flush=True)
            except Exception as _vce:
                print(f"[webapp] startup: value-choice organ warm skipped ({type(_vce).__name__}: {_vce})", flush=True)
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

# ─── INTERNAL WORLDVIEW / AFFECTIVE WORLD-MODEL (Gate-B, E2): the brain's spiking forward model predicts the
# next-turn affect and fires a surprise on an affect-trajectory violation ─────────────────────────────────────
# Per-session forward-model STATE (the affective context sign + the prediction the model held from the prior
# turn), keyed identically to the ChatBrain cache. Held host-side; the queryable PREDICTION + the surprise
# MISMATCH are the load-bearing spiking reads. Cleared on reset. See research/runners/worldmodel_production_organ.py.
_SESSION_WORLDVIEW: dict[tuple[str, str, str], dict] = {}


def _get_affect_organ():
    """The process-shared spiking affect organ (built once, ~1s on the process backend)."""
    from research.runners.affect_production_organ import get_organ
    return get_organ(seed=42)


def _get_comprehension_organ():
    """The process-shared spiking comprehension monitor (built once; the co-resident SpikingRoleCompetition)."""
    from research.runners.comprehension_production_organ import get_organ
    return get_organ(seed=42)


def _get_surprise_organ():
    """The process-shared spiking expectation-violation organ (built once; the co-resident mismatch circuit)."""
    from research.runners.surprise_production_organ import get_organ
    return get_organ(seed=42)


def _get_metacog_organ():
    """The process-shared spiking metacognition organ (built once; the co-resident balance-of-evidence monitor)."""
    from research.runners.metacog_production_organ import get_organ
    return get_organ(seed=42)


def _get_worldmodel_organ():
    """The process-shared spiking affective world-model organ (built once; the co-resident valence forward model)."""
    from research.runners.worldmodel_production_organ import get_organ
    return get_organ(seed=42)


def _get_curiosity_organ():
    """The process-shared spiking curiosity organ (built once; the co-resident from_novelty -> ASK-pool drive)."""
    from research.runners.curiosity_production_organ import get_organ
    return get_organ(seed=42)


def _get_source_provenance_organ():
    """The process-shared #129 spiking source-provenance opponent-comparator monitor (built once; rebuilt if the
    lesion flag flips). See research/runners/source_provenance_production_organ.py."""
    from research.runners.source_provenance_production_organ import get_organ, source_provenance_lesioned
    return get_organ(seed=42, lesion=source_provenance_lesioned())


def _get_pragmatic_organ():
    """The process-shared spiking scalar-implicature organ (built once; the W4 depth-2 graded RSA listener-belief)."""
    from research.runners.pragmatic_production_organ import get_organ
    return get_organ(seed=42)


def _get_self_schema_organ():
    """The process-shared DR-3 self-schema AUTHORSHIP organ (built once; the self_schema `author` sub-block that
    fires 'self' on a volunteered proposition vs 'heard' on a recalled fact). The read-time lesion is passed
    per-read (schema_access=False), so the SAME organ serves a normal request and a lesion-verification probe.
    See research/runners/self_schema_production_organ.py."""
    from research.runners.self_schema_production_organ import get_organ
    return get_organ(seed=42)


# ─── MULTI-REFERENT WORKING MEMORY (Gate-B, D6): a spiking multi-register discourse buffer that HOLDS >=2 referents
# across a span. The organ's process singleton accumulates a referent codebook across ALL sessions, so a hold-query
# would re-materialize other conversations' referents — the buffer MUST be per-session. Keyed identically to the
# ChatBrain cache; cleared on reset. See research/runners/d6_multiref_wm_production_organ.py. ────────────────────
_SESSION_MULTIREF: dict = {}

# ─── ACTIVITY-SILENT WORKING MEMORY (Gate-B, Mongillo 2008, 2026-08-26): a maintenance-mode swap on the anaphora
# referent store — the discourse FOCUS is held in short-term synaptic FACILITATION (cp_stp_u) across an intervening
# distractor turn, delay genuinely SILENT, and reactivated on the next temporal-recall query by a NONSPECIFIC ping.
# PER-SESSION (the focus + which-referent-in-which-assembly is THIS conversation's; a process singleton would leak one
# conversation's focus into another's recall), keyed identically to the ChatBrain cache; cleared on reset. DEFAULT-OFF
# (BRAIN_SILENT_WM). See research/runners/activity_silent_wm_production_organ.py. ──────────────────────────────────────
_SESSION_SILENT_WM: dict = {}

# ─── DISCOURSE EVENT REGISTER (Gate-B, D3): the running "who-was-doing-it-BEFORE across a connective" turn STATE
# (boundary_seen / heard_any), keyed identically to the ChatBrain cache. The spiking register OBJECT itself is built
# once with the agent (attached as _event_register); only this small per-conversation fold state lives here so a
# before-answer is surfaced ONLY after a connective actually opened a boundary this conversation. Cleared on reset.
# See research/runners/d3_discourse_event_register_production_organ.py. ───────────────────────────────────────────
_SESSION_DISCOURSE: dict = {}

# ─── PROSPECTIVE MEMORY (Gate-B, 2026-08-13): a spiking intention-LATCH + BA10 cue-MONITOR that HOLDS a deferred
# intention ("remind me to X when Y") across intervening turns and RELEASES it only when the cue appears. The latch
# is a PERSISTENT co-resident spiking bridge that must survive BETWEEN turns, and the held intention is THIS
# conversation's — so the organ is PER-SESSION (keyed identically to the ChatBrain cache; cleared on reset). The
# substrate calibration (homeostat bias + NMDA-plateau threshold) is process-cached per seed, so only the
# intention-holding bridge is per-session. See research/runners/prospective_memory_production_organ.py. ────────────
_SESSION_PMEM: dict = {}

# ─── SELF-INITIATED UTTERANCE (2026-08-18): the first INTERNALLY-GENERATED turn class. On an IDLE/EMPTY turn the brain
# SELECTS a stored concept ITSELF (a curiosity-biased CA3 wander on cupy; the mouth's curiosity-top decodable concept
# on the numpy-deferred path) and SPEAKS it through the OneBrainComposer mouth. PER-SESSION organ (its own self-
# contained selection substrate + mouth, like _SESSION_MULTIREF), keyed identically to the ChatBrain cache; cleared on
# reset. See research/runners/self_initiated_production_organ.py. ──────────────────────────────────────────────────
_SESSION_SELFINIT: dict = {}
# The DEFAULT-ON master switch (the production-integration anchor). BRAIN_SELF_INITIATE=0 disables the idle block
# byte-identically (the row STAYS on_by_default:YES). Flipping this to False would turn the faculty OFF by default.
_SELF_INITIATE_DEFAULT_ON = True

# ─── GNW CONFIDENCE/CONFLICT-GATED DELIBERATION (2026-08-18, T1-1 rung d): the DEFAULT-ON master switch (the production-
# integration anchor). BRAIN_GNW_DELIBERATE=0 makes the installed gate wrapper a pure pass-through (byte-identical); the
# row STAYS on_by_default:YES. Flipping this to False would turn the faculty OFF by default. See webapp/gnw_deliberation.py.
_GNW_DELIBERATE_DEFAULT_ON = True

# ─── GNW MULTI-STEP re-entrant deliberation (2026-08-19, T1-1 rung d): the DEFAULT-ON master switch (the production-
# integration anchor). Mirrors _GNW_DELIBERATE_DEFAULT_ON: the gate wrapper is ALWAYS installed here; BRAIN_GNW_MULTISTEP=0
# makes the installed wrapper a pure pass-through (byte-identical to the pre-flip default); the row STAYS on_by_default:YES.
# Flipping this to False would turn the faculty OFF by default. See webapp/gnw_multistep_deliberation.py.
_GNW_MULTISTEP_DEFAULT_ON = True

# ─── GNW NEURAL THOUGHT-SWAP OBSERVER (2026-08-19, board #77) — SUPERSEDED + RETIRED from the default handler
# (2026-08-20). This is the OLDER observe-only #77 swap tracker; board #85 `swap_drives_chat` (default-ON, below)
# SUPERSEDES it and is now the SOLE swap path — it internally reuses the SAME #77 machinery AND additionally DRIVES the
# response. The per-turn observer FALLBACK has been removed from `brain_chat` (it only ever ran when swap-drives was
# turned OFF and BRAIN_GNW_SWAP was on — a review-only combo). `_GNW_SWAP_DEFAULT_ON` is retained (unchanged, False) as
# the ledger's default-anchor (docs/PRODUCTION_INTEGRATION_LEDGER.yaml: row gnw-thought-swap, on_by_default:NO) and it
# still gates the additive, DEFAULT-OFF `gnw_swap` observability key (`BRAIN_GNW_SWAP=1`, which now surfaces
# swap-drives' own per-turn `_last_gnw_swap` read) -> no key by default -> byte-identical. See webapp/gnw_thought_swap.py
# (kept on disk; still imported by webapp/swap_drives_chat.py).
_GNW_SWAP_DEFAULT_ON = False


def _gnw_swap_flag_on() -> bool:
    """The `BRAIN_GNW_SWAP` opt-in (a lightweight env read so the DISABLED default path imports nothing / does no work).
    Truthy (1/true/on/yes) -> enable the held-topic swap tracker; unset or 0/false/off/no -> skip the block entirely."""
    return os.environ.get("BRAIN_GNW_SWAP", "0").strip().lower() in ("1", "true", "on", "yes")


# ─── SWAP DRIVES THE RESPONSE (board #85, 2026-08-19): the #77 neural thought-swap verdict made LOAD-BEARING on the
# live turn — a neural SWAP (topic change) makes the reply LEAD with a topic-transition acknowledgment naming the
# newly-held coalition ("On <newtopic>, then -- <answer>"); a HOLD stays silent (the natural discourse move). The
# production-integration anchor. When `_SWAP_DRIVES_DEFAULT_ON` is False the block is gated on the env flag alone
# (`BRAIN_SWAP_DRIVES=1` opts in for review); the response carries NO `swap_drives` key and NO transition lead ->
# byte-identical. Flipping the anchor to True installs the coupling by default (a `BRAIN_SWAP_DRIVES=0` escape reverts
# to the byte-identical oracle). This SUPERSEDES the #77 observer block below (a single neural swap runs per turn: the
# drives path when on, else the #77 observer). See webapp/swap_drives_chat.py.
_SWAP_DRIVES_DEFAULT_ON = True


def _swap_drives_on() -> bool:
    """The master switch = the anchor combined with the env override. Default-ON anchor: enabled UNLESS
    `BRAIN_SWAP_DRIVES` is an explicit off (0/false/no/off). Default-OFF anchor: enabled only on an explicit truthy
    opt-in. Kept lightweight so the disabled path does no swap work."""
    try:
        from webapp import swap_drives_chat as _SDC
    except Exception:
        return False
    if _SWAP_DRIVES_DEFAULT_ON:
        return not _SDC.swap_drives_off()
    return _SDC.swap_drives_enabled()


# ─── GNW GLOBAL-WORKSPACE STOP DRIVES THE RESPONSE (distributed-overwrite clear-all, 2026-08-26): the de-risk
# `2026-08-18-gnw-distributed-overwrite-workspace-PARTIAL.md` global-stop capability (6/6 GO) made LOAD-BEARING on the
# live turn. On a strong interrupt / hard topic-break (the gnw-deliberation acc_conflict_gate reporting n_ignited>=2,
# OR the #85 swap detector flagging a topic break) the held P1.2 coalition is driven into a divisively-normalized
# distributed workspace and a conflict-triggered depression of the SHARED recurrence CLEARS it to n_ignited=0 BEFORE
# the newcomer ignites -> a clean single-content workspace (no stale bleed). A CLEAN neural stop -> a short clearing
# lead prepended OUTERMOST; no trigger / no clean clear -> NO lead. LESION (`BRAIN_GNW_STOP_LESION=1`) zeroes the
# shared-resource-depression term -> the workspace stays >=2 co-ignited -> the clearing lead VANISHES (load-bearing).
# See webapp/gnw_global_stop.py.
# 2026-08-26 FLIPPED DEFAULT-ON (wave 3 flip, 6/6 flip-soak GO): the clearing lead ships as the production default.
# BRAIN_GNW_STOP=0 is the byte-identical escape to the pre-flip no-lead oracle.
_GNW_STOP_DEFAULT_ON = True


def _gnw_stop_flag_on() -> bool:
    """The master switch = the DEFAULT-OFF anchor combined with the env override. Enabled only on an explicit truthy
    `BRAIN_GNW_STOP` (1/true/on/yes) UNLESS `_GNW_STOP_DEFAULT_ON` is flipped True (then enabled unless an explicit
    off). Kept as a lightweight env read so the DISABLED default path imports nothing / does no stop work ->
    byte-identical to pre-wiring."""
    v = os.environ.get("BRAIN_GNW_STOP")
    if _GNW_STOP_DEFAULT_ON:
        return not (v is not None and v.strip().lower() in ("0", "false", "no", "off", ""))
    return v is not None and v.strip().lower() in ("1", "true", "on", "yes")


# ─── AFFECT DRIVES THE RESPONSE (board #84, 2026-08-19): the #81 graded-affect ladder read made LOAD-BEARING on the
# live turn — the brain's felt valence x arousal (read NEURALLY off the #81 interoceptive ladder) colors the AFFECTIVE
# EXPRESSION the reply leads with (a graded warmth/curtness marker) + its forthcomingness. The production-integration
# anchor. When `_AFFECT_DRIVES_DEFAULT_ON` is False the block is gated on the env flag alone (`BRAIN_AFFECT_DRIVES=1`
# opts in for review); the response carries NO `affect_drives` key and NO affective lead -> byte-identical. Flipping the
# anchor to True installs the coupling by default (a `BRAIN_AFFECT_DRIVES=0` escape reverts to the byte-identical
# oracle). Orthogonal to the Gate-B `BRAIN_AFFECT` prose-manner path. See webapp/affect_drives_chat.py.
_AFFECT_DRIVES_DEFAULT_ON = True


def _affect_drives_on() -> bool:
    """The master switch = the anchor combined with the env override. Default-ON anchor: enabled UNLESS
    `BRAIN_AFFECT_DRIVES` is an explicit off (0/false/no/off). Default-OFF anchor: enabled only on an explicit
    truthy opt-in. Kept lightweight so the disabled path does no ladder work."""
    try:
        from webapp import affect_drives_chat as _ADC
    except Exception:
        return False
    if _AFFECT_DRIVES_DEFAULT_ON:
        return not _ADC.affect_drives_off()
    return _ADC.affect_drives_enabled()


# >>> W5 AFFECTIVE ToM BEGIN (empathy — infer ANOTHER agent's emotion; additive, mergeable block) ───────────────
# ─── AFFECTIVE THEORY OF MIND DRIVES THE RESPONSE (W5, 2026-08-26): the 6/6-seed-GO W5 de-risk (an OTHER-tagged P0.3
# affect region, dissociable from the system's OWN affect) made LOAD-BEARING on the live turn — on a turn about
# ANOTHER agent's affectively-charged situation ("Maria is devastated", "Sam's team lost", "my friend won") the reply
# LEADS with an EMPATHIC expression whose tone is read NEURALLY off the OTHER model's `affect_out`-gated recall
# differential (reuse-by-import, NO sim/ edit). The production-integration anchor. When `_AFFECTIVE_TOM_DEFAULT_ON`
# is False the block is gated on the env flag alone (`BRAIN_AFFECTIVE_TOM=1` opts in for review); the response carries
# NO `affective_tom` key and NO empathic lead -> byte-identical. Flipping the anchor to True installs the coupling by
# default (a `BRAIN_AFFECTIVE_TOM=0` escape reverts to the byte-identical oracle). Orthogonal to the #84 self-affect
# path. LESION (`BRAIN_AFFECTIVE_TOM_LESION=1`): cut the OTHER region's `affect_out` -> the empathic tone collapses to
# neutral -> the lead VANISHES (the load-bearing proof). See research/runners/affective_tom_production_organ.py.
# 2026-08-26 FLIPPED DEFAULT-ON (wave 1/2 flip, 6-seed pool soak GO): the empathic lead ships as the production
# default. BRAIN_AFFECTIVE_TOM=0 is the byte-identical escape to the pre-flip no-lead oracle.
_AFFECTIVE_TOM_DEFAULT_ON = True


def _affective_tom_on() -> bool:
    """The master switch = the anchor combined with the env override. Default-OFF anchor (current): enabled ONLY on an
    explicit truthy `BRAIN_AFFECTIVE_TOM`. Default-ON anchor (after the parent flips it): enabled UNLESS
    `BRAIN_AFFECTIVE_TOM` is an explicit off. Kept lightweight so the disabled path does no OTHER-model work."""
    try:
        from research.runners import affective_tom_production_organ as _ATM
    except Exception:
        return False
    if _AFFECTIVE_TOM_DEFAULT_ON:
        return not _ATM.affective_tom_off()
    return _ATM.affective_tom_enabled()
# <<< W5 AFFECTIVE ToM END ──────────────────────────────────────────────────────────────────────────────────────


# ─── VISION-IDENTITY (spiking HMAX) DEFAULT-ON anchor (2026-08-26 flip, wave 1/2, 6-seed pool soak GO — vision 6/6 +
# scramble-collapsed GO): the production-integration anchor for BRAIN_VISION_IDENTITY. This path ONLY fires on a
# visual query that CARRIES a `percept` field (req.percept), so the flip is a no-op on every ordinary (non-visual)
# turn -> byte-identical there regardless of the flag. `BRAIN_VISION_IDENTITY=0` is the explicit escape back to the
# pre-flip OFF oracle even on a visual-query turn. Mirrors the _AFFECTIVE_TOM_DEFAULT_ON / _SELF_SCHEMA_DEFAULT_ON
# convention. See research/runners/vision_identity_production_organ.py.
_VISION_IDENTITY_DEFAULT_ON = True


def _vision_identity_on() -> bool:
    """Master switch = the anchor combined with the env override. Default-ON anchor (current): enabled UNLESS
    `BRAIN_VISION_IDENTITY` is an explicit off (0/false/off/no/''). Default-OFF anchor: enabled only on an explicit
    truthy opt-in."""
    v = os.environ.get("BRAIN_VISION_IDENTITY")
    if _VISION_IDENTITY_DEFAULT_ON:
        return not (v is not None and v.strip().lower() in ("0", "false", "off", "no", ""))
    return v is not None and v.strip().lower() in ("1", "true", "on", "yes")


# ─── DA-MODE DRIVES THE RESPONSE (board #79, 2026-08-19): the #76 spiking DA-mode (rest/focus/arousal) made
# LOAD-BEARING on the live turn — the brain's OWN dopamine nucleus (the snc population) self-produces a DA LEVEL from
# the message's reward/context (engagement), the level is binned to a MODE, and the mode modulates HOW forthcoming the
# reply is (a graded ENGAGEMENT SUFFIX — distinct axis + a suffix, not a third prefix, vs #84 valence / #85 topic). The
# production-integration anchor. When `_DA_DRIVES_DEFAULT_ON` is False the block is gated on the env flag alone
# (`BRAIN_DA_DRIVES=1` opts in for review); the response carries NO `da_drives` key and NO suffix -> byte-identical.
# Flipping the anchor to True installs the coupling by default (a `BRAIN_DA_DRIVES=0` escape reverts to the byte-
# identical oracle). LESION (`BRAIN_DA_DRIVES_LESION=1`): silence the spiking SNc nucleus -> the DA level collapses ->
# the mode is REST -> the suffix VANISHES (the load-bearing proof). See webapp/da_mode_drives_chat.py.
_DA_DRIVES_DEFAULT_ON = True


def _da_drives_on() -> bool:
    """The master switch = the anchor combined with the env override. Default-ON anchor: enabled UNLESS
    `BRAIN_DA_DRIVES` is an explicit off (0/false/no/off). Default-OFF anchor: enabled only on an explicit truthy
    opt-in. Kept lightweight so the disabled path does no substrate work."""
    try:
        from webapp import da_mode_drives_chat as _DAD
    except Exception:
        return False
    if _DA_DRIVES_DEFAULT_ON:
        return not _DAD.da_drives_off()
    return _DAD.da_drives_enabled()


# ─── ACTIVITY-SILENT WORKING MEMORY (Gate-B, Mongillo 2008, 2026-08-26): the DEFAULT-ON master switch (the
# production-integration anchor). `BRAIN_SILENT_WM=0` disables the maintenance-mode swap on the anaphora referent
# store byte-identically (the row STAYS on_by_default:YES). Flipping this to False would turn the faculty OFF by
# default. See research/runners/activity_silent_wm_production_organ.py. 2026-08-26 FLIPPED DEFAULT-ON (wave 3 flip,
# 6/6 flip-soak GO): the silent-hold reactivation ships as the production default.
_SILENT_WM_DEFAULT_ON = True


def _silent_wm_flag_on() -> bool:
    """The master switch = the DEFAULT-ON anchor combined with the env override. Enabled UNLESS `BRAIN_SILENT_WM`
    is an explicit off (0/false/no/off/''). Kept as a lightweight env read so the disabled path does no silent-WM
    work -> byte-identical to pre-wiring."""
    v = os.environ.get("BRAIN_SILENT_WM")
    if _SILENT_WM_DEFAULT_ON:
        return not (v is not None and v.strip().lower() in ("0", "false", "no", "off", ""))
    return v is not None and v.strip().lower() in ("1", "true", "on", "yes")


# ─── BG ACTION SELECTION (SPEAK vs STAY-SILENT, 2026-08-26): the DEFAULT-ON master switch (the production-
# integration anchor). `BRAIN_BG_SELECT=0` disables the two-channel spiking basal-ganglia race byte-identically (the
# row STAYS on_by_default:YES). Flipping this to False would turn the faculty OFF by default. See
# research/runners/bg_action_selection_production_organ.py. 2026-08-26 FLIPPED DEFAULT-ON (wave 3 flip, 6/6
# flip-soak GO): the BG SPEAK-vs-STAY-SILENT race ships as the production default.
_BG_SELECT_DEFAULT_ON = True


def _bg_select_flag_on() -> bool:
    """The master switch = the DEFAULT-ON anchor combined with the env override. Enabled UNLESS `BRAIN_BG_SELECT`
    is an explicit off (0/false/no/off/''). Kept as a lightweight env read so the disabled path does no BG-selector
    work -> byte-identical to pre-wiring."""
    v = os.environ.get("BRAIN_BG_SELECT")
    if _BG_SELECT_DEFAULT_ON:
        return not (v is not None and v.strip().lower() in ("0", "false", "no", "off", ""))
    return v is not None and v.strip().lower() in ("1", "true", "on", "yes")


# ─── COMMON GROUND DRIVES THE RESPONSE (2026-08-26): the 6-seed-GO common-ground ledger (a per-referent bistable
# NMDA-attractor store latched by grounding acts + held by recurrence; research/runners/
# _learned_common_ground_ledger_derisk.py) made LOAD-BEARING on the live turn's REFERRING EXPRESSION — a referent
# ALREADY in this conversation's common ground (mentioned earlier, its ledger slot latched + self-sustained) reads
# GROUNDED off the substrate -> audience design wins REDUCE -> the reply LEADS with a reduced/pronominal reference
# ("As for it — <answer>"); a first-mention referent reads UNGROUNDED -> the novelty prior wins INTRODUCE -> NO reduced
# lead (the reply names it in full). The production-integration anchor. When `_CG_DRIVES_DEFAULT_ON` is False the block
# is gated on the env flag alone (`BRAIN_CG_DRIVES=1` opts in for review); the response carries NO `common_ground_drives`
# key and NO reduced-reference lead -> byte-identical. LESION (`BRAIN_CG_DRIVES_LESION=1`): build the ledger recurrence
# at weight 0 -> it cannot HOLD -> even a re-mentioned referent reads ungrounded -> audience design goes static (always
# INTRODUCE) -> the reduced lead VANISHES (the load-bearing proof). See webapp/common_ground_drives_chat.py.
_CG_DRIVES_DEFAULT_ON = True   # 2026-09-01 AUTO-FLIP: PART-A 6/6 GO + PART-B end-to-end no_regression=True (load-bearing, moat-safe, byte-identical-off)


def _common_ground_drives_on() -> bool:
    """The master switch = the anchor combined with the env override. Default-OFF anchor: enabled only on an explicit
    truthy `BRAIN_CG_DRIVES` opt-in. Default-ON anchor (a later flip): enabled UNLESS `BRAIN_CG_DRIVES` is an explicit
    off. Kept lightweight so the disabled path builds no ledger + does no substrate work."""
    try:
        from webapp import common_ground_drives_chat as _CGD
    except Exception:
        return False
    if _CG_DRIVES_DEFAULT_ON:
        return not _CGD.cg_drives_off()
    return _CGD.cg_drives_enabled()


def _get_selfinit_organ(cache_key):
    """The PER-SESSION self-initiation organ (lazy build on the first idle turn). NOT a process singleton: it holds its
    own mouth + selection substrate for THIS conversation; cleared on reset. See
    research/runners/self_initiated_production_organ.py."""
    org = _SESSION_SELFINIT.get(cache_key)
    if org is None:
        from research.runners.self_initiated_production_organ import SelfInitiationOrgan
        org = SelfInitiationOrgan(seed=42)
        _SESSION_SELFINIT[cache_key] = org
    return org


def _get_episodic_organ_existing(cache_key):
    """The session's D5 episodic organ IF one was already built this conversation (do NOT build one just to
    consolidate — between-turn D5 learn-through-use only strengthens a store the live turns already formed).
    Used by the continuous tick's `consolidate_used_memory`. Returns None if no organ exists for this session."""
    try:
        import research.runners.d5_episodic_production_organ as _EP
        return _EP._ORGANS.get(cache_key)
    except Exception:
        return None


def _get_chat_existing(cache_key):
    """The session's live chat IF one was already built this conversation. Used by the continuous tick's DA-encoding
    substrate-homeostasis consolidation pass (`consolidate_substrate_homeostasis`) — never builds a chat just to
    consolidate (the Turrigiano synaptic-scaling pass only refines a store the live turns already wrote). Returns None
    if no chat exists for this session yet."""
    return _BRAIN_CHATS.get(cache_key)


def _get_multiref_organ(cache_key):
    """The PER-SESSION spiking multi-referent WM buffer (lazy build ~0.46s on the first >=2-referent turn). NOT a
    process singleton: the organ's own get_organ() shares a referent codebook across sessions, which would leak
    other conversations' referents into a hold-query read-back; one MultiReferentWMOrgan per cache_key isolates it."""
    org = _SESSION_MULTIREF.get(cache_key)
    if org is None:
        from research.runners.d6_multiref_wm_production_organ import MultiReferentWMOrgan
        # ONE-BRAIN CROSS-EDGE (opt-in): when BRAIN_ONEBRAIN_XEDGE is ON, this PER-SESSION organ shares the PROCESS
        # xedge pool's d6 slice (so the frozen w{k}->sel cross-edge is live) while KEEPING its own per-session
        # referent codebook -> referent-isolation preserved (no bleed across sessions; the shared spiking slice
        # carries only transient bumps, no cross-session meaning). OFF (default) or on any failure -> shared=None,
        # byte-identical to before (its own bridge).
        shared = None
        try:
            from research.runners.onebrain_xedge_production import xedge_enabled, get_xedge_pool
            if xedge_enabled():
                _xp = get_xedge_pool(42)
                shared = _xp.pool if _xp is not None else None
        except Exception:
            shared = None
        org = MultiReferentWMOrgan(seed=42, shared=shared)
        _SESSION_MULTIREF[cache_key] = org
    return org


def _get_silent_wm_organ(cache_key):
    """The PER-SESSION activity-silent-WM organ (Mongillo 2008). Holds this conversation's discourse FOCUS in short-term
    facilitation across a distractor and reactivates it via a nonspecific ping. PER-SESSION (not a process singleton:
    the organ's focus/binding is conversation-specific, and a shared instance would leak one conversation's focus into
    another's recall — mirrors _SESSION_MULTIREF). Lazy: no bridge is built until a recall query actually fires."""
    org = _SESSION_SILENT_WM.get(cache_key)
    if org is None:
        from research.runners.activity_silent_wm_production_organ import ActivitySilentWMOrgan
        org = ActivitySilentWMOrgan(seed=42)
        _SESSION_SILENT_WM[cache_key] = org
    return org


def _get_reconsolidation_organ():
    """The process-shared reconsolidation (belief-revision) organ. Its D2 window gate IS the shared surprise organ,
    so warming surprise warms it (built once on first use). See research/runners/reconsolidation_production_organ.py."""
    from research.runners.reconsolidation_production_organ import get_organ
    return get_organ(seed=42)


def _get_causal_organ(cache_key):
    """The PER-SESSION spiking causal why/what-if forward-model organ (lazy build ~1-2s on the first causal turn).
    Keyed by the ChatBrain cache (NOT a process singleton): the organ grounds READ-ONLY against THIS brain's live
    composer (its event set + causal curriculum are gated by that composer's moat recall), so a per-brain organ
    keeps one brain's grounding out of another's. Cleared on reset. See research/runners/causal_whatif_production_organ.py."""
    from research.runners.causal_whatif_production_organ import get_organ
    return get_organ(cache_key, seed=42)


def _get_noncontradiction_organ():
    """The process-shared non-contradiction assertion-gate organ (stateless; reads the ONE production recall composer
    directly — no co-resident bridge added). See research/runners/b3_noncontradiction_production_organ.py."""
    from research.runners.b3_noncontradiction_production_organ import get_organ
    return get_organ(seed=42)


def _get_pmem_organ(cache_key):
    """The PER-SESSION spiking prospective-memory organ (lazy build on the first intention-formation turn, ~seconds
    on numpy — the substrate calibration is process-cached per seed). NOT a process singleton: the held deferred
    intention + the persistent latch bridge that carries it across turns are THIS conversation's; one organ per
    cache_key isolates them. Cleared on reset. See research/runners/prospective_memory_production_organ.py."""
    org = _SESSION_PMEM.get(cache_key)
    if org is None:
        from research.runners.prospective_memory_production_organ import ProspectiveMemoryOrgan
        org = ProspectiveMemoryOrgan(seed=42)
        _SESSION_PMEM[cache_key] = org
    return org


def _episodic_store_ok() -> bool:
    """Whether the D5 episodic BTSP WRITE (Hook B) may run this turn. A BTSP store is ~seconds on cupy (the
    production substrate) but ~510s/topic on numpy@2000 — so on numpy the write is DEFERRED (a declared latency
    residual: the recall GATE is still spiking + load-bearing; only the WRITE is amortized to the cupy deployment).
    `BRAIN_EPISODIC_STORE` (1/true/on force-ON, 0/off force-OFF) overrides the backend gate for tests/deployments."""
    import os as _os
    v = _os.environ.get("BRAIN_EPISODIC_STORE")
    if v is not None:
        return v.strip().lower() not in ("0", "false", "no", "off", "")
    try:
        from sim.backend import get_backend
        return get_backend()[1] == "cupy"
    except Exception:
        return False


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
    the GPU; the chat endpoint must not contend for it unasked).

    EXPLICIT OVERRIDE (2026-08-21): `BRAIN_CHAT_RENDERER={qwen,stub,raw}` forces
    the renderer regardless of GPU/backend. This is how a GPU-LESS host opts INTO
    the CPU-Qwen fluent mouth (the model now loads float32 on CPU — real prose,
    just slower; see SpikingQwenFaculty). Unset -> the GPU-light auto-default
    below (byte-identical to prior behavior)."""
    _forced = os.environ.get("BRAIN_CHAT_RENDERER", "").strip().lower()
    if _forced in ("qwen", "stub", "raw"):
        return _forced
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


# The curated 15k wikidata knowledge core SHIPS as the default cortical LONG-TERM store (board #133).
# Soak verdict go=True: 6/6 seeds byte-identical vs the unsharded oracle, 0 confabulations, first-match
# recall 1.0, latency median ~0.4 s (vocab 7,032). It sits BEHIND the developed brain's small conversation
# working-set as a routed TieredFactStore LTM tier -- the buffer (WM) keeps its D5 learn-through-use
# plasticity, so this is biological memory (learn-through-use over a persistent store), not static RAG.
def _default_ltm_bundle_dir():
    """First existing candidate dir for the shipped curated core, or None if the data lake is absent.
    Candidates (in order): $BRAIN_DATA_ROOT, the sibling-of-checkout `sim-data` (the primary-checkout
    production layout), and ~/Projects/sim-data (the canonical machine path, robust when the server runs
    from a worktree). Returning None means a checkout without the data lake stays byte-identical (no LTM)."""
    roots = []
    env_root = os.environ.get("BRAIN_DATA_ROOT", "").strip()
    if env_root:
        roots.append(env_root)
    roots.append(str(REPO_ROOT.parent / "sim-data"))
    roots.append(str(Path.home() / "Projects" / "sim-data"))
    seen = set()
    for r in roots:
        if r in seen:
            continue
        seen.add(r)
        d = str(Path(r) / "knowledge_bundles" / "wikidata_core_15k")
        if os.path.isdir(d):
            return d
    return None


# 2026-08-26 FLIPPED DEFAULT-ON (board #133): the curated 15k core ships as the default cortical LTM.
# BRAIN_LTM_SHIP_DEFAULT=0 (and/or BRAIN_LTM_BUNDLE=off) is the byte-identical escape to the pre-flip no-LTM path.
_LTM_SHIP_DEFAULT_ON = True


def _ltm_ship_default_on() -> bool:
    """The ship-default knob. Unset -> _LTM_SHIP_DEFAULT_ON (the curated core is the default LTM, ON). Set
    BRAIN_LTM_SHIP_DEFAULT to 1/true/on/yes to force on, or 0/false/off/no for the byte-identical no-LTM escape."""
    env = os.environ.get("BRAIN_LTM_SHIP_DEFAULT")
    if env is None:
        return _LTM_SHIP_DEFAULT_ON
    return env.strip().lower() in ("1", "true", "on", "yes")


# PRODUCTION-DEFAULT knobs for the tiny-demo brain-chat load, promoted to NAMED constants so the CLASS-PI
# ledger (docs/PRODUCTION_INTEGRATION_LEDGER.yaml) can machine-anchor them (a flip of the default here now
# BLOCKS at commit until the ledger row moves). Same convention as _LTM_SHIP_DEFAULT_ON above.
# _COMPOSER_KIND_DEFAULT: 'onebrain' = the GENUINELY-SPIKING recall is the production default (BRAIN_COMPOSER_KIND
#   overrides; 'rf' is the numpy fast-path escape). _CONTINUOUS_DRIVES_DEFAULT: '1' = the between-turn wander
#   LEADS the next turn by default (BRAIN_CONTINUOUS_DRIVES=0 is the byte-identical escape).
_COMPOSER_KIND_DEFAULT = "onebrain"
_CONTINUOUS_DRIVES_DEFAULT = "1"

# LTM-STORE DECODE LEVERS (board #108 cluster, 2026-09-02): the #66 seed-44 recall-hole arc
# (research/FAILURE_LOG.md row 93, finding 2026-09-01-seed44-recall-hole-ROOT-CAUSED-phase-quantization-
# decode-escalation-fix.md) root-caused a genuine, seed-INDEPENDENT RF phase-readout quantization miss at
# knowledge-base scale and built two additive `RFPhasorComposer` levers to close it. Both DEFAULT ON at ANY
# bundle scale (not gated to the 100k bundle specifically): `enable_codebook_cache` is a pure memoization of a
# deterministic recomputation (byte-identical BY CONSTRUCTION -- rf_phasor_composer.py::_ensure_codebook_cache
# rebuilds exactly what the uncached path would); `enable_decode_escalation` only ever RECOVERS a fact the
# coarse readout mis-decoded (a confidence-gated finer-period re-examination of a near-tie candidate -- it can
# only turn a wrong/None answer into the correct one, never the reverse, moat-safe by construction per
# tests/test_decode_escalation_seed44_hole.py). Verified NO-OP on the shipped 15k core (0 oracle-parity
# mismatches with the flags OFF *and* ON, on the SAME `_knowledge_scale_100k_production_verify.py --bundle
# <15k core>` battery -- see research/findings/raw/_flip108_r2_wiring/). Row 94's "NEWLY FOUND" gap this
# closes: neither live LTM-load path threaded these flags at all before this change. Escape:
# BRAIN_LTM_CODEBOOK_CACHE=0 / BRAIN_LTM_DECODE_ESCALATION=0 restore the exact pre-2026-09-02 behavior.
_LTM_CODEBOOK_CACHE_DEFAULT_ON = True
_LTM_DECODE_ESCALATION_DEFAULT_ON = True


def _ltm_codebook_cache_on() -> bool:
    env = os.environ.get("BRAIN_LTM_CODEBOOK_CACHE")
    if env is None:
        return _LTM_CODEBOOK_CACHE_DEFAULT_ON
    return env.strip().lower() in ("1", "true", "on", "yes")


def _ltm_decode_escalation_on() -> bool:
    env = os.environ.get("BRAIN_LTM_DECODE_ESCALATION")
    if env is None:
        return _LTM_DECODE_ESCALATION_DEFAULT_ON
    return env.strip().lower() in ("1", "true", "on", "yes")


def _resolve_ltm_bundle():
    """Resolve the cortical LTM bundle path (returns a dir string, or None for the byte-identical no-LTM path).
      * BRAIN_LTM_BUNDLE=<path>                      -> that bundle (explicit override, unchanged).
      * BRAIN_LTM_BUNDLE=''/off/0/false/no/none      -> None (explicit disable).
      * BRAIN_LTM_BUNDLE unset  -> the shipped curated core IF the ship-default knob is on AND the bundle dir
        exists on disk (a checkout without the data lake degrades to None, byte-identical); else None.
    The escape from the shipped default is BRAIN_LTM_SHIP_DEFAULT=off (its own on/off knob)."""
    raw = os.environ.get("BRAIN_LTM_BUNDLE")
    if raw is not None:
        raw = raw.strip()
        if raw.lower() in ("", "off", "0", "false", "no", "none"):
            return None
        return raw
    if _ltm_ship_default_on():
        return _default_ltm_bundle_dir()
    return None


def _load_or_build_ltm_store(ltm_bundle: str, seed: int = 42, n_shards=None, D: int = 128,
                              enable_codebook_cache: bool = None, enable_decode_escalation: bool = None):
    """Load (fast path) or build (fallback) a `ShardedPhasorStore` LTM from `ltm_bundle`, mirroring
    `developed_brain_io.load_developed_brain`'s OWN `ltm_bundle` handling EXACTLY (same fast-path-persisted-
    store-first, then build-from-facts.json precedence) -- factored out here so the default tiny-demo brain
    (reasoning-frontier, 2026-08-25) gets the IDENTICAL LTM-attach behavior the developed-brain path already
    has, without duplicating/diverging the load-vs-build decision. Returns None if the bundle has neither a
    loadable sharded-store manifest nor a facts.json to build from (a missing/empty bundle degrades quietly;
    the caller treats None as "no LTM available").

    `enable_codebook_cache`/`enable_decode_escalation`: None (default) resolves via `_ltm_codebook_cache_on()`/
    `_ltm_decode_escalation_on()` (board #108 cluster; ON by default at any bundle scale, BRAIN_LTM_CODEBOOK_CACHE=0/
    BRAIN_LTM_DECODE_ESCALATION=0 escape) -- an explicit True/False here overrides that resolution (used by callers
    that need a specific setting, e.g. a byte-identical-comparison harness)."""
    if enable_codebook_cache is None:
        enable_codebook_cache = _ltm_codebook_cache_on()
    if enable_decode_escalation is None:
        enable_decode_escalation = _ltm_decode_escalation_on()
    from research.runners.sharded_phasor_store import ShardedPhasorStore
    manifest_path = Path(ltm_bundle) / "manifest.json"
    if manifest_path.exists():
        try:
            mani = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            mani = {}
        if isinstance(mani, dict) and "n_shards" in mani:
            extra_kwargs = {}
            if enable_codebook_cache:
                extra_kwargs["enable_codebook_cache"] = True
            if enable_decode_escalation:
                extra_kwargs["enable_decode_escalation"] = True
            return ShardedPhasorStore.load(str(ltm_bundle), extra_kwargs=extra_kwargs or None)
    from research.runners.developed_brain_io import _load_facts_json
    from research.runners.tiered_fact_store import build_ltm_from_facts, auto_n_shards
    ltm_facts = _load_facts_json(ltm_bundle)
    if not ltm_facts:
        return None
    ns = int(n_shards) if n_shards is not None else auto_n_shards(len(ltm_facts))
    cb_kwargs = {}
    if enable_codebook_cache:
        cb_kwargs["enable_codebook_cache"] = True
    if enable_decode_escalation:
        cb_kwargs["enable_decode_escalation"] = True
    return build_ltm_from_facts(ltm_facts, n_shards=ns, seed=int(seed), D=int(D), composer_kwargs=cb_kwargs or None)


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
        # BRAIN_COMPOSER_KIND (default 'onebrain' = the GENUINELY-SPIKING recall, per _COMPOSER_KIND_DEFAULT above;
        # corrected — this comment previously said "default 'rf'", stale since the flip): set 'rf' for the numpy
        # fast-path escape instead (resonate-and-fire per query, runtime new-word LEARN via vocab_headroom is the
        # onebrain path). The onebrain build is ~180s (speed secondary); it is the brain-based-only recall the
        # mission requires. Env-gated so BRAIN_COMPOSER_KIND=rf reverts to the pre-flip numpy fast path.
        _ck = os.environ.get("BRAIN_COMPOSER_KIND", _COMPOSER_KIND_DEFAULT)
        agent, aliases, _n = _build_tiny_demo(42, use_multiturn=True,
                                              enable_neural_render=False, composer_kind=_ck)
        source = "tiny-demo"
        # KNOWLEDGE-SCALE ON THE DEFAULT BRAIN (reasoning-frontier, 2026-08-25 -- board #133 extension). The
        # 2026-08-25 integrated-conversational-state diagnostic found the shipped 15k curated core attached ONLY
        # to the developed-brain path (below); the owner's DEFAULT out-of-box chat brain ('tiny-demo') had NO
        # access to it -- every knowledge query on the default brain abstained even though the store itself
        # answers correctly in isolation. Mirrors the developed-brain path's OWN attach exactly (same
        # `_resolve_ltm_bundle()` + `TieredFactStore` composition, same fast-path-persisted-store / build-from-
        # facts precedence as `developed_brain_io.load_developed_brain`): the tiny-demo composer stays the
        # small recent-conversation BUFFER; a buffer abstain falls through to the routed LTM shard. Guarded so a
        # missing/corrupt bundle degrades to the byte-identical no-LTM tiny-demo (never crashes brain load).
        # Escape: `BRAIN_LTM_SHIP_DEFAULT=off` (or `BRAIN_LTM_BUNDLE=off`) -> `_resolve_ltm_bundle()` returns
        # None -> this block is skipped entirely -> byte-identical to the pre-2026-08-25 tiny-demo.
        _tiny_ltm_bundle = _resolve_ltm_bundle()
        if _tiny_ltm_bundle is not None:
            try:
                from research.runners.developed_brain_io import _inner_agent
                from research.runners.tiered_fact_store import TieredFactStore
                _tiny_ltm = _load_or_build_ltm_store(_tiny_ltm_bundle, seed=42)
                if _tiny_ltm is not None:
                    _tiny_inner = _inner_agent(agent)
                    _tiny_inner.composer = TieredFactStore(_tiny_inner.composer, _tiny_ltm)
                    source = "tiny-demo +LTM"
            except Exception as _tiny_ltm_exc:
                print(f"[webapp] tiny-demo LTM attach failed (degrading to no-LTM tiny-demo): "
                      f"{type(_tiny_ltm_exc).__name__}: {_tiny_ltm_exc}", flush=True)
    elif brain in ("self-knowledge", "self_knowledge", "self"):
        agent, aliases, _n = _load_self_knowledge(
            _SK_CODES, _SK_CURRICULUM, 42, True, False)
        source = "self-knowledge"
    elif is_developed_brain_bundle(_resolve_bundle(brain)):
        bundle = _resolve_bundle(brain)
        # (KNOWLEDGE-SCALE) BRAIN_LTM_BUNDLE points at a separate bundle of bulk KNOWLEDGE that becomes a routed
        # cortical LONG-TERM store beside the developed brain's small conversation working-set -- so the brain can
        # answer over an LLM-scale body of facts at sub-second latency, beyond the k_max=32 co-resident cap.
        # SHIP DEFAULT (board #133): the curated 15k core is ON by default; BRAIN_LTM_SHIP_DEFAULT=off restores the
        # byte-identical no-LTM path. See _resolve_ltm_bundle() above and tiered_fact_store.py.
        _ltm_bundle = _resolve_ltm_bundle()
        agent, manifest = load_developed_brain(bundle, use_multiturn=True,
                                               enable_neural_render=False,
                                               ltm_bundle=_ltm_bundle,
                                               enable_codebook_cache=_ltm_codebook_cache_on(),
                                               enable_decode_escalation=_ltm_decode_escalation_on())
        aliases = set(manifest.get("self_aliases") or []) | set(DEFAULT_SELF_ALIASES)
        source = f"developed-brain:{brain}" + (" +LTM" if _ltm_bundle else "")
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
    # ── BEGIN faculty:vision-identity (BRAIN_VISION_IDENTITY, default ON 2026-08-26) ──
    # The ENVIRONMENT's visual percept for a 'what do you see?' turn: an object
    # descriptor the neural retina/V1 then receive ('bird'/'fish'/'0'/'1', with
    # an optional '#<exemplar>' suffix). Additive + optional; IGNORED entirely
    # when BRAIN_VISION_IDENTITY=0 or the message is not a visual query (no percept
    # on the turn is the byte-identical no-op case regardless of the flag), so
    # existing callers are byte-identical. See research/runners/vision_identity_production_organ.py.
    percept: str | None = None
    # ── END faculty:vision-identity ──


def _json_safe(obj, _path="", _bad=None):
    """Coerce a response payload to STRICT-JSON-safe values for Starlette's JSONResponse (which serializes with
    allow_nan=False and REJECTS NaN/Inf with 'Out of range float values are not JSON compliant'). A non-finite float
    -- e.g. a faculty metadata read that divided 0/0 on the cupy path -- becomes None, and its key-path is collected
    so the caller can LOG which field was non-finite (a null with a breadcrumb, never a silent 500). numpy/cupy 0-d
    scalars are coerced to python scalars. Host/transport boundary ONLY: it touches serialization, never cognition."""
    import math
    if _bad is None:
        _bad = []
    if hasattr(obj, "item") and not isinstance(obj, (str, bytes, dict, list, tuple, bool, int, float)):
        try:
            obj = obj.item()  # numpy/cupy 0-d scalar -> python scalar
        except Exception:
            _bad.append((_path or "<root>") + "(unserializable)")
            return None
    if isinstance(obj, float):
        if not math.isfinite(obj):
            _bad.append(_path or "<root>")
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _json_safe(v, ("%s.%s" % (_path, k)) if _path else str(k), _bad) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v, "%s[%d]" % (_path, i), _bad) for i, v in enumerate(obj)]
    return obj


def _safe_json_response(resp, where):
    """Sanitize a brain_chat response for strict JSON + log any non-finite fields ONCE, so a NaN/Inf from a faculty
    read (notably on the cupy substrate) degrades to null-with-a-breadcrumb instead of 500-ing the whole turn."""
    bad = []
    safe = _json_safe(resp, _bad=bad)
    if bad:
        print("[webapp] brain_chat(%s): %d non-finite field(s) nulled for JSON: %s"
              % (where, len(bad), ", ".join(str(b) for b in bad[:12])), flush=True)
    return JSONResponse(safe)


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
    try:  # CONTINUOUS-STATE ENGINE: mark this session active so the idle tick doesn't run mid-conversation
        from webapp import continuous_engine as _CE0
        _CE0.mark_request(cache_key)
    except Exception:
        pass
    if req.reset:
        _BRAIN_CHATS.pop(cache_key, None)
        _BRAIN_RICH.pop(cache_key, None)   # drop the rich discourse thread too
        try:
            from webapp import continuous_engine as _CE1
            _CE1.forget_session(cache_key)  # drop this session's between-turn inner-life + last-request
        except Exception:
            pass
        _SESSION_MOOD.pop(cache_key, None)  # clear the mood STATE (reset-between-topics)
        _SESSION_WORLDVIEW.pop(cache_key, None)  # clear the affective forward-model STATE (E2)
        _SESSION_MULTIREF.pop(cache_key, None)  # drop the multi-referent WM buffer (D6, per-session discourse state)
        _SESSION_SILENT_WM.pop(cache_key, None)  # drop the activity-silent WM focus/binding (Mongillo, per-session)
        _SESSION_SELFINIT.pop(cache_key, None)  # drop the self-initiation organ (its own mouth + selection substrate)
        _SESSION_DISCOURSE.pop(cache_key, None)  # drop the running discourse event-pair turn state (D3)
        _SESSION_PMEM.pop(cache_key, None)  # drop the held prospective intention + its latch bridge (Gate-B pmem)
        try:  # drop this conversation's episodic memory (D5, Hook C) — mirrors _SESSION_MOOD/_SESSION_WORLDVIEW
            import research.runners.d5_episodic_production_organ as _EP_reset
            _EP_reset.reset_episodic_organ(cache_key)
        except Exception:
            pass
        try:  # drop this brain's causal why/what-if organ (T1-4) so a re-taught brain re-grounds against its composer
            import research.runners.causal_whatif_production_organ as _CA_reset
            _CA_reset.reset_organ(cache_key)
        except Exception:
            pass

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

    # THE SHARED FULL-FACULTY PIPELINE (extracted 2026-08-27): everything from here — the gate-wrapper installs +
    # the entire faculty-DRIVE coupling sequence + the response assembly — now lives in `brain_reply`, so the
    # standalone TUI (research/runners/brain_chat_tui) + the OpenAI shim run the IDENTICAL sequence, not just this
    # handler. The handler above is now only request parsing + brain build/cache. See webapp/brain_reply.py.
    return brain_reply(chat, req, source, cache_key)


def brain_reply(chat, req, source, cache_key) -> JSONResponse:
    """THE SHARED FULL-FACULTY TURN PIPELINE — the entire faculty-DRIVE coupling sequence run on an already-built
    ChatBrain, returning the SAME JSONResponse the `/api/brain-chat` handler has always returned. Extracted
    VERBATIM from the old inline `brain_chat` body (so the webapp path is byte-identical); the standalone TUI +
    the OpenAI shim reach it through `webapp/brain_reply.py`, so a coupling added here reaches every surface at
    once. Faculty couplings belong HERE, never inline in a request handler (the standing discipline)."""
    # GNW N-ORGAN IGNITION BUS — DEFAULT ORGAN-COMBINATION (T1-1 Phase-B FLIP -> SCAFFOLD-RETIREMENT, 2026-08-13):
    # `install_bus_gate` idempotently wraps `chat.gate` so the SUBSTRATE (consensus-ignition + WTA) AUTHORS the organ-
    # combination verdict — the ignited patient IS the answer, no ignition IS the abstain (the moat as a substrate
    # property) — REPLACING the host `if recalled == p`. RETIREMENT: the wrapper runs `chat.gate_extract` (extraction +
    # acquisition/anaphora/open-ended side effects only) and lets `gate_via_bus` commit/veto the COVERED class WITHOUT
    # EVER COMPUTING the host combination (no `_substrate_recall` / `_gate_router_combine` on a routable factual
    # recall) — the flip only OVERRODE the host verdict; this never computes it. Both the single-fact path and the
    # default rich path funnel their direct recall through `chat.gate`, so this covers the DEFAULT turn.
    # Extraction/comprehension + the OUT-OF-SCOPE classes (self/identity, open-ended, acquisition) stay HOST-authored.
    # ESCAPE: `BRAIN_GNW_BUS_HOST=1` reverts to the original host gate() (byte-identical to pre-flip production).
    # Honest-negative lever: `BRAIN_GNW_BUS_LESION=1` collapses the answer to abstain (ignition load-bearing). Guarded
    # so a wiring failure can never crash a turn (degrades to host gate()). Earned on a broad byte-identical panel +
    # a call-count retirement proof (_gnw_bus_default_flip_verify, _gnw_bus_scaffold_retire_verify). See gnw_bus_shadow.py.
    try:
        from webapp import gnw_bus_shadow as _gnw_bus_mod
        _gnw_bus_mod.install_bus_gate(chat)
    except Exception:
        pass

    # GNW TWO-GENUINELY-DISTINCT-ORGANS coincidence bus — DEFAULT-ON (2026-08-20). The N-organ bus above combines three
    # reads that ALL come from the composer (forward + VERIFY + reverse-binding); this layers the genuinely-distinct
    # SECOND SPIKING ORGAN the `_gnw_two_distinct_organs_derisk` 6-seed GO closes caveat #1 with — the production
    # `SurpriseProductionOrgan` (a spiking predictive-coding mismatch circuit, `cp_firing_states[surprise]`). The covered
    # routable recall is authored by the COINCIDENCE of organ A (composer recall) + organ B (the surprise monitor
    # corroborating against its OWN expectation): both agree -> ignite -> commit; organ B contradicts / is lesioned ->
    # consensus-veto -> abstain. WORKS ON BOTH BACKENDS: the backend-neutral Izhikevich threshold init (default-on in
    # build_expectation_circuit) makes the surprise organ discriminate on cupy too; the bus's own discrimination gate
    # (`_organ_discriminates`) falls back to a safe inert (byte-identical) path if that fix is off, so it never runs a
    # mis-discriminating organ. Escape: `BRAIN_GNW_2ORGAN=0`/`off` disables (byte-identical to the pre-bus path). Cleared
    # the default-on gate: 6-seed GO + production wired end-to-end on cupy+numpy + zero regression off-vs-on. Load-bearing
    # levers: BRAIN_GNW_2ORGAN_WS_LESION / _ORGANB_LESION. NO sim/ edit. See webapp/gnw_two_organ_bus.py.
    if os.environ.get("BRAIN_GNW_2ORGAN", "on").strip().lower() in ("1", "true", "on", "yes"):
        try:
            from webapp import gnw_two_organ_bus as _gnw_2organ_mod
            _gnw_2organ_mod.install_two_organ_gate(chat)
        except Exception:
            pass

    # GNW THREE-GENUINELY-DISTINCT-ORGANS consensus bus — DEFAULT-OFF (2026-08-21). Extends the DEFAULT-ON 2-organ bus
    # (organ A = composer recall + organ B = the spiking surprise monitor) with a THIRD genuinely-distinct spiking
    # organ: organ C = the production COMPREHENSION monitor (`ComprehensionProductionOrgan`, the 6/6-GO D4 faculty — a
    # Wong-Wang `SpikingRoleCompetition` sel-pool WTA read off `cp_firing_states`). Organ C reads whether the RECALLED
    # PROPOSITION (agent, action, cand) is role-RESOLVABLE; on a LOW comprehension margin (role-ambiguous) it WITHHOLDS,
    # so the Q=3 consensus (all three must vote for the workspace to ignite) ABSTAINS instead of committing a recall the
    # brain did not comprehend — a consensus-veto the 2-organ bus could not make. DEFAULT-OFF (unset -> this block is
    # skipped, the module is never imported, the turn is BYTE-IDENTICAL to the 2-organ-bus production path). Escape: the
    # flag simply unset. Load-bearing lever: `BRAIN_GNW_3ORGAN_ORGANC_LESION=1` silences organ C's veto (it corroborates
    # unconditionally) -> the consensus collapses to the 2-organ decision (the low-comprehension abstain reverts to
    # commit). Guarded so a wiring failure can never crash a turn. NO sim/ edit. See webapp/gnw_three_organ_bus.py.
    # 2026-08-21 FLIPPED default-ON (real-vocab comprehension read fixed the over-veto: composed re-verify GO,
    # diverged:0 on the out-of-scope panel with BRAIN_GNW_3ORGAN isolated; genuine non-comprehension still vetoes,
    # lesion severs). BRAIN_GNW_3ORGAN=0 is the byte-identical escape (delegates to the 2-organ bus).
    if os.environ.get("BRAIN_GNW_3ORGAN", "1").strip().lower() in ("1", "true", "on", "yes"):
        try:
            from webapp import gnw_three_organ_bus as _gnw_3organ_mod
            _gnw_3organ_mod.install_three_organ_gate(chat)
        except Exception as _gnw3_exc:
            # fail-SAFE (degrade to the 2-organ path, never crash a turn) but NOT silent — a default-on faculty that
            # silently no-installs is the exact silent-failure class; log the catch so a wiring break is observable.
            print(f"[webapp] GNW three-organ install failed; degrading to 2-organ bus: "
                  f"{type(_gnw3_exc).__name__}: {_gnw3_exc}", flush=True)

    # GNW CONFIDENCE/CONFLICT-GATED DELIBERATION — THE KEYSTONE, WIRED (T1-1 rung d, 2026-08-18): after the bus commits,
    # the WORKSPACE's OWN spiking conflict read (n_ignited + the nmda_norm confidence balance) DECIDES commit-vs-abstain.
    # When the brain has >=2 genuinely-competing stored answers under the SAME (agent, action) (today's bus commits the
    # arbitrary FIRST-match), the candidates are driven EQUALLY into the P1.2 GNW workspace; a sustained co-ignition /
    # low-confidence read (the keystone acc_conflict_gate) makes the brain ABSTAIN ("I don't know") instead of committing
    # the shaky answer (deliberation-until-sure + halt-if-unsure — the "ACT on the conflict/confidence signals we only
    # REPORT" audit item). DEFAULT-ON (BRAIN_GNW_DELIBERATE=0 -> pure pass-through, byte-identical). LESION lever
    # (BRAIN_GNW_DELIBERATE_LESION=1): the conflict read runs on the recurrence-ZEROED workspace -> the conflict cannot
    # co-ignite -> the brain commits the shaky answer again (the abstain is the SPIKING competition, not a host len()).
    # MOAT-safe: it can ONLY ADD abstentions on a genuine multi-answer conflict; never un-abstains, never invents a fact,
    # never flips a confident single-answer recall (n_ignited==1). Reuse-by-import (NO sim/ edit). See webapp/gnw_deliberation.py.
    if _GNW_DELIBERATE_DEFAULT_ON:
        try:
            from webapp import gnw_deliberation as _gnw_delib_mod
            _gnw_delib_mod.install_deliberation_gate(chat)
        except Exception:
            pass

    # ── BEGIN faculty wire-in: VALUE-DRIVEN CHOICE (RANK-1 value-critic GO; DEFAULT-ON 2026-08-26) ─────────────
    # Owner directive "make the brain COMMIT [by value] instead of abstaining/guessing": on a >=2-distinct-patient
    # (agent, action) recall — the ambiguity the GNW chain today resolves by an ARBITRARY FIRST-MATCH (verified live)
    # or a halt-if-unsure abstain — COMMIT the higher-VALUE patient instead, scored by the brain's OWN LEARNED spiking
    # striosome_value critic (DA-gated STDP) and committed by a spiking value-WTA (Wang-2002 biased competition). GO:
    # research/findings/2026-07-23-value-critic-closure-RANK1-GO.md (6/6 seeds). The wrapper is installed OUTSIDE the
    # deliberation gate (INSIDE the multistep gate below, so chase-form questions keep precedence). It touches ONLY the
    # >=2-competing case (a <2-candidate turn -- confident single recall, single-patient/untaught abstain, self -- is
    # returned VERBATIM) and commits only STORED candidates (never invents a fact -> the moat holds). On decline
    # (lesion / non-decisive value) it returns the INNER pipeline result verbatim -> reverts to EXACTLY what the chain
    # would have done. FLIPPED DEFAULT-ON (wave 1/2 flip, 6-seed pool soak GO: ordinary byte-identical 6/6,
    # load-bearing 6/6, research/runners/_value_choice_flip_soak.py): BRAIN_VALUE_CHOICE=0 is the byte-identical
    # escape -> NOT installed -> chat.gate stays the pure existing chain. Load-bearing levers:
    # BRAIN_VALUE_CHOICE_LESION=1 (pin V to the mean -> the value gradient vanishes -> reverts to the inner result),
    # BRAIN_VALUE_CHOICE_UNTRAINED=1 (untrained critic -> the learned engagement-advantage vanishes). Guarded so a
    # wiring failure can never crash a turn. Reuse-by-import of the R5b organ (NO sim/ edit).
    # See research/runners/value_choice_production_organ.py.
    try:
        from research.runners import value_choice_production_organ as _vc_mod
        if _vc_mod.value_choice_enabled():
            _vc_mod.install_value_choice(chat)
    except Exception as _vc_exc:
        print(f"[webapp] value-choice install skipped ({type(_vc_exc).__name__}: {_vc_exc})", flush=True)
    # ── END faculty wire-in: VALUE-DRIVEN CHOICE ──────────────────────────────────────────────────────────────

    # GNW MULTI-STEP re-entrant deliberation — THE KEYSTONE'S DEFERRED RUNG, WIRED LIVE (T1-1 rung d, 2026-08-19; flipped
    # to DEFAULT-ON as a production-default 2026-08-19 after the 6/6-seed live GO): the single-hop deliberation gate above
    # decides ONCE (halt-if-unsure). This wires the other half — "deliberation-until-sure over a CHAIN": on an explicit
    # chase-form question ("what does X <action> all the way / to the end?") the WORKSPACE cycles the partial answer back
    # through itself, re-igniting, and the substrate's OWN spiking read (n_ignited off cp_firing_states) decides how many
    # cycles to run — NOT a host `query_chain(cue, actions)` counter — halting when the leaf collapses ignition. DEFAULT-ON
    # (the wrapper is always installed; BRAIN_GNW_MULTISTEP=0 -> pure pass-through, byte-identical to the pre-flip default).
    # LESION lever (BRAIN_GNW_MULTISTEP_LESION=1): the chase runs on the recurrence-ZEROED workspace -> ignition cannot
    # sustain -> the multi-step terminal is NOT reached (the emergent stopping is the SPIKING competition, not a host
    # loop). MOAT-safe: never un-abstains, abstains on an unstored/over-run chase, never invents a fact. Non-chase turns
    # are byte-identical (the wrapper is inert without a chase marker). Reuse-by-import of the 6/6-seed-GO keystone
    # de-risk (NO sim/ edit). See webapp/gnw_multistep_deliberation.py.
    if _GNW_MULTISTEP_DEFAULT_ON:
        try:
            from webapp import gnw_multistep_deliberation as _gnw_multistep_mod
            _gnw_multistep_mod.install_multistep_gate(chat)
        except Exception:
            pass

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

    # ── GNW NEURAL THOUGHT-SWAP — the LIVE held-topic workspace (board #77, 2026-08-19) ─────────────────────────────
    # The already-wired GNW gates RESTORE their workspace each turn -> they do not hold a thought ACROSS turns. This
    # block adds the missing cross-turn held-content register: a per-session GNW swap workspace whose one ignited
    # coalition IS the current conversational TOPIC. Each turn the user message's grounded topic (host comprehension of
    # the world input — the SAME declared boundary the SVO parser occupies) is presented to the reused 6/6-seed-GO
    # neural swap machinery: a DIFFERENT salient topic is a mismatch -> the spiking mismatch/salience detector fires ->
    # the incumbent coalition self-evicts (recurrence depression) -> the neural vacancy gate admits the newcomer (a
    # SWAP); the SAME topic MATCHES -> the pred interneuron vetoes the detector -> the current thought persists (NO
    # swap). The swap-vs-hold VERDICT is the substrate's, not a host `if`. ONE neural swap runs per turn.
    #
    # SWAP DRIVES THE RESPONSE (board #85) — the SOLE swap path. When the drives path is on (`_swap_drives_on()`,
    # default-ON anchor) it runs the single neural swap (with the lesion threaded) AND maps the verdict to a
    # topic-transition LEAD prepended to the answer OUTERMOST — on a SWAP the reply announces the shift ("On
    # <newtopic>, then -- <answer>"); a HOLD stays silent. `BRAIN_SWAP_DRIVES=0` -> the block is fully skipped (no key,
    # no lead, and NO swap observe runs at all — the correct "off") -> byte-identical oracle. LESION
    # (`BRAIN_SWAP_DRIVES_LESION=1`): silence the mismatch detector -> the swap collapses -> the lead VANISHES (the
    # load-bearing proof). The older #77 OBSERVER fallback (gnw_thought_swap.observe_turn, gated on BRAIN_GNW_SWAP) is
    # RETIRED from the default handler (2026-08-20): swap-drives SUPERSEDES it and internally reuses the SAME #77
    # machinery (swap_drives_chat imports gnw_thought_swap), so exactly one neural swap still runs per turn (never
    # double-advancing the workspace). Reuse-by-import (NO sim/ edit). Guarded so a wiring failure can never crash a
    # turn. See swap_drives_chat.py.
    swap_drives_info = None
    swap_drives_lead = ""
    if _swap_drives_on():
        try:
            from webapp import swap_drives_chat as _SDC
            swap_drives_info = _SDC.observe_turn(chat, msg)
            swap_drives_lead = str(swap_drives_info.get("lead", "") or "")
        except Exception as _sde:  # never let the swap coupling crash a turn — degrade to the un-led answer
            swap_drives_info = {"on": True, "error": f"{type(_sde).__name__}: {_sde}", "lead": ""}
            swap_drives_lead = ""

    # ── SELF-INITIATED UTTERANCE — the first INTERNALLY-GENERATED turn class (2026-08-18) ────────────────────────
    # On an IDLE/EMPTY turn (an EMPTY message, or a bare "say something / what's on your mind" lead-in — a DISJOINT
    # class, nothing else matches it) the brain SELECTS a stored concept ITSELF and SPEAKS it, rather than reacting to
    # user content. The SELECTION is the substrate's (a noise-seeded, curiosity-biased CA3 wander — the multibasin
    # self-initiation GO, 66% attributable — on cupy; on the numpy-deferred path the mouth's CURIOSITY-TOP decodable
    # concept, the heavy wander DEFERRED like d5's BTSP write); the utterance CONTENT is the OneBrainComposer mouth's
    # on-bridge RF decode (render_fact), moat-safe by construction (it speaks ONLY a stored concept; an emptied store
    # abstains). Reuse-by-import from `self_initiated_production_organ` (the loop-closing de-risk 6-seed GO). SCOPE
    # (honest): the buildable-now integration is the idle-turn SHORT-CIRCUIT — the TIMING is still HTTP-triggered
    # ("say something"), only the CONTENT is internally selected; a truly proactive NO-HTTP idle-tick is the named
    # deferred rung. This runs FIRST (before the empty-422): a self-initiated remark on an idle turn REPLACES the 422 /
    # normal handling. Default-ON; `BRAIN_SELF_INITIATE=0` -> the block is fully skipped (an empty message still 422s,
    # a "say something" turn falls to the normal path — byte-identical). LESION (`BRAIN_SELF_INITIATE_LESION=1`): the
    # CA3/mouth store NO-ENCODE control (an emptied RF store, not a host flag) -> the utterance stream collapses ->
    # n_utt=0 -> the honest neutral idle line (load-bearing). Guarded so a wiring failure can never crash a turn.
    try:
        import research.runners.self_initiated_production_organ as _SI
        _selfinit_on = _SELF_INITIATE_DEFAULT_ON and _SI.selfinit_enabled()
    except Exception:
        _SI = None
        _selfinit_on = False
    if _selfinit_on and _SI.is_selfinit_trigger(req.message):
        try:
            _si_rname = chat.renderer.name if getattr(chat, "renderer", None) is not None else "raw brain triples"
            siorg = _get_selfinit_organ(cache_key)
            si = siorg.speak(lesion=_SI.selfinit_lesioned())
            spoke = bool(si.get("n_utt", 0) >= 1 and si.get("utterance"))
            answer = _SI.self_initiated_text(si) if spoke else _SI.idle_fallback_text()
            return JSONResponse({
                "answer": answer,
                "abstained": (not spoke), "recalled_svo": None, "verified": bool(spoke),
                "renderer": _si_rname, "brain": req.brain, "source": source, "rich": False,
                "activity": None, "affect": None,
                "self_initiated": dict(si, kind="self_initiated"), "inner_state_readout": True,
            })
        except Exception:
            pass  # never let self-initiation crash a turn -> fall through to the normal empty/normal handling

    if not msg:
        raise HTTPException(422, "message must be non-empty")

    rname = chat.renderer.name if getattr(chat, "renderer", None) is not None else "raw brain triples"

    # ── BEGIN faculty:vision-identity — VISUAL OBJECT -> CATEGORY IDENTITY ("spiking HMAX"), DEFAULT-ON ─────────
    # The production consumer for the EMERGE-36 fully-spiking perception->pooler->inference GO (6 seeds). On a
    # 'what do you see?'-class turn that CARRIES a percept (the environment's retinal render, `req.percept`), the
    # brain SEES the object through the real sim.visual_cortex Gabor/V1 front end -> a spiking Marr-Albus
    # coincidence-column pooler on a real SimulationBridge (coincidence_weighted_drive, NO numpy kWTA) -> reads the
    # winning self-organized category column block as the recognized-object identity, and the recognized concept
    # SEEDS this turn's answer ('I see a <recognized-object>. It can <property>.'). Reuse-by-import (NO sim/ edit).
    # 2026-08-26 FLIPPED DEFAULT-ON (`_VISION_IDENTITY_DEFAULT_ON` above): `BRAIN_VISION_IDENTITY=0` is the explicit
    # escape back to the pre-flip OFF oracle; NO percept on the turn (the overwhelming majority of ordinary chat)
    # means this block is a no-op regardless of the flag. It ALSO short-circuits ONLY when it recognizes: an
    # unresolvable percept OR an ABSTAIN (the POOLER-LESION `BRAIN_VISION_IDENTITY_LESION=1` collapses the codon ->
    # recognize()=-1) returns None -> the turn FALLS THROUGH to the normal path -> byte-identical to flag-off, which
    # is exactly the load-bearing lesion-vanish.
    # LOAD-BEARING: vary the percept (bird<->fish) -> the answer content differs; lesion the pooler -> it vanishes.
    # SCOPE (honest): invariance is on WELL-POSED SYNTHETIC category sets, NOT natural-image translation-invariance
    # (a separate NO-GO). See research/runners/vision_identity_production_organ.py. Guarded so a wiring failure can
    # never crash a turn (degrades to the normal path).
    if _vision_identity_on():
        try:
            import research.runners.vision_identity_production_organ as _VI
            if req.percept and _VI.is_visual_query(msg):
                _vi = _VI.answer_percept(req.percept)     # None on abstain/unresolvable -> fall through (host path)
                if _vi is not None:
                    # seed the discourse grounding: write the recognized concept as the discourse referent so a
                    # follow-up ('what is it?') can reason about the seen object (best-effort, multiturn only). This
                    # runs ONLY on a triggered visual turn, so ordinary turns stay byte-identical.
                    try:
                        _ag = getattr(chat, "agent", None)
                        if _ag is not None and hasattr(_ag, "_write_referent"):
                            _ag._write_referent(_vi["noun"])
                    except Exception:
                        pass
                    return JSONResponse({
                        "answer": _vi["answer"], "abstained": False, "recalled_svo": None,
                        "verified": True, "renderer": rname, "brain": req.brain, "source": source,
                        "rich": False, "activity": None, "affect": None,
                        "vision_identity": {
                            "on": True, "recognized_category": _vi["category"], "noun": _vi["noun"],
                            "prop": _vi["prop"], "shown_category": _vi["shown_category"], "which": _vi["which"],
                            "lesioned": _VI.vision_identity_lesioned(),
                        },
                    })
        except Exception:
            pass  # never let the vision coupling crash a turn -> fall through to the normal path
    # ── END faculty:vision-identity ──

    # ── BEGIN faculty:bg-action-selection — SPEAK-vs-STAY-SILENT via the two-channel spiking BG selector ──────────
    # The production consumer for the Gate-A v2 vocal action-selection GO (research/findings/2026-08-03-neural-vocal-
    # selector-gateA-v2-4seed-GO.md). A discrete chat action decision — SPEAK this turn, or STAY-SILENT (hold) — is
    # routed through a genuine two-channel basal-ganglia RACE (channel 0 = SPEAK, channel 1 = STAY-SILENT) instead of a
    # host `if`. The composer's per-candidate salience biases each channel's striatal D1 pool; SHARED practice arousal
    # drives the proposal->D1 barrage that brings both to threshold; the FIRST channel to cross the GPi->thalamus
    # disinhibition commit burst IS the selected action (NOT a numpy argmax). Reuse-by-import (NO sim/ edit).
    # 2026-08-26 FLIPPED DEFAULT-ON (wave 3 flip, `_BG_SELECT_DEFAULT_ON` above, 6/6 flip-soak GO): `BRAIN_BG_SELECT=0`
    # is the byte-identical escape back to today's turn (the block then imports nothing + returns nothing). It is
    # CONSULTED only on a content-empty turn (STAY-SILENT is a genuine contender there — a normal content
    # message always favors SPEAK, so the selector is not even called on it), and it SHORT-CIRCUITS with a HOLD line
    # ONLY when the BG race COMMITS to STAY-SILENT. A SPEAK commit / a non-commit / an ordinary content turn all return
    # None from decide_action() -> the turn FALLS THROUGH to the normal path -> byte-identical to flag-off.
    # LOAD-BEARING: on '...' the salience favors STAY-SILENT -> the race holds (output differs); give the turn real
    # content -> SPEAK -> the hold vanishes. LESION (`BRAIN_BG_SELECT_LESION=arousal` or `=direct_path`): either lesion
    # collapses the commit -> decide_action() returns None -> the hold VANISHES (byte-identical to flag-off), proving
    # the BG cascade — not a host max — chose. See research/runners/bg_action_selection_production_organ.py. Guarded so
    # a wiring failure can never crash a turn (degrades to the normal path).
    if _bg_select_flag_on():
        try:
            import research.runners.bg_action_selection_production_organ as _BG
            _bg = _BG.decide_action(msg)   # None unless the BG race COMMITS to STAY-SILENT on a content-empty turn
            if _bg is not None:
                return JSONResponse({
                    "answer": _BG.HOLD_TEXT, "abstained": True, "recalled_svo": None,
                    "verified": False, "renderer": rname, "brain": req.brain, "source": source,
                    "rich": False, "activity": None, "affect": None,
                    "bg_select": {
                        "on": True, "action": _bg["action"], "winner": _bg["winner"],
                        "speak_salience": _bg["speak_salience"], "silent_salience": _bg["silent_salience"],
                        "decision_step": _bg["decision_step"], "lesioned": bool(_bg["lesion"]),
                    },
                })
        except Exception:
            pass  # never let the BG coupling crash a turn -> fall through to the normal path
    # ── END faculty:bg-action-selection ──

    # ── AFFECT / EMOTION coloring (Gate-B, 2026-08-12) ──────────────────────────────────────────────────────
    # Read the brain's live MOOD off the co-resident spiking graded-affect ladder and prepare (a) a CONTENT plan
    # (forthcomingness — how many gate-matched facts to volunteer) and (b) a prose MANNER template (warmer/curter
    # phrasing). Default-ON; `BRAIN_AFFECT=0` -> fully skipped (byte-identical oracle). The moat/recall/abstain
    # paths below are UNCHANGED — affect only colors an already-matched answer. `affect_info` is attached to the
    # response for the debug trace (the tone TOKEN lives here ONLY, never on the user-facing surface).
    affect_info = None
    affect_plan = None
    # `_mood_tone_level` is the additive per-turn hook the spiking recall mouth reads (2026-08-27 fix for the
    # MoodConditionedRenderer-bypass regression: BRAIN_SPIKING_MOUTH_RECALL pre-empts `chat.renderer.render_svo`
    # for a bounded-transitive recall, so the manner-coloring below never ran on that surface). Reset to neutral
    # EVERY turn before the block below (which only sets it when affect is genuinely on this turn) so a stale
    # non-neutral value can never leak from a prior turn / a different flag state into this one (`chat` is a
    # per-session cached ChatBrain) -- BRAIN_AFFECT=0 or an exception here leaves it at 0 -> the mouth is a no-op.
    chat._mood_tone_level = 0
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
            # ALSO drive the spiking recall mouth's OWN tone (2026-08-27 fix) with the SAME level, so a
            # bounded-transitive recall that never reaches `wrapped.render_svo` still carries the mood.
            chat._mood_tone_level = int(level)
        except Exception as _e:  # never let affect crash a turn — degrade to the un-colored answer
            affect_info = {"on": True, "error": f"{type(_e).__name__}: {_e}"}
            affect_plan = None
            chat._mood_tone_level = 0

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

    # ── OPEN-ENDED STATE-DRIVEN GENERATION + VERIFY POST-FILTER (default-OFF: BRAIN_OPEN_ENDED, 2026-08-21) ────────
    # The owner reframe (2026-08-19): Qwen = a FORM scaffold, honesty = STATE-FIDELITY. Two de-risks are GO on main:
    # open-ended state-driven generation READS conversational (V1 GO), and a VERIFY POST-FILTER on the free reply
    # restores honesty (fabrication 1.0 -> 0.0, known substance kept). This branch wires them as the LIVE reply path:
    # extract the topic -> retrieve the grounded facts the brain holds (the LTM / chat bundle facts.json `by_agent`,
    # the SAME source the de-risk retrieves from) -> assemble a StateContext from the LIVE affect read (valence off
    # the real affect organ's differential, computed just above) + familiarity/novelty/curiosity grounded in whether
    # the store knows the topic -> free spiking-Qwen reply (FORM) -> `post_filter` (HONESTY) -> return it. ONE Qwen:
    # the generator REUSES the server's already-warm SpikingQwenFaculty (`_get_warm_qwen_renderer()._fac`), never a
    # second model. DEFAULT-OFF: `BRAIN_OPEN_ENDED` unset/0 -> the block is fully skipped -> the strict/rich path
    # below runs byte-identically. When ON this REPLACES the strict/rich reply (the mission's conversational path,
    # behind the no-confab post-filter moat). Reuse-by-import (NO sim/ edit). See webapp/open_ended_chat.py.
    # Cheap env read FIRST so the DEFAULT-OFF path imports NOTHING (the open_ended_chat module pulls in the de-risk
    # modules, one of which disables INFO logging process-wide at import) -> off is truly byte-identical + side-effect-free.
    if os.environ.get("BRAIN_OPEN_ENDED", "0").strip().lower() in ("1", "true", "on", "yes"):
        try:
            from webapp import open_ended_chat as _OE
            # the ONE warm Qwen faculty the server already loaded for the `qwen` renderer (builds it once if the
            # host defaulted to a non-qwen renderer — still exactly one Qwen in the process).
            _warm_faculty = getattr(_get_warm_qwen_renderer(), "_fac", None)
            if _warm_faculty is None:
                raise RuntimeError("no warm Qwen faculty available for open-ended mode")
            # LIVE affect: valence off the real organ's signed differential (computed above), neutral fallback.
            if affect_info is not None and "error" not in affect_info:
                _oe_val = _OE.valence_from_affect(affect_info.get("differential", 0.0))
                _oe_aro = float(affect_info.get("appraisal_arousal", 0.3))
            else:
                _oe_val, _oe_aro = 0.0, 0.3
            _oe = _OE.answer_turn(
                msg, _warm_faculty, _oe_val, _oe_aro,
                ltm_bundle=_resolve_ltm_bundle(),
                brain_bundle=(os.environ.get("BRAIN_CHAT_BUNDLE", "").strip() or None),
                # the live, organ-wired ChatBrain (install_two_organ_gate/install_three_organ_gate already ran on
                # it above) -- consulted ONLY when BRAIN_OPEN_ENDED_GEN_TIME_HONESTY is ALSO truthy (a second,
                # independent gate; see open_ended_chat.gen_time_honesty_enabled). Passing it has NO effect while
                # that flag is off: answer_turn's one-shot path runs byte-identically, this kwarg unused.
                chat=chat,
            )
        except HTTPException:
            raise
        except Exception as e:  # surface the failure like the rich path (never mask a broken mode as strict output)
            raise HTTPException(500, f"open-ended chat turn failed: {type(e).__name__}: {e}")
        _oe_resp = {
            "answer": _oe["answer"],
            # an unknown topic (empty retrieval) is the honest abstain; a known topic is grounded + verified.
            "abstained": (not _oe["known"]),
            "recalled_svo": (_oe["facts"][0] if _oe["facts"] else None),
            "verified": bool(_oe["known"]),
            "renderer": rname, "brain": req.brain, "source": source,
            "rich": False, "mode": "open_ended", "activity": None, "affect": affect_info,
            # the open-ended trace: the raw free reply, the VERIFY-filtered reply, topic/known, retrieved facts,
            # the assembled state, and the generation latency. The moat lives in `filtered` vs `raw`. `generator`
            # (2026-08-28, crutch-burndown) names which FORM generator wrote `raw` -- "qwen" (default), "wkv_mouth"
            # (BRAIN_OPEN_ENDED_WKV_MOUTH, in-vocab prompts only), or "spiking_clause" (2026-09-02, board #112
            # residual -- BRAIN_OPEN_ENDED_FACT_CLAUSE_FALLBACK, a known topic the WKV mouth did not handle,
            # answered by the SAME brain-based fact->sentence render instead of Qwen); byte-identical omission
            # risk avoided via .get() defaults so this key is additive even against an older `_oe` dict shape.
            "open_ended": {
                "raw": _oe["raw"], "filtered": _oe["filtered"], "topic": _oe["topic"],
                "known": _oe["known"], "facts": _oe["facts"], "n_sentences": _oe["n_sentences"],
                "state": _oe["state"], "gen_seconds": _oe["gen_seconds"],
                "generator": _oe.get("generator", "qwen"), "wkv_mouth_used": _oe.get("wkv_mouth_used", False),
                "fact_clause_used": _oe.get("fact_clause_used", False),
            },
        }
        # ── D5 EPISODIC STORE now also WRITES on an open-ended turn (R1 residual, 2026-09-02) ──────────────────
        # A normal answered turn (the rich/single-fact paths below) BTSP-forms the answered topic's CA3 assembly
        # (`_EP.get_episodic_organ(...).note_topic(...)`, see the two call sites gated on `_episodic_on`) so a
        # LATER referential turn ("earlier you told me about X") can recall it. The open-ended EARLY RETURN above
        # skipped this store write entirely -- an open-ended turn was UN-RECALLABLE later (the R1 completeness-
        # audit finding: open-ended bypasses the shared pipeline's session-state writes). This mirrors the SAME
        # gating + topic convention the other two sites use (facts[0][0] -> the agent of the first supporting
        # fact), substituting the open-ended generator's OWN retrieved facts (`_oe["facts"]`) since this turn
        # never reaches the composer's `chat.gate()`. Additive/guarded: writes ONLY the episodic store, never
        # touches `_oe_resp` / the `answer` text -- the surface output stays byte-identical to before this change
        # (this is the FIRST staged rung of R1; the remaining rungs move the OTHER skipped session-state writes
        # + eventually the generation-FORM choice itself inside the shared pipeline -- see the R1 staged plan).
        # No-op (byte-identical) when BRAIN_EPISODIC=0/unavailable, the topic was unknown (nothing to note), or
        # the numpy backend (`_episodic_store_ok()` defers the write there exactly like the other two call sites).
        try:
            import research.runners.d5_episodic_production_organ as _OE_EP
            if _OE_EP.episodic_enabled() and _oe.get("known") and _oe.get("facts") and _episodic_store_ok():
                _oe_ep_topics = getattr(chat, "agents_set", None) or _brain_vocab(chat)
                _OE_EP.get_episodic_organ(cache_key, 42, _oe_ep_topics).note_topic(_oe["facts"][0][0])
        except Exception:
            pass  # never let the episodic write crash an open-ended turn -- degrade to the un-noted turn
        # ── R1 RUNG-2 (2026-09-02): the open-ended turn now ALSO runs the shared pipeline's per-turn SESSION-STATE
        #    WRITERS, closing the rest of the "open-ended bypasses the shared pipeline's session-state writes" debt
        #    (board #199). A NORMAL turn runs affect-drives (#84 felt body-state EMA), affective-ToM, DA-mode, and
        #    common-ground (the per-referent audience-design ledger) as OBSERVE-THEN-DRIVE faculties in the pipeline
        #    BELOW this block (server.py ~4687/4713/4737/4761), plus the discourse register's per-turn fold (~5217).
        #    The open-ended EARLY RETURN skipped ALL of them, so an open-ended turn left the felt body-state, the
        #    common-ground ledger, and the discourse register FROZEN (a later turn saw no trace of it -- e.g. a
        #    re-mention never reduced, the felt mood never moved). This runs the SAME faculty functions, with the
        #    SAME flag-gates + cache_key, for their STATE-WRITE side effect ONLY -- their returned tone/reference
        #    LEADS are intentionally DISCARDED here, so the open-ended free-talk surface + `_oe_resp` stay
        #    byte-identical (rung-2 moves the STATE writes, not the generation FORM -- that is a later rung). Each
        #    call is independently flag-guarded + try/excepted (the standing "never let a faculty crash a turn"
        #    convention), so with every faculty off this is byte-identical, and additive otherwise. NOTE the Gate-B
        #    `_SESSION_MOOD` EMA (`_update_session_mood`, ~4539) + the D5 episodic write (rung-1, just above) already
        #    run for an open-ended turn -- this adds the remaining named per-turn writers. DISCLOSED RESIDUAL (rungs
        #    3-5): the DEEPER query-branch state folds (worldview/multiref/prospective-memory/silent-WM ~4880-5173)
        #    and routing an open-ended turn's SPECIALIST query classes to their branches remain skipped, and these
        #    are duplicated CALL SITES, not yet a single shared pipeline both the normal + open-ended paths invoke.
        try:
            if _affect_drives_on():                        # #84 felt body-state EMA -> chat._affect_drives_workspace
                from webapp import affect_drives_chat as _OE_ADC
                _OE_ADC.observe_turn(chat, msg)
        except Exception:
            pass  # never let the affect-drives state write crash an open-ended turn
        try:
            if _affective_tom_on():                        # W5 other-agent affect region (default-OFF anchor)
                from research.runners import affective_tom_production_organ as _OE_ATM
                _OE_ATM.observe_turn(chat, msg)
        except Exception:
            pass  # never let the ToM state write crash an open-ended turn
        try:
            if _da_drives_on():                            # #79 self-produced DA-mode read -> chat._last_da_drives
                from webapp import da_mode_drives_chat as _OE_DAD
                _OE_DAD.observe_turn(chat, msg)
        except Exception:
            pass  # never let the DA-mode state write crash an open-ended turn
        try:
            if _common_ground_drives_on():                 # per-referent common-ground ledger (cache_key-keyed)
                from webapp import common_ground_drives_chat as _OE_CGD
                _OE_CGD.observe_turn(chat, msg, cache_key=cache_key)
        except Exception:
            pass  # never let the common-ground state write crash an open-ended turn
        try:                                               # D3 discourse register per-turn fold (part i ONLY -- the
            import research.runners.d3_discourse_event_register_production_organ as _OE_DR  # who-was-before query
            if _OE_DR.discourse_register_enabled() and getattr(getattr(chat, "agent", None), "_event_register", None) is not None:  # short-circuit (part ii) stays on the normal path
                _oe_dstate = _SESSION_DISCOURSE.setdefault(cache_key, _OE_DR.new_state())
                _OE_DR.note_turn(msg, chat.agent, _oe_dstate, actions=getattr(chat, "actions_set", None))
        except Exception:
            pass  # never let the discourse fold crash an open-ended turn
        return _safe_json_response(_oe_resp, "open_ended")

    # ── AFFECT DRIVES THE RESPONSE (board #84, 2026-08-19) ────────────────────────────────────────────────────
    # Read the brain's felt valence x arousal off the #81 graded-affect LADDER (the interoceptive graded read,
    # NEURAL off cp_firing_states — reuse-by-import, NO sim/ edit) and turn it into (a) an AFFECTIVE EXPRESSION the
    # reply leads with (a graded warmth/curtness marker, `affect_drives_lead`, prepended OUTERMOST to the answer
    # surface) and (b) `affect_drives_info` (the additive trace: mood, felt-arousal, graded level, the lead). The
    # moat/recall/abstain verdict runs FIRST and unchanged — this DECORATES an already-matched surface, never a fact;
    # the content fields (abstained/recalled_svo/verified) are byte-identical with it on or off. The neural read runs
    # on a PRIVATE RNG timeline (host RNG restored — the #77 footgun), so the other response fields stay byte-
    # identical. Default-ON (anchor); `BRAIN_AFFECT_DRIVES=0` -> the block is fully skipped (no key, no lead ->
    # byte-identical oracle). LESION (`BRAIN_AFFECT_DRIVES_LESION=1`): cut the interoceptive->ladder synapses -> the
    # neural mood collapses -> the lead VANISHES (the load-bearing proof). See webapp/affect_drives_chat.py.
    affect_drives_info = None
    affect_drives_lead = ""
    if _affect_drives_on():
        try:
            from webapp import affect_drives_chat as _ADC
            affect_drives_info = _ADC.observe_turn(chat, msg)
            affect_drives_lead = str(affect_drives_info.get("lead", "") or "")
        except Exception as _ade:  # never let the affect coupling crash a turn — degrade to the un-led answer
            affect_drives_info = {"on": True, "error": f"{type(_ade).__name__}: {_ade}", "lead": ""}
            affect_drives_lead = ""

    # >>> W5 AFFECTIVE ToM BEGIN (empathy — infer ANOTHER agent's emotion; additive, mergeable block) ───────────
    # ── AFFECTIVE THEORY OF MIND DRIVES THE RESPONSE (W5, 2026-08-26) ──────────────────────────────────────────
    # On a turn about ANOTHER agent's affectively-charged situation ("Maria is devastated", "Sam's team lost", "my
    # friend won"), infer the OTHER agent's EMOTION (valence) with an OTHER-tagged P0.3 affect region (dissociable
    # from the system's OWN #84 affect) and lead the reply with an EMPATHIC expression whose tone is read NEURALLY
    # off the OTHER model's `affect_out`-gated recall differential (reuse-by-import, NO sim/ edit). `tom_lead` is
    # prepended to the answer surface; `tom_info` is the additive trace (agent, appraised valence, neural tone, the
    # lead). The moat/recall/abstain verdict runs FIRST and unchanged — this DECORATES an already-matched surface,
    # never a fact; the content fields (abstained/recalled_svo/verified) are byte-identical with it on or off. The
    # OTHER-model build + read run on a PRIVATE RNG timeline (host RNG restored — the #77 footgun), so the other
    # response fields stay byte-identical even on a triggered turn. An ORDINARY turn (no other-agent, or no affective
    # word) never builds/reads the bridge -> byte-identical + no RNG perturbation (the soak-gate property). Default-OFF
    # (anchor; the parent flips it default-ON after the pool soak); `BRAIN_AFFECTIVE_TOM=1` opts in -> the block runs.
    # LESION (`BRAIN_AFFECTIVE_TOM_LESION=1`): cut the OTHER region's `affect_out` -> the neural tone collapses to
    # neutral -> the empathic lead VANISHES (the load-bearing proof). See affective_tom_production_organ.py.
    tom_info = None
    tom_lead = ""
    if _affective_tom_on():
        try:
            from research.runners import affective_tom_production_organ as _ATM
            tom_info = _ATM.observe_turn(chat, msg)
            tom_lead = str(tom_info.get("lead", "") or "")
        except Exception as _tme:  # never let the empathy coupling crash a turn — degrade to the un-led answer
            tom_info = {"on": True, "error": f"{type(_tme).__name__}: {_tme}", "lead": ""}
            tom_lead = ""
    # <<< W5 AFFECTIVE ToM END ─────────────────────────────────────────────────────────────────────────────────

    # ── DA-MODE DRIVES THE RESPONSE (board #79, 2026-08-19) ───────────────────────────────────────────────────
    # Read the brain's OWN dopamine MODE off the #76 spiking DA nucleus (the snc population self-produces the DA
    # LEVEL from this turn's reward/context engagement — NEURAL off the neuromodulator bus, reuse-by-import, NO
    # sim/ edit) and turn it into a graded ENGAGEMENT SUFFIX (`da_drives_suffix`, APPENDED to the answer surface —
    # a SUFFIX + the engagement/arousal axis, distinct from the #84 valence LEAD + the #85 topic LEAD) + a
    # `da_drives_info` trace (engagement, self-produced DA level, mode, the suffix). The moat/recall/abstain
    # verdict runs FIRST and unchanged — this DECORATES an already-matched surface, never a fact; the content
    # fields (abstained/recalled_svo/verified) are byte-identical with it on or off. The neural read runs on a
    # PRIVATE RNG timeline (host RNG restored — the #77 footgun), so the other response fields stay byte-identical.
    # Default-ON (anchor); `BRAIN_DA_DRIVES=0` -> the block is fully skipped (no key, no suffix -> byte-identical
    # oracle). LESION (`BRAIN_DA_DRIVES_LESION=1`): silence the SNc nucleus -> the DA level collapses -> the mode
    # is REST -> the suffix VANISHES (the load-bearing proof). See webapp/da_mode_drives_chat.py.
    da_drives_info = None
    da_drives_suffix = ""
    if _da_drives_on():
        try:
            from webapp import da_mode_drives_chat as _DAD
            da_drives_info = _DAD.observe_turn(chat, msg)
            da_drives_suffix = str(da_drives_info.get("lead", "") or "")
        except Exception as _dde:  # never let the DA coupling crash a turn — degrade to the un-suffixed answer
            da_drives_info = {"on": True, "error": f"{type(_dde).__name__}: {_dde}", "lead": ""}
            da_drives_suffix = ""

    # ── COMMON GROUND DRIVES THE RESPONSE (2026-08-26) ──────────────────────────────────────────────────────────
    # Read this turn's referent's common-ground state off the persistent spiking ledger (a per-referent bistable NMDA
    # store latched by earlier grounding acts + held by recurrence; the 6-seed-GO de-risk, reuse-by-import, NO sim/
    # edit) and turn the audience-design verdict into a REDUCED-REFERENCE LEAD (`cg_drives_lead`, prepended to the
    # answer surface — a grounded referent gets pronominalized, a first mention named in full) + a `cg_drives_info`
    # trace (topic, slot, decision, in_common_ground, the substrate read rates, the lead). The moat/recall/abstain
    # verdict runs FIRST and unchanged — this DECORATES an already-matched surface, never a fact; the content fields
    # (abstained/recalled_svo/verified) are byte-identical with it on or off. The ledger runs its substrate steps
    # inside a numpy global-RNG save/restore (in the organ), so the other response fields stay byte-identical.
    # Default-OFF (anchor); `BRAIN_CG_DRIVES` unset -> the block is fully skipped (no key, no lead -> byte-identical
    # oracle). LESION (`BRAIN_CG_DRIVES_LESION=1`): build the ledger recurrence at 0 -> it cannot HOLD -> a re-mentioned
    # referent reads ungrounded -> the decision goes static (always INTRODUCE) -> the reduced lead VANISHES (the
    # load-bearing proof). See webapp/common_ground_drives_chat.py.
    cg_drives_info = None
    cg_drives_lead = ""
    if _common_ground_drives_on():
        try:
            from webapp import common_ground_drives_chat as _CGD
            cg_drives_info = _CGD.observe_turn(chat, msg, cache_key=cache_key)
            cg_drives_lead = str(cg_drives_info.get("lead", "") or "")
        except Exception as _cge:  # never let the common-ground coupling crash a turn — degrade to the un-led answer
            cg_drives_info = {"on": True, "error": f"{type(_cge).__name__}: {_cge}", "lead": ""}
            cg_drives_lead = ""

    # ── DA-GATED ENCODING (board WAVE-0, Gap-4 write-side coupling) ──────────────────────────────────────────
    # The brain's OWN self-produced tonic DA (the DA-mode read just above — chat._last_da_drives["da_level"]) scales
    # the WRITE MAGNITUDE of a taught fact AT STORE TIME (Lisman-Grace hippocampal-VTA loop; Kandel D.16 — dopamine
    # gates entry into LONG-TERM memory: a salient / engaged utterance is encoded STRONGER + more stable). This
    # installs composer.encoding_gain_fn (read inside chat.gate's _maybe_acquire, which runs LATER this turn), so it
    # MUST be set here — after the DA-mode read (level fresh), before the gate. Default-OFF (`BRAIN_DA_ENCODING`
    # unset): the block is skipped, encoding_gain_fn stays None -> the store is BYTE-IDENTICAL to pre-wiring (g=1.0
    # unit-magnitude write) and no `da_encoding` key is attached. g == 1.0 at tonic (an unengaged turn is neutral)
    # and on a missing DA read. LESION (`BRAIN_DA_ENCODING_LESION=1`): pin g=1.0 regardless of DA -> the write no
    # longer rides the DA read (the coupling is severed — the load-bearing proof; distinct from BRAIN_DA_DRIVES_LESION
    # which collapses the LEVEL). The gain bites the STORED trace on a magnitude-carrying composer (the production-
    # default onebrain store_conns / rf substrate store); the rf numpy fast-path recall is magnitude-invariant. NO
    # sim/ edit (a composer-layer callable). See webapp/da_encoding_drives_chat.py.
    da_encoding_info = None
    try:
        from webapp import da_encoding_drives_chat as _DAE
        if _DAE.da_encoding_enabled():
            _g_now = _DAE.install_encoding_gain(chat)
            da_encoding_info = {"on": True, "g": _g_now, "da_level": _DAE.da_level_of(chat),
                                "lesioned": _DAE.da_encoding_lesioned()}
    except Exception as _dee:  # never let the encoding coupling crash a turn — degrade to the unit-magnitude write
        da_encoding_info = {"on": True, "error": f"{type(_dee).__name__}: {_dee}"}

    # ── THE WANDERED THOUGHT DRIVES THE RESPONSE (board #86, 2026-08-20) ─────────────────────────────────────
    # The continuous engine's idle tick (webapp/continuous_engine.py) already lets a THOUGHT wander between turns
    # (a curiosity-biased spiking selection off the self-initiation organ) — today that concept is only OBSERVED
    # in the monologue. This makes it LOAD-BEARING: if this session has a recent idle-wandered concept, the NEXT
    # real turn brings it up — a short lead PREPENDED to the reply, mirroring the #84 affect-lead / #85 swap-lead
    # pattern ("(I'd been mulling over the cat.) <answer>"). Additive; the moat/recall/abstain verdict runs FIRST
    # and unchanged — this DECORATES an already-matched surface, never a fact; the content fields
    # (abstained/recalled_svo/verified) are byte-identical with it on or off. `recent_wander()` CONSUMES the
    # record on read, so the concept surfaces once (the next turn after the tick that produced it), not on every
    # turn after. Default-ON (flipped 2026-08-21 with BRAIN_CONTINUOUS): `BRAIN_CONTINUOUS_DRIVES=0` -> no key, no
    # lead -> byte-identical escape; unset -> the default (ON). This block is byte-identical on any ORDINARY turn (no
    # pending wander -> `recent_wander()` returns None -> no key, no lead), and is additionally inert whenever the
    # continuous engine itself is off (`BRAIN_CONTINUOUS=0`) — no wander is ever recorded regardless of this flag.
    wander_drives_lead = ""
    wander_drives_info = None
    if os.environ.get("BRAIN_CONTINUOUS_DRIVES", _CONTINUOUS_DRIVES_DEFAULT).strip().lower() in ("1", "true", "on", "yes"):
        try:
            from webapp import continuous_engine as _CEW
            _wander = _CEW.recent_wander(cache_key)
            if _wander:
                wander_drives_lead = "(I'd been mulling over %s.) " % _wander
                wander_drives_info = {"on": True, "concept": _wander, "lead": wander_drives_lead}
        except Exception as _cwe:  # never let the wander coupling crash a turn — degrade to the un-led answer
            wander_drives_info = {"on": True, "error": f"{type(_cwe).__name__}: {_cwe}", "lead": ""}
            wander_drives_lead = ""

    # ── A NOVEL IDEA DRIVES THE RESPONSE (continuous IDEATION, default-OFF `BRAIN_CONTINUOUS_IDEATE`) ─────────────
    # The between-turn wander OCCASIONALLY GENERATES a NOVEL blended concept (the creativity rung): a sparse attractor
    # driven by a blend of the two curiosity-top basins settles into a recombination that was NEVER stored. If this
    # session has a recent idle IDEATION, the next real turn brings it up — but FLAGGED as a novel idea/association,
    # NEVER a recalled fact (a DISTINCT channel from the wander-recall lead above). The moat/recall/abstain verdict
    # runs FIRST and unchanged — this DECORATES an already-matched surface, never asserts a stored fact; the content
    # fields are byte-identical with it on or off. `recent_ideation()` CONSUMES the record on read (surfaces once).
    # DEFAULT-OFF: `ideation_enabled()` reads BRAIN_CONTINUOUS_IDEATE default '0' -> the whole block is skipped, no
    # `ideation_drives` key, no lead -> byte-identical to the live default-on continuous wander (the flip is untouched).
    ideation_drives_lead = ""
    ideation_drives_info = None
    try:
        from webapp import continuous_engine as _CEI
        if _CEI.ideation_enabled():
            _idea = _CEI.recent_ideation(cache_key)
            if _idea and _idea.get("sources") and len(_idea["sources"]) >= 2:
                _sA, _sB = _idea["sources"][0], _idea["sources"][1]
                ideation_drives_lead = ("(A thought occurred to me while we were apart — ‘%s’ and ‘%s’ feel "
                                        "connected; it's just an idea of mine, not something I was told.) " % (_sA, _sB))
                ideation_drives_info = {"on": True, "sources": [_sA, _sB], "kind": _idea.get("kind", "novel-association"),
                                        "flagged_as_idea": True, "is_fact": False,
                                        "novelty_max_overlap": _idea.get("novelty_max_overlap"),
                                        "blend_balance": _idea.get("blend_balance"), "lead": ideation_drives_lead}
    except Exception as _cie:  # never let the ideation coupling crash a turn — degrade to the un-led answer
        ideation_drives_info = {"on": True, "error": f"{type(_cie).__name__}: {_cie}", "lead": ""}
        ideation_drives_lead = ""

    # ── PROSPECTIVE MEMORY (Gate-B, 2026-08-13) ──────────────────────────────────────────────────────────────
    # A co-resident spiking intention-LATCH + BA10 cue-MONITOR holds a deferred intention ("remind me to X when Y")
    # across intervening turns and RELEASES it only when the cue appears — reuse-by-import from
    # `prospective_memory_production_organ` (the de-risked GO `SFANmdaProspectiveMemory`: a persistent-attractor PFC
    # intention LATCH + an NMDA/dendritic-plateau coincidence CUE-MONITOR; fire_on_cue 6/6, every silence clause 6/6).
    # Two behaviours: (i) a FORMATION turn ("remind me to X when Y" / "when Y, do X") is a DISJOINT class — it LATCHES
    # the intention (a spiking self-sustaining cortex<->dlpfc attractor) and short-circuits with an acknowledgement;
    # (ii) on later turns, when an intention is HELD, the cue-monitor is READ — a cue turn drives the cue and reads the
    # SPIKING held x cue coincidence off cp_firing_states (rel >= the frozen FIRE_THR): on a fire the reminder is
    # PREPENDED to the normal turn (which still answers what the user actually said); a non-cue turn ADVANCES the hold
    # (persistence). The fire is gated by the HELD intention (the coincidence), not the host cue-match — proven by the
    # lesion. Placed right after AFFECT so a "remind me..." formation is not mis-read as a recall/assertion by the
    # episodic/comprehension/surprise gates below. Default-ON; `BRAIN_PMEM=0` -> the whole block is skipped and no
    # `prospective` key is added (byte-identical oracle). `BRAIN_PMEM_LESION=1` -> the latch is zeroed after formation
    # (the held assembly collapses -> the SAME cue does NOT fire -> NO reminder; load-bearing). The cue->action CONTENT
    # binding is now LEARNED via a ONE-SHOT HEBBIAN potentiation at formation (Gollwitzer implementation-intention;
    # `BRAIN_PMEM_HEBBIAN=1` default; `=0` reverts to the build-time install; `BRAIN_PMEM_HEBBIAN_LESION=1` latches
    # without the event -> binding absent -> no fire). HOST-SCAFFOLD (flagged, narrowed): the intention/cue TEXT->slot
    # mapping + cue-presence are host-derived (a language/sensory boundary, like curiosity's wh-frame + novelty); the
    # HOLD + coincidence-gated RELEASE + the cue->action BINDING are spiking (wired/on-by-default; the build-time
    # binding install is RETIRED — the text/sensory boundary + operating-point calibration remain).
    pmem_prefix = ""
    prospective_info = None
    try:
        import research.runners.prospective_memory_production_organ as _PM
        _pmem_on = _PM.pmem_enabled()
    except Exception:
        _PM = None
        _pmem_on = False
    if _pmem_on:
        try:
            _pm_formation = _PM.parse_intention(msg)
            if _pm_formation is not None:
                # (i) FORMATION: latch the deferred intention (a disjoint acknowledgement turn class).
                porg = _get_pmem_organ(cache_key)
                finfo = porg.form_intention(_pm_formation["action"], _pm_formation["cue_clause"],
                                            _pm_formation["cue_keywords"], lesion=_PM.pmem_lesioned(),
                                            hebbian_lesion=_PM.pmem_hebbian_lesioned())
                return JSONResponse({
                    "answer": _PM.acknowledgement_text(_pm_formation["action"], _pm_formation["cue_clause"]),
                    "abstained": False, "recalled_svo": None, "verified": True,
                    "renderer": rname, "brain": req.brain, "source": source, "rich": False,
                    "activity": None, "affect": affect_info,
                    "prospective": dict(finfo, kind="formation"), "inner_state_readout": True,
                })
            else:
                # (ii) an intention already held? -> read the cue-monitor this turn (a fire -> a reminder PREFIX;
                # a non-cue turn advances the hold). Only reads if an intention was formed earlier this session.
                porg = _SESSION_PMEM.get(cache_key)
                if porg is not None and porg.held:
                    rd = porg.read_turn(msg)
                    prospective_info = dict(rd, kind="monitor")
                    if rd["fired"]:
                        pmem_prefix = _PM.reminder_text(rd["action"], rd["cue_clause"])
                        # NOTE: the intention is CONSUMED (porg.clear) only when the reminder is actually DELIVERED
                        # on a main answer path (below) — so if THIS turn hits a disjoint short-circuit (episodic /
                        # comprehension-repair / causal / ...) that drops the prefix, the intention STAYS held and
                        # fires again on the next main-path cue mention (the reminder is never silently lost).
        except Exception as _pme:  # never let the prospective read crash a turn — degrade to the normal turn
            prospective_info = {"on": True, "error": f"{type(_pme).__name__}: {_pme}"}

    # ── EPISODIC RECALL of PAST TURNS — Hook A: REFERENTIAL RECALL (Gate-B, D5, 2026-08-12) ──────────────────
    # On a referential turn ("earlier you told me about X", "you mentioned a cat"), decide whether topic X was
    # actually discussed THIS conversation by a genuinely-SPIKING hippocampal pattern-completion — NOT a host list
    # scan. `recall` drives the referential cue and reads the two-compartment apical dAP UP-state completion after
    # real bridge steps (reuse-by-import from `d5_episodic_production_organ` -> the kt=8 EpisodicDapMemory, 6/6-GO).
    # A completed assembly -> honest disclosure (with the host-oracle fact CONTENT the moat already governs, surfaced
    # ONLY on a completion); a non-completing cue -> honest "I don't recall discussing X" (a genuine spiking completion
    # failure, NEVER a confabulation). This owns a DISJOINT turn class (no other organ fires on 'you mentioned X') and
    # runs FIRST (referential-first, right after AFFECT) so it is NOT pre-empted by the comprehension/surprise/B3 gates
    # (which would otherwise mis-read 'you mentioned the dog' as an incomprehensible assertion). Conversation-scoped
    # (memory ACCUMULATES). LESION (`BRAIN_EPISODIC_LESION=1`): read through the UNFORMED baseline recurrent weights ->
    # every completion collapses to 0 -> the gate falls to 'not in memory' (load-bearing). Default-ON;
    # `BRAIN_EPISODIC=0` -> the referential turn falls through to the normal path (byte-identical oracle).
    episodic_info = None
    try:
        import research.runners.d5_episodic_production_organ as _EP
        _episodic_on = _EP.episodic_enabled()
    except Exception:
        _EP = None
        _episodic_on = False
    if _episodic_on and _EP.is_referential(msg):
        try:
            _ep_topics = getattr(chat, "agents_set", None) or _brain_vocab(chat)
            ref = _EP.extract_referent(msg, _ep_topics)
            if ref is not None:
                eorg = _EP.get_episodic_organ(cache_key, 42, _ep_topics)
                rec = eorg.recall(ref, lesion=_EP.episodic_lesioned())
                episodic_info = dict(rec)
                # D5 LEARN-THROUGH-USE (continuous engine, DEFAULT-ON since 2026-08-21, BRAIN_D5_CONSOLIDATE=0 is the
                # byte-identical escape): mark the topic this turn RECALLED (a genuine spiking completion) so the NEXT
                # idle tick consolidates it — the used memory becomes more robust AND recalls VISIBLY STRONGER next
                # turn. The strength is surfaced per-consolidated-topic, so a neighbour's reply is never perturbed.
                try:
                    from webapp import continuous_engine as _CEc
                    if _CEc.d5_consolidate_enabled() and rec.get("in_memory"):
                        _CEc.mark_recall(cache_key, ref)
                    # MULL (continuous engine, board #145, 2026-08-26, default-OFF `BRAIN_CONTINUOUS_WANDER_MULL`):
                    # mark the topic this turn RECALLED (the SAME genuine spiking completion gate as D5 above) so the
                    # NEXT idle wander is biased toward it IF it's one of the self-init organ's own stored concepts —
                    # what the brain unprompted brings up next genuinely tracks what it just discussed, instead of a
                    # fixed per-session draw. Independent flag/state from D5 (own enablement, own dict) so lesioning
                    # one never touches the other.
                    if _CEc.wander_mull_enabled() and rec.get("in_memory"):
                        _CEc.mark_mull(cache_key, ref)
                except Exception:
                    pass
                # CONTENT (a DECLARED host-oracle residual): surface a fact the brain holds about `ref`, rendered
                # through the SAME governed render path the moat covers — ONLY when the spiking assembly COMPLETED.
                content = None
                if rec.get("in_memory"):
                    try:
                        _comp = getattr(getattr(chat, "inner", None), "composer", None)
                        for _f, _h in (getattr(_comp, "kb", []) or []):
                            if (str(_f.get("agent", "")).lower() == ref
                                    and _f.get("action") and _f.get("patient")):
                                content = chat.render([_f.get("agent"), _f.get("action"), _f.get("patient")])
                                break
                    except Exception:
                        content = None
                return JSONResponse({
                    "answer": _EP.recall_disclosure(rec, content, cache_key=cache_key),
                    "abstained": (not bool(rec.get("in_memory"))),
                    "recalled_svo": None, "verified": bool(rec.get("in_memory")),
                    "renderer": rname, "brain": req.brain, "source": source,
                    "rich": False, "activity": None, "affect": affect_info,
                    "episodic": dict(episodic_info, kind="recall"), "referential": True,
                })
        except Exception as _epe:  # never let the episodic read crash a turn — fall through to the normal path
            episodic_info = {"on": True, "error": f"{type(_epe).__name__}: {_epe}"}

    # ── INTERNAL WORLDVIEW / AFFECTIVE WORLD-MODEL (Gate-B, E2, 2026-08-12) ──────────────────────────────────
    # The brain maintains a spiking affective FORWARD MODEL: from the current affective context it PREDICTS the
    # next-turn valence (QUERYABLE — "what do you expect / how is this going?"), and fires a genuinely-SPIKING
    # SURPRISE when the actual next turn VIOLATES that prediction (an affective prediction-error). Reuse-by-import
    # from `worldmodel_production_organ` (the E2 de-risk, 6/6 GO). The prediction read + the mismatch read are the
    # load-bearing SPIKING parts; the valence APPRAISAL + the persistence state-selection are declared host
    # boundaries (RESIDUAL, the named next rung: generic pos/neg pools, NOT bound to the ACTUAL interlocutor affect
    # — the P0.3 valence latch + the W5 ToM channel). Additive + moat-safe: E2 only READS (a queryable expectation)
    # or NOTICES (an honest surprise notice); it never manufactures a fact, flips an abstain, or changes WHICH
    # answer the recall produced. Default-ON; `BRAIN_WORLDMODEL=0` -> fully skipped (byte-identical oracle).
    worldmodel_info = None
    worldmodel_prefix = ""
    try:
        import research.runners.worldmodel_production_organ as _WM
        from research.runners.affect_production_organ import appraise_text as _wm_appraise
        _wm_on = _WM.worldmodel_enabled()
    except Exception:
        _WM = None
        _wm_on = False
    if _wm_on:
        try:
            wm_state = _SESSION_WORLDVIEW.setdefault(cache_key, {"context_sign": 1, "expected_sign": None})
            wm_lesion = _WM.worldmodel_lesioned()
            # QUERYABLE expectation: an explicit "what do you expect / how is this going" -> read the two-pool
            # spiking prediction for the current affective context and answer with an honest functional read-out.
            if _WM.is_expectation_query(msg):
                worg = _get_worldmodel_organ()
                exp = worg.expectation(int(wm_state.get("context_sign", 1)), lesion=wm_lesion)
                return JSONResponse({
                    "answer": _WM.expectation_readout(exp),
                    "abstained": False, "recalled_svo": None, "verified": True,
                    "renderer": rname, "brain": req.brain, "source": source,
                    "rich": False, "activity": None, "affect": affect_info,
                    "worldmodel": dict(exp, kind="query"), "inner_state_readout": True,
                })
            # OTHERWISE: appraise THIS turn's affect; if it VIOLATES the prediction the model held from the prior
            # turn, fire the spiking surprise + an honest notice; then update the held expectation (persistence prior:
            # a positive context expects a positive next turn, so a sign FLIP is the affect-trajectory violation).
            obs_val = float(_wm_appraise(msg).get("valence", 0.0))
            obs_sign = 1 if obs_val > 0.02 else (-1 if obs_val < -0.02 else 0)
            if obs_sign != 0:
                worg = _get_worldmodel_organ()
                held = wm_state.get("expected_sign")
                if held is not None and obs_sign != held:
                    sj = worg.read_surprise(int(wm_state.get("context_sign", 1)), obs_sign, lesion=wm_lesion)
                    worldmodel_info = dict(sj)
                    worldmodel_info["kind"] = "violation"
                    if sj["surprised"]:
                        worldmodel_prefix = _WM.worldmodel_surprise_notice(int(held))
                # update the held expectation for the NEXT turn from THIS turn's observed context (persistence).
                exp = worg.expectation(obs_sign, lesion=wm_lesion)
                wm_state["context_sign"] = int(obs_sign)
                wm_state["expected_sign"] = int(exp["pred_sign"])
                if worldmodel_info is None:
                    worldmodel_info = {"on": True, "lesioned": bool(wm_lesion), "kind": "update",
                                       "context_sign": int(obs_sign), "predicted_next_sign": int(exp["pred_sign"]),
                                       "pred_margin": float(exp["pred_margin"])}
        except Exception as _we:  # never let the world-model read crash a turn — degrade to the normal answer
            worldmodel_info = {"on": True, "error": f"{type(_we).__name__}: {_we}"}

    # ── MULTI-REFERENT WORKING MEMORY (Gate-B, D6, 2026-08-12) ───────────────────────────────────────────────
    # Hold >=2 discourse referents ACROSS a turn/span on a genuinely-SPIKING multi-register buffer (R disjoint
    # slow-NMDA bistable banks on ONE bridge, ONE shared FS pool; reuse-by-import from
    # `d6_multiref_wm_production_organ`, the 6-seed-GO MultiSlotHold + RUNG6c HebbianBinder). Two paths, mirroring
    # is_feel_query / is_expectation_query: (a) READ-OUT — an explicit "who/what are we talking about / keeping in
    # mind" query READS BACK every held referent off the buffer (what a single-attractor store can't do — it ties
    # to one) and short-circuits with an honest functional read-out; (b) MAINTAIN — an input introducing >=2 named
    # referents LOADS each into its own register and HOLDS (write-only; the turn falls through unchanged). The buffer
    # is PER-SESSION (the organ singleton's referent codebook is process-global, so a shared buffer would leak other
    # conversations' referents into a hold-query). Out-of-scope (<2 referents, no hold-query) -> None -> byte-identical.
    # LESION (BRAIN_MULTIREF_LESION=1): recur=0 kills the slow-NMDA hold -> the >=2 read-back collapses (load-bearing).
    # Additive + moat-safe: it only reads/reports ITS OWN buffer (no invented referent, no fact, no abstain flip).
    # Default-ON; `BRAIN_MULTIREF=0` -> fully skipped (byte-identical oracle).
    multiref_info = None
    # `d6org` (THIS session's own per-cache_key MultiReferentWMOrgan) is hoisted to function scope so the
    # COMPREHENSION block below can read `d6org.current_focus()` -- this session's OWN xedge focus, never another
    # session's (2026-08-27 cross-session xedge_focus leak fix, research/FAILURE_LOG.md). None when multiref is
    # disabled or the lookup failed -> the comprehension block treats that as "no focus held" (byte-identical).
    d6org = None
    try:
        import research.runners.d6_multiref_wm_production_organ as _D6
        _multiref_on = _D6.multiref_enabled()
    except Exception:
        _D6 = None
        _multiref_on = False
    if _multiref_on:
        try:
            d6org = _get_multiref_organ(cache_key)
            d6les = _D6.multiref_lesioned()
            if _D6.is_hold_query(msg):                       # READ-OUT: 'who/what are we talking about / keeping in mind'
                # ── BEGIN faculty: onebrain curiosity->d6 LEARNED CROSS-EDGE (2026-09-01), additive,
                #    default per BRAIN_ONEBRAIN_XEDGE_CURIOSITY_D6 (`_XEDGE_CD6_DEFAULT_ON`). Mirrors the
                #    PART-1/R4 frozen-cross-edge wire-in pattern, but — unlike those two (additive
                #    DIAGNOSTIC field only) — this one drives the ACTUAL reply text (2026-08-19 "faculties
                #    must drive, not observe": a neural verdict stashed as metadata + a flip is a hollow
                #    checkbox). THIS SESSION's own RECENT curiosity-crave state (`d6org.
                #    _xedge_curiosity_recent_crave`, set+CONSUMED via `_curiosity_followup` below on this
                #    session's own last abstain -- 2026-08-27 session-isolation pattern: an instance
                #    attribute on THIS session's own per-session MultiReferentWMOrgan, never on the shared
                #    process pool, so a fresh session's d6org never inherits another session's crave) drives
                #    the frozen `ask->w0` cross-edge's OWN validated instrument
                #    (`crossedge_w0_shift`, reused verbatim from the 6-seed runner-level GO,
                #    research/findings/2026-09-01-onebrain-crossedge-curiosity-to-d6wm-GO.md). When the
                #    measured shift clears the runner's own registered floor (a genuine, lesion-attributable
                #    suppression), an honest, SELF-CONSUMING functional qualifier is appended to the
                #    hold-query readout -- never changing WHICH referents are reported held, never
                #    fabricating a fact. LOAD-BEARING AT THE REPLY-TEXT LEVEL: the qualifier's presence
                #    depends on BOTH the live crave state AND the cross-edge weight; it never appears when
                #    `BRAIN_ONEBRAIN_XEDGE_CURIOSITY_D6_LESION=1` zeroes that edge, regardless of crave.
                #    DEFAULT-OFF unless flipped: unset -> no extra key, no text change, no attribute ever
                #    written (see `_curiosity_followup`'s own guard) -> byte-identical. Guarded so a build
                #    failure never crashes a turn. See
                #    research/runners/onebrain_xedge_curiosity_d6_production.py.
                #
                #    SEMANTIC-DROP rung (2026-09-01, `BRAIN_ONEBRAIN_XEDGE_CURIOSITY_D6_SEMANTIC_DROP`, default
                #    OFF): the crave gate below (ask_held + clears-floor, computed BEFORE `judge()` so the result
                #    can be threaded INTO the same load()/read() call that produces `recovered`) is translated by
                #    `semantic_drop_current` into a genuine hyperpolarizing pull on THIS session's own physical w0
                #    register, so a suppressed referent is dropped from `recovered` by the D6 substrate's own
                #    post-drive read -- not merely an appended string. `pool.cross_weight` (zeroed by the lesion)
                #    gates the magnitude, so lesioning still collapses the effect even if the boolean gate below
                #    were ever bypassed. See research/runners/onebrain_xedge_curiosity_d6_production.py.
                _cd6_ask_held, _cd6_read, _cd6_drop_current = False, None, None
                _cd6_crave_suppresses = False
                try:
                    from research.runners.onebrain_xedge_curiosity_d6_production import (
                        xedge_curiosity_d6_enabled, xedge_curiosity_d6_semantic_drop_enabled,
                        get_xedge_curiosity_d6_pool, crossedge_w0_shift, semantic_drop_current)
                    if xedge_curiosity_d6_enabled():
                        from research.runners._onebrain_crossedge_curiosity_to_d6wm import (
                            INTACT_FLOOR as _CD6_FLOOR)
                        _cd6_pool = get_xedge_curiosity_d6_pool(42)
                        _cd6_ask_held = bool(getattr(d6org, "_xedge_curiosity_recent_crave", False))
                        _cd6_read = crossedge_w0_shift(_cd6_pool, _cd6_ask_held)
                        _cd6_crave_suppresses = bool(
                            _cd6_ask_held and _cd6_read is not None and "error" not in _cd6_read
                            and _cd6_read.get("shift_w0", 0.0) <= -_CD6_FLOOR)
                        if _cd6_crave_suppresses and xedge_curiosity_d6_semantic_drop_enabled():
                            _cd6_drop_current = semantic_drop_current(_cd6_pool, d6org)
                except Exception as _cd6e0:   # never let the cross-edge read crash a turn
                    _cd6_read = {"on": True, "error": f"{type(_cd6e0).__name__}: {_cd6e0}"}
                    _cd6_crave_suppresses = False

                jq = d6org.judge(msg, lesion=d6les, xedge_drop_current=_cd6_drop_current)
                if jq is not None and jq.get("is_hold_query") and "readout" in jq:
                    try:
                        if _cd6_read is not None:
                            jq["curiosity_crossedge"] = _cd6_read
                            if _cd6_ask_held:
                                # CONSUME the crave -- a self-report qualifier fires once per crave episode,
                                # not on every subsequent hold-query indefinitely (mirrors prospective
                                # memory's own "fires once" intention consumption).
                                d6org._xedge_curiosity_recent_crave = False
                                if _cd6_crave_suppresses:
                                    jq["readout"] = jq["readout"] + (
                                        " Though a recent flash of curiosity is competing for my "
                                        "attention right now.")
                                    if _cd6_drop_current is not None:
                                        jq["curiosity_crossedge"]["semantic_drop_applied"] = True
                    except Exception as _cd6e:   # never let the cross-edge read crash a turn
                        jq["curiosity_crossedge"] = {"on": True, "error": f"{type(_cd6e).__name__}: {_cd6e}"}
                    # ── END faculty: onebrain curiosity->d6 LEARNED CROSS-EDGE ──
                    return JSONResponse({
                        "answer": jq["readout"], "abstained": False, "recalled_svo": None, "verified": True,
                        "renderer": rname, "brain": req.brain, "source": source, "rich": False, "activity": None,
                        "affect": affect_info, "worldmodel": worldmodel_info,
                        "multiref": dict(jq, kind="query"), "inner_state_readout": True,
                    })
            else:                                            # MAINTAIN: a turn introducing >=2 referents -> LOAD+HOLD
                jm = d6org.judge(msg, lesion=d6les)          # None when out-of-scope (<2 referents) -> byte-identical
                if jm is not None:
                    multiref_info = dict(jm, kind="maintain")
        except Exception as _d6e:                            # never crash a turn -> degrade to the normal answer
            multiref_info = {"on": True, "error": f"{type(_d6e).__name__}: {_d6e}"}

    # ── BEGIN faculty:activity-silent-wm (Gate-B, Mongillo 2008, 2026-08-26) ── PARENT-MERGE MARKER ───────────────
    # ACTIVITY-SILENT WORKING MEMORY as a maintenance-mode SWAP on the anaphora referent store. The live store holds the
    # discourse focus in a PERSISTENT-ACTIVITY attractor (it must keep firing); this mode holds it in short-term
    # synaptic FACILITATION (cp_stp_u) with the assembly SILENT across an intervening distractor turn, and reactivates
    # it on the next temporal-recall query by a NONSPECIFIC ping (Mongillo/Barak/Tsodyks 2008). Reuse-by-import from
    # research.runners.activity_silent_wm_production_organ (the 6/6-GO de-risk ActivitySilentWM; NO sim/ edit). Two
    # paths: (a) READ-OUT — a DISJOINT temporal-recall query ("what did we start with / the original topic / go back to
    # the beginning"; deliberately shares NO lexeme with the D6 hold-query, which runs first) reactivates the silently-
    # held focus and short-circuits with an honest functional read-out; the read-out ABSTAINS ("I don't recall ...")
    # rather than confabulate when the ping did not decisively reactivate (a no-confab gate). (b) MAINTAIN — every other
    # turn is a pure WRITE-ONLY side effect: a turn naming a referent holds it silently as the focus; a turn with no new
    # referent grows the silent delay (an intervening distractor). Neither MAINTAIN changes the reply. LESION
    # (BRAIN_SILENT_WM_LESION=1): recall builds the buffer with tau_f~5 (facilitation lesion) -> the ping cannot recover
    # the focus -> the read-out abstains (load-bearing: the correct anaphor vs abstain is caused by the SILENT hold, not
    # the host parse). 2026-08-26 FLIPPED DEFAULT-ON (wave 3 flip, `_SILENT_WM_DEFAULT_ON` above, 6/6 flip-soak GO):
    # `BRAIN_SILENT_WM=0` is the byte-identical escape (the block then imports nothing + returns nothing). Guarded so a
    # wiring failure can never crash a turn (degrades to the normal path). See
    # research/runners/_activity_silent_wm_production_soak.py (6/6 GO).
    if _silent_wm_flag_on():
        try:
            import research.runners.activity_silent_wm_production_organ as _SW
            sworg = _get_silent_wm_organ(cache_key)
            swles = _SW.silent_wm_lesioned()
            if _SW.is_silent_recall_query(msg):          # READ-OUT: disjoint temporal-recall -> ping the silent buffer
                jq = sworg.judge(msg, lesion=swles)      # None if nothing held -> falls through (byte-identical)
                if jq is not None and "readout" in jq:
                    return JSONResponse({
                        "answer": jq["readout"], "abstained": (jq.get("recovered") is None),
                        "recalled_svo": None, "verified": bool(jq.get("recovered") is not None),
                        "renderer": rname, "brain": req.brain, "source": source, "rich": False, "activity": None,
                        "affect": affect_info, "silent_wm": dict(jq, kind="recall"), "inner_state_readout": True,
                    })
            else:                                        # MAINTAIN (write-only side effect; reply unchanged)
                _sw_refs = _SW._extract_refs(msg)        # reuse the D6 host referent lexicon (declared residual)
                if _sw_refs:
                    sworg.write_referent(_sw_refs[0])    # hold the (first) named referent silently as the focus
                else:
                    sworg.note_distractor()              # no new referent -> the silent delay grows (a distractor turn)
        except Exception:
            pass  # never let the silent-WM coupling crash a turn -> fall through to the normal path
    # ── END faculty:activity-silent-wm ──

    # ── DISCOURSE EVENT REGISTER (Gate-B, D3, 2026-08-13) — who-was-doing-it-BEFORE across a connective, on spikes ──
    # Holds the running (a_curr,p_curr | a_prev,p_prev) event PAIR on the co-resident four-FS-WTA spiking register
    # (reuse-by-import from `d3_discourse_event_register_production_organ` -> the validated PairEventRegister spiking
    # twin, 6-seed GO). Two paths: (i) an ADDITIVE FOLD — an SVO discourse clause (a connective marks an event
    # BOUNDARY -> SHIFT) updates the running pair as a pure side-effect (does NOT store a fact or change the reply, so
    # the turn stays byte-identical; the normal assertion path remains the sole writer); (ii) a DISJOINT QUERY
    # short-circuit — 'who was doing it before/now?' is answered off the held spiking slot (read off cp_firing_states),
    # with the moat abstain (a before-answer is surfaced ONLY after a connective boundary actually opened this
    # conversation; a single-event register structurally abstains). Placed after D6 so the affect/episodic/worldmodel/
    # multiref read-out short-circuits keep precedence; the before/now query class is DISJOINT (no other organ handles
    # it), so every non-discourse turn is byte-identical. Default-ON; `BRAIN_DISCOURSE_REGISTER=0` -> the register is
    # built spiking=False AND this block is skipped (byte-identical). `BRAIN_DISCOURSE_REGISTER_LESION=1` -> the
    # prev-slot-silence spiking register (who-was-before collapses; NOW preserved — load-bearing).
    try:
        import research.runners.d3_discourse_event_register_production_organ as _DR
        _dr_on = _DR.discourse_register_enabled()
    except Exception:
        _DR = None
        _dr_on = False
    if _dr_on and getattr(chat.agent, "_event_register", None) is not None:
        try:
            dstate = _SESSION_DISCOURSE.setdefault(cache_key, _DR.new_state())
            # (i) additive fold (side-effect; reply stays byte-identical): a discourse SVO clause updates the pair.
            _DR.note_turn(msg, chat.agent, dstate, actions=chat.actions_set)
            # (ii) disjoint query short-circuit: 'who was doing it before/now?' -> read off the register + moat abstain.
            dr_reply = _DR.maybe_answer(msg, chat.agent, dstate)
            if dr_reply is not None:
                return JSONResponse({
                    "answer": dr_reply["answer"], "abstained": bool(dr_reply["abstained"]),
                    "recalled_svo": None, "verified": (not bool(dr_reply["abstained"])),
                    "renderer": rname, "brain": req.brain, "source": source, "rich": False,
                    "activity": None, "affect": affect_info,
                    "discourse_register": dict(dr_reply, kind=dr_reply.get("kind")), "inner_state_readout": True,
                })
        except Exception as _dre:
            pass   # never let the discourse read crash a turn — fall through to the normal path

    # ── CAUSAL WHY / WHAT-IF ORGAN (Gate-B, T1-4, 2026-08-13) ────────────────────────────────────────────────
    # A co-resident spiking CAUSAL FORWARD MODEL, grounded READ-ONLY in the brain's REAL fact store, answers a real
    # "what happens if <agent> <action>?" (forward-SIMULATION of an unseen consequence — the substrate rolls
    # A=(dog,go,east) -> B=(dog,reach,river) -> D=(dog,drink,water) though A->D was never taught) and "why did
    # <agent> <action>?" (the directed cause that survives a Pearl DO-probe — Y=(dog,wake,morning) reads C=(sun,rise,
    # sky), never the correlate X=(bird,sing)). Reuse-by-import from `causal_whatif_production_organ` (the grounded
    # de-risk, 6/6 GO). It owns a DISJOINT turn class (an explicit why-did/what-happens-if whose (agent,action) maps
    # to a known agent), placed AFTER the affect/episodic/worldmodel/multiref/discourse read-outs (so their short-
    # circuits keep precedence) and BEFORE comprehension/surprise/rich (so the causal query is not mis-read as an
    # assertion). MOAT-SAFE: the consequence/cause is emitted ONLY when `composer.query_patient` CONFIRMS it (the
    # no-confab moat the live recall uses); an unconfirmed/unmapped causal query ABSTAINS to the honest
    # `_honest_causal_answer` disclaimer (INTEGRATION #5 fallback) — 0 confabulation. A causal query about an UNKNOWN
    # agent falls THROUGH unchanged (byte-identical). LESION (`BRAIN_CAUSAL_LESION=1`): zero the learned forward
    # edges -> the forward-simulation cannot roll A->B->D and the DO-probe predecessor is no longer C -> BOTH
    # why/what-if collapse to the honest abstain (load-bearing). Default-ON; `BRAIN_CAUSAL=0` -> fully skipped
    # (byte-identical oracle). RESIDUALS (declared): grounding-by-DERIVATION not shared-substrate-merge (co-resident
    # forward-model bridge, burn-down #1); the DA sign is teacher-delivered; the causal STRUCTURE is teacher-rendered
    # (the wired scope is the validated chain-source what-if + confound why).
    causal_info = None
    try:
        import research.runners.causal_whatif_production_organ as _CA
        _causal_on = _CA.causal_enabled()
    except Exception:
        _CA = None
        _causal_on = False
    if _causal_on:
        try:
            _kind = _CA.is_causal_query(msg)
            if _kind is not None:
                evt, c_agent, c_action = _CA.extract_cue(msg, agents=getattr(chat, "agents_set", None))
                if c_agent is not None:            # a KNOWN agent -> OUR disjoint causal turn (else fall through)
                    _cause_composer = getattr(getattr(chat, "inner", None), "composer", None)
                    les = _CA.causal_lesioned()
                    c_answer = None
                    if _cause_composer is not None:
                        corg = _get_causal_organ(cache_key)
                        if _kind == "what_if" and evt == _CA.A:            # the validated chain-source what-if
                            j = corg.what_if(_cause_composer, lesion=les)
                            causal_info = dict(j, agent=c_agent, action=c_action)
                            c_answer = j["answer"]
                        elif _kind == "why" and evt == _CA.Y:              # the validated confound why
                            j = corg.why(_cause_composer, lesion=les)
                            causal_info = dict(j, agent=c_agent, action=c_action)
                            c_answer = j["answer"]
                    # ABSTAIN (0 confab): an unmapped causal target OR the forward model could not moat-confirm ->
                    # the honest disclaimer (never a confabulated cause/consequence). `stored` is the moat read for
                    # the (agent,action) — for a KNOWN fact it states EXACTLY that read; None -> the generic decline.
                    if c_answer is None:
                        from research.runners._conversation_turing_test_derisk import _honest_causal_answer
                        _stored = None
                        try:
                            if _cause_composer is not None and c_action:
                                _stored = _cause_composer.query_patient(c_agent, c_action)
                        except Exception:
                            _stored = None
                        c_answer = _honest_causal_answer(c_agent, c_action, _stored)
                        if causal_info is None:
                            causal_info = {"on": True, "lesioned": bool(les), "kind": _kind,
                                           "agent": c_agent, "action": c_action, "confirmed": False, "answer": None}
                        causal_info["abstained"] = True
                    _c_confirmed = bool(causal_info.get("confirmed"))
                    return JSONResponse({
                        "answer": c_answer,
                        "abstained": (not _c_confirmed),
                        "recalled_svo": None, "verified": _c_confirmed,
                        "renderer": rname, "brain": req.brain, "source": source, "rich": False,
                        "activity": None, "affect": affect_info, "worldmodel": worldmodel_info,
                        "multiref": multiref_info, "causal": causal_info, "inner_state_readout": True,
                    })
        except Exception as _cae:  # never let the causal read crash a turn — fall through to the normal path
            causal_info = {"on": True, "error": f"{type(_cae).__name__}: {_cae}"}

    # ── SCALAR-IMPLICATURE PRAGMATIC BELIEF (D-pragmatics, Task-#12, 2026-08-13) ─────────────────────────────
    # When the user makes a SCALAR-QUANTITY claim/probe ("I ate some of the cookies", "not all of them"), FORM the
    # brain's graded listener-belief over the interpretations {none, SBNA, all} for that utterance and PREPEND an
    # honest functional pragmatic reading ("some" +> "some but not all", with "all" left ~0.27-possible). This wires
    # the de-risk-CLOSED W4 depth-2 RSA GRADED-implicature belief (2026-08-13-w4-detector-operating-point-homeostat-GO,
    # 6/6) in as the production BELIEF SOURCE, replacing the leg2_v2 WTA ONE-HOT collapse that falsely rules "all" out.
    # The belief is a spiking read of the real Izhikevich RSA substrate (plasticity off, fixed operating point), built
    # once + frozen (reuse-by-import from `pragmatic_production_organ`). SCOPED: this is the MINIMAL genuine end-to-end
    # path — a single scalar-implicature turn class; the pipeline had NO pragmatic-implicature slot before (a general
    # pragmatic comprehension front-end is the mapped gap). Moat-safe + additive: it fires ONLY on a scalar-quantity
    # context (partitive "<scalar> of" / "not all" / a some-vs-all probe), never manufactures a fact, never flips an
    # abstain, never enters the certainty band — only PREPENDS a reading. Placed BEFORE comprehension: the scalar
    # implicature is a STRUCTURAL property of the quantifier, independent of whether the brain knows the content words,
    # so the reading is surfaced on WHATEVER path the turn takes (the comprehension-repair early-return below carries
    # it too). Default-ON; `BRAIN_PRAGMATIC=0` -> fully skipped (byte-identical oracle). LESION
    # (`BRAIN_PRAGMATIC_LESION=1`): the normalization-lesion belief (flat [0,0.5,0.5]) -> the implicature margin
    # collapses to ~0 -> no reading (load-bearing: the graded content is the substrate's FS divisive normalization,
    # not host-injected). Guarded so a read failure can never crash a turn.
    pragmatic_info = None
    pragmatic_prefix = ""
    try:
        import research.runners.pragmatic_production_organ as _PR
        _pragmatic_on = _PR.pragmatic_enabled()
    except Exception:
        _PR = None
        _pragmatic_on = False
    if _pragmatic_on:
        try:
            pj = _get_pragmatic_organ().judge_text(msg, lesion=_PR.pragmatic_lesioned())
            if pj is not None:                      # a scalar-implicature turn class was detected (else byte-identical)
                pragmatic_info = dict(pj)
                pragmatic_prefix = _PR.pragmatic_notice(pj)
        except Exception as _pe:  # never let the pragmatic read crash a turn -> the un-noticed answer
            pragmatic_info = {"on": True, "error": f"{type(_pe).__name__}: {_pe}"}

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
            # THIS session's OWN xedge focus (None if this session holds no referent, multiref is off, or xedge is
            # off) -- resolved from `d6org` (hoisted above, THIS cache_key's own MultiReferentWMOrgan) and passed
            # EXPLICITLY into every comprehension call below. 2026-08-27 cross-session leak fix: `corg` is a
            # process-shared singleton serving every session, so it must never infer "what's held" from shared
            # mutable pool state -- only from what THIS turn's own session organ reports.
            _wm_focus = d6org.current_focus() if d6org is not None else None
            cj = corg.judge(msg, brain_vocab=_brain_vocab(chat), lesion=_CO.comprehension_lesioned(),
                            wm_focus=_wm_focus)
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
                # ── ONE-BRAIN XEDGE PER-TURN LIVE PLASTICITY (PART 3, 2026-08-27) ───────────────────────────────
                # LEARN THROUGH THE CONVERSATION: on a turn where THIS session's own WM referent is HELD (`_wm_focus`
                # above, resolved from `d6org.current_focus()` -- never a shared-pool ambient global) AND
                # comprehension RESOLVED confidently, apply ONE in-brain self-supervised credited plasticity step to
                # the d6-WM->comprehension cross-edge (three-factor, DA-coincidence-gated, bounded by stdp_w_max) --
                # the SAME atom PART 2 fires over a build curriculum, now fired ONCE per REAL chat turn so the edge
                # GROWS from W0=0.05 through the conversation itself. The credit VALUE + teach DIRECTION are read
                # off the brain's OWN confident spiking resolution (no host label). Behind BRAIN_ONEBRAIN_XEDGE +
                # BRAIN_ONEBRAIN_XEDGE_LEARN (both default-ON since the 2026-08-28 flip, commit fe1911f2f --
                # corrected, this comment previously said "both default-OFF") -> byte-identical no-op when
                # EXPLICITLY set to 0 / no referent held / content inconclusive. Never crashes a turn.
                if cj["comprehended"] or known:
                    try:
                        from research.runners.onebrain_xedge_production import credit_live_turn_from_comprehension
                        _xtrace = credit_live_turn_from_comprehension(corg, cj["svo"], wm_focus=_wm_focus)
                        if _xtrace is not None:
                            comprehension_info["xedge_live_learn"] = _xtrace
                    except Exception as _xe:
                        comprehension_info["xedge_live_learn"] = {"error": f"{type(_xe).__name__}: {_xe}"}
                if (not cj["comprehended"]) and (not known):
                    # LOW margin + not a known binding -> the brain did not comprehend the roles -> honest abstain.
                    comprehension_info["abstained"] = True
                    # ── OTHER-REPAIR (T1-6): instead of a DEAD-END abstain, ask a TARGETED clarification that
                    # NAMES what did not resolve — the unresolved thematic ROLE (from the SAME D4 spiking sel-pool
                    # per-noun read, `repair_target()`) or the OOV token. The repair is TRIGGERED by the spiking
                    # comprehension signal and (for the role branch) SHAPED by it: under the D4 lesion the per-noun
                    # evidence collapses -> no target -> the byte-identical bare abstain. Moat-safe (a QUESTION,
                    # never a fact; the turn stays an abstain). Default-ON; `BRAIN_REPAIR=0` -> the bare abstain.
                    answer = _CO.didnt_follow_message(cj["svo"])
                    repair_info = None
                    try:
                        import research.runners.repair_production_organ as _RP
                        if _RP.repair_enabled():
                            tgt = corg.repair_target(
                                msg, brain_vocab=_brain_vocab(chat),
                                lesion=_CO.comprehension_lesioned(), wm_focus=_wm_focus)
                            q = _RP.clarification_question(tgt) if tgt is not None else None
                            if q:
                                answer = q
                                repair_info = dict(tgt)
                                repair_info["repaired"] = True
                            else:
                                repair_info = {"on": True, "repaired": False,
                                               "target": (dict(tgt) if tgt is not None else None)}
                    except Exception as _re:  # never let the repair read crash a turn -> the bare abstain
                        repair_info = {"on": True, "error": f"{type(_re).__name__}: {_re}"}
                    payload = {
                        # PRAGMATIC (Task-#12): the scalar-implicature reading is STRUCTURAL (independent of the OOV
                        # content words), so it is PREPENDED here too — a scalar-quantity turn whose content words are
                        # unknown still surfaces the graded reading + attaches the `pragmatic` block. Empty prefix /
                        # null block when out of scope or disabled -> byte-identical bare repair.
                        "answer": pragmatic_prefix + answer,
                        "abstained": True, "recalled_svo": None, "verified": False,
                        "renderer": rname, "brain": req.brain, "source": source,
                        "rich": False, "activity": None, "affect": affect_info,
                        "comprehension": comprehension_info, "not_understood": True,
                        "pragmatic": pragmatic_info,
                    }
                    if repair_info is not None:      # BRAIN_REPAIR=0 -> key absent -> byte-identical bare abstain
                        payload["repair"] = repair_info
                    # >>> W5 AFFECTIVE ToM BEGIN (comprehension-repair early-return; additive, mergeable block) ──
                    # This abstain-class turn RETURNS here, before the main assembly sites — so an empathy trigger
                    # whose content words are OOV ("Sam's team lost") would otherwise bypass the empathic lead. The
                    # observe block above already computed `tom_lead`/`tom_info`; prepend the lead OUTERMOST (ahead of
                    # the pragmatic prefix) + attach the additive `affective_tom` trace, so empathy also leads an
                    # OOV/unresolved abstain. Empty lead / no key when disabled, no other-agent, or lesion-collapsed
                    # -> byte-identical. The abstain content is unchanged (empathy colors the surface, never a fact).
                    if tom_lead:
                        payload["answer"] = tom_lead + payload["answer"]
                    if tom_info is not None and tom_info.get("acted"):
                        payload["affective_tom"] = tom_info
                    # <<< W5 AFFECTIVE ToM END ─────────────────────────────────────────────────────────────────
                    return JSONResponse(payload)
        except Exception as _ce:  # never let the comprehension read crash a turn — degrade to the normal answer
            comprehension_info = {"on": True, "error": f"{type(_ce).__name__}: {_ce}"}

    # ── EXPECTATION-VIOLATION / SURPRISE (Gate-B, D2, 2026-08-12) ────────────────────────────────────────────
    # When the user ASSERTS a fact (agent,action,patient) for which the brain ALREADY HOLDS a stored
    # (agent,action)->patient association, run a genuinely-SPIKING predictive-coding MISMATCH unit (the co-resident
    # circuit, reuse-by-import from `surprise_production_organ`, 6/6-GO D2 de-risk): the recalled expectation
    # delivers GABA_A SUBTRACTIVE inhibition, the asserted patient excitation; confirm cancels (~0 Hz), a
    # contradiction/novelty fires. The signal is a windowed `cp_firing_states[surprise]` rate (NO host
    # recalled==asserted compare). On a firing surprise the brain PREPENDS an honest functional NOTICE. Additive,
    # moat-safe (runs only when a stored expectation exists; never manufactures a fact or flips an abstain).
    # Default-ON; `BRAIN_SURPRISE=0` -> fully skipped (byte-identical oracle).
    surprise_info = None
    surprise_prefix = ""
    # RECONSOLIDATION (F-lane) inits — beside the surprise inits so BOTH response branches see them even when
    # the block is skipped (belief revision hooks INSIDE the surprise block below, reusing its ONE spiking read).
    reconsolidation_info = None
    reconsolidation_prefix = ""
    try:
        import research.runners.surprise_production_organ as _SO
        _surprise_on = _SO.surprise_enabled()
    except Exception:
        _SO = None
        _surprise_on = False
    try:
        import research.runners.reconsolidation_production_organ as _RC
    except Exception:
        _RC = None
    if _surprise_on:
        try:
            asrt = _SO.extract_assertion(msg)
            if asrt is not None:
                a_s, v_s, p_asserted = asrt
                # the EXPECTED patient is RECALLED by the brain's own spiking recall (not a host lookup): what_does
                # returns the stored patient for (agent,action), or falsy if the brain holds no expectation here.
                try:
                    p_stored = chat.inner.what_does(a_s, v_s)
                except Exception:
                    p_stored = None
                if p_stored:
                    sorg = _get_surprise_organ()
                    sj = sorg.judge(a_s, v_s, str(p_stored), str(p_asserted), lesion=_SO.surprise_lesioned())
                    surprise_info = dict(sj)
                    if sj["surprised"]:
                        surprise_prefix = _SO.surprise_notice(a_s, v_s, str(p_stored))
                    # ── RECONSOLIDATION / belief revision (Gate-B, F-lane, 2026-08-12) ───────────────────────
                    # When the asserted patient CONTRADICTS the stored one (the D2 spiking surprise window is OPEN),
                    # UPDATE the stored fact IN PLACE instead of appending a contradictory duplicate. The window-open
                    # decision reuses the SAME spiking `cp_firing_states[surprise]` read just computed (`sj` passed in —
                    # ZERO extra spiking reads); the in-place rewrite reuses the composer's OWN de-risked store path
                    # (rf `update_on_mismatch`; onebrain SAME-slot `_write_block`+`_compose_phases`). Reuse-by-import
                    # from `reconsolidation_production_organ` (verify-first GO: rf 6/6, onebrain 3/3, 100% attributable).
                    # Moat-safe: it only rewrites a fact the brain ALREADY HOLDS (guarded by p_stored + p_asserted!=
                    # p_stored), NEVER on a re-statement (window closed -> not surprised), NEVER fabricates a fact; it
                    # only PREPENDS an honest notice. Default-ON; `BRAIN_RECONSOLIDATION=0` -> append-only (byte-identical).
                    # LESION (`BRAIN_RECONSOLIDATION_LESION=1`): window fires but the in-place update is BLOCKED ->
                    # append-only fallback -> recall returns the STALE fact (load-bearing).
                    try:
                        _rc_on = (_RC is not None) and _RC.reconsolidation_enabled()
                    except Exception:
                        _rc_on = False
                    if (_rc_on and sj.get("surprised")
                            and str(p_asserted).lower() != str(p_stored).lower()):
                        try:
                            composer = getattr(chat.inner, "composer", None)
                            res = _get_reconsolidation_organ().reconsolidate(
                                composer, a_s, v_s, str(p_stored), str(p_asserted),
                                sj=surprise_info, lesion=_RC.reconsolidation_lesioned())
                            reconsolidation_info = res
                            if res.get("action") == "rewrite":
                                reconsolidation_prefix = _RC.reconsolidation_notice(
                                    a_s, v_s, str(p_stored), str(p_asserted))
                        except Exception as _rce:  # never let belief revision crash a turn — keep append-only
                            reconsolidation_info = {"on": True, "error": f"{type(_rce).__name__}: {_rce}"}
        except Exception as _se:  # never let the surprise read crash a turn — degrade to the un-noticed answer
            surprise_info = {"on": True, "error": f"{type(_se).__name__}: {_se}"}
    # ── BEGIN faculty: board #129 surprise->episodic/source_provenance LEARNED CROSS-EDGES — additive,
    #    DEFAULT-OFF (BRAIN_ONEBRAIN_XEDGE_SURPRISE_EPISODIC). Mirrors the PART-1 d6-WM->comprehension frozen
    #    cross-edge wire-in (research/runners/onebrain_xedge_production.py) and the R4 self_schema->source_
    #    provenance wire-in (research/runners/onebrain_xedge_selfschema_production.py) on board #129's TWO-cross-
    #    edge / divisive-ratio construction (2026-08-28-surprise-episodic-129construction-6seed-GO.md): the D2
    #    expectation-violation circuit's CONTRADICT drive is a LEARNED Hebbian cross-synapse pair onto source_
    #    provenance's perceived/generated pools, grown once on a SHARED merge pool, then frozen (no weight moves
    #    during any live turn). THIS turn's OWN live D2 surprise verdict (`surprise_info["surprised"]`, computed
    #    above) drives whether the cross-edges' presynaptic CONTRADICT drive is held during a co-temporal read of
    #    the construction's own validated ambiguous-item provenance instrument (`amb_read_ratio`, reused verbatim
    #    — not reimplemented). Attaches an ADDITIVE diagnostic field ONLY
    #    (`resp["surprise"]["source_provenance_crossedge"]`); NEVER touches `resp["answer"]` or any existing
    #    `surprise` field, and runs on an INDEPENDENT SurpriseEpisodic129Pool instance — ZERO risk to the
    #    already-default-ON D2 surprise notice/reconsolidation pipeline above. LOAD-BEARING: the shift toward
    #    GENERATED vanishes under BRAIN_ONEBRAIN_XEDGE_SURPRISE_EPISODIC_LESION=1 (BOTH cross-edges zeroed
    #    together — the SAME joint lesion the 6-seed GO's own F2 gate used). DEFAULT-OFF:
    #    BRAIN_ONEBRAIN_XEDGE_SURPRISE_EPISODIC unset -> no extra key, byte-identical. Guarded so it never
    #    crashes a turn. HONEST RESIDUAL (carried from the construction's own finding, §5, not re-litigated
    #    here): a single-edge lesion check found the confirm-side edge (not the surprise-side edge) does almost
    #    all the causal work — this wire-in exercises the already-validated JOINT (both-edges) mechanism, which
    #    is what BRAIN_ONEBRAIN_XEDGE_SURPRISE_EPISODIC_LESION lesions. See research/runners/
    #    onebrain_xedge_surprise_episodic_production.py.
    if surprise_info is not None and "surprised" in surprise_info:
        try:
            from research.runners.onebrain_xedge_surprise_episodic_production import (
                xedge_surprise_episodic_enabled, get_xedge_surprise_episodic_pool, crossedge_provenance_shift_129)
            if xedge_surprise_episodic_enabled():
                _xse_pool = get_xedge_surprise_episodic_pool(42)
                _xse_held = bool(surprise_info.get("surprised", False))
                _xse_read = crossedge_provenance_shift_129(_xse_pool, _xse_held)
                if _xse_read is not None:
                    surprise_info["source_provenance_crossedge"] = _xse_read
        except Exception as _xsee:   # never let the diagnostic read crash a turn
            surprise_info["source_provenance_crossedge"] = {"on": True, "error": f"{type(_xsee).__name__}: {_xsee}"}
    # ── END faculty: board #129 surprise->episodic/source_provenance LEARNED CROSS-EDGES ──

    # ── NON-CONTRADICTION ASSERTION-GATE (Gate-B, B3, 2026-08-12) ────────────────────────────────────────────
    # When the user ASSERTS a transitive fact whose POLARITY contradicts the brain's stored polarity for the EXACT
    # SAME SVO ("the dog eats grass" vs a stored "a dog does NOT eat grass"), REJECT the assertion instead of
    # silently overwriting a held belief. The load-bearing recall is a genuinely-SPIKING polarity WTA on the ONE
    # production composer (`chat.inner.is_it_true` == `OneBrainComposer.ask_yes_no` -> `_spiking_select` over
    # `cp_firing_states`); the gate proper is the ONE host boolean `stored != asserted` (the accepted no-confab moat
    # pattern). Reuse-by-import from `b3_noncontradiction_production_organ` (6-seed-GO de-risk gate, NOT reimplemented).
    # It runs BEFORE the store/gate so a reject returns before `_maybe_acquire` overwrites the held belief. Moat-safe:
    # an UNKNOWN SVO -> accept (never a fabricated rejection); MUTUALLY EXCLUSIVE with D2 surprise (ask_yes_no is
    # "unknown" unless the asserted PATIENT matches the stored one, so B3 fires only same-SVO/opposite-polarity).
    # LESION (`BRAIN_NONCONTRADICTION_LESION=1`): bypass the spiking recall -> every recall "unknown" -> the gate goes
    # INERT (contradictions slip through). Default-ON; `BRAIN_NONCONTRADICTION_GATE=0` -> fully skipped (byte-identical).
    noncontradiction_info = None
    try:
        import research.runners.b3_noncontradiction_production_organ as _NC
        _nc_on = _NC.noncontradiction_enabled()
    except Exception:
        _NC = None
        _nc_on = False
    if _nc_on:
        try:
            nres = _get_noncontradiction_organ().check(
                chat.inner.is_it_true, msg, lesion=_NC.noncontradiction_lesioned())
            if nres is not None:
                noncontradiction_info = dict(nres)
                if nres["reject"]:
                    return JSONResponse({
                        "answer": _NC.rejection_message(nres["svo"], nres["stored_polarity"]),
                        "abstained": False, "recalled_svo": nres["svo"], "verified": True,
                        "renderer": rname, "brain": req.brain, "source": source,
                        "rich": False, "activity": _read_activity(), "affect": affect_info,
                        "noncontradiction": noncontradiction_info, "rejected_contradiction": True,
                    })
        except Exception as _nce:  # never crash a turn -> degrade to the normal answer (gate goes silent)
            noncontradiction_info = {"on": True, "error": f"{type(_nce).__name__}: {_nce}"}

    # ── SELF-MODEL / METACOGNITION confidence read-out (Gate-B, E1, 2026-08-12) ──────────────────────────────
    # AFTER the gate/moat produces (or refuses) an answer, read a genuinely-SPIKING confidence of that answer off
    # the co-resident balance-of-evidence monitor (`|rate(asm1)-rate(asm0)|` from cp_firing_states, reuse-by-import
    # from `metacog_production_organ`, the E1 balance de-risk, 6/6 GO). The evidence is the brain's OWN mean
    # role-decode confidence (its spiking parse certainty for the answer). On a LOW-confidence answer the brain
    # honestly QUALIFIES it (an honest FUNCTIONAL hedge, never a phenomenal claim, never a content change). Moat-safe
    # + additive: it only qualifies an ALREADY-produced, moat-verified answer; an abstain (no answer) is skipped.
    # Default-ON; `BRAIN_METACOG=0` -> fully skipped (byte-identical oracle).
    try:
        import research.runners.metacog_production_organ as _MC
        _metacog_on = _MC.metacog_enabled()
    except Exception:
        _MC = None
        _metacog_on = False

    # #184 (silent-regression guard, 2026-08-27): whether THIS turn's composer even SUPPORTS tracing at all
    # (`hasattr(_composer, "last_trace")`, the same test the trace-flip block above already gated on). A composer
    # that never traces (the rate composer / an external composer) legitimately has no confidence to read — no
    # warning there. A trace-CAPABLE composer whose read still comes back empty on an ANSWERED turn is the
    # plumbing-bug signature (exactly how `TieredFactStore.__setattr__` silently ate `last_trace` for a day: the
    # answer was produced, tracing WAS supported, yet `activity` read None with nothing logged).
    _composer_traces = _composer is not None and hasattr(_composer, "last_trace")

    def _metacog_qualify(activity, no_answer):
        """Read the spiking confidence of the answer just produced. Returns (hedge_prefix, metacog_info). Skips an
        abstain / guess (no recalled answer to qualify) and any turn with no decoded-role confidence (out of scope).

        #184: when an ANSWER was given (no_answer=False) by a trace-CAPABLE composer, yet the confidence read
        still comes back empty (activity is None, or has no role carries a confidence), that is NOT a legitimate
        out-of-scope skip — it is an unexpected empty read, i.e. a plumbing regression silently disabling this
        honesty hedge. Log a WARNING (never silent) so a future regression of this shape cannot hide again; the
        turn still degrades gracefully to the un-hedged answer either way (never crashes/blocks a turn)."""
        if not _metacog_on or no_answer:
            return "", None
        try:
            mrc = _MC.mean_role_confidence(activity)
            ev = _MC.evidence_from_role_conf(mrc)
            if ev is None:
                if _composer_traces:
                    print(f"[webapp] METACOG WARNING (#184): an answer was produced by a trace-capable composer "
                          f"but the confidence read came back empty this turn (activity_is_none="
                          f"{activity is None}, roles={len((activity or {}).get('roles') or [])}) -- the honesty "
                          f"hedge is silently disabled. This is the plumbing-bug signature (TieredFactStore.__"
                          f"setattr__ ate last_trace for a day the same way); check the activity-trace plumbing.",
                          flush=True)
                return "", None
            j = _get_metacog_organ().judge(ev, lesion=_MC.metacog_lesioned())
            info = dict(j)
            info["mean_role_conf"] = (float(mrc) if mrc is not None else None)
            return (("" if j["confident"] else _MC.hedge_prefix()), info)
        except Exception as _me:  # never let the confidence read crash a turn — degrade to the un-hedged answer
            return "", {"on": True, "error": f"{type(_me).__name__}: {_me}"}

    # ── CURIOSITY follow-up (Gate-B, D3, 2026-08-12) ─────────────────────────────────────────────────────────
    # On an ABSTAIN (the no-confab moat refused -> the brain holds NO answer -> a maximal epistemic gap), read a
    # genuinely-SPIKING curiosity drive off the co-resident ASK pool (`cp_firing_states[ask]` driven by the
    # `from_novelty` neuromodulator, reuse-by-import from `curiosity_production_organ`, the DR-1 crave-drive, on-
    # bridge 6-seed / 6/6-SAFE, corr(gap,want)=+0.996). When the ASK pool CRAVES (want >= threshold) the brain
    # APPENDS an honest FOLLOW-UP QUESTION about the topic — crave, don't refuse. The moat is INVERTED, not broken:
    # the answer stays an abstain (never a confabulated fact); the added text is unambiguously a QUESTION. Moat-safe
    # + additive: it runs ONLY on an abstain (there is no answer to corrupt) and only APPENDS a question. A FAMILIAR
    # topic (a confident recall) is OUT OF SCOPE -> byte-identical. Default-ON; `BRAIN_CURIOSITY=0` -> byte-identical.
    try:
        import research.runners.curiosity_production_organ as _CU
        _curiosity_on = _CU.curiosity_enabled()
    except Exception:
        _CU = None
        _curiosity_on = False

    def _curiosity_followup(abstained):
        """Read the spiking curiosity drive on an ABSTAIN. Returns (suffix, curiosity_info). Empty on a non-abstain
        (out of scope -> byte-identical), when disabled, when no topic is extractable, or when the ASK pool is not
        craving (want < threshold, e.g. under the lesion). The suffix is unambiguously a QUESTION (moat-safe)."""
        if not _curiosity_on or not abstained:
            return "", None
        try:
            topic = _CU.extract_topic(msg)
            j = _get_curiosity_organ().judge(novelty=_CU.NOVEL_SIGNAL, lesion=_CU.curiosity_lesioned())
            info = dict(j)
            info["topic"] = topic
            curious = bool(j["curious"])
            # ── DA/ENGAGEMENT-GATED crave-threshold (WAVE-0 Gap-4 coupling (b)), default-OFF ──────────────────────
            # When the brain is ENGAGED (self-produced tonic DA above baseline, read off chat._last_da_drives set by
            # the DA-mode block earlier this turn), LOWER the effective crave-threshold so the ASK pool crosses it
            # more readily (ask a follow-up on a topic it would otherwise let pass); disengaged -> raise it. It only
            # changes WHETHER the honest follow-up QUESTION is appended — never manufactures a fact or flips the
            # abstain (moat preserved). Default-OFF: BRAIN_CURIOSITY_DA unset -> block skipped -> the organ's
            # calibrated threshold decides `curious` unchanged and NO curiosity_da key -> byte-identical to HEAD.
            # LESION (BRAIN_CURIOSITY_DA_LESION=1): the DA modulation is pinned to 0 (gain 1.0) -> the DA-dependence
            # vanishes. See webapp/da_curiosity_drives_chat.py.
            try:
                from webapp import da_curiosity_drives_chat as _DAC
                if _DAC.da_curiosity_enabled():
                    curious, _da_trace = _DAC.crave_decision(chat, j["want_hz"], j["threshold"])
                    info["curious"] = curious
                    info["curiosity_da"] = _da_trace
            except Exception as _dce:   # additive trace only; never let it change the organ's own decision on error
                info["curiosity_da"] = {"on": True, "error": f"{type(_dce).__name__}: {_dce}"}
            # ── onebrain curiosity->d6 LEARNED CROSS-EDGE (2026-09-01): persist THIS turn's OWN live crave
            # verdict onto THIS SESSION's own d6org instance (never the shared process pool -- 2026-08-27
            # session-isolation pattern), so a LATER hold-query turn in the SAME session can drive the frozen
            # ask->w0 cross-edge from it. Written ONLY when the wire-in is enabled (byte-identical footprint
            # when off: no attribute is ever set, matching every other default-OFF gate in this codebase) and
            # only when d6 is in scope for this session (`d6org` may be None: multiref disabled this session).
            try:
                from research.runners.onebrain_xedge_curiosity_d6_production import xedge_curiosity_d6_enabled
                if xedge_curiosity_d6_enabled() and d6org is not None:
                    d6org._xedge_curiosity_recent_crave = bool(curious)
            except Exception:
                pass   # never let this bookkeeping crash a turn -> the next hold-query just sees no crave
            if curious:
                return _CU.followup_question(topic), info
            return "", info
        except Exception as _cue:  # never let the curiosity read crash a turn — degrade to the bare abstain
            return "", {"on": True, "error": f"{type(_cue).__name__}: {_cue}"}

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
            # ── CONFIDENCE CAPS FORTHCOMINGNESS (board #94, 2026-08-27) ──────────────────────────────────────
            # Read the FLOOR the mood coupling (or the composer's own construction default) just decided, then
            # request ONE extra "reach" fact from the composer. The confidence read (taken ONCE, right after
            # `rich.answer()` returns below, and REUSED for the E1 hedge so the SAME spiking read drives both)
            # decides whether the reach's extra fact is KEPT (confident) or TRUNCATED back to the floor (not
            # confident / out of scope) -- NEVER overriding the floor downward, only ever granting ONE bonus
            # fact on a HIGH read (the safe direction). Default-ON (`BRAIN_CONFIDENCE_FORTHCOMING`; a
            # 2026-08-27 default-ON attempt was reverted the same day, hollow on real traffic, then RE-FLIPPED
            # 2026-09-01 once the margin-scale calibration residual was closed -- see
            # webapp/confidence_forthcoming_chat.py) -> `BRAIN_CONFIDENCE_FORTHCOMING=0` skips this whole block
            # -> byte-identical to pre-wiring.
            _cf_on = False
            _cf_floor = None
            _cf_saved_plan = None
            try:
                from webapp import confidence_forthcoming_chat as _CF
                _cf_on = _CF.confidence_forthcoming_enabled() and _metacog_on
            except Exception:
                _CF = None
                _cf_on = False
            if _cf_on:
                _cf_floor = _CF.floor_override() or (rich.max_sentences, rich.max_elaborations)
                _cf_saved_plan = (rich.max_sentences, rich.max_elaborations)
                rich.max_sentences, rich.max_elaborations = _CF.reach_plan(*_cf_floor)
            try:
                r = rich.answer(msg)
            finally:
                if _cf_saved_plan is not None:
                    rich.max_sentences, rich.max_elaborations = _cf_saved_plan
                if _saved_plan is not None:
                    rich.max_sentences, rich.max_elaborations = _saved_plan
        except Exception as e:
            raise HTTPException(500, f"rich chat turn failed: {type(e).__name__}: {e}")
        # CONFIDENCE CAPS FORTHCOMINGNESS, continued: read the SAME post-answer activity trace + metacog
        # confidence the E1 hedge below reads, ONCE (cached + reused at the hedge site so the organ's spiking
        # margin read never runs twice in one turn), and apply the cap BEFORE `facts`/`resp` are built off `r`
        # so every downstream field (recalled_svo, supporting_facts, n_sentences, the answer text) reflects the
        # (possibly truncated) content consistently. Skips an abstain / hypothesis (nothing to cap) -> byte-
        # identical there. Never crashes a turn -- degrades to the un-capped (reach-sized) answer on any error.
        _cf_mc_prefix = ""
        _cf_mc_info = None
        _cf_trace = None
        _cf_activity = None
        if _cf_on and _cf_floor is not None and (not r.get("abstained")) and (not r.get("hypothesis")):
            try:
                _cf_activity = _read_activity()
                _cf_mc_prefix, _cf_mc_info = _metacog_qualify(_cf_activity, False)
                r, _cf_trace = _CF.apply_cap(rich, r, _cf_floor[0], (_cf_mc_info or {}).get("confident"))
            except Exception as _cfe:  # never let the cap decision crash a turn -- degrade to the un-capped answer
                _cf_trace = {"on": True, "error": f"{type(_cfe).__name__}: {_cfe}"}
        facts = [list(f) for f in r.get("facts", [])]
        # DERIVED (reasoning-frontier hardening, moat-hardening audit #4/#5): True when this turn's direct fact
        # was a COMPOSED multi-hop inference (compositional_chain_route), not a directly-recalled one.
        _rich_derived = bool(r.get("derived"))
        # EPISODIC STORE (Gate-B, D5, Hook B): a normal answered turn with a VERIFIED SVO BTSP-forms the answered
        # topic's (subject/agent) CA3 assembly (write-only; changes NO reply text). The spiking BTSP write is
        # ~seconds on cupy but ~510s/topic on numpy@2000, so it is GATED behind cupy (`_episodic_store_ok`) — on a
        # numpy deployment the write is DEFERRED (a declared latency residual: the recall GATE stays spiking + load-
        # bearing; only the WRITE is amortized to the cupy deployment). Guarded so it never crashes the turn.
        # EXCLUDES a derived (ChainedSVO) direct answer (moat-hardening audit #5): a composed multi-hop
        # inference's terminal was not itself directly recalled this turn.
        if _episodic_on and (not r["abstained"]) and facts and not _rich_derived and _episodic_store_ok():
            try:
                _ep_topics = getattr(chat, "agents_set", None) or _brain_vocab(chat)
                _EP.get_episodic_organ(cache_key, 42, _ep_topics).note_topic(facts[0][0])
            except Exception:
                pass
        # OPEN-ENDED GENERATION (#3E): a generated HYPOTHESIS turn returns NO supporting facts (a guess is not a
        # recalled fact) but carries `hypothesis`/`hypothesis_svo`. Surface those so the client can render the
        # guess distinctly + so the SVO the fluent prose asserts is checkable, WITHOUT reporting it as recalled.
        is_hyp = bool(r.get("hypothesis"))
        # SURPRISE (Gate-B, D2): if the asserted fact violated a stored expectation (a firing mismatch), PREPEND the
        # honest functional notice to the turn's answer. Additive; empty prefix when not surprised / disabled.
        resp = {
            "answer": (pmem_prefix + worldmodel_prefix + surprise_prefix + reconsolidation_prefix
                       + pragmatic_prefix + r["answer"]),
            "abstained": bool(r["abstained"]),
            # the direct recall (the first supporting fact) is the gate hit,
            # surfaced for parity with the single-fact path's recalled_svo.
            # A hypothesis has no recalled fact -> null (it is a guess, not knowledge).
            # A DERIVED answer (moat-hardening audit #5) also reports null here -- a composed multi-hop
            # inference's terminal is NOT a directly-recalled fact; see `derived`/`derived_from` instead.
            "recalled_svo": (None if _rich_derived else (facts[0] if facts else None)),
            # DERIVED (reasoning-frontier hardening, moat-hardening audit #4/#5): True when the direct fact
            # above was a COMPOSED multi-hop inference (compositional_chain_route), not a directly-recalled
            # one; derived_from names the verified hop-facts it was reasoned from. False/None otherwise.
            "derived": _rich_derived,
            "derived_from": (list(r.get("derived_from") or []) if _rich_derived else None),
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
            # When the confidence-forthcoming cap ran (board #94) this is the CACHED pre-truncation activity read
            # (the one the cap decision itself used), so the trace stays consistent with `resp["metacog"]` below
            # even though a truncation's re-render may have since advanced the composer's own `last_trace`.
            "activity": (_cf_activity if _cf_mc_info is not None else _read_activity()),
            # AFFECT (Gate-B): the live mood that colored this turn's forthcomingness + prose manner (debug trace;
            # the tone token lives here, never on the surface). null when affect is disabled (BRAIN_AFFECT=0).
            "affect": affect_info,
            # COMPREHENSION (Gate-B, D4): the spiking role-binding read for an in-scope transitive (margin/threshold/
            # comprehended); null when out of scope or disabled (BRAIN_COMPREHENSION_GATE=0). A pass-through here.
            "comprehension": comprehension_info,
            # SURPRISE (Gate-B, D2): the spiking expectation-violation read for an asserted fact with a stored
            # expectation (surprise_hz/threshold/surprised); null when no stored expectation or disabled (BRAIN_SURPRISE=0).
            "surprise": surprise_info,
            # WORLD-MODEL (Gate-B, E2): the spiking affective forward-model read for this turn (kind=update with the
            # predicted next-turn valence, or kind=violation with the fired surprise); null when disabled/neutral.
            "worldmodel": worldmodel_info,
            # MULTI-REFERENT WM (Gate-B, D6): the spiking multi-register HOLD read on a >=2-referent MAINTAIN turn
            # (recovered/hold_alive_min/all_recovered); null out of scope (<2 referents) or disabled (BRAIN_MULTIREF=0).
            "multiref": multiref_info,
            # NON-CONTRADICTION (Gate-B, B3): the spiking polarity-recall read for an in-scope assertion (svo/
            # asserted/stored polarity/reject); null out of scope or disabled (BRAIN_NONCONTRADICTION_GATE=0).
            "noncontradiction": noncontradiction_info,
            # RECONSOLIDATION (Gate-B, F): the belief-revision decision (action=rewrite/restabilize/abstain/lesioned);
            # null when no stored contradicting expectation or disabled (BRAIN_RECONSOLIDATION=0).
            "reconsolidation": reconsolidation_info,
            # PRAGMATIC (Task-#12, D): the W4 graded scalar-implicature listener-belief for a scalar-quantity turn
            # (belief over {none,SBNA,all} + enriched reading); null out of scope or disabled (BRAIN_PRAGMATIC=0).
            "pragmatic": pragmatic_info,
            # EPISODIC (Gate-B, D5): null on a non-referential turn (Hook A short-circuits referential turns above).
            "episodic": episodic_info,
        }
        if is_hyp:
            # additive markers (present ONLY on a generated-hypothesis turn): the guess flag, the (a,v,p) the fluent
            # prose asserts (VERIFY re-parse target), and whether the fluent mouth (True) or the raw template
            # fallback (False) produced the surface. A non-hypothesis rich turn is byte-identical (no extra keys).
            resp["hypothesis"] = True
            resp["hypothesis_svo"] = list(r.get("hypothesis_svo") or [])
            resp["fluent_hypothesis"] = bool(r.get("fluent_hypothesis"))
            # ── BEGIN faculty: DR-3 self-schema AUTHORSHIP (self-vs-heard) — additive, DEFAULT-ON since the wave-1/2
            #    flip (BRAIN_SELF_SCHEMA; corrected — previously said "DEFAULT-OFF", stale since _SELF_SCHEMA_
            #    DEFAULT_ON=True landed) ──
            # BACK the host 'guess, not something I was taught' flag with a genuinely-SPIKING neural authorship read:
            # a generated HYPOTHESIS is a VOLUNTEERED proposition (self-authored), so the DR-3 self_schema `author`
            # sub-block is driven 'self' and fires; its readback decides whether to PREPEND an honest own-guess MARKER.
            # LOAD-BEARING: the marker rides the LIVE author-pool read — under BRAIN_SELF_SCHEMA_LESION the author
            # access is severed (schema_access=False), the pool goes silent, the read collapses to 'heard', and the
            # marker VANISHES -> the reply reverts to the host default (while the recalled/content fields stay
            # byte-identical). Reuse-by-import of the 6-seed GO de-risk (authorship acc 1.000; self-lesion collapses
            # author to chance 6/6). DEFAULT-ON (corrected — previously said "DEFAULT-OFF: BRAIN_SELF_SCHEMA unset ->
            # no `authorship` key", stale): BRAIN_SELF_SCHEMA unset -> _SELF_SCHEMA_DEFAULT_ON=True -> the organ runs;
            # BRAIN_SELF_SCHEMA=0 is the byte-identical escape (no `authorship` key, no marker). Guarded so it never
            # crashes a turn. See research/runners/self_schema_production_organ.py.
            try:
                import research.runners.self_schema_production_organ as _SS
                if _SS.self_schema_enabled():
                    _ss_read = _get_self_schema_organ().read_author(
                        authored=True, lesion=_SS.self_schema_lesioned())
                    resp["authorship"] = _ss_read
                    if _ss_read.get("is_self"):
                        resp["answer"] = _SS.authorship_marker() + resp["answer"]
            except Exception as _sse:   # never let the authorship read crash a turn — degrade to the un-marked guess
                resp["authorship"] = {"on": True, "error": f"{type(_sse).__name__}: {_sse}"}
            # ── END faculty: DR-3 self-schema AUTHORSHIP ──
            # ── BEGIN faculty: R4 self_schema->source_provenance LEARNED CROSS-EDGE — additive, DEFAULT-OFF
            #    (BRAIN_ONEBRAIN_XEDGE_SELFSCHEMA). Mirrors the PART-1 d6-WM->comprehension frozen cross-edge wire-in
            #    (research/runners/onebrain_xedge_production.py, 2026-08-27-onebrain-xedge-production-frozen-GO.md)
            #    on the R4 pairing (2026-08-27-onebrain-integration-selfschema-provenance-learned-crossedge-GO.md):
            #    self_schema's authorship axis ('did I author this thought') is a LEARNED Hebbian cross-synapse onto
            #    source_provenance's 'reads as internally-generated' pool, grown once (0.05 -> ~3) on a SHARED merge
            #    pool, then frozen (no weight moves during any live turn). This turn's OWN live authorship verdict
            #    (`_ss_read`/`resp["authorship"]`'s `is_self`, computed above) drives whether the cross-edge's
            #    presynaptic `author` pool is held during a co-temporal read of R4's own validated ambiguous-item
            #    provenance instrument (`amb_read`, reused verbatim — not reimplemented). Attaches an ADDITIVE
            #    diagnostic field ONLY (`resp["authorship"]["source_provenance_crossedge"]`); NEVER touches
            #    `resp["answer"]` or any existing `authorship` field, and runs on an INDEPENDENT R4Pool instance —
            #    ZERO risk to the already-default-ON self_schema authorship marker above. LOAD-BEARING: the shift
            #    toward GENERATED vanishes under BRAIN_ONEBRAIN_XEDGE_SELFSCHEMA_LESION=1 (the cross-edge zeroed).
            #    DEFAULT-OFF: BRAIN_ONEBRAIN_XEDGE_SELFSCHEMA unset -> no extra key, byte-identical. Guarded so it
            #    never crashes a turn. See research/runners/onebrain_xedge_selfschema_production.py.
            try:
                from research.runners.onebrain_xedge_selfschema_production import (
                    xedge_selfschema_enabled, get_xedge_selfschema_pool, crossedge_provenance_shift)
                if xedge_selfschema_enabled():
                    _xsp_pool = get_xedge_selfschema_pool(42)
                    _xsp_is_self = bool((resp.get("authorship") or {}).get("is_self", False))
                    _xsp_read = crossedge_provenance_shift(_xsp_pool, _xsp_is_self)
                    if _xsp_read is not None:
                        resp.setdefault("authorship", {})["source_provenance_crossedge"] = _xsp_read
            except Exception as _xspe:   # never let the diagnostic read crash a turn
                resp.setdefault("authorship", {})["source_provenance_crossedge"] = {
                    "on": True, "error": f"{type(_xspe).__name__}: {_xspe}"}
            # ── END faculty: R4 self_schema->source_provenance LEARNED CROSS-EDGE ──
        # METACOG (Gate-B, E1): qualify a low-confidence RECALL answer with an honest functional hedge (skip an
        # abstain or a flagged guess — no recalled answer to qualify). Additive; null when disabled/out-of-scope.
        # When the confidence-forthcoming cap (board #94) already ran the SAME read this turn, reuse it here
        # instead of running the organ's spiking margin read a second time (the cap decision + the hedge are
        # driven by ONE evidence read, not two).
        if _cf_mc_info is not None:
            _mc_prefix, resp["metacog"] = _cf_mc_prefix, _cf_mc_info
        else:
            _mc_prefix, resp["metacog"] = _metacog_qualify(resp.get("activity"), bool(r["abstained"]) or is_hyp)
        if _mc_prefix:
            resp["answer"] = _mc_prefix + resp["answer"]
        # CONFIDENCE CAPS FORTHCOMINGNESS (board #94): attach the additive trace ONLY when the coupling actually
        # ran this turn (on + in-scope) -> byte-identical (no key) when disabled or out of scope.
        if _cf_trace is not None:
            resp["confidence_forthcoming"] = _cf_trace
        # CURIOSITY (Gate-B, D3): on an ABSTAIN, APPEND an honest follow-up QUESTION when the spiking ASK pool
        # craves (crave, don't refuse). A hypothesis is a guess (not an abstain) -> out of scope. Additive; the
        # suffix is a QUESTION (moat-safe). null when disabled / not an abstain / not craving.
        _cu_suffix, resp["curiosity"] = _curiosity_followup(bool(r["abstained"]) and not is_hyp)
        if _cu_suffix:
            resp["answer"] = resp["answer"] + _cu_suffix
        # PROSPECTIVE MEMORY (Gate-B): the held-intention cue-monitor read for this turn (kind=monitor with the
        # spiking rel/held + fired), attached ONLY when the organ was active this turn (an intention held) — a turn
        # with no held intention / BRAIN_PMEM=0 adds NO key (byte-identical). The reminder is already prepended above.
        if prospective_info is not None:
            resp["prospective"] = prospective_info
        if pmem_prefix:  # the reminder was DELIVERED on this main path -> consume the intention (fires once)
            _pd = _SESSION_PMEM.get(cache_key)
            if _pd is not None:
                _pd.clear()
        # GNW BUS observability (rich path): the rich composer's DIRECT recall runs through the bus-authored
        # `chat.gate` too, so attach the last bus decision when BRAIN_GNW_BUS is on. Default-OFF -> no key (byte-
        # identical). The bus AUTHORS the combination regardless of this flag (the flag only gates the debug block).
        if os.environ.get("BRAIN_GNW_BUS", "").strip().lower() in ("1", "true", "on", "yes"):
            _rich_bus = getattr(chat, "_last_gnw_bus", None)
            if _rich_bus is not None:
                resp["gnw_bus"] = _rich_bus
        # GNW THOUGHT-SWAP observability (rich path): attach the per-turn held-topic swap read the observe block stashed
        # (`chat._last_gnw_swap`: held topic, whether this turn swapped, the spiking read of the post-swap workspace),
        # ONLY when BRAIN_GNW_SWAP is on. ADDITIVE + DEFAULT-OFF -> with the flag off there is no key -> byte-identical.
        if _gnw_swap_flag_on():
            _swap_info = getattr(chat, "_last_gnw_swap", None)
            if _swap_info is not None:
                resp["gnw_swap"] = _swap_info
        # AFFECT DRIVES THE RESPONSE (board #84): prepend the graded affective EXPRESSION lead OUTERMOST (spoken
        # first, as tone) + attach the additive `affect_drives` trace. Empty lead / no key when disabled or neutral
        # -> byte-identical. The content fields above are unchanged (affect colors the surface, never a fact).
        if affect_drives_lead:
            resp["answer"] = affect_drives_lead + resp["answer"]
        if affect_drives_info is not None:
            resp["affect_drives"] = affect_drives_info
        # >>> W5 AFFECTIVE ToM BEGIN (rich path; additive, mergeable block) ─────────────────────────────────────
        # AFFECTIVE ToM DRIVES THE RESPONSE (W5): prepend the empathic EXPRESSION lead OUTERMOST (the emotional
        # acknowledgment of the OTHER agent's situation, spoken first) + attach the additive `affective_tom` trace.
        # Empty lead / no key when disabled, no other-agent, no affective word, or lesion-collapsed -> byte-identical.
        # The content fields above are unchanged (empathy colors the surface, never a fact); the lead VANISHES under
        # the OTHER-region `affect_out` lesion (the load-bearing proof).
        if tom_lead:
            resp["answer"] = tom_lead + resp["answer"]
        if tom_info is not None and tom_info.get("acted"):   # attach ONLY on a genuine trigger (organ ran the neural
            resp["affective_tom"] = tom_info                 # read) -> an ORDINARY turn stays byte-identical (no key)
        # <<< W5 AFFECTIVE ToM END ─────────────────────────────────────────────────────────────────────────────
        # SWAP DRIVES THE RESPONSE (board #85): prepend the topic-transition lead OUTERMOST (the discourse shift is
        # announced first, ahead of the tone) + attach the additive `swap_drives` trace. Empty lead / no key when
        # disabled or on a hold -> byte-identical. The content fields above are unchanged (the swap frames the reply,
        # never a fact); the lead VANISHES under the neural mismatch-detector lesion (the load-bearing proof).
        if swap_drives_lead:
            resp["answer"] = swap_drives_lead + resp["answer"]
        if swap_drives_info is not None:
            resp["swap_drives"] = swap_drives_info
        # COMMON GROUND DRIVES THE RESPONSE (2026-08-26): prepend the reduced-reference lead (a grounded referent is
        # pronominalized) + attach the additive `common_ground_drives` trace. Empty lead / no key when disabled or on a
        # first mention -> byte-identical. The content fields above are unchanged (the ledger frames HOW the reply
        # refers, never a fact); the lead VANISHES under the neural ledger-recurrence lesion (the load-bearing proof).
        if cg_drives_lead:
            resp["answer"] = cg_drives_lead + resp["answer"]
        if cg_drives_info is not None:
            resp["common_ground_drives"] = cg_drives_info
        # THE WANDERED THOUGHT DRIVES THE RESPONSE (board #86): prepend the idle-wander lead OUTERMOST (what the
        # brain was just thinking about, announced first) + attach the additive `wander_drives` trace. Empty lead /
        # no key when disabled or no recent wander -> byte-identical.
        if wander_drives_lead:
            resp["answer"] = wander_drives_lead + resp["answer"]
        if wander_drives_info is not None:
            resp["wander_drives"] = wander_drives_info
        # A NOVEL IDEA DRIVES THE RESPONSE (continuous ideation, default-OFF): prepend the idea lead OUTERMOST +
        # attach the additive `ideation_drives` trace. Empty lead / no key when disabled or no recent ideation ->
        # byte-identical. FLAGGED as an idea (not a fact); the content fields above are unchanged.
        if ideation_drives_lead:
            resp["answer"] = ideation_drives_lead + resp["answer"]
        if ideation_drives_info is not None:
            resp["ideation_drives"] = ideation_drives_info
        # DA-MODE DRIVES THE RESPONSE (board #79): APPEND the graded engagement SUFFIX (spoken last — the brain's
        # forthcomingness) + attach the additive `da_drives` trace. Empty suffix / no key when disabled or in
        # rest/neutral -> byte-identical. The content fields above are unchanged (the mode colors how forthcoming
        # the reply is, never a fact); the suffix VANISHES under the neural SNc-nucleus lesion (the load-bearing proof).
        if da_drives_suffix:
            resp["answer"] = resp["answer"] + da_drives_suffix
        if da_drives_info is not None:
            resp["da_drives"] = da_drives_info
        if da_encoding_info is not None:   # additive trace; absent when BRAIN_DA_ENCODING off -> byte-identical
            resp["da_encoding"] = da_encoding_info
        # >>> GNW GLOBAL-STOP BEGIN (rich path; additive, mergeable block — BRAIN_GNW_STOP, default-ON 2026-08-26) ───────
        # GLOBAL-WORKSPACE STOP DRIVES THE RESPONSE (distributed-overwrite clear-all): prepend the clearing lead
        # OUTERMOST (the held coalition was cleared to n_ignited=0 before the newcomer ignited -> a clean single-content
        # workspace, spoken first) + attach the additive `gnw_stop` trace. Runs AFTER the answer is composed so the
        # gnw-deliberation / #85-swap per-turn reads exist (the interrupt trigger). Empty lead / no key when disabled,
        # no interrupt, or no clean clear -> byte-identical. The content fields are unchanged (the stop frames the
        # opening, never a fact); the lead VANISHES under the shared-resource-depression lesion (the load-bearing
        # proof). Reuse-by-import (NO sim/ edit); the RNG is isolated. See webapp/gnw_global_stop.py.
        if _gnw_stop_flag_on():
            try:
                from webapp import gnw_global_stop as _gnw_stop_mod
                _gnw_stop_info = _gnw_stop_mod.observe_turn(chat, msg)
            except Exception as _gse:   # never let the stop coupling crash a turn -> degrade to the un-led answer
                _gnw_stop_info = {"on": True, "acted": False, "error": f"{type(_gse).__name__}: {_gse}", "lead": ""}
            if _gnw_stop_info is not None:
                _gnw_stop_lead = str(_gnw_stop_info.get("lead", "") or "")
                if _gnw_stop_lead:
                    resp["answer"] = _gnw_stop_lead + resp["answer"]
                if _gnw_stop_info.get("acted"):
                    resp["gnw_stop"] = _gnw_stop_info
        # <<< GNW GLOBAL-STOP END ──────────────────────────────────────────────────────────────────────────────────
        return _safe_json_response(resp, "rich")

    # ── single-fact path (rich=False): GATE -> CONSTRAIN+VERIFY render ──
    # Peek the GATE so we can report the recalled fact (exactly what the TUI
    # smoke records), then render. gate() returns None on the moat.
    try:
        # COMPOSITIONAL CHAIN ROUTE (reasoning-frontier, 2026-08-25): checked BEFORE chat.gate(msg) for the same
        # reason the rich path does (RichAnswerComposer._direct_fact) -- chat.gate is monkeypatched by the
        # (not-yet-chain-aware) GNW ignition-bus installers, so the check must run ahead of it, not as a fallback
        # on its abstain. A non-compositional question's regex simply does not match -> byte-identical fall-
        # through to chat.gate(msg). See research/runners/compositional_chain_route.py.
        gate_svo = None
        _is_chain_route = False
        # board #140 / BRAIN_SOURCE_MONITORING_FRAMES_HONESTY (2026-09-01): the pre-`frame_derived_answer` CORE
        # text, kept ONLY so the #129-organ-driven alt-framing below (webapp/source_monitoring_honesty_chat.py)
        # can wrap the un-annotated composed answer instead of double-wrapping an already-framed one. None on
        # every branch that never reaches the chain-route arm -- byte-identical unless that new flag is on.
        _chain_raw_answer = None
        try:
            from research.runners.compositional_chain_route import resolve_compositional_chain
            gate_svo = resolve_compositional_chain(chat.inner.composer, msg)
            _is_chain_route = gate_svo is not None
        except Exception:
            gate_svo = None
            _is_chain_route = False
        if gate_svo is None:
            gate_svo = chat.gate(msg)
        if gate_svo is None:
            answer, abstained, verified = "I don't know about that.", True, False
        else:
            answer = chat.render(gate_svo)
            abstained = False
            # 'verified' = the render did NOT fall back to the raw triple
            # (ChatBrain marks an unverified render with this suffix).
            verified = "[unverified render" not in answer
            if _is_chain_route:
                # DERIVED-ANSWER framing (reasoning-frontier hardening, moat-hardening audit req #4): frame a
                # composed multi-hop inference as the brain's OWN inference, surfacing the supporting hop-facts
                # -- UNCONDITIONALLY, not gated behind the optional #129 provenance monitor below (which is
                # default-OFF and must not be the only thing standing between a derived answer and being
                # presented as a plain perceived fact).
                from research.runners.compositional_chain_route import frame_derived_answer
                _chain_raw_answer = answer
                answer = frame_derived_answer(answer, getattr(gate_svo, "derived_from", None))
    except Exception as e:
        raise HTTPException(500, f"chat turn failed: {type(e).__name__}: {e}")

    # SOURCE-PROVENANCE HONESTY (board #129, 2026-08-25): read the #129 spiking opponent-comparator provenance
    # monitor on the just-recalled fact. gate() usually returns a DIRECTLY-STORED fact, presented to the monitor
    # as PERCEIVED; the monitor's OWN LIVE JUDGED label — not this claim — decides the framing (a lesioned
    # monitor demonstrably loses the ability to keep the reply reading as confidently perceived; see
    # research/runners/source_provenance_honesty.py + tests/test_source_provenance_honesty_wirein.py). The ONE
    # EXCEPTION (reasoning-frontier hardening, moat-hardening audit req #4): a chain-route answer is a COMPOSED
    # multi-hop inference, not a directly-recalled fact — presenting it as PERCEIVED would be a provenance lie.
    # It is encoded as GENERATED instead, so the monitor's own judged label (when it fires) agrees; the text
    # framing itself already happened UNCONDITIONALLY above, so we do NOT also call `provenance_framed_text`
    # here (that would double-wrap an already-framed derived answer in a second, generic disclaimer). Applied
    # to the CORE rendered text, before any other faculty's prefix/suffix accretion below, so it composes
    # cleanly with them. Additive + DEFAULT-OFF (`BRAIN_SOURCE_PROVENANCE_HONESTY` unset -> byte-identical: the
    # organ is never built, no substrate step runs, no `provenance` key is added).
    provenance_info = None
    if gate_svo is not None:
        try:
            import research.runners.source_provenance_production_organ as _SP
            if _SP.source_provenance_enabled():
                from research.runners.source_provenance_honesty import (
                    PROVENANCE_PERCEIVED, PROVENANCE_GENERATED, provenance_framed_text,
                )
                _sp_mon = _get_source_provenance_organ()
                _sp_key = ("live_chat_known_fact",) + tuple(gate_svo)
                _sp_mon.encode_fact(_sp_key, PROVENANCE_GENERATED if _is_chain_route else PROVENANCE_PERCEIVED)
                provenance_info = _sp_mon.judge_fact(_sp_key)
                if not _is_chain_route:
                    answer = provenance_framed_text("what_does", answer, provenance_info["label"])
                else:
                    # ── BEGIN faculty: board #140 rung — SOURCE-MONITORING DRIVES HONESTY FRAMING, additive,
                    #    DEFAULT-OFF (BRAIN_SOURCE_MONITORING_FRAMES_HONESTY; webapp/source_monitoring_honesty_
                    #    chat.py). Closes the "GENERATED half has no live HTTP exposure" gap this same block's
                    #    own comment names above: let the #129 organ's OWN live readback for THIS fact — not the
                    #    `_is_chain_route` host flag — decide whether the derived answer is ALSO offered the
                    #    monitor's substrate-driven hedge ("I believe ..., but I reasoned that myself rather than
                    #    being told it directly.") in place of `frame_derived_answer`'s host-generic one. Fires
                    #    ONLY when the readback AGREES this reads GENERATED — an ambiguous tie, or a LESIONED
                    #    monitor (BRAIN_SOURCE_PROVENANCE_HONESTY_LESION=1, the de-risk's own verified failing-
                    #    direction anti-cheat) collapsing the discrimination toward chance, leaves
                    #    `frame_derived_answer`'s wording UNTOUCHED — so audit req #4's guarantee ("never the
                    #    ONLY thing standing between a derived answer and a plain-perceived presentation") holds
                    #    in every reachable state; this flag only ever SWAPS which honest hedge wording is used,
                    #    it can never remove the hedge or manufacture an unhedged assertion. LOAD-BEARING: the
                    #    swap rate collapses under the SAME lesion that collapses the #129 organ's own
                    #    discrimination — proof the wording rides the LEARNED opponent-comparator trace, not a
                    #    host if/else. Default-OFF -> this branch never touches `answer` (byte-identical to the
                    #    pre-existing `frame_derived_answer`-only behavior). See
                    #    research/runners/_source_monitoring_honesty_flip_verify.py.
                    try:
                        from webapp.source_monitoring_honesty_chat import (
                            source_monitoring_frames_honesty_enabled as _smh_enabled,
                        )
                        if (_smh_enabled() and _chain_raw_answer is not None
                                and provenance_info.get("label") == PROVENANCE_GENERATED):
                            answer = provenance_framed_text("what_does", _chain_raw_answer, provenance_info["label"])
                    except Exception as _smhe:   # never let this opt-in swap crash a turn — keep the moat framing
                        provenance_info["source_monitoring_frames_honesty_error"] = (
                            f"{type(_smhe).__name__}: {_smhe}")
                    # ── END faculty: board #140 rung ──
        except Exception as _spe:   # never let an opt-in honesty read crash a turn
            provenance_info = {"on": True, "error": f"{type(_spe).__name__}: {_spe}"}

    # EPISODIC STORE (Gate-B, D5, Hook B — single-fact path): mirror the rich-path write (gated behind cupy; on a
    # numpy deployment the write is DEFERRED). Write-only; changes NO reply text. Guarded so it never crashes a
    # turn. A chain-route (ChainedSVO) answer is EXCLUDED (moat-hardening audit req #5): note_topic marks a
    # topic as directly recalled this turn, which a synthesized chain terminal is not (each hop-fact was already
    # noted, if at all, on the turn that taught it).
    if _episodic_on and gate_svo is not None and not _is_chain_route and _episodic_store_ok():
        try:
            _ep_topics = getattr(chat, "agents_set", None) or _brain_vocab(chat)
            _EP.get_episodic_organ(cache_key, 42, _ep_topics).note_topic(gate_svo[0])
        except Exception:
            pass

    # WORLD-MODEL (Gate-B, E2) + SURPRISE (Gate-B, D2) + RECONSOLIDATION (Gate-B, F): PREPEND the honest affect-
    # trajectory-violation notice (world-model), then the expectation-violation notice (surprise), then the
    # belief-revision notice (reconsolidation). Additive; empty prefixes when not firing.
    if pmem_prefix or worldmodel_prefix or surprise_prefix or reconsolidation_prefix or pragmatic_prefix:
        answer = (pmem_prefix + worldmodel_prefix + surprise_prefix + reconsolidation_prefix
                  + pragmatic_prefix + answer)

    # METACOG (Gate-B, E1): read the spiking confidence of this recall answer; a LOW-confidence answer gets an
    # honest functional hedge PREPENDED (skip an abstain — no answer to qualify). Additive; null when disabled.
    _sf_activity = _read_activity()
    _mc_prefix, metacog_info = _metacog_qualify(_sf_activity, abstained)
    if _mc_prefix:
        answer = _mc_prefix + answer

    # CURIOSITY (Gate-B, D3): on an ABSTAIN, APPEND an honest follow-up QUESTION when the spiking ASK pool craves
    # (crave, don't refuse). Additive; the suffix is a QUESTION (moat-safe). null when disabled / not an abstain.
    _cu_suffix, curiosity_info = _curiosity_followup(abstained)
    if _cu_suffix:
        answer = answer + _cu_suffix

    # GNW N-ORGAN IGNITION BUS — observability block (T1-1 Phase-B, 2026-08-13): the substrate ignition bus now
    # AUTHORS this turn's organ-combination by DEFAULT (installed above; `gate_svo` above IS the bus's committed
    # decision unless BRAIN_GNW_BUS_HOST reverted it). When BRAIN_GNW_BUS is on, attach the per-turn bus info the
    # wrapper stashed (`chat._last_gnw_bus`: committed patient, organ reads, host-vs-bus agreement, routable/reason).
    # ADDITIVE + DEFAULT-OFF: with the flag OFF the response carries NO `gnw_bus` key -> byte-identical. Guarded.
    gnw_bus_info = None
    if os.environ.get("BRAIN_GNW_BUS", "").strip().lower() in ("1", "true", "on", "yes"):
        gnw_bus_info = getattr(chat, "_last_gnw_bus", None)

    _resp = {
        "answer": answer,
        "abstained": abstained,
        # a chain-route answer (moat-hardening audit req #5) reports null here — it is a COMPOSED multi-hop
        # inference, not a directly-recalled fact; see `derived`/`derived_from` for its distinct API shape.
        "recalled_svo": (None if _is_chain_route else (list(gate_svo) if gate_svo is not None else None)),
        # DERIVED (reasoning-frontier hardening, moat-hardening audit #4/#5): True when `gate_svo` came from the
        # compositional chain route (a composed multi-hop inference) rather than a direct `chat.gate` recall;
        # `derived_from` names the two independently-verified hop-facts it was reasoned from.
        "derived": bool(_is_chain_route),
        "derived_from": (list(getattr(gate_svo, "derived_from", None) or []) if _is_chain_route else None),
        "verified": verified,
        "renderer": rname,
        "brain": req.brain,
        "source": source,
        "rich": False,
        # B3: what the brain DID this turn -- the decoded role chips + match confidence, which engram block answered
        # (or null -> abstained), and a scalar RF firing/|Z| gauge. Read-only of the spiking recall the gate already
        # ran (composer.query_patient); null for the rate composer / when the matcher abstained before any query.
        "activity": _sf_activity,
        # SOURCE-PROVENANCE (board #129): the #129 opponent-comparator's judged label for the recalled fact
        # ({"known","label","d",...}), or null when the faculty is off / the turn abstained. See above.
        "provenance": provenance_info,
        # AFFECT (Gate-B): the live mood that colored this turn's prose manner (debug trace; single-fact path has
        # no forthcomingness lever, so only manner applies here). null when affect is disabled (BRAIN_AFFECT=0).
        "affect": affect_info,
        # COMPREHENSION (Gate-B, D4): the spiking role-binding read for an in-scope transitive; null out of scope.
        "comprehension": comprehension_info,
        # SURPRISE (Gate-B, D2): the spiking expectation-violation read for an asserted fact with a stored
        # expectation; null when no stored expectation or disabled (BRAIN_SURPRISE=0).
        "surprise": surprise_info,
        # METACOG (Gate-B, E1): the spiking balance-of-evidence confidence read for this recall answer
        # (balance/threshold/confident/mean_role_conf); null when abstained / disabled (BRAIN_METACOG=0).
        "metacog": metacog_info,
        # WORLD-MODEL (Gate-B, E2): the spiking affective forward-model read for this turn (kind=update with the
        # predicted next-turn valence, or kind=violation with the fired surprise); null when disabled/neutral.
        "worldmodel": worldmodel_info,
        # CURIOSITY (Gate-B, D3): the spiking ASK-pool crave read on an abstain (want_hz/threshold/curious/topic);
        # null when disabled or not an abstain. A follow-up QUESTION is appended to the answer when curious.
        "curiosity": curiosity_info,
        # MULTI-REFERENT WM (Gate-B, D6): the spiking multi-register HOLD read on a >=2-referent MAINTAIN turn; null
        # out of scope or disabled (BRAIN_MULTIREF=0).
        "multiref": multiref_info,
        # NON-CONTRADICTION (Gate-B, B3): the spiking polarity-recall read for an in-scope assertion; null out of
        # scope or disabled (BRAIN_NONCONTRADICTION_GATE=0). An accepted assertion attaches its read here.
        "noncontradiction": noncontradiction_info,
        # RECONSOLIDATION (Gate-B, F): the belief-revision decision for a contradicting assertion; null when no
        # stored contradicting expectation or disabled (BRAIN_RECONSOLIDATION=0).
        "reconsolidation": reconsolidation_info,
        # PRAGMATIC (Task-#12, D): the W4 graded scalar-implicature listener-belief for a scalar-quantity turn
        # (belief over {none,SBNA,all} + enriched reading); null out of scope or disabled (BRAIN_PRAGMATIC=0).
        "pragmatic": pragmatic_info,
        # EPISODIC (Gate-B, D5): null on a non-referential turn (Hook A short-circuits referential turns above).
        "episodic": episodic_info,
    }
    # GNW BUS shadow block: attached ONLY when BRAIN_GNW_BUS is on (so the flag-off response is byte-identical, with
    # no extra key). Carries the substrate's committed decision + host-vs-bus agreement for this turn.
    if gnw_bus_info is not None:
        _resp["gnw_bus"] = gnw_bus_info
    # PROSPECTIVE MEMORY (Gate-B): the held-intention cue-monitor read, attached ONLY when the organ was active this
    # turn (an intention held). No held intention / BRAIN_PMEM=0 -> no key -> byte-identical. Reminder prepended above.
    if prospective_info is not None:
        _resp["prospective"] = prospective_info
    if pmem_prefix:  # the reminder was DELIVERED on this main path -> consume the intention (fires once)
        _pd = _SESSION_PMEM.get(cache_key)
        if _pd is not None:
            _pd.clear()
    # GNW THOUGHT-SWAP observability (single-fact path): attach the per-turn held-topic swap read, ONLY when
    # BRAIN_GNW_SWAP is on. ADDITIVE + DEFAULT-OFF -> flag off there is no key -> byte-identical. See gnw_thought_swap.py.
    if _gnw_swap_flag_on():
        _swap_info = getattr(chat, "_last_gnw_swap", None)
        if _swap_info is not None:
            _resp["gnw_swap"] = _swap_info
    # AFFECT DRIVES THE RESPONSE (board #84, single-fact path): prepend the graded affective EXPRESSION lead
    # OUTERMOST + attach the additive `affect_drives` trace. Empty lead / no key when disabled or neutral ->
    # byte-identical. The content fields (abstained/recalled_svo/verified) are unchanged.
    if affect_drives_lead:
        _resp["answer"] = affect_drives_lead + _resp["answer"]
    if affect_drives_info is not None:
        _resp["affect_drives"] = affect_drives_info
    # >>> W5 AFFECTIVE ToM BEGIN (single-fact path; additive, mergeable block) ──────────────────────────────────
    # AFFECTIVE ToM DRIVES THE RESPONSE (W5, single-fact path): prepend the empathic EXPRESSION lead OUTERMOST +
    # attach the additive `affective_tom` trace. Empty lead / no key when disabled, no other-agent, no affective
    # word, or lesion-collapsed -> byte-identical. The content fields are unchanged (empathy colors the surface,
    # never a fact); the lead VANISHES under the OTHER-region `affect_out` lesion (the load-bearing proof).
    if tom_lead:
        _resp["answer"] = tom_lead + _resp["answer"]
    if tom_info is not None and tom_info.get("acted"):   # attach ONLY on a genuine trigger (organ ran the neural
        _resp["affective_tom"] = tom_info                 # read) -> an ORDINARY turn stays byte-identical (no key)
    # <<< W5 AFFECTIVE ToM END ─────────────────────────────────────────────────────────────────────────────────
    # SWAP DRIVES THE RESPONSE (board #85, single-fact path): prepend the topic-transition lead OUTERMOST + attach the
    # additive `swap_drives` trace. Empty lead / no key when disabled or on a hold -> byte-identical. The content
    # fields are unchanged (the swap frames the reply, never a fact); the lead VANISHES under the neural swap lesion.
    if swap_drives_lead:
        _resp["answer"] = swap_drives_lead + _resp["answer"]
    if swap_drives_info is not None:
        _resp["swap_drives"] = swap_drives_info
    # COMMON GROUND DRIVES THE RESPONSE (2026-08-26, single-fact path): prepend the reduced-reference lead (a grounded
    # referent is pronominalized) + attach the additive `common_ground_drives` trace. Empty lead / no key when disabled
    # or on a first mention -> byte-identical. The content fields are unchanged (the ledger frames HOW the reply refers,
    # never a fact); the lead VANISHES under the neural ledger-recurrence lesion (the load-bearing proof).
    if cg_drives_lead:
        _resp["answer"] = cg_drives_lead + _resp["answer"]
    if cg_drives_info is not None:
        _resp["common_ground_drives"] = cg_drives_info
    # THE WANDERED THOUGHT DRIVES THE RESPONSE (board #86, single-fact path): prepend the idle-wander lead
    # OUTERMOST + attach the additive `wander_drives` trace. Empty lead / no key when disabled or no recent
    # wander -> byte-identical.
    if wander_drives_lead:
        _resp["answer"] = wander_drives_lead + _resp["answer"]
    if wander_drives_info is not None:
        _resp["wander_drives"] = wander_drives_info
    # A NOVEL IDEA DRIVES THE RESPONSE (continuous ideation, default-OFF, single-fact path): prepend the idea lead
    # OUTERMOST + attach the additive `ideation_drives` trace. Empty lead / no key when disabled or no recent
    # ideation -> byte-identical. FLAGGED as an idea (not a fact); the content fields are unchanged.
    if ideation_drives_lead:
        _resp["answer"] = ideation_drives_lead + _resp["answer"]
    if ideation_drives_info is not None:
        _resp["ideation_drives"] = ideation_drives_info
    # DA-MODE DRIVES THE RESPONSE (board #79, single-fact path): APPEND the graded engagement SUFFIX + attach the
    # additive `da_drives` trace. Empty suffix / no key when disabled or in rest/neutral -> byte-identical. The
    # content fields are unchanged (the mode colors forthcomingness, never a fact); the suffix VANISHES under the
    # neural SNc-nucleus lesion (the load-bearing proof).
    if da_drives_suffix:
        _resp["answer"] = _resp["answer"] + da_drives_suffix
    if da_drives_info is not None:
        _resp["da_drives"] = da_drives_info
    if da_encoding_info is not None:   # additive trace; absent when BRAIN_DA_ENCODING off -> byte-identical
        _resp["da_encoding"] = da_encoding_info
    # >>> GNW GLOBAL-STOP BEGIN (single-fact path; additive, mergeable block — BRAIN_GNW_STOP, default-ON 2026-08-26) ──
    # GLOBAL-WORKSPACE STOP DRIVES THE RESPONSE (distributed-overwrite clear-all, single-fact path): prepend the
    # clearing lead OUTERMOST (the held coalition was cleared to n_ignited=0 before the newcomer ignited) + attach the
    # additive `gnw_stop` trace. Runs AFTER the gate/render so the gnw-deliberation / #85-swap per-turn reads exist
    # (the interrupt trigger). Empty lead / no key when disabled, no interrupt, or no clean clear -> byte-identical. The
    # content fields are unchanged (the stop frames the opening, never a fact); the lead VANISHES under the
    # shared-resource-depression lesion (the load-bearing proof). Reuse-by-import (NO sim/ edit). See gnw_global_stop.py.
    if _gnw_stop_flag_on():
        try:
            from webapp import gnw_global_stop as _gnw_stop_mod
            _gnw_stop_info = _gnw_stop_mod.observe_turn(chat, msg)
        except Exception as _gse:   # never let the stop coupling crash a turn -> degrade to the un-led answer
            _gnw_stop_info = {"on": True, "acted": False, "error": f"{type(_gse).__name__}: {_gse}", "lead": ""}
        if _gnw_stop_info is not None:
            _gnw_stop_lead = str(_gnw_stop_info.get("lead", "") or "")
            if _gnw_stop_lead:
                _resp["answer"] = _gnw_stop_lead + _resp["answer"]
            if _gnw_stop_info.get("acted"):
                _resp["gnw_stop"] = _gnw_stop_info
    # <<< GNW GLOBAL-STOP END ──────────────────────────────────────────────────────────────────────────────────────
    return _safe_json_response(_resp, "single-fact")


# ── OpenAI-API-COMPATIBLE SHIM (2026-08-19 reframe: the two-surface UX) ────────────────────────────────────────
# Expose the sim brain over `/v1/chat/completions` + `/v1/models` so ANY standard LLM client (Open WebUI, LibreChat,
# …) talks to it with no custom UI. The REPLY rides `content` (the conversation surface); the brain's honest INTERNAL
# MONOLOGUE rides `reasoning_content` (the 'thinking' panel, shown alongside the reply by default). Host TRANSPORT
# only — it computes no cognition; it re-uses `brain_chat`'s existing spiking-faculty metadata. See webapp/openai_shim.py.
class OpenAIChatRequest(BaseModel):
    model: str = "sim-brain"
    messages: list[dict] = []
    stream: bool = False
    session: str | None = None      # optional: pin a conversation; else a stable per-model default


def _shim_brain_selection(model_field: str | None) -> tuple[str, str]:
    """Resolve WHICH brain + renderer the OpenAI shim serves for a request (2026-08-21: the shim now serves a
    DEVELOPED knowledge brain, not only 'tiny-demo').

    Brain precedence:
      1. the OpenAI `model` field, IF it explicitly names a known brain ('tiny-demo'/'self-knowledge') or a
         developed-brain bundle DIRECTORY (a dir with brain.json — absolute, or repo-relative);
      2. else the `BRAIN_CHAT_BUNDLE` env — the deployment's configured default brain (a bundle path);
      3. else 'tiny-demo' (the GPU-free built-in — byte-identical to the prior shim behavior).
    A generic id ('sim-brain'/'sim'/'default'/'gpt-*'/empty) means "use the configured default", so ANY stock
    OpenAI client works unchanged. `BRAIN_LTM_BUNDLE` (the bulk cortical LONG-TERM knowledge store) is honored
    downstream in `_build_chat_brain` whenever the resolved brain is a developed bundle — so the 21k-fact LTM +
    the no-confab moat flow through automatically. Renderer follows `_default_brain_renderer()` (GPU-light, or the
    `BRAIN_CHAT_RENDERER` override)."""
    m = (model_field or "").strip()
    known = {"tiny-demo", "tiny", "demo", "self-knowledge", "self_knowledge", "self"}
    brain = None
    if m and m.lower() not in ("sim-brain", "sim", "default", "gpt-3.5-turbo", "gpt-4", "gpt-4o"):
        if m.lower() in known:
            brain = m
        else:
            # a bundle DIRECTORY path in the model field (absolute or repo-relative)
            try:
                from research.runners.developed_brain_io import is_developed_brain_bundle
                cand_rel = m if Path(m).is_absolute() else str(REPO_ROOT / m)
                if is_developed_brain_bundle(m):
                    brain = m
                elif is_developed_brain_bundle(cand_rel):
                    brain = cand_rel
            except Exception:
                brain = None
    if brain is None:
        brain = os.environ.get("BRAIN_CHAT_BUNDLE", "").strip() or "tiny-demo"
    return brain, _default_brain_renderer()


@app.get("/v1/models")
def openai_models() -> JSONResponse:
    """OpenAI-compatible model list (clients query this to select the sim brain)."""
    from webapp import openai_shim as _OAS
    return JSONResponse(_OAS.models_list())


@app.post("/v1/chat/completions")
def openai_chat_completions(req: OpenAIChatRequest):
    """OpenAI-compatible chat: the brain's reply -> `content`, its honest internal monologue -> `reasoning_content`.
    Delegates the actual turn to the existing `brain_chat` handler (same spiking faculties, same moat), then marshals
    the response into the OpenAI shape. Supports streaming (SSE) since most clients default to it."""
    import json as _json
    import time as _t
    from webapp import openai_shim as _OAS
    from fastapi.responses import StreamingResponse
    user_msg = _OAS._last_user_message(req.messages)
    session = req.session or ("openai-%s" % (req.model or "sim-brain"))
    # SERVE THE DEVELOPED KNOWLEDGE BRAIN (+ the cortical LTM via BRAIN_LTM_BUNDLE downstream), not only 'tiny-demo'.
    brain_id, renderer = _shim_brain_selection(req.model)
    r = brain_chat(BrainChatRequest(session=session, message=user_msg, brain=brain_id, renderer=renderer))
    resp = _json.loads(bytes(r.body))
    reply = resp.get("answer") or resp.get("response") or ""
    # CONTINUOUS-STATE ENGINE: this session's between-turn inner-life (mood evolution while idle), same cache_key
    # brain_chat used = (session, brain, resolved renderer). Surfaced in the monologue's 'thinking' stream.
    _il = None
    try:
        from webapp import continuous_engine as _CE
        _il = _CE.inner_life((session, brain_id, renderer))
    except Exception:
        _il = None
    monologue = _OAS.format_internal_monologue(resp, inner_life=_il)
    created = int(_t.time())
    if req.stream:
        return StreamingResponse(_OAS.stream_chunks(reply, monologue, req.model, created),
                                 media_type="text/event-stream")
    return JSONResponse(_OAS.chat_completion_object(reply, monologue, req.model, created))


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
    _SESSION_PMEM.pop(cache_key, None)  # clear the held prospective intention + its latch bridge (Gate-B pmem)
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
