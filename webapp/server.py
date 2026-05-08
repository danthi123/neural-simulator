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
    # Per-step action + reward (added 2026-04-29 for live-mode HUD).
    # action is the index 0-3 (NESW); -1 means "not parsed from this line"
    # (older runners or progress lines that predate the action= field).
    action: int = -1
    reward: float = 0.0


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
# Optional per-step action + reward fields, appended after recent_dist.
# Format: " action=N reward=+1.00" (action letter NESW, reward signed float).
# Both fields are optional for backward compat with older runner versions.
_PROGRESS_ACTION_RE = re.compile(r"action=([NESW?])\s+reward=([-+]?[\d.]+)")
_ACTION_LETTER_TO_IDX = {"N": 0, "E": 1, "S": 2, "W": 3}


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
    # long_tail). Smoke: 2/4 PASS, aggregate 0.62.
    "phase_1_5_unified": [
        "--benchmarks", "sequential_expansion", "retention_over_time",
        "interference", "long_tail",
        "--events-per-word", "200", "--n-eval-per-word", "25",
        "--silence-steps", "5000",
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
    # eval differs.
    "consolidation_synonym_medium_strict": [
        "--medium",
        "--strict-silence",
        "--n-test-per-word", "20",
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
    "phase_1_5_unified":      "research.runners.continual_eval_suite",
    "tier_2_3_phrases":       "research.runners.phrase_trainer",
    "chat_demo":              "research.runners.chat_demo",
    "chat_continual_demo":    "research.runners.chat_continual_demo",
    "chat_synonym_demo":      "research.runners.chat_synonym_demo",
    "consolidation_synonym":               "research.runners.consolidation_synonym_trainer",
    "consolidation_synonym_smoke":         "research.runners.consolidation_synonym_trainer",
    "consolidation_synonym_medium":        "research.runners.consolidation_synonym_trainer",
    "consolidation_synonym_medium_strict": "research.runners.consolidation_synonym_trainer",
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
    "phase_1_5_unified":      "--out-stats",
    "tier_2_3_phrases":       "--out-stats",
    "chat_demo":              "--out-stats",
    "chat_continual_demo":    "--out-stats",
    "chat_synonym_demo":      "--out-stats",
    "consolidation_synonym":               "--out-stats",
    "consolidation_synonym_smoke":         "--out-stats",
    "consolidation_synonym_medium":        "--out-stats",
    "consolidation_synonym_medium_strict": "--out-stats",
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
    """
    new_count = 0
    cutoff = time.time() - 86400.0  # 24h
    sidecars = sorted(
        RAW_RUNS_DIR.glob("*.cmd.json"),
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
    """Re-scan for orphan sidecars every 30s. Picks up runs spawned via
    raw `python -m research.runners.g11_bg_replicated_runner --emit-webapp-sidecar`
    (or any other process that drops a sidecar) without requiring a
    server restart."""
    async def _loop():
        while True:
            await asyncio.sleep(30)
            try:
                _scan_for_orphans()
            except Exception as e:
                print(f"[webapp] periodic orphan scan failed: {e}", flush=True)
    asyncio.create_task(_loop())


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
    for run in launched_runs.values():
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
        "out_path": run.out_path,
        "cmd": list(run.cmd),
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
        # Read last 8KB to find the latest progress line
        with log_path.open("rb") as f:
            f.seek(max(0, size - 8192))
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

        alive = _check_pid_alive(pid)
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
