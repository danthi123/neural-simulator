# Web Dashboard

A web-based research dashboard for the neural simulator, decoupled from the
existing DearPyGUI app. Built incrementally — Phase 1 added the basic
research surface; Phase 2 added 2D world playback; Phase 2.5 added live
mode for in-flight runs; the UX pass added Overview/Experiments tabs,
toasts, keyboard shortcuts, comparison view, and persistent state.

Future Phase 3 will replace the gridworld with a 3D PyBullet world for
embodied multimodal experiments.

## Why a web frontend (not just an extended DearPyGUI)

- The DearPyGUI app focuses on low-level neuron parameter tuning. It doesn't
  expose the brain-region framework, neuromodulator subsystem, plasticity
  gates, perception arc, or research runners — all of which landed in
  2026-04 and have no native UI.
- The web stack composes naturally with the planned virtual-world viz
  (Three.js for 2D, then PyBullet for 3D) and is remote-deployable.
- Decoupling means you can launch a runner on a GPU box and watch from any
  laptop. The simulator code itself doesn't need to know about the dashboard.

## Run it

```bash
# From repo root
uvicorn webapp.server:app --reload --port 8765
```

Then open http://localhost:8765/.

If `fastapi`/`uvicorn` aren't installed:

```bash
pip install -r webapp/requirements.txt
```

## Tabs

### Overview (default landing tab)

- **5 KPI cards:** Best Run, Total Runs, Mean Sum, Findings count,
  In-flight Runs. Cards click through to the relevant tab.
- **Distribution histogram** of `sum_finalQ` across all real runs (smokes
  excluded). The bin containing the flagship (4.08) is highlighted green;
  the baseline (5.88) bin is highlighted yellow. Lets you see at a glance
  where the current best sits in the distribution.
- **Recent activity feed:** last 12 runs with category badges (cheat #5,
  perception arc, PFC, sleep replay, etc.) and relative timestamps.
- **Latest findings feed:** last 10 markdown docs.

### Runs

- 222+ runs sortable + filterable.
- Filters: hide-smokes (default ON), hide-incomplete, free-text search.
- Click a run for detail: phase stats table, distance + reward learning
  curves, action-distribution histogram, raw JSON summary.
- Shift-click runs to add to a comparison set (max 3); the **Compare N**
  button overlays distance + reward curves of all selected runs in one
  chart so you can visually contrast configurations.
- "Play in World viz" jumps to the 2D animated playback.
- Auto-refreshes every 10s while open so new runs appear without manual
  reload.

### Experiments

- Auto-groups all runs by filename suffix (`g11_seed42_v3lateral.json` →
  experiment "v3lateral").
- Per-experiment table: name, category pill (color-coded by research
  area), n_seeds, mean ± std, min/max, **delta vs flagship 4.08** (red =
  worse, green = better).
- Filters: hide-smokes (default ON), only-multi-seed (default ON; hides
  one-off runs).
- Click any row to expand the seed-by-seed breakdown.
- Reveals research-direction patterns at a glance — e.g., that "sleep",
  "recencyreplay", and "adaDA" all beat the flagship by small margins
  in 3-seed mean (which would be invisible browsing 222 individual runs).

### World

- 2D top-down playback of any saved run with controls: play/pause, speed
  (1× to 100×), scrubber, run picker.
- Animated agent (heading arrow), goal beacon (intensity-falloff halo),
  landmark marker, fading trajectory trail.
- Live overlay: agent pos, goal pos, distance, action, reward, phase.
- **Live mode:** click "Live mode" to attach to any in-flight runner via
  WebSocket. Server parses runner stdout for progress events and streams
  them; viz updates every ~100 sim steps. Includes a live `recent_dist`
  line chart that grows as the run progresses.

### Findings

- Browse + read all `research/findings/*.md` files with a markdown
  renderer (escapes input then injects only its own tags — no XSS path).

### Launch

- Form for launching new runners with a preset selector:
  - `flagship` — 4-cheats-closed, 4.08 sum (current best, recommended)
  - `flagship_with_cheat5` — flagship + curriculum-staged BG cross-projections
  - `perception_only` — full perception arc, no sensed reward
  - `baseline` — minimal flags
  - `smoke` — 100-step quick check
- Custom seed.
- **Custom grid_size + n_hippocampus_per_layer** fields for varied world
  sizes (the recipe is 8×8-tuned; 16×16 underperforms baseline per the
  scaling finding — needs empirical re-tuning, not just a flag).
- Free-text "extra args" pass-through.
- Live stdout tail via WebSocket once launched. Toast confirmation on
  launch success/failure.

### About

- Repo paths, available presets, current phase, surface plan.

## Keyboard shortcuts

| Key | Action |
|---|---|
| `1`–`6` | Jump to tab (Overview, Runs, Experiments, World, Findings, Launch) |
| `r` | Refresh current tab |
| `/` | Focus search box |
| `Esc` | Blur input / clear comparison set |
| `?` | Show shortcut help (toast) |

(Shortcuts are inactive while typing in an input/textarea.)

## API surface

| Path | Purpose |
|---|---|
| `GET /` | Dashboard HTML |
| `GET /api/runs` | List runs with sum_finalQ summary |
| `GET /api/runs/{name}` | Full run JSON (parsed) |
| `GET /api/findings` | List findings |
| `GET /api/findings/{name}` | Raw markdown body |
| `GET /api/experiments` | Auto-grouped per-experiment aggregates |
| `POST /api/runs/launch` | Launch a runner subprocess |
| `GET /api/runs/launch` | List active in-flight runs |
| `GET /api/runs/launch/{run_id}` | Poll status |
| `WS  /ws/runs/{run_id}` | Stream stdout + parsed progress events |
| `GET /api/info` | Repo paths, presets, phase status |

## Architecture

```
webapp/
  __init__.py
  server.py             # FastAPI app
  requirements.txt      # fastapi, uvicorn, pydantic
  README.md             # this file
  static/
    index.html          # markup
    style.css           # dark theme
    app.js              # main bootstrap, runs/findings/launcher tabs
    world.js            # World tab + live mode
    charts.js           # canvas line + bar charts (no deps)
    ui.js               # toasts, shortcuts, persistent state, helpers
```

Out-of-process design: launched runs are subprocesses with their own GPU
contexts. Server can be restarted while a run is in flight (the run keeps
going; on next dashboard load you re-attach via `run_id`).

Frontend is plain HTML + ES modules. No build step. We can swap in
React/Vite later if the surface gets larger than ~2K lines (currently
~1.6K lines across all JS files).

All dynamic content from the filesystem (filenames, JSON values, markdown
bodies) is rendered via `textContent` or escaped before injection. Path
traversal is blocked by routing + handler. Tested.

## Tests

```bash
python -m pytest tests/test_webapp_server.py -v
```

16 endpoint smoke tests covering: info, runs listing, findings listing,
path-traversal protection, launcher 404 paths, WebSocket-stream parsing,
the experiments endpoint, and filename → experiment detection.

## Persistent state

The dashboard remembers your last active tab and filter checkbox state
(via `localStorage`, key `neural-sim-dashboard-state-v1`). Clear via
DevTools → Application → Local Storage if needed.

## Maintenance

The dashboard's contract with the simulator (field names, print formats,
flag names, filename conventions) is undocumented and brittle. The
[`keep-webapp-current` skill](../.claude/skills/keep-webapp-current/SKILL.md)
is the explicit checklist for verifying the contract still holds.
The PostToolUse hook at `.claude/hooks/check_doc_drift.py` nudges
the skill when relevant runner/sim files change.

## Phase 3 plan

- Replace gridworld with PyBullet 3D world. Agent has body + camera.
- Camera image becomes input to the V1-style sensory cortex (the missing
  piece that unlocks pixel-level vision and primitive multimodal learning).
- This is the path from "biology-grounded learning agent" → "embodied
  artificial-life creature".
