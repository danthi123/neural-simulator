# Web Dashboard (Phase 1)

A web-based research dashboard for the neural simulator, decoupled from the
existing DearPyGUI app. Phase 1 covers: browse completed runs, read findings,
launch new runners with live stdout tail. Phase 2 adds an agent-in-environment
viz; Phase 3 brings up a 3D world via PyBullet.

## Why a web frontend (not just an extended DearPyGUI)

- The DearPyGUI app focuses on low-level neuron parameter tuning. It doesn't
  expose the brain-region framework, neuromodulator subsystem, plasticity
  gates, perception arc, or research runners — all of which landed in
  2026-04 and have no UI yet.
- The web stack composes naturally with the planned virtual-world viz
  (Three.js for 2D, then Three.js + PyBullet for 3D) and is remote-deployable.
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

## Surface (Phase 1)

| Path | Purpose |
|---|---|
| `GET /` | Dashboard HTML |
| `GET /api/runs` | List `research/findings/raw/g11_bg/*.json` with sum-finalQ summary |
| `GET /api/runs/{name}` | Full run JSON (parsed) |
| `GET /api/findings` | List `research/findings/*.md` |
| `GET /api/findings/{name}` | Raw markdown body |
| `POST /api/runs/launch` | Launch a runner subprocess; returns `run_id` |
| `GET /api/runs/launch/{run_id}` | Poll status of a launched run |
| `WS  /ws/runs/{run_id}` | Stream stdout lines for a launched run |
| `GET /api/info` | Repo paths, presets, phase status |

### Presets

`POST /api/runs/launch` accepts a `preset` field. Available:

- `flagship` — current best (4 cheats closed, 4.08 / p=0.00045)
- `flagship_with_cheat5` — flagship + curriculum-staged BG cross-projections (closes cheat #5)
- `perception_only` — full perception arc, no sensed reward (4.56 / p=0.00819)
- `baseline` — minimal flags (5.88 reference)
- `smoke` — 100-step quick check

Custom flags can be added via `extra_args: ["--foo", "bar"]`.

## Architecture notes

- The server runs **out-of-process** from any active simulation. Launched
  runs are subprocesses with their own GPU contexts. This avoids GIL
  conflicts and means the dashboard is safe to restart while a run is
  in flight (the run keeps going; on next dashboard load you re-attach
  via `run_id`).
- Phase 2 will add an in-process bridge connection for live region
  activity. Until then, "live data" means tailing the runner's stdout.
- The frontend is plain HTML + ES modules. No build step. We can swap
  in React/Vite later if/when the surface gets larger than ~1K lines.
- All dynamic content from the filesystem (filenames, JSON values,
  markdown bodies) is rendered via `textContent` or escaped before
  injection. The markdown renderer escapes its input first so the only
  HTML in the output comes from this code, not from the source files.

## Phase 2 plan

- Embed the runner's `DataBus` channels via WebSocket so the dashboard
  shows live firing rates, region activity, neuromodulator concentrations,
  and plasticity gate states.
- Three.js 2D viz of the agent in its gridworld with beacon and landmark
  fields. Real-time agent position + sensor activations + cortex/BG firing.
- Add a "live mode" tab that streams data from a still-running episode.

## Phase 3 plan

- Replace gridworld with PyBullet 3D world. Agent has body + camera.
- Camera image becomes input to the V1-style sensory cortex (the missing
  piece that unlocks pixel-level vision and primitive multimodal learning).
- This is the path from "biology-grounded learning agent" → "embodied
  artificial-life creature".
