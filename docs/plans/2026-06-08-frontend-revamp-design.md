---
type: plan
status: live
date: 2026-06-08
---

# Frontend Revamp — Run-Control + Live Brain-Activity Cockpit (Design)

**Date:** 2026-06-08
**Status:** Design (no code yet — READ-ONLY deliverable)
**Supersedes / absorbs:** `docs/plans/2026-05-02-webapp-3d-visualization-design.md`
(the 3D-viz design; that doc's anatomical layout + color families are kept and
extended here — the part it never delivered was *real* per-region firing data,
which is the centerpiece of this revamp)
**Audience:** the project owner (single-user dev dashboard) + future implementers.

---

## 0. Executive summary

**Proposed architecture (one paragraph).** Keep the existing FastAPI backend
(`webapp/server.py`) and its proven out-of-process run-launcher (subprocess +
sidecar + orphan recovery), but re-organize the frontend around **four
purpose-built screens** — *Launch* (start a run), *Runs* (monitor active + past),
*Brain* (the real-time activity centerpiece), and *Environment/I-O* (gridworld +
agent + sensory-in/motor-out / words-in-out) — and **delete the three
hand-maintained status surfaces** the owner doesn't want (the 333 KB
`capability_status.json` panel, the self-reported "project progress" Overview
KPIs, and the cumulative result-distribution dashboard). The single hard problem
is **streaming live brain state to the browser**: today the simulator runs on an
isolated thread inside a detached subprocess and exposes *nothing* about
per-region activity — only a coarse `[PROGRESS]` text line on stdout. The
recommended solution is to add one small **per-region activity aggregator inside
the bridge** (mean firing rate per region + per-pathway flux, computed on-GPU,
~30 floats per frame, NOT per-neuron), have the runner emit those as a compact
**`[ACTIVITY] {json}` line on the same stdout channel** at ~5–10 Hz, and have the
server tail-parse them and re-broadcast over the **existing WebSocket** as binary
or JSON frames. This reuses the entire launch/log/orphan plumbing that already
works, requires **zero new IPC and no GPU↔browser socket**, and stays honest
about the data-rate ceiling (we stream ~30 region rates, not 5,000 neurons).

**Recommended tech stack.** Backend: **keep FastAPI + uvicorn** (unchanged
plumbing, add 2 endpoints + 1 WS message type). Frontend: **keep vanilla ES
modules + the Three.js-via-CDN-importmap approach, no build step**, but
**refactor the 12 K-line monolith into a small shared core** (a typed
`SimSocket` client, a `Store`, a `TabRegistry`) — a *consolidation*, not a
framework rewrite. Rendering: **Three.js** for the 3D region graph (already
shipped in `brain3d.js`, reuse it), **Canvas 2D** for the gridworld (already
shipped in `world.js`, reuse it) and for raster/heatmap strips, and **uPlot** (a
6 KB canvas charting lib) for the live time-series that the hand-rolled
`charts.js` currently draws. Defer React/Svelte unless the surface keeps growing
past this revamp — the justification is in §5.4.

**Phase-1 MVP scope.** (a) New top-level IA: collapse the 12 tabs to **Launch /
Runs / Brain / Environment** (+ a "More" drawer for Findings/Plans/Bridges/
Lineages, which stay as-is); **remove** the capability-status + progress +
distribution panels. (b) The bridge-side **per-region activity aggregator** +
**`[ACTIVITY]` emission** in the navigation runner, gated behind a
`--emit-activity` flag (zero overhead when off). (c) Server parses `[ACTIVITY]`
and re-broadcasts on the existing `/ws/runs/{id}`. (d) The **Brain tab driven by
real region rates** instead of synthesized action→region lighting (reuse
`brain3d.js`'s layout + color + pulse code verbatim; swap its data source). (e)
The **Environment tab** = today's `world.js` gridworld live-mode, lifted out of
the World tab as a first-class screen with the I/O strip beside it. Everything in
Phase 1 is shippable and testable on the navigation flagship preset; the deeper
viz (pathway-flux edges, neuron drill-in, conversational word-I/O, raster
strips) is Phases 2–4.

The rest of this document: §1 audits what exists (keep/drop/revamp/add); §2
captures the external research; §3 specifies the streaming architecture; §4 the
rendering approach; §5 the stack + phased plan; §6 the honest risks.

---

## 1. Audit — what already exists (keep / drop / revamp / add)

The webapp is **far more built-out than a greenfield assumption** — there are
~12,100 lines of frontend JS/CSS/HTML and a 3,422-line FastAPI server. The
revamp is mostly *re-organizing and re-pointing* existing assets, plus closing
one real data gap.

### 1.1 Backend — `webapp/server.py` (3,422 lines, FastAPI)

| Component | Lines / location | Verdict | Notes |
|---|---|---|---|
| **Run launcher** (`POST /api/runs/launch`, `PRESETS` dict ~60 presets, `PRESET_RUNNERS`, `PRESET_OUTPUT_FLAG`) | 298–1371 | **KEEP + revamp** | The out-of-process design (detached subprocess, `DETACHED_PROCESS` on Windows, per-run `.log` + `.cmd.json` sidecar) is solid and survives server restart. The *preset list* is the maintenance burden — 60 presets, many one-off experiment configs. Revamp: group presets by **run family** (navigation / conversational / consolidation / continual / semantic-memory) and expose a curated subset + an "advanced" escape hatch, instead of one 60-entry `<select>`. |
| **`[PROGRESS]` live-parse** (`_PROGRESS_RE`, `_try_parse_progress`, `sim/progress.py` `parse_last_progress`) | 118–154; `sim/progress.py` | **KEEP + extend** | This is the existing live-telemetry spine. The navigation regex parses `step/pos/goal/recent_dist/action/reward`; `sim/progress.py` is the universal `[PROGRESS] {json}` format. **Extend with a sibling `[ACTIVITY] {json}` channel** (§3) — same stdout, same tail-reader, new line prefix. |
| **Orphan recovery** (`_scan_for_orphans`, `_periodic_orphan_scan`, PID-belongs-to-runner check) | 1459–1567 | **KEEP** | Lets the dashboard re-attach to runs after a server restart. Genuinely good; keep verbatim. |
| **WebSocket stdout stream** (`/ws/runs/{id}`, replays buffered progress on connect) | 1870–1911 | **KEEP + extend** | The transport for live mode already exists. Add an `{"type":"activity", ...}` message alongside the existing `stdout`/`progress`/`done` types. |
| **Run control file** (`POST .../control`, pause/goal/reward) | 1570–1623 | **KEEP** | Interactive teleport-goal / inject-reward. Reuse for the Environment tab. |
| **Run listing + detail + trash** (`/api/runs`, `/api/runs/{name}`, soft-delete trash system) | 237–290, 2862–3061 | **KEEP** | Past-run browse + tidy-up. Keep. |
| **`/api/inflight`** (PID-file detached runs + webapp-launched, with `_parse_log_progress` for ~6 progress formats) | 2715–2831 | **KEEP, simplify later** | Powers the live panels. The 6-format legacy regex zoo (`_DETACHED_PROGRESS_RE`, `_SWR_PROGRESS_RE`, `_CES_BENCH_*`, …) is brittle; once all runners emit `sim/progress.emit_progress`, the legacy parsers can be retired (Phase 3 cleanup). |
| **`/api/capability-status`** + `webapp/capability_status.json` (**333 KB**) | 3329–3355 | **DROP** | This is exactly the "stale self-reported progress dashboard" the owner wants gone. A 333 KB hand-maintained JSON that must be re-edited every time a milestone lands. Remove the endpoint, the file, and the Home panel that renders it. |
| **`/api/experiments`** (auto-group runs by filename suffix → mean±std vs flagship) | 3064–3147 | **DROP from primary UI** | The "cumulative run data / delta-vs-flagship" surface. Auto-derived (not hand-maintained) so cheaper than capability-status, but it's the "cumulative results dashboard" the owner explicitly doesn't want as a headline. Keep the endpoint available but remove the Experiments **tab**; if anything, fold a tiny "compare selected runs" action into the Runs tab. |
| **Findings / Plans / Bridges / Lineages / Bridge-memory / Synapse-tiering / LLM-chat endpoints** | 165–229, 1944–2433, 3263–3326 | **KEEP (move to "More")** | Useful reference surfaces, low maintenance, not the cockpit. Demote from top-level tabs into a "More" drawer. The LLM-chat endpoint is the *secondary* path per CLAUDE.md; keep but don't feature. |
| **Text-I/O run endpoints** (`/api/text_io_runs`, confusion matrices) | 3161–3257 | **KEEP, re-home** | The conversational/word-I/O data. Re-home under the Environment/I-O screen's conversational mode (§2 IA), not a separate "Language" tab. |

### 1.2 Frontend — `webapp/static/` (vanilla ES modules, no build step)

| File | Lines | Verdict | Notes |
|---|---|---|---|
| `index.html` | 790 | **REVAMP** | 12 top-level tabs (`Home/Lab/Runs/Experiments/World/Brain/Language/Findings/Plans/Bridges/Lineages/About`). Collapse to 4 primary + "More". Drop the Overview capability/distribution/KPI markup. |
| `app.js` | 4,237 | **REFACTOR (keep ~70%)** | The bootstrap + `TAB_REGISTRY` extensibility pattern + run/finding loaders. The run-launcher + runs-list + findings logic is reusable. Strip the Overview KPI/capability/distribution rendering. Extract a shared `SimSocket` + `Store` (§5.3). |
| `brain3d.js` | 1,861 | **KEEP — this is the crown jewel** | Three.js scene: one sphere/region (size ∝ log neuron count), curved pathway lines colored by transmitter (exc/inh/da/ach), CSS2D labels, OrbitControls, camera presets, bloom, traveling-pulse particles, hover/pin info panel, replay scrubber + live-poll. **The only thing wrong with it: activity is *synthesized* from action+reward, not real firing.** The revamp re-points its `regionTargets[name]` from synthetic lighting to the real `[ACTIVITY]` stream. ~90% reused. |
| `world.js` | 1,869 | **KEEP — promote to Environment tab** | Canvas-2D gridworld playback + live-mode WebSocket + retina (32×32 ON/OFF) panel + interactive goal-teleport. This *is* the environment/behavior screen the owner wants. Reuse verbatim; lift out of "World" tab into first-class "Environment". |
| `charts.js` | 411 | **REPLACE with uPlot** | Hand-rolled canvas line/bar charts. Works, but a 6 KB lib (uPlot) gives pan/zoom/crosshair/legend for the live time-series for free. Low-risk swap; keep `charts.js` until uPlot is wired. |
| `ui.js` | 305 | **KEEP** | Toasts, keyboard shortcuts, `localStorage` persisted state, helpers. Reuse. |
| `style.css` | 2,650 | **KEEP + trim** | Dark/light theme + all component styles. Trim dead Overview/Experiments selectors. |

**Net frontend reuse: ~70%.** The two hard-won renderers (`brain3d.js`,
`world.js`) survive almost untouched; the work is (a) deleting the
status/progress/cumulative surfaces, (b) re-pointing the Brain renderer at real
data, (c) re-organizing the IA, (d) extracting a shared socket/store core.

### 1.3 The in-process DearPyGUI viz (`neural-simulator.py` + `viz/`)

`neural-simulator.py` (2.2 K lines) is the **DearPyGUI host** with an in-process
OpenGL 3D neuron-firing renderer (`viz/` — camera, picker, overlays). It
visualizes individual neurons as a 3D point cloud driven directly from
`cp_membrane_potential_v` / `cp_firing_states` on the same process/thread as the
sim.

**Why it can't serve a browser:** it is an *in-process desktop OpenGL* renderer
bound to the GUI event loop and the live CuPy arrays. It has (a) no network
surface, (b) no awareness of the brain-region framework, neuromodulators, the BG
cascade, the visual-cortex pipeline, or any research-runner output (per its own
README: "zero awareness of the modern brain-region framework"), and (c) it
assumes the bridge lives in *its* process — whereas research runs are detached
subprocesses the webapp never shares memory with. **Verdict: leave it alone**
(it's the parameter-tuning tool); the browser cockpit is a separate concern and
must get its data over the wire, not from shared GPU memory.

### 1.4 How live neural state is exposed *today* (the gap)

This is the load-bearing audit finding. Traced through `sim/bridge.py`,
`sim/regions.py`, `sim/data_bus.py`, and `research/runners/g11_bg_runner.py`:

- **`sim/bridge.py`** holds the live state: `cp_firing_states` (bool array, N
  neurons, `bridge.py:250/977`), `cp_membrane_potential_v` (float32 V),
  `cp_connections` (CSR sparse weights), and a `region_manager` whose
  `.indices(name)` returns the contiguous neuron-index slice per region
  (`sim/regions.py:415`). **So per-region activity is *computable* — the slices
  exist — but nothing aggregates or exposes it.**
- **`sim/data_bus.py`** (`DataChannel` pub/sub, ring buffers) is wired into the
  step loop: `bridge.py:6232` publishes `firing_rates` (total spikes + global
  rate Hz), `spike_events` (capped to ≤500 neuron indices), and an infrequent
  `weights` histogram. **But (a) it's a global rate, not per-region; (b)
  `self.data_bus` defaults to `None` (`bridge.py:465`) so it's off unless the
  in-process GUI created the bus; (c) it's an *in-process* Python pub/sub — it
  cannot cross the subprocess boundary to the webapp.**
- **The runner** (`research/runners/g11_bg_runner.py`) drives
  `bridge._run_one_simulation_step()` in a loop and emits one `[PROGRESS]
  step N/M pos=(x,y) goal=(gx,gy) recent_dist=... action=N reward=±1` line every
  `--progress-print-interval` steps (default 100). Its final JSON logs
  `trajectory / goal_log / action_log / reward_log / phase_stats / distance_log /
  snc_rate_log` — i.e. behavior + a single SNc-rate trace, but **no per-region
  firing time-series.** (`snc_rate_log` at `bridge.py:5058` is the lone existing
  per-region rate log, and only for SNc.)

**Conclusion — what it takes to stream live state to a browser:** the data is
*one host-side reduction away*. Add a method that, given the region slices,
computes `mean(cp_firing_states[slice])` per region (one GPU reduce per region,
~30 regions → ~30 float32 = 120 bytes/frame), and emit it on stdout exactly like
`[PROGRESS]`. The webapp already tails that stdout and already has a WebSocket to
the browser. **No new IPC, no GPU↔browser socket, no shared memory.** This is the
entire reason the recommended architecture (§3) is cheap.

---

## 2. External research — best-practice live-neural + monitoring frontends

Every mature browser-based neural simulator converges on the **same shape**:
client-server, server-side aggregation, WebSocket push, client-side WebGL + 2D
charts. We are not inventing; we are adopting.

### 2.1 Real-time neural-activity frontends (what works)

- **Nengo GUI** — the closest analogue. HTML5 client-server visualizer for
  spiking networks: **server-side data collection + client-side interactive
  viz, connected by WebSockets for real-time updates**; plots spike rasters,
  decoded values, and 2D activity while the sim runs, and lets you alter inputs
  live. Confirms our exact pattern: *aggregate server-side, push over WS, render
  client-side, allow live interaction.*
  ([DeepWiki: Nengo GUI value & data viz](https://deepwiki.com/nengo/nengo-gui/3.2-value-and-data-visualization),
  [nengo/nengo-gui](https://github.com/nengo/nengo-gui))
- **NEST Desktop** — the best **stack template**. Modern web app
  (Vue + Vuetify) that uses **Plotly.js for interactive activity charts, D3.js
  for the network graph, and Three.js for 3D spatial activity rendering**, talking
  to a server-side NEST instance. This is precisely the renderer division we
  propose (3D region graph + 2D charts + heatmaps), just with our own data
  source.
  ([NEST Desktop / eNeuro](https://www.eneuro.org/content/8/6/ENEURO.0274-21.2021),
  [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8638679/))
- **The Virtual Brain (TVB)** — client-server, **HTML5 + JS + WebGL** GUI for
  whole-brain network dynamics; renders network activity on 3D cortical geometry
  remotely over the internet. Validates "WebGL brain-network activity in the
  browser, server does the compute."
  ([TVB / Frontiers](https://www.frontiersin.org/journals/neuroinformatics/articles/10.3389/fninf.2013.00010/full))
- **neuroglancer** — the **scaling counter-example**: petabyte connectomics in a
  fully **client-side multi-threaded WebGL** viewer that **fetches data over HTTP
  on demand** (precomputed/N5/Zarr), separating rendering from data processing.
  Lesson for us: *don't stream everything — stream/serve aggregates and fetch
  detail on demand.* Our "per-region rate stream + fetch neuron-level detail only
  on drill-in" mirrors neuroglancer's multi-resolution philosophy.
  ([neuroglancer docs](https://connectomics.readthedocs.io/en/latest/external/neuroglancer.html),
  [Terascale-in-the-browser, arXiv 2009.03254](https://arxiv.org/pdf/2009.03254))

**Idioms that work** (synthesized across the above): (1) a **node-link region
graph** with activity as node brightness/size and edge flux; (2) **raster /
spike-rate heatmaps** (population activity, "hotter = higher Hz"); (3)
**decoded-value / metric time-series**; (4) **interactive input perturbation**
while live; (5) **same renderer for live and replay**.

### 2.2 Streaming large live state to a browser

- **Transport choice.** As of 2026, **WebTransport is Chrome/Edge-only (partial
  Firefox, absent in Safari)**, so it's not the baseline. **SSE is text-only**
  (binary needs base64 = +33% and no true framing). **WebSocket is the pragmatic
  full-duplex baseline with native binary frames.** Recommendation: **WebSocket**
  (we already have one), JSON frames in Phase 1 (tiny payloads), optional binary
  `Float32Array` frames if we ever stream neuron-level detail.
  ([RxDB transport comparison](https://rxdb.info/articles/websockets-sse-polling-webrtc-webtransport.html),
  [Aptuz: WS vs SSE vs WebTransport 2025](https://www.aptuz.com/blog/websockets-vs-sse-vs-webtransports/))
- **Rate budgeting — the iron law: you cannot stream every neuron every step.**
  5,000 neurons × sim step rate is hundreds of K events/s; the existing
  `data_bus` already caps `spike_events` at ≤500 indices for exactly this reason
  (`bridge.py:6242`). The fix everyone uses (Nengo, TVB) is **aggregate to the
  unit that matters**: per-region firing *rate* (~30 floats), per-pathway *flux*
  (~30 floats), at a **decoupled display rate (~5–10 Hz)**, not the sim rate.
  ~60 float32 × 10 Hz = **2.4 KB/s** — negligible. Neuron-level detail is
  **fetched on demand** for a single drilled-in region only (≤ a few hundred
  neurons), never the whole brain.
- **Backpressure.** WebSocket can buffer without bound if the client stalls. With
  a 2.4 KB/s aggregate stream this is a non-issue; for any future high-rate path,
  use a **latest-wins / coalescing** policy (drop intermediate frames, always
  send the freshest) — the same approach `data_bus`'s `throttle_steps` already
  encodes.

### 2.3 Performant in-browser rendering of large neural activity

- **Library landscape.** **Three.js** = the general-3D balance (hides WebGL,
  supports **GPU instancing**, LOD, frustum culling) — right for the region graph
  and for instanced neuron point-clouds on drill-in. **deck.gl** = high-volume
  data layers (1M+ pts @ 60 fps) — overkill now, the escape hatch if we ever
  render all neurons. **regl** = max control, more graphics knowledge required.
  **PixiJS** = 2D-first. **uPlot / Plotly scattergl / webgl-heatmap** for
  charts/rasters/heatmaps.
  ([deck.gl](https://deck.gl/),
  [6 WebGL libraries](https://androidexperto.com/6-best-webgl-libraries-for-perfect-3d-web-graphics/),
  [webgl-heatmap](https://github.com/pyalot/webgl-heatmap))
- **Region layout — fixed-anatomical vs force-directed.** Research on brain-network
  layout finds **force-directed** gives topological clarity but **distorts spatial/
  anatomical meaning**, while **fixed anatomical** layouts preserve the mental map;
  the recommended answer is a **hybrid / spatial-data-driven layout** (anatomical
  anchor, force-relaxation only to de-overlap).
  ([Spatial-data-driven brain layout, ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0097849322000668))
  This *exactly matches* the decision already made in the 2026-05-02 doc and
  shipped in `brain3d.js` (anatomically-inspired, "cheat anatomy where clarity
  wins": sensory-left, motor-right, cortex-top, BG cascade as parallel lanes).
  **Keep that layout** — it's both research-aligned and already built.
- **Raster/heatmap idiom.** "Topographical maps of group activity where hotter
  colors mean higher firing rate" + raster/PSTH strips are the standard
  population-activity views; WebGL heatmap libs render these at scale.
  ([Heatmap viz in browser, Medium](https://medium.com/@vibhu.anime1003/peeking-inside-neural-networks-interactive-heatmap-visualizations-in-the-browser-0c14b6674be8))

### 2.4 Run-control / job-monitoring dashboard UX

- **MLflow / W&B / TensorBoard** patterns to adopt: a **run list with status +
  live metrics**, **click-through to per-run detail**, **live charts that update
  while running**, **compare-selected-runs overlay**, customizable panels. The
  pattern to *avoid* (per the owner): the bolted-on "project status / cumulative
  scoreboard" that needs hand-curation. Adopt the **live run list + per-run live
  charts**; drop the editorialized status.
  ([DagsHub experiment-tracking comparison](https://dagshub.com/blog/best-8-experiment-tracking-tools-for-machine-learning-2023/),
  [ML experiment tracking guide](https://kindatechnical.com/deep-learning/lesson-90-experiment-tracking-with-mlflow-weights-and-biases-and-tensorboard.html))

### 2.5 Vanilla-vs-framework

- **Svelte** compiles to near-vanilla, 8–15 KB bundles (NYT graphics/election
  dashboards) and is the gentlest migration from HTML/JS — *if* we migrate.
  **React** wins for large multi-developer apps (this is single-dev). For a
  no-build, single-user, ~12 K-line dashboard where two large renderers are
  already vanilla Three.js/Canvas, the research supports **staying vanilla and
  consolidating**, with Svelte as the *only* framework worth considering later
  (no-build via CDN is harder there). Justification in §5.4.
  ([Svelte vs React 2026](https://strapi.io/blog/svelte-vs-react-comparison),
  [FrontendTools framework comparison](https://www.frontendtools.tech/blog/best-frontend-frameworks-2025-comparison))

---

## 3. The streaming architecture (sim → browser, concrete)

This is the centerpiece's foundation. Design goal: **real per-region activity in
the browser at sim speed, honest about the rate ceiling, reusing existing
plumbing.**

### 3.1 Data path (end-to-end)

```
  [ sim thread, detached subprocess ]            [ FastAPI server ]        [ browser ]
  bridge._run_one_simulation_step()                                          Brain tab
        │  (every step)                                                  (Three.js + uPlot)
        ▼                                                                      ▲
  ActivityAggregator.sample(bridge)   ── every K steps (display cadence) ──    │
        │  per-region mean firing rate (≤~30 floats)                           │
        │  per-pathway flux (≤~30 floats)                                      │
        ▼                                                                      │
  emit_activity(...)  →  "[ACTIVITY] {json}\n"  on stdout                      │
        │                                                                      │
        ▼   (already-existing per-run .log file)                              │
  webapp _drain_log() tails the .log  ──►  _try_parse_activity(line)          │
        │                                                                      │
        ▼   ring-buffer last N frames on LaunchedRun.activity_frames          │
  /ws/runs/{id}  ──── {"type":"activity", "t":…, "regions":{…}, "flux":{…}} ──┘
```

**Why this path:** it is byte-for-byte the path `[PROGRESS]` already takes
(runner stdout → `.log` file → `_drain_log` tail → WebSocket). We add **one new
line prefix** and **one new WS message type**. No socket from the GPU process, no
shared memory across the subprocess boundary, and it inherits orphan-recovery and
restart-survival *for free*.

### 3.2 The bridge-side aggregator (new, additive, default-off)

A new helper (lives in `sim/`, e.g. `sim/activity_probe.py`, or a method on the
bridge) — **purely additive, zero overhead when unused**, matching the project's
"protected sim/ edits are additive/guarded" rule:

```python
class RegionActivityProbe:
    """Host-side per-region activity reduction for live streaming.
    Computed from region slices that already exist on region_manager."""
    def __init__(self, bridge, region_names: list[str] | None = None):
        # Precompute the index arrays ONCE (region slices are static).
        self.slices = {name: xp.asarray(bridge.region_manager.indices(name))
                       for name in (region_names or bridge.region_manager.all_names())}
    def sample(self, bridge) -> dict[str, float]:
        fired = bridge.cp_firing_states           # bool[N] on device
        # One reduce per region → ~30 host floats. mean over the slice = fraction firing.
        return {name: float(fired[idx].mean()) for name, idx in self.slices.items()}
```

Cost: ~30 boolean-mean reductions per *sampled* frame (not per step). Sampled at
the display cadence (every K steps so ~5–10 Hz wall-clock), the GPU↔host transfer
is ~30 scalars. This is the **only** new GPU touch and it is tiny. Per-pathway
**flux** (Phase 2) is the analogous reduction over the *post-synaptic* region's
firing for each `RegionPathway`, optionally weighted by mean pathway weight.

**EMA smoothing** is applied host-side (display-rate, not sim-rate) so the viz
isn't strobic: `rate_ema = α·rate + (1−α)·rate_ema`.

### 3.3 The emission format (extends `sim/progress.py`)

Add a sibling to `emit_progress` in `sim/progress.py`:

```python
def emit_activity(t_ms: float, regions: dict[str, float],
                  flux: dict[str, float] | None = None, **extra) -> None:
    payload = {"t": round(t_ms, 1), "regions": {k: round(v, 4) for k, v in regions.items()}}
    if flux: payload["flux"] = {k: round(v, 4) for k, v in flux.items()}
    payload.update(extra)
    print(f"[ACTIVITY] {json.dumps(payload)}", flush=True)
```

Wire format identical to `[PROGRESS]`: **one line, JSON after a fixed prefix,
`flush=True`.** Server-side regex `r"\[ACTIVITY\]\s+(\{.*\})"`. Payload size for
30 regions ≈ 600–900 bytes/frame; at 10 Hz ≈ 6–9 KB/s on the *log*, re-broadcast
as-is on the WS. (If this ever matters, switch the WS frame to a packed
`Float32Array` with a fixed region-order header — but JSON is fine at this scale.)

### 3.4 Runner integration (gated)

The navigation runner gets a `--emit-activity` flag (and `--activity-interval N`,
default = `progress-print-interval`). When set, it constructs the probe once and
calls `emit_activity(...)` inside the existing step loop next to the
`emit_progress` call. **Default off ⇒ existing runs and benchmarks are
byte-unchanged** (critical: the project runs multi-seed science where determinism
+ wall-clock matter; the probe must never perturb a science run). The webapp
launcher adds `--emit-activity` automatically for runs launched from the **Brain**
or **Environment** screens, the same way it injects `--interactive-control-file`
today.

### 3.5 Server changes (small)

1. `ProgressEvent` gets a sibling `ActivityFrame` dataclass; `LaunchedRun` gets
   `activity_frames: deque(maxlen=600)` (a ring buffer — 60 s at 10 Hz).
2. `_drain_log` (already tailing the `.log`) additionally runs
   `_try_parse_activity(line)` and appends to the ring.
3. `/ws/runs/{id}` emits `{"type":"activity", ...}` frames alongside the existing
   `stdout`/`progress`/`done`; on connect it replays the last frame (latest-wins)
   so a late joiner sees current state immediately.
4. New `GET /api/runs/{id}/region-map` returns the **static** region graph for a
   run: region names, families/colors, neuron counts, 3D layout coords, and the
   pathway list (from/to/transmitter). This is computed once from
   `sim/regions.py` + `sim/profiles.py` and lets the Brain tab build the scene
   before any activity arrives. (Phase 1 can hardcode the navigation region map,
   matching `brain3d.js`'s current built-in layout; Phase 2 derives it per-run.)
5. **Drill-in (Phase 3):** `GET /api/runs/{id}/region/{name}/neurons` returns a
   one-shot snapshot of `cp_firing_states[slice]` / `cp_membrane_potential_v[slice]`
   for a single region — but this requires the run to expose a control channel
   that can *read* bridge state on demand. Since runs are detached subprocesses,
   the honest Phase-3 mechanism is: the runner, when `--emit-activity` is on and a
   region is "focused" (set via the existing control file), additionally emits a
   capped `[ACTIVITY_DETAIL] {region, neuron_rates:[…≤500]}` line. This reuses the
   control-file request path and the ≤500-cap already in `data_bus`.

### 3.6 Honesty about feasibility at sim speed

- Navigation runs step at ~30–60 steps/s (the interactive presets throttle to
  ~33/s via `--trial-sleep-ms 30`). Sampling every ~5 steps → ~6–10 activity
  frames/s. **Comfortable.**
- Heavy training/consolidation runs (hours, NMDA, big motor pools) step slower
  but the human only needs ~5–10 fps to perceive "which regions are firing."
  Sampling cadence is decoupled from step rate, so we just sample every Nth step.
- The aggregator's ~30 GPU reduces/frame are negligible vs a step's full neuron+
  synapse update. Measured cost is dominated by the host transfer of ~30 scalars
  (one `.get()` per frame if batched). **We will not stream per-neuron state for
  the whole brain — only per-region rate.** Drill-in detail is one region,
  on-demand, capped. This is the same multi-resolution discipline neuroglancer
  uses, and it is the only honest way to do this at sim speed.

---

## 4. The rendering approach

### 4.1 The centerpiece — real-time brain-activity view (Brain tab)

**Reuse `brain3d.js` almost wholesale**; the layout + color + pulse work is done
and research-aligned (§2.3). Changes:

1. **Data source swap (the core change).** Replace the synthesized
   `action→region` lighting with the real `[ACTIVITY]` stream: each WS activity
   frame sets `regionTargets[name] = ema(regions[name])`; the existing per-frame
   lerp into `emissiveIntensity` already animates it smoothly. Brightness = real
   firing fraction; node size stays ∝ log(neuron count).
2. **Pathway flux on edges (Phase 2).** Today pathway lines brighten when their
   pre/post region is active (a proxy). Replace with real per-pathway `flux[name]`
   → edge opacity/width, and spawn traveling-pulse particles at a rate ∝ flux
   (the particle system already exists). This is the "activity flowing along
   pathways" the owner asked for, now driven by data.
3. **Layout = the shipped hybrid anatomical** (sensory-left → motor-right, cortex
   top, subcortical below, BG cascade as 4 parallel lanes, hippocampus as a side
   subassembly). Matches the spatial-data-driven recommendation; keep it. Region
   map comes from `/api/runs/{id}/region-map` so non-navigation runs
   (conversational: `lang_input` → pools → `lang_output`; hippocampal trisynaptic
   loop) get a correct graph instead of the navigation-only built-in.
4. **Drill-in (Phase 3).** Click a region → camera flies in (already wired) →
   instanced neuron point-cloud (Three.js `InstancedMesh`) driven by
   `[ACTIVITY_DETAIL]`. Until detail arrives, show the region's aggregate rate as
   a uniform glow (graceful).
5. **2D companion strips (Phase 2).** Beside the 3D scene, a **region-rate heatmap
   strip** (regions × time, hotter = higher Hz; Canvas 2D or webgl-heatmap) and a
   **selected-region rate time-series** (uPlot). These are the "raster/heatmap"
   idioms from §2.3 and give the quantitative read the 3D glow can't.

### 4.2 The environment / behavior view (Environment tab)

**Reuse `world.js` verbatim**, promoted from a sub-tab to a first-class screen:
Canvas-2D gridworld, animated agent (heading arrow), goal beacon + intensity
falloff, landmark, fading trail, live distance chart, interactive goal-teleport +
reward injection (control file). Add the **retina (32×32 ON/OFF) panel** (already
in `world.js`) as the **sensory-input** half of the I/O strip; the **motor-output**
half shows the 4 motor-pool rates as a bar/compass (derived from the same
`[ACTIVITY]` stream — `motor_N/E/S/W`). For **conversational** runs the Environment
tab swaps the gridworld for a **word-I/O panel**: words-in (the driven
`lang_input` token) and words-out (decoded `lang_output`, reusing the existing
`/api/text_io_runs` confusion-matrix data for replay + the live `[ACTIVITY]` for
`lang_output` pool rates).

### 4.3 Run-control + monitoring views (Launch + Runs tabs)

- **Launch:** a curated, **family-grouped** picker (Navigation / Conversational /
  Consolidation / Continual / Semantic-memory), each family showing 2–4 *headline*
  presets with a one-line plain-English description + ETA, plus an "Advanced"
  disclosure for the full preset list and free-text extra-args (preserve the
  existing multi-seed `42,43,44` / `42-47` input). This replaces the current
  60-entry `<select>`.
- **Runs:** the existing live-run list + past-run list + per-run detail, with
  **live uPlot charts** (distance/reward for nav; accuracy for training) that
  update over the WS while running, and a "compare selected" overlay (fold in the
  old Experiments compare without the cumulative-scoreboard framing). Each row
  keeps the viewer-affinity buttons (open in Brain / Environment).

### 4.4 What's dropped from the rendered UI

- The **capability-status** panel (+ its 333 KB JSON + endpoint).
- The **Overview KPI cards** (Best Run / Mean Sum / Total Runs / etc.) and the
  **result-distribution histogram** (the "cumulative run data" surface).
- The **Experiments tab** as a headline (delta-vs-flagship scoreboard). Its
  endpoint may linger for a "compare selected runs" affordance, but it is not a
  top-level destination.

---

## 5. Tech stack + phased build plan

### 5.1 Recommended stack (opinionated)

| Layer | Choice | Why |
|---|---|---|
| Backend | **FastAPI + uvicorn (keep)** | Plumbing (launch/log/orphan/WS) already works; revamp adds ~2 endpoints + 1 WS type + the `[ACTIVITY]` parser. |
| Sim telemetry | **`[ACTIVITY] {json}` on stdout** (extend `sim/progress.py`) | Reuses the `[PROGRESS]` path; no new IPC; restart-survivable; additive + default-off (science-safe). |
| Transport | **WebSocket (keep `/ws/runs/{id}`)** | Native binary-capable full-duplex; already present; SSE is text-only, WebTransport not in Safari. |
| Frontend core | **Vanilla ES modules, no build step (keep)** + extract a shared `SimSocket`/`Store`/`TabRegistry` | 70% of the frontend (incl. both renderers) is already vanilla; a framework rewrite is unjustified for single-dev/no-build (§5.4). |
| 3D | **Three.js via CDN importmap (keep `brain3d.js`)** | The region graph + pulses + camera + bloom are built and research-aligned. |
| 2D world | **Canvas 2D (keep `world.js`)** | Correct tool for a flat gridworld; built. |
| Charts | **uPlot (6 KB, add)** → replaces `charts.js` | Free pan/zoom/crosshair/legend for live time-series; tiny. |
| Heatmaps | **Canvas 2D (Phase 2) → webgl-heatmap if needed** | Region×time rate heatmap; scale up only if needed. |

### 5.2 Phased plan (each phase shippable + testable)

**Phase 1 — MVP: real region activity + re-organized cockpit.** *(the headline)*
- IA: collapse 12 tabs → **Launch / Runs / Brain / Environment** + "More" drawer
  (Findings/Plans/Bridges/Lineages). Delete capability-status + Overview-KPI +
  distribution markup, `/api/capability-status`, `capability_status.json`.
- Bridge: add `RegionActivityProbe` (additive, default-off) + `emit_activity` in
  `sim/progress.py`.
- Runner: `--emit-activity` flag in `g11_bg_runner.py` (nav family first).
- Server: parse `[ACTIVITY]`, ring-buffer, re-broadcast on `/ws/runs/{id}`; add
  `GET /api/runs/{id}/region-map` (nav region map hardcoded to match `brain3d.js`).
- Frontend: re-point `brain3d.js` `regionTargets` from synthesized lighting to the
  real activity stream; promote `world.js` to the **Environment** tab; family-group
  the launcher.
- **Testable:** launch the nav flagship with `--emit-activity`; the Brain tab
  lights up regions from *real* firing, and the Environment tab shows the agent
  moving. Add `tests/test_webapp_server.py` cases for the `[ACTIVITY]` parse +
  region-map endpoint (the existing 16 endpoint tests are the template).
  Reuse the **`keep-webapp-current`** skill (verifies server/app.js/world.js/
  preset-list stay in sync) and **`sync-documentation`** (drift in README/CLAUDE).

**Phase 2 — pathway flux + 2D companion strips + conversational region map.**
- Bridge probe also computes per-pathway flux; runner emits it.
- `brain3d.js` edges + pulse-rate driven by real flux.
- Add the region-rate **heatmap strip** + selected-region **uPlot time-series**
  beside the 3D scene.
- `/api/runs/{id}/region-map` derives the graph per-run from `sim/regions.py`
  (conversational `lang_input`→pools→`lang_output`, hippocampal loop, etc.).
- I/O strip: motor-pool compass (nav) + word-in/out (conversational).
- **Testable:** conversational runs (e.g. `chat_speak_demo`) render a correct
  non-navigation region graph; flux animates on the real pathways.

**Phase 3 — neuron drill-in + detail-on-demand + legacy-parser cleanup.**
- `InstancedMesh` neuron cloud on region click, driven by `[ACTIVITY_DETAIL]`
  (capped ≤500, focused region only, via the control-file request path).
- Retire the 6 legacy progress regexes once all featured runners use
  `emit_progress`/`emit_activity` (simplifies `_parse_log_progress`).
- **Testable:** drilling a region shows live per-neuron glow; no whole-brain
  neuron streaming (verify the cap).

**Phase 4 — polish + (optional) framework migration trigger.**
- Replay parity (scrub a completed run's recorded activity — requires logging the
  activity trace into the run JSON; small additive change to the runner's final
  dump).
- uPlot fully replaces `charts.js`; trim dead CSS.
- *Decision gate:* if the frontend has grown past ~15 K lines or multiple
  contributors appear, evaluate a **Svelte** migration (§5.4); otherwise stay
  vanilla.

### 5.3 The frontend consolidation (not a rewrite)

Extract three shared modules from `app.js` so each tab is thin:
- `SimSocket` — one typed client for `/ws/runs/{id}` that demuxes
  `progress`/`activity`/`stdout`/`done` and exposes `onActivity(cb)` /
  `onProgress(cb)`; Brain + Environment + Runs all subscribe.
- `Store` — the current `localStorage`-backed state (active tab, filters,
  selected run) formalized, plus the live activity ring per run.
- `TabRegistry` — already an emergent pattern in `app.js` ("this whole tab was
  added by appending one entry to TAB_REGISTRY"); formalize it so Launch/Runs/
  Brain/Environment register declaratively and "More" holds the rest.

This is ~1–2 days of refactor that makes the renderers share one socket instead
of each polling, and it is the prerequisite that keeps "stay vanilla" honest.

### 5.4 Vanilla vs framework — the recommendation, defended

**Stay vanilla (consolidate), defer Svelte, do not adopt React.** Rationale:
- The two hardest, highest-value pieces (`brain3d.js` Three.js scene, `world.js`
  Canvas world) are **already vanilla** and would be rewritten *as-is* under any
  framework (frameworks don't help imperative WebGL/Canvas render loops). A
  rewrite risks the crown jewels for no rendering gain.
- It is a **single-user, no-build** dashboard. React's wins (multi-dev discipline,
  hiring, ecosystem) don't apply; its costs (build step, bundle) do. The research
  is explicit that vanilla/thin-framework wins at the performance edge and for
  small teams ([Svelte vs React 2026](https://strapi.io/blog/svelte-vs-react-comparison)).
- **Svelte** is the only framework worth a later look (8–15 KB bundles, gentle
  migration from HTML/JS, used for NYT live dashboards) — but it *wants* a build
  step, breaking the no-build constraint, so it's a Phase-4 *option* gated on the
  surface outgrowing vanilla, not a now-decision.
- The honest risk of staying vanilla is the 4 K-line `app.js` monolith; §5.3's
  extraction directly mitigates it.

---

## 6. Honest risks

1. **Streaming-rate ceiling is real, and the design respects it — but drill-in is
   the soft spot.** Per-region aggregate streaming is trivially cheap (§3.6). The
   honest limitation is **neuron-level detail**: because runs are detached
   subprocesses with no shared memory, on-demand per-neuron reads must round-trip
   through the control-file → `[ACTIVITY_DETAIL]` path, which is laggier than a
   direct memory read and capped at ≤500 neurons. Mitigation: drill-in shows one
   region, capped, best-effort; the whole-brain view never needs it. If true
   live per-neuron-of-the-whole-brain is ever required, that's a different
   architecture (in-process renderer, i.e. the existing DearPyGUI tool) — out of
   scope and explicitly *not* what the owner asked for.
2. **3D-in-browser performance is a solved problem at our scale, with one caveat.**
   ~30–60 region spheres + ~60 pathway curves + ~50 pulse particles is trivial for
   Three.js (the scene already runs). The caveat is **bloom postprocessing on
   low-end GPUs / integrated graphics** — `UnrealBloomPass` is the heaviest cost.
   Mitigation: bloom is already a toggle (`brain3d-bloom`); default it off on
   detected low-power contexts. Neuron drill-in (`InstancedMesh`, Phase 3) scales
   to the few-hundred-neuron focused region without issue.
3. **`[ACTIVITY]`-on-stdout is pragmatic but couples telemetry to log volume.**
   At 10 Hz × ~30 regions the `.log` grows ~6–9 KB/s (~30 MB/hr) — fine for the
   interactive runs that use it, but it must **not** be enabled on long
   multi-hour science runs by default (it would bloat logs and add host-transfer
   cost that could perturb wall-clock comparisons). Mitigation: `--emit-activity`
   is **default-off**; the webapp only adds it for runs launched from the
   Brain/Environment screens; science/multi-seed launches never get it. (If log
   volume ever bites, a dedicated `.activity` sidecar file or a binary frame is
   the upgrade — but JSON-on-stdout is right for the MVP.)
4. **Determinism / science-safety of the probe.** The probe adds GPU reductions +
   a host transfer; even tiny, this *could* in principle shift timing on a
   determinism-sensitive run. Because it is additive and default-off, the
   guarantee is simply: **science runs don't pass `--emit-activity`, so they are
   byte-identical to today.** This must be enforced (the multi-seed launchers and
   `--from-scratch`/`--deterministic` paths must never inject it). Call it out in
   the `keep-webapp-current` skill checklist.
5. **Existing-webapp reuse is high but the IA change touches a 790-line HTML + a
   4.2 K-line app.js.** Risk of regressions in the surviving tabs (Findings/Plans/
   Bridges/Lineages) during the re-org. Mitigation: the `TabRegistry` extraction
   (§5.3) is additive; move tabs into "More" rather than deleting their code;
   keep `tests/test_webapp_server.py` green; lean on the `keep-webapp-current` +
   `sync-documentation` skills after each phase. The two renderers (`brain3d.js`,
   `world.js`) are touched only at their *data source*, not their render loops,
   bounding the blast radius.
6. **Per-run region-map derivation (Phase 2) depends on `sim/regions.py` internals.**
   Deriving an arbitrary run's region graph + 3D layout from `RegionManager` for
   *non-navigation* architectures is more work than the hardcoded nav map and may
   need per-family layout heuristics (conversational vs hippocampal vs nav). Phase
   1 sidesteps this by shipping the nav map only; Phase 2 takes it on deliberately.

---

## Appendix A — file-by-file change map (for implementers)

| File | Phase 1 change |
|---|---|
| `sim/progress.py` | + `emit_activity(...)` (additive) |
| `sim/activity_probe.py` *(new)* | `RegionActivityProbe` (per-region mean firing reduce) |
| `research/runners/g11_bg_runner.py` | + `--emit-activity` / `--activity-interval`; construct probe once; call `emit_activity` in the step loop next to `emit_progress` |
| `webapp/server.py` | + `_try_parse_activity` + `ActivityFrame` + `LaunchedRun.activity_frames`; `_drain_log` parses activity; `/ws/runs/{id}` emits `{"type":"activity"}`; + `GET /api/runs/{id}/region-map`; **remove** `/api/capability-status` + `capability_status.json`; auto-inject `--emit-activity` for Brain/Environment launches |
| `webapp/static/index.html` | collapse tabs → Launch/Runs/Brain/Environment + "More"; delete Overview capability/KPI/distribution markup |
| `webapp/static/app.js` | extract `SimSocket`/`Store`/`TabRegistry`; remove Overview KPI/capability/distribution loaders; family-group launcher |
| `webapp/static/brain3d.js` | swap `regionTargets` source: synthesized → real `[ACTIVITY]` via `SimSocket.onActivity`; consume `/api/runs/{id}/region-map` |
| `webapp/static/world.js` | promote to Environment tab (no render-loop change) |
| `webapp/static/charts.js` | (Phase 4) replaced by uPlot |
| `webapp/capability_status.json` | **delete** |
| `tests/test_webapp_server.py` | + `[ACTIVITY]` parse test + region-map endpoint test |

## Appendix B — sources

Research claims above are cited inline; consolidated:

- Nengo GUI architecture — https://deepwiki.com/nengo/nengo-gui/3.2-value-and-data-visualization · https://github.com/nengo/nengo-gui
- NEST Desktop (Vue + Plotly + D3 + Three.js) — https://www.eneuro.org/content/8/6/ENEURO.0274-21.2021 · https://pmc.ncbi.nlm.nih.gov/articles/PMC8638679/
- The Virtual Brain (HTML5/JS/WebGL client-server) — https://www.frontiersin.org/journals/neuroinformatics/articles/10.3389/fninf.2013.00010/full
- neuroglancer (client-side WebGL, fetch-on-demand, terascale) — https://connectomics.readthedocs.io/en/latest/external/neuroglancer.html · https://arxiv.org/pdf/2009.03254
- Transport (WS vs SSE vs WebTransport, 2025) — https://rxdb.info/articles/websockets-sse-polling-webrtc-webtransport.html · https://www.aptuz.com/blog/websockets-vs-sse-vs-webtransports/
- WebGL libraries (Three.js / deck.gl / regl / Pixi) — https://deck.gl/ · https://androidexperto.com/6-best-webgl-libraries-for-perfect-3d-web-graphics/
- Brain-network layout (anatomical vs force-directed vs hybrid) — https://www.sciencedirect.com/science/article/pii/S0097849322000668
- WebGL heatmaps / browser neural-activity heatmaps — https://github.com/pyalot/webgl-heatmap · https://medium.com/@vibhu.anime1003/peeking-inside-neural-networks-interactive-heatmap-visualizations-in-the-browser-0c14b6674be8
- Experiment-tracking dashboard UX (MLflow/W&B/TensorBoard) — https://dagshub.com/blog/best-8-experiment-tracking-tools-for-machine-learning-2023/ · https://kindatechnical.com/deep-learning/lesson-90-experiment-tracking-with-mlflow-weights-and-biases-and-tensorboard.html
- Vanilla vs Svelte vs React (2025/26) — https://strapi.io/blog/svelte-vs-react-comparison · https://www.frontendtools.tech/blog/best-frontend-frameworks-2025-comparison
