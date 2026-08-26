---
type: plan
status: live
date: 2026-04-06
---

# Design: Interactive Exploration & In-App Experiment Prototyping

**Date:** 2026-04-06
**Status:** Approved
**Goal:** Transform the simulator from a visualization tool into a self-contained research instrument with interactive network exploration and UI-driven experiment prototyping.

**Target users:** Power users (the developer) and computational neuroscience researchers doing rapid prototyping. Dense information density, no hand-holding, but good discoverability.

**Key decision:** The monolithic `neural-simulator.py` is split into modules as part of this work.

---

## 1. Module Architecture

Split the ~14.8K-line monolith along natural boundaries. Each module owns one concern.

```
neural-simulator.py              → entry point, CLI args, thread orchestration (~500 lines)

sim/
  __init__.py
  config.py                      → CoreSimConfig, VisualizationConfig, GPUConfig, RuntimeState
  enums.py                       → NeuronModel, NeuronType, StimulusPatternType, etc.
  profiles.py                    → NEURAL_STRUCTURE_PROFILES, HH presets, trait definitions
  kernels.py                     → all @cp.fuse() GPU kernels
  bridge.py                      → SimulationBridge (engine, step loop, GPU state)
  connectivity.py                → connection generators (spatial, Watts-Strogatz, chunked, binned)
  data_bus.py                    → pub/sub data streaming with ring buffers

experiment/
  __init__.py
  engine.py                      → ExperimentEngine, phase management, logging
  stimulus.py                    → StimulusManager, StimulusChannel, patterns
  readout.py                     → ReadoutEngine, PSD, band power, synchrony
  training.py                    → TrainingProtocolEngine, reward delivery
  presets.py                     → ExperimentPresets factory
  groups.py                      → NeuronGroupManager, NeuronGroup

ui/
  __init__.py
  layout.py                      → DearPyGUI window creation, collapsing headers, parameter tables
  callbacks.py                   → all UI callback functions
  plots.py                       → raster plot, firing rate traces, weight histograms, sweep figures
  plot_manager.py                → shared plot infrastructure, synchronized time axes, update scheduling
  inspector.py                   → neuron/synapse inspection panel
  sweep_panel.py                 → in-app parameter sweep configuration and results viewer
  experiment_dashboard.py        → experiment designer with phase timeline and results
  figure_export.py               → matplotlib-based publication figure generation
  state_manager.py               → typed UI state (selection, active experiment, layout persistence)

viz/
  __init__.py
  renderer.py                    → OpenGL rendering (render_scene_gl, VBO management)
  camera.py                      → camera controls, mouse/keyboard input
  picker.py                      → GPU color-based neuron picking
  overlays.py                    → HUD text, selection highlights, group bounding boxes
```

### Dependency rules

- `sim/` has zero UI or viz dependencies — importable by headless scripts.
- `experiment/` has zero UI or viz dependencies — same.
- `ui/` depends on `sim/` and `experiment/`, never the reverse.
- `viz/` depends on `sim/` for data, not on `ui/`.
- `neural-simulator.py` is the thin orchestrator wiring everything together.
- Headless scripts (`run_experiment_headless.py`, `run_benchmarks.py`, `run_parameter_sweep.py`) import from `sim/` and `experiment/` only.

---

## 2. Interactive Exploration

### 2.1 Neuron Picking & Inspection

**Picking mechanism** (`viz/picker.py`): Color-based GPU picking. On click, render one off-screen frame where each neuron's color encodes its index as RGB. Read the pixel at click position, decode to neuron index.

**Inspector panel** (`ui/inspector.py`): A DearPyGUI collapsing header showing:

- **Identity:** Neuron index, trait name (Excitatory RS / Inhibitory FS), group membership.
- **Live state:** Membrane potential (mV), recovery variable, refractory timer, firing rate (1s window), last spike time. Updated every data bus tick.
- **Connectivity:** Incoming/outgoing synapse count, mean incoming/outgoing weight, strongest connection partner index.
- **Plasticity state:** STDP eligibility, homeostatic threshold offset, STP u/x for outgoing synapses.
- **Mini-plots:** Membrane potential trace (last 500ms), spike raster (last 2s), incoming synaptic current trace. Uses DearPyGUI line series with ring buffer data.

**Multi-select:** Shift+click adds neurons. Selection group gets aggregate stats (mean rate, pairwise synchrony, mean inter-selection weight). Selected neurons highlighted in 3D with size increase.

**Group inspection:** Click a group label in the experiment dashboard to select all neurons in that group.

### 2.2 Live Plots

**`ui/plots.py`** — DearPyGUI plots using `dpg.add_plot()` with streaming data.

| Plot | Data source | Update rate | Notes |
|------|-------------|-------------|-------|
| Spike raster | spike_events channel | Every step | Subsampled to ~200 neurons (stratified by trait), scrolling last N seconds |
| Population firing rate | firing_rates channel | Every 100ms | One line per trait or group, scrolling |
| Weight distribution | weights channel | Every 1s | Histogram of current synaptic weights |
| Spectral power | band_power channel | Every 2s | Bar chart of delta through high-gamma |
| Synchrony trace | synchrony channel | Every 100ms | Fano factor line plot per group |

All time-series plots share a synchronized time axis. Pause button freezes updates without pausing simulation.

### 2.3 Connectivity Viewer

**Adjacency heatmap:** DearPyGUI raw texture showing the connectivity matrix, downsampled for large networks. Neurons on both axes grouped by trait. Color = weight. Updated on demand.

**Selected neuron connections:** When a neuron is picked, its connections are highlighted in 3D:
- Outgoing synapses in green, incoming in blue.
- Line opacity proportional to weight.
- Toggled via checkbox in the inspector panel.

---

## 3. In-App Experiment Prototyping

### 3.1 Experiment Designer

**`ui/experiment_dashboard.py`** — replaces the current minimal experiment section.

**Phase timeline:** Horizontal bar of colored blocks (baseline=gray, stimulus=blue, training=green, testing=orange), proportional to duration. Clicking a block opens its config. During execution, a playhead cursor shows progress.

**Channel editor:** Inline list of stimulus channels with:
- Target group dropdown, pattern type dropdown, amplitude slider, onset/duration/repeat fields.
- Small inline preview plot showing the waveform shape over one period.
- Add/remove channel buttons.

**Group manager:** Visual list of neuron groups with index range, color picker, role dropdown, add/remove. "Auto-assign" button splits by trait.

**Readout config:** Checkboxes per group for firing rate, spike count, band power, synchrony. Window size slider.

**Save/Load:** Custom experiment configs to/from JSON, building a library beyond presets.

### 3.2 Parameter Sweep UI

**`ui/sweep_panel.py`** — visual version of `run_parameter_sweep.py`.

**Configuration:**
- Parameter dropdown (populated from CoreSimConfig fields).
- Value entry: start/end/num_steps, or comma-separated.
- "Add parameter" for multi-parameter grid sweeps.
- Grid vs Zip toggle.
- Experiment selector, trials per point.

**Execution:**
- Background thread runs the sweep.
- Progress bar with current run label (e.g., "Run 3/12: stdp_a_plus=0.012").
- Live results table fills as runs complete.
- Cancel button, ETA display.

**Results:**
- Auto-generated plots: line plot for 1-param sweeps (with error bars), heatmap for 2-param sweeps.
- Metric selector dropdown (delta_hz, t_statistic, cohens_d, peak_freq, etc.).
- Sortable results table.
- "Save Figure", "Export CSV", "Export JSON" buttons.

### 3.3 Results & Figures

**Single experiment results:**
- Pre/post bar chart with error bars and significance stars.
- Phase-by-phase timeline with shaded regions.
- Weight evolution per phase transition.
- Band power comparison (grouped bars).
- Statistical summary: delta, t-stat, Cohen's d, p-value, CI.

**Sweep results:**
- Dose-response curves.
- Parameter sensitivity heatmaps.
- Best/worst highlights.
- "Re-run best" button loads those parameters.

**Experiment history:** Dropdown to switch between recent results. "Compare" mode overlays two experiments on the same plot.

### 3.4 Export

- "Export Results JSON" — structured data.
- "Export Figure PNG/SVG" — matplotlib re-render for publication quality.
- "Export CSV" — tabular data.
- "Copy to Clipboard" — formatted summary text.

---

## 4. Shared Infrastructure

### 4.1 Data Bus (`sim/data_bus.py`)

Replaces the raw `sim_to_ui_queue` dict with structured pub/sub.

**Channels:** Named data streams — `"firing_rates"`, `"spike_events"`, `"weights"`, `"experiment_status"`, `"neuron_state"`, `"band_power"`, `"synchrony"`.

**Ring buffers:** Each channel keeps configurable history (e.g., 10s of firing rates, 5s of spikes). Plots read from buffers directly — no re-requests to simulation thread.

**Throttling:** Per-channel max update rate. Spike events every step, weights every 1000 steps, PSD every 2000 steps.

**Architecture:** Simulation thread publishes. UI thread reads from ring buffers. Lock-free, same pattern as existing queue.

### 4.2 Plot Manager (`ui/plot_manager.py`)

- **Registry:** Central list of active plots with per-plot update functions.
- **Synchronized time axis:** Zoom/pan on one plot applies to all linked time-series.
- **Staggered updates:** Round-robin plot refreshes to stay under 2ms total per UI frame.
- **Pause/resume:** Single button freezes all live plots.

### 4.3 Figure Export (`ui/figure_export.py`)

DearPyGUI plots are interactive but not publication-grade. Export path uses matplotlib:

- Plot data stored in structured dicts (x, y, labels, errors, annotations).
- "Export Figure" triggers matplotlib rendering to PNG/SVG/PDF.
- Journal-ready style preset (white background, serif labels, proper formatting).
- Includes title, axis labels with units, significance annotations, legend.

### 4.4 State Manager (`ui/state_manager.py`)

Typed, observable UI state replacing `global_gui_state`:

- **Selected neurons:** Set of indices. Updated by picker, read by inspector/plots/renderer.
- **Active experiment:** Current config and engine reference.
- **Sweep state:** Config, progress, results.
- **Plot visibility:** Which plots are open. Persisted to `ui_layout.json` across sessions.
- **Change callbacks:** Components register for state change notifications.

---

## 5. Implementation Phases

### Phase 1: Module Split (foundation — everything depends on this)
Extract the monolith into the module structure above. All existing tests and headless scripts must pass unchanged. No new features — pure refactor.

### Phase 2: Data Bus + State Manager (infrastructure)
Implement the pub/sub data bus and typed state manager. Wire the simulation thread to publish to channels. Existing UI continues to work via the bus.

### Phase 3: Live Plots (first visible payoff)
Implement the plot manager and initial plots: spike raster, population firing rate trace, weight histogram. These give immediate value and validate the data pipeline.

### Phase 4: Neuron Picking & Inspector (interactive exploration)
Add GPU picking to the 3D view. Build the inspector panel. Wire selection state to 3D highlights and live plots.

### Phase 5: Experiment Dashboard (experiment design)
Replace the current experiment UI with the full dashboard: phase timeline, channel editor, group manager. Save/load custom experiments.

### Phase 6: Parameter Sweep UI + Results (experiment analysis)
Build the in-app sweep panel with background execution, live results table, and auto-generated plots.

### Phase 7: Figure Export + Polish
Add matplotlib export path. Experiment history and comparison mode. CSV/JSON export. Layout persistence.

---

## 6. Testing Strategy

- **Module split:** All 38 existing tests pass. Headless scripts produce identical results. Run full benchmark suite.
- **Data bus:** Unit tests for pub/sub, ring buffer capacity, throttling.
- **Plots:** Manual verification (visual). Automated: data pipeline produces expected values.
- **Picker:** Unit test: render a known neuron layout, verify click at position returns correct index.
- **Sweep UI:** Compare results with headless `run_parameter_sweep.py` — identical metrics for same parameters.
- **Export:** Verify matplotlib figures render without error. Compare exported CSV with headless output.

---

## 7. What This Does NOT Include

- Multi-compartment neuron morphology (NEURON territory).
- Distributed simulation across multiple GPUs/machines (NEST territory).
- Jupyter notebook integration.
- Web-based UI.
- Educational tutorials or guided walkthroughs.

These are explicitly out of scope. The simulator's niche is the interactive, visual, GPU-accelerated single-machine experience.
