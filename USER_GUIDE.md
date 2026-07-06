# GPU-Accelerated Neural Network Simulator – User Guide

This document provides a complete overview of the simulator, including GUI usage,
command-line options (such as auto-tuning), and all major features.

## 1. Quick Start

### 1.1 Launching the GUI

```bash
python neural-simulator.py
```

Steps:
1. Wait for the DearPyGUI control window (left) and the OpenGL 3D window (right) to appear.
2. In the **Core Simulation Parameters** section, adjust:
   - **Number of Neurons**
   - **Connections/Neuron (Spatial Fallback)**
   - **Neuron Model** (Izhikevich, Hodgkin–Huxley, AdEx)
   - **Neural Structure Profile** (e.g. cortical, hippocampal, thalamic, basal ganglia)
   - **Note**: When switching to **Hodgkin–Huxley**, dt automatically adjusts to 0.05ms for numerical stability; it restores to 0.5ms when you switch away.
3. Click **"Apply Changes & Reset Sim"**.
4. Click **"Start"** in the **Simulation Controls** section.
5. Navigate the 3D scene with the mouse:
   - Left-drag: rotate
   - Right-drag: pan
   - Scroll: zoom

### 1.2 Headless Auto-Tuning

To automatically tune external drive scales for all model/profile/preset combinations:

```bash
# Full sweep (recommended to run once)
python neural-simulator.py --auto-tune

# Faster test sweep (subset only)
python neural-simulator.py --auto-tune --quick
```

This writes `simulation_profiles/auto_tuned_overrides.json`. When you next
use the GUI and select a tuned combination, the appropriate drive scales are
applied automatically when you click **Apply Changes & Reset Sim**.

In the GUI, look in the model-specific parameter panels for:

- **External Drive Scale (HH, auto-tuned)**
- **External Drive Scale (AdEx, auto-tuned)**

You can restore these to the auto-tuned values using:

- **Reset HH Drive to Auto-Tuned** button
- **Reset AdEx Drive to Auto-Tuned** button

Then press **Apply Changes & Reset Sim** to use the tuned values.

---

## 2. Architecture Overview

The simulator is split into focused packages (it was once a single large `neural-simulator.py`; it has since been refactored so that `neural-simulator.py` is just the GUI host and the engine lives in the modules under `sim/`):

- **Simulation core**: `SimulationBridge` in `sim/bridge.py` manages all GPU
  arrays, connectivity, neuron dynamics, and stepping. Configs are
  `@dataclass` instances in `sim/config.py`.
- **GPU backend**: CuPy + fused `@cp.fuse()` kernels in `sim/kernels.py` for
  Izhikevich, Hodgkin–Huxley, AdEx, synaptic conductance decay, STP, NMDA,
  STDP, homeostasis, eligibility traces, and extended HH currents.
- **Connectivity** (`sim/connectivity.py`):
  - Spatial 3D connectivity (distance + trait-based bias)
  - Watts–Strogatz small-world generator
  - High-level **connectivity motifs** for specific brain-region profiles
- **Brain-region framework** (`sim/regions.py`, opt-in): declarative
  multi-region simulations with `BrainRegion` + `RegionPathway`.
- **Neuromodulator subsystem** (`sim/neuromodulators.py`, opt-in): declarative
  concentration dynamics for DA / NE / 5-HT with receptor-effect targets.
- **Replicas + text I/O + visual cortex** (`sim/replicas.py`,
  `sim/text_embeddings.py`, `sim/visual_cortex.py`): replicated wiring
  for multi-network runs, language regions that turn words into neural input,
  and a model visual cortex built from edge/orientation detectors (Gabor
  filters, the standard model of early-visual-cortex receptive fields).
- **Progress events** (`sim/progress.py`): universal `[PROGRESS] {json}`
  event format consumed by the experiment runner and webapp.
- **Experiment system** (`experiment/` package): `ExperimentEngine`,
  `StimulusManager`, `ReadoutEngine`, `TrainingProtocolEngine`. Drives
  multi-phase experiments with stimulus injection and reward/eligibility
  training.
- **UI** (`ui/` package): DearPyGUI control pane drives configuration and
  monitoring.
- **Visualization** (`viz/` package): PyOpenGL-based 3D point cloud of neurons
  with synapse lines and optional synaptic pulse effects.
- **Webapp** (`webapp/` package): FastAPI + uvicorn dashboard for live
  monitoring, run history, and launcher control.
- **Research runners** (`research/runners/`): 26+ headless runners for
  research-gate experiments (G1 → G11) plus text I/O, perception, and
  diagnostic tools (`text_eval_*`, `permuted_label_check.py`,
  `eval_sanity_check.py`, `morning_briefing.py`, etc).
  See [README.md#research-runners](README.md#research-runners).
- **Sweep orchestration** (`research/experiment_runner.py`,
  `research/result_aggregator.py`): YAML-driven cross-condition
  experiments with built-in verdict aggregation.

---

## 3. Command-Line Options

### 3.1 GUI Mode (default)

```bash
python neural-simulator.py
```

Starts the GUI + OpenGL visualization. Use this for interactive configuration,
exploration, and visual debugging.

### 3.2 Auto-Tuning Mode (headless)

```bash
python neural-simulator.py --auto-tune [--quick]
```

- `--auto-tune` – run the headless tuner instead of the GUI.
- `--quick` – restricts tuning to a subset of profiles and presets for faster runs.

Output:
- `simulation_profiles/auto_tuned_overrides.json` with:
  - `tuned_combinations` keyed by `MODEL|PROFILE|HH_TYPE_OR_NONE`
  - `core_overrides` with fields like `hh_external_drive_scale` and
    `adex_external_drive_scale`
  - Tuning metrics (spike counts, rates, spiking neuron fraction, etc.)

At runtime, `SimulationBridge.apply_simulation_configuration_core` loads these
and applies any available overrides to the core configuration before
initialization.

---

## 4. GUI Panels and Controls

### 4.1 Simulation Monitor

Shows read-only runtime metrics:
- **Sim Time** (s)
- **Current Step** (time step index)
- **Spikes (Current Step)**
- **Avg Rate (Network)**
- **Plasticity Events** (Hebbian)
- **Visible Neurons / Synapses** (after filters)

### 4.2 Simulation Controls

- **Start** – start or restart the simulation.
- **Pause** – toggle pause/resume.
- **Stop** – stop the simulation.
- **Step (1ms)** – advance by 1 ms of simulation time.
- **Apply Changes & Reset Sim** – reinitialize the network based on current UI
  parameters (and auto-tuned overrides if present).
- **Simulation Speed** – real-time speed factor (0.01x–20x).

### 4.3 Core Simulation Parameters

- **Number of Neurons** – total neurons.
- **Connections/Neuron (Spatial Fallback)** – used when motif or WS generators
  produce no synapses.
- **Total Sim Time (ms)** – used for recorder and stopping logic.
- **Time Step dt (ms)** – integration time step.
- **Seed** – RNG seed (-1 = random).
- **Number of Traits** – number of population labels.
- **Neuron Model** – `IZHIKEVICH`, `HODGKIN_HUXLEY`, `ADEX`.
- **Neural Structure Profile** – high-level presets for region-specific traits,
  connectivity motifs, and (for HH) default HH presets.

### 4.4 Izhikevich Parameters

Visible when **Neuron Model = IZHIKEVICH**:
- Full set of 2007-formulation parameters (`C`, `k`, `vr`, `vt`, `vpeak`, `a`,
  `b`, `c_reset`, `d_increment`).
- Homeostasis-specific thresholds are configured under **Learning & Plasticity**.

### 4.5 Hodgkin–Huxley Parameters

Visible when **Neuron Model = HODGKIN_HUXLEY**:
- **HH Default Neuron Type** – preset selector (e.g. cortical L5 RS, thalamic
  relay, CA1/CA3 pyramidal, striatal MSN, STN, GPe, TRN, etc.).
- Base HH parameters:
  - `C_m`, `g_Na_max`, `g_K_max`, `g_L`
  - `E_Na`, `E_K`, `E_L`
  - `v_rest_hh`, `v_peak_hh`
- Extended currents:
  - M-current: `g_M_max`, `m_current_tau_ms`
  - CaT: `g_CaT_max`, `E_CaT`
  - Ih: `g_h_max`, `E_h`
  - NaP: `g_NaP_max`
- Kinetics:
  - `hh_q10_factor` (uniform Q10, applies to extended currents M/CaT/Ih/NaP only)
  - `hh_q10_m`, `hh_q10_h`, `hh_q10_n` (per-gate Q10 for main Na+/K+/leak; defaults 3.0/1.5/1.5 since 2026-04-25)
  - `hh_temperature_celsius`
- **External Drive Scale (HH, auto-tuned)**:
  - Scales the baseline HH DC input range.
  - Tuned per `(model, profile, HH preset)` by the auto-tuner if run.
- **Reset HH Drive to Auto-Tuned**:
  - If a tuned entry exists for the current combination, restores the slider to
    the tuned value.
  - You must then press **Apply Changes & Reset Sim** to apply it.

### 4.6 AdEx Parameters

Visible when **Neuron Model = ADEX**:
- `adex_C`, `adex_g_L`, `adex_E_L`, `adex_V_T`, `adex_Delta_T`
- `adex_a`, `adex_tau_w`, `adex_b`, `adex_V_r`, `adex_V_peak`
- **External Drive Scale (AdEx, auto-tuned)**:
  - Scales the AdEx DC input range.
  - Tuned per `(model=ADEX, profile)` by the auto-tuner.
- **Reset AdEx Drive to Auto-Tuned**:
  - Restores the slider to the tuned value for the current profile (if any).
  - Requires **Apply Changes & Reset Sim** to take effect in the sim.

### 4.7 Visualization Filtering

Control what neurons and synapses appear in the 3D view:
- **Filter by Neuron Type**: Show only neurons of specific traits (e.g., only
  excitatory, only inhibitory).
- **Filter by Spiking Mode**: Show only neurons that are currently spiking or
  subthreshold.
- **Min Weight Filter**: Hide synapses weaker than a threshold; useful for
  reducing visual clutter in dense networks.

Filters are applied in real-time without restarting the simulation.

### 4.8 Parameter Heterogeneity

Introduce variability within the neural population:
- **Enable**: Toggle heterogeneity on/off. When enabled, neuron parameters vary
  around their default values.
- **CV Coefficient**: Coefficient of variation (CV = std / mean). Controls the
  relative spread of parameters. Higher values = more diversity.
- Effects apply to neuron-specific parameters like resting potential, capacitance,
  and conductances (model-dependent).

### 4.9 Background Noise

Stochastic input to neurons (Ornstein–Uhlenbeck process):
- **Tau (ms)**: Time constant of noise decay; higher values = more autocorrelated.
- **Mean**: Average noise level.
- **Seed**: RNG seed for reproducibility (-1 = random).
- For Hodgkin–Huxley: Optional conductance-based noise can also be enabled,
  simulating background synaptic activity.

---

## 5. Connectivity and Profiles

### 5.1 Neural Structure Profiles

Each profile encodes:
- Trait definitions (fractions and roles: excitatory vs inhibitory).
- Optional **connectivity motif** name.
- Optional **default HH neuron type** for HH model.
- An implicit mapping between profile and HH preset(s) used for realistic HH
  simulations.

Predefined profiles:
- `CORTEX_L23_RS_FS` – Layer 2/3 cortex (regular-spiking and fast-spiking neurons)
- `CORTEX_L4_INPUT_LAYER` – Layer 4 cortex with lemniscal input characteristics
- `HIPPOCAMPUS_CA1_RS_FS` – CA1 region (pyramidal and interneurons)
- `HIPPOCAMPUS_CA3_RECURRENT` – CA3 with recurrent connectivity
- `THALAMUS_TC_TRN` – Thalamocortical and reticular nuclei
- `BASAL_GANGLIA_STRIATUM` – Striatal neurons
- `BASAL_GANGLIA_STN_GPE` – Subthalamic nucleus and globus pallidus externus
- `CEREBELLAR_CORTEX_SIMPLE` – Cerebellar cortex (Purkinje cells, granule cells, basket cells)
- `SPINAL_CORD_SEGMENT` – Spinal motor circuits
- `GENERIC_UNSTRUCTURED` – Uniform population with no explicit brain region structure

When using HH, profiles with `default_hh_neuron_type` will:
- Restrict the **HH Default Neuron Type** combo to only profile-compatible
  presets (usually a single biologically grounded choice per profile).
- Automatically set both the HH preset and UI HH parameter panel to match.
- Use profile-specific auto-tuned drive scales when available.

### 5.2 Profile Dropdown System

The **Neural Structure Profile** dropdown in the GUI automatically populates from
all `.json` files in the `simulation_profiles/` directory. This lets you:

- Select a complete brain region configuration (traits, connectivity, HH defaults)
  with a single click.
- Add custom profiles by placing new `.json` files in `simulation_profiles/`.
- Use the **Refresh** button to rescan the directory after adding new profiles.

Each profile file specifies trait definitions, connectivity motifs, and optional
HH neuron type defaults, keeping your simulation setup consistent and reusable.

### 5.3 Connectivity Generators

Priority when initializing connections:
1. **Connectivity motif** (if profile defines one)
2. **Watts–Strogatz** (if `enable_watts_strogatz` is True)
3. **3D spatial generator** (fallback)

If any generator yields zero synapses, the simulator falls back to the spatial
generator to ensure a non-empty network.

---

## 6. Learning & Plasticity

### 6.1 Hebbian Learning

Activity-dependent long-term weight updates. Pre- and post-synaptic activity
drives weight changes over longer timescales (typically seconds).

### 6.2 Short-Term Plasticity (STP)

Tsodyks–Markram model with `u` (utilization) and `x` (recovery) variables.
Synapses exhibit facilitation or depression on millisecond timescales.

**Per-Connection-Type STP**: Enable `enable_per_type_stp` to use experimentally validated,
connection-type-specific STP profiles. Configure separate `stp_U`, `stp_tau_d`, and `stp_tau_f`
for each of the four synapse types: E->E, E->I, I->E, I->I. Different brain regions exhibit
distinct STP signatures; this feature allows region-specific realism (e.g., short-term facilitation
in L4→L2/3 pyramidal synapses).

### 6.3 Spike-Timing-Dependent Plasticity (STDP)

Refines synaptic weights based on the precise timing between pre- and post-synaptic
spikes. Configurable parameters:
- **A+, A−**: Magnitude of potentiation and depression.
- **τ+, τ−**: Time windows (ms) for potentiation and depression.
- **Weight Min/Max**: Bounds on synaptic strength.
- **Nearest-Spike Mode**: Whether to use nearest spikes or all spike pairs.

### 6.4 Reward-Modulated Plasticity

Three-factor learning: combines STDP with an external reward signal (e.g.,
dopamine). Parameters:
- **Learning Rate**: Overall strength of reward modulation.
- **Eligibility Tau**: Time window (ms) for eligibility trace decay.
- **Reward Baseline**: Expected reward level.
- **Current Reward Signal**: Externally supplied reward (updated in real-time).

### 6.5 Structural Plasticity

Activity-dependent synapse formation and elimination. New synapses grow when
neurons are coactive; weak or unused synapses are pruned. Parameters:
- **Formation Rate**: Probability per timestep of creating new synapses.
- **Elimination Rate**: Probability per timestep of removing weak synapses.
- **Weight Threshold**: Minimum strength to prevent elimination.
- **Target Density**: Desired fraction of possible connections.
- **Distance Scale**: Spatial range for new synapse formation.
- **Update Interval**: How often (in timesteps) to check for formation/elimination.
- **Activity-Dependent Bias** (`struct_plast_activity_bias`): Controls how much synapse formation
  favors co-active neuron pairs. Range 0.0–1.0, default 0.5. 0 = random formation; 1 = fully
  activity-driven (Hebbian structure). Based on Cline & Haas (2008) model of activity-dependent
  neurite outgrowth.

### 6.6 Synaptic Scaling

Homeostatic multiplicative scaling: normalizes total incoming weight per neuron
to maintain activity within a target range. Parameters:
- **Rate Parameter**: Speed of scaling adjustment.

### 6.7 NMDA Receptors

Voltage-dependent Mg²⁺ block adds biological realism to synaptic transmission.
Parameters:
- **Ratio**: Fraction of current carried by NMDA.
- **Tau Decay**: NMDA current decay time constant (ms).
- **Tau Rise**: NMDA current rise time constant (ms).
- **Mg Concentration**: Extracellular Mg²⁺ concentration (mM); higher values
  increase block at rest.

### 6.8 Homeostasis

Prevents runaway activity and maintains network balance. Timescales updated to match
biological observations (seconds-to-minutes):
- **For Izhikevich**: Adaptive thresholds via exponential moving average (EMA)
  of firing rate (tau ~5 seconds at dt=1ms). If firing exceeds target, threshold is raised.
- **For Hodgkin–Huxley**: EMA-based activity monitoring only; no direct threshold
  adjustment.

All plasticity mechanisms are configurable in the **Learning & Plasticity** panel
of the GUI.

---

## 7. Recording, Playback, and Checkpoints

### 7.1 Checkpoints (.simstate.h5)

- Use **File → Save Checkpoint** / **Load Checkpoint** or the corresponding
  toolbar buttons.
- Checkpoints store full simulation state: neuron potentials, conductances,
  connections, plasticity variables, etc.
- Restart the simulation from any checkpoint to pick up where you left off.

### 7.2 Recordings (.simrec.h5)

Use the **Record** / **Playback Recording** buttons to capture simulation frames
for later analysis or visualization.

**Recording Modes:**
- **GPU Buffered** (default): Frames accumulate in GPU memory, then batch-written
  to disk. Faster and less CPU overhead; requires sufficient VRAM.
- **Streaming**: Frames written to disk frame-by-frame. Slower but lower memory
  footprint; use for very long simulations.

**Compression Options:**
- **LZ4** (default): Fast compression; good balance of speed and space savings.
- **GZIP**: Maximum compression; slower write time; useful for archival.
- **None**: No compression; fastest writes; largest file size.

**Playback:**
- Use the playback slider and controls to scrub through recorded frames.
- Hover over timeline to preview frame numbers.
- Press Play to animate through the recording.

**Memory Management:**
- Adjust **Memory Pool Fraction** (default 0.8) in GPU settings to control how
  much VRAM is reserved for CuPy operations.
- For recordings of long simulations, increase the **Viz Update Interval** to
  record fewer frames and reduce memory usage.

---

## 8. Keyboard & Mouse Shortcuts

See the **Keyboard Shortcuts** section in `README.md` for a concise list.

---

## 9. Headless / Research Workflows

The simulator can be driven entirely without the GUI. There are four headless
entry points:

### 9.1 Run a built-in experiment preset

```bash
python run_experiment_headless.py --preset rl --seed 42
```

Available presets: `stim` (basic stimulus-response), `associative`
(Pavlovian conditioning — pairing a neutral cue with a meaningful one),
`rl` (reward-driven reinforcement learning), `freq` (frequency-response
sweep). Output goes to `experiment_<preset>_<timestamp>.json`.

### 9.2 Run a parameter sweep

```bash
python run_parameter_sweep.py -e associative \
    --sweep "stdp_a_plus=0.004,0.012,0.024"
```

Runs a grid (or zip with `--zip`) of parameter combinations. Output JSON +
CSV includes Welch's t-test and Cohen's d for each condition.

### 9.3 Biological validation suite

```bash
python run_benchmarks.py --benchmark stdp-timing
python run_benchmarks.py --benchmark ei-balance
python run_benchmarks.py --benchmark stp-paired-pulse
python run_benchmarks.py --benchmark gamma-oscillations
python run_benchmarks.py --benchmark homeostasis
```

Each benchmark reproduces a known neuroscience result: the measured spike-timing
learning curve (Bi & Poo 1998), the balance between excitation and inhibition in
cortex, the way repeated synaptic pulses grow or shrink (paired-pulse ratio),
gamma-band (~40 Hz) network rhythms, and homeostatic (self-stabilizing)
adaptation of firing rates. All currently pass.

### 9.4 Research-gate runners

For specific architectural experiments. Each runner writes raw data to
`research/findings/raw/gN/` and a markdown summary to `research/findings/`.

```bash
# Encoder-decoder roundtrip (G1)
python -m research.runners.g1_v3_runner --seed 42

# Sensorimotor signed perceptron (G5)
python -m research.runners.g5_v3_runner --seed 42

# Moving-goal reinforcement learning with random motor exploration (G9)
python -m research.runners.g9_runner --seed 42 --motor-exploration-rate-hz 15

# Basal-ganglia action-selection circuit (G11) - moving-goal stress test
python -m research.runners.g11_bg_runner --moving-goal --seed 42 --n-steps 1800

# Basal-ganglia circuit - static single-action probe
python -m research.runners.g11_bg_runner --probe-action W

# Text input/output training + evaluation (learning to map a written
# word to the matching motor action)
python -m research.runners.text_train_embodied --seed 42
python -m research.runners.text_eval_embodied --seed 42

# Diagnostic: is the word-to-action signal real learning, or just
# random structure that happens to look right?
python -m research.runners.permuted_label_check <eval_file.json>

# Eval methodology validation via hand-built perfect weights
python -m research.runners.eval_sanity_check --seed 42

# Morning briefing — summarize overnight runs
python -m research.runners.morning_briefing
```

See [README.md#research-runners](README.md#research-runners) for the full
runner status table.

### 9.5 YAML-driven sweeps

For multi-condition experiments with built-in cross-condition aggregation:

```bash
# Run a YAML sweep
python -m research.experiment_runner experiments/biology_sweep.yaml

# Aggregate cross-condition results with verdict line
python -m research.result_aggregator <output_dir>
```

Built-in configs cover biology, minimum_biology, sanity_check,
b2_sparse_codes, and b4_long_training. Throughput on the 7-8x speedup
stack (dt=1.0 + parallel-3 GPU sharing + `cfg.fast_spike_reset`) brings a
6-seed batch from ~6 hours down to ~45-55 minutes. See
`research/findings/2026-05-04-perf-speedup-stack.md`.

---

## 10. Brain-Region Framework (opt-in)

For multi-region simulations (cortex + striatum + thalamus + motor on one
bridge), set `cfg.enable_brain_region_framework = True` and declare regions
+ pathways:

```python
from sim.regions import BrainRegion, RegionPathway

regions = [
    BrainRegion(name="cortex", n_neurons=400, exc_fraction=0.8,
                internal_density=0.1, exc_weight_mean=2.0),
    BrainRegion(name="striatum", n_neurons=200, exc_fraction=0.05,
                izh_neuron_type="IZH2007_STRIATAL_MSN"),
    BrainRegion(name="thalamus", n_neurons=100, exc_fraction=0.85,
                izh_neuron_type="IZH2007_THALAMIC_RELAY"),
]
pathways = [
    RegionPathway(from_region="cortex", to_region="striatum",
                  density=0.5, weight_mean=2.5, plastic=True),
    RegionPathway(from_region="striatum", to_region="thalamus",
                  density=0.8, weight_mean=5.0, plastic=False),
]

cfg.brain_regions = regions
cfg.region_pathways = pathways
cfg.num_traits = 1  # let regions own their type assignment
```

Each region gets a contiguous slice of the global neuron index space.
Pathways are CSR sparse and respect the per-pathway `plastic` flag (only
plastic pathways accept STDP / reward updates; others are frozen).

See `research/runners/g11_bg_runner.py` for a full basal-ganglia
action-selection build (30 regions, 32 pathways) and
`docs/plans/2026-04-24-brain-region-framework.md` for the design doc.

---

## 11. Neuromodulator Subsystem (opt-in)

For declarative dopamine / NE / 5-HT modeling, set
`cfg.enable_neuromodulator_subsystem = True` and add `NeuromodulatorConfig`
entries to `cfg.neuromodulators`:

```python
from sim.neuromodulators import NeuromodulatorConfig, ModulatorTarget, ProductionRule

dopamine = NeuromodulatorConfig(
    name="DA",
    baseline=0.0,
    decay_tau_ms=500.0,
    targets=[
        ModulatorTarget(target_type="plasticity_rate", scope="all", sensitivity=2.0),
    ],
    production_rules=[
        ProductionRule(rule_type="from_reward", sensitivity=1.0),
    ],
)
cfg.neuromodulators = [dopamine]
```

Effects available: `synaptic_gain`, `plasticity_rate`, `excitability_drive`.
Production rules: `manual` (set externally), `from_reward` (proportional to
`current_reward_signal`), `from_error_persistence` (EMA of |error| above
threshold).

Composes with the brain-region framework — when both are on, regions
auto-register as neuromodulator groups so
`ModulatorTarget(scope="group:PFC", ...)` resolves natively.

See `docs/plans/2026-04-24-neuromodulator-subsystem.md`.

---

## 12. Action Selection with a Basal-Ganglia Circuit

The `g11_bg_runner.py` script models the **basal ganglia** — the brain's
action-selection hub, the circuit that decides which one of several possible
moves to actually make. This circuit design resolved the "silent-motor trap"
(where the movement neurons for some actions never fire, so those actions can
never be learned or chosen).

**Architecture:** each candidate action has its own chain of neuron groups
running cortex → striatum → output nuclei → thalamus → motor. (The striatum
is the input stage of the basal ganglia; the thalamus is the relay that
releases a movement.) The circuit works by *disinhibition*: the output nuclei
normally suppress the thalamus, and choosing an action means briefly lifting
that suppression for one action's chain so its movement neurons fire. Each
action has its own gate, so the winner emerges from independent competing
gates rather than from an off-brain "pick the highest score" step.

**Two gotchas worth knowing if you build a similar circuit:**

1. **Pool size matters at deployment, not at probe scale.** If the group of
   cortex neurons feeding the striatum is too large relative to the striatum,
   it over-drives the input stage into saturation (firing so fast that the
   downstream suppression can no longer be lifted, so no action is selected).
   Sizing the cortex group down to roughly a physiological firing rate makes
   the circuit work. **Lesson:** any quick test must call the same builder
   with the same arguments as the full deployed run.

2. **The spike-timing learning rule has an upper weight bound you must set
   correctly.** That rule (STDP, defined in Section 6.3) is *soft-bounded*: it
   pulls weights toward a configured maximum, `stdp_w_max`. If a connection's
   designed weight is already *above* that maximum, every learning event pushes
   it sharply down instead of up, and the weight collapses within milliseconds.
   So set `cfg.stdp_w_max` above the largest design weight in any learning
   connection (for example, a connection built at weight 25 needs
   `stdp_w_max = 30`, not the default of 2).

**Result (3 random seeds, 1800 steps):** the agent settles about 1.7 grid
cells from the goal on average (measured as grid-step, or "city-block",
distance), versus about 5.5 for random wandering — a 74% improvement over the
earlier baseline circuit.

See `research/findings/2026-04-25-phase-b-acid-test-real-win.md` for the
full diagnosis.

### 12.1 Current navigation flagship (2026-05-01)

The action-selection circuit above was the foundation. The current best
navigation setup adds several biologically motivated extensions: a closed
feedback loop through the basal ganglia, a topographic (spatially organized)
map in cortex, a slow synaptic channel (NMDA receptors) in the cortex, motor,
and prefrontal regions that helps the network hold a decision steady, and a
model visual cortex built from edge/orientation detectors:

```bash
python -m research.runners.g11_bg_runner --moving-goal --goal-schedule multi --deterministic \
    --enable-msn-lateral-inhibition --enable-d1-d2-asymmetry --enable-striatal-pv-fsi \
    --enable-cluster-a-closed-loop --enable-cluster-e-topography \
    --enable-dlpfc-wm --enable-pfc-nmda \
    --enable-visual-cortex --visual-cortex-action-warmup-steps 600 \
    --grid-size 16 --seed N --n-steps 1800
```

Result on a 16×16 grid using simulated vision only (no built-in navigation
rule, and no direct access to the goal's coordinates): the agent ends about
3.0 grid cells from the goal on average across 3 random seeds — better than an
earlier setup managed on a grid one-quarter the size, while removing four of
the five information shortcuts the agent used to be given. See
`research/findings/2026-05-01-cluster-k-v2-breakthrough.md` for details.

---

## 13. Troubleshooting

Common issues:
- **No visible spikes or pulses**:
  - Ensure you clicked **Apply Changes & Reset Sim** after modifying params.
  - Run `--auto-tune` to generate tuned drive scales.
  - Check that **Neuron Model**, **Neural Structure Profile**, and (for HH)
    **HH Default Neuron Type** match a tuned combination.
- **Too much activity (seizure-like)**:
  - Reduce external drive scales.
  - Increase inhibitory propagation strength.
- **Out of memory**:
  - Reduce neuron count or connections per neuron.
  - Increase `Viz Update Interval` or reduce `Max Visible Neurons` / `Max Visible Connections`.
- **The action-selection circuit dies after the first trial in a research
  runner**:
  - Check `cfg.stdp_w_max` is set above your largest design weight for any
    learning pathway. The soft-bounded spike-timing rule collapses oversize
    weights silently (see Section 12).
  - Check the size ratio between a source neuron group and its target — an
    over-large source over-drives the target into saturation, which breaks the
    action-selection gating (see Section 12).
- **HH presets don't fire at 37°C**:
  - Use the per-gate Q10 values (`hh_q10_m=3.0`, `hh_q10_h=1.5`,
    `hh_q10_n=1.5`). Uniform Q10=3 over-compresses gating dynamics. The
    defaults in `sim/config.py` are correct as of 2026-04-25.
- **AdEx presets all behave the same**:
  - Verify the bridge is overlaying preset params onto `cfg.adex_*` fields.
    Fixed 2026-04-25; older builds bypassed preset loading.
