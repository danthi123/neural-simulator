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

- **Simulation core**: `SimulationBridge` in `neural-simulator.py` manages all
  GPU arrays, connectivity, neuron dynamics, and stepping.
- **GPU backend**: CuPy + custom fused kernels for Izhikevich, Hodgkin–Huxley,
  AdEx, synaptic conductance decay, STP, and extended HH currents.
- **Connectivity**:
  - Spatial 3D connectivity (distance + trait-based bias)
  - Watts–Strogatz small-world generator
  - High-level **connectivity motifs** for specific brain-region profiles
- **UI**: DearPyGUI control pane drives configuration and monitoring.
- **Visualization**: PyOpenGL-based 3D point cloud of neurons with synapse lines
  and optional synaptic pulse effects.

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
  - `hh_q10_factor` (Q10 scaling)
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
- `BASAL_GANGLIA_STN_GPE` – Subthalamic nucleus and globus pallidus externa
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

## 9. Troubleshooting

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
