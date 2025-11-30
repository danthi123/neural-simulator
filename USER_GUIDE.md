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

---

## 5. Connectivity and Profiles

### 5.1 Neural Structure Profiles

Each profile encodes:
- Trait definitions (fractions and roles: excitatory vs inhibitory).
- Optional **connectivity motif** name.
- Optional **default HH neuron type** for HH model.
- An implicit mapping between profile and HH preset(s) used for realistic HH
  simulations.

Examples:
- `CORTEX_L23_RS_FS`
- `HIPPOCAMPUS_CA1_RS_FS`
- `HIPPOCAMPUS_CA3_RECURRENT`
- `THALAMUS_TC_TRN`
- `BASAL_GANGLIA_STRIATUM`
- `BASAL_GANGLIA_STN_GPE`

When using HH, profiles with `default_hh_neuron_type` will:
- Restrict the **HH Default Neuron Type** combo to only profile-compatible
  presets (usually a single biologically grounded choice per profile).
- Automatically set both the HH preset and UI HH parameter panel to match.
- Use profile-specific auto-tuned drive scales when available.

### 5.2 Connectivity Generators

Priority when initializing connections:
1. **Connectivity motif** (if profile defines one)
2. **Watts–Strogatz** (if `enable_watts_strogatz` is True)
3. **3D spatial generator** (fallback)

If any generator yields zero synapses, the simulator falls back to the spatial
generator to ensure a non-empty network.

---

## 6. Learning & Plasticity

- **Hebbian Learning**: Activity-dependent long-term weight updates.
- **Short-Term Plasticity (STP)**: Tsodyks–Markram `u` and `x` variables.
- **Homeostasis**:
  - For Izhikevich: adaptive thresholds via EMA of firing.
  - For HH: EMA of activity only (no threshold adjustment).

All are configurable in the **Learning & Plasticity** section.

---

## 7. Recording, Playback, and Checkpoints

### 7.1 Checkpoints (.simstate.h5)

- Use **File → Save Checkpoint** / **Load Checkpoint** or the corresponding
  toolbar buttons.
- Checkpoints store full simulation state: neuron potentials, conductances,
  connections, plasticity variables, etc.

### 7.2 Recordings (.simrec.h5)

- Use the **Record** / **Playback Recording** buttons.
- Recording is GPU-buffered and written to HDF5.
- Playback mode lets you scrub through frames using the playback slider and
  controls.

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
