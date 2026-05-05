# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Repository**: https://github.com/danthi123/neural-simulator

## Project Overview

GPU-accelerated neural network simulator with real-time 3D OpenGL visualization. Uses NVIDIA CUDA/CuPy for massively parallel GPU computation, simulating large-scale networks (10K-100K+ neurons) with biologically-inspired neuron models (Izhikevich, Hodgkin-Huxley, AdEx), synaptic plasticity, and spatial connectivity.

## Common Commands

```bash
# Run the simulator (GUI mode)
python neural-simulator.py

# Run headless auto-tuning (parameter sweep)
python neural-simulator.py --auto-tune
python neural-simulator.py --auto-tune --quick  # Faster reduced sweep

# Run performance benchmarks
python benchmark.py --output results.json
python benchmark.py --quick  # Reduced configurations

# Run visualization performance benchmark
python viz_benchmark.py --output benchmarks/viz_performance_results.json
python viz_benchmark.py --quick  # Faster test

# Run biological validation suite (Bi & Poo STDP, E/I balance, STP PPR, gamma, homeostasis)
python run_benchmarks.py --benchmark stdp-timing
python run_benchmarks.py --benchmark ei-balance
python run_benchmarks.py --benchmark stp-paired-pulse
python run_benchmarks.py --benchmark gamma-oscillations
python run_benchmarks.py --benchmark homeostasis

# Run a research-gate runner (G1..G11)
python -m research.runners.g11_bg_runner --moving-goal --seed 42 --n-steps 1800 \
    --out research/findings/raw/g11_bg/g11_seed42.json
python -m research.runners.g11_bg_runner --probe-action W   # static cascade probe

# Run headless experiments (4 built-in presets)
python run_experiment_headless.py --preset rl --seed 42

# Parameter sweep
python run_parameter_sweep.py -e associative --sweep "stdp_a_plus=0.004,0.012,0.024"

# Run all tests
pytest tests/ -v

# Targeted test suites
pytest tests/test_determinism.py -v
pytest tests/test_experiment_system.py -v
pytest tests/test_neuromodulators.py -v
pytest tests/test_regions.py -v
```

## Architecture

### Modular Package Layout

The simulator was originally a single ~12K-line `neural-simulator.py`. As of 2026-04 it is refactored into modular packages — `neural-simulator.py` is now just the GUI host (2.2K lines), and the engine lives in `sim/`.

```
neural-simulator.py     # 2.2K lines — DearPyGUI host + main entry point only
sim/                    # 13 modules, ~11.8K lines — core engine
  bridge.py             # 6037 lines — SimulationBridge + GPU state orchestration
  config.py             #  718 lines — all @dataclass configs
  enums.py              #  825 lines — NeuronType (50+ presets), enums, default param managers
  connectivity.py       #  988 lines — spatial/WS/motif connection generators (GPU)
  kernels.py            #  314 lines — fused @cp.fuse() neuron + plasticity kernels
  profiles.py           #  432 lines — NEURAL_STRUCTURE_PROFILES + CONNECTIVITY_MOTIFS dicts
  regions.py            #  602 lines — BrainRegion + RegionPathway + RegionManager
  neuromodulators.py    # 1051 lines — declarative neuromodulator subsystem
  data_bus.py           #   95 lines — DataChannel pub/sub for streaming sim data
  replicas.py           #  243 lines — replicated wiring (multi-bridge support)
  text_embeddings.py    #  147 lines — token embeddings for language regions (2026-05-01)
  visual_cortex.py      #  310 lines — Gabor RFs + retina rendering (Cluster K v2, 2026-05-01)
viz/                    # OpenGL renderer, camera, picker, overlays
ui/                     # DearPyGUI panels, callbacks, layout, sweep panel, plots
experiment/             # ExperimentEngine + StimulusManager + ReadoutEngine + TrainingProtocolEngine
research/runners/       # 26 headless runners (g1..g11 + cluster/text/k_v2/etc) for research
research/findings/      # session-by-session findings docs (93+ files)
tests/                  # 40 test files (determinism, runners, kernels, plasticity, etc.)
```

### Thread Model
- **Main Thread**: DearPyGUI event loop + OpenGL rendering
- **Simulation Thread**: GPU-accelerated neural dynamics computation (fully isolated)
- **Communication**: Lock-free queues (`ui_to_sim_queue`, `sim_to_ui_queue`) for inter-thread messaging

### Key Classes

**SimulationBridge** (`sim/bridge.py:170`): Central simulation orchestrator
- Manages all GPU state arrays (CuPy)
- Simulation stepping (`_run_one_simulation_step` at line 4210)
- Initialization (`_initialize_simulation_data` at line 823)
- Recording/playback to HDF5
- Checkpoint save/restore
- Profiling and performance monitoring

**Configuration Dataclasses** (all in `sim/config.py`):
- `CoreSimConfig` (line 27): Network topology, neuron models, plasticity, biological realism
  - STP fields: `stp_U`, `stp_tau_d`, `stp_tau_f` (global defaults)
  - Per-connection-type STP: `enable_per_type_stp`, `stp_U_per_type[4]`, `stp_tau_d_per_type[4]`, `stp_tau_f_per_type[4]`
  - Structural plasticity: `struct_plast_activity_bias` (0.0–1.0) for activity-dependent synaptogenesis
  - Homeostasis: EMA alpha (~0.0002, tau ~5s) and threshold adapt rate (~0.0005)
  - Inhibitory reversal: `E_inh = -75mV`, propagation scaled 0.7x for driving force compensation
  - HH numerical stability: dt auto-adjusts to 0.05ms when HH model selected
  - **Per-gate Q10**: `hh_q10_m=3.0`, `hh_q10_h=hh_q10_n=1.5` (fixed 2026-04-25 — uniform Q10=3 over-compressed dynamics at 37°C; see Phase A below)
  - **STDP bounds gotcha**: `stdp_w_max=2.0` default. The STDP rule is **soft-bound** (`Δw_LTP = A_plus * (w_max - w) * exp(...)`) so when `weight_mean > stdp_w_max`, every "LTP" event is strongly negative and weights collapse to w_max within ms. Set `cfg.stdp_w_max` above your design weights (e.g. cortex→D1 in Phase B uses `weight_mean=25` → set `stdp_w_max=30`).
- `VisualizationConfig` (line 292): OpenGL rendering and camera parameters
- `RuntimeState` (line 312): Mutable execution state (running, paused, time tracking)
- `GPUConfig` (line 327): GPU features, memory management, recording modes
- Experiment configs (lines 440–619): `StimulusPattern`, `StimulusChannel`, `NeuronGroup`, `ReadoutConfig`, `TrainingConfig`, `ExperimentPhase`, `ExperimentConfig`

### GPU Array Naming Conventions
- `cp_*`: CuPy GPU arrays (e.g., `cp_membrane_potential_v`, `cp_firing_states`)
- `gl_*`: OpenGL handles/VBOs
- `fused_*`: GPU kernel functions decorated with `@cp.fuse()`

### Simulation Step Pipeline (in `_run_one_simulation_step()`)
1. STP (Short-Term Plasticity) update – per-connection-type if enabled
2. Synaptic conductance update – uses E_inh = -75mV with 0.7x propagation scaling
2b. **Experiment stimulus injection** – if ExperimentEngine is running, adds stimulus current
3. Background noise (OU process)
4. Neuron dynamics (model-specific: Izhikevich/HH/AdEx)
5. Plasticity updates (Hebbian, STDP, reward modulation, structural with activity bias, homeostasis)
6. Visualization updates
7. Recording (if active)

**Note on dt Auto-Adjustment**: When switching to Hodgkin–Huxley model, dt is automatically
reduced to 0.05ms for numerical stability of voltage-gated kinetics. When switching to Izhikevich
or AdEx, dt restores to 0.5ms. This occurs in `apply_simulation_configuration_core()`.

### Fused CUDA Kernels (`sim/kernels.py`)
Performance-critical GPU operations decorated with `@cp.fuse()`:
- `fused_izhikevich2007_dynamics_update()`: 9-parameter Izhikevich model
- `fused_izhikevich_legacy_dynamics_update()`: Legacy 4-param Izhikevich
- `fused_hodgkin_huxley_dynamics_update()`: Temperature-dependent HH
- `fused_adex_dynamics_update()`: Adaptive Exponential IF
- `fused_hh_m_current_update()`, `fused_hh_CaT_current_update()`, etc.: Extended HH currents
- `fused_hh_h_current_update()`: HH h-current (Ih)
- `fused_hh_NaP_current_update()`: HH persistent sodium current
- `fused_conductance_decay_and_current()`: Synaptic dynamics
- `fused_nmda_update_and_current()`: NMDA voltage-dependent Mg2+ block
- `fused_stp_decay_recovery()`: Short-term plasticity Tsodyks-Markram
- `fused_stdp_weight_update()`: Spike-timing dependent plasticity
- `fused_homeostasis_update()`: Homeostatic firing rate regulation
- `fused_eligibility_trace_decay()`: Reward modulation eligibility traces

### Hodgkin-Huxley Presets (`sim/enums.py`)
Region-specific HH parameter dicts in `DefaultHodgkinHuxleyParams`, derived from `REALISTIC_L5_PYRAMIDAL_RS_37C` base with region overrides. All retuned 2026-04-25 (per-gate Q10 fix):

Cortical: `HH_L5_CORTICAL_PYRAMIDAL_RS`, `HH_PFC_PYRAMIDAL`, `HH_CORTICAL_FS_INTERNEURON`
Hippocampus: `HH_CA1_PYRAMIDAL_BURST`, `HH_CA3_PYRAMIDAL_BURST`
Thalamus: `HH_THALAMIC_RELAY_TBURST`, `HH_TRN_BURST_INHIB`
Basal ganglia: `HH_STRIATAL_MSN`, `HH_STRIATAL_MSN_D1`, `HH_STRIATAL_MSN_D2`, `HH_STRIATAL_TAN`, `HH_STN_BURST`, `HH_GPE_PACEMAKER`, `HH_GPI_OUTPUT`, `HH_DOPAMINE_SNC`
Cerebellum: `HH_CEREBELLAR_PURKINJE`, `HH_CEREBELLAR_GRANULE`
Spinal: `HH_SPINAL_MOTOR`, `HH_SPINAL_INTERNEURON`
Other: `HH_OLFACTORY_MITRAL`, `HH_INFERIOR_OLIVE`

### Izhikevich 2007 Presets (`sim/enums.py`)
9-parameter Izhikevich-2007 presets in `DefaultIzhikevichParamsManager`, used by `cfg.default_neuron_type_izh` and per-region `izh_neuron_type` overrides. Cortical: `IZH2007_RS_CORTICAL_PYRAMIDAL`, `IZH2007_FS_CORTICAL_INTERNEURON`. BG: `IZH2007_STRIATAL_MSN`, `IZH2007_STRIATAL_MSN_D1`, `IZH2007_STRIATAL_MSN_D2`, `IZH2007_STRIATAL_TAN`, `IZH2007_GPE_PACEMAKER`, `IZH2007_GPI_OUTPUT`, `IZH2007_STN_BURST`, `IZH2007_DOPAMINE`. Thalamus: `IZH2007_THALAMIC_RELAY`, `IZH2007_THALAMIC_RETICULAR`. Hippo: `IZH2007_HIPPO_PYRAMIDAL`. (Fixed 2026-04-25: bridge previously ignored `default_neuron_type_izh` because trait-split was always-on; now opt-in only when `num_traits > 1`.)

### AdEx Presets (`sim/enums.py`)
Brette & Gerstner 2005 phenotypes in `DefaultAdExParamsManager`: `ADEX_RS`, `ADEX_FS`, `ADEX_IB`, `ADEX_CH`, `ADEX_LTS`, `ADEX_MSN`, `ADEX_DOPAMINE`. (Fixed 2026-04-25: bridge now overlays preset params onto `cfg.adex_*` fields — previously all 7 presets behaved identically because preset wasn't loaded.)

### Neural Structure Profiles (`sim/profiles.py`)
Region presets that configure traits, connectivity, and default parameters. Defined in the `NEURAL_STRUCTURE_PROFILES` dict:
- `GENERIC_UNSTRUCTURED`
- `CORTEX_L23_RS_FS`, `CORTEX_L4_INPUT_LAYER`, `CORTEX_L5_DEEP_OUTPUT`
- `PREFRONTAL_CORTEX_WM`
- `HIPPOCAMPUS_CA1_RS_FS`, `HIPPOCAMPUS_CA3_RECURRENT`
- `BASAL_GANGLIA_STRIATUM`, `BASAL_GANGLIA_STN_GPE`
- `THALAMUS_TC_TRN`
- `CEREBELLAR_CORTEX_SIMPLE`, `SPINAL_CORD_SEGMENT`
- `OLFACTORY_BULB`, `DOPAMINERGIC_MIDBRAIN`
- `CORTEX_GAMMA_FS_NETWORK`, `INFERIOR_OLIVE`

### Profile Naming Convention
Each brain region has three JSON profile variants in `simulation_profiles/`:
- `{region}_hh.json`: Full biophysics (Hodgkin-Huxley, dt=0.05ms)
- `{region}_adex.json`: Adaptive Exponential (dt=0.5ms, 10-20× faster than HH)
- `{region}_izh.json`: Izhikevich fast testing (dt=1.0ms, fastest)
- Plus `quick_demo_cortex.json` for beginners

### JSON Profile Dropdown System (`ui/`)
Full simulation profiles saved as `.json` in `simulation_profiles/` (47 files). A UI dropdown auto-populates from this directory. Key functions live in the UI package: `_scan_profile_directory()`, `_handle_full_profile_dropdown_change()`, `_refresh_full_profile_dropdown()`.

### UI-Config Roundtrip
Two critical functions must be kept in sync for profile save/load to work correctly:
- `_update_sim_config_from_ui()`: Extracts all parameter values from UI widgets and builds `CoreSimConfig`, `VisualizationConfig`, `RuntimeState`, and `GPUConfig` dataclasses
- `_populate_ui_from_config_dict()`: Takes a configuration dictionary and updates all UI widgets to reflect those values

These are inverse operations: any parameter exposed in the UI must have a corresponding getter and setter to ensure bidirectional sync between UI state and simulation configuration.

### Experiment & Stimulus System (`experiment/` package)
Programmable infrastructure for stimulus injection, I/O neuron group management, training protocols, readout/analysis, and multi-phase experiment orchestration. Configs live in `sim/config.py` (lines 440–619); engines live in `experiment/`.

**Key Classes:**
- `StimulusManager` (`experiment/stimulus.py`): Generates per-step GPU current arrays from channel definitions
- `NeuronGroupManager` (`experiment/groups.py`): Manages designated populations (input/output/hidden)
- `ReadoutEngine` (`experiment/readout.py`): Population rates, spike counts, PSD via FFT, Fano synchrony, band power
- `TrainingProtocolEngine` (`experiment/training.py`): Trial state machine for RL/supervised/associative
- `ExperimentEngine` (`experiment/engine.py`): Top-level orchestrator called once per simulation step
- `ExperimentPresets` (`experiment/presets.py`): Factory for 4 common experiment configurations

**Stimulus Pattern Types:** CONSTANT, PULSE_TRAIN, SINUSOIDAL, RAMP, POISSON_SPIKE_TRAIN, GAUSSIAN_NOISE, CUSTOM_WAVEFORM

**Training Modes:** ASSOCIATIVE_PAIRING (Rescorla-Wagner), REINFORCEMENT_LEARNING (R-STDP), SUPERVISED_TARGET, RESERVOIR_READOUT

**Built-in Presets:**
- Basic Stimulus-Response: inject current, measure output transfer function
- Associative Conditioning (CS-US): Pavlovian pairing with STDP learning
- Reinforcement Learning (R-STDP): Three-factor learning with reward/punishment
- Frequency Response Characterization: Sinusoidal sweep for bandpass analysis

**Integration Points:**
- SimulationBridge: `self.experiment_engine` initialized in `apply_simulation_configuration_core()`
- Simulation step: experiment stimulus injected after synaptic current, before OU noise
- Queue messages: LOAD_EXPERIMENT_PRESET, LOAD_EXPERIMENT_CONFIG, START_EXPERIMENT, STOP_EXPERIMENT, GET_EXPERIMENT_STATUS, SAVE_EXPERIMENT_LOG
- Checkpoint: experiment config saved/restored as JSON attribute in HDF5
- UI: "Experiment & Stimulus System" collapsing header with preset selector, controls, status display

**Running Tests:**
```bash
pytest tests/test_experiment_system.py -v
```

### Neuromodulator Subsystem (Session E.1, opt-in)

Declarative framework for hormones / neuromodulators with concentration
dynamics and configurable receptor effects on bridge state. Replaces
the one-off `current_reward_signal` and shelved `cp_synaptic_gain_modulator`
mechanisms. Default OFF for full backward compatibility.

**Module:** `sim/neuromodulators.py`

**Config (in `CoreSimConfig`):**
- `enable_neuromodulator_subsystem: bool = False` — opt-in flag
- `neuromodulators: List[NeuromodulatorConfig]` — list of declared modulators

**Three dataclasses:**
- `NeuromodulatorConfig(name, baseline, decay_tau_ms, concentration_min/max, targets, production_rules)`
- `ModulatorTarget(target_type, scope, sensitivity)` — receptor effect spec
- `ProductionRule(rule_type, sensitivity, threshold, window_ms)` — what drives concentration

**Built-in target types:**
- `synaptic_gain` — multiplies effective synaptic strength (scope=all only)
- `plasticity_rate` — multiplies reward_learning_rate (scope=all)
- `excitability_drive` — adds pA to membrane drive (scope=all, trait:N, group:NAME)

**Built-in production rules:**
- `manual` — only set externally (testing, experiments)
- `from_reward` — adds sensitivity*(current_reward_signal - reward_baseline) per step
- `from_error_persistence` — EMA of |error| > threshold drives sustained tonic increase

**Bridge integration:**
- Manager allocated in `_init_synapse_arrays_with_capacity` when subsystem enabled
- `manager.step(self)` called once per simulation step after C2 reward modulation
- `compute_synaptic_gain_multiplier()` applied in `effective_synaptic_strength`
- `compute_plasticity_rate_multiplier()` applied to `reward_learning_rate`
- `compute_excitability_drive_pA()` + `compute_excitability_drive_per_neuron()` added to `total_input_current_pA`

**Group registration:**
Runners that want `scope="group:NAME"` targets must call
`bridge.neuromodulator_manager.set_group_indices({name: indices})`
after the engine groups are known. G9 runner does this automatically
for the standard input/hidden/motor groups.

**Plan:** `docs/plans/2026-04-24-neuromodulator-subsystem.md`

**Running tests:**
```bash
pytest tests/test_neuromodulators.py -v
```

### Brain-Region Framework (Session E.2, opt-in)

Declarative framework for multiple brain regions (PFC, Motor, Hippocampus,
Striatum, etc.) on a single bridge. Each region owns a contiguous slice
of neuron indices with its own internal connectivity; cross-region
pathways are declared rather than hand-wired. Composes with the
neuromodulator subsystem from E.1 — pathways can declare
`neuromodulator_gates` and regions auto-register as neuromodulator groups.

Default OFF for full backward compatibility.

**Module:** `sim/regions.py`

**Config (in `CoreSimConfig`):**
- `enable_brain_region_framework: bool = False` — opt-in flag
- `brain_regions: List[BrainRegion]` — declared regions
- `region_pathways: List[RegionPathway]` — directed projections

**Two dataclasses:**
- `BrainRegion(name, n_neurons, exc_fraction, internal_density, exc/inh_weight_mean, weight_jitter, plastic_internal, nm_outputs)`
- `RegionPathway(from_region, to_region, density, weight_mean, weight_jitter, plastic, neuromodulator_gates)`

**Manager:** `RegionManager(regions, pathways)`
- `initialize(seed)` — allocate contiguous index slices + deterministic inh selection
- `total_neurons()` — sum across regions (auto-sets `num_neurons`)
- `indices(name)` / `inhibitory_indices(name)` — per-region lookups
- `region_indices_dict()` — for `nm_mgr.set_group_indices()`
- `build_wiring_plan(seed)` — yields plan dict consumed by `inject_explicit_wiring`

**Bridge integration:**
- Bridge allocates `region_manager` BEFORE neuron arrays (so num_neurons
  is set from `region_manager.total_neurons()`).
- Wiring is generated by `build_wiring_plan()` and fed through
  `inject_explicit_wiring()` (replacing legacy motif/WS/spatial paths).
- When BOTH frameworks are on, regions auto-register as neuromodulator
  groups so `ModulatorTarget(scope="group:PFC")` resolves natively.

**Plan:** `docs/plans/2026-04-24-brain-region-framework.md`

**Running tests:**
```bash
pytest tests/test_regions.py -v
```

### Motor Exploration Noise (Session G)

**Purpose:** Defeats the silent-motor trap (motor neurons that never fire in
phase 1 cannot acquire STDP eligibility, so reward-mediated weight updates
never reach them; agent stays glued to phase-1 winners even when reward
flips sign).

**Mechanism:** Inject independent Poisson spike trains into each output
neuron during the stimulus integration window. Each event is a strong
spike-driving current pulse, so every motor fires occasionally regardless
of upstream activity. STDP can then form positive eligibility on
hidden→silent-motor synapses; reward converts those into weight changes.

**Implementation:** Reuses existing `StimulusManager` POISSON_SPIKE_TRAIN
support — no new GPU code. The G9 runner adds a second `StimulusChannel`
alongside the sensor channel when `motor_exploration_rate_hz > 0`.

**Runner kwargs (`research/runners/g9_runner.py`):**
- `motor_exploration_rate_hz` (default 0.0 — backward compatible)
- `motor_exploration_current_pA` (default 1000.0)
- `motor_exploration_spike_ms` (default 2.0)

Typical working range: 5-30 Hz (~0.5-3 spurious spikes per motor per 100 ms
readout window). 0 disables. Above ~50 Hz starts to dominate action selection.

**Relation to ε-greedy:** Equivalent to ε-greedy / entropy regularization /
Boltzmann exploration in tabular RL, just at the spike-event level instead
of the action-distribution level. Biologically grounded in tonic dopamine
driving spontaneous striatal/cortical activity (Schultz 2007).

**Plan / findings:** `research/findings/2026-04-25-session-g-motor-exploration.md`

### Phase B BG Action Selection Module (resolved silent-motor trap)

**Status (2026-04-25):** GO. 3-seed acid test, phase 1 finalQ avg 1.76 vs G9 baseline 6.74 (74% improvement).

**Why it exists:** Sessions D–I tried 7 runner-side variants (V1–V7) of motor exploration / ε-greedy / proportional sampling to break the silent-motor trap. All were NEGATIVE. The trap was structural: a shared 200-neuron reservoir + argmax readout has a dominant-motor bias from random initial weights that no runner-side hack can fix.

**The fix:** Replace reservoir+argmax with a per-action BG cascade. Each action has its own dedicated populations: `cortex_X → str_D1_X / str_D2_X → gpi_X → thal_X → motor_X` with disinhibition gating (D1 inhibits GPi → thal released → motor fires). No shared argmax; selection happens via independent gates.

**Builder:** `research.runners.g11_bg_runner.build_bg_brain_regions(n_cortex=100)` — returns `(regions, pathways)` with 30 regions and 32 pathways (per-action cortex / D1 / D2 / GPe / GPi / Thal / Motor pools + shared STN + dopamine; ~14.5K synapses).

**Two non-obvious bugs that almost killed the architecture** (both fixed 2026-04-25):
1. `n_cortex=400` over-drove D1 to ~220 Hz (saturated, unphysiological), GPi couldn't silence past STN excitation. **Fix:** use `n_cortex=100` (25 cortex/action). The static probe used 100; the moving-goal runner shipped with 400, so the probe "passed" but the deployment failed. Lesson: probes must call the same builder with the same args as deployment.
2. `cortex→D1` weight_mean=25 against default `stdp_w_max=2` collapsed weights from 25→2 in milliseconds via soft-bound STDP. **Fix:** set `cfg.stdp_w_max = 30.0` in the runner.

**Findings:**
- `research/findings/2026-04-25-phase-b-acid-test-real-win.md` — final 3-seed GO result + diagnosis
- `research/findings/2026-04-25-phase-b-cascade-stability-fix.md` — bug 1 (n_cortex)
- `research/findings/2026-04-25-phase-b-honest-correction.md` — early overstated finding
- `research/findings/2026-04-25-phase-b-bg-acid-test.md` — initial (overstated) result kept for trail

### Phase B refinement (2026-04-26): adaptive DA, WTA, learned perception

After Phase B's structural win, an autonomous overnight session iterated
on twelve sharpening / perception / meta-modulation variants on both
2-goal (1 transition) and multi-goal (3 transitions) tasks. Full result
table in [`docs/SCIENCE_ROADMAP.md` §4.7](docs/SCIENCE_ROADMAP.md).

### 🎉 Plastic-input-layer arc RESOLVED (2026-04-27)

After 7 NEGATIVE attempts on 2026-04-26, the plastic-input-layer
problem was resolved on 2026-04-27 via per-pathway plasticity gating
infrastructure + real curriculum learning. See
[`research/findings/2026-04-27-plastic-input-layer-RESOLVED.md`](research/findings/2026-04-27-plastic-input-layer-RESOLVED.md)
and [`research/findings/2026-04-27-task-adaptive-curriculum.md`](research/findings/2026-04-27-task-adaptive-curriculum.md).

Key new infrastructure:
- `RegionPathway.plasticity_gate: str | None` — tag pathways for runtime gating
- `bridge.set_plasticity_gate(name, value)` — freeze/thaw at runtime
- `cp_plasticity_rate_gain` array — gates STDP, eligibility, Hebbian, synaptic scaling (renamed from `cp_plasticity_gain` 2026-04-29; old name is a deprecated property alias)
- NM-driven gates: `target_type="plasticity_gate", scope="gate:<name>"`

> **GOTCHA — plasticity gate vs synaptic transmission (2026-04-28):**
> `cp_plasticity_rate_gain` and `set_plasticity_gate(...)` freeze weight UPDATES
> only — STDP, eligibility, Hebbian, synaptic scaling. They do NOT freeze
> synaptic CURRENT (`g_syn × (V - E)`). A frozen pathway with non-zero
> `weight_mean` still injects current and affects forward dynamics. To
> staged-introduce a new pathway without disrupting the system before
> the thaw step, initialize it with `weight_mean=0.0` (then let STDP grow
> it from zero after thaw) — OR add a runtime weight scale per gate
> (small bridge change, not yet implemented). The cheat-5 v1 NEGATIVE
> result (2026-04-28) was caused by missing this distinction; v2 fixes
> it via zero-init.

Curriculum: phase 1 corticostriatal plastic + input layers frozen; phase 2
cortex frozen (or partial) + input layers thawed. Biologically: real
critical periods close gradually, gated by neuromodulators, allowing
sensory cortex to mature before association cortex.

### Recommended configuration (current best 2026-05-01)

**Text I/O infrastructure (2026-05-02) — STATISTICALLY SIGNIFICANT W→A.**

**Critical correction:** the previously documented "32.5% W→A baseline" was an
EAST-PREDICTION ARTIFACT on east-heavy eval data, not real learning. The
balanced-sampling fix at d961940 (May-1 19:33) was committed AFTER the baseline
file (May-1 19:22), so all prior text-IO accuracy comparisons used a biased
distribution that masked a Hebbian-decay bug. See
[`research/findings/2026-05-02-text-io-hebbian-decay-root-cause.md`](research/findings/2026-05-02-text-io-hebbian-decay-root-cause.md).

**Three biology-grounded fixes (2026-05-02 commits 144eefd + 200f73c)** restored
real plasticity. Validated across 6 independent seeds (n=600 per metric):

```
W→A (word → action via PFC-bypass): 171/600 = 28.5%  (p=0.027) ← SIGNIFICANT
I→W (image → word readout):         152/600 = 25.3%  (p=0.444) high variance
```

The 28.5% W→A is the most rigorous demonstration of working text I/O in the
project to date. Per-direction: east 6/6 LEARN, west 6/6 positive,
south 4/6 LEARN, north 4/6 REVERSED (cascade structural N-bias).

> **🚨 CRITICAL CAVEAT (2026-05-03 ~08:10 EDT, autonomous overnight):
> Permuted-label control test shows the 28.5% is NOT real word-action
> learning.** Across all 25 prior text I/O eval files (baseline / v2+SWR /
> H4 / curriculum / dpop / BigLang / BigMotor / NoLTD / NoT1 / xcouple /
> multidec / 200ep), **0/25 had the TRUE labeled mapping as the BEST of
> 24 permutations.** Best permutations consistently score 30-37% (8pp
> above chance) but the structure is randomly oriented per-seed, not
> aligned with task labels.
>
> The architecture has cascade-driven structural noise that produces
> 28-33% accuracy on SOME mapping per seed, but the mapping is
> arbitrary not learned. The 28.5% is barely-above-chance noise that
> happens to coincide with true labels marginally above mean (1/24).
>
> See [`research/findings/2026-05-03-permuted-label-control-NEGATIVE.md`](research/findings/2026-05-03-permuted-label-control-NEGATIVE.md)
> and `research/runners/permuted_label_check.py` for full analysis.
>
> The W→A binomial p=0.027 is technically correct but doesn't measure
> what we thought. It measures whether the network has ANY structure
> above chance, NOT whether that structure aligns with task labels.
> Real word-action learning requires aligned ratio ≥ 4/6.
>
> **Investigation arc (2026-05-03 evening → 2026-05-04 ongoing):**
> 1. Decisive cascade test (DONE): minimal architecture
>    (`text_minimal_isolation.py`) with NO cascade — just
>    `language_input → motor_X` — gives mean 16.7% (BELOW chance) at 3
>    seeds. **Cascade-as-cause hypothesis FALSIFIED.** The cascade was
>    a weak DAMPENER on seed-dependent random structure, not its source.
>    See [`research/findings/2026-05-04-minimal-isolation-INVERSION.md`](research/findings/2026-05-04-minimal-isolation-INVERSION.md).
> 2. Biology-grounded sweep (IN FLIGHT): tests three biology fixes
>    (topographic prior 1.5/0.7 matching Pulvermüller 2001-2003 cortical
>    somatotopy; PV-FS lateral inhibition between motor pools per Vogels
>    2011 / Hofer 2011; combined). Anti-cheat control runs FIRST:
>    topographic prior + STDP frozen — if alignment occurs without
>    learning, prior is too strong. Run via `python -m
>    research.experiment_runner experiments/biology_sweep.yaml`.
> 3. Pre-staged A/B follow-up decision chain
>    (`research/findings/raw/g11_bg/wait_biology_then_decide.ps1`)
>    auto-launches:
>    - Outcome A (any condition aligned ≥ 4/6):
>      `experiments/minimum_biology.yaml` — dose-response on biology dose
>    - Outcome B (all 0-1/6): `research/runners/eval_sanity_check.py` —
>      hand-built PERFECT weights, tests if eval methodology itself works
>    Tier-2 fallbacks if needed: `experiments/b2_sparse_codes.yaml` +
>    `experiments/b4_long_training.yaml`.
> 4. Tools shipped:
>    - `permuted_label_check.py` — definitive learning-vs-noise tool
>    - `unaligned_pattern_analysis.py` — cross-condition structural
>      bias analyzer (showed pattern is seed-dependent, +3pp motor_E
>      cascade bias)
>    - `text_minimal_isolation.py` — minimal arch + biology helpers
>      (`apply_topographic_bias`, `enable_motor_fs`, `freeze_stdp`)
>    - `eval_sanity_check.py` — eval methodology validation via hand-built
>      perfect weights
>    - `sim/progress.py` — universal `[PROGRESS] {json}` event format
>    - `research/experiment_runner.py` — YAML-driven sweep orchestrator
>    - `research/result_aggregator.py` — cross-condition aggregation +
>      verdict line (built-in configs: biology, minimum_biology,
>      sanity_check, b2_sparse_codes, b4_long_training)
>    - 7-8x speedup stack: dt=1.0 + parallel-3 GPU sharing +
>      `cfg.fast_spike_reset` (cp.where masked-update). See
>      [`research/findings/2026-05-04-perf-speedup-stack.md`](research/findings/2026-05-04-perf-speedup-stack.md).
>    - 2026-05-05 perf wave 2: three-factor GPU-port (Phase 1, ~2× on
>      3-factor runner), `cfg.fp16_synapse_state` for FP16 eligibility
>      (validated <1mV voltage drift over 1000 steps), parallel=6 in
>      YAMLs (was parallel=2 with GPU at 30-50% util). Cloud H100
>      deploy ready at `scripts/deploy_to_cloud.sh` (~$2/hr, 6-8×
>      sweep throughput vs local 3090). Full roadmap:
>      [`research/findings/2026-05-05-perf-roadmap.md`](research/findings/2026-05-05-perf-roadmap.md).

The three fixes:
1. `cfg.enable_hebbian_learning = False` (matches every g* runner) — Hebbian
   weight decay (1e-5/sub-step × 990K sub-steps = 5e-5 multiplier) was
   collapsing all weights from 2-3 design values down to 0.05 floor.
2. `cfg.stdp_w_max = 5.0` — STDP soft-bound was clipping PFC-bypass design
   weight (3.0) at default cap (2.0). CLAUDE.md gotcha (see STDP bounds note
   above).
3. Non-zero readout pathway init (0.5±0.3) — `cortex_X→language_output` and
   `IT→language_output` were initialized at 0.0; STDP couldn't grow from
   scratch with weak training signal. Non-zero seed lets STDP bidirectionally
   adjust. Biology: real cortex has spontaneous baseline weights (Barlow 1972).

5 followup architectural variations all NEGATIVE (confirming 28.5% is real
ceiling under current 100-ep config): reward shaping, stronger drives,
drive=500 reeval, bigger motor pools (10→30), longer training (100→200 ep).
See `2026-05-02-FINAL-overnight-summary.md`.

Bridge APIs: `set_token_drive()`, `read_language_output()`, `set_pathway_weights()`,
`save_checkpoint()` (does NOT save firing thresholds, STP, eligibility — see
`2026-05-02-reeval-bridge-state-limitation.md`).

Validated production config (use `--seed 42` and others 43, 44, 100, 101, 102
for 6-seed validation):
```bash
python -m research.runners.text_eval_embodied \
    --n-episodes 100 --steps-per-episode 30 --seed 42 \
    --stim-steps-per-step 200 --reset-steps 100 \
    --out-stats research/findings/raw/g11_bg/text_eval_R3R6_100ep_HebOff_v2_seed42.json
# Uses v2 config: Hebbian off + stdp_w_max=5 + readout init=0.5
# Auto-saves checkpoint to .simstate.h5
```

Diagnostic tools shipped 2026-05-02:
- `research/runners/text_eval_analyze.py` — accuracy + binomial p-value
- `research/runners/text_weight_diagnostic.py` — pathway weights + token-targeted
- `research/runners/text_weight_compare.py` — cross-checkpoint comparison
- `research/runners/text_reeval.py` — load checkpoint + re-eval (with caveat:
  cold-start state divergence — not a substitute for in-vivo post-training eval)

Pushing beyond 28.5%: requires deeper architectural changes than tested here.
Candidates: cascade N-bias compensation (reduce cluster_a/e weight to cortex_N),
different decoding (cosine on motor population vector), curriculum (visuomotor
training first → enable text I/O), or pretrained language pathways.

**🎯 LATEST BREAKTHROUGH 2026-05-01: Cluster K v2 visual cortex — 2.97 ± 0.12 at 16×16 perception-only (NO heuristic).**

```bash
# G v2.5 + K v2 — biology-grounded, perception only, scales to 16×16:
python -m research.runners.g11_bg_runner --moving-goal --goal-schedule multi --deterministic \
    --enable-msn-lateral-inhibition --enable-d1-d2-asymmetry --enable-striatal-pv-fsi \
    --enable-cluster-a-closed-loop --enable-cluster-e-topography \
    --enable-dlpfc-wm --enable-pfc-nmda \
    --enable-visual-cortex --visual-cortex-action-warmup-steps 600 \
    --grid-size 16 --seed N --n-steps 1800
```

**G v2.5 + K v2 visual-only at 16×16: 2.97 ± 0.12 (n=3)** — closes 4 of
5 original cheats (heuristic, (gx,gy), (x,y), beacon). 5.2× better than
Tier 0 vanilla perception arc at 16×16 (15.47 ± 7.06). Beats the
documented 8×8 perception arc baseline (4.08 ± 0.49) on a 4× larger grid.
38% of 1800 steps spent AT the goal. See
[`research/findings/2026-05-01-cluster-k-v2-breakthrough.md`](research/findings/2026-05-01-cluster-k-v2-breakthrough.md).

---

**Earlier breakthrough 2026-05-01: Cluster G v2.5 — per-region NMDA on cortex+motor+PFC = 2.00 ± 0.00 (n=6, with heuristic).**

```bash
# G v2.5 with heuristic (8×8/16×16/24×24 all hit 2.00):
python -m research.runners.g11_bg_runner --moving-goal --goal-schedule multi --deterministic \
    --enable-msn-lateral-inhibition --enable-d1-d2-asymmetry --enable-striatal-pv-fsi \
    --enable-cluster-a-closed-loop --enable-cluster-e-topography \
    --heuristic-single-pool \
    --enable-dlpfc-wm --enable-pfc-nmda \
    --seed N --n-steps 1800
```

**A+E + G v2.5: 2.00 ± 0.00 (n=6, multi-goal det)** — **60% improvement** over
A+E single-pool (5.02), **56% over F v2 best (4.55)**, **51% over the
documented cheats-allowed perception-arc flagship (4.08)**. ~49% of total
1800 steps spent AT the goal across all seeds. Welch t = -22.67 vs F v2,
p < 1e-15 — the most statistically significant cheat-5 result to date.

`--enable-pfc-nmda` enables NMDA per-region: dlpfc_wm + cortex_{N,E,S,W} +
motor_{N,E,S,W} (9 regions, 200 neurons total). Biology source: Wang 2002
+ Kandel ch 12 — cortical pyramidals across the neocortex express NMDA-NR2
receptors. v2.5 KEEPS NMDA OFF on hippocampus, BG, cerebellum. The original
v1 used global NMDA (every region) which gave the same 2.00 result alone
but BROKE D-stacks (22.41, 1020% worse) due to runaway recurrent excitation
in CA3. v2.5 fixes that: D v1 stack now gives 3.34 ± 0.64.

F v2 (cerebellum) doesn't compose — NMDA already gives a deterministic
attractor. Use the v2.5 config WITHOUT F v2 as the simplest strongest
config. To stack D v1 (hippocampus), expect 3.34 — still much better than
cheats-allowed flagship.

See [`research/findings/2026-05-01-cluster-g-nmda-breakthrough.md`](research/findings/2026-05-01-cluster-g-nmda-breakthrough.md).

---

**Earlier flagship 2026-04-30: `--heuristic-single-pool` (now superseded by G+NMDA above)**

```bash
python -m research.runners.g11_bg_runner --moving-goal --goal-schedule multi --deterministic \
    --enable-msn-lateral-inhibition --enable-d1-d2-asymmetry --enable-striatal-pv-fsi \
    --enable-cluster-a-closed-loop --enable-cluster-e-topography \
    --heuristic-single-pool \
    --seed N --n-steps 1800
```
**A+E single-pool: 5.02 ± 0.59 (n=6, multi-goal det)** — 28% improvement
over the documented 6.97 ± 0.83 A+E ceiling, 41% std reduction. 6/6 seeds
beat baseline, 6/6 beat A+E, 0 phase catastrophes. Found by systematic
investigation of the persistent ~50% gap between single and replicated
runners; multi-pool heuristic was creating BG-cascade arbitration noise.

**Important caveat:** the multi-pool heuristic was the default for the
two months prior (since cluster work began), so all prior cluster
"NULL" findings (B.1/B.2/B.3, A, C v1/v2, D v1/v2, F v1/v2, HER, recency
replay, RPE, surprise-LR) used a contaminated baseline. Many may have
real signal under `--heuristic-single-pool`; **all cluster results need
revisiting**.

See [`research/findings/2026-04-30-single-pool-heuristic-breakthrough.md`](research/findings/2026-04-30-single-pool-heuristic-breakthrough.md).

---

**Biology-grounded flagship (2026-04-29) — A+E + deterministic single-goal:**
```bash
python -m research.runners.g11_bg_runner --moving-goal \
    --enable-msn-lateral-inhibition \
    --enable-d1-d2-asymmetry --enable-striatal-pv-fsi \
    --enable-cluster-a-closed-loop --enable-cluster-e-topography \
    --deterministic \
    --seed N --n-steps 1800
```
Sum **3.31 ± 0.74 (n=6, single-goal)** — beats the 2026-04-27 documented
"4.08 ± 0.49 full-flagship-cheats-allowed" by **19%**. No `--hippocampus`,
no `--learned-perception`, no `--sensed-reward`, no curriculum. The
biology buildout (R-pass + Cluster B + closed BG loop A + topographic
cortex E) replaces all of those engineering shortcuts.

`--deterministic` sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` before cupy
import. Tightens seed-to-seed noise floor from ±3-5 to ±0.7. Required
to detect cluster effects below the historical noise floor. ~10-30%
slowdown.

See [`research/findings/2026-04-29-overnight-FINAL.md`](research/findings/2026-04-29-overnight-FINAL.md)
for the full eval matrix (60+ runs across 14 conditions).

**Earlier flagship (2026-04-27 — full perception arc + adaptive DA):**
```bash
python -m research.runners.g11_bg_runner --moving-goal \
    --enable-place-goal-readout --learned-perception --enable-dlpfc-wm \
    --beacon-perception --beacon-replaces-goal \
    --cue-reflex --cue-reflex-replaces-heuristic \
    --enable-landmark-sensor --landmarks-replace-place \
    --sensed-reward \
    --enable-msn-lateral-inhibition \
    --adaptive-da --adaptive-da-ema-decay-negative 0.7 \
    --curriculum --curriculum-warmup-steps 600 \
    --seed N --n-steps 1800
```
Sum 4.08 ± 0.49 (6-seed, p=0.00045, **30.6% over baseline**, 6/6 seeds beat).
With `--bg-lateral-inhibition` added 2026-04-28: 4.26 ± 0.50 (no regression).
**Agent has NO direct (gx, gy), NO direct (x, y), NO heuristic, AND NO
distance-based reward.** Reward computed from beacon-intensity gradient.
**Biology-grounded (4.08) BEATS cheats-allowed (4.41)** — closing perception/reward
cheats actually *helps* learning.

This config was the documented best until 2026-04-29's biology-grounded
A+E recipe (above) reduced to 3.31. The earlier config still has merit
for richer perception/working-memory tests; the new biology recipe is
strictly better for the cheat-5 multi-goal navigation benchmark.

**Cheat #5 — ON HOLD pending biology buildout** (reframed 2026-04-28 afternoon; original "closed by design" framing was too quick):
- **v3 (`--bg-lateral-inhibition`) — GO and shipped.** Adds MSN
  cross-pool lateral inhibition. 6-seed sum 4.26 ± 0.50 vs flagship
  baseline 4.08 (no regression). P1 (1.91) beats P0 (2.35) so
  readaptation is improved. **Permanent default in flagship config.**
  See [`research/findings/2026-04-28-cheat5-v3-results.md`](research/findings/2026-04-28-cheat5-v3-results.md).
- **v3.1 (`--bg-lateral-inhibition --bg-cross-projections ...`) — NO-GO.**
  Adult thaw at step 1200 still breaks phase-2: 6-seed 8.92 ± 2.44,
  P1 6.35.
- **v4 (`--developmental-pretraining ...`) — NO-GO.** Pre-training
  cross-projections during a 5K-trial critical period, then freezing
  for eval, is *worse* than v3.1: 3-seed 11.34 ± 1.85, P0 4.88 (even
  initial goal acquisition degrades), P1 6.46. Tier 3 (overnight 6-seed
  validation) skipped — Tier 2 was unanimous past the > 6.0 NO-GO
  threshold. See [`research/findings/2026-04-28-cheat5-v4-results.md`](research/findings/2026-04-28-cheat5-v4-results.md).
- **Option 1 (`--enable-structural-pruning`) under multi-goal — NO-GO.**
  3.2× worse than v3 baseline (22.46 vs 7.08, n=2; seed 42 hung).
  Pruning didn't reshape the topology meaningfully.
- **Option 2 (`--cross-projection-density 0.25`) under multi-goal —
  HIGH VARIANCE, partial signal.** 3-seed mean 8.76 ± 2.54 vs
  baseline 7.08 ± 0.12. Seed 44 actually *beat* baseline (5.88).
  Phase 2 (the (1,6)→(1,1) transition) shows topology-luck signal —
  std 2.09 across 3 seeds vs 0.22-0.46 on other phases. Sparse cross
  topologies sometimes work, sometimes don't, with no mechanism to
  consistently select useful pairs.
- **Reframe (2026-04-28 afternoon):** cheat #5 is **ON HOLD pending
  biology buildout**, not closed by design. Cross-projections aren't
  fundamentally broken — they're under-constrained. Real BG carves
  them via structural plasticity + closed-loop teaching + D1/D2
  asymmetry + cholinergic plasticity gating + thalamo-cortical
  feedback. The reduced model is missing all of this scaffolding.
  See [`research/findings/2026-04-28-cheat5-post-v4-reframe.md`](research/findings/2026-04-28-cheat5-post-v4-reframe.md).
- **Multi-goal eval correction:** all prior cheat-5 NO-GO calls (v1, v2,
  v3.1, v4) used a single goal change at step 300 + 1500 stable steps.
  That's a "static adult after one transition" test; cross-projections
  are theoretically useful for *rapid action-pattern switching*. Now
  using `--goal-schedule multi` (4 phases × 450 steps, 3 transitions)
  for all cheat-5 evaluation.
- **Cluster-based buildout strategy:** cheat-5 closure attempts proceed
  cluster-by-cluster per the strategy in
  [`docs/plans/2026-04-28-cheat5-real-options-survey.md`](docs/plans/2026-04-28-cheat5-real-options-survey.md):
  **Cluster B DONE (3/3)** — B.1 partial, B.2 mixed, B.3 null+infra
  (2026-04-28). **Catalog R-pass DONE (11/12 + 1 deferred, 2026-04-29)** —
  12 biology corrections from Kandel 6e + supplemental texts (per-region
  E_inh, FSI cross-action, GPe split, striosome split, neuropeptide arms,
  asymmetric reward, etc).
  **Cluster A SCAFFOLDED (`--enable-cluster-a-closed-loop`, 2026-04-29)** —
  closed BG loop (cortex→stn hyperdirect + thal→cortex feedback). Eval
  in progress.
  **Cluster C v1 SCAFFOLDED (`--enable-tonic-da`, 2026-04-29)** — tonic
  DA via neuromodulator framework (replaces signed-scalar reward
  modulation when registered). Unlocks B.3 ACh window-gating.
  **Cluster D v1 SCAFFOLDED (`--enable-cluster-d-hippocampus`, 2026-04-29)**
  — trisynaptic loop EC→DG→CA3→CA1 with FFi-mediated DG sparsity, plastic
  CA3 recurrent autoassociator, Schaffer CA3→CA1, CA1→place_cells readout.
  Eval in progress.
  **Cluster F v1 SHIPPED + NEUTRAL (`--enable-cluster-f-cerebellum`, 2026-04-29)**
  — Marr-Albus-Ito cerebellar microcircuit, reward-gated PF→PC LTD. Cheat-5
  multi-goal det: AF 7.37 ± 1.83, AEF 8.02 ± 1.81 (vs baseline 7.77 ± 3.33).
  Std reduction ~45% but no mean improvement.
  See [`research/findings/2026-04-29-cluster-f-results.md`](research/findings/2026-04-29-cluster-f-results.md).
  **Cluster F v2 SHIPPED + NEUTRAL** (`--enable-cluster-f-v2`, 2026-04-30,
  status corrected later same day)
  — CF-gated anti-Hebbian LTD per Albus 1971 §IV.C eq.4. Initial 6-seed
  replicated-runner eval showed AFv2 21.77 ± 2.35, AEFv2 24.88 ± 3.07
  (NO-GO). **Re-run on single runner shows A+F v2 = 7.20 ± 2.75, A+E+F v2 =
  8.14 ± 3.46 (n=6 each) — NEUTRAL vs A+E baseline 7.18 ± 1.58.** The
  replicated runner has a reward-modulation timing bug (~200× fewer weight
  updates per reward) that catastrophizes plasticity-sensitive evals.
  Implementation is correct (47 tests pass, biology probe verifies sign).
  **Opt-in safe; do not use replicated runner for plasticity evals until
  fixed.** See [`research/findings/2026-04-30-fv2-correction-replicated-runner-bug.md`](research/findings/2026-04-30-fv2-correction-replicated-runner-bug.md).
  **Cluster D v2 SHIPPED + PARTIAL (`--enable-cluster-d-v2-swr`, 2026-04-30)**
  — SWR-gated CA3 plasticity. Replaces D v1's implicit CA3 internal_density
  with an explicit ca3→ca3 RegionPathway tagged `ca3_swr_burst`; runner
  flips the gate between 1.0 (open) and 0.1 (suppressed) every 7th sleep
  step (~14% duty cycle, NREM ripple rate per Buzsaki 2015). Original
  endogenous-burst design pivoted to scheduled windows after empirical
  verification that 220 pA into CA3 doesn't produce sustained firing at
  our scale. 6-seed tier-3: A+E+D 29.32 ± 6.95 → A+E+D+v2 27.68 ± 4.78
  (Δmean=-1.64, std cut 31%, Welch t=-0.48 not significant at n=6).
  Both stacks are still ~4× worse than A+E alone (6.97) — sleep replay
  hurts D stacks, v2 mitigates the damage but doesn't fix the underlying
  content-quality bottleneck. **Opt-in only; do not stack on flagship.**
  See [`research/findings/2026-04-30-cluster-d-v2-results.md`](research/findings/2026-04-30-cluster-d-v2-results.md).
  **Cluster C v2 SHIPPED + NEGATIVE (`--enable-compartmentalized-da`, 2026-04-30)**
  — per-action DA channels (4 modulators dopamine_{N,E,S,W}). 6-seed
  tier-3: A+E baseline 7.18 ± 1.58 vs A+E+C v2 9.26 ± 3.91 (Δmean=+2.08,
  Δstd=+2.33). Welch t=+1.21 in the wrong direction; 4/6 seeds hurt
  (worst seed 101: 7.91 → 16.31). Likely failure modes: noisy action
  selection prevents off-policy credit; phase transitions desynchronize
  per-action DA channels; cortex inputs aren't channelized so the
  per-synapse DA tag doesn't compose. **Opt-in only; do not stack on
  flagship.** See [`research/findings/2026-04-30-cluster-c-v2-results.md`](research/findings/2026-04-30-cluster-c-v2-results.md).
  **Cluster-stacking strategy empirically falsified (2026-04-30):**
  8 attempts past A+E (A+D, A+D+E, A+F, A+E+F, A+F v2, A+E+F v2, A+E+D
  with sleep, A+E+D+v2, A+E+C v2), all NEUTRAL or NEGATIVE. A+E
  (6.97 ± 0.83) is the robust operational ceiling. Future work needs
  (a) scaling, (b) harder benchmarks, or (c) interactive eval framework
  — not more clusters.
  **Cluster C v2 DESIGNED (compartmentalized DA, 2026-04-29)** —
  fallback for if A/C v1/D evals don't close cheat-5; per-action DA
  channels with synapse action-tagging.
  Future: more of Cluster E ("Sensory perception & cortical encoding"
  per the catalog's authoritative naming — topographic maps was just
  one aspect; full cluster also covers cortical columns, plastic
  sensory→cortex, etc). Each cluster has
  independent biological merit AND collectively might shift
  cross-projection behavior.
- **Cluster B.1 (D1/D2 asymmetry, `--enable-d1-d2-asymmetry`) — PARTIAL
  SIGNAL (2026-04-28).** First piece of empirical support for the
  cluster strategy. Patch-matrix + B.1 multi-goal: 7.62 ± 1.23 (n=3) vs
  patch-matrix alone 8.76 ± 2.54. Variance halved, Phase 2 catastrophe
  eliminated (P2 mean 3.36 → 1.92, std 2.09 → 0.77). Still above v3
  baseline 7.08 ± 0.12; cheat-5 not fully closed by B.1 alone.
  See [`research/findings/2026-04-28-cluster-b1-d1d2-asymmetry-results.md`](research/findings/2026-04-28-cluster-b1-d1d2-asymmetry-results.md).
- **Cluster B.2 (striatal FSIs, `--enable-striatal-fsis`) — MIXED (2026-04-28).**
  Mean cheat-5 8.44 ± 0.62 (n=3) — slightly worse than B.1 alone (7.62)
  but **variance keeps dropping** (2.54 → 1.23 → 0.62). Phase-decomposed:
  Phases 1-3 BEAT v3 baseline (4.72 vs 4.89), Phase 0 is broken (3.72 vs
  baseline 1.83) because FSIs broadcast too eagerly before agent commits
  to a winner. Architectural issue: real FSIs have tonic baseline, burst
  dynamics, high-pass filtering on cortex drive — our model has none.
  Default str_fs_to_msn_weight retuned 8.0 → 2.0; cortex_to_str_fs_weight
  may need 30 → 10 if full cluster doesn't fix Phase 0. Proceeding to B.3
  (TANs) per unit-cluster strategy. See
  [`research/findings/2026-04-28-cluster-b2-striatal-fsis-results.md`](research/findings/2026-04-28-cluster-b2-striatal-fsis-results.md).
  - **Taxonomy note (R2.3, 2026-04-29):** `str_FS_*` regions explicitly model
    **PV-FSI** (parvalbumin-positive fast-spiking interneurons) — *one of
    eight* distinct striatal GABAergic interneuron classes catalogued in
    Tepper-2018: PV-FSI, NPY-LTS, NPY-NGF, CR, TH/THIN (4 subtypes), FAI,
    SABI, plus the cholinergic ChI/TAN class. The classes are
    **non-isomorphic to cortical taxonomy** (no chandelier-equivalent, no
    Martinotti-equivalent in striatum). The catalog (Kandel ch. 8) earlier
    misapplied cortical taxonomy to striatum; corrected per TK-2017
    pp 157–158, 174 + Tepper-2018 pp 1–2, 11–12. Future cluster work could
    add the remaining 7 classes (NPY-LTS for nitric-oxide signaling, etc.)
    but PV-FSI alone covers the dominant feedforward-WTA function.
- **Cluster B.3 (cholinergic TANs, `--enable-tans`) — NULL on cheat-5,
  shipped as infrastructure (2026-04-28 evening).** Implementation correct
  (47 unit tests pass, biology probe PASS), but TAN-on vs TAN-off is
  statistically neutral at n=3 multi-goal: B.1+B.2 alone 18.02 ± 3.68 vs
  +TANs 18.59 ± 2.64; patch-matrix variants 15.18 ± 3.44 vs 14.83 ± 3.83.
  Reason: the plasticity_window_gate fires inside the reward-modulation
  block, which is skipped when reward = 0 (between rewards). At reward
  steps, pause_on_reward drops ACh and gate ≈ 1 → no suppression. Real
  TAN function requires tonic DA-driven plasticity for ACh to gate; our
  model has only phasic DA. Real win retained: bridge step-order bug fix
  (`59dc1fc`) — `manager.step()` now runs BEFORE reward modulation,
  correcting a one-step lag for fast-dynamics modulators. The
  `pause_on_reward` rule, `plasticity_window_gate` target type, ACh
  default config, `--enable-tans` CLI, and biology probe are kept as
  reusable infrastructure for future tonic-DA experiments. **NOT
  recommended in flagship configs.** See
  [`research/findings/2026-04-28-cluster-b3-tans-results.md`](research/findings/2026-04-28-cluster-b3-tans-results.md).
- **Methodology finding (2026-04-28 evening):** multi-goal benchmark is
  regressed from documented baselines at seed 42 — v3 baseline 7.08 →
  12.05 in current code; B.1+B.2 9.50 → 22.03; patch-matrix+B.1+B.2 8.44
  → 18.87. P3 (after 3 transitions) shows the dominant regression.
  Predates Cluster B.3 changes (bisect at 714bc29 reproduces 21.22 for
  B.1+B.2 at seed 42). Future cluster work should use fresh
  current-code baselines, not the historical numbers.
- **Biology probe at `research/probes/d1_d2_asymmetry_probe.py`** validates
  the implementation: D1 weights ↑ under +reward / ↓ under −reward; D2
  weights inverted. Runnable for any future regression check.
- **`--bg-cross-projections`, `--developmental-pretraining`,
  `--enable-structural-pruning`, `--cross-projection-density`,
  `--enable-tans` all remain opt-in** for future experiments. NOT
  recommended for any current flagship configuration.

**Without sensed reward (perception arc only, 2026-04-27 night):**
```bash
... (above without --sensed-reward)
```
Sum 4.56 ± 0.70 (6-seed, p=0.00819, 22.4% over baseline). Closes 3 of 5 cheats.

**Best with cheats (engineering shortcut, no perception arc):**
```bash
python -m research.runners.g11_bg_runner --moving-goal \
    --enable-place-goal-readout --learned-perception --enable-dlpfc-wm \
    --adaptive-da --adaptive-da-ema-decay-negative 0.7 \
    --curriculum --curriculum-warmup-steps 600 \
    --seed N --n-steps 1800
```
Sum 4.41 ± 0.94 (6-seed, p=0.018, 25.0% over baseline). Uses heuristic
+ direct goal coords + distance-based reward.

Performance comparison (6-seed validated):
- **Baseline**: 5.88
- **★ Flagship (4 cheats closed)**: 4.08 (-30.6%, p=0.00045)
- **Perception arc only (3 cheats closed)**: 4.56 (-22.4%, p=0.00819)
- **Best with cheats**: 4.41 (-25.0%, p=0.018)
- **4-goal (fast-change)**: curriculum doesn't help in any variant

The PFC region adds 60 recurrent neurons modeling working memory, with
plastic pathways `goal_cells → PFC → cortex_{N,E,S,W}`. Tagged with
plasticity_gate="pfc_pathways" for future curriculum control.

For partial-freeze variant (similar performance, more flexible cortex):
```bash
... --curriculum-phase2-cortex-gain 0.2
```

For simplest robust variant without PFC (sum 4.72, p=0.02):
```bash
python -m research.runners.g11_bg_runner --moving-goal \
    --enable-place-goal-readout --adaptive-da --adaptive-da-ema-decay-negative 0.7 \
    --curriculum --curriculum-warmup-steps 600 \
    --seed N --n-steps 1800
```

For multi-goal tasks: skip the curriculum entirely. The baseline
broadcast DA (no curriculum, no hippo) handles fast-change better
because cortex stays plastic and can re-adapt to each new goal.

### Legacy configurations (pre-curriculum, less effective)

```bash
# Pre-curriculum recommendation (still works, but inferior to curriculum):
python -m research.runners.g11_bg_runner --moving-goal --surprise-lr-boost --seed N
```

This is the most reliable refinement: 6-seed mean 4.92 ± 1.07 (16% improvement over baseline 5.88, marginally significant at t=1.31). Lower variance than both baseline and asym DA.

The `--adaptive-da --adaptive-da-ema-decay-negative 0.7` config is kept opt-in but no longer recommended without seed-specific validation — it's seed-dependent (great on 42-44, bad on 100-102, pooled 6-seed mean 5.23 ± 1.90).

**Asymmetric adaptive DA** (the slow-change winner): reward-EMA-gated per-action DA targeting. When agent is winning consistently, eligibility is selectively gated to the chosen action's pathway (commit). When reward drops after goal change, eligibility broadcasts again (explore). Asymmetric ramps (slow positive tau~10, fast negative tau~3) match phasic DA biology — dips are sharper than ramps (Schultz 1998). Reverses on multi-goal (over-throttles learning during frequent changes).

**Surprise-boosted LR** (most robust across task types): when |reward - reward_ema_pre| is high (unexpected outcome), temporarily multiply `reward_learning_rate` by `(1 + α × |RPE|)`. Restored after reward hold. Models NE-like fast meta-modulation. Doesn't gate eligibility — preserves broadcast learning rate while adding "react fast" boost on surprise. Different bottleneck than asym DA: rate not gate.

### Other refinement variants (all opt-in, none beat the recommended configs)

- `--motor-lateral-inhibition`: **DEPRECATED 2026-04-29 (Wave-1 rename #11; slated for removal).** WTA microcircuit (FS interneurons). PARTIAL — exploitation+, readaptation−. Net negative when stacked with adaptive DA. Even DA-gated WTA doesn't help. Real motor-pool WTA biology is spinal Renshaw inhibition (Kandel ch 35), not cortical-FS-like inhibition; future motor-WTA work should explicitly model spinal Renshaw cells. Emits DeprecationWarning when used.
- `--per-action-da`: hard eligibility gating (always ON). Same exploitation/exploration trade-off as WTA.
- `--rpe-scaled-reward`: amplifies reward signal magnitude by RPE. Modest help, but `--surprise-lr-boost` is cleaner architecturally.
- `--learned-perception` (standalone, REPLACES heuristic): NEGATIVE in 2026-04-26 cold-start tests — random init produces no asymmetry for STDP+reward to amplify. **However, when combined with `--hippocampus`, `--pfc`, `--curriculum` and (since 2026-04-27) the perception arc flags, it composes successfully.** The flagship config uses it.
- `--bg-cross-projections`: learnable cortex_X → str_D1_Y all-to-all. NEGATIVE in v1/v2 (3-seed avg 8.40), v3.1 (6-seed 8.92), AND v4 developmental pretraining (3-seed 11.34). Phase-2 readaptation breaks across all attempts. Cheat #5 closed by design 2026-04-28: v3 lateral inhibition is the functional WTA equivalent in our reduced model. Cross-projections kept opt-in for future structural-plasticity experiments.
- `--developmental-pretraining`: critical-period analog (all gates open) for N goals × M trials, then freeze cross-projections for eval. v4 NO-GO — see `research/findings/2026-04-28-cheat5-v4-results.md`. Kept opt-in for pretraining other pathways in future experiments.
- `--bg-lateral-inhibition`: MSN cross-pool lateral inhibition. **GO 2026-04-28** (6-seed 4.26 ± 0.50, no regression). Recommended as a permanent default in all flagship runs going forward — biology-grounded WTA selection.
- Combo flags: combining adaptive DA with WTA, or adaptive DA with LR boost, doesn't compose well. Mechanisms interfere through shared reward EMA. Use one, not both.

### Refinement findings (chronological)

- `research/findings/2026-04-26-wta-lateral-inhibition-mixed.md` — WTA (4.86)
- `research/findings/2026-04-26-per-action-da-mixed.md` — hard DA (4.65)
- `research/findings/2026-04-26-adaptive-da-targeting.md` — symmetric adaptive DA (3.99)
- `research/findings/2026-04-26-asymmetric-adaptive-da.md` — asymmetric (3.53, 2-goal best)
- `research/findings/2026-04-26-da-gated-wta.md` — DA-gated WTA NEGATIVE (4.54)
- `research/findings/2026-04-26-learned-perception-cold-start-fail.md` — perception NEGATIVE
- `research/findings/2026-04-26-multi-goal-stress-test.md` — REVERSES asym DA on fast-change
- `research/findings/2026-04-26-surprise-lr-boost.md` — most robust variant (4.02 / 9.11)
- `research/findings/2026-04-26-night-summary.md` — overall overview

### Research Runner Ecosystem (`research/runners/`)

Headless runners for the research-gate progression (G1 through G11). Each is invocable as `python -m research.runners.gN_runner [args]` and writes results to `research/findings/raw/gN/`.

| Runner | Purpose | Status |
|--------|---------|--------|
| `g1_runner.py`, `g1_v2_runner.py`, `g1_v3_runner.py` | Encoder-decoder roundtrip | G1 GO (v3, 71.3% test acc) |
| `g2_runner.py` | STDP local learning | NO-GO (no epoch improvement) |
| `g3_runner.py` | Persistence/checkpointing | GO |
| `g5_runner.py`, `g5_v2_runner.py`, `g5_v3_runner.py` | Sensorimotor (signed perceptron) | GO |
| `g6_runner.py` | 2D gridworld | PARTIAL (gate metric needs redesign) |
| `g8_runner.py` | (session 8 work) | — |
| `g9_runner.py` | Moving-goal RL + motor exploration | NO-GO at runner-side |
| `g11_bg_runner.py` | BG cascade + perception arc + sensed reward + curriculum | **GO 2026-04-27/28 — flagship** |
| `aggregate_seeds.py` | Cross-seed result rollup | utility |

Findings docs in `research/findings/` document each session's outcome; **negative results are real findings** and stored alongside positives. A new runner should be added whenever a new architectural variant is being tested.

## File Formats

| Format | Extension | Purpose |
|--------|-----------|---------|
| Profiles | `.json` | Human-readable simulation configuration |
| Checkpoints | `.simstate.h5` | HDF5 compressed full simulation state |
| Recordings | `.simrec.h5` | HDF5 compressed frame-by-frame data |

Directories:
- `simulation_profiles/`: Saved configuration profiles
- `simulation_checkpoints_h5/`: State checkpoints
- `simulation_recordings_h5/`: Recorded simulations

## Units

- Time: milliseconds (ms)
- Voltage: millivolts (mV)
- Current: picoamperes (pA) or microamperes/cm² (µA/cm²)
- Conductance: nanosiemens (nS) or mS/cm²
- Capacitance: picofarads (pF) or µF/cm²

## Reproducibility

All RNG sources (CuPy, NumPy, random) are seeded together for determinism. The `RuntimeState.actual_seed_used` tracks the seed used. Separate seeds exist for heterogeneity and noise (`heterogeneity_seed`, `ou_seed`).

## GPU Memory Considerations

- Networks >100K neurons require 20GB+ VRAM
- Use `GPUConfig.memory_pool_limit_fraction` (default 0.8) to control CuPy memory pool
- Connectivity uses CSR sparse matrices to scale with actual connections, not N²

## Agent Style

See `.claude/style.md` for the recommended agent identity and communication style when working on this codebase (computational neuroscience engineer with GPU computing expertise).
