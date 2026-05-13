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
sim/                    # 21 modules (+ __init__.py), ~16.1K lines — core engine
  bridge.py             # 6704 lines — SimulationBridge + GPU state orchestration
  config.py             #  760 lines — all @dataclass configs
  enums.py              #  825 lines — NeuronType (50+ presets), enums, default param managers
  connectivity.py       #  999 lines — spatial/WS/motif connection generators (backend-pluggable)
  kernels.py            #  326 lines — fused @fuse() neuron + plasticity kernels (cupy/numpy)
  profiles.py           #  432 lines — NEURAL_STRUCTURE_PROFILES + CONNECTIVITY_MOTIFS dicts
  regions.py            #  602 lines — BrainRegion + RegionPathway + RegionManager
  neuromodulators.py    # 1052 lines — declarative neuromodulator subsystem
  data_bus.py           #   95 lines — DataChannel pub/sub for streaming sim data
  replicas.py           #  243 lines — replicated wiring (multi-bridge support)
  text_embeddings.py    #  205 lines — token embeddings for language regions (2026-05-01)
  visual_cortex.py      #  310 lines — Gabor RFs + retina rendering (Cluster K v2, 2026-05-01)
  bioparameter.py       #  231 lines — biological parameter helpers
  progress.py           #  147 lines — universal [PROGRESS] event format (2026-05-04)
  lineage.py            #  538 lines — BridgeLineage persistent continuous-learning + growth-log + shard export (2026-05-11)
  auto_growth.py        #  357 lines — TierPromoter + weight-transfer (auto-growth Phase A, 2026-05-11)
  backend.py            #  415 lines — pluggable xp abstraction + device helpers + RNG state (cupy/numpy, 2026-05-11)
  synapse_storage.py    #  415 lines — TieredSynapseStore + idle/pressure eviction (tiering Phase 3+4, 2026-05-11)
  bridge_memory.py      #  487 lines — BridgeMemory LLM-callable memory wrapper (Path 3 Phase 3.1.6, 2026-05-11)
  llm_memory_orchestrator.py #  ~440 lines — MockLLM + LLMMemoryOrchestrator tool-use loop, 5 tool schemas (Phase 3.2, 2026-05-11)
  llm_adapters.py       #  ~190 lines — OllamaLLM + LlamaCppLLM stub adapters (Phase 3.3 scaffold, 2026-05-11)
viz/                    # OpenGL renderer, camera, picker, overlays
ui/                     # DearPyGUI panels, callbacks, layout, sweep panel, plots
experiment/             # ExperimentEngine + StimulusManager + ReadoutEngine + TrainingProtocolEngine
research/runners/       # 75 headless runners (g1..g11 + cluster/text/k_v2/phase1/phase2/chat/perf_benchmark/bridge_lineage/llm_memory_demo/bootstrap_hippo_lineage/validate_ventral_semantic/etc) for research
research/findings/      # session-by-session findings docs (257+ files)
tests/                  # 79 test files (determinism, runners, kernels, plasticity, lineage, tiering, llm orchestrator, llm adapters, etc.)
```

### Thread Model
- **Main Thread**: DearPyGUI event loop + OpenGL rendering
- **Simulation Thread**: GPU-accelerated neural dynamics computation (fully isolated)
- **Communication**: Lock-free queues (`ui_to_sim_queue`, `sim_to_ui_queue`) for inter-thread messaging

### Key Classes

**SimulationBridge** (`sim/bridge.py:170`): Central simulation orchestrator
- Manages all GPU state arrays (CuPy)
- Simulation stepping (`_run_one_simulation_step` at line 4287)
- Initialization (`_initialize_simulation_data` at line 831)
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
- `VisualizationConfig` (line 357): OpenGL rendering and camera parameters
- `RuntimeState` (line 377): Mutable execution state (running, paused, time tracking)
- `GPUConfig` (line 392): GPU features, memory management, recording modes
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

### Pluggable backend (2026-05-11): NumPy backend SHIPPED end-to-end

**Status:** Phases 1+2 of the tiering design SHIPPED 2026-05-11.
SimulationBridge construction + initialization + simulation steps +
brain region framework + checkpoint save/load + bio_three_factor
training + chat_repl W→A + chat_repl :speak A→W ALL work end-to-end
under `SIM_BACKEND=numpy`. No NVIDIA/CUDA dependency required.

CuPy backend remains the production speed path (4-50× faster than
NumPy depending on workload). NumPy backend is for portability +
verification + CI + low-end hardware.

**Usage:**
```bash
# Default (CuPy if available, else NumPy)
python -m research.runners.chat_repl --mode tier1 --seed 42

# Force NumPy backend (Mac M-series, GPU-less Linux, CI)
SIM_BACKEND=numpy python -m research.runners.chat_repl --mode tier1 --seed 42

# Force CuPy explicitly (or fail if unavailable)
SIM_BACKEND=cupy python -m research.runners.chat_repl --mode tier1 --seed 42
```

Findings:
- `research/findings/2026-05-11-numpy-backend-shipped.md` (Phase 2 milestone)
- `research/findings/2026-05-11-numpy-backend-chat-repl-shipped.md` (full chat pipeline)

Design doc: [`docs/plans/2026-05-11-cpu-ram-ssd-tiering-design.md`](docs/plans/2026-05-11-cpu-ram-ssd-tiering-design.md)
Strategic context: [`docs/plans/2026-05-11-strategic-reevaluation.md`](docs/plans/2026-05-11-strategic-reevaluation.md)

**Pattern for new code:** instead of `import cupy as cp`, use:

```python
from sim.backend import get_backend, fuse, synchronize, to_host
xp, backend_name = get_backend()

@fuse()
def my_kernel(a, b):
    return a + b  # works on both cupy + numpy backends
```

**Backend selection** (in priority order):
1. Explicit `get_backend("cupy")` or `get_backend("numpy")` (test code)
2. `SIM_BACKEND` env var (`cupy` / `numpy` / `auto`)
3. Cached backend from a prior call (sticky)
4. Auto-detect: CuPy if installed AND `cp.cuda.runtime.getDeviceCount() > 0`,
   else NumPy

**Helpers exposed by `sim.backend`:**
- `get_backend()` — returns `(xp_module, backend_name)`
- `get_sparse_module()` — `cupyx.scipy.sparse` or `scipy.sparse`
- `is_gpu_backend()` — True if active backend is CuPy
- `fuse(...)` — decorator that's `cp.fuse()` on CuPy, no-op on NumPy
- `synchronize()` — `cp.cuda.Stream.null.synchronize()` on CuPy, no-op on NumPy
- `to_host(arr)` / `from_host(arr)` — D↔H transfers (passthrough on NumPy)
- `get_memory_pool_used_mb()` — CuPy memory pool stats or None

**Tests:** 27/27 pass on both NumPy and CuPy paths (`tests/test_backend.py`).
The pattern is additive — existing `import cupy as cp` code is unaffected
until refactored. No runtime behavior change for current users.

**Status of bridge.py / connectivity.py / kernels.py refactor (Phase 1 part 2, 2026-05-11):**
- `sim/kernels.py` migrated: `import cupy as cp` → backend-aware import;
  all `@cp.fuse()` decorators → `@fuse()` (no-op on NumPy backend).
- `sim/connectivity.py` migrated: `import cupy as cp` + `cupyx.scipy.sparse`
  → backend-aware via `get_sparse_module()`.
- `sim/bridge.py` migrated (import block only): backend-aware `cp` / `csp`
  / `fuse` / `synchronize`. Defensive fallback preserves CuPy code path
  exactly when `sim.backend` is unavailable (e.g. partial bootstrap).
- 19 GPU-specific call sites in bridge.py (`cp.cuda.*`,
  `cp.get_default_memory_pool()`) remain unmigrated. They work on CuPy
  backend; Phase 2 of the tiering design refactors them behind
  `is_gpu_backend()` guards. Until then, constructing a SimulationBridge
  with `SIM_BACKEND=numpy` will fail at GPU-init time — that's expected
  Phase 1 scope.
- 198 lightweight CPU-only tests pass; kernel smoke (Izhikevich) verified
  on CuPy path. No regression for current users.

### Synapse tiering (2026-05-11): pathway-grained storage + activity tracking

**Status:** Phase 3 Strategies B+C SHIPPED 2026-05-11. The bridge can
mirror its per-pathway CSRs into a `TieredSynapseStore` (`sim/synapse_storage.py`)
and track per-pathway activity each simulation step. Inference still
uses the monolithic `cp_connections`; the store is observational +
foundation for Phase 4 auto-tiering. Per-pathway shards can be
exported alongside the lineage's `current.simstate.h5` for inspection
or future SSD-tiered access.

**Opt-in usage:**

```python
# In a CoreSimConfig:
cfg.enable_brain_region_framework = True   # required (pathway names)
cfg.enable_synapse_tiering = True          # opt-in
cfg.synapse_tiering_evict_idle_steps = 1000
cfg.synapse_tiering_grace_pagein_steps = 100
cfg.synapse_tiering_root = "bridges/synapse_shards/active"

# Bridge auto-initializes self.synapse_store at end of
# _initialize_simulation_data; per-step activity tracked in
# _run_one_simulation_step.

# Inspect at runtime:
print(bridge.synapse_store.stats())
# {'n_pathways': 24, 'n_in_memory': 18, 'n_on_disk': 6,
#  'n_pageins_lifetime': 12, 'n_pageouts_lifetime': 8, ...}
```

**Lineage export (Strategy C, works with or without runtime tiering):**

```python
from sim.lineage import BridgeLineage
lineage = BridgeLineage("main")
n_shards = lineage.export_shards(bridge)
# Writes <lineage>/shards/<pathway_name>.npz per pathway
```

**CLI:**
```bash
# Inspect exported shards for a lineage
python -m research.runners.bridge_lineage list-shards main
```

**Webapp endpoint:** `GET /api/synapse-tiering/{name}` returns shard
inventory + sizes per pathway. (Active after webapp restart.)

**Design:**
- Foundational design: [`docs/plans/2026-05-11-cpu-ram-ssd-tiering-design.md`](docs/plans/2026-05-11-cpu-ram-ssd-tiering-design.md)
- Bridge integration design: [`docs/plans/2026-05-11-tiering-phase3-part2-bridge-integration-design.md`](docs/plans/2026-05-11-tiering-phase3-part2-bridge-integration-design.md)
- 3-strategy incremental plan: C (export only) → B (mirror + activity tracking) → A (per-pathway compute, 3-4 weeks scope, deferred)

**Tests:** 56 across `sim.synapse_storage` + bridge integration
(`tests/test_synapse_storage.py`, `tests/test_numpy_backend_integration.py`).
All PASS, all CPU-only.

### Path 3 LLM-callable memory (2026-05-11): BridgeMemory API

**Status:** Phase 3.1.5 SHIPPED 2026-05-11. The `BridgeMemory` class
in `sim/bridge_memory.py` wraps a SimulationBridge + BridgeLineage as
a key-value memory subsystem that an LLM can call via tool-use.

Design doc: [`docs/plans/2026-05-11-path3-bridge-memory-api-design.md`](docs/plans/2026-05-11-path3-bridge-memory-api-design.md)

**Why:** the strategic re-eval (Path 1/2/3) places this on the most
pragmatic path — a locally-runnable LLM (Phi-3-mini / Llama 3.2 1B /
Qwen2.5) handles language + cognition; the biology-grounded sim
becomes the **memory subsystem** distinguished by continuous learning
across sessions without catastrophic forgetting.

**Usage:**

```python
from sim.bridge_memory import BridgeMemory

mem = BridgeMemory(lineage_name="alice", mode="synonym")

# Bind facts — value must map to N/E/S/W (current 4-motor-pool arch)
mem.store("alice", "north", n_events=50)
# {"key": "alice", "value": "north", "target_action": "N",
#  "confidence": 1.5, "bound_correctly": True, "n_events_run": 50}

# Recall
results = mem.recall("alice", top_k=4)
# [{"action": "N", "value": "north", "confidence": 1.0, "rank": 1,
#   "raw_delta": 317}, ...]

# Extinction-style forgetting (Phase 3.2 real-ops, 2026-05-11)
mem.forget("alice", decay_rate=0.5)
# -> {"key": "alice", "decay_rate": 0.5, "n_active_neurons": 6,
#     "n_synapses_decayed": 60, "mean_weight_pre": 1.0,
#     "mean_weight_post": 0.5, "estimated_retention": 0.5}

# Long-term consolidation (Phase 3.2 real-ops, 2026-05-11)
# Requires hippocampus-enabled bridge (main lineage isn't; bootstrap
# `main_hippo` via research.runners.bootstrap_hippo_lineage)
mem.consolidate(n_sleep_cycles=3)
# -> hippo-enabled: {"n_sleep_cycles_run": 3, "n_swr_events_run": 600,
#                     "elapsed_seconds": 45.2, "hippocampus_enabled": True}
# -> no hippo:     {"n_sleep_cycles_run": 0, "hippocampus_enabled": False,
#                     "note": "Bridge lacks hippocampus..."}

# State
print(mem.stats())
```

**Webapp endpoint:** `GET /api/bridge-memory/{name}` returns memory
state aggregated from lineage growth events: n_bindings, n_forgets,
n_consolidations, the binding history (last 50), current_tier.
(Active after webapp restart; shipped commit `def96d8`.)

**What's Phase 3.2 (deferred):**
- Choose local LLM hosting (vLLM / llama.cpp / ollama)
- Wire BridgeMemory methods to tool-use handlers (OpenAI / Anthropic
  schema)
- 5-turn conversation smoke test
- Multi-session continuity test (Phase 3.3)

**Limitation:** today's bridge has 4 motor pools (N/E/S/W). Values
must map to these. Multi-modal arbitrary k/v bindings need a larger
arch (Phase 3.2+).

**Tests:** 18 across `sim.bridge_memory` (17 in test_bridge_memory.py
+ 1 real-bridge integration test in test_numpy_backend_integration.py).
All PASS.

### Engram-tagging API (P2, 2026-05-11): catalog D.14 / roadmap T1.C SHIPPED

**Status:** SHIPPED commit 29513ac + a3acb9c. 12/12 unit tests pass.
Persistence through save/load validated (2 integration tests skipped
pending fuller test bridge).

**Module:** `sim/bridge.py` (added 9 methods to SimulationBridge,
~200 lines including docstrings)

Tonegawa-style ensemble tagging — "Apple is a CA3 ensemble":

```python
bridge.start_engram_recording("apple")
# Drive lang_input("apple") + run bridge steps for the encoding window
for _ in range(encoding_steps):
    bridge._run_one_simulation_step()  # auto-accumulates spike counts
stats = bridge.commit_engram_tag("apple", top_k=50,
                                    region_filter=["ca3"])
# stats = {"n_tagged": 47, "n_recorded_steps": 100, "window_ms": 100.0,
#          "mean_spike_count": 1.4, ...}

# Later — causal recall by stimulating the tag:
bridge.stimulate_tag("apple", drive_pA=200.0)
# Now run more steps and observe downstream regions
```

Auto-tick wired into `_run_one_simulation_step` (zero overhead when
no active recordings).

Methods:
- `start_engram_recording(name)` — begin accumulating spike counts
- `commit_engram_tag(name, threshold_hz=5.0, top_k=None,
                      region_filter=None)` — finalize tag from
  accumulated counts. Two selection modes: top-K or threshold-Hz.
- `stimulate_tag(name, drive_pA, additive=False)` — drive
  `cp_external_input_current` at tagged indices
- `clear_tag_drive(name=None)` — zero per-tag or globally
- `list_engram_tags()` / `get_engram_tag_indices(name)` / `delete_engram_tag(name)`

Persistence: tags saved as HDF5 `engram_tags/` group in
`save_checkpoint`; restored in `load_checkpoint`. Concepts survive
between sessions, matching the project's continual-learning premise.

Validation: catalog D.14 (Tonegawa engram cells); roadmap T1.C
behavioral check is the Liu 2012 inception-of-fear paradigm (train
context A → reward, tag ensemble, drive ensemble in context B,
verify reward-conditioned behavior emerges). Liu 2012 reproduction
is downstream work; the API is the prerequisite.

### Positional context P4.1 substrate (2026-05-11): catalog D.01+D.02+D.11

**Status:** SUBSTRATE SHIPPED commit 11c7c53 + ea9e439. Multi-seed
validation pending GPU (after P1 two-concept aggregates).

`sim/text_embeddings.py` adds:
  `positional_drive_pattern(position, n_neurons=200, sparsity=0.1,
                              n_max_positions=16)` — deterministic
  sparse code per position. Same band-stride layout as
  `orthogonal_drive_pattern` for maximal separability.

`research/runners/text_minimal_isolation.py` adds:
  `enable_episodic_context` flag → adds `ec_context` region (default
  200 neurons) + `ec_context → dg` plastic pathway (gate
  `ec_context_to_dg`). When enabled, DG receives a combined
  (word, position) drive → distinct CA3 ensembles per (word,
  position) tuple.

`research/runners/validate_positional_binding.py` (Test runner for
P4.1):
  Encodes 4 (word, position) bindings (apple@pos_0/pos_2,
  alice@pos_0/pos_2) and measures pairwise CA3 ensemble cosines.
  PASS criteria:
    - Same word, different position: cos < 0.4
    - Different word, same position: cos < 0.4

After P4.1 PASS, the architecture supports word-order-dependent
meaning. Downstream P5/P6 can learn to distinguish sentences by
their (word, position) ensemble structure.

### Concept replay P3.1 (2026-05-11): catalog D.19 + T1.B SHIPPED

**Status:** SHIPPED commit d569848. 5/5 unit tests pass.

`run_concept_replay_phase(bridge, tag_names, n_replays_per_tag=20)`
added to `research/runners/consolidation_trainer.py`. During NREM,
drives each engram-tagged CA3 ensemble repeatedly so STDP at
ca3→ca1→cortex consolidates the specific concept.

Differs from existing `run_swr_replay_phase` (random sparse CA3
drives): concept replay is SELECTIVE to the day's tagged concepts.
After enough replay cycles, recall works from cortex without needing
hippo state (consolidated).

Graceful error handling: missing tag names + empty tags silently
skipped. Caller manages awake/sleep gate transitions.

P3.2 (sequence replay with 10-20× time compression) deferred until
P4 episodic encoder produces sequences worth replaying.

### Hippocampal trisynaptic loop (P1, 2026-05-11): catalog D.03+D.12+D.13 validated

**Status:** SINGLE-SEED PASS commit 9d9b8f3. Multi-seed (seeds 42,
43, 44) shows D.12 (separation) robust at 3/3; D.13 (completion)
1/3 on the absolute cos > 0.7 threshold (seed 42=0.748, seeds
43=0.676, 44=0.679). Seeds 43/44 within 3% of threshold —
autoassociator working but seed-variable. Two-concept discrimination
test (relative criterion, more biology-faithful per catalog D.13
"too much completion → confused episodes; too little → no
generalization") running 3 seeds; results pending.

**Runner:** `research/runners/validate_trisynaptic_loop.py`.

The trisynaptic architecture was ALREADY built in
`build_biological_brain_regions(enable_hippocampus_consolidation=True)`
(Phase 1.3 consolidation work). P1 validated the catalog's two
characteristic functional properties:

```bash
python -m research.runners.validate_trisynaptic_loop \
    --seed 42 --train-events 400 --ca3-recurrent-weight 5.0 \
    --direct-ca3-drive \
    --out research/findings/raw/g11_bg/trisynaptic_seed42.json
```

- **D.12 pattern separation** (Kandel pp 1357–1360): DG cosine 0.218
  from input cosine 0.800 — 58pp orthogonalization. ✅ PASS
- **D.13 pattern completion** (Kandel pp 1342, 1360–1361; Marr 1971):
  CA3 cosine 0.748 (target > 0.7). ✅ PASS

Methodology note: EC-driven test (drive lang_input, propagate
through trisynaptic chain) FAILED at all parameter combinations.
DIRECT-CA3 test (drive partial of stored CA3 ensemble directly) is
the cleaner Marr autoassociator test and PASSES at train=400 +
ca3_recurrent_weight=5.0.

See `research/findings/2026-05-11-P1-trisynaptic-loop-validation.md`.

### Realigned plan (2026-05-11, post-checkin): sim as STANDALONE conversational agent

After the Path 3 Phase 3.2 work shipped, the user clarified: goal is
sim as a standalone agent, **no external LLM ever**. The Phase 3.3
(real LLM swap-in) is DEPRECATED for primary path. See
[`docs/plans/2026-05-11-realigned-plan-sim-as-standalone-conversational-agent.md`](docs/plans/2026-05-11-realigned-plan-sim-as-standalone-conversational-agent.md).

Active steps:
- **Step 1 (1-2 wk):** Fix in-vivo new-vocab binding via the four-
  variant runner (`research/runners/investigate_invivo_binding_fix.py`).
  The 2026-05-11 n_events curve confirmed novel keys fail at 200/400
  events (0/4 → 1/4 correct). Variants test pre-bind anchoring,
  curriculum interleaving, recall-only fine-tune tail.
- **Step 2 (2-3 wk):** Validate synonym12 / synonym16 vocab with the
  Step 1 fix.
- **Step 3 (2-4 wk):** Compositional 2-word phrases (Tier 2.3 PFC
  verb pool reactivated).
- **Step 4 (1.5-2 mo, conditional):** Dendritic learning rewrite if
  Step 3 hits a compositional ceiling.
- **Step 5+ (months-year):** 64+ word vocab, sentence-level
  understanding, reasoning, true conversation.

Local-only commitment: every step runs on RTX 3090 or CPU. No cloud
dependencies, no external LLM.

### Path 3 Phase 3.2 (2026-05-11): LLM-memory orchestrator + chat UI (now SECONDARY)

⚠️ **The Phase 3.2 stack is now framed as the SECONDARY application
(sim as continuous-learning memory layer for external LLM agents).**
Code stays in the codebase, but the framing changed per the realigned
plan above. The MockLLM in the dashboard chat is a sim-native pattern
dispatcher — it doesn't pretend to be an LLM.

**Status:** SHIPPED 2026-05-11. MockLLM tool-use loop with end-to-end
demo + webapp chat surface. Real LLM swap-in (Phi-3 / Llama 3.2 /
Qwen2.5) via SIM_LLM_BACKEND=ollama env var is available for the
secondary path but NOT actively developed.

**Module:** `sim/llm_memory_orchestrator.py` (~440 lines)
- `TOOL_SCHEMAS` — OpenAI-compatible JSON schemas for the five
  tools (`memory_store`, `memory_recall`, `memory_speak`,
  `memory_forget`, `memory_consolidate`).
- `ToolCall` / `LLMResponse` — dataclasses for the tool-use protocol.
- `MockLLM` — regex-based pattern recognition for "remember that X
  is dir", "what's my X", "what word goes with dir", "forget my X"
  (+ "fully forget" / "erase" for decay=0.0), "consolidate" /
  "sleep on it" (+ "for N cycles" for explicit count). Direction
  synonyms (up/down/left/right). Falls back to a helpful message
  on unrecognized input. Propagates tool-dispatch errors verbatim.
- `LLMMemoryOrchestrator` — drives the tool-use loop. `chat()` adds
  the user message, queries the LLM, dispatches tool calls against
  the BridgeMemory, feeds results back, repeats until a final
  message or `max_tool_iterations` (default 5).

**Runner:** `research/runners/llm_memory_demo.py` — end-to-end demo
(MockLLM → orchestrator → BridgeMemory → SimulationBridge → lineage).
208-neuron tier1 toy bridge built in ~1.5s on CPU.

```bash
# CPU NumPy backend
SIM_BACKEND=numpy python -m research.runners.llm_memory_demo \
    --seed 42 --lineage llm_demo --out demo.json

# Scripted 5-turn chat with full transcript printed + JSON dump
```

**Webapp endpoints (Phase 3.2 UI):**

```
POST /api/llm-chat
  body: {lineage, mode, message, reset_conversation?}
  returns: {response, tool_calls[], conversation_length, n_turns}

GET /api/llm-chat/{name}/transcript?mode=...
  returns: {messages[{role,content}], n_turns, total_messages}

POST /api/llm-chat/{name}/reset?mode=...
  returns: {reset: bool}
```

In-process orchestrator cache: first call per (lineage, mode) tuple
pays the bridge-load cost; subsequent calls are fast.

**Frontend (Lineages tab):** click a lineage → "Chat with this lineage"
panel renders below the bridge-memory bindings. Mode selector
(tier1 / synonym / synonym12 / synonym16), message log (color-coded
by role), input box + Send / Enter, Reset button. Transcript
auto-loads when mode changes or the lineage is revisited.

**Tests:** 32 across the LLM stack:
- 14 in `tests/test_llm_memory_orchestrator.py` (tool schema, MockLLM
  patterns, orchestrator end-to-end, max-iter cap, error propagation)
- 2 in `tests/test_llm_memory_demo.py` (single-turn + multi-turn
  smoke against a real bridge; SIM_BACKEND=numpy for CI portability)
- 5 in `tests/test_webapp_server.py` (404 + reset idempotent +
  validation + frontend asset)

All PASS. CPU-only safe.

**Real-LLM integration (Phase 3.3 — user-blocked on LLM choice):**

```python
# Drop-in replacement for MockLLM:
def real_llm_callable(conversation: list[dict]) -> LLMResponse:
    # Call ollama / vLLM / llama.cpp with TOOL_SCHEMAS as tools.
    # Parse the response into LLMResponse(message=..., tool_calls=[...])
    ...

orch = LLMMemoryOrchestrator(memory=mem, llm_callable=real_llm_callable)
```

Candidate LLMs (all open-weights, CPU-runnable):
- **Phi-3-mini-4k-instruct** (~3.8B params, Q4 → ~2GB) — Microsoft
- **Llama-3.2-3B-Instruct** (Q4 → ~2GB) — Meta
- **Qwen2.5-3B-Instruct** (Q4 → ~2GB) — Alibaba

Findings doc: `research/findings/2026-05-11-path3-phase3.2-llm-stack-shipped.md`

### Continuous-learning workflow (2026-05-11): Bridge Lineage Manager

**Status:** SHIPPED 2026-05-11. The chat REPL now "lives" between sessions
by default. See
[`research/findings/2026-05-11-bridge-lineage-shipped.md`](research/findings/2026-05-11-bridge-lineage-shipped.md)
for the full shipping notes; design doc at
[`docs/plans/2026-05-10-bridge-lineage-design.md`](docs/plans/2026-05-10-bridge-lineage-design.md).

Persistent training state lives under `bridges/lineage/<name>/`:
`current.simstate.h5` (latest state, auto-loaded), `metadata.json`
(vocab, tier, cumulative events, accuracy_history, growth_events), and
`history/` (last 30 snapshots by default). The `BridgeLineage` class
(`sim/lineage.py`) handles atomic save (`.new` + `os.replace`),
millisecond-precision history timestamps, and schema-version migration.

**Default workflow (continuous mode):**
```bash
# Loads lineage 'main' if it exists, skips ~6-20 min training.
# Saves back on exit; previous state goes to history/.
python -m research.runners.chat_repl --mode synonym
```

**Science mode (multi-seed reproducibility):**
```bash
# Always trains from random init; does NOT touch lineage.
python -m research.runners.chat_repl --mode synonym --from-scratch --seed 42
```

**Branching for experiments:**
```bash
# Fork 'main' into a new lineage; future saves go to the fork.
python -m research.runners.chat_repl --mode synonym --fork-lineage experiment_v3
```

**Inspection / management CLI (`research/runners/bridge_lineage.py`):**
```bash
python -m research.runners.bridge_lineage list
python -m research.runners.bridge_lineage show main
python -m research.runners.bridge_lineage history main
python -m research.runners.bridge_lineage rollback main --to <snapshot_id>
python -m research.runners.bridge_lineage fork main experiment_v3
python -m research.runners.bridge_lineage prune main --keep-last 10
python -m research.runners.bridge_lineage diff main --from <snap_id> --to current
```

**Webapp endpoints (`GET /api/lineages`, `GET /api/lineages/{name}`):**
Surface the lineage data for the future Lineages tab. Endpoints are
wired + tested; frontend tab is the only remaining piece.

**Compatibility:**
- Lineage stores `mode` + arch in metadata. Loading a `tier1` lineage
  with `--mode synonym` triggers a "fallback to fresh training"
  warning — no shape-mismatch crash.
- `save_checkpoint` doesn't preserve firing thresholds / STP /
  eligibility per the CLAUDE.md gotcha above. Self-recovers in ~10ms
  of free running. Fine for inference (REPL chat); documented.
- Batch demos (`chat_demo`, `chat_synonym_demo`, `chat_speak_synonym_demo`)
  default to fresh training; opt-in to lineage via `--lineage NAME`.

**Tests:** 78 across the subsystem (21 BridgeLineage, 13 CLI, 28
chat_repl, 14 chat_demo_aggregate, 2 webapp). All PASS, all CPU-only.

### Recommended configuration (current best 2026-05-11)

**🎉 THREE multi-seed PASS validations confirmed on current code today (2026-05-11):**

1. **Tier 1 motor binding (4-word direction)**: 6/6 multi-seed PASS. Mean
   W→A 85.8%, A→W 98.2%. Recipe:
   `python -m research.runners.bio_three_factor --biological --embodied-hebbian
   --apply-topographic-bias --enable-motor-fs --n-events-per-direction 200 --seed N`.
   See `research/findings/2026-05-11-Tier1-multiseed-6of6-PASS.md`.

2. **Tier 2.1 synonym binding (8-word)**: 6/6 multi-seed PASS. Mean
   W→A 60.0%, A→W 95.3%. Recipe: Tier 1 + `--synonym-mode --n-lang-input 4096
   --n-motor-per-action 1000 --n-motor-fs-per-action 120 --n-events-per-direction 400`.
   See `research/findings/2026-05-11-Tier2.1-multiseed-6of6-PASS.md`.

3. **P5 ventral semantic comprehension (Path A iter W)**: 6/6 multi-seed
   PASS on comprehension cosine. Mean margin +0.085, ratio 1.46x. Recipe:
   `python -m research.runners.validate_ventral_semantic --seed N
   --n-train-events 400 --n-replay-cycles 40 --enable-multi-pool-wernicke
   --n-wernicke-pools 2 --n-per-wernicke-pool 100 --n-per-wernicke-pool-fs 12`.
   See `research/findings/2026-05-11-P5-iterW-BREAKTHROUGH-6of6-COMP-PASS.md`.

4. **P5 iter AA bidirectional naming (2-concept)**: 4/6 multi-seed PASS
   on pool_readout BIDIR (apple 6/6, river 4/6). Recipe: iter W +
   `--interleaved-training --enable-per-concept-lang-out-pools
   --n-per-lang-out-pool 200`. See
   `research/findings/2026-05-12-P5-iter-AA-confirmed-ceiling.md`.

**Biological-scale extension (2026-05-12) — architectural ceiling
confirmed; strategic pivot to in-vivo new-vocab binding:**

After 7 biological-scale iterations + 30+ toy-scale iterations,
**iter AA's 4/6 toy-scale BIDIR is conclusively the architectural
ceiling** for the per-concept pool design at the P5 ventral semantic
stream. No biological-scale variant tested improves on it.

| Iter | Change | Result |
|---|---|---|
| AA (toy, ref) | per-concept pools, weak dynamics | **4/6 BIDIR** |
| KK | + Tier 1 canon, biological scale | 0/seed_42 (canon amplifies bias) |
| LL | + scale only (weak) | 0/seed_42 (discrimination collapses) |
| MM | + stronger topographic | 0/seed_42 (helps river, not apple) |
| NN | + orthogonal codes | 0/seed_42 (flips winner, doesn't fix) |
| OO_visual | + sensory grounding (Cluster K v2) | 0/seed_42 (apple+23 but river flipped) |
| **PP** | **+ lang_output FS WTA** | **1/4 BIDIR** (seed 42 PASS, 43/44/100 FAIL) |

**Diagnosis:** discrimination at iter AA depends on TOPOGRAPHIC PRIOR
(selectivity_index ~0 across all seeds — STDP doesn't add concept-
specific selectivity). At biological scale, per-seed random structural
pool variance compounds through multi-hop chains and dominates the
input signal. Mitigations (canon dynamics, stronger bias, orthogonal
codes, sensory grounding, output FS WTA) all help individual seeds
but fail multi-seed robustness.

**Architecture preserved in code** (parameterized via CLI flags). iter
PP seed 42's BIDIR PASS (+1, +6 margins) is the first bidirectional
pass at biological scale and demonstrates the sensory-grounding +
output-WTA pattern CAN work — just not robustly.

See `research/findings/2026-05-12-P5-iterPP-multiseed-NEGATIVE-FINAL.md`
and `docs/plans/2026-05-12-P5-sensory-grounding-design.md` for full
arc details.

**Strategic pivot tested (2026-05-12):** Step 1 in-vivo new-vocab
binding via biology-grounded variants on main_hippo lineage.

| Variant | seed 42/43 | Notes |
|---|---|---|
| V0 vanilla | 1/4 | forest→W correct by coincidence; routing varied |
| V_HIPPO_BIO | 0/4 | Hippocampus+SWR varies pool selection but never to target |
| V_SCHEMA | 1/4 (deterministic) | **mountain→S TRUE bind via anchor reinforcement** |

**V_SCHEMA result is fully deterministic** given main_hippo state. Multi-
seed runs (42 + 43 with proper fork cleanup) produce IDENTICAL raw_delta
values (33, 22, 23, 8). The seed parameter only varies OU noise which
doesn't perturb V_SCHEMA's training outcome. To test real seed variance,
need to bootstrap multiple main_hippo lineages with different seeds.

**🎯 NON-MONOTONIC SWEET SPOT (2026-05-12):** V_SCHEMA performance
peaks at 200-event main_hippo bootstrap:

| Bootstrap | Wall | V_SCHEMA result | Bindings |
|---|---|---|---|
| 50ev (smoke) | 9 min | 1/4 | mountain→S only |
| **200ev (sweet spot)** | **53 min** | **2/4 ✓** | **apple→N + mountain→S** |
| 400ev (over-trained) | 112 min | 1/4 (REGRESSED) | mountain→S only |

400-event bootstrap REGRESSES V_SCHEMA to 1/4. Over-training creates
winner-take-all dynamics where one pool's recurrent activity
overwhelms anchor-driven STDP for new bindings. Just-right anchors
at 200ev provide balanced pool competition.

See `research/findings/2026-05-12-V_SCHEMA-2of4-strong-hippo-BREAKTHROUGH.md`
and `research/findings/2026-05-12-V_SCHEMA-non-monotonic-200ev-sweet-spot.md`.

**Best in-vivo vocab method:** V_SCHEMA + 200ev main_hippo = 2/4. To
push beyond requires different mechanism (per-direction-balanced
bootstrap, homeostasis enforcing equal pool strength, OR topographic
bias prior at binding time) — not just more events.

**Canonical main_hippo lineage:** bridges/lineage/main_hippo is now
the 200ev sweet-spot version. 400ev version preserved at
bridges/lineage/main_hippo_400ev for future comparison. Balanced
version at main_hippo_balanced (also doesn't break ceiling).

**🎉 SYNONYM12 + SYNONYM16 CHAT_SPEAK VALIDATED (2026-05-12):**

| Mode | Vocab | W→A | A→W | Verdict |
|---|---|---|---|---|
| Tier 1 | 4w | 74-98% | 58% | GO multi-seed |
| Synonym (Tier 2.1) | 8w | 31-56% | 85% | GO 6-seed |
| **Synonym12** | **12w** | **56%** | **100%** | **GO seed 42** |
| **Synonym16** | **16w** | **56%** | **100%** | **GO seed 42** |

Synonym12 and Synonym16 chat_speak both achieve 100% A→W (4/4 top-1
correct primary direction word) and 56% W→A (9x chance for 16-word).
The conversational sim now demonstrates a **16-word working vocabulary**
with reliable bidirectional binding.

See `research/findings/2026-05-12-synonym12-synonym16-chat_speak-PASS.md`.

Recipe (replicate):
```bash
python -m research.runners.chat_speak_synonym_demo --seed N \
    --vocab-size 16 --train-events 400
# Synonym16: 17K neurons, 26.8M synapses, ~42 min compute (solo)
```

Combined conversational stack:
- Pre-trained 16-word vocabulary (synonym16 GO)
- Bidirectional W→A + A→W
- :learn V_SCHEMA for in-vivo +2 words (Tse 2007 schema reinforcement)
- Phase 1.3 hippocampus consolidation (no catastrophic forgetting)
- chat_repl interactive REPL

This is a genuinely usable conversational artifact for ~16 direction-
related words. Future expansion options:
- Multi-seed validation of synonym12/16 (~3 hr)
- Synonym24/32 extension (requires vocab table expansion in text_embeddings.py)
- Tier 2.3 phrase composition (architecture-limited at 34-40%, deeper rework needed)

**Same architectural ceiling as iter PP biological scale:** per-seed
random structural variance dominates the learning signal for NOVEL
keys without pre-existing topographic prior. Hippocampus+SWR encoding
doesn't reach lang_input → motor strongly enough; schema-supported
anchor reinforcement only works on ONE direction at the current
bootstrap config.

**Bug fix preserved (commit f3308b8):** BridgeMemory was loading
hippo-enabled lineages with wrong architecture (synonym instead of
tier1_hippo). Added `tier1_hippo` mode to chat_repl helpers and
auto-detect from lineage metadata in BridgeMemory._ensure_loaded.
Any future hippo-enabled lineage work is now unblocked.

See `research/findings/2026-05-12-invivo-binding-seed42-smoke-NEGATIVE.md`
for full smoke results + strategic synthesis.

**Current demonstrated conversational capability:**
- Tier 1 (4-word direction): 6/6 BIDIR multi-seed PASS (74% W→A,
  98% A→W at seed 42)
- Tier 2.1 (8-word synonym): 6/6 BIDIR multi-seed PASS
- Phase 1.3 consolidation: 3/3 PASS (cortex retains binding after
  hippo silence)
- P5 abstract concept (2-word, toy iter AA): 4/6 BIDIR multi-seed
- In-vivo novel-key binding: 0-1/4 single-seed (architectural limit)

P5 naming still 3/6 partial. 32+ P5 iterations (A-PP) + invivo
variants exhaust parameter sweep. **iter AA 4/6 toy-scale is the P5
production capability;** biological-scale path closed pending major
architectural rethink (stronger topographic prior at binding time
OR pre-allocated novel-key pools OR unified Wernicke + sparse coding).

---

**Text I/O infrastructure (2026-05-02) — ~~STATISTICALLY SIGNIFICANT W→A~~ SUPERSEDED 2026-05-05.** See the [W→A verdict](research/findings/2026-05-05-W-to-A-VERDICT-global-scalar-feedback-fails.md): the "28.5% W→A" reported below failed the permuted-label control (2026-05-03) and was not aligned with task labels. Three subsequent investigations (3-factor with classical sign-only DA: 1/6, 3-factor with magnitude-graded DA: 0/6, B3 supervised gradient: 3/3 PERFECT) confirmed that global scalar feedback in any form cannot match per-region gradient at biological-scale W→A. Section retained below for historical context.

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
> 5. **🚨 2026-05-05 FINAL VERDICT: global scalar feedback fails at
>    biological scale.** Both classical sign-only DA (1/6 at
>    `tf_with_topo_fs`) AND magnitude-graded DA Schultz-1998-style
>    (0/6 at `tfg_with_topo_fs`) fall below dendritic-learning
>    decision gate. Gradient (B3 supervised) under identical
>    architecture: 3/3 PERFECT. Architecture is sufficient; the
>    credit-assignment rule is the bottleneck. See
>    [`research/findings/2026-05-05-W-to-A-VERDICT-global-scalar-feedback-fails.md`](research/findings/2026-05-05-W-to-A-VERDICT-global-scalar-feedback-fails.md)
>    for full chronology + three options for next direction
>    (apical-basal dendritic learning, predictive coding, or pivot
>    away from W→A). Design doc for option 1 at
>    [`docs/plans/2026-05-05-dendritic-learning-design.md`](docs/plans/2026-05-05-dendritic-learning-design.md)
>    (1.5-2 month scope).
> 6. **🎉 2026-05-06 TIER 1 BREAKTHROUGH (~6 hours after the verdict):**
>    embodied-Hebbian co-firing produces bidirectional word↔motor binding.
>    6-seed validation: **W→A 5/6 aligned, A→W 6/6 aligned** (mean 38%/45%,
>    +0pp excess on 11/12 condition×seed pairs). Compare to 3-factor
>    1/6 (noise floor) — 6× improvement just from changing the training
>    paradigm (no rule change, no rewrite). User↔sim language
>    communication achievable at 4-word vocabulary level: type "north"
>    → motor_N activates AND motor_N activates → language_output
>    produces "north". Dendritic learning rewrite (1.5-2 mo) is no
>    longer urgent for the W→A goal. See
>    [`research/findings/2026-05-06-Tier1-BREAKTHROUGH-bidirectional-binding.md`](research/findings/2026-05-06-Tier1-BREAKTHROUGH-bidirectional-binding.md)
>    for full result + Tier 2 plan (20-30 word vocab, two-word phrases,
>    ~1 month). 3-tier roadmap at
>    [`docs/plans/2026-05-05-embodied-language-3tier-design.md`](docs/plans/2026-05-05-embodied-language-3tier-design.md).
>    Use `--biological --embodied-hebbian --apply-topographic-bias
>    --enable-motor-fs --n-events-per-direction 200` on bio_three_factor.
> 7. **🎉 2026-05-06 TIER 2.1 BREAKTHROUGH: 8-word synonym vocabulary
>    works via scale-up.** 6-seed validation: **W→A 5/6 aligned, A→W 6/6
>    aligned** with synonym pairs {north,up}, {east,right}, {south,down},
>    {west,left}. A→W mean 63.7% — actually OUTPERFORMS Tier 1's 45%.
>    Solved by scaling architecture: n_lang_input 2048→4096,
>    n_motor_per_action 500→1000, n_motor_fs_per_action 60→120.
>    Total ~12K neurons, ~5M synapses, ~6GB GPU. **Capacity hypothesis
>    confirmed**: bigger motor pools give STDP enough room for
>    functional sub-populations within each motor_X (different synonyms
>    activate different sub-pops, no winner-take-all). v1-v6 with small
>    arch (500 motor neurons) all failed; v4 with scale-up
>    succeeds. User's VRAM headroom reminder + Nord-inspired
>    "more capacity wins" insight both validated. See
>    [`research/findings/2026-05-06-Tier2.1-BREAKTHROUGH-synonym-binding-via-scale.md`](research/findings/2026-05-06-Tier2.1-BREAKTHROUGH-synonym-binding-via-scale.md).
>    Use `--biological --embodied-hebbian --synonym-mode
>    --apply-topographic-bias --enable-motor-fs
>    --n-events-per-direction 400 --n-lang-input 4096
>    --n-motor-per-action 1000 --n-motor-fs-per-action 120`.
> 8. **🎉 2026-05-07 PHASE 1.4 BRANCH A CONFIRMED — biology-grounded
>    continual learning validated.** 6-seed catastrophic forgetting
>    eval (`research/runners/continual_forgetting_eval.py`): train
>    Tier 1 4-word vocab, then train 4 NEW synonym vocab, measure
>    primary retention. **5/6 PASS at >= 80% retention, mean 103%
>    (+/- 19%).** Path F's foundational premise validated: synonym
>    training preserves (often improves) primary bindings via shared
>    motor pool reinforcement. Architecture (standard Tier 1:
>    n_lang_input=2048, n_motor=500, NMDA=True). See
>    [`research/findings/2026-05-07-Phase-1.4-v3-6seed-FINAL.md`](research/findings/2026-05-07-Phase-1.4-v3-6seed-FINAL.md).
>    Critical fix during arc: `enable_nmda=True` is required (default
>    False; v2 baseline collapsed to 25% chance without it).
>    Implementation infrastructure landed during the 6-seed wait:
>    Tier 2.3 PFC verb pool builder + phrase trainer + 3-condition
>    eval (`research/runners/phrase_trainer.py`,
>    `research/runners/phrase_eval.py`); Phase 1.3 hippocampus
>    consolidation builder + awake/sleep gate helpers + trainer +
>    hippo-OFF eval (`research/runners/consolidation_trainer.py`,
>    `research/runners/consolidation_eval.py`); Phase 1.5 unified
>    eval suite dispatcher with 4 benchmarks
>    (`research/runners/continual_eval_suite.py`); 35 unit tests
>    across new code. Master plan at
>    [`docs/plans/2026-05-06-MASTER-PLAN-main-then-pathF.md`](docs/plans/2026-05-06-MASTER-PLAN-main-then-pathF.md).
> 9. **🎉 2026-05-07 PHASE 2.1 ABC TASK PASSES (path-f-hybrid branch).**
>    Surrogate-gradient BPTT validated at toy scale: 2-layer SNN
>    (3 -> 32 LIF -> 3 LIF) trained on ABC sequence task achieves
>    100% loss reduction (3.51 -> 0.0013) in 100 epochs.
>    Implementation on `path-f-hybrid` branch (NOT main):
>    `sim/surrogate_grad.py` (ATan + fast_sigmoid CuPy),
>    `sim/bptt_snn.py` (numpy reference forward + backward unroll
>    with hard-reset surrogate gradient + recurrent chain rule),
>    `research/runners/cortex_pretraining.py` (numpy BPTT trainer).
>    11 unit tests pass. Confirmed BPTT framework correctness.
>    Next (Phase 2.2): CuPy GPU port, 4-layer architecture, Tiny
>    Shakespeare corpus, ~5-10K training steps. Phase 2.3 will
>    wire the pretrained cortex back into the biology-grounded
>    Phase 1.4 BRANCH A architecture for continual learning.
>    Master plan at
>    [`docs/plans/2026-05-06-MASTER-PLAN-main-then-pathF.md`](docs/plans/2026-05-06-MASTER-PLAN-main-then-pathF.md);
>    Phase 2.1 design at
>    [`docs/plans/2026-05-06-Phase-2.1-surrogate-grad-design.md`](docs/plans/2026-05-06-Phase-2.1-surrogate-grad-design.md).
> 10. **🎉 2026-05-07 PHASE 2.2 TINY SHAKESPEARE GPU TRAINING WORKS
>    (path-f-hybrid).** 4-layer SNN (66->128->128->66) trained on
>    Tiny Shakespeare via surrogate-grad BPTT on GPU (CuPy):
>    - 50 epochs, batch 32, T=32, 500 samples
>    - Loss 14.1 -> 2.24 (84% reduction; perplexity ~9.4 vs chance 66)
>    - Wall clock 41.5s on RTX 3090
>    Implementation on path-f-hybrid:
>    `sim/bptt_snn_gpu.py` (CuPy/numpy backend abstraction, validated
>    numerical equivalence at fp32 tolerance) + `sim/char_tokenizer.py`
>    (66-char vocab from corpus, one-hot encoding, make_seq_dataset
>    for next-char prediction). 27 unit tests pass on path-f-hybrid.
>    Init notes: first layer std=2.0 (one-hot drive needs strong
>    weights), subsequent std=0.5 (sparse spike input). Phase 2.2
>    validates the full backprop-pretraining stack. Phase 2.3 next:
>    wire pretrained cortex back into Phase 1.4 BRANCH A
>    architecture for continual learning.
> 11. **🎉 2026-05-07 PHASE 1.3 CONSOLIDATION CONFIRMED.** Hippocampus
>    -> cortex memory transfer via SWR sleep replay validated at
>    single-seed smoke. Architecture: 5 hippo regions (ec/dg/
>    dg_pv_basket/ca3/ca1) + 12 pathways including ca1 -> motor_X
>    consolidation pathways. Awake/sleep gate alternation:
>    awake = encoding ON, consolidation OFF; sleep = encoding OFF,
>    ca3_swr_burst + ca1_to_motor ON, direct lang->motor frozen.
>    **Result: hippo-OFF retention 94%** (pre-silence W->A 32%,
>    hippo-OFF W->A 30%, ratio 0.94 >> 0.50 threshold). Memory
>    truly consolidated to cortex -- McClelland 1995 / Buzsaki
>    2013 complementary learning systems theory empirically
>    validated. 25min wall clock on RTX 3090. 6-seed validation
>    in flight. See [`research/findings/2026-05-07-Phase-1.3-CONSOLIDATION-CONFIRMED.md`](research/findings/2026-05-07-Phase-1.3-CONSOLIDATION-CONFIRMED.md).
>    Use `--enable-hippocampus-consolidation` flag in the
>    biological_brain_regions builder + run via
>    `research.runners.consolidation_trainer`.
> 12. **2026-05-07 Phase 2.3a NEGATIVE finding (path-f-hybrid).**
>    Pretrained cortex (Phase 2.2 next-char SNN, loss 1.016) used
>    as adapter feature extractor for Bridge: 22% W->A vs 28% with
>    random SNN init. Both sub-baseline (Phase 1.4 was 33%).
>    Char-level next-char features don't transfer to word-action
>    binding -- direction words too phonetically similar (cosine
>    0.65-0.80). Project Nord (Path F inspiration) at 1.088B
>    params + FineWeb-Edu solves this; our toy 134K-param scale
>    is ~4 orders too small. Phase 2 INFRASTRUCTURE validated; Phase
>    2 SCIENCE thesis at toy scale FALSIFIED. See
>    [`research/findings/2026-05-07-Phase-2.3a-NEGATIVE-next-char-features.md`](research/findings/2026-05-07-Phase-2.3a-NEGATIVE-next-char-features.md).
>    For full conversational sim: scale Phase 2 ~1000x OR build
>    on Phase 1.4+1.3 biology-grounded foundation alone (10-30 word
>    vocab achievable).
> 13. **🎉 2026-05-08 Phase 1.3 + Tier 2.1 COMBINED CONFIRMED 3/3 GO
>    + ANTI-CHEAT VALIDATED.** Multi-seed (medium config, 200 events/word):
>    unanimous PASS at both thresholds. **Mean primary retention 91.2% +/- 6.5%
>    (3/3 >= 80%), mean synonym retention 128.4% +/- 6.7% (3/3 >= 60%).**
>    CLS theory generalizes from Phase 1.3's 4-word Tier 1 result to
>    Tier 2.1's 8-word vocab with synonym sub-population structure.
>    Architecture: Tier 2.1 v4 scale-up (n_lang=4096, n_motor=1000) +
>    hippocampus consolidation. Per-seed wall clock ~115 min (medium);
>    3-seed total ~6 hrs.
>
>    **Anti-cheat 3-seed (`--strict-silence`, 2026-05-08):** 10x stronger
>    hippo silencing (-2000 pA) + zeroing ~194k ca1->cortex edges per
>    seed produces IDENTICAL retention to non-strict across all 3 seeds:
>    seed 42 pri 91.9% / syn 122.7% (matches non-strict EXACTLY); seed 43
>    84.4% / 126.7% (matches); seed 44 97.3% / 135.7% (matches). 3/3 GO
>    unanimous. **Hypothesis B (cortex truly retains pattern post-consolidation)
>    CONFIRMED at 3-seed.** Hypothesis A (eval-noise from imperfect
>    silencing) FALSIFIED at 3-seed. Sleep replay genuinely transfers
>    W->A binding into cortex internal recurrence (motor<->motor,
>    motor<->lang_output, topographic prior, FS lateral inhibition);
>    cortex doesn't need hippo at all post-consolidation. Use
>    `consolidation_synonym_medium_strict` webapp preset for the
>    anti-cheat eval.
>
>    Standout: word `down` improved 4x post-strict-silence (10% pre to
>    50% hippo-OFF). Without hippo input, cortex correctly produces
>    motor_S much more often -- suggests hippo input was actively
>    HURTING down's binding; cortex has cleaner pattern.
>
>    Findings:
>    [`research/findings/2026-05-08-Phase1.3-Tier2.1-combined-3seed-CONFIRMED.md`](research/findings/2026-05-08-Phase1.3-Tier2.1-combined-3seed-CONFIRMED.md)
>    (3-seed GO),
>    [`research/findings/2026-05-08-Phase1.3-Tier2.1-anti-cheat-CONFIRMED.md`](research/findings/2026-05-08-Phase1.3-Tier2.1-anti-cheat-CONFIRMED.md)
>    (single-seed anti-cheat),
>    [`research/findings/2026-05-08-Phase1.3-Tier2.1-strict-anti-cheat-3seed-CONFIRMED.md`](research/findings/2026-05-08-Phase1.3-Tier2.1-strict-anti-cheat-3seed-CONFIRMED.md)
>    (3-seed strict anti-cheat).
>    Master plan section Phase 1.3: "This is THE mechanism that
>    makes continual learning possible without catastrophic
>    forgetting at scale" -- empirically confirmed at synonym scale
>    + 3-seed anti-cheat validated.

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

**🎯 LATEST BREAKTHROUGH 2026-05-05: G v2.5 + K v2 SCALES to 32×32 at 2.57 ± 0.11 (n=6) — 13.3% BETTER than the 16×16 baseline.**

```bash
# G v2.5 + K v2 — biology-grounded, perception only, scales to 32×32:
python -m research.runners.g11_bg_runner --moving-goal --goal-schedule multi --deterministic \
    --enable-msn-lateral-inhibition --enable-d1-d2-asymmetry --enable-striatal-pv-fsi \
    --enable-cluster-a-closed-loop --enable-cluster-e-topography \
    --enable-dlpfc-wm --enable-pfc-nmda \
    --enable-visual-cortex --visual-cortex-action-warmup-steps 600 \
    --grid-size 32 --seed N --n-steps 1800
```

**Scaling result (2026-05-05 step 3):**
- **32×32 (n=6): 2.57 ± 0.11**, range 2.42–2.72. 6/6 seeds beat 16×16 baseline.
- 16×16 (n=3): 2.97 ± 0.12 (Cluster K v2 baseline)
- 32×32 random walk baseline: ~21 estimated
- 36.1% of 1800 steps at goal (650 ± 5 per seed)
- Per-quarter: Q1 ~4.3 (exploration), Q2-Q4 ~1.7-2.3 (stable AT goal)

The architecture has demonstrated unexploited capacity — handles 4×
larger grid with TIGHTER variance. Closes 4 of 5 original cheats
(heuristic, (gx,gy), (x,y), beacon). See
[`research/findings/2026-05-05-step3-32x32-scaling-success.md`](research/findings/2026-05-05-step3-32x32-scaling-success.md)
for the smoke result and [`research/findings/2026-05-05-FINAL-autonomous-arc-synthesis.md`](research/findings/2026-05-05-FINAL-autonomous-arc-synthesis.md)
for the full autonomous arc that produced this result.

Earlier breakthrough 2026-05-01 (still valid, now superseded as flagship):
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
