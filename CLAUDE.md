# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Repository**: https://github.com/danthi123/neural-simulator

## Project Overview

GPU-accelerated neural network simulator with real-time 3D OpenGL visualization. Uses NVIDIA CUDA/CuPy for massively parallel GPU computation, simulating large-scale networks (10K-100K+ neurons) with biologically-inspired neuron models (Izhikevich, Hodgkin-Huxley, AdEx), synaptic plasticity, and spatial connectivity.

## Standing practice: deep research + catalog review FIRST at roadblocks and new directions

**(2026-06-07, owner directive — make this the default first step, not an afterthought.)** Whenever the project hits a **significant roadblock** (a multiply-confirmed boundary / repeated NEGATIVE) **OR is about to begin work on a new part of the sim**, run a **deep research + reference-catalog review BEFORE committing build/GPU resources.** This has repeatedly been the decisive pivot:
- the conversational decorrelation/whitening blocker → reframed by the Mikulasch-Priesemann point-neuron limit (whitening is analog/pre-spike in biology);
- the navigation action-selection readout boundary → diagnosed as a *missing accumulator* (Wang 2002 NMDA attractor → Lo-Wang commit burst), which fixed it;
- the navigation perceptual cold-start → root-caused as a **wrong-pathway** problem (routed through the position-*invariant* ventral "what" stream / IT instead of the dorsal "where" stream + superior-colliculus orienting + place cells) via the catalog + Kandel + literature.

**The pattern:** a read-only research subagent reviews the canonical biology catalog (`E:\Documents\Projects\sim-catalog\references\feature-catalog.md`, ~323 entries across clusters A–Q — it lives in the separate `sim-catalog` worktree), plus Kandel 6e (`references/textbooks/`), `references/glossary.md`, and current literature (WebSearch + the `bio-research` MCP biorxiv tools), and produces a findings doc: **diagnosis → ranked biologically-grounded options → what existing project machinery is reusable → a recommended cheap-first de-risk → the anti-cheat controls it needs.** The controller reviews it (trust-but-verify the load-bearing claims), pushes the doc, and presents the recommendation before building. Treat this as the standing opening move for roadblocks and new-direction work.

## Standing standard: BRAIN-BASED ONLY (neurons / synapses / their communication), or it is a shortcut

**(2026-06-08, owner directive — the load-bearing bar for "a proper brain analogue".)** Anything NOT done directly by the simulated brain — **neurons firing, synapses, and the communication between them** — is a **cheat/shortcut, EVEN IF the host-side calculation is biologically correct.** A prediction error computed by a Python formula, a "reflex" that reads pixels and returns a cardinal in code, a reward computed by a distance formula, an argmax over spike counts — all are shortcuts, because the *brain* is not doing them; the simulation's bookkeeping is.

**The boundary — host code is legitimate ONLY for:**
1. **The environment** — the world's state (agent/goal positions, the grid) and rendering the agent's sensory input (the retinal image the neural retina then receives).
2. **The body** — the agent acting on its motor output (moving based on which motor pool fires).

**Everything between sensation and action is the brain's job and MUST be neurons/synapses:** perception/salience, orienting decisions, reward, value, dopamine/neuromodulators, action selection. When a capability is realized by host computation (even biologically-shaped), it is a documented shortcut to be converted to a spiking/synaptic mechanism — and an **honest negative** (the neural version underperforming the host shortcut) **IS the scientific deliverable** (it maps what the substrate can/can't do on its own). Applies **PROJECT-WIDE** (navigation AND the conversational pipeline — e.g. the VSA composer's clean exact-inverse algebra is a host shortcut for what a learned cortex would do; see the "composer-as-idealization" note). **Re-classification:** the recent nav wins (N1 SC reflex, N5 perceived reward, N6 thal/argmax readout, N9-step-1 scalar RPE) are biologically-*shaped* but partly **host-computed → they are now shortcuts**, with their spiking/synaptic versions (a spiking superior colliculus, a neural reward/value system, a spiking SNc, a neural position code, a minimal motor read-out) the real target. The host versions become the *teaching scaffolds* for their neural replacements (the innate-reflex-teaches-a-learned-circuit pattern).

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
sim/                    # 42 modules (+ __init__.py), ~20K lines — core engine
  bridge.py             # 7919 lines — SimulationBridge + GPU state orchestration (incl. transmission_gate, graded inhibition, input-mean adaptation)
  config.py             #  925 lines — all @dataclass configs
  enums.py              #  830 lines — NeuronType (50+ presets), enums, default param managers
  connectivity.py       #  999 lines — spatial/WS/motif connection generators (backend-pluggable)
  kernels.py            #  365 lines — fused @fuse() neuron + plasticity kernels (cupy/numpy)
  profiles.py           #  432 lines — NEURAL_STRUCTURE_PROFILES + CONNECTIVITY_MOTIFS dicts
  regions.py            #  733 lines — BrainRegion + RegionPathway (incl. transmission_gate, graded, input_mean_adapt) + RegionManager
  neuromodulators.py    # 1114 lines — declarative neuromodulator subsystem
  data_bus.py           #   95 lines — DataChannel pub/sub for streaming sim data
  replicas.py           #  243 lines — replicated wiring (multi-bridge support)
  text_embeddings.py    #  273 lines — token embeddings for language regions (2026-05-01)
  visual_cortex.py      #  310 lines — Gabor RFs + retina rendering (Cluster K v2, 2026-05-01)
  bioparameter.py       #  231 lines — biological parameter helpers
  progress.py           #  214 lines — universal [PROGRESS] event format (2026-05-04)
  lineage.py            #  538 lines — BridgeLineage persistent continuous-learning + growth-log + shard export (2026-05-11)
  auto_growth.py        #  357 lines — TierPromoter + weight-transfer (auto-growth Phase A, 2026-05-11)
  backend.py            #  415 lines — pluggable xp abstraction + device helpers + RNG state (cupy/numpy, 2026-05-11)
  synapse_storage.py    #  415 lines — TieredSynapseStore + idle/pressure eviction (tiering Phase 3+4, 2026-05-11)
  bridge_memory.py      #  721 lines — BridgeMemory LLM-callable memory wrapper (Path 3 Phase 3.1.6, 2026-05-11)
  llm_memory_orchestrator.py #  452 lines — MockLLM + LLMMemoryOrchestrator tool-use loop, 5 tool schemas (Phase 3.2, 2026-05-11)
  llm_adapters.py       #  204 lines — OllamaLLM + LlamaCppLLM stub adapters (Phase 3.3 scaffold, 2026-05-11)
  # (plus ~20 newer modules from the language-generation + learned-cortex arc:
  #  surrogate_grad / bptt_snn / bptt_snn_gpu / char_tokenizer / bpe_tokenizer /
  #  tiny_transformer / ngram_* / td_value_critic / predictive_coding /
  #  dendritic_* / compose_temporal_bind / song_hvc / activity_probe / …)
viz/                    # OpenGL renderer, camera, picker, overlays
ui/                     # DearPyGUI panels, callbacks, layout, sweep panel, plots
experiment/             # ExperimentEngine + StimulusManager + ReadoutEngine + TrainingProtocolEngine
research/runners/       # 350+ headless runners (g1..g11 + cluster/text/k_v2/phase1/phase2/chat/perf_benchmark/bridge_lineage/llm_memory_demo/multibridge_chat/g20_multibridge/g20_sparse/order_intrinsic/generator_S-D-E-F-G/mode-unification/content_selection+content_selection_spiking+dialogue_agent/nested_composition+phasor_associative_memory+phasor_chat+gated_compose/unified_agent_benchmark+spiking_unified_agent/multibridge_graded_derisk+cortex_conversation_ensemble+phase1_composer_ab/etc) for research
research/findings/      # session-by-session findings docs (750+ files)
tests/                  # 291 test files (determinism, runners, kernels, plasticity, lineage, tiering, llm orchestrator, multibridge, g20-sparse, generator/BPTT, order-intrinsic, mode-unification, content-selection/dialogue, nested-composition, transmission-gate/gated-compose, unified-agent-benchmark/spiking-unified-agent, core-sim-composition + brain-conversational-agent, learned-graded cortex, etc.)
```

### Thread Model
- **Main Thread**: DearPyGUI event loop + OpenGL rendering
- **Simulation Thread**: GPU-accelerated neural dynamics computation (fully isolated)
- **Communication**: Lock-free queues (`ui_to_sim_queue`, `sim_to_ui_queue`) for inter-thread messaging

### Key Classes

**SimulationBridge** (`sim/bridge.py:210`): Central simulation orchestrator
- Manages all GPU state arrays (CuPy)
- Simulation stepping (`_run_one_simulation_step` at line 5456)
- Initialization (`_initialize_simulation_data` at line 1002)
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
- `VisualizationConfig` (line 500): OpenGL rendering and camera parameters
- `RuntimeState` (line 520): Mutable execution state (running, paused, time tracking)
- `GPUConfig` (line 535): GPU features, memory management, recording modes
- Experiment configs (lines 648–845): `StimulusPattern`, `StimulusChannel`, `NeuronGroup`, `ReadoutConfig`, `TrainingConfig`, `ExperimentPhase`, `ExperimentConfig`

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
>
> **UPDATE (2026-06-03): the complement now EXISTS — `transmission_gate`.**
> `RegionPathway(transmission_gate="name")` + `bridge.set_transmission_gate(name, value)`
> scale a pathway's effective synaptic **CURRENT** in [0,1] at runtime
> (the `cp_transmission_gain` per-synapse multiplier in `_run_one_simulation_step`,
> mirroring `cp_plasticity_rate_gain` but on current, not weight updates).
> Pre-wire a route with a fixed weight, hold it CLOSED (gate=0, no current,
> no STDP cold-start), OPEN it on command → **thalamocortical dynamical
> gating**: binding = which gate is open, not which weight grew
> (Logiaco-Abbott-Escola 2021). Validated in spikes
> (`tests/test_transmission_gate.py`): closed → target silent; open → target
> fires; re-binding reroutes the same source with **zero weight change**,
> where grown weights could not. Default `None` = always-on (additive, zero
> overhead unused). See `2026-06-03-deep-research-surpassing-the-blockers-synthesis.md`.

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

### 🎉 OPPONENCY ESCAPED — FHRR-on-bridge composer is the conversational PRODUCTION DEFAULT (2026-06-05)

**`BrainConversationalAgent` now defaults to the FHRR-on-bridge `RFPhasorComposer` (opponency-free).** The composer's
last numpy op — `onoff(bon−boff)` opponency (common-mode removal of a small signed difference of correlated channels)
— was confirmed a FUNDAMENTAL rate-coded SNR wall: 3 independent spiking mechanisms NEGATIVE (simple accumulator
0.41, NEF integrator 0.90-aggregate/0.077-unbind, bipolar/WTA 0.385), because biology removes the common mode in the
ANALOG stage BEFORE spiking — rate codes physically can't (Kandel 6e Ch 22, the retina). Fix: pivot the bound-vector
representation from the ±1 Hadamard to **spiking-phasor FHRR** (Frady-Sommer 2019 resonate-and-fire phasor neurons +
complex synapses) — unit-magnitude, info in PHASE, so there is no common mode and no small signed difference and the
opponency simply does not exist. Realized ON the bridge: new `NeuronModel.RESONATE_AND_FIRE` (complex state
Z=re+i·im reusing v/u; rotate `exp(λ+iω)`; Im zero-crossing spike = phase) + complex synaptic matvec
(`rf_set_complex_weights`, SPARSE) + a dedicated `rf_resonate_steps` fast loop + `rf_kick`/`rf_read_phases` (all
ADDITIVE/guarded protected `sim/` edits — Izhikevich/HH/AdEx byte-unchanged; the bind/unbind/bundle happen THROUGH
complex synapses, Frady-Sommer). `RFPhasorComposer` (`research/runners/rf_phasor_composer.py`) reproduces the full
capability matrix (who/what Q&A, abstention, negation/yes-no, one-attribute, recursive clauses, dialogue, generation)
multi-seed; 320-concept correctness GO (8/8/8). The agent's FULL existing suite (`tests/test_brain_conversational_agent.py`)
passes VERBATIM on the RF default — behavioral parity, no-confab moat intact, ZERO regression (29 GPU tests). Rate
composer = explicit opt-in (`composer_kind='rate'`); the separate 320-concept retrieval pipeline is untouched. The
F=3 two-attribute resonator (which the ±1 scheme provably can't do) is now available to lift the K=5 boundary
(follow-on). Findings: `2026-06-05-fhrr-production-switch-DONE.md`, `-fhrr-layer-{a,b,c}-*`,
`-B-opponency-rate-coded-SNR-wall-CONFIRMED.md`, `-FHRR-pivot-derisk.md`,
`-spiking-opponency-literature-synthesis.md`. Plan: `docs/plans/2026-06-05-full-fhrr-on-bridge-feature-plan.md`.

**Known limitation — composer is a principled idealization, not a functional cortex (2026-06-06):** the
FHRR/VSA composer is a *principled idealization* (Eliasmith Spaun / Semantic Pointer Architecture — a
serious hypothesis that cortex binds VSA-like), NOT a functional reproduction of cortex. Its binding is
a clean, exactly-invertible ALGEBRA that DEMANDS decorrelated full-precision codes (the whole whitening
requirement is downstream of this); a real cortex has LEARNED, lossy, redundant read-outs that learn to
read whatever messy code arrives. The binding OPERATIONS are already on-substrate spiking (FHRR
resonate-and-fire + complex synapses); the residual idealization is the exact-inverse algebra + the
clean-code demand. The spike-native robustness ladder (a phase-encoded handoff, b temporal integration,
c population redundancy + attractor cleanup) makes the scaffold spike-FAITHFUL; the genuine-cortical
conversion (d: learned read-outs replacing the fixed algebra) is **BENCHED** below the planned work
(cheat/shortcut removal → single-brain consolidation → capability addition + scaling). NOT labelled a
"cheat," but stay cognizant it is not functionally identical to the cortex it stands in for. Trade-off:
the algebra buys the no-confab moat + compositional reliability ~free; a learned cortex does not.
See `research/findings/2026-06-06-composer-vsa-idealization-known-limitation.md`.

### Conversational pipeline CONSOLIDATED onto the core sim (2026-06-04)

**The production conversational agent runs ON the core `SimulationBridge` (the brain), not on a
bolted-on numpy simulator.** Per the owner's directive ("the core sim IS the simulated brain;
capabilities realized through it, no bolted-on modules"), the conversational loop —
comprehend / store / recall / who-what Q&A / abstention / negation / clauses / one-attribute /
dialogue planning — was consolidated onto three interacting core-sim bridges:

- **`research/runners/core_sim_composition.py` (`CoreSimComposer`)** — role-filler VSA composition
  computed by **spiking coincidence neurons** on a real ~6400-neuron Izhikevich bridge (the ±1
  Hadamard: `bound_ON=AND(role_ON,fill_ON)+AND(role_OFF,fill_OFF)`), reused for unbind; SVO fact
  memory, who/what Q&A, abstention (the no-confab moat → `None` when no fact's agent matches),
  negation/yes-no (a bound polarity tag). Concept codes are the substrate's own (`denoise64`).
- **`research/runners/brain_conversational_agent.py` (`BrainConversationalAgent`, `BridgeParser`)** —
  the full loop: a **Hebbian-learned parser bridge** (comprehension: `(word-position × voice) → role`,
  voice-invariant — active "dog go north" and its passive frame assign the same agent) + the composer +
  recursive **clauses** + **dialogue planning** (`elaborate(topic)` via the dlPFC spiking
  content-selection Control over an association graph built from the agent's own facts).
- 10 on-brain regression tests pass: `tests/test_core_sim_composition.py` (5) +
  `tests/test_brain_conversational_agent.py` (5). All build a real bridge; they skip gracefully if
  the `denoise64` concept-code cache is absent.

**Honest residual:** the ±1 coincidence scheme cannot invertibly bind two concept codes (adj⊗noun) —
attributes use a feature-binding ATTRIBUTE role-tag: **1-attribute RESOLVES, 2-attribute is a
documented K=5-load BOUNDARY**, and the FHRR F=3 resonator stays a **numpy reference**. Vocab is the
validated probe scale (V=16); production 320-concept on the brain agent is a follow-on.

**COMPOSER CLEANUP SHORTCUT CLEARED — spiking NEF cleanup (2026-06-05):** the composer's last numpy readout
(the `np.argmax([concepts[w]·est])` nearest-concept cleanup in `unbind`/`_render_filler`) now has a validated,
fully-spiking, biology-grounded replacement: the **NEF thresholded cleanup** (Stewart-Tang-Eliasmith 2011, the
cleanup inside Spaun). Opt-in `CoreSimComposer(enable_spiking_cleanup=True)` builds a persistent cleanup bridge from
the codebook (operating point `NEF_CLEANUP_OP`: input-normalized matched filter + per-concept firing threshold placed
so off-target emits ZERO spikes + n_per=12 noise averaging) and routes the cleanup through it; **== numpy on the
capability matrix at production D=2048 multi-seed (27/27 seeds 42/43/44, no regression, NO sim/ edits).** Reached via
owner-steered deep research after 3 hand-tuned mechanisms plateaued/failed (divisive-norm 0.84, two-stage 0.91,
hand-WTA 0.13 — the last violated the Rutishauser α>1 WTA-stability condition). Key insight: a rate readout is a
LINEAR reconstructor (off-target leak caps it ~0.91); a placed threshold discretizes it to argmax parity. The grounded
agent enables it; numpy stays the fast default. Findings: `2026-06-05-composer-cleanup-NEF-GO.md` +
`-spiking-cleanup-memory-literature-synthesis.md`. The deeper **(B) memory shortcut** (the numpy-held bound fact +
numpy superposition/opponency) is the remaining full-clear piece (options: `docs/plans/2026-06-05-composer-B-substrate-held-memory-options.md`).

**ONE-BRIDGE UNIFICATION COMPLETE (2026-06-04):** the three conversational regions now run as disjoint
persistent slices on ONE interacting `SimulationBridge` — `research/runners/unified_brain_bridge.py`
(`UnifiedBrainBridge`). Step 1: parser + composer share the bridge (no capability regression at
production D=2048 multi-seed; a `plastic=False` population still drifts under global Hebbian, so the
composer's fixed bind population is frozen by a per-synapse plasticity gate, `cp_plasticity_rate_gain=0`).
Step 2: the parser→composer hand-off is SYNAPTIC — comprehension routes composition in spikes via a
parser-gated transmission route (`hear_synaptic`); a transmission gate coupled to a BURSTY control needs a
working-memory LATCH to hold routing during the downstream read (comprehend→latch→compose). Step 3: the
dlPFC dialogue-planning loop (`enable_dlpfc=True`) merges at dt=1.0 — its NMDA-dependent WM latch survives
dt=1.0 (de-risked at the genuinely NMDA-dependent attractor weight 30, not the saturated 50 = AMPA
ping-pong); a per-region NMDA mask isolates NMDA to the dlPFC slice; `elaborate` reproduces the dlPFC's
validated dialogue-planning function with no regression. QUALIFIED nuance: rank-order (latency) coding
RESOLUTION is dt-bound, so at dt=1.0 equidistant direct neighbours tie and the tie-break may pick a
different-but-equally-valid associate than the dt=0.5 oracle (the GATE asserts the validated function, not
the tie-break). NO `sim/` edits anywhere in the unification (reuse-by-import). Findings:
`2026-06-04-one-bridge-unification-step1-capability.md`, `-step2-DONE.md`, `-step3-dlpfc-dt-survives.md`,
`-step3-dlpfc-MERGED.md`.

**The two standalone numpy phasor simulators are REFERENCE-only, NOT the production substrate:**
`research/runners/spiking_phasor_fhrr.py` + `resonate_fire_fhrr.py` (and the unified agents that import
them — `nested_composition_agent` / `spiking_unified_agent` / `unified_agent_*`) carry a NUMPY-REFERENCE
header and are retained only as the FHRR validation ceiling. Do not treat them as "the brain analogue."

Finding: `research/findings/2026-06-04-conversational-pipeline-consolidated-onto-core-sim.md`.
Audit: `research/findings/2026-06-04-conversational-pipeline-substrate-audit.md`.
Plan: `docs/plans/2026-06-04-consolidate-conversational-pipeline-onto-core-sim-design.md`.

### 🧠✅ NAVIGATION + CONVERSATIONAL merged onto ONE bridge — roadmap step 2 DONE (2026-06-10)

**STATUS: roadmap step 2 COMPLETE (2026-06-10).** The navigation cascade, the conversational parser, the dlPFC
dialogue planner, AND the resonate-and-fire (RF) composer now run as **disjoint neuron-index slices on ONE
`SimulationBridge` with one step loop**, capability-equivalent to the separate brains (STEP 2a + 2b both
COMPLETE, all acceptance gates GREEN — see the per-step bullets below). The remaining frontier is step 3 (the
true learned cortex), deferred to its own arc.

After navigation was fully biologized (every cognitive computation between sensation and action is a
validated neural mechanism — N1 spiking superior colliculus, N5 neural reward, N6/N8/N9 spiking selection +
disinhibition + dopamine RPE, N2/N7 defensible perception), the arc was **consolidating the navigation
brain and the conversational brain onto ONE `SimulationBridge`** (the owner's "one brain" directive). Builder:
`research/runners/nav_conv_merged_bridge.py` (`build_merged_nav_conv_bridge` + `MergedNavConvAgent`). The whole
arc was de-risked cheapest-first BEFORE any protected edit:

- **De-risk 5a (plasticity isolation) — PASS + one characterized gap.** The per-synapse plasticity gate
  (`cp_plasticity_rate_gain=0`) isolates weight UPDATES against the full navigation stressor (reward-STDP +
  the global dopamine `scope="all"` + Hebbian) — a frozen conversational slice stays byte-identical, controls
  change, a conversational read is unchanged across a navigation burst. THE ONE GAP: the two global weight
  CLIPS (`bridge.py:6261` Hebbian, `:6566` reward) are UNGATED, so a frozen weight OUTSIDE the active rule's
  clip bounds is moved by the clip. **Mitigation:** raise `stdp_w_max` + `hebbian_max_weight` above the frozen
  conversational real-valued weight (~300); the RF composer's COMPLEX binding weights (`cp_rf_w_re/im`) are
  array-disjoint from `cp_connections` so they are IMMUNE. Findings:
  `2026-06-10-unification-5a-plasticity-isolation-PASS-with-clip-caveat.md`.
- **De-risk 5b (RF vs Izhikevich) — KILL confirmed → the minimal protected edit.** RF stores its complex
  phasor in the same `v`/`u` arrays Izhikevich uses; one Izhikevich step destroys a phasor (|z| 1.0 → 16.3).
  But the composer is stateless-per-op (re-kicks each op) and stores memory in complex synapses, so the
  minimal edit is to **slice the RF ops** (not a core-step-loop dual-dispatch): `rf_kick(..., neuron_mask=)`
  + `_rf_advance_one` mask all `v`/`u` writes to the RF slice. **Default `None` = byte-identical** (18/18
  conversational tests pass verbatim incl. the no-confab moat); validated co-residence (an RF op on a masked
  slice == a standalone RF bridge exactly, the Izhikevich slice byte-isolated). **OWNER-APPROVED** for the
  strict (RF co-resident) merge. `tests/test_rf_neuron_mask_coexistence.py`. Findings:
  `2026-06-10-unification-5b-*` + `2026-06-10-unification-sliced-RF-ops-edit-byte-review.md`.
- **STEP 2a (merged bridge, RF composer external) — COMPLETE.** The framework path IS a wrapper around
  `inject_explicit_wiring` (`bridge.py:2196`), so the parser + dlPFC are appended as framework regions.
  The conversational gate (b) passes VERBATIM on the merged bridge — `tests/test_nav_conv_merged_agent.py`
  8/8 incl. the three `is None` no-confab assertions (`what_does`/`elaborate`/`describe`). The navigation gate
  (a) uses a HYBRID `run_moving_goal_episode` integration (4 additive no-op-default params + an index-based
  `finalize_conv_for_nav_gate` hook that runs AFTER the V1/SC post-init `set_pathway_weights(add_missing=True)`
  CSR rebuild — which re-sorts the data + stales gate-index maps + the Hebbian decay would erode the fixed
  perception weights; the hook handles all three by masking by index, not gate name). The **nav-on-merged
  smoke PASSES**: the merged bridge navigates AND the conversational populations stay byte-frozen in vivo
  under the live navigation reward-STDP + dopamine stressor. A `stdp_w_max=400` cheap-check confirmed the
  navigation score is byte-identical to 150 (the actor is ceiling-bound, not soft-bound — over-grows to 311 —
  but inert because the spiking WTA readout saturates). **Navigation gate (a) = PASS (GREEN_INERT):** the
  standalone-vs-merged score is BYTE-IDENTICAL (sum 2.0, per-phase `[0.496,0.504,0.496,0.504]`) at every completed
  seed (3/6; the remaining 3 cancelled by owner authorization to free the GPU for 2b — byte-identity is exact +
  mechanistically seed-independent for this inertness/null gate, so 3 byte-identical = conclusive, distinct from
  the standing 6-seed rule for variable effects). Tool: `research/runners/nav_gate2a_aggregate.py` (9 tests).
  Design: `docs/plans/2026-06-10-nav-conv-merge-implementation-design.md` +
  `docs/plans/2026-06-10-nav-episode-integration-design.md`. Findings:
  `2026-06-10-step2a-nav-gate-a-PASS-3of6-byte-identical.md`, `2026-06-10-nav-on-merged-smoke-PASS-*`.
- **STEP 2b (RF composer co-resident on the one bridge) — COMPLETE.** Via the owner-approved masked RF ops (an
  `rf` region with no `cp_connections` out-edges; the composer driven through `rf_kick(neuron_mask=rf_mask)`). Opt-in
  `MergedNavConvAgent(co_resident_composer=True)` (default off = STEP-2a byte-preserved); `MergedRFComposer`
  overrides only `_resonate` to address the rf slice. All three acceptance gates GREEN: (1) CPU bit-exactness +
  byte-isolation `tests/test_merged_rf_composer_coresident.py` 5/5 (== standalone composer to atol 1e-9; the
  co-resident Izhikevich slice byte-identical across the op); (2) the full conversational matrix co-resident at
  production D=128 `tests/test_nav_conv_step2b_coresident.py` 7/7 on GPU (incl. the `is None` no-confab moat + the
  co-residence anti-cheat); (3) nav-not-regressed-with-rf = 2.0 byte-identical (Δ=0). NO sim/ edit (beyond the
  default-off masked op). **⇒ ROADMAP STEP 2 (consolidate nav + conversation onto ONE bridge) DONE** — nav + parser
  + dlPFC + composer all on one `SimulationBridge`, capability-equivalent. HONEST SCOPE: a consolidation of EXISTING
  capabilities, not a new one; the composer's exact-inverse VSA binding stays the principled idealization (= step 3).
  Finding: `2026-06-10-step2b-rf-composer-coresident-COMPLETE.md`.
- **Step 3 (true cortex) — DE-RISKED to a FORK (2026-06-11); flat-cortex (A) no-confab moat validated.** The
  arc to replace the composer's exact-inverse vector-symbolic-algebra (Fourier Holographic Reduced Representation,
  "FHRR") idealization with a learned spiking-cortical binder was run to ground cheap-first. **Core finding:** the
  brain's own concept codes are CORRELATED (carry semantic similarity), and **four mechanistically-distinct
  brain-based mechanisms FAILED to decorrelate them on the point-neuron substrate** — vanilla Hopfield
  (common-mode collapse), Storkey local covariance (locality wall: only a NON-local matrix inverse removes the
  common mode), spiking dentate-gyrus (sub-reproducible read), and a fixed random expansion / Marr-Albus granule
  recoding (the common mode survives the linear expansion; threshold units flip under realistic noise). All four
  converge on the **documented Mikulasch-Priesemann point-neuron limit: decorrelation/whitening is an ANALOG /
  pre-spike (dendritic) computation a point-neuron substrate fundamentally cannot do** (the project's prior
  conversational whitening blocker, "Standing practice" above). Conversely, on DECORRELATED codes everything
  works: the distributed attractor cleanup recovers 1.000, AND a LEARNED binder generalizes SYSTEMATICALLY to
  never-seen role-filler combinations (Fodor-Pylyshyn held-out test, held-out=1.000=train, 3 seeds, leakage-
  asserted, vs memorization-floor 0.000). **⇒ THE FORK (owner decision, `docs/plans/2026-06-11-cortex-build-plan-decorrelate-then-bind.md`):
  (A) a semantically-FLAT cortex** (generated decorrelated codes + the validated binder + cleanup + no-confab
  gate) is **achievable now** and already passes the full conversational matrix at V=320, but cannot generalize
  across similar concepts; **(B) a semantically-STRUCTURED cortex** (preserve the correlated semantic codes →
  generalization) needs the **deferred dendritic-substrate rewrite** (months-scale, Mikulasch-Priesemann-mandated)
  — the path to a proper, biology-translatable brain analogue that generalizes. **Flat-cortex (A)'s last
  brain-based gap closed:** the no-confab abstention moat (currently a host check) now has a VALIDATED neural
  replacement — the learned Bogacz-Brown familiarity gate matches the host abstention decision at V=320 multi-seed
  (agreement 168/168 every seed, **zero moat-breaches**, zero abstention-floor false-accepts; validated ALONGSIDE
  the host, moat NOT weakened). Findings: `2026-06-11-cortex-{storkey-ca3,dg-ratekwta,fixed-expansion-decorrelation}-*.md`,
  `2026-06-11-cortex-sparse-attractor-poscontrol-GO.md`, `2026-06-11-cortex-learned-binder-systematicity-NEGATIVE-ON-CORRELATED.md`,
  `2026-06-11-familiarity-gate-v320-GO.md`, `2026-06-11-cortex-core-learned-binder-research.md`. The (B)
  dendritic rewrite remains the deepest/highest-variance open problem and a deliberate owner call.

- **UPDATE (2026-06-15) — the GENERALIZING learned cortex is achievable WITHOUT the (B) dendritic rewrite,
  and is REALIZED on the spiking substrate, learned from the conversation stream.** The fork's (B) framing
  ("decorrelate the correlated codes → needs the dendritic rewrite") was superseded by the CYCLE-88 reframe:
  the off-diagonal decorrelation was a **red herring**. A generalizing cortex needs **feedforward LOCAL
  normalization** (PPMI = log + per-hub + per-concept mean-subtraction + threshold, all local ops), NOT
  cross-neuron decorrelation (which would *destroy* generalization). PPMI codes reach host (+0.518) AND
  generalize (held-out 0.86), land in the binding sweet spot, and pass the full who/what + no-confab pipeline
  (CYCLE 88-90, numpy). The biology-faithful **online STREAM** version — a cortex that hears the corpus
  word-by-word (online Hebbian co-occurrence + running-frequency, NO preprocessing, NO global matrix) —
  reaches the target (CYCLE 94, +0.513). And it is now **realized ON THE REAL SPIKING SUBSTRATE** (CYCLE
  95-96): rate-Hebbian co-occurrence learning (6-seed `corr(M,C) +0.686`; STDP is the WRONG rule — measured
  656k events / 0 weight change at `delta_t≈0`, because symmetric co-occurrence has no pre→post order) +
  the **population code** (lifts the single-neuron read-out from 47% → **100-108%** of host-ref, the
  documented rate-code-wall lift) + the full conversation on the **stream-learned** codes (3-seed who/what
  recall **1.00**; no-confab moat **0.96** — 1 tail false-accept on the lowest-fidelity seed = the
  code-fidelity cost, NOT a moat-mechanism weakening; the lever is more stream → wider familiarity gap,
  never a looser gate). HONEST SCOPE: validated at 64 concepts; the on-bridge absolute fidelity is
  window-budget-bounded (a wall-clock cap, not a substrate limit — `corr(M,C) 0.885` shows faithful
  learning). The **320-concept stream-scaling** (needs a corpus-grounded 320-word taxonomy) and the
  **on-bridge log-domain normalization CIRCUIT** (the read-out double-centring is currently a host-side
  scaffold; CYCLE 93b builds it as per-concept feedforward inhibition + per-hub adaptation, POST-f-I) are
  the remaining build. ⇒ a generalizing, biology-faithful, **learned-from-conversation** cortex on the
  point-neuron substrate (with population coding); the months-scale dendritic rewrite is NOT required for
  this generalizing cortex. Finding:
  `research/findings/2026-06-15-on-bridge-hebbian-co-occurrence-learning-mechanism-GO.md` (+ the CYCLE
  88-94 PPMI/stream findings: `2026-06-15-off-diagonal-red-herring-ppmi-local-normalization-reaches-host.md`,
  `2026-06-15-biology-faithful-online-stream-cortex-reaches-target.md`).

`SIM_BACKEND=cupy` (GPU) is required for the merged-bridge runs (numpy is a tiny-smoke / CI path only).

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

### Concept pool architecture (2026-05-13): diversity beyond 4 motor pools

**User mandate 2026-05-12:** "those scaling axes are 100% what need to
be given our full focus currently, as the blocker for reaching
conversational capabilities... it needs concepts, composition, and
diversity."

Root-cause diagnosis: every conversational ceiling (P5 2/4, Tier 2.3
34-40%, in-vivo 2/4 fixed capacity, synonym32 W→A 44%) shares ONE
common cause — only 4 motor pools. Every concept must collapse onto
one of the four cardinal directions.

**Solution:** mirror the proven Tier 1 6/6 multi-seed recipe (500-neuron
pool + paired teacher current + FS cross-inhibition + reciprocal
lang_output) for non-direction concept categories. Each kind gets its
own pools alongside the existing 4 motor pools.

**Architecture additions in `sim/research/runners/text_minimal_isolation.py`:**

`build_biological_brain_regions` parameters:
- `enable_noun_pools` + `noun_pool_names` + `n_noun_per_pool` + `n_noun_fs_per_pool`
- `enable_verb_pools` + `verb_pool_names` + `n_verb_per_pool` + `n_verb_fs_per_pool`
- `concept_to_language_output_density` / `_weight` / `_jitter`

Internal helper `_add_concept_kind(kind, names, ...)` builds:
- Per-pool BrainRegion (Tier 1 cortical canon)
- `lang_input → pool` plastic pathway (gate-tagged `language_input_to_{kind}_pool`)
- Reciprocal `pool → language_output` plastic pathway (gate-tagged `{kind}_pool_to_language_output`)
- FS interneurons WITHIN kind (no cross-kind FS — deliberate design
  choice so "go north" can fire verb_pool_GO + motor_N together)

**Three demo runners:**
- `research/runners/concept_pool_demo.py` — Phase 1 cross-category
  isolation (typing "apple" → noun_APPLE, NOT motor_N or verb_GO)
- `research/runners/concept_compose_demo.py` — Phase 2 composition
  (sequential + co-fire merging; tests NMDA bistability)
- `research/runners/concept_speak_demo.py` — Phase 3 A→W readout (drive
  pool → decode "spoken" word from language_output cosine)

**Supporting infrastructure:**
- `research/runners/concept_pool_repl.py` — interactive shell
- `research/runners/concept_weight_probe.py` — diagnose trained weights
- `research/runners/concept_pool_aggregate.py` — multi-seed analysis
- 21 tests (15 unit + 6 integration) all PASS, CPU-only

**Default vocab v2 (12 distinct output pools, 3× diversity over Tier 1):**

| Kind | Pool count | Words |
|---|---|---|
| Motor (existing Tier 1) | 4 | north, east, south, west |
| **Noun (NEW)** | **4** | apple, river, dog, cat |
| **Verb (NEW)** | **4** | go, come, stop, look |

Optional 3rd kind:
| Adjective (opt-in via `--enable-adjective`) | 4 | big, small, hot, cold |

→ 14 pools with adjectives = 3.5× Tier 1 diversity.

**Webapp wire-up:**
- PRESETS["concept_pool_demo"] / ["concept_compose_demo"] / ["concept_speak_demo"]
- ui.js category "Concept pool architecture" (sky blue)
- index.html launcher dropdown options

**Seed 42 v1 (10 pools, 2 verb): 0/10 PASS** — verb_pool_COME structurally
dominated 9/10 words due to FS within-kind imbalance (2 verb pools = 1
cross-FS edge per FS vs 3 for 4-pool kinds).

**Seed 42 v2 (12 pools, 4 verb + tighter topographic + target-only STDP
gating): IN FLIGHT (~2-3 hr/seed at full scale, ~35 min at smoke scale).**

Wall-clock note: with 12 pools and NMDA dynamics, training per word is
~150-180s at biological scale (vs Tier 1's 100s for 4 pools). 200
events × 12 words = 2-3 hours per seed. Full multi-seed (4 seeds) ≈
10 hours. Use --n-train-events 50 + smaller pools for faster smoke
testing if needed.

**v7 production recipe (2026-05-13, post-iteration arc):**

After 8 architectural iterations (v1→v7), the production recipe is:
- 12 pools (4 motor + 4 noun + 4 verb) for FS symmetry
- Target-only STDP gating (only target kind's gate open per word)
- Weak dynamics for concept pools (0.05/0.3/0.8); motor canon
- Topographic prior 3.0/0.3 (10x ratio), target-priority bias
- Interleaved training (shuffled events match Tier 1 pattern)
- 200 events per word, 2048 lang_input, 200 per pool

```bash
python -m research.runners.concept_pool_demo --seed N \
    --n-train-events 200 --n-lang-input 2048 --n-per-pool 200 \
    --n-fs-per-pool 24 --weak-concept-dynamics --interleaved \
    --topographic-factor 3.0 --off-target-factor 0.3
```

**v7 5-seed (W→A only):** mean **6.4/12 (53%)**, std 0.89,
range 5-7. All 5 seeds PARTIAL on Phase 1 cross-category isolation.

**🎉 v9 BREAKTHROUGH (2026-05-13): bidirectional binding multi-seed validated**

v9 adds reciprocal topographic bias to pool → language_output. v9
5-seed result:
- Phase 1 W→A: mean 6.2/12 (52%), std 0.89
- Phase 3 A→W: **60/60 = 100% UNANIMOUS** across all 5 seeds

| Variant | Phase 1 W→A | Phase 3 A→W |
|---|---|---|
| v7 (forward bias only) | 6/12 | 0/12 |
| v8 (+ weight 0.5→2.0) | 6/12 | 0/12 |
| **v9 (+ reciprocal bias)** | **6/12 (mean 6.2 5-seed)** | **12/12 unanimous 5/5** |

**🎉 v11 SCALE BREAKTHROUGH (2026-05-13 evening): 16 pools work BETTER**

Added adjective pools (BIG/SMALL/HOT/COLD) via --enable-adjective.
Total 16 distinct output pools = 4 motor + 4 noun + 4 verb + 4 adj.

Single-seed result on seed 42:
- Phase 1 W→A: **11/16 PASS (69%)** — BETTER than v9's 6/12 (50%)
- Phase 3 A→W: **16/16 PASS (100%)** — every pool speaks its word

The architecture's discrimination IMPROVES with more output diversity.
v9's failed words (south, west, come) now PASS in v11. Hypothesis:
larger off-target set spreads structural bias; no single pool can
dominate ALL competitors when there are 15 of them.

Multi-seed v11 in flight.

```bash
python -m research.runners.concept_pool_demo --seed N \
    --n-train-events 200 --n-lang-input 2048 --n-per-pool 200 \
    --n-fs-per-pool 24 --weak-concept-dynamics --interleaved \
    --topographic-factor 3.0 --off-target-factor 0.3 \
    --enable-adjective  # ← v11 16-pool architecture
```

V10 (NMDA tau 250ms uniformly) NEGATIVE: collapses both Phase 1 and
A→W via "canon amplifies bias" mechanism. Lesson: per-pool persistence
needs dedicated holding region (dlpfc_verb pattern, Tier 2.3), not
uniform NMDA extension. v12 = dlpfc_verb integration for sequential
composition is queued.

**🎉🎉🎉 v14 5-SEED MULTI-SEED GO (2026-05-13 night): orthogonal codes
+ 16 pools, NEW PRODUCTION RECIPE**

Hash-based vocab_to_drive_pattern produced ~10% pairwise overlap.
Per-word structural overlap → seed-dependent fragile words (v11
multi-seed: half words fragile across seeds). v14 uses
orthogonal_drive_pattern: each word gets a non-overlapping band.

**5-seed FINAL result (seeds 42-46):**
- Phase 1 W→A: **mean 12.4/16 (77.5%), std 1.52, range 11-15** —
  up from v11 mean 9.0/16 (56%) and v9 mean 6.2/12 (52%)
- Phase 3 A→W: **80/80 = 100% UNANIMOUS** across all 5 seeds
- TOTAL: **142/160 = 88.75% bidirectional binding multi-seed**
- 5 GO + 0 PARTIAL + 0 FAIL

**Per-word robustness (5 seeds):**
- Robust 5/5: west, apple, cat, come, hot, cold (6 words)
- Robust 4/5: east, south, go, stop (4 words)
- Mixed 3/5: north, river, dog, small (4 words)
- Fragile 2/5: look, big (2 words)

10 of 16 words robust at 4-5 seeds; orthogonal codes lift the W→A mean
+22pp over v11. The architectural ceiling at this scale appears to be
around mean 80% W→A with consistent 100% A→W.

```bash
python -m research.runners.concept_pool_demo --seed N \
    --n-train-events 200 --n-lang-input 2048 --n-per-pool 200 \
    --n-fs-per-pool 24 --weak-concept-dynamics --interleaved \
    --topographic-factor 3.0 --off-target-factor 0.3 \
    --enable-adjective --orthogonal-codes --sparsity 0.05 \
    --save-bridge <out.h5> --out <out.json>

python -m research.runners.concept_speak_demo --seed N \
    --n-lang-input 2048 --n-per-pool 200 --n-fs-per-pool 24 \
    --weak-concept-dynamics --enable-adjective --orthogonal-codes \
    --sparsity 0.05 --load-bridge <out.h5> --out <speak.json>
```

Total wall-clock 5-seed: ~85 min (~17 min/seed). 16-pool architecture
(4 motor + 4 noun + 4 verb + 4 adjective) trains with 200 events/word
× 16 words = 3200 interleaved events. Demonstrates 4× concept diversity
over Tier 1's 4-motor-only ceiling with reliable bidirectional binding.

**Findings:** `research/findings/2026-05-13-concept-pool-architecture-Phase1.md` §v14

## v15 (unidirectional verb→dlpfc→motor): NEGATIVE multi-seed (2026-05-13)

Tried per the sequential composition design note. v15 adds
`verb_pool_X → dlpfc_verb` (forward, plastic) + `dlpfc_verb → motor_X`
(forward, plastic, gated) wiring — no back-feedback to verb_pool
(the v12 leakage source). Three iterations:

- v15a: weight 2.0 + jitter 0.2 + canon dlpfc → 8/16 seed 42
- v15b: zero weight + zero jitter + canon dlpfc → 8/16 (proves
  not weight_jitter)
- v15c: weak dlpfc dynamics + zero-init pathways + skip lang→dlpfc
  → 11/16 seed 42, 5-seed mean 11.2/16

But Phase 3 A→W collapsed 5-seed mean: 3.2/16 (25%) vs v14's
16/16 unanimous (100%). A→W failure pattern shows off-by-1 (each
pool speaks the previously-trained word), suggesting the 200-neuron
dlpfc_verb region perturbs eligibility-trace state across training
events.

**Conclusion:** adding the dlpfc_verb region (with any dynamics
strength tested) breaks v14's reciprocal binding. v15 cannot be a
drop-in.

For sequential composition, pivoted to v16: direct verb_pool → motor
plastic pathways (no new region). Hebbian "go + north co-firing"
hypothesis. See findings doc §v15 NEGATIVE.

## 🎉 v16 5-seed MULTI-SEED GO (2026-05-13 night): compositional substrate validated

V16 adds **direct verb_pool_X → motor_Y plastic pathways** (16 total =
4 verbs × 4 motors), zero-init + zero-jitter so Phase 1 is preserved.
No new region (v15's downfall). 5-seed validation:

| Seed | P1 W→A | P3 A→W | Total |
|---|---|---|---|
| 42 | 13/16 | 16/16 | 29/32 |
| 43 | 12/16 | 16/16 | 28/32 |
| 44 | 11/16 | 16/16 | 27/32 |
| 45 | 12/16 | 16/16 | 28/32 |
| 46 | 11/16 | 16/16 | 27/32 |
| **Mean** | **11.8/16 (74%)** | **80/80 = 100% UNANIMOUS** | **27.8/32 (87%)** |

vs v14 5-seed: P1 -0.6/16 (-3.75pp); A→W identical (both 100%); total
-1.85pp. **v16 is a near-drop-in for v14** — minor P1 regression from
added structural pathways is within noise, A→W perfect across all 5 seeds.

**5/5 GO** with 5 robust + 9 mixed + 2 fragile words (look=2/5, big=2/5).

**Critical:** v16 pathways are STRUCTURALLY present but FUNCTIONALLY
silent (zero weights) until compose training opens the
`verb_to_motor_direct` gate and drives (verb, motor) co-firing.
Compose training is the next step (will grow weights from 0 via STDP).

```bash
# v16 production recipe (Phase 1 with compositional substrate):
python -m research.runners.concept_pool_demo --seed N \
    --n-train-events 200 --n-lang-input 2048 --n-per-pool 200 \
    --n-fs-per-pool 24 --weak-concept-dynamics --interleaved \
    --topographic-factor 3.0 --off-target-factor 0.3 \
    --enable-adjective --orthogonal-codes --sparsity 0.05 \
    --enable-direct-verb-to-motor \
    --save-bridge <out.h5> --out <out.json>
```

**v16 is the new production recipe for Phase 1 + composition-ready.**

**Findings:** `research/findings/2026-05-13-concept-pool-architecture-Phase1.md` §v16

## 🎉🎉 v16 + compose-training: FIRST COMPOSITIONAL BINDING (2026-05-13 night)

Implemented `research/runners/concept_compose_train.py`: loads a v16
bridge, freezes Phase 1 plasticity gates, opens `verb_to_motor_direct`
gate, and trains (verb, motor) co-fire pairs with temporal offset
(verb fires before motor for LTP-favorable STDP).

**Single-seed (seed 42) results, 4 compose pairs:**

| Events/pair | Compose PASS | A→W (post-compose) |
|---|---|---|
| 100 | 2/4 ('come', 'look' strong) | **16/16 PERFECT** |
| 400 | 2/4 ('go' 1.62x, 'come' 1.23x) | **16/16 PERFECT** |

**Key invariant — frozen-gate strategy works:** Compose-training does
NOT disturb v14/v16's reciprocal binding. 16/16 A→W preserved across
both 100 and 400 event scales.

**Compositional binding emerges from Hebbian STDP** on direct
verb_pool → motor pathways. After training, driving the verb word
ALONE preferentially activates the trained motor pool (vs other 3
motor pools).

**This is the first demonstration of all three user-stated blockers:**
- Concepts (16 distinct pools, v14)
- Composition (verb-alone → motor pool, v16+compose-train)
- Diversity (4× over Tier 1)

**Production recipe (v16 + compose):**
```bash
# Step 1: Phase 1 training (v16 architecture)
python -m research.runners.concept_pool_demo --seed N \
    --n-train-events 200 --n-lang-input 2048 --n-per-pool 200 \
    --n-fs-per-pool 24 --weak-concept-dynamics --interleaved \
    --topographic-factor 3.0 --off-target-factor 0.3 \
    --enable-adjective --orthogonal-codes --sparsity 0.05 \
    --enable-direct-verb-to-motor \
    --save-bridge <v16.h5>

# Step 2: Compose-training
python -m research.runners.concept_compose_train \
    --load-bridge <v16.h5> --seed N \
    --compose-pairs "go:north,come:south,stop:west,look:east" \
    --n-events-per-pair 400 --orthogonal-codes --sparsity 0.05 \
    --save-bridge <v16_composed.h5> --out compose.json

# Step 3: Verify A->W still works
python -m research.runners.concept_speak_demo --seed N \
    --enable-adjective --orthogonal-codes --sparsity 0.05 \
    --load-bridge <v16_composed.h5> --out speak.json
```

**Open work:** scale compose-training (multi-seed) and refine to
push PASS rate above 2/4. Cross-seed variance + competing-pair
interference (later pairs may overwrite earlier ones' STDP) are
the main investigation targets.

### Multi-seed compose-training (seeds 42-46, 100 events/pair)

Result on existing v16 bridges (no Phase 1 retrain, just compose):

| Seed | Compose PASS | A→W (post-compose) |
|---|---|---|
| 42 | 2/4 | 16/16 |
| 43 | 0/4 | 16/16 |
| 44 | 1/4 | 16/16 |
| 45 | 0/4 | 16/16 |
| 46 | 2/4 | 16/16 |
| **Total** | **5/20 (25%)** | **80/80 (100%)** |

**Verdict: A→W PERFECTLY preserved across all 5 seeds** — the
frozen-gate strategy is robust. **Compose-binding is PARTIAL** —
signal emerges but seed-variable at 100 events. PASS rate 25%
multi-seed (vs 50% at seed 42 alone).

Hypothesis: at 100 events/pair, the v16 verb→motor weights grow
just enough to be visible at strong seeds (42, 46) but not enough
to overcome random structural bias at weaker seeds (43, 45). Need
either:
1. More events/pair (200, 400, 800)
2. Better temporal protocol (longer verb-only window, varied
   compose-pair interleaving)
3. Direct weight initialization (set verb→motor to a small prior)
   then refine via STDP.

**Status: compositional architecture validated multi-seed for
binding preservation; compositional signal requires further
iteration to robust multi-seed PASS.**

### Anti-cheat (permuted-mapping control, 5 seeds, 400 events/pair)

Compose-training trains TRUE mapping {go→north, come→south,
stop→west, look→east}. Anti-cheat tests all 4! = 24 permutations
of verb→motor mapping. A REAL learning result would have TRUE
mapping uniquely ranked 1/24 (highest PASS count).

| Seed | True rank | True PASS | Best perm PASS |
|---|---|---|---|
| 42 | 2/24 | 2/4 | 4/4 |
| 43 | 3/24 | 2/4 | 3/4 |
| **44** | **1/24** | **2/4** | **2/4 (TRUE IS UNIQUELY BEST)** |
| 45 | 17/24 | 0/4 | 2/4 |
| 46 | 19/24 | 0/4 | 2/4 |

**Honest verdict:** Mean true rank 8.4/24 vs chance 12.5/24. 3/5
seeds show above-chance signal (ranks ≤3), but only seed 44 is
UNIQUELY best. Seeds 45/46 actively learn WRONG associations
(rank 17-19 = anti-learning). This pattern matches the 2026-05-03
permuted-label control finding: weak signal that's seed-dependent
and not robust enough to claim composition is "working."

**Compositional binding multi-seed verdict: BOUNDARY (weak signal,
not robust). Some learning is happening (seed 44 unique best is
not chance) but the current architecture+training doesn't reliably
produce composition across seeds.**

**What would push above BOUNDARY:**
1. Much stronger compose-training (2000+ events, teacher current
   on motor during co-fire)
2. Pre-bind initialization (set verb→motor weight to non-zero
   prior so STDP refines rather than grows from 0)
3. Different test methodology (verb+motor co-drive → motor
   modulated, vs. verb-alone → motor)
4. Deeper architecture rework (e.g., proper PFC region with
   correct dynamics that doesn't break A→W)

The compose-training framework + anti-cheat tools are now in place
for future iteration. v14/v16 binding (concepts + diversity) is
unconditionally validated; composition is the open frontier.

### Root cause diagnosis (2026-05-13 late): Phase 1 word-level binding is the bottleneck

Direct probe of verb_pool firing during verb-alone inference reveals
the actual mechanism:

| Drive | verb_GO | verb_COME | verb_STOP | verb_LOOK | motor_N |
|---|---|---|---|---|---|
| "go" | **0.00** | 0.14 | 0.17 | 0.24 | 0.08 |
| "come" | 0.03 | **0.26** | 0.07 | 0.04 | 0.10 |
| "north" | 0.13 | 0.24 | 0.12 | 0.14 | **0.42** |
| "apple" | 0.23 | 0.20 | 0.20 | 0.22 | 0.11 |

**At seed 42, "go" drive activates verb_pool_GO at 0.00 rate** —
Phase 1 simply didn't bind "go" → verb_pool_GO at this seed. Yet
"come" → verb_pool_COME works (0.26 rate). "north" → motor_N works
(0.42 rate).

This explains the compose BOUNDARY pattern: compose-binding requires
Phase 1 to have bound BOTH the verb word AND the motor word reliably.
With Phase 1 W→A multi-seed mean 74% (5 robust + 9 mixed + 2 fragile
words), the pairwise (verb, motor) Phase 1 success rate is 0.74² = 55%
expected. Multi-seed compose result 5/20 = 25% is BELOW this — but
the strict "verb alone → motor uniquely fires" criterion also
requires the verb_pool's downstream motor pathway to dominate over
random structural bias, which is a HARDER test.

**Manual weight test** (set verb_pool → motor weights to 5.0, 30.0):
even huge weights don't fix the compose test, because verb_pool_GO
itself isn't firing during "go" drive at this seed. The bottleneck
is NOT the v16 pathway weight magnitude — it's Phase 1 binding.

**Strategic implication:** To improve compose multi-seed, first
improve Phase 1 W→A robustness (push mean from 11.8/16 → 14+/16).
Compose-binding currently inherits Phase 1's seed-dependent
per-word fragility.

**Pivot direction for future iteration:**
1. Push v14/v16 Phase 1 W→A robustness (more training, better
   topographic prior, etc.)
2. Cherry-pick compose pairs to words that DID Phase 1 bind well
   (anti-cheat against per-word bias)
3. Re-test compose only on the robust word subset

This is the honest architectural reality. Tonight's compose arc has
exhausted the v16-direct-pathway approach within the constraints of
Phase 1's existing word-level fragility.

### End-of-arc summary (2026-05-13 night)

**14 iterations across the day:**
- v1 (0/10) → v2-v6: cumulative dampening bug fixes
- v7 (multi-seed): topographic target-priority
- v9 (5/5 GO): reciprocal bias
- v11 (5/5 GO): adjective pools (16 total)
- v14 (5/5 GO): orthogonal codes — production for concepts+diversity
- v15a/b/c (NEGATIVE): dlpfc_verb region addition collapses A→W
- v16 (5/5 GO): direct verb→motor pathways — production for binding
- v16 + compose-training (BOUNDARY): compose works architecturally
  but binding is bottlenecked by Phase 1 per-word fragility

**Final status table:**

| User blocker | Status | Validation |
|---|---|---|
| Concepts | ✅ VALIDATED multi-seed | v14: 16-pool, 5/5 seeds |
| Diversity | ✅ VALIDATED | 4× over Tier 1 |
| Composition | ⚠️ BOUNDARY | Framework + anti-cheat tools in place; signal weak (mean TRUE rank 8.4/24 vs chance 12.5/24); bottlenecked by Phase 1 |

**Tools shipped:**
- `concept_pool_demo.py --enable-direct-verb-to-motor`: v16 Phase 1
- `concept_compose_train.py --motor-teacher-pA`: compose-training
- `v16_compose_permuted_check.py`: 24-permutation anti-cheat
- `v16_manual_compose.py`: manual weight installation for arch tests
- `v16_dlpfc_probe.py`: dlpfc_verb diagnostic (for v15 forensics)
- `concept_pool_aggregate.py`: multi-seed aggregator (extended)
- Multiple PowerShell multi-seed launchers

**Future work priorities (in order):**
1. Push Phase 1 W→A multi-seed mean from 11.8/16 → 14+/16. This
   directly improves compose-binding capacity since compose
   inherits Phase 1 per-word success.
2. Validate that v16+compose works architecturally on
   Phase-1-passing pairs (would need ≥4 such pairs per seed).
3. Consider alternative composition mechanisms not bottlenecked
   by single-direction Phase 1 binding (e.g., engram tags for
   compositional binding, catalog D.14).
4. Scale to 24/32 word vocabs with v14 recipe (no Phase 1
   architecture change needed).

### Direct-drive test: v16 compose pathway is essentially silent

Performed architectural validation by driving verb_pool directly
with 1500 pA (bypassing Phase 1's lang_input → verb_pool), then
measuring motor pool firing. This isolates the v16 verb_pool →
motor pathway from the Phase 1 binding bottleneck.

**Result (seed 42, v16+compose400):**

| Verb pool drive | motor_N | motor_E | motor_S | motor_W |
|---|---|---|---|---|
| verb_pool_GO | 0.003 | 0.004 | 0.003 | 0.005 |
| verb_pool_COME | 0.002 | 0.001 | 0.002 | 0.001 |
| verb_pool_STOP | 0.003 | 0.002 | 0.002 | 0.002 |
| verb_pool_LOOK | 0.004 | 0.004 | 0.004 | 0.003 |

- **0/4 compose pairs PASS** on direct-drive test (verb_pool drive
  doesn't preferentially activate trained motor pool)
- **TRUE mapping rank 15/24** — BELOW chance (12.5/24)
- Motor firing rates 0.001-0.005 (essentially zero — 100× lower
  than lang_input-driven rates of 0.30-0.45)

**Definitive finding:** The v16 verb_pool → motor pathway is
ESSENTIALLY SILENT even after compose-training. The previous
verb-alone-via-lang_input test results (2/4 PASS at ratios 1.1-1.6x)
were artifacts of lang_input → motor pathways (trained during
Phase 1) + structural noise, NOT real composition via the v16
pathway.

**The architecture fails to compose at the pathway level.** Possible
reasons:
- STDP weight growth from zero-init is insufficient (compose-training
  events don't drive enough LTP to produce functionally meaningful
  weight magnitudes)
- Motor pool requires multi-modal input (lang_input + verb context)
  to fire; verb context alone is insufficient
- The 200 lang_input neurons firing → 100 verb_pool neurons firing →
  via v16 pathway → 200 motor neurons effectively NEVER causes
  motor pool to threshold

**Revised verdict:**
- v16 + compose-training: **NEGATIVE on real composition**
  (BOUNDARY claim retracted further — direct-drive proves no
  pathway-level binding)
- v16 architecture as a SUBSTRATE for future compose mechanisms:
  still useful (pathways structurally exist, gates work)
- Future compose must use either:
  1. Pre-initialized non-zero weights (skip STDP cold start)
  2. Rate-based Hebbian instead of STDP timing-based
  3. Different connectivity (e.g., higher density, lower neuron count)
  4. Engram-tagging mechanism (catalog D.14) — bind sets of
     co-fired neurons as a unit rather than train pathway weights

## ⚠️ v17 28-word scaling LIMIT (2026-05-14 PM): structural imbalance, not training-event count

Tested two approaches to push the conversational capability beyond
v16's 16-word vocab:

| Variant | Phase 1 PASS | Multitag FULL | Multitag PARTIAL |
|---|---|---|---|
| v17 original (200 events) | 14/28 = 50% | 0/36 = 0% | 15/36 = 41.7% |
| v17 stronger (400 events) | 6/28 = 21% | 0/9 = 0% | 5/9 = 55.6% |

**400 events makes Phase 1 WORSE.** Diagnosis (from per-word ratios):
motor pools dominate. 4 motor pools × ~150 concentrated lang_input
weights vs 24 concept pools × 60 spread weights. More training shifts
balance further toward motors. 22/28 words have a MOTOR pool as the
top off-target winner.

Vocab scaling to 28 words requires **architectural rework**, not
longer training:
- Smaller motor pools, or
- 8192+ lang_input neurons, or
- Topographic prior favoring concept-pool selectivity, or
- Concept-only architecture (drop motor pools entirely)

For 16-word vocab, the validated multitag mechanism delivers
genuine 90% multi-seed conversational retrieval. 28-word is an
open frontier requiring deeper architectural design.

## 🎉🎉 Multi-tag cue retrieval: 90% FULL / 100% PARTIAL multi-seed (2026-05-14 PM)

The "real concept-concept conversation" capability the user wanted is
GENUINELY ACHIEVED. After the bug retraction and engram-stim-recall
finding, a simple aggregator mechanism delivers cue-driven retrieval
at multi-seed reliability:

**Mechanism:** For cue word X, stim every engram tag containing X
(each at 87.5% per-tag stim-recall reliability), aggregate
`lang_output` cosines, rank associates.

**Multi-seed result (5 seeds × 8 cues, 16-word vocab):**

| Seed | FULL (all associates in top-2) | PARTIAL (any in top-2) |
|---|---|---|
| 42 | 7/8 | 8/8 |
| 43 | 7/8 | 8/8 |
| 44 | 8/8 | 8/8 |
| 45 | 8/8 | 8/8 |
| 46 | 6/8 | 8/8 |
| **Total** | **36/40 = 90.0%** | **40/40 = 100%** |

Chance for FULL (top-2 of 15 covering 2 specific words): ~0.95%.
**Result is ~95× chance.**

**Demo (seed 44):**
```
> apple
  matched 2 tag(s): ['apple_big', 'apple_cat']
  top-5: [big=0.20, cat=0.17, stop=0.06, go=0.06, come=0.06]
  ✓ both big and cat in top-2

> dog
  matched 2 tag(s): ['dog_small', 'dog_river']
  top-5: [river=0.43, small=0.34, cat=0.06, big=0.06, look=0.05]
  ✓ both river and small in top-2

> big
  matched 2 tag(s): ['apple_big', 'big_hot']
  top-5: [apple=0.45, hot=0.40, stop=0.07, small=0.06, cat=0.06]
  ✓ both apple and hot in top-2
```

**Three-mode chat REPL** (`compose_concept_chat.py`):
- `<word>` → multitag (default, 90% FULL multi-seed) ✓ RECOMMENDED
- `/stim <tag>` → direct tag stim-recall (87.5% per-tag)
- `/cue <word>` → raw pool firing (~28%, experimental)

**See findings:**
- [`research/findings/2026-05-14-multitag-cue-retrieval-90pct-VALIDATED.md`](research/findings/2026-05-14-multitag-cue-retrieval-90pct-VALIDATED.md)
- [`research/findings/2026-05-14-engram-stim-recall-multi-seed-VALIDATED.md`](research/findings/2026-05-14-engram-stim-recall-multi-seed-VALIDATED.md)

## 🎉 Engram-tag stim-recall: 87.5% multi-seed (2026-05-14 PM, corrected re-test)

After the bug retraction (below), the same engram-tagging mechanism was
re-tested with corrected bridge architecture AND stronger encoding
settings (teacher current 500 pA + 500 encoding events). The validated
result:

**5-seed × 8-pair concept-concept stim-recall: 35/40 = 87.5%**

| Seed | Stim-recall (both A,B in lang_output top-5) | Assoc-recall (B in non-A top-3) |
|---|---|---|
| 42 | 7/8 | 2/8 |
| 43 | 6/8 | 3/8 |
| 44 | 8/8 | 2/8 |
| 45 | 8/8 | 3/8 |
| 46 | 6/8 | 1/8 |
| **Total** | **35/40 = 87.5%** | 11/40 = 27.5% |

Chance baseline for stim-recall (both in top-5 of 16): 8.3%. Result is
~10× chance.

**Recipe (validated):**
```bash
# 1. Train v16 bridge (standard 16-pool architecture)
python -m research.runners.concept_pool_demo --seed N \
    --n-train-events 200 --n-lang-input 2048 --n-per-pool 200 \
    --n-fs-per-pool 24 --weak-concept-dynamics --interleaved \
    --topographic-factor 3.0 --off-target-factor 0.3 \
    --enable-adjective --orthogonal-codes --sparsity 0.05 \
    --save-bridge bridges/v16/seed${N}.simstate.h5

# 2. Encode + test (per seed)
python -m research.runners.compose_concept_engram \
    --load-bridge bridges/v16/seed${N}.simstate.h5 --seed N \
    --n-lang-input 2048 --n-per-pool 200 --n-fs-per-pool 24 \
    --n-words-for-orthogonal 16 --encoding-steps 500 --sparsity 0.05 \
    --pairs "apple:big,dog:small,cat:hot,river:cold,go:look,come:stop,big:hot,small:cold" \
    --balanced-teacher-pA 500.0
```

**What works:**
- Tonegawa-style engram tagging (catalog D.14): bind a co-fired
  ensemble across two concept pools, stim it later, both concept
  pools reactivate. lang_output spelling for both words appears in
  top-5 with high reliability.

**What doesn't work (yet):**
- Cue-only associative recall (drive A alone, expect B in top-3):
  27.5% multi-seed, barely above chance (20%). Cross-pool plastic
  pathways (v18/v19) don't add measurable improvement.

**v19 (cross-pool pathways) verdict:** NEGATIVE. Side-by-side at
seed 42 with same strong encoding, v19 gives 6/8 stim vs v16's 7/8.
Cross-pool architecture adds complexity without lift.

**See finding:** [`research/findings/2026-05-14-engram-stim-recall-multi-seed-VALIDATED.md`](research/findings/2026-05-14-engram-stim-recall-multi-seed-VALIDATED.md)

## ⚠️ RETRACTION (2026-05-14): concept-concept results were architecture-mismatch artifacts

**Critical bug discovered 2026-05-14:** The 65% pool-firing readout and
90% transitive inference claims below were measurement artifacts caused
by a module-level monkey-patch in `compose_engram_demo_v2` that silently
corrupted bridge architecture during evaluation. With corrected
architecture matching, the strict top-1 collapses from claimed 25% to
0/8 (chance ~6%), and chain transitive collapses from claimed 90%
multi-seed to 1/4 on seed 42.

**See** [`research/findings/2026-05-14-CRITICAL-bug-compose-concept-architecture-mismatch.md`](research/findings/2026-05-14-CRITICAL-bug-compose-concept-architecture-mismatch.md)
for full diagnosis.

**What's still real (unaffected by bug):**
- Tier 1 (4-word direction): 6/6 BIDIR multi-seed (bio_three_factor)
- Tier 2.1 (8-word synonym): 6/6 BIDIR multi-seed (bio_three_factor)
- Synonym32 (32-word multi-language) chat_speak: 100% A→W seed 42
- Phase 1.3 hippocampus consolidation: 3/3 strict anti-cheat multi-seed
- P5 ventral semantic comprehension: 6/6 multi-seed (iter W)
- Phase 1.4 BRANCH A no-forgetting: 5/6 retention ≥ 80%
- Encoding-axis 64-word: 3/3 GO unanimous, 35× speedup

The motor-routed compose ladder (4-pair, 12-pair, 48-pair, 96-pair) also
uses `compose_engram_demo` (NOT `compose_concept_engram`) and is
unaffected by the bug.

**Section below preserved for historical context** (showing what was
claimed before the bug was discovered). All "65%" / "90%" / "30%"
numbers are RETRACTED.

---

## 🔥 ARCHITECTURAL PIVOT (2026-05-14): real semantic memory via concept-concept engrams [RETRACTED 2026-05-14 PM]

**User correctly identified** that all "compositional capacity" results
above (48/96/240 cross-pair compose) were **vocabulary-rich motor-direction
routing**: every engram terminated in N/E/S/W. That's not real conversation.

**Pivoted to concept-concept engrams (NO motor in tag):**
- Encode (apple, big): both concepts simultaneously drive lang_input,
  engram tag includes only concept-pool neurons (motor pools excluded
  from region_filter).
- Recall: stimulate tag → read lang_output → cosine-match to word
  spelling patterns. Output is a CONCEPT WORD, not motor direction.

**Multi-seed v16 result (5 seeds × 8 pairs) — RETRACTED:**

| Test | Pass count (CLAIMED) | Re-test (CORRECTED) |
|---|---|---|
| Stim-recall (both concepts in lang_output top-5) | 23/40 = 57.5% | not re-tested |
| Associative-recall (drive a alone, b in non-a top-3) | 12/40 = 30% | not re-tested |

vs chance (~30%/~20%). Both above chance — real semantic memory.
NOTE: These numbers measured with mismatched 16-pool vs 28-pool bridge
architecture. Not real signal — see retraction notice above.

**Concept chat REPL transcript (compose_concept_chat.py):**

```
> apple   → [cat=0.13, big=0.13, come=0.09]   (apple↔cat, apple↔big trained — both retrieved)
> big     → [hot=0.11, small=0.09, go=0.08]   (big↔hot top!)
> river   → [go=0.10, cold=0.09, hot=0.09]    (river↔cold in top-3)
> hot     → [stop=0.14, big=0.10, cold=0.09]  ("big" retrieved from big↔hot)
> small   → [dog=0.10, big=0.09, cat=0.07]    ("dog" retrieved from dog↔small)
```

**Limitations honest:**
- v16 30% associative-recall multi-seed is above chance but not robust
- Quality depends on Phase 1 binding strength of EACH concept
- v17 28-word vocab has weak Phase 1 (50%) so concept-concept doesn't
  work well there (1/8 stim, 2/8 assoc at seed 42)
- Strong Phase 1 (v16 81%) gives clean cosine readout for spelling

**Production tools (semantic memory):**
- `compose_concept_engram.py` — encode (concept, concept) engrams + test
  stim-recall and associative-recall via lang_output
- `compose_concept_chat.py` — interactive concept chat (user types
  concept, system associates)

**Path forward to robust conversation:**
1. Strengthen Phase 1 W→A (improves lang_output spelling weights)
2. Denser association graphs (each concept paired with multiple)
3. Confidence threshold + multi-association retrieval
4. CA3-style pattern completion (catalog D.13) for cleaner partial-cue recall

**Bottom line:** the user's three blockers (concepts/composition/diversity)
have BOTH motor-routed validation (96/96 PERFECT compose) AND now a
genuine concept-output validation (30% associative, 57% stim-recall).
The latter is the real conversational foundation.

## ⚠️ RETRACTED 2026-05-14 PM: Pool-firing readout 65% + TRANSITIVE 90%

The two breakthroughs reported below were BOTH measurement artifacts of
the same architecture-mismatch bug. With corrected bridge architecture:

| Metric | CLAIMED | CORRECTED (seed 42) |
|---|---|---|
| Pool-firing readout (compose_concept_strict) | 26/40 = 65% multi-seed | 0/8 top-1 = 0% |
| Transitive inference (compose_concept_chain_test) | 18/20 = 90% multi-seed | 1/4 chains = 25% |

Section preserved for historical context. See [`2026-05-14-CRITICAL-bug-compose-concept-architecture-mismatch.md`](research/findings/2026-05-14-CRITICAL-bug-compose-concept-architecture-mismatch.md).

## 🎉🎉🎉 Pool-firing readout + TRANSITIVE INFERENCE (2026-05-14 morning) [RETRACTED]

After initial concept-concept demo (30% assoc with lang_output cosine),
two breakthroughs improved semantic memory quality: [RETRACTED — bug artifacts]

**1. Pool-firing readout (compose_concept_pool_readout.py):** [RETRACTED]
Instead of cosine-matching lang_output spelling patterns, rank concept
pools by firing rate during recall. Top non-cue pool = associated
concept. Multi-seed v16 8-pair: **26/40 = 65% associative recall**
(up from 12/40 = 30% with cosine).
**Re-test with corrected architecture: 0/8 top-1 strict.**

**2. TRANSITIVE INFERENCE (compose_concept_chain_test.py):** [RETRACTED]
Train apple↔big AND big↔hot. Query apple. Top-3 non-apple includes
"hot" (the chained association via big). Multi-seed v16 4 chains:
**18/20 = 90% chained recall** (vs 17/20 = 85% direct).
**Re-test with corrected architecture: 1/4 chains on seed 42.**

Chained > direct! Distributed activation propagates through learned
association graph. apple → noun_APPLE (Phase 1) AND adj_BIG (cross-pool
STDP from apple_big encoding) AND adj_HOT (chained via big_hot
encoding's cross-pool STDP). Multiple pathways converge.
[RETRACTED — the multi-pathway convergence was firing-pattern coincidence
due to architecture mismatch, not learned cross-pool STDP.]

**Capability ladder for semantic conversation:** [RETRACTED]
- Direct association (apple → big): 65% multi-seed → CORRECTED 0%
- Transitive inference (apple → hot via big): 90% multi-seed → CORRECTED 25% seed 42
- Chat REPL (compose_concept_chat.py) operational with pool-firing
  readout, all 8 trained pairs retrieval visible in top-3.
  [Chat REPL infrastructure still operational; ranking results suspect.]

**This is genuine semantic conversation.** [RETRACTED] Not vocabulary-rich motor
routing — the system stores learned word-word associations, retrieves
them via pool firing, AND infers indirect connections through the
graph.
[RETRACTED — measurements were bug artifacts. Genuine semantic
conversation at the concept-concept level remains an open problem.]

## 🎉 v17 (2026-05-14): extended 28-word vocab + 96-pair compose PERFECT seed 42

Extended vocabulary from v16's 16 words to **28 words**:
- 8 nouns (added: tree, bird, sun, moon)
- 8 verbs (added: walk, run, eat, sleep)
- 8 adjectives (added: red, blue, fast, slow)
- 4 motors (unchanged: N/E/S/W)

Bridge size 14464 neurons (vs 7680 v16), 16M synapses, 2.5GB GPU.
Training: 28 words × 200 events = ~44 min/seed.

**Seed 42 results:**

| Test | Result |
|---|---|
| Phase 1 W→A | 14/28 = 50% (degraded from v16's 81%) |
| 4-pair engram (NEW verbs walk/run/eat/sleep) | 4/4 PERFECT |
| 24-cue engram (every cue → one motor) | 24/24 PERFECT |
| **96-pair cross (24 cues × 4 motors)** | **96/96 PERFECT** |

**Key finding:** engram-based composition is INDEPENDENT of Phase 1
binding quality. Even at 50% W→A, every (cue, motor) compose pair
recalls its target motor at 100%.

**Compositional capacity progression:**
- v16 (16-word): 48 cross-pairs PERFECT multi-seed
- v17 (28-word): 96 cross-pairs PERFECT seed 42 (2× v16)

**Multi-seed v17 in flight** (seeds 43-46, ~3.5 hours).

**Production tools (v17):**
- `concept_pool_demo_v2.py` — 28-word bridge training
- `compose_engram_demo_v2.py` — extended-vocab compose
- `compose_chat_repl_v2.py` — chat REPL with 24-cue + motor vocab
- `compose_5word_engram.py` — 5-word phrase test
- `run_v17_multiseed.ps1` — multi-seed launcher

## 🎉🎉🎉 ENGRAM-BASED COMPOSITION (catalog D.14): 5-seed VALIDATED

After the STDP-pathway compose approaches went NEGATIVE, switched to
the biology-grounded Tonegawa engram-tagging mechanism (catalog D.14,
already shipped as a bridge API). This BYPASSES pathway-weight
learning entirely.

**Mechanism (`research/runners/compose_engram_demo.py`):**
1. ENCODING: For each (verb, motor) pair, drive lang_input(verb) +
   lang_input(motor) simultaneously for ~100ms with
   `bridge.start_engram_recording()`. Commits a top-K=100 engram tag
   spanning verb_pool + motor regions via `commit_engram_tag()`.
2. RECALL: `bridge.stimulate_tag()` drives the tagged neurons at
   1500 pA. The tag spans verb_pool + motor co-firing neurons —
   stimulation reactivates the original compositional ensemble.
3. ANTI-CHEAT: 24 permutations of verb→motor mapping.

**5-seed result:**

| Seed | PASS | TRUE rank |
|---|---|---|
| 42 | 3/4 | 1/24 |
| 43 | **4/4** | 1/24 |
| 44 | 3/4 | 1/24 |
| 45 | **4/4** | 1/24 |
| 46 | 2/4 | 1/24 |
| **Total** | **16/20 (80%)** | **1/24 UNANIMOUS** |

vs chance: PASS 25%, TRUE rank 12.5/24
vs v16 STDP-pathway (NEGATIVE): 5/20 PASS, TRUE rank 8.4/24

**Why engram succeeds where STDP failed:**
- STDP grows synaptic weights over training events; doesn't reach
  functional magnitude before training ends. The pathway is
  essentially silent (direct-drive test: 0/4 PASS).
- Engram tags STORE neuron indices directly in one encoding pass.
  No weight growth needed; recall stimulates the bound ensemble.

**Honest caveats:**
- TRUE rank 1/24 has TIES at the lower-PASS seeds (seed 46 has
  ~5 permutations at 2/4). Strict unique-best is 2/5 seeds
  (43, 45 at 4/4 max). But TRUE is consistently AMONG top
  permutations at all seeds — 16/20 PASS = 3.2× chance.

**Production recipe (engram-composition):**
```bash
# Step 1: Phase 1 v14/v16 training (any concept_pool_demo bridge works)
python -m research.runners.concept_pool_demo --seed N \
    --n-train-events 200 --n-lang-input 2048 --n-per-pool 200 \
    --n-fs-per-pool 24 --weak-concept-dynamics --interleaved \
    --topographic-factor 3.0 --off-target-factor 0.3 \
    --enable-adjective --orthogonal-codes --sparsity 0.05 \
    --save-bridge <bridge.h5>

# Step 2: Engram-encode (verb, motor) pairs + test composition
python -m research.runners.compose_engram_demo \
    --load-bridge <bridge.h5> --seed N \
    --compose-pairs "go:north,come:south,stop:west,look:east" \
    --encoding-steps 200 --top-k 100 \
    --recall-stim-pA 1500 --recall-steps 100 \
    --save-bridge <bridge_with_engrams.h5> --out engram.json
```

Wall clock: ~17 min/seed Phase 1 + ~30s/seed engram encoding+recall.
Total full demo: ~18 min/seed.

**Status: ALL THREE user-stated blockers VALIDATED multi-seed:**
- Concepts: ✅ VALIDATED (v14, 5/5 GO)
- Diversity: ✅ VALIDATED (4× over Tier 1)
- Composition: ✅ VALIDATED (engram, 16/20 PASS, TRUE rank 1/24 unanimous)

The architectural arc that started with 14 iterations of STDP-based
attempts finally succeeded by pivoting to a completely different
biology-grounded mechanism. Catalog D.14 paid off.

## 🎉 Composition refinement (2026-05-14): PERFECT recall + sequential + chat REPL

After the initial engram breakthrough, four iterative improvements
pushed composition from 80% → 100% multi-seed:

**1. Motor teacher during encoding (`--motor-teacher-pA 1500`):**
Injects teacher current on target motor pool during encoding window.
Ensures the engram tag contains enough motor neurons for clean recall.
Result: 5-seed **20/20 PERFECT** (vs 16/20 = 80% without).

**2. Cosine-match retrieval (`compose_engram_retrieval.py`):**
Stores firing pattern (per-neuron spike count) during encoding alongside
the tag. At query time, drive lang_input(cue) → measure firing pattern →
cosine-match to stored patterns → identify best engram. Enables
automatic retrieval (no need to specify tag by name).
5-seed: 17/20 (85%) cosine-match accuracy.

**3. Full pipeline (`compose_full_pipeline.py`):**
End-to-end: query → cosine-match → stimulate matched engram → motor.
Phase separation critical: ALL queries first, THEN all recalls (otherwise
recall stim perturbs subsequent query firing patterns).
5-seed: 17/20 cosine + 16/20 motor = effective chat-usable accuracy 80%.

**4. Chat REPL (`compose_chat_repl.py`):**
User-facing demo. Type "go north" → system identifies engram → outputs
NORTH action. 4-pair: 4/4 trained pairs PASS at seed 42 with motor
teacher. 12-pair (verb+noun+adj→motor): 5-seed motor recall
**55/60 = 91.7%** (3/5 seeds achieve perfect 12/12).

**5. Sequential composition (`compose_sequential_engram.py`):**
Real conversation has temporal sequences. Encoding drives lang_input(verb)
for verb_steps, gap, then lang_input(motor) for motor_steps. Engram
captures spikes across all 3 windows. 5-seed:
- Retrieval: 16/20 (80%) via cosine on sequential drive
- Recall: **20/20 (100% PERFECT)** via engram stimulation

**Practical chat capability achieved:**
- 12-word vocabulary multi-seed at 92% motor accuracy
- Sequential word input handled (real conversation pattern)
- Cosine retrieval for automatic engram identification
- Motor teacher ensures perfect engram quality
- v14/v16 substrate preserved (A→W 100% unanimous across all tests)

**Production tools (`research/runners/compose_*.py`):**
- `compose_engram_demo.py` — encode engrams + recall (multi-seed VALIDATED)
- `compose_engram_retrieval.py` — cosine-match retrieval
- `compose_full_pipeline.py` — end-to-end pipeline with phase separation
- `compose_sequential_engram.py` — temporal sequence encoding
- `compose_chat_repl.py` — user-facing interactive REPL
- `v16_compose_permuted_check.py` — anti-cheat (24 permutations)

Sequential composition still open (v12 NEGATIVE bidirectional dlpfc;
v13 PARTIAL per-kind NMDA: +3x persistence but -5x isolation). Real
architectural tension between holding (NMDA bistability) and selection
(clean isolation); biology solves this with PFC as separate region.

Per-word robustness from v7 multi-seed:
- 80% robust (4/5 seeds): north, east, cat
- 60% mixed (3/5): apple, river, look
- 40% fragile (2/5): south, west, dog, go, come
- 20% fragile (1/5): stop

v7 weight probe shows consistent 4x weight ratio across seeds —
variability is dynamics, not weights. v9 architecture demonstrates
concepts + diversity + bidirectional binding per user mandate.
3-word robust trio (north, east, cat) demonstrates the recipe CAN
reliably bind specific pairs across seeds.

Phase 3 A→W readout currently 0/12 on v7 bridge (separate issue:
concept_to_language_output_weight=0.5 vs motor's 2.0 = 4x weaker
readout). Fix queued for v8 batch.

Trajectory: v1 0/10 → v7 6.5/12. Each iteration identified and fixed
a specific bug:
- v2: FS topology asymmetry (verb count)
- v2.x: target-only STDP gating
- v3: weak dynamics fixes canon bias amplification
- v4: stronger topographic prior
- v6: cross-kind topographic dampening
- v7: target-priority assignment (no cumulative dampening)

Findings: `research/findings/2026-05-13-concept-pool-architecture-Phase1.md`

Diagnosis + fix: `docs/plans/2026-05-13-concept-pool-FS-design-note.md`
Findings: `research/findings/2026-05-13-concept-pool-architecture-Phase1.md`

## 🎉🎉🎉 160-concept sparse-distributed G.20 ensemble — end-to-end SHIPPED (2026-05-15)

The catalog G.20 (Pulvermüller distributed cortical word ensembles)
shared-pool architecture, in its **true Kanerva-SDM sparse form**
(each concept = a scattered K-of-N random pattern, K=100 in a
2000-neuron pool, NOT a contiguous slice), is the vocab-scaling unlock.
Per-bridge: **64 concepts @ 100% discrimination, multi-seed 288/288**
(see `2026-05-15-sparse-distributed-capacity-curve.md`). The 256-concept
single-bridge run is **training-bound, not prior-bound** (after the
GPU-vectorized topographic-prior fix, prior is 0.3s; training is ~6 hr)
→ **multi-bridge is the production scaling route, linear in bridge
count** (`2026-05-15-256-concept-training-bound-conclusion.md`).

**SHIPPED + validated end-to-end 2026-05-15:** 5 sparse bridges
(A nouns / B verbs / C adj / D spatial / E functional) × 32 concepts
= **160 unique concepts, every bridge 100%**, loaded through a new
`g20_multibridge --sparse` mode. Seed-42 scripted demo PASSED with
zero failures:

- Cross-bridge associative memory: `remember apple is big`
  (apple∈nouns, big∈adj) → querying `apple` returns **big at rate
  662**, decisively above the ~400 noise floor, via
  `bridgeC_adj/apple_big`. This exercises the new sparse recall +
  cross-bridge sparse engram capture + deterministic pattern regen.
- Exact cross-bridge tag match (`is apple big?` → YES).
- N-word sentence spanning 3 bridges (`remember dog run fast`).
- Tag-name role queries (`who run fast?` → dog; `what did dog run?`
  → fast) — the v16-validated 100% multi-seed mechanism,
  architecture-independent.

```bash
# Train 5 sparse bridges (chain waits for GPU, ~17 min/bridge):
pwsh research/runners/g20_sparse_5bridge_chain.ps1
# End-to-end ensemble demo (loads all 5, scripted exercise):
pwsh research/runners/g20_sparse_ensemble_demo.ps1
# Or directly: python -m research.runners.g20_multibridge --sparse \
#   --pattern-size 100 --n-shared-pool 2000 --n-lang-input 8192 \
#   --sparsity 0.02 --seed 42 --bridges <5 *.simstate.h5> \
#   --vocab-files <5 vocab.txt> --names bridgeA_nouns ... --scripted "..."
```

`--sparse` builds via `build_sparse_pool_bridge`, regenerates per-bridge
patterns from `--seed` (verified byte-identical to training; 16 CPU
tests pin this — a drift would silently read wrong neurons), and routes
recall/encode through sparse analogues in `shared_pool_chat.py`. The
sparse-vs-contiguous branch is centralized in `SharedPoolMember`
methods so the sentence/tokenizer/hierarchy dispatch is reused
unchanged (contiguous path preserved; 96 multibridge tests still green).

**Honest scope:** per-bridge 100% is multi-seed; the *ensemble
integration* (cross-bridge + sentences through `--sparse`) is seed-42.
Findings:
`research/findings/2026-05-15-G20-sparse-ensemble-160concept-end-to-end-SHIPPED.md`.

**320-concept production tier — SHIPPED 2026-05-16 (98.4% per-bridge):**
5 bridges × 64 sparse concepts = 320 (the documented "age-5" target).
Per-bridge **98.4% (315/320)** — honest: NOT 100% like the 32-tier.
One **deterministic, characterized** gap: every bridge fails at concept
index 12 (rank 18), because all 5 train with `--seed 42` → identical
`generate_sparse_patterns(64,2000,100,42)` set; pattern-12 fails
identically (vocab-independent; NOT raw overlap — idx 8/17 overlap more
yet pass). Ensemble integration validated end-to-end seed 42 incl. the
**+160 extension vocab** (querying new word `horse` retrieves sentence
co-members `run` 882 + `fast` 508 cross-bridge across 3 bridges).
`g20_vocab_spec_320.py` (frozen-160 base + curated +160, global-
uniqueness assert) + `g20_sparse_5bridge_chain_320.ps1` (**sparsity
0.007** required: orthogonal-drive needs n_active 57 < stride
8192/64=128; the 160 chain's 0.02 → 164 would crash every bridge).
Cheap recovery path (deferred): per-bridge distinct seeds (42–46) /
overlap-rejection in `generate_sparse_patterns`. Findings:
`research/findings/2026-05-16-G20-sparse-ensemble-320concept-SHIPPED.md`.

**320-concept BIOLOGICAL COMPOSITION — RESOLVES multi-seed 2026-06-02 (honest flat-distinct path):**
Distinct from the retrieval tier above (multitag/ranking): this is the brain-analogue spiking COMPOSITION
(VSA coincidence bind/unbind) at full 320-concept scale. The hierarchical-320 shortcut (shared seed-42 codes
+ a 2nd bridge-role binding level) catastrophically FAILED on STRUCTURED facts (0.000 at seed 42 — the
nesting/multi-hop SNR wall) and was retracted. The fix is exactly the "per-bridge distinct seeds (42–46)"
recovery path noted above: 5 bridges at seeds 42–46 → 320 DISTINCT flat codes (between-cos mean 0.045, max
0.604) composed at a SINGLE binding level. Structured SVO (noun/verb/adj, cleanup over all 320) =
**1.000/1.000/1.000** (seeds 42/43/44); the harder ANY-BANK (any concept, any role) = **0.992 mean 6-seed**
(42–47, min 0.950, single miss localised to the spatial bridge, 119/120 facts); conversational demo 6/6 +
absent-cue ABSTAINS (anti-artifact). Scope: codes GIVEN by sparse encoding (cheating-audit); the composition
on top is GENUINE + robust at 320. Per-bridge retrain ~73–75 min (64c×400ev×8192lang — the docs' "~17 min"
was wrong for the 64-concept tier; NOT fragmentation). Tools: `research/findings/raw/_run_flatdist_DE.sh`,
`_insubstrate_flatdistinct320_test.py`, `_insubstrate_flatdist320_anybank_test.py`,
`research/runners/compose_flatdist320_conversation_demo.py`. Incremental/resumable training
(`concept_pool_sparse_distributed --resume-from`, shipped same day) lets such retrains chunk across breaks.
Finding: `research/findings/2026-06-02-full-320-flat-distinct-composition-RESOLVES-multiseed.md`.

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

**🎉 FULL VOCAB TIER LADDER CHAT_SPEAK VALIDATED (2026-05-12/13):**

| Mode | Vocab | n_motor | W→A | A→W | Verdict |
|---|---|---|---|---|---|
| Tier 1 | 4w | 500 | 74-98% | 58% | GO multi-seed |
| Synonym (Tier 2.1) | 8w | 1000 | 31-56% | 85% | GO 6-seed |
| Synonym12 | 12w | 2000 | 56% | 100% | GO seed 42 |
| Synonym16 | 16w | 2000 | 56% | 100% | GO seed 42 |
| Synonym24 | 24w | 2000 | 56% | 100% | GO seed 42 (multi-lang ES+DE) |
| **Synonym32** | **32w** | **3000** | **44%** | **100%** | **GO seed 42 (multi-lang ES+DE+JP+AR)** |

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
