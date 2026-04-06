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

# Run determinism tests
pytest tests/test_determinism.py -v
```

## Architecture

### Single-File Design
The entire simulator is contained in `neural-simulator.py` (~12,000 lines). This is intentional for easy distribution. Code is organized into clear sections with comment blocks.

### Thread Model
- **Main Thread**: DearPyGUI event loop + OpenGL rendering
- **Simulation Thread**: GPU-accelerated neural dynamics computation (fully isolated)
- **Communication**: Lock-free queues (`ui_to_sim_queue`, `sim_to_ui_queue`) for inter-thread messaging

### Key Classes

**SimulationBridge** (line ~2115): Central simulation orchestrator
- Manages all GPU state arrays (CuPy)
- Simulation stepping and dynamics updates
- Recording/playback to HDF5
- Checkpoint save/restore
- Profiling and performance monitoring

**Configuration Dataclasses**:
- `CoreSimConfig` (~line 485): Network topology, neuron models, plasticity, biological realism
  - STP fields: `stp_U`, `stp_tau_d`, `stp_tau_f` (global defaults)
  - Per-connection-type STP: `enable_per_type_stp`, `stp_U_per_type[4]`, `stp_tau_d_per_type[4]`, `stp_tau_f_per_type[4]`
  - Structural plasticity: `struct_plast_activity_bias` (0.0–1.0) for activity-dependent synaptogenesis
  - Homeostasis: EMA alpha (~0.0002, tau ~5s) and threshold adapt rate (~0.0005)
  - Inhibitory reversal: `E_inh = -75mV`, propagation scaled 0.7x for driving force compensation
  - HH numerical stability: dt auto-adjusts to 0.05ms when HH model selected
- `VisualizationConfig` (~line 695): OpenGL rendering and camera parameters
- `RuntimeState` (~line 715): Mutable execution state (running, paused, time tracking)
- `GPUConfig` (~line 729): GPU features, memory management, recording modes

### GPU Array Naming Conventions
- `cp_*`: CuPy GPU arrays (e.g., `cp_membrane_potential_v`, `cp_firing_states`)
- `gl_*`: OpenGL handles/VBOs
- `fused_*`: GPU kernel functions decorated with `@cp.fuse()`

### Simulation Step Pipeline (in `_run_one_simulation_step()`)
1. STP (Short-Term Plasticity) update – per-connection-type if enabled
2. Synaptic conductance update – uses E_inh = -75mV with 0.7x propagation scaling
3. Background noise (OU process)
4. Neuron dynamics (model-specific: Izhikevich/HH/AdEx)
5. Plasticity updates (Hebbian, STDP, reward modulation, structural with activity bias, homeostasis)
6. Visualization updates
7. Recording (if active)

**Note on dt Auto-Adjustment**: When switching to Hodgkin–Huxley model, dt is automatically
reduced to 0.05ms for numerical stability of voltage-gated kinetics. When switching to Izhikevich
or AdEx, dt restores to 0.5ms. This occurs in `apply_simulation_configuration_core()`.

### Fused CUDA Kernels (~lines 1813-2101)
Located in the main file, these are performance-critical GPU operations:
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

### Hodgkin-Huxley Presets (~lines 300-490)
Region-specific HH parameter dictionaries in `DefaultHodgkinHuxleyParams`, all derived from
`REALISTIC_L5_PYRAMIDAL_RS_37C` base with region-appropriate overrides:
- `HH_L5_CORTICAL_PYRAMIDAL_RS`: L5 pyramidal regular spiking (base preset)
- `HH_L23_CORTICAL_PYRAMIDAL`: L2/3 pyramidal (lower g_Na, higher g_M)
- `HH_CORTICAL_FS_INTERNEURON`: Fast-spiking interneuron (high g_Na/g_K, no adaptation)
- `HH_HIPPOCAMPAL_CA1_PYRAMIDAL`: CA1 pyramidal (prominent Ih, moderate CaT)
- `HH_HIPPOCAMPAL_CA3_PYRAMIDAL`: CA3 pyramidal (high CaT for bursting)
- `HH_THALAMOCORTICAL_RELAY`: TC relay (strong CaT rebound, strong Ih)
- `HH_TRN_RETICULAR`: TRN (very high CaT for oscillatory bursting)
- `HH_STRIATAL_MSN`: Medium spiny neuron (low g_Na, very negative E_L)
- `HH_STN_NEURON`: Subthalamic nucleus (autonomous pacemaker, strong NaP)
- `HH_CEREBELLAR_PURKINJE`: Purkinje cell (high g_Na, strong CaT for complex spikes)
- `HH_CEREBELLAR_GRANULE`: Granule cell (compact, low capacitance, minimal Ca²⁺)
- `HH_SPINAL_MOTOR`: Motor neuron (high C_m, strong NaP for plateau potentials)
- `HH_SPINAL_INTERNEURON`: Spinal interneuron (moderate channels, no NaP)

### Neural Structure Profiles (~lines 1463-1641)
Brain region presets that configure trait definitions, connectivity, and default parameters:
- GENERIC_UNSTRUCTURED
- CORTEX_L23_RS_FS, CORTEX_L4_INPUT_LAYER
- HIPPOCAMPUS_CA1_RS_FS, HIPPOCAMPUS_CA3_RECURRENT
- BASAL_GANGLIA_STRIATUM, BASAL_GANGLIA_STN_GPE
- THALAMUS_TC_TRN
- CEREBELLAR_CORTEX_SIMPLE, SPINAL_CORD_SEGMENT

### JSON Profile Dropdown System (~lines 8863-8956)
Full simulation profiles saved as `.json` in `simulation_profiles/`. A UI dropdown auto-populates from this directory, allowing one-click loading of complete parameter sets. Key functions:
- `_scan_profile_directory()`: Scans for `.json` files, builds display name map
- `_handle_full_profile_dropdown_change()`: Loads selected profile into UI
- `_refresh_full_profile_dropdown()`: Rescans directory and updates dropdown

### UI-Config Roundtrip
Two critical functions must be kept in sync for profile save/load to work correctly:
- `_update_sim_config_from_ui()`: Extracts all parameter values from UI widgets and builds `CoreSimConfig`, `VisualizationConfig`, `RuntimeState`, and `GPUConfig` dataclasses
- `_populate_ui_from_config_dict()`: Takes a configuration dictionary and updates all UI widgets to reflect those values

These are inverse operations: any parameter exposed in the UI must have a corresponding getter and setter to ensure bidirectional sync between UI state and simulation configuration.

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
