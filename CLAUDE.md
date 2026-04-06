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
The entire simulator is contained in `neural-simulator.py` (~9000 lines). This is intentional for easy distribution. Code is organized into clear sections with comment blocks.

### Thread Model
- **Main Thread**: DearPyGUI event loop + OpenGL rendering
- **Simulation Thread**: GPU-accelerated neural dynamics computation (fully isolated)
- **Communication**: Lock-free queues (`ui_to_sim_queue`, `sim_to_ui_queue`) for inter-thread messaging

### Key Classes

**SimulationBridge** (line ~1633): Central simulation orchestrator
- Manages all GPU state arrays (CuPy)
- Simulation stepping and dynamics updates
- Recording/playback to HDF5
- Checkpoint save/restore
- Profiling and performance monitoring

**Configuration Dataclasses**:
- `CoreSimConfig` (~line 415): Network topology, neuron models, plasticity, biological realism
- `VisualizationConfig` (~line 557): OpenGL rendering and camera parameters
- `RuntimeState` (~line 577): Mutable execution state (running, paused, time tracking)
- `GPUConfig` (~line 591): GPU features, memory management, recording modes

### GPU Array Naming Conventions
- `cp_*`: CuPy GPU arrays (e.g., `cp_membrane_potential_v`, `cp_firing_states`)
- `gl_*`: OpenGL handles/VBOs
- `fused_*`: GPU kernel functions decorated with `@cp.fuse()`

### Simulation Step Pipeline (in `_run_one_simulation_step()`)
1. STP (Short-Term Plasticity) update
2. Synaptic conductance update
3. Background noise (OU process)
4. Neuron dynamics (model-specific: Izhikevich/HH/AdEx)
5. Plasticity updates (Hebbian, STDP, reward modulation, structural, homeostasis)
6. Visualization updates
7. Recording (if active)

### Fused CUDA Kernels (~lines 1374-1630)
Located in the main file, these are performance-critical GPU operations:
- `fused_izhikevich2007_dynamics_update()`: 9-parameter Izhikevich model
- `fused_hodgkin_huxley_dynamics_update()`: Temperature-dependent HH
- `fused_adex_dynamics_update()`: Adaptive Exponential IF
- `fused_hh_m_current_update()`, `fused_hh_CaT_current_update()`, etc.: Extended HH currents
- `fused_conductance_decay_and_current()`: Synaptic dynamics
- `fused_stdp_weight_update()`: Spike-timing dependent plasticity

### Neural Structure Profiles (~lines 1023-1200)
Brain region presets that configure trait definitions, connectivity, and default parameters:
- CORTEX_L23_RS_FS, CORTEX_L4_INPUT_LAYER
- HIPPOCAMPUS_CA1_RS_FS, HIPPOCAMPUS_CA3_RECURRENT
- BASAL_GANGLIA_STRIATUM, BASAL_GANGLIA_STN_GPE
- THALAMUS_TC_TRN
- CEREBELLAR_CORTEX_SIMPLE, SPINAL_CORD_SEGMENT

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
