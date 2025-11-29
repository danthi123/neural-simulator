# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

A high-performance GPU-accelerated 3D neural network simulator using NVIDIA CUDA/CuPy for massively parallel computation. Simulates large-scale networks (10K-100K+ neurons) with biologically-inspired models, real-time OpenGL visualization, and comprehensive recording/playback capabilities.

**Key Technologies**: CuPy (CUDA), PyOpenGL, DearPyGUI, NumPy, H5PY

## Architecture

### Single-File Design
The entire simulator is in `neural-simulator.py` (~9500 lines). This monolithic structure keeps all GPU kernels, UI, and simulation logic together for easier CUDA optimization and deployment.

### Thread Model
- **Main Thread**: DearPyGUI event loop + OpenGL rendering (60 FPS target)
- **Simulation Thread**: GPU-accelerated neural dynamics computation
- **Communication**: Lock-free queues (`ui_to_sim_queue`, `sim_to_ui_queue`)
- **Shutdown**: Global `threading.Event` flag (`shutdown_flag`)

### GPU Memory Architecture
- **CuPy Arrays**: All neuron states, connectivity, and dynamics on GPU
- **CSR Sparse Matrices**: Synaptic connectivity (`cupy.sparse.csr_matrix`)
- **Memory Pool**: Configured to use 80% of available VRAM
- **Zero-Copy**: CUDA-OpenGL interop for visualization (when available)
- **GPU Buffering**: Entire recordings can be held in VRAM for instant playback

### Key Data Structures

#### Neuron State Arrays (prefix: `cp_` for CuPy)
- `cp_membrane_potential_v`: Membrane voltage (mV)
- `cp_recovery_variable_u`: Recovery/adaptation (Izhikevich)
- `cp_gating_variable_m/h/n`: Ion channel gates (Hodgkin-Huxley)
- `cp_conductance_g_e/g_i`: Excitatory/inhibitory conductances
- `cp_firing_states`: Boolean spike flags
- `cp_traits`: Integer neuron population IDs
- `cp_neuron_type_ids`: Integer type IDs (GPU-optimized filtering)
- `cp_neuron_positions_3d`: 3D spatial coordinates (N x 3)

#### Connectivity
- `cp_connections`: CSR matrix (weights)
- `cp_stp_u/x`: Short-term plasticity variables (Tsodyks-Markram)

#### Configuration Classes
- `CoreSimConfig`: Simulation parameters (dt, neurons, plasticity)
- `VisualizationConfig`: Rendering settings (camera, filters)
- `RuntimeState`: Dynamic state (time, running, paused)

## Development Commands

### Run Simulator
```bash
python neural-simulator.py
```

### Install Dependencies
```bash
# Install CuPy for your CUDA version (critical!)
pip install cupy-cuda12x  # For CUDA 12.x
pip install cupy-cuda11x  # For CUDA 11.x

# Install other dependencies
pip install -r requirements.txt
```

### Verify CUDA Setup
```bash
python -c "import cupy; print(f'GPU: {cupy.cuda.runtime.getDeviceProperties(0)[\"name\"]}')"
python -c "import cupy; print(f'Devices: {cupy.cuda.runtime.getDeviceCount()}')"
```

### Check GPU Memory
```bash
nvidia-smi
# Or within Python:
python -c "import cupy as cp; free, total = cp.cuda.Device().mem_info; print(f'{free/1e9:.1f}GB free / {total/1e9:.1f}GB total')"
```

## Network Size Guidelines

### Memory Usage Estimates
| Neurons | Connections/N | VRAM Usage | Expected FPS | Notes |
|---------|---------------|------------|--------------|-------|
| 1K | 100 | ~500 MB | 60+ FPS | Interactive prototyping |
| 10K | 500 | ~2 GB | 60 FPS | Recommended default |
| 50K | 1000 | ~8 GB | 30-60 FPS | High-performance GPU |
| 100K | 1000 | ~20 GB | 15-30 FPS | Research/enthusiast |
| 200K+ | 1000+ | 40+ GB | <15 FPS | Multi-GPU future work |

### With 24GB VRAM (Your Setup)
**Maximum Practical Configuration:**
- **~120K neurons** with 1000 connections/neuron
- **~80-100K neurons** with 1500 connections/neuron (leaves headroom for recording)
- **Frame size**: ~5-10MB per recording frame
- **GPU buffer capacity**: ~1500-2000 frames (90-120s at 60 FPS recording)

**Memory Breakdown (100K neurons, 1000 conn/neuron):**
- Neuron states (V, u, g_e, g_i, etc.): ~100K * 20 arrays * 4 bytes = ~80 MB
- Connectivity matrix (sparse): ~100M synapses * 8 bytes = ~800 MB
- STP/Hebbian plasticity: ~100M * 8 bytes = ~800 MB
- Visualization buffers: ~100K * 16 bytes = ~2 MB
- Recording buffer (optional): ~10 MB/frame * N frames
- **Total baseline**: ~2-3 GB, leaving 20+ GB for recording or larger networks

## Key Parameters

### Performance Tuning
- `dt_ms`: Time step (0.025-1.0 ms)
  - Izhikevich: 1.0 ms typical
  - Hodgkin-Huxley: 0.025-0.05 ms (needs smaller dt)
- `viz_update_interval_steps`: Steps between visual updates (17 for ~60fps)
- `MAX_NEURONS_TO_RENDER`: Cap on rendered neurons (default 1M)

### Plasticity (Major VRAM Impact)
- `enable_hebbian_learning`: LTP/LTD (doubles weight array memory)
- `enable_short_term_plasticity`: Tsodyks-Markram (adds U, X arrays)
- Disable both to maximize network size with limited VRAM

### Network Topology
- `enable_watts_strogatz`: Small-world network with high clustering + short paths
  - `connectivity_k`: Number of nearest spatial neighbors (must be even)
  - `connectivity_p_rewire`: Rewiring probability (0=regular, 1=random)
- `connections_per_neuron`: Average outgoing synapses (for spatial connectivity)
- `num_traits`: Neuron population types (affects trait-based connectivity)

## File Formats

### Profiles (.json)
- Human-readable JSON
- Stores all `CoreSimConfig`, `VisualizationConfig` parameters
- Location: `simulation_profiles/`
- **Does NOT store**: Runtime state, network connectivity

### Checkpoints (.simstate.h5)
- HDF5 format with full simulation state
- Includes connectivity, all neuron states, plasticity variables
- Compressed (gzip level 4)
- Location: `simulation_checkpoints_h5/`
- Can restore exact simulation state

### Recordings (.simrec.h5)
- HDF5 format with frame-by-frame data
- **GPU-buffered mode**: All frames stored in VRAM during recording, then batch-written to disk
- Groups: `initial_state`, `metadata`, `frames/frame_NNNNNN`
- Location: `simulation_recordings_h5/`
- Enables scrubbing through recorded simulations

## Common Workflows

### Adding New Neuron Model
1. Add enum to `NeuronType` (line ~66-79)
2. Define parameters in `DefaultIzhikevichParamsManager` or `DefaultHodgkinHuxleyParams` (line ~82-150)
3. Update `NeuronTypeIDMapper` to include new type (line ~152-200)
4. Implement dynamics in `_run_one_simulation_step()` (search for dynamics kernel)
5. Add initialization logic in `_initialize_simulation_data()` (line ~936+)

### Debugging GPU Memory Issues
1. Check available memory: `self._get_gpu_memory_info()` (line ~904)
2. Force garbage collection: `cp.get_default_memory_pool().free_all_blocks()`
3. Reduce network size or disable plasticity
4. Monitor with: `nvidia-smi dmon -s mu`

### Optimizing Visualization Performance
1. Reduce `MAX_NEURONS_TO_RENDER` in OpenGL config
2. Increase `viz_update_interval_steps` (reduce GPU→CPU transfers)
3. Disable synaptic pulses: `ENABLE_SYNAPTIC_PULSES = False`
4. Use "Show Only Spiking" filter mode (GPU-side filtering)
5. Disable synapse rendering for very large networks

### Recording Long Simulations
1. Estimate frame size: `self._estimate_frame_size_bytes()` (line ~1434)
2. Check capacity: `self._check_gpu_recording_capacity(estimated_frames)` (line ~1466)
3. GPU buffer holds frames until finalized, then batch-writes to disk
4. For >24GB recordings, disable GPU buffering (set `gpu_buffered_recording = False`)

## GPU Kernel Optimization Notes

### Fused Operations
The simulator uses fused CUDA kernels (via CuPy) to minimize kernel launches:
- Dynamics update combines: current injection + conductances + gating variables
- Connectivity application uses sparse matrix-vector products (cuSPARSE)
- Plasticity updates fused with spike detection

### Memory Access Patterns
- Neuron states: Coalesced access (contiguous arrays)
- Connectivity: CSR format optimized for row-wise access (each neuron's outputs)
- Avoid Python loops over neurons—always vectorize with CuPy

### Performance Profiling
Enable profiling (line ~829): `self._enable_profiling = True`
Check timings: `self._profile_timings` dict with deques for step timing

## Troubleshooting

### "Out of memory" errors
```bash
# Check current usage
nvidia-smi

# Reduce network size or connections
# Disable Hebbian + STP to cut memory usage ~60%
# Clear memory pool manually in Python console:
import cupy as cp; cp.get_default_memory_pool().free_all_blocks()
```

### Simulation crashes on start
```bash
# Verify CuPy installation
python -c "import cupy; print(cupy.__version__)"

# Check GPU compute capability (need 6.0+)
python -c "import cupy; print(cupy.cuda.Device(0).compute_capability)"
```

### Poor visualization FPS
- Increase `viz_update_interval_steps` from 17 to 34 (30fps) or 51 (20fps)
- Reduce `MAX_NEURONS_TO_RENDER`
- Ensure CUDA-GL interop is enabled (check console output on startup)

### Playback/Recording issues
- HDF5 files must have `.simrec.h5` extension
- Checkpoints use `.simstate.h5` extension
- Ensure H5 file is properly closed before re-opening
- GPU buffer size limits recording duration (check free VRAM)

## Code Navigation Tips

### Key Function Locations
- Main event loop: `render_dpg_frame()` (search for this)
- Simulation step: `_run_one_simulation_step()` (line ~1650+)
- Connectivity generation: `_generate_spatial_connections_3d()` (line ~1180+)
- OpenGL rendering: `render_gl()` (search for this)
- Recording logic: `_write_gpu_frames_to_disk()` (line ~1546+)

### Global State
- `global_simulation_bridge`: Main simulation interface (accessed by both threads)
- `global_gui_state`: UI thread state dict
- `global_viz_data_cache`: Visualization data shared with OpenGL
- All OpenGL globals prefixed with `gl_`

### Search Patterns
- CuPy arrays: Search `cp_`
- Configuration: Search `CoreSimConfig`, `VisualizationConfig`
- UI callbacks: Search `handle_` or `_button_click`
- File I/O: Search `.h5`, `h5py`, `HDF5`

## Performance Benchmarks

On RTX 3090 (24GB):
- 100K neurons, 1000 conn/neuron: ~45 FPS (sim + viz)
- 50K neurons, 1000 conn/neuron: ~60 FPS
- Recording overhead: ~5-10% slowdown with GPU buffering
- Playback: Instant frame seeking with GPU cache

## Future Enhancements
- Multi-GPU support via CuPy device management
- AMD ROCm port (replace CuPy with equivalent)
- SONATA/NeuroML export formats
- Synaptic delay queues (currently simplified)
- More neuron models (LIF, AdEx, multi-compartment)
