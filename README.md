# GPU-Accelerated Neural Network Simulator

A high-performance 3D neural network simulator with real-time OpenGL visualization, leveraging NVIDIA CUDA/CuPy for massively parallel GPU computation. Simulates large-scale networks (10K-100K+ neurons) with biologically-inspired models including Izhikevich and Hodgkin-Huxley neurons, synaptic plasticity, and spatial connectivity.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![CUDA](https://img.shields.io/badge/CUDA-CuPy-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## Features

### GPU-Accelerated Simulation
- **CuPy-based computation**: All neural dynamics run on GPU using fused CUDA kernels
- **Scalable**: Efficiently handles 10K-100K+ neurons with millions of synaptic connections
- **Real-time performance**: 60 FPS visualization with parallel simulation updates
- **Memory optimized**: Smart GPU memory management with automatic garbage collection

### Neuron Models
- **Izhikevich 2007 formulation**: Fast, biologically plausible spiking neurons
  - Regular Spiking (RS) Cortical Pyramidal cells
  - Fast Spiking (FS) Cortical Interneurons
  - Custom parameter configurations
- **Hodgkin-Huxley model**: Detailed biophysical neuron dynamics
  - Layer 5 Cortical Pyramidal neurons
  - Temperature-dependent kinetics (Q10 scaling)
  - Multi-compartment gating variables (m, h, n)

### Synaptic Plasticity
- **Hebbian Learning (LTP/LTD)**: Activity-dependent weight modification
- **Short-Term Plasticity (STP)**: Tsodyks-Markram depression and facilitation
- **Homeostatic Plasticity**: Adaptive firing thresholds for network stability
- **Conductance-based synapses**: Separate excitatory (AMPA) and inhibitory (GABA) channels

### Network Architecture
- **3D spatial connectivity**: Distance-dependent connection probabilities
- **Trait-based organization**: Neuron populations with shared properties
- **Watts-Strogatz networks**: Small-world topology generation
- **Inhibitory interneurons**: Configurable E/I balance

### Visualization
- **Real-time 3D OpenGL rendering**: Hardware-accelerated graphics
- **Interactive camera**: Orbit, pan, zoom controls
- **Activity visualization**: Color-coded neuron states (firing, silent, recent activity)
- **Synaptic pulse rendering**: Visual feedback for spike propagation
- **GPU-efficient filtering**: Type-based and activity-based neuron filtering on GPU

### Recording & Playback
- **HDF5-based recording**: Efficient compression and streaming to disk
- **GPU-buffered recording**: Zero-copy recording of entire simulations in VRAM
- **Frame-accurate playback**: Scrub through recorded simulations
- **State checkpointing**: Save and restore full simulation state

### User Interface
- **DearPyGUI control panel**: Comprehensive parameter configuration
- **Live monitoring**: Real-time metrics (firing rate, spike counts, plasticity events)
- **System logs panel**: Real-time console output with search, export, and auto-scroll
- **Profile management**: Save/load simulation configurations as JSON
- **Performance testing**: Built-in benchmarking with stop/start controls
- **Keyboard shortcuts**: Quick access to common operations

## Requirements

### Core Dependencies
```
python >= 3.8
cupy-cuda12x >= 12.0.0  # For CUDA 12.x (adjust for your CUDA version)
numpy >= 1.21.0
h5py >= 3.7.0
dearpygui >= 1.9.0
```

### Graphics Dependencies
```
PyOpenGL >= 3.1.6
PyOpenGL-accelerate >= 3.1.6
```

### Hardware Requirements
- **GPU**: NVIDIA GPU with CUDA compute capability 6.0+ (Pascal or newer)
- **VRAM**: 
  - 4GB minimum (for 1K-10K neurons)
  - 8GB recommended (for 10K-50K neurons)
  - 16GB+ for large networks (50K-100K+ neurons)
- **RAM**: 8GB+ system memory
- **Display**: OpenGL 3.3+ compatible graphics

## Installation

1. **Install CUDA Toolkit** (if not already installed):
   - Download from [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)
   - Version 11.x or 12.x recommended

2. **Install CuPy** (match your CUDA version):
   ```bash
   # For CUDA 12.x
   pip install cupy-cuda12x
   
   # For CUDA 11.x
   pip install cupy-cuda11x
   ```

3. **Install other dependencies**:
   ```bash
   pip install numpy h5py dearpygui PyOpenGL PyOpenGL-accelerate
   ```

4. **Clone and run**:
   ```bash
   git clone https://github.com/danthi123/neural-simulator.git
   cd neural-simulator
   python neural-simulator.py
   ```

## Quick Start

### Basic Usage (GUI Mode)
1. Launch the simulator: `python neural-simulator.py`
2. Configure parameters in the DearPyGUI control panel (left side)
3. Click **"Apply Changes & Reset Sim"** to initialize the network
4. Click **"Start"** to begin simulation
5. Use mouse in OpenGL window to navigate:
   - **Left click + drag**: Rotate camera
   - **Right click + drag**: Pan camera
   - **Scroll wheel**: Zoom in/out

### Quick Auto-Tuning (Headless Mode)

The simulator includes a headless **auto-tuning** workflow that scans combinations of
neuron model, neural structure profile, and Hodgkin–Huxley preset to pick sensible
external drive scales so networks are active but not saturated.

- **Run a full tuning sweep** (all profiles and presets):
  ```bash
  python neural-simulator.py --auto-tune
  ```
- **Run a faster, reduced sweep** (for testing):
  ```bash
  python neural-simulator.py --auto-tune --quick
  ```

This produces `simulation_profiles/auto_tuned_overrides.json`, which contains
per-combination overrides (e.g. `hh_external_drive_scale`, `adex_external_drive_scale`).
These are automatically loaded and applied whenever you:

- Select the corresponding **Neuron Model / Neural Structure Profile / HH preset** in the UI
- Click **"Apply Changes & Reset Sim"**

In the HH and AdEx parameter panels you can see and adjust these as:

- **External Drive Scale (HH, auto-tuned)** – scales baseline HH DC input
- **External Drive Scale (AdEx, auto-tuned)** – scales baseline AdEx DC input

You can also click the **"Reset HH Drive to Auto-Tuned"** or
**"Reset AdEx Drive to Auto-Tuned"** buttons to restore the slider to the
auto-tuned value for the current combination, then press **Apply & Reset**
to use it in the simulation.

### Example Configurations

#### Small Network (Fast Preview)
```
Neurons: 1,000
Connections per neuron: 100
Model: Izhikevich (RS Cortical Pyramidal)
dt: 1.0 ms
```

#### Medium Network (Balanced)
```
Neurons: 10,000
Connections per neuron: 500
Model: Izhikevich (mixed RS/FS)
Enable Hebbian Learning: Yes
Enable STP: Yes
dt: 1.0 ms
```

#### Large Network (High Performance GPU)
```
Neurons: 50,000+
Connections per neuron: 1,000+
Model: Izhikevich
dt: 1.0 ms
Note: Requires 16GB+ VRAM
```

## Key Parameters

### Core Simulation
- **Total Simulation Time**: Duration in milliseconds
- **dt (Time Step)**: Integration timestep (0.025-1.0 ms)
  - Smaller values = more accurate but slower
  - Izhikevich: 1.0 ms typical
  - Hodgkin-Huxley: 0.025-0.05 ms recommended
- **Seed**: Random seed for reproducibility (-1 for random)

### Network Structure
- **Connections per Neuron**: Average outgoing synapses
- **Num Traits**: Number of neuron populations/types
- **Enable Watts-Strogatz**: Use small-world topology vs spatial

### Plasticity
- **Hebbian Learning Rate**: LTP strength (0.0001-0.001 typical)
- **STP Parameters**: 
  - U: Baseline utilization (0.1-0.5)
  - tau_d: Depression timescale (50-500 ms)
  - tau_f: Facilitation timescale (20-200 ms)
- **Homeostasis Target Rate**: Desired firing rate for stability (0.01-0.05)

### Visualization
- **Viz Update Interval**: Steps between visual updates (17 for ~60fps at dt=1.0ms)
- **Point Size**: Neuron rendering size
- **Max Neurons to Render**: Performance cap for visualization
- **Spiking Mode**: 
  - Highlight Spiking: Show recent activity
  - Show Only Spiking: Filter inactive neurons
  - No Spiking Highlight: Static colors

## Command-Line Usage

The main entry point is `neural-simulator.py`.

### GUI Mode (default)

```bash
python neural-simulator.py
```

Starts the DearPyGUI control window and the OpenGL 3D visualization (if PyOpenGL is
available). All configuration is done through the UI.

### Auto-Tuning Mode (headless)

```bash
python neural-simulator.py --auto-tune [--quick]
```

- `--auto-tune` – run the headless tuning workflow instead of launching the GUI.
- `--quick` – optional; restricts the sweep to a smaller subset of profiles/presets
  for faster runs (useful while developing).

The tuner:
- Iterates through Hodgkin–Huxley + AdEx model combinations with all defined
  **Neural Structure Profiles**.
- For HH, only uses **profile-compatible HH presets** (e.g. striatum → MSN,
  CA1 → CA1 pyramidal, STN–GPe → STN bursting). Generic/unstructured profiles
  still allow all HH presets.
- For each combination, tests several external drive scales.
- Measures spike activity, fraction of neurons that spiked, and connectivity.
- Chooses the best scale according to simple criteria (network is alive but not
  seizure-like).
- Saves the results into `simulation_profiles/auto_tuned_overrides.json`.

At runtime, whenever a matching combination is selected in the GUI, these values
are applied automatically before initialization.

## Keyboard Shortcuts

### OpenGL Window
- **ESC**: Exit application
- **R**: Reset camera position
- **N**: Toggle synapse visibility
- **P**: Pause/resume simulation
- **S**: Step simulation (1ms forward)

### General
- **Ctrl+S**: Save profile
- **Ctrl+L**: Load profile
- **Ctrl+Shift+S**: Save checkpoint
- **Ctrl+Shift+L**: Load checkpoint

## File Formats

### Profiles (.json)
- Stores simulation configuration parameters
- Includes GUI settings and visualization preferences
- Human-readable, can be edited manually
- Location: `simulation_profiles/`

### Checkpoints (.simstate.h5)
- HDF5 format with full simulation state
- Includes neuron states, connectivity, plasticity variables
- Compressed for efficient storage
- Location: `simulation_checkpoints_h5/`

### Recordings (.simrec.h5)
- HDF5 format with frame-by-frame simulation data
- GPU-buffered recording for maximum performance
- Includes initial state and all dynamic variables per frame
- Location: `simulation_recordings_h5/`

## GPU Configuration and Profiling

### GPU Feature Flags

Configure GPU-specific features via `GPUConfig` dataclass:

**Recording Modes:**
- `gpu_buffered` (default): Store all frames in VRAM for maximum speed
- `streaming`: Write frames directly to disk as generated
- `disabled`: No recording

**Playback Modes:**
- `gpu_cached` (default): Load entire recording into VRAM for instant seeking
- `streaming`: Read frames from disk on demand
- `auto`: Automatically choose based on available GPU memory

**CUDA-OpenGL Interop:**
Enable zero-copy GPU→OpenGL transfers for visualization with no CPU roundtrip.

**Memory Management:**
- `memory_pool_limit_fraction`: Max GPU memory for CuPy pool (default: 0.8)
- `memory_pressure_threshold`: Trigger cleanup above this usage (default: 0.9)
- `enable_adaptive_quality`: Reduce quality under memory pressure

### Performance Profiling

Enable detailed profiling in your simulation:

```python
from neural_simulator import GPUConfig, SimulationBridge, CoreSimConfig

gpu_config = GPUConfig(
    enable_profiling=True,
    profiling_detailed=True,
    profiling_window_size=1000  # Keep last 1000 step timings
)

sim = SimulationBridge(
    core_config=CoreSimConfig(num_neurons=10000),
    gpu_config=gpu_config
)
```

Get profiling statistics:
```python
stats = sim.get_profiling_stats()
print(f"Mean step time: {stats['step_total']['mean']*1000:.2f}ms")
print(f"P95 step time: {stats['step_total']['p95']*1000:.2f}ms")
```

Export full profiling report:
```python
sim.export_profiling_report("profile_report.json")
```

### Benchmarking

Run comprehensive benchmark suite:
```bash
python benchmark.py --output results.json
```

Quick benchmark (reduced configurations):
```bash
python benchmark.py --quick
```

Compare against baseline:
```bash
python benchmark.py --compare benchmarks/baseline_v1.json
```

The benchmark suite:
- Sweeps network sizes (1K, 10K, 50K neurons)
- Tests different connection densities
- Runs all neuron models (Izhikevich, HH, AdEx)
- Measures step time, GPU memory, throughput
- Outputs JSON report with system info

### Determinism and Reproducibility

For reproducible simulations:

1. **Set explicit seed:**
```python
config = CoreSimConfig(seed=42)
```

2. **Track actual seed used:**
```python
sim._initialize_simulation_data()
print(f"Actual seed: {sim.runtime_state.actual_seed_used}")
```

3. **Run determinism tests:**
```bash
pytest tests/test_determinism.py -v
```

All RNG sources (CuPy, NumPy, random) are initialized together for full determinism.

## Performance Optimization

### GPU Memory Management
The simulator automatically manages GPU memory, but you can optimize:
- Configure `GPUConfig.memory_pool_limit_fraction` (default: 0.8)
- Adjust `memory_pressure_threshold` for earlier cleanup
- Reduce `Max Neurons to Render` if visualization is slow
- Disable synaptic pulses for better performance
- Use larger `Viz Update Interval` to reduce GPU-CPU transfers
- Close other GPU-intensive applications

### Network Size Guidelines
| Network Size | VRAM Usage | Expected FPS | Notes |
|-------------|------------|--------------|-------|
| 1K neurons  | ~500 MB    | 60+ FPS      | Real-time interactive |
| 10K neurons | ~2 GB      | 60 FPS       | Smooth, recommended |
| 50K neurons | ~8 GB      | 30-60 FPS    | High performance GPU |
| 100K neurons| ~20 GB     | 15-30 FPS    | Enthusiast/research |

### CPU vs GPU Bottlenecks
- **GPU bottleneck**: Reduce network size or disable STP/Hebbian learning
- **CPU bottleneck**: Increase visualization update interval
- **Transfer bottleneck**: Enable GPU-buffered recording

## Known Limitations

1. **Memory constraints**: Networks >100K neurons require very large VRAM (20GB+)
2. **Visualization performance**: Rendering >50K neurons simultaneously may reduce FPS
3. **Platform**: Currently Windows/Linux with NVIDIA GPUs only (CUDA requirement)
4. **Precision**: Uses float32 for performance; float64 not currently supported

## Troubleshooting

### "Out of memory" errors
- Reduce number of neurons or connections per neuron
- Close other applications using GPU memory
- Check GPU memory with `nvidia-smi`
- Disable Hebbian learning and STP for reduced memory usage

### Poor visualization performance
- Reduce `Max Neurons to Render`
- Increase `Viz Update Interval`
- Disable synaptic pulse rendering
- Use "Show Only Spiking" filter mode

### Simulation crashes on start
- Ensure CUDA drivers are up to date
- Verify CuPy installation: `python -c "import cupy; print(cupy.cuda.runtime.getDeviceCount())"`
- Check that GPU compute capability is 6.0+

## Architecture

### Thread Model
- **Main Thread**: DearPyGUI event loop and OpenGL rendering
- **Simulation Thread**: GPU-accelerated neural dynamics computation
- **Communication**: Lock-free queues for inter-thread messaging

### GPU Optimization Techniques
- **Fused CUDA kernels**: Minimized kernel launches for dynamics updates
- **Sparse matrix operations**: CSR format for connectivity
- **Vectorized operations**: Full GPU parallelization of neuron updates
- **Zero-copy transfers**: CUDA-OpenGL interop for visualization
- **Memory pooling**: Reduced allocation overhead with CuPy memory pool

## Contributing

Contributions are welcome! Areas for improvement:
- Additional neuron models (LIF, AdEx, multi-compartment)
- Network analysis tools (connectivity statistics, firing patterns)
- Export to SONATA/NeuroML formats
- Multi-GPU support for larger networks
- AMD ROCm/HIP port for non-NVIDIA GPUs

## License

MIT License - See LICENSE file for details

## Citation

If you use this simulator in your research, please cite:

```bibtex
@software{neural_simulator_2025,
  title = {GPU-Accelerated Neural Network Simulator},
  author = {danthi123},
  year = {2025},
  url = {https://github.com/danthi123/neural-simulator}
}
```

## Acknowledgments

- Izhikevich neuron model: Izhikevich, E. M. (2007). Dynamical Systems in Neuroscience
- Hodgkin-Huxley model: Hodgkin & Huxley (1952). J. Physiol.
- STP model: Tsodyks & Markram (1997). PNAS
- CuPy library for GPU acceleration
- DearPyGUI for UI framework
- OpenGL for 3D visualization

## Contact

For questions, issues, or suggestions:
- GitHub Issues: [Project Issues](https://github.com/danthi123/neural-simulator/issues)

---

**Note**: This is a research/educational tool. For production neuroscience simulations, consider established frameworks like NEST, Brian2, or NEURON.
