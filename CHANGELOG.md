# Changelog

All notable changes to the GPU-Accelerated Neural Network Simulator will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **G1 pipeline GO** - First end-to-end dataset → encoder → sim → decoder → loss round-trip
  - `research/datasets/tiny_patterns.py` + canonical `.npz` - K=4 Poisson-rate synthetic dataset
  - `RATE_VECTOR_POISSON` stimulus pattern - per-neuron Poisson rate encoding
  - `SimulationBridge.inject_explicit_wiring()` - injectable explicit CSR connectivity for research networks
  - Three runners explored: v1 teacher-forced STDP (NO-GO), v2 external perceptron (NO-GO), v3 reservoir + external LogReg (**GO** - mean 71.3% test acc across 3 seeds, threshold 55%)
  - v1/v2 post-mortem: sim's default `propagation_strength=0.05` is calibrated for ~1000 converging synapses per neuron; the 68-neuron tiny architecture needs non-default params. v3 uses a 264-neuron reservoir in the sim's calibrated regime.
  - Full findings in `research/findings/2026-04-20-g1.md`

- **Profile System** - Biologically accurate brain region presets and UI integration
  - 9 brain region profile JSONs with realistic neuron models and connectivity: Cortex L2/3, Cortex L4, Hippocampus CA1, Hippocampus CA3, Thalamus TC-TRN, Basal Ganglia Striatum, Basal Ganglia STN-GPe, Cerebellar Cortex, Spinal Cord
  - Quick Demo profile for rapid testing
  - Full profile dropdown menu in UI that auto-populates from `simulation_profiles/*.json` files
  - Refresh button to reload profiles from disk without restarting

- **Plasticity Parameters in UI** - STDP, reward modulation, and structural plasticity
  - 28 new fields added to SimulationConfiguration for complete plasticity roundtrip (save/load)
  - Support for STDP timing windows, reward modulation learning rates, and structural synapse thresholds
  - Full persistence in simulation profiles and checkpoint files

- **Per-Connection-Type STP** - Biologically realistic short-term plasticity heterogeneity
  - New `enable_per_type_stp` parameter and per-type arrays `stp_U_per_type`, `stp_tau_d_per_type`, `stp_tau_f_per_type`
  - Each is indexed by connection type [E->E, E->I, I->E, I->I]
  - Different brain regions now use experimentally validated STP profiles per connection type
  - UI table exposes all 12 parameters (4 connection types × 3 STP variables)

- **Activity-Dependent Structural Synaptogenesis** - Cline & Haas 2008 model
  - New `struct_plast_activity_bias` parameter (0.0-1.0, default 0.5)
  - Biases new synapse formation toward co-active neuron pairs using activity EMA
  - 0 = random synapse formation; 1 = fully activity-driven (Hebbian structuring)

- **COO Cache Invalidation** - Fixed stale data handling in GPU memory
  - Cache invalidation in `clear_simulation_state_and_gpu_memory()` prevents stale sparse matrix data across reinitializations

### Fixed
- **STP/Connection Shape Mismatch at Scale** - CSR matrix deduplication bug
  - Fixed shape mismatch occurring at 100K+ neurons caused by CSR matrix addition deduplicating overlapping (pre,post) pairs
  - Now uses `cp_connections.nnz` as authoritative size instead of stale shape values
  - Structural plasticity synapse count now properly synced after CSR addition

- **Synaptic Scaling Crash** - Stale COO cache surviving reinitialization
  - COO cache no longer persists across simulation reinitializations, preventing crashes when synaptic scaling is active

- **Unicode Handling on Windows** - JSON I/O encoding issue
  - UnicodeDecodeError on Windows (cp1252) when loading profile JSONs with Unicode characters (em dashes, etc.)
  - All JSON I/O now uses UTF-8 encoding explicitly

- **Em Dash Rendering** - DearPyGui font limitation
  - Em dashes rendered as question marks in DearPyGUI default font
  - Replaced with regular hyphens in all UI text and profile names

### Changed
- **Hodgkin-Huxley Numerical Stability** - Automatic time step adjustment
  - dt automatically reduces to 0.05ms when switching to HH model for improved numerical stability
  - dt automatically restores to 0.5ms when switching away from HH model
  - Prevents instabilities in voltage-gated kinetics at larger time steps

- **Homeostatic Plasticity Timescale** - Biologically realistic adaptation
  - EMA alpha reduced from 0.01 (tau ~100ms) to 0.0002 (tau ~5s at dt=1ms)
  - Threshold adapt rate reduced from 0.015 to 0.0005
  - Homeostatic mechanisms now operate on seconds-to-minutes timescale, matching experimental observations

- **Inhibitory Reversal Potential** - Corrected Nernst equilibrium
  - E_inh changed from -70mV to -75mV (matches Cl- Nernst potential at 37°C)
  - Inhibitory propagation strength scaled by 0.7 to compensate for increased driving force
  - Improves accuracy of GABAergic synaptic transmission

- **.gitignore** - Profile tracking and auto-tuning separation
  - Now tracks `simulation_profiles/*.json` to include biologically accurate presets in repository
  - Excludes `auto_tuned_overrides.json` to prevent auto-tuned parameters overwriting checked-in profiles

- **Profile Files** - Superseded files removed
  - Removed 7 old profile files replaced by new standardized brain region profiles

- **System Logs Panel** - Comprehensive log viewing and management
  - Real-time display of all console output within the GUI
  - Auto-scroll functionality using DearPyGUI's `tracked` parameter
  - Search functionality with previous/next navigation through matches
  - Export logs to timestamped text files
  - Clear logs functionality
  - Thread-safe `LogCapture` class for zero-overhead console mirroring
  
- **Performance Test Controls**
  - Stop button for halting running benchmarks and auto-tuning mid-execution
  - Proper state tracking to preserve existing result files
  - Informative logging showing which test type was stopped
  - Located above "Reload Auto-Tuned Overrides" button in GUI

### Changed
- **VRAM Utilization for Initialization** - Increased chunking from 40% to 70% of free VRAM
  - ~2x faster initialization for networks with 50K+ neurons
  - Example: With 18GB free VRAM, now uses 12.6GB instead of 7.2GB for chunking
  - Maintains 30% safety margin for stability
  
- **GUI Layout Improvements**
  - Auto-tuning button now stretches to fill available width (width=-80)
  - Better space utilization when window is resized wider
  - "Quick" checkbox properly positioned at right edge

### Fixed
- Auto-scroll in System Logs now works correctly using DPG best practices
  - Replaced manual scroll manipulation with `tracked=True` and `track_offset=1.0`
  - Dynamic height adjustment based on text size for proper scrolling
  - Toggle auto-scroll on/off via checkbox callback
  
- Performance test stop functionality prevents corrupted result files
  - Benchmark and auto-tuning only save results at completion
  - Stopping mid-run preserves any previously existing result files

### Technical Details
- Log capture uses thread-safe deque with 5000-line rolling buffer
- System logs display widget uses `child_window` with `input_text` for proper scrolling
- Auto-scroll implementation follows official DearPyGUI documentation patterns
- Stop flags (`performance_test_stop_flag`, `performance_test_running_type`) properly managed in try/finally blocks

## [Previous Versions]

See git commit history for details on earlier changes.
