---
type: plan
status: live
date: 2026-04-06
---

# Phase 1: Module Split — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extract the ~14.8K-line `neural-simulator.py` monolith into a clean module structure (`sim/`, `experiment/`, `ui/`, `viz/`) without changing any behavior. All existing tests and headless scripts pass unchanged.

**Architecture:** Bottom-up extraction — start with zero-dependency modules (enums, config), then move up to modules with internal dependencies (kernels, connectivity), then the engine (bridge), then experiment system, then UI/viz. Each task extracts one module, adds an import bridge in the monolith, and verifies tests pass.

**Tech Stack:** Python 3.12, CuPy, DearPyGUI, OpenGL, HDF5. No new dependencies.

**Key constraint:** The monolith must remain functional after EVERY task. We use re-exports so that any code doing `from neural_simulator import SimulationBridge` continues to work. The headless scripts are updated at the end to import from the new paths.

**Reference:** Design doc at `docs/plans/2026-04-06-interactive-exploration-and-experiment-prototyping-design.md`

---

### Task 1: Create package structure and __init__ files

**Files:**
- Create: `sim/__init__.py`
- Create: `sim/enums.py`
- Create: `experiment/__init__.py`
- Create: `ui/__init__.py`
- Create: `viz/__init__.py`

**Step 1: Create directories and empty init files**

```bash
mkdir -p sim experiment ui viz
touch sim/__init__.py experiment/__init__.py ui/__init__.py viz/__init__.py
```

**Step 2: Extract enums to `sim/enums.py`**

Move all Enum classes from `neural-simulator.py`:
- `NeuronModel`
- `NeuronType`
- `DefaultHodgkinHuxleyParams` (the dict of HH parameter presets)
- `StimulusPatternType`
- `NeuronGroupRole`
- `ExperimentPhaseType`
- `TrainingMode`

Keep imports minimal — enums should have zero internal dependencies.

In `neural-simulator.py`, replace the enum definitions with:
```python
from sim.enums import (NeuronModel, NeuronType, DefaultHodgkinHuxleyParams,
                        StimulusPatternType, NeuronGroupRole, ExperimentPhaseType,
                        TrainingMode)
```

**Step 3: Run tests**

```bash
pytest tests/test_experiment_system.py tests/test_determinism.py -v
python run_benchmarks.py -b stdp-timing
python run_experiment_headless.py -e stimulus-response --num-neurons 1000
```

Expected: All pass — enums are the same objects, just imported from a different location.

**Step 4: Commit**

```bash
git add sim/ experiment/ ui/ viz/
git add neural-simulator.py
git commit -m "refactor: create package structure and extract enums to sim/enums.py"
```

---

### Task 2: Extract config dataclasses to `sim/config.py`

**Files:**
- Create: `sim/config.py`
- Modify: `neural-simulator.py`

**Step 1: Move dataclasses to `sim/config.py`**

Move these dataclass definitions:
- `CoreSimConfig`
- `VisualizationConfig`
- `RuntimeState`
- `GPUConfig`
- `ReadoutConfig`
- `TrainingConfig`
- `StimulusPattern`
- `StimulusChannel`
- `NeuronGroup`
- `ExperimentPhase`
- `ExperimentConfig`

`sim/config.py` imports from `sim.enums` for type references.

In `neural-simulator.py`, replace definitions with:
```python
from sim.config import (CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig,
                         ReadoutConfig, TrainingConfig, StimulusPattern, StimulusChannel,
                         NeuronGroup, ExperimentPhase, ExperimentConfig)
```

**Step 2: Run tests**

```bash
pytest tests/test_experiment_system.py tests/test_determinism.py -v
```

Expected: All 38 pass. The test file extracts experiment classes by exec'ing the source — it needs to find these classes in the namespace.

**Step 3: Fix test import if needed**

`tests/test_experiment_system.py` reads `neural-simulator.py` and exec's a block. If the dataclass definitions are now imports, the exec'd block won't find them. May need to update the test to import from `sim.config` directly, or ensure the exec'd block includes the imports.

**Step 4: Commit**

```bash
git add sim/config.py neural-simulator.py tests/test_experiment_system.py
git commit -m "refactor: extract config dataclasses to sim/config.py"
```

---

### Task 3: Extract neural profiles to `sim/profiles.py`

**Files:**
- Create: `sim/profiles.py`
- Modify: `neural-simulator.py`

**Step 1: Move profile definitions**

Move these to `sim/profiles.py`:
- `NEURAL_STRUCTURE_PROFILES` dict (all 15+ brain region profiles)
- `CONNECTIVITY_MOTIFS` dict (if separate from profiles)
- `get_compatible_hh_type_names_for_profile()` function
- Any helper functions used only by profile lookup

`sim/profiles.py` imports from `sim.enums` for NeuronModel, NeuronType references.

**Step 2: Update imports in monolith**

```python
from sim.profiles import NEURAL_STRUCTURE_PROFILES, get_compatible_hh_type_names_for_profile
```

**Step 3: Run tests**

```bash
pytest tests/test_determinism.py -v
python run_experiment_headless.py -e associative --num-trials 5 --num-neurons 1000
```

Expected: All pass. Profiles are looked up by name, same data.

**Step 4: Commit**

```bash
git add sim/profiles.py neural-simulator.py
git commit -m "refactor: extract neural structure profiles to sim/profiles.py"
```

---

### Task 4: Extract GPU kernels to `sim/kernels.py`

**Files:**
- Create: `sim/kernels.py`
- Modify: `neural-simulator.py`

**Step 1: Move all `@cp.fuse()` kernel functions**

Move these fused kernels:
- `fused_izhikevich2007_dynamics_update`
- `fused_izhikevich_legacy_dynamics_update`
- `fused_hodgkin_huxley_dynamics_update`
- `fused_adex_dynamics_update`
- `fused_hh_m_current_update`, `fused_hh_CaT_current_update`, etc.
- `fused_hh_h_current_update`, `fused_hh_NaP_current_update`
- `fused_conductance_decay_and_current`
- `fused_nmda_update_and_current`
- `fused_stp_decay_recovery`
- `fused_stdp_weight_update`
- `fused_homeostasis_update`
- `fused_eligibility_trace_decay`

`sim/kernels.py` imports only `cupy as cp` at module level.

**Step 2: Update imports in monolith**

```python
from sim.kernels import (fused_izhikevich2007_dynamics_update,
                          fused_hodgkin_huxley_dynamics_update, ...)
```

**Step 3: Run tests and benchmarks**

```bash
pytest tests/test_determinism.py -v
python run_benchmarks.py -b stdp-timing
python run_benchmarks.py -b ei-balance
```

Expected: All pass. Kernels are pure functions — same inputs, same outputs regardless of where they're defined.

**Step 4: Commit**

```bash
git add sim/kernels.py neural-simulator.py
git commit -m "refactor: extract GPU fused kernels to sim/kernels.py"
```

---

### Task 5: Extract connectivity generators to `sim/connectivity.py`

**Files:**
- Create: `sim/connectivity.py`
- Modify: `neural-simulator.py`

**Step 1: Extract connection generation methods**

These are currently methods on SimulationBridge. Extract them as standalone functions that take positions, traits, and config as arguments:
- `_generate_spatial_connections_3d_vectorized` → `generate_spatial_connections_gpu(n, k, positions, traits, config)`
- `_generate_spatial_connections_3d_chunked` → `generate_spatial_connections_chunked(...)`
- `_generate_spatial_connections_3d_binned` → `generate_spatial_connections_binned(...)`
- `_generate_random_connections_large` → `generate_random_connections(...)`

SimulationBridge keeps thin wrapper methods that call these functions with `self.cp_neuron_positions_3d`, `self.cp_traits`, etc.

**Step 2: Run tests**

```bash
pytest tests/test_determinism.py::TestDeterministicConnectivity -v
python run_benchmarks.py -b gamma-oscillations
```

Expected: Connectivity generation produces same results. Gamma benchmark passes (cross-type connections present).

**Step 3: Commit**

```bash
git add sim/connectivity.py neural-simulator.py
git commit -m "refactor: extract connectivity generators to sim/connectivity.py"
```

---

### Task 6: Extract experiment system to `experiment/`

**Files:**
- Create: `experiment/engine.py`
- Create: `experiment/stimulus.py`
- Create: `experiment/readout.py`
- Create: `experiment/training.py`
- Create: `experiment/presets.py`
- Create: `experiment/groups.py`
- Modify: `neural-simulator.py`

**Step 1: Extract in dependency order**

1. `experiment/groups.py` — `NeuronGroupManager` (depends on sim.config)
2. `experiment/stimulus.py` — `StimulusManager` (depends on sim.config)
3. `experiment/readout.py` — `ReadoutEngine` with band power and synchrony (depends on sim.config, groups)
4. `experiment/training.py` — `TrainingProtocolEngine` (depends on readout, groups)
5. `experiment/engine.py` — `ExperimentEngine` (depends on all above)
6. `experiment/presets.py` — `ExperimentPresets` factory (depends on sim.config)

Also move `experiment_config_from_dict()` and `experiment_config_to_dict()` serialization helpers.

**Step 2: Update `experiment/__init__.py` to re-export key classes**

```python
from experiment.engine import ExperimentEngine
from experiment.presets import ExperimentPresets
from experiment.readout import ReadoutEngine
# etc.
```

**Step 3: Update imports in monolith and headless scripts**

In `neural-simulator.py`:
```python
from experiment import ExperimentEngine, ExperimentPresets, ReadoutEngine
```

**Step 4: Run tests**

```bash
pytest tests/test_experiment_system.py -v
python run_experiment_headless.py -e associative --num-trials 10 --num-neurons 1000
python run_experiment_headless.py -e reinforcement --num-trials 10 --num-neurons 1000
python run_benchmarks.py -b homeostasis
```

Expected: All pass. Experiment system behavior unchanged.

**Step 5: Commit**

```bash
git add experiment/ neural-simulator.py
git commit -m "refactor: extract experiment system to experiment/ package"
```

---

### Task 7: Extract SimulationBridge to `sim/bridge.py`

**Files:**
- Create: `sim/bridge.py`
- Modify: `neural-simulator.py`

**Step 1: Move SimulationBridge class**

This is the largest single extraction (~3-4K lines). `sim/bridge.py` imports from:
- `sim.config` (all config dataclasses)
- `sim.enums` (NeuronModel, NeuronType)
- `sim.kernels` (all fused kernels)
- `sim.profiles` (NEURAL_STRUCTURE_PROFILES)
- `sim.connectivity` (generator functions)
- `experiment` (ExperimentEngine, etc.)

Keep the `_run_one_simulation_step()`, `_initialize_simulation_data()`, checkpoint save/load, recording, and all GPU state management.

Remove from SimulationBridge: any UI-specific methods (like `get_latest_simulation_data_for_gui`). These become adapter functions in `ui/callbacks.py` or stay in the monolith temporarily.

**Step 2: Update `sim/__init__.py`**

```python
from sim.bridge import SimulationBridge
from sim.config import CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig
from sim.enums import NeuronModel, NeuronType
```

**Step 3: Run full test suite**

```bash
pytest tests/test_experiment_system.py tests/test_determinism.py -v
python run_benchmarks.py -b all
python run_experiment_headless.py -e associative --num-trials 10 --num-neurons 1000
python run_parameter_sweep.py -e associative --sweep "stdp_a_plus=0.008,0.012" --num-trials 10 --num-neurons 1000
```

Expected: All pass. This is the critical validation — the bridge is the heart of the simulator.

**Step 4: Commit**

```bash
git add sim/bridge.py sim/__init__.py neural-simulator.py
git commit -m "refactor: extract SimulationBridge to sim/bridge.py"
```

---

### Task 8: Extract OpenGL visualization to `viz/`

**Files:**
- Create: `viz/renderer.py`
- Create: `viz/camera.py`
- Create: `viz/overlays.py`
- Modify: `neural-simulator.py`

**Step 1: Extract rendering code**

- `viz/renderer.py` — `render_scene_gl()`, `update_gl_data()`, `fast_vbo_update()`, VBO management, `init_gl()`, neuron color computation, filter functions
- `viz/camera.py` — camera state (azimuth, elevation, radius), mouse callbacks, keyboard callbacks, `reshape_gl()`
- `viz/overlays.py` — `render_text_gl()`, HUD footer rendering, `opengl_viz_config` defaults, `TRAIT_COLOR_MAP`

These depend on OpenGL (PyOpenGL) and CuPy but NOT on DearPyGUI.

**Step 2: Handle the global state**

The viz code currently uses many globals (`gl_neuron_pos_vbo`, `gl_num_neurons_to_draw`, `opengl_viz_config`, etc.). Bundle these into a `VizState` class or keep as module-level state in `viz/renderer.py`.

**Step 3: Verify GUI still works**

This requires manually launching the app:
```bash
python neural-simulator.py
```
- Verify 3D rendering works
- Verify camera controls (orbit, pan, zoom)
- Verify neuron filtering
- Verify spike highlighting

Also run headless (no GL):
```bash
python run_experiment_headless.py -e stimulus-response --num-neurons 1000
```
Expected: Headless still works (viz modules only loaded when OpenGL is available).

**Step 4: Commit**

```bash
git add viz/ neural-simulator.py
git commit -m "refactor: extract OpenGL visualization to viz/ package"
```

---

### Task 9: Extract DearPyGUI UI to `ui/`

**Files:**
- Create: `ui/layout.py`
- Create: `ui/callbacks.py`
- Modify: `neural-simulator.py`

**Step 1: Extract UI code**

- `ui/layout.py` — `create_gui_layout()` function and all widget creation code (~2-3K lines of dpg.add_* calls)
- `ui/callbacks.py` — all callback functions (`_handle_start_sim`, `_handle_experiment_preset_change`, `_update_sim_config_from_ui`, `_populate_ui_from_config_dict`, etc.)

These depend on DearPyGUI, `sim/`, and `experiment/`.

**Step 2: Wire the entry point**

`neural-simulator.py` becomes the thin orchestrator:
```python
from sim import SimulationBridge, CoreSimConfig
from experiment import ExperimentEngine
from ui.layout import create_gui_layout
from ui.callbacks import setup_callbacks
from viz.renderer import init_gl, render_scene_gl

def main():
    # Parse args
    # Create bridge
    # Start sim thread
    # Create GUI (if not headless)
    # Main loop
```

**Step 3: Verify full GUI works**

```bash
python neural-simulator.py
```
- All parameter controls work
- Experiment system works
- Profile loading works
- Checkpoints work
- Recording/playback works

**Step 4: Run all tests**

```bash
pytest tests/ -v
python run_benchmarks.py -b all
```

**Step 5: Commit**

```bash
git add ui/ neural-simulator.py
git commit -m "refactor: extract DearPyGUI UI to ui/ package"
```

---

### Task 10: Update headless scripts and final cleanup

**Files:**
- Modify: `run_experiment_headless.py`
- Modify: `run_parameter_sweep.py`
- Modify: `run_benchmarks.py`
- Modify: `tests/test_experiment_system.py`

**Step 1: Update headless script imports**

Replace the `importlib.util.spec_from_file_location` hack with clean imports:

```python
from sim import SimulationBridge, CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig
from sim.enums import NeuronModel
from experiment import ExperimentEngine, ExperimentPresets
```

Remove the `load_simulator()` function from all three scripts.

**Step 2: Update test imports**

`tests/test_experiment_system.py` currently exec's a block of the monolith. Replace with:
```python
from sim.config import *
from sim.enums import *
from experiment import *
```

Or keep the MockCuPy approach but import classes directly.

**Step 3: Run everything**

```bash
pytest tests/ -v
python run_benchmarks.py -b all
python run_experiment_headless.py -e associative --num-trials 10 --num-neurons 1000
python run_experiment_headless.py -e stimulus-response --num-neurons 1000
python run_experiment_headless.py -e frequency-response --num-neurons 1000
python run_experiment_headless.py -e reinforcement --num-trials 10 --num-neurons 1000
python run_parameter_sweep.py -e associative --sweep "stdp_a_plus=0.008,0.012" --num-trials 10 --num-neurons 1000
```

Expected: All pass with clean imports.

**Step 4: Remove dead code from monolith**

After extraction, `neural-simulator.py` should be ~500 lines: imports, CLI argument parsing, thread orchestration, and main loop. Remove any code that was extracted but accidentally left behind.

**Step 5: Final commit**

```bash
git add -A
git commit -m "refactor: complete module split - update headless scripts and tests to use new imports"
```

---

## Verification Checklist (run after all tasks)

```bash
# Unit tests
pytest tests/ -v

# All 5 biological benchmarks
python run_benchmarks.py -b all

# All 4 experiment presets
python run_experiment_headless.py -e associative --num-trials 20 --num-neurons 5000
python run_experiment_headless.py -e stimulus-response --num-neurons 5000
python run_experiment_headless.py -e frequency-response --num-neurons 5000
python run_experiment_headless.py -e reinforcement --num-trials 20 --num-neurons 5000

# Parameter sweep
python run_parameter_sweep.py -e associative --sweep "stdp_a_plus=0.008,0.016" --num-trials 10 --num-neurons 5000

# GUI (manual)
python neural-simulator.py
# → Verify 3D rendering, parameter controls, experiment system, recording, profiles
```

---

## Phase 2-7 Plans

Each subsequent phase gets its own plan document, written after the prior phase lands:

- **Phase 2:** `2026-XX-XX-phase2-data-bus-state-manager.md`
- **Phase 3:** `2026-XX-XX-phase3-live-plots.md`
- **Phase 4:** `2026-XX-XX-phase4-neuron-picking-inspector.md`
- **Phase 5:** `2026-XX-XX-phase5-experiment-dashboard.md`
- **Phase 6:** `2026-XX-XX-phase6-sweep-ui-results.md`
- **Phase 7:** `2026-XX-XX-phase7-figure-export-polish.md`

Each plan follows the same TDD structure: failing test → implement → verify → commit.
