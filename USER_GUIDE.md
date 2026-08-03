# User Guide

Neural Simulator is active research software. Its stable user surface is the
simulation engine and its inspection tools; the developing brain is not yet an
open-ended conversational system. See [Current State](docs/CURRENT-STATE.md) for
the present capability boundary.

Follow the [Quickstart](QUICKSTART.md) before using this guide.

## Choose The Backend

| Backend | Set | Appropriate uses | Current limits |
|---|---|---|---|
| CPU through NumPy | `SIM_BACKEND=numpy` | Focused tests, small backend-compatible probes, and parallel independent runs | Slower; the desktop GUI and several older launchers import CuPy directly |
| NVIDIA GPU through CuPy | `SIM_BACKEND=cupy` | Desktop simulator, large spiking runs, GPU-specific experiments, and legacy headless launchers | Requires a working NVIDIA driver and compatible CuPy wheel |

Always set the backend explicitly for a recorded run. The default mode tries
CuPy first and falls back to NumPy, which is convenient for exploration but can
hide an unintended device change.

## Desktop Simulator

The current desktop entry point is an NVIDIA GPU workflow:

```bash
SIM_BACKEND=cupy python neural-simulator.py
```

It opens a Dear PyGui control window and, when OpenGL is available, a separate 3D
network view.

### Basic Workflow

1. Choose a neuron count, random seed, neuron model, and neural structure profile
   under **Core Simulation Parameters**.
2. Change connectivity, synaptic, plasticity, or visualization settings as
   needed.
3. Select **Apply Changes & Reset Sim**. Configuration changes do not alter an
   already initialized network until they are applied.
4. Use **Start**, **Pause**, **Stop**, or **Step (1ms)** under **Simulation
   Controls**.
5. Inspect the spike raster, population firing rate, weight distribution, neuron
   inspector, and 3D activity view.

Begin with a small network and a fixed seed. Increase the network only after the
configuration behaves as expected and GPU memory use is known.

### Profiles, Checkpoints, And Recordings

The **File** menu provides three distinct forms of persistence:

| Item | Purpose | Default location |
|---|---|---|
| Profile (`.json`) | Reusable configuration; does not contain learned state | `simulation_profiles/` |
| Checkpoint (`.simstate.h5`) | Simulation state that can be resumed or inspected later | `simulation_checkpoints_h5/` |
| Recording (`.simrec.h5`) | Time-series playback data | `simulation_recordings_h5/` |

Use **Save Profile** for settings, **Save Checkpoint** for a particular brain
state, and **Record** followed by **Playback Recording** for visual review. A
profile is not a trained brain and a recording is not a resumable checkpoint.

The GUI exposes many parameter combinations. Their presence in the interface is
not evidence that every combination has been scientifically validated. Start
from a supplied profile or a focused experiment when testing a biological claim.

## CPU Verification

CPU users can verify the portable backend without installing CuPy:

```bash
SIM_BACKEND=numpy python - <<'PY'
from sim.backend import get_backend, get_sparse_module

print(get_backend()[1])
print(get_sparse_module().__name__)
PY
```

Run a focused simulation error-handling test:

```bash
SIM_BACKEND=numpy python -m pytest tests/test_strict_step_errors.py -q
```

Research modules vary in backend support. Before running one on CPU, inspect its
module header and imports. A direct `import cupy` or a documented GPU requirement
means it is not a CPU workflow even if the core simulator has a NumPy backend.

## Headless Experiment Runner

The maintained root headless launcher currently uses CuPy directly. A verified
small GPU run is:

```bash
SIM_BACKEND=cupy python run_experiment_headless.py \
  --experiment stimulus-response \
  --num-neurons 1000 \
  --seed 42 \
  --output /tmp/stimulus-response.json
```

Available presets are shown by:

```bash
SIM_BACKEND=cupy python run_experiment_headless.py --help
```

The output is an experiment log, not a general claim that the simulated brain
has learned a human-like faculty. Repeat important experiments across seeds and
add controls before interpreting them.

`run_parameter_sweep.py` and `run_benchmarks.py` are also research tools. Read
their `--help` output and the experiment source before use; their metrics and
presets answer specific questions rather than providing a universal quality
score.

## Research Dashboard

Install and start the local dashboard:

```bash
python -m pip install -r webapp/requirements.txt
SIM_BACKEND=numpy python -m uvicorn webapp.server:app --port 8765
```

Open `http://127.0.0.1:8765` in a browser. The dashboard can inspect findings,
completed artifacts, active runs, saved bridges, and selected experiment views.
Some panels can launch research runners.

The dashboard is a local research interface, not a hardened multi-user service.
Use `SIM_BACKEND=cupy` when a launched runner needs the GPU. A dashboard page or
button can expose an experimental path without establishing that the underlying
capability is complete.

## Research Runners

Most experimental programs live in `research/runners/`. They are specialist
tools, not a stable command-line application. A runner may require a particular
backend, checkpoint, corpus, optional dependency, or long runtime.

Before using a runner:

1. Read its module-level description and command example.
2. Check its imports and backend setting.
3. Check the latest related report in `research/findings/`, including its status.
4. Start with one fixed-seed smoke run and a disposable output path.
5. Record the exact command, commit, backend, device, seed, and output artifact.
6. Use multiple seeds and causal controls before making a capability claim.

Some conversation demonstrations use fixed parsers, decoders, hand-designed
codes, conventional language training, or other temporary components. They are
valuable experiments but should not be read as natural open-ended conversation.

## Tests

Install the development dependencies, then run a focused file:

```bash
python -m pip install -r requirements-dev.txt
SIM_BACKEND=numpy python -m pytest tests/<test_file>.py -q
```

Use `SIM_BACKEND=cupy` for tests marked or documented as GPU-specific. The full
suite is large and includes optional research dependencies, so start with tests
covering the changed or used path and broaden deliberately.

## Output And Evidence

- GUI profiles, checkpoints, and recordings use the directories listed above.
- Root headless tools write to the path supplied with `--output` or to a
  timestamped file in the repository root.
- Banked research artifacts live under `research/findings/raw/` with provenance
  information.
- Dated interpretations live under `research/findings/` and may later be
  corrected, superseded, or retracted.

For exact measurements, trust the raw artifact and its provenance. For current
capability status, trust [Current State](docs/CURRENT-STATE.md), not an old demo
name or screenshot.

## Troubleshooting

**The backend is not the one expected.** Set `SIM_BACKEND` explicitly and rerun
the backend probe from the quickstart. Backend selection happens during import,
so change it in a fresh Python process.

**CuPy cannot see the GPU.** Confirm `nvidia-smi` works, verify that only one CuPy
package is installed, and install a wheel compatible with the CUDA environment.

**A CPU command imports CuPy.** That entry point is not backend-portable yet. Use
a backend-compatible module or an NVIDIA environment; setting
`SIM_BACKEND=numpy` cannot override a direct `import cupy`.

**The GPU runs out of memory.** Reduce neuron count, connectivity, recording
detail, or experiment size. Restart the Python process after a failed large run
if the memory pool remains allocated.

**The desktop opens without 3D visualization.** Check the OpenGL packages and
display drivers. The control window can run without the optional OpenGL view on
some systems.

**A result conflicts with a document.** Follow the authority order in the
[Documentation Index](docs/INDEX.md): raw artifacts first for measurements, then
the latest non-retracted finding, current-state document, and roadmap.
