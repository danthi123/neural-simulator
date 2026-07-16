# Quick Start — Neural Simulator in 60 Seconds

> **Goal:** get you to a running simulation as fast as possible. Total
> time: ~60 seconds after you have CUDA + Python.

## TL;DR

```bash
# 1. Install
pip install -r requirements.txt

# 2. Run the GUI with live 3D visualization
python neural-simulator.py
```

That's it. The GUI loads with a working brain — neurons firing, synapses
pulsing, agent navigating a gridworld. Click anywhere in the gridworld
to teleport the goal; watch the brain reorient.

## Requirements

- **Python 3.10+**
- **NVIDIA GPU** with CUDA support (RTX 3090 ideal; runs on smaller GPUs) —
  optional; the engine also runs on the CPU via NumPy (see below)
- **CuPy** matching your CUDA version (`pip install cupy-cuda12x` or
  `cupy-cuda11x`)
- ~6 GB GPU memory for default configurations

**No NVIDIA GPU?** The engine also runs on the CPU — set one environment
variable and it swaps its GPU array library (CuPy) for NumPy with no code
change. It is slower (roughly 4–50× depending on the workload) but
numerically equivalent, and the headless demos all work:

```bash
SIM_BACKEND=numpy python -m research.runners.chat_demo --seed 43
```

The live 3D GUI (`python neural-simulator.py`) still needs a GPU.

## Four things to try

### 1. The flagship navigation experiment

Watch the agent learn to navigate using only its simulated retina (no
shortcuts, no direct goal coordinates). This is the current best:
A+E + Cluster G v2.5 + Cluster K v2 visual cortex.

```bash
python -m research.runners.g11_bg_runner --moving-goal --goal-schedule multi --deterministic \
    --enable-msn-lateral-inhibition --enable-d1-d2-asymmetry \
    --enable-striatal-pv-fsi --enable-cluster-a-closed-loop \
    --enable-cluster-e-topography --enable-dlpfc-wm --enable-pfc-nmda \
    --enable-visual-cortex --visual-cortex-action-warmup-steps 600 \
    --grid-size 16 --seed 42 --n-steps 1800 \
    --out research/findings/raw/g11_bg/quickstart_navigation.json
```

Takes ~10 min on RTX 3090. The headline metric is `sum_finalQ` — the SUM
over the 4 goal phases of each phase's final-quarter mean Manhattan distance
(`g11_bg_runner.py:8158`), i.e. ~4 terms of ~0.74, NOT a mean distance.
Validated 2.97 ± 0.12 on the 16×16 grid, ~38% of timesteps at goal. Do not
compare it against `mean_distance_overall` (also printed) — they differ ~4×.
The move decision is made in spikes by default — the action emerges from a
race between competing neural populations, not an off-brain shortcut. (Use
`--readout-source motor` to swap in the old shortcut as a baseline.)

### 2. The text-to-action experiment (under investigation)

Word→action mapping is the active research question. The previously
claimed "28.5% W→A, p=0.027" result was falsified by a 2026-05-03
permuted-label control test (TRUE labels were never the best of 24
permutations across 45+ runs). The runner still works for current
biology-grounded experiments:

```bash
python -m research.runners.text_eval_embodied \
    --n-episodes 100 --steps-per-episode 30 --seed 42 \
    --stim-steps-per-step 200 --reset-steps 100 \
    --out-stats research/findings/raw/g11_bg/quickstart_textio.json
```

To check whether output reflects real label-aligned learning vs random
above-chance structure, also run the permuted-label check:

```bash
python -m research.runners.permuted_label_check \
    research/findings/raw/g11_bg/quickstart_textio.json
```

See [`research/findings/2026-05-03-permuted-label-control-NEGATIVE.md`](research/findings/2026-05-03-permuted-label-control-NEGATIVE.md)
for the full story.

### 3. Autonomous overnight experiment runs

Queue YAML-driven sweeps and let them run unattended:

```bash
# Run a multi-condition sweep (e.g. biology fixes vs anti-cheat controls)
python -m research.experiment_runner experiments/biology_sweep.yaml

# Aggregate results across conditions and seeds with a verdict line
python -m research.result_aggregator biology_sweep

# Morning summary of any overnight run
python -m research.runners.morning_briefing --short
```

Built-in YAMLs in `experiments/`: `biology_sweep`, `minimum_biology`
(dose-response), `eval_sanity_check` (eval-methodology validator via
hand-built perfect weights), `b2_sparse_codes`, `b4_long_training`.

### 4. Interactive control via the webapp

```bash
# Install webapp deps (if not already)
pip install -r webapp/requirements.txt

# Start the webapp (separate terminal)
uvicorn webapp.server:app --host 127.0.0.1 --port 8765

# Open browser to http://127.0.0.1:8765
```

Browse runs, launch experiments, view 3D visualization, monitor
in-flight training jobs.

## What you just ran (in plain language)

The flagship experiment trains a spiking neural network with:

- **A retina** (32×32 ON + 32×32 OFF cells) — sees the gridworld image
- **Visual cortex (V1, V2, IT)** — extracts edges, shapes, object identities
  from the retinal image
- **Premotor cortex** (4 pools, one per direction) — competes for action
- **Basal ganglia** (D1/D2 striatum, GPe, GPi, thalamus) — selects one
  action from the competing options
- **Motor cortex** (4 pools) — fires the chosen action
- **Prefrontal cortex** (working memory) — holds the goal across delays
- **Dopamine midbrain** — reinforces successful actions

The agent learns from **dopamine-modulated spike-timing plasticity** only.
No backpropagation. No symbolic optimization. Same learning rules real
brains use (Schultz 1998, Bi & Poo 1998).

After 1800 simulation steps (~10 min wall time on RTX 3090), the agent
goes from random behavior to consistent goal-seeking — comparable
trajectories to animals on similar tasks.

For the deep technical view, see [docs/CURRENT-STATE.md](docs/CURRENT-STATE.md).
For the biology view, see [docs/biology.md](docs/biology.md).

## Common operations

```bash
# Run all tests
pip install -r requirements-dev.txt   # pytest -- NOT in requirements.txt (the sim runs without it)
pytest tests/ -v

# Run a smoke test of any runner
python -m research.runners.g11_bg_runner --n-steps 200 --seed 42

# Analyze a result file
python -m research.runners.text_eval_analyze <result.json>

# Cross-checkpoint weight comparison
python -m research.runners.text_weight_compare \
    label1:diag1.json label2:diag2.json

# Aggregate text I/O experiments meta-analysis
python -m research.runners.text_io_meta_analysis
```

## Troubleshooting

**"CUDA out of memory"** — Reduce `--n-steps` or use smaller grid
(`--grid-size 8`). Default config uses ~1.3 GB.

**"No CuPy available"** — Install matching version: `pip install
cupy-cuda12x` (for CUDA 12) or `pip install cupy-cuda11x` (for CUDA 11).

**Visualization runs slowly** — Disable visualization for headless
runs (no `python neural-simulator.py`, just runners).

**Window doesn't open on Linux** — Need OpenGL drivers. On WSL2,
use `pyopengl` with `LIBGL_ALWAYS_SOFTWARE=1` for software rendering.

## Where to next

- See **[README.md](README.md)** for project overview
- See **[docs/CURRENT-STATE.md](docs/CURRENT-STATE.md)** for what works today
- See **[docs/biology.md](docs/biology.md)** for the neuroscience
- See **[USER_GUIDE.md](USER_GUIDE.md)** for detailed configuration options
- See **[CONTRIBUTING.md](CONTRIBUTING.md)** to extend the codebase
