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

- **Python 3.8+**
- **NVIDIA GPU** with CUDA support (RTX 3090 ideal; runs on smaller GPUs)
- **CuPy** matching your CUDA version (`pip install cupy-cuda12x` or
  `cupy-cuda11x`)
- ~6 GB GPU memory for default configurations

If you don't have CUDA, you can browse the code and read findings docs,
but the simulator itself needs GPU.

## Three things to try

### 1. The flagship navigation experiment

Watch the agent learn to navigate using only its simulated retina (no
shortcuts, no direct goal coordinates):

```bash
python -m research.runners.g11_bg_runner --moving-goal --goal-schedule multi --deterministic \
    --enable-msn-lateral-inhibition --enable-d1-d2-asymmetry \
    --enable-striatal-pv-fsi --enable-cluster-a-closed-loop \
    --enable-cluster-e-topography --enable-dlpfc-wm --enable-pfc-nmda \
    --enable-visual-cortex --visual-cortex-action-warmup-steps 600 \
    --grid-size 16 --seed 42 --n-steps 1800 \
    --out research/findings/raw/g11_bg/quickstart_navigation.json
```

Takes ~10 min on RTX 3090. The agent should reach mean Manhattan
distance ~3 to goal by the end (38% of timesteps spent at goal).

### 2. The text-to-action experiment

Test the statistically validated word→action capability (28.5% W→A,
p=0.027 across 6 seeds):

```bash
python -m research.runners.text_eval_embodied \
    --n-episodes 100 --steps-per-episode 30 --seed 42 \
    --stim-steps-per-step 200 --reset-steps 100 \
    --out-stats research/findings/raw/g11_bg/quickstart_textio.json
```

Takes ~50 min. Result printed at end: I→W and W→A accuracy with
per-direction breakdown.

To reproduce the full 6-seed validation:

```bash
for seed in 42 43 44 100 101 102; do
    python -m research.runners.text_eval_embodied \
        --n-episodes 100 --steps-per-episode 30 --seed $seed \
        --stim-steps-per-step 200 --reset-steps 100 \
        --out-stats research/findings/raw/g11_bg/quickstart_seed${seed}.json
done

python -m research.runners.text_io_meta_analysis
```

### 3. Interactive control via the webapp

```bash
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
