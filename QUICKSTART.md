# Quick Start — Neural Simulator in 60 Seconds

> **Goal:** get you to a running simulation as fast as possible. Total time: ~60 seconds after you have CUDA + Python.

## TL;DR

```bash
# 1. Install
pip install -r requirements.txt

# 2. Run the GUI (recommended for first-timers)
python neural-simulator.py

# 3. Or run the flagship research experiment headlessly:
python -m research.runners.g11_bg_runner --moving-goal \
    --hippocampus --learned-perception --pfc \
    --beacon-perception --beacon-replaces-goal \
    --cue-reflex --cue-reflex-replaces-heuristic \
    --enable-landmark-sensor --landmarks-replace-place \
    --sensed-reward \
    --enable-msn-lateral-inhibition \
    --adaptive-da --adaptive-da-ema-decay-negative 0.7 \
    --curriculum --curriculum-warmup-steps 600 \
    --seed 42 --n-steps 1800
```

## What you just ran

Option 3 above runs our **biology-grounded learning agent** — a spiking
neural network that learns to navigate a 2D gridworld using only:
- **Goal-beacon perception** (8 directional sensors detect a beacon — replaces direct goal coords)
- **Landmark perception** (8 directional sensors to a fixed cue — replaces direct (x,y))
- **Hippocampal place + goal cells** (self-organize from sensors, not coordinates)
- **Prefrontal cortex** (recurrent working memory for goal context)
- **Innate cue-following reflex** (direction-only, like phototaxis in real animals — replaces hand-coded heuristic)
- **Sensed reward** (beacon-intensity gradient — replaces ground-truth distance reward)
- **Curriculum learning** (cortex matures, then input layers train via per-pathway plasticity gates)

It's the result of weeks of architectural work to close 4 of 5 "magic GPS"
cheats real animals don't have. **6/6 seeds beat baseline by 30.6%
with this config (p=0.00045)** — biology-grounded actually beats the
cheats-allowed version (4.08 vs 4.41 sum).

## Three things to try next

### A. I want to see it learn (visual)

```bash
python neural-simulator.py
```

Then in the GUI:
1. Open the **Experiment & Stimulus System** panel
2. Choose preset: "Pavlovian Conditioning" or "RL"
3. Click **Start**
4. Watch neurons fire in 3D in real-time

→ See **[USER_GUIDE.md](USER_GUIDE.md)** for the full GUI walkthrough.

### B. I'm a researcher / I want to read findings

→ **[research/findings/INDEX.md](research/findings/INDEX.md)** — all results
chronologically, including negatives.

The most exciting recent ones:
- 🎉 [Item 1: Full Perception Arc Complete](research/findings/2026-04-27-FULL-PERCEPTION-ARC-COMPLETE.md)
  (4.56 sum, p=0.00819) — agent navigates without coordinate cheats
- 🎉 [PFC Working Memory](research/findings/2026-04-27-pfc-working-memory.md)
  (4.41 sum, p=0.018) — recurrent prefrontal region adds value
- 🎉 [Plastic-Input-Layer Arc Resolved](research/findings/2026-04-27-plastic-input-layer-RESOLVED.md)
  (4.72 sum, p=0.02) — closed a 7-NEGATIVE architectural arc

### C. I'm a developer / I want to build on this

```bash
# Run the test suite
pytest tests/ -v

# Read the architecture
cat CLAUDE.md  # or open in your editor
```

Key files:
- **[sim/bridge.py](sim/bridge.py)** — main simulation engine (`SimulationBridge` class)
- **[sim/regions.py](sim/regions.py)** — declarative brain region framework
- **[sim/neuromodulators.py](sim/neuromodulators.py)** — DA/NE/ACh subsystem
- **[research/runners/g11_bg_runner.py](research/runners/g11_bg_runner.py)** — flagship runner with all opt-in flags

→ See **[CONTRIBUTING.md](CONTRIBUTING.md)** for dev setup, code style, PR template.

## Requirements

- **Python** 3.8+
- **NVIDIA GPU** with CUDA support
- **CuPy** (matched to your CUDA version)
- ~2 GB GPU memory minimum (10K neurons), 20+ GB for 100K+ networks

See **[README.md](README.md)** for full installation details.

## Troubleshooting

If something doesn't work:
1. **GPU not detected?** → check `nvidia-smi` shows your card
2. **Import errors?** → `pip install -r requirements.txt --upgrade`
3. **g11 runner fails?** → check **[research/runners/TROUBLESHOOTING.md](research/runners/TROUBLESHOOTING.md)** for known gotchas
4. **Tests fail?** → most tests need GPU; if GPU works but tests don't, file an issue

## Next steps

| If you want to... | Go to... |
|---|---|
| Use the GUI | [USER_GUIDE.md](USER_GUIDE.md) |
| Run experiments programmatically | [README.md#programmable-api](README.md#programmable-api) |
| Read all the research | [research/findings/INDEX.md](research/findings/INDEX.md) |
| See the project's scientific arc | [docs/SCIENCE_ROADMAP.md](docs/SCIENCE_ROADMAP.md) |
| Contribute code | [CONTRIBUTING.md](CONTRIBUTING.md) |
| Understand the architecture | [README.md#system-architecture](README.md#system-architecture) + [CLAUDE.md](CLAUDE.md) |
| Build on the research-runner framework | [research/runners/](research/runners/) + [TROUBLESHOOTING.md](research/runners/TROUBLESHOOTING.md) |
