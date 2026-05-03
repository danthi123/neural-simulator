# Current State

A comprehensive snapshot of what the neural simulator can do today,
how it does it, and what's known about its limitations.

This document is the **authoritative current-state reference**.
Update it whenever capabilities change. For the journey of how we
got here, see `research/findings/`.

**Last meaningful update:** 2026-05-02

---

## At a glance

The simulator is a GPU-accelerated spiking neural network with:
- ~5,000 neurons in 50+ brain regions
- ~175,000 synapses with multiple plasticity rules (STDP, STP, Hebbian, homeostasis, reward-modulated)
- Real-time 3D visualization
- Biology-grounded models: Izhikevich-2007, Hodgkin-Huxley, AdEx
- ~375 biological mechanisms catalogued from Kandel + 12 specialty texts

The agent currently solves two main tasks:

1. **Gridworld navigation** (the "main task") — find a goal on a 8×8
   to 24×24 grid using only retinal input
2. **Word-to-action mapping** (the "language task") — hear "north",
   move motor cortex toward north

---

## Validated capabilities

### Navigation

**Best result:** 16×16 grid, 1800-step session, **2.87 ± 0.19 mean
Manhattan distance to goal** across 6 seeds. Agent spends ~38% of
its time at the goal. Beats the 8×8 baseline (4.08 ± 0.49) on a 4×
larger grid.

**How it works (in plain language):** the agent's "eyes" (32×32
retina) see the gridworld image. Visual cortex extracts edges (V1),
combines them into shapes (V2), then identifies what's where (IT).
This information flows to multiple cortex pools competing for action
selection. The basal ganglia picks one — say "go north" — and the
motor cortex fires accordingly. If the move reduced distance to the
goal, dopamine reinforces those particular neuron firings. Over time,
the right vision-to-action mappings get strengthened.

**Configuration:**

```bash
python -m research.runners.g11_bg_runner --moving-goal --goal-schedule multi --deterministic \
    --enable-msn-lateral-inhibition --enable-d1-d2-asymmetry \
    --enable-striatal-pv-fsi --enable-cluster-a-closed-loop \
    --enable-cluster-e-topography --enable-dlpfc-wm --enable-pfc-nmda \
    --enable-visual-cortex --visual-cortex-action-warmup-steps 600 \
    --grid-size 16 --seed 42 --n-steps 1800
```

**No shortcuts:** agent does NOT have direct access to (agent_x,
agent_y), (goal_x, goal_y), or a hand-coded heuristic. Everything
must come through the visual pathway.

### Word-to-action mapping

**Best result:** **28.5% accuracy across 6 seeds, n=600 trials,
p=0.027** vs 25% chance baseline.

**How it works (in plain language):** the agent's "ears" hear a word
(language_input region with sparse pattern coding). The word activates
specific cortex pools via the language→cortex pathway, AND directly
activates motor cortex via a "PFC bypass" pathway that mimics the
Wernicke→arcuate fasciculus→Broca→M1 anatomy in real human brains.
After 100 episodes of training where the agent navigates while
hearing the corresponding direction word, motor cortex starts to
fire preferentially for the correct direction when given a fresh
test word.

**Configuration:**

```bash
python -m research.runners.text_eval_embodied \
    --n-episodes 100 --steps-per-episode 30 --seed 42 \
    --stim-steps-per-step 200 --reset-steps 100 \
    --out-stats results.json
```

**Critical fixes** (commit 144eefd + 200f73c):
- `cfg.enable_hebbian_learning = False` — Hebbian global decay
  (1e-5/sub-step over ~990K sub-steps) was collapsing all weights to
  the floor (0.05) and erasing learning. This isn't a "feature off",
  it's a bug fix matching what every other research runner already does.
- `cfg.stdp_w_max = 5.0` — STDP soft-bound was clipping
  language→motor design weights (3.0) at 2.0, eliminating headroom.
- Non-zero readout pathway init (0.5 ± 0.3) — pathways from cortex
  to language_output were initialized at zero; STDP couldn't grow
  them from scratch with weak training signal.

**Known weakness:** the inverse direction (image → word readout) is
unreliable. Per-seed range 21-33%, cumulative ~25% (chance). The
multi-step image→V1→V2→IT→language_output pathway has too many
plastic stages to consolidate cleanly.

### What 9 architectural variations did NOT improve

Tested on top of the v2 baseline at seed=42, none beat 28.5%:

| Variation | I→W | W→A | Verdict |
|---|---|---|---|
| Reward shaping (no LTD penalty) | 33% | 25% | Negative |
| Stronger drives (lang_in 200→400) | 33% | 25% | Negative |
| Eval-time drive 500 reeval | 25% | 24% | Negative |
| Bigger motor pools (10→30 neurons) | 24% | 24% | Negative |
| Longer training (100→200 ep) | 22% | 24% | Negative |
| Bigger language regions (256→512) | 25% | 18% | Negative |
| Curriculum (visuomotor first 200 ep) | 24% | 23% | Negative |
| Alternative decoders (4 variants) | 33% | 27% (delta) | Negative |
| Motor cross-coupling (90° adj) | 29% | 22% | Negative |

The most diagnostic finding came from the curriculum experiment:
even when Phase 2 cascade reached 43% correct moves (vs. v2's 30%),
the language pathway weights were *identical to v2 to 3 decimal places*.
This shows the language pathway weights converge to a steady state
determined by cascade STRUCTURE and STDP parameters, not by cascade
ACCURACY. The 28.5% is a true architectural ceiling, not a tuning issue.

### Working memory

PFC holds goal location for ~10 seconds in NMDA-bistable activity
patterns. Damage to PFC region (set NMDA conductance to zero) eliminates
this and the agent can no longer hold goals in mind across delays.

### Real-time interactive control

Click anywhere in the gridworld during a live run to teleport the goal
to that position. The agent will reorient and approach. Works at
any grid size, any moving-goal cadence.

---

## How the architecture is organized

The brain has 50+ regions wired in biology-grounded pathways. Major
regions:

### Sensory front-end
- **retina** (2,048 neurons) — 32×32 ON-channel + 32×32 OFF-channel
  representing the gridworld image
- **cortex_v1_simple** (1,024) — Gabor-tuned simple cells (Hubel-Wiesel)
- **cortex_v1_complex** (512) — phase-pooled complex cells
- **cortex_v2** (256) — feature integration
- **cortex_it** (64) — high-level object/scene representation

### Action selection
- **cortex_{N,E,S,W}** (4 pools, 25 each) — premotor cortex,
  one per cardinal direction
- **str_D1/D2_{N,E,S,W}** (8 pools, 50 each) — direct/indirect
  pathway striatum
- **gpe_{N,E,S,W}**, **stn**, **gpi_{N,E,S,W}** — basal ganglia loops
- **thal_{N,E,S,W}** (4 pools, 10 each) — thalamic relay
- **motor_{N,E,S,W}** (4 pools, 10 each) — primary motor cortex output
- **snc** (10) — substantia nigra dopamine cells

### Working memory + language
- **dlpfc_wm** (60 neurons) — dorsolateral PFC working memory
- **language_input** (256) — Wernicke-like, hears words
- **language_output** (256) — Broca-like, produces words
- Pathways: language_input → dlpfc_wm, → cortex_X, → motor_X (PFC bypass);
  cortex_X → language_output, IT → language_output

### Memory system (optional, currently not integrated with text I/O)
- **dg, ca3, ca1** — hippocampal trisynaptic loop with sharp-wave-ripple
  consolidation infrastructure

---

## Plasticity rules

The brain learns through several mechanisms running concurrently:

| Mechanism | What it does | Real biology |
|---|---|---|
| **STDP** | Strengthens connections when pre-spike precedes post-spike by 1-20 ms; weakens otherwise | Bi & Poo 1998 |
| **Reward modulation** | Three-factor learning: STDP × eligibility × reward | Schultz 1998 dopamine RPE |
| **Short-term plasticity (STP)** | Synapses depress with repeated firing, recover slowly | Tsodyks & Markram 1997 |
| **Homeostasis** | Adjusts firing thresholds to keep average rate near target (~2 Hz) | Turrigiano 1999 |
| **Hebbian (currently disabled in research runs)** | "Neurons that fire together wire together" | Hebb 1949 |

**Note:** Hebbian is disabled in all research runners because the
default global weight-decay constant (1e-5/sub-step) catastrophically
erodes weights over hundreds of thousands of simulation steps. STDP
+ reward modulation provide the actual learning signal.

---

## Performance

### Throughput

- ~6 ms per simulation sub-step on RTX 3090
- 100-episode text training run: ~50 minutes (with stim_steps=200)
- 1800-step navigation run: ~15 minutes

### Memory

- ~5K neurons + 175K synapses uses ~1.3 GB GPU memory
- Scales linearly to 100K+ neurons (network of 50K well within RTX 3090 limits)

### Reproducibility

All random sources seeded together. Same `--seed` on same hardware
produces bit-identical trajectories. Use `--deterministic` flag for
maximum reproducibility (sets `CUBLAS_WORKSPACE_CONFIG`).

---

## Limitations

### Computational
- 4-direction action space only (no diagonal moves)
- Single goal at a time (no multi-goal compositional planning)
- 8×8 to 24×24 grid sizes only
- ~5K neurons (real brain regions have 10⁴–10⁶ each)

### Biological
- No developmental phases (no synaptic pruning, layer formation)
- No protein-synthesis-dependent late-LTP for long-term consolidation
- No spatial conduction delays beyond fixed per-pathway values
- No neurovascular coupling, no glia
- No multi-time-scale plasticity (only fast STDP)

### Tasks
- Vocabulary: 4 cardinal directions
- No compositional language ("go north then east")
- No multi-modal tasks (smell, touch, sound modalities not modeled)
- No social interaction (joint attention not implemented)

---

## Active research directions

Current open experiments (as of 2026-05-02):

1. **Distributed motor pool architecture (Pulvermüller G.20)**
   — running. 8 motor sub-pools at 45° intervals with cosine-tuned
   thal pathways and population vector decoding. Tests whether
   labeled-line motor pools are the bottleneck for the 28.5% W→A
   ceiling.

2. **Sharp-wave-ripple consolidation (Wilson-McNaughton 1994)**
   — implementation ready. Replays recent (token, action) tuples
   during sleep windows. Composes with both labeled-line and
   distributed-pool architectures.

Detailed roadmap: `docs/plans/2026-05-02-text-io-next-directions-biology-grounded.md`.

---

## Dependencies

- Python 3.8+
- CuPy (CUDA 11+ or 12+)
- NumPy, h5py, dearpygui, PyOpenGL
- (Optional) FastAPI + uvicorn for the webapp

NVIDIA GPU with 6GB+ VRAM recommended. Can run on smaller GPUs by
reducing network sizes.

---

## Where to go from here

| If you want to... | Read this |
|---|---|
| Run the simulator | [QUICKSTART.md](../QUICKSTART.md) |
| Understand the biology in plain language | [biology.md](biology.md) |
| Modify or extend the codebase | [../CONTRIBUTING.md](../CONTRIBUTING.md) |
| Reproduce a specific result | [USER_GUIDE.md](../USER_GUIDE.md) |
| Read scientific findings chronologically | [`research/findings/`](../research/findings/) |
| See the full biology catalog | [`references/feature-catalog.md`](../references/feature-catalog.md) (catalog-build branch) |
| Understand AI agent guidelines | [CLAUDE.md](../CLAUDE.md) |
