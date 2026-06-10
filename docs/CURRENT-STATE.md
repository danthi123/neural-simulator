# Current State

A comprehensive snapshot of what the neural simulator can do today,
how it does it, and what's known about its limitations.

This document is the **authoritative current-state reference**.
Update it whenever capabilities change. For the journey of how we
got here, see `research/findings/`.

**Last meaningful update:** 2026-06-10 (see "Recent milestones (2026-06)" below).

**Recent milestones (2026-06):**
- **The conversational "composer" is now fully spiking** — the part that binds
  words into facts uses phase-based spiking neurons (a resonate-and-fire model
  with complex-valued synapses), which sidesteps a noise barrier that the
  earlier firing-rate version hit. It handles who/what question answering,
  refusing to answer when it doesn't know, negation and yes/no, embedded
  clauses, dialogue, and sentence generation; correct at the 320-concept scale.
- **The whole conversational pipeline runs on the core simulator** —
  comprehension (a learned sentence parser), fact memory + question answering,
  and dialogue planning all run as genuine spiking neurons on the core network,
  with no separate maths-only module bolted on.
- **Navigation is fully biology-based** — every step between seeing and acting
  is done by simulated neurons (a spiking superior colliculus for orienting, a
  neural reward signal, and a spiking basal-ganglia decision and dopamine
  system), with no hand-coded shortcut in between.
- **Navigation and conversation now share one network (roadmap step 2)** — each
  brain is its own group of neurons on a single network. The conversational
  behaviour works unchanged on the shared network (including its refusal to make
  up answers), and navigation runs on it while the conversational neurons stay
  exactly unchanged during navigation's live learning. See
  [`ARCHITECTURE_nav_conv_merge.md`](ARCHITECTURE_nav_conv_merge.md). (A
  six-seed confirmation that the navigation score is statistically unchanged is
  the final check, currently running.)

> The "At a glance" and detailed sections below predate the 2026-06 work and
> describe the earlier multi-tag-retrieval era; the milestones above are the
> current frontier. A full rewrite of the detailed sections is pending.

**Earlier (2026-05-14):** concept-concept semantic conversation validated at 90%
FULL / 100% PARTIAL multi-seed via multi-tag cue retrieval; bug retraction for
prior compose_concept claims documented.

---

## At a glance

The simulator is a GPU-accelerated spiking neural network with:
- ~5,000 neurons in 50+ brain regions
- ~175,000 synapses with multiple plasticity rules (STDP, STP, Hebbian, homeostasis, reward-modulated)
- Real-time 3D visualization
- Biology-grounded models: Izhikevich-2007, Hodgkin-Huxley, AdEx
- ~375 biological mechanisms catalogued from Kandel + 12 specialty texts

The agent currently solves three main tasks:

1. **Gridworld navigation** (the "main task") — find a goal on a 8×8
   to 24×24 grid using only retinal input
2. **Word-to-action mapping** (the "language task") — hear "north",
   move motor cortex toward north
3. **Concept-concept semantic conversation** (2026-05-14, NEW) —
   user types "apple", system retrieves "big" AND "cat" via
   multi-tag engram cue retrieval. 90% FULL / 100% PARTIAL multi-seed
   at 16-word vocab. Chat REPL with `remember`/`what is`/`forget`/`and`
   commands. See [`research/findings/2026-05-14-multitag-cue-retrieval-90pct-VALIDATED.md`](../research/findings/2026-05-14-multitag-cue-retrieval-90pct-VALIDATED.md).

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

> **🚨 CRITICAL CAVEAT (2026-05-03 evening, autonomous overnight):
> Permuted-label control test shows the 28.5% is NOT real word-action
> learning.** Across 45+ runs spanning baseline / v2+SWR / v2+SWR-balanced
> (H1) / H4 PFC isolation / fundamentals sweep (heb_only, drive_5x,
> stdp_wmax_10, heb_drive, heb_stdp, drive_stdp) / H4 dose-1000 — **0/45
> had the TRUE labeled mapping as the BEST of 24 permutations.** The
> binomial p=0.027 measures whether the network has ANY structure above
> chance, not whether that structure aligns with task labels. Best
> permutations score 30-37% (8pp above chance) but are seed-dependent
> and arbitrary, not aligned with N/E/S/W. See
> [`research/findings/2026-05-03-architecture-fundamentally-cant-align.md`](../research/findings/2026-05-03-architecture-fundamentally-cant-align.md)
> for the full analysis. **A biology-grounded sweep (topographic prior
> 1.5/0.7 + motor PV-FS lateral inhibition) is in flight to test if
> biology fixes can break the 0/N alignment streak.**

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
- **6-seed minimal-arch batch:** ~45-55 minutes (was ~6 hours pre-2026-05-03)

### Speedup stack (shipped 2026-05-03 through 2026-05-05)

Layered optimizations across multiple shipping waves:

| Layer | Date | Speedup | Mechanism |
|---|---|---|---|
| dt = 0.5 → 1.0 ms | 2026-05-03 | ~2x | substep count halved, dynamics still stable |
| Parallel-3 GPU sharing | 2026-05-03 | ~1.7x | 3 procs at ~70% efficiency each |
| `fast_spike_reset` (cp.where masked-update) | 2026-05-03 | 1.29x | eliminates per-step GPU-CPU sync in spike-reset |
| Three-factor GPU-port (Phase 1) | 2026-05-05 | ~2x (3-factor only) | eliminates per-event 6 MB CSR round-trip |
| **`cfg.fp16_synapse_state`** | 2026-05-05 | 1.05-1.15x | fp16 storage for `cp_eligibility_trace` (validated <1mV drift) |
| **Parallel=6** (use VRAM headroom) | 2026-05-05 | ~3x sweep throughput | 30-50% GPU util at parallel=2 → bumped per user observation |

Numerical equivalence + drift verified at:
- `tests/test_fast_spike_reset.py` (6 tests, dt path)
- `tests/test_three_factor_update.py` (7 tests, GPU-port logic)
- `tests/test_fp16_drift.py` (4 tests, voltage drift <1mV over 1000 steps)

Default flags (all opt-in for backward compat):
- `cfg.fast_spike_reset = False` (default-on in modern runners)
- `cfg.fp16_synapse_state = False` (validated, ship default-on after sweep)
- `gpu_eligibility = True` in three_factor runner

Honest perf-roadmap: `research/findings/2026-05-05-perf-roadmap.md`.
Cloud H100 deploy: `docs/plans/2026-05-05-cloud-h100-deployment.md`
(~$2/hr for 6-8× sweep throughput, ready for activation).

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

Current open experiments (as of 2026-05-04 ~00:30 EDT autonomous):

1. **Biology-grounded sweep on minimal isolation arch** — IN FLIGHT.
   The 0/N alignment finding (permuted-label control 2026-05-03) is
   architectural, not a tuning issue. Testing whether two
   biology-grounded fixes break the streak: topographic Wernicke→motor
   prior (1.5/0.7 ratio per Pulvermüller 2001-2003) + cortical
   PV-FSI lateral inhibition (~12% of motor pool per Vogels 2011).
   4 conditions (baseline / +FS only / +Topo only / +Topo+FS) ×
   6 seeds = 24 runs in parallel-3 (~3 hours). Auto-launches when
   minimal-iso batch 2 finishes. See
   `experiments/biology_sweep.yaml` and
   `research/findings/2026-05-04-biology-sweep-followup-plan.md`.

2. **Minimal-isolation INVERSION finding (2026-05-03 evening)**
   The cascade-as-cause hypothesis is FALSIFIED. With cascade
   stripped (no cluster_a, no cluster_e, no PFC, no visuomotor —
   just language_input → motor_X under paired-stim training),
   alignment ratio is 0/3 not 4/6. The cascade was a weak DAMPENER
   on seed-dependent random structure, not its source. See
   `research/findings/2026-05-04-minimal-isolation-INVERSION.md`.

3. **Pre-staged A/B follow-up decision chain.** A PowerShell
   waiter polls for biology-sweep completion, runs the result
   aggregator, and auto-launches the appropriate next experiment:
   - Outcome A (≥ 4/6 aligned): `minimum_biology.yaml` —
     dose-response (weak topo, minimal FS, strong topo, combo half)
   - Outcome B (0-1/6): `eval_sanity_check.py` — hand-built
     PERFECT weights, tests if the eval methodology itself works
   - Tier-2 fallbacks pre-staged: `b2_sparse_codes.yaml` +
     `b4_long_training.yaml` (sparse-code overlap + dose-response)

4. **Distributed motor pool architecture (Pulvermüller G.20)**
   — implemented. 8 motor sub-pools at 45° intervals with
   cosine-tuned thal pathways and population vector decoding. Tested
   2026-05-02 at n=1, didn't beat the (now-debunked) 28.5% baseline.
   No clear path forward without first establishing real W→A.

### New tooling shipped 2026-05-03

- `sim/progress.py` — universal `[PROGRESS] {json}` event format.
  All runners emit structured progress (kind, current, total, phase,
  elapsed_seconds). Webapp parses for live progress display.
- `research/experiment_runner.py` — YAML-driven experiment orchestrator
  replacing per-experiment PowerShell scripts. Parallel-N batching,
  master log + COMPLETE marker, auto pid management.
- `research/result_aggregator.py` — parameterized cross-condition
  result rollup with permuted-label aligned ratio + verdict line.
  Built-in configs: swr-investigation, fundamentals, biology,
  minimum_biology, sanity_check, b2_sparse_codes, b4_long_training.
- `research/runners/profile_step.py --arch {v2|minimal}` — section
  profiler for hot-path identification.
- `research/runners/bench_parallel_gpu.py` — parallel-process GPU
  contention benchmark.

Detailed roadmap:
* `docs/plans/2026-05-03-autonomous-overnight-plan.md`
* `research/findings/2026-05-04-biology-sweep-followup-plan.md`
* `research/findings/2026-05-04-perf-speedup-stack.md`

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
