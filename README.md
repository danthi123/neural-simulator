# Neural Simulator

**A brain-inspired AI that learns by spike timing, not gradient descent.**

This is a research project building a virtual brain that learns from
experience the way real brains do — through individual neurons firing
together, dopamine reinforcing successful actions, and connections
strengthening or weakening based on millisecond-precise timing.

No backpropagation. No supervised training labels. No symbolic
optimization. Just real neurons firing in real time, learning from
reward and biological plasticity rules.

```
┌─────────────────────────────────────────────────────────────────┐
│  Agent sees the world → cortex activates → BG selects action    │
│            ↑                                          ↓          │
│      Reward shapes  ←  Move toward goal   ←  Motor cortex fires  │
│      future choices       (or away)                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## What it does today

**Navigates a gridworld** using only what it sees through a simulated
retina. The agent locates a goal, picks a direction, moves, gets a
reward signal, and learns. After 1800 steps on a 16×16 grid, the agent
spends 38% of its time at the goal — well above random behavior.

**Word-action mapping (under investigation).** A 2026-05-03 permuted-label
control falsified the previously-claimed "28.5% W→A" result: across 45+
runs spanning every variant tested, the TRUE labeled mapping was NEVER
the best of 24 permutations of (token → action). The architecture
produces structure above chance, but it's seed-dependent and unaligned
with task labels. A biology-grounded investigation (topographic
Wernicke→motor priors + motor PV-FS lateral inhibition) is the active
research question. See
[`research/findings/2026-05-03-permuted-label-control-NEGATIVE.md`](research/findings/2026-05-03-permuted-label-control-NEGATIVE.md).

**Visualizes its own brain.** Live 3D OpenGL view of every neuron
firing, every synapse pulsing, with click-to-teleport-goal interactive
control. You can watch the brain learn in real time.

---

## Why this matters

Modern AI uses gradient descent — a powerful but biologically implausible
optimization technique. Real brains have no equivalent of backpropagation
through time. They learn from local rules: "neurons that fire together,
wire together" (Hebb 1949), refined by dopamine reward (Schultz 1998).

This project tests how far you can go with **only those biological
rules**. The answer so far: navigation works well, language learning
works modestly. The system has clear limits, and finding those limits
is itself the contribution.

---

## What the brain can do

| Capability | How it works | Status |
|---|---|---|
| **See** | Retina → V1 (Gabor edge detectors) → V2 → IT (object) | ✅ Working |
| **Decide** | Basal ganglia cascade picks one action from competing options | ✅ Working |
| **Move** | Motor cortex pools fire, agent moves on grid | ✅ Working |
| **Learn from reward** | Dopamine modulates spike-timing plasticity | ✅ Working |
| **Hold goals in mind** | Prefrontal cortex working memory (NMDA bistability) | ✅ Working |
| **Understand words** | Language input → motor cortex via Wernicke→Broca pathway | ⚠️ Under investigation — see [permuted-label control finding](research/findings/2026-05-03-permuted-label-control-NEGATIVE.md) |
| **Speak** | Cortex/visual cortex → language output | ⚠️ Partial (high variance) |
| **Remember & dream** | Hippocampus + sharp-wave-ripple replay | ⚠️ Implemented, integration pending |

---

## Try it in 60 seconds

```bash
git clone https://github.com/danthi123/neural-simulator
cd neural-simulator
pip install -r requirements.txt   # CuPy + DearPyGUI + PyOpenGL + h5py + ...
python neural-simulator.py        # GUI mode with live 3D viz
```

Or run the navigation flagship headless (current best on 16×16 perception-only):

```bash
python -m research.runners.g11_bg_runner --moving-goal --goal-schedule multi --deterministic \
    --enable-msn-lateral-inhibition --enable-d1-d2-asymmetry --enable-striatal-pv-fsi \
    --enable-cluster-a-closed-loop --enable-cluster-e-topography \
    --enable-dlpfc-wm --enable-pfc-nmda \
    --enable-visual-cortex --visual-cortex-action-warmup-steps 600 \
    --grid-size 16 --seed 42 --n-steps 1800
```

Or talk to the sim — type a direction word and watch the motor pool
activate (Tier 1 4-word vocab, ~6 min single seed):

```bash
python -m research.runners.chat_demo --seed 43 --train-events 200
```

Or with synonyms — both `north` and `up` activate motor_N (Tier 2.1
8-word vocab, ~10 min single seed):

```bash
python -m research.runners.chat_synonym_demo --seed 42 --train-events 400
```

Or interactive REPL with checkpoint save (train once, reload instantly
in future sessions):

```bash
# First time: train + save (~6 min for tier1, ~10-20 min for synonym)
python -m research.runners.chat_repl --mode tier1 --seed 43 \
    --save-bridge simulation_checkpoints_h5/repl_tier1.simstate.h5

# Subsequent sessions: load saved bridge (~30 sec, REPL ready)
python -m research.runners.chat_repl --mode tier1 --seed 43 \
    --load-bridge simulation_checkpoints_h5/repl_tier1.simstate.h5

# Or 16-word vocab with Unicode arrows:
python -m research.runners.chat_repl --mode synonym16 --seed 42
# Type ↑ or "north" or "up" or "n" → all activate motor_N
```

See [`docs/CHAT-DEMO-GUIDE.md`](docs/CHAT-DEMO-GUIDE.md) for all
conversational demos (Tier 1 / Tier 2.1 synonym / Phase 1.4 continual
learning).

Full setup in [QUICKSTART.md](QUICKSTART.md).

---

## What real biology looks like inside

The simulator implements ~375 biological mechanisms documented in
[`references/feature-catalog.md`](references/feature-catalog.md), drawn from
Kandel et al. *Principles of Neural Science* (6th ed, 2021) plus 12
specialty texts.

A few highlights, in plain language:

**The retina** — 32×32 grid of light-sensitive cells, each firing more
when light hits its receptive field. Just like real retinal ganglion
cells (Hubel & Wiesel 1962).

**Visual cortex (V1)** — neurons tuned to oriented edges using Gabor
receptive fields. Detects horizontal lines, vertical lines, diagonals.
Same wiring real V1 develops in the first weeks of life.

**Basal ganglia** — the brain's "action selector". Multiple options
compete; the strongest one wins; the loser pathways are silenced.
Damage here causes Parkinson's (loss of selection) or Tourette's
(over-selection).

**Working memory (PFC)** — neurons that keep firing after their input
stops, holding the goal in mind. Real prefrontal cortex does this
via NMDA receptors creating "bistability" — the same property our
model implements (Wang 2002).

**Plasticity** — connections between neurons strengthen when both
fire together within 20 ms (LTP, Long-Term Potentiation), weaken
when one fires before the other (LTD). Reward-modulated: only the
right pairings get the reinforcement (Schultz 1998 dopamine RPE).

**Hippocampus** — pattern separator (DG), pattern completer (CA3),
memory readout (CA1). Damage = no new memories (Henry Molaison "H.M.").

**No symbolic shortcuts.** No `agent.pick_best_action()`. The action
emerges from spike rates in motor pools, just like real motor commands
emerge from M1 firing.

For the deep technical view, see [docs/biology.md](docs/biology.md).

---

## What's known to work, what's not

### Working

- **Navigation:** 38% time at goal on 16×16 grid (perception only,
  no shortcuts) — see [docs/CURRENT-STATE.md](docs/CURRENT-STATE.md)
- **Multi-region cascading:** 50+ brain regions, ~175,000 synapses,
  fully connected through biology-grounded pathways
- **Real-time visualization:** every neuron + synapse rendered in 3D
- **Reproducibility:** all RNG sources seeded, deterministic mode
  available
- **Performance:** ~7-8x speedup vs original (dt=1.0 + parallel-3 GPU
  sharing + cp.where masked-update spike-reset). 6-seed minimal-arch
  batch in ~45-55 min. See [`research/findings/2026-05-04-perf-speedup-stack.md`](research/findings/2026-05-04-perf-speedup-stack.md).

### Modest results

- **Image→word readout:** ~25% (chance) on average across seeds; some
  seeds reach 33%, but high variance
- **Working memory across long delays:** PFC holds goals for ~10s
  but not minutes
- **Vocabulary size:** 4 cardinal directions only

### Known limitations

- **Scale:** ~5K neurons. Real brain regions have 10⁴–10⁶ neurons each
- **Training time:** ~100 episodes. Real children see 10⁵+ examples
- **Static brain structure:** developmental changes (synaptic pruning,
  cortical layer formation) not modeled
- **Single time scale:** millisecond STDP only. No protein-synthesis-
  dependent late-LTP for long-term consolidation

---

## Project structure

```
neural-simulator/
├── README.md              ← you are here
├── QUICKSTART.md          ← 60-second setup
├── neural-simulator.py    ← GUI host + main entry point
├── benchmark.py           ← GPU throughput benchmark
├── viz_benchmark.py       ← visualization performance benchmark
├── run_benchmarks.py      ← biological validation suite
├── run_experiment_headless.py
├── run_parameter_sweep.py
├── requirements.txt
├── docs/
│   ├── CURRENT-STATE.md   ← what works today, technical details
│   ├── biology.md         ← what biology we model, plain language
│   ├── plans/             ← architecture decision records
│   └── project-history-archive.md  ← prior README content (preserved)
├── sim/                   ← engine: bridge, regions, kernels, plasticity
├── experiment/            ← ExperimentEngine + Stimulus + Readout + Training
├── experiments/           ← YAML configs for autonomous sweeps
├── research/
│   ├── runners/           ← experiment scripts (g1-g11, text_*, k_v2)
│   ├── findings/          ← chronological session findings
│   ├── experiment_runner.py ← YAML-driven sweep orchestrator
│   └── result_aggregator.py ← cross-condition rollup + verdict line
├── references/
│   ├── feature-catalog.md ← biology mechanism encyclopedia (375+ entries)
│   └── glossary.md
├── webapp/                ← FastAPI dashboard (server.py + static/)
├── simulation_profiles/   ← 47 brain-region JSON profiles
├── tests/                 ← pytest test suite (60+ files)
├── viz/                   ← 3D OpenGL rendering
└── ui/                    ← DearPyGUI controls
```

---

## How autonomous research runs

The project ships a YAML-driven experiment runner so the operator can
queue overnight sweeps without writing one-off scripts. Each YAML file
declares conditions (CLI flag combinations) × seeds; runs emit a
universal `[PROGRESS] {json}` event format and write per-run JSON
outputs.

```bash
# Run a sweep (anti-cheat controls + biology variants)
python -m research.experiment_runner experiments/biology_sweep.yaml

# Aggregate results with a verdict line
python -m research.result_aggregator biology_sweep

# Morning summary of any overnight run
python -m research.runners.morning_briefing --short
```

Built-in YAMLs in `experiments/`: `biology_sweep`, `minimum_biology`
(dose-response), `eval_sanity_check` (eval-methodology validator),
`b2_sparse_codes`, `b4_long_training`. The pre-staged decision chain
auto-launches the appropriate follow-up based on the verdict line.

A 7-8x speedup stack (`dt=1.0` + parallel-3 GPU sharing +
`cfg.fast_spike_reset`) ships with the runner; a 6-seed minimal-arch
batch finishes in ~45-55 minutes on an RTX 3090.

---

## Latest validated result

**Navigation on 16×16 gridworld with biology-grounded perception** —
agent reaches goal 38% of the time using only retinal input, no
shortcuts (no direct (x,y) access, no heuristic, no distance reward).

```
Configuration: A+E + G v2.5 + Cluster K v2 visual cortex
Validation:    6 independent seeds, 1800 steps each
Result:        Mean 2.87 ± 0.19 Manhattan distance to goal
```

See [`docs/CURRENT-STATE.md`](docs/CURRENT-STATE.md) for the configuration
and [`research/findings/2026-05-01-cluster-k-v2-breakthrough.md`](research/findings/2026-05-01-cluster-k-v2-breakthrough.md)
for the breakthrough writeup.

**Word→action mapping was UNDER INVESTIGATION as of 2026-05-03**
(permuted-label control falsified the prior 28.5% claim). Resolved
on 2026-05-06 by changing the training paradigm to embodied Hebbian
co-firing — Tier 1 BREAKTHROUGH (W→A 5/6 + A→W 6/6 aligned), Tier 2.1
BREAKTHROUGH (8-word + 12-word synonyms 5/6 + 6/6 aligned). The
permuted-label control was the right anti-cheat — it caught a real
problem with the previous training approach.

**Continual learning + sleep consolidation validated multi-seed
(2026-05-08 → 2026-05-09):**
- Phase 1.4 BRANCH A: 5/6 PASS, mean 103% retention (no catastrophic
  forgetting when learning new vocab)
- 8-word Phase 1.3 + Tier 2.1 combined: **3/3 GO** at multi-seed
  + 3-seed strict anti-cheat IDENTICAL to non-strict (cortex truly
  retains binding post-consolidation)
- 12-word vocab default arch: **2/3 GO PARTIAL** (defines capacity
  boundary; seed 43 fails at 71% primary)
- 12-word vocab **scaled arch** (n_motor=2000): **3/3 GO unanimous**
  multi-seed (mean primary 95.5%, mean synonym 115.0%) — capacity
  hypothesis confirmed at multi-seed (2026-05-09)
- Phase 1.5 unified 4-benchmark suite at scaled arch (3-seed FINAL,
  2026-05-09): **DEMOTED** from milestone gate to tier report. Mean
  **0.629 ± 0.056** — below 0.70 master plan threshold. All 3 hypothesis
  tests refuted; architectural ceilings real:
  - **2/4 PASS** all 3 seeds (sequential_expansion 0.95,
    retention_over_time 0.94) — sequential continual learning regime
    validated at scaled arch
  - **2/4 architectural ceiling** (interference 0.39, long_tail 0.26
    after 3-lever sweep): under-training REFUTED (+0.005), capacity
    REFUTED (+0.045), dose+teacher REFUTED (+0.090). All small
    sub-threshold lifts. Per-word bimodal pattern (some words bind,
    others don't) consistent across all tests suggests drive-pattern
    collision under sparse encoding — architectural not tunable.
- Track 3 v1 conversational scaffolding **feature-complete**
  (2026-05-09): 4 layers shipped (`--learn` primitive, `chat_learn_demo`
  runner, `:again`/`:opposite`/`:history`/`:forget` dialog state,
  `:speak` generative decoder). chat_speak_demo single-seed validated
  at 75% A2W (3/4 actions decoded correctly). Track 3 v1 conversation
  example: read words, write words, remember last action, learn new
  vocab online — all biology-grounded.
- **Phase 2 path-f-hybrid scale thesis REFUTED (2026-05-09 evening):**
  Phase 2.2b v3 at 50M params (375× larger than Phase 2.3a's 134K)
  produced inter-word cosine 0.85 — WORSE than Phase 2.3a's 0.72.
  Bigger model packs direction-word features MORE alike, not less.
  Char-level next-char objective is wrong for word-action transfer;
  scale doesn't fix it. **Phase 2 path-f-hybrid is a documented
  dead-end at single-3090 scale class.** The path forward to
  conversational capability is fully **Path A (biology-grounded):**
  Phase 1.4 BRANCH A + Phase 1.3 consolidation + Tier 2.1 12-word
  scaled multi-seed + Track 3 v1 (chat_repl `--learn` / `:speak`
  generative decoder / dialog state). Currently: chat_speak_demo
  6-seed multi-seed running on freed GPU to validate Track 3 layer 4
  robustness.

This characterizes the architecture's empirical capability: the
biology-grounded continual learning works at multi-seed (4/8/12-word
vocab tiers), sleep consolidation transfers binding to cortex
genuinely (anti-cheat validated), and motor pool capacity scales
linearly with vocab size (rule: ~333 neurons per sub-population).

See:
- [`research/findings/2026-05-09-Phase1.3-Tier2.1-12word-scaled-3seed-CONFIRMED.md`](research/findings/2026-05-09-Phase1.3-Tier2.1-12word-scaled-3seed-CONFIRMED.md) (12-word multi-seed)
- [`research/findings/2026-05-09-Phase-1.5-interference-undertraining-hypothesis.md`](research/findings/2026-05-09-Phase-1.5-interference-undertraining-hypothesis.md) (Phase 1.5 partial + hypothesis)
- [`research/findings/2026-05-08-Phase1.3-Tier2.1-strict-anti-cheat-3seed-CONFIRMED.md`](research/findings/2026-05-08-Phase1.3-Tier2.1-strict-anti-cheat-3seed-CONFIRMED.md) (anti-cheat)
- [`docs/CHAT-DEMO-GUIDE.md`](docs/CHAT-DEMO-GUIDE.md) (capacity scaling table)

---

## License

MIT. See [LICENSE](LICENSE).

## Mirrors

- GitHub: https://github.com/danthi123/neural-simulator
- Gitea: https://git.dant123.com/dant123/neural-simulator

## How this differs from typical AI

| Typical AI (e.g., LLMs) | This project |
|---|---|
| Gradient descent on millions of parameters | Spike-timing plasticity at each synapse |
| Trained once, then frozen | Always learning from interaction |
| Symbolic input/output (tokens) | Continuous neural activity |
| Massive corpora, no embodiment | Embodied agent in a world |
| Capabilities far exceed biology | Capabilities limited by biology faithfulness |

We're not trying to compete with GPT. We're trying to understand
**how much of intelligence emerges from biology alone**.
