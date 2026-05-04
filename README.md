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

**Word-action mapping (under investigation).** Originally claimed at
28.5% accuracy across 6 seeds, p=0.027. The 2026-05-03 permuted-label
control test revealed this is *not* real word-action learning — across
45+ runs spanning every variant tested, the TRUE labeled mapping is
NEVER the best of 24 permutations of (token → action). The architecture
has structure above chance but it's seed-dependent and unaligned with
task labels. A biology-grounded investigation (topographic Wernicke →
motor priors + motor PV-FS lateral inhibition) is the active research
question. See `research/findings/2026-05-03-architecture-fundamentally-cant-align.md`.

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
| **Understand words** | Language input → motor cortex via Wernicke→Broca pathway | ⚠️ 28.5% accuracy is *not real learning* per 2026-05-03 permuted-label control — under active biology-grounded investigation |
| **Speak** | Cortex/visual cortex → language output | ⚠️ Partial (high variance) |
| **Remember & dream** | Hippocampus + sharp-wave-ripple replay | ⚠️ Implemented, integration pending |

---

## Try it in 60 seconds

```bash
git clone https://github.com/danthi123/neural-simulator
cd neural-simulator
python neural-simulator.py        # GUI mode with live 3D viz
```

Or run a research experiment headless:

```bash
# Watch the agent navigate, no language training
python -m research.runners.g11_bg_runner --moving-goal --deterministic \
    --enable-msn-lateral-inhibition --enable-d1-d2-asymmetry \
    --enable-cluster-a-closed-loop --enable-cluster-e-topography \
    --enable-dlpfc-wm --enable-pfc-nmda --enable-visual-cortex \
    --grid-size 16 --seed 42 --n-steps 1800

# Test the text-to-action capability (28.5% W→A baseline)
python -m research.runners.text_eval_embodied \
    --n-episodes 100 --steps-per-episode 30 --seed 42 \
    --stim-steps-per-step 200 --reset-steps 100 \
    --out-stats results.json
```

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
├── docs/
│   ├── CURRENT-STATE.md   ← what works today, technical details
│   ├── biology.md         ← what biology we model, plain language
│   ├── plans/             ← architecture decision records
│   └── project-history-archive.md  ← prior README content (preserved)
├── sim/                   ← engine: bridge, regions, kernels, plasticity
├── research/
│   ├── runners/           ← experiment scripts (g1-g11, text_*)
│   └── findings/          ← chronological session findings
├── references/
│   ├── feature-catalog.md ← biology mechanism encyclopedia (375+ entries)
│   └── glossary.md
├── tests/                 ← pytest test suite
├── viz/                   ← 3D OpenGL rendering
└── ui/                    ← DearPyGUI controls
```

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

**Word→action mapping is currently UNDER INVESTIGATION** — earlier
"28.5% accuracy" claim was debunked by a permuted-label control test
on 2026-05-03 (the structure was real but seed-dependent and not
aligned with N/E/S/W labels). A biology-grounded sweep (topographic
Wernicke→motor prior + motor PV-FS lateral inhibition) is in flight
to test whether biology fixes break the alignment streak. Pre-staged
decision chain auto-launches the appropriate follow-up based on
outcome. See [`research/findings/2026-05-03-permuted-label-control-NEGATIVE.md`](research/findings/2026-05-03-permuted-label-control-NEGATIVE.md).

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
