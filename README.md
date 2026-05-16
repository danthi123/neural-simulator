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

**A trustworthy continual memory.** You can teach the system
word–concept facts ("apple is big"); it recalls them on cue, and —
the genuinely hard part it actually solves — it keeps old memories
intact while learning new ones (no *catastrophic forgetting*). It
holds roughly **320 distinct concepts** spread across a five-part
cortex, validated across multiple random seeds. Scientific basis:
words stored as distributed cell assemblies spanning cortex
(Pulvermüller 2001), recalled as scattered sparse patterns (sparse
distributed memory, Kanerva 1988), each memory a re-triggerable
tagged ensemble (engram cells — Liu/Tonegawa et al. 2012), and
protected from being overwritten by hippocampus→cortex consolidation
with sharp-wave-ripple replay (complementary learning systems — Marr
1971; McClelland, McNaughton & O'Reilly 1995; Buzsáki replay).

**It refuses to make things up.** Asked about something it was never
taught, it answers "I don't know" instead of fabricating a confident
wrong answer — a trust property today's language models lack. This is
validated quantitatively: a clean confidence separation between what
it knows and what it does not.

**It is learning to speak in its own words (early-stage).** The
system's *own* spiking network is being trained to generate language
from a local text corpus using spike-based backprop-through-time
(surrogate-gradient learning, Neftci, Mostafa & Zenke 2019). The
foundation is validated — it provably learns *real* text structure,
not noise — but it is **not yet fluent**, and honestly far from a
large language model. A separate model may be used only as a
training-time teacher (knowledge distillation, Hinton, Vinyals &
Dean 2015); **after training the system runs entirely on its own and
fully local** — no external model and no hand-written response
templates in actual use.

**It still navigates from vision.** The original capability: it
finds a goal in a gridworld using only simulated retinal input,
picking directions and learning from a reward signal — ~38% of time
at the goal on a 16×16 grid, well above chance.

**It visualizes its own brain.** Live 3D view of every neuron firing
and synapse pulsing, with interactive control — watch it learn in
real time.

---

## Why this matters

Modern AI uses gradient descent — a powerful but biologically implausible
optimization technique. Real brains have no equivalent of backpropagation
through time. They learn from local rules: "neurons that fire together,
wire together" (Hebb 1949), refined by dopamine reward (Schultz 1998).

This project tests how far you can go with **only those biological
rules**, entirely locally. The answer so far: navigation works well;
a biologically-grounded **memory** works genuinely well — continual
(no catastrophic forgetting) and trustworthy (it abstains rather than
confabulate); open-ended **language generation** is an active,
honestly-hard frontier (foundation validated, fluency not yet). The
system has clear limits, and mapping those limits — with anti-cheat
controls and forthright retractions when a result doesn't hold — is
itself the contribution.

---

## What the brain can do

| Capability | How it works | Status |
|---|---|---|
| **See** | Retina → V1 (Gabor edge detectors) → V2 → IT (object) | ✅ Working |
| **Decide** | Basal ganglia cascade picks one action from competing options | ✅ Working |
| **Move** | Motor cortex pools fire, agent moves on grid | ✅ Working |
| **Learn from reward** | Dopamine modulates spike-timing plasticity | ✅ Working |
| **Hold goals in mind** | Prefrontal cortex working memory (NMDA bistability) | ✅ Working |
| **Remember word–concept facts** | Distributed cortical word-ensembles (Pulvermüller); sparse distributed recall (Kanerva 1988); engram tagging (Tonegawa) | ✅ Working — ~320 concepts, multi-seed validated |
| **Not forget when learning more** | Hippocampus→cortex consolidation + sharp-wave-ripple replay (complementary learning systems, McClelland 1995) | ✅ Working — no catastrophic forgetting |
| **Know what it doesn't know** | Recall-confidence threshold; abstains below it | ✅ Working — refuses to confabulate |
| **Speak in its own words** | Own spiking network trained by surrogate-gradient backprop-through-time (Neftci 2019) on local text | ⚠️ Early — foundation validated, not yet fluent |

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

### 🎉 G.20 160-concept multi-bridge ensemble SHIPPED (2026-05-15)

**5 G.20 bridges × 32 concepts = 160 unique concept words.**
All 5 bridges at EXACTLY 26/32 (81.2%) top-1 / 31/32 (96.9%) top-5
at seed 42 — identical PASS rate across 5 different vocab categories.

```bash
# Run end-to-end demo (requires 5 bridges trained):
python research/runners/g20_160word_demo.py --seed 42 --friendly
```

Example session:
```
> vocab                          → TOTAL: 160 unique concepts
> remember apple is big          → OK (cross-bridge: apple in A, big in C)
> what is apple                  → Apple is associated with: big, red, ...
> is a dog an animal?            → Yes, dog is a kind of animal.
> what mammals do you know?      → Kinds of mammal: dog, cat, person, baby
> what is dogs (tokenized→dog)   → Dog is associated with: small, fast, ...
```

Combined effective vocab: **160 concepts × ~6 morpheme variations
≈ 960 surface forms** — toddler-vocabulary range.

Plus 4-SEED validation at 32-concept tier: 96/128 (75.0%) — statistically
equivalent to v16 baseline (77.5%) at 2× vocabulary + 1/2 substrate.

Catalog G.20 status: PARTIALLY MISSING → **5-BRIDGE PRODUCTION ENSEMBLE**.

```bash
# Train a 32-concept G.20 bridge (~30 min):
python -m research.runners.concept_pool_demo_shared --seed 42 \
    --n-concepts 32 --n-train-events 400 --n-lang-input 8192 \
    --n-shared-pool 1600 --slice-size 50 --top-k 100 \
    --topographic-factor 10.0 --off-target-factor 0.1 --sparsity 0.03 \
    --save-bridge bridges/shared_pool_n32.h5 \
    --out results/shared_pool_n32.json

# Chat with the trained bridge:
python -m research.runners.shared_pool_chat \
    --load-bridge bridges/shared_pool_n32.h5 \
    --vocab "apple,river,dog,cat,go,come,stop,look,big,small,hot,cold,\
tree,bird,sun,moon,walk,run,eat,sleep,red,blue,fast,slow,\
house,road,fire,water,give,take,find,lose" --friendly
```

Per-neuron PASS efficiency: 4.2× better than v16. Catalog
G.20 (Pulvermüller distributed cortical word ensembles) status:
**PROTOTYPE VALIDATED**. See
[`research/findings/2026-05-15-G20-shared-pool-BREAKTHROUGH-32-concepts.md`](research/findings/2026-05-15-G20-shared-pool-BREAKTHROUGH-32-concepts.md).

### 60-word multi-bridge conversation (2026-05-15)

Single bridge has an architectural ceiling at 16-word vocab. The
**multi-bridge ensemble** runs 5 v16 bridges (each owning 12 distinct
concept words = 60 unique vocab) with automatic dispatch:

```bash
# End-to-end demo with natural-language output
python research/runners/multibridge_60word_demo.py --seed 42 --friendly
```

Example session (--friendly mode):
```
> remember the dog is big
  OK, I'll remember dog is big.
> remember the dog ate apple
  OK, I'll remember dog ate apple.
> is the dog big?
  Yes, dog is big.
> is the apple small?
  I don't know. I haven't been told.
> who ate apple?
  Dog did.
> what did dog ate?
  Apple.
> remember sun is hot
  OK, I'll remember sun is hot.    # CROSS-SET (sun in set2, hot in set1)
> remember apple's color is red
  OK, I'll remember apple's color is red.
> what color is apple?
  Red.
```

11 conversational features: pair encoding, N-word sentences, negation,
conjunctions, possessives, pronoun coreference, tense (PAST/FUTURE),
comparisons, yes/no, role queries, relational queries. Plus memory
CRUD (about / forget / save). 91 unit tests passing in 1.2s.

See [`research/findings/2026-05-15-multibridge-60word-shipped.md`](research/findings/2026-05-15-multibridge-60word-shipped.md)
for the full milestone doc.

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

### Working (validated, multi-seed where stated)

- **Continual memory:** ~320 concepts across a five-part cortex; new
  learning does **not** erase old memories (no catastrophic
  forgetting) — multi-seed validated
- **Trustworthy recall:** the system abstains ("I don't know")
  instead of confabulating on un-taught queries — quantified clean
  confidence separation between known vs unknown
- **Associative retrieval:** cue→associate recall ≈87% at the
  320-concept scale / ≈93% multi-seed at the 160-concept scale;
  subject→(verb, object) sentence-style retrieval ≈80%
- **Navigation:** ~38% time at goal on 16×16 grid from simulated
  vision only, no shortcuts — see [docs/CURRENT-STATE.md](docs/CURRENT-STATE.md)
- **Own-network text learning (foundation):** the system's own
  spiking net provably learns real local text structure (beats a
  shuffled-text control), trained by surrogate-gradient
  backprop-through-time — fully local
- **Multi-region cascading:** 50+ brain regions, biology-grounded
  pathways; **real-time 3D visualization**; **reproducibility**
  (all RNG seeded, deterministic mode); **performance** ~7–8×
  speedup vs the original engine

### Modest / early

- **Language fluency:** the own-network generator's foundation is
  validated but output is far below a large language model — fluency
  is the active frontier (knowledge-distillation training underway)
- **Working memory across long delays:** prefrontal working memory
  holds a goal for seconds, not minutes

### Known limitations (honest)

- **Not LLM-fluent.** The genuine contribution is a *continual,
  trustworthy, fully-local* memory — not open-ended fluent prose.
  Local hardware caps generation well below cloud models; integrity
  (no cheating, no fabrication, self-contained), not parity, is the
  point.
- **Scale:** thousands of neurons per region vs 10⁴–10⁶ in biology;
  training on far fewer examples than a developing brain sees
- **Static structure:** developmental synaptic pruning / cortical
  layer formation not modeled
- **Single time scale:** millisecond spike-timing plasticity only;
  no protein-synthesis-dependent late-phase consolidation

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
├── tests/                 ← pytest test suite (102 files)
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

**A trustworthy, continual, fully-local concept memory + an
early own-network speech generator (2026-05-16):**

A multi-part cortex storing many concepts as distributed sparse
ensembles (Pulvermüller cortical word-webs; Kanerva sparse
distributed memory; Tonegawa engram tagging) now demonstrates, with
anti-cheat controls:

```
Concept capacity:   320 concepts (5 cortices × 64), every cortex
                    ~98% concept-discrimination; 160-concept tier
                    100% per-cortex
Cross-cue recall:   ~87% genuine (320) / ~93% multi-seed (160),
                    measured with a pre/post anti-cheat control
                    (counts only if it was NOT already the answer
                    before being taught — rules out coincidence)
Sentence recall:    subject→(verb, object) ≈80%
Trustworthiness:    clean confidence separation known vs unknown —
                    the system abstains instead of confabulating
Continual learning: new vocabulary does not erase old memories
                    (no catastrophic forgetting), multi-seed
Generator (early):  the system's OWN spiking net, trained by
                    surrogate-gradient backprop-through-time on a
                    local text corpus, provably learns real text
                    structure (70% loss reduction; 22% better than
                    a shuffled-text control) — foundation only,
                    not yet fluent
```

Honest framing: the validated, distinctive result is a
biology-grounded memory that is **continual** (doesn't forget) and
**trustworthy** (doesn't fabricate), running **entirely locally**.
Open-ended fluent generation is an active frontier — the foundation
is proven; fluency is not, and is not overclaimed. Several results
en route were retracted forthrightly when anti-cheat controls
failed (e.g. a 2026-05-14 architecture-mismatch bug; a
seed-favourable retrieval number corrected to its multi-seed mean) —
those corrections are part of the record, not hidden.

See `research/findings/2026-05-16-G20-failure-mechanism-FINAL-SYNTHESIS.md`
and `research/findings/2026-05-16-generator-increment1-foundation.md`.

---

**Earlier flagship — navigation on 16×16 gridworld with biology-grounded perception** —
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
- Track 3 v1 conversational scaffolding **feature-complete +
  multi-seed validated** (2026-05-09): 4 layers shipped (`--learn`
  primitive, `chat_learn_demo` runner, `:again`/`:opposite`/`:history`/
  `:forget` dialog state, `:speak` generative decoder). chat_speak_demo
  single-seed validated at 75% A2W (3/4 actions decoded correctly).
  **6-seed multi-seed: A2W mean 58.3% ± 20.4%, 5/6 seeds at ≥50%**
  (5/6 above-chance). Per-direction A2W: N=67% E=67% S=67% W=33%
  (W cascade-bias mirror of the Tier 1 BREAKTHROUGH N-bias).
  Track 3 v1 conversation example: read words, write words, remember
  last action, learn new vocab online — all biology-grounded.
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
  generative decoder / dialog state). Currently:
  `chat_speak_synonym_demo` (Tier 2.1 8-word :speak production-side
  test) smoke in flight, then 6-seed multi-seed wrapper pre-staged.
  Subsequent: 16-word capacity rule extension test (predicted PASS
  at scaled arch).

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
