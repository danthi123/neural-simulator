# Neural Simulator

**A GPU-accelerated simulator for biologically realistic spiking neural
networks — with a real-time 3D view of every neuron firing.**

It learns the way brains do: individual neurons fire in real time,
a dopamine-like reward signal reinforces successful actions, and connections
strengthen or weaken based on millisecond-precise spike timing. The core
runs on local biological rules — *not* backpropagation through a static
graph, no supervised labels, no symbolic optimizer.

The distinguishing goal: a *single* simulated brain that navigates a world
and holds a simple, grounded conversation, with **every cognitive step done
by spiking neurons** rather than ordinary code. Where the biology genuinely
can't do something on this substrate, that limit is measured and reported
rather than papered over.

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![Backend](https://img.shields.io/badge/backend-CuPy%20(CUDA)%20%2F%20NumPy%20(CPU)-orange.svg)
![Status](https://img.shields.io/badge/status-active%20research-yellow.svg)

*The simulated brain — one engine, two configurations (a **navigating** and a **conversing** brain) that share it, plus one memory and one dopamine core, joined by validated synaptic links:*

```mermaid
flowchart TB
    World(["🌍 Simulated world — renders what the agent sees, enacts its movements"]):::io
    Turn(["💬 A sentence — the user's turn"]):::io

    subgraph ENGINE["🧠 One brain — spiking neurons + synapses on a single update loop"]
      direction TB
      subgraph SENSE["Sensing"]
        direction LR
        VIS["Vision — retina to primary visual cortex<br/><small>orientation-selective edge detectors · what and where streams</small>"]:::sense
      end
      subgraph SM["The navigating brain — reach goals by moving"]
        direction LR
        NAV["Action selection and navigation<br/><small>basal-ganglia go/no-go loops · superior-colliculus orienting · place cells · goal working-memory</small>"]:::nav
      end
      subgraph LANG["The conversing brain — understand, think, speak"]
        direction LR
        COMP["Understanding<br/><small>parser + reservoir: word order to who-did-what</small>"]:::conv
        CONCEPT["Concepts and meaning ✅<br/><small>categories learned from experience · reasoning</small>"]:::conv
        SPEAK["Speaking ✅<br/><small>self-organized grammar · every word on spikes</small>"]:::gen
        DISC["Conversation<br/><small>tracks who/what across turns · the 'I don't know' guard</small>"]:::plan
      end
      subgraph SHARED["Shared core (used by both brains)"]
        direction LR
        MEM["Memory — hippocampus<br/><small>separate and complete patterns · tag · replay in 'sleep'</small>"]:::mem
        REW["Reward and drive — dopamine<br/><small>one shared limbic core for both brains</small>"]:::reward
        LRN["Learning rules<br/><small>spike-timing · Hebbian · three-factor · dendritic</small>"]:::learn
      end
    end

    Body(["🌍 Body — carries out the chosen movement"]):::io
    Reply(["🗣️ Spoken reply — grounded, checked, or 'I don't know'"]):::io

    World ==>|pixels| VIS
    VIS ==> NAV
    NAV ==>|movement| Body
    Turn ==>|a sentence| COMP
    World -->|a spoken command steers movement| NAV
    VIS -.->|what it saw while moving| MEM
    COMP ==> CONCEPT ==> DISC
    CONCEPT --> MEM
    DISC ==> SPEAK ==> Reply
    MEM -.-> DISC
    REW -.->|modulates learning and confidence| SM
    REW -.->|modulates learning and confidence| LANG
    LRN -.-> SHARED

    classDef io fill:#eef1f4,stroke:#7a8794,color:#1d1d1f;
    classDef sense fill:#d6eaf8,stroke:#2e6da4,color:#10303f;
    classDef nav fill:#fdebd0,stroke:#c87f2e,color:#5b3a10;
    classDef conv fill:#d6eaf8,stroke:#2471a3,color:#10303f;
    classDef gen fill:#d1f2eb,stroke:#138d75,color:#0c3d33;
    classDef plan fill:#e9dcf5,stroke:#7d3c98,color:#3b1d4e;
    classDef mem fill:#d4efdf,stroke:#1d8049,color:#0f3d23;
    classDef reward fill:#fcf3cf,stroke:#b9770e,color:#5b3a10;
    classDef learn fill:#eae3f3,stroke:#5b4a8a,color:#2c2247;
```

**Full flowcharts (GitHub-rendered, plain-language, kept current):**
[overview](docs/diagrams/brain_architecture_current.md) ·
[detailed diagrams — every region &amp; pathway](docs/diagrams/brain_architecture_detailed.md) ·
[development roadmap](ROADMAP.md)
&nbsp;·&nbsp; *(the earlier hand-drawn per-synapse [SVGs](docs/diagrams/brain_master.svg) are an archived 2026-06 snapshot, superseded by the detailed diagrams)*

**Jump to:** [What it is](#what-it-is) ·
[Roadmap](#️-roadmap--the-full-development-path) ·
[Key features](#key-features) ·
[Quick start](#quick-start) ·
[Architecture](#architecture) ·
[Project status](#project-status--research-direction) ·
[Glossary](#glossary) ·
[Docs](#documentation--further-reading)

---

## 🗺️ Roadmap — the full development path

**[`ROADMAP.md`](ROADMAP.md) is the source of truth for progress toward the goal** — artificial life whose conversational ability approaches a large language model, built the honest way: one simulated spiking brain, learning from experience, with no permanent external AI model doing the thinking. It's written to be read without knowing the codebase, and lays out the whole developmental path stage by stage — each mapped to the brain region/function it reproduces (with textbook and paper citations), a status, what's done, what's open, and the next step — plus the temporary stand-ins still to be replaced, the honest remaining walls, and a no-hype assessment of the distance left.

At a glance (see [`ROADMAP.md`](ROADMAP.md) for the plain-language detail and citations):

| Stage | Brain function it reproduces | Status |
|---|---|---|
| Perception | Retina → visual cortex edge detectors; what/where streams | 🟨 Partial |
| Attention & orienting | Superior colliculus "look here" map; arousal | 🟩 Done (orienting) |
| Action selection | Basal-ganglia go/no-go loops; evidence → commitment burst | 🟩 Done |
| Reward & value | Dopamine "actual minus expected" signal; one shared drive | 🟩 Done · 🟨 value critic |
| **Navigation & spatial cognition** | Place cells + goal-directed movement from perception alone | 🟩 **Done** (flagship behavior) |
| Memory | Hippocampal loop; memory tags; sleep replay (replaces backprop) | 🟩 Done · 🟧 deep consolidation |
| Concept formation | Concept hub; categories discovered from experience | ✅ **Emergent** |
| Comprehension | Dual-stream language; word-order → roles, learned, in spikes | ✅ **Emergent** · 🟧 deep nesting |
| Semantic reasoning | Inference (inheritance, transitivity) emerges from shared codes | ✅ **Emergent** |
| Production | Broca's area; self-taught grammar; every word spoken in spikes | ✅ **Emergent** · 🟧 open prose |
| Conversation | Working memory of who's-being-discussed; "I don't know" guard | ✅ **Emergent** · 🧩 fluent chat |
| Working memory & recursion | Persistent-activity slots; graded fading memory | 🟩 Done · 🟧 nesting depth |
| Artificial life | Develop over time; one merged brain; one drive; persistence | 🟩 Done (pieces) · 🟨 unified |

Legend: ✅ emergent (learned from experience) · 🟩 done (with one hand-designed part) · 🟨 partial · 🟧 a mapped limit · 🧩 temporary stand-in · ⬜ open. **The two honest gaps to a large language model:** open-ended fluent speech without the small conventional-AI crutch (its first home-grown rung just landed — an emergent, on-brain, no-backpropagation next-word model that beats the standard baselines), and a deeper learning rule to lift the remaining ceilings. A real, bounded, multi-month distance — not a demo away, and not blocked. The roadmap also covers the body, supporting systems (cerebellum, sleep), the 3D viewer and interactive consoles, and future directions.

---

## What it is

This is a research platform for building brains out of simulated neurons.
Networks are made of biologically realistic model neurons that communicate
with discrete electrical pulses ("spikes") over time, the way real neurons
do — not the continuous numbers used in mainstream machine learning. The
computation runs in parallel on an NVIDIA graphics card (via CUDA and the
CuPy array library), or on an ordinary CPU when no GPU is available. An
optional real-time 3D view shows every neuron firing and every synapse
pulsing as the network learns.

The project asks a single question: **how much of intelligence emerges from
biological rules alone?** Instead of gradient descent — the powerful but
biologically implausible optimizer behind modern AI — it uses only local
learning rules that real brains plausibly use: "neurons that fire together
wire together" (Hebb 1949), refined by dopamine-driven reward (Schultz
1998). It is a working platform *and* an active research program: the
simulation engine, brain-region framework, and navigation agent are mature
and demonstrated, while a small neuron-built conversational agent is a
growing capability with validated core behaviors.

It is **not** a large language model and does not try to be one. Its
contribution is a memory and reasoning system that is continual (it keeps
learning without forgetting), trustworthy (it declines to answer rather than
fabricate), and self-contained (after training it runs entirely on local
hardware with no external model).

**Who might find it useful:**

| If you are a… | You'll care that… |
|---|---|
| **Software developer** | It's a scriptable Python engine: 50+ brain regions, sparse GPU kernels, a YAML-driven experiment runner, a web dashboard, and a NumPy fallback that runs with no GPU at all. |
| **Computational neuroscientist** | Neurons use published models (Izhikevich, Hodgkin–Huxley, adaptive exponential integrate-and-fire); learning reproduces measured timing curves (Bi & Poo 1998); regions, pathways, and neuromodulators are declared as data, not hand-wired — and results are validated across multiple random seeds with controls against accidental success. |
| **Biologist** | Mechanisms are grounded in *Principles of Neural Science* (Kandel, 6th ed.) plus a dozen specialty texts, explained in plain language — see [`docs/biology.md`](docs/biology.md). |
| **Curious reader** | You can watch a virtual brain learn to navigate and to remember words, in 3D, in real time — and see exactly how far "biology alone" gets you. |

---

## Key features

- **Large-scale spiking simulation.** Networks from a few thousand to over a
  hundred thousand neurons, organized into named brain regions (visual
  cortex, basal ganglia, thalamus, motor cortex, prefrontal cortex,
  hippocampus, language areas) wired together by biologically motivated
  pathways. Runs are seed-controlled and reproducible.

- **A vision-based navigation agent.** A simulated animal finds a goal on a
  grid using only simulated vision — no direct coordinates and no hand-coded
  distance signal. It chooses moves through a modeled basal-ganglia decision
  circuit and improves through reward-driven learning. Every step *between*
  seeing and acting is performed by neurons by default. This is a
  demonstrated capability, validated across multiple random seeds.

- **A small conversational agent built entirely from simulated neural
  circuits** — not a bolted-on external language model. It parses short
  sentences with a learned neural parser, stores facts, answers who/what and
  yes/no questions (including negation), combines concepts into structured
  facts ("who did what to whom", including nested clauses), chains several
  facts to answer multi-step questions, and tracks what was just mentioned so
  a later pronoun resolves correctly. Core question-answering is demonstrated
  and validated across random seeds; several richer abilities are exploratory
  or documented as current limits.

- **A trustworthy, continual memory.** You can teach it word–concept facts;
  it recalls them on cue and — the genuinely hard part — keeps old memories
  intact while learning new ones (avoiding *catastrophic forgetting*, the
  usual failure mode of continually trained neural networks). It has been
  validated holding a few hundred distinct concepts across a multi-part model
  cortex.

- **It refuses to make things up.** Asked about something it was never
  taught, it answers "I don't know" rather than fabricating a plausible-but-
  wrong reply — a trust property that mainstream language models notably
  lack. This is measured: there is a clear confidence gap between what it
  knows and what it does not.

- **Generalization across similar things.** Shown a *novel* object through
  its simulated eyes, it can recognize a never-seen shape as belonging to a
  known category, then recall a fact about that category and answer. The
  end-to-end demonstration works; per-run fidelity is the current open edge.

- **Learning and plasticity mechanisms** drawn from the neuroscience
  literature: spike-timing-dependent plasticity, short-term plasticity,
  homeostatic regulation, reward-modulated learning, memory consolidation via
  simulated-sleep replay, and short-term working memory.

- **Real-time 3D visualization** of every neuron firing and synapse pulsing,
  with interactive camera control.

- **Runs with or without a GPU.** The GPU path (CuPy/CUDA) is the speed path;
  a NumPy path runs the same code on any CPU — slower, but numerically
  equivalent.

### One honest caveat up front

The most recent research is the conversational and language-generation
frontier, and it is genuinely early. The system's *own* spiking network is
being trained to generate language, and its foundation is validated — it
provably learns real text structure rather than noise — but it is **not yet
fluent**, and far from a large language model. When a separate model is used,
it serves only as a training-time teacher; after training the system runs
entirely on its own, fully local, with no external model and no hand-written
reply templates in the loop.

---

## Quick start

### Install and launch the GUI

```bash
git clone https://github.com/danthi123/neural-simulator
cd neural-simulator
pip install -r requirements.txt   # CuPy + DearPyGUI + PyOpenGL + h5py + ...
python neural-simulator.py        # GUI with live 3D visualization
```

Full setup, prerequisites, and troubleshooting are in
[QUICKSTART.md](QUICKSTART.md).

### No NVIDIA GPU? Run on the CPU

Set one environment variable and the engine swaps its GPU array library
(CuPy) for NumPy with no code change. It is slower, but every demo below
works:

```bash
SIM_BACKEND=numpy python -m research.runners.chat_demo --seed 43
```

### Talk to it

Type a direction word and watch the matching group of movement neurons
(a "motor pool") light up (four-word vocabulary, ~6 min to train on one
seed):

```bash
python -m research.runners.chat_demo --seed 43 --train-events 200
```

With synonyms — both `north` and `up` activate the same motor pool
(eight-word vocabulary, ~10 min):

```bash
python -m research.runners.chat_synonym_demo --seed 42 --train-events 400
```

An interactive shell that saves its trained brain so you reload instantly
next time:

```bash
# First time: train, then save (~6–20 min depending on vocabulary)
python -m research.runners.chat_repl --mode tier1 --seed 43 \
    --save-bridge simulation_checkpoints_h5/repl_tier1.simstate.h5

# Later sessions: load the saved brain (~30 sec) and start chatting
python -m research.runners.chat_repl --mode tier1 --seed 43 \
    --load-bridge simulation_checkpoints_h5/repl_tier1.simstate.h5
```

A trained, larger-vocabulary brain holds a short conversation grounded in
its own memory:

```
> remember the dog is big
  OK, I'll remember dog is big.
> is the dog big?
  Yes, dog is big.
> is the apple small?
  I don't know. I haven't been told.
> who ate the apple?
  Dog did.
```

See [`docs/CHAT-DEMO-GUIDE.md`](docs/CHAT-DEMO-GUIDE.md) for every
conversational demo.

### Watch it navigate

Run the navigation agent headless (vision-only, 16×16 grid):

```bash
python -m research.runners.g11_bg_runner --moving-goal --goal-schedule multi --deterministic \
    --enable-msn-lateral-inhibition --enable-d1-d2-asymmetry --enable-striatal-pv-fsi \
    --enable-cluster-a-closed-loop --enable-cluster-e-topography \
    --enable-dlpfc-wm --enable-pfc-nmda \
    --enable-visual-cortex --visual-cortex-action-warmup-steps 600 \
    --grid-size 16 --seed 42 --n-steps 1800
```

It prints live progress (`step 800/1800  pos=(6,1)  goal=(1,6)
recent_dist=2.6 …`) and writes a per-run JSON summary.

### Run the tests

```bash
pytest tests/ -v
```

### Run a benchmark or validation suite

```bash
python benchmark.py --quick                      # GPU throughput
python run_benchmarks.py --benchmark stdp-timing # biological validation (Bi & Poo STDP)
```

### File formats

| Format | Extension | Purpose |
|---|---|---|
| Profiles | `.json` | Human-readable simulation configuration |
| Checkpoints | `.simstate.h5` | Compressed (HDF5) full simulation state |
| Recordings | `.simrec.h5` | Compressed (HDF5) frame-by-frame data |

---

## Architecture

Two threads keep the interface responsive while the brain computes:

```
Main thread                         Simulation thread
───────────                         ─────────────────
DearPyGUI event loop      ◄──────►  GPU neural dynamics (CuPy/CUDA)
OpenGL 3D rendering    lock-free     spike + plasticity kernels
camera / interaction     queues      recording / checkpointing
```

The engine lives in the `sim/` package (43 modules, ~21K lines). The central
object is the **`SimulationBridge`**, which owns all neuron and synapse state
as GPU arrays and advances the network one millisecond-scale step at a time:
synaptic currents → background noise → neuron-model update → plasticity →
visualization → recording.

| Package | What's in it |
|---|---|
| `sim/` | The engine: the bridge, neuron/plasticity GPU kernels, brain-region framework, neuromodulators, connectivity, checkpointing |
| `viz/` | OpenGL 3D renderer, camera, neuron picker, overlays |
| `ui/` | DearPyGUI control panels and the configuration round-trip |
| `experiment/` | Stimulus injection, neuron-group management, readout/analysis, training protocols |
| `research/runners/` | Hundreds of headless experiment scripts (navigation, conversation, consolidation, and more) |
| `webapp/` | Web dashboard for launching runs and watching them live |

Networks are stored as sparse connection matrices, so memory grows with the
number of *actual synapses*, not with neurons-squared. Brain regions and the
pathways between them are *declared* as data and wired automatically — adding
a region does not mean hand-editing the engine.

For the visual big picture, see the architecture diagrams under
[`docs/diagrams/`](docs/diagrams/README.md). For the deep technical view of
the modeled biology, see [`docs/biology.md`](docs/biology.md).

### Performance envelope

Measured on a single NVIDIA RTX 3090 (24 GB):

| Network size | Backend | Notes |
|---|---|---|
| 1K–10K neurons | CuPy or NumPy | Runs on a laptop CPU in NumPy mode; comfortable on any CUDA GPU |
| 10K–100K neurons | CuPy (CUDA) | The everyday research range; fits in 8–24 GB depending on connectivity |
| 100K+ neurons | CuPy (CUDA) | Needs ~20 GB+ VRAM; use sparse connectivity and the memory-pool limit |

The GPU path is roughly 4–50× faster than the CPU path depending on the
workload, but the two are numerically equivalent. The simulation time step is
0.5 ms for the fast neuron models, automatically tightened to 0.05 ms for
full Hodgkin–Huxley biophysics.

---

## Project status & research direction

This is an **active research project.** The parts that are mature and
demonstrated:

- **The simulation engine, region framework, plasticity rules, and
  visualization** — the working platform.
- **Vision-based navigation** — reaches a goal from simulated vision only,
  with no coordinate or distance shortcuts, validated across multiple random
  seeds. Every step between seeing and acting runs in neurons by default:
  orienting toward the goal, a self-organizing position code, a neural
  reward/value/dopamine system, and the move decision itself (which emerges
  from a race between competing action circuits rather than an off-brain
  "pick the best option" step).
- **A continual, trustworthy memory** — teach it word–concept facts; it
  recalls them, keeps old memories intact while learning new ones, and
  declines to answer about anything it was never taught. Validated at a
  few-hundred-concept scale across multiple random seeds.

The active frontier is **conversation and language built from neurons.** The
whole conversational pipeline — parsing, storing facts, recalling them,
combining them into structured facts, and answering — is being pushed to run
as spiking neurons within a single shared network, and to *learn its concept
representations from experience* (hearing words in context) rather than
having them hand-assigned. Open-ended, fluent language generation remains a
genuinely hard, honestly acknowledged frontier: the foundation is proven, but
fluency is not there yet.

The guiding principle throughout is strict: **everything between sensing the
world and acting on it must be done by simulated neurons and synapses.** Only
the external world (the environment) and the body (turning a motor-neuron
output into a movement) may be handled by ordinary code. Anything computed by
a plain formula — a reward, a decision, a word choice — is treated as a
placeholder to be replaced by a genuine neural mechanism. When the neural
version underperforms the shortcut, that gap is recorded as an honest
scientific result rather than hidden.

### Known limitations

- **Not a large language model.** Local hardware caps language generation
  well below cloud models. The contribution is integrity (no cheating, no
  fabrication, self-contained), not fluency parity.
- **The cost of doing everything in neurons is speed.** With all the spiking
  machinery switched on at full scale, a conversation runs much slower than
  the same pipeline would with the (now-retired) ordinary-code shortcuts. The
  single-query path has already been sped up substantially; extending that to
  the rest of the fully-spiking loop is the next engineering step.
- **One deep substrate limit stays open and is named as such.** Separating a
  position cell's *intrinsic* near/far geometry from the *value* it has
  learned is something a simple point-neuron cannot fully do — it would need
  the branching input structure (dendrites) that real neurons have. This is
  the project's deepest open *neural* problem, pursued as its own research
  arc, not a hidden shortcut.
- **Scale.** Thousands of neurons per region versus far more in biology; far
  fewer training examples than a developing brain sees.
- **Simplifications.** Developmental synaptic pruning and cortical-layer
  formation are not modeled; learning is at the millisecond spike-timing
  scale only, without slower protein-synthesis-dependent consolidation.
- **Research software.** APIs may change; this is not peer-reviewed for any
  clinical or diagnostic use.

### How the record is kept

The project keeps a detailed, dated record. Each research session writes a
findings document — **including negative results, which are treated as real
findings** — and milestones are summarized in the changelog. Results are
reported with the random seeds and conditions used, and several promising
numbers have been **retracted** when a control later showed they did not hold
up. Those corrections are part of the record, not hidden.

- **History & milestones:** [CHANGELOG.md](CHANGELOG.md)
- **What works today, in technical detail:** [`docs/CURRENT-STATE.md`](docs/CURRENT-STATE.md)
- **The science roadmap:** [`docs/SCIENCE_ROADMAP.md`](docs/SCIENCE_ROADMAP.md)
- **Per-session findings:** `research/findings/` (chronological)

---

## How autonomous research runs

A YAML-driven runner queues parameter sweeps without one-off scripts. Each
file declares conditions (flag combinations) × seeds; runs emit a uniform
progress event and write per-run JSON.

```bash
python -m research.experiment_runner experiments/biology_sweep.yaml   # run a sweep
python -m research.result_aggregator --config biology                 # roll up + verdict line
python -m research.runners.morning_briefing --short                   # summarize an overnight run
```

---

## Project structure

```
neural-simulator/
├── README.md              ← you are here
├── QUICKSTART.md          ← full setup + troubleshooting
├── CHANGELOG.md           ← dated history & milestones
├── neural-simulator.py    ← GUI host + main entry point
├── benchmark.py           ← GPU throughput benchmark
├── run_benchmarks.py      ← biological validation suite
├── sim/                   ← engine (43 modules)
├── viz/                   ← 3D OpenGL rendering
├── ui/                    ← DearPyGUI controls
├── experiment/            ← stimulus, groups, readout, training
├── experiments/           ← YAML configs for autonomous sweeps
├── research/
│   ├── runners/           ← headless experiment scripts (navigation, chat, …)
│   └── findings/          ← chronological session findings
├── references/            ← glossary + source textbooks
├── docs/                  ← biology, current state, roadmap, guides, diagrams
├── webapp/                ← web dashboard
├── simulation_profiles/   ← 47 brain-region JSON profiles
└── tests/                 ← pytest suite (425 files)
```

---

## Glossary

A few terms recur in this project's documentation. In plain language:

| Term | What it means |
|---|---|
| **Spike / firing** | A neuron's brief electrical pulse — the basic unit of activity. The brain "computes" by which neurons spike, and when. |
| **Plasticity** | How connections (synapses) change strength as the network learns. Here it is local and timing-based, not gradient descent. |
| **Spike-timing-dependent plasticity (STDP)** | The main learning rule: a connection strengthens when the sending neuron fires just *before* the receiver, and weakens when the order reverses. |
| **Catastrophic forgetting** | The usual failure mode where a network learning something new overwrites what it already knew. Avoiding it is a core goal here. |
| **Refusal to fabricate** | The system's measured refusal to answer ("I don't know") when asked about something it was never taught, instead of inventing a wrong answer. |
| **Composition / binding** | Combining separate concepts into a structured fact ("the dog ate the apple" = who-did-what), and later pulling them back apart — computed in spikes. |
| **Model cortex** | The part of the system that stores concepts. A current effort is to make its internal codes carry *meaning-similarity*, learned from experience. |
| **Multi-seed** | A result re-run with several different random starting seeds, so it isn't a one-off fluke. Claims here are reported with the seeds used. |
| **Findings document** | A dated write-up of one research session's result — including negative ones — kept under `research/findings/`. |

---

## Documentation & further reading

- [QUICKSTART.md](QUICKSTART.md) — install, prerequisites, first run
- [`docs/biology.md`](docs/biology.md) — the modeled biology, in plain language
- [`docs/CURRENT-STATE.md`](docs/CURRENT-STATE.md) — what works today, technically
- [`docs/CHAT-DEMO-GUIDE.md`](docs/CHAT-DEMO-GUIDE.md) — every conversational demo
- [`docs/SCIENCE_ROADMAP.md`](docs/SCIENCE_ROADMAP.md) — where it's going
- [`docs/diagrams/README.md`](docs/diagrams/README.md) — how to read & regenerate the architecture diagrams
- [CONTRIBUTING.md](CONTRIBUTING.md) — how to contribute
- [CHANGELOG.md](CHANGELOG.md) — dated history

---

## Contributing

Contributions are welcome — bug fixes, new brain-region profiles,
biological-validation tests, and documentation especially. Please read
[CONTRIBUTING.md](CONTRIBUTING.md) and run the test suite first:

```bash
pytest tests/ -v
```

---

## How to cite

If you use this simulator in research, please cite the repository:

```bibtex
@software{neural_simulator,
  title  = {Neural Simulator: a GPU-accelerated biologically realistic
            spiking neural network simulator with real-time 3D visualization},
  author = {Thiberge, Daniel},
  year   = {2026},
  url    = {https://github.com/danthi123/neural-simulator}
}
```

---

## License

MIT — see [LICENSE](LICENSE).

## Mirrors

- GitHub: https://github.com/danthi123/neural-simulator
- Gitea: https://git.dant123.com/dant123/neural-simulator
