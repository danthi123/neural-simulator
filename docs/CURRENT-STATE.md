# Current State

A plain-language snapshot of what this project can do today, how it does it, and
what it honestly cannot do yet.

This document is the **authoritative current-state reference** for readers. It is
meant to be *holistically* current — a description of the project as it stands, not
an old snapshot with new notes bolted on. For the session-by-session journey of how
we got here, see [`research/findings/`](../research/findings/) (internal research
notes, including negative results).

**Last comprehensive rewrite:** 2026-07.

---

## What this project is

A **GPU-accelerated simulator for biologically realistic spiking neural networks**,
with a real-time 3-D view of the neurons firing. Unlike a conventional deep-learning
model, it learns the way brains are thought to:

- Individual neurons fire in real time, moment to moment.
- Connections change from the **millisecond timing of spikes** (spike-timing-dependent
  plasticity) and from **local co-activity** (Hebbian rules).
- A **dopamine-like reward signal** reinforces firings that led to useful behaviour.

There is **no backpropagation through a static graph, no supervised labels, and no
symbolic optimizer** inside the core brain. The learning is local and biological.

The distinguishing long-term goal is a **single simulated brain** that both **navigates
a world** and **holds a simple, grounded conversation**, with **every cognitive step
performed by spiking neurons** rather than ordinary program code. This is an **active
research project** aimed at an artificial-life "brain" that learns, grows, and can be
talked to. Where the biology genuinely cannot yet do something on this substrate, the
limit is **measured and reported honestly** — an honest negative is treated as a real
scientific finding, not a failure to hide.

**How to read this doc.** Capabilities below are tagged:

- **[Robust]** — validated across multiple random seeds and wired into the working
  system.
- **[Research-stage]** — demonstrated in validating experiments, often at smaller
  scale, and still part of active research. Real, but not a polished feature.

---

## The simulator (the foundation) — [Robust]

The engine underneath everything:

- **Spiking neuron models:** Izhikevich (2007 and a legacy variant),
  Hodgkin–Huxley (temperature-dependent), and Adaptive Exponential Integrate-and-Fire
  (AdEx) — plus a **resonate-and-fire** model with complex-valued synapses used by the
  conversational binding system.
- **Real-time 3-D visualization** (OpenGL) of the network firing, with a desktop GUI app.
- **A declarative brain-region grammar.** Brain regions and the pathways between them are
  described in configuration; the engine assembles a brain to match. This is how a single
  codebase builds a navigation brain, a conversation brain, or one combined brain.
- **A declarative neuromodulator subsystem** — dopamine-like and other modulators with
  their own concentration dynamics and configurable effects on the network.
- **Multiple concurrent plasticity rules:** spike-timing-dependent plasticity (STDP,
  Bi & Poo 1998), three-factor reward-modulated learning (Schultz-style dopamine), short-term
  plasticity (Tsodyks & Markram), homeostasis (Turrigiano), and structural plasticity.
- **An experiment / stimulus system** for classic paradigms — Pavlovian conditioning,
  reinforcement learning, frequency-response characterization, and more.
- **Checkpointing and "lineages"** so a brain can keep learning across sessions rather than
  starting fresh each run.
- **A biological-validation suite** that checks the substrate reproduces textbook phenomena:
  the STDP timing curve, excitation/inhibition balance, short-term-plasticity paired-pulse
  behaviour, gamma oscillations, and homeostatic rate regulation.

**Under the hood (verified):** the core engine (the `sim/` package) is **43 Python modules**;
the largest is the central simulation orchestrator. Two interchangeable backends run the same
code — **CuPy** (NVIDIA CUDA GPU) for speed and **NumPy** (CPU) for portability and continuous
integration — selectable with the `SIM_BACKEND` environment variable (`cupy` / `numpy` /
`auto`). **No GPU is required** to run on the NumPy backend. The engine scales to large
networks (roughly 10K–100K+ neurons) using sparse connectivity, though the individual task
brains described below are smaller. There are **472 test files** (mostly CPU-only),
**~1,250 headless research runners**, and **47 saved simulation profiles**. Python 3.10+;
**MIT-licensed**.

---

## What the brain can do today

### Navigation — [Robust]

One brain drives an agent through a 2-D gridworld toward a goal — including a goal that
**moves** on a schedule, which the agent must re-acquire. Everything between seeing and
acting is done by simulated neurons:

- A **spiking visual cortex** (Gabor/V1-style edge filters feeding higher visual areas)
  turns the gridworld image into a neural representation of what is where.
- A biologically-structured **basal-ganglia action-selection circuit** — the direct and
  indirect pathways, with dopamine — chooses which way to step.
- The move choice is made by a **spiking decision step**: competing neural populations
  race, integrating evidence like a working-memory accumulator, until the winner fires an
  all-or-none committing "burst." This spiking decision is now the default; an older,
  hand-coded pick-the-best-option step is retired to an optional comparison baseline.
- A **neural reward/value/dopamine core** reinforces the firings that reduced distance to
  the goal.

Performance is characterized across grid sizes and multiple random seeds, and doing the
decision in neurons instead of code carries a modest, honestly-reported extra cost in steps.

> **A note on older navigation numbers.** Some widely-copied navigation claims from earlier
> notes (a specific "X% better than baseline," a "navigates with no shortcuts / all cheats
> closed" configuration, and a cross-grid-size percentage comparison) were found on internal
> audit to be wrong or overstated — a favourable-seed selection, a config that still had a
> shortcut enabled, or a comparison of two different metrics. They are **not repeated here**.
> Navigation is described qualitatively on purpose.

### Grounded conversation, with a no-fabrication safeguard — [Robust]

The brain holds a simple, grounded conversation entirely on the spiking network:

- **Comprehension.** It parses a simple sentence into who-did-what-to-whom roles, and this
  works for both active and passive phrasings ("the dog chased the cat" / "the cat was
  chased by the dog" assign the same roles).
- **Memory and question answering.** It stores facts and answers *who* and *what* questions
  about them, handles **yes/no and negation**, does **simple multi-step reasoning** (following
  a chain of stored facts), and **tracks referents across turns** (a later "it" resolves to
  the earlier subject).
- **The no-fabrication safeguard.** When it has no stored fact matching a question, it
  **says it does not know** rather than guessing. This is the central honesty guarantee, and
  it is built in by construction — the fluent generator (below) is never even invoked when the
  brain decides to abstain.

This is validated at a **few-hundred-concept vocabulary, across multiple random seeds**, with
no fabricated answers.

### Learning meaning and discovering categories from experience — [Research-stage]

Rather than being handed a dictionary, the brain can learn from experience:

- **Learning word meaning by listening.** As it "hears" a stream of text, it builds
  word-meaning representations from context — words that appear in similar contexts end up
  represented similarly. This is unsupervised, with a plain local learning rule; there is no
  pre-processing of the text.
- **Discovering categories and simple taxonomies on its own.** From co-occurrences — or from
  actually *seeing* objects through the visual front end — it groups things into categories and
  even multi-level taxonomies (for example, learning that several things are a kind of "bird,"
  which is a kind of "animal") without being told the structure.
- **Inheritance with exceptions.** Once it has a taxonomy, it can **inherit** a property it was
  never directly told (a never-taught robin "can fly" because birds fly), while honouring
  **exceptions** (a penguin walks). You can then converse with it about what it discovered.

These are demonstrated in validating experiments (the emergent-language work) and are an
active research frontier rather than a finished feature.

### Fluent speech — two layers

Turning the brain's grounded decisions into English happens in two layers:

- **(a) A small, locally-trained language generator — [Robust, but a deliberate scaffold].**
  A compact generator (tens of millions of parameters — far smaller than a typical large
  language model, trained locally) supplies fluent English *phrasing only*. It does **not**
  decide what is true or whether to answer — the brain does that first, and the generator is
  never invoked when the brain abstains, so the no-fabrication safeguard holds. This generator
  is explicitly a **temporary scaffold** on the way to fully brain-produced speech.

- **(b) The brain's own spiking speech production — [Research-stage].** Increasingly, the
  brain's **own spiking circuitry** produces the actual words and their order for a **bounded
  set of sentence forms** — modelled on the human speech-production region (Broca's area). It
  even learns the sentence structure (which words are function words, what order slots go in,
  which slots a construction has) from a text stream rather than having it hand-written. This
  is real for a limited inventory of sentence shapes; open-ended prose is a frontier (see below).

### Development over simulated time, without forgetting — [Research-stage]

The brain can live a simple simulated life:

- It **forages under a hunger drive**, and it **perceives and remembers** the objects it
  encounters as it moves.
- Over simulated days it **grows its vocabulary and factual knowledge**, day over day, **without
  catastrophically forgetting** what it already knew.
- It **persists across restarts** (save and resume). A person can load the brain at a given
  "day" and talk to it about what it has lived through.

Validated across multiple random seeds in the artificial-life experiments; the *learned* spatial
policy behind survival still uses a simplified stand-in and is part of the ongoing research.

### One brain, one shared core — [Robust]

Navigation, the conversational parser, a planning / working-memory region (modelling prefrontal
cortex), the conversational binding system (which combines concepts into facts and reads them
back), a hippocampus-style memory, and a **shared dopamine reward/drive core** all run as **one
network on one update loop** — not separate programs stitched together. They are joined by
validated **cross-region synaptic links**. Demonstrated interactions include:

- A **spoken command can steer movement** (language routes to action).
- An object **seen while moving can be recalled and talked about later** (perception routes to
  memory and speech).
- A **hungry brain's raised dopamine tightens both its actions and its conversational
  confidence** — one drive modulating both halves of the same brain.

When the two halves are combined, the conversational behaviour (including the no-fabrication
safeguard) works unchanged, and navigation's live learning does not disturb the conversational
neurons. See [`ARCHITECTURE_nav_conv_merge.md`](ARCHITECTURE_nav_conv_merge.md) for the
architecture of the combined brain.

---

## Shipped/robust vs. research-stage — at a glance

| Capability | Status |
|---|---|
| The simulation engine, neuron models, plasticity, visualization, experiment system, validation suite | **Robust** |
| Checkpointing / continuous-learning lineages | **Robust** |
| Gridworld navigation (fully neural, spiking decision) | **Robust** |
| Grounded who/what conversation + no-fabrication safeguard | **Robust** |
| One shared brain (navigation + conversation) with cross-region synaptic routes | **Robust** |
| Small local generator for fluent phrasing (a temporary scaffold) | **Robust (scaffold)** |
| Learning word meaning by listening; discovering categories/taxonomies; inheritance with exceptions | **Research-stage** |
| The brain's own spiking speech production (bounded sentence forms) | **Research-stage** |
| Living/developing over simulated days + persistence | **Research-stage** |

---

## Current research frontiers

The project is doing **autonomous, biology-grounded research** toward the single-brain,
fully-spiking, conversational artificial-life goal. The near-term aim is to close the remaining
gaps so that **every step is done by spiking neurons on one brain**. The active frontiers:

1. **Open-ended fluent generation** — moving beyond a bounded set of sentence forms toward free
   conversation, produced by the brain's own circuitry rather than the temporary generator
   scaffold.
2. **Learned concept binding** — replacing today's fixed, hand-designed scheme for combining
   concepts into facts with one the brain actually *learns*.
3. **Resolving ambiguous references** — deciding which of several candidate things a bare
   pronoun refers to.
4. **Biological credit assignment** — a **dendrite-based** local learning rule (how a single
   neuron works out which of its inputs to strengthen) that does not rely on backpropagation.
   This is the likely enabler for open-ended generation and is the project's deepest open
   *neural* problem.
5. **Memory replay and imagination** — the brain internally replaying and recombining stored
   sequences (as the hippocampus does during rest) to support planning and imagination.

These are **open frontiers being actively worked on**, not solved features.

---

## Known limitations (honest)

**Task scope**
- Navigation uses a 4-direction action space (no diagonal moves) and small grids.
- Navigation handles a goal that moves on a schedule, but not multi-step compositional
  *planning*.
- Conversation is grounded in stored/learned facts; **open-domain, free-topic conversation and
  free open-world inference remain the field's unsolved walls**, and are managed here with
  domain constraints, grounded retrieval, and honest abstention rather than claimed as solved.
- Combining *two* attributes into one referent (e.g. "big red ball") is not yet reliable on the
  learned codes — a documented boundary.

**Scaffolding still in place** (being actively converted to neural mechanisms)
- The fluent generator is a conventional (non-spiking) model used for phrasing only.
- Some structure the brain uses is still hand-designed rather than self-organized; replacing it
  with developmentally self-organized structure is ongoing.

**Biological realism not yet modelled**
- No full developmental phases (synaptic pruning, layer formation) beyond what auto-growth
  provides.
- No protein-synthesis-dependent late-stage consolidation; no glia or neurovascular coupling;
  limited multi-time-scale plasticity.
- No smell/touch/sound modalities and no social interaction (joint attention) yet.

**Honest cost of purity**
- Doing every step in spiking neurons is slower than the equivalent ordinary code would be.
  Reducing that latency (so a fully-spiking conversation runs comfortably in real time) is an
  ongoing engineering effort; single-query answering is already fast on a desktop GPU.

---

## Reproducibility and performance

All random sources are seeded together, so the same `--seed` on the same hardware produces
bit-identical trajectories; a `--deterministic` option maximizes reproducibility. Long
research runs use the GPU (CuPy) backend; the NumPy backend is for portability, CI, and
low-end hardware. Concrete throughput depends heavily on network size and configuration, so
specific timings are best taken from the relevant runner rather than quoted here.

---

## Dependencies

- **Python 3.10+**
- **CuPy** (CUDA 11 or 12) for the GPU backend — optional; the NumPy CPU backend needs no GPU
- **NumPy**, **h5py**, **DearPyGui**, **PyOpenGL**
- *(Optional)* FastAPI + Uvicorn for the web console

An NVIDIA GPU with several GB of VRAM is recommended for the heavier runs; smaller networks
run on modest hardware or CPU-only.

---

## Where to go from here

| If you want to… | Read this |
|---|---|
| Run the simulator | [QUICKSTART.md](../QUICKSTART.md) |
| See the brain's layout as a diagram (plain language) | [`diagrams/brain_architecture_current.md`](diagrams/brain_architecture_current.md) |
| Understand the biology in plain language | [biology.md](biology.md) |
| Understand the combined navigation + conversation brain | [ARCHITECTURE_nav_conv_merge.md](ARCHITECTURE_nav_conv_merge.md) |
| Reproduce a specific result | [USER_GUIDE.md](../USER_GUIDE.md) |
| Modify or extend the codebase | [../CONTRIBUTING.md](../CONTRIBUTING.md) |
| Read the research findings chronologically | [`research/findings/`](../research/findings/) |
| Understand the guidelines Claude follows on this repo | [../CLAUDE.md](../CLAUDE.md) |
