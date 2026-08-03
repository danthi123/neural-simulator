# Neural Simulator

Neural Simulator is a research codebase for growing a simulated brain from
spiking neurons and synapses.

The aim is not to wrap a chatbot in neuroscience vocabulary. The aim is to build
one integrated brain that can eventually perceive a world, act in it, remember
what happened, develop emotion and curiosity, learn from people, and speak from
its own internal state.

This is active research. Some parts are solid infrastructure. Some are promising
but narrow. Some are temporary scaffolds that must be replaced by brain-native
mechanisms before they count as the real capability.

## The Goal

The long-term target is a fully spiking brain that:

- lives in a small world through sensors and a body;
- learns from interaction, not only from prewritten text;
- builds concepts tied to perception, action, memory, value, and emotion;
- develops persistent emotional state that changes attention, memory, speech,
  and behavior;
- has curiosity as a drive to learn what is learnable;
- can read useful parts of its own state, such as attention, confidence, and
  authorship;
- says it does not know, hedges, or asks when its grounding is weak;
- keeps learning and growing without wiping itself.

In ordinary terms: we want a small artificial brain that can grow toward the
conversational usefulness of a small language model, but through grounded lived
experience rather than text imitation.

The end state is one shared spiking substrate. Specialized regions are expected,
because real brains are specialized. The important rule is that perception,
attention, value, memory, affect, reasoning, self-modeling, and language should
be computed by simulated neurons and synapses. Host code is allowed for the
outside world, the body interface, files, visualization, and temporary scaffolds
that are explicitly tracked.

The project does not claim consciousness or felt emotion. It builds and measures
functional ingredients: access to information, self-report, confidence,
affective bias, learning drive, memory, and integrated behavior.

## Why This Is Not Just An LLM

A language model learns patterns in text and predicts likely continuations. This
project is trying to build the machinery beneath useful language: a stateful
world-facing brain that has something to perceive, want, remember, question, and
express.

That distinction matters. Earlier parts of this repo showed that narrow tests can
produce template-like behavior even when the mechanism is spiking. A corpus
predictor can sound fluent without grounded meaning. A no-fabrication rule can
decline unknowns without being a real self-model. The current roadmap therefore
focuses on a closed loop:

```text
sense -> internal state -> speech or action -> consequence -> learning
```

Language should become one action the brain can take, not the central shortcut
doing the brain's work.

## What Exists Today

| Area | Current state |
|---|---|
| Simulation engine | Mature research infrastructure with CPU and NVIDIA GPU backends, sparse spiking networks, multiple neuron models, plasticity, neuromodulators, checkpointing, and visualization. |
| Shared-brain wiring | Brain regions and pathways can be declared and run together in one simulation loop. This supports combined navigation, memory, reward, conversation, and modulation experiments. |
| Navigation and action | Spiking visual, basal-ganglia-like, motor, and reward loops exist for gridworld tasks. Some older headline claims were corrected, so use current findings before quoting numbers. |
| Simple conversation | The system can store simple facts, parse basic questions, answer from memory, and abstain when no matching memory exists. This is useful, but it is still narrow question answering, not open conversation. |
| Language circuits | Bounded spiking speech paths and larger local recurrent language experiments exist. Fluent open-ended speech remains the largest unfinished capability. |
| Memory and replay | The repo contains episodic memory, pattern completion, replay, consolidation, and directional replay experiments. The open goal is integrated reconstructive memory rather than database-like recall. |
| Affect and drive | Reward prediction, value, persistent mood-like state, neuromodulator axes, and curiosity circuits exist. Rich emotion that develops over time and colors speech/behavior remains open. |
| Self-monitoring | Confidence, attention, authorship, and source-monitoring circuits exist in research runners. A default-off conversation hook can use self-schema confidence to hedge; a newer independent RF source-memory echo catches a current high-confidence wrong-recall failure mode in six-seed tests. This is progress, not final biological honesty. |
| Research workflow | Findings are expected to cite raw artifacts, provenance, controls, and multi-seed runs when practical. Negative and corrected results remain part of the record. |

## The Main Open Problems

1. **Make a live loop.** The brain needs a small world, a body, internal state,
   speech/action, consequences, and learning all running together.
2. **Make language grounded action.** Speech should come from curiosity, affect,
   need, memory, or social intent inside the brain, not from a host-written query
   template.
3. **Ground meaning.** Words need to share neural machinery with what the brain
   sees, does, remembers, and values.
4. **Move honesty into the self-model.** Abstention and hedging should be driven
   by brain-state confidence and source-monitoring, not by a final host-side
   safety rule.
5. **Make emotion developmental.** Current affect is useful but too simple. The
   target is graded emotion learned from experience.
6. **Support continual learning and growth.** The system needs to keep learning
   through interaction without catastrophic forgetting.
7. **Stay ownable.** The target is high-end consumer hardware, with sparse,
   local, event-driven computation and a long-term path toward neuromorphic
   hardware.

## Temporary Scaffolds

Scaffolds are allowed when they help build the real brain, but they are not the
product. Important current scaffolds include:

- conventional language-model training used by some fluency experiments;
- host-side parsing, discourse planning, and routing in older conversation paths;
- host-computed novelty, appraisal, confidence, or learning-progress signals in
  some probes;
- exact source metadata and engineered source echoes used to make current
  honesty tests safer;
- hand-designed concept codes, grammar frames, memory slots, and pathway weights;
- an AI teacher used as an early social environment.

The replacement direction is always the same: move the computation into the
spiking brain, connect it to perception/action/affect/memory, and test it in the
closed loop. The public scaffold ledger is
[docs/SCAFFOLD-LEDGER.md](docs/SCAFFOLD-LEDGER.md).

## Run It

```bash
git clone https://github.com/danthi123/neural-simulator
cd neural-simulator
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

Run the GUI on a CUDA-capable machine:

```bash
python neural-simulator.py
```

Run a CPU conversation smoke test:

```bash
SIM_BACKEND=numpy python -m research.runners.chat_demo --seed 43
```

Run tests:

```bash
pytest tests/ -q
```

More setup detail is in [QUICKSTART.md](QUICKSTART.md).

## Repository Map

| Path | Purpose |
|---|---|
| `sim/` | Core simulation engine: neurons, synapses, pathways, plasticity, neuromodulators, checkpointing, and backends. |
| `experiment/` | Stimuli, training helpers, readouts, and reusable experiment support. |
| `research/runners/` | Headless experiment scripts where most new scientific work starts. |
| `research/findings/` | Dated research writeups, including negative, corrected, and superseded results. |
| `research/findings/raw/` | Raw artifacts and provenance sidecars. |
| `docs/` | Current state, roadmap support, biology notes, diagrams, and historical plans. |
| `tools/gates/` | Checks that catch unsupported claims, stale docs, missing provenance, weak controls, and repeated refuted ideas. |
| `ui/`, `viz/`, `webapp/` | Desktop and web visualization/control surfaces. |

## How To Read The Project

Start with:

- [ROADMAP.md](ROADMAP.md) for what we are building next;
- [docs/CURRENT-STATE.md](docs/CURRENT-STATE.md) for what works today;
- [docs/SCAFFOLD-LEDGER.md](docs/SCAFFOLD-LEDGER.md) for what is still temporary;
- [research/findings/](research/findings/) for the full evidence trail.

Progress is judged by whether a mechanism helps the whole brain, not only by
whether it passes an isolated test. A strong result should have raw artifacts,
provenance, controls, and an honest statement of what remains scaffolded.
