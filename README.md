# Neural Simulator

Neural Simulator is a research codebase for building a simulated brain from
spiking neurons and synapses. The long-term aim is not to wrap a chatbot in
brain-themed code. The aim is to grow one integrated, biologically grounded
system that can eventually perceive, act, remember, develop emotions, learn from
people, and speak from its own internal state.

The project is active research. Some pieces are strong and well tested. Some are
temporary scaffolds. Some earlier results are useful but were aimed at narrower
tests than the current goal. This README is the plain-language front door; the
detailed experiment record lives in [`research/findings/`](research/findings/).

## The Goal

Build one simulated spiking brain that:

- lives in a small world through sensors and a body;
- learns from interaction instead of only from prewritten text;
- forms grounded concepts tied to perception, action, memory, and affect;
- develops persistent emotional state that changes attention, memory, speech,
  and behavior;
- has curiosity as a drive to learn what is learnable;
- can read parts of its own state, including attention, confidence, and
  authorship;
- says "I do not know" or asks when its grounding is weak instead of fabricating;
- grows over time while keeping earlier memories and skills.

The end state is one fully spiking brain on a shared substrate. Dedicated brain
regions are fine, because real brains are organized that way. The important rule
is that perception, attention, value, memory, emotion, reasoning, self-modeling,
and language must be computed by the simulated neurons and synapses, not by
ordinary host-side shortcuts. Host code is allowed for the world, the body, file
I/O, visualization, and temporary training scaffolds that are explicitly tracked.

The project does not claim consciousness or felt emotion. It builds and measures
functional correlates: access to information, self-report, confidence, affective
bias, learning drive, and integrated behavior. Whether subjective experience has
appeared is treated as the long-horizon wager, not as a claim we can casually
assert.

## Why This Is Not Just An LLM

A large language model learns statistical patterns in text and generates likely
continuations. This project is trying to build the process underneath language:
a world-facing brain that has something to perceive, want, remember, feel,
question, and express.

The current lesson of the project is blunt: passing a narrow test can produce
template-like behavior even when the mechanism is spiking and learned. A language
network trained only to predict a corpus is still a corpus predictor. A
no-fabrication rule bolted on at the edge is still a rule. The roadmap has been
realigned around the closed loop: senses, internal state, action or speech,
consequence, learning, and back again.

## What Exists Today

| Area | Current state |
|---|---|
| Spiking simulation engine | Mature research infrastructure. It supports GPU via CuPy and CPU via NumPy, sparse connectivity, multiple neuron models, neuromodulators, plasticity, checkpointing, and visualization. |
| Brain-region framework | Regions and pathways can be declared and wired into one shared simulation loop. This supports navigation, conversation, memory, reward, and combined-brain experiments. |
| Navigation | The system has spiking visual, decision, basal-ganglia, motor, and reward loops for gridworld navigation. Older shortcut-closed claims were audited; use current findings before quoting old numbers. |
| Grounded question answering | The brain can store and retrieve simple facts, parse basic who/what and yes/no questions, handle some grammar variants, and abstain when the matching memory is absent. |
| Word and category learning | Several experiments show local, experience-driven formation of word meaning, categories, simple taxonomies, and inheritance with exceptions. These are promising but still small-scale. |
| Memory and replay | The repo contains hippocampus-style episodic memory, replay, consolidation, directional replay readers, and several tested routes for moving memory toward cortical storage. Some older memory claims were corrected or retired. |
| Affect and drive | Reward prediction, value, persistent mood-like state, dissociable neuromodulator axes, and basic curiosity circuits exist. Rich graded emotion, appraisal, and speech coloring remain open. |
| Self-monitoring | Separate test runners show circuits that can report confidence, attention, and self-related state with lesion controls. These are not yet fully integrated into the production conversation loop. |
| Language generation | There are bounded spiking speech generators and larger local language-circuit experiments. Fluent open-ended conversation is still the largest gap. A conventional generator remains a temporary phrasing scaffold in some demos. |

Current repo scale, measured in this checkout: 43 `sim/` modules, 489 test
files, 1389 research runners, 1938 dated finding docs, and 46 saved simulation
profiles.

## The Main Open Problems

1. **Close the whole loop.** The project has many validated pieces, but the next
   step is making a small brain live continuously: perceive, form internal state,
   speak or act, receive consequences, and learn from them.
2. **Make language an action.** Speech should come from an internal reason to
   communicate, not from a query string or text-completion objective.
3. **Ground meaning in lived experience.** Words need to point to the same neural
   assemblies used for seeing, acting, remembering, and valuing.
4. **Move honesty into the self-model.** "I do not know" should emerge from
   confidence and grounding signals inside the brain, not remain a separate
   host-side guard.
5. **Make affect richer and developmental.** The current mood/value machinery must
   become graded, learned, embodied, and able to shape speech and behavior.
6. **Support continual learning and growth.** The brain must learn from a stream of
   interaction without overwriting earlier knowledge, using consolidation and
   developmental growth.
7. **Scale without becoming datacenter-only.** The target is high-end consumer
   hardware, with sparse, local, event-driven computation and a long-term path
   toward neuromorphic hardware.

## Temporary Scaffolds

Scaffolds are allowed when they help develop the real brain, but they are not the
product. They must be tracked and replaced.

Current important scaffolds include:

- conventional language-model training used by some language experiments;
- host-side parsing and discourse planning in older conversation paths;
- host-computed novelty, appraisal, or confidence signals in some probes;
- hand-designed concept codes, weights, grammar frames, and memory layouts;
- the AI teacher used as an early social environment.

The replacement direction is the same in each case: move the computation into the
spiking brain, ground it in perception/action/affect/memory, and test it in the
whole loop.

## Running It

```bash
git clone https://github.com/danthi123/neural-simulator
cd neural-simulator
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
python neural-simulator.py
```

Use the CPU backend when CUDA is unavailable:

```bash
SIM_BACKEND=numpy python -m research.runners.chat_demo --seed 43
```

Interactive conversation demo:

```bash
python -m research.runners.chat_repl --mode tier1 --seed 43 \
  --save-bridge simulation_checkpoints_h5/repl_tier1.simstate.h5
```

Run tests:

```bash
pytest tests/ -v
```

More setup detail is in [QUICKSTART.md](QUICKSTART.md).

## Repository Map

| Path | Purpose |
|---|---|
| `sim/` | Core simulation engine: bridge, neurons, synapses, plasticity, regions, neuromodulators, checkpointing. |
| `experiment/` | Stimuli, readouts, training utilities, and reusable experiment helpers. |
| `research/runners/` | Headless experiment scripts. Most scientific work enters here before being promoted into shared code. |
| `research/findings/` | Dated experiment writeups, including negative and corrected results. This is the audit trail. |
| `docs/` | Current state, biology notes, diagrams, plans, and historical docs. |
| `tools/gates/` | Automated checks that block unsupported claims, stale documentation, missing provenance, weak controls, and other known failure modes. |
| `ui/`, `viz/`, `webapp/` | Desktop and web visualization/control surfaces. |

## How Progress Is Judged

The project values honest, reproducible progress over impressive phrasing.

- Results should run across multiple random seeds when practical.
- Claims need raw artifacts and provenance.
- Controls should remove, lesion, scramble, or permute the mechanism being
  credited.
- A passed isolated test is a floor, not proof that a human-like faculty exists.
- Every mechanism needs a short statement of its role in the whole brain.
- Every scaffold needs a named biological replacement and a burn-down trigger.

Start with [ROADMAP.md](ROADMAP.md) for the plan,
[docs/CURRENT-STATE.md](docs/CURRENT-STATE.md) for capability status, and
[HANDOFF.md](HANDOFF.md) for the development workflow.
