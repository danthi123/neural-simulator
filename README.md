# Neural Simulator

Neural Simulator is an open research project attempting to grow an artificial
brain from spiking neurons and synapses. The long-term goal is one integrated
brain that starts small, develops through embodied and social experience, and
eventually holds natural conversation from its own perceptions, memories,
needs, emotions, and uncertainty.

This is not a finished mind or a chatbot wrapped in neuroscience terminology.
It is a simulator, a collection of tested brain mechanisms, and a research
program working to join those mechanisms into a continuously developing whole.

## What The Project Is Building

The target system has five defining properties:

- **Fully spiking:** cognition is carried by simulated neurons, synapses, and
  their changing activity.
- **One shared substrate:** specialized regions are welcome, but perception,
  memory, value, emotion, thought, and language must interact within one brain.
- **Grounded development:** the brain learns by sensing a world, acting in it,
  experiencing consequences, and interacting with people. Text can be part of
  that experience, but isolated text prediction is not the goal.
- **Causal inner state:** needs, emotions, curiosity, confidence, and memory must
  change what the brain attends to, learns, says, and does. They are not labels
  added to an answer after it has been generated.
- **Individual ownership:** early brains should run locally and grow only as
  learning requires it. The engineering target is high-end consumer hardware,
  with sparse and local computation that could later suit neuromorphic hardware.

The intended developmental loop is:

```text
world and people -> sensation -> internal state -> action or speech
                 -> consequence -> learning -> changed future behavior
```

Ordinary host code may provide the outside world, body interface, storage,
visualization, and scientific instruments. It must not remain a substitute for
a cognitive function that the simulated brain is meant to perform.

Read the [project charter](docs/PROJECT-CHARTER.md) for the commitments behind
this work and the [technical overview](docs/TECHNICAL-OVERVIEW.md) for the
architecture, current evidence, and major gaps.

## Current State

The simulation foundation is substantial. It supports central processing unit
and NVIDIA graphics processing unit execution, sparse spiking networks,
multiple neuron models, declared brain regions and pathways, several forms of
local plasticity, neuromodulation, homeostasis, structural change,
checkpointing, experiment runners, and visualization.

The cognitive system is much earlier:

| Area | Honest status |
|---|---|
| Shared brain | Multiple regions and pathways can occupy one simulation and update together. Several combined experiments use this architecture, and the current conversational work co-locates several faculties in one live loop. A remaining step merges two co-resident spiking bridges into a single substrate. |
| Grounded action | Navigation, action selection, reward learning, perception, memory, and replay have working research results, usually in constrained tasks. |
| Grounded communication | A six-seed result joins learned visual association, hunger, a request-or-silence choice, and a world consequence. A newer six-seed result learns a tiny external vocal convention with two intents and two referents and succeeds on both untrained cross-combinations. It still uses injected motor exploration, fixed neural channels, and a host listener; it is preverbal learning, not natural language. |
| Conversational integration (current frontier) | Work now wires validated faculties into a live fourteen-turn conversational test and asks whether the conversation itself improves. In a toy world (two agents, three actions, a small fact set) the brain gives grounded replies on in-domain turns and correctly stays silent on out-of-domain ones, which is the no-confabulation moat working rather than a gap. Landed and six-seed verified on the real chat: a sub-clausal moat check that drops ungrounded clauses the generator invents (confabulations three to zero); honest inner-state read-outs for affect, self-model, and a graded certainty band (functional read-outs, never felt states); an honest disclaimer when asked for a cause the brain has no faculty to compute; and grounded facts learned from a heard corpus (subject breadth two to nine) and, at small scale, taught by corrective interaction as spiking weight changes gated by a learned, now fully spiking no-confabulation check. Episodic recall of the prior dialogue runs on the spiking substrate. The sentence-forming generator is a declared articulation scaffold; several fact stores and self-report templates are named host scaffolds with brain-native replacements pending, and multi-fact continual learning is still open. |
| Language | The repository contains simple question answering and bounded spiking language experiments, plus a live conversational loop that composes honest grounded replies and honest silences in a toy world. Fluent, open-ended conversation grounded in ongoing life is not yet achieved. |
| Emotion and drives | Reward, value, persistent mood-like state, neuromodulator signals, and curiosity mechanisms exist in limited forms. A rich emotional system that develops through experience and broadly shapes behavior remains open. |
| Memory and self-monitoring | Episodic memory, replay, confidence, authorship, and source-monitoring mechanisms exist. The episodic composition seam is now mechanistically closed at the readout level: an emergent loop selects an assembly, one-shot plasticity forms the attractor, and an intrinsic per-cell dendritic plateau completes it cue-specifically (six-seed GO de-risk at one density, read during the cue). That spiking recall path now runs on the standard numpy substrate in the conversation's memory turn. Some fact content and self-report wording remain host scaffolds with brain-native replacements pending. |
| Growth | Structural plasticity and capacity-growth infrastructure exist. A brain-native policy that safely grows a whole developing brain is not yet complete. |

The central problem is integration. Passing a small test does not show that a
mechanism can serve its role in a living brain, so the current frontier is a
continuous-integration arc: validated faculties are wired into a live
conversational loop one dependency at a time and kept only when the conversation
itself measurably improves under causal controls and independent seeds. Running
the real conversation, rather than an isolated probe, is what exposes a
mechanism that was mis-scoped in isolation. Broader causal loops joining
perception, body state, memory, affect, communication, consequence, and learning
remain the goal; a first cluster of them now runs inside one conversational test.

For a dated status report, see [Current State](docs/CURRENT-STATE.md). For the
planned build order, see the [Roadmap](ROADMAP.md).

## Research Discipline

Temporary scaffolds are allowed to make a question testable, but every scaffold
must remain visible, have a brain-native replacement, and have a condition for
removal. Examples include host-written parsers, fixed concept codes, hand-set
pathways, conventional language training, and external teaching systems. See
the [Scaffold Ledger](docs/SCAFFOLD-LEDGER.md).

Capability claims are expected to survive causal controls, independent random
seeds when practical, and comparison with simpler explanations. Findings cite
raw artifacts and provenance. Negative, corrected, and superseded results stay
in the record because they constrain future work. The chronological evidence is
under [`research/findings/`](research/findings/).

## Run The Project

Start with [QUICKSTART.md](QUICKSTART.md) for installation and backend setup.

Run a small central-processing-unit test:

```bash
SIM_BACKEND=numpy python -m pytest tests/test_strict_step_errors.py -q
```

Launch the interactive simulator on a configured NVIDIA system:

```bash
python neural-simulator.py
```

Most scientific experiments are headless modules in `research/runners/`. They
are research instruments rather than polished product demos.

## Repository Guide

| Path | Purpose |
|---|---|
| `sim/` | Simulation engine, neuron and synapse state, learning, brain regions, neuromodulation, growth, and backends. |
| `experiment/` | Reusable stimulus, training, and readout support. |
| `research/runners/` | Executable experiments and diagnostic runs. |
| `research/findings/` | Dated interpretations of results, including failures and corrections. |
| `research/findings/raw/` | Raw measurements and run provenance. |
| `tools/gates/` | Automated checks for unsupported or stale research claims. |
| `ui/`, `viz/`, `webapp/` | Interactive control and visualization surfaces. |
| `docs/` | Current status, architecture, research standards, and historical plans. |

## Boundaries And License

The project studies functional mechanisms associated with cognition, affect,
self-monitoring, and communication. It does not claim consciousness, sentience,
felt emotion, human equivalence, or reliable general intelligence. Outputs from
research demos are experimental and must not be treated as authoritative or as
a safety-critical decision system.

The code is released under the [MIT License](LICENSE) and is provided without
warranty. Contributions should preserve the distinction between measured
behavior, interpretation, and speculation.
