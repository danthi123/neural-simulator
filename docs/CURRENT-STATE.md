# Current State

Last rewritten: 2026-08-03.

This document is the concise status snapshot. It describes what the project can
currently do, what remains scaffolded, and what is still open. For the plan, read
[ROADMAP.md](../ROADMAP.md). For the detailed audit trail, read
[`research/findings/`](../research/findings/).

## One Sentence

Neural Simulator is a GPU/CPU spiking-brain simulator plus an active research
program trying to grow one grounded, continuously learning conversational brain.

## What Is Solid Infrastructure

- **Simulation substrate.** The core engine supports spiking neurons, synapses,
  sparse pathways, several neuron models, plasticity rules, neuromodulators,
  checkpointing, and continuous stepping.
- **Backends.** The same model can run on NVIDIA GPUs through CuPy or on CPUs
  through NumPy. The CPU path is slower but useful for tests and parallel seed
  sweeps.
- **Brain regions and pathways.** Regions can be declared and wired together as
  one brain rather than hand-calling separate modules.
- **Research harness.** The repo has many runners, raw artifacts, provenance
  files, and gates that check claims before results are banked.
- **Visualization and control.** A desktop GUI, 3D visualization, and a web app
  exist for inspecting runs.

Measured in this checkout: 43 core `sim/` modules, 489 test files, 1389 research
runners, 1938 finding documents, and 46 simulation profiles.

## Current Brain Capabilities

| Capability | Honest status |
|---|---|
| Navigation | Spiking circuits can support visual/gridworld navigation, action selection, motor output, and reward learning. Some older headline numbers were later corrected, so current findings should be checked before quoting a specific score. |
| Simple grounded conversation | The brain can store simple facts, parse basic questions, answer from memory, and decline when it lacks a matching fact. This is useful but still more like a narrow grounded dialogue system than open conversation. |
| Word/category learning | Local learning experiments show word-meaning structure, categories, taxonomies, inheritance, and exceptions at small scale. The strongest versions still need deeper grounding in perception and action. |
| Language production | Bounded spiking speech paths exist. Fluent open-ended language still relies partly on temporary conventional training or small corpus-prediction setups. The strategic goal is to make speech an internally caused communicative action. |
| Memory and replay | Episodic memory, completion, replay, consolidation, and directional replay reading have substantial tested support. The unfinished part is reconstructive, integrated memory that predicts and consolidates without database-like shortcuts. |
| Reward and value | Dopamine-like reward prediction and value signals are among the better grounded pieces. They can drive action learning and interact with other regions. |
| Affect | Persistent mood-like state and neuromodulator axes exist. Rich, graded, learned emotion that changes perception, memory, speech, and behavior remains open. |
| Curiosity | Basic novelty/learning-drive circuits exist and can avoid some unlearnable noise. Learning-progress tracking over time is still being strengthened. |
| Self-monitoring | Separate circuits can read confidence, attention, and self-related state in test runners. A default-off production hook now preserves the hard moat and downgrades low-confidence wrong recalls. A named source-consistency scaffold catches the current high-confidence mismatch failures, but the final version still needs a brain-native learned correctness/source-monitoring signal. |
| One-brain integration | Several merged-brain demos show navigation, conversation, memory, and shared modulation co-residing. The next step is a continuously running lived loop, not just a combined demo. |

## What Is Still Scaffolded

These are useful stand-ins, not final biology:

- external or conventional language-model training for fluency;
- host-side query parsing, discourse planning, verification, or routing in older
  conversation paths;
- host-computed novelty, appraisal, confidence, or learning-progress scalars in
  some experiments;
- exact composer source metadata used as a temporary safety floor for known-fact
  honesty;
- hand-designed concept codes, grammar frames, memory slots, or pathway weights;
- teacher/assistant systems used as an early social environment;
- measurement scripts that read internal state for evaluation.

Evaluation instruments are allowed. The problem is when an instrument becomes the
thing doing cognition.

## The Biggest Open Gaps

1. **Open-ended natural conversation.** The brain can answer and produce bounded
   speech, but it does not yet speak freely from a rich internal life.
2. **Grounding language in perception, action, and affect.** Words need to share
   circuitry with what the brain sees, does, wants, and remembers.
3. **A complete live loop.** The project needs a small but continuous world/body/
   speech/learning loop where consequences change the brain.
4. **Self-based honesty.** Abstention and hedging are partly wired into the
   self-model now. A scaffold can catch source mismatches, but the confidence
   signal still needs to predict correctness from brain activity rather than
   exact metadata.
5. **Developing emotion.** Current affect is useful but too narrow. The target is
   graded, learned, embodied affect that shapes behavior and speech.
6. **Continual learning without forgetting.** The project has consolidation pieces,
   but the full "learn from human interaction over time" story is not yet closed.
7. **Scaffold burn-down.** The codebase has accumulated many shortcuts. They need
   a public ledger and replacement path.

## Current Highest-Value Direction

The highest-value next build is not another isolated language benchmark. It is a
minimal grounded conversation loop:

1. Put the brain in a small world with a few objects, actions, needs, and a social
   partner.
2. Let internal state produce a reason to speak, such as curiosity, surprise,
   uncertainty, need, or affect.
3. Make the utterance refer to grounded internal representations, not just text.
4. Feed the result of the speech act back into reward, surprise, confidence, and
   learning.
5. Test whether removing the internal state, grounding, or contingent feedback
   breaks the behavior.

That loop is the shortest path from "collection of working mechanisms" to
"small brain that says simple things of its own."

## Compute Status

The project is designed to use all available compute without making the final
brain datacenter-dependent.

- Local GPU: NVIDIA/CUDA via CuPy for heavy single runs and large spiking-forward
  checks.
- Local CPU: NumPy backend for tests, smoke runs, and development.
- Mini-PC CPU pool: useful for independent seeds and parameter sweeps.
- Cloud GPU: optional and not currently part of the normal local loop.

The long-term target remains high-end consumer hardware, with sparse and local
computation that could eventually map to neuromorphic hardware.

## How To Interpret Claims

- A multi-seed result means the measured behavior survived several random
  initializations. It does not by itself prove the whole faculty is human-like.
- A lesion, shuffle, or permutation control is more important than a high score:
  it checks whether the credited mechanism is actually doing the work.
- A negative result is useful when it identifies which method failed and what
  should be tried next.
- A scaffolded result can still be valuable, but it should not be described as
  final brain-native cognition.

## Good Entry Points

- [README.md](../README.md) - plain project overview and run commands.
- [ROADMAP.md](../ROADMAP.md) - compact plan and priorities.
- [HANDOFF.md](../HANDOFF.md) - workflow for autonomous development.
- [2026-08-02 project charter](plans/2026-08-02-PROJECT-CHARTER-grounded-emergence-realignment.md) - the
  current goal realignment.
- [2026-08-02 structural mechanism map](plans/2026-08-02-structural-mechanism-map.md) - technical faculty map with
  biological references.
- [`research/findings/`](../research/findings/) - full chronological research
  record.
