# Current State

Last rewritten: 2026-08-03.

This is the concise status snapshot. It says what works today, what is still
temporary, and what blocks the next jump. For the plan, read
[ROADMAP.md](../ROADMAP.md). For evidence, read
[`research/findings/`](../research/findings/).

## One Sentence

Neural Simulator is a CPU/GPU spiking-brain simulator plus an active research
program trying to grow one grounded, continuously learning conversational brain.

## Solid Infrastructure

- **Simulation substrate.** Spiking neurons, synapses, sparse pathways, several
  neuron models, local plasticity, neuromodulators, checkpointing, and continuous
  stepping.
- **Backends.** NVIDIA GPU through CuPy and CPU through NumPy. CPU is slower but
  good for tests and parallel seed sweeps.
- **Brain regions.** Regions and pathways can run as one shared brain rather
  than separate scripts manually calling each other.
- **Research harness.** Many runners, raw artifacts, provenance sidecars, and
  claim/gate checks are already in place.
- **Visualization.** Desktop GUI, 3D visualization, and a web app exist for
  inspecting runs.

## Current Brain Capabilities

| Capability | Honest status |
|---|---|
| Navigation and action | Spiking loops for visual/gridworld navigation, action selection, motor output, and reward learning exist. Treat older headline scores cautiously because several were later corrected. |
| Simple conversation | The system can store simple facts, parse basic questions, answer from memory, and decline when no matching memory exists. This is narrow grounded dialogue, not open conversation. |
| Language production | Bounded spiking speech paths and larger recurrent language experiments exist. A mature 267M-parameter WKV checkpoint now has six-seed RF spiking-forward parity under the current test. Fluent open-ended language is still unfinished and partly scaffolded by conventional training. |
| Word and category learning | Local learning experiments show word-meaning structure, categories, taxonomies, inheritance, and exceptions at small scale. The strongest versions still need deeper sensory/action grounding. |
| Memory and replay | Episodic memory, completion, replay, consolidation, and directional replay reading have substantial tested support. The production path still has database-like pieces to replace. |
| Reward and value | Dopamine-like reward prediction and value signals are among the better grounded pieces. They can drive action learning and interact with other regions. |
| Affect | Persistent mood-like state and neuromodulator axes exist. Rich, graded, learned emotion that changes perception, memory, speech, and behavior remains open. |
| Curiosity | Curiosity circuits can seek learnable unknowns and avoid some unlearnable noise. Learning-progress memory is improving but not fully brain-native yet. |
| Self-monitoring and honesty | Confidence, attention, authorship, and source-monitoring pieces exist. The default-off production hook is moat-safe. A new spiking Hebbian path learns whether a complete proposition was externally experienced and feeds that support into the self-schema without reading an expected answer at inference. It downgraded all 46 stressed wrong recalls across six seeds, but unnecessarily hedged 5/133 correct recalls and still uses a separate, partly host-defined circuit. |
| One-brain integration | Several combined demos show navigation, conversation, memory, and modulation co-residing. The next step is a continuously running lived loop, not another combined demo. |

## What Is Still Scaffolded

These are useful stand-ins, not final biology:

- external or conventional language-model training for fluency;
- host-side query parsing, discourse planning, verification, or routing in older
  conversation paths;
- host-computed novelty, appraisal, confidence, or learning-progress scalars in
  some experiments;
- exact source metadata and engineered source-memory echoes retained as honesty
  comparisons, plus host-hashed proposition codes and explicit source events in
  the newer learned monitor;
- hand-designed concept codes, grammar frames, memory slots, or pathway weights;
- teacher/assistant systems used as an early social environment;
- measurement scripts that inspect internal state for evaluation.

The detailed public list is [SCAFFOLD-LEDGER.md](SCAFFOLD-LEDGER.md).

## Biggest Open Gaps

1. **Open-ended natural conversation.** The brain can answer and produce bounded
   speech, but it does not yet speak freely from a rich internal life.
2. **Grounded meaning.** Words need to share circuitry with what the brain sees,
   does, wants, and remembers.
3. **A complete live loop.** The project needs a small but continuous
   world/body/speech/learning loop where consequences change the brain.
4. **Self-based honesty.** Source support can now be learned in spiking synapses,
   but the source circuit, ACC-like monitor, self-schema, and speech decision are
   not yet one co-resident learned pathway driven by lived sensory activity.
5. **Developing emotion.** Current affect is useful but too narrow. The target is
   graded, learned, embodied affect that shapes behavior and speech.
6. **Continual learning without forgetting.** Consolidation pieces exist, but the
   full "learn from human interaction over time" story is not closed.
7. **Scaffold burn-down.** The codebase has accumulated shortcuts that must stay
   visible until replaced.

## Highest-Value Direction

The next major build should be a minimal grounded conversation loop. In parallel,
the new learned source signal should move onto the same brain as the confidence
and self-schema circuits:

1. Put the brain in a small world with a few objects, actions, needs, and a social
   partner.
2. Let internal state produce a reason to speak: curiosity, surprise,
   uncertainty, need, or affect.
3. Make the utterance refer to grounded internal representations.
4. Feed the result of the speech act back into reward, surprise, confidence, and
   learning.
5. Test whether removing internal state, grounding, source-monitoring, or
   contingent feedback breaks the behavior.

That is the shortest path from a collection of working mechanisms to a small
brain that says simple things of its own.

## Compute Status

The project should use available compute aggressively without making the final
brain datacenter-dependent.

- **Local GPU.** Best for heavy single runs, large spiking-forward checks, and
  GPU kernels.
- **Local CPU.** Best for tests, smoke runs, smaller de-risking, and docs/gates.
- **Mini-PC CPU pool.** Best for independent seeds and parameter sweeps when
  available.
- **Cloud GPU.** Optional burst capacity, not the normal development path.

The long-term target remains high-end consumer hardware and, eventually,
hardware-friendly sparse spiking computation.

## How To Interpret Claims

- A multi-seed result means behavior survived several random starts. It does not
  prove the whole faculty is human-like.
- A lesion, shuffle, or permutation control is often more important than a high
  score because it checks whether the credited mechanism is doing the work.
- A negative result is useful when it identifies which method failed and what
  should be tried next.
- A scaffolded result can still be valuable, but it should not be described as
  final brain-native cognition.

## Good Entry Points

- [README.md](../README.md) - plain project overview and run commands.
- [ROADMAP.md](../ROADMAP.md) - compact plan and priorities.
- [SCAFFOLD-LEDGER.md](SCAFFOLD-LEDGER.md) - temporary shortcuts and replacement
  paths.
- [HANDOFF.md](../HANDOFF.md) - autonomous development workflow.
- [Project charter](plans/2026-08-02-PROJECT-CHARTER-grounded-emergence-realignment.md)
  - current mission and anti-template realignment.
- [Structural mechanism map](plans/2026-08-02-structural-mechanism-map.md) -
  technical faculty map with biological references.
- [`research/findings/`](../research/findings/) - full chronological research
  record.
