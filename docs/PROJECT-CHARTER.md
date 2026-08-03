# Project Charter

## Mission

Build a growing artificial brain whose capabilities arise from one shared
spiking neural system developing through embodied and social interaction.

The brain should begin small enough to run locally. It should learn from what it
perceives, does, feels, remembers, and is taught; form new connections and gain
capacity as development requires; and grow toward fluent, open-ended
conversation. Its speech should express its own grounded internal state rather
than imitate likely text or retrieve a prepared response.

This is a long-term research aim, not a description of the current system.

## What Success Means

The project succeeds by closing a continuing causal loop:

```text
perception -> internal state -> choice -> action or speech
           -> consequence -> learning -> later change
```

A capability counts only when it plays its required part in that loop. A memory
system must influence later decisions. An emotion system must alter attention,
learning, recall, speech, or action. An uncertainty system must change whether
the brain asserts, qualifies, asks, or investigates. A language system must
express meanings that already exist in the brain's experience and current
state.

Isolated benchmark success is evidence about a component. It is not evidence
that the whole faculty has been achieved.

## Architectural Commitments

### One fully spiking brain

The end state is one substrate of simulated neurons and synapses. Brain regions
may be specialized and use different cell types, time scales, connectivity, and
learning rules. They must still communicate as regions of one continuously
running brain, through neural activity and shared modulatory state.

Perception, attention, value, motivation, emotion, memory, planning,
self-monitoring, language choice, and language production belong inside that
substrate. A conventional program performing one of those functions is a
temporary scaffold, not the completed capability.

### A strict host boundary

Host software may:

- simulate the environment and body;
- turn external events into sensory input;
- enact neural motor output in the environment;
- save state and provide visualization;
- measure, perturb, and compare experiments;
- provide an explicitly temporary teaching or research scaffold.

Host software must not be credited as brain function. If it parses meaning,
chooses an intent, computes an emotion, selects a fact, plans an answer, or
decides confidence, that contribution must be named and tracked until a neural
mechanism replaces it.

### Grounded development

The main source of learning is lived interaction. The brain should encounter a
world through a body, experience needs and consequences, and learn with social
partners. Language is part of this world: words become meaningful by sharing
neural structure with perception, action, memory, affect, and social outcomes.

An artificial teacher may help establish an early curriculum. It must act as an
external social environment, not secretly supply the brain's reasoning or
answers. Teacher support should diminish as learning becomes self-sustaining and
human interaction becomes sufficient.

### Emotion and motivation are causal neural state

Emotions, drives, and curiosity are not output styles. They must be persistent,
graded neural processes shaped by body state, prediction, memory, and
experience. They should alter what is noticed, remembered, learned, chosen, and
said. Development should produce individual history and changing dispositions,
not a fixed menu of named moods.

### Growth is earned

The brain starts with limited capacity. New synapses, neurons, or regions should
be added because experience and measurable demand justify them. Growth must
preserve useful learning, remain stable, and avoid allocating a datacenter-scale
network before it is needed.

The practical target is high-end consumer hardware. Sparse activity, local
learning, limited data movement, and event-oriented computation are design
requirements. They also preserve a long-term route to energy-efficient
neuromorphic hardware.

## Scaffold Policy

Scaffolds are sometimes necessary to isolate a question or make early behavior
possible. They are technical debt with an explicit purpose, not permanent
architecture.

Every scaffold must record:

1. the function it temporarily supplies;
2. why the present implementation is not brain-native;
3. the biological or neural replacement being pursued;
4. the evidence required to remove it;
5. a test that reveals continued dependence on it.

New work should reduce total scaffold dependence or clearly enable a named
replacement. A more fluent result is not progress if it moves additional
cognition into the host. The current inventory is maintained in the
[Scaffold Ledger](SCAFFOLD-LEDGER.md).

## Scientific Standard

Research is organized around falsifiable claims.

Before building a mechanism, state what it must contribute to the whole brain
and what observation would show the proposed mechanism is unnecessary or wrong.
After building it:

- preserve raw measurements and enough provenance to reproduce the run;
- test multiple random initializations when the claim depends on learning;
- remove, silence, shuffle, or replace the proposed cause;
- compare against simpler mechanisms and host-side leakage;
- test transfer beyond the exact training case;
- report scope, scaffolds, failures, and unresolved alternatives;
- retain negative and corrected results as part of the evidence record.

A result that survives a narrow test remains narrow. Promotion requires evidence
that the mechanism affects the integrated loop in the predicted way.

## Development Priorities

Near-term work should broaden the smallest complete lived loop: several
percepts, needs, intents, consequences, memories, and reasons to communicate in
one continuous brain. The next gains should come from learned relationships and
causal interaction, not additional fixed decoders.

Medium-term work should join grounded language, continual memory, developing
affect, curiosity, self-monitoring, social learning, and safe structural growth.
The brain should keep learning through ordinary interaction without erasing its
history.

Long-term work should produce fluent conversation from a rich world model and
self-model, reduce external teaching and scaffolding, and improve efficiency so
the developed system remains individually ownable.

## Epistemic And Safety Boundaries

The project may investigate functional correlates of consciousness,
self-awareness, emotion, and agency. Behavioral similarity or a neural analogy
does not establish subjective experience. Documentation must distinguish
measured behavior from interpretation and interpretation from speculation.

The software is experimental. It does not provide reliable factual authority,
clinical judgment, autonomous safety guarantees, or evidence of human-level
intelligence. As capabilities grow, evaluation must include misuse, control,
privacy, welfare, and shutdown considerations rather than treating them as
future documentation tasks.

The repository is distributed under the [MIT License](../LICENSE), without
warranty.
