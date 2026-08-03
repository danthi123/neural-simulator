# Roadmap: One Grounded Spiking Brain

Last rewritten: 2026-08-03.

This is the compact plan of record. It keeps the project aimed at the actual
goal: one integrated spiking brain that learns from lived interaction and can
eventually speak from its own grounded internal state.

The detailed experiment record lives in `research/findings/`. The dense live lab
board lives in `GAP_CLOSURE_MISSION.md`. This document is for orientation and
priorities.

## North Star

Build a simulated brain that starts small, runs on ownable hardware, lives in a
simple world, learns through interaction, develops memory and emotion, becomes
curious about learnable unknowns, understands its own uncertainty, and grows
toward open-ended conversation.

This is an artificial-life and neuroscience-engineering project, not a chatbot
project. Language matters because it is one action a mind can take.

## Rules That Keep Us Pointed Correctly

1. **One shared spiking brain.** Specialized regions are fine; separate programs
   pretending to be regions are not the end state.
2. **Host code is not cognition.** The host may run the outside world, the body
   interface, files, visualization, and explicit scaffolds. Perception, memory,
   value, affect, reasoning, self-modeling, and speech should move into neurons
   and synapses.
3. **Grounding before polish.** A simple grounded sentence is worth more than a
   fluent sentence with no lived reference.
4. **Integration before proxy scores.** A mechanism is useful when it serves its
   role in the whole brain, not merely when it passes a narrow test.
5. **Scaffolds must burn down.** Every shortcut needs a name, a reason, a
   biological replacement, and a removal trigger.
6. **Performance matters.** Faithful biology may be slow, but lazy slow code is
   not faithfulness. Prefer sparse, local, event-driven computation.
7. **Be honest about claims.** Build and measure functional correlates of mind;
   do not assert subjective experience.

## Where The Project Is Now

The simulator is strong enough to support real work: CPU and GPU backends, sparse
spiking networks, brain-region wiring, neuromodulators, local plasticity,
checkpointing, visualizers, many tests, and a large research harness.

The brain pieces are real but incomplete. Navigation, memory, replay, reward,
curiosity, affect-like state, simple conversation, and self-monitoring all have
validated results. The correction is that many were validated in narrow settings
or with scaffolds. They are building blocks, not finished human-like faculties.

The immediate crux is a minimal lived loop:

```text
small world + body + internal state + speech/action + consequence + learning
```

Without that loop, language keeps sliding back into text prediction and memory
keeps sliding back into lookup.

## Phase 0: Keep The Record Honest

Goal: make drift hard.

Build and maintain:

- concise public docs for goal, status, and roadmap;
- a scaffold ledger that names temporary shortcuts and their replacements;
- finding templates that require role-in-the-whole, artifacts, provenance, and
  controls;
- gates that catch stale summaries, unsupported claims, missing artifacts, and
  re-proposed dead ends;
- clear separation between public orientation docs and dense lab boards.

Current status: partially in place. The README, roadmap, current-state snapshot,
and scaffold ledger are the public spine. The gates and finding workflow exist,
but they still need stricter role-in-the-whole enforcement.

## Phase 1: Build The Minimal Lived Loop

Goal: one small brain runs continuously in a simple environment.

Build:

- a small world with objects, actions, simple needs, and a social partner;
- a body interface where host code only enacts motor outputs and provides
  sensory input;
- a continuous shared loop rather than isolated experiment calls;
- internal causes for speech, such as curiosity, surprise, uncertainty, need, or
  affect;
- feedback from the world or partner that changes future behavior.

Pass condition:

The same external prompt should not be enough to determine the utterance. Speech
must depend on internal state, recent experience, and learned grounding.

## Phase 2: Make Language A Grounded Action

Goal: words flow into and out of grounded brain state.

Build:

- comprehension from words or sounds into grounded concepts;
- overlapping word, percept, action, and affect assemblies;
- a preverbal-message path where internal state chooses what is worth saying;
- a speech-production path that turns that message into ordered words and later
  articulation-like motor output;
- a contingent teacher loop where success, correction, and misunderstanding
  reshape language.

Current assets:

- bounded spiking speech paths;
- concept-pool to word-output pathways;
- basic question parsing and fact answering;
- early contingent-teacher learning experiments;
- larger recurrent language checkpoints used to test local spiking-forward
  fidelity.

Main risk: a corpus-trained generator can look like progress while still being
ungrounded. It can remain as a scaffold only when clearly named.

## Phase 3: Make Uncertainty Brain-Native

Goal: the brain answers, hedges, asks, or abstains because its own self-model and
source-monitoring state support that behavior.

Build:

- a self-schema region that reads attention, confidence, and authorship;
- metacognitive monitors that predict whether first-order answers are reliable;
- source-monitoring that distinguishes self-generated, heard, remembered, and
  uncertain content;
- routing from confidence/source state into speech behavior;
- tests with missing facts, familiar-but-wrong facts, source mismatches, and
  ambiguous memories.

Current assets:

- familiarity-based no-fabrication behavior;
- self-schema, attention, authorship, and metacognition runners with lesion and
  permutation controls;
- a default-off production conversation hook that can hedge matched answers
  without weakening the hard unknown-fact moat;
- an exact source-metadata floor that caught the current mismatch failure mode;
- a newer independent RF source-memory echo that catches the same six-seed
  stressed failures without reading the exact source fact.

Open point:

The RF source echo is still engineered and written at store time. The next step
is a learned, plastic source-monitoring circuit feeding the self-schema.

## Phase 4: Make Memory Predictive And Reconstructive

Goal: memory behaves like memory, not like a database.

Build:

- episodic memory for lived events with self/other/source tags;
- hippocampal completion and replay for consolidation and imagination;
- cortical memory that stores distributed patterns without fixed fact slots;
- reconsolidation so corrected memories update rather than duplicate;
- retrieval that can return ambiguity instead of forcing one answer.

Current assets:

- pattern completion, replay, consolidation, and directional replay experiments;
- wake/sleep round-trip infrastructure;
- category and semantic-memory experiments;
- known boundaries where exact stores or frozen reads were too scaffolded.

## Phase 5: Develop Affect, Drive, And Curiosity

Goal: emotion and motivation become continuous internal forces that shape what
the brain notices, remembers, learns, says, and does.

Build:

- graded valence and arousal rather than a binary mood latch;
- learned appraisal from events, goals, social feedback, and body state;
- interoceptive/body signals tied into affect;
- affective bias on attention, memory, speech choice, and action vigor;
- curiosity based on learning progress over time, not raw novelty alone.

Current assets:

- dopamine-like reward prediction and value circuits;
- persistent affect-like state with causal behavior effects in test settings;
- dissociable neuromodulator axes;
- curiosity circuits that can prefer learnable unknowns and avoid some
  unlearnable noise.

## Phase 6: Support Continual Learning And Growth

Goal: the brain keeps learning from interaction without wiping itself, and grows
only as it needs capacity.

Build:

- developmental growth of neurons, regions, and connections;
- consolidation that protects old knowledge while admitting new experience;
- local learning rules that work at useful scale on the chosen spiking substrate;
- teacher-as-caregiver scaffolding that fades over time;
- resource-aware scheduling across local CPU, local GPU, and available CPU pool
  machines.

Hardest science:

Deep local credit assignment on real spikes remains a frontier. The practical
route is to use local input learning, recurrent reservoirs, shallow readouts,
one-shot memory, replay, and consolidation where they work, while continuing to
measure the deeper learning-rule boundary honestly.

## Phase 7: Scale While Staying Ownable

Goal: grow toward richer conversation without becoming datacenter-only.

Build:

- larger grounded vocabularies and richer sensory/social experience;
- sparse and event-driven implementations of heavy pathways;
- GPU kernels that preserve the model rather than changing it;
- CPU-pool fanout for independent seeds and parameter sweeps;
- explicit memory/runtime/throughput accounting;
- design choices compatible with future neuromorphic hardware.

Success means the brain becomes larger because experience and growth require it,
not because a giant static model was preallocated.

## Highest Priority Work Now

1. **Finish the large language-circuit spiking-forward promotion.** Complete the
   multi-seed RF spiking-forward check for the local recurrent language
   checkpoint and update the finding only when the artifact exists.
2. **Burn down the honesty scaffold.** The production self-schema hook is safe,
   and the independent source-memory echo is a strong interim step. The next
   work is learned/plastic source-monitoring that feeds the self-schema without
   exact source labels or engineered echoes.
3. **Build the minimal grounded speech-action loop.** Let curiosity, surprise,
   need, or affect cause a simple utterance; feed the result back into learning.
4. **Keep the scaffold ledger current.** New shortcuts should never enter the
   repo unnamed.
5. **Use compute in parallel.** Keep the GPU on the highest-value long run while
   CPU lanes handle independent seeds, tests, docs, and smaller de-risking work.

## Short Success Target

A narrow but real prototype:

- one brain;
- one small world;
- a body;
- grounded perception of a few objects and actions;
- internal needs, affect, and curiosity;
- a simple self-confidence/source read;
- speech caused by internal state;
- interaction with a teacher or person that changes future speech and behavior;
- honest abstention or asking when grounding is missing.

This prototype may speak simply. It does not need to sound like a large language
model yet. It needs to be stateful, grounded, learning, internally driven, and
integrated.
