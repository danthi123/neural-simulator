# Roadmap: One Grounded Spiking Brain

Last rewritten: 2026-08-02.

This roadmap is the compact plan of record. It replaces the older habit of
tracking progress as a long list of narrow capability tests. Those tests still
matter, but they are no longer the goal. The goal is an integrated brain that can
learn and speak from its own grounded life.

Detailed lab history remains in `research/findings/`. Dense implementation notes
remain in `GAP_CLOSURE_MISSION.md`, `HANDOFF.md`, and `docs/plans/`.

## North Star

Build one simulated spiking brain that starts small, lives in a simple world,
learns through interaction, develops memory and emotion, becomes curious about
what it can learn, understands its own uncertainty, and eventually holds
free-flowing conversation that is genuinely its own.

This is an artificial-life project, not a chatbot project. Language matters
because it is one action a mind can take, not because text benchmarks define the
mind.

## Non-Negotiable Direction

1. **One shared spiking substrate.** Brain regions may specialize, but they remain
   parts of one simulated brain communicating through spikes and synapses.
2. **No permanent host-side cognition.** Ordinary code may implement the world,
   body, files, visualizers, and temporary scaffolds. Perception, attention,
   reward, memory, affect, reasoning, self-modeling, and speech must move into
   the brain.
3. **Grounded before fluent.** Words must be tied to the brain's own perception,
   action, affect, and memory. A smooth sentence without grounding is a wrong
   target.
4. **Integration before proxy perfection.** A narrow result is useful only if the
   mechanism can serve its role in the whole loop.
5. **Scaffolds burn down.** Every temporary shortcut needs a named biological
   replacement and a trigger for removing it.
6. **Performance matters.** Faithful biology can be slow, but lazy design is not
   faithfulness. Prefer sparse, local, event-driven computation.
7. **No unsupported consciousness claims.** Build and measure functional
   correlates. Do not assert subjective experience.

## Current State In One Page

**Strong foundation.** The simulator itself is capable: GPU and CPU backends,
spiking neurons, sparse connectivity, region/pathway wiring, neuromodulators,
local plasticity, checkpointing, visualizers, and a large test/research runner
surface.

**Useful brain pieces exist.** The repo has validated work in navigation, simple
conversation, memory, category learning, reward/value, replay, curiosity,
affect-like state, and self-monitoring.

**The big correction.** Many pieces were built to pass narrow tests. Some remain
template-like or host-assisted. That is not wasted work, but it must be judged as
substrate and evidence, not as finished human-like faculties.

**The immediate crux.** Build a small closed loop where the brain has a world, a
body, internal state, and a reason to speak. The first meaningful conversational
advance is not "more corpus training"; it is speech caused by the brain's own
grounded state and changed by the consequences of interaction.

## Phase 0: Keep The Record Honest

Status: active and partially implemented.

Purpose: make sure the project does not drift back into test-passers and
scaffold accumulation.

What is in place:

- raw artifacts and provenance sidecars for research results;
- multi-seed expectations for nontrivial claims;
- gates that catch stale documentation, unsupported numbers, weak controls,
  missing provenance, and repeated refuted ideas;
- a new charter and structural mechanism map from the 2026-08-02 realignment.

Still needed:

- a lightweight public scaffold ledger;
- a standard "role in the whole brain" section for new mechanism work;
- an integration-test requirement for any claim that a faculty meaningfully
  advanced;
- clearer separation between public docs and dense lab boards.

## Phase 1: A Minimal Lived Loop

Goal: a small brain in a small world that perceives, wants, acts, speaks, receives
consequences, and learns.

Build:

- a minimal world with objects, actions, simple needs, and a social partner;
- a body interface where host code only enacts motor outputs and provides senses;
- a shared loop that runs continuously rather than as isolated experiments;
- a simple internal-state source for speech, such as curiosity, surprise, need,
  or affect;
- feedback from the world or partner that changes the brain's future behavior.

Pass condition:

The same external prompt should not be enough to produce speech. The utterance
must depend on internal state, learned grounding, and recent consequences.

Why this comes first:

Without this loop, language keeps sliding into text prediction and memory keeps
sliding into lookup.

## Phase 2: Grounded Language As Action

Goal: words flow into and out of grounded brain state.

Build:

- a comprehension path from words or sounds into grounded concepts;
- overlapping word, percept, action, and affect assemblies instead of isolated
  labels;
- a preverbal-message path: internal state chooses what is worth saying before
  words are selected;
- a speech-production path that renders that message into word order and
  articulation-like motor output;
- a contingent teacher loop where correction, misunderstanding, and successful
  communication reshape the pathway.

Current assets:

- bounded spiking speech generation;
- concept-pool to word-output pathways;
- simple parsers and fact-question answering;
- early contingent-teacher learning experiments;
- larger recurrent language-circuit checkpoints, including a recent local
  spiking-forward fidelity check at 267M parameters for two seeds.

Main risks:

- corpus prediction can look fluent while staying ungrounded;
- grammar templates can pass tests without becoming flexible language;
- an abstention gate can look honest while staying outside the self-model.

## Phase 3: Self-Model And Honest Uncertainty

Goal: uncertainty, confidence, attention, and authorship are read from the brain's
own state and used to decide whether to answer, hedge, ask, or stay silent.

Build:

- a self-schema region that reads attention, confidence, and source of thought;
- a metacognitive monitor that predicts when first-order answers are reliable;
- routing from confidence to behavior: answer, hedge, ask, explore, or abstain;
- tests where familiar-but-wrong cues, missing facts, and uncertain memories are
  separated.

Current assets:

- familiarity-based no-fabrication behavior;
- self-schema and metacognition test runners with lesion and permutation
  controls;
- a six-seed self-schema relay result in the research runner;
- a default-off production conversation hook that preserves the hard moat and
  downgrades low-confidence familiar-wrong recalls in the current stressed
  battery;
- a named source-consistency safety scaffold that catches the current RF
  source/answer mismatch failure mode, while remaining marked for burn-down.

Open point:

The hook is wired, but the signal is not yet strong enough: raw trace confidence
still asserts some wrong recalls. The source-consistency floor prevents those
assertions in the current production wrapper, but it reads exact composer source
metadata. The next step is a neural source-monitoring or learned correctness
signal from the metacognitive monitor.

## Phase 4: Predictive And Reconstructive Memory

Goal: memory is not a database. It predicts, reconstructs, consolidates, forgets,
and is colored by salience and emotion.

Build:

- a predictive world model over states, actions, and consequences;
- episodic memory that stores lived events with self/other/source tags;
- hippocampal replay that supports consolidation and imagination;
- cortical memory that stores distributed patterns without fixed fact slots;
- memory retrieval that can honestly return ambiguity instead of forcing one
  answer.

Current assets:

- hippocampus-style completion and replay experiments;
- sleep/replay/consolidation infrastructure;
- directional replay readers;
- shared-population memory routes under active testing;
- category and semantic memory experiments.

Main risks:

- exact host-side bind/retrieve routines can behave like a database;
- memory handoff can appear successful while still reading from the old store;
- a single correct answer can hide genuinely ambiguous recall.

## Phase 5: Developing Affect, Drive, And Curiosity

Goal: emotion and motivation become continuous internal forces that shape what
the brain notices, remembers, learns, says, and does.

Build:

- a graded valence and arousal state instead of a binary good/bad latch;
- learned appraisal from events and goals, not host-set mood values;
- interoceptive/body signals tied into affect;
- affective bias on attention, memory, speech choice, and action vigor;
- curiosity based on learning progress over time, not just novelty in the moment.

Current assets:

- reward-prediction and value circuits;
- persistent affect-like state with causal bias in test settings;
- dissociable neuromodulator axes;
- curiosity circuits that can seek learnable unknowns and avoid some unlearnable
  noise.

Open point:

The pieces need to become one live system. Rich emotion means state that develops
from experience and changes behavior over time, not a scalar label on a concept.

## Phase 6: Continual Learning And Growth

Goal: the brain keeps learning from interaction without wiping itself, and grows
only as needed.

Build:

- developmental growth of neurons, regions, and connections;
- consolidation that protects old knowledge while admitting new experience;
- curriculum and teacher-as-caregiver scaffolding that gradually fades;
- local learning rules that work at useful scale on the chosen spiking substrate;
- resource-aware scheduling across local CPU, local GPU, and available CPU pool
  machines.

Current assets:

- many local plasticity rules;
- wake/sleep/growth/persistence infrastructure;
- checkpoint lineages;
- CPU and GPU backends;
- mini-PC CPU pool support for parallel experiment sweeps.

Hardest science:

Continual learning without forgetting is difficult beyond this project. Deep
credit assignment on real spikes is partly mapped and remains a side frontier.
The preferred route is to rely on local input learning, recurrent scaffolds,
readouts, replay, and consolidation where they fit, while continuing to measure
the harder learning-rule boundary honestly.

## Phase 7: Scale And Ownership

Goal: grow toward open-ended conversation while keeping the project ownable on
high-end personal hardware.

Build:

- larger grounded vocabularies and richer sensory/social experience;
- sparse, event-driven implementations of heavy pathways;
- GPU kernels where they preserve the same model;
- CPU-cluster fanout for independent seeds and parameter sweeps;
- explicit accounting of memory use, runtime, and throughput;
- architecture choices compatible with future neuromorphic hardware.

What success looks like:

The system should become larger because the brain has grown and earned it, not
because a giant static model was preallocated.

## Highest Priority Work Now

1. **Bank the large-language-circuit spiking-forward promotion.** Finish the
   multi-seed check for the 267M-parameter local recurrent language checkpoint,
   then update the record honestly.
2. **Strengthen the self-schema honesty signal.** The production hook exists and
   is moat-safe. A source-consistency scaffold catches the current mismatch
   failures; now burn that down into calibrated learned correctness confidence
   instead of exact composer metadata or raw trace confidence.
3. **Build the minimal grounded speech-action loop.** Use internal curiosity,
   surprise, or affect as the cause of speech; use contingent partner feedback as
   the learning signal.
4. **Create the scaffold ledger.** Make every host-side shortcut visible, named,
   and paired with a replacement.
5. **Retune docs and gates around role-in-the-whole.** New claims should state what
   the mechanism does for the whole brain, not just which narrow score improved.

## How To Read Old Milestones

Older documents often use strong labels for narrow experimental results. Read
them as evidence that a mechanism can do a specific measured thing under specific
conditions. Do not read them as proof that a whole human-like faculty is complete.

Corrections and negative results are part of the record. If a summary and a
finding disagree, trust the specific finding and update the summary.

## Short Success Target

A narrow but real prototype:

- one brain;
- one small world;
- a body;
- grounded perception of a few objects and actions;
- internal needs, affect, and curiosity;
- a simple self-confidence read;
- speech caused by internal state;
- interaction with a teacher or person that changes future speech and behavior;
- honest abstention or asking when grounding is missing.

This prototype may speak simply. It does not need to sound like a large language
model. It needs to be alive in the project-specific sense: stateful, grounded,
learning, internally driven, and integrated.
