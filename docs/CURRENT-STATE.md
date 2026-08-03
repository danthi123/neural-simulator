# Current State

Status reviewed: 2026-08-03.

Neural Simulator is a capable CPU/GPU spiking-network simulator and a large
collection of neuroscience-inspired experiments. It is **not yet an autonomous
conversational mind**. It cannot currently live in a world, develop through
ordinary human interaction, hold fluent open-ended conversations, or produce
all of its cognition through one self-organizing spiking brain.

This page separates evidence for working components from partial integration
and from abilities that have not been achieved.

## Demonstrated

These statements have direct code, tests, and recorded experimental evidence.
They do not imply human-level versions of the named ability.

| Demonstrated result | Evidence and boundary |
|---|---|
| **Spiking simulation substrate** | The repository supports NumPy CPU and CuPy/NVIDIA GPU execution, multiple neuron models, sparse region-to-region wiring, continuous stepping, checkpoints, local plasticity, reward modulation, and configurable neuromodulators. Tests cover the shared simulation network, plasticity isolation, checkpoint behavior, and modulatory dynamics. |
| **Grounded communication learning** | One six-seed experiment joins a learned food percept, hunger, request, consequence, and satiety. A second six-seed experiment learns an external two-intent by two-referent raw vocal convention from contingent listener responses and correctly composes both combinations withheld from training. Fresh brains also learn a swapped convention; no-consequence, unrelated-reward, dopamine-lesion, context-lesion, and perception-lesion controls fail as predicted. The validated path still uses injected motor babbling, fixed channels, and a host listener/readout. A first intrinsic-exploration and same-brain reversal follow-up passed only 1/4 development seeds. An isolated neural selector now passes its fixed four-seed physiology gate without host channel choice or fallback, but local reward credit is not yet implemented. These are preverbal causal loops, not natural language. See the [single-request finding](../research/findings/2026-08-03-grounded-speech-action-loop-6seed-GO.md), [learned-convention finding](../research/findings/2026-08-03-developmental-vocal-convention-6seed-GO.md), [intrinsic-reversal negative](../research/findings/2026-08-03-intrinsic-vocal-reversal-4seed-NO-GO.md), and [selector finding](../research/findings/2026-08-03-neural-vocal-selector-gateA-v2-4seed-GO.md). |
| **Learned source-support signal** | A separate spiking memory can learn that a complete proposition was paired with an external experience and can reduce confidence for unsupported recalls. Its six-seed production test removed all 46 stressed wrong assertions while preserving the hard unknown-answer boundary, at the cost of unnecessarily downgrading 5 of 133 correct recalls. Proposition codes, source events, normalization, and final routing remain partly host-defined. See the [source-memory finding](../research/findings/2026-08-03-laneC-plastic-source-memory-GO-learned-source-support-production-wirein.md). |
| **Memory and replay components** | The repository contains tested episodic storage, pattern completion, replay, reconsolidation, and wake/sleep integration experiments. A six-seed shared-network sleep-cycle experiment shows that these parts can coexist in a bounded setup. This does not establish general autobiographical memory or lifelong consolidation. See the [sleep-cycle finding](../research/findings/2026-07-25-gap5-onebrain-production-sleepcycle-merge-6seed-GO.md). |
| **Reward, affect, and curiosity components** | Dopamine-like reward prediction, persistent affect-like state, separable modulatory axes, active affect clearing, and curiosity-related circuits have passed narrow causal tests. Some curiosity and appraisal quantities are still computed by host code, and the parts do not yet form a developing emotional life. See the [affect-axis result](../research/findings/2026-08-02-laneA-affect-axes-DISSOCIATE-6seed-GO-first-attempt-negative-was-a-measurement-artifact.md), [affect-clear result](../research/findings/2026-08-01-affect-BRAIN-BASED-active-clear-spiking-quench-fs-gate-6seed-GO-operating-region-mapped.md), and [learning-progress result](../research/findings/2026-08-02-laneB-curiosity-learning-progress-slope-CPU-proxy-6seed-GO-next-onbridge-realization.md). |
| **Large language-circuit forward execution** | A conventionally trained 267-million-parameter recurrent language checkpoint matched its non-spiking reference output through the tested resonate-and-fire spiking read path across six GPU seeds. This demonstrates numerical forward-path fidelity, not grounded language learning, biological training, or conversational ability. See the [language-circuit finding](../research/findings/2026-08-02-gap1-wkv-width-ladder-scale-read-run4-d2048-is-the-next-spiking-forward-target.md). |

## Partially Achieved

| Capability | What exists | What prevents the full claim |
|---|---|---|
| **A shared brain** | Several experiments place perception, memory, drives, action selection, and communication-related regions in one simulation network. | Most validated faculties were developed and tested in separate runners. They do not yet operate together as one continuous developing agent. |
| **Conversation** | Demos can parse constrained questions, retrieve simple stored facts, render bounded answers, and abstain on unknown cues. | Host parsing, fixed data structures, templates, confidence thresholds, and conventionally trained language components still perform important work. Dialogue is narrow and prompted rather than autonomous and open-ended. |
| **Grounded language** | Preverbal experiments now show one fixed request loop and learned selection among two intent and two referent channels from consequences. The tiny learned factors compose in combinations absent from training. Other experiments connect words with visual or conceptual patterns. | The learned signals are raw channels rather than words. Motor exploration, regional structure, perception currents, listener semantics, and readout remain scaffolded. The first intrinsic same-brain reversal attempt failed repeatability. Natural input, sequence learning, and conversation are unvalidated. |
| **Honest uncertainty** | Familiarity, decision confidence, self-monitoring, authorship, and learned source-support mechanisms exist, with lesion and permutation controls in bounded tests. | They are not one co-resident learned pathway from lived sensory experience through speech choice. Production still uses host-side thresholds and source representations. The brain is not a general truth checker. |
| **Memory-guided behavior** | Episodic storage, completion, replay, consolidation, and correction mechanisms work in small controlled tasks. | Some memories use engineered codes or slots; replay has known write/selectivity limits; retention and useful transfer have not been shown over an open-ended lifetime. |
| **Emotion and motivation** | Body-drive, reward, mood-like persistence, modulatory axes, and affect clearing can causally alter selected behaviors. | Appraisal is not broadly learned from lived social history, emotional state is not richly integrated with language and memory, and current tests should not be interpreted as human-like emotion. |
| **Curiosity** | Narrow circuits can favor learnable novelty and suppress some unproductive exploration. | Learning progress and action selection still rely partly on host arithmetic and task policy. Curiosity does not yet autonomously choose questions and sustain learning in the full brain. |
| **Continual learning** | Local Hebbian and spike-timing rules, reward-gated learning, replay, homeostasis, structural plasticity, and growth-related experiments exist. | Robust delayed credit assignment in real spiking networks remains unresolved, and the repository has not demonstrated long-running learning without destructive interference. See the [current deep-credit boundary](../research/findings/2026-08-02-gap4-production-bridge-deep-credit-NOT-closed-by-XOR-the-wall-is-deeper-than-task-decodability-on-bridge-forward.md). |

## Not Yet Achieved

- An autonomous agent that initiates and sustains fluent, free-form conversation.
- Human-like language acquisition from grounded social interaction.
- A single continuously operating brain in which all claimed faculties are
  learned, co-resident, and mutually influential.
- Rich emotions that develop through experience and consistently shape
  perception, memory, learning, speech, and behavior.
- Reliable self-based honesty without host fact checks, fixed confidence rules,
  or symbolic source events.
- General reasoning and imagination grounded in lived experience.
- Lifelong learning, developmental growth, and forgetting resistance at useful
  scale.
- Evidence that the full target system can run in real time on consumer
  hardware.

## Main Blockers

1. **Grounded communication is still too narrow.** The latest result learns a
   two-intent by two-referent convention, but exploration is injected, channel
   anatomy is fixed, and there is no learned word or sequence production.
2. **Learning does not yet span the whole behavior.** Local learning works in
   several shallow or specialized circuits, but delayed credit through deeper
   spiking pathways is unreliable.
3. **Language fluency and biological learning are disconnected.** The largest
   language circuit was trained conventionally; grounded loops remain small and
   template-like.
4. **Self-monitoring is not in the production path end to end.** Useful
   confidence and source signals exist, but host code still joins them to the
   final answer policy.
5. **Memory, affect, and curiosity are components rather than a life history.**
   Their tests are informative but too bounded to establish an enduring person-
   like state.
6. **Scaffolds are widespread.** Fixed representations, routing, thresholds,
   teachers, and host-computed psychological signals must be removed under
   explicit tests.

## Highest-Value Work

The immediate priority is executed-action-local reward credit. A corrected
600-neuron vocal selector now passes Gate A in all four development seeds: it
makes balanced choices, commits cleanly on 98-100% of trials, and stops when
shared arousal or its direct basal-ganglia route is lesioned. Gate B must now
show that only the winning route retains eligibility for a later global
dopamine signal, that contingent reward changes later choices, and that yoked
reward, action-collateral lesions, and dopamine lesions do not teach the same
preference. Held-out seeds stay untouched until later gates. After local credit
passes, the selector should return to same-brain reversal and then expand to
more needs, percepts, intents, and consequences. In parallel, learned source
signals should move into the same network as confidence, self-monitoring, and
speech selection. A larger language circuit is useful only after the brain has
selected a grounded message, certainty, and decision to speak.

This order tests the central project claim early: whether a small integrated
spiking brain can learn to say something because of what it perceives, needs,
remembers, and expects, while recognizing when its evidence is weak.

## Reading The Evidence

- Passing several random seeds shows repeatability within a tested setup, not a
  general cognitive faculty.
- A lesion, shuffle, reversal, or no-consequence control tests whether the named
  mechanism caused the result.
- A scaffolded success stays partial until the behavior survives the scaffold's
  removal test.
- Negative findings define real scientific boundaries and should not be hidden
  behind the larger number of unit tests or demos.

See the [roadmap](../ROADMAP.md), [scaffold ledger](SCAFFOLD-LEDGER.md), and
[`research/findings/`](../research/findings/) for plans, temporary shortcuts,
and the detailed experimental record.
