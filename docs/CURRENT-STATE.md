# Current State

Status reviewed: 2026-08-04.

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
| **Grounded communication learning** | Controlled experiments have learned a food-request loop, a two-intent by two-referent vocal convention, and a neural action selector. They have not yet joined clean completed actions to reliable delayed reward credit. V13 adds autonomously active GPi/SNr-like output neurons, but its first calibration and dependent replication are procedurally undefined and their seeds are consumed. The large CPU/GPU split was traced to backend-specific initialization; the correction passed exact step-zero parity for all 15 checked arrays. After removing GPU flush-to-zero from the opt-in strict path, replay v2 kept voltage, recovery state, and spikes byte-exact for all 1,200 rows on NumPy and the RTX 3090. Process correction v1 found one diagnostic NumPy passing point but stopped on an evidence-contract defect. V2 ran fresh calibration once on NumPy and CuPy; both found only `100 pA` passing, and the merge observed calibration GO. Its command omitted an explicit NumPy backend, however, so the selection could not be sealed and v2 is also undefined. No v2 replication or held-out command ran. The emitters are corrected and covered by envelope-level tests; v3 must be preregistered with fresh partitions before more scientific execution. Held-out seed `1021` and Stage-1 seed `1031` remain sealed. Injected exploration, fixed channels, and a host listener/readout remain, so these are preverbal causal loops rather than natural language. See the [learned-convention finding](../research/findings/2026-08-03-developmental-vocal-convention-6seed-GO.md), [selector finding](../research/findings/2026-08-03-neural-vocal-selector-gateA-v2-4seed-GO.md), [V13 process correction](../research/findings/2026-08-04-neural-vocal-credit-gateB-v13-calibration-order-UNDEFINED.md), [initialization parity result](../research/findings/2026-08-04-neural-vocal-credit-gateB-v13-backend-neutral-izh-initialization-correction-DIAGNOSTIC-RESULT.md), [replay v2 result](../research/findings/2026-08-04-neural-vocal-credit-gateB-v13-backend-neutral-izh-arithmetic-replay-v2-DIAGNOSTIC-RESULT.md), [v1 undefined result](../research/findings/2026-08-04-neural-vocal-credit-gateB-v13-stage0-process-correction-v1-UNDEFINED.md), and [v2 undefined result](../research/findings/2026-08-04-neural-vocal-credit-gateB-v13-stage0-process-correction-v2-UNDEFINED.md). |
| **Learned source-support signal** | A zero-initialized spiking pathway can learn whether episode activity was seen, heard, or self-generated and propagate recalled source activity into neural aPFC and ACC populations on the same bridge. Both initial calibration seeds passed every causal control, but one of three preregistered development seeds missed the fixed source-margin floor. Local inhibitory competition then cleared every absolute margin on two fresh seeds but slightly harmed one strong source. A third version's threshold homeostasis failed both fresh seeds. A fourth version's formal calibration is undefined because its evaluator mishandled a bound method signature; its intact and learning-lesion behavior was also identical on both consumed seeds. Episode allocation, source-afferent identity, fixed competition anatomy, and host-timed learning windows are still scaffolded. See the [development no-go](../research/findings/2026-08-03-source-monitor-coresidency-development-NO-GO.md), [competition no-go](../research/findings/2026-08-03-source-monitor-coresidency-v2-calibration-NO-GO.md), [homeostasis no-go](../research/findings/2026-08-03-source-monitor-coresidency-v3-calibration-NO-GO.md), and [adaptive-inhibition undefined result](../research/findings/2026-08-03-source-monitor-coresidency-v4-calibration-UNDEFINED.md). |
| **Visual representation components** | Host top-k selection has been removed from the tested V1/V2/IT path, and all deadline-fired cells are measured. A hierarchical retinotopic-part and temporal-trace candidate failed both valid calibration seeds: intact inhibition silenced V2/IT and all permanence changes, while removing V2 inhibition created activity without above-chance identity and saturated IT on one seed. Fixed Gabor filters, host V1 normalization, fixed receptive fields, synthetic identity-pure tracks, and host-timed readout remain scaffolds. See the [hierarchical visual no-go](../research/findings/2026-08-03-visual-hierarchical-part-identity-calibration-NO-GO.md). |
| **Auditory sensory construction** | A streaming CPU front end now converts normalized microphone or WAV audio into a tonotopic auditory-nerve spike raster. Channel-aligned auditory-nerve, excitatory A1, and inhibitory A1 populations initialize on the real shared bridge, and the adapter can drive only the auditory-nerve regions. This is a tested construction boundary, not evidence of calibrated A1 responses, speech perception, or learned auditory concepts; reduced cochlear mechanics and missing brainstem/thalamic stages remain explicit scaffolds. See the [construction finding](../research/findings/2026-08-04-auditory-cochlea-tonotopic-a1-frontend-v1-CONSTRUCTION.md). |
| **Memory and replay components** | The repository contains tested episodic storage, pattern completion, replay, reconsolidation, and wake/sleep integration experiments. Uncued hippocampal replay can causally change cortical weights on one bridge. Local fast-spiking competition made one fresh seed much more selective, but hippocampus-independent recall remained seed-fragile and did not reliably depend on learned target identity or replay order. A learned index-relay successor produced no intact recovery on either fresh seed because its required sleep relay and inhibitory-loop activity never appeared, making that calibration invalid rather than negative. This does not establish general autobiographical memory or lifelong consolidation. See the [sleep-cycle finding](../research/findings/2026-07-25-gap5-onebrain-production-sleepcycle-merge-6seed-GO.md), [replay v2 no-go](../research/findings/2026-08-03-replay-cortical-consolidation-v2-calibration-NO-GO.md), and [replay v3 invalid result](../research/findings/2026-08-03-replay-cortical-consolidation-v3-calibration-UNDEFINED.md). |
| **Reward, affect, and curiosity components** | Dopamine-like reward prediction, persistent affect-like state, separable modulatory axes, active affect clearing, and curiosity-related circuits have passed narrow causal tests. A fresh two-seed diagnostic retained neural clearing and restart but found no recurrent weight that produced graded, neutral-crossing valence; the state remained latch-like and formal testing stayed sealed. Some curiosity and appraisal quantities are still computed by host code, and the parts do not yet form a developing emotional life. See the [affect-axis result](../research/findings/2026-08-02-laneA-affect-axes-DISSOCIATE-6seed-GO-first-attempt-negative-was-a-measurement-artifact.md), [affect-clear result](../research/findings/2026-08-01-affect-BRAIN-BASED-active-clear-spiking-quench-fs-gate-6seed-GO-operating-region-mapped.md), [graded-affect boundary](../research/findings/2026-08-04-laneA-graded-affect-quench-v1-DIAGNOSTIC-RESULT.md), and [learning-progress result](../research/findings/2026-08-02-laneB-curiosity-learning-progress-slope-CPU-proxy-6seed-GO-next-onbridge-realization.md). |
| **Large language-circuit forward execution** | A conventionally trained 267-million-parameter recurrent language checkpoint matched its non-spiking reference output through the tested resonate-and-fire spiking read path across six GPU seeds. This demonstrates numerical forward-path fidelity, not grounded language learning, biological training, or conversational ability. See the [language-circuit finding](../research/findings/2026-08-02-gap1-wkv-width-ladder-scale-read-run4-d2048-is-the-next-spiking-forward-target.md). |

## Partially Achieved

| Capability | What exists | What prevents the full claim |
|---|---|---|
| **A shared brain** | Several experiments place perception, memory, drives, action selection, and communication-related regions in one simulation network. | Most validated faculties were developed and tested in separate runners. They do not yet operate together as one continuous developing agent. |
| **Conversation** | Demos can parse constrained questions, retrieve simple stored facts, render bounded answers, and abstain on unknown cues. | Host parsing, fixed data structures, templates, confidence thresholds, and conventionally trained language components still perform important work. Dialogue is narrow and prompted rather than autonomous and open-ended. |
| **Grounded language** | Preverbal experiments now show one fixed request loop and learned selection among two intent and two referent channels from consequences. The tiny learned factors compose in combinations absent from training. Other experiments connect words with visual or conceptual patterns. | The learned signals are raw channels rather than words. Motor exploration, regional structure, perception currents, listener semantics, and readout remain scaffolded. The first intrinsic same-brain reversal attempt failed repeatability. Natural input, sequence learning, and conversation are unvalidated. |
| **Honest uncertainty** | Familiarity, decision confidence, self-monitoring, authorship, and learned source-support mechanisms exist. Episode, source, aPFC, and ACC activity now coexist on one tested bridge. | Source strength missed its first development repeatability gate and is not yet connected from lived sensory experience through speech choice. Production still uses host-side thresholds. The brain is not a general truth checker. |
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
4. **Self-monitoring is not yet robust or end to end.** Local competition and
   threshold homeostasis have not produced stable source margins across fresh
   seeds, and host code still joins monitoring signals to the answer policy.
5. **Memory, affect, and curiosity are components rather than a life history.**
   Their tests are informative but too bounded to establish an enduring person-
   like state.
6. **Scaffolds are widespread.** Fixed representations, routing, thresholds,
   teachers, and host-computed psychological signals must be removed under
   explicit tests.

## Highest-Value Work

The immediate priority remains reward credit for actions the brain actually
completed. V13 tests whether autonomously active GPi/SNr-like output neurons on
the shared spiking substrate can support clean action completion and later
learning. Its original calibration cannot be used because the locked CPU-first
order was violated. The backend correction is complete for the matched-state
path: initialization and the corrected 1,200-step replay are exact across NumPy
and the RTX 3090.

The first replacement NumPy calibration found `100 pA` as its only passing
point, but its v1 evidence chain failed closed on digest and sidecar sealing
defects. V2 corrected those controls and ran fresh calibration partitions once
on NumPy and the RTX 3090. Both backends again found only `100 pA` passing. The
selection merge observed GO but could not be sealed because its command omitted
the explicit NumPy backend required by the manifest contract. V2 is undefined,
its calibration seed is consumed, and its unexecuted replication seed is
retired. The command emitters and tests are corrected. The next action is to
preregister v3 with fresh mechanically derived partitions before freezing a new
source and configuration. Held-out and Stage-1 seeds remain sealed.

In parallel, replay needs a different mechanism after its first target-plateau
correction failed smoke seed `216` and was retired. Source-monitor v4 is retired
after an undefined formal run and a measured lack of effect versus its learning
lesion. Hierarchical visual identity is also retired after both valid formal
seeds produced silent intact V2/IT populations and chance decoding.
A larger language circuit is useful only after the brain has selected a
grounded message, certainty, and decision to speak.

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
