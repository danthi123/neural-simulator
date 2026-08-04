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
| **Grounded communication learning** | One six-seed experiment joins a learned food percept, hunger, request, consequence, and satiety. A second learns a two-intent by two-referent vocal convention from listener consequences, including combinations withheld from training. An isolated neural selector passes its four-seed physiology gate without host channel choice. Three local reward-credit circuits failed the yoked-reward control. A fourth host-timed trace was retired before formal testing. A fifth circuit produces an audited neural action trace and learns a separate action-local reward expectation on CPU and GPU without unintended weight changes. Its prediction-error output fails the reserved-seed smoke: repeated reward suppresses dopamine by only `5.56%` on CPU and `8.86%` on GPU against a fixed `20%` minimum, and omission recruits neither LHb-like nor RMTg-like neurons. Increasing sparse-route weight or a single learned afferent's population did not create a prediction. A dual-afferent successor was silent at its only subthreshold point; all higher fixed-input points produced expectation without learning and failed their causal learning-lesion control. Formal seeds remain sealed. Injected exploration, fixed channels, and a host listener/readout remain. These are preverbal causal loops, not natural language. See the [learned-convention finding](../research/findings/2026-08-03-developmental-vocal-convention-6seed-GO.md), [selector finding](../research/findings/2026-08-03-neural-vocal-selector-gateA-v2-4seed-GO.md), [v5 trace smoke](../research/findings/2026-08-03-neural-vocal-credit-gateB-v5-cross-backend-smoke-QUALIFIED.md), [v5 learning smoke](../research/findings/2026-08-03-neural-vocal-credit-gateB-v5-learning-smoke-NO-GO.md), [v6 result](../research/findings/2026-08-03-neural-vocal-credit-gateB-v6-engagement-NO-GO.md), [v7 result](../research/findings/2026-08-03-neural-vocal-credit-gateB-v7-dense-convergence-NO-GO.md), and [v8 result](../research/findings/2026-08-03-neural-vocal-credit-gateB-v8-dual-afferent-NO-GO.md). |
| **Learned source-support signal** | A zero-initialized spiking pathway can learn whether episode activity was seen, heard, or self-generated and propagate recalled source activity into neural aPFC and ACC populations on the same bridge. Both initial calibration seeds passed every causal control, but one of three preregistered development seeds missed the fixed source-margin floor. Local inhibitory competition then cleared every absolute margin on two fresh seeds but slightly harmed one strong source. A third version's threshold homeostasis failed both fresh seeds. A fourth version's formal calibration is undefined because its evaluator mishandled a bound method signature; its intact and learning-lesion behavior was also identical on both consumed seeds. Episode allocation, source-afferent identity, fixed competition anatomy, and host-timed learning windows are still scaffolded. See the [development no-go](../research/findings/2026-08-03-source-monitor-coresidency-development-NO-GO.md), [competition no-go](../research/findings/2026-08-03-source-monitor-coresidency-v2-calibration-NO-GO.md), [homeostasis no-go](../research/findings/2026-08-03-source-monitor-coresidency-v3-calibration-NO-GO.md), and [adaptive-inhibition undefined result](../research/findings/2026-08-03-source-monitor-coresidency-v4-calibration-UNDEFINED.md). |
| **Visual representation components** | Host top-k selection has been removed from the tested V1/V2/IT path, and all deadline-fired cells are measured. A hierarchical retinotopic-part and temporal-trace candidate failed both valid calibration seeds: intact inhibition silenced V2/IT and all permanence changes, while removing V2 inhibition created activity without above-chance identity and saturated IT on one seed. Fixed Gabor filters, host V1 normalization, fixed receptive fields, synthetic identity-pure tracks, and host-timed readout remain scaffolds. See the [hierarchical visual no-go](../research/findings/2026-08-03-visual-hierarchical-part-identity-calibration-NO-GO.md). |
| **Memory and replay components** | The repository contains tested episodic storage, pattern completion, replay, reconsolidation, and wake/sleep integration experiments. Uncued hippocampal replay can causally change cortical weights on one bridge. Local fast-spiking competition made one fresh seed much more selective, but hippocampus-independent recall remained seed-fragile and did not reliably depend on learned target identity or replay order. A learned index-relay successor produced no intact recovery on either fresh seed because its required sleep relay and inhibitory-loop activity never appeared, making that calibration invalid rather than negative. This does not establish general autobiographical memory or lifelong consolidation. See the [sleep-cycle finding](../research/findings/2026-07-25-gap5-onebrain-production-sleepcycle-merge-6seed-GO.md), [replay v2 no-go](../research/findings/2026-08-03-replay-cortical-consolidation-v2-calibration-NO-GO.md), and [replay v3 invalid result](../research/findings/2026-08-03-replay-cortical-consolidation-v3-calibration-UNDEFINED.md). |
| **Reward, affect, and curiosity components** | Dopamine-like reward prediction, persistent affect-like state, separable modulatory axes, active affect clearing, and curiosity-related circuits have passed narrow causal tests. Some curiosity and appraisal quantities are still computed by host code, and the parts do not yet form a developing emotional life. See the [affect-axis result](../research/findings/2026-08-02-laneA-affect-axes-DISSOCIATE-6seed-GO-first-attempt-negative-was-a-measurement-artifact.md), [affect-clear result](../research/findings/2026-08-01-affect-BRAIN-BASED-active-clear-spiking-quench-fs-gate-6seed-GO-operating-region-mapped.md), and [learning-progress result](../research/findings/2026-08-02-laneB-curiosity-learning-progress-slope-CPU-proxy-6seed-GO-next-onbridge-realization.md). |
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

The immediate priority is still executed-action-local reward credit. The
selector is reliable, but three credit circuits reinforced arbitrary actions
under yoked reward and the first dendritic successor was retired for host-
timing and backend defects. Its corrected v5 replacement has an audited CPU/GPU
action trace and learns a separate action-local reward expectation. The current
output circuit nevertheless fails before formal calibration: reward suppression
is too weak and the omission path is silent. The repaired evidence search
localized the boundary: neither a complete sparse-route weight ladder nor a
`24/64/128/200` single-route convergence ladder made the striatal expectation
cells fire before reward. A preregistered dual-afferent successor then tested a
fixed convergent state input plus separate plastic context. Its weight-2 point
was subthreshold but silent after learning; every higher point predicted without
learning and failed the learning-lesion control. The completed evidence gate
selected the simulator's existing graded dendritic plateau as a genuinely
different postsynaptic integration mechanism. A bounded v9 reserved-seed smoke
is preregistered to test centers `16/8/4/2` with causal learning and dendritic-
route lesions. Only after learned expectation engages will the existing slow
GABA-B/GIRK output be tested. Formal seeds remain sealed.

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
