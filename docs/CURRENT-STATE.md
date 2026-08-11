# Current State

Status reviewed: 2026-08-11.

Neural Simulator is a capable CPU/GPU spiking-network simulator and a large
collection of neuroscience-inspired experiments. It is **not yet an autonomous
conversational mind**. It cannot currently live in a world, develop through
ordinary human interaction, hold fluent open-ended conversations, or produce
all of its cognition through one self-organizing spiking brain.

The current frontier is **continuous integration**: validated faculties are now
wired into one live fourteen-turn conversational loop, and each change is judged
by whether the conversation actually improves, because running the real chat is
what exposes results that only looked good in isolation. That chat still runs in
a small toy world (two agents, three actions, a small fact set), so most turns
correctly abstain, and the language "mouth" is a conventionally trained spiking
language model kept as an explicit articulation scaffold. Episodic memory (the
"gap#5" seam) is now mechanistically closed at the completion-readout level and
wired into that loop.

This page separates evidence for working components from partial integration
and from abilities that have not been achieved.

## Demonstrated

These statements have direct code, tests, and recorded experimental evidence.
They do not imply human-level versions of the named ability.

| Demonstrated result | Evidence and boundary |
|---|---|
| **Spiking simulation substrate** | The repository supports NumPy CPU and CuPy/NVIDIA GPU execution, multiple neuron models, sparse region-to-region wiring, continuous stepping, checkpoints, local plasticity, reward modulation, and configurable neuromodulators. Tests cover the shared simulation network, plasticity isolation, checkpoint behavior, and modulatory dynamics. |
| **Grounded communication learning** | Controlled experiments have learned a food-request loop, a two-intent by two-referent vocal convention, and a neural action selector. They have not yet joined clean completed actions to reliable delayed reward credit. V13 adds autonomously active GPi/SNr-like output neurons. After correcting GPU flush-to-zero, replay v2 kept voltage, recovery state, and spikes byte-exact for all 1,200 rows on NumPy and the RTX 3090. V6 sealed cross-backend calibration at `100 pA`, replication GO on fresh seed `890220`, and held-out GO on blind seed `1021`. V7 measured performance without rerunning physiology and missed the default/old and v2 active/default limits; its receipt failed after measurement. V8 then fixed the v2 active-path overhead and captured a complete receipt: v2 active/default passed at `1.006836`, but normal default/old still failed at `1.059092` against `1.02`, so Stage 0 remains unpromoted. Injected exploration, fixed channels, and a host listener/readout remain, so these are preverbal causal loops rather than natural language. See the [learned-convention finding](../research/findings/2026-08-03-developmental-vocal-convention-6seed-GO.md), [selector finding](../research/findings/2026-08-03-neural-vocal-selector-gateA-v2-4seed-GO.md), [replay v2 result](../research/findings/2026-08-04-neural-vocal-credit-gateB-v13-backend-neutral-izh-arithmetic-replay-v2-DIAGNOSTIC-RESULT.md), and [v8 performance result](../research/findings/2026-08-04-neural-vocal-credit-gateB-v13-stage0-performance-confirmation-v8-NO-GO.md). |
| **Learned source-support signal** | A zero-initialized spiking pathway can learn whether episode activity was seen, heard, or self-generated and propagate recalled source activity into neural aPFC and ACC populations on the same bridge. Both initial calibration seeds passed every causal control, but one of three preregistered development seeds missed the fixed source-margin floor. Local inhibitory competition then cleared every absolute margin on two fresh seeds but slightly harmed one strong source. A third version's threshold homeostasis failed both fresh seeds. A fourth version's formal calibration is undefined because its evaluator mishandled a bound method signature; its intact and learning-lesion behavior was also identical on both consumed seeds. Episode allocation, source-afferent identity, fixed competition anatomy, and host-timed learning windows are still scaffolded. See the [development no-go](../research/findings/2026-08-03-source-monitor-coresidency-development-NO-GO.md), [competition no-go](../research/findings/2026-08-03-source-monitor-coresidency-v2-calibration-NO-GO.md), [homeostasis no-go](../research/findings/2026-08-03-source-monitor-coresidency-v3-calibration-NO-GO.md), and [adaptive-inhibition undefined result](../research/findings/2026-08-03-source-monitor-coresidency-v4-calibration-UNDEFINED.md). |
| **Visual representation components** | Host top-k selection has been removed from the tested V1/V2/IT path, and all deadline-fired cells are measured. A hierarchical retinotopic-part and temporal-trace candidate failed both valid calibration seeds: intact inhibition silenced V2/IT and all permanence changes, while removing V2 inhibition created activity without above-chance identity and saturated IT on one seed. Fixed Gabor filters, host V1 normalization, fixed receptive fields, synthetic identity-pure tracks, and host-timed readout remain scaffolds. See the [hierarchical visual no-go](../research/findings/2026-08-03-visual-hierarchical-part-identity-calibration-NO-GO.md). |
| **Auditory sensory construction** | A streaming CPU front end now converts normalized microphone or WAV audio into a tonotopic auditory-nerve spike raster. Channel-aligned auditory-nerve, excitatory A1, and inhibitory A1 populations initialize on the real shared bridge, and the adapter can drive only the auditory-nerve regions. This is a tested construction boundary, not evidence of calibrated A1 responses, speech perception, or learned auditory concepts; reduced cochlear mechanics and missing brainstem/thalamic stages remain explicit scaffolds. See the [construction finding](../research/findings/2026-08-04-auditory-cochlea-tonotopic-a1-frontend-v1-CONSTRUCTION.md). |
| **Memory and replay components** | The repository contains tested episodic storage, pattern completion, replay, reconsolidation, and wake/sleep integration experiments. Uncued hippocampal replay can causally change cortical weights on one bridge. Local fast-spiking competition made one fresh seed much more selective, but hippocampus-independent recall remained seed-fragile and did not reliably depend on learned target identity or replay order. A learned index-relay successor produced no intact recovery on either fresh seed because its required sleep relay and inhibitory-loop activity never appeared, making that calibration invalid rather than negative. This does not establish general autobiographical memory or lifelong consolidation. See the [sleep-cycle finding](../research/findings/2026-07-25-gap5-onebrain-production-sleepcycle-merge-6seed-GO.md), [replay v2 no-go](../research/findings/2026-08-03-replay-cortical-consolidation-v2-calibration-NO-GO.md), and [replay v3 invalid result](../research/findings/2026-08-03-replay-cortical-consolidation-v3-calibration-UNDEFINED.md). |
| **Reward, affect, and curiosity components** | Dopamine-like reward prediction, persistent affect-like state, separable modulatory axes, active affect clearing, and curiosity-related circuits have passed narrow causal tests. A fresh two-seed diagnostic retained neural clearing and restart but found no recurrent weight that produced graded, neutral-crossing valence; the state remained latch-like and formal testing stayed sealed. Some curiosity and appraisal quantities are still computed by host code, and the parts do not yet form a developing emotional life. See the [affect-axis result](../research/findings/2026-08-02-laneA-affect-axes-DISSOCIATE-6seed-GO-first-attempt-negative-was-a-measurement-artifact.md), [affect-clear result](../research/findings/2026-08-01-affect-BRAIN-BASED-active-clear-spiking-quench-fs-gate-6seed-GO-operating-region-mapped.md), [graded-affect boundary](../research/findings/2026-08-04-laneA-graded-affect-quench-v1-DIAGNOSTIC-RESULT.md), and [learning-progress result](../research/findings/2026-08-02-laneB-curiosity-learning-progress-slope-CPU-proxy-6seed-GO-next-onbridge-realization.md). |
| **Large language-circuit forward execution** | A conventionally trained 267-million-parameter recurrent language checkpoint matched its non-spiking reference output through the tested resonate-and-fire spiking read path across six GPU seeds. This demonstrates numerical forward-path fidelity, not grounded language learning, biological training, or conversational ability. See the [language-circuit finding](../research/findings/2026-08-02-gap1-wkv-width-ladder-scale-read-run4-d2048-is-the-next-spiking-forward-target.md). |
| **Episodic memory completion (gap#5)** | The episodic-composition seam is now mechanistically closed at the completion-readout level. In one emergent loop, a dentate-gyrus-like stage selects a small cell assembly, one-shot behavioral-timescale plasticity forms a recurrent attractor, and an intrinsic per-cell dendritic plateau read-out completes the memory from a partial cue in a way that does not depend on assembly size. Six seeds passed; permuted-cue, silent-rest, no-encoding, and recurrence-zeroed controls were all exactly zero, and a linear coincidence-off control failed, so the plateau (not the weights alone) is the completer. Honest boundary: this is a de-risk result at one density read during the cue; the shipped default completion path is still the recurrent read, so making the dendritic read the default is the remaining integration step, which the live-chat turn-7 recall (below) then took. It is distinct from the still-open deep-credit-on-spikes boundary (gap#4). See the [dendritic-readout finding](../research/findings/2026-08-10-gap5-lever-B-dendritic-dAP-readout-completes-emergent-small-assembly-6seed-GO.md). |
| **Live conversational integration** | Validated faculties are wired into one live fourteen-turn chat, with grounded content read from the brain's own vector-symbolic memory and the language model kept off or as a declared articulation scaffold. Six changes each improved the real conversation. A no-confab check now verifies every main and subordinate clause against neural memory, dropping invented causal clauses (confabulations three to zero, six seeds). Turn 7 recalls the prior topic from a per-turn episodic store instead of falling silent, and after a corrected operating point that recall flows through the spiking dendritic completion path on both NumPy and GPU. Turn 5 ("how do you feel?") returns a functional affect read-out from the spiking valence differential, and turn 13 ("are you a simulated brain?") an honest structural self-affirmation with a graded certainty band; both are stated as functional read-outs, never as feeling or experience, and six seeds clear the certainty bar. Turn 4 ("why did the dog go east?") confirms the stored fact and honestly discloses the absent causal faculty rather than inventing a reason (six seeds). Corpus-mined relational facts raise grounded-subject breadth from two to nine (six seeds), and a demo-scale variant learns three facts as spiking weight changes gated by a learned, now fully spiking, no-confab check (six seeds). Boundaries: a toy world where most turns correctly abstain (silence is the moat working); two still-separate spiking bridges rather than one merged brain; host-side fact mining and storage, response templates, and an argmax read-out remain named scaffolds. See the [sub-clausal moat](../research/findings/2026-08-10-INTEGRATION1-subclausal-moat-live-chat-confab-6seed.md), [spiking turn-7 recall (corrected)](../research/findings/2026-08-10-episodic-dialogue-recall-wired-to-spiking-dAP-readout-numpy-backend-honest-negative.md), [certainty band](../research/findings/2026-08-10-INTEGRATION-3c-certainty-band-opponent-margin-robust-turn13-all6-clear-002.md), [causal disclaimer](../research/findings/2026-08-10-INTEGRATION-5-honest-causal-query-disclaimer-turn4-6seed.md), [corpus-learned facts](../research/findings/2026-08-10-INTEGRATION-6-corpus-learned-facts-into-live-chat-6seed.md), [plasticity-learned facts](../research/findings/2026-08-10-INTEGRATION-7-plasticity-learned-facts-into-live-chat-6seed.md), and [fully spiking moat](../research/findings/2026-08-10-INTEGRATION-7-burndown2-spiking-familiarity-gate-moat-fully-spiking-6seed.md). |

## Partially Achieved

| Capability | What exists | What prevents the full claim |
|---|---|---|
| **A shared brain** | A live conversational loop now runs several validated faculties together — grounded fact recall, a no-confab check, episodic dialogue memory, functional affect and self-model read-outs, and a curiosity ask — and judges changes by whether the whole conversation improves. | Integration has begun but is not complete: the loop still spans two separate spiking bridges rather than one merged substrate, many faculties were developed and tested in separate runners, and the world is a small toy set. Merging the co-resident bridges into one brain is the named next arc. |
| **Stage B experiment screening** | The V14 engine completed a fresh 512-candidate, five-arm GPU engineering screen for Sobol indices 512-1023 and selected batch width 512 through a separate benchmark. Strict triage classified 421 candidates as engineering failures, 91 as inconclusive, and 0 as passes. A follow-up authenticated 36-trace diagnostic retired the old packet. The corrected successor's fused Stage 1 fast-Na/Kv3 clamp now runs on CPU and GPU, re-fits all 18 source endpoints, and produces an authenticated verdict with little manual handling. Eleven endpoints passed and seven failed. The exact failure set now opens a fixed, authenticated research gate automatically. | The current packet and the successor's first fast-channel state equations are structural NO-GOs, so no candidate is eligible for confirmation or compartment integration. Fast-Na activation/deactivation and Kv3 deactivation must be redesigned from focused current-level evidence before another search. The engine can plan, seal, dispatch, persist, resume, analyze, compile bounded observations, and route a preregistered failure into research, but it cannot accept scientific claims or preregister a valid replacement architecture by itself. Candidates 284 and 404 remain closed, and the heterogeneous 12-cell SK cohort is unavailable. See the [fresh screen](../research/findings/2026-08-05-v14-stageB-sobol-v3-fresh-gpu-screen-ENGINEERING-NO-GO.md), [failure diagnostic](../research/findings/2026-08-05-v14-stageB-v3-failure-diagnostic-STRUCTURAL-NO-GO.md), [Stage 1 source transfer](../research/findings/2026-08-05-v14-stageB-fast-channel-source-transfer-STRUCTURAL-NO-GO.md), and [V2 successor contract](../research/specs/v14_snr_stageB_structural_successor_v2.json). |
| **Conversation** | The live fourteen-turn chat parses constrained questions, recalls grounded facts and the prior dialogue topic, reports functional affect and self-model read-outs with a graded certainty band, asks a curiosity question, and abstains on out-of-domain cues. A sub-clausal no-confab check keeps invented causal clauses out of every reply. | The world is a small toy set, so most turns correctly abstain rather than converse. The language "mouth" is a conventionally trained spiking model kept as an articulation scaffold, and host fact mining and storage, response templates, and some read-outs still perform important work. Dialogue is prompted and bounded, not autonomous and open-ended. |
| **Grounded language** | Preverbal experiments now show one fixed request loop and learned selection among two intent and two referent channels from consequences. The tiny learned factors compose in combinations absent from training. Other experiments connect words with visual or conceptual patterns. | The learned signals are raw channels rather than words. Motor exploration, regional structure, perception currents, listener semantics, and readout remain scaffolded. The first intrinsic same-brain reversal attempt failed repeatability. Natural input, sequence learning, and conversation are unvalidated. |
| **Honest uncertainty** | Familiarity, decision confidence, self-monitoring, authorship, and learned source-support mechanisms exist. In the live chat, a sub-clausal no-confab check verifies each clause against neural memory, a graded certainty band accompanies self-report, one demo fact-gate is now fully spiking, and unsupported turns abstain — all stated as functional read-outs, never as claims about experience. | Earlier source-strength work missed a development repeatability gate, and honesty is not yet driven end-to-end from lived sensory experience through speech choice. Some gates and thresholds remain host-side, and the brain is not a general truth checker. |
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
4. **Self-monitoring is improving but not yet end to end.** The live chat now
   runs a sub-clausal no-confab check and a graded certainty band, and one demo
   fact-gate is fully spiking, but earlier source-margin work stayed
   seed-fragile, host code still joins some monitoring to the answer policy, and
   honesty is not yet driven from lived experience through speech choice.
5. **Memory, affect, and curiosity are components rather than a life history.**
   Their tests are informative but too bounded to establish an enduring person-
   like state.
6. **Integration is partial and scaffolds are widespread.** The live loop still
   spans two separate spiking bridges rather than one merged brain, and host
   fact mining and storage, response templates, an argmax read-out, and other
   fixed representations, routing, thresholds, and teachers must still be removed
   under explicit tests. Continual, sequential learning of many facts without
   forgetting remains unsolved and blocks scaling the learned-fact demo.

## Highest-Value Work

The immediate priority is **continuous integration**: keep wiring validated
faculties into the one live conversational loop and judge every change by
whether the actual fourteen-turn chat gets better, because running the real
conversation is what exposes results that only looked good in isolation. Recent
integrations added a sub-clausal no-confab check, spiking episodic recall of the
prior topic, functional affect and self-model read-outs with a graded certainty
band, an honest causal-query disclaimer, corpus-learned grounded breadth, and a
demo-scale fact learned as spiking weight changes behind a fully spiking
no-confab gate. Each held across six seeds with zero confabulations and no
regression on the other turns.

The named next steps, each a mechanism rather than a deferral, are: merge the
two co-resident spiking bridges into one brain (the "one brain" step); replace
the host-side fact mining and vector-symbolic storage with a stream cortex that
learns co-occurrence in synapses; make the dendritic completion read the default
episodic path; replace the argmax read-out and remaining response templates with
neural mechanisms; and reach continual, sequential learning of many facts
without forgetting, which currently blocks scaling the learned-fact demo up to
the corpus-learned breadth.

Component arcs continue in support of this frontier — reward credit for actions
the brain actually completed, replay-driven consolidation, source monitoring,
and the larger conventionally trained language circuit — but a larger language
circuit is useful only after the brain has selected a grounded message,
certainty, and decision to speak.

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
