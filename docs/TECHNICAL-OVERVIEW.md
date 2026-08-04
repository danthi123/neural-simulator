# Technical Overview

## Scope

Neural Simulator combines a general spiking-network engine with experiments in
perception, action, memory, value, affect, learning, self-monitoring, and
language. Its research target is a growing brain in which those functions share
one neural substrate and develop through interaction.

The codebase is more capable than the present integrated brain. Many mechanisms
work in constrained experiments; far fewer have been shown to cooperate in one
continuous life. This document separates implemented infrastructure, banked
behavior, and the target architecture.

## System Boundary

The intended system has three layers:

```text
outside world and body
    | sensory events                         ^ enacted movement or sound
    v                                       |
one shared spiking substrate: perception <-> memory <-> affect <-> action
    ^                                       |
    | experimental perturbation             v measured neural activity
research instruments, storage, and visualization
```

The world and body are ordinary software because they are outside the brain.
Research tools may observe or perturb neural state. Between sensation and
action, the final architecture requires neural computation. A host routine that
parses a sentence, calculates novelty, chooses an answer, or maps a symbolic
intent to words is therefore a scaffold even when its formula is inspired by
biology.

"Shared substrate" does not mean a uniform network. The engine assigns regions
contiguous neuron populations and declares pathways between them. Regions can
use distinct neuron types, connectivity, inhibition, receptor effects,
plasticity, and time scales while participating in one simulation step and one
synapse space.

## Implemented Foundation

The central engine is `SimulationBridge` in [`sim/bridge.py`](../sim/bridge.py),
configured through [`sim/config.py`](../sim/config.py). The implementation
includes:

- NumPy execution on the central processing unit and CuPy execution on NVIDIA
  graphics processors;
- sparse synapse storage and continuous spiking simulation;
- Izhikevich, Hodgkin-Huxley, and adaptive exponential neuron models;
- fast and slow synaptic dynamics, short-term plasticity, spike-timing and
  coactivity learning, reward-modulated eligibility, and homeostasis;
- structural plasticity, pruning support, checkpoints, recording, and
  profiling;
- declared `BrainRegion` and `RegionPathway` structures in
  [`sim/regions.py`](../sim/regions.py);
- dopamine-like and other modulatory concentration systems with scoped effects
  in [`sim/neuromodulators.py`](../sim/neuromodulators.py);
- experiment, visualization, and headless research interfaces.

These are available mechanisms, not a claim that every experiment uses them
together or that each is a complete biological reproduction.

The repository also contains automatic capacity-growth support in
[`sim/auto_growth.py`](../sim/auto_growth.py). Structural change and resizing are
possible; deciding when and how a whole developing brain should grow remains an
open research problem.

## Current Evidence

The most reliable summary is [Current State](CURRENT-STATE.md), backed by dated
findings and raw artifacts. At a high level:

| Capability | Evidence boundary |
|---|---|
| Simulation and regional wiring | Implemented and broadly exercised. Regions and pathways can share one bridge, backend, update loop, and modulatory state. |
| Perception, navigation, and action | Working in constrained visual and grid-world experiments. Results do not yet amount to general embodied intelligence. |
| Auditory input | Microphone or WAV audio can now be transformed into tonotopic auditory-nerve spikes, with A1 regions and pathways declared on the shared bridge. Functional A1 calibration, ascending brainstem stages, speech perception, and learned auditory meaning remain open. |
| Reward and learning | Local plasticity, eligibility traces, value learning, and dopamine-like modulation have working cases. Credit assignment across deep, changing spiking circuits remains difficult. |
| Memory | Episodic storage, completion, replay, consolidation, and persistent state have tested components. Some production paths still resemble host-managed records or use engineered codes. |
| Grounded communication | Controlled experiments have learned a food-request loop, a two-intent by two-referent raw vocal convention, and an isolated neural selector. They have not yet produced reliable delayed credit for clean completed actions. V13's first calibration and dependent replication are procedurally undefined. Backend-neutral initialization is exact at step zero; after correcting GPU flush-to-zero at subnormal values, a full 1,200-step matched-state replay is byte-exact for voltage, recovery state, and spikes on NumPy and the RTX 3090. Process correction v1 later found one diagnostic NumPy passing point but stopped on an evidence-contract defect. V2 ran fresh calibration once on NumPy and CuPy; both found `100 pA` as the only passing point, but the seed-free selection merge omitted its required explicit NumPy environment and could not be sealed. No replication or held-out run followed. The emitters are corrected and tested; v3 is preregistered with fresh partitions and awaits source freeze. Injected motor babbling, fixed channels, and host listener/readout remain; this is preverbal communication learning, not natural language. |
| Language | Simple question answering, bounded spiking production, and larger sequence experiments exist. Some fluent paths depend on conventional training, host parsing, fixed roles, or constrained rendering. Free-flowing grounded conversation has not been achieved. |
| Affect and motivation | Reward, drive, mood-like persistence, curiosity, and several modulatory axes have narrow positive results. Active neural clearing and restart survive a fresh two-seed diagnostic, but no tested recurrent weight produced smoothly graded, neutral-crossing valence. Rich emotions that develop from experience and jointly shape perception, memory, learning, speech, and action are not complete. |
| Self-monitoring | Confidence, attention, authorship, and learned source-support experiments exist. Important representations and routing remain engineered or separate from the main lived loop. |
| Developmental growth | The engine can alter connectivity and capacity. Stable, brain-native growth driven by lived need, with continual learning and no destructive forgetting, is open. |

No committed result establishes general vocal learning, words, syntax, or
open-ended speech through ordinary social interaction. The demonstrated vocal
convention has only two intents and two referents under a controlled curriculum.

## Intended Developmental Architecture

The target is organized by interacting functions rather than a fixed list of
anatomical labels:

1. **Sensation and interoception** encode external events and body state.
2. **Perception and association** form stable, overlapping representations of
   objects, actions, contexts, people, and words.
3. **Value, affect, and drives** estimate significance, maintain internal
   pressures, and modulate attention, learning, recall, and action selection.
4. **Memory and predictive state** connect current input with past episodes,
   learned regularities, expected outcomes, and offline replay.
5. **Working state and action selection** choose what to inspect, remember,
   learn, do, or communicate under competing needs.
6. **Self and social models** track confidence, information source, authorship,
   other agents, and the likely effects of interaction.
7. **Language comprehension and production** connect sound or text to grounded
   concepts and turn a brain-selected message into ordered motor output.
8. **Local plasticity and growth** change pathways and capacity from the
   consequences of experience while preserving useful older learning.

These functions may occupy specialized regions, but their states must remain
causally coupled. For example, emotion is incomplete if it changes a displayed
label but not memory or choice; language is incomplete if a host parser supplies
the meaning; honesty is incomplete if a final string filter supplies all doubt.

## Learning Through Interaction

The shortest route from the current system to the goal is a small continuous
world with a body and a social partner. A developmental episode should contain:

```text
sensory context + body state
    -> neural competition over attention and action
    -> movement, silence, or a raw vocal action
    -> contingent response from the world or partner
    -> local eligibility plus modulatory outcome
    -> changed neural response on a later encounter
```

The host can deliver the consequence after observing raw motor activity. It
cannot tell the brain which neural channel is the correct answer or clamp the
desired output during evaluation. Strong controls include withholding the
consequence, separating reward from the chosen action, silencing the proposed
learning signal, removing the relevant percept or drive, permuting external
meanings, and testing novel combinations.

The food-request experiment closes this loop once with fixed semantic
structure. The developmental-vocal successor learns two intent and two
referent channel meanings from consequences and composes held-out pairs. The
remaining generalization requires intrinsic exploration, adaptation within the
same brain, broader meanings and contexts, and neural message-to-word
production.

## Practical Interaction Boundary

The intended user-facing body does not require a physical robot. Available
interaction can combine webcam pixels, microphone audio, speaker output, and a
2D or 3D virtual body that can move, orient, point, select, or manipulate
virtual objects. This is enough to provide continuous perception, action,
social consequence, and a shared environment for early development.

The host may convert devices and virtual-world events into sensory streams and
carry neural motor output back into sound or virtual action. Object
recognition, speech understanding, intent selection, appraisal, and learning
must remain brain functions rather than hidden services in that interface.

Simulated biological needs are mechanism tests, not necessarily deployment
requirements. Hunger can test persistent interoception and drive reduction,
but practical development should emphasize pressures the system can genuinely
encounter: uncertainty, learning progress, social response, unresolved goals,
prediction conflict, consolidation demand, sensory overload, communication
success, and real system-health signals. Findings from biological-need probes
should be retained when they transfer to those practical pressures, rather than
turning literal hunger or similar states into required product features.
Human-like names should be used only when the resulting state has the claimed
persistence and causal reach.

## Emotion, Curiosity, And Honesty

The project treats these as coupled control systems:

- **Emotion** should emerge from body state, appraisal, prediction, memory, and
  neuromodulation, persist over useful time scales, and alter later processing.
- **Curiosity** should favor uncertainty for which interaction produces learning
  progress, while reducing effort on irreducible noise.
- **Honesty** should depend on source memory, familiarity, confidence, and the
  distinction between perceived, recalled, inferred, and self-generated
  content.

Current experiments provide parts of each chain. The target requires those
parts to become learned, co-resident pathways that influence the same action and
speech selection system.

## Growth And Compute

Consumer ownership is an architectural constraint. The software therefore
favors sparse connectivity, local state, local learning, backend-resident data,
and parallel independent experiments. The central processing unit backend is
useful for tests and small parallel runs; the NVIDIA backend is intended for
large simulations and long runs.

Performance work must preserve measured behavior. Dense global optimization,
centralized symbolic state, or a required external language service may improve
a demonstration while moving away from the target. Conversely, slow code is not
biological fidelity: avoidable data transfer, serial independent work, and
unprofiled bottlenecks should be removed.

Growth poses an additional constraint. Adding capacity must be based on neural
or developmental demand, fit available hardware, preserve checkpoints, and be
validated against forgetting and instability. Existing resizing support is an
engineering base, not evidence that this policy is solved.

## Evidence And Falsification

Research claims should connect four records:

1. an executable runner or test;
2. raw measurements with command, seed, configuration, and environment
   provenance;
3. controls that distinguish the proposed mechanism from leakage, coincidence,
   and simpler alternatives;
4. a dated finding that states the scope, interpretation, scaffolds, and next
   unresolved question.

The repository keeps negative and corrected findings because they prevent
failed mechanisms from being rediscovered as progress. Automated checks under
[`tools/gates/`](../tools/gates/) detect several classes of missing provenance,
weak controls, stale summaries, unsupported numbers, and citations to
superseded work. These checks support scientific judgment; they do not replace
it.

## Main Technical Gaps

The highest-value sequence is:

1. make neural action selection and local reward eligibility reliable enough to
   remove injected babbling and pass same-brain convention reversal;
2. broaden the grounded lived loop across several needs, percepts, actions,
   intents, and contingent social outcomes;
3. learn grounded concepts that are shared by perception, action, memory,
   affect, comprehension, and production;
4. develop persistent, graded affect and curiosity that causally change the
   same decisions and memories;
5. integrate source memory, confidence, and self-monitoring with speech
   selection;
6. support continual learning, replay, and safe structural growth over long
   interaction histories;
7. scale brain-native message formation and word production toward natural,
   open-ended conversation;
8. optimize the integrated system within a high-end consumer hardware envelope.

The [Project Charter](PROJECT-CHARTER.md) defines why these constraints are
non-negotiable. The [Scaffold Ledger](SCAFFOLD-LEDGER.md) records current
shortcuts, and [`research/findings/`](../research/findings/) contains the
evidence behind current claims.

## Interpretation And Safety

This architecture is a research hypothesis. Biological names indicate modeled
functions or inspiration, not anatomical completeness. A successful behavioral
test does not establish consciousness, sentience, felt emotion, human
equivalence, or general intelligence. Experimental outputs are not reliable
authority and should not be used for safety-critical decisions.

The software is distributed under the [MIT License](../LICENSE), without
warranty.
