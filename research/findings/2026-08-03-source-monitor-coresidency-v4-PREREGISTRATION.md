---
type: preregistration
status: live
date: 2026-08-03
mechanism: source-monitor-coresidency-v4
runner: research/runners/_laneC_source_monitor_coresidency_gate_v4.py
---

# Source monitoring v4: adaptive inhibitory competition

**Filed before any scientific v4 seed was run.** Seed `600` is reserved for a
synthetic-spike wiring smoke test. It cannot produce scientific evidence or a
formal verdict.

## Functional requirement

The same remembered event can carry evidence that it was seen, heard, or
self-generated. The source populations must remain distinct when one source is
weak and when two valid sources occur together. Competition should adapt from
local activity rather than applying one fixed suppressive weight to every
source.

## Mechanism under test

V4 inherits v2's co-resident episode, source-memory, fast-spiking, aPFC, and ACC
populations. It does not inherit v3's threshold homeostasis. The only new
mechanism is homeostatic inhibitory spike-timing-dependent plasticity on
fast-spiking-to-rival-source GABA-A synapses.

Each neuron keeps a local spike trace. Every step first decays that trace and
adds the current spike. On an inhibitory presynaptic spike, the local update is
`eta * (post_trace - alpha)`; on a postsynaptic spike it is
`eta * pre_trace`. Here `alpha = 2 * rho * tau_steps`. Weights are positive
inhibitory conductance magnitudes and remain within fixed bounds.

The rule follows Vogels, Sprekeler, Zenke, Clopath, and Gerstner (2011),
*Inhibitory Plasticity Balances Excitation and Inhibition in Sensory Pathways
and Memory Networks*. The repository's prior review identifies this local
inhibitory-plasticity family in
`research/findings/2026-06-15-L1-SM-on-spiking-deep-research.md`.

## Fixed mechanism and protocol

- Trace time constant: `20 ms`.
- Target firing probability: `0.02` per simulation step.
- Learning rate: `0.001`. <!--derived-->
- Inhibitory weight bounds: `0.0` through `6.0`.
- Initial inhibitory weight: `3.0`, inherited from v2.
- Balanced rehearsal budget: at least `5,000` simulation steps, cycling equally
  through three single-source episodes and one mixed-source episode.
- Episode-to-source Hebbian learning is frozen during inhibitory rehearsal.
- Firing thresholds remain fixed; intrinsic homeostasis is disabled.

The substrate updates a synapse only when it is plastic, uses GABA-A, is emitted
by an inhibitory neuron, receives a positive pathway plasticity gain, and has a
presynaptic or postsynaptic spike on that step. A closed gain is exactly inert.

## Seed and phase lock

- Non-scientific smoke only: `600`.
- Calibration, open only as the exact ordered tuple: `601`, `607`.
- Development, locked: `613`, `617`, `619`.
- Held out, locked: `631`, `641`, `643`.

The aggregate runner rejects incomplete, duplicate, or reordered calibration
partitions. Both calibration seeds must pass every primary criterion with no
parameter changes. Any failure keeps later phases closed.

## Matched arms and controls

The intact and inhibitory-learning-lesion arms begin with identical networks,
receive identical source learning, and receive the same balanced rehearsal.
Only the intact arm opens the inhibitory-plasticity gate. The expression lesion
loads the intact learned weights and then closes cross-source inhibitory
transmission during recall.

The inherited controls remain mandatory: episode-route lesion, ACC-route
lesion, full competition-expression lesion, source-afferent swap, mixed-source
recall, unseen episode, and episode-learning-off. Recall receives episode
activity without source metadata.

## Fixed validity preconditions

A result is `UNDEFINED` if the exact calibration partition is absent, any seed
is reused from smoke or an earlier gate, matched arms do not begin with equal
episode and inhibitory weights, non-inhibitory weights or firing thresholds
change during inhibitory rehearsal, the learning lesion changes inhibitory
weights, a pathway gate fails to reach its declared synapses, or any scored
numeric value is non-finite.

## Fixed scientific criteria

Every item must pass on both calibration seeds:

1. Seen, heard, and self-generated source margins are each at least `0.15`.
2. The weakest intact margin exceeds the matched learning lesion by at least one
   spike quantum, `1 / (100 * 12)`.
3. For each source, any loss relative to the learning lesion cannot exceed that
   lesion source's surplus above the `0.15` floor.
4. Total rival-source spike burden is lower than both the learning lesion and
   the expression lesion.
5. Inhibitory weights change in the intact arm only. Episode-to-source weights
   and all firing thresholds remain identical across inhibitory rehearsal.
6. Episode-route and ACC-route attributable fractions are each at least `0.90`.
7. Unseen episodes and episode-learning-off recall produce zero source-memory
   spikes.
8. Source-afferent swap follows physical activity rather than episode identity.
9. Mixed visual-auditory recall activates both valid source populations.
10. All inherited structural, inference-interface, and causal controls pass.

## Smoke boundary

The smoke test supplies synthetic population spike events to one fast-spiking
pool and one rival source pool. It checks that the larger weight change follows
the coactive rival when activity is swapped, the silent rival differs, the
learning lesion remains bit-identical, and excitatory weights and thresholds do
not move. This diagnoses rule scope and wiring only; it does not assess recall,
behavior, or scientific acceptance.

## Host boundary and remaining scaffolds

The host still supplies sparse episode assemblies, source-afferent activity,
rehearsal order and timing, learning-window boundaries, rest boundaries, spike
count readout, lesion switches, and scoring. Source anatomy and initial
inhibitory weights are predefined. V4 tests adaptive neural competition within
that bounded circuit; it does not claim natural episode formation, language,
confidence, response policy, or a complete self-model.
