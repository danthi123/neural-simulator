---
type: preregistration
status: live
date: 2026-08-03
mechanism: source-monitor-coresidency-v4
runner: research/runners/_laneC_source_monitor_coresidency_gate_v4.py
---

# Source monitoring v4: adaptive inhibitory competition

**Filed before any scientific v4 seed was run.** Seed `600` is reserved for
non-scientific mechanism smoke. It cannot produce scientific evidence or a
formal verdict.

**Pre-formal amendment after independent audit.** The original synthetic-spike
smoke proved that the local rule could update the declared route, but an audit
then ran the real rehearsal circuit and found that its fast-spiking populations
never fired. No calibration, development, or held-out seed had been run. The
real reserved-seed circuit was therefore added as a mandatory smoke condition,
and the source-memory-to-FS afferent was mapped on seed `600` only. The inherited
weight `1.0` was silent. Aggregate activity initially hid that values through
`2.1` could leave one FS pool silent. Per-pool and per-route telemetry identified
that boundary; `2.2` was frozen as the first mapped value that recruited all
three FS pools and changed all six routes on both CPU and GPU. This amendment changes
the operating point before formal evidence rather than treating synthetic
spikes as proof that the biological route operates.

## Functional requirement

The same remembered event can carry evidence that it was seen, heard, or
self-generated. The source populations must remain distinct when one source is
weak and when two valid sources occur together. Competition should adapt from
local activity rather than applying one fixed suppressive weight to every
source.

## Mechanism under test

V4 inherits v2's co-resident episode, source-memory, fast-spiking, aPFC, and ACC
populations. It does not inherit v3's threshold homeostasis. Its adaptive
mechanism is homeostatic inhibitory spike-timing-dependent plasticity on
fast-spiking-to-rival-source GABA-A synapses. The source-memory-to-FS afferent is
fixed at `2.2` rather than v2's silent `1.0` operating point so the tested route
actually carries source activity.

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
- Source-memory-to-FS afferent weight: `2.2`, selected on smoke seed `600` only.
- Balanced rehearsal budget: a minimum design budget of `5,000` elapsed
  simulation steps, cycling
  equally through three single-source episodes and one mixed-source episode.
  Each block has `20` plasticity-open drive steps and `80` plasticity-closed rest
  steps. The executed protocol is exactly `13` cycles, `5,200` elapsed steps,
  and `1,040` plasticity-open steps per arm. The `5,000` value records the design
  minimum; validity requires the exact executed counts.
- Episode-to-source Hebbian learning runs only while inhibitory STDP is disabled.
  Inhibitory rehearsal runs only while ordinary Hebbian learning is disabled.
- Inhibitory traces are cleared when entering and leaving the host-separated
  episode-learning phase and before inhibitory rehearsal. This prevents recent
  activity from one learning rule becoming stale eligibility for the other.
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
parameter changes. Formal execution requires enabled provenance from an
immutable Git archive with a source manifest. Any failure keeps later phases
closed.

## Matched arms and controls

The intact and inhibitory-learning-lesion arms begin with identical networks,
receive identical source learning, and receive the same balanced rehearsal.
Their excitatory and inhibitory weights are compared again immediately before
rehearsal. Only the intact arm opens the inhibitory-plasticity gate.

Immediately after intact and learning-lesion rehearsal, the runner snapshots synaptic weights,
membrane and recovery variables, conductances, spike and refractory state,
inhibitory traces, thresholds, input currents, pathway gains, and simulation
time. Before every intact, learning-lesion, and expression-lesion recall, it
restores and verifies the corresponding exact post-rehearsal state. It then
changes only cross-source inhibitory transmission for the expression lesion.
Thus each source comparison begins from its arm's matched learned neural state
rather than from state left by an earlier recall trial.

The inherited controls remain mandatory: episode-route lesion, ACC-route
lesion, full competition-expression lesion, source-afferent swap, mixed-source
recall, unseen episode, and episode-learning-off. Recall receives episode
activity without source metadata.

## Fixed validity preconditions

A result is `UNDEFINED` if the exact calibration partition is absent, any seed
is reused from smoke or an earlier gate, matched arms do not begin with equal
episode and inhibitory weights, those weights differ immediately before
rehearsal, non-inhibitory weights or firing thresholds change during inhibitory
rehearsal, the learning lesion changes inhibitory weights, or any intact or
expression recall fails exact post-rehearsal restoration. A result is also
`UNDEFINED` unless the inhibitory route is exactly the six declared
FS-to-rival-source routes. That check enforces per-route count, plastic and
transmission gate membership, the complete plastic mask, inhibitory
presynaptic identity, GABA-A routing, and current learning/transmission gains.
Any non-finite scored value is also `UNDEFINED`.
The real rehearsal must also contain source-memory activity in every source
population and fast-spiking activity in every corresponding FS pool in both
matched arms, execute exactly `5,200` elapsed and `1,040` plasticity-open steps,
change all six inhibitory routes in the intact arm, and change none in the
learning lesion.

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

The mandatory smoke first trains the real episode-to-source routes, then runs
the exact balanced rehearsal on matched intact and learning-lesion brains. Both
must show activity in every source-memory and fast-spiking pool; all six intact
inhibitory routes must change while every lesion route stays fixed; excitatory
weights and thresholds must remain identical.
A separate synthetic-spike diagnostic checks rule scope by swapping the
coactive rival and requiring the larger local change to follow it. Together
these diagnose circuit engagement and update scope only; they do not assess
recall, behavior, or scientific acceptance.

## Host boundary and remaining scaffolds

The host still supplies sparse episode assemblies, source-afferent activity,
rehearsal order and timing, learning-window boundaries, phase-boundary trace
clearing, rest boundaries, spike
count readout, lesion switches, and scoring. Source anatomy and initial
inhibitory weights are predefined. V4 tests adaptive neural competition within
that bounded circuit; it does not claim natural episode formation, language,
confidence, response policy, or a complete self-model.
