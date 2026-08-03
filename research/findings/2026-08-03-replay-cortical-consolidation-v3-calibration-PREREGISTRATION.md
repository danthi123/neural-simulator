---
type: preregistration
status: live
date: 2026-08-03
mechanism: replay-driven-cortical-consolidation-v3
runner: research/runners/_replay_cortical_consolidation_gate_v3.py
---

# Replay consolidation v3: learned cortical index and local replay balance

**Filed after non-scientific smoke seed `216` was used for implementation checks
and before any v3 scientific seed was run.** Smoke seed `216` is mechanically
excluded from every scientific partition and cannot produce a calibration
verdict.

## Functional requirement

Sleep replay must let the same brain recover an episode after the hippocampal
route is disabled. Both memories must survive, false recall must stay bounded,
and improvement must depend on learned event identity and replay order. Broad
coactivity or a host-selected cortical target is not sufficient.

## Mechanism under test

V2's local fast-spiking opponent inhibition remains. V3 adds a neutral,
broad-fan-in CA1-to-cortical-index pathway whose weights change from local wake
coactivity. During sleep, weighted dendritic coincidence and local recurrence
reactivate the learned index assembly, which drives its cortical target.
Index assemblies also recruit local slow GABA-B feedback so the stronger memory
adapts across replay events instead of monopolizing consolidation.

The biological role is a bounded approximation of hippocampal indexing,
dendritic coincidence, cortical recurrent reinstatement, and slow local
inhibitory adaptation. It does not claim complete hippocampal or sleep
microcircuit anatomy.

## Seed and phase lock

- Non-scientific smoke only: `216`.
- Calibration, currently open: `228`, `229`.
- Development, locked: `230`, `231`, `326`.
- Held out, locked: `327`, `328`, `329`.

Both calibration seeds must pass every fixed criterion without tuning between
them. A failure stops the gate and leaves later partitions closed.

## Fixed protocol

Every seed runs separately initialized copies of these conditions: intact,
no-sleep, exact-content shuffled replay order, shuffled learned target index,
CA3-to-CA1 lesion, cortical-plasticity-off, target-inhibition lesion,
index-relay lesion, and index-balance lesion. Every condition uses one bridge
through wake encoding of A, wake encoding of B, uncued sleep, and cortical
retest with hippocampal retrieval disabled.

The shuffled-order condition permutes the exact sleep event list. It must
preserve every stimulated cell and the event-content multiset while changing
order and adjacent overlap. No host process chooses an episode or cortical
target during sleep.

## Fixed validity preconditions

A result is `UNDEFINED` unless every condition preserves one bridge and the
fixed phase sequence; wake current reaches CA3, CA1, cue, target, and index but
never directly drives the index; intact sleep recruits uncued replay, the
learned relay, target fast-spiking inhibition, and local slow balance; the
temporal control preserves content while changing order; no-sleep is
quiescent; every lesion reaches its declared gate; and cortical-plasticity-off
holds cortical weights fixed.

## Fixed scientific criteria

Every item must pass on both calibration seeds:

1. Both memories replay during intact sleep.
2. The index has neutral all-to-all CA1 fan-in before learning.
3. Index weights change by more than `1e-5` during wake and less than `1e-7`
   during sleep.
4. Cortical association weights change by more than `1e-5` during sleep, less
   than `1e-7` during wake, and less than `1e-7` with cortical plasticity off.
5. Intact mean recovery is at least `0.03`, mean margin at least `0.015`, each <!--derived-->
   memory's recovery at least `0.015`, and the weaker memory at least `35%` of <!--derived-->
   the stronger.
6. Intact mean false recall is at most `0.15`.
7. Intact recovery exceeds no-sleep by `0.015`, shuffled order by `0.01`, <!--derived-->
   shuffled target index by `0.015`, CA3-to-CA1 lesion by `0.015`, <!--derived-->
   cortical-plasticity-off by `0.015`, and index-relay lesion by `0.015`. <!--derived-->
8. The weaker intact memory exceeds the index-balance lesion by `0.005`, while <!--derived-->
   intact false recall is no more than `0.025` above that lesion. <!--derived-->
9. Target inhibition lowers false recall by at least `0.05` versus its lesion
   while retaining at least `75%` of the lesion's recovery.

## Host boundary and scaffolds

The host defines wake episode populations, partial probes, fixed relay and
inhibitory channel membership, sleep down-state boundaries, and
episode-agnostic CA3 background current. It reads known assemblies for scoring.
It may not rank memories, choose a replay event, or stimulate a selected
cortical target during sleep. Fixed anatomy, a wake teacher pathway,
rate-window Hebbian learning, and scheduled sleep remain explicit scaffolds.

If calibration fails, record the negative result and identify the failed
mechanism. Do not tune against the same seeds or open development.
