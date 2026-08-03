---
type: finding
status: negative
date: 2026-08-03
mechanism: replay-driven-cortical-consolidation
runner: research/runners/_replay_cortical_consolidation_gate.py
artifacts:
  - research/findings/raw/parallel_gates/replay_cortical_consolidation_calibration_seed212.json
  - research/findings/raw/parallel_gates/replay_cortical_consolidation_calibration_seed213.json
---

# Replay consolidation is causal but not yet reliable

<!--derived-->
**Verdict: NO-GO at calibration.** Uncued hippocampal replay changed cortical
synapses and produced a small amount of hippocampus-independent recall, but the
effect was weak, noisy, and inconsistent across the two calibration seeds.
Development and held-out seeds remain locked.

## Question

Can one continuously persisted spiking brain encode two interfering episodes,
reactivate both during uncued sleep, and leave a cortical memory that can later
be recalled from a partial cue while hippocampal transmission is disabled?

The host provided episode input during wake, weak episode-agnostic CA3 noise
during sleep, and a partial cortical cue during retest. It did not choose which
episode replayed, copy a memory into cortex, label the correct answer at
inference, or compute recall outside the spike record.

## Calibration Design

Seeds 212 and 213 were run independently on the mini-PC CPU pool. Every
condition preserved one bridge through encoding A, interfering encoding B,
uncued sleep, and hippocampus-disabled retest. The four controls removed sleep,
permuted the learned CA1-to-target index weights, lesioned CA3-to-CA1
transmission during sleep, or disabled cortical sleep plasticity.

This calibration tests whether replay activity can form a cortical
cue-to-target association. It does not test the temporal order of replay. The
`shuffled_target_index` arm permutes the CA1 target-index weight assignment; it
is not a temporally shuffled replay control.

## Result

Artifacts:
`research/findings/raw/parallel_gates/replay_cortical_consolidation_calibration_seed212.json`
and
`research/findings/raw/parallel_gates/replay_cortical_consolidation_calibration_seed213.json`.
Both artifacts were produced from clean source commit `3e81de810` with the
NumPy backend; their provenance sidecars record the commands and environments.

| seed | intact recovery | intact margin | false recall | no sleep | shuffled index | CA3-CA1 lesion | plasticity off | result |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 212 | 0.0333 | 0.0153 | 0.3581 | 0.0000 | 0.0201 | 0.0000 | 0.0000 | no-go | <!--derived-->
| 213 | 0.0021 | 0.0014 | 0.1250 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | no-go | <!--derived-->

Both intact brains spontaneously reactivated both episodes during all 24 sleep
events. Cortical association weights changed only during sleep in the intact
condition, by a mean 1.393 on seed 212 and 0.069 on seed 213. <!--derived--> Disabling sleep,
the CA3-to-CA1 route, or cortical plasticity eliminated cortical recall.

Those causal controls establish a real replay-to-cortex path, but they do not
establish useful consolidation. Seed 212 exceeded the minimum recovery and
margin floors, then failed the false-recall bound and retained only 39.6% of
its recovery advantage over the permuted-index control. Seed 213 produced four
correct target spikes for memory A and none for memory B, missing the minimum
recovery, two-memory, margin, and causal-difference criteria. The 16-fold
difference in intact recovery between seeds is too large to treat as a stable
mechanism.

## Root Cause

The current CA1-to-cortex reinstatement is below a reliable operating regime.
Seed 212 drove enough cortical coactivity to strengthen broad associations,
including incorrect targets; seed 213 drove almost no cortical learning. A
single strong cortical learning rate therefore alternates between diffuse
false recall and near-inert learning depending on the sampled network.

The present control battery also cannot support a claim about ordered replay.
It isolates the learned CA1 target index and the replay-to-cortex causal chain,
but never changes replay timing while preserving the same events.

## Decision

Do not open development seeds 214, 215, or 310. Preserve this runner and the
negative artifacts as the baseline. The next attempt must first make
CA1-to-cortex reinstatement reliable without increasing false recall, using a
local competitive or inhibitory mechanism rather than a larger host-selected
drive. It must then add a true temporal-order control before claiming that
replay sequence, rather than replayed coactivity alone, is load-bearing.

The next calibration remains bounded to seeds 212 and 213. Its fixed criteria
must retain two-memory recovery, an absolute false-recall ceiling, the no-sleep
and pathway lesions, cortical-plasticity necessity, and a stronger advantage
over the permuted-index control.
