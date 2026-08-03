---
type: finding
status: negative
date: 2026-08-03
mechanism: replay-driven-cortical-consolidation-v2
runner: research/runners/_replay_cortical_consolidation_gate_v2.py
artifacts:
  - research/findings/raw/parallel_gates/replay_cortical_consolidation_v2_calibration_seed212.json
  - research/findings/raw/parallel_gates/replay_cortical_consolidation_v2_calibration_seed212.json.prov.json
  - research/findings/raw/parallel_gates/replay_cortical_consolidation_v2_calibration_seed213.json
  - research/findings/raw/parallel_gates/replay_cortical_consolidation_v2_calibration_seed213.json.prov.json
---

# Local inhibition improves replay specificity but consolidation remains fragile

<!--derived-->
**Verdict: NO-GO at v2 calibration.** Opponent fast-spiking inhibition made
replayed cortical memories substantially more selective and a true replay-order
control now preserves event content while changing order. One seed still
under-recovered its second memory, and the target-index and temporal-order
advantages were not repeatable. Development and held-out seeds remain locked.

## Mechanism Change

V1 alternated between diffuse false recall and nearly inert cortical learning.
V2 leaves host drive magnitudes unchanged and adds local cortical competition:
each target assembly recruits a fast-spiking interneuron channel that inhibits
the competing assembly. The desired assembly can use recurrence while diffuse
competitors are suppressed.

The same bridge persists through wake encoding, interfering memory encoding,
uncued sleep, and hippocampus-disabled cortical retest. The new
`shuffled_replay_order` arm permutes the exact same CA3 background events. It
changes their order without changing any event's stimulated cells.

## Result

Artifacts:
`research/findings/raw/parallel_gates/replay_cortical_consolidation_v2_calibration_seed212.json`
and
`research/findings/raw/parallel_gates/replay_cortical_consolidation_v2_calibration_seed213.json`.
Both ran on the mini-PC pool from clean source commit `52786d103`, the NumPy
backend, and one revision-addressed source manifest. All eight validity
preconditions passed on both seeds.

| seed | intact recovery | margin | false recall | shuffled order | shuffled target index | inhibition lesion recovery / false | result |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 212 | 0.0729 | 0.0722 | 0.0192 | 0.0576 | 0.0639 | 0.0306 / 0.3417 | no-go | <!--derived-->
| 213 | 0.0194 | 0.0174 | 0.1875 | 0.0167 | 0.0174 | 0.0167 / 0.4000 | no-go | <!--derived-->

<!--derived-->
Seed 212 improved over v1 from `0.0333` to `0.0729` mean recovery while mean
false recall fell from `0.3581` to `0.0192`. Memory A recalled at `0.1111` with
zero false spikes; memory B recalled at `0.0347` with `0.0385` false recall.
The inhibition lesion sharply increased false recall and reduced recovery,
showing that the new circuit is load-bearing for specificity.

Seed 213 improved mean recovery from `0.0021` to `0.0194`, but memory B reached
only `0.0069` correct rate with `0.3750` false recall. Its intact result did not
beat shuffled replay order by the fixed margin. Neither seed beat the
shuffled target-index arm by the required `0.015` recovery difference. Thus
the experiment does not yet establish that the learned index, rather than
broad coactivity, reliably carries the transfer.

Both temporal controls preserved the exact event-content digest and changed
the order digest and adjacent-overlap structure. No-sleep, CA3-to-CA1 lesion,
and cortical-plasticity-off arms produced zero recovery. This makes the no-go
well-defined rather than an instrument failure.

## Decision

Keep opponent inhibition and the true temporal-order control. Do not open
development seeds 214, 215, or 310. The next change should stabilize the weak
memory-B CA1-to-cortex reinstatement and make the learned target index
causally stronger without increasing false recall or host-selected drive.
Another global learning-rate or current sweep is not justified by these data.

Remaining scaffolds include host-defined episode assemblies and partial cues,
fixed inhibitory channel membership, scheduled sleep down-state boundaries,
host-provided episode-agnostic CA3 current, and measurement against known
calibration assemblies.
