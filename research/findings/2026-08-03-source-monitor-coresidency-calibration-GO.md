---
type: finding
status: positive
date: 2026-08-03
mechanism: source-monitor-coresidency
runner: research/runners/_laneC_source_monitor_coresidency_gate.py
artifacts:
  - research/findings/raw/parallel_gates/source_monitor_coresidency_calibration_seed212.json
  - research/findings/raw/parallel_gates/source_monitor_coresidency_calibration_seed212.json.prov.json
  - research/findings/raw/parallel_gates/source_monitor_coresidency_calibration_seed213.json
  - research/findings/raw/parallel_gates/source_monitor_coresidency_calibration_seed213.json.prov.json
---

# Source memory works on the shared monitoring substrate

<!--derived-->
**Verdict: GO at two-seed calibration.** A single 304-neuron spiking bridge
learned whether an episode was seen, heard, or self-generated, then reinstated
that source from episode activity alone and propagated it into neural aPFC and
ACC populations. All 12 calibration checks passed on both seeds.

## Why This Matters

The previous source-memory rung learned support on a dedicated bridge and
passed a host scalar into the speech-safety path. This experiment moves the
episode, source-memory, aPFC, and ACC populations onto one continuously
persisted bridge. It is a step toward a brain that can distinguish remembered
experience from its own generated activity before speaking.

## Mechanism

During an experience, a sparse episode assembly and physical source afferents
co-fire. The source afferents represent visual activity, auditory activity, or
motor corollary discharge. Zero-initialized episode-to-source synapses learn
from that coactivity. During recall, the caller supplies only the episode
activity; the learned source spikes then travel to source-specific aPFC pools
and ACC on the same bridge.

The inference function has no source label, answer, proposition, confidence
score, or response decision. Population spike counts are read only to evaluate
the experiment.

## Result

Both cluster runs used clean source commit `3e81de810`, the NumPy backend, and
the same `git_archive` source manifest. Their provenance sidecars also preserve
the scientific dispatch rationale.

Artifacts:
`research/findings/raw/parallel_gates/source_monitor_coresidency_calibration_seed212.json`
and
`research/findings/raw/parallel_gates/source_monitor_coresidency_calibration_seed213.json`.

| seed | seen margin | heard margin | self-generated margin | source spikes | source lesion | ACC spikes | ACC lesion | checks |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 212 | 0.2200 | 0.1967 | 0.1883 | 264 | 0 | 390 | 0 | 12/12 | <!--derived-->
| 213 | 0.1942 | 0.1867 | 0.2208 | 233 | 0 | 264 | 0 | 12/12 | <!--derived-->

The learned pathways began at exactly zero. Experience made 607 synapses
nonzero on seed 212 and 629 on seed 213. Disabling learning left every source
weight at zero and produced no source recall. Swapping the physical afferents
made source recall follow the swapped experience. Mixed visual-auditory
episodes reinstated both sources. Lesioning the episode-to-source path removed
100% of the tested source activity, while lesioning source-to-ACC transmission
preserved source recall and removed 100% of ACC activity.

## Honest Boundary

This does not yet establish robust source monitoring or truthful speech. The
upstream sparse episode activity is supplied by the test world, source-afferent
identity and the learning window remain developmental scaffolds, and this gate
does not choose words or decide whether to speak. It tests a small isolated
mechanism inside one bridge, not the complete continuously learning brain.

## Decision

Freeze the operating point and open development seeds 214, 215, and 310 under
the separately filed gate. Development must add an unseen-episode control and
meet stronger margin and causal-attribution floors. Held-out seeds 311, 312,
and 313 remain mechanically inaccessible until the development result is
recorded.
