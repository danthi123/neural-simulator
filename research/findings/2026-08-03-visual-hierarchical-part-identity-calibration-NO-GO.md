---
type: finding
status: negative
date: 2026-08-03
mechanism: visual-hierarchical-part-identity
runner: research/runners/_laneD_visual_hierarchical_part_identity_gate.py
artifacts:
  - research/findings/raw/visual_hierarchical_identity_v1/calibration.json
  - research/findings/raw/visual_hierarchical_identity_v1/calibration.json.prov.json
---

# Hierarchical visual identity remains silent under intact inhibition

<!--derived-->
**Verdict: NO-GO at calibration.** Both preregistered seeds were valid
scientific failures. The intact hierarchy produced no V2 or IT spikes, changed
no V2 or IT permanence, and decoded four held-out identities at chance (`0.25`)
with zero cosine margin. Development and held-out seeds remain locked.

## Mechanism tested

The candidate replaced host-selected V1 features with all deadline-fired V1
cells, local retinotopic V1-to-V2 part learning, per-hypercolumn fast-spiking
competition, and a sparse IT population trained by a presynaptic temporal trace.
It used identity-pure synthetic tracks as disclosed weak supervision, while
identity labels were restricted to scoring.

The exact battery compared intact, all-learning-off, V2-learning-off,
IT-trace-off, temporal-shuffle, V2-FS-lesion, IT-FS-lesion,
receptive-field-scramble, and pixel-scramble arms. The RF scramble preserved
synapse count, orientation composition, and overlap statistics while disrupting
retinotopy.

## Result

All ten validity preconditions passed on seeds `503` and `509`. Scientific
partitions were disjoint, labels did not enter encoding or learning, every
deadline-fired V1/V2/IT cell was used without top-k or first-k truncation, RF
controls were structurally matched, and every measurement was finite.

Both seeds produced the same intact result:

| seed | intact decode | intact margin | intact V2 changed | intact IT changed | intact V2 fired fraction | intact IT fired fraction |
|---:|---:|---:|---:|---:|---:|---:|
| 503 | 0.25 | 0.0 | 0 | 0 | 0.0 | 0.0 |
| 509 | 0.25 | 0.0 | 0 | 0 | 0.0 | 0.0 |

Every learning-off, trace-off, temporal, RF, and pixel control also decoded at
`0.25` with zero margin. The intact hierarchy therefore had no learned signal
for those controls to remove.

The V2 fast-spiking lesion did activate the otherwise silent pathway. On seed
503 it changed `13,824` V2 and `5,760` IT permanences, with V2/IT fired fractions
`0.202895/0.156351`. On seed 509 it changed `13,824` V2 and `24,576` IT
permanences, with fired fractions `0.223245/1.0`. Decoding nevertheless remained
at chance with zero margin. Removing IT inhibition alone did not activate the
silent V2 input.

<!--derived-->
This localizes the failure upstream: the intact V2 inhibitory operating point
silences representation learning, while removing it creates activity without
identity information and can saturate IT. The result does not support another
selector or scoring change.

## Provenance

Artifact: `research/findings/raw/visual_hierarchical_identity_v1/calibration.json`
with its `.prov.json` sidecar.

The aggregate ran for `407.1` seconds on the NumPy backend on `pool41`, from
immutable Git archive commit `b976f89459f50cd29664533c7149d36bc58f728d`,
manifest `17eadc7a732d1c35a904bb6857b8db0c469853e527291bc9fb1cac3d8b06fef9`,
run ID `1785788847-495468`. The provenance sidecar records successful full
source-manifest verification at both start and exit.

## Decision

Do not open development seeds `521/523/541` or held-out seeds `547/557/563`.
Do not tune against calibration seeds `503/509`, remove inhibition and accept a
saturated uninformative code, or change the decoder. Retire this candidate. A
fresh mechanism should first establish nonzero, non-saturated locally learned
V2 and IT activity on a reserved smoke seed, with inhibition causally improving
selectivity rather than suppressing all learning, before receiving a new seed
partition.
