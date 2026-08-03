---
type: finding
status: negative
date: 2026-08-03
mechanism: visual-identity-temporal-binding
runner: research/runners/_laneD_visual_identity_temporal_binding_gate.py
artifacts:
  - research/findings/raw/parallel_gates/visual_temporal_binding_calibration.json
  - research/findings/raw/parallel_gates/visual_temporal_binding_calibration.json.prov.json
---

# Temporal binding does not yet produce stable visual identity

<!--derived-->
**Verdict: NO-GO at calibration.** Both preregistered calibration seeds failed,
and neither result was undefined. The strict, audit-corrected criteria were
frozen before seeds `224` and `225` ran. Development and held-out seeds remain
locked.

## Result

Artifact: `research/findings/raw/parallel_gates/visual_temporal_binding_calibration.json`
with provenance sidecar
`research/findings/raw/parallel_gates/visual_temporal_binding_calibration.json.prov.json`.

The intact mechanism did not reach the required `0.50` held-view identity
decode on either seed. Its learned representation also failed the required
causal gaps against learning-off and pixel-scrambled controls.

| seed | intact decode | no learning | pixel scramble | additional failed causal criteria | verdict |
|---:|---:|---:|---:|---|:---:|
| 224 | 0.2188 | 0.2812 | 0.1562 | V1 FS; identity FS | NO-GO |
| 225 | 0.3125 | 0.2500 | 0.2500 | persistence; trace; V1 FS | NO-GO |

<!--derived-->
Seed `224` failed decode, learning necessity, the required intact-to-pixel
decode gap, and both fast-spiking-pathway causality checks. Seed `225` failed
decode, persistence and presynaptic-trace load-bearing checks, the required
learning gap, the required intact-to-pixel gap, and V1 fast-spiking-pathway
causality. Both seeds had empty undefined-reason lists, so these are scientific
failures rather than invalid runs.

## Provenance

The formal aggregate ran on a mini-PC with the NumPy backend from clean commit
`d24548b630a6c43012baa67bbed7bdde10add5a9`. The source manifest was
`f4561464b4e03be3a8ac9b024624c8bd7ab67fa56348f4eddd3764a42caad723`, and the
corpus check was fresh. After repairing the provenance record, an exact repeat
produced identical scientific results.

## Decision

Do not promote this temporal-binding mechanism or open development seeds
`226`, `227`, and `322`, or held-out seeds `323`, `324`, and `325`. The next
attempt must improve invariant identity learning while retaining the frozen
learning-off, pixel-structure, temporal, trace, persistence, and neural
fast-spiking causal controls.
