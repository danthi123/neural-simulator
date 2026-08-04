---
type: finding
status: no-go
date: 2026-08-03
mechanism: neural-vocal-action-credit-v8-dual-afferent-upstate
backend: cupy
seed: 0
---

# Gate B v8: fixed convergence does not provide a learned operating point

## Result

Gate B v8 is retired at its complete preregistered engagement ladder. On
reserved smoke seed `0` and the RTX 3090, a distinct fixed state trace and a
plastic action trace both reached the MSN-D1 expectation population. Weight
`2` kept the fixed arm subthreshold, but the learned route still produced no
pre-outcome expectation spikes. Weights `4`, `8`, `12`, and `16` produced
expectation activity before reward, but the fixed arm already produced activity
without learning. No point passed the causal learning-lesion requirement.

Artifacts, each stamped with source commit `1e8251816` in the provenance log:

- `research/findings/raw/vocal_action_credit_gate_v8/engagement_fixed2_seed0_cupy.json`
- `research/findings/raw/vocal_action_credit_gate_v8/engagement_fixed4_seed0_cupy.json`
- `research/findings/raw/vocal_action_credit_gate_v8/engagement_fixed8_seed0_cupy.json`
- `research/findings/raw/vocal_action_credit_gate_v8/engagement_fixed12_seed0_cupy.json`
- `research/findings/raw/vocal_action_credit_gate_v8/engagement_fixed16_seed0_cupy.json`

| Fixed weight | Clean trials, intact / learning lesion / fixed lesion | Intact route mean, `0.100 ->` | Baseline expectation, intact | Late action-0 expectation, intact / learning lesion / fixed lesion |
|---:|---:|---:|---:|---:|
| `2` | `11 / 11 / 12` | `0.549` | `0` | `0 / 0 / 0` | <!--derived-->
| `4` | `12 / 12 / 12` | `0.686` | `30` | `213 / 113 / 0` | <!--derived-->
| `8` | `12 / 7 / 12` | `80.000` | `236` | `1252 / 629 / 0` | <!--derived-->
| `12` | `7 / 6 / 12` | `79.620` | `362` | `988 / 1126 / 0` | <!--derived-->
| `16` | `11 / 11 / 12` | `45.658` | `451` | `725 / 1266 / 0` | <!--derived-->

All runtime prerequisites held at every point. The learning lesion kept its
route exactly at `0.100`, the fixed-arm lesion gate was zero, both presynaptic
trace populations fired before outcome, and no synapse outside the declared
plastic routes changed. The 8- and 12-weight points also missed the 90% clean-
action requirement in at least one condition, but their causal learning checks
had already failed.

## Interpretation

The tested architecture has no valid operating point on this coarse bounded
ladder. At weight `2`, learned plastic input is not enough to cross the MSN
threshold. At weight `4`, the fixed route has already crossed it without
learning: the learning-lesion arm retains `113/213`, or `53.1%`, of intact
late expectation activity. At weights `12` and `16`, the no-learning control
exceeds intact. This is fixed-drive dominance, not learned convergence.

This result also sharpens the boundary exposed by v6 and v7. Scaling one
plastic input was silent; adding a second fixed input changes the response too
abruptly between weights `2` and `4` to give the tested point-neuron circuit a
causally learned, subthreshold operating region. The apparent gap is not
permission for post-result interpolation. Weight interpolation was not
preregistered, and the v8 stop rule explicitly retires this bootstrap
mechanism when no point passes.

The capability remains open. A successor must enter a new research and
preregistration gate and change the mechanism, not merely tune this ladder.
The evidence now favors mechanisms that can regulate or localize nonlinear
integration, such as dendritic plateau/clustered-input dynamics or a locally
maintained membrane-state mechanism, while preserving the same learning-off,
fixed-arm-off, action-cleanliness, and plastic-leakage controls. That is a
research target, not yet an approved implementation.

## Controls and scope

- CuPy used `NVIDIA GeForce RTX 3090` for every run.
- Only reserved seed `0` ran; no development or held-out seed was assigned.
- All three conditions were independently initialized with matched seed and
  topology at each weight.
- The fixed route was nonplastic and disabled by a transmission gate in its
  lesion; the population and weights remained present.
- The expectation-learning lesion disabled only the local plastic route.
- The full GABA-B/GIRK suppression and omission smoke did not run because the
  engagement prerequisite never passed.
- The ladder was not interpolated, extended, or repeated with another seed.

## Decision

`ENGAGEMENT_FAIL / NO-GO`. Retire the v8 fixed-plus-plastic dual-afferent
bootstrap and keep formal execution sealed. Before another Gate B build, query
the project record and primary literature for a biologically distinct way to
produce graded, learning-dependent MSN integration. Do not reopen v6-v8
weight, size, or fixed-drive ladders.
