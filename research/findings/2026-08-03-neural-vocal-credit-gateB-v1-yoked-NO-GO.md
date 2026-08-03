---
type: finding
status: negative
date: 2026-08-03
mechanism: neural-vocal-action-credit
runner: research/runners/_vocal_action_credit_gate.py
artifacts:
  - research/findings/raw/vocal_action_credit_gate/calibration_v2_seed7.json
  - research/findings/raw/vocal_action_credit_gate/calibration_v2_seed11.json
---

# Vocal credit v1 fails the yoked-reward control

<!--derived-->
**Verdict: NO-GO at Gate B v1 calibration.** Delayed contingent reward
strengthened only locally eligible vocal-action routes and both causal lesions
blocked learning. A matched noncontingent reward schedule nevertheless learned
the same action on seed 11. The circuit carries raw reward rather than an
action-conditioned reward prediction error, so development and held-out seeds
remain locked.

## Question

Can a shared cue, local executed-action collateral, and a later spiking SNc
dopamine broadcast reinforce the action that actually occurred without a host
channel label?

The host presented one identical cue to both channels, observed the first
motor threshold crossing, and later presented a reward-US event. It did not
set an eligibility trace, dopamine concentration, neural action, or synaptic
weight by channel. Only two cue-to-D1-actor pathways were plastic.

## Calibration Corrections

Early smoke runs exposed two measurement problems and one circuit problem
before this fixed comparison:

1. The first measured trial began after warmup but before the reset and
   washout used between later trials. The final runner applies the same reset
   before every measured trial.
2. Motor collateral weights of 60-100 formed a recurrent
   motor-to-actor-to-GPi loop that could survive the inter-trial interval. A
   measured subthreshold value of 20 retained winner-local actor firing without
   that recurrence.
3. A learned cue could trigger a real motor action before shared arousal, but
   the first runner discarded that action. The final runner scores the earliest
   neural threshold crossing across cue-lead and arousal phases.

Calibration used seeds 7 and 11 only. Development seeds 42, 43, 44, and 100
and held-out seeds 101 and 102 were not inspected.

## Result

Each condition used 20 baseline, 40 training, and 30 frozen evaluation trials
on the CuPy production bridge. The probe contained 716 neurons, 45 declared
pathways, and about 31,500 synapses. Automatic provenance sidecars record the
commands, backend, source state, and run IDs.

Artifacts:
`research/findings/raw/vocal_action_credit_gate/calibration_v2_seed7.json` and
`research/findings/raw/vocal_action_credit_gate/calibration_v2_seed11.json`.

| seed | condition | baseline action 0 | evaluation action 0 | cue-led evaluation | actor weights 0 / 1 | outside changes |
|---:|---|---:|---:|---:|---:|---:|
| 7 | contingent | 0.474 | 1.000 | 1.000 | 33.953 / 0.236 | 0 | <!--derived-->
| 7 | yoked | 0.474 | 0.000 | 1.000 | 9.330 / 24.555 | 0 | <!--derived-->
| 7 | collateral lesion | 0.474 | 0.633 | 0.000 | 0.100 / 0.100 | 0 | <!--derived-->
| 7 | dopamine-path lesion | 0.474 | 0.567 | 0.000 | 0.100 / 0.100 | 0 | <!--derived-->
| 11 | contingent | 0.650 | 1.000 | 1.000 | 37.268 / 0.255 | 0 | <!--derived-->
| 11 | yoked | 0.650 | 1.000 | 1.000 | 25.475 / 12.537 | 0 | <!--derived-->
| 11 | collateral lesion | 0.650 | 0.500 | 0.000 | 0.100 / 0.100 | 0 | <!--derived-->
| 11 | dopamine-path lesion | 0.650 | 0.467 | 0.000 | 0.100 / 0.100 | 0 | <!--derived-->

Contingent learning met the local-credit observations on both seeds. The
executed-to-losing eligibility ratio was at least 10:1 on every rewarded trial,
all evaluation choices became cue-led action 0, and no synapse outside the two
declared routes changed. Closing the collateral path left no eligibility or
weight change. Closing reward-US-to-SNc preserved selection and left both
route means exactly at 0.1.

The yoked control is decisive. It preserved the contingent run's reward count
and shifted its trial schedule, yet seed 11 still learned action 0 and seed 7
learned action 1. The learned action followed early local action-reward
coincidence rather than the experiment's stable contingency. This is valid
local three-factor plasticity, but it is not reliable contingent learning.

## Root Cause

Reward-US excitation raises SNc firing directly. There is no neural estimate
of expected reward for the executed action, so every reward remains a positive
teaching event. Under yoked feedback, whichever route is active during enough
early rewards can strengthen, bias later choices, and create a self-reinforcing
loop. Local eligibility solves *where* a broadcast applies; it does not by
itself solve whether the outcome was better or worse than expected.

## Decision

Gate B v1 does not advance. Preserve the local eligibility and causal lesion
mechanisms, then add a spiking action-conditioned value critic. The intended
next circuit uses executed motor activity to learn a local striosome value,
striosome-to-SNc GABA-B inhibition to subtract expected value, tonic SNc firing
to support both bursts and dips, and the resulting shared dopamine prediction
error to update actor and critic routes. Freeze that v2 design on calibration
seeds before opening development seeds.
