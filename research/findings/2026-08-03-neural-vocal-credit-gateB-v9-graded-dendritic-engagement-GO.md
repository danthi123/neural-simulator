---
type: finding
status: qualified
date: 2026-08-03
mechanism: neural-vocal-action-credit-v9-graded-dendritic-readout
backend: cupy
seed: 0
---

# Gate B v9: a learned action route recruits pre-reward expectation through a dendrite

## Result

Gate B v9 passes its preregistered engagement smoke at plateau center `2`, the
last and first passing point in the fixed descending ladder `16/8/4/2`. The
result is qualified because it establishes a learned, action-specific neural
expectation before reward, not yet dopamine suppression or omission.

Artifacts, each stamped from implementation commit `8d69069e7`:

- `research/findings/raw/vocal_action_credit_gate_v9/engagement_center16.json`
- `research/findings/raw/vocal_action_credit_gate_v9/engagement_center8.json`
- `research/findings/raw/vocal_action_credit_gate_v9/engagement_center4.json`
- `research/findings/raw/vocal_action_credit_gate_v9/engagement_center2.json`
- `research/findings/raw/vocal_action_credit_gate_v9/construction_center16.json`
- `research/findings/raw/vocal_action_credit_gate_v9/construction_center8.json`
- `research/findings/raw/vocal_action_credit_gate_v9/construction_center4.json`
- `research/findings/raw/vocal_action_credit_gate_v9/construction_center2.json`

| Center | Status | Late rewarded expectation, intact / learning lesion / plateau lesion | Rewarded / other channel, intact | Dendritic attribution |
|---:|---|---:|---:|---:|
| `16` | fail | `2458 / 0 / 664` | `2458 / 1606` | `73.0%` | <!--derived-->
| `8` | fail | `1770 / 0 / 493` | `1770 / 1475` | `72.1%` | <!--derived-->
| `4` | fail | `865 / 0 / 68` | `865 / 1007` | `92.1%` | <!--derived-->
| `2` | **pass** | `167 / 0 / 1` | `167 / 48` | `99.4%` | <!--derived-->

At center `2`, every preregistered check passes:

- all three independently initialized conditions complete 12 trials with one
  clean neural action on every trial;
- no expectation neuron fires before learning;
- the rewarded route grows from `0.100` to `6.790`, while the other route ends
  at `1.800`; <!--derived-->
- the rewarded expectation population emits `167` late pre-outcome spikes,
  versus `48` in the other channel, clearing the fixed 3-to-1 rule;
- disabling expectation learning leaves both routes at `0.100` and removes all
  `167` rewarded-channel expectation spikes;
- clearing only the learned route's dendritic mask leaves its synapses, weights,
  ordinary transmission, and upstream trace plateau intact, but reduces the
  rewarded expectation from `167` spikes to `1`;
- no synapse outside the declared actor and expectation routes changes; and
- the expectation output gate remains zero in every condition, so the result
  cannot alter its own dopamine teaching signal.

## Interpretation

V6-v8 treated the MSN-D1 expectation population as a point-neuron receiver.
More learned weight, more presynaptic cells, and a separate fixed convergent
input did not produce a valid learned operating point. V9 changes the cellular
integration mechanism instead. The same local plastic route now contributes a
weighted dendritic signal through the bridge's smooth NMDA-like plateau before
the MSN soma.

The ladder maps a useful boundary rather than merely finding a high-activity
setting. Centers `16` and `8` produce abundant expectation, but ordinary
transmission retains more than 20% of it when the dendritic mask is removed.
Center `4` becomes dendrite-dependent but loses action selectivity. Center `2`
is both action-selective and causally dependent on learning and the dendritic
route. This is the first Gate B candidate to satisfy all three properties at
once.

The shared transfer also changes the upstream action-trace dendrite. That was
declared before execution and measured in all conditions: the trace plateau
remains identical when only the expectation-route mask is lesioned. The result
therefore supports the locked integrated operating point, not a claim that
center `2` is a universal striatal parameter.

## Controls and scope

- CuPy used the local NVIDIA GeForce RTX 3090 for all eight construction and
  engagement artifacts.
- Only reserved seed `0` ran; no formal partition exists.
- Every condition was independently initialized from the same seed and
  topology.
- The learning lesion changed only the named plasticity gate.
- The plateau lesion changed exactly the learned expectation-route mask bits
  and no weight or other dendritic route.
- Analog plateau conductance was recorded as telemetry. Expectation spikes,
  not a host read of that conductance, are the capability signal.
- Downstream GABA-B/GIRK, reward suppression, omission, NumPy/CuPy agreement,
  and formal seeds were not tested in this engagement ladder.

## Decision

`ENGAGEMENT_PASS / QUALIFIED-GO`. Lock plateau center `2`; do not revisit the
ladder. Supersede the engagement preregistration with a separately filed
reserved-seed output phase. Keep formal execution sealed until the output
phase, cross-backend agreement, and independent audit pass.
