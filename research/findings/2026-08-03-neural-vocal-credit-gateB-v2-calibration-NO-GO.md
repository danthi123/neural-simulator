---
type: finding
status: negative
date: 2026-08-03
mechanism: neural-vocal-action-credit-v2
runner: research/runners/_vocal_action_credit_gate.py
artifacts:
  - research/findings/raw/vocal_action_credit_gate_v2/calibration_seed7.json
  - research/findings/raw/vocal_action_credit_gate_v2/calibration_seed7.json.prov.json
  - research/findings/raw/vocal_action_credit_gate_v2/calibration_seed11.json
  - research/findings/raw/vocal_action_credit_gate_v2/calibration_seed11.json.prov.json
---

# Action-value critic changes the arbitrary winner but does not neutralize yoked reward

<!--derived-->
**Verdict: NO-GO at Gate B v2 calibration.** Both clean calibration seeds
learned the contingently rewarded action on every frozen evaluation trial, but
unrelated reward still produced a completely dominant action. Seed 7 selected
action 1 on every yoked trial; seed 11 selected action 0 on every yoked trial.
Development and held-out seeds remain locked.

## Mechanism change

V1 established executed-action-local eligibility but reinforced arbitrary
actions under delayed yoked reward. V2 added two competing spiking action-value
populations. Each executed motor population trained its local value route, and
value activity inhibited SNc through slow GABA-B so expected outcomes could
reduce later dopamine teaching.

The fixed test retained contingent, reward-count-matched yoked,
executed-collateral-lesion, reward-to-SNc-lesion, and yoked-value-lesion arms.
Only the declared cue-to-actor and motor-to-value routes could change.

## Result

Both artifacts came from source commit `4fddb43e0` on the CuPy/NVIDIA backend.
Seed 11 ran from a clean worktree. Seed 7 was repeated from an immutable Git
archive with manifest
`f22e67f32d63a32f0e6326da66f886a9bc4f7182e09a590a63b9571cafc0a166`.
Both provenance sidecars report `git_dirty=false`. All five validity
preconditions passed on both seeds.

Artifacts: `research/findings/raw/vocal_action_credit_gate_v2/calibration_seed7.json`
and `research/findings/raw/vocal_action_credit_gate_v2/calibration_seed11.json`.
Their `.prov.json` sidecars record the source revision, manifest, backend, and
cleanliness checks.

| seed | contingent action 0 | cue-led | yoked action 0 | distance from balanced | collateral lesion | dopamine lesion | result |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 7 | 1.00 | 1.00 | 0.00 | 0.50 | 0.35 | 0.35 | no-go | <!--derived-->
| 11 | 1.00 | 1.00 | 1.00 | 0.50 | 0.60 | 0.50 | no-go | <!--derived-->

<!--derived-->
The critic did learn action-conditioned routes. In the yoked arm, final value
weights were `55.74 / 43.47` on seed 7 and `80.00 / 49.57` on seed 11. Actor
weights ended at `3.71 / 3.57` and `10.69 / 3.62`, respectively. These states
did not represent neutral behavior: the frozen selector saturated in opposite
directions across seeds.

The value lesion also changed which arbitrary action won without restoring
neutrality: its yoked evaluation selected action 0 at `1.00` on seed 7 and
`0.00` on seed 11. Thus the critic is behaviorally load-bearing, but not in the
required direction.

<!--derived-->
Average minimum dopamine on unrewarded yoked training trials was only `0.4702`
and `0.4772`, while rewarded peaks averaged `0.5482` and `0.5459`. V2 has no
dedicated frozen omission probe, and this shallow decrease did not prevent
self-reinforcing actor learning. Every other scientific check passed: local
eligibility, collateral and dopamine lesions, contingent cue-led acquisition,
and zero synaptic changes outside declared routes.

## Decision

Do not open v2 development seeds or search another value-to-SNc GABA-B gain on
these data. The successor is preregistered as Gate B v3. It keeps the v2
operating point and adds a spiking expected-omission path through LHb- and
RMTg-like populations, a reward-driven neural veto, and local fast-spiking
critic normalization. It must create a lesionable negative dopamine error for
expected omission while preserving reward bursts and contingent learning.

Fixed trial boundaries, outcome timing, action-channel anatomy, and global
plasticity windows remain explicit scaffolds. This result concerns a small
vocal action-credit circuit, not natural speech or general agency.
