---
type: preregistration
status: draft
date: 2026-08-03
mechanism: neural-vocal-action-credit-v5-learning
runner: research/runners/_vocal_action_credit_gate_v5_learning.py
---

# Gate B v5 learning: action trace to local reward prediction

**Filed before the learning runner exists and before any scientific seed is
assigned.** Reserved seed `0` may be used only for implementation smoke. Formal
entry points remain sealed until the corrected v5 smoke audit is clear, this
document is committed, and a separate seed manifest is committed.

## Question

Can a selected action leave a neural trace that trains a local reward-
expectation circuit, so a later shared reward event reinforces the responsible
action while an equal number of action-independent rewards does not create an
arbitrary preference?

The whole-brain role is narrower than conversation but load-bearing for it: a
brain must connect delayed social or environmental consequences to what it just
did without a host program carrying the action label across the delay.

## Mechanism

The frozen v5 smoke establishes only an action-selective trace. The learning
runner must separate that trace from learned value:

1. The existing commit/arousal dendritic circuit loads fixed action-trace
   populations during one fixed action epoch. These routes remain nonplastic.
2. Each trace converges with the shared outcome event on its own reward-
   expectation population. Only these trace-to-expectation routes and the
   existing cue-to-actor routes may change.
3. Reward-expectation populations inhibit the shared SNc-like dopamine circuit
   and the tonic omission gate through local pathways. Reward can therefore
   produce a positive prediction error, while expected omission can recruit the
   existing LHb-like to RMTg-like negative path.
4. The signed dopamine signal updates only synapses carrying local eligibility.
   The host never selects a route, stores the winner across the delay, computes
   prediction error, or writes a weight.

This is an engineering synthesis constrained by the local source record. It is
not a claim that one biological circuit implements the exact population split.
The relevant priors are three-factor striatal eligibility, subtractive expected-
reward inhibition in VTA/SNc circuitry, and LHb-to-RMTg signaling for outcomes
worse than expected. The committed implementation plan must cite the exact
local primary-source passages it uses.

## Host boundary

The experimental world may schedule a cue, one fixed action epoch, a fixed
delay, a generic outcome event, and the presence or absence of sensory reward.
It may record spikes and score a winner only after the action epoch. The
environment may make reward contingent on action `0`, but reward and outcome
afferents must remain shared and channel-neutral.

The host may freeze or open declared learning windows between protocol phases.
It may not stimulate a desired action, use winner-dependent timing or routing,
set eligibility or dopamine, calculate expected value, copy an action label
into the outcome phase, or apply a synaptic update.

## Seed lock

- Implementation smoke: reserved seed `0` only.
- Calibration: unassigned and sealed.
- Development: unassigned and sealed.
- Held out: unassigned and sealed.

After audit clearance and this preregistration commit, a deterministic seed-
assignment script must reject every seed used by vocal-credit v1-v4 or by any
other active gate. It must commit two calibration seeds, four development
seeds, and two held-out seeds before the learning runner can execute a formal
phase. No result may influence that assignment.

## Fixed protocol

Each seed constructs separately initialized brains for these arms:

1. contingent reward;
2. reward-count-matched, one-trial-shifted yoked reward;
3. all learning rates zero;
4. executed-action collateral lesion;
5. action-trace plateau lesion;
6. reward afferent to dopamine lesion;
7. expectation plasticity lesion;
8. expectation output lesion;
9. LHb-like to RMTg-like omission-path lesion; and
10. fixed action-channel permutation.

Every arm uses 20 frozen baseline trials, 40 training trials, frozen rewarded
and omitted outcome probes, and 40 frozen evaluation trials. The yoked arm uses
the contingent arm's exact reward count and a fixed one-trial schedule rotation.
No threshold, duration, current, or weight may change between arms or backends.

Calibration runs on NumPy and exact CuPy with one configuration. Development
and held-out execution require the same committed source revision and
configuration digest. A failed calibration retires its mechanism; it does not
unlock later partitions or permit threshold tuning on consumed seeds.

## Validity preconditions

A seed is `UNDEFINED`, never a scientific pass or fail, unless all conditions
hold:

1. Selector, action trace, expectation, actor, omission, RMTg-like, and SNc-like
   populations occupy one bridge.
2. Every action epoch, delay, and outcome window is fixed before neural activity
   is observed.
3. Reward and outcome afferents are symmetric across action channels.
4. The intact and permutation arms make one clean neural commit on at least 90%
   of baseline trials; bilateral commit or value output is rejected.
5. Coincidence synapses exist only on the declared action-trace routes.
6. Only cue-to-actor and trace-to-expectation synapses change, and every lesion
   reports both route engagement and zero unintended weight changes.
7. The yoked reward count exactly equals the contingent reward count.
8. Intact value and dopamine populations are active but remain below the
   committed physiological firing bound on NumPy and CuPy.
9. Every provenance sidecar names a clean source revision, exact backend,
   command, configuration digest, and fresh corpus check.

## Scientific criteria

Both fresh calibration seeds must pass every criterion:

1. Contingent frozen evaluation selects rewarded action `0` on at least 90% of
   clean trials, and at least 90% of those choices occur from the learned cue
   before shared arousal.
2. Yoked action-0 preference remains in `[0.25, 0.75]`; dominance in either
   direction fails.
3. Contingent training increases the rewarded trace-to-expectation route by at
   least 25% of its initial mean and separates it from the unrewarded route by
   at least 20% of that initial mean. In yoked learning, the absolute route-
   mean difference divided by their mean stays at or below 10%.
4. Repeated expected reward reduces the late dopamine response by at least 20%
   relative to early reward. Expectation-output lesion restores at least half
   of that reduction.
5. Frozen expected omission activates LHb-like and RMTg-like populations and
   lowers dopamine below its pre-outcome level. Omission-path lesion removes
   RMTg-like firing and at least 80% of the dip.
6. Learning-rate-zero, collateral-lesion, plateau-lesion, and reward-path-
   lesion arms do not acquire the contingent preference. Learning-rate-zero
   changes no declared weight.
7. Expectation-plasticity lesion prevents learned reward suppression and
   omission prediction. Expectation-output lesion increases either yoked
   preference distance from `0.5` by at least `0.15` or actor-route mean
   asymmetry by at least 20% of its initial mean relative to intact yoked
   learning, establishing that the critic is behaviorally load-bearing rather
   than decorative.
8. Channel permutation moves trace, learned expectation, and acquired action
   preference together; host relabeling alone cannot pass.
9. No synapse outside the two declared plastic route families changes.

The calibration verdict is GO only if both seeds pass every precondition and
criterion on both backends. Backend agreement is compatibility evidence, not
independent replication, because the operating point is developed jointly.

## Limits and successor rule

This gate does not demonstrate natural speech, social understanding, intrinsic
motivation, or general agency. Trial boundaries, outcome segmentation, the
binary environmental contingency, and global phase-level plasticity windows
remain scaffolds that later integrated work must replace or burn down.

Do not weaken yoked neutrality, omit the no-learning arm, accept an arbitrary
yoked preference because its direction varies by seed, or let the fixed action
trace itself count as learned value. If the smoke cannot establish active,
bounded expectation learning and a causal prediction-error signal, retire the
candidate before assigning formal seeds.
