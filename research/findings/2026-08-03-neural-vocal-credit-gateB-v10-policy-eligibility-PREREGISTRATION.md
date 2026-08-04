---
type: preregistration
status: active
date: 2026-08-03
mechanism: neural-vocal-action-credit-v10-policy-eligibility
runner: research/runners/_vocal_action_credit_gate_v10_policy.py
---

# Gate B v10: action-local eligibility in the selector policy

**Filed before the v10 runner exists.** This is a reserved-seed engagement
smoke, not a policy-learning result. Seed `0` is the only executable seed.
Formal calibration, development, and held-out seeds are unassigned and sealed.

The question is narrow: when the existing neural selector commits an action,
does its normal activity leave a sufficiently selective eligibility trace on
the same proposal-to-D1/D2 synapses that control future choices? No reward,
dopamine teaching signal, or weight learning is allowed in this phase.

## Fixed construction

1. Start from the unchanged Gate A v2 selector: 600 neurons, shared practice
   arousal, proposal noise, two D1/D2 basal-ganglia channels, thalamic commit,
   and motor output.
2. Change only the four same-channel `proposal -> str_d1` and
   `proposal -> str_d2` pathway declarations from fixed to reward-plastic and
   assign them to `vocal_selector_policy_learning`. Keep density `1.0`, mean
   weight `400`, jitter `0.05`, ordinary transmission, and all other selector
   parameters unchanged.
3. Enable the existing local pre-trace/post-spike coactivity eligibility with
   eligibility decay `300 ms`, presynaptic trace decay `80 ms`, coactivity
  threshold `0.001`, and scale `20.0`. <!--derived-->
4. Enable cell-identity D1/D2 signs. Every proposal-to-D1 synapse must have
   sign `+1`; every proposal-to-D2 synapse must have sign `-1`.
5. Set reward learning rate to exactly `0`, current reward signal and reward
   baseline to exactly `0`, and `hebbian_min_weight` /
   `hebbian_max_weight` to `[0, 600]`. Also set the inactive STDP bounds to the
   same values. Disable STDP, Hebbian learning, homeostasis, structural
   plasticity, and the neuromodulator subsystem.
6. Include no credit cue, actor, expectation, outcome, reward-US, SNc, LHb,
   RMTg, or omission population or pathway. The reward-eligible and plastic
   synapse sets must both equal exactly the four policy routes.
7. Add one engagement-only scalar `reward_coactivity_trace_input_gain`. It
   multiplies only the new fired-neuron term entering the presynaptic
   coactivity trace; exponential decay continues at every step. Its default is
   `1.0`, it never changes neural transmission or any firing state, and it may
   not inspect a winner or differ by channel. This is measurement
   instrumentation, not part of the policy candidate.

The runner must assert that every jittered initial policy weight lies strictly
inside `[0, 600]` and report the minimum and maximum. Rescaling the policy or
changing Gate A physiology is forbidden.

## Fixed protocol

Run two separately constructed, seed-identical CuPy conditions:

- `intact`: native coactivity eligibility enabled;
- `coactivity_lesion`: only coactivity eligibility disabled. Policy
  transmission remains `1.0`, and action-phase plasticity gain remains `1.0`;
  every other configuration, initial weight, current, timing step, and noise
  seed is identical.

In both conditions, policy transmission stays at `1.0`. The policy plasticity
gain is `0` during warmup, reward delay, reset, and washout, and `1.0` only
during the fixed action window. This phase gate is symmetric across all four
routes and never depends on the winner. It prevents neutral OU activity from
creating new eligibility while preserving neural dynamics and trace decay.
The coactivity-trace input gain follows the same fixed phase schedule: `0`
during every neutral phase and `1` for the complete action window. It is never
changed after observing a neural event.

Warm up for exactly `80` neutral steps. Run `12` trials per condition. Each
trial has:

1. exactly `600` action steps with shared practice arousal and equal tonic
   currents;
2. an eligibility snapshot on the exact step of the first unique neural motor
   threshold crossing, without stopping the action window;
3. an eligibility snapshot after all `600` action steps, then policy
   plasticity gain closes symmetrically;
4. exactly `100` neutral reward-delay steps followed by the pre-outcome
   eligibility snapshot;
5. exactly `35` selector-reset steps;
6. exactly `3000` neutral washout steps, equal to ten eligibility time
   constants.

The simulation never stops a phase after observing a winner. The action is the
first motor channel to cross the unchanged `12`-spike commitment threshold
alone. It is clean only if the losing channel has at most `25%` of the winner's
motor spikes at that crossing and does not cross the threshold later in the
fixed action window. There is no argmax, fallback, forced action, channel mask,
winner-index trace edit, outcome, or reward. Decision-time selectivity is the
primary measure; action-end and pre-outcome selectivity show whether the local
tag persists to the fixed delayed-consequence time.

Record a pretrial eligibility baseline for every route on every trial. All
four means and every individual trace value must be exactly zero before trial
one. Before later trials, each route's mean must be at most `0.1%` of that same
route's preceding pre-outcome mean, and the maximum absolute trace over all
policy synapses must be at most `0.01`. If a preceding route mean is zero, its
next baseline must also be zero. Failure is `UNDEFINED_WASHOUT`, not evidence
against action-local eligibility.

Also record the complete presynaptic coactivity trace immediately before every
action window. It must be exactly zero before trial one and have maximum
absolute value at most `1e-6` thereafter. No host assignment may clear either
trace array. Failure is `UNDEFINED_PRETRACE`, not an engagement result.

## Locked measurements

For each snapshot, each clean intact trial, and separately for D1 and D2,
remove only analytically expected carryover:

`net = max(raw route mean - pretrial route mean * exp(-elapsed_ms / 300), 0)`.

This host-side calculation is analysis only and never alters simulator state.
From those net route means define:

- `selected`: mean absolute eligibility on the matching route for the neural
  winner;
- `loser`: mean absolute eligibility on the other action's same-class route;
- `margin = selected - loser`;
- `ratio = selected / max(loser, 1e-12)`.

Report raw and net values at all three snapshots for every row. Unclean rows
remain in the artifact but are excluded from winner-labeled GO aggregates.
Also report proposal, D1, D2, thalamic, commit, and motor spikes; first crossing;
decision step; cleanliness reason; loser ratio; residual pretrial eligibility;
a hash of the complete all-neuron firing vector after every step; complete
weight hashes; elapsed time; GPU identity; memory use; neuron count; and
synapse count.

## Validity prerequisites

All must hold before the selectivity verdict is interpreted:

1. backend is CuPy on seed `0`, with no formal phase open;
2. construction matches the fixed topology, plastic/eligible ownership,
   policy gate, D1/D2 signs, bounds, and host boundary above;
3. all initial policy weights are strictly within bounds;
4. `intact` and `coactivity_lesion` start with byte-identical complete weight
   arrays and have identical whole-run per-step firing hashes. Consequently,
   every first crossing, decision step, cleanliness value, motor count, and
   12-entry action sequence, including unclean `null` rows, must match exactly;
5. at least `11/12` trials are clean in both conditions;
6. each action is selected cleanly at least three times in intact; otherwise
   return `UNDEFINED_ACTION_COVERAGE` and do not extend or repeat the run;
7. the exact-zero first baseline and every later locked washout rule pass;
8. every complete weight array remains byte-identical to its own initial
   array in both conditions;
9. the lesion's maximum absolute policy eligibility is at most `1%` of the
   intact mean selected eligibility, separately for D1 and D2.
10. the coactivity-input gate follows the fixed symmetric schedule, every
    pre-action coactivity-trace bound passes, and a unit regression proves gain
    `0` decays without adding spikes while the default `1` path is unchanged.

Run one additional `clip_path_control` construction with coactivity disabled,
reward learning rate `0`, reward baseline `0`, and diagnostic current reward
signal `1`. Take one neutral simulation step. This scalar exists only to enter
the bridge's update-and-clip branch; the control is not a teaching or behavior
condition. It must start within the declared Hebbian bounds and retain a
byte-identical complete weight array. Any weight movement in any condition is
`UNDEFINED_WEIGHT_MOVEMENT`; it cannot count as learning or engagement failure.

## GO criteria

Evaluate each criterion using only clean trials, separately for D1 and D2,
separately within trials that selected action `0` and action `1`, and
separately at decision time and pre-outcome. All eight
route-class/action/snapshot groups must satisfy all of:

1. every trial's net selected eligibility is greater than zero;
2. median selected-to-loser ratio is at least `4.0`;
3. mean loser eligibility is at most `25%` of mean selected eligibility;
4. selected eligibility exceeds loser eligibility on at least `80%` of the
   group's trials;
5. mean winner-minus-loser margin is positive.

`ENGAGEMENT_GO` requires every validity prerequisite and every GO criterion.
If validity holds but any selectivity criterion fails, return
`ENGAGEMENT_FAIL` for this locked seed and configuration. One reserved-seed
smoke cannot establish cross-seed mechanism retirement. Do not tune timing,
currents, threshold, scale, trace decay, policy weights, or selector parameters
after seeing the result, and do not open another seed without a new committed
preregistration justified by this smoke.

No policy-training condition may run from this preregistration. A GO permits a
new, separately committed policy-learning preregistration with contingent,
yoked, acquisition-lesion, dopamine-path-lesion, and exact weight-restoration
controls. A failure requires a new research gate for a biologically justified
eligibility mechanism rather than host attribution.

## Evidence read before filing

- `2026-08-03-neural-vocal-credit-gateB-v10-corticostriatal-policy-RESEARCH-GATE.md`
- `2026-08-03-neural-vocal-credit-gateB-v9-output-v2-UNDEFINED-no-trained-action.md`
- `2026-06-19-fsg-watermaze-trial-structured-derisk.md`
- Kandel, *Principles of Neural Science*, 6e, chapter 38, pp. 943-946.
- Reynolds, Hyland and Wickens (2001),
  [Nature](https://www.nature.com/articles/35092560).
- Shen et al. (2008),
  [Science/PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC2833421/).
- Yagishita et al. (2014),
  [Science/PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC4225776/).
- Shindou et al. (2019),
  [OIST repository](https://oist.repo.nii.ac.jp/record/1004/files/Shindou-2019-A%20silent%20eligibility%20trace%20enable.pdf).
