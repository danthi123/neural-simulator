---
type: preregistration
status: live
date: 2026-08-03
mechanism: neural-vocal-action-credit-v6-persistent-expectation
runner: research/runners/_vocal_action_credit_gate_v6_persistent_expectation.py
---

# Gate B v6: learned expectation must precede and overlap reward

**Filed before the v6 runner exists.** This is a reserved-seed implementation
smoke, not formal calibration. Seed `0` is the only executable seed. No v6
formal seed is assigned, and every formal entry point must remain sealed.

## Why v5 stopped

Gate B v5 learned an action-local expectation on NumPy and CuPy with no
unintended weight changes. It still failed because expectation neurons emitted
zero spikes before outcome. They began firing only after the generic outcome
input arrived. Fast GABA-A then reduced the reward response by only `5.56%` on
NumPy and `8.86%` on CuPy, below the fixed `20%` minimum, while omission
recruited neither LHb-like nor RMTg-like neurons.

The repaired project search recovered the already established substrate rule:

- `2026-06-08-spiking-snc-stageB-critic-derisk.md` found learned value but weak,
  sign-sensitive direct GABA-A subtraction at dopamine neurons.
- `2026-06-08-gabab-girk-stageB-derisk-GO.md` showed that the implemented
  GABA-B/GIRK route can produce prediction-dependent dopamine suppression when
  expectation activity precedes and overlaps reward.
- `2026-06-09-N9-SNc-rV-subtraction-research.md` showed why timing, conductance
  reset, engagement, and a graded operating point matter. More GABA-B is not a
  valid fix for a late or clamped prediction signal.

The biological constraint matches that record. VTA GABA neurons show sustained
delay activity proportional to expected reward (Cohen et al. 2012,
[PMC3271183](https://pmc.ncbi.nlm.nih.gov/articles/PMC3271183/)), and their
activity subtracts an approximately constant amount from dopamine reward
responses (Eshel et al. 2015,
[PMC4567485](https://pmc.ncbi.nlm.nih.gov/articles/PMC4567485/)). Schultz's
omission result likewise requires an internal expectation at the expected time,
not a response that starts after outcome (`Schultz-1998-JNeurophysiol-`
`PredictiveReward.txt:375-445`).

## Mechanism

V6 keeps the successful v5 action trace and local trace-to-expectation
plasticity. It changes the output timing and receptor:

1. Fixed neural commit-plus-arousal routes load an action-selective excitatory
   trace. The host does not copy or retain the winner.
2. Reward trains only the local trace-to-expectation route. After learning, the
   trace alone must make the responsible expectation population spike during
   the fixed pre-outcome delay.
3. The expectation population projects to SNc-like dopamine neurons through
   the existing GABA-B/GIRK pathway. Its slow conductance must build before
   reward and overlap the reward response.
4. The same pre-outcome expectation suppresses the tonic omission gate. A
   generic outcome without sensory reward must therefore disinhibit LHb-like
   neurons, excite GABAergic RMTg-like neurons, and lower dopamine.
5. Trial reset clears residual GIRK conductance and relevant membrane state
   before the next measured trial. This is experimental isolation, not a
   cognitive calculation.

Only cue-to-actor and trace-to-expectation synapses may change. No host value,
winner label, prediction error, dopamine assignment, eligibility assignment,
or weight update is permitted.

## Bounded implementation calibration

Only seed `0` may tune implementation engagement. The search is bounded to the
smallest mechanism-relevant levers:

- trace-to-expectation learning gain or bound, solely to make learned
  expectation fire during the delay;
- expectation-to-SNc GABA-B pathway weight or propagation strength, solely to
  place suppression in a graded, non-clamped range; and
- expectation-to-omission-gate strength, solely to recruit the declared
  LHb-to-RMTg path.

For each lever, use a monotonic ladder of at most four values and stop at the
first bounded point that satisfies its local engagement check. Do not tune
action-selection weights, reward drive, trial timing, success thresholds, or
formal seeds. More than two failed levers against one defect triggers another
record and external-source check before a new mechanism is proposed.

## Fixed dynamics smoke

With one locked configuration, run separately initialized intact,
expectation-learning-lesion, expectation-output-lesion, and
omission-path-lesion conditions. Each condition uses 12 fixed contingent
training trials followed by frozen rewarded and omitted probes. Run the exact
configuration on NumPy and CuPy.

The implementation passes only if every check passes on both backends:

1. At least 90% of action epochs contain one clean neural action.
2. The rewarded trace-to-expectation route grows by at least 25% of its initial
   mean and separates from the other route by at least 20% of that mean.
3. During late rewarded-action delays, the learned expectation emits spikes
   before outcome and exceeds the other channel by at least `3:1`.
   Expectation-learning lesion removes both the route change and pre-outcome
   expectation firing.
4. SNc-targeted GIRK conductance is above zero immediately before reward in the
   intact late trials and zero after the explicit reset. Output lesion removes
   that pre-reward conductance effect.
5. Repeated expected reward reduces the late dopamine burst by at least 20%
   relative to early reward. The late response remains above zero and above 25%
   of the early response, rejecting an all-or-none clamp. Output lesion restores
   at least half of the reduction.
6. Frozen expected omission produces LHb-like and RMTg-like spikes and a
   dopamine dip. Omission-path lesion removes RMTg-like firing and at least 80%
   of the dip.
7. No synapse outside cue-to-actor and trace-to-expectation routes changes.
   Every named learning and transmission gate owns exactly its declared route.
8. Reward and outcome afferents remain symmetric across action channels, and
   fixed channel permutation moves trace, expectation, and output together.

A same-initialization GABA-A output control is diagnostic and must remain below
the intact GABA-B suppression if the receptor mechanism is credited. Backend
agreement is compatibility evidence, not independent replication.

## Stop and successor rule

Any failed check is `DYNAMICS_FAIL` or `DYNAMICS_PARTIAL`, never a pass. Do not
weaken thresholds, accept a dopamine dip without LHb/RMTg firing, or count
outcome-evoked expectation as a prediction. If the bounded ladder cannot make
expectation fire before reward, retire the trace-to-expectation cell/learning
mechanism. If expectation fires but graded GIRK suppression fails, consult the
banked saturating/cooperative GABA-B design before changing the simulator.

The v5 calibration, development, and held-out seeds remain unexecuted and
retired with v5. Only after v6 passes this smoke and an independent structural
audit may a new collision-checked v6 manifest and formal contingent/yoked
preregistration be committed.
