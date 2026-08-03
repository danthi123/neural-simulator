---
type: preregistration
status: live
date: 2026-08-03
mechanism: neural-vocal-action-credit-v7-dense-convergence
runner: research/runners/_vocal_action_credit_gate_v7_dense_convergence.py
---

# Gate B v7: dense convergent neural action context

**Filed before the v7 runner exists.** This is a reserved-seed implementation
smoke. Seed `0` is the only executable seed. No v7 scientific manifest exists,
and all formal execution remains sealed.

## Why this successor exists

V5 learned the correct action-local route but its 24-cell trace did not make
the MSN-D1 expectation cells fire before outcome. V6 then tested the complete
allowed weight ladder (`0.1`, `1.0`, `2.0`, `4.0`) and still observed zero
delay expectation spikes. More weight is not the next experiment.

The project already diagnosed this cellular regime. Striatal projection
neurons require convergent excitation to leave their KIR-like down state;
dedicated dense input made the same MSN-D1 model fire with noise disabled in
`2026-06-08-striatal-value-critic-firing-research.md`. The established N9
de-risk used 200-cell context populations rather than changing the value cell
to a biologically easier generic neuron. V7 transfers that specific finding to
action-local reward prediction.

## Mechanism

1. The selected commit population and shared arousal population load an
   action-specific excitatory trace through the existing fixed coincidence
   mechanism. The host neither reads nor retains the winner.
2. V7 increases the number of trace afferents that can converge on each MSN-D1
   expectation cell. It does not inject expectation current or change the
   expectation cell type.
3. Generic outcome activity makes expectation cells fire during initial
   learning. Reward-modulated local plasticity changes only the active
   trace-to-expectation route. After learning, convergent trace activity must
   make the responsible expectation population fire before outcome.
4. The learned expectation reaches dopamine neurons through GABA-B/GIRK and
   reaches the omission comparator through the existing neural pathway.
5. Measured-trial reset clears residual GIRK conductance and target membrane
   state. No host value, prediction error, action label, eligibility assignment,
   or weight update is allowed.

## Bounded engagement ladder

Use seed `0` and otherwise one locked configuration. Test trace population
sizes `24`, `64`, `128`, and `200` in that order, stopping at the first size
that satisfies all local engagement checks after the fixed 12 contingent
trials:

- at least one clean neural action on at least 90% of trials;
- the rewarded trace-to-expectation route grows at least 25% and separates
  from the other route;
- the responsible expectation population emits pre-outcome delay spikes in
  late trials at least three times the other channel;
- expectation-learning lesion removes route growth and at least 80% of that
  pre-outcome firing; and
- no synapse outside cue-to-actor and trace-to-expectation routes changes.

Do not tune route weight, learning rate, reward drive, trial timing, dopamine
thresholds, action-selection parameters, or pass thresholds during this
ladder. If no size passes, retire dense convergence rather than extending it.

## Fixed full dynamics smoke

Lock the first locally engaging population size, then run separately
initialized intact, expectation-learning-lesion, expectation-output-lesion,
omission-path-lesion, and same-initialization GABA-A diagnostic conditions on
NumPy and CuPy. The full smoke passes only if:

1. action cleanliness, route learning, pre-outcome expectation, learning
   lesion, and plasticity isolation satisfy the ladder checks;
2. SNc-targeted GIRK conductance is positive immediately before late reward,
   zero after reset, and removed by output lesion;
3. late expected reward suppresses the dopamine burst by at least 20% versus
   early reward while remaining above zero and above 25% of the early burst;
4. output lesion restores at least half the suppression;
5. frozen expected omission recruits LHb-like and RMTg-like spikes and creates
   a dopamine dip, while omission-path lesion removes RMTg firing and at least
   80% of the dip;
6. GABA-A output suppresses less than intact GABA-B output; and
7. generic reward/outcome afferents remain channel-symmetric and a fixed
   channel permutation moves trace, expectation, and output together.

Any failed check is `DYNAMICS_PARTIAL` or `DYNAMICS_FAIL`, never a pass.
Backend agreement is compatibility evidence, not replication. Formal seeds may
be assigned only after a passing smoke and an independent structural audit.

## Stop rule

If dense convergence creates pre-outcome expectation but graded GABA-B output
still fails, consult the banked cooperative/saturating GABA-B design before a
simulator edit. If omission remains silent after expectation engagement, test
only the declared expectation-to-omission strength ladder under a new filed
amendment. Never hide a failed component behind the whole-smoke summary.
