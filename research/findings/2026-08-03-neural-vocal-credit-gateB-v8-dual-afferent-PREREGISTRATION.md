---
type: preregistration
status: retired
date: 2026-08-03
mechanism: neural-vocal-action-credit-v8-dual-afferent-upstate
runner: research/runners/_vocal_action_credit_gate_v8_dual_afferent.py
---

# Gate B v8: subthreshold state plus learned action context

> **Retired:** the complete ladder is NO-GO. See
> [the v8 finding](2026-08-03-neural-vocal-credit-gateB-v8-dual-afferent-NO-GO.md).

**Filed before the v8 runner exists.** This is a reserved-seed implementation
smoke. Seed `0` is the only executable seed. No v8 scientific manifest exists,
and all formal execution remains sealed.

## Why this is a different mechanism

V6 exhausted the allowed sparse-route weight ladder. V7 then exhausted the
allowed `24/64/128/200` population-size ladder for one plastic afferent. In both
cases the action trace fired, but the MSN-D1 expectation cells did not fire
before reward. Neither ladder may be reopened.

The repaired project search recovered the June N9 dual-afferent result. A dense
fixed afferent made the same MSN-D1 model fire, allowing a separate dopamine-
gated afferent to learn from `0.20` to about `5.0` across three CuPy seeds. That
experiment still failed its place-grading objective because the fixed afferent
was suprathreshold and position-blind. V8 tests only the transferable cellular
hypothesis: a **subthreshold** fixed input may place the MSN near its up-state so
that a separate learned action input can cross threshold. It does not copy the
failed position-blind critic or claim that the N9 mechanism passed as a whole.

Primary evidence is consistent with this split. Striatal projection neurons
require coordinated cortical and thalamic excitation to leave the down-state,
and clustered glutamatergic input can trigger NMDA-dependent dendritic plateau
potentials. Reward-related and associative cortical inputs converge in
striatum, while dopamine gates corticostriatal plasticity.

## Fixed architecture

1. Retain v7's 200-cell action-specific plastic trace populations, 50% route
   density, MSN-D1 expectation cells, local reward modulation, action selector,
   symmetric outcome drive, and all reset and host-boundary controls.
2. Add one distinct 200-cell fixed-state trace population per action. The same
   neural commit-plus-arousal coincidence that loads the plastic trace loads
   this population; Python does not choose a channel or extend its duration.
3. Project each fixed-state population to its matching expectation population
   at 80% density with no plasticity. Project the original trace through the
   existing local plastic route. These are distinct source populations because
   the simulator cannot represent two pathways with the same source and target.
4. Keep the fixed arm subthreshold before learning. It may depolarize the
   expectation cells, but it must not produce a pre-outcome expectation spike
   by itself or when the plastic route remains at its initial weight.
5. Keep expectation output on the existing GABA-B/GIRK route, but do not assess
   reward suppression or omission until the engagement prerequisite passes.

No neuron type, plastic learning rate, trial timing, action-selection setting,
plastic-route density or initial weight, outcome drive, dopamine threshold, or
pass threshold may change during the ladder.

## Bounded fixed-arm ladder

Use seed `0`, 12 contingent trials, CuPy, 200 cells per fixed and plastic trace,
and fixed-route density `0.80`. Test fixed-route mean weights `2`, `4`, `8`,
`12`, and `16` in order. This brackets v7's silent 200-cell plastic route and
the lower edge of the prior N9 up-state sweep. Stop at the first point that
satisfies every engagement check. Do not interpolate, extend the ladder, or
repeat a failed point with a changed seed.

Run separately initialized intact, expectation-learning-lesion, and fixed-arm-
lesion conditions with matched seed and topology. Lesions disable updates or
transmission after construction; they do not remove populations or redraw the
remaining network.

Every passing point must satisfy all of the following:

1. both trace populations fire before outcome and at least 90% of trials have
   one clean neural action;
2. before reward learning, the executed channel has zero delay expectation
   spikes in intact and learning-lesion conditions;
3. after learning, the rewarded plastic route grows at least 25% from its
   initial mean and separates from the other route;
4. the responsible expectation population fires before outcome on late
   rewarded trials and produces at least three times the other channel's delay
   spikes;
5. the learning lesion blocks route growth and removes at least 80% of intact
   pre-outcome expectation firing;
6. the fixed-arm lesion removes at least 80% of intact pre-outcome expectation
   firing, proving the learned route did not merely repeat v7 at a later point;
7. no synapse outside the declared actor and plastic expectation routes changes;
   and
8. formal phases remain sealed and all runtime preconditions are recorded in
   the result artifact.

A failed runtime prerequisite yields `UNDEFINED`, never `ENGAGEMENT_FAIL`.
Attribution is reported only when the intact arm has a nonzero effect; two
silent arms are a null, not a percentage attribution.

## Successor smoke and stop rules

If one fixed weight passes, lock the first passing value. Under a filed
amendment, run the inherited full-dynamics conditions on NumPy and CuPy and
require graded expected-reward suppression, causal GABA-B output, causal
LHb/RMTg omission, channel permutation, and an independent structural audit
before assigning formal seeds.

If no fixed weight passes, retire the dual-afferent bootstrap mechanism. Do not
change MSN type, add direct expectation current, add host winner memory, turn
the fixed arm suprathreshold, or extend the weight ladder. A negative retires
this method, not executed-action credit as a project capability.

## Sources read before filing

- `research/findings/2026-06-08-striatal-value-critic-firing-research.md`
- `research/findings/2026-06-09-N9-faithful-value-cell-design.md`
- `research/findings/2026-06-09-N9-convergent-upstate-derisk.md`
- `research/runners/n9_convergent_upstate_derisk.py`
- Local catalog B.02 and Kandel 6e Chapter 38.
- Plotkin et al. (2011), synaptically driven dendritic state transitions in
  striatal spiny neurons,
  [PNAS/PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC3235762/).
- Haber et al. (2006), convergence of reward-related and associative cortical
  input in primate striatum,
  [Journal of Neuroscience/PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC6673798/).
- Tseng et al. (2007), dopamine modulation of cortically driven striatal
  plateaus,
  [Psychopharmacology/PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC2213368/).
