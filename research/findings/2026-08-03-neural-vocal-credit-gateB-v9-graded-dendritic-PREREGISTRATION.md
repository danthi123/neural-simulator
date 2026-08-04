---
type: preregistration
status: superseded
date: 2026-08-03
mechanism: neural-vocal-action-credit-v9-graded-dendritic-readout
runner: research/runners/_vocal_action_credit_gate_v9_graded_dendritic.py
---

# Gate B v9: learned action context through a graded dendrite

> **Engagement complete:** center `2` is the first passing point and is locked.
> See the [qualified finding](2026-08-03-neural-vocal-credit-gateB-v9-graded-dendritic-engagement-GO.md)
> and the [Phase-2 preregistration](2026-08-03-neural-vocal-credit-gateB-v9-output-PREREGISTRATION.md).

**Filed before the v9 runner exists.** This is a reserved-seed engagement
smoke. Seed `0` is the only executable seed. No scientific manifest exists and
all formal execution remains sealed.

## Fixed architecture

1. Inherit v7's 200-cell action-specific cortical traces, 50% plastic
   trace-to-expectation routes, 24-cell MSN-D1 expectation populations, neural
   commit-plus-arousal trace loading, local reward modulation, selector,
   resets, and host-boundary controls.
2. Remove v8's fixed-state populations and routes entirely.
3. Mark only each plastic trace-to-matching-expectation route as input to the
   existing graded dendritic plateau. Keep ordinary synaptic transmission and
   local plasticity on the same synapses.
4. Retain the upstream trace-loading plateau. Both route classes share the
   bridge's existing transfer parameters, but the strong commit/arousal loader
   is expected to remain saturated while the weaker learned route traverses
   the graded middle.
5. Hold plateau slope `0.33`, strength `2.0`, rise `2 ms`, and decay `500 ms`.
   Change no learning rate, neuron type, route density or initial weight,
   action setting, outcome drive, dopamine threshold, or pass threshold.
6. Hold the expectation-to-dopamine output gate at zero in every engagement
   condition. Do not test GABA-B/GIRK, omission, or reward suppression yet.

No host process may select an action channel, inject expectation current,
extend a route after observing a winner, copy plateau conductance into dopamine,
or update a synaptic weight.

## Bounded center ladder

Use seed `0`, 12 contingent trials, CuPy, and plateau centers `16`, `8`, `4`,
and `2` in that order. The descending order tests the least excitable transfer
first. Center `8` is the shipped Stage-1 default; the bounded neighbors bracket
the expected weighted drive of the learned trace route. Stop at the first point
that satisfies every check. Do not interpolate, extend, repeat with another
seed, or change slope or strength.

Run separately initialized matched conditions:

- `intact`;
- `expectation_learning_lesion`, with only expectation-route plasticity off;
- `expectation_plateau_lesion`, with only the trace-to-expectation dendritic
  mask off after construction. Its synapses, weights, ordinary transmission,
  and upstream trace loaders remain present.

## Required engagement checks

Every passing point must satisfy all of the following:

1. at least 90% of trials have one clean neural action in every condition;
2. the upstream action trace fires before outcome in every condition;
3. before reward learning, intact and learning-lesion expectation populations
   emit zero delay spikes;
4. the rewarded plastic route grows at least 25% from its initial mean and
   separates from the other route by at least 20% of the initial mean;
5. on the last four rewarded trials, the responsible expectation population
   fires before outcome and emits at least three times the other channel's
   delay spikes;
6. the learning lesion blocks route growth and removes at least 80% of intact
   pre-outcome expectation firing;
7. the expectation-plateau lesion removes at least 80% of intact pre-outcome
   expectation firing while the upstream action trace still engages;
8. plateau conductance appears in the responsible expectation population in
   intact, and the lesion removes that conductance only from expectation cells;
9. no synapse outside the declared actor and expectation routes changes;
10. expectation output remains closed, formal phases remain sealed, and all
    runtime prerequisites are recorded.

A failed runtime prerequisite yields `UNDEFINED`, not `ENGAGEMENT_FAIL`.
Attribution is reported only when intact expectation is nonzero.

## Stop rules

If one center passes, lock the first passing center. File an amendment before
opening the existing GABA-B/GIRK output and testing graded reward suppression,
omission, plateau-output causality, channel permutation, NumPy/CuPy agreement,
or formal seeds.

If no center passes, retire this learned-route graded-dendritic mechanism. Do
not tune the v6-v8 routes, add host analog output, weaken the MSN threshold, or
change plateau slope/strength after seeing the ladder.

## Evidence read before filing

- `2026-08-03-neural-vocal-credit-gateB-v9-graded-dendritic-RESEARCH-GATE.md`
- `2026-08-03-neural-vocal-credit-gateB-v4-smoke-NO-GO.md`
- `2026-08-03-neural-vocal-credit-gateB-v5-learning-smoke-NO-GO.md`
- `2026-08-03-neural-vocal-credit-gateB-v6-engagement-NO-GO.md`
- `2026-08-03-neural-vocal-credit-gateB-v7-dense-convergence-NO-GO.md`
- `2026-08-03-neural-vocal-credit-gateB-v8-dual-afferent-NO-GO.md`
- `2026-06-20-dendrite-derisk-A-graded-plateau-readout.md`
- `2026-06-20-dendrite-stage1-onbridge-graded-plateau.md`
- Plotkin et al. (2011), [PNAS/PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC3235762/).
- Du et al. (2017), [PNAS/PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC5594658/).
