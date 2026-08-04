---
type: preregistration
status: superseded
date: 2026-08-03
mechanism: neural-vocal-action-credit-v9-gabab-output
runner: research/runners/_vocal_action_credit_gate_v9_graded_dendritic.py
---

# Gate B v9 Phase 2: neural expectation must change reward and omission

> **Superseded after an undefined run:** all fixed probes selected action `1`,
> while only action `0` had learned reward expectation. No output claim was
> evaluated. See the [replacement preregistration](2026-08-03-neural-vocal-credit-gateB-v9-output-v2-PREREGISTRATION.md).

**Filed after engagement passed and before any v9 output code or run.** Center
`2` is locked. Seed `0` remains the only executable seed, no scientific
manifest exists, and every formal phase remains sealed.

## Fixed protocol

1. Retain the passing v9 architecture, parameters, timing, topology, local
   plasticity, 12 contingent training trials, and all engagement checks.
2. Hold the expectation output gate at zero during training in every condition.
   This prevents prediction output from changing its own dopamine teaching
   signal while the expectation route is acquired.
3. After training, disable actor and expectation plasticity, reset the measured
   neural state, and run fixed probes. No probe may update a weight.
4. In `output_intact`, set the existing expectation output gate to one. The
   expectation-to-SNc route remains GABA-B/GIRK and the expectation-to-omission-
   gate route remains GABA-B.
5. In `expectation_output_lesion`, leave only that output gate at zero. In
   `expectation_learning_lesion`, disable expectation-route learning during
   training, then open the same output gate for probes.
6. Use matched independently initialized conditions. Do not calculate dopamine,
   omission, or a probe response from host-read expectation or plateau state.

## Fixed probes and checks

Run one expected-reward probe and, after a complete reset, one expected-omission
probe in each condition. A pass requires all of the following:

1. the trained intact route still clears the complete v9 engagement checks;
2. expectation fires before outcome in intact reward and omission probes, while
   the learning lesion removes at least 80% of that firing;
3. pre-reward expectation creates nonzero GABA-B/GIRK conductance in SNc in
   `output_intact`, and the output lesion removes it;
4. the intact dopamine reward burst is at least 20% smaller than both the output
   lesion and learning lesion, and intact SNc outcome spikes do not increase;
5. expected omission recruits both LHb-like and RMTg-like populations and
   creates a dopamine dip in intact;
6. the output lesion and learning lesion each remove at least 80% of intact
   LHb and RMTg omission spikes and at least 50% of the dopamine dip;
7. all conditions retain clean neural actions and action-specific expectation;
8. no weight changes during either probe, plasticity remains confined during
   training, host-boundary controls remain false, and formal phases remain
   sealed.

A missing runtime prerequisite is `UNDEFINED`, not a failed mechanism. If the
CuPy reserved smoke passes, run the identical seed-0 protocol on NumPy and
require the same pass/fail decisions, each spike count within 10% or two spikes
when counts are small, and dopamine metrics within 10%. Then perform an
independent evidence audit before assigning formal seeds.

If CuPy output fails, record the boundary and retire this output realization.
Do not reopen the center ladder, alter GABA-B strength or time constants, add a
host prediction-error calculation, or inspect formal seeds.
