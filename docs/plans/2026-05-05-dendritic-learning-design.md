# Apical-basal dendritic learning — design doc (CONDITIONAL on
# global-DA-feedback failing)

**Date:** 2026-05-05
**Status:** DESIGN ONLY (not yet implemented)
**Trigger condition:** Fires only if BOTH classical 3-factor AND
graded-DA 3-factor fail at `tf_with_topo_fs` (i.e., global scalar
feedback in any form is insufficient for biology-grounded W→A learning).

## Problem statement

If experiments confirm that scalar global feedback (sign-only DA OR
magnitude-graded DA, both biology-plausible) cannot match supervised
gradient at our W→A task, we need richer per-region feedback. The
gold-standard frameworks in the field:

1. **Apical-basal dendritic learning** (Bono & Clopath 2017,
   Sacramento et al 2018): pyramidal neurons have two compartments
   computing different signals. Basal dendrites integrate bottom-up
   sensory drive. Apical dendrites integrate top-down feedback (predictions
   or error signals from higher regions). Plasticity at basal synapses is
   gated by apical activity.

2. **Predictive coding** (Rao & Ballard 1999, Whittington & Bogacz 2017):
   each region predicts its inputs from internal state; deviations
   propagate as error signals. Mathematically equivalent to backprop
   under certain assumptions.

This doc scopes (1), the cheaper of the two to implement in our
codebase.

## What's needed

### A. Multi-compartment Izhikevich neurons

Currently we have **point neurons** — a single (V, u) state per neuron.
For apical-basal learning we need at minimum:

```
Per neuron, we track:
  V_basal      : membrane potential at basal compartment
  V_apical     : membrane potential at apical compartment
  V_soma       : output potential (drives spikes)
  u            : recovery variable (Izhikevich)
```

The apical compartment receives top-down inputs; basal receives
bottom-up. The soma decision (fire / don't fire) integrates both
plus recovery. The Larkum 2013 BAC firing rule says: somatic spikes
driven by basal alone trigger only at high threshold; basal + apical
coincidence drives easier firing. This creates the credit-assignment
signal.

Implementation cost: 2-3 weeks. Touches `sim/bridge.py` neuron
allocation + `sim/kernels.py` integration kernels. NEW kernel:
`fused_apical_basal_dynamics_update`.

### B. Apical-driven plasticity gate

The classical Bono-Clopath rule: STDP at basal-pyramidal synapses is
gated by apical compartment activity at the post-synaptic neuron.

```
For each basal synapse (pre, post):
  Δw = STDP_kernel(t_pre, t_post) × apical_activity[post]
```

When apical compartment is depolarized (top-down feedback says "you
should fire more"), basal synapses learn fast. When apical is silent,
no plasticity.

Implementation: ~1 week. Adds `cp_apical_activity` array, modifies
`fused_stdp_weight_update` to read it as a gate.

### C. Top-down feedback pathway

For apical to carry useful signal, we need a feedback path from
"output" regions to "input" regions. In our W→A architecture, this
means: motor_X → language_input via a SEPARATE apical-pathway
infrastructure.

Two design options:

**Option C1**: dedicated apical-only pathway. New `RegionPathway`
property `target_compartment="apical"`. The pathway delivers current
to post-synaptic apical compartments rather than basal.

**Option C2**: per-pathway compartment routing. Existing pathways
unchanged; add a new "apical" connection class that's separate from
the standard CSR.

Option C1 is cleaner. ~1 week.

### D. The teaching signal

For apical to drive useful learning, it needs a meaningful signal.
Three candidate sources:

1. **Reward-modulated apical drive** (cheapest): when reward is
   delivered, drive apical compartments of motor_correct pool
   directly. Simulates a top-down "you got it right" signal. Bridges
   to existing reward infrastructure.

2. **Reciprocal connectivity** (more biological): the motor pool's
   own activity feeds back to language_input via a dedicated apical
   channel. Encodes "what motor pool fired given this input."

3. **Hippocampal replay** (most biologically rich): SWR replay drives
   apical signals during quiet periods to consolidate memory. Long-tail
   research direction.

Start with (1). 1 day to wire.

## Total scope estimate

- Multi-compartment Izhikevich: 2-3 weeks
- Apical-driven STDP gate: 1 week
- Apical pathway routing: 1 week
- Reward-modulated apical drive: 1 day
- Validation tests + benchmarks: 1 week

**Total: 1.5-2 months of focused engineering.**

## Why this might fail

The Bono-Clopath rule assumes bottom-up → top-down → bottom-up
information flow with appropriate timing. In our W→A task:

- Bottom-up: language_input drives motor pools (current path)
- Top-down: would need motor → language feedback (new infrastructure)
- The "feedback" carries... what? Action-correctness signals? Reward?

If we naively route reward back to language_input apical, we're
basically reinventing R-STDP with extra steps. The real value of
apical-basal is that the feedback carries PER-REGION error magnitude —
which is what gradient has and global DA doesn't.

So the question becomes: how do we COMPUTE the per-region error
magnitude in a biology-plausible way? Three candidates:

1. **Predictive coding from above**: each region above predicts
   the region below; deviation drives apical signals. This pulls
   us back toward predictive coding architecture.

2. **Lateral comparison**: motor pools compare to each other (WTA
   already gives this); the local "you didn't win" signal becomes
   the apical input. Limited info.

3. **Trial-relative comparison**: motor pools track running
   averages; deviations from average drive apical. Provides
   magnitude info but at slow timescale.

Option 1 is the cleanest theoretical answer but it's predictive
coding under another name. Option 2 is cheap but information-poor.
Option 3 is biologically supported (Tobler 2005 measured neurons
encoding RPE-like signals against running averages) and computationally
tractable.

## Recommended sequencing if pursued

1. **Week 0**: prototype `cp_apical_activity` array allocation +
   `cfg.enable_apical_compartment` flag. No-op if disabled.
2. **Week 1-2**: `fused_apical_basal_dynamics_update` kernel +
   tests. Compare to single-compartment Izhikevich for stability.
3. **Week 3**: apical pathway routing. Add `target_compartment`
   to `RegionPathway`.
4. **Week 4**: reward-modulated apical drive (Option 1, cheapest
   teaching signal). Smoke-test on minimal arch.
5. **Week 5**: integrate with W→A bio_topo_fs config. Run sweep.

If sweep gives >= 4/6 aligned: ship. If still failing: move to
predictive coding architecture (next-tier rewrite, 2-3 months).

## Decision gate

DO NOT START this work until BOTH:
- `bio_three_factor` (classical 3-factor): aligned ≤ 1/6 at
  tf_with_topo_fs (CONFIRMED currently emerging as 1/6)
- `bio_three_factor_graded_da` (magnitude-graded): aligned ≤ 2/6
  at tf_with_topo_fs (PENDING — auto-fires after current sweep)

If graded-DA gives 4-6/6 aligned, the cheap fix solved it and
dendritic learning isn't needed.

If graded-DA gives 0-2/6, scalar feedback is fundamentally
insufficient at this scale, and dendritic learning becomes the
recommended next direction.

## Alternative: predictive coding (not detailed here)

If apical-basal also fails or proves architecturally too disruptive,
predictive coding is the next-tier framework. Scope: 2-3 months.
Different network organization (paired generative + recognition
pathways at every region). Out of scope for this doc; a separate
design doc would be written if/when this becomes the active
direction.
