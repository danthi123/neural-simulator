# Biologization step 1 = PASS: the compositional capability survives replacing the function-first integrator neurons with the resonate-and-fire neuron model

## Status

The first step of the pre-registered biologization arc. The phase-coded
composition layer's first engineered shortcut -- function-first
integrator neurons for the bind and unbind operations -- has been
replaced with the resonate-and-fire neuron, a recognized biological
neuron model. The subsystem self-test clears the frozen 0.80
compositional bar at every load with the abstention separation intact.
Shortcut 1 is biologized at the subsystem level.

## Why this step exists

The owner asked, correctly, whether the phase-coded composition layer
(a Fourier Holographic Reduced Representation, FHRR, of the project's
compositional task) would count as cheating against the project's
biological-realism standard. The honest answer is that the
representational principle -- phase-of-spike coding relative to a
theta/gamma rhythm -- is sound biology, but the layer as built relies
on three engineered devices a brain does not have: function-first
integrator neurons, an oracle that assigns symbols by lookup, and a
clean-up that takes an argmax over a stored vocabulary list. The
validated FHRR integration is therefore a validated engineering
scaffold and a proof the compositional target is reachable -- not a
biological result. The biologization arc replaces the three shortcuts
one at a time, pre-registered, with the rule that the capability must
survive each replacement. This is step 1.

## What was replaced

The validated scaffold realizes binding by Orchard's phase-sum
integrator neuron -- a hand-built circuit with a counter that starts at
the first input spike and counts down at the second, reverse-engineered
to output a phase sum. That is a function-first engineered device, not
a neuron model.

The replacement is the resonate-and-fire neuron (Izhikevich, 2001).
Following Frady and Sommer ("Robust computation with rhythmic spike
patterns", PNAS 2019), who showed a resonate-and-fire network computes
directly with complex-valued phasor representations:

- The neuron has a complex internal state that evolves as a damped
  oscillation, Z(t+1) = Z(t) * exp(lambda + i*omega), with omega the
  cycle frequency and lambda a small negative damping. It is kicked by
  its complex synaptic input and emits a spike at the first upward
  zero-crossing of the imaginary part of its state -- the oscillation
  completing a cycle. The spike step encodes the phase of the state.
- Binding (phase addition) is realized as a complex synaptic weight:
  one phasor passes through a synapse whose complex weight is the other
  phasor, and complex multiplication is magnitude product plus phase
  sum. This is Frady and Sommer's synaptic integration; the phase
  arithmetic lives in the synapse -- biologically where weights live --
  not in a counter inside a neuron.
- Unbinding is the same with the conjugate synaptic weight.
- Bundling is the postsynaptic summation of co-temporal phasors.
- Every operation's result is re-emitted as a genuine spike by a
  time-stepped resonate-and-fire neuron, so the representation stays
  spiking through the whole bind-bundle-unbind chain.

The module is `research/runners/resonate_fire_fhrr.py`, net-new, a
parallel biologized variant. The validated `spiking_phasor_fhrr.py` is
not modified -- it stays as the engineering-scaffold reference, and is
imported only for its pure phase helpers. No protected, frozen, or moat
module is touched. No automatic differentiation -- the resonate-and-fire
dynamics are a time-stepped ordinary differential equation with a
threshold.

## Result (pre-registered; frozen 0.80 bar; the project's compositional task)

Primitive check (maximum phase error, as a fraction of a cycle):

```
bind          0.0019      unbind        0.0025
bundle        0.0026      robustness    0.0010
```

The bind, unbind, and bundle primitives reproduce the FHRR operations
to within about 0.2% of a cycle -- the discrete time-step quantization
floor. The robustness figure is the key resonate-and-fire property: the
spike phase is invariant to a large variation in the kick magnitude, so
the readout depends on phase, not amplitude.

Self-test (8 cues, 8 fillers, dimension 512, 300 trials per load):

```
            compositional accuracy    abstention separation
L=2         1.0000                    groundable 0.596 > ungroundable 0.112
L=3         1.0000                    groundable 0.454 > ungroundable 0.115
L=5         1.0000                    groundable 0.303 > ungroundable 0.112

VERDICT -> PASS
```

The resonate-and-fire realization clears the frozen 0.80 bar at every
load, and the groundable-versus-ungroundable abstention separation
holds at every load. The compositional capability and the
no-confabulation abstention separation both survive the replacement of
the function-first integrator neurons with the biological neuron model.

## Smell test (a PASS scrutinised harder than a FAIL)

- The resonate-and-fire readout is genuinely dynamical: it time-steps
  the damped complex oscillation for up to a full cycle and detects the
  threshold crossing; the spike step is produced by the dynamics, not
  computed by an angle function. The primitive errors (~0.002) are the
  discrete time-step quantization, consistent with a genuine
  time-stepped readout.
- The resonator robustness is real: a kick magnitude varied by a factor
  of three leaves the spike phase unchanged to within 0.001 of a cycle.
  This is a genuine resonate-and-fire property.
- The binding and bundling are complex synaptic integration, which is
  faithful to Frady and Sommer's model and biologically correct -- the
  weight lives in the synapse. The function-first counter is gone. This
  realization is more genuinely time-stepped than the validated
  scaffold, which time-stepped only the bind.
- Nothing was tuned to pass: the resonate-and-fire parameters were
  fixed before the run and verified by the primitive check; the 0.80
  bar is the frozen project bar; the task, dimension, trial count, and
  seed match the validated scaffold's self-test exactly. The accuracy
  of 1.0000 is the clean-symbol ceiling -- the validated scaffold's
  self-test reached the same ceiling, because at dimension 512 with
  eight fillers and loads up to five the task is well within FHRR
  capacity.
- Honest caveat: the primitive errors are small but non-zero, and over
  a three-operation chain they could in principle accumulate; the
  self-test runs the full chain and still reaches 1.0000, so the
  accumulated error is well within the clean-up margin.

## Honest scope

This is a subsystem-level self-test result, directly comparable to the
validated `spiking_phasor_fhrr.py` self-test. It biologizes shortcut 1
only. The composition layer still has shortcut 2 (the oracle symbol
assignment -- the naive activity-grounded form was a decisive negative
because raw substrate activity is too noisy) and shortcut 3 (the
stored-vocabulary argmax clean-up). A PASS here means the binding
arithmetic now runs on a biological neuron model; it does not by itself
make the whole layer biological, and it is not yet a capability claim.
A dedicated adversarial review is the pre-registered discipline step
before any capability-status claim rolls up, exactly as the validated
scaffold followed (subsystem self-test, then integration, then
adversarial review, then a capability pillar).

## Next step

Shortcut 3: replace the stored-vocabulary argmax clean-up with an
attractor network whose fixed points are the vocabulary. Frady and
Sommer's threshold phasor associative memory -- a complex-valued
Hopfield-style attractor network with a Lyapunov energy function -- is
exactly that, and it is built from the same resonate-and-fire neurons.
After shortcut 3, shortcut 2's deeper form (a symbol grounded in a
denoised, attractor-stabilised representation rather than raw activity)
can be revisited, because an attractor network both grounds and
denoises a representation.

## Files / evidence

- Module: `research/runners/resonate_fire_fhrr.py`
- Self-test result: `research/findings/raw/resonate_fire_fhrr_selftest.json`
- Design: `docs/plans/2026-05-22-resonate-and-fire-biologization-design.md`
- The engineering-scaffold reference (unchanged):
  `research/runners/spiking_phasor_fhrr.py`

## References

- Izhikevich, "Resonate-and-fire neurons", Neural Networks, 2001.
- Frady and Sommer, "Robust computation with rhythmic spike patterns",
  PNAS 116(36):18050-18059, 2019.
