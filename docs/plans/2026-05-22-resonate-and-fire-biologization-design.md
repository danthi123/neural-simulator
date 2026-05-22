# Biologizing the phase-coded composition layer, step 1: resonate-and-fire neurons -- design

## Context and the reframe this arc acts on

The project has a validated compositional capability: a two-system
pipeline in which the rate-coded concept-recognition substrate is the
front-end and a phase-coded composition layer is the back-end. The
composition layer is a Fourier Holographic Reduced Representation
(FHRR): each concept is a high-dimensional vector of phases, binding is
phase addition, bundling is the phase of a complex sum, and clean-up is
a nearest-match over a vocabulary.

Under the project's biological-realism standard, that composition layer
as built is a validated **engineering scaffold** -- a proof the
compositional target is reachable -- not yet a biological result. It
relies on three engineered devices a brain does not have:

1. **Function-first integrator neurons.** The bind and unbind
   operations are realized by hand-built integrator circuits (a counter
   that starts at the first input spike and counts down at the second)
   reverse-engineered to output a phase sum. This is not a biological
   neuron model.
2. **Oracle symbol assignment.** Each concept is given a fixed phase
   vector by lookup. The brain has no external table of codes.
3. **Stored-vocabulary clean-up.** The clean-up takes an argmax over an
   explicitly enumerated vocabulary list. The brain does not keep an
   enumerated answer list.

The biologization arc replaces these one at a time, each its own
pre-registered step, with the standing rule that the compositional
capability and the no-confabulation abstention separation must survive
each replacement -- or the honest finding is which biological
constraint breaks it. A prior step already attacked shortcut 2
(grounding the symbol in the substrate's own activity); the naive form
was a decisive negative because raw single-observation substrate
activity is too noisy. This document designs **step 1: replace the
function-first integrator neurons with resonate-and-fire neurons**, the
best-defined and highest-priority replacement.

## The biological replacement: resonate-and-fire neurons

A resonate-and-fire neuron is a recognized biological neuron model
(Izhikevich, "Resonate-and-fire neurons", 2001). Unlike an
integrate-and-fire neuron, its subthreshold dynamics oscillate: the
membrane state is naturally described by a complex number whose damped
rotation is the subthreshold oscillation. Frady and Sommer ("Robust
computation with rhythmic spike patterns", PNAS 2019) showed that a
network of resonate-and-fire neurons computes directly with
complex-valued (phasor) representations -- the same representation FHRR
uses -- with spike timing carrying the phase.

The model, following Frady and Sommer:

- Each neuron has a complex internal state Z = V + iU. Between inputs
  it evolves as a damped oscillation, dZ/ds = (lambda + i*omega)*Z,
  where omega = 2*pi/T sets the cycle and lambda < 0 is the damping.
- A presynaptic spike arriving through a synapse kicks Z by the
  synapse's complex weight. The synapse weight is itself a complex
  number; its phase translates into a synaptic transmission delay
  (delay = T * synapse_phase / 2*pi).
- The neuron emits a spike when its state crosses threshold (the real
  part exceeds a threshold while the imaginary part is positive). The
  spike time, relative to the global T-periodic cycle, is the phase of
  the neuron's complex state.

This is a genuine, recognized neuron model, not a function-first
device. The phase arithmetic that FHRR needs is not hand-built into a
circuit -- it emerges from the resonant dynamics plus ordinary
synaptic delays.

## How binding, unbinding, and bundling map to resonate-and-fire dynamics

- **Bundling (the phase of a complex sum).** Native. A
  resonate-and-fire neuron that receives several presynaptic spikes
  sums their kicks in its complex state; coherent inputs reinforce the
  oscillation, incoherent inputs partly cancel. The neuron's resulting
  spike phase is the phase of the complex sum of its inputs -- exactly
  the FHRR bundle. No special mechanism is needed; bundling is what
  postsynaptic summation already does.
- **Binding (phase addition).** A spike carrying phase phi_a, passed
  through a synapse whose complex weight has phase phi_b, kicks the
  postsynaptic resonate-and-fire neuron; the neuron resonates and
  spikes at phase phi_a + phi_b. The phase addition is realized by the
  synaptic delay (delay = T * phi_b / 2*pi) plus the resonant dynamics
  -- a biological transmission delay, not a counter circuit.
- **Unbinding (phase subtraction).** The same mechanism with the
  complementary delay (delay = T * (-phi_b) / 2*pi, taken modulo T):
  passing the composite's spike through a synapse delayed by the
  negative of the cue's phase recovers phase phi_composite - phi_cue.

So the engineered integrator neuron is replaced by a resonate-and-fire
neuron driven through a delayed synapse. Both pieces are biological.

## The build

`research/runners/resonate_fire_fhrr.py` -- a net-new module, a
biologized parallel variant of the validated `spiking_phasor_fhrr.py`
composition subsystem.

- The validated `spiking_phasor_fhrr.py` is **not modified**. It stays
  as the engineering-scaffold reference. The resonate-and-fire module
  is a separate file so the comparison between the two is clean.
- Reuse-by-import only; no protected, frozen, or moat module is
  imported or modified. No automatic differentiation -- the
  resonate-and-fire dynamics are a time-stepped ordinary differential
  equation with a threshold, which is neuron dynamics, not gradients.
- The module implements: a time-stepped resonate-and-fire neuron (the
  complex damped-oscillation ordinary differential equation, integrated
  over one to two cycles, with the threshold-crossing spike detector);
  bind, unbind, and bundle realized as resonate-and-fire dynamics as
  described above; the same abstention-thresholded clean-up the
  validated subsystem uses (the clean-up itself is shortcut 3 and is
  biologized in a later step -- this step changes only the neurons).
- The module carries the same pre-registered self-test as the
  validated subsystem: the project's compositional task at loads
  {2, 3, 5}, against the frozen 0.80 compositional bar, plus the
  groundable-versus-ungroundable abstention separation.

## Pre-registered reading (fixed before the run, never tuned)

- **PASS** -- the resonate-and-fire realization clears the frozen 0.80
  bar at loads {2, 3, 5} AND the abstention separation holds. The
  compositional capability survives the replacement of the
  function-first integrator neurons with a recognized biological neuron
  model. Shortcut 1 is biologized. The arc proceeds to shortcut 3 (the
  attractor clean-up) and then shortcut 2's deeper attractor-grounded
  form.
- **NEGATIVE** -- it does not clear the bar, or the abstention
  separation breaks. The honest finding is which property of the
  resonate-and-fire dynamics breaks the capability (damping eroding
  phase precision; threshold-crossing jitter; the synaptic-delay
  realization of phase addition). That is a biology-translatable result
  about what a phase-coded binding system costs in a realistic neuron
  model, and it routes to a mitigation question, not to abandoning the
  arc.

Either outcome is propagated honestly (a findings document, both git
remotes). A negative is a real finding.

## What this step does and does not claim

This step replaces shortcut 1 only. After it, the composition layer
still has the oracle symbol assignment (shortcut 2) and the
stored-vocabulary clean-up (shortcut 3). A PASS here means the binding
arithmetic runs on a biological neuron model -- it does not yet make
the whole layer biological. The honest status of the composition layer
remains "engineering scaffold being biologized, step by step" until all
three shortcuts are replaced and the capability has survived each.

Frady and Sommer's threshold phasor associative memory -- a
complex-valued attractor network whose fixed points are the stored
patterns -- is the biological replacement for shortcut 3, and it is the
designed next step after this one. Shortcut 2's deeper form (a symbol
grounded in a denoised, attractor-stabilised representation) couples
with shortcut 3, because an attractor network both grounds and denoises
a representation.

## References

- Izhikevich, "Resonate-and-fire neurons", Neural Networks, 2001 -- the
  resonate-and-fire neuron model.
- Frady and Sommer, "Robust computation with rhythmic spike patterns",
  PNAS 116(36):18050-18059, 2019 -- complex/phasor computation with
  resonate-and-fire neurons; the threshold phasor associative memory.
- Orchard and Jarvis, "Hyperdimensional Computing with Spiking-Phasor
  Neurons", 2023 -- the engineered spiking-phasor FHRR the validated
  scaffold is built on.
