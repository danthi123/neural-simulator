# Phase-coded vector-symbolic composition: design

> **For Claude / autonomous continuation:** Pre-registered design for
> the next major arc, after the dynamics-gating fix class was
> exhausted and the staged-recurrence variant converged with the SPEAR
> negative. Acts on owner-directed external research (biology + open
> source). Continue straight through: design -> cheap-first probe ->
> propagate; no hand-back.

## Why this arc exists

The compositional investigation has exhausted the entire "fix the
network dynamics" class: 8 architectures plus the difference-readout
probe plus the ca1->concept-pool wire plus the ACh-staged-recurrence
variant. Every one operates on network dynamics / wiring / gain;
every one hit the same wall. The SPEAR arc's own conclusion, reached
independently, was that gating dynamics is insufficient -- the
compositional readout needs a STRUCTURED DECODABLE object, not the
firing rate of a sum of partially-active sub-populations. SPEAR's
own pre-registered next step, never built, was phase-coded
vector-symbolic composition.

Owner-directed external research (biology + open-source projects)
confirms the direction and supplies a concrete mechanism:

- **Biology**: theta-gamma phase coding is observed and
  well-characterized (Lisman & Jensen, "The theta-gamma neural
  code") -- the timing of a spike within an oscillation cycle carries
  information, and different ensembles occupy different gamma cycles
  within a theta cycle.
- **Open source**: Orchard & Jarvis 2023 ("Hyperdimensional Computing
  with Spiking-Phasor Neurons", ICONS) implement a full vector-
  symbolic architecture -- Fourier Holographic Reduced Representation
  (FHRR) -- in spiking neurons, where a vector dimension is one
  spiking-phasor neuron and its value is the PHASE of its spike
  within a global cycle. Torchhd (Heddes et al., JMLR 2023) is the
  reference open-source FHRR algebra library.

## The mechanism (FHRR, from Orchard & Jarvis 2023)

A symbol is an N-dimensional vector of unit-modulus complex numbers;
element k is e^(i*phi_k). The composition operators:

- **Binding** (x): elementwise complex multiply = phase ADDITION.
  bound = noun (x) adjective.
- **Unbinding** (/): multiply by conjugate = phase SUBTRACTION.
  bound / noun ~= adjective.
- **Bundling** (+): vector addition, modulus then discarded (keep
  phase). Stores many facts in one vector. LOSSY -- the loss grows
  with the number of facts bundled (the paper's honest caveat).
- **Similarity**: complex inner product. Random vectors ~ 0.
- **Clean-up**: restore a noisy result to the nearest vocabulary
  vector (an attractor / associative memory; in the spiking version,
  two resonate-and-fire populations with mutual inhibition ->
  winner-take-all).

Crucially this is a STRUCTURED representation: a composed fact
(noun (x) adjective) is a precise algebraic object that the unbind
operator decomposes exactly, and clean-up denoises. It is NOT a
firing-rate sum -- it is exactly the "structured decodable object"
SPEAR identified as missing.

The spiking-phasor realization (Orchard & Jarvis): each dimension is
a spiking-phasor neuron firing once per global cycle; binding is a
"phase-sum" neuron (two integrators), unbinding a "phase-subtraction"
neuron, bundling a "phase-midpoint" neuron, clean-up the two RF
populations. Their state-transition model (705 neurons) and spatial
memory (3406 neurons) both worked at >95% confidence. No automatic
differentiation anywhere -- the operators are integrator dynamics.

## Honest scoping: biology-inspired engineering, and what this reuses

This must be stated plainly. FHRR is a biology-INSPIRED engineering
framework. Theta-gamma phase coding is real biology; the FHRR
algebra is a designed mathematical framework; the spiking-phasor
neuron models (phase-sum integrator etc.) are function-first
engineered devices, not derived from a biological neuron model. Under
the owner's reframed goal (artificial life with a proper brain
analogue; biology-translatable insights are the deliverable), a full
spiking-phasor FHRR composition layer is a NEW representational
substrate that does not reuse the v14/v16 concept pools' rate-coded
representation -- it is a parallel mechanism, not a variant of the
validated substrate.

This is a genuine paradigm consideration. It is surfaced honestly,
not buried. The arc is therefore staged so the cheap, decisive,
engineering-only question is answered FIRST, before any paradigm
commitment:

## Cheap-first falsification probe (pre-registered; engineering-only, ceiling-clarification)

Before designing or building any spiking-phasor network, a numpy
FHRR reference probe answers the decisive question: is the FHRR
ALGEBRA even capable of the project's compositional task at the
project's required load and vocabulary, against the project's frozen
bar? This is explicitly an ENGINEERING ceiling-clarification test
(the owner's standing rule permits clearly-marked engineering
baselines for ceiling clarification; insights from it are about
engineering, not biology, and it is non-load-bearing). It is ~30
lines of numpy, minutes to run.

Protocol (fixed before the run):
- Vocabulary: the project's 4 nouns {apple, river, dog, cat} + 4
  adjectives {big, small, hot, cold}. Each symbol = a random
  N-dimensional unit-modulus complex vector.
- For load L in the project's frozen ladder {2, 3, 5}: pick L
  (noun, adjective) facts; encode each as noun (x) adjective; BUNDLE
  the L bound facts into one FHRR vector.
- Query: for each of the L facts, compute bundle / noun, clean up
  against the adjective vocabulary by similarity, check the recovered
  adjective == the bound one.
- Sweep dimension N in {64, 128, 256, 512, 1024} to find the
  dimension needed.
- Accuracy = fraction of facts correctly recovered; the project's
  frozen compositional bar is 0.80.
- Multi-trial (many random vocab draws per (L, N) cell) for a stable
  estimate.

Pre-registered decision rule (fixed; never tuned):
- If numpy FHRR clears 0.80 at L = 2, 3, 5 at a tractable dimension
  (N <= 1024): the FHRR algebra is sufficient for the project's
  compositional task. The next arc is the biology-grounded
  spiking-phasor implementation -- designed then, with its own
  pre-registered fixed-bar gate, adversarial review, decisive run.
  The honest framing is locked: the algebra works (engineering); the
  open scientific question is the biology-grounded spiking
  realization and its integration with the validated subsystems
  (the no-confabulation moat especially).
- If numpy FHRR does NOT clear 0.80 at L = 5 within N <= 1024: FHRR
  bundling capacity is insufficient for the project's task at the
  required load -- ruled out cheaply, before any spiking build. The
  honest finding routes to a different structured-representation
  mechanism (resonator networks; sparse block codes; or a capacity
  analysis of what load FHRR does support).

## Honest ceiling (binding)

- The numpy FHRR probe is an engineering reference. A PASS on it does
  NOT achieve compositional capability in the project's substrate --
  it only establishes the algebra is capable, motivating the
  biology-grounded build.
- The biology-grounded spiking-phasor arc, if built and if it passes
  its own gate, would be the first compositional retrieval clearing
  the trustworthy bar -- still not fluent open-ended language, and
  pending the honest question of whether a spiking-phasor layer
  composed with the no-confabulation moat preserves the moat.
- No bar tuned. The numpy probe touches no protected/frozen/moat
  module (standalone numpy). No autograd. Honest propagation both
  remotes.

## Next step

Write `research/findings/raw/fhrr_numpy_probe.py` (standalone numpy
FHRR; the pre-registered protocol above), run it, apply the decision
rule, propagate. Continue straight through.
