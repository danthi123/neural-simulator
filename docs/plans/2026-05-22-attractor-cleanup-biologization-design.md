# Biologizing the phase-coded composition layer, step 3: attractor clean-up -- design

## Context

The phase-coded composition layer (a Fourier Holographic Reduced
Representation of the project's compositional task) is being biologized
one engineered shortcut at a time. Step 1 replaced the function-first
integrator neurons with resonate-and-fire neurons and passed. This
document designs step 3: replace the clean-up.

## The shortcut

The clean-up takes a recovered (noisy) phasor and identifies which
vocabulary item it is. As built, it does this by computing the
similarity of the recovered vector to every item in an explicitly
enumerated vocabulary list and taking the argmax. A brain keeps no such
enumerated list and runs no such search. The vocabulary, biologically,
is stored in recurrent synaptic weights, and identifying an item is a
settling of recurrent dynamics, not a table lookup.

## The biological replacement: a complex-valued attractor network

Frady and Sommer (PNAS 2019) give exactly the replacement: the
Threshold Phasor Associative Memory (TPAM), a complex-valued
Hopfield-style attractor network whose stable fixed points are the
stored phasor patterns. It is built from the same resonate-and-fire
neurons as step 1.

- The vocabulary is stored in a recurrent weight matrix, the outer
  product of the stored patterns: W = S S* (S is the matrix whose
  columns are the stored phasor patterns; here normalised by the
  dimension so a clean match drives each unit to about unit magnitude).
- A settling step is the recurrent synaptic integration u = W z
  followed by the resonate-and-fire threshold transfer: each neuron
  whose drive magnitude exceeds the threshold spikes at the phase of
  its drive (the resonate-and-fire readout from step 1, which is
  magnitude-invariant); each neuron whose drive is below threshold
  stays silent.
- Iterating the settling step drives the state to a fixed point. Frady
  and Sommer give the Lyapunov energy function that guarantees
  convergence: E(z) = -1/2 sum W_ij z_i z_j* + Theta * ||z||_1.

The clean-up is then: initialise the network with the recovered noisy
phasor, settle, and read out which stored pattern the fixed point is.
The denoising is done by the attractor dynamics; the vocabulary lives
in the recurrent weights, not in a list.

## Abstention as a basin-of-attraction property

The no-confabulation abstention moat is biologized at the same time. A
groundable query's recovered vector lies in the basin of one stored
attractor, so the network settles onto a full stored pattern (almost
every neuron above threshold and spiking). An ungroundable query's
recovered vector lies in no attractor's basin: the recurrent drive
never exceeds threshold for enough neurons, so the state collapses to
silence. Abstention is therefore the network failing to reach any
stored attractor -- a structural property of the dynamics, not a
hand-set similarity cutoff. The abstention signal is the fraction of
neurons still active after settling.

## The build

Extend `research/runners/resonate_fire_fhrr.py` with a
`ResonateFireTPAM` clean-up: build the recurrent weight matrix from the
vocabulary phasors; settle a recovered phasor through repeated
recurrent integration plus the resonate-and-fire threshold transfer
(reusing the step-1 resonate-and-fire readout); read out the attractor
reached and the fraction of neurons still active.

Reuse-by-import only; no protected, frozen, or moat module touched; no
automatic differentiation -- the settling is recurrent attractor
dynamics, a time-stepped iteration, not gradients. The threshold is a
structural parameter of the attractor network, set in advance from the
drive-magnitude analysis (a clean match drives a unit to about unit
magnitude; an ungroundable input drives it well below); it is not the
0.80 compositional bar and the bar is not tuned.

## Pre-registered reading (fixed before the run, never tuned)

Re-run the project's compositional task self-test with the attractor
network as the clean-up, against the frozen 0.80 compositional bar.

- PASS: the attractor clean-up clears the 0.80 bar at loads {2,3,5}
  AND the abstention separation holds (every groundable query settles
  to a stored attractor with a high active fraction; every ungroundable
  query collapses to a low active fraction; the two ranges separate).
  Shortcut 3 is biologized -- the clean-up is now an attractor settling,
  and the no-confabulation moat is a basin-of-attraction property.
- NEGATIVE: it does not clear the bar, or the abstention separation
  breaks. The honest finding is which property of the attractor
  dynamics breaks the capability (spurious attractors; the recovered
  vector falling outside every basin at high load; the collapse not
  being clean). That routes to a mitigation question, not to abandoning
  the arc.

Either outcome is propagated honestly to both git remotes.

## Scope after this step

After step 3 the composition layer runs on resonate-and-fire neurons
(step 1) with an attractor clean-up (step 3). Shortcut 2 -- the symbol
still assigned by oracle lookup -- remains. Its naive form (deriving the
symbol from raw substrate activity) was a decisive negative because raw
activity is too noisy. Its deeper form is to ground the symbol in an
attractor-stabilised representation: an attractor network both grounds
a representation in learned recurrent weights and denoises it, which is
exactly what the activity-level negative said was missing. So step 3's
attractor machinery is also the substrate for step 2's deeper form.

## References

- Frady and Sommer, "Robust computation with rhythmic spike patterns",
  PNAS 116(36):18050-18059, 2019 -- the Threshold Phasor Associative
  Memory.
- Izhikevich, "Resonate-and-fire neurons", Neural Networks, 2001.
