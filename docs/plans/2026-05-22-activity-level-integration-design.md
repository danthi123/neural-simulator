# Activity-level integration of the composition layer with the concept substrate -- design

## Context

The project now has a validated, multi-seed, adversarially-reviewed
compositional capability: a two-system pipeline in which the project's
rate-coded concept-recognition substrate is the front-end and a
spiking-phasor composition layer is the back-end. The composition layer
is a Fourier Holographic Reduced Representation (FHRR): each concept is
represented by a high-dimensional vector of phases (a "phasor" per
dimension), binding is phase addition, unbinding is phase subtraction,
bundling several bound pairs is the phase of their complex sum, and
clean-up is a nearest-match over a vocabulary of known concept vectors.
The pipeline clears the project's frozen 0.80 compositional bar at
multi-seed mean 0.96-0.99 across loads of 2, 3, and 5 facts.

That validated integration joins the two systems at the **concept-
identity level**. The substrate recognizes an input word and reports a
single discrete label -- the identity of the concept pool that fired
most strongly. A fixed lookup table then maps that discrete label to a
pre-assigned phasor vector. The substrate's actual neural activity --
the graded firing-rate pattern across the concept-pool population --
does not itself flow into the composition layer; only the argmax label
does.

## The gap this arc addresses

The identity-level interface has two honest limitations:

1. **It is a lookup table.** A discrete label indexes a fixed
   dictionary of phasor vectors. This is a hand-built correspondence,
   not a learned or emergent one -- a small homunculus sitting between
   the two systems.

2. **It discards information.** The substrate's population activity is
   a graded, distributed pattern. Collapsing it to a single argmax
   label throws away the partial activations, the relative firing of
   competing pools, and any graded confidence the substrate expresses.

A more biologically faithful interface would let the substrate's
population activity pattern itself **be** the input to the composition
layer -- the phasor symbol **derived from** the activity vector, with
no discrete-label bottleneck and no lookup table. This is
**activity-level integration**.

## The honest new failure mode

Activity-level integration introduces a failure mode that the
identity-level interface does not have.

With the identity-level lookup, the same recognized label always
produces a byte-identical phasor symbol. The cue symbol used when a
fact is stored and the cue symbol used when that fact is queried are
exactly equal, so unbinding is exact.

With activity-level derivation, the phasor symbol is a function of the
substrate's population activity vector -- and that activity vector has
**trial-to-trial variability**. Real cortical populations show
substantial variability in firing rate from one presentation of the
same stimulus to the next. So the activity vector observed when a fact
is stored and the activity vector observed when that fact is queried
differ, even for the same concept and even when recognition is correct
both times. The derived phasor symbols therefore differ, and unbinding
leaves a residual phase error.

This is structurally different from the spike-timing jitter already
shown tolerable: spike-timing jitter perturbs each phasor dimension
independently, whereas activity-vector noise projects through a fixed
derivation function into **correlated** phase error across all
dimensions, and the encode-time and query-time errors are independent
draws that do not cancel.

The arc's question is therefore precise: **does deriving the phasor
symbol from a noisy population activity vector still support FHRR
binding, unbinding, and clean-up well enough to clear the frozen 0.80
compositional bar?**

## The cheap-first probe

Following the de-risking discipline that the FHRR arc used (three
cheap numpy probes before any substrate build), this arc starts with a
single cheap-first probe before any real-substrate runner is built.

`research/findings/raw/activity_level_integration_probe.py` -- a
standalone numpy probe, explicitly an engineering ceiling-clarification
(non-load-bearing), no protected/frozen/moat module touched, no
automatic differentiation.

It models the activity-level interface:

- **Concept activity centroids.** Each concept has a "true" population
  activity pattern -- its centroid. Two activity representations are
  tested: a coarse 16-dimensional per-pool vector (what the substrate's
  pool-firing readout returns directly) and a richer 256-dimensional
  distributed population code (closer to the substrate's per-neuron
  activity). A concept's centroid has a high firing rate (drawn in the
  realistic 0.2-0.8 range) on its active dimensions and a low leakage
  rate (0.0-0.05) elsewhere -- the firing levels the codebase documents
  for correctly-recognized words.

- **Trial noise.** Each time a concept is observed, the activity vector
  is its centroid plus zero-mean Gaussian noise, clipped to be
  non-negative. The noise standard deviation is swept.

- **Symbol derivation.** A fixed, deterministic function maps an
  activity vector to a phasor symbol: a fixed random complex projection
  of the (normalized) activity vector, taking the phase of each
  projected component. This is smooth -- a small change in activity
  produces a small change in phase -- and deterministic, so the same
  activity vector always yields the same symbol.

- **Composition.** Facts are stored by binding an activity-derived cue
  symbol to an activity-derived filler symbol and bundling. Each fact
  is queried by unbinding with a cue symbol derived from an
  **independent** activity trial of the same cue concept. Clean-up
  matches the recovered vector against a vocabulary of concept symbols
  derived from the stable centroids (the consolidated, stable concept
  identities).

Recognition is held correct throughout -- this probe isolates the one
new variable, activity-derived vs lookup-derived symbols. Recognition
error propagation was already characterized by the validated
identity-level integration.

## Pre-registered reading (fixed before the run, never tuned)

The probe sweeps the activity-noise standard deviation over
{0.0, 0.05, 0.10, 0.20} and the phasor dimension over {256, 1024}.

- **PASS** -- activity-level integration is reachable -- if the probe
  clears the frozen 0.80 bar at all loads {2, 3, 5} at activity noise
  <= 0.10 (a coefficient of variation of about 20% on a mean firing
  rate near 0.5, which is within the realistic range of cortical
  trial-to-trial rate variability) at some dimension <= 1024. In this
  case the arc proceeds to design and build the real activity-level
  integration runner, which captures the substrate's actual population
  activity vectors and derives the phasor symbols from them, under the
  full standard discipline (frozen verdict module, dedicated
  adversarial review, honest propagation).

- **NEGATIVE** -- the identity-level interface is the validated ceiling
  -- if it does not clear 0.80 at all loads at activity noise <= 0.10.
  In this case the honest finding is the noise-corruption mechanism:
  activity-derived symbols carry the in-flight trial-to-trial noise of
  the population activity, which the discrete-label lookup discards,
  and that noise is enough to break FHRR composition. This is itself a
  biology-translatable result -- it says the discrete-label bottleneck
  is doing real work (it denoises), and a faithful activity-level
  interface would need an explicit denoising or averaging stage. The
  parallel scaling arc then becomes the next step.

Either outcome is propagated honestly (a findings document, both git
remotes). A negative outcome is a real finding, not a failure.

## Honest scope

This is one cheap-first probe and a pre-registered decision. It does
not by itself build anything load-bearing. It de-risks (or rules out)
the activity-level integration arc the same way the FHRR numpy probe
de-risked the composition layer. The validated identity-level
integration stands regardless of this probe's outcome; this arc asks
only whether a more biologically faithful interface is reachable on
top of it.
