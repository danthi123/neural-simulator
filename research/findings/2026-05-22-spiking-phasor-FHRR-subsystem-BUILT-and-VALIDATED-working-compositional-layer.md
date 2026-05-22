# Spiking-phasor FHRR composition subsystem BUILT and VALIDATED: a genuine time-stepped spiking implementation of vector-symbolic composition clears the project's frozen 0.80 compositional bar at 100% at all loads {2,3,5} with a clean no-confabulation abstention signal -- the first working compositional layer of the project, after eight architectures could not produce one

## Status

Build milestone. After the cheap-first trilogy (FHRR algebra
sufficient / spiking-phasor noise-tolerant / abstention preservable,
all green), this is the working subsystem: a genuine time-stepped
spiking implementation of Fourier Holographic Reduced Representation
(FHRR) vector-symbolic composition, after Orchard & Jarvis 2023.

Net-new module `research/runners/spiking_phasor_fhrr.py`. Standard
library + numpy only; no protected/frozen/validated module imported
or modified; no automatic differentiation (the operators are
integrator dynamics, not gradients).

## What was built

A symbol of dimension N is N spiking-phasor neurons, each firing once
per global cycle of T=1000 steps; the value is the PHASE of the spike.
The composition operators are populations of genuine time-stepped
integrator neurons, vectorized over the N dimensions:

- **bind** -- the phase-sum neuron, Orchard's Algorithm 1, implemented
  as real p/q integrators stepped over two cycles (the two-integrator
  design handles a phase sum exceeding one period). `phase_sum_neuron`
  steps t = 0..2T with p counting up to the first input spike, q
  taking p's value then counting down, the neuron spiking when q
  crosses zero.
- **unbind** -- the phase-subtraction neuron (the elapsed time between
  the two input spikes).
- **bundle** -- the phase-midpoint / FHRR bundle (phase of the complex
  sum).
- **clean-up** -- winner-take-all over the vocabulary by phase-
  similarity, WITH an abstention threshold: a query whose top
  similarity falls below threshold returns ABSTAIN (index -1) rather
  than a confident nearest-vocabulary answer. This is the
  no-confabulation moat, carried natively in the composition layer.

The module exposes a `SpikingPhasorFHRR` class (encode = bind+bundle,
query = unbind) and a frozen-bar self-test.

## Result (pre-registered frozen verdict; self-test)

The self-test runs the project's compositional task -- 8 cue x 8
filler symbols, bundle L facts, query each, clean up -- as a genuine
time-stepped spiking-phasor simulation, 300 trials per load:

```
| load | compositional accuracy | groundable sim min | ungroundable sim max | abstention separates |
|------|------------------------|--------------------|----------------------|----------------------|
| L=2  | 1.0000                 | 0.597              | 0.114                | YES                  |
| L=3  | 1.0000                 | 0.454              | 0.114                | YES                  |
| L=5  | 1.0000                 | 0.303              | 0.112                | YES                  |

Pre-registered verdict -> PASS.
```

The subsystem clears the project's frozen 0.80 compositional bar at
100% at every load, AND at every load the minimum groundable
clean-up similarity sits well above the maximum ungroundable
similarity -- the abstention signal cleanly separates "answer" from
"I don't know" with margin.

## What this is

This is the first working compositional layer the project has. Eight
architectures plus four diagnostic probes plus two substrate variants
of the biology-grounded rate-coded substrate could not produce
compositional retrieval that cleared the bar -- they plateaued at
~0.46, the readout mostly at the noise floor. A spiking-phasor FHRR
composition layer clears it at 100% with a clean abstention moat.

## What this is NOT, and the honest open work

The subsystem operates on abstract random spiking-phasor symbols. It
is NOT yet integrated with the project's biological substrate. The
project's validated v14/v16 substrate recognizes concept words
(direct binding, 88.75% multi-seed) -- a rate-coded representation.
The spiking-phasor FHRR layer is phase-coded. The genuine open arc is
the INTERFACE: the project's substrate as the concept-recognition
front-end, the spiking-phasor FHRR layer as the composition back-end,
joined at the concept-identity level. This is a two-system
architecture, and whether that is the brain-analogue direction the
owner wants -- versus pushing the rate-coded substrate further -- is
the standing paradigm question, surfaced honestly in the phase-coded
VSA design doc.

Also honest: the spiking-phasor neuron models, while genuine
time-stepped integrators, are function-first engineered devices
(Orchard's design), not derived from a biological neuron model.
Theta-gamma phase coding is real biology; this realization of it is
biology-inspired engineering. The subsystem's value is established
and concrete: a working, noise-tolerant, abstention-carrying
compositional layer. Its biological faithfulness is partial and
stated plainly.

## Discipline check

Pre-registered frozen 0.80 bar; not tuned. Net-new module; no
protected/frozen/moat module imported or modified; no autograd.
Honest propagation both remotes.

## Files / evidence

- Subsystem: `research/runners/spiking_phasor_fhrr.py`
- Self-test result: `research/findings/raw/spiking_phasor_fhrr_selftest.json`
- Cheap-first trilogy that de-risked it: `fhrr_numpy_probe`,
  `spiking_phasor_fhrr_probe`, `fhrr_abstention_probe`.

## Next arc

Interface the spiking-phasor FHRR subsystem with the project's
concept substrate: the validated v14/v16 substrate recognizes the
concept words and supplies concept identities; the FHRR layer
composes them; the composed result is read out with the abstention
moat. The arc: a design pass pinning the interface (how a recognized
concept maps to a spiking-phasor symbol, and how a composed result is
verified), then the build and a biological-scale decisive run under
the standard discipline (frozen verdict module, dedicated adversarial
review, honest propagation).
