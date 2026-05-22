# Activity-level integration decisive run = NEGATIVE: the substrate's raw per-neuron activity is too noisy to derive a composition symbol from directly; the discrete-label lookup's denoising is doing real work

## Status

The decisive multi-seed run of the activity-level integration arc.
Result: NEGATIVE. This is the first attempt to replace one of the
phase-coded composition layer's engineered shortcuts -- the oracle
symbol lookup -- with a biologically faithful alternative, and the
naive form of that replacement does not work. That is an honest,
pre-registered, biology-translatable finding.

## What was tested

The validated compositional pipeline joins the project's concept-
recognition substrate to the spiking-phasor composition layer at the
concept-identity level: the substrate reports one discrete recognized
pool label, and a fixed lookup table maps that label to a pre-assigned
phasor symbol. Activity-level integration is the more biologically
faithful interface: derive the phasor symbol from the substrate's
actual per-neuron concept-pool population activity, with no discrete
label and no lookup table.

The cheap-first probe established this was reachable in principle, and
handed the build one requirement: use the distributed per-neuron
population activity, not the coarse per-pool aggregate. The decisive
runner (`research/findings/raw/activity_level_integration.py`) did
exactly that -- it captured the substrate's full 3200-dimensional
per-neuron concept-pool activity vector, derived phasor symbols from
it via a fixed random projection, and composed them through the
validated spiking-phasor composition subsystem, byte-unchanged.

## Result (pre-registered; frozen 0.80 bar; seeds 42/43/44; 300 trials/load)

```
                     L=2      L=3      L=5
integrated mean      0.378    0.361    0.331    (all << 0.80)
composition-only     0.416    0.406    0.359    (all << 0.80)
measured activity coefficient of variation: 1.61-1.68 (mean ~1.63)
recognition-clean rate: ~0.66

VERDICT -> NEGATIVE
```

The activity-derived interface does not clear the frozen 0.80
compositional bar at any load. Critically, the composition-only
accuracy (restricted to facts whose words were all recognized
correctly) is also far below the bar -- so this is not the
"recognition is the bottleneck, composition works" case. Even on
correctly-recognized facts, the activity-derived symbol is too noisy
to compose.

## Why -- the measured mechanism

The runner measured the substrate's trial-to-trial activity
variability directly: the coefficient of variation of the genuinely-
active population, across repeated observations of the same word, is
about 1.63 -- roughly 160%. The cheap-first probe had modelled
activity noise and shown that activity-derived symbols compose above
the bar only at a coefficient of variation up to about 20%, and
degrade sharply by 40%. The real substrate's per-neuron activity is
four to eight times noisier than the regime where the derivation
survives.

The reason is structural. A single neuron's firing over a 100-step
observation window is a small, highly variable spike count, and the
substrate's background noise and recurrent dynamics make the whole
population's activity level swing from one observation to the next.
The activity-derived symbol carries all of that swing. The discrete-
label lookup interface never sees it: the argmax that picks the
recognized pool label discards the entire graded activity pattern and
its trial noise, and then the symbol is a fixed, noise-free vector.

## The control: this negative is cleanly attributed

This run has a built-in control. The identity-level integration --
same substrate, same composition subsystem, same task, same seeds --
cleared the bar at 0.96-0.99. The activity-level run differs from it
in exactly one step: the symbol is derived from the raw activity
vector instead of looked up from the discrete label. Identity-level
0.96 versus activity-level 0.36 therefore isolates the cause
unambiguously to the symbol-derivation step. The composition
subsystem is not at fault (it is independently validated and scores
0.96 on the identical task with lookup symbols); the substrate is not
at fault (it is the same substrate); the task is not at fault (it is
the identical task). The activity-derived symbol is too noisy. The
negative is genuine, not an artifact.

## What this means -- and why it is not a dead end

This arc is the first concrete attempt to biologize one of the
composition layer's three engineered shortcuts. The phase-coded
composition layer, as built, is a validated engineering scaffold that
proves the compositional target is reachable; it is not yet a
biological result, because it relies on three devices a brain does
not have:

1. Function-first integrator neurons for the bind and unbind
   operations (engineered to output a phase sum, not a biological
   neuron model).
2. An oracle that assigns each concept a fixed symbol by lookup
   (the brain has no external table of orthogonal codes).
3. A clean-up that takes an argmax over an explicitly stored
   vocabulary list (the brain does not keep an enumerated answer
   list).

Activity-level integration attacked shortcut 2: replace the oracle
lookup with a symbol grounded in the substrate's own activity. The
honest result is that the naive form of this replacement fails -- raw
per-neuron population activity, read once, is too noisy to be a
symbol. This is biology-translatable: a brain cannot use a raw,
single-observation population snapshot as a stable symbolic token
either. It must first stabilise the representation. Biology does this
with attractor dynamics -- a representation that settles into a clean
fixed point -- and with temporal integration over a sustained
encoding window. That is the same mechanism as shortcut 3's biological
replacement (an attractor clean-up network): a denoiser. The two
shortcuts are coupled. Grounding symbols in substrate activity and
replacing the stored-vocabulary clean-up with an attractor network are
not independent fixes; a biological attractor representation does
both jobs at once -- it grounds the symbol in learned activity and it
denoises it.

So the negative does not close the activity-level direction; it
re-specifies it. A faithful activity-derived symbol needs an attractor
or temporal-integration denoising stage between the substrate's raw
activity and the composition layer. That is the next pre-registered
arc.

## Honest scope

This is one decisive multi-seed run with a clear pre-registered
verdict. It says the naive single-observation activity-to-symbol
derivation fails on the real substrate; it does not say activity-level
integration is impossible -- it says it needs a denoising stage, and
names what kind. The validated identity-level integration stands as an
engineering scaffold. No protected, frozen, or moat module was
modified; the composition subsystem was reused by import, byte-
unchanged; no automatic differentiation was used.

## Files / evidence

- Runner: `research/findings/raw/activity_level_integration.py`
- Result: `research/findings/raw/activity_level_integration_full.json`
- Probe + design: `research/findings/raw/activity_level_integration_probe.py`,
  `docs/plans/2026-05-22-activity-level-integration-design.md`
- The control (identity-level integration):
  `research/findings/2026-05-22-INTEGRATED-compositional-capability-multi-seed-PASS-substrate-recognition-plus-spiking-phasor-FHRR.md`
