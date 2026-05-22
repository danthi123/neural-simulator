# Vocabulary scaling to 64 concepts = NEGATIVE: the G.20 sparse substrate's captured activity is far too sparse (0.5% active) for the activity-grounded composition pipeline validated on the 15x-denser v14/v16 substrate

## Status

The decisive multi-seed run of the vocabulary-scaling arc. Result:
NEGATIVE. The biologized grounded-composition pipeline, which clears
the frozen 0.80 compositional bar at multi-seed 0.98 on the 16-concept
v14/v16 substrate, does not transfer to a 64-concept G.20
sparse-distributed substrate as captured. The cause is diagnosed: the
captured G.20 sparse activity is near-silent -- about fifteen times
sparser than the v14/v16 activity the pipeline was validated on -- so
the grounded symbols derived from it are noise-dominated and do not
compose. This is an honest, pre-registered NEGATIVE, and the diagnosis
localises it precisely to the substrate's activity, not the pipeline.

## What was run

Following the vocabulary-scaling design and TDD plan: a runner builds a
64-concept G.20 sparse-distributed bridge, captures per-neuron
shared-pool activity for the 64 concepts, and runs the biologized
grounded-composition pipeline on it against the frozen 0.80 bar
(seeds 42/43/44, loads {2,3,5}). The runner was adversarially reviewed
CLEAR before the run -- genuine reuse, a broken run cannot score a
PASS, the bar is frozen.

## Result (pre-registered; multi-seed 42/43/44; frozen 0.80 bar)

```
            integrated mean    composition-only mean
L=2         0.106              0.116
L=3         0.117              0.113
L=5         0.101              0.107

recognition (reported separately): per-observation 0.265, temporally
averaged 0.841

VERDICT -> NEGATIVE (far below the 0.80 bar)
```

Per-seed integrated accuracy is highly variable -- seed 42 about 0.22,
seeds 43 and 44 about 0.03-0.07 -- the fragility expected when the
underlying signal is near the noise floor. Composition-only is also
about 0.11, so this is not recognition-bounded: even on
recognition-clean facts the composition fails.

## Diagnosis -- the captured activity is near-silent

A direct comparison of the captured activity vectors:

```
                              mean       fraction of neurons nonzero
G.20 64-concept (this run)    0.00015    0.0051   (0.5% active)
v14/v16 16-concept (validated 0.00099    0.0754   (7.5% active)
  substrate the pipeline
  passed 0.98 on)
```

The G.20 sparse substrate's captured activity is about fifteen times
sparser by active-fraction and about six times lower by mean rate. Only
about ten neurons of the 2000-neuron shared pool fire in a given
observation. At that firing level the per-observation activity vector
is dominated by Poisson spiking noise: which neurons fired is close to
random from one observation to the next.

This explains the result precisely. The grounded symbol is derived from
the concept's consolidated (common-mode-removed) activity. When the
activity is near-silent and noise-dominated, the consolidated vector
still carries some concept-specific structure -- enough that the
recognition readout, which averages eight observations and takes a
scale-invariant cosine match, recovers the concept about 84% of the
time. But the FHRR composition needs each grounded symbol to bind and
unbind precisely; symbols derived from near-silent noise-dominated
activity do not, and the composition collapses toward chance.

So recognition is not the bottleneck here (0.84, comparable to the 0.93
at 16 concepts). The composition is, and the composition fails because
the substrate's captured activity is too sparse to ground a clean
symbol.

## Why the G.20 sparse activity is near-silent

Two contributing factors, both honest to state:

1. The G.20 sparse architecture is by design sparse -- each concept is a
   scattered K-of-N pattern and the shared pool fires sparsely. Some
   sparsity is intrinsic and intended.
2. The bridge in this run was freshly built, not trained. The G.20
   sparse ensemble's validated 100%-discrimination results are on
   bridges that go through an encoding/training stage; the runner built
   a fresh bridge and evoked each concept with a teacher current on its
   pattern neurons. On a fresh bridge the lang_input-to-pool pathway is
   untrained, so only the teacher current drives the pattern, and it
   drives it weakly (mean rate 0.00015).

The honest consequence: this run captured activity from an untrained
G.20 sparse bridge. The design doc specified "the project's validated
large-vocabulary substrate", and the validated G.20 sparse substrate is
the trained one. So the decisive run, as executed, did not test the
biologized pipeline on the validated substrate -- it tested it on an
untrained, near-silent one. That is a setup gap, surfaced by the
diagnosis, and it is recorded honestly here.

## What this is, and what it is not

This is a genuine NEGATIVE for the pipeline as run: the activity-grounded
composition does not transfer to the G.20 sparse substrate captured
this way. It is NOT a demonstration that 64-concept compositional
retrieval is impossible -- the diagnosis shows the run did not exercise
the validated trained substrate, and the FHRR capacity curve already
showed the composition algebra itself scales far past 64. The honest,
narrow finding is: the activity-grounded pipeline needs a substrate
whose concept activity is dense enough to ground a clean symbol, and
the G.20 sparse substrate as freshly built and teacher-evoked does not
provide that.

## Next step

The diagnosis routes the next pre-registered step precisely. Two
distinct candidates, to be weighed honestly in the next design:

1. Capture from a TRAINED G.20 sparse bridge (run the validated G.20
   sparse encoding first, so a concept's drive evokes its pattern
   strongly), and/or a stronger capture drive -- then re-run the
   pre-registered test on the substrate it was meant to use.
2. Ground the symbol in the G.20 sparse PATTERN itself -- the concept's
   K-of-N code is the substrate's actual concept representation, clean
   and near-orthogonal by construction -- rather than in the noisy
   per-observation firing. This must be weighed honestly against
   whether it is still "grounded in substrate activity" or closer to
   the oracle lookup.

Either is a new pre-registered step; this NEGATIVE is propagated first.

## Honest scope

A multi-seed decisive run with a clear pre-registered NEGATIVE verdict.
The runner was adversarially reviewed CLEAR, so the NEGATIVE is not a
runner artifact; it reflects the substrate's near-silent captured
activity, diagnosed by direct comparison to the validated v14/v16
activity. No protected, frozen, or moat module was modified; the FHRR
and substrate modules were reused by import; no automatic
differentiation; the 0.80 bar was not tuned. The completed,
twice-reviewed FHRR-biologization arc (16-concept, multi-seed 0.98)
stands and is unaffected.

## Files / evidence

- Runner: `research/findings/raw/vocabulary_scaling_run.py`
- Result: `research/findings/raw/vocabulary_scaling_run_full.json`
- Design + plan: `docs/plans/2026-05-22-vocabulary-scaling-design.md`,
  `docs/plans/2026-05-22-vocabulary-scaling-implementation.md`
