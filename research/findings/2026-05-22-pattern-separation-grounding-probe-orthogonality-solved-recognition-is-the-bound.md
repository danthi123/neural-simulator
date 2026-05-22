# Pattern-separation grounding probe: pattern separation solves the symbol-orthogonality half of shortcut 2, but grounding then reduces to concept recognition -- which is the substrate's own bound

## Status

The cheap-first probe for the routing out of the shortcut-2 NEGATIVE.
Result: a clear, two-part finding. Pattern separation solves the
orthogonality problem decisively -- the separated symbols are
near-orthogonal and compose perfectly. But recognizing which concept a
noisy observation is, by separating it, fails -- the classic
separation-versus-completion tension. The honest synthesis: grounding
the composition symbol is reducible to two things -- orthogonality,
which pattern separation supplies, and concept recognition, which is
the substrate's own (imperfect, known) capability. The composition
layer is recognition-bounded whether its symbols are oracle-assigned or
substrate-grounded.

## Why this probe exists

Biologization shortcut 2 (ground the composition symbol in the
substrate's activity) was a terminal NEGATIVE: the substrate's concept
representations overlap by a mean pairwise similarity of about 0.45,
and the composition layer requires near-orthogonal atomic symbols. The
project's own validated biology -- dentate-gyrus pattern separation
(catalog D.12, an input cosine of 0.80 measured down to 0.218) -- is
the candidate fix. This probe tested it.

## Result (pre-registered; multi-seed 42/43/44; frozen 0.80 bar)

```
raw-derived symbols, mean pairwise similarity        0.433
pattern-separated symbols, mean pairwise similarity  0.170   (target < 0.30)
FHRR composition with separated symbols, L=2/3/5     1.000 / 1.000 / 1.000
noisy-observation recognition (separate-then-match)  0.457   (target >= 0.80)

VERDICT -> grounding via pattern separation INSUFFICIENT (recognition)
```

The probe modelled the dentate gyrus as a fixed random expansion (a
four-fold expansion) followed by a two-percent k-winners-take-all
sparsification -- a fixed, deterministic transform; the same activity
always maps to the same separated code.

## What pattern separation solved

The orthogonality problem is solved, decisively. Pattern separation
reduced the mean pairwise similarity of the concept symbols from 0.433
to 0.170 -- into the range the project's D.12 validation measured for
the dentate gyrus. And the separated, near-orthogonal symbols clear the
frozen 0.80 compositional bar at every load, perfectly (1.000). The
orthogonality that the oracle lookup supplied for free, pattern
separation supplies from the substrate's own representations. The
nonlinearity of the sparsification is what decorrelates: two
overlapping inputs activate mostly-disjoint sparse sets.

So the symbol-orthogonality half of shortcut 2 is genuinely solved by a
biology-grounded mechanism the project had already validated.

## What it did not solve, and why

Recognizing which concept a noisy activity observation is, by passing
that observation through the same dentate-gyrus separation and matching
its separated code to the concepts' separated codes, fails at 0.457 --
near chance.

The reason is the classic, well-characterized separation-versus-
completion tension. Pattern separation makes SIMILAR inputs into
DISSIMILAR codes. A noisy observation of a concept and that concept's
consolidated representation are similar inputs -- so the dentate gyrus
separates them too, and the noisy observation's separated code no
longer matches its own concept's separated code. The dentate gyrus
separates across memories; it does not, by itself, map a noisy cue back
onto a stored memory. That is pattern completion -- the role of the
CA3 recurrent network, not the dentate gyrus.

## The honest synthesis

Grounding the composition symbol decomposes into two requirements:

1. The symbols must be near-orthogonal. Pattern separation supplies
   this -- validated here (0.43 to 0.17, composition 1.000).
2. A noisy activity observation must be mapped to the correct concept.
   This is concept recognition. It cannot be done by separating the
   observation (separation pushes it away from the concept). In the
   raw activity space, recognition of a noisy observation is the
   substrate's own capability -- a direct measurement during the
   shortcut-2 work put soft nearest-match recognition at about 0.74,
   and the substrate's documented direct-binding recognition is in the
   0.74-0.88 range depending on training and measurement.

So a fully substrate-grounded symbol pipeline is achievable in
principle -- recognize the concept, then use that concept's
pattern-separated code as the symbol -- but it is RECOGNITION-BOUNDED.
Its accuracy is the substrate's recognition accuracy, not 1.0.

This is exactly the bound the validated identity-level integration
already operates under. That integration recognizes a word with the
substrate and looks up an oracle symbol; its honest scope statement
already says "integrated accuracy is recognition-bounded." The
biologization arc now shows this is not an artifact of the oracle
lookup: whether the symbol is oracle-assigned or substrate-grounded via
pattern separation, the composition layer's bound is the substrate's
concept recognition. The oracle lookup was never the limiting
shortcut; recognition is the bound.

## The biologization arc -- terminal synthesis

The arc set out to replace three engineered shortcuts in the phase-
coded composition layer. The honest, complete outcome:

- Shortcut 1, the function-first integrator neurons: replaced with
  resonate-and-fire neurons (a recognized biological neuron model).
  PASS.
- Shortcut 3, the argmax-over-a-stored-list clean-up: replaced with an
  attractor settle for identification plus a separate familiarity gate
  for abstention; the structural finding that a pure attractor settle
  confabulates. RESOLVED.
- Shortcut 2, the oracle-assigned symbols: the symbol's required
  property is near-orthogonality; pattern separation supplies it from
  the substrate's own representations; what remains is concept
  recognition, which is the substrate's own bound. Shortcut 2 is
  closed: the symbol is groundable, the pipeline is recognition-bounded.

The composition layer can be biologized in its neurons and its
clean-up. Its symbols can be grounded in the substrate via pattern
separation, but the grounded pipeline is recognition-bounded -- and so
is the oracle-symbol pipeline. The whole compositional line converges
on one bound: the substrate's concept-recognition accuracy.

## Next step

The pre-registered discipline step: a dedicated adversarial review of
the biologized composition layer (shortcuts 1 and 3) and this arc's
findings, before any capability-status synthesis. The convergent
finding -- that the composition capability is recognition-bounded --
relocates the open problem precisely: improving the compositional
capability further means improving the substrate's concept recognition
(reducing its activity noise, sharpening its concept representations),
which is the project's standing recognition-substrate problem.

## Honest scope

Cheap-first probe (numpy, models the dentate gyrus as a fixed expansion
+ sparsification). It establishes that pattern separation solves
symbol orthogonality and that grounding is recognition-bounded. It does
not by itself build a grounded pipeline. The validated identity-level
integration stands and is unaffected.

## Files / evidence

- Probe: `research/findings/raw/pattern_separation_grounding_probe.py`
- Result: `research/findings/raw/pattern_separation_grounding_probe.json`
- The shortcut-2 NEGATIVE this routes from:
  `research/findings/2026-05-22-biologization-shortcut-2-NEGATIVE-the-oracle-supplies-orthogonality-the-substrate-cannot.md`
