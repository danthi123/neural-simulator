# Fully-biologized grounded composition = NEGATIVE: the biologized attractor clean-up and the biologized grounded symbols have incompatible orthogonality requirements

> **CORRECTION (2026-05-22, same day): this NEGATIVE is SUPERSEDED.**
> It was specific to dentate-gyrus pattern separation as the grounding
> transform, which floors the symbol overlap at ~0.07 -- too correlated
> for the attractor clean-up. The correct transform is common-mode
> removal (mean-centering): subtracting the across-concept mean activity
> drops the grounded-symbol overlap to ~ -0.05 (near-orthogonal), and
> the same end-to-end pipeline then clears the frozen 0.80 bar at 0.98
> multi-seed. The "incompatible orthogonality requirements" framing was
> wrong: the requirements are compatible once the symbol is grounded by
> common-mode removal rather than dentate-gyrus separation. See
> `2026-05-22-biologized-grounded-composition-PASS-mean-centering-closes-the-arc-and-corrects-the-premature-negatives.md`.
> This document is kept as the honest trail.

## Status

The end-to-end test of a compositional pipeline that is biology-grounded
at every stage -- longer-integration recognition, dentate-gyrus
pattern-separated grounded symbols, resonate-and-fire FHRR composition,
attractor clean-up, no oracle symbol table. Result: NEGATIVE, with a
precise, diagnostic-confirmed cause. Two of the biologized pieces, each
validated in isolation, are mutually incompatible: the attractor
clean-up requires near-orthogonal symbols, and the pattern-separated
grounded symbols are not orthogonal enough.

## What was tested

Every engineered shortcut of the phase-coded composition layer had been
addressed individually. This runner composed them into one pipeline,
reusing the real activity cache (numpy, no simulation run): recognise a
concept by averaging its activity over K=8 observations; take the
recognised concept's dentate-gyrus pattern-separated code, projected to
a phasor, as the grounded symbol; compose with the resonate-and-fire
FHRR layer; clean up with the attractor. Measured against the frozen
0.80 compositional bar, multi-seed, loads {2,3,5}.

## Result (pre-registered; multi-seed 42/43/44; frozen 0.80 bar)

```
            integrated mean    composition-only mean
L=2         0.353              0.356
L=3         0.327              0.329
L=5         0.326              0.331

VERDICT -> NEGATIVE (far below the 0.80 bar)
```

Integrated accuracy is about 0.33 -- only a little above the 0.25
chance level for the 4-way clean-up. Critically, the composition-only
accuracy (restricted to facts whose cue and filler were both recognised
correctly) is essentially equal to the integrated accuracy. So this is
not recognition-bounded: even on correctly-recognised facts, with the
correct grounded symbols, the composition fails.

## Diagnosis -- the precise, confirmed cause

The pattern-separation probe had shown the pattern-separated grounded
symbols compose at 1.000 -- but with a numpy FHRR and a soft
nearest-match clean-up. This pipeline differs in using the
resonate-and-fire FHRR layer and the attractor clean-up. A direct
diagnostic isolated the cause:

```
grounded symbols, mean pairwise similarity                 0.187
attractor settle of a CLEAN grounded symbol -> identifies it   1/16
soft argmax of a CLEAN grounded symbol -> identifies it       16/16
```

The attractor clean-up cannot identify even a clean, un-composed,
un-noised grounded symbol -- it identifies one of sixteen at 1/16,
exact chance. A soft argmax over the identical symbols identifies all
sixteen. The grounded symbols carry the information; the attractor
specifically cannot use it.

The reason is the same mechanism found for the raw concept
representations. A Threshold Phasor Associative Memory weight matrix
is the outer product of the stored patterns; if the patterns are
mutually correlated, the attractor is degenerate -- the correlated
patterns merge into one dominant basin and every input settles there.
The raw concept representations overlap by 0.45 and the attractor was
fully degenerate. Dentate-gyrus pattern separation reduces the overlap
to 0.19 -- and the attractor is STILL degenerate at 0.19. The attractor
clean-up needs near-orthogonal symbols, of the order the oracle's
random vectors have (~0.04). Pattern separation orthogonalises the
substrate's representations only part of the way -- from 0.45 to 0.19,
not to 0.04.

## The honest finding

Two of the biologized pieces have incompatible requirements:

- The biologized clean-up (shortcut 3, the attractor) requires
  near-orthogonal symbols. Its self-test passed because it was tested,
  honestly, on random near-orthogonal symbols. That result stands for
  near-orthogonal symbols; the end-to-end pipeline now adds the honest
  condition that the attractor clean-up is degenerate once the symbols
  are even moderately correlated (0.19).
- The biologized grounded symbol (shortcut 2's pattern-separation
  route) produces symbols at 0.19 mutual similarity at the biological
  dentate-gyrus setting, and pattern separation cannot reach the ~0.05
  the attractor needs. A multi-seed dentate-gyrus sweep
  (`dg_orthogonality_sweep.py`) over expansion ratios {4, 8, 16, 32}
  and sparsities {2%, 1%, 0.5%, 0.2%} -- far past biological values --
  confirms this: the stored-symbol mean similarity floors at 0.070,
  and the attractor stays degenerate (1-of-16 self-identification) at
  every one of the sixteen settings. Recognition in this pipeline is
  done by temporal averaging in the raw activity space; it does not
  pass through the dentate gyrus, so a stronger separation does not
  cost recognition -- the limit is purely that the substrate's
  sixteen concept activity vectors are too mutually structured to
  orthogonalise below ~0.07, while the attractor needs below ~0.05.

So a composition layer that is biology-grounded at every stage does not
compose end-to-end on this substrate. The neurons biologize
unconditionally (the resonate-and-fire layer works regardless of symbol
orthogonality). But the symbol and the clean-up cannot both be
biologized: the attractor clean-up needs an orthogonality the
substrate's grounded representations do not have, and pattern
separation cannot supply it.

The validated compositional capability -- the identity-level
integration, multi-seed 0.96-0.99 -- works precisely because it sidesteps
this: it uses oracle near-orthogonal symbols and a soft-similarity
clean-up. Each is, by the project's biological-realism standard, an
engineered choice; this finding shows the two are not independently
removable. The oracle symbol supplies orthogonality; the soft-similarity
clean-up tolerates non-orthogonality. Replace the oracle symbol with a
grounded one and the soft clean-up still works (the pattern-separation
probe: composition 1.000) -- but replace the soft clean-up with the
biologized attractor and it requires the orthogonality the grounded
symbol lacks.

## Terminal synthesis of the compositional-biologization line

- The project has a validated compositional retrieval capability (the
  identity-level integration, 0.96-0.99 multi-seed, adversarially
  reviewed).
- The composition layer's NEURONS are biologizable unconditionally
  (resonate-and-fire, shortcut 1, PASS).
- The composition layer's SYMBOL and CLEAN-UP cannot both be biologized
  end-to-end on this substrate: the attractor clean-up requires
  near-orthogonal symbols; the substrate's grounded representations,
  even pattern-separated, are 0.19-correlated; the two requirements
  conflict.
- The root cause, which the whole compositional line converged on, is
  that the substrate's concept representations are fundamentally
  mutually overlapping (~0.45 raw). Every downstream mechanism that
  needs orthogonality -- FHRR binding, the attractor clean-up -- inherits
  that overlap as a bound.

This is a complete, honest, biology-translatable result set. The
biology-translatable insight: a vector-symbolic compositional layer
and a Hopfield-type attractor clean-up both require near-orthogonal
atomic symbols; a biological substrate whose concept representations
share common-mode structure does not natively provide them, and
pattern separation closes the gap only partially. The honest
engineering consequence: the validated compositional capability's
oracle symbols + soft clean-up are a coupled pair that, on this
substrate, cannot both be replaced by biological mechanisms.

## Next step

The compositional-biologization line is at a thorough, honest terminus.
The root cause is now precisely named: the substrate's concept
representations overlap by ~0.45. The genuinely next arc, per the
project's goal, is the substrate itself -- whether the concept-pool
architecture can be trained to produce more separable concept
representations (for example by routing the concept pools through the
validated dentate gyrus during training, so the STORED representations
are separated, rather than separating them after the fact). That is a
substrate-training investigation, a distinct and substantial arc.

## Honest scope

This NEGATIVE does not retract any prior result. The identity-level
integration stands. Shortcut 1 (resonate-and-fire neurons) stands
unconditionally. Shortcut 3 (the attractor clean-up) stands for
near-orthogonal symbols, with the honest added condition that it is
degenerate for correlated symbols. Shortcut 2's pattern-separation
route stands as "orthogonality partially solved (0.45 to 0.19)". The
new finding is that these do not compose into a fully-biological
end-to-end pipeline, and exactly why.

## Files / evidence

- Runner: `research/findings/raw/biologized_grounded_composition.py`
- Result: `research/findings/raw/biologized_grounded_composition.json`
- The pattern-separation probe (composition 1.000 with soft clean-up):
  `research/findings/raw/pattern_separation_grounding_probe.py`
