# Where the vocabulary-scaling limit actually lives (2026-05-31)

Prompted by the owner's pushback ("scaling past 320 feasible despite the limits we saw?").
Measured the bind/cleanup recovery vs vocabulary size V to locate the limit precisely.

## Measurement (bind/unbind K=3, overlapping codes between-cos ~0.70)
numpy algebra:  V=16/32/64/160/320/640 -> recovery 1.000 at every V.
spiking (bias-500, D=800): V=16/64/160/320 -> recovery 1.000 at every V.

The target's self-match (~1.0) beats every distractor's UNIFORM 0.70 overlap regardless of how
many distractors there are -> the bind/cleanup COMPOSITION layer is vocabulary-robust.

## Conclusion: the 320-scale limit is in the RECOGNITION FRONT-END, not composition
- Front-end (text -> distinct concept codes): the real ceiling. v17 28-word = structural
  imbalance (motor pools dominate); G.20 reached 320 only at 98.4% via sparse multi-bridge
  (per-bridge 64 = clean 100%; deterministic pattern-12 gap). Getting hundreds of clean,
  distinguishable concept codes is the hard part.
- Load per structure (K, items bound at once): separately capped ~6 (firing-rate capacity).
- Direction-7 "envelope bends at V=320" was the COMBINATION (load L=5 AND vocab 320 AND the
  sparse-distributed cross-bridge ORDERING mechanism), not vocabulary alone, and not this new
  bind/relational-memory pipeline.

## Honest correction of two overstatements
- "Scaling vocab is tractable, pieces exist" -- WRONG: the front-end is a real 320 wall (98.4%).
- "Cleanup degrades with vocabulary" -- ALSO WRONG: it does not (robust to V=320+ in spiking).

The new pipeline would consume a larger vocabulary fine IF the front-end supplied clean codes;
the realistic ceiling is ~64/bridge at 100%, ~320 at 98.4%, gated by recognition.
