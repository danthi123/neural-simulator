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

## Refinement (2026-05-31, owner follow-up): the front-end is less limiting than v17's 50% implied

The v17 28-word 50% pool-label recognition was the OLD concept-pool architecture (motor pools dominate
the argmax readout). Two things refine the picture upward:
- The ENCODING-AXIS architecture (finding 2026-05-10-encoding-axis-64word-3SEED-GO, n_lang=8192,
  n_motor=2000) validated 64-word RECOGNITION at 3/3 GO -- not 50%. So recognition scales cleanly to 64.
- Insight #5 (the bind uses the DISTRIBUTED activity, not the pool LABEL): the pool-label argmax is the
  lossy part; the distributed code the bind consumes is more separable (live-text 15/16 label but 1.000
  bind). So the bind's effective recognition exceeds the pool-label accuracy.

Demonstrated bind-side (sparse codes, spiking wh-QA): V=64 -> 1.000 (seed 42). So:
- 64-word conversation: CLEAN (encoding-axis recognition 3/3 GO + vocabulary-robust bind, QA 1.000).
- 320-word: feasible at ~98% (G.20 sparse multi-bridge 98.4% + robust bind).
The honest "limit" is the soft 98.4% at 320, not a hard wall, and 64 is clean. Scaling richer conversation
is front-end work (the documented arc), but the encoding-axis result shows it is tractable to ~64+.
