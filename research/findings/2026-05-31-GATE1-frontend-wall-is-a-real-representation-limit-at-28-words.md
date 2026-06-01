> **RETRACTED 2026-06-01 (verdict only):** the "representation limit" verdict was confounded by an UNDERTRAINED bridge (the _v17 28-word bridge was ~50 events; the 16-word control was 200). A matched 150-event 28-word bridge gives clean recognition 0.893, not 0.64. See 2026-06-01-GATE2-overturns-GATE1-28word-wall-is-undertraining-not-representation-limit.md. The PIPELINE + 16-word control remain valid; only the cross-vocab representation-limit conclusion is retracted.

# GATE 1 (validated): the 28-word front-end wall is a REAL representation limit, not a cheap readout fix

**Direction (A) preparation, compute-protecting cheap-first gate.** Decisive + validated with a positive
control. 2026-05-31.

## Question
Is the 28-word recognition wall (pool-argmax ~0.57) a LOSSY-READOUT artifact (codes are separable, the
readout is bad -> cheap fix, NO 100hr) or a genuine representation limit (codes are inseparable -> source
representation learning needed)? The internal asset map established the 16-word activity is 100% NN-
identifiable though pool-argmax recognition is only 81% -- i.e. at 16 words it IS a lossy readout. Does that
escape extend to 28 words?

## Method (and the faithfulness bug-hunt it required)
Capture multi-sample per-neuron concept activity, then compare, at matched averaging level k, pool-argmax
(the documented readout) vs a full-code leave-one-out nearest-centroid decoder. CRITICAL faithfulness work:
single 100-step captures are OU-noise-DOMINATED (two captures of the same word have cosine ~0.13-0.18, even
after a full checkpoint reload) -- so codes must be AVERAGED over many observations (the project's validated
denoise64 methodology, M_OBS=16). Round 0 (fresh from the loaded checkpoint) reproduces the front-end
probe's pool-argmax 0.571 exactly; later rounds drift, so each round restores the clean loaded state. Several
intermediate Gate-1 runs were CAUGHT AS BUGS (capture saturation from 16-in-a-row, cold reset, missing mean-
centering, p>>n logreg overfit) and corrected before any verdict.

## Result -- with a VALIDATED pipeline (16-word positive control)

16-word CONTROL (_learned16 bridge), fair head-to-head:
| k-avg | pool-argmax | full-code NN |
|------:|------------:|-------------:|
| 1 | 0.801 | **0.910 (NN wins = lossy readout)** |
| 4 | **1.000** | **1.000** |
| 8 | 1.000 | 1.000 |
-> reproduces the internal map exactly: at 16 words the full code is MORE decodable than pool-argmax (lossy
readout), and with averaging the clean codes are PERFECTLY separable (100%). The pipeline is VALID.

28-word (_v17 bridge), same fair head-to-head:
| k-avg | pool-argmax | full-code NN |
|------:|------------:|-------------:|
| 1 | 0.397 | 0.290 |
| 4 | 0.527 | 0.402 |
| 8 | 0.589 | 0.161 |
clean 16-avg codes: pool-argmax 0.643; between-concept cos 0.606; 8-avg within 0.517 vs between 0.460.
-> the full-code decoder is WORSE than pool-argmax at every level, and averaging plateaus at ~0.53-0.64
(NOT 1.000 like 16 words). The codes are genuinely weakly-separable (overlap, not noise -- averaging doesn't
fix it).

## Verdict: REPRESENTATION LIMIT at 28 words (validated)
The lossy-readout escape that works at 16 words (clean codes -> 100%) does NOT extend to 28 words (clean
codes -> ~64%). This is a genuine representation-capacity transition: the substrate's learned concept
representations become too overlapping between 16 and 28 words. No cheap readout/decoder fix recovers it
(the full code carries no extra reliably-decodable identity beyond what pool-argmax already extracts). So the
front-end wall is real, NOT a cheap fix -- consistent with the internal map's lesson that transforming the
SAME activity is bounded.

## Implication for the compute decision (Direction A)
- A cheap readout fix is OUT (Gate 1 validated). Representation learning is genuinely needed past ~16-28 words.
- BPTT is OUT as the big bet (internal map: char-level Phase 2.3a/2.3b NEGATIVE, scale makes it WORSE).
- Post-hoc transforms on the captured activity are OUT (internal map: DG/Foldiak/random all bounded).
- So the lever must be ACQUISITION-LEVEL (how the substrate learns concept reps), and the evidenced-better
  bets are: (1) scale the validated G.20 sparse-distributed architecture (160/320 -> 640, D8 infra
  scaffolded); (2) the VSA gain-field role-binding (the only composition path with a positive result); (3)
  biologically-grounded acquisition-level local learning the project has NOT tried -- e-prop (three-factor)
  or expansion+Hebbian applied DURING acquisition (Lindsay 2017), both NON-100hr.

## Next (pre-registered)
GATE 2 (cheap, ~1 GPU-hr): test an ACQUISITION-level local-learning method (e-prop, or acquisition-level
expansion+Hebbian) on 28-word recognition. Does it beat the ~0.64 clean-code wall? If yes -> a cheaper-than-
100hr acquisition lever exists. If no -> the 100hr is earned and should target G.20-scaling or VSA-roles
(NOT BPTT). The cheap-first gates have already redirected the 100hr away from the bounded BPTT.
