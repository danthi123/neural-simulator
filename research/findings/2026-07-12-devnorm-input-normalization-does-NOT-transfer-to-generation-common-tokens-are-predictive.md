# The developmental input normalization does NOT transfer from classification to GENERATION: down-weighting common tokens HURTS next-token CE (common tokens are locally predictive) — sharpening the R3↔generation convergence back onto the LEARNED-input / deep-credit frontier

**Date:** 2026-07-12
**Status:** ❌ HONEST NEGATIVE (quick, mechanistically-sensible) — the session's validated developmental normalization (PPMI-style divisive input gain) is the WRONG lever for the emergent generator; it consistently hurts. A first-class negative that maps where the generator's input-scale lever actually is. Reuse-by-import; NO `sim/` edit, NO shared-runner edit.
**Frontier:** the R3↔emergent-generation-ladder convergence — testing whether the session's spiking-compatible developmental input normalization improves the real-text reservoir generator (the input-scale lever the ESN fixed-input invariant leaves unused).

## The hypothesis + the test
The RUNG-1 generator uses a FIXED input projection (ESN invariant); my R3 arc showed learning the input beats fixed at rate but the credit-based learning is coarseness-bound on spikes; the developmental normalization WORKS on spikes (validated on count-corpus). Hypothesis: apply the developmental normalization (down-weight common tokens, PPMI/TF-IDF-style) to the generator's input → improve real-text next-token CE. Test (`_reslm_realcorpus_devnorm_derisk.py`, no shared edit): pre-scale each token's input code by `g_v = scale/(σ+freq_v)^k` (mean-normalized), so `W_in @ (onehot·g) = g·(W_in @ onehot)` is the divisive input gain; baseline (fixed input) vs dev-norm vs permuted-gain, same reservoir, held-out next-token CE.

## Result — dev-norm consistently HURTS (TinyStories, V=200, seed 42)
| k | baseline CE | dev-norm CE | dev − base | bigram |
|---|---|---|---|---|
| 0.5 | 3.667 | 3.731 | **+0.064** | 3.939 |
| 1.0 | 3.667 | 3.742 | **+0.075** | 3.939 |
| 2.0 | 3.667 | 3.772 | **+0.105** | 3.939 |

Monotone with k: the more the common tokens are down-weighted, the WORSE the next-token CE (though both still beat the bigram). Clean, mechanistically-sensible negative.

## Why (the load-bearing mechanism) — the tasks are OPPOSITE
The developmental normalization down-weights COMMON inputs. In CLASSIFICATION/generalization (the session's confound task), the common (identity) dims are class-IRRELEVANT confounds → down-weighting them HELPS. In next-token GENERATION, the common tokens (the/a/is — function words) carry the strongest LOCAL predictive structure (they strongly constrain what comes next) → down-weighting them REMOVES predictive signal → hurts. So the *same* mechanism that suppresses a confound for classification suppresses *predictive signal* for generation — the roles are opposite. The developmental confound-suppression normalization is the WRONG lever for generation.

## ⇒ The sharpened R3↔generation convergence
The generator's input-scale lever is NOT the developmental (frequency-based) normalization — it is the LEARNED input representation: learning WHICH tokens' context is predictive (not a fixed frequency heuristic). That is exactly R3's learn-W_in (rate-validated, beats fixed) and exactly the spiking-credit-coarseness boundary (this session's convergence). So the generation scale frontier points BACK onto the LEARNED-input / **deep-credit-on-spikes (dendritic)** problem — the owner's standing priority — as the genuine path past the fixed-input ESN ceiling, NOT a developmental normalization. The developmental normalization stays validated for its own domain (classification/generalization confound-suppression on count-corpus); it does not generalize to generation.

## Files
`research/runners/_reslm_realcorpus_devnorm_derisk.py` (no-edit reuse of ReservoirStates/train_readout/eval_ce/bigram); `raw/_devnorm_smoke.json`, `_devnorm_k{0.5,2}.json`. Builds on the R3↔generation convergence (AUTONOMOUS_STATE 2026-07-12) + the SCALE findings.
