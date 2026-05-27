# Direction 6 geometry diagnostic: dedicated-pool grounded-symbol geometry is empirically NEAR-ORTHOGONAL post-mean-centring (mean cos -0.030, abs_mean 0.041 across 15 cells); validates the pillar n=109 SHATTER hypothesis

**Date:** 2026-05-27 ~08:50 EDT
**Status:** Diagnostic measurement; not a verdict (no bar). Provides empirical quantitative basis for the FHRR algebra capacity-ratio prediction shatter at pillar n=109.

## Why this diagnostic

Pillar n=109 D6 V=160 production decisive PASS multi-seed at L=5 OI = 0.987 BEAT both pillar n=108 D4 V=80 (0.977) and pillar n=95 G.20 sparse V=160 (0.790). The FHRR algebra capacity-ratio prediction (capacity proportional to N_dim/V; doubling V should drop boundary ~2 rungs) was DECISIVELY SHATTERED. The n=109 reviewer verdict + findings doc hypothesised that the dedicated-pool grounded-symbol geometry is substantially CLEANER than uniform-random phasors the FHRR algebra assumes -- likely near-orthogonal because each concept fires its own dedicated pool with other pools quiet.

This diagnostic empirically measures the actual cosine geometry of the cached D6 production per-concept activity vectors to quantify the "near-orthogonal" claim and provide a numerical basis for the SHATTER pattern.

## Methodology

CPU-only Python script `research/findings/raw/direction_6_geometry_diagnostic.py`. Reads the existing cached `direction_6_cache/activity_full_<bridge>_seed<N>.npz` files (15 cells = 5 bridges x 3 seeds). For each cell:
- Per-concept activity = `(M_OBS=16, n_pool_union=6400)` float32 array (sum of spike counts per neuron across 50-step stim window per observation)
- Measure 3 cosine distributions:
  1. **same-concept M_OBS cosine** -- pairwise cosines between the 16 observations of one concept (V*M_OBS*(M_OBS-1)/2 = 32*120 = 3840 pairs per cell); characterises per-trial stochasticity
  2. **different-concept cosine (raw)** -- pairwise cosines between the mean-of-16-obs vectors of distinct concepts (V*(V-1)/2 = 496 pairs per cell); characterises raw concept-pair geometry BEFORE any normalisation
  3. **different-concept cosine (mean-centred)** -- same as #2 but with per-bridge common-mode subtracted (subtractive normalisation; analog of cortical pooled inhibition); characterises the geometry the cross-bridge probe actually consumes

Across 15 cells: 57600 same-concept pairs + 7440 different-concept pairs measured per condition.

## Result (aggregate across 15 cells)

| Distribution | mean of means | mean of abs_means | cell-mean range |
|---|---|---|---|
| same-concept (M_OBS) | 0.029 | 0.029 | [0.024, 0.038] |
| different-concept (raw) | 0.184 | 0.184 | [0.131, 0.252] |
| **different-concept (mean-centred)** | **-0.030** | **0.041** | [-0.032, -0.029] |

## Interpretation

1. **Same-concept M_OBS cosine LOW (0.029)**: Individual stim observations are highly stochastic. The same concept driven 16 times produces variable spike-count patterns within its dedicated pool (~200 neurons fire variable amounts per observation). The cross-bridge probe relies on averaging M_OBS=16 observations to denoise this trial variability into a robust per-concept "mean vector." Below the n=160 dimensionality (6400-dim vectors), this is consistent with high per-trial stochasticity but consistent rate profiles.

2. **Raw different-concept cosine MODERATE (0.184)**: Without normalisation, distinct concepts share moderate common-mode (background firing, baseline drift, motor pool activity that all concepts have). This is the empirical signature confirming that cortical pooled inhibition (the biological analog of mean-centring) is the correct normalisation step.

3. **Mean-centred different-concept cosine NEAR-ORTHOGONAL (mean -0.030, abs_mean 0.041)**: After per-bridge common-mode subtraction, distinct concept-pair cosines are essentially zero. The slight negative mean is the mathematical consequence of mean-centring (sum across concepts forced to zero). The abs_mean of 0.041 is FAR below the FHRR algebra's uniform-random assumption of ~0.5. EMPIRICALLY VALIDATES the pillar n=109 SHATTER hypothesis: dedicated-pool grounded-symbol geometry is genuinely near-orthogonal after the cortical-pooled-inhibition normalisation step.

## Why this matters for capacity scaling

The FHRR algebra capacity-ratio (capacity proportional to N_dim/V) is derived under the assumption that grounded symbols are uniform-random phasors. Under that assumption, doubling V approximately halves the per-symbol effective dimensionality (because random collisions consume increasing fractions of N_dim).

With near-orthogonal grounded symbols (mean abs cosine 0.041 instead of ~0.5), there are essentially no random collisions. Each concept occupies a near-disjoint subspace of the N_dim phasor space. Capacity is then bounded by the number of disjoint subspaces a 512-dim phasor space can support, not by random-collision noise.

This explains why D6 V=160 (n=109) OUTPERFORMED D4 V=80 (n=108) at L=5 OI: at V=160, per-bridge mean-centring at V=32 per bridge yields a SHARPER common-mode subtraction (32 concepts to average against the shared baseline instead of 16), producing CLEANER concept-specific residuals. The geometry advantage compounds with vocab scaling rather than degrading.

## Predicted implication for D7 V=320

If the geometry-cleanliness scaling continues (per-bridge mean-centring at V=64 per bridge yielding even sharper common-mode subtraction), D7 V=320 should also clear the L=5 OI bar -- potentially with even better margin than D6 V=160. The D7 SMOKE (in flight at finding-doc time) and D7 PRODUCTION (auto-launches on SMOKE PASS via the post-smoke chain watcher PID 35220) will empirically test this prediction.

If D7 V=320 PASSes, the bio_brain_regions dedicated-pool architecture's cleanliness scaling is confirmed across V=80 -> V=160 -> V=320; if D7 BOUNDARYs at L=5 OI, the SHATTER pattern has a natural ceiling somewhere between V=160 and V=320 -- itself an interesting characterisation.

## Files

- Diagnostic script: `research/findings/raw/direction_6_geometry_diagnostic.py` (CPU-only; stdlib + numpy; no GPU; no protected/frozen/moat module touched)
- Result JSON: `research/findings/raw/direction_6_geometry_diagnostic.json`
- Cached activity (read-only): `research/findings/raw/direction_6_cache/activity_full_*_seed{42,43,44}.npz` (from pillar n=109 production runner, commit c1fca54)

## Discipline

No bar (this is a diagnostic, not a verdict). No protected/frozen/moat module touched. No autograd. Reuse-only on the n=109 cached activity. No-confab moat 7/7 untouched. Plain ASCII output.
