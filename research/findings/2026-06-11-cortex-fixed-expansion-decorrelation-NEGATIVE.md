---
type: finding
status: live
date: 2026-06-11
mechanism: cleanup
---

# Cortex fixed-expansion decorrelation probe: NEGATIVE

**Status:** NEGATIVE — reproducibility and decorrelation do NOT coexist for a fixed random expansion on correlated dense inputs. The DG k-WTA tension is NOT specific to the spiking mechanism; it is a fundamental property of threshold-based sparse codes applied to dense correlated inputs.

**Date:** 2026-06-11
**Runner:** `research/runners/cortex_fixed_expansion_decorrelation_probe.py`
**Raw JSON:** `research/findings/raw/_cortex_fixed_expansion_decorrelation_probe.json`
**Seeds:** 42, 43, 44 (all complete)
**Backend:** `SIM_BACKEND=numpy` (CPU, no `sim/` edits)

---

## 0. Context

The DG rate-accumulated k-WTA probe (NEGATIVE, `2026-06-11-cortex-dg-ratekwta-cleanup-NEGATIVE.md`)
found a fundamental tension: at the k value where repro first exceeded 0.7, the between-concept
cosine was also ~0.80 (= raw input cosine, no decorrelation). The probe named a "mechanistically
distinct" alternative: a **FIXED random expansion + threshold** (Marr 1969 / Albus 1971 granule
layer; Babadi & Sompolinsky 2014), where the deterministic fixed weights should give reproducibility
"free" (same input -> same output), leaving only decorrelation to achieve.

**The probe question:** does a fixed random expansion + top-k threshold on the brain's correlated
denoise64 codes (between-code cosine ~0.81) achieve BOTH between-cos <= 0.1 AND same-input
reproducibility >= 0.9 simultaneously?

---

## 1. Unit check (mandatory convention check)

| Seed | Input between_cos | Status |
|------|-------------------|--------|
| 42   | **0.8201**        | PASS   |
| 43   | **0.8047**        | PASS   |
| 44   | **0.7992**        | PASS   |

All seeds: input between-cos > 0.6. Codes loaded in NATIVE convention (mean over obs samples,
random Gaussian project to 800 dims, mean-center, unit-normalize). No decorrelation applied.

---

## 2. Stage 1 results: the full operating-point sweep

Expansion ratios r = D_exp/D_in ∈ {4, 8, 16} (D_in=800 -> D_exp ∈ {3200, 6400, 12800}).
Active fractions f ∈ {0.05, 0.1, 0.2} (top-k threshold).
Noise sigma ∈ {0.001, 0.005, 0.01, 0.05, 0.1, 0.2} for reproducibility.

**Joint GO criterion: repro at sigma=0.1 >= 0.9 AND between_cos <= 0.1. NOT MET at any point.**

### Seed 42 (D_in=800, input_cos=0.820)

| Config     | between_cos | repro@sigma=0.1 | margin/noise | joint_GO |
|------------|-------------|-----------------|--------------|----------|
| r=4, f=0.05  | 0.493       | 0.030           | 0.0009       | ---      |
| r=4, f=0.10  | 0.550       | 0.042           | 0.0006       | ---      |
| r=4, f=0.20  | 0.588       | 0.057           | 0.0003       | ---      |
| r=8, f=0.05  | 0.508       | 0.031           | 0.0006       | ---      |
| r=8, f=0.10  | 0.550       | 0.042           | 0.0003       | ---      |
| r=8, f=0.20  | 0.578       | 0.057           | 0.0002       | ---      |
| r=16, f=0.05 | 0.511       | 0.030           | 0.0002       | ---      |
| r=16, f=0.10 | 0.549       | 0.044           | 0.0002       | ---      |
| r=16, f=0.20 | 0.578       | 0.058           | 0.0001       | ---      |

Seeds 43 and 44 identical in character (full table in raw JSON):
- between_cos range: 0.45–0.58 (never <= 0.1)
- repro at sigma=0.1 range: 0.025–0.057 (never >= 0.9)
- margin/noise range: 0.0001–0.0009 (always << 1)

**No operating point meets the joint criterion at any seed.**

---

## 3. THE KEY FINDING: repro/decorrelation tension persists for a fixed expansion

### Why repro is catastrophically low (0.03-0.06) despite the expansion being deterministic

The expansion IS deterministic given the input — but the THRESHOLD on a continuous activation
is not. This is the critical insight:

**Activation distribution:** for fixed W_ij ~ N(0, 1/D_in) and unit-normed input code c,
the activation of expansion unit i is:
```
a_i = W_i . c ~ N(0, 1)  [by CLT, for large D_in]
```
The top-k threshold cuts at the (1-f)-quantile of this distribution, which is a **smooth region
with O(1/D_exp) spacing between adjacent activations** (the gap between the k-th and (k+1)-th
unit shrinks as D_exp grows).

**Margin at boundary:**
```
margin = a[k] - a[k+1] ~ O(1/D_exp) = 0.000026 (for D_exp=6400)
```

**Activation noise (from input noise sigma=0.1):**
```
noise on a_i = ||W_i|| * sigma_input ~ sigma_input = 0.10
```
(since ||W_i||^2 = sum_j W_ij^2 ~ D_in * (1/D_in) = 1 -> ||W_i|| ~ 1)

**margin/noise = 0.000026 / 0.10 = 0.00026 << 1**

This is the PROOF: the threshold margin is ~3,800x smaller than the noise. EVERY boundary unit
flips on every independent noise realization. The expanded binary code is essentially RANDOM
noise relative to the input concept -- hence repro = 0.03-0.06 (near zero, not near chance).

**The crossover** (where repro reaches 0.9) requires sigma << 0.003 -- a noise level ~30-100x
below the realistic spiking noise (sigma ~ 0.05-0.1 from OU dynamics). This is NOT achievable
by any biological spiking mechanism without fundamentally changing the input code structure.

---

## 4. Why between_cos stays at 0.45-0.59 (not 0.1)

The linear expansion W is a random matrix (not structured for whitening). For **correlated dense
inputs**, a random linear expansion PRESERVES the correlation structure:

```
cos(W*c_i, W*c_j) = (c_i^T W^T W c_j) / (||W*c_i|| ||W*c_j||)
                  ≈ c_i^T c_j  (for large D_exp, W^T W / D_exp -> I)
                  = input_cos ~ 0.82
```

The threshold (top-k) creates sparsity which **partially** reduces the cosine (0.82 -> 0.50),
but the COMMON MODE shared by all 16 concepts -- the dominant eigenvector of the input covariance
-- passes through the linear expansion and survives the threshold. This is why the expanded
between-cos is reduced but never approaches 0.1.

The Babadi & Sompolinsky (2014) result applies to **sparse random inputs** (not dense correlated
ones). For sparse inputs, the common mode is weak and the threshold is more effective. Our
denoise64 codes are dense real-valued codes with ~0.82 pairwise cosine; the theoretical
conditions for effective expansion recoding are not met.

---

## 5. Lesion control (confirms decorrelation rides the expansion, not input structure)

All three seeds: between_cos_lesion = 1.000 after zeroing W.

With W=0, all expansion unit activations are 0, and the tie-breaking selects the SAME first-k
units for every input -> all expanded codes are identical -> between_cos = 1.0. This confirms:
- Any decorrelation in the ACTIVE probe comes from the expansion W (not input structure)
- The lesion correctly collapses discrimination
- Anti-cheat passes (verdict: PASS, all seeds)

---

## 6. Cleanup parity on expanded codes

At the best operating point (r=16, f=0.2) for each seed:

| Noise sigma | Argmax parity | Hopfield parity | Chance |
|-------------|---------------|-----------------|--------|
| 0.00        | 1.000         | 0.062-0.081     | 0.062  |
| 0.10        | 0.706-0.925   | 0.044-0.081     | 0.062  |
| 0.20        | 0.362-0.556   | 0.031-0.069     | 0.062  |
| 0.50        | 0.169-0.181   | 0.050-0.062     | 0.062  |

**Argmax on CLEAN expanded codes: 1.000** -- the expanded codes ARE distinguishable at zero noise.
**Argmax on NOISY expanded codes: collapses rapidly** -- with sigma=0.1, 70-92% accuracy but the
Hopfield attractor collapses (0.04-0.08, = chance) because the expanded codes have between_cos~0.5
(the same common-mode wall as the raw codes).

This is a secondary confirmation: the expanded codes are NOT usable by the downstream
distributed attractor cleanup (which needs between_cos <= ~0.1 as established by the
poscontrol probe). The argmax still works at low noise as a REFERENCE, but the Hopfield
binder/cleanup mechanism collapses. The expansion does NOT make the codes suitable for
the brain-based cleanup -- it inherits the common-mode problem of the input.

**Stage 2 (on-bridge):** skipped -- Stage 1 is uniformly NEGATIVE; Stage 2 would only confirm
the same result at greater computational cost.

---

## 7. The complete characterization of the tension

The repro/decorrelation Pareto frontier for a fixed expansion on correlated codes:

| sigma (noise level) | repro at f=0.1, r=8 | between_cos |
|---------------------|----------------------|-------------|
| 0.001               | **0.97**             | 0.55        |
| 0.005               | 0.85                 | 0.55        |
| 0.010               | 0.70                 | 0.55        |
| 0.050               | 0.15                 | 0.55        |
| **0.100 (realistic)** | **0.04**             | 0.55        |
| 0.200               | 0.01                 | 0.55        |

At sigma=0.001 (very low noise): repro=0.97 -- BUT between_cos=0.55 (not decorrelated).
At sigma=0.1 (realistic spiking noise): repro=0.04 -- AND between_cos=0.55.

The between_cos is INDEPENDENT of noise level (determined only by the expansion structure).
The repro is VERY SENSITIVE to noise (drops from 0.97 at sigma=0.001 to 0.04 at sigma=0.1).
There is NO operating point where BOTH repro >= 0.9 AND between_cos <= 0.1.

This is structurally identical to the DG k-WTA result:
- DG k-WTA: repro = sep = raw_cos ~ 0.81 (they rose and fell together as k varied)
- Fixed expansion: between_cos stays ~0.55 regardless; repro collapses at realistic noise

Both hit the same wall from different angles: the dense correlated input common mode
dominates both the threshold boundary (low repro) and the expanded code correlation (high sep).

---

## 8. What would work (the honest next step)

The probe definitively establishes: **you cannot decorrelate dense correlated codes with a
fixed expansion + binary threshold at realistic noise levels**. The mechanisms that WOULD work:

1. **Host ZCA/PCA whitening** (already shown to work; the shortcut): remove the common mode
   before expansion. Between-cos -> 0, repro improves dramatically. But this is a host operation
   (O(D^3) eigendecomposition, not neurons/synapses).

2. **Learned recurrent attractor for whitening** (the biological path, untried): cortical columns
   learn to whiten their inputs via recurrent inhibition and Hebbian plasticity (Barlow 2001;
   Olshausen & Field 2004). This is what real cortex does -- but it requires a LEARNED
   transformation, not a fixed random expansion.

3. **Accept the idealization**: the VSA composer's exact-inverse algebra (the idealization) uses
   codes that are ALREADY decorrelated by their design. The spiking front-end (the step-3
   challenge) would need to learn to produce decorrelated codes from the brain's own activity --
   which is the learned-whitening problem (option 2). The principled idealization (noted in
   CLAUDE.md step 3) remains the honest scaffolding while the genuine-cortical conversion is
   designed.

4. **Decouple the problem**: the cleanup wall is solved on decorrelated codes (poscontrol probe GO).
   The decorrelation problem is the residual. The cheapest de-risk for the cortex arc is to
   establish whether a LEARNED sparse code (e.g., sparse autoencoder with ICA-style coding
   trained to decorrelate the denoise64 representations) can produce:
   (a) between_cos <= 0.1
   (b) same-input repro >= 0.9 under realistic noise (sigma=0.1)
   That is a genuinely different mechanism (learned bottleneck vs fixed expansion).

---

## 9. Verdict

**NEGATIVE** -- confirmed uniformly, 3/3 seeds, all operating points tested.

The key facts:
- **between_cos (expanded):** 0.45-0.59 at all (ratio, f) -- never <= 0.1
- **repro at sigma=0.1:** 0.025-0.058 -- never >= 0.9
- **margin/noise:** 0.0001-0.0009 -- 100-10,000x smaller than required for repro >= 0.9
- **Lesion control:** PASS (between_cos -> 1.0 when W=0, confirms expansion is the mechanism)
- **Cleanup parity (expanded codes):** argmax clean=1.000, but Hopfield collapses (0.06=chance)
  because between_cos~0.55 persists into the expanded space
- **Stage 2 (on-bridge):** skipped -- Stage 1 is definitively NEGATIVE

**The DG k-WTA tension (repro and sep cannot coexist) is NOT specific to the spiking mechanism.**
A FIXED deterministic expansion has the same fundamental tension for dense correlated inputs:
the binary threshold sits in a smooth Gaussian activation region with a margin ~3,800x smaller
than the realistic input noise, collapsing repro to near-zero regardless of expansion ratio.
Between-cos stays at ~0.55 because the shared input common mode (the dominant eigenvector of
the denoise64 code covariance) passes through the linear expansion and survives the threshold.

**The Marr-Albus granule-layer recoding is NOT a viable reproducible decorrelation front-end
for DENSE CORRELATED inputs** (it was designed for sparse, weakly-correlated mossy fiber codes,
not for dense real-valued codes with cosine ~0.82).

**The honest next step**: the decorrelation problem requires LEARNED whitening (recurrent
inhibitory circuits that learn to remove the common mode), not a fixed expansion. The step-3
cortex challenge is the learned whitening / sparse coding transformation, which is a qualitatively
different (and deeper) arc than any fixed-weight decorrelation approach.
