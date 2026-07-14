# AIRTIGHT controls (6-seed, the adversarial-verify's named test): over a fixed reservoir, the trained selective channel provides GENUINE selective-specific distal holding at the deep tail — sel beats the ORDERED bigram + a random-gate control + a no-hold control, all 6/6, the selective-specific component GROWING toward the deepest tail

**Date:** 2026-07-13
**Runner:** `research/runners/_reslm_scale_trained_selssm_vectorized_derisk.py` (now with per-depth `sel` vs ORDERED-bigram + no-hold(λ=0) + random-gate controls) · raw `research/findings/raw/_trainedsel_scale/airtight_V300_nt5000_seed*.json`. numpy (GPU-capable); NO `sim/` edit.
**Status:** ✅ the properly-controlled re-test PASSES — reinstates a genuinely-earned deep-tail long-range result after the adversarial-verify walked back the loosely-controlled framings.

## Why (the adversarial-verify's named follow-on)

The 3-skeptic adversarial-verify of the margin-over-bag scale probe retracted its "deep-distributed" framing (a metric bug: the per-depth `sel_lift=(bag−sel)−(bag−res)` has the BAG cancel → it measured sel-vs-reservoir, not a margin over the bag) and demanded the AIRTIGHT controls: per-depth sel vs the **ORDERED bigram** (not the order-blind bag) + a **no-hold (λ=0, current-token-only)** control (isolates the shallow readout fix) + a **random-gate** control (isolates the input-dependent SELECTIVE holding). Built + run 6-seed at V=300/nt=5000 (vectorized trainer, transport-free; per-depth CE per arm).

## Result — 6-seed (V=300, nt=5000, np=200 fixed echo-state reservoir)

| control (deep-tail CE reduction, LOWER-is-better so >0 = sel wins) | d≥6 | d≥10 |
|---|---|---|
| **sel < random-gate** (selective-SPECIFIC — the load-bearing control) | **+0.294 (min +0.250, 6/6)** | **+0.322 (min +0.262, 6/6)** |
| sel < no-hold (held context helps) | +0.333 (min +0.273, 6/6) | — |
| sel < ORDERED bigram | +0.089 (min +0.035, 6/6) | +0.170 (min +0.099, 6/6) |
| aggregate sel − bigram (all depths) | — | −0.054 (0/6, shallow-dominated) |

- **The selective-SPECIFIC control (sel<random-gate) is POSITIVE 6/6 at the deep tail and GROWS toward the deepest tail** (+0.294 at d≥6 → +0.322 at d≥10). `sel` and `rand` share the identical selective channel + current-token injection; the ONLY difference is the gate reads the CURRENT token (sel) vs a RANDOM token (rand). So this isolates the value of INPUT-DEPENDENT selective holding — and it is genuine, growing with depth (the opposite of a shallow spillover, which would shrink with depth).
- **sel beats the ORDERED bigram at the deepest tail** (+0.170 at d≥10, 6/6) — the proper long-range baseline, not the order-blind bag.
- **sel beats the no-hold control** (+0.333 at d≥6, 6/6) — accumulation/holding (vs pure current-token) helps at depth.
- **HONEST aggregate caveat:** the ALL-depths `sel−bigram` is −0.054 (0/6) — the overall model at this scale is bigram-level (the shallow depths dominate the aggregate, the null-discriminator regime). So the model is not yet a bigram-beater overall; the genuine result is the DEEP-TAIL STRUCTURE: the selective mechanism provides real selective-specific distal holding where long-range lives.

## ⇒ honest read (properly controlled this time)

Over a fixed reservoir, the trained selective channel provides **genuine input-dependent selective distal holding at the deep tail** — it beats the ordered bigram AND the random-gate AND the no-hold controls, all 6/6, with the selective-specific component growing toward the deepest tail. This is the airtight test the adversarial-verify demanded, and it passes: the deep-tail benefit is NOT a shallow readout fix (sel≫noheld) and NOT generic capacity (sel≫rand, identical channel/capacity, only the gate's input differs). The overall model at this tractable scale is still bigram-level (aggregate), so this is a DEEP-TAIL-STRUCTURE result, not an overall-fluency result.

**Reconciles the sibling walk-back:** the frozen coupling found sel−rand ~0 at d≥10 over an e-prop-TRAINED (strong) reservoir; here over a FIXED (weak) reservoir the selective does the deep lifting, so its selective-specific contribution is large. Both are honest — the selective mechanism's marginal deep value scales inversely with how much the reservoir already holds.

## Vocab-scaling of the CLEAN selective-specific controls (V=300→600→1000, the airtight controls, NOT bigram-confounded)

| V | d≥10 sel<random-gate | d≥10 sel<ordered-bigram | d≥6 sel<no-hold |
|---|---|---|---|
| 300 | +0.322 (6/6) | +0.170 (6/6) | +0.333 (6/6) |
| 600 | +0.359 (6/6) | +0.279 (6/6) | +0.390 (6/6) |
| 1000 | **+0.436 (3/3)** | +0.426 (3/3) | +0.461 (3/3) |

**The selective-specific deep-tail holding (sel<random-gate) GROWS MONOTONICALLY with vocab** (+0.32→+0.36→+0.44) — and this control is NOT bigram-dependent (it compares sel to its own random-gate control), so it is NOT confounded by the bigram-starvation Skeptic B flagged. As the language gets richer (bigger vocab, more deep structure), the genuine input-dependent distal holding the selective gate provides grows. (The aggregate `sel−bigram` also grows with vocab at fixed nt — but that IS partly bigram-starvation; the clean data-axis test is the running data-scaling batch at fixed V=300.)

## CLEAN data-axis test (fixed V=300, bigram WELL-SAMPLED — defeats the bigram-starvation confound Skeptic B flagged)

| nt (V=300) | d≥10 sel<random-gate | d≥10 sel<ordered-bigram | AGG sel−bigram |
|---|---|---|---|
| 5000 | +0.322 (6/6) | +0.170 (6/6) | −0.054 |
| 10000 | +0.332 (3/3) | +0.085 | **−0.069 (0/3)** |
| 20000 | +0.387 (3/3) | +0.096 | **−0.084 (0/3)** |

- **The deep-tail selective-specific holding (`sel<random-gate`) HOLDS/GROWS with data** (+0.32→+0.33→+0.39) — robust, clean.
- **`sel` still beats the ordered bigram at the deepest tail** (+0.09-0.10 at nt=10K/20K), positive but SHRINKING as the bigram gets better-sampled (expected — the honest bigram is a stronger baseline at more data).
- **The AGGREGATE stays bigram-level-to-slightly-worse and does NOT improve with data** (−0.05→−0.07→−0.08, 0/3 at nt=10K/20K). ⇒ the "aggregate beats the bigram" seen at higher V/fixed-nt was BIGRAM STARVATION (exactly Skeptic B's warning); with a well-sampled bigram, the overall model is bigram-level at V=300 regardless of data.

## PROPER-reservoir test (np=500, 6-seed — directly addresses Skeptic B's "weak reservoir" concern)

Skeptic B noted my np=200 reservoir was WEAK (worse than a bigram at V=120). Re-run over a PROPER np=500 reservoir (V=300/nt=5000, 6-seed): **d≥10 `sel<random-gate +0.269 (6/6)`, `sel<ordered-bigram +0.201 (6/6)`, `sel<no-hold +0.269 (6/6)`.** The deep-tail selective-specific holding SURVIVES a properly-sized reservoir — slightly smaller than over the weak np=200 reservoir (+0.322), the EXPECTED clean gradient: **the selective's marginal deep contribution decreases as the reservoir strengthens** (weak np=200 +0.32 → proper np=500 +0.27 → e-prop-TRAINED sibling ~0). This RECONCILES the sibling frozen-coupling's ~0-deep (its reservoir was even stronger) — one honest monotone story: the selective helps MORE when the reservoir holds LESS, and it is genuinely positive over a proper fixed reservoir.

## ⇒ THE HONEST, PROPERLY-CONTROLLED CONCLUSION (the first-class deliverable)

The selective mechanism provides **genuine deep-tail long-range holding** — validated by the airtight controls (`sel<random-gate`, `sel<no-hold`), robust across vocab (grows +0.32→+0.44) AND data (grows +0.32→+0.39), and it still edges the *ordered* bigram at the deepest positions. This is a real, clean, adversarially-survived MECHANISM result. **BUT the overall model is bigram-level at tractable scale (V≤1000)** — the aggregate does NOT beat a well-sampled bigram, and more DATA at fixed small vocab does not change that, because the deep-context tokens are too small a fraction of the corpus at small vocab. **Moving the aggregate (overall fluency) requires the deep-context signal to dominate more tokens — i.e. larger vocab / richer language (the documented ~23.7M-word / V=2000 validated regime), not just more data at small vocab.** That regime is a genuine scale lever (a multi-day run; the small launch-bound reservoir makes GPU no faster than CPU here — the honest compute constraint). ⇒ the deep-tail long-range MECHANISM is the honest overnight deliverable; the aggregate-FLUENCY-at-validated-scale is the named, bounded, scale-gated open frontier.

## Next (running)
- **GPU validated-scale** run (np=500 PROPER reservoir per Skeptic B, V=2000, nt=40000): does the deep-tail selective-specific holding survive a properly-sized reservoir at the non-null-discriminator scale, and does the AGGREGATE beat the bigram there?
- **CPU:** the airtight controls at np=500 (proper reservoir, tractable) + a V=600/V=1000 vocab grid — does the deep-tail selective-specific hold over a strong reservoir + grow with vocab.
- raw `research/findings/raw/_trainedsel_scale/airtight_V300_nt5000_seed*.json`.
