---
type: finding
status: live
date: 2026-06-11
mechanism: cleanup
---

# Cortex Step 3: Storkey CA3 Cleanup Probe — NEGATIVE (Locality Wall)

**Date:** 2026-06-11
**Status:** NEGATIVE — Storkey (local rule) fails on correlated codes; pseudo-inverse (host ceiling) succeeds.
**Diagnosis:** LOCALITY WALL (not a capacity wall)
**Verdict:** 3/3 seeds NEGATIVE, unanimous.
**Next step:** See section below — the brain's biological answer to correlated codes is
*preprocessing* (DG pattern separation), not a stronger local rule. The recommended path is
a continuous-attractor / spiking-phasor approach or a dedicated decorrelation circuit.

---

## TL;DR table (multi-seed mean, 3 seeds 42/43/44)

| Mechanism | flip=0.0 | flip=0.1 | flip=0.2 | flip=0.3 | Label |
|---|---|---|---|---|---|
| **argmax** | 1.000 | 1.000 | 1.000 | 0.903 | IDEALIZATION (god's-eye codebook; host shortcut) |
| **hopfield_vanilla** | 0.038 | 0.065 | 0.073 | 0.068 | collapse reference (≈ chance 0.062) |
| **hopfield_storkey** | 0.150 | 0.143 | 0.132 | 0.142 | **THE probe — local rule; FAILS** |
| **hopfield_pinv** | 1.000 | 1.000 | 1.000 | 1.000 | linear ceiling (host matrix inverse) |
| chance | 0.062 | 0.062 | 0.062 | 0.062 | — |

Storkey is ~2–3× above vanilla (not chance) but remains ~7× below argmax/pinv parity.
The gate (Storkey ≥ argmax − 0.10 = 0.90) is NOT met at any flip fraction.

## Completion test (TEST 2, seed 42)

| keep_frac | argmax_partial | storkey | storkey_edge | pinv |
|---|---|---|---|---|
| 0.50 | 1.000 | 0.205 | −0.795 | 1.000 |
| 0.35 | 1.000 | 0.125 | −0.875 | 1.000 |
| 0.25 | 0.970 | 0.190 | −0.780 | 1.000 |
| 0.15 | 0.895 | 0.180 | −0.715 | 0.995 |

No completion edge at any keep_frac; Storkey is uniformly well below argmax-on-partial.

## Capacity sweep (at flip=0.1, seed 42)

| V stored | argmax | vanilla | storkey | pinv | chance |
|---|---|---|---|---|---|
| 8 | 1.000 | 0.110 | 0.240 | 1.000 | 0.125 |
| 16 | 1.000 | 0.070 | 0.230 | 1.000 | 0.062 |

*(Only 16 unique denoise64 codes are available; V > 16 not testable on this cache.)*

At V=8 Storkey reaches 0.24–0.53 (seed-variable), below the 0.90 gate even at half the vocabulary.
The failure is NOT a counting/capacity issue — pinv achieves 1.000 at all tested V on all seeds.

## Storkey sanity check (random near-orthogonal patterns, N=200)

| Load | P | vanilla | storkey |
|---|---|---|---|
| 0.14N | 28 | 0.983 | 1.000 |
| 0.25N | 50 | 0.133 | **1.000** |
| 0.35N | 70 | 0.000 | 0.717 |

**Implementation is CORRECT.** Storkey comfortably beats vanilla at 0.25N on orthogonal patterns
(the textbook benchmark). The failure is specific to CORRELATED inputs.

## Anti-cheat results

### Anti-cheat 1: lesion (zero recurrent weights)

| | intact | lesioned | chance |
|---|---|---|---|
| mean acc (3 seeds) | 0.172 | **1.000** | 0.062 |
| collapses | **False** (REVERSED) | — | — |

The lesion anti-cheat gives a *reversed* result: zeroed weights produce HIGHER accuracy than
intact Storkey. This is not a test-design flaw; it is a diagnostic: with W=0 the attractor settle
is a no-op (the cue is returned unchanged) and the subsequent cosine-scoring against the bipolar
codebook performs nearest-bipolar-neighbor search — effectively argmax on the binarised codes —
which achieves ~1.000 at 10% noise. Intact Storkey is *actively corrupting* recall by pushing all
inputs into a single dominant attractor (see Root-cause below).

### Anti-cheat 2: shuffle (permuted codebook)

| | true codebook | shuffled | chance |
|---|---|---|---|
| mean acc (3 seeds) | 0.203 | 0.130 | 0.062 |
| drops_to_chance | **True** (all 3 seeds) | — | — |

Shuffle anti-cheat passes: Storkey weights built from a permuted codebook drop from 0.20 to 0.13,
confirming the weights store THESE specific codes (not generic structure). The stored signal is
real, just insufficient and dominated by a single attractor.

---

## Root-cause diagnosis

### Why Storkey collapses on correlated codes

The denoise64 bip patterns have mean inter-pattern cosine **0.61** (between-cosine of the raw
codes is 0.81; binarisation at the median compresses this to 0.61 but the codes remain highly
correlated). Direct trace of the synchronous update (seed 42):

- **14 of 16 stored patterns** all converge to bip[15] ("west") when settled from themselves.
  Only bip[2] ("cat") and bip[15] ("west") are true fixed points.
- After 3–4 synchronous steps, any cue starting near bip[0]–bip[14] ends in the "west" attractor.

This is the **common-mode collapse** that was documented for vanilla Hopfield in the prior probe
(`2026-06-10-cortex-learned-cleanup-derisk-PARTIAL.md`), now in its Storkey form:

- **Vanilla** acquires one enormous common-mode eigenvalue (~10.1 vs next ~0.56); every
  pattern's mean value (the shared common-mode direction) is amplified 18× over the signal.
- **Storkey** spreads the spectrum more evenly (top 5 eigenvalues ~0.69, 0.63, 0.61, 0.56, 0.53;
  no 10× outlier), but the correlated patterns still carve a **dominant spurious attractor** in
  the corner of the hypercube that is most consistent with the common mode of all 16 patterns.
  The local update rule reduces but does NOT eliminate the common-mode basin.

### Locality wall (not a capacity wall)

**Pseudo-inverse achieves 1.000** on all seeds at all noise levels. The pinv rule `W = C(C^TC)^-1 C^T`
is the *exact* solution to the projection operator onto the stored pattern span — it effectively
orthogonalises by inverting the full Gram matrix G = C^TC. That inversion is a **global host
computation** over the entire codebook correlation structure.

Storkey's local rule `ΔW_ij = (1/N)[xi_i xi_j - xi_i h_j - h_i xi_j]` applies a *first-order*
covariance correction using only the current weight-induced field h, but without the full G^{-1}
correction it cannot flatten the correlated landscape into independent basins.

**The wall is locality, not capacity.** The patterns are well within the Storkey capacity for
random patterns (16 ≪ 0.39×512 ≈ 200); the failure is entirely due to the high pairwise
correlation (mean cos=0.61), which the local rule cannot compensate for.

---

## Decision logic (as specified)

> If Storkey collapses toward vanilla (well below argmax) → the local rule does NOT solve
> correlated-code capacity → verdict NEGATIVE → characterize precisely.

- Storkey does NOT match argmax (0.14 vs 1.000 at flip=0.1 multi-seed mean). NEGATIVE.
- pinv SUCCEEDS → it is a LOCALITY WALL (not a capacity wall).
- The local covariance correction is insufficient for codes with mean cosine 0.61.

---

## Recommended next step

The probe has now definitively mapped the boundary: the problem is the **common-mode
structure of the denoise64 correlated codes**. This is precisely what the brain-based DG
pattern separation was designed to remove. Two paths forward:

### Path A (brain-based, recommended): Fix the DG-irreproducibility blocker first, then re-run

The prior probe (`2026-06-10-cortex-DG-CA3-cleanup-NEGATIVE.md`) confirmed that DG pattern
separation CAN decorrelate the codes (D.12 validated: input cosine 0.80 → DG cosine 0.05 at
single seed). The blocker was that the DG spike read is **sub-reproducible** (~15 spikes/600
neurons; same-input cosine ≈ 0.05 — the codes are decorrelated but the read is too noisy to
store reliably).

The resolution is not a different weight rule — it is **denser DG readout** or **longer
integration windows** to accumulate enough spikes for a reproducible DG code. With reproducible
DG codes, vanilla Hebbian on the DG-space codes should work (the ZCA-restored parity in the
prior probe confirmed this). Options:
- Increase n_dg (600→1500) and/or integration window (80→200ms) to improve DG code SNR
- Use a **rate-coded DG average** (mean firing rate over a longer window rather than a single
  80-step binary snapshot) as the stored pattern, giving a real-valued DG code rather than
  the noisy binary one
- Add an explicit **DG population-vector cleanup** (k-WTA thresholding on the DG rate code)
  before storing in the Hebbian CA3 weights

This stays fully brain-based (the DG circuit is already validated and on-bridge), just requires
tuning the readout window/density rather than a new circuit.

### Path B (reference; host shortcut): Use the pinv as the IDEALIZATION replacement

The pseudo-inverse achieves 1.000 on all conditions but requires the full G^{-1} computation —
a host matrix operation over the entire codebook. This is exactly as much a shortcut as argmax.
It is reported here as the ceiling reference only; do NOT use as the production mechanism.

### Path C (deferred, deeper architecture): Continuous-attractor / phasor-interference cleanup

The real cortex likely avoids this problem by never operating in a bipolar/synchronous Hopfield
attractor mode for correlated patterns. Two validated alternatives:
- **RF/FHRR phasor attractor**: the existing `ResonateFireTPAM` operates in phase-space, where
  the interference metric is the mean phase-similarity (a circular quantity immune to the DC
  common mode). The prior de-risk showed this works with ZCA codes; whether it works on raw
  codes is untested (it uses the Hebbian outer product in complex space, so the common-mode
  problem may resurface as a common-phase-angle problem — a separate probe is warranted).
- **Attractor networks in the DG output space** (once DG readout is made reproducible): the
  DG-space codes are approximately orthogonal (between-cosine ≈ 0.05 post-separation); any
  standard Hopfield (even vanilla) would work at V=16 well within the 0.14N capacity limit.

**The recommended action**: run a targeted DG-readout-density probe (a few minutes on CPU) to
find the minimum n_dg / integration-steps that makes DG codes reproducible to within cosine ≥ 0.7
same-input. If that threshold is achievable, the path to brain-based CA3 cleanup via
vanilla Hopfield on DG-space codes is open. That is the cheapest-first next step.

---

## Files

- Runner: `research/runners/cortex_storkey_ca3_cleanup_probe.py`
- Raw multi-seed JSON: `research/findings/raw/_cortex_storkey_ca3_cleanup_multiseed.json`
- Raw per-seed: `research/findings/raw/_cortex_storkey_ca3_cleanup_seed{42,43,44}.json`

## Prior probes in this arc

1. `2026-06-10-cortex-learned-cleanup-derisk-PARTIAL.md` — vanilla Hopfield collapses on
   correlated codes; ZCA (host shortcut) restores parity. PARTIAL.
2. `2026-06-10-cortex-DG-CA3-cleanup-NEGATIVE.md` — spiking DG decorrelates (D.12 passed)
   but the DG spike read is sub-reproducible (same-input cos ≈ 0.05). CA3 training fails.
   NEGATIVE. Recommended this Storkey probe as next step.
3. **This probe** — Storkey fails on correlated codes (locality wall); DG path remains
   the brain-based solution; needs readout improvement. NEGATIVE.
