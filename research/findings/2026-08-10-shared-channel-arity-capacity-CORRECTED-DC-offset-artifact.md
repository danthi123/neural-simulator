---
title: "CORRECTED (supersedes the retracted M*~sqrt(d) finding): the shared-channel 'capacity break' was a removable readout DC-offset artifact — with common-mode removal, shared-channel neural superposition composes zero-shot at 1.00 through arity 6 (N=729) at d=8 and d=16; NO capacity limit found in range"
date: 2026-08-10
type: finding
status: contributing
lane: composer
seeds: [42, 43, 44]
supersedes: research/findings/2026-08-10-shared-channel-arity-capacity-located-M-star-grows-with-dimension.md
seed-waiver: 3-seed x {d=8, d=16} x M in {2..6}. The result is a clean 3/3 (corrected recall 1.00 nearly everywhere; the confound demonstrated on every seed), and the deliverable is a CORRECTION + an upper bound (no break in range), not a fitted constant — more seeds only tighten an already-unanimous ceiling.
---

# The shared-channel "capacity break" was a readout DC-offset artifact — corrected, superposition is far more robust

## What this corrects (and how it was caught)

<!--derived-->

The retracted finding (`2026-08-10-shared-channel-arity-capacity-located-...`, commit `353f2e64`) claimed a
bundling-capacity limit M* that "grows ~sqrt(d)", located by a disjoint no-crosstalk control. **An adversarial
verification workflow caught it the same day**: the shared-channel collapse was a **removable readout DC-offset
(common-mode) artifact, NOT bundling crosstalk.** A skeptic re-ran an isolating probe; I reproduced it independently
(seed44/M=3/d=16: as-is recall 0.20 -> 1.00 after one label-free constant-offset removal; |C|=8.2 > inter-prototype
spacing 7.25). This finding is the corrected measurement.

## The mechanism of the artifact

<!--derived-->

The shared generator's per-primitive readout is a HEBBIAN RUNNING-MEAN cleanup: `readout_m(v)` = mean over the facts
sharing (family m, value v) of the full d-channel engram. The co-occurring OTHER terms average to their family
centroids (independent of v), so `readout_m(v) = primM[v] + sum_{m'!=m} centroid_{m'}` and `regenerate = proto + C`
with **C = (M-1)*sum_m centroid_m a FACT-INDEPENDENT constant**. `|C|` grows ~`(M-1)*sqrt(M)` with arity while the
inter-prototype spacing does NOT, so once `|C|` exceeds ~half the spacing (here by M>=3) Euclidean nearest-prototype
is dominated by the offset and CRATERS with M -- an artifact that mimics a capacity break exactly. The disjoint
control read 1.00 everywhere only because concatenation accumulates NO such offset -- so it never isolated crosstalk;
it also differed in whether a DC bias piled up. The smoking gun: cosine-to-true stayed ~0.75-0.85 (direction correct)
while biased recall went to ~0 -- the signature of a removable DC offset, not genuine fidelity loss.

## The fix (a biological common-mode removal) and the corrected result

<!--derived-->

Remove the common-mode LABEL-FREE (a subtractive inhibitory reference / divisive-normalization analogue): center each
regeneration by the mean TAUGHT regeneration (inductive; NO held-out labels) and the ruler by its own set mean. This
uses no held-out information; the verify's oracle-offset variant gave the identical result, confirming the label-free
estimate captures the whole offset. Runner `research/runners/_teacher_loop_arity_capacity_derisk.py` now reports the
CORRECTED recall as headline and keeps the BIASED metric + the DC-offset ratio as self-documenting witnesses.

**Corrected shared-channel held-out recall (3-seed means), d in {8,16}, K=3:**

| arity M | N | corrected @ d=8 | corrected @ d=16 | biased @ d=8 | DC-ratio @ d=8 | disjoint |
|---|---|---|---|---|---|---|
| 2 | 9 | 1.00 | 1.00 | 0.83 | 0.40 | 1.00 |
| 3 | 27 | 0.93 | 1.00 | 0.07 | 0.96 | 1.00 |
| 4 | 81 | 1.00 | 1.00 | 0.08 | 1.86 | 1.00 |
| 5 | 243 | 1.00 | 1.00 | 0.02 | 2.79 | 1.00 |
| 6 | 729 | 1.00 | 1.00 | 0.02 | 4.06 | 1.00 |

**corrected-M* = None on all 3 seeds at both d** (no break in range); the confound is demonstrated on all 3 seeds
(biased craters while corrected holds; DC-ratio grows monotonically). GO 3/3 at both d.

## The honest reframe (this REOPENS the residual, it does not close it)

<!--derived-->

- **Shared-channel neural superposition is far more capable than the artifact suggested:** with proper common-mode
  removal it composes never-taught combinations zero-shot at ~1.00 through arity 6 (N=729) even at the smallest
  tested dimension d=8. The naive ~1/sqrt(#terms) VSA break did NOT manifest in the tested range.
- **The true capacity limit is NOT located -- it is beyond M=6 at d>=8.** Why the break is so far out: the readout
  reconstructs each primitive accurately (running-mean denoising over the facts sharing it), so the effective SNR is
  high and the reconstruction error stays below the min inter-prototype distance. Locating the real limit needs a
  HARDER regime: much larger arity, a weaker/noisier readout, or genuinely interfering SAME-value codes (not just
  more disjoint families). That is the open next probe.
- **Consequence for the composer arc: it is NOT "map complete."** The capacity edge -- where bundling must hand off
  to binding for same-type composition -- is reopened and currently unmeasured. The four mechanism proof-of-concepts
  (bundle, bind, faithful-spiking-bind, arity-3) survive adversarial verification but are all on disjoint-by-
  construction or idealized-matched-world setups; they show the OPERATOR CLASS works, not naturalistic capacity.

## Housekeeping

<!--derived-->

NEURAL (lesion perturbs regeneration); 0 stored raw patterns; generator never read the ruler; disjoint split +
coverage + no-leakage all M; cfg.seed byte-identical substrate; `git diff main -- sim/` EMPTY (NO sim edit); the
common-mode removal is label-free (taught-only + ruler-set-mean). The dead-constant witness note from the verifier
(`_used_ruler`/`_stored_raw_patterns` set once) applies to the shared generator too and is a separate hygiene item.

Artifacts: `research/findings/raw/teacher_loop_arity_capacity_corrected_d8_AGG.json`,
`research/findings/raw/teacher_loop_arity_capacity_corrected_d16_AGG.json` (+ per-seed). SIM_BACKEND=numpy.
