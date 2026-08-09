---
title: "Population-scaled normalization RESCUES the read-out collapse at scale (n_pool=1000: 0.13→0.32) but does NOT reach the ceiling — the residual is UNDER-DRIVE, not competition"
date: 2026-08-09
type: finding
status: contributing
lane: world-model-readout
seeds: [42, 43, 44]
---

# Population-scaled normalization rescues the large-reservoir read-out collapse, but the residual is under-drive (PARTIAL)

## Claim

<!--derived-->

Following the capacity sweep (`e249b13d`: the evidence ceiling rises with reservoir size but the FIXED WTA
inhibition collapses at large n_pool), a **neural population-scaled divisive normalization** (Carandini-Heeger,
built runner-side; `cb12ed51f`) was tested. It **partially rescues** the collapse — at n_pool=1000 the neural
read-out rises from the fixed-WTA 0.133 to **0.320** (~2.4×) — confirming population-scaling is a real lever. But
it does **NOT** close the gap to the 0.973 evidence ceiling, and it is seed-variable. Verdict **PARTIAL / NO-GO on
the strict bar.** The decisive residual is **UNDER-DRIVE of the discriminative margin**, not over-competition — so
the next lever is scaling the **drive/excitability UP** with capacity, not adding inhibition.

## Data (`_fm_scaled_norm_readout_derisk`, G=5, 3-seed, de-clamped)

<!--derived-->

| n_pool | scaled-norm read-out | fixed-WTA read-out (prior) | evidence ceiling |
|---|---|---|---|
| 250 | 0.347 | 0.373 | 0.747 |
| 1000 | **0.320** | 0.133 | 0.973 |

Per-seed scaled-norm @ n_pool=1000: 0.56 / 0.16 / 0.24 (high variance). At n_pool=250 scaled-norm ≈ fixed-WTA
(normalization adds nothing when the fixed competition already resolves); at n_pool=1000 it rescues the collapse
but tops out ~0.32 vs the 0.973 ceiling.

## Read

<!--derived-->

- **Population-scaling is a real lever at scale:** 0.133 → 0.320 at n_pool=1000 (the fixed inhibition, calibrated
  at n_pool=250, collapses on the larger population; the scaled pool — driven by real reservoir→pool synapses whose
  count grows with the reservoir — recovers ~2.4×). So the direction identified by the capacity sweep is correct.
- **But it is NOT sufficient.** 0.32 « 0.97 ceiling. The read-out still discards most of the (capacity-improved)
  evidence. This is NOT a winner-op problem (prior finding) and NOT closed by more competition/normalization.
- **The residual = UNDER-DRIVE.** The banked capacity sweep shows ensemble firing *drops* with reservoir size
  (ens_mean_spk 0.0102→0.0069) — the ensembles are starved at large n_pool. Adding inhibition (even population-
  scaled) cannot recover a margin that is under-driven; it can only stop the fixed inhibition from over-suppressing.
  The true companion process we replaced with a constant is a **gain / excitability set-point that scales the
  discriminative drive UP as the population grows** (a homeostatic excitability target, or a divisive-CONDUCTANCE
  normalization that renormalizes gain rather than subtracting current).

## Next lever (precisely scoped)

Scale the ensemble **drive/excitability** with reservoir capacity (per-ensemble gain homeostat toward a target
firing rate, or shunting/conductance divisive normalization that preserves the relative margin) so the
capacity-improved evidence (ceiling 0.97) actually reaches the ensembles — THEN the population-scaled WTA reads a
resolvable margin. The subtractive-current inhibition tested here is the wrong sign for the under-drive residual.

## Artifacts + rigor

Decisive raws (3-seed): `research/findings/raw/fm_scaled_norm_decisive_np250.json`,
`research/findings/raw/fm_scaled_norm_decisive_np1000.json`. Runner + smoke banked `cb12ed51f`. Content path is the
VERBATIM imported neural-WTA `_neural_predict` (grep-clean, winner from cp_firing_states); the normalization is a
real inhibitory pool (reservoir→pool E_TO_E drive, pool→ensembles I_TO_E output), not a host divide; seeded
byte-identical; no `sim/` edit; backend recorded numpy. 3-seed scoping (not a 6-seed headline).

NO-EXTERNAL-NEEDED: read-out lane external round DR-recorded (Carandini-Heeger normalization, Wang-2002); the
under-drive → excitability-set-point lever is the newly-named companion process.
