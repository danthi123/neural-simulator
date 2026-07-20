# gap#4 RUNG 4 — PRE-REGISTRATION: adjacent-band depression (filed BEFORE the run)

**Filed 2026-07-20. No rung-4 result exists at time of writing.** Committed before the runner is launched so the
prediction cannot be adjusted to the outcome — the same discipline that made rung 3d meaningful, and whose absence
is exactly what the adversarial audit punished in rung 1.

## What is being tested

The measured defect: BTSP's learned map has **healthy far-field contrast (2.60x) and a deficient adjacent-field
contrast (1.21x)** — and neighbours are what localize a field. Cause: the committed thresholded gate
`max(theta - E, 0)/theta` depresses LOW-eligibility (= FAR) synapses. Milstein 2021's depression instead fires in a
window BETWEEN two thresholds, i.e. at the lags ADJACENT to the peak.

The fix (built, default-off, byte-identity asserted on both backends at `54cfc3e2`): `band_lo < Etilde < band_hi`.

## The band is DERIVED, not tuned

From the rule (`tau = btsp_elig_tau_ms = 1000 ms`) and the geometry (bin = 200 ms, field spacing = 4 bins), with
`E(D) = exp(-D/tau)`:

| lag | relative eligibility |
|---|---|
| peak (0 fields) | 1.0000 |
| **adjacent (1 field, 800 ms)** | **0.4493** ← depression must fire here |
| far (2 fields, 1600 ms) | 0.2019 ← and not here |

Band edges = geometric midpoints, centring the band on ADJACENT in log-eligibility:
`band_lo = sqrt(far*adj) = 0.3012 * E_max`, `band_hi = sqrt(adj*peak) = 0.6703 * E_max`.
Against the measured layer-1 eligibility range (0.00052 .. 0.02310, on record): **band_lo = 0.006958,
band_hi = 0.015484**. No parameter here was fitted to any contrast outcome.

## PRE-REGISTERED PREDICTIONS

1. **P1 — adjacent contrast rises:** response contrast vs the ADJACENT field goes from 1.21x to **>= 1.60x**
   on >= 5/6 seeds.
2. **P2 — far contrast is not sacrificed:** contrast vs the FAR field stays **>= 2.0x** on >= 5/6 (the fix must
   not simply trade one for the other).
3. **P3 — the trough moves to the adjacent band:** in the weight map, the minimum deviation from baseline occurs
   at a cell **1 field from the peak**, not 2, on >= 5/6.
4. **P4 — band is load-bearing:** with the band OFF the run reproduces the current numbers (adjacent ~1.21x),
   6/6 — i.e. the effect is attributable to the band and not to drift.

**FALSIFIED if** P1 fails (the adjacent-band hypothesis is wrong — the deficit has another cause), or if P2 fails
(the mechanism only redistributes contrast rather than adding it).

## The bar this must clear, stated in advance

Already on record from the compression measurement: **weight contrast 1.73x yields only 1.09-1.21x response
contrast — the transfer function eats ~1.5x.** So P1's 1.60x response target implies roughly **>= 2.5x adjacent
weight contrast**. A mechanism that merely doubles weight contrast will NOT clear this gate. Stated before the run
precisely so it cannot be softened afterwards.

## Seeds

**300-305** — never used in any gap#4 rung (42/43/44, 100/101/102, 200-205 are all now contaminated).

## Honest scope, in advance

- This tests whether adjacent-band depression raises adjacent contrast. It does NOT test whether that suffices to
  pass rung 3's original read/selectivity gate; that is a separate question and will not be claimed from this run.
- `map_ok` must hold; any seed where stage 1 fails is reported as excluded, never silently dropped.
- Both the band-ON and band-OFF arms are run in the same invocation, so the comparison cannot drift across configs.
