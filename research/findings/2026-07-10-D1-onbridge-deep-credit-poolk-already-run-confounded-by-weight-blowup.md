# D1 on-bridge deep credit — the pool-k population-coding sweep was ALREADY RUN (read-your-own-record) and shows NO clean help, but it is CONFOUNDED by a ~K² weight-scale blowup. The fair (K-normalized) test is the clean closer.

**Date:** 2026-07-10
**Runner:** `research/runners/_semantic_inheritance_onbridge_spiking_derisk.py`. Prior results: `research/findings/raw/_onbridge_kfast_K{1,8,16}_s{42,43,44}.json`, `_onbridge_ksweep_K1_s*.json` (already on disk — I nearly re-ran an answered experiment).
**Verdict:** on-bridge deep multi-layer credit (BDSP/Burstprop/microcircuit) does NOT train at cheap scale, and population coding as-run does NOT fix it — but the as-run pool-k test is confounded by a weight-scale blowup, so it is not yet a clean read-variance test. Fair test launched.

## The prior pool-k result (seed 42; chance 0.278; oracle 1.0, permuted→chance, lesion collapses — all valid)
| arm | K=1 | K=8 | K=16 |
|---|---|---|---|
| plain-FA inherit | 0.241 | 0.167 | 0.148 |
| Burstprop inherit | 0.222 | 0.111 | 0.222 |
| microcircuit inherit | 0.167 | 0.056 | 0.333 |
| **read_snr_corr_mean** | **0.512** | **0.160** | **0.123** |
| **ff_weight_moved_fa** | **899** | **89 258** | **403 507** |
| trains_at_all | False | False | True(barely) |

Across 3 seeds `trains_at_all` is 2/9, uncorrelated with K. So as-run, population coding does NOT help — the deep-credit arms sit at chance at every K, and the microcircuit does not beat plain-FA on spikes (consistent with the 2026-07-07 `does-not-train-at-cheap-scale` 0/6).

## The confound (a0 read of the pooling wiring)
The K neurons per logical unit DO receive identical drive (fair setup, `...derisk.py:272-275`), so pooling *should* reduce read variance and RAISE `corr(pooled E, soma_rate)` with K. Instead corr DROPS (0.512→0.123) and `ff_weight_moved_fa` scales ~**K²** (899→403 507 ≈ K²·const — the K×K dense FF block has K² synapses, each moved by an unnormalized lr). ⇒ at higher K the forward weights BLOW UP → the net saturates/destabilizes → the read degrades → accuracy stays at chance. This is a **weight-scale confound**, NOT a clean demonstration that read-variance is irreducible. The population-coding surpass has therefore NOT been fairly tested.

## The fair test (launched): normalize lr AND ff-w-init by 1/K so the per-postsynaptic-neuron drive + movement are K-invariant
`--lr 0.25/K --ff-w-init 4.5/K` at K∈{1,8,16}, seeds 42/43/44 (like-for-like otherwise). Decisive read: does `corr` now RISE with K (read-variance residual → population coding IS the fix) or stay flat / does accuracy stay at chance (credit-STRUCTURE residual → the microcircuit clean-error channel / learned apical feedback [2026-07-07 D2 rung2 Kolen-Pollack] is the next mechanism)? bv7k56qq7 (the un-normalized 6-seed confirmation at defaults) runs alongside.

## Strategic framing (honest, per the emergence bar)
Deep multi-layer credit on real spikes is a GENUINE open frontier (the field hasn't cleanly solved it either), but it is NOT the blocker for a FIRST emergent language cortex: a fixed recurrent RESERVOIR + a learned SHALLOW read-out (the validated EMERGE-78..85 reservoir arc; Hinaut-Dominey) and the validated stream-cortex rate-Hebbian POPULATION learner both LEARN structure from experience WITHOUT deep credit. So the emergence engine proceeds on the validated shallow/reservoir substrate NOW, while deep-credit-on-spikes is pursued as a parallel frontier (this arc). The `bdsp_w_max`/bursting/transient/silent-pool a0 findings this session stand as the mechanism map; the fair pool-k test + (if flat) learned apical feedback are the next rungs.

## Files
prior `_onbridge_kfast_*`/`_onbridge_ksweep_*` jsons; `_semantic_inheritance_onbridge_spiking_derisk.py`; relates to `2026-07-07-deep-credit-onbridge-spiking-6seed-does-not-train-at-cheap-scale.md`, `-D2-rung2-learned-apical-feedback-lifts-deep-credit-alignment-transport-free.md`.
