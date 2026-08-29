---
type: finding
status: partial
lane: mouth
board: 80
date: 2026-08-28
mechanism: mouth read-power wall (#80 / gap#4) rank-2 de-risk — variance-weighted / regularized LOCAL read
  (input-channel reliability reweighting) vs the FIXED (uniform, 1:1) gain the batched-substrate read-out uses today
verdict: >
  INDICATIVE ONLY, NOT the pre-registered 6-seed GO. One seed (42), SMOKE scale (B=4, read_window=25, 2-3 repeats,
  k_perm=3), shows the variance-weighted (input-channel-reliability-reweighted) read beating the fixed uniform read
  held-out (delta +0.0518, fixed_recov 0.9326 -> var_recov 0.9844, z=+5.0 vs a permutation null) — but with only 1
  seed and a k_perm=3 null, this is a single positive data point, not a validated result. LOWER-STAKES now: a
  parallel diagnostic found the mouth read is already recovering ~0.85 post-cache-fix (not the deep ~0.37 plateau
  this lever targeted), so the full 6-seed run at decisive scale is parked as a backup lever pending that readout
  run's own verdict, per owner/coordinator direction — not run in this session.
seed-waiver: 1 seed only (42), SMOKE scale — explicitly NOT the pre-registered 6-seed decisive gate. Session was
  cut bounded before the full-scale 6-seed run (already in flight) could finish; killed cleanly, no partial/corrupt
  artifact left running.
artifacts:
  - research/findings/raw/_wkv_mouth_readout_variance_weighted_read/iter2_seed42_smoke.json
  - research/findings/raw/_wkv_mouth_readout_variance_weighted_read/iter2_seed42_smoke.json.prov.json
runner: research/runners/_wkv_mouth_readout_variance_weighted_read_derisk.py
external:
  - Salinas & Abbott, "Vector reconstruction from firing rates", J Comput Neurosci 1:89-107 (1994) — cited via
    research/findings/2026-08-28-mouth-read-power-wall-deep-research-ranked-shortlist.md (rank 2), not re-searched
    here (corpus check via `tools/before_you_build.sh` surfaced that finding directly, no new external round run)
---

# Mouth read-power rank-2 de-risk: variance-weighted read — INDICATIVE (1 seed, smoke scale), not a validated GO

Runs rank 2 of `research/findings/2026-08-28-mouth-read-power-wall-deep-research-ranked-shortlist.md` §3: does a
variance-weighted / regularized LOCAL read beat the fixed mean-difference/1:1 estimator the mouth e-prop
batched-substrate read-out uses today (`_wkv_mouth_readout_eprop_batched_substrate_derisk.py`)? Session was cut
bounded mid-run by owner/coordinator direction (a parallel diagnostic found the mouth read is already at ~0.85
post-cache-fix, not the deep ~0.37 wall this lever targets, so the decisive run is now a backup, not load-bearing)
— reporting the one real result in hand rather than waiting on the in-flight 6-seed run.

## VERIFY-FIRST: the "already-recorded traces" premise is FALSE

Checked `research/findings/raw/_wkv_mouth_readout_snr_ensemble/`, `_mouth_readout_tuning/`, and
`_wkv_mouth_hid_correlation_diagnostic/` — all hold aggregate summary JSON (recov numbers, gain, seed hashes),
**no raw per-(repeat,block,word) margin traces**. A small fixed-weight capture was required (never a retrain —
`head_w`, the checkpoint's own ground truth, is read repeatedly, never gradient-updated).

## Design iteration 1 (documented negative, not silently dropped): per-output-word gain is ill-conditioned here

The first attempt fit an independent gain per OUTPUT WORD channel (v in 0..999) from a handful of repeated
substrate reads of a small TRAIN stimulus batch, empirical-Bayes-shrunk toward a global gain. It collapsed
catastrophically — held-out AND in-sample recov near zero against a ~0.95 fixed baseline — for two compounding
reasons, both now recorded in the runner's own docstring (`_wkv_mouth_readout_variance_weighted_read_derisk.py`):
(1) with only a handful of TRAIN stimuli, most words' `host_ideal[b,v]` is small for every sampled stimulus (a
word is only strongly implicated by whichever stimulus targets it), so a per-word slope is fit from an
ill-conditioned few points, dominated by which stimuli happened to be sampled rather than genuine repeat-noise;
(2) this is a WINNER-TAKE-ALL argmax over V independently-calibrated channels (unlike Salinas & Abbott's
population-VECTOR decode, which combines channels into one continuous estimate) — any single channel's gain
landing near a small value from small-sample noise makes `margin/gain` explode and that channel wins every
argmax regardless of true evidence. A trust-region clip contained the explosion but the estimate underneath was
still built on too little, badly-conditioned data. **This is filed as a real, informative negative**: naive
per-output-channel gain fitting is the wrong level for this read's structure.

## Design iteration 2 (this file's method): reweight the D=128 INPUT channels, not the V=1000 output words

Moves the variance-weighting to the input side, where it is well-conditioned and structurally matches
Salinas-Abbott (many noisy channels combined into ONE downstream estimate, not V independent single-shot
calibrations): a one-hot PROBE SWEEP drives each of the D=128 host-feature dimensions alone (with `head_w`
already set), reads the resulting V-word margin pattern R times, and fits a per-dimension gain + noise variance
pooled over V*R samples (thousands of samples per channel, vs a handful per output-word in iteration 1).
`reliability_d = SNR_d/(SNR_d+c)` (c = TRAIN-only median SNR, fixed before any held-out number is read) builds
`head_w_reweighted[v,d] = head_w[v,d] * reliability_d` — down-weighting noisy input channels uniformly across
every output word. Both arms (fixed `head_w` vs `head_w_reweighted`) get their own pooled least-squares gain
(the SAME formula `_calibrate_gain` uses in production) fit on a disjoint TRAIN stimulus batch, then are scored
ONCE on a disjoint HELD-OUT (different sentences) TEST batch never touched by the probe sweep or either gain fit.
A channel-identity permutation null (K_PERM draws, `reliability_d` scrambled across dimension index) tests
whether the lift is dimension-specific.

## The one result in hand: seed 42, SMOKE scale (not the pre-registered gate)

`research/findings/raw/_wkv_mouth_readout_variance_weighted_read/iter2_seed42_smoke.json` — B=4, read_window=25,
repeats_probe=2, repeats_train/test=3, k_perm=3, n_sentences=4000 (all cut from the pre-registered de-risk
defaults B=8/read_window=45/repeats=3-5/k_perm=20 for a fast sanity check, not the decisive scale):

| | fixed (today's uniform gain) | variance-weighted (input-reliability-reweighted) | delta |
|---|---|---|---|
| held-out recov_argmax | 0.9326 | 0.9844 | **+0.0518** |
| held-out argmax_agree | 0.5833 | 0.75 | +0.1667 <!--derived--> |
| in-sample recov_argmax | 1.0 | 1.0 | 0.0 |

Permutation null (K_PERM=3 draws, channel-identity scrambled): null_mean=+0.0044, null_std=0.0094,
**z=+5.035** — the observed lift clears Z_FLOOR=2.0 comfortably even against this null, and `go_seed=True` on
the module's own pre-registered per-seed gate. Build-twice seed-trap hash confirmed seeded
(`a45d2385f84619f0`==`a45d2385f84619f0`). The weight-matrix lever check (`lever()`) confirms the two arms'
read weights and recov actually differ (not a void A/B): `||head_w||`=37.53 -> `||head_w_reweighted||`=18.94,
mean per-dimension reliability ~0.497 <!--derived--> (`probe_diag.w_d_mean`=0.4969; roughly half the input
channels down-weighted by roughly half on average).

**Why this is NOT a claimed GO:** (1) one seed, not the pre-registered >=5/6 of 42/43/44/100/101/102; (2) smoke
scale (B=4, read_window=25, 2-3 repeats) — a fast sanity/correctness check, not the decisive-scale capture
(B=8, read_window=45, repeats_probe=3, repeats_train/test=5); (3) k_perm=3 gives a very small null sample (the
pre-registered k_perm=20 was not run), so the null's own mean/std are themselves noisy estimates. The GO-gate in
the runner's `go_seed`/`go_5of6` fields is real and will fire honestly on a proper run — it just has not been
given that run yet.

## Why this is now lower-stakes (context that changed mid-session)

A parallel diagnostic (per owner/coordinator update, not independently reverified in this session) found the
mouth substrate read is already recovering ~0.85 post-stale-cache-fix — i.e. the residual this rank-2 lever was
built to close may be the SHALLOW coverage/epochs residual (§5 of the ranked-shortlist finding), not the deep
~0.34-0.37 plateau. If that holds, this lever answers a question that a pure data/compute-budget fix may already
be closing on its own, and the decisive 6-seed run at full scale becomes a backup for the deep plateau rather
than load-bearing for the near-term mouth crutch-burndown. It is NOT superseded as a mechanism — iteration 2's
design is sound and the one seed run is a genuine positive signal — it is de-prioritized.

## What is still open (the next action, when this lever is picked back up)

Run the pre-registered decisive capture (module defaults: B=8, read_window=45, repeats_probe=3,
repeats_train/test=5, k_perm=20; ~5-10 min/seed estimated from the smoke timing, CPU-only, no GPU) via
`SIM_BACKEND=numpy .venv/bin/python -m research.runners._wkv_mouth_readout_variance_weighted_read_derisk
--seeds 42,43,44,100,101,102 --json <a fresh 6seed output path, not yet run -- no artifact exists for it yet>`. GO-gate (pre-registered, unchanged): held-out delta>0 AND z>=2.0 beyond the permutation null
at >=5/6 seeds. A partial timing run for seed 42 at full scale was started this session and killed incomplete
(mid-probe-sweep, no output) when the session was cut bounded — no corrupted artifact was left behind, nothing to
clean up on resume.
