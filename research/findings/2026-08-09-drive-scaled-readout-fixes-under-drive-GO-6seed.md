---
title: "Drive/excitability scaling FIXES the world-model read-out under-drive — neural read-out lifts 0.18→0.82 at n_pool=1000 (6-seed GO), closing the content-path burn-down"
date: 2026-08-09
type: finding
status: contributing
lane: world-model-readout
seeds: [42, 43, 44, 45, 46, 47]
---

# Homeostatic drive-scaling closes the read-out under-drive: 0.18→0.82 at n_pool=1000 (6-seed GO)

## Claim

<!--derived-->

The world-model read-out arc's precisely-named residual was **UNDER-DRIVE**: at large reservoirs the ensembles
starve (ens_mean_spk drops), so the fixed-WTA read-out collapses (0.40@250 → 0.17@1000) and adding inhibition
(scaled-norm) was the wrong sign. This de-risk scales the ensemble **drive/excitability UP** with capacity via a
NEURAL homeostatic intrinsic-excitability set-point (Desai/Turrigiano): a per-ensemble controller reads each
ensemble's firing from `cp_firing_states` and writes a tonic `cp_external_input_current` toward a target rate.
**Result (6-seed GO):** the neural read-out at n_pool=1000 rises from **0.18 (fixed-WTA collapse) to 0.820**
(per-seed 0.92/0.84/0.76/0.72/0.88/0.80, all 6 well above baselines), **85% of the 0.96 ridge ceiling**, beating
fixed-WTA by +0.64 and scaled-norm (0.24) by +0.58 — and it does it through the named mechanism (ensemble firing
recovers from the 0.0067 starvation toward target). No loss where ensembles aren't starved (n_pool=250: drive
0.387 ≈ fixed 0.367).

## Data (`_fm_drive_scaled_readout_derisk`, 6-seed, in-run baselines)

<!--derived-->

| n_pool | drive_scaled | fixed_wta | scaled_norm | ridge ceiling |
|---|---|---|---|---|
| 1000 | **0.820** | 0.18 | 0.24 | 0.96 |
| 250 | 0.387 | 0.367 | 0.35 | 0.69 |

Raws: `research/findings/raw/fm_drive_scaled_np1000_s42.json` (+ s43..s47 and np250). Runner: `0f346fc6d`.

## Why it's real (anti-cheats, adversarially designed)

<!--derived-->

- **The drive is NEURAL, not a host logit rescale:** the set-point reads `cp_firing_states` and injects tonic
  `cp_external_input_current` (intrinsic excitability) into the ensemble neurons; the winner is argmax over competed
  `cp_firing_states` counts — no logits, no `np.divide`/logit rescale (`_drive_is_neural()`=True).
- **The mechanism does what it claims:** ens_mean_spk @1000 rose 0.0067→0.0091 WITH drive-scaling (it recovered the
  starvation toward target) — the lift is causally the drive, not spurious.
- **Content path clean** (VERBATIM-imported `_neural_predict`, grep-clean of map-matmul/logit-argmax); drive_scaled
  and fixed_wta share IDENTICAL wiring (only the set-point term differs, isolating the variable); wp-lesion→collapse
  (floors carry no per-(s,a) content); untrained control→chance; matched-sham unchanged; two-path==ridge exact;
  ceiling/fixed/scaled-norm ALL measured in-run. cfg.seed byte-identical; no `sim/` edit; backend numpy.

## What this closes + the honest residual

<!--derived-->

**The world-model content-path read-out burn-down is closed:** the last host op (argmax) sits on a fully-neural,
capacity-SCALING read-out (0.82@n_pool=1000, near ceiling) — the winner is read from competed spiking ensembles, the
resolving is neural, and the read-out no longer collapses as the reservoir grows. HONEST GUARD (built in): at small
reservoirs where ensembles are NOT starved, over-driving PAST the operating point hurts — so the controller sweeps
the target and selects on TRAIN, keeping n_pool=250 at ~parity (no loss). Residual: the remaining ~0.14 gap to the
0.96 ridge ceiling (the read-out reaches 0.82, not the full evidence ceiling) — a smaller residual than the
under-drive collapse, and the next read-out refinement rather than a wall.

NO-EXTERNAL-NEEDED: homeostatic intrinsic plasticity (Desai/Turrigiano) + Carandini-Heeger were DR-recorded on the
world-model-readout lane.
