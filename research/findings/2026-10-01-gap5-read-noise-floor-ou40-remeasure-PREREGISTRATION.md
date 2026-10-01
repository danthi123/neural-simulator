---
type: finding
status: live
date: 2026-10-01
lane: memory (learn-through-use, gap#5)
mechanism: re-measure the gap#5 read-noise-floor instrument (K18) at the OU noise it states (40 pA), now that T15 makes
  the Ecker store builder set the field the engine reads; no code change, the registered runner unchanged
seeds: [42, 43, 44, 100, 101, 102]
runner: research/runners/_gap5_read_noise_floor_ltu_derisk.py
builds_on:
  - research/findings/2026-09-17-gap5-read-noise-floor-instrument-lane-is-read-noise-dominated-K18.md
  - docs/drafts/ecker-ou-field/AFFECTED.md
---

# PRE-REGISTRATION: gap#5 read-noise floor re-measured at the stated 40 pA OU noise (T15 follow-up)

Registered by Claude (cloud session) on 2026-10-01, before any run at 40 pA. Runbook RULING `ecker-ou-merge`.

## Why

The 2026-09-17 K18 finding measured the weak-cue read noise of the gap#5 learn-through-use lane. It found:
- `sigma_read_bar` = 0.0745 against `sigma_substrate` = 0.0446;
- `read_SNR` = 0.60, classified READ-NOISE-DOMINATED;
- 18 reads per arm needed to resolve the 0.05 gain bar.

T15 found that the store builder (`_gap5_ecker_adex_ca3_stdp_band_derisk.build_store`) wrote the OU amplitude to
`cfg.ou_noise_sigma_pa`, a field the engine never reads. So that run used the engine default of 100 pA, not the 40 pA
it states. Noise is the quantity this instrument measures, so its classification may be an artifact of 2.5 times the
stated noise. Two weight-lever NO-GOs were re-scoped on it: 2026-09-16 forward-band and 2026-08-27 reverse-edge.

## Protocol (unchanged except the noise that now reaches the engine)

- Revision: a main revision that carries the T15 rename. Precondition P0, checked before scoring: `git grep -n
  "ou_std_current_pA" <rev> -- research/runners/_gap5_ecker_adex_ca3_stdp_band_derisk.py` shows the builder setting the
  real field. P0 fails: UNDEFINED.
- Smoke first, one seed, for time and memory:
  `SIM_BACKEND=numpy .venv/bin/python -m research.runners._gap5_read_noise_floor_ltu_derisk --seeds 42
  --n-read-trials 12 --out research/findings/raw/gap5_ecker_adex/read_noise_floor_ou40_s42.json`
- The gate run, six seeds:
  `SIM_BACKEND=numpy .venv/bin/python -m research.runners._gap5_read_noise_floor_ltu_derisk --seeds 42 43 44 100 101
  102 --n-read-trials 12 --out research/findings/raw/gap5_ecker_adex/read_noise_floor_6seed_ou40.json`
- The K18 run took 966 s for six seeds on numpy. Pool lines carry ` ; prereg <this file> ; mem_gb=4 ; max_age_h=72`
  in `--checked`. The smoke's peak memory replaces the `mem_gb` estimate if it is larger.
- Every default is the runner's own, as in K18: `--ou-sigma` 40, `--n-read-trials` 12, `--resolve-bar` 0.05.

## Reading (fixed now; numbers come from the artifact only)

The runner's own instrument-validity verdict is unchanged: graded 6/6, weights frozen 6/6, and `sigma_read > 0` on at
least 5 of 6 seeds (anti-cheat).

| outcome | reading | next |
|---|---|---|
| P0 holds, instrument VALID, `classification` READ-NOISE-DOMINATED (`read_SNR` < 1) | K18 holds at the stated noise. The weight-lever NO-GOs stay re-scoped as under-sampled. | the lane's next LTU test averages `K_reads_needed` reads per arm, as K18 says |
| P0 holds, instrument VALID, `read_SNR` >= 1 | K18's classification was a property of 100 pA, not of the store at 40 pA | re-run the 2026-09-16 forward-band NO-GO at 40 pA, six seeds, before any new weight lever (Claude registers it) |
| P0 fails, or the instrument is not VALID | UNDEFINED, never a verdict on the lane | NEEDS-CLAUDE with the artifact path |

Reported beside the reading, never gated: `sigma_read_bar`, `sigma_substrate`, `read_SNR`, `MDG` and `K_reads_needed` at
40 pA, next to K18's values at 100 pA (0.0745, 0.0446, 0.60, 0.2107, 18).

## Prediction

Lower OU noise should lower `sigma_read_bar` more than `sigma_substrate`, so `read_SNR` should rise. Whether it
crosses 1 is not predicted.

## What it cannot show

It says nothing about any weight lever's mechanism. It measures how many reads a lever test needs at the stated
noise. One store size and one cue operating point are used, the same as K18.
