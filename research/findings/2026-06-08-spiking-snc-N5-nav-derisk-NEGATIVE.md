# Fully brain-based reward + dopamine REGRESSES navigation (honest negative)

**Date:** 2026-06-08
**Status:** NEGATIVE (honest deliverable per the BRAIN-BASED-ONLY standard) — diagnosis in flight
**Runner:** `research/runners/g11_bg_runner.py` flagship multi-goal (SC reflex + N8 + N6 back-end)
**Analyzer:** `research/findings/raw/_biorda_derisk_analyze.py`
**Raw:** `_biorda_{neural,cheat}_s{42,43,44}.json`

## What was tested

The first FULL-NAV test of the brain-based reward + dopamine stack:
- **neural** = N5 coordinate-free perceived-approach reward + the **spiking-SNc actor-critic
  dopamine** (Stage A, merged `ea42e9ad`: the dopamine RPE is computed by a substantia-nigra
  dopamine pool FIRING, `I_snc = tonic + k_r·max(0,r) − k_v·V`, V = host reward_ema scaffold).
- **cheat** = the host shortcut baseline: coordinate Manhattan-distance reward + raw-scalar dopamine.

Both inside the identical biologized flagship config (`--moving-goal --goal-schedule multi`,
SC orienting reflex, genuine GPi→thal disinhibition, spiking-WTA readout). Metric =
`sum_finalQ` (sum of per-phase final-quarter mean distance; LOWER better). 3 seeds.

## Result — strong regression, 3/3 seeds

| seed | neural (N5+spiking-SNc) | cheat (coord + raw DA) | Δ(neural−cheat) |
|---|---|---|---|
| 42 | 25.70 | 3.64 | +22.06 |
| 43 | 20.40 | 4.02 | +16.38 |
| 44 | 23.36 | 3.85 | +19.51 |
| **mean** | **23.15** | **3.83** | **+19.32** |

The neural reward+dopamine agent scores **~6× worse** than the cheat, and **23.15 ≈ the
non-navigating floor** (~18–27 for this config). So this is not a subtle cost — with brain-based
reward+dopamine as currently realized, the agent **barely navigates at all**.

## Interpretation (honest)

This is exactly the kind of result the owner's BRAIN-BASED-ONLY standard names as the
scientific deliverable: *"an honest negative — the neural version underperforming the host
shortcut — IS the deliverable; it maps what the substrate can/can't do on its own."* The
spiking SNc passes the **isolated** Pavlovian falsifiers (omission dip, acquisition burst-shrink)
but **fails when deployed in the full navigation cascade** — a classic works-in-isolation /
fails-in-the-whole-system gap.

**NOT yet concluded fundamental.** Before calling this a substrate limit it must be diagnosed:
the de-risk combined TWO changes (N5 reward + spiking-SNc dopamine), and the SNc has untuned
gains (`--snc-tonic-pa 220 --snc-reward-gain 400 --snc-value-gain 400`). Candidate causes:
1. **Spiking-SNc dopamine** broken at nav scale (the DA broadcast from SNc firing doesn't track
   r−V well enough to drive corticostriatal plasticity → near-floor). Most likely.
2. **N5 coordinate-free reward** the culprit (less likely — it was 8/8 label-agreement with the
   Manhattan reward on CPU, i.e. behaviorally equivalent in isolation).
3. **Tuning** (tonic/gain) rather than a structural limit.

## Diagnosis in flight (`bx7bv2g5z`, isolation pool, ~24 min)

Two single-change isolations × seeds 42/43/44 (`_bioiso_{n5alone,sncalone}_s{42,43,44}.json`):
- **N5-alone** (`--perceived-approach-reward`, raw DA): if ≈ cheat → N5 is fine; SNc is the regressor.
- **SNc-alone** (`--spiking-snc`, coord reward): if ≈ floor → the spiking SNc is the regressor.

Then (if SNc is the culprit) a tonic/gain sweep to separate tuning from a real limit. The honest
negative stands regardless; the diagnosis localizes it.

## What this does NOT change

The nav PERCEPTION leg is independent and solid: Rank-1 SC orienting reflex (6-seed GO) +
Rank-2 learned-from-vision generalization (3-seed GO indicator, 6-seed pending). This negative
is specifically about the **reward+dopamine** leg being brain-based.
