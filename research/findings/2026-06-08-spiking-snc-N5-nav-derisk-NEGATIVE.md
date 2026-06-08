# Fully brain-based reward + dopamine REGRESSES navigation (honest negative)

> ## ⚠️ CORRECTION (2026-06-08, same day): this NEGATIVE is CONFOUNDED by a bridge BUG — do NOT cite as a substrate limit.
> The diagnosis (chasing 137 MB run logs from the frontend work) found the neural runs throw a
> **silent shape-mismatch every step** at `sim/bridge.py:5964`
> (`weight_updates * cp_d1_d2_sign[:actual_nnz]`, shapes `(241047,)` vs `(173888,)`): `cp_d1_d2_sign`
> is allocated once at init to the initial synapse count but the synapse arrays later grow, so it is
> never regrown → the reward-modulated weight update **raises and is silently caught every step it
> runs**, dropping reward-driven plasticity. It hits BOTH conditions but ~11× more in the neural runs
> (377k vs 34k errors) because the continuous N5 reward + tonic SNc dopamine run the reward block far
> more often. So the "regression" is the neural agent's learning being **silently broken**, NOT the
> spiking SNc failing on its own merits. **The spiking-SNc Stage A verdict must be RE-RUN after the
> bug fix.** Also: this bug silently degrades ANY reward-modulated nav run — a broader audit is warranted.
> Fix = size `cp_d1_d2_sign` (and `cp_transmission_gain`, same pattern at `:2150`) to match the synapse
> arrays / regrow on structural growth (protected `sim/` edit, owner byte-review).

**Date:** 2026-06-08
**Status:** ~~NEGATIVE~~ → **CONFOUNDED by a bridge bug (above); re-run pending the fix**
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
