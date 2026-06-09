# N9 SNc r−V subtraction de-risk — RESULT: diagnosis-driven fixes land; subtraction validation blocked by critic-learning non-determinism

**Date:** 2026-06-09
**Backend:** CuPy (RTX 3090), deterministic-regime (OU/conductance-noise/global-homeostasis OFF — but NOT cuBLAS/cusparse pinned, which is the blocker below).
**Scope:** the remaining N9 core — the spiking SNc reward-prediction-error δ = r − V realized as a GABA_B subtraction of the learned V from the dopamine cell (gate-2e in `n9_place_graded_critic_stage2_derisk.py`). The critic VALUE half (learns + grades V) is validated separately (`2026-06-09-N9-weighted-coincidence-plateau-RESULT.md`).
**Precedes:** `2026-06-09-N9-SNc-rV-subtraction-research.md` (the read-only diagnosis this executes).

## What landed (the diagnosis-driven fixes — all runner-only, no sim/ edit)

1. **The residual-conductance bug is FIXED (verified).** `_reset_snc_subtraction_state` zeros `cp_conductance_g_gabab` + resets the SNc membrane/recovery at every phase boundary (`_calibrate_da`, `_snc_test`). Before: the FS-gating critic (33–53 Hz) left a huge standing GIRK that suppressed the SNc in the next calibration → SNc tonic read **0.0000**. After: SNc tonic reads **0.0400** (≈40 Hz = the correct IZH2007_DOPAMINE tonic at 180 pA). The diagnosis's failure-mode-2 is resolved.

2. **A calibration sanity check** (`tonic_frac ∈ [0.015, 0.12]`, a per-step fraction at dt=1 ms) that would have caught the bug — now silent (passes at 0.0400).

3. **The gate-2e test toggles** (the diagnosis's two requirements for a state-specific arithmetic subtraction): during the SNc-gap test only (training keeps the validated config so the critic learns), (a) activate the WEIGHTED graded critic so it fires near≫far during the lead → a *differential* GABA_B (the count form is weight-blind → equal GABA_B near+far → no gap), and (b) `--gate2e-gabab-scale` detunes GABA_B toward the Eshel-2015 arithmetic band (the default CLAMPS the SNc all-or-none). Both wrap the `_snc_test` calls.

## The blocker: critic-learning non-determinism (CuPy sparse matvec)

The gate-2e subtraction is only testable when the critic has **learned V** (so it fires near≫far during the lead). But the critic-learning quality **varies between process invocations**:

| invocation | place diff-cos | w_near (seed 42) | critic |
|---|---|---|---|
| earlier (validated 3-seed, n50, moderated) | 0.120 / 0.065 | 6.41 / 8.15 / 4.78 | STRONG (fires 16–33 Hz) |
| recent (v2 / v3 / 3-seed gate-2e) | 0.138 | 1.16 | WEAK (silent, 0.0–0.14 Hz) |

Same seed, same config — the **place-code self-organization draws a different code** between invocations because the de-risk's "deterministic regime" pins OU/conductance-noise/homeostasis but **not** the CuPy sparse-matvec (cusparse atomics) or cuBLAS workspace, which are non-deterministic. Recent invocations consistently drew the weaker code (0.138), and the critic training is too fragile to learn strongly from it → silent critic → gate-2e reads pred = unpred = 100 Hz (the bare reward burst, no differential GABA_B). My edits are all *post*-self-org, so they cannot be the cause — this is a pre-existing reproducibility gap.

## Exact next steps (the two levers)

1. **Pin CuPy determinism** — set `CUBLAS_WORKSPACE_CONFIG=:4096:8` before the CuPy import (the g11 `--deterministic` pattern) and CuPy's deterministic algorithms, so the place-code self-org is reproducible. Then verify whether the pinned draw gives a strong critic; if it's the weak draw, apply lever 2.
2. **Robustify the critic training** so it learns V strongly regardless of the place-code draw (more `--n-train`, a stronger value-teacher, or a place-code-quality gate that re-rolls a weak self-org). The validated runs show strong critics are common — the training just needs to not be fragile to the 0.120-vs-0.138 difference.

Then: on a reproducible strong-critic draw, sweep `--gate2e-gabab-scale` to the arithmetic band (predicted > 0 AND unpredicted > 1.3 × predicted), and run the anti-cheats — `--lesion` (GABA_B-zero → gap → 1.0, the decisive control against host arithmetic) and `--shuffle` (place→location permute → state-specificity breaks). Fallback ladder if the phenomenological detune is seed-fragile: A(detune) ≻ C(FS-WTA bounds the critic output) ≻ B(normal-reversal GABA relay) ≻ D(a Destexhe saturating GABA_B kernel — a protected `sim/` edit, byte-review).

## Honest status

The diagnosis was correct and its load-bearing fix (the reset bug) is verified working. The gate-2e measurement infrastructure is built and correct (the two test-only toggles). The subtraction itself is **not yet validated** — blocked by the critic-learning non-determinism, which is a reproducibility-infrastructure fix (pin determinism + robustify training), not a mechanism question. The subtraction stays strictly synaptic (a host-computed r−V would be a cheat per BRAIN-BASED-ONLY).
