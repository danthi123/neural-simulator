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

## 🎉 RESOLVED — the SNc r−V subtraction MECHANISM is validated (seed 42, lesion-confirmed)

The reproducibility blocker was overcome with the **critic-teacher scaffold** (`--critic-teacher-pa 500`, the endorsed innate-reflex-teaches-a-learned-circuit pattern): it drives the critic during the value-training LEARN window (removed at read-out) so place→value STDP LTP forms even on the weak (non-deterministic) place-code draw. With it (+ CUBLAS_WORKSPACE_CONFIG pinned + read-out θ=12), the critic FIRES (12.2 Hz) + GRADES (22×) + LEARNS (w_near 2.77) on the weak draw, making gate-2e testable.

**The GABA_B subtraction is arithmetic + state-specific (seed 42):**

| `--gate2e-gabab-scale` | predicted (NEAR) | unpredicted (FAR) | gap | state-specific |
|---|---|---|---|---|
| 0.1 | 100.0 | 100.0 | 1.00 | no (too weak) |
| 1.0 | 83.3 | 100.0 | 1.20 | no |
| 1.5 | 71.7 | 100.0 | 1.40 | **YES** |
| **2.0** | **64.2** | 100.0 | **1.56** | **YES** |
| 2.0 + **LESION** (GABA_B zeroed) | **100.0** | 100.0 | **1.00** | **no ✅** |
| 2.0 + shuffle | 63.3 | 100.0 | 1.58 | yes (see note) |

The subtraction is the **Eshel-2015 arithmetic signature**: predicted is shifted DOWN by a constant (36 Hz at scale 2.0) but stays **> 0** (not the all-or-none clamp) — because the GABA_B is proportional to the critic's V-firing (near 12 Hz → subtraction; far 0.56 Hz → none). **The LESION anti-cheat is decisive and HOLDS:** zeroing the GABA_B mask collapses the gap to exactly 1.00 → the subtraction *is* the synaptic GABA_B, not host arithmetic.

**Anti-cheat honesty:** the `--lesion` control (the decisive one) holds. The `--shuffle` control *survives* (gap 1.58) but this is a **test-validity** limitation, not a real failure: `--shuffle` only permutes the cell-sets used for weight *tracking* (gate-2c), it does not perturb the training or the gate-2e firing, so the subtraction is unaffected by construction. The subtraction's value-of-location is **inherited from the critic**, whose LTP *does* break under shuffle (validated in the value-grading arc).

## Multi-seed (teacher 350, scale 2.0, θ=12) — 2/3 state-specific + 3/3 lesion-confirmed

| seed | place diff-cos | critic | gate-2e primary | gate-2e + LESION |
|---|---|---|---|---|
| 42 | 0.138 | fires | pred 60 < unpred 100, gap 1.67 **state-specific** | gap 1.00 ✅ collapses |
| 43 | — | fires | pred 42.5 < unpred 100, gap 2.35 **state-specific** | gap 1.00 ✅ collapses |
| 44 | 0.077 | **silent** (w_near 1.04, DA-gated LTP didn't grow on this draw despite the teacher firing it 7 Hz) | pred 100 = unpred 100, gap 1.00 (no critic → no subtraction) | gap 1.00 ✅ |

**The decisive LESION control holds 3/3** — on every seed, zeroing the GABA_B mask collapses the gap to exactly 1.00, so wherever a gap exists it IS the synaptic GABA_B (not host arithmetic). The **state-specific arithmetic subtraction holds 2/3** — seed 44's critic simply did not learn V on its draw (silent → no differential GABA_B → no gap to measure; the subtraction mechanism is not implicated, the upstream critic-learning is). So the SNc r−V subtraction MECHANISM is validated; the 2/3 (vs 3/3) is the same critic-learning-robustness residual as the value-grading arc (some draws under-learn V).

## Honest residual caveats (multi-seed robustness)

1. **The operating point is critic-rate-dependent.** This validation is on the WEAK teacher-bootstrapped draw (critic 12 Hz, w_near 2.77) at scale 2.0. A STRONG-critic draw (33 Hz, the diagnosis's clamp scenario) would need a LOWER scale. A robust multi-seed gate needs either consistent critics (determinism + a phase-locked teacher giving stronger, more uniform LTP) or a critic-rate-normalized GABA_B. The MECHANISM is validated; the multi-seed operating-point calibration is the remaining engineering.
2. The continuous supra-threshold teacher gives unphased (weaker, w_near ~2.8) LTP vs the phase-locked place volley (~6.4). A sub-threshold phase-locked teacher would be cleaner + more biological.

The diagnosis was correct (the GABA_B is the subtraction; it must be in the arithmetic band, not the clamp) and its load-bearing fix (the reset bug) is verified. The subtraction stays strictly synaptic — the GABA_B does the r−V, no host arithmetic.
