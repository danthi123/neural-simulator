---
type: finding
status: qualified
lane: lane-D
date: 2026-07-31
note: "the test ran but is rate-confounded; the matched version is staged"
---

# lane D: both normalization rules degrade every metric — but the comparison is RATE-CONFOUNDED, so this is NOT a clean refutation

**Date:** 2026-07-31 · **Status:** pre-registered test RUN (3 seeds × 3 arms), result **CONFOUNDED by design** ·
the matched test is staged · **do not read this as "subtractive normalization is refuted"**

---

## 1. What was pre-registered, and what came back

## 0. Evidence

**Aggregate (every number in the tables is IN this file):**
`research/findings/raw/laneD_norm/AGG_norm_arms.json`
**Per-seed raw:** `research/findings/raw/laneD_norm/base_*.json`, `research/findings/raw/laneD_norm/meansub_*.json`, `research/findings/raw/laneD_norm/oja_*.json`

## Derived

Ratios to the homeostatic target, percentages, and configuration values quoted as context (homeo_target, hebb_max, drive, dev_steps), plus figures quoted from the prior record.

Pre-registered in [`research/DEFERRED_GPU_WORK.md`](../DEFERRED_GPU_WORK.md) N-1 before any result was in:
*"`HEBB_MEAN_SUB=1.0` raises `|on_minus_off_mean|` above baseline and raises `osi_post_frac` above 0.0104"*, with
the kill criterion *"if `|on_minus_off_mean|` does NOT separate from baseline, the common-mode diagnosis is
refuted in turn."*

Result at the known operating point (init 120 / `hebb_max` 1200 / drive 1200 / `dev_steps` 6000, seeds 42/43/44):

| metric | base | meansub (1.0) | oja (0.01) |
|---|---|---|---|
| `\|on−off\|` | 0.0678 | 0.0164 | 0.0026 |
| `osi_post_frac` | 0.0052 | 0.0026 | 0.0039 |
| `orient_decode` | 0.281 | 0.120 | 0.036 |
| `rsa_vs_host` | 0.827 | 0.473 | 0.339 |
| `l2_mean` (incoming) | 2038 | 837 | 758 |
| `on_mean` | 9.154 | 3.011 | 3.099 |

Every metric moves the WRONG way under both rules. Taken at face value, the kill criterion fires.

## 2. Why taking it at face value would be wrong

<!--derived-->

**Miller-MacKay subtractive normalization makes `sum_j dw_ij = 0` BY CONSTRUCTION.** The baseline rule is
potentiation-only (`dw = lr·(w_max − w)`, always positive), so it continuously ADDS weight mass. That net addition
was what balanced the homeostatic scaling running underneath (`--homeo-target 0.002`). Remove the net addition and
the homeostatic term wins unopposed:

| arm | `v1_firing_rate` | relative to the 0.002 homeo target |
|---|---|---|
| base | 0.00433 | **2.17× (overshoots)** |
| meansub | 0.00117 | **0.58× (undershoots)** |
| oja | 0.00063 | **0.32× (undershoots)** |

**The arms sit on OPPOSITE SIDES of the homeostatic target.** So the comparison is not
"competition vs no competition" — it is also "3× the weight mass and 4–7× the firing rate vs not". A drop in
orientation selectivity is exactly what less firing predicts, independently of whether competition works.

This is the **ONE FLAG ≠ ONE VARIABLE** failure: `hebbian_mean_subtract` changes the fixed-point structure AND the
net weight mass AND the operating firing rate. A single flag, three functional variables. The lever was verified
to be *doing something* (it plainly is), but not to be doing only *one* thing.

**The implementation is NOT at fault** — `sim/bridge.py:7849-7863` correctly subtracts each postsynaptic cell's
mean increment over the coactive set, which is the rule as specified. The confound is in the EXPERIMENT DESIGN,
mine, not in the mechanism.

## 3. What is still validly established

<!--derived-->

- The **raw-weight diagnosis is unchanged and holds in all three arms**: `weight_diagnosis` reads
  COMMON-MODE CONVERGENCE for base, meansub and oja alike. ON and OFF converge; the signed RF cancels.
- **A subset of weights is pinned at the bound in every arm** — `on_absmax` is **exactly 1200.0 = `hebb_max`**
  throughout, while `on_mean` is 3–9. So partial saturation is real even though mean saturation is not; my earlier
  phrasing "nowhere near saturation" was true of the MEAN and false of the MAX, and is corrected here.
- The earlier record's claim that Oja is the strongest `osi_post_frac` lever (0.0104 → 0.0385 → 0.1112) is **not
  reproduced** at this operating point (Oja gives 0.0039 against a 0.0052 baseline). Those numbers came from a
  different configuration and should not be carried forward without re-measurement.

## 4. The matched test (staged)

<!--derived-->

Sweep the mean-subtract STRENGTH rather than switching it fully on. `hebbian_mean_subtract` is a float: partial
subtraction retains part of the net potentiation, so there exists a setting whose firing rate matches baseline.

`HEBB_MEAN_SUB ∈ {0.25, 0.5, 0.75, 0.9}` × seeds 42/43/44, everything else identical.

- **Read the arm whose `v1_firing_rate` is closest to the baseline 0.00433**, and compare `osi_post_frac` and <!--derived-->
  `|on−off|` THERE. That is the rate-matched comparison this test should have been.
- **Prediction:** if competition is genuinely the missing ingredient, the rate-matched arm beats baseline on
  `osi_post_frac`. If OSI tracks firing rate monotonically across the whole sweep and nothing beats baseline at
  matched rate, then competition is refuted *on its own terms* — and that WOULD be the clean negative this run
  was supposed to deliver.

## 5. The transferable lesson

**A pre-registered prediction protects against retrofitting the conclusion; it does not protect against a
confounded design.** I registered the prediction and the kill criterion honestly, ran it, and got a clean-looking
answer in the wrong direction — and the number that exposed the problem (`on_mean` falling 9.15 → 3.01) was sitting
in the same table as the result. A sum-conserving rule cannot lower the mean weight; noticing that it *had* is
what turned a false refutation into a design fix.

**Before comparing two learning rules, check they left the network at the same operating point.**
