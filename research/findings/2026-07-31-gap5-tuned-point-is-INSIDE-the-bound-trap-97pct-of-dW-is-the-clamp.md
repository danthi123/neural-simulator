---
type: finding
status: contributing
mechanism: btsp-place-field-formation
lane: gap#5
date: 2026-07-31
---

# ⛔⛔ gap#5: the TUNED operating point sits INSIDE the documented bound trap — 97% of its weight change is the CLAMP, and `lr=0` "reads exactly 0.0" only because of rectification

**Date:** 2026-07-31 · **Status:** MEASURED, 6/6 seeds · **CORRECTS a claim I wrote earlier TODAY** and a
"controls clean" line on the board · **the FIFTH instance of the trap CLAUDE.md already documents for four rules**

---

## 1. The measurement

## 0. Evidence

**Aggregate (every number in the tables is IN this file):**
`research/findings/raw/gap5_density/AGG_clamp_budget.json`
**Per-seed raw:** `research/findings/raw/gap5_density/g5fix_d025_*.json`
**Field-quality reference:** `research/findings/raw/gap5_reader/fieldquality_permcheck_cpu.json`

## Derived

Values computed FROM the artifacts, or quoted as configuration context rather than measured here: percentages, ratios, the 5.05x/4.53x reference ratios, and config settings (w_max, W0, lr, laps).

The tuned operating point uses `--w-max 150`. The runner's initial weight is **`W0 = 250.0`**
(`_gap5_btsp_place_field_derisk.py:39`). **The BTSP clamp sits BELOW the initial weight**, so the clamp drags
every weight down and every increment is negative.

| seed | `lr=0` mean\|dW\| | `lr=0.005` mean\|dW\| | attributable to lr | % identical to the no-learning control |
|---|---|---|---|---|
| 400 | 21.94 | 22.61 | 0.67 | **97.0%** |
| 401 | 25.21 | 26.10 | 0.89 | 96.6% |
| 402 | 23.48 | 23.79 | 0.32 | 98.7% |
| 403 | 25.39 | 26.28 | 0.88 | 96.6% |
| 404 | 25.58 | 26.39 | 0.81 | 96.9% |
| 405 | 23.36 | 24.25 | 0.89 | 96.3% |

**Mean: 97.0% of the weight change at the tuned point is IDENTICAL in the `lr=0` arm. The learning-rate lever
moves 3.0% of it.**

## 2. ⛔ The correction to my own finding, written earlier today

In [`2026-07-31-gap5-tuned-circdW-is-concentration-not-place-specificity.md`](2026-07-31-gap5-tuned-circdW-is-concentration-not-place-specificity.md)
§3 I listed, as instrument validation:

> `lr=0` arm | `circ_dW` **exactly 0.0** — no learning, no weight change

**The inference is WRONG.** `circ_resultant` clips negatives internally
(`w = np.maximum(w, 0.0)`) and returns `0.0` when the clipped sum is ≤ 0. So `circ_dW == 0.0` means **every
increment was negative**, NOT that nothing changed. The same artifact records that arm's mean `|dW|` as **21.94**.

The board carries the same misreading — *"Controls clean: `lr=0` reads EXACTLY 0.0000 at every seed"*. That zero
is **rectification, not cleanliness**. A control that reads 0.0 because its signal was clipped away is not a
validated control; it is an unread one.

**This is the metric hiding destructive weight change**, and it is why `circ_dW` at the tuned point is computed
over only the small positive residual that survives the clamp.

## 3. Why this is the FIFTH instance of an already-documented trap

`CLAUDE.md` carries a standing warning that a plasticity bound set below the design weights does not merely fail
to learn — it destroys weights uniformly, which reads as a substrate limitation. It records four rules hit:
**STDP** (`stdp_w_max`), **BDSP** (`bdsp_w_max`), **BTSP** (`btsp_w_max`, 2026-07-25 — "saturation silently
crushed a rank-1 write to a flat null"), and **Hebbian** (`hebbian_max_weight`). Its stated pre-flight is:

> compare its bound against the ACTUAL weight, and verify the trained pathway moves DIFFERENTLY from an
> untrained control.

**Both halves would have caught this.** `w_max=150` vs `W0=250` fails the first. `lr=0` vs `lr=0.005` differing
by 3% fails the second. The check exists, is written down, and was not run — and the *tuning* walked deeper into
the trap: the board records `w_max` rising 110 → **150** → 220 with 150 selected as the interior optimum. What
was being optimized was how much clamp-driven destruction the metric would reward.

## 4. This explains every gap#5 observation at once

<!--derived-->

| observation | explanation |
|---|---|
| `circ_dW` 0.6572 with a position-shuffled null of 0.6486 (ratio 1.01) | the surviving positive residual is a few synapses; its concentration is arbitrary, its position carries nothing |
| treat and randset agreeing to ~1e-7 | both are dominated by the same clamp, which is drive-independent |
| the legacy `circ` control degenerate in 29/36 arm-runs | same cause |
| `lr` measured "~inert across 16×" (banked as an inert lever) | it **is** inert here — it controls 3% of the weight change |
| tuning 0.2474 → 0.7050 "improving" the field | it tracked clamp depth, not place-specificity |

**And it explains the contrast with the field-quality config**, which uses `w_max=2500` against `w0=600` — the
bound is **above** the weight, no clamp trap — and which scores a **5.05× position-shuffled ratio at p=0.0025,
6/6 seeds**, exceeding the σ=5 oracle's 4.53×. Two configurations, one inside the trap and one outside it, with
exactly the results that predicts.

## 5. Consequences

<!--derived-->

- **The tuned operating point is RETIRED.** It is not a better field; it is a deeper clamp. The field-quality
  configuration (`w_max=2500, w0=600, lr=0.002, laps=1, dwell=30, drive=8000, elig_tau_ms=1000, hetero_dep=0.2, <!--derived-->
  elig_exp=4.0`) is the correct gap#5 operating point.
- **`lr=0` is not a sufficient control when a clamp is active** — it holds the *learning* fixed while the *clamp*
  runs identically in both arms. The control that discriminates is `mean|dW|` per arm, which was already being
  recorded and never read.
- **Any metric built on `circ_resultant` must report the sign budget** (what fraction of `|dW|` is positive), or
  it silently scores the residual of a destructive process.

## 6. The transferable lesson

**A control reading exactly zero is a claim, not a reassurance.** `lr=0 → circ_dW = 0.000000` looked like the
cleanest possible control — an exact zero, reproducible across every seed — and I quoted it as instrument
validation in the same document where I was otherwise arguing that controls must be verified. The number that
refutes it (`dW = 21.94`) was in the same JSON object, one key away.

**When a control reads a perfect zero, ask what operation could produce a zero other than the absence of the
effect.** Here it was a `np.maximum(w, 0.0)` three functions away.
