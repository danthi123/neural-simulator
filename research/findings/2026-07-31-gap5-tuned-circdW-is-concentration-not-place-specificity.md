---
status: live
claim_check: synthesis
lane: gap#5
date: 2026-07-31
---

## 0. Evidence

> ⛔ **CORRECTION found by `claim_check.py` on this very document:** the density-1.0 row originally read
> null **0.5894**, ratio **0.83**. The aggregate computed from the artifacts gives **0.4977** and
> **0.98**. The published table was wrong; it is corrected above. The conclusion is unchanged (still
> not place-specific) but the number was not what the data said.

**Aggregate (the section-1 table is IN this file):** `research/findings/raw/gap5_density/AGG_density_perm.json`
**Per-seed raw, both densities:** `research/findings/raw/gap5_density/g5fix_d025_*.json`, `research/findings/raw/gap5_density/g5fix_d100_*.json`
**Clamp budget aggregate:** `research/findings/raw/gap5_density/AGG_clamp_budget.json`
**Field-quality reference:** `research/findings/raw/gap5_reader/fieldquality_permcheck_cpu.json`

## Derived

Means over seeds where not in the aggregate; all ratios; the SYNTHETIC sigma=5 oracle and the 60-draw power/FPR figures (computed in-probe, stored in no artifact); the 0.2474/0.7050 board quotes (unverified, see the note in section 3b of the laps finding); and config values quoted as context.
<!--derived-->

# ⛔ gap#5: the TUNED `circ_dW` headline is increment CONCENTRATION, not place-specificity — position-shuffling changes it by 1.3%

**Date:** 2026-07-31 · **Status:** MEASURED, 6 seeds × 2 densities, instrument validated in BOTH directions ·
**Touches a number on the board** (the 2026-07-31 06:15 entry) · **Does NOT touch the banked field-quality GO**

---

## 1. The result

At the tuned operating point the board reports as the gap#5 best (`w_max=150, dwell=180, density=0.25`,
`circ_dW 0.7050 = 105% of headline`), the weight change's spatial arrangement is **indistinguishable from a random
permutation of the same increments**:

| condition | observed `circ_dW` | position-shuffled null | ratio | permutation p |
|---|---|---|---|---|
| **measured, density 0.25** (n=6) | 0.6572 | 0.6486 | **1.013** | **0.42** |
| **measured, density 1.0** (n=6) | 0.4877 | 0.4977 | **0.98** | **0.60** |
| **σ=5 ORACLE** (positive control) | 0.8887 | 0.1964 | **4.525** | **0.0025** |

The null holds increment **magnitude and concentration exactly fixed** and shuffles only **position**. Shuffling
positions moves the measured value by **1.3%**; it moves a genuine place field by **4.5×**.

⇒ **`circ_dW` at this operating point is measuring how CONCENTRATED the increments are, not WHERE they are.**
The quantity that was tuned from 0.2474 → 0.7050 is one that a position-blind process reproduces.

## 2. Why this was invisible until now

`circ_resultant` rewards concentration: mass piled on a few place indices gives a high resultant **wherever those
indices are**. The controls in use could not separate the two:

- the **randset** control is structurally weak for a cumulative measure — over 5 laps × 60 positions, the place
  sweep and the random-set drive deliver the **same total mass to every place cell**; randset only scrambles
  contiguity *within a step*. Measured consequence: `treat_circ_dW` and `randset_circ_dW` agree to **seven decimal
  places** (~1e-7) at both densities. It was never going to fail.
- the **legacy `circ`-based control** was outright degenerate (no power at all) — see
  [`2026-07-31-gap5-stepC-control-void-at-small-dW-and-the-fix.md`](2026-07-31-gap5-stepC-control-void-at-small-dW-and-the-fix.md).
  The new degeneracy guard fires on **4/6** runs at density 0.25 and **6/6** at density 1.0.

The board already carried the warning in weaker form — *"Do NOT quote the raw circ 0.846 combination — 59-30% of it
is place-INDEPENDENT concentration."* **At the tuned point it is not 30-59%. It is ~100%.**

## 3. The instrument was validated in BOTH directions before this negative was accepted

<!--derived-->

A negative needs its instrument verified exactly as much as a positive does.

| control | result |
|---|---|
| POSITIVE — σ=5 oracle field | detected, **p = 0.0025** (obs 0.8887 vs null 0.1964) |
| POSITIVE — weak field (amp 5.0, 1.0) | detected, p = 0.0025 (the metric is scale-free, so amplitude is not the issue) |
| NEGATIVE — scattered increments, 60 independent draws | **FPR 0.000**, median null p 0.679 |
| POWER — contiguous increments, 60 draws | **1.000** |
| `lr=0` arm | `circ_dW` **exactly 0.0** — ⛔ **THIS ROW IS WRONG, see below** |

> ⛔ **CORRECTION (same day).** "`circ_dW` exactly 0.0 ⇒ no weight change" is a **false inference**.
> `circ_resultant` clips negatives internally and returns 0.0 when the clipped sum is ≤ 0, so an exact zero means
> **every increment was NEGATIVE** — and that arm's own recorded mean `|dW|` is **21.94**. At this operating point
> the clamp (`w_max=150`) sits **below** the initial weight (`W0=250`), so **97% of the weight change is
> clamp-driven and identical in the `lr=0` control**; the lr lever moves 3%. The zero was rectification, not a
> clean control. Full analysis:
> [`2026-07-31-gap5-tuned-point-is-INSIDE-the-bound-trap-97pct-of-dW-is-the-clamp.md`](2026-07-31-gap5-tuned-point-is-INSIDE-the-bound-trap-97pct-of-dW-is-the-clamp.md).
> This does not change any other row, nor the concentration result — it strengthens it by explaining the cause.

The test has power on realistic place-field structure and does not cry wolf.

## 4. Scope — what this does and does NOT touch

<!--derived-->

- **DOES touch:** the 2026-07-31 06:15 board entry's *"gap#5 field quality: `circ_dW` 0.7050 ± 0.0605 at 6 seeds
  = 105% of the 0.6705 headline, 81% of the σ=5 oracle"*, and the tuning progression
  `0.2474 → 0.3852 → 0.5897 → 0.7050` that produced it. Those numbers are **real as measurements** and
  **misdescribed as place-specificity**. The tuning optimized a concentration statistic.
- **Does NOT touch** the banked **field-quality GO** (`research/findings/raw/gap5_reader/fieldquality_gpu6.json`,
  a different code path): there `circ 0.664` against `randset 0.122` is a **ratio of 5.4**, comparable to the
  oracle's 4.5. **That measurement does show place-specificity and stands.**
- **The tension is informative, not contradictory:** two different operating points. Place-specificity is present
  at the field-quality configuration and absent at the configuration that maximizes `circ_dW`. That is consistent
  with the tuning having walked *away* from place-specificity while walking *up* a concentration metric.

## 5. Consequence

<!--derived-->

`circ_dW` **alone is not a valid gate for place-field formation.** Any gate on it must be accompanied by the
position-shuffled permutation test, which is now computed and stored on every run
(`perm_p_value_median`, `perm_null_p95_circ_dW`) at no extra simulation cost.

**The open question is now well-posed and was not before:** at which operating point does the BTSP write become
place-specific *above a concentration-matched null*? The field-quality configuration is the place to look, since
it is the one with a 5.4× randset ratio.

### ✅ ANSWERED SAME DAY — the field-quality config IS place-specific, 6/6 seeds

Ran the field-quality configuration (`lr=0.002, w_max=2500, laps=1, dwell=30, drive=8000, w0=600,
elig_tau_ms=1000, hetero_dep=0.2, elig_exp=4.0`) through the position-only permutation test:

| configuration | observed | position-shuffled null | ratio | median p |
|---|---|---|---|---|
| **field-quality** (6 seeds) | **0.6511** | **0.1289** | **5.05×** | **0.0025** |
| σ=5 oracle (reference) | 0.8887 | 0.1964 | 4.53× | 0.0025 |
| tuned point (6 seeds) | 0.6572 | 0.6486 | 1.01× | 0.42 |

Per-seed ratios **4.99–5.12×**, every seed at p=0.0025 — no seed carries the result. Saturation is low
(`sat` 0.010–0.025), width 16.3, peaks 4.04.

⇒ **The banked field-quality GO is VINDICATED on a STRICTER control than it originally used.** Its randset null
varies the drive; this one holds magnitude and concentration exactly fixed and varies only position, and the
effect not only survives — its ratio **exceeds the σ=5 oracle's** (5.05× vs 4.53×).

⇒ **And the two configurations are now cleanly separated.** Place-specificity is present at the field-quality
operating point and absent at the one that maximizes `circ_dW`. The tuning that raised `circ_dW`
0.2474 → 0.7050 **walked away from place-specificity while walking up a concentration statistic** — which is
exactly the failure mode this finding names, now demonstrated rather than hypothesized.

**Practical consequence: the field-quality configuration is the correct operating point for gap#5, and the
"tuned" one should not be carried forward.** The higher `circ_dW` at the tuned point is not a better field.

Verification note: running this required fixing three call sites broken by an earlier `run()` return-arity change
(5→6) — including `_gap5_fieldquality_gpu6.py`, the runner that PRODUCES the banked artifact, which had been
unrunnable since that change. All now use a `*_`-tolerant unpack so future diagnostics cannot repeat it. The
banked artifact was backed up and the output path made overridable (`GAP5_FQ_OUT`) before running, so a CPU check
cannot clobber a banked GPU result.

## 6. The transferable lesson

<!--derived-->

**A metric that goes up under tuning is not thereby measuring what its name says.** The number moved 0.2474 →
0.7050 across four steps of honest, controlled tuning, and every step was real — the increments genuinely became
more concentrated. Nothing in that progression could reveal that position had dropped out, because no control in
the loop held concentration fixed and varied position. The permutation null is one line of extra arithmetic on
increments that were already computed.
