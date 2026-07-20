# gap#4 RUNG 8 — RESULT: the adjacent-contrast deficit is GEOMETRY-DETERMINED, and even spacing hid that

Pre-registered at `c54195a1` with the interpretation for all three outcomes fixed in advance. Seeds 1800-1805,
per-seed Poisson field placement (Rich 2014), rule OFF (plain BTSP — the valid, well-characterized arm).

## All three predictions PASS

| prediction | bar | result |
|---|---|---|
| **P1** `c_adj` varies across seeds | std > 0.05 | **std = 0.329 — PASS** |
| **P2** >= 1/6 seeds reach 1.60x | >= 1 | **1802 gives 1.902 — PASS** |
| **P3** measurability retained | `map_ok` >= 4/6 | **5/6 — PASS** |

| seed | field centres | `c_adj` |
|---|---|---|
| 1800 | [10, 13, 17, 19] | 1.105 |
| 1801 | [1, 9, 13, 16] | 1.242 |
| **1802** | **[4, 13, 15, 18]** | **1.902** |
| 1803 | [8, 13, 15, 19] | 0.965 |
| 1805 | [6, 10, 13, 17] | 1.122 |

**Against the even-spacing baseline of 1.213 IDENTICALLY on every seed across FIVE independent runs (std = 0.000).**

## The precise finding — and it is NOT "even spacing caused the deficit"

The mean under Poisson placement is **1.267**, against even spacing's **1.213**. Those are barely different.
**So even spacing was not a pathological geometry — it sits near the middle of the distribution.**

What even spacing WAS is **artificially deterministic**. It pinned `c_adj` to a single value and thereby concealed
two facts:

1. **Contrast is GEOMETRY-DETERMINED**, ranging 0.965 to 1.902 — nearly a factor of two — purely from where the
   fields happen to land.
2. **Favourable geometries clear the bar unaided.** Seed 1802 reaches **1.902** with `dw = 5.529` — a *tiny* weight
   change. Good contrast there is essentially free, requiring no mechanism at all.

⇒ **Eight mechanisms were tuned against a single arbitrary point in a distribution, and treated its value as a
property of the rule.** The deficit is real at that point, but it is not a fixed property of BTSP — it is a
property of the layout, and the layout had no empirical basis.

## What this does and does not overturn

**Does NOT overturn:** the measurement itself. `c_adj = 1.213 / c_far = 2.609` under even spacing is correct and
reproducible. The seven separation mechanisms genuinely failed. PF-5's fixed point genuinely holds.

**DOES overturn:** the framing that adjacent-contrast is a deficit *of the rule* to be fixed *by a better rule*.
It is substantially set by the task's geometry — and the geometry I used was inherited from theory papers that
adopt equal spacing for tractability, not from measurement.

## Honest limitations

- One seed (1804) returned `c_adj = nan` and one (1800) `map_ok = 0`; the statistics are over 5 usable seeds.
- `c_far` is `nan` on several seeds — with 4 randomly-placed cells the "far" set is often degenerate, so the
  far-field leg is not assessable here.
- `min_gap = 2` departs from a true Poisson draw (which permits coincident fields), a concession for measurability
  stated in the code.
- **This does not show any mechanism works.** It shows the target was mis-specified.

## What should have happened, and when

The literature check that produced this was cheap and available from the start. Eight mechanisms were built against
an assumption — evenly-spaced fields — that **no measurement in the record ever supported**, and that a single
question to the primary sources would have flagged. The lesson is not "check the literature"; it is that **the
parameters nobody questions are the ones worth questioning**, and in this arc that was the task geometry rather
than any rule.
