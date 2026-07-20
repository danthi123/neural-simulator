# gap#4 RUNG 3d — RESULT: all four pre-registered predictions CONFIRMED 6/6 on fresh seeds

**Pre-registration committed at `b2950290` BEFORE this run** (`2026-07-20-gap4-rung3d-PREREGISTRATION-offset-zero.md`).
Seeds **200-205**, never used in any prior gap#4 rung. Nothing below was adjusted after seeing the data.

## Scored against the pre-registration

| prediction | bar | result |
|---|---|---|
| **P1** `l2_peak - plateau_bin == 0 +/- 1` | >= 5/6 | **6/6** (offset exactly +0 every seed) |
| **P2** plateau moves -> read moves 1:1 | >= 5/6 | **6/6** (plateau 7->11, `l2_peak` 7->11) |
| **P3** freezing L2 plasticity collapses read, `dw==0` | 6/6 | **6/6** |
| **P4** removing the L2 plateau collapses read, `dw==0` | 6/6 | **6/6** |

`map_ok=1` on all six; no seed excluded.

⇒ **One-shot local credit COMPOSES ACROSS A LAYER, and the backward window is applied ONCE, at the input layer.**
The mechanism predicted in advance is the mechanism observed.

## The seed question, settled properly

Per-seed outputs are near-identical, which in this project is the signature of the n=1 trap ("a 6-seed GO that was
n=1 repeated"). **It is not that here, and I verified rather than assumed:** building the same net at seeds 200 and
201 in separate processes gives DIFFERENT substrates —

| | seed 200 | seed 201 |
|---|---|---|
| firing-threshold md5 | `a840f639587a` | `d792bd188590` |
| threshold mean | -42.258198 | -42.827675 |
| weight md5 / abs-sum | `5bd36fb9e280` / 4995.8237 | `ec025239200f` / 5000.3574 |

The substrate genuinely varies; the RESULT is invariant to that variation. **That is robustness, not n=1** — the
opposite reading from the one I initially suspected, and the distinction is only visible because the hashes were
checked.

## ⚠️ THE HONEST LIMITATION — correct LOCATION, poor CONTRAST

The pre-registration deliberately said nothing about response MAGNITUDE. Measuring it now (reported as an
observation, NOT as a passed prediction):

| arm | response AT plateau bin | next-best field | ratio |
|---|---|---|---|
| MAIN (plateau bin 7) | 0.19048 | 0.15782 | **1.21x** |
| C2 (plateau bin 11) | 0.20196 | 0.18481 | **1.09x** |

L2's response IS maximal at the plateau bin in both arms — so the read is genuinely selective in the sense of
"argmax lands in the right place". **But the margin over the next-best field is only 1.09-1.21x**, far below the
2x the gate demands.

**This is the SAME signature the layer-1 BTSP work recorded on 2026-07-19: "correct structure, poor contrast"
(potentiation moved 20/20 bins, peak in the right place, contrast too low to localize).** The defect REPRODUCES at
layer 2. That is a genuine cross-layer finding: the contrast limitation is a property of the RULE as configured,
not of any one layer's read-out.

## Why the NO-GO stands — and why re-centring would NOT have rescued it

The original gate is `read_acc >= 0.80 AND selectivity >= 0.80`, with selectivity = fraction of non-target fields
beaten by >= 2x. I refused to re-centre the window on the data that revealed offset 0, because the record already
warns I have mis-centred this metric twice.

**It now turns out that refusal was also immaterial: re-centring would not have produced a GO.** Even scored
against the CORRECT (plateau) reference, the best margin is 1.21x, so the selectivity leg evaluates to 0.00
regardless of where the window is centred. **The gate fails on contrast, not on centring.**

So: the pre-registered rung-3 **NO-GO STANDS**, and it stands for a real reason rather than a metric artifact.

## Net state of gap#4

- **Rung 1 (6-seed GO):** a place field from ONE plateau; eligibility-tau ablation load-bearing.
- **Rung 2 (6-seed GO):** 4 cells, 4 distinct fields, one lap, shared inputs; shuffle control 0.00.
- **Rung 3 (gate NO-GO; mechanism CONFIRMED 6/6 pre-registered on fresh seeds):** credit composes across a layer,
  window applied once, plateau-locked and causally verified — but **contrast-limited**.
- **The blocker is now named and singular: CONTRAST.** Not credit assignment (the credit lands in the right place,
  repeatedly, across two layers), not the read-out (graded read works: 0.92 vs 0.000000 for spikes). The next
  method must raise the contrast of the learned code — which is a research-gate question about the depression
  term, not a sweep.

## Process note

Rung 3c handed me a re-centring that would have converted a filed NO-GO into a GO with a plausible mechanistic
story attached. Pre-registering on fresh seeds instead cost one extra run and produced a strictly better outcome:
the mechanism is now confirmed by a test that could have failed, AND the gate's failure is traced to a real cause
(contrast) that the re-centring would have hidden.
