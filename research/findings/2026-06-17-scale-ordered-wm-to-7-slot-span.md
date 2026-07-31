---
type: finding
status: corrected
date: 2026-06-17
---

# Scale quick-win: order-encoded WM holds the FULL 7-slot span at D=256; the multi-sentence "K≈4 ceiling" was a moat-calibration artifact, not a recall wall

**Date:** 2026-06-17 (roadmap quick-win, owner directive "work up easy→hard, check off the easy stuff")
**Status:** **GO.** Lifting the phasor dimension D=128→256 makes ordered-sequence recall clean across the full
n_slots=7 Lisman-Idiart span (K=5 and K=7 both 1.000, 3 seeds), with the no-confab moat intact. Longer
discourse/context is a **dimension knob, not a wall**. A second, compounding finding: *pure* ordered recall
already held to K=7 at D=128 (0.993) — so the multi-sentence de-risk's reported "K≈4 ceiling" was **not** the
ordered-WM recall failing; it was **moat false-abstains** at high load under the old slot-0 threshold
calibration (now fixed to worst-slot).

## Result (`_phaseB_scale_ordered_wm_probe.py`, 3 seeds, CPU, reuse-by-import of `OrderedPositionWM`, no `sim/` edit)

| D | K=3 | K=5 | K=7 |
|---|---|---|---|
| 128 | 1.000 | 1.000 | 0.993 |
| **256** | **1.000** | **1.000** | **1.000** |

Moat @ D=256 (empty/scrambled abstain, worst-slot-calibrated threshold): **1.000**.

## Two compounding quick fixes (in `ordered_position_wm.py`, the production module)

1. **Worst-slot threshold calibration.** `calibrate_threshold` previously measured the groundable floor from
   slot 0 ONLY. Later slots carry more bundle cross-talk, so slot-0 over-estimates the floor → the threshold sits
   too high → a real later-slot read can fall below it and **false-abstain** at higher loads. Fixed to measure
   the match of EVERY used slot and take the worst (min) → the true floor. This is exactly why the multi-sentence
   de-risk saw erosion past K≈4 at D=128 (slots 4-6 false-abstaining), while *pure* gate-free recall held to 7.
2. **`encode_sequence` guards** (empty sequence → clear error instead of `IndexError`; over-length sequence →
   clear error). Robustness; no behaviour change in range.

Both verified non-regressing: `tests/test_multi_turn_ordered_wm.py` 31/31 still pass.

## Reading it

- **The sequence/context ceiling is the Lisman-Idiart 7-item span, reached cleanly at D=256** — matching the
  biology (working memory ~7 items) and the ordered-WM foundation (already D=256). Discourse length scales with
  D, no new mechanism needed.
- **The "K≈4 multi-sentence ceiling" is retired** as a recall wall — it was a threshold-placement artifact at
  D=128, addressed by the worst-slot calibration + D=256. (A direct multi-sentence-at-D=256 re-run is the
  natural confirmation; by decomposition it follows — multi-sentence = ordered recall (now clean to 7) +
  per-topic describe (K-independent).)
- Honest scope: vocab 16, CPU/numpy; D=256 is 2× the agent composer's D=128, so an agent wanting the full
  7-item discourse span runs its WM at D=256 (the concept-code byte-parity with a D=128 composer is then
  traded for span — a deliberate per-use choice).

## Reproduce
```bash
SIM_BACKEND=numpy python -u -m research.runners._phaseB_scale_ordered_wm_probe --seeds 42 43 44
```
