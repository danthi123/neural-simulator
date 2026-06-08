# Rank 2 learned-from-vision navigation GENERALIZES to never-trained goals (3-seed GO indicator)

**Date:** 2026-06-08
**Status:** 3-seed GO **indicator** (6-seed validation queued — not a validated claim yet, per the project's 6-seed rule)
**Runner:** `research/runners/g11_bg_runner.py --goal-schedule generalize2`
**Analyzer:** `research/findings/raw/_rank2_generalize2_analyze.py`
**Raw:** `research/findings/raw/_rank2_generalize2_s{42,43,44}.json`

## The question

Rank 2 ("learned vision circuit") replaced the position-INVARIANT ventral/IT
"what" stream — wrong for navigation, it collapses to a ~6.1 mean-distance floor
(equivalent to a random walk, because it discards *where* the goal is) — with a
**learned-from-vision dorsal "where" map**: the superior-colliculus orienting reflex
*teaches* a plastic `(offset dx, dy) -> action` circuit during an early window, then
the reflex is **weaned off** and the learned circuit must drive navigation alone.

The decisive test of whether this is a genuine position-preserving code (vs. a
memorized per-goal lookup) is **generalization to goals it was never trained on**.

## The test (`generalize2` schedule)

Train (reflex on) by rotating through all **four corners** so the learned offset->action
map covers the full direction space, wean the reflex off at steps 2000-3000, then test on
**three NEW non-corner goals** with the reflex fully OFF:

| phase | steps | goal | role |
|---|---|---|---|
| 0 | 0-700 | (6,6) | TRAIN (reflex on) |
| 1 | 700-1400 | (1,6) | TRAIN |
| 2 | 1400-2100 | (1,1) | TRAIN (wean begins @2000) |
| 3 | 2100-3000 | (6,1) | TRAIN (wean completes @3000) |
| **4** | **3000-4000** | **(4,6) mid-top** | **TEST — NEW goal, reflex OFF** |
| **5** | **4000-5000** | **(1,4) mid-left** | **TEST — NEW goal, reflex OFF** |
| **6** | **5000-6000** | **(6,4) mid-right** | **TEST — NEW goal, reflex OFF** |

Metric = mean of phases 4-6 `final_quarter_mean_distance` (lower = closer to goal).
Reference points: reflex single-goal **precision ceiling ~2.0**; position-invariant
IT-only **floor ~6.1** (cannot navigate, ≈ random walk).

## Result (3 seeds)

| seed | train (0-3) | NEW-goal (4-6) | per-phase [4/5/6] | toward reflex | verdict |
|---|---|---|---|---|---|
| 42 | 3.33 | **3.88** | 3.82 / 4.10 / 3.71 | 54% | GENERALIZES |
| 43 | 4.25 | **4.32** | 3.53 / 5.87 / 3.55 | 43% | GENERALIZES |
| 44 | 3.57 | **3.56** | 3.08 / 4.13 / 3.46 | 62% | GENERALIZES |

**NEW-goal mean = 3.92 ± 0.31, 3/3 generalize (≤4.5).** The circuit lands **53% of the
way** from the IT "cannot-navigate" floor to the reflex precision ceiling — on goals it
was **never trained on**, driving entirely from its own learned weights.

## Interpretation

The learned-from-vision map is **goal-agnostic** (offset -> action), not a memorized
per-goal policy: it transfers to never-seen goals with the teaching reflex removed. That
is the signature of a **position-preserving dorsal "where" code** — exactly the biology
Rank 2 was built to capture (Pouget-Denève population codes; Kawato feedback-error-learning,
where an innate reflex teaches a learned predictive controller and is then weaned).

The one soft phase (seed 43, mid-left 5.87) keeps this **between** the reflex ceiling and the
IT floor, not at the ceiling — the learned circuit is real but lower-precision than the
hard-wired reflex, the honest expected cost of replacing a host shortcut with a learned
neural circuit.

## Honest scope

- **3 seeds = GO indicator, NOT a validated claim.** Per the project rule (3-seed indicators
  are unreliable), the 6-seed extension (seeds 100/101/102) is queued; this doc will be
  updated with the 6-seed result before "generalizes" is asserted as validated.
- This is the **perception** leg of full nav biologization. The **reward + dopamine** leg
  (N5 coordinate-free perceived-approach reward + the spiking-SNc actor-critic dopamine) is
  validated separately by the neural-reward+DA nav de-risk (running alongside this extension).
- Position-preserving means *better than the IT floor and transfers to new goals*, not
  *matches the reflex*. The reflex remains the precision ceiling; the value here is biological
  provenance (learned from vision, generalizes) replacing a host-coded perception shortcut.
