---
type: finding
status: contributing
date: 2026-08-01
mechanism: deep-credit-on-spikes
artifacts:
  - research/findings/raw/gap4/supervised_plateau/gap4_supervised_6seed_aggregate.json
---

# gap#4: DIRECTED error (DFA) on the movable plateau hidden TRAINS it (train 0.81 vs tonic 0.34) but does NOT beat unsupervised sharpening on held-out — 6-seed NULL, it overfits

**One-line verdict:** the companion to the plastic-plateau finding. Having shown an UNSUPERVISED local rule on the
movable plateau hidden beats a frozen reservoir (5/6), the question was whether adding a DIRECTED output error —
the thing that could not move the tonic-pinned hidden — pushes held-out generalization further now that the
hidden is movable. 6-seed answer: **no.** Supervised `deep_credit_share` mean **0.108 vs unsupervised 0.139**,
and the supervised arm beats the unsupervised one on only **1/6** seeds. The directed error's extra fit is
**overfitting** — it does not convert to held-out inheritance. All anti-cheats hold; no `sim/` edit.

Artifact: `research/findings/raw/gap4/supervised_plateau/gap4_supervised_6seed_aggregate.json` (backend cupy/GPU).

## Two findings in one, both important

**1. The movable-substrate reframe HOLDS for directed credit — the wall genuinely broke.** All three arms fit the
TRAINING set on the plateau hidden — frozen 0.753, unsupervised 0.827, supervised 0.808 — versus the tonic-pinned
hidden where NO credit rule (fixed-DFA, converged microcircuit, or the true gradient) could move past 0.34. So a
directed, transport-free credit signal genuinely **reaches and moves a deep spiking hidden on the production
bridge**, which the located-wall finding said was impossible. That is the real conceptual advance: the blocker
was the non-movable hidden, not the credit rule, and it is confirmed for the supervised case.

**2. But directed error adds NO held-out benefit over unsupervised sharpening (a clean null).** Supervised dcs
0.108 < unsupervised 0.139 (beats 1/6). The extra train-fit (0.808 vs the unsupervised 0.827 — comparable) does
not generalize better; the held-out gain is carried by the label-free covariance sharpening, not the directed
error. Held-out is **capped by the small/coarse task** (k=8 classes, ~24 held-out examples), not by the credit —
the directed error has nothing more to buy where a label-free rule already saturates the task's generalizable
structure.

## Anti-cheats (6-seed)
No-transport verified 6/6 (code + B≠Wᵀ + B immutable + runtime); shuffle-DFA (scramble the error routing across
the batch) degrades below the unsupervised arm on every seed — the error routing is load-bearing, not decorative;
sup-on-permuted-labels collapses to ≈frozen — the directed benefit (such as it is) is label-dependent; rate
reservoir fails (op-point genuine); reproducibility 1.0.

## Next
The residual is now precise: does a **larger / richer task** (more classes, finer held-out, deeper compositional
demand) let the directed error's train-fit convert to a held-out advantage the small task masks? That is the
lever to test whether supervised deep credit on the movable hidden is genuinely null or task-limited. The
parallel Deep Feedback Control arm (research-gate fallback) remains untested. The unsupervised plateau result
(5/6) stands as the best on-bridge deep-credit signal; this narrows what the directed variant adds (train-fit,
not generalization) on THIS task.
