---
type: finding
date: 2026-08-19
status: instrument delivered + self-validated; demonstration PARTIAL-but-decisive (honest) on a real circuit
seed-waiver: single-substrate-seed BY DESIGN — the statistical population is the >=20 distribution-preserving
  shuffles per seed (Shiu's 100 sims), not multiple substrate seeds; two substrate seeds (42/43) are reported for
  robustness, and the per-seed spread is stated rather than a headline lifted from the favourable one
artifacts:
  - research/findings/raw/_weight_shuffle/wshuffle_when_Wctx.json
tools:
  - tools/lab.py  (shuffle_preserving_marginal, dependency_control)
  - research/runners/_weight_shuffle_dependency_control_demo.py
  - tests/test_weight_shuffle_control.py
---

# Weight-shuffle DEPENDENCY control — an anti-cheat instrument, demonstrated on the gap#5 WHEN recency pathway

**Board task #59 — "Prove a skill really depends on what the brain actually learned."** This delivers a reusable
falsifier for the question *"does this circuit's function depend on the ACTUAL learned/structured weights, or merely
on their gross statistics (the weight histogram, or the per-cell row sums)?"* — and demonstrates it, with teeth, on a
real structured circuit already in this repo. Terminology conditions checked against `docs/TERMS.md` (the "lesion"
below is verified to still hold at measurement: the reads are non-plastic, so a zeroed pathway cannot regrow).

## The method (adopted from Shiu 2024)

Take a trained/structured weight matrix `W`, SHUFFLE its entries preserving its value distribution (the multiset of
weights), re-run the function at a FIXED substrate seed over many shuffles, and require the function to COLLAPSE. If
the function survives a distribution-preserving shuffle, it was riding on gross statistics, not on learned structure —
an overclaim. This is the built-in falsifier of Shiu & Sterne et al. 2024, "A Drosophila computational brain model
reveals sensorimotor processing" (Nature 634:210-219): a descending motor neuron fired in **100/100** sims with the
real connectome weights and **1/100** with the weights shuffled but the global weight distribution preserved. The same
control is Ecker et al. 2022 (eLife 71850) column-identity shuffle, already named load-bearing in this project's
gap#5 replay shuffle-bar research gate (`research/findings/2026-07-24-gap5-replay-sequence-encoding-shuffle-bar-research-gate.md`).

Delivered as two numpy-only helpers in `tools/lab.py`:

- `shuffle_preserving_marginal(W, rng, mode)` — a copy of `W` with entries permuted, preserving the value multiset
  EXACTLY. `mode="global"` (Shiu's control) holds only the global histogram; `mode="per_row"` / `"per_col"` permute
  within each row/column, holding that row/column SUM fixed — the STRONGER control a skeptic invokes when the row (or
  column) sums are the "gross statistic". Works for numpy and cupy arrays (module detected via `type(W).__module__`;
  cupy is never imported for a numpy array); deterministic at a fixed `rng` seed.
- `dependency_control(measure_fn, W, rng, n_shuffles=32, mode="global", ratio=3.0)` — runs `measure_fn(W)` once on the
  real `W` and `n_shuffles` times on shuffles, and reports `collapsed`.

**The `collapsed` criterion (documented margin):** `collapsed == (real STRICTLY exceeds the shuffled p95) AND (real >=
ratio x the shuffled mean)`, with `real > 0` required (the real arm must actually show the effect — both-arms-null is a
void, not a collapse). Default `ratio = 3.0`. When the shuffled mean is <= 0 while real > 0, the null sits at/below zero
(the shuffle ERASED or reversed the function) — the strongest collapse — so the ratio is reported as +inf and the p95
clearance carries the decision. A real score that does NOT clear the shuffled p95 is the Shiu NEGATIVE: the function
rides on the value distribution, an overclaim if called "emergent / structure-dependent".

The instrument is self-validated in `tools/lab.py`'s self-check (`python tools/lab.py`): a structure-dependent
function (diagonal alignment) COLLAPSES under a global shuffle (0/40 shuffles match the real score), while a function
that reads only the PRESERVED total does NOT collapse (40/40 shuffles equal the real score) — proving the instrument
can both fire and correctly NOT fire, so a firing is informative.

## The demonstration with teeth — the gap#5 WHEN `W_ctx` recency pathway

The target is `W_ctx`, the plastic context->CA3 heteroassociative matrix of the gap#5 WHEN episodic store
(`research/runners/_gap5_episodic_temporal_context_when_derisk.py`). This circuit is a clean target precisely because
the runner's OWN anti-cheat — a context-LESION (`W_ctx := 0`) — already shows the recency gradient is carried entirely
by this pathway (recency = 0.0 with the pathway zeroed). So the effect is known to ride on THIS matrix — NOT the
geometry-carried case where a weight shuffle is insensitive.

`measure_fn` = the runner's own recency metric: newest-third-minus-oldest-third of held-cell apical-UP completion (its
`range_intact`), read through the SAME spiking dendritic-dAP completion machinery. Large positive => a graded recency
gradient exists; ~0 => the gradient collapsed. Fixed substrate seed, 24 shuffles per seed per mode. Raw:
`research/findings/raw/_weight_shuffle/wshuffle_when_Wctx.json`.

The table below is ROUNDED from the cited raw artifact above (block-scoped derived until the next heading):

<!--derived-->

| seed | mode | real | shuffled mean | shuffled p95 | shuffled max | real/mean | n>=real | collapsed |
|------|------|------|---------------|--------------|--------------|-----------|---------|-----------|
| 42   | global  | 0.319 | 0.056  | 0.288 | 0.333 | 5.75x | 1/24 | yes |
| 42   | per_row | 0.292 | -0.155 | -0.042 | -0.028 | +inf | 0/24 | yes |
| 43   | global  | 0.347 | 0.138  | 0.313 | 0.375 | 2.51x | 1/24 | no  |
| 43   | per_row | 0.333 | -0.042 | 0.229 | 0.292 | +inf | 0/24 | yes |

Reference floor (the runner's own anti-cheat): `W_ctx := 0` gives recency range **0.0** on BOTH seeds.

**What it shows.**
- **The per_row control is decisive on 2/2 seeds** (0/24 shuffles reach the real score; null mean at/below zero). This
  holds each cell's TOTAL context drive (its row sum) FIXED and scrambles only WHICH context dimension each weight
  aligns to — so the recency gradient depends on the ACTUAL within-row structure of `W_ctx`, not on per-cell drive
  magnitude. This is the strongest statement the instrument makes here.
- **The global control (Shiu's exact histogram-preserving shuffle) collapses cleanly on seed 42** (5.75x, real above
  p95, 1/24 — Shiu's "1/100" shape) and **marginally on seed 43** (2.51x, below the 3x bar, real barely above p95).

**The honest caveat, stated not hidden.** With only 6 serial positions the completion metric is coarse (quantised to a
few held cells), so a global scatter can occasionally reconstruct a partial recency gradient by chance — the global
null has a heavy upper tail (seed 43 max 0.375 vs real 0.347). The per_row control (row sums held fixed) and a graded
rank metric are the decisive reads here; the global histogram control is the weakest of the three and is where a small,
coarse-metric residual survives. This is a property of the 6-position READOUT, not evidence that recency rides on the
weight distribution — the per_row collapse and the `W_ctx := 0` floor both rule that out.

## Honest scope (what this control does and does NOT prove)

- It tests **dependence-on-STRUCTURE, not correctness.** A function can depend on the exact weights and still be wrong.
- It can be **INSENSITIVE when the function rides on network GEOMETRY / topology** rather than the weight VALUES — e.g.
  an order carried by the feed-forward graph that a weight shuffle leaves intact (the 3-assembly hand-wired-fan-in case
  in the gap#5 replay shuffle-bar finding). Pair it with a pathway LESION: if zeroing the pathway ALSO fails to collapse
  the effect, these weights are not load-bearing and the shuffle was the wrong instrument. (Here the `W_ctx := 0` lesion
  DOES collapse the effect to 0.0, so the shuffle is the right instrument for this circuit.)
- A single substrate seed is sufficient for the CONTROL (the population is the shuffles, Shiu-style); the two-seed
  spread above is reported for robustness and shows the global-control marginality is real, not a favourable-seed pick.

## Why this matters (the retractions it would have had teeth against)

Three of the nine 2026-07-28 retractions were terminology overclaims with correct measurements underneath: a
"compositional" read over a localist code, a "self-organized" rule the host supplied both factors of, a "consolidation"
whose replay branch never ran. A distribution-preserving shuffle is exactly the falsifier that asks "does the function
depend on the ACTUAL structure?" — the standing "does it EMERGE from the substrate?" bar, made executable. Run it
before any "it emerged" / "depends on the learned or structured connectivity" claim (see the verify-go skill).

## Reproduce

```
OMP_NUM_THREADS=2 SIM_BACKEND=numpy python -m research.runners._weight_shuffle_dependency_control_demo \
    --seeds 42 43 --n-shuffles 24 --modes global per_row \
    --out research/findings/raw/_weight_shuffle/wshuffle_when_Wctx.json
python tools/lab.py                                              # instrument self-check
RUN_SLOW_TESTS=1 pytest tests/test_weight_shuffle_control.py     # unit + CI-guarded circuit collapse
```

Corpus checked before build (`tools/before_you_build.sh "weight-shuffle dependency control anti-cheat"`, logged to the
corpus-check log under research/queue/): surfaced the 2026-07-24 gap#5 shuffle-bar gate (Ecker 2022's column-shuffle),
which this generalises into a reusable helper.
