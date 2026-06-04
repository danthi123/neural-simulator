# Vocab scaling (iii): the capacity curve refines the cost model — single-step ops scale free, multi-step (composition-depth) ops don't — 2026-06-04

**One line:** Measured the unified-agent benchmark (numpy agent, with the clause-depth2 fix) at growing vocabulary
(320 → 640 → 1280) at fixed dimension D=2048. The **retrieval/trust core — fact memory (flat), one-attribute,
who/what, abstention — holds at 100% to 4× vocabulary at fixed D**; the **composition-DEPTH categories
(two-attribute AND clauses) degrade** as distractor concepts grow. This **corrects** the prior cost model's claim
that "clause holds to 4× vocab" — the numpy agent's recursive clause decode compounds the larger-codebook
distractor crosstalk and degrades, just like the F=3 resonator.

## The curve (2 seeds, D=2048, frozen test set; larger vocab = more distractor concepts)

| vocab (n/v/a) | flat | 1-attr | 2-attr | clause-d1 | clause-d2 | who | abstain | overall |
|---|---|---|---|---|---|---|---|---|
| 320 (200/60/60) | 100% | 100% | 100% | 100% | 100% | 100% | 100% | **100%** |
| 640 (400/120/120) | 100% | 100% | **0%** | **50%** | **67%** | 100% | 100% | 78% |
| 1280 (800/240/240) | 100% | 100% | **0%** | **0%** | **0%** | 100% | 100% | 67% |

```
HOLDS to 4x at fixed D:  flat, 1-attribute, who-query, abstain
DEGRADES with vocab:     2-attribute (0% @ 640+), clause-depth1 (50% @ 640 -> 0% @ 1280), clause-depth2 (67% -> 0%)
```

## What scales free vs what needs dimension

- **Single-step operations hold at fixed D** — fact retrieval (one cleanup), one-attribute (one F=2 resonator
  factoring), who/what (one cleanup), abstention (one threshold). The right code stays nearest even among 4× more
  distractors; the only cost is the cleanup's compute (already vectorized to a matmul).
- **Multi-step operations compound the larger-codebook crosstalk and degrade** — two-attribute (the F=3 resonator
  needs D ∝ M², so at 120 adjectives it wants D=8192, not 2048 → 0%); clauses (the recursive decode does a cleanup
  at every level, and a 400-800-noun codebook's distractor crosstalk compounds across levels → clause-depth1 and
  -depth2 fall off at 640 and collapse at 1280).

## The correction (honest)

The earlier capacity finding (`2026-06-04-capacity-curve-scaling-cost-model.md`) reported "Memory/retrieval/
who-what/abstention/ONE-attribute/CLAUSE all HOLD at 100% to 4× vocabulary at fixed D=2048" — that measurement was
on the spiking agent's core-benchmark harness and did NOT separate clauses' behavior at the numpy agent's larger
noun codebook. This run shows the numpy agent's **clauses DO degrade at 640+**. So the honest cost model is:

| capability | scales at fixed D? | lever to scale |
|---|---|---|
| fact memory / 1-attribute / who / abstain | **yes**, to ≥4× | none (free at fixed D) |
| two-attribute (F=3 resonator) | no | D ∝ M² (GPU resonator; or sparse block codes) |
| clauses (recursive cleanup) | no | higher D, or bound the per-codebook vocab |

## Production implication

Beyond ~320 concepts on a SINGLE codebook at fixed D, the composition-depth categories need D to grow (expensive:
D∝M² for two-attribute). The documented production scaling route avoids this by **keeping each bridge ≤320
concepts** (where every category is 100%) and scaling by ADDING bridges (the sparse-distributed G.20 multi-bridge
ensemble — 160/320-concept tiers validated). So: per-bridge ≤320 = full-capability; cross-bridge scaling is linear
in bridge count, not D∝M². For a single growing codebook, the two near-term levers are the GPU resonator (two-
attribute) and raising D for clauses; the far lever is sparse block codes.

## Files

- `research/runners/unified_agent_capacity_curve.py` — the numpy capacity-curve runner (reuse-by-import).
- `research/findings/raw/unified_capacity_curve.json` — raw per-category-per-vocab.
