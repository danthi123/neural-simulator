# Emergent HTM word-LM vocabulary-scale lever: the high-order advantage SCALES (accuracy) but the DENSE pool blows up; naive random-SPARSE subsampling is a decisive NEGATIVE → the canonical HTM GROW-TO-ACTIVE-CONTEXT (structural plasticity) is the mechanism

**Date:** 2026-07-14
**Runners:** `research/runners/_emerge15_word_sequence_lm_derisk.py` (the word-LM, corpus extended to scale) + `research/runners/_emerge15_sparse_pool_scale_derisk.py` (the sparse-pool de-risk). Raw `research/findings/raw/_htmscale/*` + `_htmsparse/*`. numpy CPU; NO `sim/` edit.
**Status:** the mechanism SCALES in accuracy (GO); the DENSE pool is the scale wall (measured); naive random-sparse is a decisive NEGATIVE that names the next mechanism.

## Why (the emergent language cortex's documented vocabulary-scale frontier)
The emergent HTM Temporal-Memory word-LM (emerge15, `2026-07-02-emerge15-word-sequence-lm-GO.md`) predicts a high-order, context-dependent branch word (the branch depends on the SUBJECT carried through a shared middle — an n-gram is stuck at 1/n_subj). The research gate (`2026-07-02-emergent-sequence-cortex-to-language-research-gate.md`) named the vocabulary-scale bottleneck: the potential pool is DENSE cross-column = O((vocab·nE)²), infeasible at real vocab; the canonical HTM fix (path d) is per-cell sparse distal segments.

## Result 1 — the high-order advantage SCALES in accuracy (reference/ceiling, GO)
| n_subj | HTM branch-acc | best n-gram (=1/n_subj) | advantage | dense synapses |
|---|---|---|---|---|
| 4 | 1.000 | 0.250 | +0.750 | 63,360 |
| 8 | 1.000 | 0.125 | +0.875 | 547,200 |

The HTM stays at 1.000 while the n-gram floor drops as 1/vocab, so the **advantage GROWS with vocab** — the mechanism scales in accuracy. (n≥16 needed a trivial corpus-generator fix — a hardcoded-8-word-list `IndexError`, NOT a substrate wall, caught by a0-reading the crash; the word lists now extend with synthetic tokens.) **The dense pool blows up ~8.6× for a 2× vocab** (63k→547k, since vocab AND nE both grow) — the scale wall the research gate flagged.

## Result 2 — naive random-SPARSE subsampling is a decisive NEGATIVE (the cheap-first rung)
Replacing the dense all-to-all cross-column pool with a SPARSE one (each post cell samples K random pre-cells from other columns; runner-side, NO `sim/` edit), K-swept at n_subj=8 (dense HTM 1.0), 3 seeds each, 18-way fanned:

| K (syn/post) | sparse HTM (3-seed) | sparse synapses | % of dense |
|---|---|---|---|
| 60 | 0.042 | 45,600 | 8.3% |
| 100 | 0.083 | 76,000 | 13.9% |
| 150 | 0.000 | 114,000 | 20.8% |
| 200 | 0.042 | 152,000 | 27.8% |
| 300 | 0.000 | 228,000 | 41.7% |
| 400 | 0.000 | 304,000 | 55.6% |

**NEGATIVE at every K — even K=400 (55% of the dense synapses) gives HTM 0.000.** This is not a coverage tradeoff (bigger K does not help; it is if anything worse). ⇒ RANDOM subsampling of the potential pool breaks the HTM: the winner-selection + allocation (`OnBridgeLearner._match_count`) scores each cell by how many of its synapses connect to the SPECIFIC active-context cells (the prev-winner SDR); a random subsample rarely connects a given post cell to the specific context it must predict from, and the match-count signal degrades → wrong winners → no learned prediction. **Coverage of a fixed-size context by random synapses is the wrong model.**

## ⇒ the mechanism (canonical HTM, cited) — GROW-TO-ACTIVE-CONTEXT, not random subsample
Hawkins-Ahmad 2016 (HTM) + Bouhadjar-Diesmann 2022 (the ported substrate): a cell's distal segments are NOT random — each segment GROWS potential synapses to the cells that were ACTIVE when that segment learned to predict (structural plasticity / activity-dependent synaptogenesis). This makes the pool sparse (a cell connects only to the contexts it has actually seen) AND correct-by-construction (it connects to exactly the active-context cells it must detect). So the scale fix is **grow-to-active-context**: start with an empty/small pool, and on each learning step ADD potential synapses from the just-fired post-winners to the currently-active prev-winners (then the committed permanence rule matures them). Synapses then scale as (distinct contexts seen)×(k_win) = LINEAR in experience, not vocab². This is `enable_structural_plasticity` in spirit (activity-dependent synapse growth), catalog D.18 (the three-term permanence rule already committed) + structural synaptogenesis.

## NEXT (the concrete de-risk)
Build the grow-to-active-context sparse pool: `OnBridgeLearner` grows a synapse (pre_winner → post_winner) when one does not exist and the pair co-fires (capped per post cell), instead of relying on the pre-allocated dense pool. De-risk: does it MATCH the dense pool's HTM branch-accuracy (~1.0) at a SUB-QUADRATIC synapse count, AND stay vocab-independent (K_effective ~ contexts, not vocab²) as n_subj → 16/32/64? Anti-cheats: the n-gram floor, dAP-lesion, the dense-pool parity, permuted-corpus, 6-seed. Likely runner-side first (grow in the learner); a `sim/` structural-plasticity path only if the runner-side proves it.
