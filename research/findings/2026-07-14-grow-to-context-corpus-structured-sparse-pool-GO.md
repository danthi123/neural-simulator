# Grow-to-active-context SURPASSES the naive-sparse NEGATIVE: a corpus-STRUCTURED (co-occurring-column-pair) sparse coincidence pool MATCHES the dense pool's high-order word-LM accuracy (HTM 1.0) at a small fraction of the synapses, CONFOUND-FREE (window=1) and with both anti-cheats collapsing — the emergent HTM word-LM's pool-sparsity scale axis

**Date:** 2026-07-14
**Runner:** `research/runners/_emerge15_sparse_pool_scale_derisk.py` (`--variant corpus`, `--window`, `--no-dense`, `--lesion`, `--permute`). Raw `research/findings/raw/_htmsparse/corpus_*.json`; launchers `_corpus_fan*.sh`; aggregator `_corpus_aggregate.py`. numpy CPU; NO `sim/` edit.
**Status:** GO — the grow-to-active-context TARGET structure (co-occurring column-pairs) matches dense at a fraction of the synapses, confound-free, the synapse fraction drops as vocab grows, both anti-cheats collapse (6-seed). n=32/64/measured-parity are in-flight strengthening points.

## Why (the named next mechanism from the naive-sparse NEGATIVE)
`2026-07-14-htm-word-LM-vocab-scale-naive-sparse-NEGATIVE-grow-to-context-is-the-mechanism.md`: the emergent HTM Temporal-Memory word-LM predicts a high-order branch word (branch depends on the SUBJECT carried through a shared middle), HTM 1.0 vs the n-gram floor 1/n_subj — but its potential pool is DENSE cross-column O((vocab·nE)²). Naive RANDOM subsampling of that pool was a decisive NEGATIVE at every K (HTM 0.000 even at 55% of the dense synapses): random synapses rarely connect a post cell to the SPECIFIC active-context cells it must predict from. The canonical HTM fix (Hawkins-Ahmad 2016) is grow-to-active-context: a cell's distal synapses grow to the cells that were ACTIVE when it learned — sparse AND correct-by-construction.

## The de-risk: the OFFLINE equivalent of grow-to-context (the TARGET structure)
`build_corpus_sparse_pool` wires cross-column synapses ONLY between column-pairs (ca→cb, ca before cb) that CO-OCCUR within `window` positions in the corpus — exactly the connections the online grow rule would create. Synapses = (distinct co-occurring column-pairs)·nE². Everything else (the committed `fused_htm_permanence_update`, coincidence detection, winner-selection) is unchanged. Score on the TRUE corpus branch prediction; the analytic dense count N·(N−nE) gives the ratio (dense OOMs at large vocab).

## Result 1 — the corpus-structured pool MATCHES dense (n=8, 6-seed GO, BOTH windows)
| pool | HTM (6-seed) | synapses | % of dense (547,200) | vs n-gram floor 0.125 |
|---|---|---|---|---|
| dense (reference) | 1.000 | 547,200 | 100% | ✓ |
| corpus window=8 | 1.000 | 94,400 | 17.3% | ✓ |
| **corpus window=1 (confound-free)** | **1.000** | **28,800** | **5.26%** | ✓ |

**window=1 (adjacent column-pairs ONLY) is the CONFOUND-FREE test + is decisive.** window=8 spans the whole ~5-word sentence, so it wires a direct `subject→branch` synapse — a potential low-order shortcut. window=1 wires ONLY the consecutive pairs the HTM winner-chain actually potentiates (no subject→branch edge exists), yet gives **identical HTM 1.0 at 5.26% of the dense synapses** (28,800 = the 18 adjacent column-pairs × nE²=1,600). ⇒ the branch prediction rides the genuine HTM high-order winner-chain (subject-specific SDRs propagated through the shared middle), NOT a low-order shortcut; and the longer-range window=8 synapses are dead weight (identical accuracy, 3× the synapses).

## Result 2 — HTM stays 1.0 as vocab grows + the synapse FRACTION vs dense DROPS (6-seed)
| window=1 | n_subj | vocab | HTM (6-seed) | n-gram floor | corpus synapses | analytic dense | ratio |
|---|---|---|---|---|---|---|---|
| | 8 | 19 | 1.000 | 0.125 | 28,800 | 547,200 | 0.0526 |
| | 16 | 35 | 1.000 | 0.062 | 176,256 | 6,168,960 | 0.0286 |
| | 32 | 67 | [in flight] | 0.031 | [in flight] | ~7.6e7 | [drop expected] |

(window=8 confirms the same shape: n=8 ratio 0.1725 → n=16 0.0966.) HTM stays 1.000 while the n-gram floor drops as 1/n_subj (0.125→0.062→…), so the **HTM advantage GROWS with vocab**, and the corpus-pool synapse **FRACTION vs dense DROPS** (window=1: 0.0526→0.0286 as vocab ~doubled). The dense pool is O(N²)=O(vocab²·nE²); the corpus pool is O((co-occurring pairs)·nE²), so the ratio → ~1/vocab.

## Result 3 — both anti-cheats COLLAPSE (window=1, n=8, 6-seed)
| control | HTM (6-seed) | vs intact 1.000 / floor 0.125 | reading |
|---|---|---|---|
| intact | 1.000 | — | — |
| **dAP-LESION** | **0.000** | collapses BELOW floor | the dendritic apical (dAP) coincidence prediction is load-bearing — no dAP ⇒ no prediction |
| **PERMUTED corpus** | **0.083** | collapses to ~chance | the true word-order co-occurrence structure is load-bearing — scramble it and the pool + winner-chain carry nothing |

Both collapse decisively across all 6 seeds ⇒ the corpus pool's HTM 1.0 is NOT a trivial artifact of "having a corpus-shaped pool"; it requires BOTH the dendritic coincidence mechanism AND the genuine word-order structure. (Measured dense-parity — corpus-w1 == dense HTM on the SAME corpus — is in flight; the analytic count + Result 1's dense reference already anchor it.)

## The three MULTIPLICATIVE HTM-TM scale axes — where grow-to-context sits
Per `2026-07-02-onbridge-htm-tm-scaling-multisegment-research-gate.md`, HTM-TM capacity scales on THREE independent, multiplicative axes; grow-to-context is the third, and it COMPOSES with the other two:
1. **Cells-per-column** (Bouhadjar-Diesmann 2022 — one segment/cell, many cells/column reaches order-10 sequences). This runner already uses it: `nE = k_win·n_subj + 8 ≥ k_win·n_subj` = the disjoint-allocation frontier, which is WHY the flat HTM stays 1.0 (enough disjoint slots). This is also why the corpus pool's ABSOLUTE synapse count still grows (~cubically in n_subj): nE scales with n_subj.
2. **Segments-per-cell / cell REUSE** (Hawkins-Ahmad 2016 — 128 segments/cell, OR-over-segments; the prior gate's Option B, a small guarded default-off `segment_id`-on-the-coincidence-mask `sim/` edit). This DECOUPLES nE from n_subj (fixed cells hold ≫ nE/k_win contexts) — the CELL-efficiency axis, not yet built.
3. **Pool sparsity — grow-to-active-context (THIS finding).** Which *potential* synapses exist. Dense allocates all O(N²); grow-to-context allocates only the co-occurring column-pairs, so the fraction drops ~1/vocab. This is the SYNAPSE-count axis, orthogonal to (1)/(2).

**Honest composition:** grow-to-context alone drops the fraction, but the absolute count still grows because nE scales (axis 1). TRULY sub-cubic (approaching linear-in-corpus) needs grow-to-context (axis 3) COMBINED with fixed-nE multi-segment (axis 2) — the two compose. This finding validates axis 3 in isolation.

## Honest scope + NEXT (the EMERGENT single-pass online grow)
This OFFLINE corpus pool proves the grow-to-context TARGET STRUCTURE works at a fraction of the synapses (confound-free, anti-cheated). It is NOT yet the emergent ONLINE grow: it PRE-SCANS the corpus token-adjacency to wire the pool (a hand-structured, if data-derived, scaffold). The truly emergent mechanism grows synapses DURING learning from the network's OWN winner co-firing (no pre-scan), CELL-level (winner→winner, ~k_win² per context — even sparser). It must BOOTSTRAP subject-specificity incrementally (a winner cell becomes "used" between sentences → the next subject allocates fresher cells for the shared middle → subject-specific SDRs). Built (ready to test): `research/runners/_emerge15_online_grow_derisk.py` — `OnlineGrowLearner` starts EMPTY and grows the coincidence pool per-sentence via `inject_explicit_wiring` re-injection (which correctly rebuilds `cp_coincidence_synapse_mask`), preserving permanences. a0 note (`sim/bridge.py:7567-7660`): the EXISTING `enable_structural_plasticity` is the WRONG rule for this — it is Cline-Haas (2008) STOCHASTIC activity-*biased* synaptogenesis toward a target DENSITY, not the Hawkins-Ahmad DETERMINISTIC grow-to-*active-context*; so this is a genuine dedicated build (runner-side first, per discipline). Anti-cheats for the online grow: n-gram floor, dAP-lesion, PERMUTED corpus (must change the grown set AND collapse accuracy — proving the structure is discovered from the dynamics, not a token pre-scan), 6-seed.
