# The EMERGENT online grow-to-active-context: the HTM word-LM's coincidence pool GROWS from the network's OWN winner co-firing DURING learning (no corpus pre-scan) → HTM 1.0 at 848 grown synapses (0.155% of dense), cell-level, 6-seed GO, both anti-cheats collapse — the emergence-bar close of the vocab-scale mechanism

**Date:** 2026-07-14
**Runner:** `research/runners/_emerge15_online_grow_derisk.py` (`OnlineGrowLearner`). Raw `research/findings/raw/_htmsparse/online_*.json`; launcher `_online_fan.sh`. numpy CPU; NO `sim/` edit.
**Status:** GO (6-seed) — the offline corpus pool (`2026-07-14-grow-to-context-corpus-structured-sparse-pool-GO.md`) proved the TARGET structure; this is its EMERGENT realization: the pool is DISCOVERED from experience, not pre-scanned. n=16/32 scaling in flight.

## Why (the emergence bar)
The offline corpus pool matched dense (HTM 1.0) at a fraction of the synapses — but it PRE-SCANS the corpus token-adjacency to wire the pool (a hand-structured, if data-derived, scaffold). Per the emergence bar, the mechanism must EMERGE from a learning substrate, not be hand-installed. The canonical HTM grow-to-active-context (Hawkins-Ahmad 2016; Poirazi-Mel 2001 activity-dependent structural stabilization) grows a cell's distal synapses to the cells that were ACTIVE when it learned — during learning, from the network's own activity. This finding realizes that.

## The mechanism (`OnlineGrowLearner`)
Start with an EMPTY coincidence pool. Each sentence: (1) select winners on the CURRENT pool; (2) for each prev-winner→cur-winner pair in the winner-chain, GROW a coincidence synapse (CELL-level, winner→winner) if it does not exist; (3) re-inject the grown pool (`inject_explicit_wiring` REPLACES + correctly rebuilds `cp_coincidence_synapse_mask`), preserving permanences; (4) potentiate the winner-chain on the now-grown pool. The structure is DISCOVERED from the winner dynamics — no corpus pre-scan.

**The load-bearing subtlety — subject-specificity BOOTSTRAPS incrementally:** sentence 1 (subject A) grows + potentiates its winner-chain; sentence 2 (subject B) then sees A's cells as "committed" (`_committed_count` reads perm>p_init+0.02) → allocates FRESHER cells for the shared middle → subject-specific SDRs → distinct branch prediction. This is why grow+potentiate must be VISIBLE between sentences (the per-sentence re-inject).

## Result — 6-seed GO + both anti-cheats collapse (n=8)
| arm | HTM (6-seed) | grown synapses (mean) | reading |
|---|---|---|---|
| **online grow (intact)** | **1.000** (6/6) | **848** (0.155% of dense 547,200) | matches dense/offline-corpus, discovered from winner dynamics |
| **dAP-LESION** | **0.000** (6/6) | 512 | no dendritic prediction ⇒ collapse (a DIFFERENT, smaller grown set) |
| **PERMUTED corpus** | **0.083** (6/6, ≈chance 0.125) | 640 | scrambled word order ⇒ collapse (a DIFFERENT grown set) |

**Cell-level grow is even sparser than the offline column-level pool** (848 vs 28,800 window=1 synapses = 34× sparser), because it grows only the specific winner→winner synapses that co-fire (~k_win² per context), not all cells of the co-occurring columns (nE² per pair).

**The grown-synapse COUNT differs across conditions (848 / 512 / 640)** — decisive evidence the grown structure is genuinely DISCOVERED from the specific dynamics (different dynamics ⇒ different grown set), NOT a fixed template. The permute control grows a different set AND collapses accuracy — it isn't a token pre-scan.

## a0 note (why this needed a dedicated build, not a config flip)
The existing `enable_structural_plasticity` (`sim/bridge.py:7567-7660`) is Cline-Haas (2008) STOCHASTIC activity-*biased* synaptogenesis toward a target DENSITY (samples pre/post ∝ firing-EMA, forms ordinary non-coincidence synapses) — the WRONG rule for HTM grow-to-*active-context* (grow exactly prev_winner→post_winner, coincidence-tagged, deterministic). So the online grow is a genuine dedicated build, done runner-side (per the "runner-side first" discipline) via per-sentence re-injection. NO `sim/` edit.

## ⇒ the emergence bar is met for the vocab-scale mechanism
The HTM word-LM's pool-sparsity scale axis now EMERGES from experience: the coincidence pool grows from the network's own winner co-firing during learning, cell-level, HTM 1.0 at 0.155% of the dense synapses, both anti-cheats collapsing, 6-seed. The offline corpus pool (the hand-structured pre-scan) is retired as a de-risk scaffold — the online grow is the real mechanism.

## NEXT
- Vocab-scaling of the online grow (n=16/32/64): does HTM stay 1.0 + the grown count stay sub-quadratic (cell-level ~ contexts·k_win², not vocab²)? [in flight]
- Compose with the multi-segment cell-efficiency axis (Option B of `2026-07-02-onbridge-htm-tm-scaling-multisegment-research-gate.md`) — fixed nE + cell reuse — for truly sub-cubic scaling.
- Port the per-sentence re-inject to a genuine on-substrate structural-plasticity grow step (a guarded, default-off `sim/` grow-coincidence-synapse path) once the runner-side mechanism is fully validated.
