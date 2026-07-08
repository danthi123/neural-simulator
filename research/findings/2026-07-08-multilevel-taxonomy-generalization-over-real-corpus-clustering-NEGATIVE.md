# Multi-level taxonomic generalization over a real-corpus clustering-hierarchy is NEGATIVE (6-seed, honest boundary): a NEVER-TAUGHT sub-category does NOT inherit its super-category's property when the hierarchy is built by clustering co-occurrence-code centroids — the super-grouping is not load-bearing (real ≈ deranged). Co-occurrence gives FLAT categories, not a clean nested is-a hierarchy. The next mechanism is a LEARNED hierarchy (EMERGE-44/45/50). NO `sim/` edit.

**Date:** 2026-07-08
**Runner:** `research/runners/_realcorpus_inheritance_multilevel_derisk.py` (reuse-by-import: breadth discovery + the emergent-cluster k-means + rung-1's inheritance read). numpy-only, offline. NO `sim/` edit.
**Verdict:** NEGATIVE (robust across k settings and 6 seeds) — an honest boundary that maps where the breadth→knowledge arc's single-level inheritance stops.

## What was tested (2-level generalization — the harder test)
The single-level rungs (1–5, all GO) inherit within a flat category: a held-out MEMBER inherits a property taught to its category's OTHER members. This tests 2-LEVEL generalization: build a 2-level taxonomy by hierarchical clustering (fine clusters of the codes → coarse super-clusters of the fine-cluster centroids), teach a SUPER-category property to SOME fine-clusters, HOLD OUT an ENTIRE different fine-cluster of the same super, and test whether the held-out sub-category's members inherit the super-property purely via the super-cluster structure. That is genuine taxonomic generalization: a never-seen sub-category inheriting from its super-category.

## The result — NEGATIVE, robust
| config | real multi-level inherit | deranged (super-labels) | chance |
|---|---|---|---|
| 6-seed (K=512, 20 fine → 5 super) | 0.482 | **0.512** | 0.208 |
| seed-42 k_fine=30/coarse=6 | 0.338 | 0.350 | 0.200 |
| seed-42 k_fine=12/coarse=3 | 0.554 | 0.551 | 0.333 |
| seed-42 k_fine=24/coarse=4 | 0.351 | 0.324 | 0.250 |

**In every configuration, real ≈ deranged (often deranged is HIGHER).** The super-cluster grouping is NOT load-bearing — shuffling the super labels does not degrade the inheritance, so the "inheritance" is not riding a genuine super-category structure. Robust across k-fine/k-coarse and 6 seeds.

## The diagnosis (why — reading the substance)
2-level generalization requires the discovered super-clusters to be semantically COHERENT: sibling fine-clusters within a super must be mutually similar AT THE MEMBER LEVEL, so a never-taught sub-category's members are predictable from its taught siblings. Clustering the fine-cluster CENTROIDS into supers does NOT guarantee that — TinyStories co-occurrence gives FLAT categories (words that co-occur), but NOT a clean nested is-a HIERARCHY (super-categories whose sub-categories mutually resemble each other at the member level). So a held-out fine-cluster's members are as similar to other supers' taught members as to their own super's → real ≈ deranged. Single-level inheritance works precisely because a held-out MEMBER IS similar to its taught siblings; a held-out whole SUB-CATEGORY is not similar to its sibling sub-categories under flat co-occurrence.

## The next mechanism (boundary = next mechanism)
Real taxonomic hierarchy needs the IS-A signal, which flat co-occurrence clustering lacks. The project's lever: a LEARNED hierarchy — **EMERGE-44/45's stacked competitive pooler** (a second pooler layer discovers a super-column BLOCK shared across sub-categories) bound by **EMERGE-50's Földiák temporal-continuity trace** (same-superordinate codons presented in temporal proximity bind to SHARED super-columns). That LEARNS the nested structure from the is-a signal (shared superordinate context / temporal proximity), rather than clustering centroids post-hoc. The de-risk: feed the real-corpus codes through the EMERGE-44/45 stacked pooler and re-test 2-level generalization (honest data note: TinyStories may lack a clean is-a signal at this scale — a taxonomy-richer or is-a-annotated corpus may be needed, mirroring the breadth data-lever).

## What this establishes
An honest boundary: the breadth→knowledge arc's inheritance is SINGLE-LEVEL (a held-out member of a discovered category, GO). Multi-level taxonomic generalization (a never-seen sub-category inheriting from its super) does NOT come free from clustering co-occurrence codes — it needs a learned is-a hierarchy. This maps exactly where flat co-occurrence stops and a learned-hierarchy mechanism is required. Single-level inheritance (rungs 1–5) stands unaffected.

## Files
`research/runners/_realcorpus_inheritance_multilevel_derisk.py`; 6-seed `research/findings/raw/_rc_ml_s*.json`. Prior: the single-level rungs 1–5 (GO); EMERGE-44/45 (stacked pooler), EMERGE-50 (Földiák trace) — the named next mechanism.
