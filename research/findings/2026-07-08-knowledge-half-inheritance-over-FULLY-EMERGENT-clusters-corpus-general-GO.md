# The KNOWLEDGE half of breadth, PROBE-FREE (GO, multi-seed): inheritance rides FULLY-EMERGENT categories — the brain DISCOVERS its categories by clustering its own experience-learned codes (NO hand-labeled probe), and a held-out cluster member inherits its cluster's property. Works on TWO corpora (TinyStories AND WikiText) — corpus-general. The last hand-designed scaffold in the inheritance pipeline is removed. NO `sim/` edit.

**Date:** 2026-07-08
**Runner:** `research/runners/_realcorpus_inheritance_emergent_clusters_derisk.py` (reuse-by-import: the breadth discovery + rung-1's inheritance read + a simple cosine k-means). numpy-only, offline. NO `sim/` edit.
**Verdict:** GO on both corpora — inheritance over categories DISCOVERED by clustering (no a-priori probe), master-directive-aligned (emergent, not hand-designed) and corpus-agnostic.

## Why this ran (removing the last hand-designed scaffold)
Rungs 1–4 defined the categories with the hand-labeled `TAXONOMY_8x8` probe — the last hand-designed piece in the inheritance pipeline (vocab discovery, category structure, and the inheritance itself are all emergent). And that probe is child-concept-specific: on the encyclopedic WikiText corpus it covers 0 words in the top-256 (game/season/song, not animals/toys). The master-directive fix: DISCOVER the categories by clustering the real-corpus co-occurrence codes (k-means, NO labels), and ride inheritance over the fully-emergent clusters — which also removes any probe-to-corpus mismatch.

## The mechanism
Discover the vocab + co-occurrence codes (breadth) → cosine k-means the codes into emergent clusters (no labels) → keep clusters with ≥4 members → for each cluster, teach a distinct property to HALF its members → a HELD-OUT cluster member inherits ITS cluster's property (argmax over cluster properties, rung-1's read). Anti-cheat: label-DERANGEMENT (shuffle cluster assignments) must collapse it.

## The result (multi-seed)
| corpus | scale | held-out inherit | deranged | chance | mean cluster coherence | verdict |
|---|---|---|---|---|---|---|
| **TinyStories** | K=256, 10 clusters, 6-seed | **0.664 ± 0.043** (6.6× chance) | 0.119 | 0.100 | +0.095 | **GO** |
| **WikiText** | K=1024, 12 clusters, 3-seed | **0.642 ± 0.035** (7.7× chance) | 0.093 | 0.083 | +0.048 | **GO** |

**GO on BOTH corpora** — every seed beats chance AND the derangement control by ≥0.15. A held-out member of a cluster the brain DISCOVERED (never told the grouping) inherits a property taught only to other cluster members; label-derangement collapses it to chance → the DISCOVERED clustering is load-bearing. It works on a children's-story corpus AND an encyclopedic corpus with no probe change — the mechanism is corpus-general.

## The emergent categories are semantically real (if noisy) — TinyStories, seed 42
```
cluster 3 (coh +0.131, n=20): mom dog cat boy man ...   (animate nouns / characters)
cluster 9 (coh +0.117, n=42): said help can did want make ...   (verbs / modals / dialogue)
cluster 1 (coh +0.110, n=53): day big saw named wanted went played ...  (past-tense narrative)
cluster 7 (coh +0.081, n=26): hey fun toys things more ...   (activity / objects)
```
The clusters recover meaningful semantic/syntactic groupings (animate nouns, verbs, narrative words) with no labels — noisy at the edges (subword fragments like 'ily'/'ucy' from "Lily"/"Lucy" leak through the tokenizer; some function words mix in), but the inheritance is robust to the noise (6.6–7.7× chance, derangement collapses).

## Honest scope
Rate-level read (the spiking realization is rung-2, which ports to the committed HTM kernel). The clusters are noisy (tokenizer fragments + function-word contamination — a cleaner tokenizer/content-filter would sharpen them). WikiText's clusters are looser (mean coherence +0.048 vs TinyStories +0.095 — encyclopedic codes cluster less tightly) but inheritance still GOes (7.7× chance). k-means is a host clustering step (the spiking realization is the EMERGE-33/38 competitive self-organizing pooler — this shows the CAPABILITY probe-free; the spiking pooler already discovers clusters on-substrate).

## What this establishes
The inheritance pipeline is now FULLY emergent end-to-end: discover the vocab from real experience → learn category structure → DISCOVER the categories by clustering (no hand-labeled probe) → reason (inherit) over the discovered categories → and it generalizes across corpora. The last hand-designed scaffold (the probe) is removed; the categories are the brain's own. Combined with rungs 1–4: discover a broad vocab from any real corpus → discover its categories → teach a fact → answer about a held-out word → abstain on the unknown, transformer-free, moat intact. Next: the spiking self-organizing pooler (EMERGE-33/38) over these real-corpus codes for a fully-spiking probe-free pipeline; a cleaner tokenizer to sharpen the emergent clusters.

## Files
`research/runners/_realcorpus_inheritance_emergent_clusters_derisk.py`; `research/findings/raw/_rc_emerclust_ts_s*.json` (TinyStories 6-seed) + `_rc_emerclust_wiki_s*.json` (WikiText 3-seed). Prior: rungs 1–4; breadth `...-open-domain-breadth-is-a-data-scale-lever-...md`; the WikiText probe-mismatch that motivated removing the probe.
