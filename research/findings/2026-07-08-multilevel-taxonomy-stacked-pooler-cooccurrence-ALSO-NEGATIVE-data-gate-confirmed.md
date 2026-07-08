# Multi-level taxonomy: the NAMED surpass mechanism (EMERGE-44/45 stacked-pooler CO-OCCURRENCE super) ALSO fails on TinyStories (6-seed NEGATIVE, head-to-head) — real ≈ deranged, same as the centroid baseline. The mechanism is RIGHT; the DATA is the gate: TinyStories lacks a nested is-a signal even in second-order fine-cluster co-occurrence. The surpass is a taxonomically-structured corpus. NO `sim/` edit.

**Date:** 2026-07-08
**Runner:** `research/runners/_realcorpus_multilevel_stacked_derisk.py` (co-occurrence super vs the centroid-super NEGATIVE baseline, head-to-head, same seeds/data). numpy. NO `sim/` edit.
**Verdict:** NEGATIVE (6-seed) — confirms the multi-level taxonomy is a DATA gate, not a mechanism gate.

## Why this ran (the disciplined research-gate move)
The multi-level NEGATIVE (CYCLE 986) showed CENTROID clustering of fine-cluster centroids into supers is not load-bearing (real ≈ deranged) and NAMED the untested next mechanism: EMERGE-44/45's stacked competitive pooler, which groups fine clusters by their CO-OCCURRENCE (a second-order signal — which fine-clusters appear together in the same stories) rather than centroid similarity. Rather than accept the boundary or re-tread, this TESTS the named mechanism head-to-head.

## The mechanism tested (vs the NEGATIVE baseline)
- **CO-OCCURRENCE super (the named mechanism):** build the fine-cluster co-occurrence matrix (for each story, the fine clusters of its words all co-occur), PPMI-normalize, cluster the fine clusters by their co-occurrence PROFILES → supers from what appears together (the EMERGE-44 stacked-pooler signal).
- **CENTROID super (the NEGATIVE baseline):** cluster the fine-cluster centroids by similarity.
Both feed the SAME 2-level generalization test (teach a super property to some fine-clusters, hold out a WHOLE fine-cluster, check its members inherit the super) + super-derangement.

## The result — 6-seed (K=512, 20 fine, 5 coarse)
```
CO-OCC super:  inherit 0.291 ~ deranged 0.278  (beats-chance/deranged all-seeds = False)
CENTROID super: inherit 0.482 ~ deranged 0.512  (the NEGATIVE baseline, confirmed)
```
**Both real ≈ deranged, every seed.** The co-occurrence super is NO more load-bearing than the centroid super — shuffling the super labels does not degrade the inheritance in either case. The stacked-pooler's second-order co-occurrence signal does NOT carry a nested is-a hierarchy in TinyStories.

## The diagnosis (mechanism right, data is the gate)
A nested is-a hierarchy needs same-super sub-categories to be MORE related to each other (at the member level OR in co-occurrence) than to other supers' sub-categories. TinyStories — a children's-story corpus — has FLAT category structure: different animals co-occur (in animal stories) but NOT in a way that separates "animals" from "toys" as a super above the flat level. So NEITHER the fine codes' centroid similarity NOR the fine-clusters' second-order co-occurrence forms a super above the flat categories. The mechanism (stacked pooler) is correct (it works on synthetic data where same-super items DO co-occur, EMERGE-44 GO); the corpus is the gate.

## The surpass (the data lever, not a mechanism)
Multi-level taxonomy needs a corpus with a genuine is-a signal — where same-super items co-occur / are described together MORE than cross-super (an encyclopedic or taxonomically-structured corpus: "animals include dogs, cats, and birds"; "a robin is a kind of bird"), OR explicit is-a definitional text with POS-clean genus extraction (the dictionary genus was too noisy, CYCLE 986). The mechanism (EMERGE-44/45 stacked pooler) is ready; it awaits is-a-structured DATA. This is the breadth data-lever pattern (more/richer data), not a new mechanism.

## What this establishes
The multi-level taxonomy boundary is now RIGOROUSLY a DATA gate: the named surpass mechanism (stacked-pooler co-occurrence) was tested head-to-head and ALSO fails on TinyStories (real ≈ deranged, 6-seed) — so it is NOT a mechanism gate. Single-level reasoning (inherit + cancel, comprehensively GO) stands; multi-level taxonomy awaits an is-a-structured corpus. An honest, decisive boundary confirmation.

## Files
`research/runners/_realcorpus_multilevel_stacked_derisk.py`; `research/findings/raw/_ml_stacked_s*.json`. Prior: the multi-level centroid NEGATIVE `2026-07-08-multilevel-taxonomy-generalization-over-real-corpus-clustering-NEGATIVE.md`; the dictionary-genus data-gate `2026-07-08-emergent-is-a-from-dictionary-genus-NOISY-taxonomy-data-gate-confirmed.md`; EMERGE-44/45 (the stacked pooler, GO on synthetic is-a-structured data).
