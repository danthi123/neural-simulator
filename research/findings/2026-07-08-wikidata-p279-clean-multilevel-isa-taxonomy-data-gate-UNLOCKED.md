# The taxonomy DATA-gate is UNLOCKED with real data (GO): Wikidata P279 (subclass-of) yields a CLEAN MULTI-LEVEL is-a graph — 8 hub superordinates (mammal/tree/tool/fish/insect/fruit/vehicle/bird, up to 25 members each) + 110 is-a chains (fish→animal, tree→plant, fruit→food, ...) — exactly the clean multi-level structure that distributional + copular extraction from TinyStories/WikiText could NOT produce. The data-acquisition premise (CYCLE 1039) holds; the ready EMERGE-44/45 stacked pooler now has a clean is-a corpus. NO `sim/` edit.

**Date:** 2026-07-08
**Runner:** `research/runners/_realcorpus_wikidata_taxonomy_derisk.py` (reuses the fluid-conversation P15 Wikidata SPARQL fetcher `_sparql`). numpy + network. NO `sim/` edit.
**Verdict:** GO (data-acquisition first-look) — real curated is-a (Wikidata P279) yields a clean multi-level taxonomy; the taxonomy is DATA-unlockable, not mechanism-blocked.

## Why this ran (the taxonomy data-acquisition de-risk, per CYCLE 1039)
The multi-level taxonomy was triply-confirmed as a DATA gate (distributional centroid + co-occurrence + explicit copular extraction all NEGATIVE on TinyStories/WikiText; the mechanism EMERGE-44/45 is ready). The research gate (CYCLE 1039) scoped the acquisition path. This is the first-look: does a REAL is-a-structured resource yield the clean multi-level graph the corpora couldn't?

## The result — Wikidata P279 (fetch subclasses of 8 superordinates for BREADTH + their parents for DEPTH)
```
is-a graph: 150 pairs, 144 children, 19 parents
HUB superordinates (>=3 children): 8  ->  mammal(20), tree(25), tool(25), fruit(16), insect(15), fish(14), vehicle(14), bird(9)
multi-level chains (child->parent->grandparent): 110  (e.g. fish->animal, insect->arthropod, tree->plant, fruit->food, vehicle->object, tool->object)
VERDICT: GO
```
Real Wikidata P279 yields a **clean multi-level is-a graph**: genuine hub superordinates with many members AND real is-a chains — precisely the structure the distributional/copular extraction from children's-story / raw-encyclopedic text could NOT produce (all NEGATIVE, CYCLE 986/1038). ⇒ the taxonomy is a DATA gate, and a clean is-a resource surpasses it. The EMERGE-44/45 stacked pooler is validated on this exact shape (clean is-a groups, synthetic 6-seed GO); it now has a real corpus.

## Emergent-defensibility (the honest path)
Feeding the P279 GRAPH directly (child→parent edges) to the pooler would inject a curated structure (a shortcut per the emergent directive). The emergent-defensible consumption (per CYCLE 1039): text-convert P279 to is-a SENTENCES ("a dog is a mammal") → the existing copular miner extracts the is-a from the text → the stacked pooler DISCOVERS the grouping → inheritance. This first-look establishes the DATA is clean + multi-level; the full emergent de-risk (text-convert → copular miner → stacked pooler → held-out-member inheritance vs permuted-super control, 6-seed) is the immediate follow-on.

## What this establishes
The multi-level taxonomy data-gate — characterized as blocked across 3 extraction approaches × 2 corpora — is UNLOCKABLE via data acquisition: real curated is-a (Wikidata P279) yields the clean multi-level graph the mechanism needs. This converts the deep taxonomy boundary into a solved data-acquisition + a scoped mechanism-consumption follow-on. Follow-on: the emergent-path consumption (text-convert → copular miner → EMERGE-44 stacked pooler → inheritance, 6-seed); more superordinates for breadth; wire multi-level inheritance into the conversational console.

## Files
`research/runners/_realcorpus_wikidata_taxonomy_derisk.py`; `research/findings/raw/_wikidata_taxonomy.json` (the mined is-a graph). Reuses `_fluidconv_phase15_wikidata_breadth_derisk._sparql`. Prior: the acquisition research gate `2026-07-08-taxonomy-acquisition-research-gate-simple-wiki-definitions-path.md`; the triply-confirmed data-gate `2026-07-08-copular-isa-extraction-NEGATIVE-taxonomy-data-gate-confirmed-3rd-approach.md`; the ready mechanism `2026-07-02-emerge44-stacked-pooler-multilevel-taxonomy-GO.md`.
