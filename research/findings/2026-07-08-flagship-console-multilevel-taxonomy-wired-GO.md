# The flagship console now answers MULTI-LEVEL TAXONOMIC property questions (GO): a default-off `taxonomy_qa` flag on `UnifiedTalkableConsole` routes "can a <leaf> <superordinate-property>?" through the CYCLE-1043 chained inheritance (member→super→grandparent) over the real Wikidata is-a graph — with level-specificity (a mismatched property → no) + the no-confab moat (unknown leaf → abstain). Default path BYTE-PRESERVED (the discovered-cluster reasoner + all existing paths unchanged). NO `sim/` edit.

**Date:** 2026-07-08
**Files:** `research/runners/_realcorpus_unified_talkable_console.py` (additive `taxonomy_qa` flag + one gated `ask()` branch), `_realcorpus_taxonomy_qa_console_derisk.py` (added `hold_out=False` deployed mode), `tests/test_realcorpus_unified_console.py` (+2 CI tests). numpy. NO `sim/` edit.
**Verdict:** GO — the multi-level taxonomy is wired into the flagship conversational console; targeted CI 4/4 (2 taxonomy + 2 core regression).

## What landed
The CYCLE-1044 taxonomy property-QA (chained multi-level inheritance + level-specificity + moat) is now a path in the flagship `UnifiedTalkableConsole`:
- **Flag:** `taxonomy_qa=False` (default off, mirroring the existing `spiking_gen`/`multi_bridge`/`neural_route`/`rich_gen` opt-ins). When on, the console builds a `TaxonomyQA(seed, wikidata_3level_tree, hold_out=False)` — the deployed mode where every leaf is answerable (no held-out super).
- **Routing (gated, non-shadowing):** in `ask()`, a "can/does a `<X>` `<prop>`?" query routes to the taxonomy ONLY when `X` is a known taxonomy leaf AND `prop` is a taught superordinate property. Otherwise it falls through to the existing discovered-cluster property reasoner / moat — so the new path never shadows the emergent reasoner.
- **Answers:** "can a `<leaf>` `<prop>`?" → "yes -- the `<leaf>` can `<prop>`" (inherited 2-up via the chain) / "no -- a `<leaf>` does not `<prop>`" (level-specificity: a different grandparent's property) / "I don't know" (no-confab moat, unknown leaf).

## Validation (targeted CI, numpy)
```
test_taxonomy_multilevel_inheritance_qa      PASS  -- >=2 grandparents: leaf inherits its property (yes),
                                                       a mismatched property is denied (no), unknown leaf abstains (moat)
test_taxonomy_default_off_byte_identical     PASS  -- default console has _tax is None; a taxonomy-only leaf
                                                       question falls through to the moat when the path is off
test_property_inheritance_and_cancellation   PASS  -- the discovered-cluster reasoner unregressed
test_relational_answer_and_moat              PASS  -- the relational path + moat unregressed
```
The default (no `taxonomy_qa`) path is byte-preserved: `_tax is None`, no new knowledge source loaded, all existing paths identical.

## Honest scope
The taxonomy currently covers 4 grandparents (plant/vehicle/food/tool) from the CYCLE-1042 Wikidata 3-level fetch — the canonical "animal→...→breathe" chain needs an `animal` grandparent with ≥2 supers (a breadth follow-on: refetch with animal's sub-hierarchy). The is-a STRUCTURE is Wikidata-curated (encyclopedic experience); the codes + inheritance are learned/emergent. The knowledge source is distinct from the discovered-cluster (TinyStories) reasoner — the console now carries BOTH, gated by question content.

## What this establishes
The flagship talkable console now reasons over a MULTI-LEVEL taxonomy from real-world knowledge: it answers property questions about never-taught members via chained inheritance, denies mismatched properties, and abstains on the unknown — all default-off, byte-preserving, NO `sim/` edit. Follow-on: broaden the taxonomy (more grandparents incl. animal, so "can a robin breathe?" works); member-specific cancellation over the chain; a natural-text is-a source (Simple Wikipedia definitions).

## Prior
`2026-07-08-taxonomy-qa-multilevel-inheritance-conversational-GO.md` (the QA de-risk, CYCLE 1044), `-wikidata-2up-chained-multilevel-inheritance-GO.md` (the chained read, 1043).
