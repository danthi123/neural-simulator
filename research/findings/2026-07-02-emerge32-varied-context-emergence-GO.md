# EMERGE-32 / toward-semantics — GO (6/6 seeds): EMERGENT STRUCTURE FROM VARIED EXPERIENCE. The emergent category is NOT keyed to a single shared context token — each member is observed with a DIFFERENT overlapping feature subset (no universal token), yet a held-out member still inherits a property taught via one exemplar, via the feature overlap. Closes the "one shared token = a provided superordinate" critique of EMERGE-30. NO `sim/` edit.

**2026-07-02 (autonomous; the EMERGE-30 robustness follow-on).** Runner `research/runners/_emerge32_varied_context_emergence_derisk.py`; CI guard `tests/test_emerge32_varied_context_emergence.py` (4 tests). Reuse-by-import (`_emerge14` + `_emerge12`); NO `sim/` edit; CPU numpy-backend; 6-seed.

## What it strengthens
EMERGE-30 discovered the category from co-occurrence, but each category's members shared ONE identical context token — a skeptic could call that token a provided superordinate. EMERGE-32 removes it: each member is observed with its OWN random subset of its category's feature pool (robin sees `[18,19,20]`, sparrow `[18,19,21]`, canary `[18,19,21]`, trout `[23,24,25]`, salmon `[22,23,25]`, pike `[22,23,24]` — no single feature shared by all; the CI asserts no universal token). The category still emerges and a held-out member still inherits.

## The claim (6/6 seeds)
Teach the property via ONE exemplar per category ("robin can fly", "trout can swim"), binding it to the exemplar + its observed features. Then:
- **Held-out inheritance 1.00:** the held-out members (sparrow, canary, salmon, pike — different feature subsets, property never taught) inherit their category property via the OVERLAP of their subset with the exemplar's — `sparrow/canary → fly`, `salmon/pike → swim`.
- **Moat 1.00:** a never-observed member abstains.

## Mechanism (Rogers-McClelland feature overlap + the next-state predictor)
Each category has a feature POOL (n=4); each member is streamed with its own random 3-subset, guaranteeing pairwise overlap ≥ ACT_TH (since 3+3 > 4). The committed `sim/` three-term kernel learns member-content → its feature subset (on-bridge Hebbian). The property, taught via an exemplar, binds to the exemplar's features; a held-out member (a different subset) inherits because its subset overlaps the exemplar's taught features by ≥ ACT_TH → the shared features prime the property. No single feature is universal; the overlap carries the category.

## Anti-cheats (all airtight, 6/6)
- **PERMUTED-POOL** (each member draws its subset from the MIXED union of both pools → subsets overlap only by chance, ~K²/8 < ACT_TH → no category structure): held-out inheritance collapses to **0.29 mean** (0.00–0.75), well below the intact 1.00 — isolating the LEARNED feature-overlap structure as the cause.
- **NO-LEARNING** (skip the stream): **0.00**. **dAP-LESION**: **0.00**. **MOAT 1.00.** No-universal-token asserted. 6-seed.

## Significance
The emergence-from-experience result (EMERGE-30) is now robust to realistic, varied, non-identical contexts — the category is carried by feature OVERLAP (Rogers-McClelland / Lambon Ralph feature-based semantics), not a hand-provided shared token. The substrate discovers category structure from varied experience and infers over it, emergent + unsupervised on one spiking brain, NO `sim/` edit.

## Honest scope + next
- The features are still the environment (legitimate world/experience); the GROUPING is discovered from overlap — the permuted-pool control isolates exactly that.
- Next: an HTM Spatial Pooler that forms a NEW shared column block from the varied inputs (a fully-emergent superordinate representation, not the feature overlap directly); cancellation on emergent codes; couple varied-context emergence into the experiential console (EMERGE-31).

## Artifacts
`research/runners/_emerge32_varied_context_emergence_derisk.py`, `tests/test_emerge32_varied_context_emergence.py`, `research/findings/raw/_emerge32_varied_context_emergence.json`. Prior: `2026-07-02-emerge30-emergent-superordinate-GO.md`, `2026-07-02-emerge31-experiential-console.md`.
