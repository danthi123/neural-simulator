# EMERGE-45 / toward-semantics — GO (6/6 seeds): a THREE-LEVEL discovered taxonomy + TRANSITIVITY. Stacking the competitive pooler 3 deep (member features → sub-category → genus → order) discovers a 3-level hierarchy from co-occurrence, and inheritance chains through TWO learned levels so a held-out sub-category inherits its ORDER property while the sibling order's property stays FALSE. Extends EMERGE-44 (2-level) per the research gate. NO `sim/` edit.

**2026-07-02 (autonomous).** Runner `research/runners/_emerge45_three_level_hierarchy_derisk.py`; CI guard `tests/test_emerge45_three_level_hierarchy.py` (3 tests). Reuse-by-import (`_emerge14` + `_emerge12` + the EMERGE-44 pooler helper); NO `sim/` edit; CPU numpy-backend; 6-seed.

## The claim (6/6 seeds)
8 sub-categories group into 4 genera, which group into 2 orders. Three stacked competitive poolers discover the levels:
- **L1** (features → sub-category codons), **L2** (L1 codons → genus codons, via same-genus co-occurrence), **L3** (L2 codons → order codons, via same-order co-occurrence).
- **Two-level inheritance:** an entire held-out sub-category (never taught the order property) inherits its ORDER **2 discovered levels up** — order-acc **0.97 mean** (0.92/1.00/1.00/0.88/1.00/1.00 across seeds 42/43/44/100/101/102, chance 0.50).
- **Transitivity discrimination:** the SIBLING order's property is NOT inherited — the member infers its own order, not the other (robin is-an animal-that-breathes; robin is NOT a fish-that-swims). **transitivity 1.00 every seed.**
- **Anti-cheats:** PERMUTED-co-occurrence **0.36** (random cross-order pooling → the hierarchy isn't discovered → collapses), dAP-LESION **0.00**.

## Mechanism
Each level reuses the EMERGE-38 competitive pooler, but its INPUT is the codons of the level below, and it is trained on the CO-OCCURRENCE at that level (same-genus members for L2; same-order members for L3). So L2 columns tune to what co-occurs within a genus → genus codons; L3 columns tune to what co-occurs within an order → order codons. A member's features → L1 → L2 → L3 codon; an order property taught (committed three-term kernel) on the training members' L3 codons is inherited by a held-out sub-category through the two discovered levels; the graded-drive read picks the member's own order over the sibling (transitivity). Biology: the ventral hierarchy's successive pooling stages with growing abstraction (Kandel 6e Ch 21) + ATL convergence zones (Patterson–Lambon Ralph; Damasio 1989) — each cortical level pools the one below.

## Significance
The stacking mechanism (EMERGE-44) generalizes to THREE levels: the brain discovers a genuine multi-level taxonomy from experience and inherits through it WITH discrimination (a member gets its own branch's property, not the sibling's) — the transitive structure of Collins-Quillian semantic memory, learned not hand-designed, on one spiking brain, no transformer. Combined with EMERGE-42/43 (cancellation + multi-override), the discovered-taxonomy substrate now supports the full inference repertoire across multiple levels.

## Honest scope + next
- The pooler LEARNING is a rate-reference (fully-on-substrate at EMERGE-39/40; k-WTA spiking at EMERGE-41); the inheritance chain runs on the spiking bridge over the discovered L3 codons. Held-out at the sub-category level (its members co-occurred within the genus/order during pooler training). Two discovered levels of inheritance (sub→genus→order).
- Next: EMERGE-46 — the fully-spiking stacked hierarchy (replace the numpy pooler layers with the on-substrate EMERGE-40 kernel + EMERGE-41 FS-WTA for each layer).

## Artifacts
`research/runners/_emerge45_three_level_hierarchy_derisk.py`, `tests/test_emerge45_three_level_hierarchy.py`, `research/findings/raw/_emerge45_three_level_hierarchy.json`. Prior: `2026-07-02-emerge44-stacked-pooler-multilevel-taxonomy-GO.md`.
