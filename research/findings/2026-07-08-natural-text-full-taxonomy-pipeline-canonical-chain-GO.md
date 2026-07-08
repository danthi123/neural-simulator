# The WHOLE taxonomy pipeline runs on NATURAL definitional text (GO, 6-seed): REAL Simple-Wikipedia definitions → a DISCOVERED 3-level is-a tree → the chained multi-level inheritance QA answers the canonical "can a bear/duck/cod breathe?" (leaf→mammal/bird/fish→animal→breathe), YES=1.000, moat=1.000. No curated graph anywhere — the structure is discovered from the definitions. NO `sim/` edit.

**Date:** 2026-07-08
**Runner:** `research/runners/_realcorpus_simplewiki_taxonomy_qa_derisk.py` (multi-sentence natural-copula extractor → 3-level tree builder → the CYCLE-1044 `TaxonomyQA` chained read). numpy + cached defs. NO `sim/` edit.
**Verdict:** GO (6-seed) — natural-text structure discovery + canonical multi-level inheritance + moat.

## The discovered natural-text 3-level tree + canonical demo
```
animal (breathe): mammal->[bear,cat,dog,lion,rabbit]; bird->[duck,owl,parrot,penguin,sparrow]; fish->[cod,goldfish,salmon,trout]
plant  (grow):    tree->[birch,maple,oak,willow]
machine(work):    vehicle->[bicycle,bus,car,motorcycle,truck]

Q: can a bear breathe?  (bear->mammal->animal)  -> yes
Q: can a duck breathe?  (duck->bird->animal)    -> yes
Q: can a cod breathe?   (cod->fish->animal)     -> yes

6-seed: YES(inherit)=1.000  moat=1.000
```
The entire tree — leaves→supers AND supers→grandparent — is MINED from the fetched Simple-Wikipedia definitions ("Dogs are mammals", "Mammals ... They are a group of vertebrate animals"), then fed to the validated chained inheritance QA.

## What the extraction needed (beyond CYCLE 1049)
Super definitions state their genus in a LATER sentence ("Mammals are in the class Mammalia. They are a group of vertebrate animals."). So `mine_multi` reads the first ~3 sentences with the TITLE as the persistent subject (resolving pronoun subjects "They are.../It is..."). `build_tree` then assembles {grandparent: {super: [leaves]}} where a super is a node with both children and a grandparent. Fixed a `_sing` over-strip ("class"→"clas") with an -ss/-us guard.

## Honest scope (adversarial self-check — no overclaim)
- This ~65-title fetch yields ONE rich property-grandparent (**animal**, with 3 supers mammal/bird/fish). Because animal is the only grandparent with ≥2 supers, the chained read's L2 step (super→grandparent) has a single routing candidate here — so this result does NOT re-test multi-grandparent DISCRIMINATION. The mismatch-NO cross-property test is therefore N/A on this tree.
- The multi-grandparent DISCRIMINATION + mismatch-NO + grandparent-derangement collapse were validated separately on the 4-grandparent curated P279 tree (CYCLE 1043/1044/1046). What THIS result adds is that the STRUCTURE is discoverable from NATURAL definitional text (not a curated graph), and the canonical animal chain + moat work on it.
- The breadth follow-on (a 2nd rich property-grandparent from natural text — more plant/machine titles) would re-demonstrate discrimination on a natural-text tree.

## What this establishes
The complete taxonomy pipeline — structure discovery + multi-level chained inheritance + the no-confab moat — runs end-to-end on NATURAL Simple-Wikipedia definitions, the most emergent-defensible source (the brain reads real definitions and discovers the is-a; no curated graph is injected). The canonical dog/bear→mammal→animal→breathe chain, the demonstrative example of the whole taxonomy arc, answers correctly on a fully-natural-text-discovered tree. Follow-on: a 2nd rich natural-text grandparent (breadth) → natural-text discrimination + mismatch-NO; wire the natural-text tree into the console's `taxonomy_qa` path as an alternative to the P279 tree.

## Files
`research/runners/_realcorpus_simplewiki_taxonomy_qa_derisk.py`; reuses `_realcorpus_simplewiki_isa_derisk` (the CYCLE-1049 fetch+extractor) + `_realcorpus_taxonomy_qa_console_derisk.TaxonomyQA`. Prior: `2026-07-08-simplewiki-natural-text-isa-taxonomy-source-GO.md` (the source, 1049), `-taxonomy-qa-multilevel-inheritance-conversational-GO.md` (the QA on P279, 1044).
