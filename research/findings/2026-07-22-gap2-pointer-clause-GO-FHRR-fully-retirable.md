# gap#2 — POINTER-CLAUSE (depth-1 embedded clauses by indirection): 6-seed GO → the learned spiking slot-binder now covers the COMPLETE deployed FHRR set (FHRR exact-inverse algebra fully retirable)

**2026-07-22, CPU/numpy, coexisting with the fluency training.** Step 2/2 of retiring the FHRR exact-inverse algebra
(step 1/2 = the attribute slot, GO + committed 9b0cdbe4). Per the research gate `2026-07-22-recursive-slotbinder-
research-gate.md` MOVE 3 #1: recursion by INDIRECTION (point-don't-copy — Neural Blackboard Architecture /
assembly-projection multilevel pointers / Frankland-Greene factored registers), NOT by copying the clause composite
into a filler (which would re-import the FHRR ~2 superposition cap the slot-binder was built to escape).

## The mechanism (`research/runners/slotbinder_composer.py` ONLY; additive, default-preserving; NO `sim/` edit)
A depth-1 embedded clause "dog saw (cat chase bird)" is stored by indirection, not composition:
- Dedicated pointer filler-pools `__CLAUSE0__ .. __CLAUSE{max_clauses-1}__` appended like AFFIRM/NEGATE/NOATTR; the
  pointer LITERALLY names the fact-group index (`CLAUSE_j <-> group j`, bijective) — **the pointer identity IS the
  address; no host address table is load-bearing.**
- `store(dog, saw, Clause(cat,chase,bird))`: (1) store the INNER clause `(cat,chase,bird)` as its OWN fact at group `j`
  (the existing flat GO mechanism — its own near-orthogonal slots, so NO clause-level superposition; the gap-#2 win
  preserved at depth-1); (2) store the MATRIX fact `(dog, saw, PTR=CLAUSE_j)` — bind the CLAUSE_j pointer pool into the
  matrix patient slot.
- Recall = read matrix patient slot → recover CLAUSE_j → FOLLOW → read group `j` with the SAME neural scan (`query_clause`
  returns the inner (a,v,p)). Depth-1 only (the parser abstains beyond; depth-2 = human center-embedding limit).

## Result — 6-seed GO (independently controller-reproduced)
- **MAIN:** embedded-clause roles (inner a/v/p) recovered **6/6**, matrix roles (outer a/v) **6/6** (bar ≥0.90 on ≥5/6);
  flat SVO + attribute + polarity + multi-hop un-regressed (**1.000**).
- **AC1 permuted-pointer:** embedded recovery vs-TRUE **0.000**, vs-PERMUTED **1.000** ⇒ the indirection is load-bearing
  structure, not a coincidence.
- **AC2 lesion-the-second-hop:** reading the matrix patient WITHOUT following the pointer returns the POINTER code, NOT
  clause content (**1.000**) ⇒ the follow-hop is the mechanism (mirrors the FHRR 2-level unbind).
- **AC3 wrong-clause distractor:** with ≥2 clauses stored, the matrix pointer selects the RIGHT group (right **1.000**,
  wrong **0.000**, all clauses distinct) ⇒ not a "read the only clause" artifact.
- **AC4 moat:** a matrix fact whose pointer names no stored group → None; never-stored cues → None; a real clause still
  recovered. All 6 seeds.
- **0 regression:** `tests/test_slotbinder_composer.py` + the step-1 attribute de-risk still GO.

## ⇒ the FHRR exact-inverse algebra is now FULLY RETIRABLE
The learned spiking slot-binder covers the COMPLETE capability set the FHRR ships in production: flat SVO + polarity +
multi-hop (`query_chain`) + single-attribute (step 1) + depth-1 embedded clause (step 2). The FHRR's own 2-attribute-F3
(~29%) and depth-2 are boundaries the FHRR itself does not cross. So the hand-designed exact-inverse VSA/FHRR
algebra — the project's #1 documented idealization shortcut (flagged by the 2026-07-22 field-novelty assessment) — is
replaced end-to-end by a LEARNED, self-organizing, fully-spiking, one-substrate binder whose no-confab moat is the
intrinsic neural content-addressable scan (not a VSA-cleanup shortcut). Next: make the slot-binder the production
DEFAULT (retire the FHRR/rf fallback) — a wire-in + a 320-scale GPU re-verify (gated on the fluency training). De-risks:
`research/runners/_gap2_{attribute_slot,pointer_clause}_derisk.py`.
