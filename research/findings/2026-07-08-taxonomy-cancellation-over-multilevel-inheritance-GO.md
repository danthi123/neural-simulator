# Member-specific CANCELLATION over multi-level (chained) inheritance — the canonical "penguin" case, on real Wikidata is-a (GO, 6-seed): a member inherits its grandparent's property 2-up UNLESS it has a taught OWN property that contradicts it, in which case the own property overrides (cancel=1.000); a NORMAL sibling still inherits (1.000); the override is LOAD-BEARING (removing it flips YES→NO, ctrl-flip=1.000); the no-confab moat holds. NO `sim/` edit.

**Date:** 2026-07-08
**Runner:** `research/runners/_realcorpus_taxonomy_cancellation_derisk.py` (`CancellingTaxonomyQA` extends the CYCLE-1044 `TaxonomyQA` with member-specific own-property overrides). numpy. NO `sim/` edit.
**Verdict:** GO (6-seed) — cancellation over multi-level inheritance, with a load-bearing control.

## The result — 6-seed (4 grandparents plant/vehicle/food/tool)
```
cancel        (member's OWN property beats the inherited grandparent property) = 1.000 every seed
inherit-intact(a NORMAL sibling still inherits the grandparent property)       = 1.000 every seed
ctrl-flip     (WITHOUT the override -> YES; WITH it -> NO; the override flips)  = 1.000 every seed
moat          (unknown token abstains)                                         = 1.000 every seed
```

## The mechanism (the penguin case)
Teach a grandparent property P (all descendants inherit P via the CYCLE-1043 chain) + a member M an OWN property Q≠P. Then:
- "can a M P?" → **NO** — M's proximal own fact Q cancels the distal inherited P (graded-specificity precedence: a closer taught fact beats a more distant inherited one; EMERGE-26/54 cancellation, Dusek-Eichenbaum).
- "can a M Q?" → **YES** (M's own).
- "can a M' P?" → **YES** — a normal sibling M' (no override) still inherits P; the exception does NOT leak to siblings.
- "can a zzz P?" → **MOAT**.
Cancellation = a priority check: a member's own taught property is consulted BEFORE the chained inheritance.

## Anti-cheat / validity
- **Load-bearing control (ctrl-flip=1.000):** BEFORE teaching M's override, "can a M P?" inherits → YES; AFTER, → NO. The SAME member's answer flips only because of the override → the cancellation is the taught exception, not a coincidence.
- **Sibling non-leakage (inherit-intact=1.000):** the exception is member-specific — a sibling of the same super/grandparent still inherits, so the override doesn't corrupt the category.
- **Moat preserved:** an unknown token still abstains.

## What this establishes
The multi-level taxonomy now supports EXCEPTIONS: a member inherits its ancestors' properties by default but a taught member-specific property correctly overrides (the penguin case), without leaking to siblings and without weakening the moat — the full inherit + cancel reasoning the single-level discovered-cluster reasoner has, now over the real Wikidata multi-level chain. Follow-on: wire `CancellingTaxonomyQA` into the flagship console's `taxonomy_qa` path (+ a `teach_taxonomy_exception` method); broaden the taxonomy breadth; a natural-text is-a source.

## Prior
`2026-07-08-taxonomy-qa-multilevel-inheritance-conversational-GO.md` (the QA, CYCLE 1044), `-flagship-console-multilevel-taxonomy-wired-GO.md` (console wiring, 1045), `-wikidata-2up-chained-multilevel-inheritance-GO.md` (the chained read, 1043).
