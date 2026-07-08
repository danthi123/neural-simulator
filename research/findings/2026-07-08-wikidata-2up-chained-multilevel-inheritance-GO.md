# TRUE multi-level (2-up) taxonomy inheritance on real Wikidata is-a (GO, 6-seed): a HELD-OUT super's leaves inherit their GRANDPARENT 2 levels up via a CHAINED read (leaf→super→grandparent), 1.000, with the grandparent-derangement control collapsing to 0.000. The CYCLE-1042 flat-code 2-up NEGATIVE is resolved: 2-up needs the CHAINED/stacked mechanism (both single-level steps chained), not flat codes. NO `sim/` edit.

**Date:** 2026-07-08
**Runner:** `research/runners/_realcorpus_wikidata_2up_chained_derisk.py` (3-level Wikidata tree grandparent→supers→leaves; learn_stream_codes; chained nearest-prototype read). numpy. NO `sim/` edit.
**Verdict:** GO (6-seed) — true 2-up multi-level inheritance on real is-a, valid collapsing control.

## The result — 6-seed (4 grandparents: plant/vehicle/food/tool)
```
2up-chained (leaf->super->grandparent) = 1.000 every seed  -- held-out super's leaves inherit their grandparent
grandparent-DERANGEMENT                = 0.000 every seed  -- collapses (each grandparent's proto = a WRONG gp's supers)
chance                                 = 0.250
```

## The mechanism (resolving the CYCLE-1042 flat-code NEGATIVE)
CYCLE 1042 found the FLAT read (leaf→grandparent directly) NEGATIVE — the co-occurrence codes don't propagate 2 hops. The fix (stacked-pooler logic): CHAIN the two single-level steps that DO work — (L1) leaf→super (the leaf's code matches its super's other leaves), then (L2) super→grandparent (the super's code matches same-grandparent supers). A held-out super's leaves route leaf→(true super)→(true grandparent)→property. Both steps are single-level (each validated); chaining gives true 2-up.

## Adversarial-verify (3 self-caught control bugs this session)
The first two 2-up passes gave 1.000 AND deranged=1.000 (invalid controls: a consistent relabel, then a derangement that didn't touch the tested prototypes). Rather than accept the false result, the control was fixed each time until valid: the final derangement assigns each grandparent a DIFFERENT grandparent's supers as its prototype → a super's code no longer matches its own grandparent → the chain routes wrong → 0.000. The GO stands on the valid collapsing control. (Same discipline caught the CYCLE-1041 derangement bug.)

## The taxonomy, fully resolved end-to-end
1. DATA-gate: unlocked — real Wikidata P279 yields a clean multi-level is-a graph (CYCLE 1040), where distributional+copular extraction from natural corpora could not (triply-confirmed NEGATIVE).
2. Single-level inheritance (member→direct-super) on real is-a: GO (CYCLE 1041).
3. True 2-up (multi-level) inheritance: NEGATIVE via flat codes (CYCLE 1042) → GO via the chained/stacked read (this, CYCLE 1043).
⇒ the multi-level taxonomy deep boundary is fully surpassed: real is-a data + emergent code-learning + the chained (stacked-pooler-logic) read → true multi-level inheritance, valid collapsing controls, 6-seed. Honest scope: the is-a STRUCTURE is Wikidata-curated (encyclopedic experience); DISCOVERY from raw distributional experience remains gated (the natural-corpus NEGATIVE).

## Files
`research/runners/_realcorpus_wikidata_2up_chained_derisk.py`; `research/findings/raw/_wikidata_3level.json`. Prior: `2026-07-08-wikidata-taxonomy-inheritance-END-TO-END-GO.md` (single-level), CYCLE 1042 (flat-code 2-up NEGATIVE), the ready EMERGE-44 stacked pooler.
