# The multi-level taxonomy is CONVERSATIONALLY usable (GO, 6-seed): "can a <held-out member> <property>?" is answered by the CYCLE-1043 chained read (member→super→grandparent → inherited grandparent property = YES), a DIFFERENT grandparent's property is correctly DENIED (NO=1.000 — level-specificity), an unknown token hits the no-confab MOAT (abstain=1.000), and the grandparent-derangement control collapses the YES-answers (0.000). The multi-level taxonomy on real Wikidata is-a is now usable in dialogue. NO `sim/` edit.

**Date:** 2026-07-08
**Runner:** `research/runners/_realcorpus_taxonomy_qa_console_derisk.py` (Wikidata 3-level tree + learn_stream_codes + the CYCLE-1043 chained read wrapped as property Q&A). numpy. NO `sim/` edit.
**Verdict:** GO (6-seed) — multi-level property-QA with level-specificity, moat, and a collapsing control.

## The result — 6-seed (4 grandparents: plant/vehicle/food/tool; properties grow/move/nourish/help)
```
YES  (held-out member inherits its GRANDPARENT's property, 2-up)   = 1.000 every seed
NO   (member correctly DENIES a DIFFERENT grandparent's property)  = 1.000 every seed   <- level-specificity
moat (unknown token "zzz" -> "I don't know what a zzz is")         = 1.000 every seed
grandparent-DERANGEMENT (property bound to a WRONG gp's supers)    -> YES collapses to 0.000
```

## What this adds beyond CYCLE 1043 (the honest delta)
CYCLE 1043 validated the chained read (leaf→super→grandparent) as a classification accuracy. This wires it into a CONVERSATIONAL property-Q&A and validates two NEW pieces:
- **Level-specificity (NO=1.000):** a member inherits ONLY its own ancestors' properties. Asking "can a `<plant-leaf>` move?" (move = vehicle's property) → the chain routes the leaf to `plant` ≠ `vehicle` → NO. A member does NOT spuriously inherit every taught property — the inheritance is bound to the correct is-a chain.
- **Conversational moat:** an unknown token abstains ("I don't know what a zzz is") — the no-confab moat holds in the QA form.
The YES metric re-confirms the CYCLE-1043 chained inheritance; the query form + level-specificity + moat are the new capability layer (the taxonomy usable in dialogue).

## Anti-cheat / validity
- **Held-out:** queries are leaves of the HELD-OUT super per grandparent; the grandparent prototypes are built from the OTHER (taught) supers only — the queried leaves' super is excluded from every prototype.
- **Derangement (valid collapsing control):** binding each grandparent's prototype to a WRONG grandparent's supers routes the chain to the wrong grandparent → the YES-answers (which require the correct grandparent) collapse to 0.000, while the un-deranged path stays 1.000.
- (Same discipline as CYCLE 1041/1043 — the control must collapse for the GO to be real.)

## What this establishes
The multi-level taxonomy — data-unlocked (Wikidata P279, CYCLE 1040), single-level-inherited (1041), 2-up-chained (1043) — is now CONVERSATIONALLY usable: the brain answers a property question about a never-taught member via multi-level inheritance, denies mismatched properties (level-specificity), and abstains on the unknown. Follow-on: wire this path into the flagship `UnifiedTalkableConsole` (route a "can a X <prop>?" query through the taxonomy QA when X is a known leaf + the property is a taught superordinate property, else the existing discovered-cluster reasoner / moat); member-specific cancellation over the chained inheritance; more grandparents/breadth.

## Files
`research/runners/_realcorpus_taxonomy_qa_console_derisk.py`; reuses `research/findings/raw/_wikidata_3level.json`, `_emergent_vocab_breadth_scale_derisk.learn_stream_codes`, `_realcorpus_inheritance_rung1_derisk._unit_rows`. Prior: `2026-07-08-wikidata-2up-chained-multilevel-inheritance-GO.md` (the chained read), `-wikidata-taxonomy-inheritance-END-TO-END-GO.md` (single-level).
