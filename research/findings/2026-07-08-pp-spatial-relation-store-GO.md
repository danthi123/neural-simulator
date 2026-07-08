# PP (spatial/directional) relation store (GO, 6-seed, perfect): a 4-role FHRR store recovers "the owl flies TO the pond" (goal) / "ON the rock" (location) AND distinguishes GOAL from LOCATION, abstaining on the unstored — completing the relational schema breadth (SVO + ditransitive + PP). Production = EMERGE-72 C_PPGOAL/C_PPLOC on spikes. NO `sim/` edit.

**Date:** 2026-07-08
**Runner:** `research/runners/_realcorpus_pp_relation_store_derisk.py` (reuse-by-import: the validated SVOStore FHRR `_phasors`/`_role`; the corpus stream codes). numpy. NO `sim/` edit.
**Verdict:** GO (6-seed, perfect) — recovers PP spatial relations, distinguishes goal/location, moat clean.

## Why this ran (completing the relational schema breadth)
After the ditransitive ternary store (CYCLE 1028), the last construction family is the PP (spatial/directional) relation: "the owl flies TO the pond" (a GOAL) vs "the owl flies ON the rock" (a LOCATION). The production already renders both (EMERGE-72 C_PPGOAL/C_PPLOC on spikes); this de-risks the STORE side so the brain can DISCUSS spatial relations.

## The mechanism + result — 6-seed, perfect
A 4-role FHRR store (AGENT/VERB/GOAL/LOCATION; each fact binds AGENT+VERB + the GOAL *or* the LOCATION role):
```
answer-acc          = 1.000 every seed   (where does the owl fly to? -> pond;  ...fly on? -> rock)
goal/location-discrim = 1.000 every seed (querying GOAL on a location-fact -> MISS, and vice versa)
moat-abstain        = 1.000 every seed   (an unstored agent-verb -> abstain)
chance = 0.004 (V=256)
```
The store recovers the destination for each fact's own spatial kind, and the goal-vs-location DISCRIMINATION is perfect (the distinct role phasors keep "fly-to X" and "fly-on X" separable), with the no-confab moat abstaining on the unstored.

## What this establishes
The brain's relational schema now spans SVO (binary) + ditransitive (ternary, CYCLE 1028) + PP (spatial goal/location) — a comprehensive relational breadth, each with store + query + moat, 6-seed, and each with an EMERGE production that renders it on spikes. The brain can store + discuss actions, transfers, and spatial relations. Follow-on: wire the ditransitive + PP stores + their producers into the console (comprehend + generate on spikes); richer query forms ("where does X fly?").

## Files
`research/runners/_realcorpus_pp_relation_store_derisk.py`. Reuses the SVOStore FHRR + EMERGE-72 (C_PPGOAL/C_PPLOC producers). Prior: the ditransitive store `2026-07-08-ditransitive-ternary-relation-store-GO.md`; the binary SVO store `_realcorpus_svo_qa_derisk.py`.
