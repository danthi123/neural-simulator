# DITRANSITIVE answer GENERATION on spikes (GO, 3/3): "the dog GIVES the cat a bone" — the 7-slot C_DITRANS order produced ON SPIKES by EMERGE-77's 8-pool 2-stage-calibrated registry producer, and every word (incl. the 3sg verb via PRODUCTIVE inflection give→gives) spelled ON SPIKES by the productive multi-bridge A→W. This COMPLETES the relational-generation schema breadth on spikes: property + transitive + spatial + ditransitive. NO `sim/` edit.

**Date:** 2026-07-08
**Runner:** `research/runners/_realcorpus_spiking_broca_ditransitive_answer_derisk.py` (reuse-by-import: EMERGE-77 `DitransRegistry`/`DitransRegistryProducer` 8-pool 2-stage read; the `ProductiveMultiSpeaker` reset_steps=150). numpy. NO `sim/` edit.
**Verdict:** GO (3/3, 3-seed 42/43/44) — the ditransitive answer is produced fully on spikes (order + productive morphology + every word).

## Why this ran (completing the schema-breadth generation)
The relational schema breadth was de-risked as STORES (SVO + ditransitive + PP, CYCLE 1028/1029). The generation-on-spikes covered property (F_MODAL), transitive (C_TRANS), and spatial (C_PPGOAL/C_PPLOC, CYCLE 1030/1031). The last construction is the DITRANSITIVE ("the dog gives the cat a bone", 7 slots) — the richest core relation (agent + recipient + theme). EMERGE-77 surpassed the 7-slot capacity boundary (n_slot_pools 6→8 + a 2-stage per-pool bias-calibrated read) but rendered the WORDS as host tokens; this wires it with the A→W (words on spikes) + productive inflection ("gives" = spell("give")+spell("s")).

## The result — seed 42
```
C_DITRANS(dog,  give,  cat, bone) -> "the dog gives the cat a bone"    [exact, 7-slot order + words ON SPIKES]
C_DITRANS(cat,  bring, dog, gift) -> "the cat brings the dog a gift"   [exact]
C_DITRANS(bear, send,  fox, seed) -> "the bear sends the fox a seed"   [exact]
VERDICT: GO (3/3 exact)
```
- The **7-slot C_DITRANS order** (the/subj/verb/the/recipient/a/theme) is produced by EMERGE-77's 8-pool 2-stage-calibrated spiking read (the primacy gradient → per-pool rate ranking, with the per-pool bias calibration for the tightly-packed 8-rank case + the EMERGE-61 wash-out).
- Every **word** is spelled on spikes by the productive multi-bridge A→W: the/a (function words), dog/cat/bear/fox (subjects/recipients, BRIDGE-1/3), bone/gift/seed (themes, BRIDGE-4), and the **3sg verb via PRODUCTIVE inflection** (gives = spell("give")+spell("s"), brings, sends — never a stored 3sg lexeme).
- The EMERGE-75b history-independent read (reset_steps=150) keeps every word correct across the deep 7-slot render.

## What this establishes
The relational-generation schema breadth is now COMPLETE on spikes — the brain produces, fully on the spiking substrate:
- **property** ("the owl can fly", F_MODAL)
- **transitive** ("the dog eats the cat", C_TRANS)
- **spatial** ("the owl runs to the pond" / "on the rock", C_PPGOAL/C_PPLOC)
- **ditransitive** ("the dog gives the cat a bone", C_DITRANS)

— each with the slot order + every word (incl. productive morphology) on spikes, transformer-free, gate-first moat. Combined with the de-risked stores (SVO + ternary + spatial), the brain can store AND speak the full relational breadth. Follow-on: a 6-seed confirmation of the ditransitive render; wire the richer relations (ditransitive/PP) into the console (comprehend + generate); the -ies allomorphy (fly→flies, the RANK-3 phonological mechanism).

## Files
`research/runners/_realcorpus_spiking_broca_ditransitive_answer_derisk.py`. Reuses EMERGE-77 (`DitransRegistry`, 8-pool 2-stage read), the `ProductiveMultiSpeaker` (BRIDGE-1/2/3/4 + affix, reset_steps=150), CYCLE-1024 productive inflection. Prior: the PP generation `2026-07-08-pp-spatial-generation-on-spikes-mechanism-GO-render-3of4.md`; the ditransitive store `2026-07-08-ditransitive-ternary-relation-store-GO.md`; EMERGE-77.
