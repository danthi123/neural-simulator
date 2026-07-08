# DITRANSITIVE (ternary) relation store (GO, 6-seed, perfect): a 4-role FHRR store binds "the dog GIVES the cat a bone" = give(agent=dog, recipient=cat, theme=bone) and answers BOTH argument queries ("what does the dog give the cat?" → bone; "who does the dog give a bone to?" → cat), abstaining on the unstored (moat). Schema expansion beyond binary SVO — the brain can now STORE + DISCUSS ternary relations; the production side (EMERGE-77 C_DITRANS) already renders it on spikes. NO `sim/` edit.

**Date:** 2026-07-08
**Runner:** `research/runners/_realcorpus_ditransitive_store_derisk.py` (reuse-by-import: the validated SVOStore FHRR `_phasors`/`_role`, extended 3→4 roles; the corpus stream-code learning). numpy. NO `sim/` edit.
**Verdict:** GO (6-seed, perfect) — 4-role FHRR recovers ditransitive relations, both argument queries, moat + permuted controls clean.

## Why this ran (a capability advance beyond the fully-spiking core)
This session made the whole conversational turn on binary SVO fully spiking (comprehend + reason + speak, CYCLE 1021-1027). The natural CAPABILITY advance is a richer relational SCHEMA. The production already handles multi-argument constructions (EMERGE-74/77: C_TRANS transitive, C_DITRANS ditransitive at n_slot_pools=8, C_PPGOAL/C_PPLOC); the console's STORE + comprehension was binary SVO. This de-risks the STORE side of ditransitive relations (agent-verb-recipient-theme), so the brain can DISCUSS them, not just render them.

## The mechanism (SVOStore 3→4 roles)
The validated SVOStore binds an SVO fact as an FHRR superposition `AGENT*z_s + VERB*z_v + PATIENT*z_o`. The ditransitive store adds a RECIPIENT role: `f = AGENT*z_s + VERB*z_v + RECIPIENT*z_i + THEME*z_o`. Each argument is recovered by unbind (`f*conj(ROLE)`) + cleanup. D=512 gives ample FHRR SNR for 4 superposed terms (√(512/4) ≈ 11).

## The result — 6-seed (42/43/44/100/101/102), perfect
```
theme-acc     = 1.000 every seed   (what does the dog give the cat? -> bone)
recipient-acc = 1.000 every seed   (who does the dog give a bone to? -> cat)
moat-abstain  = 1.000 every seed   (an unstored agent-verb-recipient -> abstain)
permuted      = 0.000 every seed   (query with a WRONG verb -> miss)
chance        = 0.004 (V=256)
```
Both argument queries recover perfectly; the no-confab moat abstains on the unstored; the permuted-verb control collapses. The 4-role FHRR handles ternary relations with no loss at D=512.

## What this establishes
The brain's relational schema extends beyond binary SVO to ternary (ditransitive) relations — store + both argument queries + moat, 6-seed perfect. With the EMERGE-77 C_DITRANS producer (which renders "the dog gives the cat a bone" on spikes), the full ditransitive conversation is reachable: teach a ditransitive fact → query either argument → generate the answer on spikes. Follow-on: wire the ditransitive store + the C_DITRANS producer into the console (comprehend "does the dog give the cat a bone", generate on spikes); PP relations (C_PPGOAL/C_PPLOC, goal/location roles — the same 4-role pattern with GOAL/LOCATION).

## Files
`research/runners/_realcorpus_ditransitive_store_derisk.py`. Reuses the SVOStore FHRR (`_realcorpus_svo_qa_derisk`/`_realcorpus_svo_compose_probe`) + EMERGE-77 (the C_DITRANS producer, generation on spikes). Prior: the binary SVO store `_realcorpus_svo_qa_derisk.py`; the relational generation `2026-07-08-fully-spiking-relational-transitive-answer-generation-GO.md`.
