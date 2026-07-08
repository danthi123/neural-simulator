# The FLAGSHIP console GENERATES its ditransitive answer ON SPIKES (opt-in `--rich-gen`, GO): "the dog gives the cat a bone" — the 7-slot order via EMERGE-77's 8-pool 2-stage producer + every word (incl. the 3sg verb via PRODUCTIVE inflection) via the ProductiveMultiSpeaker (BRIDGE-1/2/3/4 + affix, reset_steps=150). Default OFF byte-preserved. NO `sim/` edit.

**Date:** 2026-07-08
**Runner:** `research/runners/_realcorpus_unified_talkable_console.py` (`rich_gen=True` / CLI `--rich-gen`; uses `ProductiveMultiSpeaker` + a `DitransRegistry` 8-pool producer + `_gen_ditrans`). CI guard `tests/test_realcorpus_unified_console.py::test_rich_gen_ditransitive_answer_on_spikes`. numpy. NO `sim/` edit.
**Verdict:** GO — the console's ditransitive answer is produced on spikes (order + every word incl. productive morphology), default paths byte-preserved.

## Why this ran (fully-spiking richer-relation answers in production)
The console converses about ditransitive relations (CYCLE 1034) but the ANSWER was host-rendered. The ditransitive generation ON SPIKES was de-risked (CYCLE 1032, 3-seed) + the EMERGE-75b read-path surpass (CYCLE 1031). This wires them into the console so its ditransitive answer is fully-spiking-generated — the "one brain" goal for the richer relations.

## What was built (opt-in `rich_gen`, default off)
- When `rich_gen`, the console's speaker is the **`ProductiveMultiSpeaker`** (BRIDGE-1/2/3/4 + the affix bridge; reset_steps=150 — the EMERGE-75b history-independent read; productive 3sg spell(stem)+spell("s")) — a drop-in for the console's speaker contract (spell/vocab/speak_frame/speakers).
- `rich_gen` builds the **`DitransRegistry`** 8-pool 2-stage-calibrated producer (EMERGE-77) alongside the property + C_TRANS producers.
- `_gen_ditrans(subj, verb, recip, theme)` routes the console's ditransitive answer through the 8-pool producer + the productive A→W (via the de-risk's `_emit_ditrans_aw`) when every filler is spellable; else the host template.
- Default OFF: `rich_gen=False` → `_ditrans_producer is None` → `_gen_ditrans` falls to the host template → default paths BYTE-PRESERVED (the 20-test arc CI is green with these additive edits).

## The result (console smoke, seed 42)
```
rich_gen: ditrans producer built = True; speaker vocab = 61 (BRIDGE-1/2/3/4)
teach "<s> give <r> <t>"          -> stored
what does the <s> give the <r>?   -> "the <s> gives the <r> a <t>"   MATCH the exact C_DITRANS spiking surface
```
The ditransitive answer's 7-slot order is produced by the 8-pool 2-stage spiking read, and every word — including the 3sg verb "gives" via productive inflection (spell("give")+spell("s")) — is spelled on spikes. MATCH=True to the expected surface.

## What this establishes
The flagship console now GENERATES its ditransitive answer fully on spikes (opt-in), so the richer-relation conversation is spiking end-to-end for the ditransitive: teach → query (ternary FHRR store) → generate (8-pool spiking-Broca producer + productive A→W). Combined with the property/transitive spiking generation, the console speaks the relational breadth on spikes. Follow-on: route the PP (spatial) answer through the registry producer's C_PPGOAL/C_PPLOC (same pattern); a 6-seed rich_gen confirm; the -ies allomorphy.

## Files
`research/runners/_realcorpus_unified_talkable_console.py` (`rich_gen` flag + `ProductiveMultiSpeaker` speaker + the ditransitive producer + `_gen_ditrans`); `research/runners/_realcorpus_productive_multi_speaker.py` (`speak_frame` + `speakers`/`_of` drop-in); `tests/test_realcorpus_unified_console.py`. Reuses EMERGE-77 (8-pool ditransitive), CYCLE-1024 productive inflection, CYCLE-1031 (reset_steps=150). Prior: the console ditransitive teach/query `2026-07-08-console-ditransitive-teach-query-GO.md`; the ditransitive generation `2026-07-08-ditransitive-generation-on-spikes-GO-schema-breadth-complete.md`.
