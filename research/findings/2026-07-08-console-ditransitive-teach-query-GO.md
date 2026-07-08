# The FLAGSHIP console now converses about DITRANSITIVE (ternary) relations (GO): teach "the dog gives the cat a bone" live → query the theme ("what does the dog give the cat?" → bone) + the recipient ("who does the dog give a bone?" → cat), abstaining on the unstored (moat). Extends the console's relational conversation beyond binary SVO, reusing the CYCLE-1028 4-role FHRR store. NO `sim/` edit.

**Date:** 2026-07-08
**Runner:** `research/runners/_realcorpus_unified_talkable_console.py` (`teach_ditransitive` + a ditransitive query branch in `ask()` + a co-resident `DitransStore`). CI guard `tests/test_realcorpus_unified_console.py::test_ditransitive_teach_query_moat`. numpy. NO `sim/` edit.
**Verdict:** GO — the console stores + queries + abstains on ternary relations, reusing the de-risked 4-role FHRR store.

## Why this ran (richer relations in the actual conversation)
The ditransitive (ternary) relation store was de-risked standalone (CYCLE 1028, 6-seed perfect) and the ditransitive GENERATION on spikes was de-risked (CYCLE 1032). This wires the STORE + COMPREHENSION + QUERY into the FLAGSHIP `UnifiedTalkableConsole`, so the console can actually CONVERSE about ternary relations (a genuine "talk-like-an-LLM" expressivity advance beyond binary SVO).

## What was built
- A co-resident **`DitransStore`** (the CYCLE-1028 4-role FHRR: agent/verb/recipient/theme) built in the console `__init__` over the SAME phasor codes Z as the binary `SVOStore`.
- **`teach_ditransitive(subj, verb, recip, theme)`** — stores a ternary fact live (all fillers must be in the discovered vocab + the verb resolvable); persists as `["ditrans", s, v, r, t]`.
- A **ditransitive query branch** in `ask()` (before the binary what/who): a ditransitive verb (give/show/bring/send/tell/offer) + TWO nouns → the ternary store. "what does the X give the Y?" → theme (Y = recipient given); "who does the X give a Z?" → recipient (Z = theme given); an unstored ternary fact abstains (moat).
- REPL: a 4-content declarative ("the dog gives the cat a bone") teaches a ditransitive fact.

## The result (console smoke, seed 42)
```
teach "<s> show <r> <t>"           -> stored
what does the <s> show the <r>?    -> "the <s> shows the <r> a <t>"   (ditransitive; theme recovered)
who does the <s> show a <t>?       -> "the <s> shows the <r> a <t>"   (ditransitive; recipient recovered)
what does the <other> show the <r>?-> "I don't know"                  (moat; unstored subject)
```
(The store/query/moat mechanism is validated; the answer is host-rendered here — the ditransitive generation ON SPIKES is de-risked separately, CYCLE 1032, and is the follow-on wire-in.)

## What this establishes
The flagship talkable console now converses about TERNARY relations (teach + both argument queries + moat), extending its relational conversation beyond binary SVO — the brain can discuss transfers ("gives the cat a bone"), grounded in the de-risked 4-role FHRR store, moat intact. Default binary/property paths byte-preserved (the ditransitive branch only fires for a ditransitive verb + 2 nouns). Follow-on: wire the ditransitive/PP GENERATION on spikes (CYCLE 1032/1031) into the console's ditransitive/relational answers (currently host-rendered); PP (spatial) teach+query in the console.

## Files
`research/runners/_realcorpus_unified_talkable_console.py` (`DitransStore` + `teach_ditransitive` + the ditransitive `ask()` branch + persistence); `tests/test_realcorpus_unified_console.py`. Reuses the CYCLE-1028 `DitransStore`. Prior: the ditransitive store `2026-07-08-ditransitive-ternary-relation-store-GO.md`; the ditransitive generation `2026-07-08-ditransitive-generation-on-spikes-GO-schema-breadth-complete.md`.
