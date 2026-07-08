# The FLAGSHIP talkable console now GENERATES its property answers ON SPIKES (opt-in `--spiking-gen`, GO): the answer's slot ORDER ("the X can Y") is produced by the EMERGE-65 self-organized spiking-Broca producer (competitive queuing + wash-out), replacing the host `speak_frame` f-string; every word already spelled on spikes by the A→W. Gate-first moat intact; default path byte-unchanged. NO `sim/` edit.

**Date:** 2026-07-08
**Runner:** `research/runners/_realcorpus_unified_talkable_console.py` (`UnifiedTalkableConsole(..., spiking_gen=True)` / CLI `--spiking-gen`). CI guard `tests/test_realcorpus_unified_console.py::test_spiking_gen_property_answer_on_spikes`. numpy. NO `sim/` edit.
**Verdict:** GO — the console's property answer is generated fully on spikes (order + words), moat holds, default off is byte-identical.

## Why this ran (the fluency fork, wired into production)
The fully-spiking property-answer GENERATION was de-risked GO (3-seed; the EMERGE-65 `SelfOrganizedProducer` fixes the base-CQ order tail). That de-risk proved the capability in isolation; this wires it into the FLAGSHIP `UnifiedTalkableConsole` so the console ITSELF generates on spikes. Before: the console's property answer used `ConceptFrameSpeaker.speak_frame`, which spells each WORD on spikes (A→W) but assembles the slot ORDER with a host f-string `f"{the} {subj} {can} {verb}"`. Now: an opt-in `spiking_gen` builds the self-organized spiking-Broca producer (from the corpus stream) and routes the property answer's frame through it (`_gen_frame` → `producer.speak({F_MODAL, subject, verb})`), so the ORDER is a spiking read-out.

## The result — the console generating on spikes (seed 42)
```
does a bird run? -> "no -- the bird can sleep"   [override, order+words ON SPIKES]   (bird = taught exception -> sleep)
does a cat  run? -> "yes -- the cat can run"     [inherit]
does a fish run? -> "yes -- the fish can run"    [inherit]
does a bear run? -> "yes -- the bear can run"    [inherit]
does a frog run? -> "yes -- the frog can run"    [inherit]
does a zzz  run? -> "I don't know"               [moat -- producer NOT invoked (0 productions on the abstain)]
```
- All five property answers (1 exception + 4 inherit) render exact, the slot ORDER from the spiking-Broca producer + every word from the A→W read-out.
- The gate-first MOAT holds: exactly 5 productions for 5 answers, **0 on the abstain** (the producer is never invoked when the reasoner abstains — moat by construction).

## What's load-bearing / design
- `spiking_gen` is opt-in (default OFF). When off, `_producer is None` → `_gen_frame` falls through to the prior `speak_frame` → the default console path is BYTE-UNCHANGED (the 9 existing CI tests are unaffected). When on, the property answer's two frame renders (inherit + exception override) route through `_gen_frame` → the producer.
- Both the A→W speaker bridge AND the producer's slot bridge co-execute in ONE numpy process (the SelfOrganizedProducer's wash-out + exact-order CQ handle the co-execution).
- Reuse-by-import: EMERGE-65 `SelfOrganizedProducer`, EMERGE-62 `build_stream`, the breadth A→W. NO `sim/` edit.

## What this establishes
The flagship real-corpus talkable console now SPEAKS its property answers with the sentence STRUCTURE produced on spikes (not a host template) — "simulate Broca" realized in production, gate-first moat intact, default path preserved. Follow-on: route the RELATIONAL (SVO transitive) answer + the describe/contrast discourse through the extended producer (EMERGE-72/74 C_TRANS + morphology); more spoken vocab.

## Files
`research/runners/_realcorpus_unified_talkable_console.py` (`spiking_gen` flag + `_gen_frame`); `tests/test_realcorpus_unified_console.py` (`test_spiking_gen_property_answer_on_spikes`). Prior: the de-risk `2026-07-08-fully-spiking-generation-real-corpus-property-answer-GO.md`; EMERGE-65/66 (the self-organized producer + the console spike-render).
