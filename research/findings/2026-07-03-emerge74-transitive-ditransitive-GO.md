# EMERGE-74 — CORE SVO production: TRANSITIVE renders on spikes (GO, 6-seed); DITRANSITIVE is mined but hits the honest N_SLOT_POOLS=6 capacity boundary

**Date:** 2026-07-03
**Verdict:** **GO** — the self-organized spiking-Broca producer BROADENS to the CORE SVO argument-structure constructions. TRANSITIVE "the dog chases the cat" is mined from the corpus + rendered EXACT on spikes every seed (6 constructions total: the 5 EMERGE-72 + TRANSITIVE). DITRANSITIVE "the dog gives the cat a bone" is genuinely MINED (its 7-role signature discovered + routed) but its render is CAPACITY-GATED at 7 slots > N_SLOT_POOLS=6 — a precisely-named, honest SPIKING-SUBSTRATE CAPACITY boundary (NOT a data/label wall), whose fix is a bounded scale lever (more slot pools).
**Runner:** `research/runners/_emerge74_transitive_ditransitive_derisk.py`
**CI:** `tests/test_emerge74_transitive_ditransitive.py` (14 tests, CPU/numpy, offline — all pass)
**Raw:** `research/findings/raw/_emerge74_transitive_ditransitive.json`
**Research gate:** `2026-07-03-broaden-construction-inventory-research-gate.md` (MOVE 3 / RANK 2 + MOVE 4: "the project ALREADY has a richer inventory on the same FrameCQ engine — argstructure_composer.FRAME_LEXICON transitive/ditransitive + the GO _bucketB corpus frame miner — so broadening is largely routing the existing inventory through the self-organized S1a/S1b/S2 pipeline; one label_sentence extension for post-verbal content unifies the argstructure/_bucketB transitive/PP inventory").
**Reuse-by-import; NO `sim/` edit; the gate-first moat untouched.**

---

## What EMERGE-74 does

Broadens the EMERGE-72/73 signature-keyed `ConstructionRegistry` from the 5-construction inventory (3 EMERGE frames + PP-goal/PP-location) to the **CORE SVO argument-structure constructions** — TRANSITIVE (DET SUBJ VERB DET OBJ) and DITRANSITIVE (DET SUBJ VERB DET IOBJ DET OBJ) — by routing the project's **already-GO** argument-structure inventory through the self-organized mining + spiking-render pipeline, corpus-driven.

**The two inventories UNIFIED (provenance cross-checked, `provenance_consistent: True`):**
- `argstructure_composer.FRAME_LEXICON` (`research/runners/argstructure_composer.py:65-77`): transitive `_default` = agent-action-**patient**; ditransitive `give`/`send` = agent-action-**THEME-RECIPIENT**.
- The `_bucketB` corpus verb-frame miner (`research/findings/raw/_bucketB_corpus_mined_frames.json`): `chase` → [agent, action, patient] (transitive); `give`/`send`/`bring`/`carry` → [agent, action, THEME, RECIPIENT] (ditransitive).
- EMERGE-74's own constructions: transitive = 2 content roles (SUBJ, OBJ); ditransitive = 3 content roles (SUBJ, IOBJ, OBJ). All three sources agree.

**The ONLY additions over EMERGE-72** (the mine/order/spell/moat were already construction-agnostic — EMERGE-64/63/59):
1. **`label_sentence_svo`** — the one bounded, precedented label extension: admits a POST-VERBAL CONTENT structure with ONE (transitive OBJ) or TWO (ditransitive IOBJ + OBJ) post-verbal content words, and recognizes the core-SVO verb lexemes (chase/give/… — not in EMERGE-62's base `_VERBS`, so the SVO labeller supplies its own SVO-aware verb + 3sg-inflection test). This is the "second post-verbal argument" (IOBJ) the RANK-2 gate flagged — argstructure ditransitive RECIPIENT; Goldberg's ditransitive argument-structure construction ("X causes Y to receive Z").
2. **`build_stream_svo`** — the EMERGE-62 base stream + de-risk-local transitive + ditransitive SVO sentences, CORPUS-mined by the EMERGE-64 signature machinery (NOT host-listed frame definitions — the CONSTRUCTIONS template is validation-only, never read by the miner).

`IOBJ` is one new open-class slot type (the second post-verbal content noun). The `RegistryProducer` render, the gate-first moat, and the EMERGE-61 inter-utterance wash-out are all reused unchanged.

---

## Result (6 seeds: 42/43/44/100/101/102, CPU/numpy)

Every seed is identical:

| metric | value (all 6 seeds) |
|---|---|
| constructions registered (mined) | **7** (5 EMERGE-72 + C_TRANS + C_DITRANS) |
| constructions rendered EXACT on spikes | **6** (the 5 EMERGE-72 + C_TRANS) — render **1.000** |
| TRANSITIVE mined + rendered exact | **True / True** every seed |
| DITRANSITIVE mined | **True** every seed |
| DITRANSITIVE fits N_SLOT_POOLS=6 | **False** (7 > 6 — the capacity boundary) |
| PERMUTED-CORPUS render | **0.000** (n_registered 0.0) |
| CROSS-CONSTRUCTION render | **0.000** |
| NO-CORPUS registered | **0** |
| held-out shared det+subj+verb backbone | **1.000** |
| gate-first moat calls on abstain | **0** |
| provenance_consistent (argstructure + _bucketB) | **True** |

**GO gates met:** ≥6 constructions rendered exact every seed with render 1.000; every input-destruction control collapses with margin ≥0.30; held-out generalizes on the shared SVO backbone; moat 0; 6 seeds. `go: True`.

**Live transcript (spikes, gate-first moat):**
```
you> what does the wolf chase?    broca> the wolf chases the ball          [TRANS; producer INVOKED]
you> where does the owl fly?      broca> the owl flies to the pond         [PPGOAL; producer INVOKED]
you> can a penguin fly? [deny]    broca> the penguin does not fly          [NEGMOD; producer INVOKED]
you> what does the wolf give the cub?
                                  broca> [mined but > N_SLOT_POOLS=6 -- capacity BOUNDARY; more slot pools is the fix]
                                                                           [DITRANS; producer NOT invoked (capacity-gated)]
you> can a zzz fly?               broca> I don't know.                     [MOAT; producer NOT invoked]
```

---

## The honest CAPACITY boundary — DITRANSITIVE (7 slots > 6 pools), named precisely

The ditransitive "the dog gives the cat a bone" is `det subj verb:3sg det iobj det:a obj` = **7 slots**. `N_SLOT_POOLS=6` (`_emerge59_spiking_broca_frame_slots_derisk.py:118`). So it **exceeds the spiking substrate's pool count**.

Critically, this is **NOT a data/label wall** — the corpus mine DISCOVERS the ditransitive's full 7-role signature every seed (`ditransitive_mined_all_seeds: True`), its signature matches the ground-truth ditransitive template exactly, and it is routed to `C_DITRANS`. The S1a/label side works: the ditransitive is genuinely attested + labellable. It is an honest **SPIKING-SUBSTRATE CAPACITY** wall (the FrameCQ pool count), and the registry correctly refuses to render it (it would overflow the slot pools) rather than force it. `ditransitive_capacity_boundary: True`.

**The fix is a bounded SCALE lever:** bump `N_SLOT_POOLS` 6 → 8, after which the ditransitive renders with **ZERO further mechanism** (the mine already found it). This de-risk therefore validates the ditransitive end-to-end **up to the capacity gate**. Named, not hidden; do not force it.

**The named residual (held-out):** the ditransitive's DISTINCTIVE part — the IOBJ (a SECOND post-verbal content noun) — is attested ONLY by the ditransitive itself. Holding it out, no other construction has two post-verbal content nouns, so the IOBJ scaffold is not recoverable from the others (`ditransitive_distinctive_recovered: False`) — the same precisely-named shared-vs-distinctive residual as EMERGE-63/64/72. The shared SVO backbone (det+subj+verb) DOES generalize (1.000).

---

## Anti-cheats (all collapse — the broadening is genuinely corpus-derived)

- **PERMUTED-CORPUS** — shuffling each exemplar's word order before mining dilutes every construction's dominant ordering below the dominance threshold → **0 registered** (render 0.000). The broadening is corpus-order-driven, not host-smuggled.
- **CROSS-CONSTRUCTION** — rendering construction A's fact through a DIFFERENT construction B's mined structure is **wrong** (0.000): the transitive rendered through the ditransitive's structure ≠ the transitive surface. Dominey-Hinaut form-specificity — construction A's order must not render construction B.
- **NO-CORPUS** — empty stream → 0 registered.
- **HELD-OUT-CONSTRUCTION** — a fully-held-out construction's shared det+subj+verb backbone is recovered from the OTHERS (1.000); the distinctive post-verbal scaffold is the named residual.

## The moat (gate-first, untouched)

The no-confab moat holds by construction: on ABSTAIN the producer is NEVER invoked (0 productions on abstains, 6 seeds). The capacity-gated ditransitive ALSO never invokes the producer (it is not loaded into the spiking producer). Not weakened.

---

## A note on the corpus stream design (the Goldilocks precondition)

The SVO content nouns (subjects/objects/recipients/themes) are **base-disjoint + large + per-subject selectionally restricted**, so each stays context-narrow (low coverage → CONTENT, not closed). This matters: an SVO noun in a fixed determiner-flanked slot ("the dog CHASES the CAT"), if small-vocab and shared with the base stream, would CONCENTRATE → high frequency AND high coverage → the EMERGE-62 Goldilocks discovery would MISLABEL it CLOSED (the documented EMERGE-72/73 adjective-ambiguity failure mode), and the labeller would correctly SKIP the SVO sentence. Verified: **0 SVO nouns mislabelled closed, all 6 seeds** (`test_svo_vocab_is_base_disjoint_and_content`); the only residual false-positives are the pre-existing EMERGE-72 base-stream adjectives, which are not in any SVO slot and do not affect SVO mining. (An early iteration that shared/small-pooled the SVO nouns confirmed this failure mode directly — the objects/subjects went closed and transitive/ditransitive did not mine.)

---

## Honest scope / what this is NOT

- BROADENS the bounded, corpus-attested, router-selected inventory to the CORE SVO constructions (transitive; ditransitive up to the capacity gate) — the biggest expressivity jump toward richer conversation (arguments AFTER the verb). NOT open prose (R4, the separate deferred wall; the from-scratch spiking LM is ~4 orders too small).
- The A→W SPELL stays the token surface for THIS de-risk. The fully-spiking A→W of the NEW content words (the object/indirect-object nouns) is the batched EMERGE-75 follow-on (its own spiking validation is `concept_speak_demo`).
- The corpus mining is offline syllabus prep (BRAIN-BASED-ONLY compliant — like rendering a retinal image the neural retina reads); the STRUCTURE is rendered on REAL spikes.

## Regression

EMERGE-59..73 defaults preserved (no edits to those files; EMERGE-74 is additive reuse-by-import). Verified: EMERGE-59..65 CI 83 passed; EMERGE-66..73 CI 52 passed + 4 pre-existing skips. EMERGE-72 de-risk still GO. NO `sim/` edit anywhere.

## Sources
- Goldberg, *Constructions* — argument-structure constructions (transitive; ditransitive "X causes Y to receive Z").
- Hinaut & Dominey (PLoS ONE 2013; Brain & Language 2015) — production = SELECTING the construction to express predicate + thematic roles; the reservoir generalizes to NEW constructions from closed-class order/position (the construction-router warrant).
- Tomasello usage-based / Goldberg construction grammar — the inventory grows by abstracting more usage-based constructions.
- Project precedents: `argstructure_composer.FRAME_LEXICON:65` (transitive/ditransitive), `_bucketB_corpus_mined_frames_derisk` (corpus verb-frame mining, GO), `_emerge72_construction_registry_derisk.py` (the signature-keyed registry + OBJ + label_sentence_ext), `_emerge59_spiking_broca_frame_slots_derisk.py:118` (N_SLOT_POOLS), catalog G.12 (Broca grammatical encoding).
