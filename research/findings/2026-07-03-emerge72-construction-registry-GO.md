# EMERGE-72 — BROADEN the self-organized spiking producer's CONSTRUCTION INVENTORY beyond the 3 EMERGE frames (signature-keyed construction registry) — GO (6-seed), + honest adjective boundary named

**Date:** 2026-07-03
**Verdict:** **GO, 6/6 seeds.** The fully-self-organized spiking-Broca producer (EMERGE-56..71) BROADENS from **3 to 5 corpus-mined, router-selected constructions** with ~zero new mechanism: de-hard-code the 3-frame router into a **signature-keyed construction REGISTRY**, plus ONE bounded, precedented label extension (a post-verbal OBJECT slot) so the transitive-motion constructions the corpus **already attests** (PP-goal, PP-location) are mined + rendered instead of discarded. All controls collapse; the held-out backbone generalizes; the gate-first moat holds. The **adjective-based** templates the research gate initially named (predicative-adjective / adj+ability / existential) are an **honest BOUNDARY** carried alongside the GO — named precisely, not forced.

**Runner:** `research/runners/_emerge72_construction_registry_derisk.py` (`--demo` / `--derisk` / `--derisk --seeds ...`)
**CI:** `tests/test_emerge72_construction_registry.py` (10 tests, CPU/numpy, offline)
**Raw:** `research/findings/raw/_emerge72_construction_registry.json`
**Research gate:** `research/findings/2026-07-03-broaden-construction-inventory-research-gate.md` (RANK 1 / EMERGE-72)
**Reuse-by-import; NO `sim/` edit; the gate-first no-confab moat is untouched.**

---

## The headline

The producer now renders **5 distinct constructions** (up from 3), all discovered from the same corpus stream and rendered EXACT on real spikes:

| id | surface | slots (corpus-mined) |
|---|---|---|
| F_MODAL | "the owl can fly" | det:the · SUBJ · func:can · VERB:bare |
| F_INTR | "the penguin walks" | det:the · SUBJ · VERB:3sg |
| F_NEGMOD | "the penguin does not fly" | det:the · SUBJ · func:does · func:not · VERB:bare |
| **C_PPGOAL** (NEW) | "the owl flies to the pond" | det:the · SUBJ · VERB:3sg · func:to · func:the · **OBJ** |
| **C_PPLOC** (NEW) | "the owl flies on the rock" | det:the · SUBJ · VERB:3sg · func:on · func:the · **OBJ** |

The two NEW constructions add **arguments AFTER the verb** (the transitive-motion / Goldberg motion argument-structure construction) — the biggest expressivity jump. The 6-slot PP constructions fit the existing `N_SLOT_POOLS=6` spiking substrate exactly (NO `sim/` edit).

Sample transcript (seed 42, on spikes, gate-first):
```
you> can an owl fly?            broca> the owl can fly              [MODAL;  producer INVOKED]
you> what does a penguin do?    broca> the penguin walks           [INTR;   producer INVOKED]
you> can a penguin fly? [deny]  broca> the penguin does not fly    [NEGMOD; producer INVOKED]
you> where does the owl fly?    broca> the owl flies to the pond   [PPGOAL; producer INVOKED]
you> where does the owl fly?    broca> the owl flies on the rock   [PPLOC;  producer INVOKED]
you> can a zzz fly?             broca> I don't know.               [MOAT;   producer NOT invoked]
```

## 6-seed results

| seed | registered | rendered-exact | render | permuted-corpus | cross-construction | held-out backbone | no-corpus | moat |
|---|---|---|---|---|---|---|---|---|
| 42 | 5 | 5 | 1.000 | 0.000 (n 0) | 0.000 | 1.000 | 0 | 0 |
| 43 | 5 | 5 | 1.000 | 0.000 (n 0) | 0.000 | 1.000 | 0 | 0 |
| 44 | 5 | 5 | 1.000 | 0.000 (n 0) | 0.000 | 1.000 | 0 | 0 |
| 100 | 5 | 5 | 1.000 | 0.000 (n 0) | 0.000 | 1.000 | 0 | 0 |
| 101 | 5 | 5 | 1.000 | 0.000 (n 0) | 0.000 | 1.000 | 0 | 0 |
| 102 | 5 | 5 | 1.000 | 0.000 (n 0) | 0.000 | 1.000 | 0 | 0 |

**GO gates, all met, every seed:** ≥5 constructions rendered EXACT (render 1.000) · PERMUTED-CORPUS collapses (0.000, margin ≥ 0.30) · CROSS-CONSTRUCTION collapses (0.000) · NO-CORPUS empty (0 registered) · HELD-OUT-CONSTRUCTION shared det+subj+verb backbone generalizes (1.000) · gate-first MOAT 0 productions on abstains.

## The mechanism (the ONE residual the gate pinned, closed)

The mine (`_emerge64:262-298 mine_inventory`) was **already construction-agnostic** — it keys constructions by their ordered role-type signature (`_emerge64:197-209 _slot_signature`). The EMERGE-62 corpus stream (`_emerge62:145-181 build_stream`) **already attests ~10 templates**. The producer rendered only 3 because two seams were hard-keyed to the 3 EMERGE frames:
- `_emerge64:325-336 match_inventory_to_frames` mapped mined signatures ONLY against `{_frame_signature(fr): fr for fr in FRAME_NAMES}` (`:328`) — any other mined construction was **silently discarded**.
- `_emerge59:316-329 decision_from_emerge` emitted ONLY F_MODAL / F_INTR / F_NEGMOD.

EMERGE-72 replaces both with a general **`ConstructionRegistry`**: `{mined-signature → construction id + render route}` — any dominance-clearing mined construction gets a stable id, and a general `decision(gate, construction, subject, verb, obj)` selector (the Dominey-Hinaut construction-router) replaces the 3-way branch. The 3-frame path is preserved byte-identical (EMERGE-59..71 defaults untouched; all 106 prior EMERGE CI tests still pass).

**The one bounded, precedented label extension.** `_emerge64:label_sentence` skips ALL post-verbal content (`:170-174`), so it can mine only single-content-verb constructions. `label_sentence_ext` (this file, ADDITIVE — EMERGE-64 untouched) admits exactly ONE post-verbal CONTENT/OBJECT slot with its closed-class scaffold (PP preposition + determiner) — the `argstructure_composer.FRAME_LEXICON` motion frame + the `_bucketB_corpus_mined_frames` verb-frame precedent (Goldberg motion/caused-motion). The object filler flows through the SAME gated-decision path as subject/verb; the OBJECT slot is spelled by the SAME A→W read-out. This is RANK-2-precedented and cheap, exactly the gate's characterization.

**One implementation note (found + fixed during the de-risk):** the initial `RegistryProducer.emit` override bypassed the EMERGE-61 inter-utterance wash-out, so the 5-/6-slot constructions' two lowest-primacy adjacent slots occasionally swapped in the spiking rate-ranking (render 0.95-0.975, the known EMERGE-59/61 spike-frequency-adaptation tail — `does`/`not` or `the`/`obj` swap on ~1/8 facts). Applying the inherited `self._reset_substrate()` (EMERGE-61's validated position-independence wash-out) at the top of `emit` restored render to 1.000 on every seed. This is a **reuse of the already-validated mechanism**, not a gate relaxation — the gate still requires bit-exact order on every fact.

## The anti-cheats (all collapse, decisively)

- **PERMUTED-CORPUS (input-destruction):** shuffle each exemplar's word order before mining → **0 constructions registered**, render 0.000, every seed. The shuffle-invariant bag key (EMERGE-64b, extended to the OBJ role) dilutes every construction's dominant ordering below the dominance threshold, so nothing is confidently mined. ⇒ the broadening is **corpus-ORDER-derived**, not host-smuggled.
- **CROSS-CONSTRUCTION (form-specificity, Dominey-Hinaut):** render construction A's fact through construction B's mined slot structure → **0.000** exact. The constructions are genuinely form-specific.
- **HELD-OUT-CONSTRUCTION:** hold one construction out of the mining corpus (drop its exemplars by ground-truth signature) → its SHARED det+subj+verb backbone is recovered from the OTHERS (1.000). The distinctive PP scaffold (func:to/on + OBJ) is the named residual (the transitive-motion scaffold generalizes across the two PP constructions; the specific preposition is per-construction).
- **NO-CORPUS:** empty stream → 0 registered.
- **MOAT (gate-first):** an ABSTAIN never invokes the producer (0 productions on abstains); an ANSWER does (5 invocations over 6 probes in the demo). By construction.

## The honest BOUNDARY carried alongside the GO (named, NOT hidden)

The research gate's RANK-1 candidate named three ADJECTIVE-based templates — predicative-adjective "the owl is big", adj+ability "the big owl can fly", existential "it is a big owl". **These do NOT cleanly mine from THIS corpus's distributional statistics**, and I did NOT force them:

- The corpus's adjectives (`big`, `fast`, `grey`, `tall`, …) appear across MANY frames (conjunction / existential / adj+ability / predicative) → **high frequency AND high context-coverage** → EMERGE-62's Goldilocks discovery (`freq-pct ≥ 0.90 AND cover-pct ≥ 0.60`) labels **2-4 of them CLOSED-class per seed** (verified: seed 42 `{big, fast}`, seed 44 `{small, fast, red, tall}`, seed 101 `{red, grey, warm, cold}`, …).
- The PPMI-content cue does NOT separate adjectives from true function words (adjective content-prank `[0.01, 0.26]` overlaps function-word `[0.00, 0.45]`).

So an adjective's CONTENT role is **statistically ambiguous with the closed class** here — the copular-predicative + existential constructions are the precisely-named residual. `label_sentence_ext` correctly **skips** "the owl is big" (an adjective sitting as the sole predicate is not cleanly labellable), rather than forcing a wrong label. This is exactly the gate's "if the copular/existential role shape doesn't cleanly mine → honest BOUNDARY" branch.

**This is NOT a wall.** The next single distributional signal is the ADJECTIVE's OWN attributive pre-nominal signature: an adjective sits immediately left of a content noun with which it has selectional affinity ("grey owl", "big fish") — a phrase-internal cue the closed class lacks. That is **EMERGE-73**'s argument-structure / attributive-modifier labelling (a third distributional cue, à la Yang-Getz phrase-boundary alignment). Named, not gated, not forced.

## Honest scope

- BROADENS the **bounded, corpus-attested, router-selected** construction inventory from 3 to 5 — NOT open prose (R4, the separate deferred wall; a from-scratch spiking LM is ~4 orders too small, `2026-05-07-Phase-2.3a-NEGATIVE`).
- The A→W SPELL stays the token surface for THIS de-risk. The fully-spiking A→W of the NEW content words (the OBJECT nouns) is the EMERGE-67/68-style follow-on (its own spiking validation is `concept_speak_demo`, 100% multi-seed). This de-risk validates the CONSTRUCTION-inventory broadening (the registry + the render on spikes), not the spell.
- The corpus mining is offline syllabus prep (BRAIN-BASED-ONLY compliant — like rendering a retinal image the neural retina reads); the structure is rendered on REAL spikes; the gate-first moat is untouched (0 productions on abstains, by construction).

## Regression / integrity

- **NO `sim/` edit** anywhere (`git status` shows only the two new EMERGE-72 files added; no prior runner or `sim/` file modified).
- **EMERGE-59..71 defaults preserved** — the 3-frame path is byte-identical: EMERGE-59/60/61/62/62b/63/64/64b/65/66/67/68/69/70 CI **all pass** (61 + 40 + 5 = 106 prior EMERGE tests green; 3 skips are pre-existing GPU-render-skip-if-no-ckpt).
- EMERGE-72 CI: 10 tests pass.

## Sources / precedents

- Hinaut & Dominey (PLoS ONE 2013; Brain & Language 2015) — the fronto-striatal reservoir reads thematic roles from the closed-class ORDER/POSITION, **generalizes to NEW constructions**, and production = SELECTING the construction to express predicate + thematic roles (the construction-router warrant).
- Tomasello usage-based / Goldberg argument-structure construction grammar — the inventory grows by abstracting MORE usage-based constructions (transitive/motion/caused-motion as separate item-based constructions).
- Catalog G.12 (Kandel 6e Ch 55) — Broca as the grammatical-frame + closed-class engine; G.07/H.19 (pre-SMA/SMA serial order) — the FrameCQ slot-order substrate.
- Project precedents: `_emerge64_mine_slot_inventory_derisk.py` (`_slot_signature`/`mine_inventory`/`match_inventory_to_frames:328`), `_emerge59_spiking_broca_frame_slots_derisk.py` (`FRAMES`/`FrameSlotCQ`/`decision_from_emerge:316`/`realize_slot`), `_emerge62_discover_function_words_derisk.py` (`build_stream:145`), `_emerge63` (order + `CorpusOrderFrameSlotCQ`), `_emerge61` (the inter-utterance wash-out), `_emerge65` (the composed pipeline + anti-cheats), `argstructure_composer.FRAME_LEXICON`, `_bucketB_corpus_mined_frames_derisk.py` (corpus verb-frame mining, GO).
