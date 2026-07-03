# EMERGE-73 — CLOSE the adjective boundary EMERGE-72 named: the adjective's ATTRIBUTIVE PRE-NOMINAL POSITION reclassifies it OPEN, so the self-organized spiking producer ADMITS the adjective constructions (>= 7 total) — GO (6-seed)

**Date:** 2026-07-03
**Verdict:** **GO, 6/6 seeds.** The honest ADJECTIVE boundary EMERGE-72 named is **SURPASSED** by adding the adjective's OWN third distributional cue — its **ATTRIBUTIVE PRE-NOMINAL POSITION** (the `DET _ NOUN` slot). This corpus's adjectives are high-FREQUENCY AND high-context-COVERAGE, so EMERGE-62's 2-cue Goldilocks discovery mislabels 2–4 of them CLOSED-class per seed; the position cue **reclassifies them OPEN** (F1 1.00, ZERO true function words promoted), so the producer now MINES + RENDERS the corpus-attested attributive ("the big owl can fly") + predicative ("the owl is grey") constructions — **7 distinct constructions total** (the 5 from EMERGE-72 + 2 adjective ones), all rendered EXACT on real spikes. Every input-destruction control collapses; the held-out backbone generalizes; the gate-first no-confab moat holds.

**Runner:** `research/runners/_emerge73_adjective_position_cue_derisk.py` (`--demo` / `--derisk` / `--derisk --seeds ...`)
**CI:** `tests/test_emerge73_adjective_position_cue.py` (11 tests, CPU/numpy, offline)
**Raw:** `research/findings/raw/_emerge73_adjective_position_cue.json`
**Boundary closed:** `research/findings/2026-07-03-emerge72-construction-registry-GO.md:71-80` (the named RANK-1 next signal)
**Reuse-by-import; NO `sim/` edit; the gate-first no-confab moat is untouched.**

---

## The boundary EMERGE-72 named (and how EMERGE-73 closes it)

EMERGE-72 broadened the producer from 3 to 5 constructions but carried an honest boundary: the ADJECTIVE constructions do NOT cleanly mine because THIS corpus's adjectives (`big`/`small`/`fast`/`tall`/`grey`/…) are **high-FREQUENCY AND high-context-COVERAGE**, so EMERGE-62's 2-cue Goldilocks discovery (`freq-pct >= 0.90 AND cover-pct >= 0.60`) **mislabels 2–4 of them CLOSED-class per seed**. With the adjectives wrongly in the closed class, `label_sentence_ext` correctly SKIPS the adjective sentences rather than mislabelling — an honest boundary, not forced. The EMERGE-72 finding named the exact next signal: *"the ADJECTIVE's OWN attributive pre-nominal signature."*

**EMERGE-73 adds exactly that cue.** An adjective is OPEN-class (content) BUT positionally constrained: it reliably sits in the ATTRIBUTIVE slot BETWEEN the determiner and the head noun (`the BIG owl`: `DET adj NOUN`). A true closed-class word has a different position profile — a true determiner (`the`, `a`) sits at the NP ONSET (not itself preceded by another closed word inside the NP), whereas an adjective sits INSIDE the NP, preceded by the determiner AND followed by the content noun. So the discriminator is the **attributive-pre-nominal rate**:

> `attribscore[w]` = fraction of `w`'s occurrences where `w` is (i) immediately PRECEDED by a closed-class word AND (ii) immediately FOLLOWED by a content noun (subject/object) — the `DET _ NOUN` attributive slot.

A Goldilocks-CLOSED word with `attribscore >= TP_ATTRIB (0.50)` is an ADJECTIVE → reclassified OPEN. The reclassification is **ASYMMETRIC + SAFE**: it only ever PROMOTES a Goldilocks-CLOSED word to OPEN — it never demotes an already-open word or touches a true determiner. This is the **inverse-position cue** to EMERGE-62b's function-word cue (which picks out the closed class by phrase-edge / pre-content position; EMERGE-73 picks out the CONTENT word whose position is constrained inside the NP).

## The empirical separation (why a fixed threshold works, every seed)

Measured `attribscore` on the controlled stream (per-seed), among the words 2-cue Goldilocks called CLOSED:

| word class | attributive rate | example |
|---|---|---|
| **adjectives** (mislabelled CLOSED) | **0.68 – 0.74** | `big` 0.74, `fast` 0.71, `grey` 0.69, `warm` 0.70 |
| true determiner `a` | **0.36** | (from "and a nest" / "is a big") |
| true determiner `the` | 0.26 | |
| auxiliaries/copula/prepositions (`is`,`can`,`does`,`to`,`on`,`in`,`and`,`it`,`not`) | 0.00 – 0.04 | |

The gap between the highest true-function-word rate (`a` at ~0.36) and the lowest adjective rate (~0.68) is clean and consistent across all 6 seeds, so the pre-registered `TP_ATTRIB = 0.50` (chosen ONCE on seed-42, then FROZEN) separates them cleanly every seed.

## The two adjective constructions this admits (>= 7 total)

| id | surface | slots (corpus-mined) |
|---|---|---|
| F_MODAL | "the owl can fly" | det:the · SUBJ · func:can · VERB:bare |
| F_INTR | "the penguin walks" | det:the · SUBJ · VERB:3sg |
| F_NEGMOD | "the penguin does not fly" | det:the · SUBJ · func:does · func:not · VERB:bare |
| C_PPGOAL | "the owl flies to the pond" | det:the · SUBJ · VERB:3sg · func:to · func:the · OBJ |
| C_PPLOC | "the owl flies on the rock" | det:the · SUBJ · VERB:3sg · func:on · func:the · OBJ |
| **C_ATTRIB** (NEW) | "the big owl can fly" | det:the · **ADJ** · SUBJ · func:can · VERB:bare |
| **C_PRED** (NEW) | "the owl is grey" | det:the · SUBJ · func:is · **ADJ** |

The 5-slot `C_ATTRIB` fits the existing `N_SLOT_POOLS = 6` spiking substrate exactly (**NO `sim/` edit**).

Sample transcript (seed 42, on spikes, gate-first):
```
you> can an owl fly?            broca> the owl can fly           [MODAL;  producer INVOKED]
you> what does a penguin do?    broca> the penguin walks         [INTR;   producer INVOKED]
you> can a penguin fly? [deny]  broca> the penguin does not fly  [NEGMOD; producer INVOKED]
you> where does the owl fly?    broca> the owl flies to the pond [PPGOAL; producer INVOKED]
you> where does the owl fly?    broca> the owl flies on the rock [PPLOC;  producer INVOKED]
you> what can the big owl do?   broca> the big owl can fly       [ATTRIB; producer INVOKED]  <- NEW
you> what is the owl like?      broca> the owl is grey           [PRED;   producer INVOKED]  <- NEW
you> can a zzz fly?             broca> I don't know.             [MOAT;   producer NOT invoked]
```

## 6-seed results

| seed | registered | rendered-exact | adj rendered | adj-reclass F1 | promoted (OPEN) | POS-SHUFFLE adj | FREQ-ONLY adj | no-corpus | held-out | moat |
|---|---|---|---|---|---|---|---|---|---|---|
| 42 | 7 | 7 | ✔✔ | 1.00 | big, fast | 0 (5 constr.) | 0 (5 constr.) | 0 | 1.000 | 0 |
| 43 | 7 | 7 | ✔✔ | 1.00 | grey, tall, wet | 0 | 0 | 0 | 1.000 | 0 |
| 44 | 7 | 7 | ✔✔ | 1.00 | fast, red, small, tall | 0 | 0 | 0 | 1.000 | 0 |
| 100 | 7 | 7 | ✔✔ | 1.00 | big, tall, wet | 0 | 0 | 0 | 1.000 | 0 |
| 101 | 7 | 7 | ✔✔ | 1.00 | cold, grey, red, warm | 0 | 0 | 0 | 1.000 | 0 |
| 102 | 7 | 7 | ✔✔ | 1.00 | big, fast, slow, tall | 0 | 0 | 0 | 1.000 | 0 |

**GO gates, all met, every seed:** ≥7 constructions rendered EXACT (render 1.000, both adjective constructions incl.) · adjective reclassification F1 1.000 (all mislabelled adjectives OPEN, 0 true function words promoted) · POSITION-SHUFFLE collapses (0 adjectives reclassified → 0 adjective constructions, falls back to the EMERGE-72 5) · FREQUENCY-ONLY reproduces the EMERGE-72 state (5 constructions, 0 adjective) · NO-CORPUS empty · HELD-OUT-CONSTRUCTION backbone 1.000 · gate-first MOAT 0 productions on abstains.

## The anti-cheats (all collapse, decisively)

- **POSITION-SHUFFLE (input-destruction, LOAD-BEARING):** scramble each sentence's word order BEFORE computing the attributive statistic → the `DET _ NOUN` cue is destroyed → **0 adjectives reclassified** → **0 adjective constructions mined** → the registry falls back to the EMERGE-72 5 constructions. ⇒ the broadening is genuinely **corpus-POSITION-derived**, not a spurious lift or host-smuggled label.
- **FREQUENCY-ONLY (the EMERGE-62 2-cue baseline = the EMERGE-72 state):** skip the position reclassification → the adjectives stay CLOSED → **0 adjective constructions**, the EMERGE-72 5-construction state (still renders all 5). ⇒ the position cue is exactly what ADDS the adjective constructions (single-variable isolation).
- **NO-CORPUS:** empty stream → no statistics → no reclassification → 0 registered.
- **HELD-OUT-CONSTRUCTION:** hold ONE construction out of the mining corpus (drop its exemplars by ground-truth signature) → its SHARED `det+subj+verb` backbone is recovered from the OTHERS (1.000).
- **MOAT (gate-first):** an ABSTAIN never invokes the producer (0 productions on abstains); an ANSWER does (7 invocations over 7 answer probes in the demo). By construction.

## The mechanism (what was added, minimally)

Two ADDITIVE pieces on top of EMERGE-72 (`_emerge72_construction_registry_derisk.py` untouched; the 5 EMERGE-72 constructions preserved byte-identically):

1. **`compute_attributive_stats` + `reclassify_adjectives`** (`_emerge73:...`) — the attributive-pre-nominal statistic over EMERGE-62b's sentence-segmented stream + the asymmetric reclassification. Reuses EMERGE-62's `discover_closed_class` for "preceded by a closed word" (the closed set is DISCOVERED, not host-listed) and the open-class noun lexicon for "followed by a content noun" (the Redington immediate-neighbour role).
2. **`label_sentence_adj`** — an ADDITIVE extension of EMERGE-72's `label_sentence_ext` that admits a pre-nominal ADJ slot (`DET adj NOUN`) + a predicative post-copular ADJ slot (`DET SUBJ COP adj`), keyed by the DISCOVERED adjective set. The `AdjConstructionRegistry` reuses EMERGE-72's `RegistryProducer` (spiking order read-out over the EMERGE-61 wash-out) + `RegistryBrocaProducer` (gate-first moat) verbatim; the ADJ slot is spelled by the same A→W read-out as OBJ/SUBJ.

## Honest scope

- BROADENS the **bounded, corpus-attested, router-selected** inventory from 5 to 7 by adding the adjective's attributive-position cue — NOT open prose (R4, the separate deferred wall; a from-scratch spiking LM is ~4 orders too small, `2026-05-07-Phase-2.3a-NEGATIVE`).
- The A→W SPELL stays the token surface for THIS de-risk. The fully-spiking A→W of the NEW adjective content words is the EMERGE-67/68-style follow-on (its own spiking validation is `concept_speak_demo`, 100% multi-seed). This de-risk validates the CONSTRUCTION-inventory broadening (the reclassification + the render on spikes), not the spell.
- The corpus mining is offline syllabus prep (BRAIN-BASED-ONLY compliant — like rendering a retinal image the neural retina reads); the structure is rendered on REAL spikes; the gate-first moat is untouched (0 productions on abstains, by construction).

## Regression / integrity

- **NO `sim/` edit** anywhere (`git status sim/` clean; only 3 new EMERGE-73 files added).
- **EMERGE-59..72 defaults preserved** — no prior runner/`sim/` file modified. CI re-run green: EMERGE-59/62/62b/63/64/72 (50 tests) + EMERGE-65/66/67/68/69/70 (39 passed, 4 pre-existing GPU-render skips) all pass.
- EMERGE-73 CI: 11 tests pass.

## Sources / precedents

- **Tomasello (usage-based construction grammar) / Goldberg (argument-structure grammar)** — the attributive construction: an adjective abstracts into the pre-nominal MODIFIER slot inside the NP, an item-based construction distinct from the closed class.
- **Yang & Getz (2026, arXiv 2601.21191)** — the 3rd universal property of function words = phrase-boundary / syntactic-position alignment; EMERGE-73 applies the INVERSE cue to isolate the positionally-constrained CONTENT word (the adjective).
- **Redington, Chater & Finch (1998); Cartwright & Brent (1997)** — distributional POS induction from immediate-neighbour role profiles.
- **Catalog G.12** (Kandel 6e Ch 55 pp 1382–1384) — Broca open/closed dissociation (retained noun selection, lost function-word use); the neurolinguistic basis for a separable open/closed statistical population.
- **Research gate:** the EMERGE-72 finding's named RANK-1 next signal (`2026-07-03-emerge72-construction-registry-GO.md:80`).
- **Project precedents:** `_emerge72_construction_registry_derisk.py` (the ConstructionRegistry + spiking RegistryProducer + gate-first moat), `_emerge62_discover_function_words_derisk.py` (`build_stream:145` / `discover_closed_class` / the lexicons), `_emerge62b_function_words_position_cue_derisk.py` (`sentences_from_controlled` — the phrase-boundary front end), `_emerge59_spiking_broca_frame_slots_derisk.py` (`N_SLOT_POOLS`/`emerge_v3`/the FrameSlotCQ substrate), `_emerge63` (corpus-order slot CQ), `_emerge64` (`_verb_inflection`).
