# Stage 1a — corpus-extracted facts: the brain states REAL knowledge (GO)

**Date:** 2026-06-26 | **Deep-knowledge build, Stage 1a** (plan: `2026-06-26-deep-knowledge-brain-fluency-research.md`).

## The gap this closes

The first-chat console worked mechanically but said gibberish ("curry describes pine") because its facts were **uniform-random sampled** — `_make_svo_facts` (`_curriculum_step1_320_real_corpus.py:521-523`: `a=nouns[rand]; v=verbs[rand]; p=nouns[rand]`) never reads the corpus. The brain had a 1,454-word *vocabulary* but no *knowledge*.

## What was built

`research/runners/_corpus_svo_extract.py` — spaCy (3.8.11 + `en_core_web_sm`) dependency-parse extractor: for each `nsubj` of a VERB/AUX, pair with its `dobj`/`dative`/`attr`/`oprd` or a `prep→pobj`, lemma-or-raw matched to the brain's learned vocab, **frequency-ranked + corpus-attested** (each kept fact logged with a source sentence — the anti-cheat). Host-side curriculum preprocessing (legitimate per BRAIN-BASED-ONLY: preparing the syllabus); the brain still stores/recalls/generalizes via spikes/binding.

Console wiring: `first_chat_console.py --facts-json <path>` (new `_load_real_facts`: loads, dedups to one patient per `(agent,action)`/`(action,patient)` for unambiguous cues, builds real absent-cue moat sets — same return shape as `_make_svo_facts`).

## Result — GO

Full TinyStories extraction → **558 attested facts**, e.g. `(boy,go,park)` 169×, `(girl,go,park)` 97×, `(bird,fly,sky)` 63×, `(cat,stick,tree)` 25×, `(bird,see,cat)` 15×, `(dog,take,ball)` 11× — 165 unique-cue after dedup. All meaningful (vs the random gibberish).

Console demo on 60 real facts: **58/60 recall, moat 0 LEAKS.** The CERTAIN channel now states corpus-TRUE facts:
- "what does boy go?" → *"The boy goes park."* (`boy go park`, 169×)
- "what does bird fly?" → *"The bird flies sky."* (`bird fly sky`, 63×)
- "what is head?" → real adjacent facts (*"The lily throws ball. The lily asks friend."*)
- unknown word → abstains. The proposer now recombines REAL arguments.

## F1 surface-morphology polish

The renderer produced naive `verb+"s"` ("boy gos park", "bird flys sky"). Fixed with a **display-layer** polish (`_surface_morphology` in the console: `go→goes`, `fly→flies`, sibilant→`es`, applied to the final paragraph only) — body-level emission, leaving the internally-consistent VERIFY chain untouched (no fact-drop risk). The moat is unaffected.

## Honest residuals

1. **Extraction noise** — a few mis-parses survive (`(spot,go,friend)` = `pobj`-of-"with"; `(rocket,make,fire)` = a quoted hypothetical). The brain is only as truthful as the extractor; mitigated by frequency-threshold + attestation + the moat (a wrong-but-unqueried fact still abstains, never fabricates).
2. **Abstract-topic flagged noise** — the novel/opinion channel still recombines noisily for abstract topics ("head peeks rocket") because the PPMI graph is noise there — but it stays HEDGED ("I'd guess"), never asserted. Frequent-entity topics are clean. The fix is Stage 1b coverage (multi-bridge).

## Next

Stage 1b (multi-bridge concept coverage for the rare/abstract tail; Simple-Wikipedia extraction in flight for a broader encyclopedic base) → Stage 1c (develop-loop syllabus, cumulative/resumable) → Stage 0 (latency: build ~40s + per-turn). The moat held 0-leak throughout.
