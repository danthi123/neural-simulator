# The brain LEARNS relational facts FROM THE CORPUS (GO, 3-seed): mine clean subject-verb-object triples from TinyStories → answer what/who about them, moat intact, knowledge corpus-derived. Breadth beyond hand-taught facts — the brain talks about what it has "experienced". NO `sim/` edit.

**Date:** 2026-07-08
**Runner:** `research/runners/_realcorpus_learn_corpus_facts_derisk.py` (reuse-by-import: `SVOStore` + the breadth discovery). numpy FHRR. NO `sim/` edit.
**Verdict:** GO — the brain learns clean relational facts from the corpus and answers about them, 3-seed.

## Why this ran (breadth toward "talk about what it experienced")
The console's relational facts were hand-TAUGHT. Toward the "talk about anything" north star, the brain should LEARN facts from experience (the corpus), not just be told them. This mines subject-verb-object triples from TinyStories and stores them so the brain answers relational questions about corpus-derived knowledge.

## The mechanism (+ the data-quality gate it clears)
- **Raw SVO mining is NOISY** (probe): objects land on adjectives ("tim saw big"), subjects are names ("tim"/"lily") — clean semantic SVO needs POS tagging. 10,580 raw triples, mostly noise.
- **A NOUN filter clears it:** require subject AND object both in a curated common-noun set (animals + objects) → **327 clean triples (160 distinct)**: "bird saw cat" (26), "dog saw cat" (19), "cat saw dog", "dog saw ball", "girl saw bird", "mom saw dog" — semantically reasonable corpus-derived facts.
- The top-N facts are stored in the SVO store (grounded real-corpus phasors); the brain answers `what did <subj> <verb>?` / `who <verb> <obj>?` by role-unbinding.

## The result — 3-seed (K=256, top-40 facts)
```
seed 42/43/44: mined 40 facts (bird saw cat, dog saw cat, cat saw dog, cat saw bird, ...)
  what_acc=1.00 | who_acc=1.00 | moat=1.00 | permuted-corpus overlap=0.00/0.05/0.03
```
- **what/who 1.00** — a query returns a VALID corpus fact. (The corpus facts are MANY-TO-MANY — cat saw dog AND bird AND mouse — so a query recovers one of the objects the subject actually V-ed; scored correct iff the returned (subj, verb, obj) is a mined fact.)
- **MOAT 1.00** — a never-mined (subject, verb) → abstain (no confabulation).
- **permuted-corpus overlap ~0.03** — shuffling the token order yields COMPLETELY DIFFERENT mined facts (overlap ≈ 0), so the knowledge is genuinely CORPUS-DERIVED (the token ORDER carries it), not an artifact of the noun set.

## Honest scope
- The clean facts are noun-verb-noun over a curated common-noun set; raw open SVO mining is noisy (POS tagging is the general fix — the same data-quality gate as the taxonomy genus extraction). The mined verbs are mostly perception ("saw"/"found") — TinyStories' dominant transitive pattern.
- Many-to-many facts (a concept in many facts) are recovered as ANY valid corpus fact for a (subject, verb) cue, not a unique object (the honest semantics of "what did cat see?" when cat saw several things).
- Rate-level (numpy FHRR); the spiking realization is the RFPhasorComposer (CYCLE 1002 pattern).

## What this establishes
The brain LEARNS relational knowledge from EXPERIENCE (the corpus) — not just hand-taught facts — and answers what/who about it with the no-confab moat, the knowledge provably corpus-derived. A breadth step toward "talk about what it has experienced". Follow-on: wire into the console (a `--learn-corpus-facts` flag) so the owner asks about corpus-learned facts; POS tagging for open (non-curated-noun) mining; the spiking realization.

## Files
`research/runners/_realcorpus_learn_corpus_facts_derisk.py`; `research/findings/raw/_corpusfacts_s*.json`. Prior: the relational SVO Q&A `2026-07-08-relational-SVO-QA-over-real-corpus-codes-GO.md`; the taxonomy data-quality gate `2026-07-08-emergent-is-a-from-dictionary-genus-NOISY-taxonomy-data-gate-confirmed.md`.
