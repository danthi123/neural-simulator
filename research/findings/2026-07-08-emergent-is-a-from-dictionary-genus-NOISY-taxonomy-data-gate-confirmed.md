# Emergent is-a from a public-domain dictionary (Webster's 1913) genus extraction is NOISY (probe, honest research-gate): the archaic text + sense ambiguity derail the genus chains, so it does NOT yield a clean multi-level is-a tree without heavy NLP. Combined with the flat-co-occurrence NEGATIVE and the declined WordNet graph, emergent multi-level taxonomy is DATA-gated on all tractable fronts. NO `sim/` edit.

**Date:** 2026-07-08
**Probe:** inline (fetch Webster's 1913 `data/corpus/websters1913.json`, public-domain, word→definition, 86,036 entries; genus-differentia extraction + recursive chains). Read-only research-gate probe. NO `sim/` edit. NO build.
**Verdict:** the definitional-corpus path to emergent is-a is NOISY — an honest boundary confirming the multi-level-taxonomy data-gate.

## Why this ran (the named next mechanism needs is-a data)
The multi-level-taxonomy NEGATIVE (`2026-07-08-multilevel-taxonomy-generalization-over-real-corpus-clustering-NEGATIVE.md`) diagnosed: flat co-occurrence gives FLAT categories, not a nested is-a hierarchy; the named next mechanism (EMERGE-44/45 stacked pooler + EMERGE-50 Földiák trace) needs an IS-A SIGNAL IN THE DATA, which TinyStories lacks. Per the master directive (boundaries = undiscovered mechanisms; get the data with the signal), the tractable legitimate is-a source is a DEFINITIONAL corpus — extract is-a from "X is a [genus]" definitional syntax (emergent from text, the classic dictionary-hypernym method, NOT the WordNet hypernym graph which the owner declined as a shortcut).

## The probe — Webster's 1913 genus extraction
Direct (1-level) genus is MIXED (~7/12 clean): `dog→quadruped`, `cat→animal`, `bird→vertebrate`, `lion→mammal`, `tree→plant`, `oak→tree`, `horse→quadruped` — but `fish→"counter, used in various games"` (wrong sense — the archaic first sense), `frog→"1"` / `rose→"imp"` (numbered-sense / abbreviation parsing), `robin` absent.

RECURSIVE genus chains (the key test for a nested tree) DRIFT INTO GARBAGE:
```
dog   -> quadruped                 (stops; quadruped's genus not extracted)
cat   -> animal                    (clean, 1 step)
bird  -> blooded                   (WRONG: head noun of "warm-blooded" = the adjective)
oak   -> shrub -> composed -> agitation        (drifts)
tree  -> plant -> business -> position -> method -> classification   (drifts: "plant" has a business sense)
apple -> pyrus                     (the Latin genus name, not a common superordinate)
rose  -> imp -> graft -> portion   (drifts: "imp." = imperfect-verb abbreviation)
```

## The diagnosis
Three compounding failures make the dictionary-genus path noisy: (1) **sense ambiguity** — the first/parsed sense is often not the taxonomic one ("plant"→business, "fish"→game-counter); (2) **archaic text** — abbreviations ("imp."), Latin genera ("Pyrus"), and dated head nouns ("quadruped") that don't nest cleanly; (3) **crude head-noun extraction** — no POS tagging, so adjectives ("blooded") and prepositions derail it. A clean is-a tree from this source would require substantial NLP (POS tagging, word-sense disambiguation, better head-noun/genus parsing) — a large sub-arc with uncertain payoff, and the genus terms themselves are at INCONSISTENT levels (quadruped vs mammal vs vertebrate vs animal all appear as direct genera of animals).

## What this establishes (the honest boundary)
Emergent multi-level taxonomy is DATA-gated on ALL tractable fronts under the project's constraints:
- **Flat co-occurrence** (TinyStories) → no is-a signal (the multi-level NEGATIVE, diagnosed: sibling sub-categories not mutually similar at the member level).
- **Public-domain dictionary genus** (Webster's 1913) → too noisy (this probe: sense ambiguity + archaic text derail the chains).
- **WordNet hypernym graph** → declined as a hand-provided shortcut (the owner directive).
The MECHANISM (EMERGE-44/45 stacked pooler + Földiák trace) exists; the gate is clean is-a DATA, which is not tractably available emergently. This maps exactly where the breadth→knowledge arc's single-level reasoning (inherit + cancel, comprehensively GO) stops and multi-level taxonomy would begin. Surpass paths (future, larger effort): a modern clean-genus definitional corpus + a POS-tagged/sense-disambiguated genus extractor; or a temporal-proximity is-a signal from a taxonomy-structured corpus (Földiák trace). Single-level property reasoning stands unaffected.

## Files
Probe inline (dictionary cached at `data/corpus/websters1913.json`, gitignored/regenerable). Prior: the multi-level NEGATIVE `2026-07-08-multilevel-taxonomy-generalization-over-real-corpus-clustering-NEGATIVE.md`; the taxonomic-is-a-from-definitional-syntax-on-wikitext2 (weak); EMERGE-44/45/50 (the stacked-pooler mechanism awaiting clean is-a data).
