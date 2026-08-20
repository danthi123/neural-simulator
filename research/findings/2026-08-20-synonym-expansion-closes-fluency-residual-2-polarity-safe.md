---
type: finding
status: live
date: 2026-08-20
mechanism: open-text-moat-verifier
lane: D-pragmatics
seeds: [42]
seed-waiver: A deterministic ROUTING/lexicon fix + its before/after check — does synonym expansion let a true fluent rewording ("circles" for stored "orbits") be entailed without loosening the polarity/negation reject. The evidence is per-item keep/suppress correctness on a fixed labelled set; single seed is the substrate build seed.
instrument: research/runners/_synonym_expansion_verify_derisk.py — a synonym/lemma expansion in front of the UNCHANGED FactStore lookup (spiking NPHeadBinder extraction + classify_claim comparison unchanged)
runner: research/runners/_synonym_expansion_verify_derisk.py
artifacts:
  - research/findings/raw/_synonym_expansion_verify_derisk.json
external: NO-EXTERNAL-NEEDED — an in-repo lexical-resource fix for residual #2 named by our own 2026-08-20 fluency de-risk (the same host-preprocessing category as the existing HEDGES/STOPWORDS lexicons); no capability wall or paradigm claim.
---
# Fluency residual #2 CLOSED: synonym expansion lets true rewordings entail, without loosening the polarity reject

Artifact: research/findings/raw/_synonym_expansion_verify_derisk.json

**One line.** The fluency de-risk's residual #2: `FactStore` keys on the exact `(agent, action)` string, so a true
fluent rewording "Mercury CIRCLES the sun" extracts a correct spiking triple `(mercury, circles, sun)` but is wrongly
SUPPRESSED because the store only holds `(mercury, orbits)` (recall 0/3 on synonym-faithful). This lands the fix: a
small declared synonym/lemma expansion of the EXTRACTED verb before the store lookup — entailment succeeds iff ANY
expansion matches WITH THE SAME POLARITY. The spiking extraction (NPHeadBinder) and the `classify_claim` polarity
comparison are imported UNCHANGED; only the pre-lookup verb-candidate set is new.

## Before / after (numpy, seed 42; reproduced identically)
| metric | before | after |
|---|---|---|
| recall-on-synonym-faithful | 0/3 | **3/3** — "Mercury circles the sun" etc. now KEPT |
| precision-on-synonym-confab | 3/3 | 3/3 — a synonym of a FALSE claim ("Mercury circles Neptune") still SUPPRESSED |
| negated-synonym faithful ("Fish do not inhale air") | 0/1 | **1/1** KEPT |
| negated-synonym confab ("Whales do not inhale air") | 2/2 | 2/2 — false negation of a true fact still SUPPRESSED |
| regression (plain/passive/hedge/embedded styles) | — | byte-identical |

The negation-guard set is the important half: it proves the expansion did NOT loosen the same-SVO-opposite-polarity
reject — a false negation routed through a synonym is still suppressed. `expand_action_candidates` = a stem-only
lemmatizer + a declared `SYNONYM_LEMMA_MAP` (circle→orbit, fertilize→pollinate, inhale→breathe), re-inflected to
bare and +s forms; `classify_claim_synonym_aware` runs the UNCHANGED polarity comparison over the candidate set.

## Scope / residual
Closed for the declared 3-verb lexicon without loosening polarity. The remaining residual (out of scope, unchanged):
this is a small CLOSED lexicon, not general lemmatization / a learned entailment — a synonym verb NOT in the map
still fails SAFE (defaults to SUPPRESS, never fail-open). Generalizing to a learned/embedding synonym set is the next
lever; the fail-safe direction means it is a coverage cost, never a safety hole. During the build the agent caught +
fixed a real lemmatizer bug ("circles" mis-stemmed to "circl"). With the hedge-bypass fix (dd1b76b6) + this, two of
the three fluency residuals are closed; only reporting-clause segmentation ("Scientists confirmed that X") remains
before widening Qwen is both safe and fluent. Wiring the corrected routing + expansion into the LIVE verifier is the
integration step (#99). (Agent-built, independently re-run + reproduced.)
