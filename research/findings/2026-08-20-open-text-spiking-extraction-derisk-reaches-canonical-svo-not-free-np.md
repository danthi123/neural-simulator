---
type: finding
status: live
date: 2026-08-20
mechanism: open-text-spiking-extraction
lane: D-pragmatics
seeds: [42]
instrument: BridgeParser (spiking position x voice -> role read-out) run over 13 clauses from 12
  hand-authored Qwen-style prose items, feeding the SAME entailment layer as
  2026-08-20-open-text-moat-verifier-derisk-extract-then-entail-feasible.md
runner: research/runners/_open_text_spiking_extraction_derisk.py
artifacts:
  - research/runners/_open_text_spiking_extraction_derisk.py
  - research/findings/raw/_open_text_spiking_extraction_derisk.json
---
# Open-text spiking-extraction de-risk: BridgeParser reaches canonical 3-word SVO, not free noun phrases

Artifact: research/findings/raw/_open_text_spiking_extraction_derisk.json

**One line.** The prior de-risk (`_open_text_moat_verifier_derisk.py`, 2026-08-20) proved the entailment layer
but used host regex to build the (agent, action, patient) triple. This file swaps ONLY that step for the actual
on-brain parser the production 'rf'-composer chat path builds (`BridgeParser`,
`research/runners/brain_conversational_agent.py:28`, invoked from `hear()` at :672 whenever the composer has no
`hear()` of its own). Result on 12 items / 13 clauses: **extraction coverage 9/11 = 0.818** <!--derived--> on assertion clauses
(2 clauses were hedge/opinion and correctly never sent to extraction). On the 9 clauses that DID extract, the
reused-unchanged entailment layer scored **precision=1.0, recall=1.0, F1=1.0** (5/5 ungrounded caught, 4/4
grounded correctly passed, catching confabulation for false-attribution, wrong-patient, and role-reversed cases,
plus one explicit same-SVO-opposite-polarity negation pair). Of 6 gold-false clauses total, **5 were caught
ungrounded and 1 slipped through as unparsed-and-suppressed** (never asserted, but also never flagged).

## What BridgeParser actually is, and what it is not
`BridgeParser` is a genuinely spiking, genuinely learned component: 6 conjunction units (word-position x voice)
project onto 3 Hebbian-trained role ensembles on a 126-neuron Izhikevich `SimulationBridge`; `parse(words, voice)`
drives one position's conjunction unit alone and reads out which role ensemble fires hardest
(`brain_conversational_agent.py:147-167`). It carries **zero lexical/verb knowledge** — role assignment is 100%
positional. Verified interactively before building this set: `parser.parse(['the','big','apple'])` returns
`{'agent':'the','action':'big','patient':'apple'}` — a confident, wrong answer, because "confident" here only
means "position 1's ensemble fired hardest when driven alone," which it always does. `parse()` also hard-asserts
`len(words) == 3`: it has no clause-boundary detector, no verb lexicon, no NP-boundary logic, and (confirmed by
inspection, not a scored item here) no morphological normalization — an inflected form used in a positive
sentence ("orbits") and the bare form the same fact takes under "does not" negation ("orbit") are DIFFERENT store
keys to it, exactly as they would be to a real cortex needing separate learned codes for each surface form.

## The extraction pipeline actually used
Per clause (from `split_clauses`/`is_opinion`, reused unchanged from the prior de-risk): tokenize, remove
STOPWORDS (`{the,a,an,to,that,this,these,those}`, reused unchanged) and a small NEGATORS set
(`{not,doesn't,don't,does,do,didn't,never}`, new to this file, needed because "do"/"don't" were absent from the
plain-negator handling that lived inside the OLD regex extractor). This is a lexical FILTER — it deletes tokens,
it never decides which surviving token plays which role. If exactly 3 content words remain, IN THEIR ORIGINAL
ORDER, they are handed to `BridgeParser.parse` and the role assignment (which content word is subject/verb/object)
is entirely the spiking positional read-out. If not exactly 3, the clause is ABSTAIN-and-SUPPRESS: recorded as
`unparsed`, no triple fabricated, never silently treated as agreeing or disagreeing with anything. Entailment
(`Claim`, `FactStore`, `classify_claim`, `AFFIRM`/`NEGATE`) is imported and used byte-unchanged from the prior
de-risk, which itself mirrors the production `routed_composer.ask_yes_no` / `query_patient` abstain-on-unknown /
same-SVO-opposite-polarity-reject semantics.

## The 9 parsed clauses (why they parsed) vs. the 2 unparsed (why they didn't)
Parsed: `"Mercury orbits the sun"`, `"moon orbits the earth"`, `"Newton discovered gravity"`, `"sun orbits the
moon"`, `"Fish breathe air"`, `"Darwin discovered relativity"`, `"Flowers pollinate bees"`, `"Wasps pollinate
flowers"`, `"Wasps do not pollinate flowers"` — every one is already a bare proper-noun-or-plural subject + a
single verb + a bare single-word object, i.e. Qwen prose that happens to already read like a textbook SVO
sentence. Unparsed: `"The Great Barrier Reef is the largest coral reef system in the world"` (multi-word subject,
copula, multi-word predicate nominal — dozens of content words survive the stopword filter, nowhere near 3) and
`"The Eiffel Tower was built in London"` (multi-word subject, passive-voice verb phrase, preposition). These two
are exactly the shape of REAL free-generated assistant prose (definite-article proper-noun phrases, copulas,
passive constructions) — and they are also exactly what a 3-slot positional reader cannot touch. The honest
takeaway: **BridgeParser reaches artificially-canonical SVO, not free noun phrases.** The 0.818 <!--derived--> coverage number on
this set is an OPTIMISTIC upper bound — it was obtained on a hand-built mix that is roughly half deliberately
clean and half deliberately messy; a large real-Qwen sample skewed toward multi-word entities and copulas would
score materially lower, and this file makes no claim about that unmeasured distribution.

## The safety-relevant residual
The `false_claims_slipped_unparsed_suppressed` count (1/6, the Eiffel Tower / London item) is the number that
matters for production risk: it demonstrates the CORRECT failure mode (suppress, never assert) but also that
extraction failure is a coverage hole, not a safety hole — a false claim that fails to extract is silently
dropped, not silently confirmed. It is not caught either. Whether that is acceptable depends on the downstream
policy (does an un-checkable clause get spoken with a hedge, or dropped from the response entirely) — out of
scope here, flagged for whoever wires this into the response pipeline.

## Concrete next mechanism (smallest lever, in order of expected marginal coverage)
1. **Clause-internal determiner/copula stripping is already the extent of host preprocessing here — the next
   lever is NOT more regex.** The single largest coverage gap is multi-word noun phrases (`"The Great Barrier
   Reef"`, `"the Eiffel Tower"`) collapsing to more than one content token. A spiking NP-boundary mechanism
   (analogous to the existing `AttributedBridgeParser`'s adjective+noun binding, `attributed_parser.py`, but
   generalized to bind an arbitrary-length determiner-headed span into ONE role-fillable unit before the
   position x voice read-out) is the natural next de-risk — it is the same population-binding idea already
   validated for `adjective(s) + noun`, just needing to cover proper-noun multi-word spans and copula predicate
   nominals too.
2. **Passive-voice detection is currently HOST-ASSUMED, not measured.** Every clause here was fed `voice="active"`
   by construction; `BridgeParser` DOES support a `voice="passive"` flip (position 0<->2), but nothing in this
   pipeline detects passive surface form ("was built") from the text to select it automatically. That is a second,
   independent, smaller lever (a lexical passive-auxiliary detector feeding the existing `voice` argument, not a
   new parser).
3. **Morphological normalization was NOT needed for the entailment result to come out correct here** (the two
   negation items used a plural subject + invariant verb form specifically to avoid the inflection mismatch), but
   it is a real, separate gap: an affirmative "X orbits Y" and its negated "X does not orbit Y" use DIFFERENT
   store keys (`orbits` vs `orbit`) unless the KB stores both forms or a lemmatizer sits between BridgeParser's
   output and the store lookup. Not exercised as a scored item; documented from direct inspection so it is not
   silently rediscovered later.

## What this does NOT prove
The entailment precision/recall=1.0 result is unsurprising and expected — it is the SAME code path already
validated in the prior de-risk; this file changes only the extractor feeding it. The 0.818 <!--derived--> coverage figure is
over a 12-item hand-built set skewed toward the parser's known strengths (half the items were purpose-built to be
canonical SVO), not a held-out sample of real Qwen output. Before this extraction path is wired into any live
chat/response pipeline, run it on an actual generated-prose sample (not hand-authored) to measure real coverage,
and build lever #1 above first — it is the mechanism most likely to move the number honestly.
