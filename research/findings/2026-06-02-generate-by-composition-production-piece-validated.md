# Generate-by-composition: the missing PRODUCTION piece validated (principle) -- honest assessment -- 2026-06-02

## What was tested
Grounded in the primary source (Kandel Ch 55 dual-stream production; Hagoort MUC; read directly from the local
Kandel PDF, not just the catalog): the brain produces language as retrieve distributed concepts -> assemble
(bind/unify) -> SEQUENCE to ordered output. The project has the first two (validated VSA concept codes +
bind/unbind); the missing piece is the ordered SEQUENCE READ-OUT. This probe
(research/findings/raw/_generate_by_composition_probe.py, numpy) tests it: compose a meaning (role (x) filler
superposition) -> for each role in grammatical order, unbind + cleanup -> emit word -> ordered sentence.

## Result
Correct-novel-sentence = **1.000 multi-seed** (seeds 42/43/44) at sentence lengths 3, 4, 5; scrambled-role
control 0.000. Novel meanings (role-filler combinations never composed together) are produced correctly IN
ORDER with ZERO training. A memorizer (lookup over seen triples) is 0.000 on novel by construction.

## Honest framing (scrutinise the PASS)
This RESOLVES the principle but is an EASY validation: 1.000 is EXPECTED -- unbind+cleanup recovering bound
fillers is the known VSA property, re-confirmed in the production framing. It establishes the production read-
out is SOUND and generalises to novel meanings for free (unlike the overfit next-token LM), but it did not test
anything uncertain. The genuinely HARD parts of generation remain untested:
- CONTENT SELECTION / planning (deciding WHAT to say) -- here the meaning was GIVEN. This is where tiny-LLM
  intelligence actually lives (contextually-appropriate continuation); the biology-faithful equivalent
  (PFC/frontal response planning + retrieval) is the real frontier and is unsolved.
- VARIABLE / flexible structure (different grammatical orders, optional/recursive roles).
- The SPIKING in-substrate version (this is numpy; the project has the spiking bind/unbind to reuse).
- FLUID conversational integration + SCALE (vocabulary, varied utterances).

## Where the conversational pieces now stand (biology-faithful)
- Comprehension (parse input -> meaning): validated (Hebbian conjunctive parser, voice-invariant).
- Memory + composition (retrieve/compose meaning): validated (VSA bind/unbind, KB >=30 facts, 320 concepts).
- PRODUCTION (meaning -> ordered output): validated here (principle).
- CONTENT SELECTION / planning (what to say): NOT addressed -- the real frontier.
- Integration + scale: partial (chat REPLs exist; not a fluid full loop at scale).

So the project has the conversational COMPONENTS biology-faithfully; the frontier for tiny-LLM-like
conversation is CONTENT SELECTION + INTEGRATION + SCALE, not a missing low-level mechanism. This refines the
"missing structure" question: it's now a higher-level (planning/control -- Hagoort's "Control"; PFC dialogue
management) + integration problem, on validated biological components.

## Next
(1) Spiking in-substrate generate-by-composition (reuse _insubstrate_relational_memory_probe bind/unbind +
ordered read-out) -- faithfulness. (2) Design a biology-faithful conversational LOOP integrating the validated
components (comprehend -> retrieve/compose -> produce), with simple content selection (answer/respond from KB),
as the tangible artifact. (3) Then the hard frontier: biology-faithful CONTENT SELECTION / dialogue planning
(PFC/frontal control over what to compose-and-produce). Brainstorm/design before building. Honest, biology-
faithful, both remotes.
