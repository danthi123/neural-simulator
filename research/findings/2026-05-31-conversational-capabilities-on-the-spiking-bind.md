# Conversational capabilities on the spiking bind (2026-05-31)

Built directly on the validated spiking compositional bind + relational fact-memory (finding
`2026-05-31-composition-in-spiking-substrate-SYNTHESIS.md`), without touching the recognition
front-end (which is the real vocabulary ceiling -- see `_vocab_scaling_locus_note.md`). Each
capability stays within the composition layer's strengths (16-word vocab, <=6 bindings/structure)
and is multi-seed validated in spiking.

## Capabilities (all reuse-by-import, all spiking, multi-seed)

1. **End-to-end learned syntactic understanding** (`_insubstrate_parser_bind_e2e_probe.py`) --
   the Hebbian-learned parser assigns roles, the bind stores the sentence, a relational query
   extracts the agent VOICE-INVARIANTLY. Multi-seed (42,43,44): parse 6/6, voice-invariant 1.000,
   scrambled-parse control 0.000. "dog chases cat" (active) = "cat is chased by dog" (passive).

2. **Question-answering** (`_insubstrate_qa_probe.py`) -- wh-questions over an SVO knowledge base:
   "who chases cat?" (-> agent), "what does dog chase?" (-> patient), "what does dog do to cat?"
   (-> action). A wh-word marks the query slot; the other content words are cues; a multi-cue
   relational match finds the fact and reads the query role. Multi-seed (42,43,44): QA 1.000/1.000/
   0.900 (mean 0.967; all-3-questions-correct per trial), unknown-question control -> none (1.000).

3. **Persistent knowledge base across sessions** (`_insubstrate_persistent_kb_probe.py`) -- store
   facts, persist the bound vectors, reload in a FRESH substrate, add new facts, answer across the
   accumulated KB. Multi-seed 3/3: session-1 facts SURVIVE the reload (no forgetting -- by separate-
   fact construction a new fact cannot disturb prior ones) AND the session-2 fact answers. This is
   the artificial-life / continual-learning premise realized on validated pieces.

4. **Negation + yes/no questions** (`_insubstrate_negation_probe.py`) -- "dog does NOT chase cat" =
   the SVO bind + a POLARITY role bound to a NEGATE filler (K=4, within capacity); "does dog chase
   cat?" finds the fact and unbinds POLARITY -> yes/no. Multi-seed (42,43,44): yes/no 1.000, control 1.000 -- RESOLVES.
   Insight: negation is an explicit bound POLARITY ENSEMBLE (a distinct tag), not the absence of a
   fact -- consistent with separate-ensemble storage.

Owner-facing artifact: `compose_conversation_demo.py` -- a scripted agent that stores statements,
answers wh-questions, and persists its knowledge across a session boundary, all spiking.

## Genuine composition vs conventional glue (honest audit, owner-prompted)

Asked directly whether this is templates/cheating or real composition. Audit:

GENUINE composition (the load-bearing claim, verified): the bind/unbind FORMS and RETRIEVES sentences
that GENERALIZE to novel combinations. Test: bind 8 random nonsense SVO sentences ("go cold hot", "dog hot
dog") + query each role -> 8/8 recovered; there are 16^3 = 4096 possible SVO sentences, NONE enumerated or
stored -- only the bind/unbind operation is reused (60/60 in compose_vsa_demo; multi-seed bind/unbind;
adversarial reviewer CLEAR). A template can only echo stored sentences; this forms+answers arbitrary new
ones, computed by spiking coincidence neurons.

CONVENTIONAL GLUE / one template (honest, not hidden): (1) the interactive REPL's text->role parsing is a
hardcoded POSITIONAL template (store(word[0],word[1],word[2])); (2) the relational query is a Python loop
over a fact-list with == matching (the unbinds inside are spiking; the search is control logic); (3) cleanup
is argmax over the stored concept-code vocabulary (standard VSA cleanup, not an attractor net); (4)
generation prints unbound words in a fixed role order (retrieval genuine, ordering a template). Scope: 3-slot
SVO frames + voice-invariance, not fluent grammatical language.

ONE TEMPLATE CLOSED (compose_learned_parse_demo.py): the LEARNED parser (Hebbian conjunction->role, 6/6
incl the active<->passive flip) assigns roles in the pipeline -- active "dog go north" AND passive "north
is go by dog" both -> agent=dog (voice-invariant), where a positional template would wrongly call "north"
the agent of the passive. So the parse step is now LEARNED, not positional-hardcoded; bind/unbind remain
genuine spiking composition.

## Honest scope

Roles/cues are mapped from the (validated separately) learned parser; the vocabulary is 16 words
and the recognition front-end -- not the composition -- caps growth (~64/bridge clean, ~320 at
98.4%). These capabilities demonstrate that the spiking bind composes into the building blocks of
conversation (statements, questions, negation, persistent memory) at that scale; they do not claim
fluent open-ended language or front-end scaling. The biology-translatable thread: each conversational
operation reduces to bind (coincidence) + unbind + cleanup over separately-stored ensembles, with
polarity/role/query all carried as bound tags -- a cell-assembly account of simple language.
