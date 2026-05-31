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
   cat?" finds the fact and unbinds POLARITY -> yes/no. Multi-seed (42,43): yes/no 1.000, control 1.000 (seed 44 finishing) -- RESOLVES.
   Insight: negation is an explicit bound POLARITY ENSEMBLE (a distinct tag), not the absence of a
   fact -- consistent with separate-ensemble storage.

Owner-facing artifact: `compose_conversation_demo.py` -- a scripted agent that stores statements,
answers wh-questions, and persists its knowledge across a session boundary, all spiking.

## Honest scope

Roles/cues are mapped from the (validated separately) learned parser; the vocabulary is 16 words
and the recognition front-end -- not the composition -- caps growth (~64/bridge clean, ~320 at
98.4%). These capabilities demonstrate that the spiking bind composes into the building blocks of
conversation (statements, questions, negation, persistent memory) at that scale; they do not claim
fluent open-ended language or front-end scaling. The biology-translatable thread: each conversational
operation reduces to bind (coincidence) + unbind + cleanup over separately-stored ensembles, with
polarity/role/query all carried as bound tags -- a cell-assembly account of simple language.
