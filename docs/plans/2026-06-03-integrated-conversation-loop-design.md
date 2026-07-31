---
type: plan
status: live
date: 2026-06-03
---

# Integrated conversation loop — design (comprehend → decide-what-to-say → produce)

**Date:** 2026-06-03
**Status:** design approved (owner delegated: "whatever leads to our goals soonest and most efficiently").

## Goal

Unify the project's three validated conversational abilities into one fluid loop:

1. **Comprehend** — parse a user utterance into concepts (existing: SVO parse; faithful = Hebbian
   conjunctive parser).
2. **Decide what to say** — the content-selection Control (PFC "Control"), validated end-to-end this
   session, choosing the most relevant thing to bring up next.
3. **Produce** — generate-by-composition (existing, validated): an ordered sentence read-out from a
   composed meaning.

The existing `_integrated_conversation_loop_demo.py` already wires comprehend → memory → produce for
**factual Q&A** (store SVO facts; answer "what does X do / who does Y / tell me about Z" with composed
sentences). What it cannot do — and what the Control adds — is **dialogue planning**: deciding what to
elaborate when the user merely raises a topic, and sustaining a coherent, progressing multi-turn
conversation. That is exactly the capability validated this arc.

## Approach (staged; this doc = milestone 1, numpy)

Milestone 1 builds the full loop in **numpy**, reusing every validated piece, so a tangible end-to-end
conversational agent runs fastest and the architecture is de-risked before the slower spiking version.
Milestone 2 swaps in the spiking content-selection Control (`SpikingSpreadingController`) + spiking
production. This mirrors the structured→spiking staging that worked for the Control arc itself.

## Mechanism — the Control's graph is the agent's own KB

The key idea: **the agent's association graph is built from its KB**, so dialogue planning is grounded
in what the agent actually knows.

- Each stored fact `{agent, action, patient}` makes its three words pairwise-associated (a `build_
  association_graph` input like `"dog_chase"`, `"chase_cat"`, `"dog_cat"`).
- When the user raises a **topic** (a known word, not a question or a 3-word statement), the agent:
  1. rebuilds the Control over the current KB graph,
  2. `Control.turn([topic])` → the most relevant unsaid associated word,
  3. finds a KB fact containing **both** the topic and that word (guaranteed to exist — the edge came
     from a fact) that has not been elaborated yet,
  4. **produces** that fact as an ordered sentence (generate-by-composition).
- `"more"` continues: `Control.turn([focus])` → next pick → next fact → produce. The Control guarantees
  on-topic coherence + non-repetition (inhibition-of-return) + progression across the conversation.
- Direct **questions** and **statements** keep the existing factual-Q&A / bind behavior.

So elaborating "dog" walks the agent's associative memory of dog — each turn a coherent, non-repeating,
produced sentence about a dog fact — which is the dialogue-planning capability the factual-Q&A loop lacked.

## Components / files

- **Reuse (no change):** `_generate_by_composition_probe.build_world/compose/generate` (production);
  `research.runners.content_selection.build_association_graph` + `ContentSelectionController` (Control).
- **Create:** `research/runners/integrated_conversation_loop.py` — the `ConversationalAgent` + a scripted
  demo + a `--repl`.
- **Test:** `tests/test_integrated_conversation_loop.py` — comprehend/bind, factual Q&A, and the NEW
  elaboration path (topic → produces a related-fact sentence; `more` → progresses, non-repeating,
  on-topic; coherent topic shift).

## Data flow

```
user text ──► parse ──┬─ statement (SVO) ─► bind into KB ─► echo composed sentence
                      ├─ question (what/who/tell) ─► retrieve fact ─► produce sentence
                      └─ topic / "more" ─► Control over KB-graph picks next word
                                          ─► find KB fact (topic + word) ─► produce sentence
```

## Error handling

- Unknown topic / no facts → honest "(i don't know about X)".
- Control returns None / no unsaid fact left → "(that's all i know about X)".
- Unparseable input → "(i didn't understand)".

## Testing / honesty

- The elaboration must stay on-topic (every produced fact contains the focus topic) and non-repeating
  (each fact elaborated once) — asserted in tests.
- Honest scope in the runner docstring: numpy substrate + simple SVO parse (the faithful spiking Control
  + Hebbian parser are milestone 2); this milestone demonstrates the INTEGRATION + dialogue-planning.

## Milestone 2 (follow-up, faithful)

Swap `ContentSelectionController` → `SpikingSpreadingController` (already validated, same `.turn` API);
optionally the spiking production. The KB-graph + produce wiring is unchanged — only the Control backend.
