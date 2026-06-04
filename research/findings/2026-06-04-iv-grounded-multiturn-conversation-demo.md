# (iv) A grounded multi-turn conversational artifact — the session's capabilities in one dialogue — 2026-06-04

**One line:** A single INTERLEAVED multi-turn dialogue (statements and questions mixed, as a real conversation
flows), on concept codes GROUNDED in the real V1 Gabor bank + ventral-hierarchy decorrelation, exercising every
capability the session validated: comprehend-and-learn, answer-by-composition (flat / one- and two-attribute /
embedded clause, auto-detected), answer who-queries, ABSTAIN on the unknown (no confabulation), and elaborate on a
topic (dialogue planning). `research/runners/unified_agent_conversation_demo.py` + 2 smoke tests (pass).

## The transcript (concept codes from a real biological V1 receptive-field bank)

```
user>  dog chase cat.                          sim>  ok, learned.
user>  what does dog chase?                    sim>  cat.
user>  child hold red ball.                    sim>  ok, learned.
user>  what does child hold?                   sim>  red ball.            (one attribute)
user>  what does dog eat?                       sim>  (I don't know -- you never told me.)   (ABSTAIN)
user>  cat want big red ball.                  sim>  ok, learned.
user>  what does cat want?                      sim>  big red ball.        (two attributes)
user>  bird see cat chase cold river.          sim>  ok, learned.
user>  what does bird see?                      sim>  cat chase cold river. (clause, attributed inner arg)
user>  dog see child give bread.               sim>  ok, learned.
user>  what does dog see?                        sim>  child give bread.    (clause, flat inner args)
user>  who chase cat?                            sim>  dog.
user>  what does river chase?                    sim>  (I don't know -- you never told me.)   (ABSTAIN)
user>  tell me about dog.                         sim>  dog see child give bread.  dog chase cat.   (elaborate)
```

## What it demonstrates

This is the conversational payoff of the session's resolutions, in one artifact:

- **Comprehension + continual learning** mid-dialogue (facts are learned as stated, then immediately queryable).
- **Composition** at every depth the benchmark covers, auto-detected from the query alone — flat patient, one-
  attribute ("red ball"), two-attribute ("big red ball"), and an embedded clause with either an attributed inner
  argument ("cat chase cold river") or flat inner arguments ("child give bread").
- **The no-confabulation moat** — in-vocabulary agent+action pairs that were never stated return "I don't know",
  not a plausible guess.
- **Dialogue planning** — "tell me about dog" brings up the on-topic facts via the content-selection Control over
  the agent's own knowledge graph.
- All on **sensory-grounded** concept codes (real V1 Gabor + decorrelation), not random/constructed codes.

## Honest scope

- Composition + the no-confab moat are validated IN GENUINE SPIKES (the #4 grounded-spiking capstone = 72/72 = 100%
  core); the dialogue-PLANNING layer in this demo is the numpy content-selection Control (validated separately in
  spikes as the spiking content-selection arc). So the demo is the richest numpy showcase running on grounded
  codes; the spiking realization of the composition+moat is the capstone, and the spiking dialogue planning is the
  prior content-selection arc.
- The per-concept stimuli are synthetic distinct textures (no natural images for abstract words — the
  embodied-cognition limit); the grounding pipeline (V1 Gabor) is real.
- Scripted (reproducible, headless, testable). An interactive REPL variant already exists (`phasor_chat`).

## Files

- `research/runners/unified_agent_conversation_demo.py` — the interleaved grounded multi-turn demo.
- `tests/test_unified_conversation_demo.py` — 2 smoke tests (composition + abstention; topic elaboration).
