# Roadmap phase 2 — the conversational AGENT runs on the `OneBrainComposer` (the whole turn on one brain): GO

**Date:** 2026-06-18 (the real "one brain" headline arc). **Status:** **GO** — uniform across 3 seeds (3/3). This is
the **production wiring** of the integrated one-brain composer (STEP A3) into `BrainConversationalAgent`: the agent now
runs the whole who/what conversational turn on ONE persistent spiking bridge.

**Runner:** `research/runners/_phaseB_onebrain_agent_integration_derisk.py` | **CI guard:**
`tests/test_one_brain_composer_agent.py` | **Raw:** `research/findings/raw/_phaseB_onebrain_agent_integration.json`

## What changed (additive; the rf / rate default is byte-unchanged)

- **`research/runners/one_brain_composer.py`** — the production `OneBrainComposer` extracted from the A3 de-risk into a
  clean importable module. (Fixed a latent bug while extracting: the cleanup now iterates the composer's **actual**
  vocabulary `self.words`, not the module-level `VOCAB` constant — harmless in the de-risk where they coincided, wrong
  for an arbitrary agent vocabulary.)
- **`research/runners/brain_conversational_agent.py`** — two additive edits (a production runner, no protected `sim/`
  module touched):
  1. `composer_kind="onebrain"` builds the `OneBrainComposer`.
  2. `hear(sentence)` **delegates comprehension to the composer** when the composer carries its own parser (has a
     `hear` method) — so for the one-brain composer there is literally **one parser on the one brain**, and the parse
     result flows operand→bind as spikes rather than through the agent's separate parser. The agent's own parser is now
     built **only** when the composer lacks `hear` (the rf / rate / external paths — byte-unchanged; verified neither
     the rf nor the rate composer defines `hear`).

## Result — 3 seeds (the agent is the unit)

| metric | result (mean, 3/3 seeds) |
|---|---|
| `what_does` == ground truth | **1.000** |
| `who_does` == ground truth | **1.000** |
| `is_it_true` == ground truth ("yes" for stored) | **1.000** |
| == the rf reference agent (what / who / yes-no) | **1.000 / 1.000 / 1.000** |
| moat: unheard cue → `what_does` None | **1.00** |
| moat: unheard fact → `is_it_true` abstains | **1.00** |

Every seed is full GO on all metrics.

Both agents (onebrain + the rf reference) hear the same sentences, including one via its **passive frame** (voice
flips the first and third positions), with `polarity="AFFIRM"` (the affirmative-fact scope).

## Reading

- **The conversational agent runs the whole who/what turn on one brain.** With `composer_kind="onebrain"`, the agent
  comprehends (the composer's on-bridge parser), stores the fact in synapses, queries it by cue, and abstains when
  there is no fact — every answer matching the rf reference agent and the ground truth, the no-confab moat intact.
- **One parser on the one brain.** The agent delegates comprehension to the composer's parser; it does not build a
  separate parser for the one-brain path. Comprehension and storage are on the same persistent bridge.
- **The default is untouched.** The rf composer has no `hear`, so the default agent builds its own parser and uses the
  byte-unchanged parse+store path; a non-GPU guard test pins that `hear()` keeps both branches.

## Honest scope + next

- This first cut handles **affirmative** facts (who / what / affirmative yes-no + abstention). **Negation** (a bound
  polarity tag = a 4th role; the 4-role coherence is already GO) and the **richer capabilities** (`render_fact` for
  `describe`, `query_chain` for `reason_chain`, the dialogue `elaborate`) are bounded follow-ons that extend the
  `OneBrainComposer` to the agent's full composer interface.
- **Next:** run the rf-default agent suite (`tests/test_brain_conversational_agent.py`) as a regression check (the
  edits are additive, expected green) + the new CI guard. Then the richer composer capabilities, and A5 — megakernel
  the persistent loop so the one-brain path is speed-competitive, then make it a documented option and (once
  competitive) retire the legacy numpy runtime, keeping numpy as the test oracle.

## Reproduce
```bash
SIM_BACKEND=cupy python -u -m research.runners._phaseB_onebrain_agent_integration_derisk --seeds 42,43,44
SIM_BACKEND=cupy python -m pytest tests/test_one_brain_composer_agent.py -v
```
