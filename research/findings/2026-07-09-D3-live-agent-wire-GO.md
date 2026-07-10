# D3 → the LIVE MultiTurnAgent (the production wire-in END-TO-END): the deployed agent binds "it" to D3's composed discourse center, never to recency

**Date:** 2026-07-09
**Runner:** `research/runners/_d3_agent_centering_wire_derisk.py` (reuse-by-import: `MultiTurnAgent` + `make_centering_task` + `discrete_attractor_rnn`; numpy; NO `sim/` edit).
**Verdict:** GO (6-seed) — the deployed conversational agent resolves pronouns via D3's brain-based composed focus.

## What this closes
The D3→conversation integration validated the resolution MECHANISM (D3's focus → the biased-competition → the pronoun). This is the **end-to-end deployment**: a **real `MultiTurnAgent`** hears SVO facts, a D3 **Centering-Cb adapter** (`D3CenteringFocusSource`) tracks the composed discourse center over those facts, and the agent's `focus_bias_source` hook uses it in `_resolve_biased` **in place of the host `content_bias_target`** feature-lookup. So the deployed brain binds "it/he/she" to *who we are talking about* (the composed center), not recency or a host feature match.

## The result (6-seed; NO `sim/` edit)
Focus-shifted discourses (the center CONTINUES as subject while new objects are mentioned → the true Cb ≠ the most-recent object):

| the LIVE agent resolves "it" to… | 6-seed |
|---|---|
| **the composed Centering-Cb (D3, brain-based)** | **0.611** (5/6 seeds 0.667) |
| RECENCY (the most-recently-mentioned) | **0.000** (every seed) |

**GO (load-bearing):** the deployed agent binds "it" to the composed discourse center and **NEVER to recency** — it resolves the Cb OR abstains (moat-safe), never mis-resolving to the recent referent. Live transcript (seed 42): *"bird chase worm. dog chase cat. dog chase fish. dog chase ball. → 'it' → **dog**"* (the continued center, not the recent "ball").

## What the anti-cheats + a0 established
- **resolves-to-Cb, NEVER-to-recency** (0.611 vs 0.000): the composed focus, not recency, drives the deployed resolution.
- **moat-safe:** the 1/3 non-resolutions are ABSTENTIONS (the no-confab moat), never wrong answers — the agent refuses to bind rather than pick the recent.
- **the resolution RATE (0.611) inherits the buffer's OWN ~5/6 per-referent competition decisiveness** (a referent whose fixed bias doesn't overcome its rival's intrinsic strength → the moat abstains) — the `BiasedCompetitionContextBuffer`'s characterized property, not the wire's.
- **A0 deployment detail (found + handled):** the agent's WM holds the PATIENT of each fact, but the Centering Cb is SUBJECT-preferred — so the wire also holds the subject (`_write_referent`) and bounds the WM (Centering maintains the center + recent) so the Cb is a held candidate.

## ⇒ the production wire-in is end-to-end
The recurrent sequence/language cortex (D3) — built, learned, spiking — now drives the DEPLOYED conversational agent's anaphora resolution: it binds a pronoun to the composed discourse center it tracks over the facts the agent hears, replacing the host `content_bias_target` shortcut with a brain-based composed focus, never confabulating to recency. This is the mission payoff, deployed.

## Honest scope + next
- The Cb tracker is a numpy discrete-attractor (the spiking transition+FS-WTA drops in). The `graded_bias` over-abstained here (probes+scales over the large held set); the fixed bias + a bounded WM is the working config.
- The resolution RATE is buffer-limited (~5/6 per-referent) — improving the biased-competition's decisiveness (a stronger/adaptive bias, or a cleaner attractor codebook) is the buffer's own arc. Combine Cb + feature-compatibility (`content_bias_target`) for the fullest resolution.

## Files
`research/runners/_d3_agent_centering_wire_derisk.py` (+ the `focus_bias_source` hook on `multi_turn_agent.py`); the D3 arc `2026-07-09-D3-*.md`.
