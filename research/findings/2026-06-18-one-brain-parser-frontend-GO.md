# Roadmap phase 2, GAP B (step B1) — the PARSER FRONT-END drives the composition on one bridge: GO

**Date:** 2026-06-18 (the real "one brain" headline arc). **Status:** **GO** — uniform across 3 seeds × 2 D (6/6). This
is **GAP B** from the production scoping (`2026-06-18-production-one-brain-composer-scoping.md`): the composer's operand
is driven by the **parser's neural role decision**, not a host `{role: word}` dictionary — comprehension becomes
synaptic.

**The question.** The prior de-risks HOST-set which role each operand binds to (the host knew "word 1 is the agent").
Production must instead drive that from the parser's firing — the comprehension (which word is the agent vs the patient)
must be a neural computation, not a Python dict the host then orchestrates.

**Runner:** `research/runners/_phaseB_onebrain_parser_frontend_derisk.py` (+ construction smoke
`_phaseB_onebrain_parser_coresident_smoke.py`) | **Raw:** `research/findings/raw/_phaseB_onebrain_parser_frontend.json`

## The mechanism — B-ii (the parser's firing selects the bind)

The GAP-B scope (`2026-06-18-onebrain-gapB-parser-frontend-scoping.md`, controller-verified) settled the design:

- **B-i (gate an RF→RF complex synapse) is ruled out.** Transmission gates multiply the Izhikevich connection matrix
  `cp_connections`, NOT the resonate-and-fire (RF) complex matvec `cp_rf_w_re`/`cp_rf_w_im` (verified `bridge.py:5528`).
  So a gate cannot open/close an RF→RF route directly.
- **B-ii is the way, and it is brain-based-compliant.** The parser **fires** for a word's role (the neural decision —
  `BridgeParser.role_of` reads which of the three role ensembles fires most), and that decision **selects the bind's
  complex weight** = that role's fixed phasor. The decision is neural; the projected code is a fixed wiring constant
  (like an axon's developmental target). Under the project's brain-based-only standard, a neural decision selecting a
  fixed downstream projection is legitimate; only a host *computing* the decision would be a shortcut.

**Co-residence (construction smoke PASS).** The Izhikevich Hebbian parser (state in `v`/`u` as voltage, stepped by the
full simulation step) and the RF registers (state in `v`/`u` as a complex phasor, stepped by the masked resonate loop)
co-exist on ONE bridge, each un-regressed — the same regime the merged navigation+conversation bridge already proved
(step 2b). The parser's incidental firing corrupts the RF registers' `v`/`u` between ops, but the composer re-kicks
every op, so it is harmless; the masked RF ops leave the parser slice untouched.

**The flow (one fact, one bridge).** Comprehend the sentence with the parser → for each position, the role it fires
selects that word's bind complex weight → bind all three through complex synapses → bundle into a composite register →
unbind the cued role → cleanup → answer. The parser's decision is the only thing that says which word is the agent.

## Result — 3 seeds × {D=64, D=128}

| metric | result (mean, 6/6) |
|---|---|
| parser-driven recall == ground truth | **1.000** |
| parser-driven recall == host-routed oracle | **1.000** |
| voice-invariance (active & passive store the SAME fact) | **1.000** (5/5 every seed/D) |
| moat clean-separation (bound role vs unbound role peak) | **1.000** (bound ~4–10×10⁸ vs unbound ~1.2–2.6×10⁸) |
| anti-cheat: permuted parser→role collapses | **1.00** (the agent query returns the action word) |
| anti-cheat: lesioned parser collapses | **1.00** (garbled comprehension → wrong agent) |

Every one of the 6 configs is recall 1.00 self/host, voice-invariance 5/5, moat-sep 1, both anti-cheats collapsing.

## Reading

- **Comprehension is synaptic.** The fact that gets stored is determined by the parser's firing, not a host dict — the
  recall matches the host-routed oracle (which is *told* the parse) exactly, so the parser's neural decision is
  faithful.
- **Voice-invariance is the parser's signature.** An active sentence "dog go north" and its passive frame (voice flips
  the first and third positions) store the **same** fact — both recover agent=dog, action=go, patient=north. That is
  only possible if the parser genuinely comprehends the role structure (a positional shortcut would store different
  facts).
- **The anti-cheats collapse.** Permuting the parser→role map binds the wrong roles (the agent query returns the
  action word); lesioning the parser's learned weights garbles comprehension. So the parser's specific learned mapping
  is load-bearing, not incidental.
- **The moat holds under parser-driven composition.** Querying a role the fact lacks gives a low cleanup peak
  (abstain), cleanly separated from a bound role's high peak — the no-confab moat is preserved.

## Honest scope + next

- The **query side** is a host-specified cue (the store side is parser-driven). Driving the query cue from the parser
  comprehending the *question* (wh-questions) is the natural follow-on (STEP B2), deferred per the scope.
- This B1 stores the parser-driven fact in a register (a single fact). Combining it with the **GAP-A persistent
  synapse-store** (many facts, GO to K=32) is the integration: that is **STEP A3** — wrap the parser front-end + the
  persistent store as the production `OneBrainComposer` (an `RFPhasorComposer` API-sibling) and run the full agent
  capability matrix (who / what / yes-no / moat) on it, swapped into `BrainConversationalAgent`.
- Then A4 (the optional spiking winner-take-all selection) and A5 (make it the default + megakernel the persistent
  loop + retire the legacy numpy runtime, keeping numpy as the test oracle).

## Reproduce
```bash
SIM_BACKEND=cupy python -u -m research.runners._phaseB_onebrain_parser_coresident_smoke --seed 42 --D 64
SIM_BACKEND=cupy python -u -m research.runners._phaseB_onebrain_parser_frontend_derisk --seeds 42,43,44 --dims 64,128
```
