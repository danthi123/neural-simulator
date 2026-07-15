# Wire-in TARGET CORRECTION (a0 read-the-real-code): the EMERGE-58 ability router is a set-membership FAMILIARITY GATE (already-GO learned moat) + a regex, NOT compositional dispatch — the real compositional-dispatch target is the FLUID console's multi-intent router (wh-half already neural; discourse-marker-half is the wire-in)

**Date:** 2026-07-15 · **Status:** honest target-correction from BUILDING the wire-in against the real console code (not a re-skin of the synthetic task). Redirects the deployment; the de-risked SCIENCE (interpolative compositional dispatch GO) stands and now points at the correct target.

## What I built + what it exposed
`_learned_dispatch_console_wire_derisk.py` instantiates the REAL `UnifiedFluentConsole(build_fluid=False)`, introspects its ACTUAL taxonomy (`reasoner.member_idx` + the is-a script), and trains the GO deep-credit classifier to reproduce the router's decision. 1-seed smoke:
```
[wire s42] parity=1.000 heldout->reasoner=0.000 moat(unk->non)=1.000 FA=0 permuted=0.867 n_mem=16 roots=['animal']
```
- **parity=1.000, moat 0-FA:** the learned classifier TRIVIALLY reproduces the router's decision + never routes an unknown to the reasoner.
- **roots=['animal'], pure-membership route:** the tell. The EMERGE-58 `turn()` route is `X in self.reasoner.member_idx` — a **set-membership test** (member→reasoner; else→fluid/abstain, renderer never invoked). Every member routes the same regardless of category; the only distinction is member-vs-unknown. That is a **familiarity / novelty gate**, which the project ALREADY solved as the learned Bogacz-Brown familiarity moat (`2026-06-11-familiarity-gate-v320-GO`) — plus a regex frame-match (`_ABILITY_RE`).
- **`heldout->reasoner=0.000` is not a bug — it's ill-posed for THIS router:** the hand router itself routes a non-member (a held-out member) to NON-reasoner (abstain). So "generalise a held-out member to the reasoner" is a CAPABILITY CHANGE, not parity — and it only pays off if the reasoner ALSO generalises to that member (a separate, entangled mechanism). `permuted=0.867` likewise just reflects the 16-vs-3 route-1/route-0 class imbalance under the parity-on-members metric.

## The correction
The ROADMAP-sync named "the hand membership-aware router (EMERGE-58)" as the compositional-dispatch wire-in target. Reading the real code shows that specific router is **not** a compositional dispatch — it is a familiarity gate + regex. The rich multi-intent compositional dispatch (the shape `_learned_dispatch_derisk` validated: subject-category × question-type → response-frame, interpolative GO) lives in the **FLUID console** (`_fluidconv_chat_repl.py` `FluidChat.turn`): QUESTION (what-patient / who-agent / yes-no) · DISCUSS (tell-about / compare / share) · LEARN (fetch+ingest) · TAXONOMY (trace/ancestry) · UNTAUGHT→moat.

Crucially, the fluid router is **already PART neural**: the wh→query-type half is the Phase-7 neural interrogative parser (`_neural_parse`, "composer wh->type + BridgeParser roles", not a host keyword). The remaining HAND-KEYWORD routing is the **discourse-marker triggers** — `{compare, different, share, common, classify, trace, ancestry}` (line ~64) + the `learn about` / `tell me about` triggers. THOSE are the emergence-bar wire-in target: learn the discourse-marker→intent routing from a stream instead of the keyword set.

## Honest status of the arc
- **Science:** DONE — interpolative compositional dispatch GO (`_learned_dispatch_derisk`, 6-seed); true compositional EXTRAPOLATION is a structural boundary (binder wall recurs).
- **Deployment target:** CORRECTED — not the EMERGE-58 familiarity gate (already a learned moat), but the fluid console's discourse-marker intent routing (the wh half is already neural).
- **Next build:** learn the discourse-marker→intent dispatch from the fluid console's own routing + wire it into `FluidChat` (additive, default-off, moat-preserved), so the intent structure is LEARNED not keyword-coded. The wh→type half needs no wire-in (Phase-7 already neural).

The probe runner `_learned_dispatch_console_wire_derisk.py` is retained as the artifact that produced this correction (its parity/moat result is a valid confirmation that the membership gate is trivially learnable; the held-out/permuted metrics are ill-posed for a pure-membership route and are not gates).
