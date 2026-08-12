---
type: finding
status: contributing
date: 2026-08-12
mechanism: GENERATE-channel (#3E) hypothesis SURFACE rendered by the composed spiking BROCA (EMERGE-59/61 order read-out × the #3E draw), transformer-free — burn-down A1a
lane: production-integration / articulation (A1)
seeds: [42]
artifacts:
  - research/runners/brain_chat_tui.py
  - research/runners/_spiking_fluent_surface_derisk.py
instrument: synchronous numpy-CPU verify harness over a real ChatBrain (tiny-demo, rf composer, GPU-free stub renderer) — end-to-end #3E generation + direct HypothesisSVO render + moat/lesion/no-regression, plus the canonical `--smoke` compared pristine==modified.
---

# The GENERATE channel's hypothesis SURFACE is now spoken by FIRING NEURONS (spiking Broca), not the external transformer — burn-down A1a WIRED

## What changed

The production `/api/brain-chat` GENERATE channel (#3E) VOLUNTEERS a novel grounded hypothesis on an open-ended prompt ("what might a dog chase") as a moat-verified, flagged `HypothesisSVO`. Its SURFACE previously came from the agrammatic host f-string ("perhaps bear walk foot") or, post the fluent-generation wire, the external Qwen mouth. This wire routes a STRUCTURED (transitive SVO) hypothesis's surface through the **composed spiking BROCA render**, reused-by-import from `research/runners/_spiking_fluent_surface_derisk.py` (the 6-seed GO de-risk) — the word ORDER is the per-pool spiking-RATE ranking on a real Izhikevich `SimulationBridge` (EMERGE-59/61), every word via the A→W read-out, productive 3sg inflection. NO transformer on that path.

Edit is additive + guarded, in `ChatBrain.render_hypothesis_verified` (`research/runners/brain_chat_tui.py`): a supported hypothesis (single-word transitive SVO, subject≠object) with `BRAIN_SPIKING_MOUTH != "0"` renders on the cached spiking producer (built + competitive-queuing-learned once, ~0.35 s; ~5 ms/emit via the EMERGE-61 inter-utterance wash-out), then is re-parse VERIFIED by the SAME production moat the recall path uses (`_verify` → `_extract_svo_from_prose` recovers the drawn SVO). A verify miss falls back to the raw flagged template — never a leak. The guess stays clearly FLAGGED ("perhaps … [a guess from what I've learned — not something I was taught]"). The escape flag `BRAIN_SPIKING_MOUTH=0`, OR content the spiking Broca can't frame (open/multi-word prose), falls through to the UNCHANGED pre-spiking mouth (Qwen / stub / template). The GATE decision + the moat are unchanged; only the SURFACE render of a GENERATED hypothesis changes. No `sim/` edit.

## Verification (synchronous, numpy-CPU; all PASS)

- **(a) spiking render + moat re-parse.** End-to-end through the production gate: `gate("what might a dog chase")` → `HypothesisSVO ['dog','chase','hare']` → **"perhaps the dog chases the hare  [a guess …]"** (`fluent_verified=True`; the moat re-parse recovered the drawn SVO). Four direct verbs all render grammatically + faithfully: "perhaps the dog chases the hare", "perhaps the fox chases the ball", "perhaps the cat eats the fish", "perhaps the owl eats the mouse" — the INDEPENDENT held-out parser (`parse_hedged_transitive`) also recovers each SVO. The path is transformer-free (no `torch`/`transformers` in-process).
- **(b) MOAT — 0 leaks.** An unverifiable render (subject not a known noun) → verify miss → falls back to the raw FLAGGED template (`verified=False`), never asserting a wrong fact; every rendered hypothesis is flagged as a guess; `_verify` is DISCRIMINATIVE (surface(dog,chase,hare): verify(correct)=True, verify(wrong-object)=False, verify(wrong-subject)=False — no trivial pass); a plain recall still returns the TAUGHT fact (the generator's own moat intact).
- **(c) LESION — load-bearing.** `BRAIN_SPIKING_MOUTH=0` → the GENERATE surface is NO LONGER the spiking clause (4/4 revert to the stub/Qwen mouth, e.g. "Maybe the dog chases hare — that's a guess …"); the flag-off output is byte-identical to the pre-spiking mouth path.
- **(d) NO REGRESSION.** The canonical `--smoke` verdict is byte-identical with and without the change (the worktree's pre-existing PARTIAL state is unchanged by this edit — self-reference, fluent recall, in-turn anaphora "what does it eat" → "The cat eats fish", the no-confab moat abstentions, and the discourse before/now tracking all hold in the transcript). The change touches only the `HypothesisSVO` surface branch; recall/abstain/learn/anaphora/affect are untouched.

## Honest residual

The RENDER-ORDER is on spikes; the DRAW's content sampling remains host bookkeeping over the brain's learned graph (the fully-spiking SWR-CA3 draw is the banked `_followon1` negative — burn-down B1), and the A→W spell is the identity-surface callback. Open ARBITRARY prose the spiking Broca can't frame still falls back to the Qwen mouth — the banked deep-context wall (A1 proper). So A1a narrows A1: the STRUCTURED transitive-SVO GENERATE surface is now brain-native spiking; the Qwen residual is the open-prose wall.
