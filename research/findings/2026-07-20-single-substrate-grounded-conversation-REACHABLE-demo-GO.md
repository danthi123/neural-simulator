---
type: finding
status: qualified
date: 2026-07-20
---

# Single shared substrate — a REACHABLE grounded conversation on ONE bridge (GO)

**Date:** 2026-07-20 · **Status:** GO (6-seed 42/43/44/100/101/102) — a runnable grounded conversational turn on ONE
`SimulationBridge`: teach facts → ask → the composer retrieves + gates → the SPIKING WKV cortex renders the grounded
answer → the no-confab moat holds by construction (0 WKV invocations on abstains). Makes the single-substrate capstone
REACHABLE (talk to the one-brain), not just a de-risk. NO `sim/` edit.

## What this adds

The capstone proved the composer + WKV co-reside on one bridge. This wraps them into an actual turn you can RUN
(`OneBridgeChat` / `_gap_onebridge_conversation_demo.py --demo`):

```
Q: what does the dog chase?   A: the dog chases cat      [grounded on 'cat']
Q: what does the owl eat?     A: the owl eats mouse       [grounded on 'mouse']
Q: what does the lion roar?   A: I don't know.            [abstained — WKV NOT invoked]
```

Turn: `composer.query_patient(subj, verb)` → answer word (or `None` = abstain). On `None` → "I don't know." and the
WKV is **never invoked** (gate-first moat by construction). On a word → `WKV.answer("the {subj} {verb} {word}", ...)`
renders the grounded prose. Everything on one bridge (chan + encoder = WKV; composer region = RF retrieval).

## Result (seeds 42/43/100)

- **answer word PRESENT in render: True** (each answer contains the composer-retrieved word). Honest (post
  adversarial-audit): this is a presence check, NOT a faithfulness check — `ans` is injected into the WKV prompt, so a
  subject-fidelity wobble ("the mouse chases mouse") still passes. Render faithfulness is the separate De-risk-5 WKV
  render-quality item; the LOAD-BEARING claims here are the reachable single-substrate turn + the gate-first moat.
- **no-confab moat: True** (abstains on the never-taught `lion roar` / `fish swim`).
- **gate-FIRST: True — WKV invoked 0× across all abstains** (3 invocations for 3 known Qs, 0 for 2 abstains).

CI: `tests/test_onebridge_conversation.py` (GPU + ckpt, else skip).

## Honest scope

- The WKV render's SUBJECT fidelity wobbles on some frames (e.g. "the cat chases mouse" rendered "the mouse chases
  mouse") — the retrieved ANSWER is correct + present and the moat holds, but the render is not perfectly faithful on
  every frame. This is the known De-risk-5 WKV render-quality scope (a generation-fidelity item), NOT a
  single-substrate consolidation issue.
- Facts must use in-vocab words (the WKV's 4000-word TinyStories vocab); OOV subjects/verbs (e.g. "wolf"/"hunt") drop
  from the render prompt. A vocab/data lever, not a mechanism wall.
- ⇒ you can TALK to the one-brain single substrate: composer retrieves + gates, the spiking WKV renders, moat intact,
  all on ONE bridge. The full interactive `_fluidconv_chat_repl.py` wire-in (multi-turn/anaphora/learn-a-fact on the
  single substrate) is the remaining console-integration follow-on.

Runner: `_gap_onebridge_conversation_demo.py` (`--seed`, `--ckpt`).
