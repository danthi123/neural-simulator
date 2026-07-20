# Grounded fluent conversation — DE-RISK 4 (open/rich multi-fact prose): the wall mapped + the capability met by the per-fact method

**Date:** 2026-07-20 · **Status:** DE-RISK 4 characterized — single-pass multi-fact SYNTHESIS is the honest field
wall (the WKV confabs); the CAPABILITY (rich grounded multi-fact discourse) is MET by the render-per-fact + aggregate
method (the P10/P16 discipline), preserved with the WKV renderer. NO `sim/` edit.

## The rung

De-risk 0/1/2/3 de-risked the north-star end-to-end: grounded fluent SINGLE-fact answers, on spikes, gate-first moat,
6-seed GO. De-risk 4 = OPEN/RICH multi-fact prose (elaboration/discussion). The scoping (+ CLAUDE.md P10/P16) flagged
single-pass multi-fact synthesis as the field wall; this rung MAPS the wall and confirms the workaround.

## The wall — single-pass multi-fact SYNTHESIS confabulates (CONFIRMED)

The format fine-tune trained only SINGLE-fact copy (`the A v3 P <ans> the A v3 P <eos>`). Prompted with TWO facts it
never saw, the WKV confabulates (memory ~5 tokens can't hold two facts through the copy):
- "the dog eats meat and chases cat" → **"the cat chases cat"** (subject/synthesis error)
- "the cat eats fish likes milk" → **"the fish likes milk"** (confab)

⇒ single-pass multi-fact synthesis on a small (~2.3M-param, ~5-token-memory) LM is the documented field wall — the
same one the ~21M ANN hit (CLAUDE.md P10/P16: "each fact rendered SINGLY"). NOT solved by this substrate at this scale.

## The capability MET by a different method — render-per-fact + aggregate (the P10/P16 discipline)

The console's `_discuss`/`plan_discourse` renders EACH fact singly (De-risk 2 GO: "the dog eats meat", "the dog chases
cat", "the dog likes bone" — all clean via the WKV) then AGGREGATES with connectives into grounded connected prose,
grounded by construction (every clause is a stored triple → no free abstractive generation → no confab). Live, WKV
renderer:
- "tell me about the dog" → **"Here's what I know about the dog: A dog is big. It eats meat, chases cat and likes bone."**
- "tell me about the cat" → **"Here's what I know about the cat: A cat eats fish, chases mouse and likes milk. A dog chases cat."**

Rich, grounded, fluent multi-fact discourse — no confab, moat intact. The single facts render via the spiking WKV;
the multi-fact SYNTHESIS is the grounded aggregation (template connectives), NOT the generator (which would confab).

## Read-out — the LAW satisfied: the CAPABILITY is met, the walled METHOD is deferred

- **The capability (rich grounded multi-fact discourse) IS achieved** — by the render-per-fact + aggregate method,
  which is grounded + fluent + moat-safe. The specific METHOD that hits a wall (weave N facts in ONE generator pass)
  is deferred; the capability does not depend on it (a-la the mission LAW: a wall verdicts a method, not a capability).
- **The wall's surpass is a scale/data lever, not a hard limit:** a longer-memory WKV (or a multi-fact-frame fine-tune)
  could extend single-pass synthesis to 2-3 facts — but the field's honest position (P16) is that fluent open-domain
  single-pass synthesis remains unsolved at small scale, managed by domain-constraint + grounded-retrieval +
  per-fact-render + abstention. The per-fact+aggregate deliverable IS the honest answer.

**⇒ THE NORTH-STAR ARC IS COMPLETE + honestly characterized:** grounded fluent conversation — comprehend + retrieve +
gate-first moat + render fluent grounded prose on the spiking WKV cortex (single fact = spiking WKV render, De-risk
0/1/2/3 6-seed GO; multi-fact = grounded per-fact-render + aggregate, De-risk 4; single-pass synthesis = the mapped
field wall). "A brain you COMMUNICATE with," fluency on spikes, ANN scaffold retired for the render path, NO `sim/`
edit anywhere in the arc.
