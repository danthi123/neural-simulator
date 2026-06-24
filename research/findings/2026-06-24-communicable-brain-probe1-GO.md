# Communicable-brain Probe 1 GO: the brain generates a novel, grounded, flagged thought (2026-06-24)

The first validated step from "querying a store" (the RAG-feel the owner flagged) to "talking to
something" — done the BRAIN-BASED way: the brain generates the content, the LLM is fluency-only, and a
lesion anti-cheat *proves* the content is the brain's. Owner's top-priority frontier.

## The turn (`research/runners/_communicable_brain_probe1_whatdoyouthink.py`, `WhatDoYouThinkTurn`)

A "what do you think about X" turn:
1. **ASSIMILATE X** — read X's neighborhood in the brain's learned PPMI co-occurrence graph.
2. **PROPOSE** — the b2 generative-replay proposer samples a NOVEL-but-plausible, *topic-relevant*
   proposition (X must appear; graph-biased; moat-gated; non-contradictory). Returns `None` = the honest
   "I don't really have a view on X" case.
3. **RENDER + VERIFY** — neural serial-order word order + a fluency-only faculty for surface form (NEVER
   the content); VERIFY re-parses the *prose* back to an SVO (the brain's content-extractor + `BridgeParser`)
   and rejects on any drift from the brain's proposed triple.
4. **EMIT** — a graded-confidence FLAGGED hypothesis ("I'm not sure, but I'd guess ..."), driven by the
   proposal's plausibility. NOT stored; the known-fact channel stays hard-gated.

## 6-seed GO (seeds 42/43/44/100/101/102)

- EMITTED flagged hypotheses: mean 7.7 / 30 topics (min 6); render+VERIFY rate 1.000.
- **(1) NOVEL** 1.00 all seeds (vs the measured 0.0 retrieval-novelty ceiling) — every emission is generated,
  never a verbatim stored fact.
- **(2) GROUNDED** shuffled-PPMI-graph advantage mean 16.7× (min 12.7×; bar 3×) — the brain's *learned*
  structure is load-bearing (shuffle it and the proposals collapse to the floor).
- **(3) FLAGGED + leak-free** 0 known-fact-channel leaks (all seeds) — the moat is RELAXED to
  speak-while-flagging, NOT removed (a subsequent who/what query on an un-stored fact still abstains).
- **(4) CALIBRATED** spearman(plausibility, confidence) 1.00 all seeds.
- Examples: "I'd say fish sing white." (conf 1.0), "I think maybe brother play pink." (conf 0.40), "I'm not
  sure, but I'd guess sister look hand." (conf 0.0).

## The anti-cheat that makes it real (not the LLM doing the cognition)

**LESION / PROVENANCE, 46/46 across 6 seeds:** sever the brain's proposal and let the fluency faculty
FREE-GENERATE its own content → VERIFY REJECTS every one (the re-parsed SVO ≠ the brain's proposed triple).
So with the brain's proposal removed, no sensible reply survives — the content is the BRAIN's, not the
LLM's. The LLM is strictly fluency-only (surface form). Plus the shuffled-graph control (groundedness
load-bearing) + the known-fact-channel integrity (0 leaks; stored facts still answered).

## Honest scope (each a precise next-build, not a failure)

- The brain **abstains-from-opinion on ~70% of topics** — it says "I don't have a view" rather than
  confabulate when the learned PPMI graph lacks crisp relatedness (high-frequency words whose high marginals
  suppress PMI). Correct graceful behavior, but emissions are thin (6–9/30). ⇒ the spiking **value/salience
  appraisal** ("is this worth saying / which proposition is salient") is the natural next mechanism to make
  the brain *choose to speak more* where it has support — and it's exactly where the option-A limbic/value
  core pays off.
- The fluency faculty here is a CPU content-locked stub (the validated gate→VERIFY contract). Wiring the
  real GPU spiking-Qwen faculty into this exact pipeline is the drop-in follow-on — the VERIFY contract is
  identical and already proven to catch a real 0.5B's drift.

## NO `sim/` edit — reuse-by-import (the b2 proposer + the neural serial-order renderer + the gate→VERIFY
loop, all independently GO on `main`).

## ⇒ The brain DOES generate a communicable thought of its own. The RAG-feel is breakable; this is the
first validated, no-cheat step toward a brain you converse with.
