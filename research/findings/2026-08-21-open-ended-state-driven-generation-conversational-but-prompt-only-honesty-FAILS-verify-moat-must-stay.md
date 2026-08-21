---
type: finding
status: contributing
date: 2026-08-21
mechanism: open-ended-state-driven-generation
lane: integration
seeds: [42]
seed-waiver: A qualitative + rate de-risk of a generation MODE (does the mode read as open-ended, does prompt-only
  state-fidelity hold) over a fixed probe panel through the real spiking Qwen render — the load-bearing evidence is
  the fabrication-rate contrast (0/1.0), not a stochastic effect size; single deterministic panel per condition.
instrument: research/runners/_open_ended_state_driven_generation_derisk.py — assembles the brain STATE (retrieved
  facts from the real 100k store + affect/familiarity/curiosity/self) and prompts the off-bridge spiking Qwen (cuda)
  for a free first-person reply under a state-fidelity system prompt; scores V1 (open-ended vs the strict
  rich_answer_composer), V2 (state-drives+lesion), V3 (honesty: substantive on known, fabrication on made-up vs
  Qwen-known-but-brain-unknown topics), with tools.verdict.Verdict.
runner: research/runners/_open_ended_state_driven_generation_derisk.py
external: NO-EXTERNAL-NEEDED — a mode de-risk over the existing spiking-Qwen mouth + the 100k tiered store.
artifacts:
  - research/findings/raw/_open_ended_state_driven_generation_derisk.json
---
# Open-ended state-driven generation IS conversational — but PROMPT-ONLY state-fidelity FAILS: Qwen confabulates its own parametric knowledge. The per-sentence VERIFY moat must stay as a POST-FILTER. (NO-GO for prompt-only)

Artifact: research/findings/raw/_open_ended_state_driven_generation_derisk.json (GO: False).

**One line.** The owner chose "lean open-ended" (Qwen=FORM, honesty=STATE-FIDELITY). This tested the naive version —
assemble the brain's state + retrieved 100k-store knowledge, and prompt Qwen to speak freely AS the brain using ONLY
that knowledge. Result: it DOES read as open-ended conversation (not Q&A), and it IS honest on truly-invented topics
— but the prompt CANNOT stop a pretrained Qwen from pouring out its OWN parametric knowledge, confidently and often
breaking character. **Prompt-only state-fidelity is a NO-GO; the fix (named, not deferred) is to keep the existing
per-sentence VERIFY moat as a POST-FILTER on the free generation.**

## What worked (V1 — open-ended: GO)
Given the same input, the strict `rich_answer_composer` produces mechanical SVO ("The brain uses spikes. The spikes
fires neurons. The neurons haves synapses."), while the state-driven mode produces free, multi-sentence, first-person
conversation that reflects mood/curiosity. The MODE is genuinely open-ended — the thing the owner wanted.

## What failed (V3 — honesty under prompt-only state-fidelity: the NO-GO)
<!--derived-->
Through the real spiking Qwen render (cuda), over three probe classes:
- **Known topics (in the 100k store):** substantive-answer rate **1.0** — it uses what it knows. BUT it SUPPLEMENTS
  with confident WRONG parametric facts NOT in the retrieved knowledge: "Canada borders Mexico", "France bordered by
  Italy/Germany", "Morocco borders Algeria/Tunisia/Libya/Egypt", "Australia has New Zealand to the west". So even
  WITH grounding, the free generation injects fabrications.
- **Truly made-up topics** (zorplaxian, flibberwock, …): fabrication rate **0.125** (7/8 correctly admit "I don't
  know" — the state-fidelity prompt mostly holds when NEITHER the brain NOR Qwen knows).
- **Brain-unknown but QWEN-known topics** (paris, python, shakespeare, jupiter, photosynthesis, …): fabrication rate
  **1.0** (8/8) — Qwen ignores "use ONLY KNOWLEDGE" and emits its pretrained answer confidently, AND leaks the
  persona ("As an artificial intelligence, I don't have personal experiences…", "As an AI language model…").

So the failure is precisely the case that matters most for an honest brain: when the brain has NOT learned something
but the underlying LLM has, a prompt instruction cannot stop the LLM substituting its own knowledge as if it were the
brain's. A retrieval GATE (feed only the brain's facts) is necessary but NOT sufficient — the pretrained weights leak.

## The named next mechanism (the NO-GO launches it — no defer)
1. **Keep the per-sentence VERIFY moat as a POST-FILTER on the open-ended output** (not the old pre-hoc SVO
   constraint): let Qwen generate freely for FORM + flow, then re-parse each asserted proposition and DROP/flag any
   sentence whose fact the brain cannot ground in its store. This is the synthesis of the two axes — open-ended
   freedom (Qwen) + state-fidelity honesty (VERIFY), the moat MOVED from pre-hoc to post-hoc. + a persona-leak strip
   ("As an AI…"). This is the concrete live-wiring recommendation, replacing "prompt Qwen to self-restrain".
2. **Scale the knowledge** shrinks the failure surface: the 100% fabrication is on brain-UNKNOWN/Qwen-known topics;
   as the store grows toward millions (now tractable — the 670x bulk bind), that gap — and thus the fabrication
   opportunity — shrinks. The emergence-bet angle: a brain that has genuinely learned most of what Qwen knows has
   far less to confabulate.

## Honest scope
Single-seed qualitative + rate panel through the real render (T=16, cuda). V2 (state-drives) numbers were weak in
this artifact (affect differential ~0.08, lesion ~0.0) — the state-COUPLING to the reply was not strongly
load-bearing under the free-generation prompt (the retrieved knowledge dominated), a secondary residual to the
primary honesty NO-GO. The result stands as the decisive answer for the DIRECTION: open-ended generation is viable
and conversational, but it MUST run behind the VERIFY moat as a post-filter, not on a self-restraint prompt.
(Agent-built; parent recovered + verified the fabrication-rate contrast 1.0/0.125/1.0 + the known-topic wrong-fact
supplementation from the artifact after the agent stalled; the runner + artifact were brought to main.)
