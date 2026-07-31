---
type: plan
status: live
date: 2026-05-16
---

# Generative-conversation frontier — options survey (research input, NOT a design)

> **Status:** Research survey. This document does **not** choose a
> direction or authorize implementation. Per the project's
> brainstorming discipline, the generative-conversation direction is
> a design-class decision requiring a collaborative brainstorming
> session + explicit approval. This survey exists so that session is
> well-informed. It enumerates what is proven, the precise gap, and
> candidate approaches with honest tradeoffs — nothing here is built.

## 1. What the validated retrieval substrate provably provides

The G.20 sparse-distributed ensemble is now rigorously characterized
(all anti-cheat, mostly seed-42, 160 also multi-seed; see the
2026-05-16 findings docs):

| Property | Result | Confidence |
|---|---|---|
| Vocabulary | 320 concepts (5×64) | strong |
| Per-bridge discrimination | 98.4% (320) / 98.1% (160 ms) | strong |
| Pair cross-bridge recall | 86.7% s42 / 92.7% (160, 5-seed) | strong |
| Sentence 3-way recall | 80.0% s42 | moderate (s42) |
| **Abstention / no-confabulation** | **AUC 0.990** | strong |
| Retention under 30-fact load | 80%, no catastrophic forgetting | moderate (s42) |
| Failure mode | index-intrinsic seed-42 pattern weakness (fixable via flagged per-seed/overlap-rejection) | understood |

Net: a **trustworthy associative memory** — store (concept,concept)
and (subj,verb,obj) bindings cross-bridge, retrieve by cue, answer
role queries (`who X Y?`), and *reliably abstain* when it doesn't
know. Continuous learning without catastrophic forgetting holds at
conversational scale.

## 2. The precise gap to "proper conversation"

The substrate does **retrieval**, not **generation**:

- ✅ "remember apple is big" → "what is apple?" → "big"
- ✅ "who run fast?" → "dog" (role query over stored sentence tags)
- ✅ correctly says "I don't know" for un-encoded queries
- ❌ produce a *novel* utterance not previously encoded
- ❌ multi-turn dialogue STATE (coreference across turns: "what
  about its color?")
- ❌ intent → response mapping (question vs statement vs request)
- ❌ sequencing concepts into grammatical output beyond fixed
  template slots

"Proper conversation" needs a **generative/dynamics layer on top of
the proven retrieval substrate** + a **dialogue-state mechanism**.
The realigned plan (2026-05-11) constrains this: standalone agent,
**no external LLM, local-only**.

## 3. Candidate approaches (honest tradeoffs; precedent-grounded)

### A. Retrieval-composed templated generation (lowest risk)
Drive output by the validated cross-bridge retrieval + role-query
machinery into grammatical templates ("X is Y", "the X Vs the Z").
Dialogue state = a small slot memory of recent tags.
- **Builds on:** 100% of what's proven; no new learning mechanism.
- **Precedent:** the g20_multibridge `--friendly` mode + role
  queries already do primitive versions.
- **Cost/risk:** low cost, low risk. **Ceiling:** expressivity
  bounded by template inventory — "conversational" in a structured
  QA/assistant sense, not free dialogue. Honest: this is the
  reliable, shippable floor, not open-ended conversation.

### B. Learned tag→tag transition dynamics (medium)
Learn next-ensemble prediction over the engram-tag space (a sequence
model at the concept-ensemble level), so the system can continue an
utterance rather than only recall a binding.
- **Builds on:** the engram substrate; the project has surrogate-
  grad/BPTT infra (Phase 2.1/2.2, `path-f-hybrid` branch) — but
  toy-scale, and Phase 2.3a was a documented NEGATIVE (char-features
  didn't transfer).
- **Cost/risk:** medium-high. Honest precedent: prior STDP-pathway
  composition attempts (v12–v16) were BOUNDARY/NEGATIVE; engram
  tagging (D.14) is what finally worked for binding. A transition
  layer must avoid repeating the composition-via-STDP trap.

### C. Biology-grounded hub/attractor sequencing (higher research risk)
Patterson hub-and-spoke / catalog hub dynamics: sequential attractor
traversal through the shared pool produces concept sequences.
- **Builds on:** the project's biology thesis + shared-pool arch.
- **Cost/risk:** highest research uncertainty; most aligned with the
  project's scientific identity (biology-grounded, no LLM). Could
  fail like the dlpfc_verb sequential-composition NEGATIVE (v12/v15)
  — that's the cautionary precedent.

### D. Sim-as-memory-for-LLM (explicitly deprecated for primary)
Path 3.2 stack exists (orchestrator, tool schemas). The realigned
plan **deprecated this for the primary path** (no external LLM).
Listed only for completeness as a fallback framing; not recommended
given the standalone-agent mandate.

## 4. Honest recommendation for the brainstorming session

- The **substrate is a solid, validated foundation** — whatever
  direction is chosen builds on proven, trustworthy associative
  memory with quantified limits. That de-risks the next phase.
- **A is the reliable floor**, shippable, low-risk, but bounded
  expressivity. **B/C are the routes to genuine generation** with
  honest precedent that concept-sequencing has repeatedly been the
  project's hardest, most-NEGATIVE-prone problem (composition arc).
- Decision criteria to weigh *with the user*: expressivity target vs
  biology-fidelity vs research-risk tolerance vs the local-only
  constraint. This is exactly a brainstorming-skill decision — it
  should NOT be made unilaterally or implemented before approval.
- Prerequisite regardless of route: the flagged index-weakness
  recovery (per-seed/overlap-rejection) lifts the substrate ceiling
  the generative layer inherits.

## Files / grounding

- Characterization: `research/findings/2026-05-16-G20-*-benchmark.md`,
  `-cross-benchmark-failure-analysis.md` (corrected)
- Constraints: `docs/plans/2026-05-11-realigned-plan-sim-as-standalone-conversational-agent.md`
- Cautionary precedent: CLAUDE.md composition arc (v12–v16
  BOUNDARY/NEGATIVE; engram D.14 the win)
- This survey is **input to**, not a substitute for, a brainstorming
  session.
