# Stage-2 (concept-sequence replay learning) — feasibility probe: NO-GO, reframed

> **Outcome of a brainstorming context-exploration step.** A cheap,
> zero-GPU, anti-cheat feasibility probe was run BEFORE designing
> Stage 2 (the project's falsify-before-investing discipline,
> demanded by the design doc's "falsifiable success gate"). It
> decisively killed the premise. This is a high-integrity outcome:
> a months-class NEGATIVE research arc was avoided for the cost of
> one analysis. **No design follows; this is the honest reframe.**

## The probe

Harvested every concept-sequence the project actually possesses
(engram tag names from committed g20 benchmark JSONs + the Stage-1
agent JSONL `concept_sequence` fields), built a bigram next-concept
model, evaluated held-out next-concept prediction vs a unigram
baseline and a permuted-sequence anti-cheat control.

| Metric | Value |
|---|---|
| Corpus | 86 unique sequences, **192 tokens, vocab 141** |
| REAL bigram next-concept acc | **0.000** |
| Unigram baseline | 0.000 |
| PERMUTED control bigram | 0.000 |
| Bigram lift over unigram | REAL **+0.000** == PERMUTED +0.000 |

**Verdict: NO learnable sequential structure.** Not a marginal
signal — literally zero, identical to the permuted control.

## Why (the real finding — corpus, not mechanism)

The "sequences" are **random-by-construction**: every multi-concept
tag the substrate ever saw (`apple_when`, `because_toward`,
`bottom_nine`, `box_ask_sour`) was a *randomly sampled* cross-bridge
pair/triple created by benchmark samplers to stress-test associative
memory. Vocab 141 over 192 tokens ⇒ almost no concept ever follows
another concept twice. There is nothing for any sequence learner —
replay, BPTT, or otherwise — to learn, because **the project has
never possessed a natural-language concept-sequence corpus**.

Stage-2's blocker is therefore upstream of the replay mechanism: it
is the **total absence of a concept-sequence corpus**, and acquiring
one under the project's hard constraints is the genuine hard problem.

## The corpus problem is fundamental under the constraints

Constraints: no external LLM, no cheating, local-only, biology-
grounded, 320 hand-curated concepts. Every obvious corpus source
fails:

- **Public text corpus** (the Phase-2 Tiny-Shakespeare approach):
  320 hand-curated concepts cover ≈0% of real running text;
  tokenizing real text to these concepts discards essentially all
  of it → no usable sequences. (This is structurally why Phase 2.3a
  char-level was NEGATIVE and a word-level version never had
  coverage.)
- **Hand-authored concept sentences:** tiny, non-scaling; and the
  human author is effectively supplying the language model — small
  and borderline against the "no cheating / standalone" intent.
- **The agent's own user conversations:** chicken-and-egg (no users;
  and Stage-1 only emits retrieved pairs, not novel sequences).
- **Procedural generation from a hand-written grammar:** that
  grammar *is* a hand-built language model; Stage-2 learning it back
  is **circular** — it would relearn the generating grammar with no
  emergent generalization. This is the exact v12–v16 / dlpfc
  composition-circularity trap the project already hit and
  documented NEGATIVE.

## Honest conclusion

"LLM-like generation in-sim without cheating" is blocked at the
**data layer**, not merely the mechanism layer — consistent with the
earlier honest blocker analysis ("no learned language distribution;
scale gap"). Stage-2 as scoped in
`2026-05-16-generative-conversation-design.md` (replay-learn concept
transitions) cannot proceed: it has no fuel, and no non-circular,
non-cheating, sufficiently-large fuel source exists under the
constraints. **Recommendation: do NOT implement Stage-2
replay-learning.** It would be a months-class NEGATIVE.

## What this leaves (honest)

- **Stage 1 stands as the genuine, shipped, hardened result**:
  a trustworthy grounded conversational agent (no confabulation
  moat, robust margin on the remediated substrate). That is the
  real, defensible conversational capability obtainable under the
  constraints — and it is delivered.
- The generative frontier's true gate is **corpus acquisition under
  no-LLM/no-cheating/local-only**. That is a *strategic* decision
  for the user, not an autonomous implementation task: it likely
  requires relaxing a constraint (e.g. sanctioning a scoped public
  text corpus mapped onto a *much larger* learned vocabulary, which
  is a different project than 320 hand-curated concepts), and that
  trade-off is the user's to make. Surfaced, not decided unilaterally.

## Process note

This is the brainstorming skill working as intended: the
context-exploration probe invalidated the idea, so **no design doc
and no writing-plans follow** (the skill's terminal is
writing-plans only on a GO). Killing a doomed design at the
brainstorm stage for one cheap probe is the highest-value possible
outcome here.

## Files

- Probe: inline analysis over committed
  `research/findings/raw/g11_bg/g20_*bench*320*.json` +
  `g20_generative_agent_smoke*.jsonl`.
- Supersedes the Stage-2 portion of
  `docs/plans/2026-05-16-generative-conversation-design.md` with a
  NO-GO + reframe (corpus problem).
