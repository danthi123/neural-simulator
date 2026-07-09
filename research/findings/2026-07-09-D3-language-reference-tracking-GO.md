# D3 → LANGUAGE (the mission payoff): the discrete-attractor does genuine DISCOURSE-REFERENT tracking — "who holds the object / who are we talking about now" across an unbounded narrative — the iterative-compose niche between retention and stack

**Date:** 2026-07-09
**Runner:** `research/runners/_d3_reference_tracking_derisk.py` (reuse-by-import: `train_endstate` [RANK-1] + `discrete_attractor_rnn`; numpy; NO `sim/` edit).
**Verdict:** MECHANISM GO (6-seed) + an honest relational-learning residual.

## The mission payoff, research-gated
The D3 arc validated the recurrent-composition substrate on an abstract group-composition proxy. The language-application research gate (`2026-07-09-D3-...-research-gate` brief) identified the cheapest GENUINELY-LINGUISTIC iterative-compose task — **discourse-referent state tracking** — the niche BETWEEN the project's own two reservoir results: EMERGE-83 (retention: hold a feature, ignore distractors) and EMERGE-84 (stack: nested pair-matching, bounded). It is the airtight linguistic face of state-tracking = permutation composition (Merrill-Petty-Sabharwal 2404.08819, "tracking entities in a long narrative"; Centering, Grosz-Joshi-Weinstein 1995; the boxes task, Kim-Schuster 2305.02363), and it is the project's own multi-turn anaphora upgraded from holding-one-referent to tracking-a-SHIFTING-one.

## The task (a possession narrative)
"Alice gives it to Bob. Bob gives it to Carol. Dave nods. …" → who has it now? The running HOLDER state updates per clause:
`δ(holder, (subj=a, obj=b)) = b if holder==a else holder` — a real transfer FROM the current holder, else a NO-OP/distractor. **State-dependent** (needs the composed history), **non-commutative** ([A→B,B→C]=C but [B→C,A→B]=B), with **built-in distractors** (the last clause is forced to a no-op so "last-named entity" reveals nothing — the markov shortcut fails while the body's transfers still move the holder off its start, so retention fails). Clause code = [subj-half = noisy ±1 pool code of a ; obj-half = code of b] (the XOR-over-pool anti-lookup — the model must read BOTH slots nonlinearly and COMPARE the subj to the tracked holder).

## The result (6-seed 42/43/44/100/101/102; K-way floors, chance 1/6=0.167; NO `sim/` edit)
| K=6, held-out-DEEPER (lengths 6/7/8) | value |
|---|---|
| **DISCRETE-ATTRACTOR holder-track (per-step δ)** | **0.881** (step-delta **0.999**, every seed) |
| markov / last-object-mentioned floor | **0.167** (= chance) |
| last-named-entity floor | 0.167 (= chance) |
| retention floor (initial holder) | 0.173 (≈ chance) |
| order-change fraction (non-commutative) | **0.600** (order genuinely matters) |
| weak (end-state-only) supervision — *residual* | 0.289 |

**MECHANISM GO (all 6 seeds):** the discrete-attractor tracks WHO-HOLDS-IT to held-out-DEEPER narrative lengths (0.881, per-step δ 0.999), and every LINGUISTIC shortcut FAILS at chance — "last object mentioned" 0.167, "last-named entity" 0.167, "initial holder" (retention) 0.173 — while 60% of narratives change their referent under reordering (non-commutative). ⇒ this is **genuine iterative discourse-referent composition**, not retention (EMERGE-83), not last-mention, not a stack (EMERGE-84). (The K-way floors are seed-independent; an earlier 2-way-property version was coincidentally inflated on some seeds — fixed by scoring the floors against the K-way holder.)

## ⇒ D3 tracks who/what we are talking about across an unbounded conversation
This is the mission payoff: the discrete-attractor (= the brain's CA3 attractor, which we realized on spikes) is the mechanism for **unbounded discourse-referent / entity tracking** — the load-bearing operation behind multi-turn anaphora and coherent long conversation — where a reservoir's fading memory and a transformer's bounded depth cannot follow the referent chain. It fills the genuine iterative-finite-state-compose slot that had no linguistic exemplar in the project.

## The honest residual (the next mechanism)
Learning the **RELATIONAL** reference-δ (compare the tracked holder to a clause slot, then conditionally read) from **WEAK (end-state-only) supervision** reaches only ~0.26 — much weaker than the group-task's LOOKUP DFA, which RANK-1 learned from weak supervision to 1.0. **The relational state-update needs stronger supervision than end-state-only provides** (the per-step-supervised "teacher-forced" arm learns it fine — step-delta 0.999). This is a precise, honest boundary: the discrete-attractor EXECUTES genuine reference-tracking given the δ; LEARNING the relational δ from weak supervision is the open sub-problem. Candidate next mechanisms (research-gate the choice): per-step self-supervised OBSERVATION prediction (RANK-3 / TEM — the agent perceives the current holder each step), a comparison-coverage curriculum (randomized initial holder so the relational δ is seen across all holders at short lengths), or a relational inductive bias. NOTE the contrast with the group task is itself the finding: LOOKUP composition learns from weak supervision (RANK-1 GO); RELATIONAL composition (the linguistic case) does not yet.

## Honest scope + next
- MECHANISM GO on the possession/entity-tracking task; the relational-δ-from-weak-supervision residual is the next mechanism.
- Escalation (if the relational-learning residual is surpassed): the pronoun-threading variant (fullest anaphora), a non-solvable transformation monoid over K≥5 referents (the theorem-backed NC¹ version), and the spiking CA3-attractor + FS-WTA port (already built for the group task).

## Files
`research/runners/_d3_reference_tracking_derisk.py`; research gate brief (dispatched 2026-07-09); the D3 arc `2026-07-09-D3-*.md`.
