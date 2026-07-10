# D3 EVENT QA → MULTI-TURN coherence (6-seed GO, host + fully-spiking): the composed running event persists across conversational turns and survives intervening questions

**Date:** 2026-07-10
**Runner:** `research/runners/_d3_event_multiturn_qa_derisk.py` (reuse-by-import: `D3EventRegister` + `MultiTurnAgent.what_does_agent_now`; numpy; NO `sim/` edit).
**Verdict:** GO (6-seed dev 42/43/44 + blind 100/101/102, BOTH host and `--spiking`).

## What this closes
The single-turn QA rungs answered "what does HE eat?" over ONE discourse. Real conversation is multi-turn: the running meaning must **carry across turn boundaries**, and **asking a question must not disturb it**. Both are tested on the LIVE `MultiTurnAgent`:

```
TURN 1:  "dog chase cat."  "he chase fish."   -> running event: agent=dog (coref-deep), patient=fish
ASK:     "what does he eat?"  -> dog's eat-fact                (a QUESTION between turns)
TURN 2:  "he chase bird."  "it flee worm."    -> the coref carries DOG across the boundary; then PROMOTE
                                                 (agent <- bird, the prev patient), patient=worm
ASK:     "what does he eat?"  -> BIRD's eat-fact               (the answer CHANGED with the composed event)
```
Every turn after the first **opens with a coref/promote**, so its first clause has no in-turn antecedent — it can only resolve through state carried across the boundary.

## The load-bearing metric (a dilution caught before gating)
An all-query average is **diluted**: a turn whose LAST clause is an INTRODUCE (names the agent) or a PROMOTE (binds the *in-turn* observed patient) needs no cross-boundary state at all. So the gate rides on the **CROSS-TURN subset** — queries whose correct answer provably requires carried state (computed by re-simulating the turn from a reset state and checking the resulting agent differs).

**Tautology avoided:** on that subset the RESET control is ~0 **by construction** (the subset is *defined* as "the reset-simulated agent differs"). It is therefore reported as a **consistency check, never a gate term** — a tautological gate metric is precisely the defect this project's own adversarial audit caught before. The gate rides on the **independent** floors (recency, flat-fact) plus the **non-rigged** all-query reset drop.

## Result (6-seed; identical host and `--spiking`)
| cross-turn subset (load-bearing) | dev 42/43/44 | blind 100/101/102 |
|---|---|---|
| **CROSS-TURN QA (live agent)** | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| RECENCY (independent floor) | 0.206 / 0.156 / 0.229 | 0.077 / 0.095 / 0.065 |
| FLAT-FACT (independent floor) | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 |
| n (cross-turn queries) | 34 / 32 / 35 | 26 / 42 / 31 |
| query-invariance | True | True |

**Non-rigged reset evidence (all queries):** 1.00 → **0.65–0.78** when the event is reset at each turn boundary.
**Query-invariance:** the running `(agent, patient)` is byte-identical before and after every question (asserted on all 720 queries) — asking does not perturb the composed meaning.
The `--spiking` variant (event maintained on two FS-WTA Izhikevich attractor slots) is **identical on all six seeds**.

## Anti-cheats
- **(a)** cross-turn QA (1.00) ≫ RECENCY (0.14 mean) and ≫ FLAT-FACT (0.00) — both independent of the subset definition.
- **(b)** the all-query reset drop (1.00 → 0.72 mean) is the honest, non-definitional reset evidence.
- **(c)** query-invariance asserted at every query.
- **(d)** fully-spiking variant reproduces exactly.

## Honest scope + next
- The reset control on the cross-turn subset is definitionally ~0 and carries no independent evidence (stated above, excluded from the gate).
- Deployment demonstration on generated 3-turn / 2-clause conversations; the composition mechanism's held-out-DEEPER generalization is the earlier rate + spiking QA rungs.
- **Next:** discourse connectives (relate two composed events — Contrast/Cause/Sequence, not just carry one); the self-supervised δ feeding the multi-turn register (a fully-emergent situation model carrying across turns).

## Files
`research/runners/_d3_event_multiturn_qa_derisk.py`; the QA rungs `2026-07-09-D3-event-QA-{unification,fully-spiking,live-agent-wire}-GO.md`.
