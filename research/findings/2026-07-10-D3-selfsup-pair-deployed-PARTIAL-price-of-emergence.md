# D3 SELF-SUPERVISED EVENT PAIR → the LIVE agent: an honest **PARTIAL** — and the quantified price of removing state labels

**Date:** 2026-07-10
**Runner:** `research/runners/_d3_event_selfsup_pair_agent_derisk.py` (numpy; NO `sim/` edit).
**Verdict:** **PARTIAL (2/6 seeds GO).** The mechanism is directionally real and every floor is beaten, but the deployed accuracy is weak and seed-variable.

## What was attempted
Deploy the **self-supervised** event pair (δ learned from an agent-emission cross-entropy + replay, **no `(agent,patient)` state label anywhere**) into the live `MultiTurnAgent`, so it answers *"who was doing it BEFORE?"* — the same question the **labelled** pair register answers at **0.928** (`2026-07-10-D3-event-pair-live-agent-BEFORE-GO.md`).

## Naming both emergent slots without labels (and a real job for a refuted hypothesis)
The emergent slots are *permutations* of entity identity, so a deployed register must **name** them:
- **`a_curr`** ← fitted from INTRODUCE clauses: the subject is **spoken**, so `(slot-state-after-an-introduce, named subject)` is an observable pair.
- **`a_prev`** ← nothing ever speaks it. But a **RETURN clause (the discourse pop)** copies `a_curr ← a_prev`, so the already-calibrated *current* read-out **reads aloud** whatever the prior slot was holding. Fitting `perm_prev` from `(a_prev before a return, name decoded from a_curr after it)` is therefore also label-free.

⇒ The discourse pop is **not** what teaches the held slot (that hypothesis was refuted by its own control: replay *without* pops scores as well or better). But it **is** what lets the brain **name what it is holding**. A discourse habit of returning to the prior protagonist is a **read-out calibrator, not a teacher.**

## A distribution shift I caused, found by reading my own generators
The first deployment scored **0.311** (1/6 GO). Diagnosis, from reading the two generators rather than tuning: the register is **trained** on discourses containing RETURN clauses (20%) but was **deployed** on discourses containing none — the connective only ever mapped to a boundary. Self-inflicted, and doubly so, because the RETURN is exactly what calibrates `perm_prev`.

The linguistically correct fix: a connective + a **named** subject opens a new event ("**then** *bird* chase worm" → BOUNDARY); a connective + a **pronoun** subject is a discourse pop ("**meanwhile** *he* chase ball" → RETURN). With deployment matched to training, BEFORE rose 0.311 → **0.367** and the replay ablation sharpened decisively.

## Result (6-seed, informative discourses only: a real prior event, distinguishable from the current agent)
| | mean | range |
|---|---|---|
| **BEFORE (self-supervised + replay)** | **0.367** | 0.167 – 0.633 |
| REPLAY-ABLATED register (prediction alone) | 0.139 | 0.000 – 0.267 |
| SINGLE-EVENT register | 0.000 | structurally cannot answer |
| RECENCY | 0.167 | 0.067 – 0.267 |
| naive "answer the CURRENT agent" | 0.028 | 0.000 – 0.100 |
| NOW (current event) | 0.783 | 0.700 – 0.933 |

**2/6 seeds GO** (42: 0.633; 44: 0.533). Seeds 43 and 100 sit at or below their own replay-ablated arm.

## What is nonetheless established
- **Replay is load-bearing at deployment**, not just at the probe: ablating it (prediction alone) drops BEFORE from 0.367 → 0.139, and to **0.000** on the best seed.
- **The single-event register cannot answer at all** (0.000, all seeds) — structural, not gradual.
- **Recency (0.167) and naive-current (0.028) are both beaten**, so the answer is not a listener's shortcut.
- **Both slot names were learned label-free**, including the prior slot — via the discourse pop.

## The price of emergence, quantified
| register | BEFORE |
|---|---|
| **labelled** pair (per-step state labels) | **0.928** |
| **self-supervised** pair (no state label) | **0.367** |

Removing the state labels costs roughly **0.56 absolute** on this deployed question. The loss compounds: the emergent held slot decodes at 0.597 on its *own* training distribution (79% of a one-emission ceiling), and deployment then stacks two label-free read-outs and a discourse the model must generalize to. **This gap is the deliverable** — it maps precisely what this substrate can and cannot do without being told who the agent is, and it is reported rather than tuned away.

## Next
Close the held-slot decode itself (the 0.597 → ceiling residual, and the seed-43 weakness) before re-deploying; replay on the substrate (the project's SWR machinery); a richer emergent read-out than a linear permutation.

## Files
`research/runners/_d3_event_selfsup_pair_agent_derisk.py`; the mechanism `2026-07-10-D3-event-pair-selfsup-NEGATIVE-then-replay-mechanism.md`; the labelled deployment `2026-07-10-D3-event-pair-live-agent-BEFORE-GO.md`.
