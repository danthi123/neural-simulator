# D3 EVENT PAIR → the LIVE MultiTurnAgent (6-seed GO, host + spiking): the deployed brain answers *"who was doing it BEFORE?"*

**Date:** 2026-07-10
**Runner:** `research/runners/_d3_event_pair_agent_derisk.py` (reuse-by-import: `_d3_event_connective_derisk` + `build_fswta_score_bridge`/`fswta_drive` + `MultiTurnAgent`; numpy; NO `sim/` edit).
**Verdict:** GO (6-seed dev 42/43/44 + blind 100/101/102, BOTH host and `--spiking`).

## What this deploys
The connectives rungs established the mechanism (a discourse connective marks an **event boundary** that *shifts* the running event into a previous slot) at rate and on four spiking FS-WTA attractor slots. This is the conversational payoff: `PairEventRegister` is a drop-in for `D3EventRegister` that holds `(a_curr, p_curr | a_prev, p_prev)` on the substrate, and the live agent gains **`who_agent_before()`**.

```
"dog chase cat."  "he chase fish."      -> current event: agent = dog
"THEN bird chase worm."                 -> the connective SHIFTS dog's event into the prior slot
"he chase ball."                        -> current agent = bird (coref), prior agent = dog (HELD)

ASK "who is doing it now?"     -> bird
ASK "who was doing it BEFORE?" -> dog     <- a single-event register overwrote dog and cannot answer at all
```

The agent's `hear` strips a leading connective ("then"/"but"/"meanwhile") and calls `mark_boundary()` on any register that supports it. Single-event registers simply lack the method and are unaffected — **backward-compatible** (`tests/test_multi_turn_agent.py` 3/3 green).

## Result (6-seed, 30 discourses/seed; NO `sim/` edit)
| | spiking (mean, range) | host (mean) |
|---|---|---|
| **"who was doing it BEFORE?" (pair register)** | **0.928** (0.867 – 1.000) | 0.939 |
| SINGLE-EVENT register | **0.000** (all seeds) | 0.000 |
| RECENCY (most-recently-mentioned) | 0.172 | 0.172 |
| naive "answer the CURRENT agent" | 0.178 | 0.172 |
| **"who is doing it NOW?" (unharmed)** | **0.967** | 0.978 |

## The load-bearing contrast
The single-event register scores **0.000 on every seed** — not because it is weaker, but because it **structurally cannot answer**: it overwrote the prior event and has nowhere to hold it, so `who_agent_before()` returns `None`. Meanwhile the pair register answers at 0.928 **while its current-event answer stays at 0.967** — holding a prior event costs nothing in the present.

Two further floors rule out the shortcuts a listener might take: **recency** (0.172) and **naively answering the current agent** (0.178) both fail.

## Anti-cheats (all pass)
- **(a)** pair (0.928) ≫ single-event (0.000) — structural, not gradual.
- **(b)** ≫ RECENCY (0.172) and ≫ naive-current (0.178).
- **(c)** the CURRENT-event answer is not degraded (0.967).
- **(d)** `--spiking`: all four event slots maintained on FS-WTA Izhikevich attractors, with the normalized drive (the diagnosed f-I saturation fix). Host and spiking agree (0.939 vs 0.928).
- **(e)** the connective hook is additive and default-safe; the existing agent tests pass verbatim.

## ⇒ the claim
**The deployed brain relates two composed meanings and can be asked about either.** It answers *"who is doing it now?"* and *"who was doing it before?"* across a discourse connective, on spikes — a question its own single-event predecessor could not answer at all.

## Honest scope + next
- The pair δ is per-step supervised (state labels); the **self-supervised** pair is a separate, honest arc — forward prediction does not teach the held slot, and **replay** does (`2026-07-10-D3-event-pair-selfsup-NEGATIVE-then-replay-mechanism.md`). Deploying the *self-supervised* pair register is the natural follow-on.
- Two events (depth-2); Contrast/Cause semantics beyond Sequence + "who was before" remain open.
- 30 generated discourses per seed, 6 clauses each.

## Files
`research/runners/_d3_event_pair_agent_derisk.py`; `research/runners/multi_turn_agent.py` (`who_agent_before` + the connective→`mark_boundary` hook); rate `2026-07-10-D3-event-discourse-connectives-GO.md`; spiking `2026-07-10-D3-event-connectives-ON-SPIKES-GO.md`.
