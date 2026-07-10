# D3 GATED COPY → the LIVE agent: the price of emergence is nearly **halved** (BEFORE 0.367 → 0.711)

**Date:** 2026-07-10
**Runner:** `research/runners/_d3_event_gatedcopy_agent_derisk.py` (numpy; NO `sim/` edit).
**Verdict:** 5/6 seeds GO. The deployed label-free "who was doing it BEFORE?" rises from **0.367 → 0.711**, against a labelled ceiling of 0.928.

## What changed
The self-supervised pair deployed on the **REPLAY** mechanism answered BEFORE at only **0.367** — the quantified *price of removing state labels* (`2026-07-10-D3-selfsup-pair-deployed-PARTIAL-price-of-emergence.md`). The reason was upstream: replay's held-slot decode was 0.492/0.597, because replay teaches the held slot from a single noisy emission symbol.

The **BOUNDARY-GATED COPY** removes that bottleneck by making the copy **structural** — a pre-wired route opened by an observable marker, exactly `sim/`'s own `transmission_gate` (PBWM/BG output gating). `a_prev` never had to *infer* an agent; it only had to *copy* one.

## Result (6-seed, informative discourses only)
| deployed register | BEFORE | note |
|---|---|---|
| **labelled** pair (per-step state labels) | **0.928** | the ceiling |
| **GATED COPY (no state label)** | **0.711** (0.433 – 0.833) | this rung |
| replay mechanism (no state label) | 0.367 | prior |
| gate-lesion (gate never opens) | 0.172 | ≈ chance |
| SINGLE-EVENT register | 0.000 | structurally cannot answer |
| recency | 0.167 | |
| naive "answer the current agent" | 0.050 | |
| NOW (current event) | 0.767 | unharmed |

**The gap to the labelled register falls from ~0.56 to ~0.22.** Making the copy structural nearly halves the cost of removing `(agent, patient)` state labels.

## A simplification the gated copy buys for free
With replay, `a_prev` was a **learned head**, so its slot basis differed from `a_curr`'s and the register needed a **second**, separately-calibrated slot→name read-out — fitted from RETURN clauses, where the discourse pop "reads the held slot aloud."

With a structural copy, `a_prev = g·a_curr + (1−g)·a_prev` lives in the **same basis** as `a_curr`. **One** label-free read-out (fitted from INTRODUCE clauses, where the subject is *spoken*) names both slots, and the discourse-pop calibrator is **no longer needed at all**. A mechanism that is right in one place tends to delete machinery elsewhere.

## Label-free throughout
The δ is learned from an agent-emission cross-entropy **alone**; the gate reads only the **observable** clause code; the single slot→name read-out is fitted from clauses whose subject is spoken. **No `(agent, patient)` state label anywhere.** Identity labels appear only in the evaluation.

## Anti-cheats (all pass)
- **gate-lesion collapses (0.172).** With the gate held shut nothing is ever shifted into the held slot — so the deployed BEFORE answer rides the gate, not the register's mere existence.
- **the SINGLE-EVENT register cannot answer at all (0.000, every seed)** — structural, not gradual.
- **recency (0.167)** and **naive-current (0.050)** both fail, ruling out a listener's shortcuts.
- **NOW is not degraded (0.767)** — holding a prior event costs little in the present.

## Honest reporting
- **5/6 seeds GO; seed 102 lands at 0.433** (its gate-lesion arm is also its highest, 0.300 — the gate learned poorly on that seed). This tracks the rate rung's known residual: gate learnability is seed-variable under honest `gate_cost` selection.
- The upstream headline is the adversarially-corrected one: the gated copy's held-slot decode is **≈0.63 under held-out selection** (0.693 at the tuned constant), and is **comparable to** — not far past — replay's 0.597. The *deployment* gain (0.367 → 0.711) is nonetheless large, because deployment compounds the decode with two read-outs, and the structural copy removes one of them entirely.
- 30 informative discourses per seed.

## ⇒ the claim
A prior event held by a **structural, biologically-grounded gate** rather than a learned head lets the deployed brain answer *"who was doing it before?"* at **0.711 with no state label**, versus 0.367 for the replay mechanism and 0.928 for the fully-labelled register. **The price of emergence is nearly halved by getting the mechanism right, not by adding supervision.**

## Honest scope + next
- The register runs the gated copy at rate; the held slot's **spiking** version (a persistent slow-NMDA attractor, read out of spikes at rate-model fidelity) is validated separately (`2026-07-10-D3-spiking-boundary-gated-copy-...md`). Wiring the *spiking* register into the live agent is the next rung.
- Seed 102's gate-learnability failure is the named residual.

## Files
`research/runners/_d3_event_gatedcopy_agent_derisk.py`; the mechanism `2026-07-10-D3-boundary-gated-copy-the-held-event-is-gated-not-learned.md`; the spiking hold `2026-07-10-D3-spiking-boundary-gated-copy-held-event-is-a-spiking-memory.md`; the replay deployment it supersedes `2026-07-10-D3-selfsup-pair-deployed-PARTIAL-price-of-emergence.md`.
