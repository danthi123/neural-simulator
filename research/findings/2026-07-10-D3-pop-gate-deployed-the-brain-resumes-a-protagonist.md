# D3 POP GATE → the LIVE agent: the brain **resumes a protagonist it had set aside**

**Date:** 2026-07-10
**Runner:** `research/runners/_d3_event_popgate_agent_derisk.py` (numpy; NO `sim/` edit).
**Verdict:** GO (6-seed, all anti-cheats).

## The new capability

The push gate let the deployed brain answer *"who was doing it before?"*. The pop gate lets it do something it could not
do **at all**: after a discourse pop, resume the earlier protagonist.

```
"dog chase cat."   "he chase fish."        -> current event: dog
"THEN bird chase worm."                    -> the boundary PUSHES dog's event into the held slot
"he chase ball."                           -> current agent = bird
"MEANWHILE he chase worm."                 -> a discourse POP: the return marker OPENS the read gate

ASK "who is doing it now?"   -> dog        <- the register RESUMED the agent it had set aside
```

## Result (6-seed, 30 resumption discourses per seed)

| "who is doing it now?" immediately after a discourse pop | |
|---|---|
| **pop-gated register** | **0.778** (0.533 – 1.000) |
| push-only register (**same model**, `r` forced to 0) | 0.139 |
| shortcut: keep answering the **pre-pop** agent | 0.050 |
| recency (most recently mentioned) | 0.084 |

| the other questions (30 informative discourses/seed) | pop-gated | push-only |
|---|---|---|
| "who was doing it BEFORE?" | 0.767 | 0.745 (prior deployment: 0.711) |
| "who is doing it NOW?" (all discourses) | 0.817 | 0.817 |
| SINGLE-EVENT register, BEFORE | 0.000 | — structurally cannot answer |

## The transfer risk was named before the run, and it did not bite

The register is **trained** on `make_pair_task` (whose clause code carries an explicit return mark) and **deployed** on
`make_discourse`, where a connective + a **pronoun** subject is a pop and a connective + a **named** subject is a
boundary. This repo has been bitten by exactly that train/deploy generator mismatch before, so the gate's opening was
measured separately on each deployed clause kind rather than assumed:

**deployed gate opening `r`: 0.845 on pops, 0.031 on boundaries.**

That separation is the load-bearing safety property. Both clause kinds carry a connective; a gate that keyed on the
connective would **pop at a boundary**, overwriting the present with a stale past. It does not — it discriminates the
pronoun subject from the named one.

## A defect I found in my own metric

The first version of this runner reported the resumption metric as `0.000` on all six seeds — with `n_pop = 0`. The cause
was mine: a pop sets `a_curr ← a_prev`, so `true_now == true_before`, and the "informative discourse" filter
(`skip if tb == tn`) **discards every discourse that ends in a pop, by construction.** I was measuring resumption on a
pool from which resumption had been excluded.

Fixed with a second, purpose-built pool: discourses whose last clause is a pop **and** whose resumed agent differs from
the pre-pop agent — otherwise "keep answering the same agent" would be trivially correct. That shortcut is now an
explicit control, and it scores 0.050.

## Anti-cheats (all pass)
- **vs the pop-lesion register (+0.639)** — the same trained model with `r` forced to 0. Single-variable: nothing else differs.
- **vs "keep answering the pre-pop agent" (+0.728)** and **vs recency (+0.694)** — the two shortcuts a listener could take.
- **the gate separates on the deployed generator** (0.845 on pops vs 0.031 on boundaries).
- **ordinary NOW is not degraded** (0.817, identical to push-only) and **BEFORE does not regress** (0.767).
- **a SINGLE-EVENT register still cannot answer BEFORE at all** (0.000, every seed) — structural, not gradual.
- No `(agent, patient)` state label anywhere: the transition is learned from an agent-emission cross-entropy alone, both
  gates read only the observable clause code, and the single slot→name read-out is fitted from clauses whose subject is
  spoken.

## Honest reporting
- Seed 101 is the weak seed (resumption 0.533; its BEFORE is 0.400, below the prior deployment's 0.833 on that seed —
  BEFORE is roughly a wash across seeds, +0.056 on the mean, and it is **not** the claim of this rung).
- The pop-lesion arm (0.745) is not identical to the prior `GatedCopyPairRegister` deployment (0.711): they use different
  trainers (55 vs 40 epochs, and the push gate's opening cost differs). The pop-lesion arm is the correct single-variable
  control; the prior deployment is reported only for continuity.
- Rate model. The spiking port is the open rung.

## ⇒ the claim
A brain that was **never told who any agent is** learns a discourse transition from prediction alone; a **boundary opens a
write gate** and the running event transfers into a held slot; a **return marker opens a read gate** and the brain
**resumes the protagonist it had set aside** — 0.778, against 0.139 for the identical register with the read gate shut,
and 0.050 for the shortcut of simply carrying on. Push on a boundary, pop on a return: one register, two gates, one
attentional stack.

## Next
The **spiking** pop. The held slot is a persistent slow-NMDA attractor; the push writes it by clearing for longer than
τ_NMDA and re-loading. The pop must do the opposite — **read it without disturbing it** — which the substrate makes
non-trivial: a read that drives the attractor risks re-igniting or erasing what it is reading. That asymmetry is the next
mechanism.

## Files
`research/runners/_d3_event_popgate_agent_derisk.py`; the mechanism
`2026-07-10-D3-pop-gate-the-discourse-pop-is-a-gated-copy-OUT.md`; the push gate it completes
`2026-07-10-D3-gatedcopy-deployed-price-of-emergence-halved.md`.
