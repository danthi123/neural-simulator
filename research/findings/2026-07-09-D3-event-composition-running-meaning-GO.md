# D3 → INCREMENTAL EVENT COMPOSITION (the anti-RAG running meaning): the discrete-attractor maintains a running FACTORED (agent, patient) EVENT state, composing relational role-shifts

**Date:** 2026-07-09
**Runner:** `research/runners/_d3_event_composition_derisk.py` (reuse-by-import of the D3 discrete-attractor pattern; numpy; NO `sim/` edit).
**Verdict:** GO (6-seed: dev 42/43/44 + blind 100/101/102) — the cheap-first de-risk of the research-gated TOP-pick next capability.

> **ADVERSARIAL-VERIFY CORRECTION (2026-07-09):** an independent skeptic pass returned SURVIVES-WITH-SCOPE-FIXES — the core mechanism (no leakage, held-out length, fair floors) was CONFIRMED by re-running, but the FIRST task version was **2-token-shallow**: forcing the last turn to be a promote + `p←o` every turn made the answer a deterministic function of only the last two objects (a static "last-2-objects" reader scored **1.0** with no recurrence), so the "deep discourse-length composition" framing was overstated. **FIXED** by adding an **AGENT-COREF op** ("he V o": the agent PERSISTS): the final agent now traces back through a variable-length coref run to a random-depth setting, so the last-2-objects reader FAILS (~0.38) while the model holds ~0.98 — the task is now genuinely DEEP. All numbers below are the corrected deep-task 6-seed results.

## Why this (the research-gated next frontier after anaphora)
The next-direction deep-research gate ranked **incremental semantic/EVENT composition** the #1 highest-leverage step: the conversational stack's only "meaning" object is a discrete STORED FACT (a bound triple in the composer) — so the loop is *"retrieve-a-set → render-each → concatenate"* = **RAG-like however spiking the internals** (2026-07-01 gap assessment §3e; `project_communicable_brain_not_rag`). What's missing is a **running, maintained, updatable** meaning — the composed "who-did-what-to-whom, now." D3 just proved the exact substrate primitive for that (a discrete-attractor maintains + updates a running state to unbounded depth where a continuous RNN drifts), but only for **one referent** (anaphora). This lifts it to the **factored EVENT** — the anti-RAG middle layer, upstream of generation (speak FROM a meaning) + reasoning (query a meaning) + coherence, and the shared representation that unifies the pieces (composer/reservoir/D3/HTM/Broca).

## The mechanism
The running EVENT state = **(agent a, patient p)**, each ∈ 0..K-1 = a **FACTORED two-slot register** (Frankland-Greene 2015, PNAS/PMC4577152: distinct neighboring lmSTC subregions carry the current agent + patient, *"data registers of computers"*). Two discrete-attractor slots, each RE-DISCRETIZED to a clean K-way attractor per step (D3's drift-removal = the brain's CA3). Each utterance is one of THREE relational ops that UPDATE the event:
- `"s V o"` (INTRODUCE) → (a, p) = (s, o)
- `"he V o"` (AGENT-COREF) → (a, p) = (**a**, o) — the AGENT PERSISTS (a runs deep across a coref run)
- `"it V o"` (PROMOTE) → (a, p) = (**p_prev**, o) — the current PATIENT promotes to AGENT (the new agent = the PREVIOUS patient)

The **AGENT-COREF op makes the task genuinely DEEP** (the adversarial-verify fix): the final agent traces back through a variable-length coref run to a random-depth introduce/promote, so the model must TRACK the running agent across the discourse — a static "last-2-objects" reader cannot. Trained per-step-supervised on len ≤3 discourses; tested on **held-out-DEEPER** len 6-8. MUST be factored, not one attractor (agent×patient blows up K; D3's A5 lesson: sub-perfect per-step compounds over depth).

## The result (6-seed; deep task; NO `sim/` edit)
| held-out-DEEPER (len 6-8; trained ≤3), K=6 | mean | per-seed range |
|---|---|---|
| **FACTORED event (a,p) — the running meaning** | **0.980** | 0.974–0.984 |
| **LAST-2-OBJECTS shallow reader (the skeptic's control)** | **0.383** | 0.365–0.413 |
| RECURRENCE-lesion (current-token-only) | 0.372 | 0.349–0.396 |
| RECENCY floor (bind to last-mentioned) | 0.375 | 0.352–0.390 |
| JOINT-K² capacity (reported, not gated) | 0.928 | 0.921–0.934 |

**GO (all 6 seeds, dev + blind):** the factored discrete-attractor maintains the running (agent, patient) event to held-out-deeper lengths (**0.980**, agent 0.98 / patient 0.997), composing the relational agent-coref persistence + it-promotes role-shifts, where the **static LAST-2-OBJECTS reader FAILS (0.383** = the task is genuinely DEEP, not a 2-token lookup), **RECENCY** fails (0.375), and a **RECURRENCE-LESION** COLLAPSES (0.372 = the running state IS the mechanism). ⇒ D3 composes a running **who-did-what-to-whom MEANING across a discourse** — the anti-RAG middle layer the conversational loop was missing.

## The anti-cheats (adversarial-verify-hardened)
- **LAST-2-OBJECTS shallow reader** (the skeptic's decisive control): guess a = 2nd-to-last object, p = last object. In the shallow v1 it scored 1.0 (the answer WAS the last 2 objects); with AGENT-COREF it FAILS (0.383) — the agent traces back deep, so a 2-token reader is wrong on the agent → the composition is genuinely deep.
- **held-out-DEEPER** genuine: train_lens (1,2,3) vs test_lens (6,7,8), no overlap → real length-generalization.
- **RECENCY fails** (0.375): the agent is the deep-tracked entity, not the last-mentioned.
- **RECURRENCE-lesion collapses** (0.372): with the running state zeroed, the model sees only the current token and cannot track the agent across the coref run → the recurrence is load-bearing.
- **order** is non-commutative (the promote reads the evolving patient); **6-seed** dev+blind unanimous.

## Honest scope + the named next escalation
- **Per-step-supervised** here (each step's (a,p) is the teacher). The genuinely-open crux (the D3 `-language-reference-tracking-GO` lines 29-30 residual) is learning the **relational UPDATE from weak / self-supervised signal** — the ~0.29 residual. The cheapest de-risk of that is **self-supervised next-observation prediction (TEM factorization**, Whittington et al. 2020, Cell/PMC7707106, §Factorization: separate the structural/transition code from content) — the NEXT rung, and where the adversarial-verify earns its keep.
- **JOINT-K² is a CAPACITY note, not a gate:** at K=6 the 36-pair joint memorizes (0.99), so it does NOT discriminate factoring here. Factoring's true advantage is **held-out COMBINATIONS** (Fodor-Pylyshyn systematicity) + larger K/more slots (K³) — the scaling follow-on.
- The **fixed FHRR bind stays load-bearing** (2026-06-16: multi-attribute bundling is un-learnable on point neurons; a fixed self-inverse algebra bundles 0.989). So the target is *wrap the composer's fixed per-slot bind in D3's re-discretized recurrent maintenance + a learned relational update*, NOT replace FHRR.
- Then the **spiking port** (the validated transition-LIF + FS-WTA re-discretization, two slots) → on-substrate.

## Files
`research/runners/_d3_event_composition_derisk.py`; the D3 arc `2026-07-09-D3-*.md`; the next-direction gate result (in AUTONOMOUS_STATE).
