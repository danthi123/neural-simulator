# D3 → INCREMENTAL EVENT COMPOSITION (the anti-RAG running meaning): the discrete-attractor maintains a running FACTORED (agent, patient) EVENT state, composing relational role-shifts

**Date:** 2026-07-09
**Runner:** `research/runners/_d3_event_composition_derisk.py` (reuse-by-import of the D3 discrete-attractor pattern; numpy; NO `sim/` edit).
**Verdict:** GO (6-seed: dev 42/43/44 + blind 100/101/102) — the cheap-first de-risk of the research-gated TOP-pick next capability.

## Why this (the research-gated next frontier after anaphora)
The next-direction deep-research gate ranked **incremental semantic/EVENT composition** the #1 highest-leverage step: the conversational stack's only "meaning" object is a discrete STORED FACT (a bound triple in the composer) — so the loop is *"retrieve-a-set → render-each → concatenate"* = **RAG-like however spiking the internals** (2026-07-01 gap assessment §3e; `project_communicable_brain_not_rag`). What's missing is a **running, maintained, updatable** meaning — the composed "who-did-what-to-whom, now." D3 just proved the exact substrate primitive for that (a discrete-attractor maintains + updates a running state to unbounded depth where a continuous RNN drifts), but only for **one referent** (anaphora). This lifts it to the **factored EVENT** — the anti-RAG middle layer, upstream of generation (speak FROM a meaning) + reasoning (query a meaning) + coherence, and the shared representation that unifies the pieces (composer/reservoir/D3/HTM/Broca).

## The mechanism
The running EVENT state = **(agent a, patient p)**, each ∈ 0..K-1 = a **FACTORED two-slot register** (Frankland-Greene 2015, PNAS/PMC4577152: distinct neighboring lmSTC subregions carry the current agent + patient, *"data registers of computers"*). Two discrete-attractor slots, each RE-DISCRETIZED to a clean K-way attractor per step (D3's drift-removal = the brain's CA3). Each utterance is a relational op that UPDATES the event:
- `"x V y"` (INTRODUCE) → (a, p) = (x, y)
- `"it V z"` (PROMOTE) → (a, p) = (**p_prev**, z) — "it" = the current PATIENT promotes to AGENT (a genuine role-shift: the new agent = the PREVIOUS patient, a composed state a recency resolver cannot track).

Trained per-step-supervised on len ≤3 discourses; tested on **held-out-DEEPER** len 6-8 (the D3 length-generalization test). MUST be factored, not one attractor (agent×patient blows up K; D3's A5 lesson: sub-perfect per-step compounds over depth).

## The result (6-seed; NO `sim/` edit)
| held-out-DEEPER (len 6-8; trained ≤3), K=6 | mean | per-seed range |
|---|---|---|
| **FACTORED event (a,p) — the running meaning** | **0.993** | 0.990–0.995 |
| RECURRENCE-lesion (current-token-only) | **0.163** | 0.153–0.172 |
| RECENCY floor (bind to last-mentioned) | **0.166** | 0.158–0.177 |
| JOINT-K² capacity (reported, not gated) | 0.990 | 0.985–0.995 |

**GO (all 6 seeds, dev + blind):** the factored discrete-attractor maintains the running (agent, patient) event to held-out-deeper lengths (**0.993**, agent 0.99 / patient 0.998), composing the relational `it→patient-promotes-to-agent` role-shift, where a **RECENCY** resolver FAILS (0.166) and a **RECURRENCE-LESION** (zero the running state → current-token-only) COLLAPSES (0.163 = the running state IS the mechanism). ⇒ D3 extends from tracking one referent to composing a running **who-did-what-to-whom MEANING** — the anti-RAG middle layer the conversational loop was missing.

## The anti-cheats (each self-checked as fair)
- **held-out-DEEPER** genuine: train_lens (1,2,3) vs test_lens (6,7,8), no overlap → real length-generalization, not memorized length.
- **RECENCY fails** because the last utterance is a FORCED promote — recency gets the patient right (=last object) but the agent WRONG (guesses the object; the true agent = the composed previous patient) → joint ≈ 1/K coincidence (0.166 ✓).
- **RECURRENCE-lesion collapses** (0.163 ≈ recency): with the running state zeroed, the model can read the current (s,o) but cannot recover p_prev for the promote → the recurrence is load-bearing.
- **order** is non-commutative (the promote reads the evolving patient); **6-seed** dev+blind unanimous.

## Honest scope + the named next escalation
- **Per-step-supervised** here (each step's (a,p) is the teacher). The genuinely-open crux (the D3 `-language-reference-tracking-GO` lines 29-30 residual) is learning the **relational UPDATE from weak / self-supervised signal** — the ~0.29 residual. The cheapest de-risk of that is **self-supervised next-observation prediction (TEM factorization**, Whittington et al. 2020, Cell/PMC7707106, §Factorization: separate the structural/transition code from content) — the NEXT rung, and where the adversarial-verify earns its keep.
- **JOINT-K² is a CAPACITY note, not a gate:** at K=6 the 36-pair joint memorizes (0.99), so it does NOT discriminate factoring here. Factoring's true advantage is **held-out COMBINATIONS** (Fodor-Pylyshyn systematicity) + larger K/more slots (K³) — the scaling follow-on.
- The **fixed FHRR bind stays load-bearing** (2026-06-16: multi-attribute bundling is un-learnable on point neurons; a fixed self-inverse algebra bundles 0.989). So the target is *wrap the composer's fixed per-slot bind in D3's re-discretized recurrent maintenance + a learned relational update*, NOT replace FHRR.
- Then the **spiking port** (the validated transition-LIF + FS-WTA re-discretization, two slots) → on-substrate.

## Files
`research/runners/_d3_event_composition_derisk.py`; the D3 arc `2026-07-09-D3-*.md`; the next-direction gate result (in AUTONOMOUS_STATE).
