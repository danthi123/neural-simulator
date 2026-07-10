# D3 EVENT COMPOSITION — the crux escalation: the running (agent, patient) MEANING is LEARNED from END-STATE-only (weak) supervision

**Date:** 2026-07-09
**Runner:** `research/runners/_d3_event_weak_supervision_derisk.py` (reuse-by-import of `make_event_task` + the RANK-1 detached-rollout weak-supervision realization; numpy; NO `sim/` edit).
**Verdict:** GO (6-seed: dev 42/43/44 + blind 100/101/102) — an independent adversarial skeptic returned SURVIVES-WITH-SCOPE-FIXES; the scope fix (deepen the task) is applied and re-validated below.

> **ADVERSARIAL-VERIFY (2026-07-09):** the skeptic CONFIRMED by re-running that the supervision is genuinely end-state-only (no per-step leakage — the only targets are STA/STP at index L-1; intermediates are the model's own detached rollout), the length split is held-out, and the shuffle/recency floors are fair. Its scope-fix: the FIRST task was **2-token-shallow** (a static last-2-objects reader scored 1.0). **FIXED** by the AGENT-COREF op (the agent persists → the final agent traces back to a random-depth setting), making the task genuinely DEEP. Re-validated: the weak-supervision learning holds on the deep task (STATE 0.970) while the last-2-objects reader FAILS (0.383).

## What this closes
The RANK-1 event de-risk (`2026-07-09-D3-event-composition-running-meaning-GO`, 0.993) built the running FACTORED (agent, patient) event but supervised it **per-step**. The genuinely-open crux (the D3 `-language-reference-tracking-GO` lines 29-30 residual) is whether the composed running MEANING can be **LEARNED without dense supervision**. This answers the sharp question the reference-tracking finding posed — is the event UPDATE a **LOOKUP composition** (a routing/copy op, learnable from the final answer alone) or a **RELATIONAL** one (needs dense per-step signal, the 0.289 residual)?

## The result (6-seed; deep task; NO `sim/` edit)
| held-out-DEEPER (len 6-8; trained ≤3), K=6, END-STATE-ONLY supervision | mean | per-seed range |
|---|---|---|
| **STATE-endpoint (K-way final a AND p) — weak supervision** | **0.970** | 0.961–0.979 |
| **LAST-2-OBJECTS shallow reader (the skeptic's control)** | **0.383** | 0.365–0.413 |
| SHUFFLE-label (memorization-floor) | **0.032** | 0.028–0.039 |
| PROPERTY-endpoint (2-color low-bit, reported) | 0.732 | 0.650–0.842 |
| RECENCY floor | 0.375 | 0.352–0.390 |

**GO (all 6 seeds, dev + blind):** the factored (agent, patient) event UPDATE — including the relational agent-coref persistence + it-promotes role-shifts — is **LEARNED from END-STATE-ONLY supervision** (only the final (a,p) is a target; the intermediate (a,p) are the model's OWN detached argmax rollout), via a short-length (1→2→3) curriculum, and length-generalizes to held-out-DEEPER discourses (**0.970**), where the static **LAST-2-OBJECTS reader FAILS (0.383** = genuinely deep tracking, not a 2-token lookup), the SHUFFLE-label memorization-floor collapses to chance (0.032 ≈ 1/36), and RECENCY fails (0.375).

## The interpretation (the LOOKUP-vs-RELATIONAL distinction, sharpened)
The reference-tracking finding established that a **relational-comparison δ** (is the holder the giver? — a conditional over values) is NOT weak-supervisable on this substrate (0.289 from end-state-only; needs dense per-step signal). The event's **promote op is different**: `a ← p_prev` is a fixed **routing/copy** (move the patient slot into the agent slot) — the transition table δ((a,p), promote) = (p, o) is a fixed function, not a value-comparison. **⇒ the LOOKUP/routing op class of the running-meaning update IS weak-supervisable** (learned from just the final answer), surpassing the dense-supervision residual for this class. The composed who-did-what-to-whom MEANING is learnable, not merely per-step-taught — the anti-RAG middle layer is a LEARNABLE capability.

## The anti-cheats (each self-checked + adversarially verified)
- **Genuinely weak (no per-step leakage):** the ONLY gradient targets are `fa_all`/`fp_all` = STA/STP at index L-1 (the final endpoints); the prev-(a,p) entering the final-step gradient come from `roll_hard` (the model's own argmax rollout), never ground-truth intermediate states.
- **held-out-DEEPER genuine:** train_lens (1,2,3) vs test_lens (6,7,8), no length overlap → the δ is length-independent (learned from short, applied at depth via the drift-free re-discretized rollout).
- **SHUFFLE collapses** (0.030 ≈ chance 1/36): the model learns the transition, not the endpoints.
- **RECENCY fails** (0.166): the forced-promote-last makes the true agent = the composed previous patient ≠ last-mentioned.
- **PROPERTY (low-bit) partial** (0.777): honestly reported — 2 color bits partially pin the K-way state (more than the reference-tracking's 1 bit, less than the full K-way endpoint), consistent with the log₂K-bits reframe.

## Honest scope + the named next escalation
- **End-state-LABEL supervision** here (weak, but the final (a,p) is still a LABEL). The fully-self-supervised rung — remove the endpoint label entirely and learn from **next-OBSERVATION prediction** (TEM factorization, Whittington et al. 2020, Cell/PMC7707106: separate the structural/transition code from content) — is the NEXT step (the research-gated route).
- **The RELATIONAL-comparison op class stays the dense-supervision residual** (0.289) — a genuinely different, harder op than the routing/promote here; its weak-supervision is the separate open problem.
- The **fixed FHRR bind stays load-bearing** (2026-06-16): the target is *wrap the composer's fixed per-slot bind in D3's re-discretized recurrent maintenance + this learned relational update*, then the **spiking port** (the validated transition-LIF + FS-WTA re-discretization, two slots).

## Files
`research/runners/_d3_event_weak_supervision_derisk.py`; the RANK-1 `2026-07-09-D3-event-composition-running-meaning-GO.md`; the D3 arc `2026-07-09-D3-*.md`.
