# D3 EVENT COMPOSITION — the fully-spiking ONE-LOOP: the WHOLE two-slot event step in ONE spiking loop (learned + executed on spikes)

**Date:** 2026-07-09
**Runner:** `research/runners/_d3_event_onbridge_loop_derisk.py` (reuse-by-import: `train_event_spiking_weak` weights + `build_fswta_score_bridge`/`fswta_drive` + `lif_rate`; numpy, real Izhikevich bridges; NO `sim/` edit).
**Verdict:** GO (6-seed: dev 42/43/44 + blind 100/101/102).

## What this closes
The event arc validated the two halves of the recurrent step SEPARATELY: the transition LEARNED through a spiking LIF hidden (the spiking-forward learning rung) and the re-discretization ON SPIKES (the FS-WTA port). This composes them into **ONE spiking loop** for the FACTORED event — each rollout step: (i) the LEARNED spiking LIF transition-forward → two K-way score vectors (agent, patient); (ii) each drives its own K-pool Izhikevich FS-WTA attractor bridge → the spiking winner; (iii) the two spiking winners = the next (a, p), FED BACK. So the running who-did-what-to-whom MEANING is maintained by a single spiking loop whose transition is **learned-on-spikes** and whose re-discretization is **on-spikes** — the master-directive *fully spiking, one loop* for the composed event. Mirrors the single-slot one-loop integration milestone.

## The result (6-seed; deep task; real Izhikevich bridges; NO `sim/` edit)
Transition trained SHALLOW (len 1/2/3); the loop rolled out held-out DEEP (len 6/7/8), genuinely-deep (AGENT-COREF):

| held-out-DEEPER, K=6, full spiking loop | mean | per-seed range |
|---|---|---|
| **LOOP event (a,p) DEEPER — learned LIF transition + FS-WTA, one loop** | **0.948** | 0.917–0.967 |
| per-slot host-agree — agent WTA | 0.995 | 0.990–0.998 |
| per-slot host-agree — patient WTA | 0.999 | 0.998–1.000 |
| LAST-2-OBJECTS shallow reader (the skeptic's control) | 0.376 | 0.360–0.390 |

**GO (all 6 seeds, dev + blind):** the whole two-slot event step runs in ONE spiking loop — the LEARNED-on-spikes LIF transition-forward + the FS-WTA re-discretization, both slots, feeding back — length-generalizing to the genuinely-DEEP task (**LOOP event-track 0.948 ≫ the LAST-2-OBJECTS shallow reader 0.376**), with **both per-slot WTAs faithful == host argmax** (agent 0.995 / patient 0.999). ⇒ the running composed MEANING is maintained by a single spiking loop that is learned-on-spikes AND executed-on-spikes = the simulated recurrent sequence/language cortex step for a composed EVENT, fully realized on spikes end-to-end.

## The event-composition arc — comprehensively delivered, fully-spiking end-to-end (7 GO rungs)
1. **Mechanism** — factored discrete-attractor (agent, patient), genuinely deep (AGENT-COREF), adversarially verified (skeptic SURVIVES-WITH-SCOPE-FIXES → task deepened).
2. **Weak-supervisable** — the (a,p) update learned from END-STATE-only supervision.
3. **Spiking re-discretization** — the FS-WTA port (two slots).
4. **Deployed standalone** — the anti-RAG payoff (answer who/what from the composed event vs flat-fact/recency).
5. **Production wire** — the deployed `MultiTurnAgent` answers from the running event (additive default-off hook, byte-identical default).
6. **Spiking-forward learning** — the transition learned through a spiking LIF hidden (fully-spiking-incl-learning forward).
7. **Fully-spiking ONE-LOOP** — the whole event step (learned LIF transition + FS-WTA re-discretization) in one spiking loop, feeding back.

All 6-seed, NO `sim/` edit. Honest scope: the surrogate-gradient BACKWARD is host BPTT (a local rule is the separate deep wall); the fully-self-supervised TEM signal (removes the endpoint label) + the relational-comparison op class (the dense-supervision residual) are the remaining research frontiers.

## Files
`research/runners/_d3_event_onbridge_loop_derisk.py`; the event arc `2026-07-09-D3-event-*.md` (7 findings).
