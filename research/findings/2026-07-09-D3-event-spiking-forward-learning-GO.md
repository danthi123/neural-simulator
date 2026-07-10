# D3 EVENT COMPOSITION — the DEEPEST fully-spiking rung: the two-slot event transition LEARNED through a spiking LIF hidden from weak supervision

**Date:** 2026-07-09
**Runner:** `research/runners/_d3_event_spiking_learning_derisk.py` (reuse-by-import: `make_event_task` + `last2_objects_floor` + `lif_rate` from the single-slot rung; `sim.surrogate_grad`; numpy; NO `sim/` edit).
**Verdict:** GO (6-seed: dev 42/43/44 + blind 100/101/102).

## What this closes
The event-composition arc had the re-discretization on spikes (the FS-WTA port) and the transition δ learned from weak supervision (through a RATE tanh hidden). This composes them for the FACTORED event — the transition's hidden is a **SPIKING LIF pool** (rate-coded, T=16 steps, surrogate gradient), trained from **END-STATE-only supervision** via the detached-rollout curriculum (roll both slots with the LIF-argmax, supervise ONLY the final (a,p)). So the event transition FORWARD is spiking THROUGHOUT the weak-supervision learning — the master-directive *fully spiking including the learning*, not just at execution.

## The result (6-seed; deep task; NO `sim/` edit)
| held-out-DEEPER (len 6-8; trained ≤3), K=6, END-STATE-only, SPIKING-forward | mean | per-seed range |
|---|---|---|
| **SPIKING-FORWARD event (a,p) — learned through a LIF hidden** | **0.960** | 0.945–0.974 |
| LAST-2-OBJECTS shallow reader (the skeptic's control) | 0.376 | 0.360–0.390 |
| SHUFFLE-label (memorization-floor) | 0.028 | 0.018–0.041 |

**GO (all 6 seeds, dev + blind):** the factored (agent, patient) event transition is LEARNED from END-STATE-only supervision THROUGH a spiking LIF hidden (surrogate gradient) and length-generalizes to the genuinely-DEEP task (**0.960**), where the static LAST-2-OBJECTS reader FAILS (0.376) and the SHUFFLE-label memorization-floor collapses (0.028). ⇒ the event-composition LEARNING has a **spiking forward throughout**; with the FS-WTA re-discretization port, BOTH the transition-learning-forward AND the execution of the running MEANING are on spikes.

## Honest scope
- Same as the single-slot spiking-forward rung: the surrogate-gradient **BACKWARD is still host BPTT** — a biologically-plausible LOCAL learning rule is the separate deep wall (EMERGE-6..8's 5×-confirmed dead-end; the research gate says don't re-attack). This rung makes the FORWARD spiking during the two-slot learning (matching the rung-2 "transition on spikes" sense, now from weak supervision).
- The re-discretization is on spikes separately (the FS-WTA port); a single unified on-bridge loop that runs the LIF-transition-learning-forward + the FS-WTA re-discretization together is the natural composition (mirrors the single-slot one-loop integration).

## ⇒ the event-composition arc is fully-spiking end-to-end (learning + execution)
Mechanism (factored discrete-attractor, deep, adversarially verified) → weak-supervisable → spiking re-discretization (FS-WTA port) → deployed (standalone + production MultiTurnAgent wire) → **the transition LEARNED through a spiking LIF forward from weak supervision**. The anti-RAG running MEANING is found, learned-on-a-spiking-forward, executed on spikes, and deployed — on the project's substrate.

## Files
`research/runners/_d3_event_spiking_learning_derisk.py`; the event arc `2026-07-09-D3-event-*.md` (6 findings).
