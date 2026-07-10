# D3 EVENT COMPOSITION — the SPIKING port: the running FACTORED (agent, patient) MEANING re-discretized ON SPIKES

**Date:** 2026-07-09
**Runner:** `research/runners/_d3_event_spiking_derisk.py` (reuse-by-import: `factored_event_rnn` weights + `build_fswta_score_bridge`/`fswta_drive`; numpy backend, real Izhikevich bridges; NO `sim/` edit).
**Verdict:** GO (6-seed: dev 42/43/44 + blind 100/101/102) — re-validated on the adversarial-verify-DEEPENED task (see the RANK-1 finding's correction note).

> **ADVERSARIAL-VERIFY CORRECTION (2026-07-09):** the shared `make_event_task` was deepened (an AGENT-COREF op defeats the shallow last-2-objects reader — see `2026-07-09-D3-event-composition-running-meaning-GO.md`). Re-run on the deep task: the spiking two-slot port holds (SPIKING event 0.975, both WTAs faithful). Numbers below are the corrected deep-task results.

## What this closes
The rate de-risks built the running FACTORED (agent, patient) event (per-step GO 0.993; weak-supervisable GO 0.996) in numpy. The master directive requires it **fully spiking on one brain**. This ports the re-discretization onto the project's OWN spiking substrate for the factored event: each step's transition produces TWO K-way score vectors (agent, patient); each drives its own **K-pool Izhikevich attractor bridge with a shared FS lateral-inhibition pool** (the CA3/NEF clean one-of-K winner); the two spiking winners = the next (a, p); iterate. So the running who-did-what-to-whom MEANING is maintained as **two co-evolving spiking attractors**, composing the relational role-shift to held-out-DEEPER depth.

## The result (6-seed; deep task; real Izhikevich bridges; NO `sim/` edit)
| held-out-DEEPER (len 6-8; trained ≤3), K=6 | mean | per-seed range |
|---|---|---|
| **SPIKING event (a,p) DEEPER — two FS-WTA attractor slots** | **0.975** | 0.933–1.000 |
| per-slot host-agree — agent WTA | 0.996 | 0.993–1.000 |
| per-slot host-agree — patient WTA | 0.999 | 0.983–1.000 |
| (rate event DEEPER, reference) | 0.977 | 0.971–0.987 |

**GO (all 6 seeds, dev + blind):** the running factored (agent, patient) MEANING is re-discretized ON SPIKES (two co-evolving FS-WTA Izhikevich attractor slots) and composes the relational agent-coref persistence + it-promotes role-shifts to held-out-DEEPER depth (**spiking event-track 0.975 ≈ the rate 0.977**, event chance 1/36 = 0.028; the deep task's last-2-objects shallow reader fails at 0.383), with **both per-slot WTAs faithful == host argmax** (agent 0.996 / patient 0.999). ⇒ the anti-RAG running who-did-what-to-whom MEANING runs on the project's spiking substrate — a **simulated recurrent sequence/language cortex maintaining a composed event**.

## The mechanism + anti-cheats
- **Two FS-WTA slots:** each slot's K Izhikevich attractor pools + a shared inhibitory FS pool with lateral inhibition resolve a clean one-of-K winner (the project's shared-FS / concept-pool WTA = CA3 pattern completion). The two slots use distinct bridges (seed, seed+7) so they re-discretize independently.
- **RUNG SCOPE (mirrors the group spiking port):** the TRANSITION δ is the rate-learned `factored_event_rnn` weights; only the RE-DISCRETIZATION is on-spikes. Learning the transition on-substrate is the next rung.
- **Per-slot faithfulness** (host-agree 0.99/0.999): each spiking WTA winner == the host argmax → the on-spikes re-discretization is faithful, not a lossy approximation.
- **held-out-DEEPER** (train ≤3, test 6-8) == the rate result: the spiking rollout length-generalizes (the two attractors are drift-free per step, so error doesn't compound over depth).
- The **running state is load-bearing** (inherited: the rate recurrence-lesion collapses to 0.16, `2026-07-09-D3-event-composition-running-meaning-GO`).

## Honest scope + next
- **Rate-learned transition** ported to spiking re-discretization (rung-1 scope). The on-substrate LEARNING of the two-slot transition is the follow-on.
- **The fixed FHRR bind stays load-bearing:** the target is *wrap the composer's fixed per-slot bind in D3's spiking re-discretized recurrent maintenance*, then integrate into the conversational loop (the running event becomes the composed meaning the who/what answers + generation read from).

## Files
`research/runners/_d3_event_spiking_derisk.py`; the rate `2026-07-09-D3-event-composition-running-meaning-GO.md` + `-event-weak-supervision-lookup-op-GO.md`.
