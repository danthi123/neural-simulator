# D3 EVENT → the SELF-SUPERVISED δ, EXECUTED ON SPIKES (6-seed GO): the running meaning both EMERGES from prediction and RUNS on the spiking substrate

**Date:** 2026-07-10
**Runner:** `research/runners/_d3_event_selfsup_spiking_derisk.py` (reuse-by-import: `_d3_event_selfsup_derisk` + `build_fswta_score_bridge`/`fswta_drive`; numpy backend, small Izhikevich bridge; NO `sim/` edit).
**Verdict:** GO (6-seed dev 42/43/44 + blind 100/101/102).

## What this closes
Two master-directive requirements, simultaneously, for the event composition:

- **EMERGENT** — δ is learned from an agent-emission cross-entropy ALONE, with **no `(agent, patient)` state label anywhere** (`2026-07-10-D3-event-selfsupervised-delta-GO.md`, 6-seed, adversarially verified by three skeptics).
- **SPIKING** — that learned δ's running agent slot is re-discretized by the project's own **spiking one-of-K FS-WTA Izhikevich attractor** (K excitatory pools + a shared FS inhibitory pool), not a host softmax/argmax. The spiking winner is fed back as the next state.

⇒ the running who-did-what-to-whom **MEANING** both **emerges from predicting what the brain hears** and **runs on the spiking substrate**.

## Result (6-seed; NO `sim/` edit)
| | mean | range |
|---|---|---|
| **SPIKING self-sup (held-out-DEEPER)** | **0.875** | 0.818 – 0.973 |
| **…on the coref-DEEP subset (≥3 trailing corefs)** | **0.873** | 0.766 – 0.971 |
| **…on promote-bound finals** | **0.885** | 0.798 – 0.975 |
| per-step host-agree (FS-WTA winner == host argmax) | 0.987 | 0.980 – 0.995 |
| EMISSION-SEVERED model, same spiking rollout | 0.226 | 0.193 – 0.265 |
| FAIR reservoir (ESN+ridge) on coref-DEEP | 0.147 | — |
| honest label-free floor on promote-bound | 0.169 | — |

Chance is 0.167.

## Anti-cheats (all pass)
- **(a) coref-DEEP: 0.873** while a **fair echo-state reservoir sits at 0.147** (chance) — the depth regime is exactly where the learned δ earns its keep.
- **(b) promote-bound: 0.885** vs the honest label-free `last-named-subject` floor at **0.169** — that floor structurally cannot bind a promoted patient (~51% of finals).
- **(c) host-agree 0.987:** the FS-WTA winner *is* the state (it is rolled forward), not a spiking check on a host argmax. No host argmax in the state path.
- **(d) EMISSION-SEVERED collapses to 0.226 through the SAME spiking rollout** — so the spikes are executing a *learned* δ, not a generic attractor that would sort any scores.

## The discrete-attractor thesis, confirmed again
The spiking coref-DEEP number (0.873) **matches or exceeds** the rate version's (0.841) — hard FS-WTA re-discretization removes the drift a soft recurrent state accumulates with depth. This is the D3 thesis (a discrete attractor length-generalizes where a continuous state drifts), now observed on a δ that was never given a state label.

## Honest scope
- Evaluated on a 600-item subsample of the held-out-DEEPER split (the spiking rollout runs a real bridge per clause).
- The probe's slot→agent permutation is fitted on the host rollout of the TRAIN split, then applied to the spiking test states (both one-hot); labels are used only to read the slot.
- Inherits the rate rung's scope: K=6 at the shipped capacity (K=10 needs proportional capacity and `M ≥ K`, now enforced); robust to emission noise and to `p_coref=0.8`.

## Next
Feed the self-supervised spiking state into the RANK-3 QA (a fully-emergent situation model answering questions on spikes); discourse connectives; the self-supervised δ inside the deployed `D3EventRegister`.

## Files
`research/runners/_d3_event_selfsup_spiking_derisk.py`; the rate rung `2026-07-10-D3-event-selfsupervised-delta-GO.md`; the QA rungs `2026-07-09-D3-event-QA-*.md`; multi-turn `2026-07-10-D3-event-multiturn-coherence-GO.md`.
