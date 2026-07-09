# D3 integration rung — the WHOLE recurrent step in ONE spiking loop (spiking LIF transition → spiking FS-WTA re-discretization → feedback), length-generalizing to ~5× training depth

**Date:** 2026-07-09
**Runner:** `research/runners/_d3_spiking_onbridge_loop_derisk.py` (reuse-by-import of rung-1 FS-WTA + rung-2 LIF transition + the group task; numpy; NO `sim/` edit).
**Verdict:** GO (S3, 6-seed) — both spiking halves compose into one recurrent loop that length-generalizes.

## Context — closing the "validated separately" gap
The D3 arc found the mechanism (discrete-attractor recurrence = re-discretize the running group state to a clean attractor each step = the project's CA3/NEF-cleanup substrate) and realized each half on spikes SEPARATELY:
- **Rung 1** (`_d3_spiking_attractor_derisk`): the RE-DISCRETIZATION on the Izhikevich FS-WTA bridge (S3 host-agree 1.0).
- **Rung 2** (`_d3_spiking_transition_derisk`): the TRANSITION (δ: state×input→next-state scores) learned THROUGH a spiking LIF hidden pool (surrogate grad, step-delta 1.0).

But rung 1's rollout still computed the transition with the **rate** tanh weights. This integration rung composes the two: **every step of the autoregressive rollout is (i) the spiking LIF transition forward → (ii) the spiking FS-WTA re-discretization → (iii) the spiking winner's `emb` feeds back as the next state.**

## The result — the full-spiking loop length-generalizes to ~5× training depth
Transition trained on SHALLOW sequences (lengths 1/2/3); the full-spiking loop rolled out on held-out DEEP sequences (lengths **8/12/16**, up to ~5× training depth — the transition is per-step teacher-forced → depth-agnostic, so depth is a genuine held-out generalization test). S3 (K=6, chance 0.167), seeds 42/43/44 (dev) + 100/101/102 (blind):

| S3, 6 seeds | held-out-DEEPER state-track |
|---|---|
| **FULL-SPIKING loop** (LIF transition + FS-WTA re-discretization + feedback) | **0.954** (host-agree **0.999**) |
| NO-REDISCRETIZE control (carry soft `softmax(scores)@emb`, no attractor) | 0.583 |

per-seed full-spiking: 1.0 / 0.975 / 0.975 / 0.875 / 0.925 / 0.975; control: 0.5 / 0.675 / 0.675 / 0.475 / 0.575 / 0.6. **GO all 6 seeds** (gap 0.371 ≫ 0.15).

## What the anti-cheats establish
- **(a) length-generalization:** the full-spiking loop holds at depth 8–16 (mean 0.954) ≫ chance 0.167 — the discrete-attractor's whole point (arbitrary depth) is realized on spikes end-to-end.
- **(b) faithful re-discretization:** spiking-winner == host-argmax over the LIF scores at **0.999** — the FS-WTA cleanly re-discretizes the spiking transition it reads (not a lucky host read).
- **(c) re-discretization is LOAD-BEARING:** the NO-REDISCRETIZE control (same trained transition, but carry the soft continuous state forward instead of the clean attractor winner) DRIFTS to 0.583 at depth. Note it was only ~0.9 at depth 4–6 (a near-perfect transition makes `softmax` nearly one-hot, so the soft carry ≈ the clean attractor when the transition is confident) — the re-discretization becomes load-bearing precisely at DEPTH, where drift accumulates. This is the honest mechanism: the clean-attractor re-discretization is what prevents drift over many steps.

## ⇒ the simulated recurrent sequence/language cortex step is realized END-TO-END on spikes
Both halves of the discrete-attractor recurrent step `s_t = re-discretize(δ(s_{t-1}, g_t))` now run on spiking neurons **in one loop, feeding back**, composing to held-out-deep depth. This is the mission-central "simulated recurrent sequence/language cortex" step, on the project's own spiking substrate, grounded in the mechanism the rate arc found (discrete-attractor = CA3), transformer-free, NO `sim/` edit.

## Honest scope + open
- Validated on S3 at depth 8/12/16. The theorem-backed **A5** (non-solvable group, K=60) one-loop is a NEGATIVE at this budget with a **singular, clear root cause: the spiking LIF transition only reaches step-delta 0.512** (it did not learn the 60-way A5 DFA at epochs=40 / n_hid=192) → the loop can't hold because the transition is near-chance PER STEP, compounding to ~0 over depth 6–10 (0.51⁸≈0.004). The FS-WTA re-discretization is NOT the problem (host-agree 0.92, itself downstream of the weak small-margin scores). This is exactly rung 2's named-open "A5-scale the spiking transition" item, now quantified: the spiking LIF transition (60-way classification through a surrogate-gradient spiking hidden pool) needs the capacity/training lever (more epochs / wider hidden / cleaner codes) to reach step-delta→1.0, at which point the loop follows (S3 already shows the loop holds once step-delta≈1.0). The RATE transition DID learn A5 (rung-1: step-delta 1.0 at n_pool=256 → FS-WTA deeper 0.933) — so the residual is specifically the SPIKING transition's A5 capacity, not the mechanism. **In progress: an A5 spiking-transition capacity sweep** (wider hidden + more epochs) to lift step-delta→1.0, then re-run the loop.
- The transition is still trained on teacher-forced per-step (state,input→next-state) triples; **learning the transition from weak (end-label) supervision remains the residual credit wall** (rung 3, `2026-07-09-D3-endlabel-supervision-boundary.md`) — the next mechanism to find (curriculum / self-supervised / reward-RL / hippocampal SWR-replay).
- The two spiking pieces run as a transition-forward (numpy LIF) + an on-bridge FS-WTA in a Python-orchestrated loop; folding the LIF transition ONTO a `SimulationBridge` region (one literal bridge) is the deeper consolidation.

## Files
`research/runners/_d3_spiking_onbridge_loop_derisk.py`; the D3 arc `2026-07-09-D3-*.md`.
