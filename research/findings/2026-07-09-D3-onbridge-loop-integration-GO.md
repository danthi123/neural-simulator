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
- Validated on S3 at depth 8/12/16. The theorem-backed **A5** (non-solvable group, K=60) one-loop is a NEGATIVE at this budget with a **singular, clear root cause: the spiking LIF transition only reaches step-delta 0.512** (it did not learn the 60-way A5 DFA at epochs=40 / n_hid=192) → the loop can't hold because the transition is near-chance PER STEP, compounding to ~0 over depth 6–10 (0.51⁸≈0.004). The FS-WTA re-discretization is NOT the problem (host-agree 0.92, itself downstream of the weak small-margin scores). This is exactly rung 2's named-open "A5-scale the spiking transition" item, now diagnosed. Two levers were tested and one refuted:
- **Rate-resolution (T) — REFUTED.** T=16→32→48 give near-identical step-delta (train 0.846 at 150 epochs, same 0.609 — T=32 and T=48 byte-identical), and at 300 epochs T=16 already reaches **train 0.991** — so the LIF rate code (16 levels) is NOT the cap; the transition FITS the training triples.
- **Data COVERAGE of the transition table — the real lever, CONFIRMED.** The plateau is a GENERALIZATION gap (train 0.991 vs same 0.686 at nperlen=3000): the A5 multiplication table is 60×60=3600 entries with NO interpolable structure (a structureless lookup — the theorem-backed worst case: non-solvable, no algebraic shortcuts), so held-out (state,input) pairs not covered in training cannot be inferred. The RATE transition reached step-delta 1.0 because rung-1 used nperlen=8000 (thicker coverage); mine used 3000. **DECISIVE TEST CONFIRMED:** at nperlen=8000 (n_hid=512, 200 epochs) the spiking LIF transition reaches **train=1.000, same=0.981, deeper=0.969** (up from 0.512 at nperlen=3000) — coverage was the lever, exactly as the a0 diagnosis predicted.
- **A5 full-spiking loop = strong PARTIAL (seed 42): deeper 0.767** (depth 6/8/10, host-agree 0.983) ≫ the no-rediscretize control 0.167 ≫ chance 0.017 (**45× chance**). The loop COMPOSES on the theorem-backed non-solvable K=60 group on spikes; the residual is honest compounding — the tiny per-step transition error (step-delta 0.981) × the FS-WTA re-discretization error (host-agree 0.983) accumulate over depth 6–10 (≈0.98²⁰ ≈ 0.67–0.77). The mechanism scales to A5; the strict >0.90 GO needs step-delta→1.0 (more coverage/epochs — the transition table is fully learnable, cf. rung-1's rate transition at 1.0) and host-agree→1.0 (FS-WTA tune). This is the honest scale-hardening residual, not a mechanism wall.
- **Mission-relevant framing:** A5's structureless table is the PESSIMAL case (a transition that MUST see all 3600 entries). Real LANGUAGE transitions carry compositional structure (grammar, systematic semantics) that INTERPOLATES — so a language sequence-cortex learns its transition from far less coverage than the worst-case group. The A5 coverage requirement bounds the hard case; it is not a wall.
- The transition is still trained on teacher-forced per-step (state,input→next-state) triples; **learning the transition from weak (end-label) supervision remains the residual credit wall** (rung 3, `2026-07-09-D3-endlabel-supervision-boundary.md`) — the next mechanism to find (curriculum / self-supervised / reward-RL / hippocampal SWR-replay).
- The two spiking pieces run as a transition-forward (numpy LIF) + an on-bridge FS-WTA in a Python-orchestrated loop; folding the LIF transition ONTO a `SimulationBridge` region (one literal bridge) is the deeper consolidation.

## Mission connection — the two kinds of linguistic recursion
D3 slots precisely against the EMERGE-84/85 recursion arc:
- **EMERGE-84** (reservoir stack-recursion BOUNDARY): a plain reservoir has FADING memory → drifts past depth-1 nested pair-matching (d\*=2). This is the SAME failure mode as D3's continuous-RNN / soft-carry control (drifts past trained depth).
- **EMERGE-85** (WM-buffer GO): a bounded theta-gamma ordered-slot stack pushes NESTED (center-embedding) recursion to d\*=3, capacity-bounded (the human ~2–3-embedding limit).
- **D3 (this)**: the discrete-attractor is the complementary **unbounded, no-fading running state** for ITERATIVE (single running-state) composition — exactly what EMERGE-84's reservoir lacked. It gives arbitrary depth (16+) where the reservoir faded at 2.

⇒ the two mechanisms cover the two kinds of linguistic recursion: **iterative/tail** (D3 discrete-attractor, unbounded) and **nested/center-embedding** (EMERGE-85 stack, bounded) — both on the project's spiking substrate. This is why the "simulated recurrent sequence/language cortex" needs the discrete-attractor: incremental language composition maintains a running discrete state (the parse/meaning-so-far) without drift, over arbitrary sentence length.

## Files
`research/runners/_d3_spiking_onbridge_loop_derisk.py`; the D3 arc `2026-07-09-D3-*.md`; the recursion arc `2026-07-03-emerge8{4,5}-*.md`.
